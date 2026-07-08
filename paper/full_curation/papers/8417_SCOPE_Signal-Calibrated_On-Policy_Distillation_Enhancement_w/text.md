# arXiv:2604.10688v2[cs.LG]30May2026

## SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting

Binbin Zheng1,2∗† Xing Ma2∗‡ Yiheng Liang2,3 Jingqing Ruan2 Xiaoliang Fu4 Kepeng Lin5 Benchang Zhu2‡ Ke Zeng2 Xunliang Cai2

1University of Science and Technology of China 2Meituan LongCat Interaction Team

3Nanjing University 4Fudan University 5Huazhong University of Science and Technology https://github.com/machine981/SCOPE

### ABSTRACT

On-policy reinforcement learning has become the dominant paradigm for reasoning alignment in large language models, yet its sparse, outcome-level rewards make token-level credit assignment notoriously difficult. On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model, but typically applies this supervision uniformly across all rollouts, ignoring fundamental differences in signal quality. We propose Signal-Calibrated OnPolicy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths. For incorrect trajectories, SCOPE performs teacher-perplexity-weighted KL distillation to prioritize instances where the teacher demonstrates genuine corrective capability, while down-weighting unreliable guidance. For correct trajectories, it applies student-perplexity-weighted maximum likelihood estimation to concentrate reinforcement on low-confidence samples at the capability boundary rather than over-reinforcing already mastered ones. Both paths employ a group-level normalization to adaptively calibrate weight distributions, accounting for the intrinsic difficulty variance across prompts. Extensive experiments on six mathematical reasoning benchmarks show that SCOPE achieves average relative improvements of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines, with extended experiments demonstrating its broader applicability.

### 1 Introduction

In the reasoning alignment of large language models (LLMs), on-policy reinforcement learning has become the dominant paradigm, where the model samples rollouts and updates its policy based on outcome correctness (Guo et al., 2025; Shao et al., 2024; Yu et al., 2025). However, the sparse, outcome-level nature of these rewards makes token-level credit assignment notoriously difficult, often demanding massive iterations to converge (Peng et al., 2026; Wei et al., 2025). On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model on the student’s self-sampled rollouts (Min et al., 2024; Fu et al., 2026b), striking a balance between distribution consistency and training efficiency.

Despite its effectiveness, OPD assumes the teacher’s dense supervision is uniformly reliable across rollouts (Ko et al., 2024; Agarwal et al., 2024; Fu et al., 2026b), problematic in two respects. (1) For incorrect trajectories, low teacher perplexity (PPL) signifies a strong reasoning grasp, enabling reliable post-error guidance. Conversely, high PPL indicates unfamiliarity, rendering the teacher’s token-level distribution an unreliable signal. Distillation must therefore prioritize teacher-confident instances. As shown in §2, low teacher PPL strongly correlates with successful error recovery (Xiong et al., 2024; Kadavath et al., 2022), validating this perplexity as a proxy for genuine corrective capability. (2) For correct trajectories, teacher KL supervision risks suppressing valid yet unconventional reasoning paths where the student diverges (Agarwal et al., 2024). While Maximum Likelihood Estimation (MLE) selfreinforcement is a natural alternative, equal-weight MLE disproportionately reinforces stably mastered samples (Zhu et al., 2025), marginalizing low-confidence instances at the capability boundary. Correct trajectories should thus be weighted adaptively by the student’s perplexity to maximize learning value.

∗Equal contribution. †This work was done during an internship at Meituan. ‡Corresponding author.

PSR Training on Qwen2.5-7B OPD Training on Distill-Qwen-1.5B

|Q1 (avg PPL=1.36) Q2 (avg PPL=1.57) Q3 (avg PPL=1.71) Q4 (avg PPL=2.38)<br><br>Lower teacher PPL achieves higher recovery rates<br><br>Recovery rate drops with higher truncation level<br><br>|
|---|

|Accuracy Improve<br><br>63.2<br><br>74.1<br><br>Diversity Degrade<br><br>93.7<br><br>84.9<br><br>Baseline<br><br>PSR Training|
|---|

|Accuracy Improve<br><br>29.4<br><br>40.2<br><br>76.5<br><br>75.0<br><br>Baseline<br><br>OPD Training<br><br>Diversity Degrade|
|---|

70

90

90

RecoveryRate(%)

60

70

70

Pass@k(%)

Pass@k(%)

50

50

50

40

30

30

30

10

10

20 40 60 80

Pass@1 Pass@32

Pass@1 Pass@32

Truncation Level (%)

(a) Accuracy and Reasoning Diversity Trade-Off (b) Teacher Recovery Rate by Perplexity Bucket

- Figure 1: (a) Performance changes on the AIME24 benchmark before and after training. Both PSR and OPD training enhance pass@1 at the expense of pass@32, highlighting a clear trade-off between accuracy and reasoning diversity. (b) Recovery rate of the teacher model across varying truncation levels, conditioned on truncated student error trajectories as prefixes and stratified by perplexity.

The above analysis reveals a shared structural flaw in standard OPD: the absence of signal quality awareness. For incorrect trajectories, OPD fails to distinguish reliable teacher guidance from unreliable supervision. For the correct ones, it treats all samples as equally valuable regardless of learning utility. Notably, the two paths require complementary weighting perspectives, with teacher perplexity applied to the former and student perplexity to the latter, motivating a unified framework that routes trajectories by correctness and applies adaptive weighting tailored to each scenario.

To this end, we propose Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework. Specifically, SCOPE routes on-policy rollouts by correctness into two supervision paths. For incorrect trajectories, it performs selective KL distillation weighted by teacher perplexity, up-weighting instances where the teacher demonstrates genuine corrective capability. For correct trajectories, it applies weighted MLE based on the student’s perplexity, concentrating reinforcement on samples at the capability boundary. Finally, both paths employ a normalization mechanism that adaptively calibrates the weight distribution within each group.

Our main contributions are as follows:

- • Empirical analysis of signal quality heterogeneity in OPD. We uncover an overlooked quality variance in OPD: teacher and student perplexity reliably predict corrective capability on incorrect trajectories and capability-boundary samples on correct ones, respectively.
- • The dual-path adaptive framework. By routing rollouts based on correctness, SCOPE directs incorrect trajectories to teacher-perplexity-weighted OPD and correct trajectories to student-perplexity-weighted MLE, achieving qualityaware supervision within a unified objective.
- • Extensive experimental validation. SCOPE achieves an average relative improvement of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines on mathematical benchmarks, with extended experiments demonstrating its broader applicability.

### 2 Preliminary Analysis

Before presenting our framework, we conduct two empirical studies that reveal fundamental limitations of existing on-policy optimization paradigms: the degradation of reasoning diversity when optimizing successful trajectories, and the inefficiency of rectifying failed ones. These findings directly motivate the dual-path design of SCOPE.

#### 2.1 Diversity Degradation

Uniformly reinforcing the student’s self-generated correct trajectories amplifies its dominant reasoning paths, marginalizing valid but low-probability alternatives (Zhu et al., 2025; Li et al., 2025; Liang et al., 2025). Meanwhile, imposing dense teacher signals forces it to match the teacher’s distribution strictly, thereby suppressing the student’s own valid and diverse explorations (Yuan et al., 2025). Ultimately, both paradigms inevitably lead to severe mode collapse.

The Pass@k Paradox. Zhu et al. (2025) report that uniformly reinforcing a model’s own correct answers (Positive Sample Reinforcement, PSR) on Qwen2.5-7B yields a stark paradox: Pass@1 improves, yet Pass@32 severely degrades from 93.7% to 84.9%. To investigate whether dense teacher supervision can circumvent this issue, we apply OPD to all generated trajectories of DeepSeek-R1-Distill-Qwen-1.5B, and observe a similarly striking pattern: Pass@1 increases, but Pass@32 drops from 76.5% to 75.0% (Figure 1a). Both results confirm that uniform optimization of

correct trajectories inevitably sharpens the policy toward dominant reasoning modes at the expense of diversity. The underlying cause is intuitive: among correct rollouts for the same prompt, some follow the dominant reasoning mode while others arrive at the answer through rare, unconventional paths. Uniform optimization treats all trajectories equally, over-reinforcing the former and extinguishing the latter. This calls for a weighting mechanism that distinguishes wellmastered solutions from under-explored ones, allocating greater importance to the latter to preserve reasoning diversity.

#### 2.2 Rectification Inefficiency

When a model generates incorrect trajectories, relying solely on self-exploration to find the correct path is highly inefficient in complex reasoning tasks (Lu & Lab, 2025; Zhao et al., 2026). Although introducing a teacher for corrective supervision is intuitive, the on-policy nature creates a severe bottleneck: the teacher risks being conditioned on flawed prefixes generated by the weaker student. If such prefixes are logically corrupted, the teacher’s guidance may degenerate into noise, making rectification inefficient and unstable.

The Flawed Prefix Trap. To investigate the efficiency of external rectification on flawed on-policy student prefixes, we conduct an Error Recovery Experiment. Specifically, we sample 2,000 problems from the DeepMath dataset (He et al., 2025b) and generate reasoning trajectories using the student model (Distill-R1-Deepseek-Qwen-1.5B), retaining the incorrect ones. We then compute the perplexity of these flawed trajectories using the teacher model (Skywork-OR17B) and stratify them into distinct buckets based on their perplexity scores. Finally, we truncate these prefixes at various length ratios and evaluate the teacher model’s recovery accuracy when prompted to complete the generation (More details and case studies are provided in Appendix C.1 and C.4, respectively).

As shown in Figure 1b, low-PPL prefixes (Q1) consistently yield significantly higher recovery rates than their high-PPL counterparts (Q4) across all truncation levels, outperforming them by a margin of up to +19.4%. Furthermore, as the truncation level increases, recovery rates drop rapidly across all groups, with even the best-performing group declining to approximately 35% at an 80% truncation ratio. This reveals a critical mechanism: high teacher PPL indicates severe context degradation, which disrupts the teacher’s reasoning process and generates high-entropy noise. Learning from these degraded regions renders rectification extremely inefficient. Conversely, low PPL ensures the prefix remains structurally coherent, allowing the teacher to provide high-quality corrective signals. This motivates our belief that efficient rectification requires down-weighting samples with high teacher PPL to filter out misleading noise.

### 3 Methodology

To overcome the aforementioned degradation of reasoning diversity and the inherent inefficiency of prefix rectification, we present SCOPE, a dual-path training framework. As illustrated in Figure 2, SCOPE routes on-policy rollouts based on trajectory outcomes, and filters out misleading teacher noise through perplexity-calibrated adaptive weighting. We first describe the outcome-driven group branching (§3.1), then detail the dual-path adaptive weighting mechanism (§3.2), and finally formulate the overall objective (§3.3).

#### 3.1 Outcome-Driven Group Branching

During the on-policy rollout, for each input prompt x, the student model generates a group of N responses, denoted as Y x = {y1,y2,...,yN}, where each response is a sequence of tokens yi = (ai,1,ai,2,...,ai,|y

i|). Each response yi ∈ Y x is subsequently evaluated by a verifier to yield a binary reward Ri ∈ {0,1}. We utilize these binary rewards as a routing signal to explicitly partition the generated trajectories into two disjoint subsets: the correct set Ωxc = {yi ∈ Y x |Ri = 1} and the incorrect set Ωxw = {yi ∈ Y x |Ri = 0}.

On-Policy Surrogate Formulation. To optimize the current policy πθ using trajectories generated by the behavior policy πold, we account for the distribution shift by defining the token-level importance sampling ratio:

πθ(ai,t |x,ai,<t) πold(ai,t |x,ai,<t)

(1)

ρi,t(θ) =

Building upon this, we design two distinct surrogate objectives for the partitioned subsets:

Valid Trajectory Exploitation (i ∈ Ωxc): Correct trajectories encapsulate direct, valid reasoning steps. Rather than relying on teacher guidance, we explicitly leverage these self-generated successful attempts. By maximizing their

- (a) Standard OPD

- (b) SCOPE (Ours)

[Figure 1]

[Figure 2]

[Figure 3]

Teacher

[Figure 4]

uniform KL Supervision

[Figure 5]

[Figure 6]

sample Reversed KLD

[Figure 7]

[Figure 8]

Student

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

PPL: 1.2

0.158

Norm()Norm()PPL1/PPL

[Figure 20]

[Figure 21]

[Figure 22]

PPL: 2.6

0.342

trajectory-level PPL

[Figure 23]

[Figure 24]

0.500

PPL: 3.8

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

sample

[Figure 30]

[Figure 31]

[Figure 32]

Student

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

PPL: 4.5

0.192

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

0.298

PPL: 2.9

[Figure 41]

trajectory-level PPL

PPL: 1.7

0.510

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Stage 1: Outcome-Driven Group Branching Stage 2: Dual-Path Adaptive Weighting

- Figure 2: (a) Standard OPD applies uniform supervision to all samples. (b) Our SCOPE framework refines the learning

process by first dividing trajectories into correct ΩC and incorrect ΩW sets, applying dual-path perplexity-based weighting, and finally optimizing the weighted branches via a unified objective.

likelihood through the surrogate objective, we reinforce the model’s intrinsic capabilities:

|yi|

ρi,t(θ) (2)

LMLE(x,yi;θ) = −

t=1

Flawed Trajectory Rectification (i ∈ Ωxw): Incorrect trajectories lack inherent supervisory targets. To enable effective rectification, we leverage the teacher policy πT to provide external guidance. The on-policy distillation objective minimizes the reverse KL divergence by treating the token-level log-ratio as a negative advantage, yielding the following surrogate loss:

|yi|

ρi,t(θ) log πθ¯(ai,t |x,ai,<t) − log πT(ai,t |x,ai,<t) , (3) where θ¯denotes parameters detached from the computational graph.

LOPD(x,yi;θ) =

t=1

#### 3.2 Dual-Path Adaptive Weighting

To mitigate diversity degradation and improve rectification efficiency, we introduce a novel mechanism termed Dual-Path Adaptive Weighting (DPAW), which operates strictly within the candidate response group of each

prompt. Let log π(yi |x) = |ty=1i| log π(ai,t |x,ai,<t) denote the sequence-level log-probability. To intuitively quantify the trajectory’s intrinsic uncertainty, we formulate our weighting mechanism using sequence perplexity, where

PPL(yi|x) = exp(−|y1

i| log π(yi |x)).

Student-guided Weight: Amplifying “Unconventional Valid Paths”. For correct trajectories (i ∈ Ωxc), we want the student model to focus on instances where it successfully reaches the correct outcome through low-probability, alternative routes. To assign higher weights to these low-confidence trajectories, we apply a group-relative softmax over the length-normalized negative log-probabilities. Using the sequence probability πS(yi |x), the student-guided weight wistu is computed as:

exp −τ|1y

i| log πS(yi |x) j∈Ωxc exp −τ|1y

wistu =

=

j| log πS(yj |x)

PPLS(yi |x)1/τ

, ∀i ∈ Ωxc. (4)

j∈Ωxc PPLS(yj |x)1/τ

- As shown in the rightmost term, this formulation can be elegantly expressed as a direct group-level normalization of the student’s perplexity (scaled by a temperature τ). Thus, correct but high-perplexity samples naturally receive amplified supervision. A theoretical analysis is provided in Appendix A.2.

Teacher-guided Weight: Filtering Out “Prefix-Induced Noise”. For incorrect trajectories (i ∈ Ωxw), conditioning the teacher on flawed prefixes often leads to high-entropy noise (as demonstrated in Section 2.2). To prevent the student from inheriting this noise, we rely on the teacher only when it provides highly confident corrections. Specifically, we apply the softmax directly over the teacher’s length-normalized log-probabilities:

exp τ| 1y

i| log πT(yi |x) j∈Ωxw exp τ| 1y

PPLT(yi |x)−1/τ

witea =

j∈Ωxw PPLT(yj |x)−1/τ , ∀i ∈ Ωxw. (5)

=

j| log πT(yj |x)

This formulation selectively down-weights instances where the teacher exhibits high perplexity, thereby effectively filtering out the noise induced by flawed prefixes. We formalize this filtering criterion in Appendix A.1.

- 3.3 The Overall SCOPE Objective

Finally, we integrate the outcome-driven branches and the adaptive weights into an overall objective over the dataset D. The overall SCOPE loss JSCOPE is formulated as:

JSCOPE = Ex∼D

i∈Ωxc

wistu · LMLE(x,yi) +

i∈Ωxw

witea · LOPD(x,yi) . (6)

The group-level normalization stabilizes adaptive updates across prompts of varying difficulty. A formal analysis is provided in Appendix A.3. Within this framework, SCOPE adaptively calibrates supervision signals at the group level: it reinforces the student’s boundary capabilities on valid paths, while distilling only informative corrections from the teacher on flawed ones.

- 4 Experiment

- 4.1 Experimental Setup

Training Settings and Baselines. We employ DeepSeek-R1-Distill-Qwen-1.5B (Guo et al., 2025) and Qwen3-1.7BBase (Yang et al., 2025) as student models, paired with SkyWork-OR1-7B (He et al., 2025a) and Qwen3-8B-Instruct (Yang et al., 2025) as their respective teachers, all trained on DeepMath (He et al., 2025b). We compare SCOPE against the following baselines:

- • Group Relative Policy Optimization (GRPO) (Shao et al., 2024): Optimizes the policy via group-relative advantages based on outcome rewards across multiple sampled responses.
- • Knowledge Distillation (KD) (Kim & Rush, 2016): Trains the student on static teacher-generated sequences from an offline dataset via supervised learning.
- • On-Policy Distillation (OPD) (Lu & Lab, 2025): Applies dense, token-level KL divergence supervision from the teacher on student-sampled trajectories.

Evaluation Benchmarks and Metrics. To comprehensively assess the reasoning capabilities of our model, we measure its performance across a wide range of datasets, including MATH500 (Hendrycks et al., 2021), AIME24 (MAA, 2024), AIME25 (MAA, 2025), AMC 2023 (MAA, 2023), Minerva (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). Our evaluation employs two key metrics: Avg@32, which reflects the model’s expected stability, and Pass@32, which highlights its upper-bound capability.

Implementation Details. During training, we employ a global batch size of 256, a maximum prompt length of 4,096 tokens, a completion length of 12,288 tokens, a rollout temperature of 0.6, and a weight temperature of 1.0. For evaluation, we report performance based on a rollout temperature of 0.6, top-p sampling with p = 0.95, and a maximum response length of 32,768 tokens. More details are provided in Appendix B.1.

- Table 1: Main results on mathematical reasoning benchmarks under different teacher–student configurations. We report Avg@32 (A@32) and Pass@32 (P@32) for each benchmark. Bold denotes the best performance and underlined the second-best.

Model AIME24 AIME25 AMC23 MATH500 Minerva Olympiad Average

A@32 P@32 A@32 P@32 A@32 P@32 A@32 P@32 A@32 P@32 A@32 P@32 A@32 P@32

###### Teacher: SkyWork-OR1-7B −→ Student: DeepSeek-R1-Distill-Qwen-1.5B

R1-Distill-Qwen-1.5B 29.4 76.5 23.9 46.9 72.7 94.7 84.6 97.3 32.3 55.9 44.2 67.5 47.9 73.1 w/ GRPO 35.5 68.3 24.5 45.1 75.1 95.0 87.0 96.7 35.1 53.5 40.5 67.7 49.6 71.1 w/ KD 26.6 71.4 22.2 45.6 69.1 96.3 84.1 97.4 30.7 54.1 39.7 66.9 45.4 72.0 w/ OPD 40.2 75.0 28.9 48.5 75.9 95.0 89.0 97.7 34.9 53.0 44.9 69.3 52.3 73.1 w/ SCOPE (Ours) 42.7 77.9 30.4 50.9 80.9 97.2 89.8 97.9 37.8 55.1 49.7 70.9 55.2 75.0 ∆% vs. OPD +6.22% +3.87% +5.19% +4.95% +6.59% +2.32% +0.90% +0.20% +8.31% +3.96% +10.69% +2.31% +5.54% +2.60%

###### Teacher: Qwen3-8B-Instruct −→ Student: Qwen3-1.7B-Base

Qwen3-1.7B-Base 3.7 26.7 3.2 20.7 24.6 69.4 54.8 88.6 11.2 41.6 16.3 48.0 19.0 49.2 w/ GRPO 7.5 27.2 4.2 24.5 33.1 60.0 62.9 89.8 25.3 49.8 25.9 50.9 26.5 50.4 w/ KD 4.6 18.9 5.8 24.8 29.8 66.1 50.2 86.2 16.6 43.2 13.3 41.4 20.1 46.8 w/ OPD 12.2 31.5 10.6 29.7 43.1 80.1 67.9 90.7 24.2 49.4 25.3 53.7 30.6 55.9 w/ SCOPE (Ours) 13.3 31.5 12.1 35.6 46.3 83.0 70.9 93.0 27.9 54.0 24.6 54.6 32.5 58.6 ∆% vs. OPD +9.02% +0.00% +14.15% +19.87% +7.42% +3.62% +4.42% +2.54% +15.29% +9.31% -2.77% +1.68% +6.21% +4.83%

- 0.1

- 0.2

- 0.3

- 0.4

32

42

| |
|---|

30

| |
|---|

38

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

| |
|---|

| |
|---|

| |
|---|

EntropyLoss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

28

Avg@32(%)

Avg@32(%)

| |
|---|

| |
|---|

34

| |
|---|

26

30

24

26

GRPO

GRPO

22

GRPO

KD

KD

OPD

OPD

OPD

22

20

SCOPE(Ours)

SCOPE (Ours)

SCOPE (Ours)

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Training Steps

Training Steps

Training Steps

(a) Dynamics of Entropy Loss

(b) Dynamics of AIME24 Avg@32

(c) Dynamics of AIME25 Avg@32

- Figure 3: Training dynamics comparing GRPO, OPD, and SCOPE (Ours): (a) entropy loss across training steps, and Avg@32 (%) performance on (b) AIME24 and (c) AIME25.

#### 4.2 Main Results

Performance on Mathematical Reasoning. Table 1 illustrates the evaluation results across challenging mathematical reasoning benchmarks. Under the primary configuration (Teacher: SkyWork-OR1-7B, Student: Distill-Qwen-1.5B), SCOPE consistently achieves the best Avg@32 performance. Compared to strong baselines, SCOPE yields an average relative improvement of +5.54% over standard OPD, with notable gains of +10.69% on Olympiad and +6.59% on AMC23. These gains stem from our teacher-guided weighting, which adaptively penalizes high-perplexity failed trajectories to bypass the “flawed prefix trap” and thereby extract precise corrective signals. Furthermore, Pass@32 results demonstrate SCOPE’s unique capability to preserve reasoning diversity and overcome the Pass@k paradox, a challenge that is especially severe when optimizing raw base models. As shown in the bottom half of Table 1, experiments on the Qwen3-1.7B-Base reveal that standard paradigms (e.g., GRPO, KD) drastically degrade the base model’s inherent exploration ability, indicating severe mode collapse. In contrast, SCOPE effectively prevents this degradation and significantly elevates the multi-sample pass rate. This improvement can be attributed to our studentguided weighting, which actively amplifies “unconventional valid paths” by assigning higher weights to correct but high-perplexity trajectories.

Training Dynamics. Figure 3 illustrates the training dynamics of SCOPE alongside GRPO and OPD. Figure 3(a) shows a clear contrast in entropy evolution. While GRPO exhibits continuous entropy decay (a direct driver of the Pass@k paradox via premature exploitation), both OPD and SCOPE sustain a healthy policy entropy. However, standard OPD soon reaches a performance plateau as its uniform supervision ignores signal quality. With dual-path adaptive weighting, SCOPE achieves consistently superior performance and sample efficiency (Figures 3b and c). Overall, while

81

73

65

Pass@K(%)

57

49

41

Baseline

GRPO

33

KD

OPD

25

SCOPE (Ours)

1 2 4 8 16 32

K (log2 scale)

(a) Pass@k on AIME24

52

44

Pass@K(%)

36

Baseline

28

GRPO

KD

OPD

20

SCOPE (Ours)

1 2 4 8 16 32

K (log2 scale)

(b) Pass@k on AIME25

95

90

Pass@K(%)

85

80

75

Baseline

GRPO

KD

70

OPD

SCOPE (Ours)

65

1 2 4 8 16 32

K (log2 scale)

(c) Pass@k on AMC23

- Figure 4: Pass@k (%) performance comparison of GRPO, OPD, and SCOPE (Ours) on the AIME24, AIME25, and AMC23 benchmarks using the DeepSeek-R1-Distill-Qwen-1.5B model.

Avg@32

(a) Performance on AIME24 (b) Performance on AIME25

SCOPE (Ours)

w/o Teacher-guided Weight w/o Student-guided Weight w/o DPAW

w/ OppositeTeacher-guided Weight w/ Opposite Student-guided Weight

Removing Weights

Reversing Weights

[Figure 47]

[Figure 48]

Pass@32 Avg@32 Pass@32

- Figure 5: Ablation study on the dual-path adaptive weighting mechanism (DPAW) of SCOPE. We report the performance of Avg@32 (%) and Pass@32 (%) on AIME24 (a) and AIME25 (b).

GRPO underperforms due to collapsed exploration and OPD is limited by noisy distillation, SCOPE effectively breaks this bottleneck by combining diversity-preserving exploration with quality-aware error rectification.

Pass@k Performance. As illustrated in Figure 4, SCOPE consistently outperforms all baselines in Pass@k metrics across AIME24/25 and AMC23. Notably, standard methods such as GRPO and OPD suffer from the Pass@k paradox, which is particularly evident in the AIME24 evaluation. They exhibit restricted diversity scaling, where performance gains from multiple samples diminish or plateau at larger k. In contrast, SCOPE amplifies unconventional valid trajectories near the student’s capability boundary, enabling the model to preserve multiple complementary reasoning routes. As a result, its pass rates continue to improve with larger sample sizes up to k = 32, demonstrating stronger diversity scaling. These results show that SCOPE improves not only Pass@1 performance, but also the coverage of diverse reasoning modes, validating the effectiveness of signal-quality-aware supervision in preserving exploration diversity.

#### 4.3 Ablation Study

Effectiveness of DPAW Mechanism. To validate the efficacy of the DPAW mechanism, we conduct an ablation study on the AIME24/25 benchmarks, as shown in Figure 5. Removing the entire DPAW module results in severe performance degradation. For example, the AIME25 Pass@32 drops significantly from 50.9% to 45.7%. This demonstrates that standard uniform weighting, which fundamentally ignores signal quality, fails to optimally leverage on-policy rollouts. Furthermore, omitting the student-guided weight reduces multi-sample pass rates, as AIME24 Pass@32 drops from 77.9% to 74.1%, while reversing its direction also hurts performance. Likewise, removing or reversing the teacherguided weight compromises overall accuracy. Notably, reversing the teacher-guided weight causes a drastic drop in the AIME24 Avg@32 from 42.7% to 38.6%. This empirically verifies that dynamically penalizing unreliable teacher guidance effectively filters out prefix-induced noisy distillation signals. Collectively, these findings indicate that the two components of DPAW are highly complementary: the student-guided weighting maximizes exploration diversity on successful trajectories, while the teacher-guided weighting rigorously mitigates distillation noise on failed ones.

- Table 2: Results on code generation benchmarks based on the 1.5B student model. We report Avg@32 (A@32) and Pass@32 (P@32) for each benchmark. Bold denotes the best performance and underlined the second-best.

Model HumanEval Codeforces LiveCodeBench Average

A@32 P@32 A@32 P@32 A@32 P@32 A@32 P@32 Teacher: SkyWork-OR1-7B −→ Student: DeepSeek-R1-Distill-Qwen-1.5B

R1-Distill-Qwen-1.5B 57.8 74.1 20.3 39.4 21.2 35.2 33.1 49.6 w/ GRPO 61.5 75.2 32.4 45.3 34.5 45.7 42.8 55.4 w/ KD 49.5 73.2 12.9 22.9 17.1 25.1 26.5 40.4 w/ OPD 64.3 78.2 33.3 52.3 30.2 47.0 42.6 59.2 w/ SCOPE (Ours) 67.2 80.2 34.4 53.7 32.3 47.0 44.6 60.3 ∆% vs. OPD +4.51% +2.56% +3.30% +2.68% +6.95% +0.00% +4.69% +1.86%

- 4.4 Extended Experiments

Table 2 further evaluates the generality of our SCOPE on code generation tasks, including HumanEval (Chen et al., 2021), LiveCodeBench (Jain et al., 2025) and Codeforces problems. Compared with standard OPD, SCOPE consistently improves Avg@32 across all benchmarks, yielding an average relative gain of +4.69%, and also outperforms GRPO by +4.21% on average. The gains over OPD are particularly notable on HumanEval and LiveCodeBench, where SCOPE increases Avg@32 by +4.51% and +6.95%, respectively. In addition, SCOPE improves the average Pass@32 from 59.2% to 60.3%, further indicating that dual-path adaptive weighting remains effective beyond mathematical reasoning. These results suggest that by distinguishing reliable teacher corrections from noisy ones while reinforcing low-confidence successful trajectories, SCOPE provides broadly applicable supervision signals for various verifiable tasks. More details can be found in Appendix B.2.

- 5 Related Work

- 5.1 Reinforcement Learning with Verified Rewards

Reinforcement learning with verified rewards (RLVR) has recently driven major advances in the reasoning capabilities of LLMs (Guo et al., 2025; Fu et al., 2026a; Yu et al., 2025), leveraging deterministic outcome verifiers in objective domains (e.g., mathematics and code generation) to provide unambiguous signals that prevent reward hacking and incentivize autonomous exploration (Bin Tarek & Beheshti, 2025; Dong et al., 2025). However, standard RLVR algorithms such as GRPO (Shao et al., 2024) rely on sparse, scalar outcome rewards dispensed only at the terminal step of long reasoning trajectories, severely exacerbating the credit-assignment problem (Peng et al., 2026; Wei et al., 2025) and depriving the model of granular process supervision (Hübotter et al., 2026). This difficulty is further amplified for smaller LLMs, whose limited representational capacity leaves less room for autonomous credit propagation (Xu et al., 2025; Ko et al., 2026). While Process Reward Models (PRMs) (Lightman et al., 2023; Cui et al., 2025) can offer step-wise feedback, they demand costly human annotation and generalize poorly across domains. This bottleneck motivates seeking dense, token-level supervision from capable teacher models through distillation.

#### 5.2 Knowledge Distillation

Knowledge distillation (KD) (Hinton et al., 2015) has become a primary paradigm for transferring teacher capabilities to compact student LLMs, predominantly through token-level logit alignment (Gu et al., 2023; Agarwal et al., 2024; Jung et al., 2025). Off-policy KD trains on static teacher-generated trajectories (Guo et al., 2025; Yang et al., 2025) but inherently suffers from exposure bias and distribution mismatch (Agarwal et al., 2024; Hsieh et al., 2023). OnPolicy Distillation (OPD) addresses this by optimizing student-sampled rollouts with teacher feedback via reverse KL divergence, yielding stronger convergence (Ko et al., 2024; Lu & Lab, 2025). Recent RL-KD hybrids further unify verified rewards with teacher supervision within a single training loop: KDRL (Xu et al., 2025) jointly optimizes reward and KL objectives, while RLAD (Zhang et al., 2026) and REOPOLD (Ko et al., 2026) inject teacher signals through dynamic reward shaping. Despite their progress, all these methods implicitly assume that teacher supervision is uniformly reliable across all rollouts, overlooking the fact that teachers can be confidently wrong on specific trajectories, turning indiscriminate distillation into a vehicle for confirmation bias propagation. This limitation calls for a trajectory-level adaptive mechanism that differentiates supervision strategies based on both rollout correctness and signal reliability, which is precisely the design principle behind our proposed framework.

### 6 Conclusion

In this work, we proposed Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that incorporates signal quality awareness into on-policy distillation. SCOPE routes rollouts by correctness into two complementary supervision paths: teacher-perplexity-weighted KL distillation for incorrect trajectories to prioritize reliable corrective guidance, and student-perplexity-weighted MLE for correct trajectories to reinforce under-explored reasoning paths at the capability boundary. A unified group-level normalization adaptively calibrates weight distributions across prompts of varying difficulty. Extensive experiments on six mathematical reasoning benchmarks show that SCOPE achieves average relative improvements of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines. Additional results on three code generation benchmarks further demonstrate its effectiveness and broader applicability across various tasks.

### Limitation

We acknowledge two main limitations of the current study. First, SCOPE depends on automatically verifiable outcome signals to divide trajectories into correct and incorrect branches. Accordingly, our experiments are conducted in domains where correctness can be reliably checked, including mathematical reasoning and code generation. Applying SCOPE to tasks with subjective, incomplete, or preference-driven feedback, such as open-ended dialogue or creative writing, may require additional reward models or more sophisticated verification mechanisms. Second, our empirical evaluation remains constrained by computational resources. Although we evaluate SCOPE across multiple benchmarks and teacher–student configurations, further experiments on larger foundation models, MoE architectures, and more diverse verifiable domains remain important directions for future work.

### Ethical Considerations

We recognize the ethical responsibilities associated with research on large language models. Our experiments are conducted using publicly available datasets and benchmarks, without involving private, sensitive, or personally identifiable information. We use all datasets, benchmarks, and pretrained models under their original licenses and terms of use, and our released resources will follow the corresponding licensing requirements. We report the main training configurations, evaluation settings, and implementation details to support transparency and reproducibility. Although our method is designed to enhance model reasoning capabilities, outputs may still be unreliable in certain cases and should not be relied upon as the sole basis for high-stakes decisions. The released resources are intended for research purposes, and we encourage responsible use.

### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations, 2024.

Mirza Farhan Bin Tarek and Rahmatollah Beheshti. Reward hacking mitigation using verifiable composite rewards. In Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics, pp. 1–6, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, et al. Agentic entropy-balanced policy optimization. arXiv preprint arXiv:2510.14545, 2025.

Xiaoliang Fu, Jiaye Lin, Yangyi Fang, Binbin Zheng, Chaowen Hu, Zekai Shao, Cong Qin, Lu Pan, Ke Zeng, and Xunliang Cai. Maspo: Unifying gradient utilization, probability mass, and signal reliability for robust and sample-efficient llm reasoning. arXiv preprint arXiv:2602.17550, 2026a.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562, 2026b.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081): 633–638, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025a.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025b.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob

Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint

arXiv:1503.02531, 2015.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 8003–8017, 2023.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Naman Jain, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, volume 2025, pp. 58791–58831, 2025.

Seongryong Jung, Suwan Yoon, DongGeon Kim, and Hwanhee Lee. Todi: Token-wise distillation via fine-grained divergence control. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 8089–8102, 2025.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 1317–1327, 2016.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. Distillm: Towards streamlined distillation for large language models. arXiv preprint arXiv:2402.03898, 2024.

Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation. arXiv preprint arXiv:2603.11137, 2026.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

Long Li, Zhijian Zhou, Jiaran Hao, Jason Klein Liu, Yanting Miao, Wei Pang, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, et al. The choice of divergence: A neglected key to mitigating diversity collapse in reinforcement learning with verifiable reward. arXiv preprint arXiv:2509.07430, 2025.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Xiao Liang, Zhongzhi Li, Yeyun Gong, Yelong Shen, Ying Nian Wu, Zhijiang Guo, and Weizhu Chen. Beyond pass@ 1: Self-play with variational problem synthesis sustains rlvr. arXiv preprint arXiv:2508.14029, 2025.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi:

10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation. MAA. American mathematics competitions - amc, 2023. Accessed: 2023.

- MAA. American invitational mathematics examination - aime, 2024. Accessed: 2024.
- MAA. American invitational mathematics examination - aime, 2025. Accessed: 2025.

Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.

Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, and Mingyi Hong. Hiper: Hierarchical reinforcement learning with explicit credit assignment for large language model agents. arXiv preprint arXiv:2602.16165, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Quan Wei, Siliang Zeng, Chenliang Li, William Brown, Oana Frunza, Wei Deng, Anderson Schneider, Yuriy Nevmyvaka, Yang Katie Zhao, Alfredo Garcia, et al. Reinforcing multi-turn reasoning in llm agents via turn-level reward design. arXiv preprint arXiv:2505.11821, 2025.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In International Conference on Learning Representations, volume 2024, pp. 23650–23678, 2024.

Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, and Fei Mi. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning. arXiv preprint arXiv:2506.02208, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Xiaoyang Yuan, Yujuan Ding, Yi Bin, Wenqi Shao, Jinyu Cai, Jingkuan Song, Yang Yang, and Heng Tao Shen. More than one teacher: Adaptive multi-guidance policy optimization for diverse exploration. arXiv preprint arXiv:2510.02227, 2025.

Zhaoyang Zhang, Shuli Jiang, Yantao Shen, Yuting Zhang, Dhananjay Ram, Shuo Yang, Zhuowen Tu, Wei Xia, and Stefano Soatto. Reinforcement-aware knowledge distillation for llm reasoning. arXiv preprint arXiv:2602.22495, 2026.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

### A Theoretical Motivation and Derivation

In this section, we provide a theoretical motivation and derivation for SCOPE. We analyze how uniform distillation on flawed prefixes may amplify noisy teacher signals, explain why student-guided weighting helps preserve low-probability but valid reasoning paths, and show how group-level normalization stabilizes adaptive updates.

#### A.1 Perplexity-Aware Filtering on Flawed Prefixes

We first analyze why uniform distillation can be detrimental on incorrect trajectories. In on-policy distillation, the teacher signal is evaluated on trajectories sampled by the behavior policy πold, rather than under an explicit expectation over the full teacher distribution. Following the surrogate formulation in Section 3, the detached token-level log-ratio serves as an advantage-like signal:

|y|

ρt(θ) log πθ¯(at | x,a<t) − log πT(at | x,a<t) , (7)

LOPD(θ) = Ey∼π

old

t=1

where ρt(θ) = πθ(at | x,a<t)/πold(at | x,a<t), and θ¯ denotes parameters detached from the computational graph. Evaluating the gradient near the behavior policy, i.e., θ ≈ θold, gives

|y|

∇θ log πθ(at | x,a<t) · log πθ¯(at | x,a<t) − log πT(at | x,a<t) . (8)

∇θLOPD(θ) ≈ Ey∼π

old

t=1

Let At = log πθ¯(at | x,a<t) − log πT(at | x,a<t) denote the detached distillation signal. For a single sampled trajectory, the corresponding stochastic gradient estimator is

|y|

∇θ log πθ(at | x,a<t) · At. (9)

gOPD(x,y) =

t=1

When the prefix a<t forms a flawed reasoning context, the teacher may become less reliable on the continuation. It may assign lower probability to the sampled token or produce a less informative distribution, increasing −log πT(at | x,a<t) (Xiong et al., 2024). As a result, At may be dominated by low-confidence evaluations rather than meaningful corrective directions.

To see how this affects optimization, consider the squared norm of the single-trajectory gradient estimator. By Cauchy’s inequality,

2

|y|

|y|

∥∇θ log πθ(at | x,a<t)∥2 · A2t. (10)

∥gOPD(x,y)∥2 =

∇θ log πθ(at | x,a<t) · At

≤ |y|

t=1

t=1

If the score function is bounded as ∥∇θ log πθ(at | x,a<t)∥ ≤ G, then

|y|

E ∥gOPD(x,y)∥2 ≤ |y|G2

E A2t . (11)

t=1

This bound shows that the second moment of the stochastic update is controlled by the magnitude of the detached distillation signal. Thus, low teacher likelihood on flawed prefixes can enlarge the update scale and introduce noisy gradients.

Trajectory-level teacher perplexity summarizes this low-confidence regime:

1 |y|

log πT(y | x) , (12)

PPLT(y | x) = exp −

where log πT(y | x) = |ty=1| log πT(at | x,a<t). A high PPLT(y | x) indicates that the teacher assigns low average likelihood to the sampled trajectory. For incorrect trajectories, this often suggests that the teacher is conditioned on a

degraded prefix, making its token-level guidance less reliable. Motivated by this interpretation, SCOPE down-weights incorrect trajectories with high teacher perplexity:

PPLT(yi | x)−1/τ

witea =

j∈Ωxw PPLT(yj | x)−1/τ . (13) Therefore, incorrect trajectories with high teacher likelihood are emphasized, while high-perplexity trajectories, which are more likely to contain prefix-induced teacher noise, are suppressed. This explains how teacher-guided weighting improves rectification efficiency by reducing the influence of likely noisy teacher signals.

#### A.2 Exploration of Valid Trajectories

We next analyze why student-guided weighting helps alleviate the Pass@k paradox caused by uniform reinforcement of successful trajectories. For correct trajectories Ωxc, standard MLE reinforces all successful samples uniformly:

∇θLMLE ∝ −

∇θ log πθ(yi | x). (14)

i∈Ωxc

However, under on-policy sampling, correct trajectories are not equally represented. Dominant reasoning paths are sampled more frequently, while rare but valid paths appear with much lower probability. Uniform reinforcement can therefore amplify dominant modes and weaken the contribution of low-probability valid paths.

SCOPE counteracts this effect by assigning larger weights to correct trajectories with higher student perplexity:

PPLS(yi | x)1/τ j∈Ωxc PPLS(yj | x)1/τ

wistu =

. (15)

Ignoring the within-group normalization constant and focusing on relative weighting, this behaves like an inverseprobability-style correction:

1 τ|yi|

1

log πS(yi | x) = πS(yi | x)−

wistu ∝ exp −

τ|yi| . (16)

The length normalization is retained in implementation to avoid systematically favoring shorter or longer trajectories. At the population level, omitting length normalization only for notational clarity, the weighted gradient can be written as

1 τ|y|

πS(y | x)1−

wstu(y)∇θ log πθ(y | x) ∝

Ey∼π

· ∇θ log πθ(y | x). (17)

S

y∈Ωxc

This weighting partially offsets the sampling-frequency bias of on-policy trajectories by reducing the exponent of πS(y | x) from 1 to 1 − τ|1y|. A smaller τ strengthens this compensation, whereas a larger τ moves the weighting closer to the uniform case. Although rare paths are not guaranteed to be preserved, this weighting reduces the dominance of frequent reasoning modes over low-probability valid trajectories. This explains how SCOPE preserves reasoning diversity while still exploiting successful on-policy samples.

#### A.3 Group-Level Normalization and Stability

Finally, we analyze the stabilizing effect of group-level normalization. Different prompts can exhibit different perplexity scales due to variations in difficulty, length, and reasoning structure. Using raw perplexity-based weights may therefore make the update scale sensitive to prompt-specific statistics.

SCOPE avoids this issue by applying local softmax normalization within each trajectory group:

exp(Si/τ) j∈Ω exp(Sj/τ)

, (18)

wi =

where Si denotes the path-specific score. For correct trajectories, Si corresponds to the student-perplexity score; for incorrect trajectories, it corresponds to the inverse teacher-perplexity score. This normalization ensures

wi = 1, 0 < wi < 1. (19)

i∈Ω

For a given prompt x, the SCOPE gradient can be written as ∇θJSCOPE(x) =

wistu∇θLMLE(x,yi) +

i∈Ωxc

i∈Ωxw

By the triangle inequality,

witea∇θLOPD(x,yi). (20)

∥∇θJSCOPE(x)∥ ≤

wistu ∥∇θLMLE(x,yi)∥ +

i∈Ωxc

i∈Ωxw

witea ∥∇θLOPD(x,yi)∥. (21)

Table 3: Training config for GRPO, OPD, and SCOPE

Parameter GRPO OPD SCOPE Optimization Parameters

Learning Rate 5 × 10−5 5 × 10−5 5 × 10−5 LR Scheduler Constant Constant Constant Weight Decay 0.01 0.01 0.01 Train Batch Size (Prompts) 256 256 256 Mini Batch Size (Prompts) 256 256 256

RL & Generation Parameters

Max Prompt Length 4096 4096 4096 Max Completion Length 12288 12288 12288 Number of Generations (G) 8 1 8 Temperature 0.6 0.6 0.6 KL Coefficient (β) 0.0001 – – Clip Ratio (ϵ) 0.2 – – Update Epochs per Batch 1 1 1

Since the weights are normalized within each group, each branch forms a convex combination of trajectory-level gradients rather than an unbounded rescaling by raw perplexity values. In particular,

∥gi∥. (22)

wigi ≤

wi ∥gi∥ ≤ max

i∈Ω

i∈Ω

i∈Ω

Thus, group-level normalization controls the update scale while preserving the intended adaptive weighting effect. It does not make the estimator unbiased; instead, it introduces a signal-calibrated bias that emphasizes reliable teacher corrections on flawed trajectories and underexplored valid paths on correct trajectories.

### B Experimental Details

#### B.1 Implementation Setup

Training Infrastructure. All experiments were conducted on a high-performance distributed cluster using a total of 20 NVIDIA A100 (80GB) GPUs. Specifically, 16 GPUs (across two nodes) were allocated for training the student model, while the remaining 4 GPUs (on a single node) were dedicated to deploying the teacher model.

Hyperparameter Configuration. The detailed experimental settings for our study are presented in two parts. Table 3 outlines the specific training configurations, including optimization and reinforcement learning hyperparameters, for GRPO, OPD, and SCOPE. For the evaluation phase, we adopt a consistent set of generation parameters across all models, as detailed in Table 4. For Qwen3-1.7B-Base, due to severe repetition issues observed during evaluation, we increased the repetition penalty to mitigate this problem.

#### B.2 Evaluation Benchmarks

Mathematical Reasoning Experiment. To provide a comprehensive assessment of mathematical reasoning, we evaluate our method on a broad suite of benchmarks covering various problem sources, difficulty levels, and reasoning patterns. These benchmarks range from standard high-school competition problems to challenging olympiad-style tasks, with detailed characteristics in Table 5.

Code Generation Experiment. For code generation experiments, we utilize the TACO dataset (Li et al., 2023) as our primary training corpus, which comprises 25,202 programming problems sourced from diverse competitive programming platforms including Codeforces, AtCoder, Aizu Online Judge, and GeeksforGeeks. For evaluation, we employ the LiveCodeBench (from 2024.08 to 2025.02) and HumanEval benchmarks. Additionally, we sample 500 Code contests problems (Codeforces) from the TACO dataset as an evaluation set. To prevent data leakage, these sampled problems are explicitly excluded from our training corpus. A detailed description of the datasets is provided in Table 5.

Table 4: Evaluation parameters for all models.

Parameter Value Max Tokens 32768 Temperature 0.6 Top-p 0.95 Top-k 20 Repetition Penalty (Qwen3-1.7B) 1.08 Repetition Penalty (Distill-1.5B) 1.0 Samples per Prompt 32

Table 5: Overview of evaluation benchmarks and their key characteristics, covering mathematical reasoning and code generation tasks.

Benchmark Evaluation Focus & Characteristics

- AIME 2024 Evaluates precise arithmetic and algebraic reasoning under strict answer-format constraints, where solutions are typically given as integer-valued responses.
- AIME 2025 Serves as a held-out benchmark for assessing whether models can handle unseen and challenging mathematical reasoning patterns.

AMC 2023 Includes foundational competition-style problems in algebra, geometry, and combinatorics, serving as a

measure of core mathematical problem-solving ability.

MATH-500 Uses a representative subset of the MATH benchmark spanning multiple mathematical areas, requiring

symbolic manipulation, formula interpretation, and robust answer parsing.

Minerva Consists of STEM-oriented mathematical problems, emphasizing technical notation, quantitative

reasoning, and multi-step solution construction.

OlympiadBench Collects difficult olympiad-style problems, targeting rigorous logical reasoning, advanced mathematical

generalization, and complex problem-solving skills.

HumanEval Comprises 164 hand-crafted Python programming problems with unit-test-based evaluation, requiring

syntactically correct code generation and functional correctness against hidden test assertions.

Codeforces Sourced from competitive programming platform with 500 problems, featuring varying test coverage

(1–343 I/O pairs per problem) that demands robust algorithmic reasoning and edge-case handling.

LiveCodeBench Curated from recent competitive programming contests , containing 279 problems with consistent

multi-case evaluation, emphasizing solution reliability under diverse input conditions.

### C Supplementary Experimental Analysis

#### C.1 Preliminary Experiment

We sample 2,000 problems from the DeepMath dataset and generate 4 reasoning trajectories per problem using the student model (DeepSeek-R1-Distill-Qwen-1.5B) with temperature τ = 0.6, top-k = 20, top-p = 0.95, and a maximum response length of 32,768 tokens.

For each incorrect trajectory, we compute its perplexity score under the teacher model (SkyWork-OR1-7B) over the response tokens only (excluding the prompt), defined as:

|yw|

1 |yw|

log PT(yt |x)) (23)

PPL(yw |x) = exp(−

t=1

where yw represents the wrong samples. They are stratified into four equal-sized buckets (Q1–Q4) based on their PPL scores via quartile splitting. Table 6 reports the PPL and negative log-likelihood (NLL) statistics for each bucket.

For the prefix truncation experiment, each incorrect trajectory is truncated at the nearest newline boundary to the target truncation ratio r ∈ {0.2,0.4,0.6,0.8}, yielding a flawed prefix yprefix. The teacher is then prompted to complete the generation from yprefix using the completions API with temperature τ = 0.6. Each prefix is completed n = 4 times, and the recovery rate is computed as the mean accuracy over these completions. Table 7 presents the complete error recovery rates across all truncation ratios and PPL buckets.

#### C.2 Impact of Weight Temperature

To examine how the sharpness of the weight distribution affects optimization, we vary the temperature parameter τ among 0.5, 1.0, and 2.0. Figure 6 shows the effect of τ in the group-wise softmax normalization of DPAW (Eq. 4 and

Table 6: Teacher PPL and NLL statistics for each perplexity bucket over incorrect student trajectories.

Bucket PPL Mean NLL Mean

- Q1 (Lowest PPL) 1.361 0.305
- Q2 1.565 0.448
- Q3 1.710 0.536
- Q4 (Highest PPL) 2.383 0.688 Overall 1.755 0.494

- Table 7: Teacher error recovery rate (%) under different truncation ratios and PPL buckets. Each cell reports the mean accuracy over n = 4 completions per sample. The Q1–Q4 gap (rightmost column) quantifies the within-truncation-level spread attributable to PPL stratification.

Trunc. Ratio Q1 Q2 Q3 Q4 Q1–Q4 Gap

0.2 64.9 59.7 54.0 45.4 +19.4 0.4 55.8 53.2 50.1 40.2 +15.6 0.6 44.8 43.1 41.0 34.5 +10.3 0.8 35.8 35.3 32.6 28.6 +7.2

- = 0.5

| |
|---|

- = 1.0

- = 2.0

Avg@32 Pass@32

| |
|---|

| |
|---|

100

97.2

95.0

94.7

80.9 77.9

80

77.0 76.4

75.2 75.9

###### Scores(%)

60

50.9

49.1

48.6

42.7

41.0

40.4

40

30.4

29.9

28.9

20

0

AIME24 AIME25 AMC23

- Figure 6: Impact of the temperature hyperparameter τ on model performance across AIME24, AIME25, and AMC23. The results indicate that the default configuration of τ = 1.0 consistently yields the best performance across all benchmarks compared to τ = 0.5 and τ = 2.0.

Eq. 5). We adopt τ = 1.0 as the default setting. A smaller temperature overly sharpens the weight distribution, making the model concentrate on trajectories with extreme perplexity values. This aggressive weighting can amplify outlier noise instead of extracting reliable corrective signals, leading to unstable training. In contrast, a larger temperature flattens the distribution and weakens DPAW toward the uniform weighting used in standard OPD. As discussed in Section 2, uniform weighting lacks signal-quality awareness: it neither filters noisy teacher guidance on failed trajectories nor sufficiently emphasizes underexplored valid paths on successful ones. This can further induce the Pass@k paradox and diversity collapse. Overall, τ = 1.0 provides a balanced calibration of signal-quality variance, enabling effective noise filtering for incorrect paths while preserving reasoning diversity for correct ones.

#### C.3 Computational Cost

As illustrated in Table 8, we analyze the per-step wall-clock time of SCOPE against the two primary baselines, GRPO and OPD, to characterize the additional overhead introduced by our method. All experiments are conducted on the same configuration and timing statistics are collected over the stable training region.

While the rollout generation time is comparable to GRPO, our approach incurs an additional time overhead primarily due to teacher model queries. Notably, we use a naive synchronous training architecture where rollout and teacher logprob acquisition time do not overlap. By implementing an asynchronous strategy, the training efficiency is expected

- Table 8: Per-step wall-clock time breakdown (seconds) for each training method. Values are means over the stable training region. The number of rollouts is set to 8 for GRPO and SCOPE, and 1 for OPD.

Component GRPO OPD SCOPE (Ours) Generation 264.5 164.5 247.7 Old Logprob 34.0 5.2 31.4 Reward Computation 4.8 0.7 4.6 Actor Update 151.8 22.9 154.1 Teacher Scoring — 31.2 200.0 Total (step) 459.0 227.5 641.9

Table 9: Case 1: Numerical collapse.

2 i

Problem Let a1, a2, . . . , a100 be integers such that a

ai = 100. Find the maximum possible

value of a1. Ground Truth 550 Student Answer 100 Error Type Numerical collapse Key Excerpt ...10100m2 + 202md + 2d2 − 2000000m − 20000d+

9999...9 [thousands of repeated digits] × 6999× 9999...9 [continued overflow] That seems like I’m not doing this... The problem is just to be solved.

...\boxed{100}

to be comparable to that of GRPO. Furthermore, the computational overhead introduced by the weight calculation itself is minimal.

#### C.4 Case Study

We present representative incorrect student trajectories from the highest-PPL bucket to illustrate the qualitative nature of high perplexity errors. The cases in Tables 9–12 illustrate the Flawed Trajectory Trap across high-perplexity (PPL ≥ 1.80) errors, encompassing both structural collapse (e.g., numerical overflow in Case 1, infinite loops in Case 2) and logical hallucinations (e.g., flawed premises in Case 3, self-contradictions in Case 4). In all such instances, the severely degraded reasoning context disrupts the teacher model, flattening its predictive distribution and forcing it to output high-entropy, uninformative noise. Standard on-policy distillation on these samples disastrously compels the student to mimic this confusion. Our Dual-Path Adaptive Weighting (DPAW) explicitly circumvents this trap. By scaling distillation weights inversely with teacher perplexity, DPAW assigns near-zero weights to these toxic trajectories. This mechanism inherently filters out context-induced hallucinations, ensuring the student only learns from structurally coherent prefixes that elicit precise corrective signals.

Table 10: Case 2: Infinite reasoning loop.

Problem Determine whether the perimeter of the Mandelbrot set is finite or infinite. Ground Truth infinite Student Answer finite Error Type Repetitive reasoning loop; fails to commit to correct conclusion Key Excerpt Okay, so I have this question... I remember that the Mandelbrot set

is this intricate fractal... [paragraph repeated verbatim 4 times] But the question isn’t about whether a point is in the set, it’s about the perimeter. I know that fractals often have infinite perimeters, but I’m not entirely sure about the Mandelbrot set... [loop continues without resolution]

Table 11: Case 3: Incorrect application of a theorem.

Problem Is a real-valued function F on [a, b] with bounded variation and the intermediate

value property guaranteed to be continuous? Ground Truth Yes Student Answer No Error Type Incorrect application of Cantor function as counterexample Key Excerpt ...the Cantor function is constant on the middle-thirds Cantor set,

so it’s not continuous, but it’s still of bounded variation. [...] Therefore, it’s clear that such functions can be discontinuous. So, the answer is no, it’s not guaranteed to be continuous.

Note The Cantor function does not satisfy the intermediate value property, making this a flawed counterexample. The student fails to verify the premise.

Table 12: Case 4: Off-by-one arithmetic error in integral approximation.

√r−1 dx .

Problem Find the integer value of K = 80r=1 0 1 x

Ground Truth 16 Student Answer 17 Error Type Incorrect integral bounds in Euler–Maclaurin approximation Key Excerpt Wait, that seems contradictory because 16 is greater than 16.888.

That can’t be. Wait, maybe I misapplied the bounds. [...] But in the initial manual addition, I had the sum as about 16.888, but in reality, it’s about 17.28. Therefore, the answer is 17.

Note The student correctly computes ≈ 16.888 at one point, then contradicts this with an erroneous re-approximation, ultimately reporting ⌊17.28⌋ = 17 instead of ⌊16.888⌋ = 16.

