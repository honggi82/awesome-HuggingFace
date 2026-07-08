# arXiv:2605.26844v1[cs.LG]26May2026

## Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation

Yuanyi Wang1, Su Lu1, Yanggan Gu1, Pengkai Wang1, Yifan Yang1, Zhaoyi Yan2, Congkai Xie2, Jianmin Wu1, Hongxia Yang1,2,3* † 1The Hong Kong Polytechnic University, PolyU 2InfiX.ai 3Hong Kong Polytechnic University Daya Bay Technology and Innovation Research Institute

Code: https://github.com/wyy-code/TA-OPD

### Abstract

On-policy distillation (OPD) trains a student on its own rollouts with token-level teacher supervision. Recent selective OPD methods exploit the non-uniformity of OPD signals by prioritizing high-entropy or high-disagreement tokens. We revisit this principle and ask: which token-level teacher signals are actually learnable? Using a fixed-context diagnostic that measures same-context teacher–student KL reduction, we show that raw KL disagreement is a coarse proxy for learning value. It conflates learnable disagreement, where the teacher as-

signs corrective mass to the student’s top-K candidates, with incompatible disagreement, where the teacher places mass mostly off the student’s current support. We formalize this local compatibility as token teachability and show that it better predicts fixed-context improvement than raw KL alone. Motivated by this finding, we propose Teachability-Aware OPD (TA-OPD), a lightweight token-position selection method that applies OPD loss to highteachability positions without reward models or verifiers. Across Qwen2.5 and Qwen 3 teacher– student settings, TA-OPD often surpasses fulltoken OPD with only 5% retained tokens and improves over entropy- and divergence-based baselines. Our results reframe selective OPD as selecting learnable teacher signals rather than merely salient tokens.

### 1 Introduction

Knowledge distillation has long been used to transfer the behavior of a large teacher model to a smaller student model (Hinton, 2014; Zhou et al., 2026; Fang et al., 2026). For large language models (LLMs), a particularly effective variant is on-policy distillation (OPD), which trains a student policy on its own rollouts using token-level supervision from a teacher policy (Agarwal et al., 2024; Gu et al.,

*Corresponding author. †wangyuanyi713@gmail.com,

hongxia.yang@polyu.edu.hk

Same raw KL, different learnability

Prior work: Q3 = low-entropy with high-KL selector region

###### A

Within low-entropy with high-KL tokens, raw KL alone is ambiguous.

Learnable DL

Incompatible DI

teacher student

teacher student

teacher reweights local support

teacher points off support

Student local support

Student local support

TA-OPD keeps locally learnable supervision

###### B

Use support-aligned disagreement rather than raw salience alone.

###### 2 Top-K support stats

###### 3 Teachability gate

###### 4 Budgeted OPD

###### 1 On-Policy states

compute Dl & Ci

score learnable KL

keep top-p postitions

student rollout postitions

[Figure 1]

###### keep

[Figure 2]

support-aligned

[Figure 3]

[Figure 4]

###### filter

[Figure 5]

[Figure 6]

[Figure 7]

support-aligned positions rank higher

off-support

candidate positions

disagreement D

Figure 1: Token teachability. A: Low-entropy, highKL tokens can contain learnable disagreement DL, which stays within the student’s local support, and incompatible disagreement DI, which shifts off support. B: TA-OPD computes disagreement Dt and compatibility Ct, ranks positions by steacht = D˜tC˜t, and keeps only high-teachability OPD supervision.

2026c; Zhang et al., 2025). Compared with offpolicy distillation (Zhou et al., 2025; Wang et al., 2026e), OPD supervises the student on states that it actually visits, reducing distribution mismatch between training and deployment. However, this also exposes a central challenge: the teacher provides dense supervision at every token, but not every token-level teacher signal is equally useful.

Recent work shows that OPD supervision is highly non-uniform (Jin et al., 2026; Wang et al., 2026b). Selective OPD therefore allocates token budget to salient positions, such as high-entropy tokens or low-entropy tokens with large teacher– student disagreement (Guo et al., 2026; Xie et al., 2026; Xu et al., 2026). However, these criteria measure uncertainty or disagreement, not the learning effect of the induced token-level supervision. In on-policy training, two tokens with similar KL disagreement can lead to very different updates, depending on how the teacher distribution interacts with the student’s current predictive state (Fig 1.A). This raises a basic question: which token-level teacher signals in OPD are actually learnable?

To answer this question, we need to separate

token-level learning value from the rollout in which a token appears. End-to-end OPD performance entangles token supervision with sampling noise, downstream context shifts, and interactions across positions. We therefore introduce a fixed-context diagnostic: we collect student-generated prefixes, freeze them as a context bank, and rescore both the initial and trained students against the same teacher distribution. For each token, we measure the same-context reduction in teacher–student KL and ask which pre-update disagreement signals predict this local improvement. This diagnostic allows us to study whether a token-level teacher signal is aligned with useful learning, rather than merely salient under entropy or raw KL.

Our diagnostic shows that high KL disagreement is not a uniform learning signal. Even in the lowentropy, high-divergence regime targeted by prior selectors, raw disagreement mixes two qualitatively different cases: In learnable disagreement, the teacher assigns corrective mass to the student’s topK candidates, yielding high student-support coverage and better alignment with KL-disagreement reduction. In incompatible disagreement, teacher mass falls mostly outside the student’s current support, producing a large KL gap but weak local improvement. We refer to this local compatibility between teacher correction and student predictive support as token teachability. Fig 1.A illustrates this distinction.

Motivated by this diagnostic, we reformulate selective OPD as selecting learnable supervision rather than merely salient tokens. We instantiate this principle as Teachability-Aware OPD (TAOPD), which scores each response position by support-aligned teacher–student disagreement and applies the OPD loss only to high-teachability positions. TA-OPD uses teacher and student token probabilities available during OPD, optionally approximated with a lightweight top-K support, and requires no reward model or verifier (Fig 1.B).

Across Qwen3 and Qwen2.5 teacher–student settings varying in scale, reasoning ability, and backbone, TA-OPD shows that token quality can outweigh token count. With only 5% retained tokens, TA-OPD often matches or surpasses full-token OPD and improves over budget-matched entropyand divergence-based methods. These results support our central claim: OPD gains depend on teachable teacher signals, not merely dense supervision.

Our contributions are summarized as follows. (i) we introduce a fixed-context diagnostic that

measures same-context KL disagreement and links pre-update token signals to local OPD improvement. (ii) using this diagnostic, we formalize token teachability and show that raw KL disagreement conflates learnable and incompatible disagreement; only support-aligned disagreement reliably predicts useful learning. (iii) we propose Teachability-Aware OPD (TA-OPD), a lightweight token-position selection method that applies OPD loss to high-teachability positions without reward models or verifiers. (iv) we validate both the finding and the method across Qwen3 and Qwen2.5 settings, showing signficant performance with only 5% retained tokens.

### 2 Related Work

LLM distillation. Knowledge distillation transfers a stronger teacher model’s behavior to a smaller student by matching outputs, intermediate representations, or generated trajectories (Hinton, 2014; Zhou et al., 2026; Fang et al., 2026). For LLMs, distillation compresses reasoning, instruction-following, and alignment capabilities under limited compute (Zhou et al., 2025; Wang et al., 2026e). Offpolicy distillation trains on teacher- or humangenerated trajectories, which may diverge from student-visited states, while on-policy distillation supervises the student on its own rollouts with token-level teacher distributions (Agarwal et al., 2024; Gu et al., 2026c; Zhang et al., 2025). We focus on this OPD setting to examine which teacher signals are truly learnable.

Selective OPD. OPD provides dense token-level supervision, but recent work shows that useful signal is highly non-uniform (Yang et al., 2026). Entropybased criteria prioritize uncertain tokens (Jin et al., 2026; Guo et al., 2026), and divergence-based criteria highlight positions where the teacher strongly disagrees with the student (Xie et al., 2026; Wang et al., 2026b). TIP (Xu et al., 2026) formalizes this with a two-axis taxonomy of student entropy and teacher–student divergence, enabling tokenefficient training. Our work complements this line by assessing whether informative tokens are also learnable, showing that high divergence can mix support-aligned and off-support supervision.

Teacher–student compatibility. Distillation quality depends on both teacher strength and compatibility with the student policy. Recent OPD analyses (Li et al., 2026) suggest gains arise from alignment on high-probability student tokens and com-

patible reasoning patterns. Self-distillation (Zhao et al., 2026; Agarwal et al., 2024) and contextconditioned distillation (Ye et al., 2026; Zhang et al., 2026) further emphasize that effective supervision depends on information state and trajectory distribution. We operationalize compatibility at the token level by decomposing disagreement into learnable and incompatible components, reframing selective OPD as identifying signals that can be absorbed by the student.

### 3 Diagnosing Token Teachability

This section diagnoses which OPD tokens are actually learnable. Using TIP’s entropy–KL plane as a reference, we focus on Q3, the low-entropy with high-KL region of confident teacher–student disagreement (Xu et al., 2026). We show that Q3 is heterogeneous: raw KL mixes support-aligned corrections with off-support mismatch. We first define fixed-context token gain, then decompose disagreement by local support.

#### 3.1 Fixed-Context Token Gain

OPD trains on student-visited contexts ct = (x,y<t), the prefix generated by the current student. Raw KL at ct does not by itself reveal learning value, since end-to-end performance also depends on rollout noise and downstream context changes. We therefore freeze a bank of on-policy prefixes and rescore checkpoints on the same states.

For pi,tθ = pθ(· | ci,t), define the empirical token gain

− DKL pi,tT ∥pi,tθ

Gfixi,t = DKL pi,tT ∥pi,tθ

. (1)

τ

0

Positive Gfixi,t means the trained student is closer to the teacher on the same prefix. This diagnostic controls for rollout resampling variance and measures local KL reduction rather than answer-level success.

Proposition 1 (Gradient alignment) Let Lfix be differentiable and β-smooth. Let ℓt be a token OPD loss. For the update induced by ℓt, let Gt be the fixed-context loss reduction. Then

Gt = η ⟨∇θLfix(θ),∇θℓt(θ)⟩ + Rt, |Rt| ≤

(2)

βη2 2 ∥∇θℓt(θ)∥22.

Thus, useful tokens are not merely high-KL tokens; their gradients must align with fixed-context improvement. Appendices C and I give the diagnostic protocol and proof.

- 3.2 Local-Support Decomposition We now ask why high KL can fail to be learnable. At context ct, define the student’s local sup-

port as its top-K token set StS(K); these are the candidates the student currently considers plau-

sible. Let StT(K) be the teacher top-K set and Ut = StS(K) ∪ StT(K). We measure local disagreement on Ut:

Dt = DKL p ¯UTt ∥p¯Uθt , (3)

where p¯Ut is p(· | ct) renormalized on Ut. To measure whether the teacher correction is reachable, define compatibility mass

Ct =

v∈StS(K)

pT(v | ct). (4)

High Ct means the teacher mostly reweights tokens already in the student’s local support; low Ct means the teacher points off-support. After robust normalization to [0,1], we decompose disagreement as

DtL = Dt Ct, DtI = Dt(1 − Ct). (5)

DtL is learnable disagreement: the correction is large and locally reachable. DtI is incompatible disagreement: the correction is large but off-support. Fig 2A links this split to fixed-context gain, and Fig 2B–C show that Q3 region mixes both signals. Thus, teachability measures whether disagreement is locally absorbable, rather than whether the token is uncertain or high-KL. Appendix D gives details.

- 3.3 Learnable Disagreement Predicts Fixed-Context Gain

We next test whether this decomposition predicts Gfixi,t. For each K, we fit standardized token-level regressions with student entropy, local disagreement, token position, and teacher entropy as controls, and report prompt-cluster bootstrap intervals.

Fig 3A summarizes the regression evidence. Across K = 8,16,32, DL has roughly twice the coefficient of DI (0.086–0.087 vs. 0.043–0.045), with positive bootstrap gaps (+0.041–+0.044). Thus, raw KL is a coarse proxy: it mixes useful corrections with off-support mismatch, while learnable disagreement carries the stable gain signal. Full analysis are reported in Appendix F.

- 3.4 The Low-Entropy with High-Divergence Region is Not Uniformly Teachable

Following TIP, we use Q3 to denote the lowentropy with high-KL quadrant, where the student

B. Dl in TIP quadrants

C. Di in TIP quadrants

A. gain over decomposition

1.0

1.0

4

1.0

1.0

1.0

|[Figure 8]| | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 9]

[Figure 10]

[Figure 11]

Q3

[Figure 12]

[Figure 13]

Q3: low H high KL

high H high KL

forwardKLpercentile

forwardKLpercentile

0.8

0.8

0.8

0.8

0.8

2

percentileDi

asinhgain

0.6

0.6

0.6

0.6

0.6

pct.Dl

pct.Di

0

0.4

0.4

0.4

0.4

0.4

2

0.2

0.2

0.2

0.2

0.2

low H low KL

high H low KL

0.0

0.0

0.0

4

0.0

0.0

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

Dl percentile

student entropy percentile

student entropy percentile

- Figure 2: Local-support decomposition. A: fixed-context gain over learnable disagreement DL and incompatible disagreement DI. B–C: DL and DI projected onto TIP’s entropy–KL plane; Q3 denotes the low-entropy/high-KL region. Teachability separates support-aligned corrections from off-support mismatch within Q3.

| | | |2.0x<br><br>2.0x<br><br>DL D<br><br>|I|
|---|---|---|---|---|
| | | | | |

0.04 0.06 0.08 0.10

std. coefficient

K=8

K=16

K=32

1.9x

A DL explains gain

hi DL lo DL hi DI

−0.02

0.00

0.02

0.04

gain

+0.015

-0.010

+0.006

same Q3 budget

B Q3 is heterogeneous

matched

0.00 0.05 0.10 0.15

high-support gain gap

4B held-out

4B GSM8K

8B teacher

14B K=8

14B K=16

14B K=32

+0.078

+0.087

+0.118

+0.098

+0.084

+0.078

C Robust across contexts

matched

bootstrap

- Figure 3: Fixed-context evidence for token teachability. A: DL has about twice the standardized coefficient of DI. B: within Q3, high-DL tokens are beneficial, while low-DL and high-DI tokens are weak or harmful. C: high-support gain gaps remain positive across held-out contexts, GSM8K-COT, larger teachers, and support sizes.

is confident yet disagrees with the teacher. This region is a natural target for selective OPD, but our decomposition asks whether its disagreement is uniformly teachable.

Fig 3B shows that it is not. In a Q3-restricted intervention with matched budgets, high-DL tokens yield positive gain, whereas low-DL tokens are negative and high-DI tokens are weak. Thus, Q3 is informative but coarse: teachability separates learnable high-KL supervision from off-support mismatch. Fig 4 controls for token count. Q3+TA outperforms TIP, entropy, and random selection under exact top-N matching (A–B), the effect holds across support proxies (C), and support mass tracks gain better than raw KL or DI across buckets (D). Bucket trends further show that support mass tracks gain, while raw KL and incompatible disagreement decline. These controls indicate that Q3 needs a teachability filter, not merely a larger token budget.

#### 3.5 Robustness Across Contexts and Teachers

Finally, we test whether the separation depends on a single prompt shard or teacher. We repeat the fixed-context diagnostic on held-out math prompts,

GSM8K-COT prompts (Cobbe et al., 2021; Gao et al., 2024), and stronger 8B/14B teachers. Fig 3C and Table 1 show positive high-support gaps across contexts, teachers, and support sizes, with all main prompt-cluster intervals above zero. This supports our claim that teachability is a local property of OPD supervision rather than an artifact of one run. Appendix E reports the full exact-budget, bucket, and proxy checks.

### 4 Teachability-Aware OPD

The fixed-context diagnostic in Section 3 suggests selecting token positions by learnable disagreement rather than raw salience. We therefore use teachability as the main selection signal. Student entropy is kept separate, serving only as a baseline axis and an optional mixture to test complementarity.

#### 4.1 Budgeted OPD Objective

For a rollout batch, let I be all valid response positions after padding and trainer masks. At position t, let ct = (x,y<t) denote the on-policy context, and let pθ(· | ct) and pT(· | ct) be the student and teacher next-token distributions over vocabulary V .

###### A. exact top-N

###### B. efficiency

###### C. support proxy

###### D. bucket trend

Q3+C Q3+TA Q3+KL

K=16 K=32

| |
|---|

0.12

0.1

C mass

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Q3gaingap

gainshift

TA-OPD KL DI

0.0

0.06

TIP Ent. Rand.

−0.1

DI KL

● 3%, ■ 5%

0.00

−0.1 0.0 0.1

−4 0 4

0 1 2 3

overlap mass CBC

selected-token gain

gain / keep

Q3 bucket

- Figure 4: Q3 controls. A–B: exact top-N comparisons under matched token counts; B reports gain per kept token. C: support proxies yield positive high–low gain gaps inside Q3 at K = 16,32. D: bucket trends (thin: individual diagnostics; bold: mean) show support mass tracks gain, whereas raw KL and DI decline.

Table 1: Context and teacher-scale robustness. The gap compares high- vs. low-support tokens inside Q3, the low-entropy/high-KL diagnostic region. Bootstrap intervals are prompt-clustered; matched gap denotes the prompt-matched estimate. All main 300-context diagnostics have positive intervals.

Setting Context / support Contexts Tokens Bootstrap gap 95% interval Matched gap Qwen3-4B→1.7B held-out math, K = 16 300 57,600 +0.078 [+0.060, +0.095] +0.016 Qwen3-4B→1.7B GSM8K-COT, K = 16 300 57,600 +0.087 [+0.066, +0.109] +0.037 Qwen3-8B→1.7B held-out math, K = 16 300 57,600 +0.118 [+0.094, +0.144] +0.042 Qwen3-14B→1.7B held-out math, K = 8 300 57,600 +0.098 [+0.080, +0.117] +0.071 Qwen3-14B→1.7B held-out math, K = 16 300 57,600 +0.084 [+0.069, +0.098] +0.058 Qwen3-14B→1.7B held-out math, K = 32 300 57,600 +0.078 [+0.062, +0.095] +0.046

Full reverse-KL OPD uses ℓOPDt = DKL(pθ(· | ct)∥pT(· | ct))

pθ(v | ct) log pθ(v | ct) − log pT(v | ct) .

=

v∈V

(6) Sampled-token OPD may instead use ℓOPDt = log pθ(yt | ct) − log pT(yt | ct); our selector is agnostic to this estimator. Given a binary mask mt ∈ {0,1}, budgeted OPD optimizes

1 t∈I mt t∈I

mt ℓOPDt (θ), (7)

Lm(θ) =

with ℓOPDt substituted when used by the trainer. This reverse-KL loss is the training objective; the fixed-context KL reduction in Section 3 is used only for diagnosis. Token budgets refer to KLsupervised response positions, not proportional wall-clock savings.

#### 4.2 Teachability Score

For each position, define the student and teacher top-K sets

- StS(K) = TopK pθ(· | ct),K ,
- StT(K) = TopK pT(· | ct),K ,

(8)

and let Ut = StS(K) ∪ StT(K). We measure local teacher–student disagreement on Ut:

Dt = DKL p ¯UTt ∥p¯Uθt , p¯Ut(v | ct) =

(9)

p(v | ct) u∈Ut p(u | ct)

, v ∈ Ut.

The forward direction emphasizes teacherpreferred candidates underweighted by the student. We define student-support coverage as the teacher probability mass on the student’s top-K support:

pT(v | ct). (10)

Ct =

v∈StS(K)

High Ct indicates support-aligned correction; low Ct indicates off-support disagreement.

We put all token-level scores on a common scale using batch-wise robust normalization. Let IB be the valid response positions in rollout batch B. For a raw score zt at position t, such as Dt or Ct, let zB = {zj : j ∈ IB}. We define

NormB(zt) = clip

zt − Q0.05(zB) Q0.95(zB) − Q0.05(zB) + ϵ

,0,1 ,

(11)

where Qq is the q-quantile and ϵ > 0 prevents division by zero. We apply this operator to disagreement and compatibility:

Dt = NormB(Dt), Ct = NormB(Ct). The teachability score is

steacht = DtL = Dt Ct. (12)

The complementary diagnostic term is DtI = Dt(1− Ct), which captures large but locally incompatible disagreement. Thus, teachability captures locally learnable disagreement, whereas entropy captures uncertainty.

#### 4.3 Token Selection and Baselines

Given retention ratio ρ, let n = ⌈ρ|I|⌉. TA-OPD keeps the top-n valid positions by teachability:

Tρteach = Topn {steacht : t ∈ I} , mt = 1[t ∈ Tρteach].

(13)

To isolate teachability from uncertainty and raw disagreement, we compare against budget-matched selectors:

sHt = Ht, sDt = Dt, (14) sTIPt = Ht + Dt − Ht Dt, sCt = Ct, (15)

where Ht = H(pθ(· | ct)). We also report an optional entropy–teachability mixture,

sH+teacht = Ht + steacht − Htsteacht , (16) to test complementarity with the TIP entropy axis. Full OPD sets mt = 1 for all t ∈ I, and random selection samples n valid positions uniformly.

In practice, Dt and Ct are computed from teacher and student top-K log-probabilities. If teacher scores on StS(K) are available, Eq. 10 is exact; otherwise we use

pT(v | ct), (17)

Ct =

v∈StS(K)∩StT (K)

a lower bound on Ct. Unless stated otherwise, K = 16. TA-OPD requires no reward model, verifier, or additional labels.

Fixed-context selector check. Before downstream evaluation, we verify that the diagnostic score can serve as a training mask. Table 2 compares budgeted selectors on the same fixed-context bank. TA-OPD gives the largest KL reduction and gain per kept token at both 3% and 5% budgets, supporting teachability as an OPD allocation rule rather than only a post-hoc diagnostic.

Table 2: Fixed-context selector check. G3% and G5% are same-context KL reductions at nominal budgets; G/K normalizes by actual retained ratio (Keep). TAOPD gives the largest reduction and efficiency. C is teacher mass on student support, and Q3 is TIP’s lowentropy/high-divergence region.

labeltab:selector-intervention

Selector Keep G3% G/K3% G5% G/K5% Q3 frac. TA-OPD 3.4 0.047 1.40 0.049 0.95 4.4 Raw KL 3.4 0.033 0.97 0.036 0.69 2.7 Incompat. DI 3.5 0.025 0.72 – – 1.4 TIP 3.3 0.023 0.69 0.026 0.50 1.6 Random 3.3 0.021 0.65 0.021 0.41 3.3 TA-OPD+Ent. 3.3 0.021 0.63 0.016 0.31 1.0 Entropy 3.3 0.018 0.53 0.009 0.18 0.3 Q3+high-C 3.2 0.016 0.51 0.009 0.23 94.6 Q3+low-C 3.2 0.010 0.32 – – 94.7 Q3-only 3.2 0.010 0.31 0.002 0.05 91.7

### 5 Experiments

5.1 Setup Models. We evaluate four teacher–student pairs:

- (1) Qwen3-4B to Qwen3-1.7B
- (2) Qwen3-8B-GRPO to Qwen3-4B
- (3) Qwen3-14B to Qwen3-4B
- (4) DeepSeek-R1-Distill-Qwen-14B (Guo et al.,

- 2025) to Qwen2.5-3B (Qwen: et al., 2025). The Qwen3-8B teacher is GRPO-tuned (Shao et al.,

- 2024), so the settings vary scale, capability, and backbone architecture. Training data and benchmarks. Training prompts are sampled from DAPO (Yu et al.,

2026). We evaluate math, coding, factual QA, and instruction following on AIME24 (Zhang and Math-AI, 2024), AIME25 (Zhang and Math-AI,

- 2025), GPQA-Diamond (Rein et al., 2023), HumanEval (Chen et al., 2021), IFEval (Zhou et al., 2023), and MATH-500 (Hendrycks et al., 2021), based on the EvalScope (Team, 2024). Each result reports the mean and standard deviation over five evaluation seeds from the same trained checkpoint. Baselines and budgets. We compare full OPD with budgeted token selectors: entropy-only (Jin et al., 2026), TIP-style entropy+divergence (Xu et al., 2026), Teachability-Aware OPD (TA-OPD), and TA-OPD+Entropy. Unless otherwise stated, support statistics use K = 16, and budgets refer to KL-supervised response-token positions rather than wall-clock savings. Implementation. All runs use the same sampledtoken OPD pipeline as the fixed-context analysis and are trained on 64 NVIDIA H800 GPUs.

Table 3: Main benchmark results at a 10% supervised-token budget. We evaluate Qwen3 teacher–student pairs and a cross-backbone DeepSeek-R1-Distill-Qwen-14B to Qwen2.5-3B pair. Entries report mean±std over five evaluation seeds; Avg. is the average score over six benchmarks. TA-OPD denotes Teachability-Aware OPD, and TA-OPD+Ent. adds entropy.

Setting Method Avg. AIME24 AIME25 GPQA-D. HumanEval IFEval MATH-500

Base 40.77 10.00±3.33 10.00±3.33 31.56±0.76 60.97±1.83 59.52±1.29 72.60±1.00 Pure OPD 42.37 11.67±5.00 15.00±1.67 30.55±0.25 59.15±1.22 62.85±0.74 75.00±1.20 Entropy 41.46 13.34±3.34 8.33±5.00 32.58±1.77 59.45±0.30 60.63±0.01 74.40±1.00 TIP 43.05 18.34±1.66 11.67±5.00 29.29±2.02 62.50±0.30 60.91±0.27 75.60±1.00 TA-OPD 44.89 20.00±0.01 15.00±5.00 34.59±2.27 60.98±0.01 64.05±1.02 74.70±0.50 TA-OPD+Ent. 42.32 15.00±1.67 8.34±1.67 32.58±1.77 61.59±0.61 61.73±0.92 74.70±0.10

Qwen3-4B

→ Qwen3-1.7B

Base 52.98 21.66±1.66 13.33±0.01 47.97±0.51 77.75±0.30 73.57±2.22 83.60±0.20 Pure OPD 53.71 20.00±3.33 21.67±1.67 47.22±1.26 78.36±0.92 70.33±1.20 84.70±1.30 Entropy 56.69 26.33±3.10 23.33±3.30 47.47±1.01 77.65±1.21 79.85±0.37 85.50±0.10 TIP 56.81 26.33±3.60 23.67±3.33 47.47±0.49 80.49±0.61 76.71±1.11 86.20±0.30 TA-OPD 56.87 30.00±1.40 21.31±1.32 47.47±0.51 79.88±0.60 75.13±5.30 87.40±0.80 TA-OPD+Ent. 56.52 30.00±3.33 25.00±1.67 45.70±2.78 78.05±1.22 73.38±2.22 87.00±0.40

Qwen3-8B (GRPO)→ Qwen3-4B

Base 52.98 21.66±1.66 13.33±0.01 47.97±0.51 77.75±0.30 73.57±2.22 83.60±0.20 Pure OPD 54.64 23.34±3.34 21.66±1.66 42.17±0.76 78.05±0.61 79.11±0.74 83.50±0.10 Entropy 53.39 21.67±5.00 15.00±1.67 46.46±0.01 79.27±0.61 73.94±4.06 84.00±0.80 TIP 53.62 20.00±0.01 16.66±3.33 47.98±2.53 78.05±0.61 76.62±1.57 82.40±0.01 TA-OPD 54.65 23.34±3.34 18.34±1.66 45.20±1.77 79.57±0.92 78.93±0.01 82.50±1.70 TA-OPD+Ent. 54.10 21.67±5.00 21.67±1.67 42.17±0.25 78.35±1.53 77.54±1.02 83.20±0.60

Qwen3-14B

→ Qwen3-4B

Base 29.98 3.34±3.33 1.67±1.66 27.81±0.25 64.63±1.22 26.16±0.27 56.31±0.20 Pure OPD 28.76 5.00±1.67 1.67±1.66 27.78±1.51 62.20±1.21 23.48±0.18 52.50±0.90 Entropy 19.94 1.67±1.66 1.67±1.66 23.98±1.02 54.88±3.66 24.68±0.83 12.83±1.40 TIP 28.11 3.33±0.01 1.67±1.66 27.55±0.01 58.54±0.60 26.16±0.28 51.40±0.50 TA-OPD 30.62 5.00±1.66 1.67±1.66 29.59±0.51 66.77±0.31 25.42±1.20 55.31±0.60 TA-OPD+Ent. 30.28 8.34±1.60 3.33±0.01 31.63±0.51 62.81±1.22 23.39±1.02 52.20±0.70

DeepSeek-R1 Distill-Qwen-14B

→ Qwen2.5-3B

#### 5.2 Main Results

- Table 3 compares all methods at a 10% supervisedtoken budget. Across the four teacher–student settings, TA-OPD obtains the best average score in every group: 44.89 for Qwen3-4B to Qwen31.7B, 56.87 for Qwen3-8B-GRPO to Qwen3-4B, 54.65 for Qwen3-14B to Qwen3-4B, and 30.62 for the cross-backbone DeepSeek-R1-Distill-Qwen to Qwen2.5 setting. This supports the main claim that low-budget OPD can preserve useful supervision when the retained tokens are teachable.

The gains are strongest when the student is smaller or the teacher–student pair is mismatched. For Qwen3-4B to Qwen3-1.7B, TA-OPD improves the average over Full OPD from 42.37 to 44.89 and leads on AIME24, GPQA-Diamond, and IFEval. For the cross-backbone setting, TA-OPD improves over both Base and Full OPD in average score, while TA-OPD+Entropy gives the best AIME24, AIME25, and GPQA-Diamond scores. These results suggest that teachability is especially useful when dense teacher supervision contains more incompatible or noisy token signals.

For stronger Qwen3 teachers with a 4B student, the results have similar trend. With the GRPOtuned 8B teacher, TA-OPD has the best average and leads on AIME24 and MATH-500, while TIP or entropy remain competitive on HumanEval and IFEval. With the 14B teacher, TA-OPD nearly matches Full OPD in average score and leads HumanEval, while TA-OPD+Entropy is strongest on AIME25. Thus, teachability is not a universal perbenchmark booster; it is a token-allocation rule that improves the quality of supervised positions under a constrained OPD budget.

#### 5.3 Budget Sensitivity

Table 4 sweeps 5%, 10%, 30%, and 50% budgets for the two Qwen3-4B student settings. The results are not monotonic in budget, indicating that more KL-supervised tokens do not always yield better downstream performance.

For Qwen3-8B-GRPO to Qwen3-4B, low budgets are already competitive: TA-OPD+Entropy reaches the best 5% average score (57.89), while TA-OPD gives the best MATH-500 score at 5%

- Table 4: Budget sweep for the two Qwen3-4B student settings. Benchmark entries report mean±std; Avg. is the arithmetic mean of the six benchmark means, excluding the ± terms. TA-OPD denotes Teachability-Aware OPD; TA-OPD+Ent. adds the entropy axis.

Setting Budget Selector Avg. AIME24 AIME25 GPQA-D. HumanEval IFEval MATH-500

Entropy 56.95 31.66±1.66 26.66±0.01 47.47±0.15 76.83±1.22 73.47±0.64 85.60±0.20 TIP 56.20 21.65±8.35 24.98±1.68 48.23±0.25 78.66±0.01 76.80±0.28 86.90±0.50 TA-OPD 57.35 31.66±1.67 26.67±0.01 46.47±0.51 78.97±0.92 72.46±0.37 87.90±1.10 TA-OPD+Ent. 57.89 33.34±6.67 28.33±1.67 47.22±2.28 79.58±0.92 72.37±0.28 86.50±0.30

5%

Entropy 56.69 26.33±3.10 23.33±3.30 47.47±1.01 77.65±1.21 79.85±0.37 85.50±0.10 TIP 56.81 26.33±3.60 23.67±3.33 47.47±0.49 80.49±0.61 76.71±1.11 86.20±0.30 TA-OPD 56.87 30.00±1.40 21.31±1.32 47.47±0.51 79.88±0.60 75.13±5.30 87.40±0.80

10%

Qwen3-8B (GRPO)→ Qwen3-4B

- TA-OPD+Ent. 56.52 30.00±3.33 25.00±1.67 45.70±2.78 78.05±1.22 73.38±2.22 87.00±0.40

30%

Entropy 57.56 28.33±5.00 25.00±5.00 48.49±1.52 79.58±0.31 77.45±3.33 86.50±0.10 TIP 58.46 31.67±5.00 28.33±1.66 45.70±0.25 79.88±0.01 78.56±1.85 86.60±1.00 TA-OPD 57.28 28.34±1.67 28.33±1.67 49.19±0.30 77.75±0.31 74.96±3.24 85.10±0.50

- TA-OPD+Ent. 57.46 25.00±1.67 28.34±1.67 47.98±3.03 78.97±0.92 78.75±0.93 85.70±0.50

Entropy 53.90 20.00±3.33 20.00±3.33 43.68±1.76 78.35±0.91 75.69±4.34 85.70±1.50 TIP 55.74 23.33±3.33 24.99±1.66 44.69±0.75 78.96±0.91 76.15±2.40 86.33±0.33 TA-OPD 57.90 31.67±4.34 23.34±0.01 47.73±0.76 78.97±0.92 79.67±1.11 86.00±1.80 TA-OPD+Ent. 57.25 23.33±0.01 30.00±3.33 46.97±1.01 78.05±0.61 78.93±0.19 86.20±0.40

50%

Entropy 53.56 23.34±3.34 16.67±0.01 43.94±2.02 77.75±0.92 75.97±5.36 83.70±0.30 TIP 53.83 23.33±0.01 16.67±0.01 44.70±1.77 79.88±0.61 75.50±2.49 82.90±0.70 TA-OPD 53.58 18.34±1.66 18.34±1.66 45.70±1.27 77.75±0.30 79.66±0.93 81.70±1.30 TA-OPD+Ent. 54.47 21.66±1.66 16.67±0.01 48.23±2.27 77.75±0.91 79.02±0.65 83.50±0.70

5%

Entropy 53.39 21.67±5.00 15.00±1.67 46.46±0.01 79.27±0.61 73.94±4.06 84.00±0.80 TIP 53.62 20.00±0.01 16.66±3.33 47.98±2.53 78.05±0.61 76.62±1.57 82.40±0.01 TA-OPD 54.65 23.34±3.34 18.34±1.66 45.20±1.77 79.57±0.92 78.93±0.01 82.50±1.70 TA-OPD+Ent. 54.10 21.67±5.00 21.67±1.67 42.17±0.25 78.35±1.53 77.54±1.02 83.20±0.60

10%

Qwen3-14B

→ Qwen3-4B

Entropy 53.62 20.00±3.33 21.66±1.66 46.97±0.01 78.97±0.92 70.52±0.46 83.60±0.20 TIP 53.93 26.67±0.01 21.66±5.00 43.18±1.26 79.57±0.30 69.97±0.09 82.50±0.50 TA-OPD 52.81 21.66±8.33 15.00±1.67 43.18±0.76 79.57±0.30 74.77±3.42 82.70±0.10 TA-OPD+Ent. 53.15 20.00±3.33 18.34±1.66 45.70±2.77 79.88±0.01 72.45±1.30 82.50±0.10

30%

Entropy 52.76 18.34±1.66 18.33±1.67 44.95±0.51 78.66±0.01 73.38±2.59 82.90±0.50 TIP 52.12 21.66±1.66 13.33±0.01 45.95±1.52 78.05±0.01 70.34±0.27 83.40±1.40 TA-OPD 52.16 13.33±0.01 18.34±1.66 45.20±0.25 77.75±0.31 73.75±3.51 84.60±0.20 TA-OPD+Ent. 52.95 18.34±1.66 16.66±3.34 44.19±2.78 78.97±0.91 76.43±1.94 83.10±0.10

50%

and the best average among teachability-based selectors at 50% (57.90). TIP performs best at 30% average, showing that uncertainty and divergence can still help on some benchmark mixtures. However, the strong 5% results indicate that much of the useful OPD signal is concentrated in a small set of teachable positions.

For Qwen3-14B to Qwen3-4B, the best average appears at 10% with TA-OPD (54.65), while TAOPD+Entropy is strongest at 5% (54.47). Increasing the budget to 30% or 50% does not consistently improve the average, and several metrics regress relative to the 5–10% settings. This is consistent with the fixed-context analysis: OPD is token quality limited rather than purely budget-limited, and the optimal ratio depends on the teacher, student, and benchmark mix. Appendix G provides fixedcontext budget curves, macro-average budget summaries, and early downstream boundary checks.

Remark 1: Appendix G reports the ablation study.

### 6 Conclusion

We studied which token-level teacher signals are actually learnable in on-policy distillation. Using a fixed-context diagnostic, we showed that raw KL disagreement is a coarse proxy for learning value: it conflates learnable disagreement, where teacher mass has high coverage over the student’s top-K support, with incompatible disagreement, where teacher mass falls mostly off support. We formalized this support-aware learning value as token teachability and used it to design TeachabilityAware OPD, a lightweight token-position selection method without reward model or verifier. Across Qwen3 and Qwen2.5 teacher–student settings, TA-OPD often surpasses full-token OPD and improves over entropy- and divergence-based selectors. These results suggest that selective OPD should prioritize locally learnable teacher signals rather than merely salient tokens, reframing OPD as a compatibility-aware supervision problem.

### 7 Limitations

Our analysis focuses on OPD for math-heavy reasoning prompts and Qwen-family teacher–student pairs, with one cross-backbone distillation setting. Although the fixed-context diagnostic is modelagnostic, broader coverage over multilingual data, dialogue tasks, code-specialized teachers, and nonQwen backbones would further test the generality of token teachability. TA-OPD also selects supervised token positions rather than pruning transformer computation, so the reported token budget should be interpreted as a supervision budget, not a proportional wall-clock speedup. Finally, our diagnostic measures same-context KL reduction; it is designed to explain local learning signals and should be paired with downstream evaluation when making deployment claims.

### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246–21263.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Luyang Fang, Xiaowei Yu, Jiazhang Cai, Yongkai Chen, Shushan Wu, Zhengliang Liu, Zhenyuan Yang, Haoran Lu, Xilin Gong, Yufang Liu, and 1 others. 2026. Knowledge distillation and dataset distillation of large language models: Emerging trends, challenges, and future directions. Artificial Intelligence Review, 59(1):17.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The language model evaluation harness.

Yanggan Gu, Shuo Cai, Zihao Wang, Wenjun Wang, Yuanyi Wang, Pengkai Wang, Sirui Huang, Su Lu,

Jianmin Wu, and Hongxia Yang. 2026a. Featcal: Feature calibration for post-merging models. arXiv preprint arXiv:2605.13030.

Yanggan Gu, Yuanyi Wang, Zhaoyi Yan, Yiming Zhang, Qi Zhou, Fei Wu, and Hongxia Yang. 2026b. Infifpo: Implicit model fusion via preference optimization in large language models. Advances in Neural Information Processing Systems, 38:15645–15672.

Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2026c. Miniplm: Knowledge distillation for pre-training language models. In The Thirteenth International Conference on Learning Representations.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Yiju Guo, Wenkai Yang, Zexu Sun, Ning Ding, Zhiyuan Liu, and Yankai Lin. 2026. Learning to focus: Causal attention distillation via gradient-guided token pruning. Advances in Neural Information Processing Systems, 38:24921–24948.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. In Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

G Hinton. 2014. Distilling the knowledge in a neural network. In Deep Learning and Representation Learning Workshop in Conjunction with NIPS.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. 2026. Entropy-aware on-policy distillation of language models. In The 1st Workshop on Scaling Posttraining for LLMs.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, and 1 others. 2026. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016.

Qwen:, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, and 24 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

ModelScope Team. 2024. EvalScope: Evaluation framework for large models.

Pengkai Wang, Pengwei Liu, Yuanyi Wang, Guanyu Chen, Xingyu Ren, Xiaolong Li, Zhongkai Hao, Yuting Kong, Qixin Zhang, and Dong Ni. 2026a. Discovering physical directions in weight space: Composing neural pde experts. arXiv preprint arXiv:2605.14546.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, and 1 others. 2026b. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. Advances in Neural Information Processing Systems, 38:115452–115486.

Wenjun Wang, Yanggan Gu, Shuo Cai, Yuanyi Wang, Pengkai Wang, Jianmin Wu, and Hongxia Yang. 2026c. E-pmq: Expert-guided post-merge quantization with merged-weight anchoring. arXiv preprint arXiv:2605.16882.

Yuanyi Wang, Yanggan Gu, Zihao Wang, Kunxi Li, Yifan Yang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, and Hongxia Yang. 2026d. Mergepipe: A budgetaware parameter management system for scalable llm merging. arXiv preprint arXiv:2602.13273.

Yuanyi Wang, Yanggan Gu, Yiming Zhang, Qi Zhou, Zhaoyi Yan, Congkai Xie, Xinyao Wang, Jianbo Yuan, and Hongxia Yang. 2025. Model merging scaling laws in large language models. arXiv preprint arXiv:2509.24244.

Yuanyi Wang, Zhaoyi Yan, Yiming Zhang, Qi Zhou, Yanggan Gu, Fei Wu, and Hongxia Yang. 2026e. Infigfusion: Graph-on-logits distillation via efficient gromov-wasserstein for model fusion. Advances in Neural Information Processing Systems, 38:119677– 119713.

Yuanyi Wang, Yifan Yang, Su Lu, Yanggan Gu, Pengkai Wang, Wenjun Wang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, Jialun Cao, and 1 others. 2026f. Geometry conflict: Explaining and controlling forgetting in llm continual post-training. arXiv preprint arXiv:2605.09608.

Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, and Jiangning Zhang. 2026. Llm-oriented token-adaptive knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 34070– 34078.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. 2026. Tip: Token importance in on-policy distillation. arXiv preprint arXiv:2604.14084.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. 2026. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. 2026. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2026. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244.

Songming Zhang, Xue Zhang, Tong Zhang, Bojie Hu, Yufeng Chen, and Jinan Xu. 2025. Aligndistil: Token-level language model alignment as adaptive policy distillation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19791– 19807.

Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, and Jingnan Gu. 2026. Opsdl: On-policy self-distillation for long-context language models. arXiv preprint arXiv:2604.17535.

- Yifan Zhang and Team Math-AI. 2024. American invi-

- tational mathematics examination (aime) 2024.

Yifan Zhang and Team Math-AI. 2025. American invi-

- tational mathematics examination (aime) 2025.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover.

- 2026. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv preprint arXiv:2601.18734.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Qi Zhou, Yiming Zhang, Yanggan Gu, Yuanyi Wang, Zhijie Sang, Zhaoyi Yan, Zhen Li, Shengyu Zhang, Fei Wu, and Hongxia Yang. 2025. Democratizing ai through model fusion: A comprehensive review and future directions. Nexus.

Qi Zhou, Yiming Zhang, Yanggan Gu, Yuanyi Wang, Zhaoyi Yan, Zhen Li, Chi Yung Chung, and Hongxia Yang. 2026. Model fusion for scalable and sustainable artificial intelligence: A review and outlook. Journal of Modern Power Systems and Clean Energy, 14(1):37–49.

### A Use of AI Assistants.

We used large language model assistants for language polishing, grammar checking, and formatting refinement during paper preparation. All technical ideas, method design, experiments, analyses,

claims, and final writing were developed, verified, and approved by the authors. The authors take full responsibility for the content of the submission.

### B Ethical Considerations

TA-OPD is a training-time method for filtering teacher supervision. It can reduce exposure to noisy or incompatible teacher signals, but it does not provide guarantees about factuality, safety, bias, or harmful content in the distilled model. If the teacher produces unsafe behavior, selective distillation may still transfer parts of that behavior when those tokens appear locally learnable to the student. Practical use should therefore combine TA-OPD with standard data filtering, safety evaluation, and task-specific deployment checks. Our experiments use public benchmark-style data and do not rely on private user data.

### C Fixed-Context Diagnostic Protocol

Context bank. For each prompt, we sample student rollouts and store valid response prefixes ci,t = (xi,yi,<t). All checkpoints are scored on this same bank, so diagnostic differences cannot come from resampling different rollouts.

Support metrics. For support size K, StS(K) and StT(K) denote the student and teacher top-K token sets. We compute disagreement on StS(K) ∪ StT(K), and compatibility Ct(K) as teacher mass on StS(K). Scalars are robustly normalized within a batch by clipping the 5–95 percentile range to [0,1].

Fixed-context gain. When full-vocabulary probabilities are available, we compute Eq. 1. Otherwise, top-K support diagnostics are used only for robustness checks. Because token gains are heavytailed and correlated within a rollout, all confidence intervals use prompt-cluster bootstrap.

Gradient view. Proposition 1 gives the local optimization view used in Section 3.1. The proof is included in Appendix I.

### D Local-Support Decomposition

Top-K supports. For each fixed context ct, we store the student and teacher top-K sets StS(K) and StT(K). The union Ut = StS(K) ∪ StT(K) is used for local disagreement. For v ∈ Ut, the

restricted distribution is

p(v | ct) u∈Ut p(u | ct)

p¯Ut(v | ct) =

.

This keeps the diagnostic focused on locally salient alternatives rather than full-vocabulary tail noise.

Normalization. All scalar token scores are normalized within a rollout batch:

zt − Q0.05(zB) Q0.95(zB) − Q0.05(zB) + ϵ

zt = clip

,0,1 ,

where Qq is the q-quantile. We use the same normalization for entropy, disagreement, and compatibility-derived scores.

Support proxies. The main compatibility mass is Ct = v∈SS

t (K) pT(v | ct). We also audit simpler proxies: top-K overlap fraction, top-K Jaccard, shared teacher top-K mass, and a contrastive binary compatibility score (CBC). These proxies ask the same question: how much of the teacher correction remains near the student’s local support?

### E Low-Entropy with High-Divergence Heterogeneity Diagnostics

Live quadrant intervention. We train three masks restricted to the low-entropy with highdivergence (Q3) region: high DL, low DL, and high DI. All keep roughly the same token ratio. The high-DL mask is the only clearly positive intervention; low-DL is negative, and high-DI is weak.

Exact-budget fixed-context controls. Fig 4 summarizes the main Q3 control evidence. Here we report the full exact top-N table behind Panels A– B. Every selector keeps the same number of target tokens; support-aligned selectors remain positive, while random selection is consistently negative. Appendix Fig 5C shows the bucket-level gain shapes used as a nonlinear sanity check.

Support-definition audit. The quadrant result is stable across support proxies. Teacher mass on student support, top-K overlap, Jaccard overlap, and CBC all produce positive high–low support gaps inside the low-entropy with high-divergence region. Appendix Fig 5A gives a compact heatmap of the robustness gaps in Table 1. Appendix Fig 5B visualizes the support-proxy audit.

Table 5: Section 3 diagnostic datasets. The short diagnostic run is retained only as a boundary check.

Dataset Contexts Tokens Use 4B → 1.7B held-out 300 300 57,600 main 4B → 1.7B GSM8K-COT 300 300 57,600 main 8B → 1.7B 300 300 57,600 main 14B → 1.7B 300 K=8 300 57,600 main 14B → 1.7B 300 300 57,600 main 14B → 1.7B 300 K=32 300 57,600 main 8B → 1.7B short diagnostic 8 1,024 boundary only

- Table 6: Support proxy audit inside the low-entropy with high-divergence (Q3) region. Different support definitions produce positive high–low gain gaps.

Setting Proxy Tokens Gap 95% interval 14B → 1.7B, K=16 Ct mass 3342 +0.086 [+0.064, +0.104] 14B → 1.7B, K=16 CBC 3342 +0.105 [+0.078, +0.125] 14B → 1.7B, K=16 Jaccard 3342 +0.038 [+0.020, +0.055] 14B → 1.7B, K=16 overlap 3342 +0.038 [+0.020, +0.055] 14B → 1.7B, K=32 Ct mass 3342 +0.090 [+0.074, +0.106] 14B → 1.7B, K=32 CBC 3342 +0.105 [+0.078, +0.125] 14B → 1.7B, K=32 Jaccard 3342 +0.052 [+0.032, +0.069] 14B → 1.7B, K=32 overlap 3342 +0.052 [+0.032, +0.069]

### F Regression Diagnostics for Learnable Disagreement

Specification. For each token, the dependent variable is the fixed-context gain Gfixi,t in Eq. 1. All scalar predictors are standardized within the diagnostic dataset. The baseline includes student entropy, local disagreement, their interaction, normalized token position, and teacher entropy. We then compare two extensions: adding compatibility Ct alone, or replacing raw disagreement with the decomposed pair DtL,DtI. Confidence intervals use prompt-cluster bootstrap.

Why coefficients, not only R2. Token gains are heavy-tailed and noisy because a single local update can affect later prefixes and formatting tokens. We therefore use standardized coefficients and prompt-cluster gaps as the main diagnostic; R2 is reported only as an incremental sanity check.

Nonlinear sanity check. We also bin residualized gains by DL, DI, and Ct using the same fixed-context bank. The binned curves do not indicate that the separation in Fig 3A is an artifact of a purely linear model. The source artifacts are archived as standardized_regression_coefficients.csv and spline_sanity_bins.csv under figs/.

### G Budget and Downstream Boundary Checks

Budget sweep. We sweep the effective KLsupervised token budget for TA-OPD and low-

entropy with high-divergence compatibility controls. The fixed-context gain peaks around 3–5% and saturates at 10%, suggesting that selected supervision is quality-limited rather than simply budget-limited.

Macro-average budget view. Table 11 compresses the Qwen3-4B student budget sweep from Table 4 into benchmark macro-averages. The trend is not monotonic: TA-OPD or TA-OPD+Ent. is strongest at several low-budget points, while larger budgets can help or hurt depending on the teacher and benchmark mix.

Downstream selector ablation. Table 12 compares alternative token selectors in the Qwen38B-GRPO to Qwen3-4B setting at the same 10% supervised-token budget. TA-OPD obtains the best average score (54.65), ahead of raw KL (53.76), Conly selection (54.19), Q3-only selection (52.46), and Q3+high-C selection (53.32). This supports the main interpretation: useful OPD targets require both a meaningful teacher correction and local support alignment.

Early downstream smoke checks. Before the full benchmark suite in Section 5.2, we ran small downstream smoke checks on GSM8K-COT, MATH-hard, and capped AIME. GSM8K-COT followed the fixed-context trend, while MATH-hard and capped AIME were low-resolution boundary checks; we therefore report the complete downstream evidence only in the main benchmark tables and omit the sparse smoke table.

- Table 7: Low-entropy with high-divergence live intervention. Even inside this TIP region, high teachability is beneficial while low teachability is harmful.

Q3-region mask Keep % Boot. gain Match. gain DL DI C Q3+DL high 3.26 0.015 ± 0.010 0.010 0.072 0.005 0.972 Q3+DL low 3.24 -0.010 ± 0.015 -0.006 0.015 0.011 0.933 Q3+DI high 3.23 0.006 ± 0.003 -0.003 0.053 0.012 0.919

- Table 8: Exact top-N matched fixed-context analysis at K = 16. Useful as a confound check: all selectors retain exactly the same number of tokens.

Selector Ratio Gain 95% interval Gain/keep Q3 frac. H D C DL 3 0.130 [0.123, 0.138] 4.34 7.4 0.31 1.00 0.99 DL 5 0.135 [0.128, 0.143] 2.70 5.5 0.37 0.94 0.97 KL 3 0.120 [0.112, 0.129] 4.00 6.1 0.40 1.00 0.66 KL 5 0.119 [0.113, 0.126] 2.38 5.4 0.41 1.00 0.65 TIP-soft 3 0.093 [0.075, 0.110] 3.11 2.6 0.70 0.69 0.59 TIP-soft 5 0.094 [0.080, 0.107] 1.88 2.6 0.69 0.69 0.61 Entropy 3 0.076 [0.055, 0.095] 2.53 0.0 1.00 0.42 0.54 Entropy 5 0.066 [0.050, 0.082] 1.32 0.0 1.00 0.42 0.56 Random 3 -0.130 [-0.200, -0.072] -4.35 4.8 0.18 0.13 0.93 Random 5 -0.105 [-0.149, -0.057] -2.09 4.8 0.19 0.13 0.93 DI 3 0.105 [0.096, 0.113] 3.50 1.8 0.64 0.87 0.03 DI 5 0.111 [0.104, 0.117] 2.22 1.6 0.66 0.72 0.11

### H Impact to Model Fusion

Model fusion aims to aggregate capabilities from multiple LLMs under limited compute, either by parameter-space merging or by distillation-based fusion. Recent merging studies characterize scaling behavior and budget-aware parameter management (Wang et al., 2025, 2026d), while related work studies post-merge calibration, quantization, and geometry conflict in weight space (Wang et al., 2026f; Gu et al., 2026a; Wang et al., 2026c,a). Another line performs implicit or logit-level fusion through preference optimization and distillation, treating model outputs as transferable supervision signals (Gu et al., 2026b; Wang et al., 2026e). Our work is complementary to both views. Rather than asking how to combine models globally, TAOPD studies which token-level teacher signals are locally absorbable by the current student policy. This suggests a compatibility-aware perspective on distillation-based fusion: effective fusion should not only aggregate stronger sources, but also identify the subset of teacher signals that the target model can learn from.

### I Proof of Proposition 1

Let gt = ∇θℓt(θ) and ∆t = −ηgt. By βsmoothness of Lfix,

Lfix(θ + ∆t) = Lfix(θ) + ⟨∇θLfix(θ),∆t⟩ + rt,

with

β 2∥∆t∥22.

|rt| ≤

Substituting ∆t = −ηgt gives

Lfix(θt+) = Lfix(θ) − η ⟨∇θLfix(θ),gt⟩ + rt. Rearranging,

Gt = η ⟨∇θLfix(θ),gt⟩ − rt. Setting Rt = −rt yields the result.

###### B. support proxy audit

###### A. compact robustness heatmap

###### C. bucket-level shape

0.7

[Figure 14]

+0.078 +0.016 0.036

4B→1.7B held-out

K=32 / CBC

0.6

+0.087 +0.037 0.042

4B→1.7B GSM8K-COT

K=32 / mass

0.5

trimmedtokengain

0.4

+0.118 +0.042 0.050

8B→1.7B math

K=32 / overlap

0.3

+0.098 +0.071 0.037

K=16 / CBC

14B→1.7B K=8

C math

0.2

DI math DL math C GSM8K

K=16 / mass

+0.084 +0.058 0.029

14B→1.7B K=16

0.1

DI GSM8K DL GSM8K

K=16 / overlap

+0.078 +0.046 0.033

14B→1.7B K=32

0.0

bootgap matchgap CItight

0.00 0.05 0.10

0 1 2 3

within-Q3 gain gap

Q3 bucket

Figure 5: Additional low-entropy with high-divergence and robustness evidence. The panels visualize the supportproxy audit, compact robustness statistics, and bucket-level gain shapes used by Sections 3.4–3.3.

- Table 9: Prompt-cluster bootstrap regression decomposition. Learnable disagreement has a consistently larger

standardized coefficient than incompatible disagreement. ∆Rdecomp2 is reported in 10−3 units relative to the entropy+divergence baseline.

K Q3 tokens βL βI Gap 95% interval ∆Rdecomp2

8 2393 0.086 0.045 0.041 [0.039, 0.044] 0.897 16 2393 0.087 0.043 0.044 [0.041, 0.047] 0.790 32 2393 0.087 0.043 0.044 [0.041, 0.047] 0.743

- Table 10: Fixed-context budget sweep for TA-OPD. The teachability selector peaks around 3–5% effective targettoken budget and saturates at 10%.

Selector Nominal % Actual % Bootstrap gain Matching gain Seeds TA-OPD 1 1.89 0.034 ± 0.016 0.017 ± 0.015 3 TA-OPD 3 3.35 0.047 ± 0.006 0.030 ± 0.003 3 TA-OPD 5 5.14 0.049 ± 0.002 0.027 ± 0.003 3 TA-OPD 10 10.16 0.035 ± 0.013 0.022 ± 0.007 3 Q3+high-C 1 1.93 0.009 ± 0.004 0.002 ± 0.002 2 Q3+high-C 3 3.20 0.016 ± 0.001 0.007 ± 0.006 3 Q3+high-C 5 3.93 0.009 ± 0.015 0.002 ± 0.009 3 Q3+low-C 3 3.23 0.010 ± 0.003 0.006 ± 0.006 3 DI 3 3.52 0.025 ± 0.000 0.014 ± 0.005 3

- Table 11: Macro-average budget sweep on the two Qwen3-4B student settings. Scores average the six downstream benchmark means in Table 4; the full per-benchmark results are kept in the main table. TA-OPD denotes TeachabilityAware OPD, and TA-OPD+Ent. adds the entropy axis.

Setting Budget Entropy TIP TA-OPD TA-OPD+Ent.

5% 56.95 56.20 57.35 57.89 10% 56.69 56.81 56.87 56.52 30% 57.56 58.46 57.28 57.46 50% 53.90 55.74 57.90 57.25

Qwen3-8B (GRPO)→4B

5% 53.56 53.83 53.58 54.47 10% 53.39 53.62 54.65 54.10 30% 53.62 53.93 52.81 53.15 50% 52.76 52.12 52.16 52.95

Qwen3-14B

→4B

- Table 12: Downstream selector ablation for Qwen3-8B-GRPO → Qwen3-4B at a 10% supervised-token budget. Entries report mean±std over five evaluation seeds; Avg. averages the six benchmark means. TA-OPD is not reducible to raw KL, compatibility mass C, or the low-entropy with high-divergence Q3 region.

Selector Avg. AIME24 AIME25 GPQA-D. HumanEval IFEval MATH-500 Raw KL 53.76 21.66±1.66 20.00±0.01 44.95±1.52 78.97±0.30 73.38±2.22 83.60±0.40 C-only 54.19 21.66±1.66 18.33±8.33 43.94±3.03 77.75±0.91 80.13±0.83 83.30±0.30 Q3-only 52.46 21.66±1.66 15.00±1.67 44.70±1.27 79.57±0.91 70.52±0.28 83.30±0.10 Q3+high-C 53.32 21.67±5.00 16.66±3.33 44.69±3.29 78.97±0.30 75.05±5.36 82.90±1.10 Entropy 53.39 21.67±5.00 15.00±1.67 46.46±0.01 79.27±0.61 73.94±4.06 84.00±0.80 TIP 53.62 20.00±0.01 16.66±3.33 47.98±2.53 78.05±0.61 76.62±1.57 82.40±0.01 TA-OPD+Ent. 54.10 21.67±5.00 21.67±1.67 42.17±0.25 78.35±1.53 77.54±1.02 83.20±0.60 TA-OPD 54.65 23.34±3.34 18.34±1.66 45.20±1.77 79.57±0.92 78.93±0.01 82.50±1.70

- Table 13: MATH-hard category deltas relative to the base model. Values are percentage-point changes. This table is best treated as boundary analysis rather than the central claim.

Model Alg. C&P Geo. IntAlg. Num. PreAlg. PreCalc. DL +1.6 0.0 +3.0 -1.4 -3.9 -0.5 +1.5 TIP +2.9 +0.8 +2.3 +3.2 -7.1 +3.1 +0.7 DI +1.0 +4.9 +0.8 +1.8 -3.9 +2.6 0.0 Random +2.0 -0.8 -0.8 +0.4 -3.9 +2.1 0.0 Entropy+DL 0.0 +0.8 +3.8 +1.4 -5.2 0.0 +0.7 Entropy 0.0 0.0 +2.3 -2.1 -2.0 +0.5 -0.7

