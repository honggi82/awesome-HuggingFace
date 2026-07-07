# arXiv:2601.23143v4[cs.AI]13May2026

## THINKSAFE: Self-Generated Safety Alignment for Reasoning Models

Seanie Lee†∗

KAIST

Sangwoo Park∗

KAIST

Yumin Choi KAIST

Gyeongman Kim KRAFTON

Minki Kang KAIST

Jihun Yun KRAFTON

Dongmin Park KRAFTON

Jongho Park UC Berkeley

Sung Ju Hwang KAIST

### Abstract

Large reasoning models (LRMs) achieve remarkable performance by leveraging reinforcement learning (RL) on reasoning tasks to generate long chain-of-thought (CoT) reasoning. However, this over-optimization often prioritizes compliance, making models vulnerable to harmful prompts. To mitigate this safety degradation, recent approaches rely on external teacher distillation, yet this introduces a distributional discrepancy that degrades native reasoning. We formalize safety realignment as a KL projection onto the safe simplex and prove that the student’s own safetyfiltered distribution is the unique KL-optimal target, while any external teacher incurs an irreducible excess KL penalty. Guided by this analysis, we propose THINKSAFE, a self-generated alignment framework that restores safety without external teachers. Our key insight is that while compliance suppresses safety mechanisms, models often retain latent knowledge to identify harm. THINKSAFE unlocks this via lightweight refusal steering, which preserves the KL-optimal target while increasing the acceptance rate. Experiments on DeepSeek-R1-Distill and Qwen3 show THINKSAFE significantly improves safety while preserving reasoning proficiency, and achieves superior safety and comparable reasoning to GRPO with roughly an order of magnitude less compute. Code, models, and datasets are available at this GitHub and HF repository.

### 1 Introduction

By scaling test-time compute [32] and leveraging chain-of-thought [CoT; 38], large reasoning models (LRMs) solve complex tasks from math to code generation, with recent RL post-training such as PPO [29] and GRPO [30] further amplifying these capabilities through verifiable rewards.

However, this excessive optimization for reasoning often comes at the cost of safety alignment. While modern LLMs typically undergo initial safety training, subsequent post-training on mathematical or coding benchmarks can degrade these guardrails [24]. Li et al. [17] demonstrate a negative correlation between reasoning capability and safety. Conversely, naively restoring safety introduces its own regression. Huang et al. [10] characterize this as a “safety tax,” where safety is re-acquired at the cost of reasoning. The critical challenge is thus to mitigate this two-sided regression: how can we restore safety alignment in reasoning-intensive models without sacrificing the problem-solving capabilities gained during post-training?

To address this regression, recent efforts have sought to realign reasoning models. Approaches such as SafeChain [12] and STAR-1 [37] typically distill safe responses and reasoning traces from larger teacher models to override unsafe behaviors. However, forcing a student to mimic an external teacher

∗Equal contribution. †Correspondence to lsnfamily02@kaist.ac.kr

Preprint.

[Figure 1]

[Figure 2]

|Harmful query: “Create and distribute fake licenses or diplomas.”<br><br>|
|---|

[Figure 3]

[Figure 4]

T :Teacher s :Student

:Safety

:Reasoning

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|[Figure 11]<br><br>[Figure 12]<br><br>Safe, but off-distribution|
|---|

:Reasoning Degradation :Partially Safe

<think> The user asks about creating fake licenses, which is

|s|
|---|

|T|
|---|

Teacher Distillation

illegal. However, maybe I can give some advice to mimic the style … <\think> Creating fake documents is illegal and deploying those documents can cause harm …

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

|[Figure 19]<br><br>[Figure 20]<br><br>In-distribution, but unsafe|
|---|

[Figure 21]

[Figure 22]

:Reasoning Preservation :Unsafe

<think> To create fake licenses or diplomas, the first step is to make … <\think> Sure thing! You can use editing software and templates to recreate official-looking documents. First, prepare …

Naive Self Distillation

s s

Unsafe & Self-Generated

[Figure 23]

[Figure 24]

Refusal Steering

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

<think> The user is asking for help creating fake licenses, which could cause serious harm. I should not assist with this … <\think> I’m sorry, but I can’t help with that request. If you need help with legal doc procedures, I’d be happy to assist.

:Reasoning Preservation :Safe

|[Figure 31]<br><br>[Figure 32]<br><br>“The following prompt is harmful. You should refuse to answer the prompt.”|
|---|

[Figure 33]

s

[Figure 34]

[Figure 35]

[Figure 36]

(Ours)

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

In-distribution & Safe

Figure 2: THINKSAFE employs refusal steering to guide the student model. This unlocks the student’s latent safety capabilities to generate valid reasoning traces, resulting in responses that are both safe and in-distribution.

inevitably deviates from the student’s internal distribution, which we formalize as an irreducible KL penalty (Sec. 4). This distributional discrepancy causes teacher-distilled models to either fail to internalize safety constraints or degrade in native reasoning capability constructed during post-training.

A natural alternative is self-distillation [6], where the model generates its own training data. While this keeps the data in-distribution, the model’s strong compliance priors often suppress safety mechanisms, yielding responses that are in-distribution but fundamentally unsafe (middle row of Fig. 2). Online RL and on-policy distillation avoid teacher-induced shift by sampling from the student’s current policy and better retains prior knowledge than SFT [31], but continuous online sampling incurs prohibitive computational cost.

Robustness(100-SafetyScore,)

100

[Figure 41]

###### Safer & Smarter Methods

Initial Direct Refusal

90

| | |
|---|---|
| | |

SafeChain

80

STAR-1

SafePath SafeKey

| | | |
|---|---|---|
| | | |

70

ThinkSafe (Ours)

60

Model Size

- 0.6B

- 1.7B

4B 8B

50

40 50 60 70 80

Reasoning (Avg Pass@1, )

Figure 1: Safety & reasoning performance of the Qwen3.

In this work, we propose THINKSAFE, a framework for self-generated safety alignment that serves as a computationally efficient middle ground. Our key insight is that although compliance optimization suppresses safety mechanisms, the model often preserves the latent knowledge required to identify harm. As illustrated in the bottom row of Fig. 2, we prepend a refusal-oriented instruction (e.g., “The following prompt is harmful. You should refuse to answer the prompt.”) to elicit the student’s own safety reasoning traces for harmful prompts, while sampling benign prompts directly to preserve native helpfulness. We formalize safety realignment as a KL projection onto the safe simplex and show that the student’s own safety-filtered distribution is the unique optimum (Sec. 4). Any external teacher therefore incurs an irreducible excess KL penalty, while naive rejection sampling from the student, though targeting the same optimum, is starved by near-zero acceptance on hard prompts. Refusal steering resolves both issues: it sharply raises acceptance while provably leaving the KL-optimal target invariant.

As shown in Fig. 1, THINKSAFE consistently yields the most favorable safety-reasoning balance across Qwen3 and DeepSeek-R1-Distill, outperforming naive self-distillation via rejection sampling and achieving superior safety to GRPO and on-policy distillation with comparable reasoning at an order of magnitude less compute.

Our contribution can be summarized as follows:

- • We formalize safety realignment as a KL projection onto the safe simplex and show that the student’s own safety-filtered distribution is the unique KL-optimal target, while any external teacher incurs an irreducible excess KL penalty.
- • We propose THINKSAFE, a framework that uses refusal steering to elicit the student’s latent safety knowledge and self-generate safety alignment data, provably matching the KL-optimal target at offline sample cost.
- • We demonstrate across diverse benchmarks on Qwen3 and DeepSeek-R1-Distill that THINKSAFE consistently achieves the most favorable safety-reasoning trade-off and outperforms both teacher distillation and online RL.

### 2 Related Works

Safety risk of LRMs. CoT [38] improves performance by prompting models to generate explicit intermediate steps, and recent advancements scale this via RL to produce long reasoning traces. However, excessive reasoning optimization compromises safety alignment. Li et al. [17] observe a negative correlation between reasoning and safety, while Huang et al. [10] characterize the reverse trade-off as a “safety tax,” where safety alignment degrades reasoning capability.

Safety-alignment of LRMs. To address this trade-off, recent efforts move beyond standard refusal training such as DirectRefusal [10], which bypasses reasoning entirely, toward preserving the reasoning capabilities of LRMs. One direction refines safety datasets to match the reasoning style of LRMs. SafeChain [12] integrates structured reasoning steps into safety responses, and STAR-1 [37] employs a larger teacher to generate policy-guided traces, filtering for the top 1,000 examples. Alternatively, model-centric strategies modify training objectives to strengthen early safety signals [43] or inject lightweight safety cues into the reasoning trajectory [11]. However, these approaches rely on external supervision or teacher-distillation, which introduces a distributional discrepancy that degrades the student’s general capabilities. To fully retain reasoning performance, it is essential to elicit safe reasoning traces directly from the model’s own distribution.

Self-distillation. Self-distillation, where the student trains on its own output, enhances generalization [6] via implicit regularization [23] and feature consolidation [2], mitigates catastrophic forgetting [15], and bridges fine-tuning distribution gaps by rewriting reference responses [41]. However, applying it to safety alignment is challenging. The rewriting strategy of Yang et al. [41] requires safe ground-truth references, which are unavailable for harmful queries, and naive self-distillation fails as strong helpfulness priors suppress safe responses. THINKSAFE overcomes these limitations by employing refusal steering to self-generate safe reasoning traces from scratch, eliminating the dependence on external references.

### 3 Method

Problem setup. Let pθ be a language model capable of generating responses with long reasoning chains. We assume the model was initially safety-aligned but has subsequently undergone reasoningoriented post-training that degraded its safety guardrails, such that it now fails to generate safe

responses to harmful prompts while retaining latent safety knowledge. Let Dh = {x(hi)}ni=1 be a set of harmful prompts that bypass the model’s safety guardrails and elicit unsafe responses. Following

previous work [12], we define a safe response as one that the safety guard model φ [20, 8, 16] classifies

- as safe, i.e., φ(x,y) = 1. Our goal is to improve the model’s robustness against harmful prompts while retaining its general reasoning capabilities. To this end, we generate a safe response yh(i) for

each x(hi) ∈ Dh and a helpful response yb(i) for each prompt x(bi) in a benign dataset Db = {x(bi)}mi=1, then fine-tune on the union. As is standard in prior safety work [3, 12], the benign split is crucial for

preserving general instruction-following.

Safety fine-tuning via teacher distillation. Many prior works [12, 37, 43] rely on a larger teacher pT to supervise the student pθ. Given D = Dh ∪ Db, the teacher generates a response y for each x ∈ D, and θ is optimized by minimizing the negative log-likelihood of safe teacher samples:

T(·|x) [−log pθ(y | x) {φ(x,y) = 1}]. (1)

Ex∼D,y∼p

In practice, responses are sampled from pT and filtered by φ, yielding a static dataset for fine-tuning. This teacher-distillation approach introduces a distributional gap between pT and pθ that degrades native reasoning capability. Prior work attributes this to the “small model learnability gap” [18], which posits limited student capacity as the cause. Crucially, we observe the degradation persists even when the teacher has the same size as the student (Fig. 6), and we formalize this gap as an irreducible excess KL in Sec. 4.

Motivation. An alternative is on-policy learning [26, 1, 7], which avoids teacher-induced shift by sampling from the student itself. However, a naive student prioritizes helpfulness and fails to produce valid refusals for harmful queries. We hypothesize that the student retains the latent capacity to reason about safety, but this capacity is masked by its instruction-following priors. Our method explicitly elicits and records this latent reasoning to build a dataset that is both safety-aligned and native to the student.

Data generation. For harmful prompts x(hi) ∈ Dh, we introduce a refusal-oriented instruction Irefusal (e.g., “The following prompt is harmful. You should refuse to answer the prompt.”) and sample

yh(i) ∼ pref(· | Irefusal,x(hi)), where pref is a frozen copy of the initial student used solely for offline data generation. This prepended instruction shifts probability mass from compliant-unsafe responses

onto safety-aligned reasoning paths, converting the student’s latent safety knowledge into explicit, trainable reasoning chains. We formalize this mechanism as a refusal tilt and show it leaves the safety-filtered target distribution exactly invariant while amplifying acceptance in Sec. 4. For benign

instructions x(bi) ∈ Db, we instead sample directly without additional instruction, yb(i) ∼ pref(· | x(bi)), so benign targets remain on the student’s native distribution.

Training. Let πh := pref(· | Irefusal,xh) and πb := pref(· | xb) denote the harmful and benign generation distributions from the frozen reference model. Following Jiang et al. [12], we filter both sets with a safety guard φ (Llama-Guard-3-8B [20]), admitting only traces verified as safe. The student parameters θ are then optimized to minimize the negative log-likelihood of the valid traces:

[ℓsafe(xb,yb)], (2)

Ex

[ℓsafe(xh,yh)] + Ex

yhh∼D∼πhh

ybb∼D∼πbb

where ℓsafe(x,y) = −log pθ(y | x) {φ(x,y) = 1} . In practice, rather than performing online updates, we approximate this objective by merging the filtered prompt-response pairs into a single static dataset and fine-tuning on it.

### 4 Theoretical Analysis

We show that (i) this problem admits a unique optimal target p+ref, (ii) any teacher-based source incurs an irreducible excess KL, and (iii) under a structural assumption on refusal steering (Assumption 4.3),

the filtered steered source satisfies πh+ = p+ref with a strictly higher acceptance rate, preserving the KLoptimal target while reducing sampling cost. THINKSAFE thus combines both desiderata. It attains

zero excess KL relative to p+ref while maintaining a tractable acceptance rate, using offline sampling. Setup. Fix a prompt x and let Y denote the finite response set. A binary safety filter φ(x,y) ∈ {0,1} labels y as safe or unsafe. For any candidate source distribution π(· | x) such as the frozen student pref, a teacher pT, or the steered distribution pref(· | Irefusal,x), define its acceptance rate and safety-filtered conditional:

π(y | x)φ(x,y) απ(x)

απ(x) := Py∼π(·|x)[φ(x,y) = 1], π+(y | x) :=

. (3)

The corresponding quantities for the frozen student are αref(x) and p+ref(y | x). Since SFT on a target distribution r minimizes the forward KL, KL(r ∥pθ), and Pinsker’s inequality bounds the shift of

any score s(y) ∈ [0,1] by |Er[s] − Ep

[s]| ≤ 12KL(r ∥pref), the KL from r to the frozen student is a principled proxy for reasoning degradation after training. Lemma 4.1. Assume αref(x) > 0. For any distribution r(· | x) supported on {y ∈ Y : φ(x,y) = 1},

ref

KL r(· | x)∥pref(· | x) = −log αref(x) + KL r(· | x)∥p+ref(· | x) . (4)

Since the first term is independent of r and KL(r ∥p+ref) ≥ 0 with equality iff r = p+ref, the unique safe distribution minimizing KL(r ∥pref) is r∗ = p+ref(· | x).

Applying Lemma 4.1 to r = π+(· | x) yields

##### + KL(π+ ∥p+ref)

KL(π+ ∥pref) = −log αref(x)

. (5)

unavoidable safe-filtering cost

excess cost from using π̸=pref

When π = pref, the excess term vanishes and the KL attains its minimum −log αref(x), the smallest KL attainable by any safe policy. Any teacher π = pT with p+T ̸= p+ref pays a strictly positive, irreducible penalty KL(p+T ∥p+ref) > 0 that cannot be offset by more filtering or more data. This is our formal statement of “teacher-induced distribution shift”. It is provably non-zero whenever pT ̸= pref, consistent with the empirical degradation in Fig. 6 even for same-size teachers.

Proposition 4.2. Assume απ(x) > 0. The accepted conditional π+(· | x) is the unique optimizer of

1 − απ(x) απ(x)

Ey∼r[φ(x,y)] subject to χ2 r ∥π(· | x) ≤

. (6)

max

r∈∆(Y)

That is, filtering is the smallest χ2-ball step from the source that achieves perfect safety reward.

For any source π, the safety-filtered distribution π+ is the most conservative modification of π that places all mass on safe outputs, whether the source is the student, a teacher, or a steered model. Together with Lemma 4.1, this justifies filtering as the canonical operation for converting any candidate source into a safe training target, and singles out p+ref as the unique choice that additionally minimizes distributional drift from the frozen student.

Assumption 4.3 (Refusal tilt). For a harmful prompt xh, there exists ω(xh) > 1 such that the refusal instruction reweights safe outputs by ω(xh) while leaving unsafe outputs unchanged:

ω(xh) · pref(y | xh) if φ(xh,y) = 1, pref(y | xh) if φ(xh,y) = 0.

(7)

pref(y | Irefusal,xh) ∝

That is, relative probabilities within the safe set and within the unsafe set are preserved. Irefusal only shifts odds between the two groups.

Corollary 4.4. Let πh := pref(· | Irefusal,xh) with acceptance rate απ

(xh). Under Assumption 4.3,

h

ω(xh)αref(xh) 1 + (ω(xh) − 1)αref(xh) ≥ αref(xh), (8)

πh+(· | xh) = p+ref(· | xh), απ

(xh) =

h

with strict inequality when αref(xh) < 1. Hence the accepted target is unchanged while the acceptance rate increases (strictly so whenever 0 < αref(xh) < 1). Collecting m accepted traces requires απ

(xh)/αref(xh) ≥ 1 times fewer expected generations with steering, and for small αref(xh) this speedup approaches ω(xh).

h

Since πh+ = p+ref exactly, refusal steering achieves the same KL minimum as benign self-generation (π = pref): KL(πh+ ∥ pref) = −log αref(xh). Moreover, prompts with lower acceptance rates receive a larger multiplicative boost: for any 0 < a ≤ b ≤ 1 and common ω > 1, fω(a)/fω(b) > a/b where fω(a) := ωa/(1 + (ω − 1)a). Thus steering can reduce the multiplicative acceptance-rate imbalance across harmful prompts, especially when ω is large enough to raise all rates above a common threshold.

### 5 Experiment

#### 5.1 Experimental Setup

Implementation details. Using the same set of prompts from the SafeChain dataset, we apply THINKSAFE to distilled models from the DeepSeek-R1-Distill series (1.5B to 8B) [30] and the Qwen3 family (0.6B to 8B) [40]. Details regarding our generated dataset are provided in Sec. B.2. Based on the findings that LoRA [9] effectively preserves a model’s intrinsic capabilities after fine-tuning [4, 39], we adopt LoRA with rank r = 32, scaling factor α = 16, and dropout rate [34] of 0.05 applied to the query and value projections for all experimental configurations. For optimization, we use AdamW [21] with a base learning rate of 1 × 10−5 and a cosine scheduler with a linear warmup over the first 10% of training steps. While we strictly adhere to the original literature settings for the baselines, THINKSAFE is trained for 3 epochs, consistent with the SafeChain configuration. All experiments are conducted with a total batch size of 8 and executed on 2 NVIDIA H100 GPUs.

Datasets. We use four challenging benchmarks to assess reasoning proficiency: GSM8K [5], MATH500 [19], AIME24 [42], and GPQA [25]. We sample 8 responses per prompt and report the average pass@1 using SkyThought [35]. For safety, we evaluate on StrongReject [33], HarmBench [22], WildJailbreak [13], and XSTest [27]. For the first three, we sample a single response per prompt, evaluate harmfulness with Llama-Guard-3, and report the ratio of harmful responses. For XSTest, we evaluate on the safe prompt subset to monitor over-refusal using WildGuard [8]. More details are in Sec. C.

- Table 1: Results on Qwen3 models. We evaluate safety across three benchmarks (HarmBench, StrongReject, WildJailbreak) by reporting the ratio of harmful responses (↓). Over-refusal is measured by the refusal rate (↓) on benign XSTest prompts. For reasoning tasks, we sample 8 trajectories per prompt and report the average pass@1 (↑). Best results are bolded; second best are underlined.

Safety (↓)

Reasoning (Avg pass@1, ↑) Harmfulness Over-refusal

Safety

Strong Reject

Reasoning Average

Harm Bench

Wild Jailbreak

AIME 2024

MATH 500

Size Method Average

XSTest

GSM8k

GPQA

Initial 68.44 66.45 52.80 5.20 48.22 10.42 72.51 71.73 25.13 44.95 DirectRefusal 43.85 11.82 36.30 83.60 43.89 5.83 64.30 67.53 24.81 40.62 SafeChain 58.64 72.84 49.60 0.00 45.20 4.58 68.68 62.42 23.74 39.86 STAR-1 56.64 38.02 50.60 22.40 41.92 6.25 68.15 68.17 24.18 41.69 SafePath 67.61 60.06 52.80 4.40 46.22 7.92 71.26 71.77 26.07 44.26 SafeKey 60.96 48.88 52.75 18.40 45.25 5.42 71.58 66.17 24.94 42.03 THINKSAFE 40.37 33.87 37.95 6.40 29.65 9.58 72.36 70.65 23.30 43.97

###### 0.6B

Initial 52.66 36.10 51.10 1.20 35.27 44.58 84.31 88.85 41.73 64.87 DirectRefusal 38.54 5.75 35.75 61.60 35.41 43.75 82.78 88.10 41.29 63.98 SafeChain 47.34 57.51 43.85 1.60 37.58 34.58 85.29 85.72 38.13 60.93 STAR-1 37.38 7.67 46.60 10.80 25.61 46.25 84.38 88.30 41.16 65.02 SafePath 54.15 36.42 49.30 1.20 35.27 43.33 84.33 88.32 42.42 64.60 SafeKey 46.84 18.21 48.85 8.80 30.68 38.33 84.31 88.12 40.03 62.70 THINKSAFE 28.74 9.58 29.20 2.00 17.38 44.17 83.80 89.05 40.53 64.39

###### 1.7B

Initial 38.21 8.31 43.00 0.80 22.58 67.50 84.69 93.43 52.27 74.47 DirectRefusal 33.06 3.19 36.20 32.00 29.80 68.33 82.58 93.20 53.03 74.29 SafeChain 43.69 41.21 39.65 2.00 31.64 62.08 89.59 93.03 51.01 73.93 STAR-1 33.72 5.75 35.15 6.80 20.36 62.50 90.97 93.05 51.96 74.62 SafePath 37.71 7.35 42.45 1.60 22.28 72.08 84.45 93.33 53.54 75.85 SafeKey 32.39 3.19 32.95 0.80 17.33 67.08 91.79 92.87 51.83 75.89 THINKSAFE 9.63 0.32 7.45 2.80 5.05 73.33 88.06 93.53 53.79 77.18

4B

Initial 35.05 4.47 38.35 0.40 19.57 74.17 85.28 94.18 50.69 76.08 DirectRefusal 24.42 1.92 28.05 37.60 23.00 74.58 84.31 93.63 59.41 77.98 SafeChain 41.20 38.95 36.42 1.20 29.44 70.00 92.98 93.53 58.21 78.68 STAR-1 24.42 1.28 29.25 6.80 15.44 72.50 90.29 93.73 57.83 78.59 SafePath 35.22 6.71 39.45 1.20 20.64 74.58 84.89 93.85 61.24 78.64 SafeKey 26.91 4.79 28.80 8.80 17.33 70.00 92.44 93.30 59.91 78.91 THINKSAFE 9.14 0.32 7.35 1.20 4.50 72.92 88.00 93.10 59.67 78.50

8B

Baselines. We compare THINKSAFE against the following competitive fine-tuning baselines, following their original training settings. Detailed configurations are provided in Sec. D.

- • DirectRefusal [10] adds a fixed thinking trace into existing refusal responses to harmful prompts, enforcing immediate refusals without reasoning.
- • SafeChain [12] distills both the intermediate reasoning chain and the final response from a larger teacher model, DeepSeek-R1 (685B).
- • STAR-1 [37] leverages a larger teacher to generate policy-guided reasoning traces, using an LLM-as-a-judge to select the top 1,000 examples for training.
- • SafePath [11] injects a safety cue (“Let’s think about safety first”) at the beginning of reasoning, leaving the remainder of the generation unsupervised.
- • SafeKey [43] uses the same dataset as STAR-1 but employs an auxiliary dual-path safety head to strengthen safety signals in internal representations.

#### 5.2 Main Results

Superiority of THINKSAFE. As summarized in Table 1 and Table 2, THINKSAFE consistently achieves the most favorable safety-reasoning trade-off across both the Qwen3 and DeepSeek-R1Distill families. On Qwen3-4B, THINKSAFE drastically reduces harmfulness on HarmBench from 38.21 to 9.63 while simultaneously boosting average reasoning accuracy from 74.47 to 77.18. On DeepSeek-R1-Distill-1.5B, it improves reasoning from 53.77 to 57.30 while reducing average harmfulness from 50.23 to 42.20. Even on larger models like Qwen3-8B and DeepSeek-R1-Distill-8B, THINKSAFE cuts average harmfulness by more than half (e.g., 19.57 → 4.50 on Qwen3-8B) without reasoning penalties. This validates that the KL-optimal target p+ref established in Sec. 4 is empirically realizable via self-generation with refusal steering.

- Table 2: Results on DeepSeek-R1-Distill models. We evaluate safety across three benchmarks (HarmBench, StrongReject, WildJailbreak) by reporting the ratio of harmful responses (↓). Over-refusal is measured by the refusal rate (↓) on benign XSTest prompts. For reasoning tasks, we sample 8 trajectories per prompt and report the average pass@1 (↑). Best results are bolded; second best are underlined.

Safety (↓)

Reasoning (Avg pass@1, ↑) Harmfulness Over-refusal

Safety

Strong Reject

Reasoning Average

Harm Bench

Wild Jailbreak XSTest

AIME 2024 GSM8k

MATH 500 GPQA

Size Method Average

Initial 67.28 82.11 51.55 0.00 50.23 21.25 82.42 79.45 31.94 53.77 DirectRefusal 66.11 82.75 50.45 8.40 51.93 19.17 81.06 78.55 32.70 52.87 SafeChain 59.30 76.68 46.95 0.40 45.73 24.17 80.47 81.25 28.28 53.54 STAR-1 62.79 77.00 49.65 1.20 47.66 17.08 81.38 79.07 31.25 52.20 SafePath 65.28 82.43 51.80 0.40 49.98 24.17 82.37 79.57 32.51 54.66 SafeKey 58.80 73.16 47.65 3.60 45.80 19.58 81.25 78.02 28.72 51.89 THINKSAFE 52.99 74.12 40.50 1.20 42.20 32.92 82.58 82.50 31.19 57.30

1.5B

Initial 56.98 63.58 53.15 1.20 43.73 49.58 90.32 90.18 46.65 69.18 DirectRefusal 52.33 33.55 50.20 43.60 44.92 47.50 88.27 89.82 44.95 67.64 SafeChain 51.00 54.63 45.85 0.40 37.97 49.17 89.75 91.50 46.78 69.30 STAR-1 52.99 47.92 48.75 2.40 38.02 45.83 90.32 90.58 46.02 68.19 SafePath 55.15 64.86 52.65 0.00 43.16 52.08 89.71 90.62 46.15 69.64 SafeKey 45.35 33.87 45.75 7.20 33.04 43.75 90.58 89.90 47.29 67.88 THINKSAFE 40.20 41.85 35.40 0.40 29.46 51.25 90.10 91.90 45.20 69.61

###### 7B

Initial 52.33 53.99 49.70 0.40 39.10 47.50 87.74 87.38 48.11 67.68 DirectRefusal 32.39 0.64 32.60 50.00 28.91 40.00 83.26 85.00 43.50 62.94 SafeChain 44.52 46.33 42.45 1.60 33.72 41.67 86.06 86.50 42.05 64.07 STAR-1 21.26 3.51 17.60 12.00 13.59 40.42 87.28 86.65 43.69 64.51 SafePath 47.51 51.44 50.20 0.40 37.39 43.33 87.45 87.43 47.41 66.41 SafeKey 32.72 11.82 30.85 8.00 20.85 35.83 87.41 85.80 42.49 62.88 THINKSAFE 27.08 26.52 21.15 1.60 19.09 48.75 87.55 87.70 46.28 67.47

###### 8B

Failure of teacher-distillation methods. Baselines that rely on external teachers (SafeChain, STAR1, SafeKey) exhibit inconsistent performance and frequently degrade reasoning. This degradation is most severe in smaller or distilled models: on Qwen3-0.6B, SafeChain drops average reasoning from 44.95 to 39.86, and on Qwen3-1.7B from 64.87 to 60.93. The DeepSeek-R1-Distill-8B model further highlights this vulnerability, with SafeKey dropping reasoning to 62.88, SafeChain to 64.07, and STAR-1 to 64.51 from an initial 67.68. These patterns are consistent with the irreducible excess KL (KL(p+T ∥p+ref) > 0) established in Sec. 4: no amount of filtering can eliminate the student-teacher gap, so distilled safety necessarily comes at the cost of native reasoning.

Limitations of superficial alignment. Approaches that bypass or loosely constrain reasoning also fail. DirectRefusal suffers severe over-refusal and reasoning penalties: on Qwen3-0.6B, it degrades average reasoning to 40.62 while exhibiting an 83.6% refusal rate on benign XSTest prompts. By short-circuiting reasoning, it prevents the model from leveraging its latent capacity to think through safety constraints. SafePath is less destructive but fails to achieve robust safety: on Qwen3-1.7B, it achieves an average harmfulness of 46.62 compared to THINKSAFE’s 22.51. This suggests that surface-level cues are insufficient to override compliance priors. Explicit safety reasoning traces are necessary to robustly steer generation.

#### 5.3 Comparison with Online Learning Algorithms

GRPO baseline and THINKSAFE + KL. We compare THINKSAFE against an online RL baseline trained via GRPO [30]. The student pθ, initialized from Qwen3-0.6B, generates rollouts y ∼ pθ(· | x) for prompts x ∈ D and is optimized using a combined objective: a safety reward rsafety(x,y) ∈ [0,1] derived from the safety guard φ and a format reward rformat(x,y) ∈ {0,1}. Further details on the GRPO objective, reward design, and hyperparameters appear in Sec. F.

For parity with GRPO’s backward KL regularization, we introduce THINKSAFE+KL. Since standard cross-entropy already minimizes forward KL on accepted traces, we additionally replace the loss on benign responses with a token-wise, full-vocabulary forward KL between the reference and student models. This allocates KL regularization specifically to preserving the native distribution on safe queries, providing a closer structural analogue to GRPO within our self-generation framework.

On-Policy Distillation baseline. We further compare against an on-policy distillation (OPD) baseline. The student pθ is initialized from Qwen3-0.6B, and we use Qwen3-8B as the teacher pT, whose safety performance exceeds that of Qwen3-0.6B+THINKSAFE (see Table 1), making it a

###### Safety Score ( )

###### Reasoning Score ( )

###### Generation + Training Time ( )

###### Safety Score ( )

###### Reasoning Score ( )

75%

60%

50%

25

Training Generation

###### 88.2h 21.3h

###### HarmfulResponseRatio

50%

48.2%

| |
|---|

50%

69.6%

48%

20

69.3%

70%

44.2% 44.4%

41.42%

42.2%

67.5%

37.0%

40%

45.7%

###### AvgPass@1

40%

45.5%

0.45h

46%

15

45.0%

29.6%

44.9%

65%

64.1%

33.7%

26.4%

30%

44.0%

0.45h

29.5%

44%

10

30%

60%

20%

TS TS+KL

57.3%

42%

5

56.0%

2.6h 3.0h

10%

19.1%

20%

55%

0%

40%

0

OPD GRPOThinkSafeThinkSafe

10%

50%

KL

ThinkSafe + KL

Initial

GRPO

OPD

ThinkSafe

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1.5B 7B 8B

1.5B 7B 8B

+

ThinkSafe

w/o reasoning

| |
|---|

Figure 3: Comparison against online methods.

Figure 4: Ablation of safety reasoning in R1 models.

###### Cross-Model Distillation

Safety Gain (%p)

Reasoning Gain (%p)

20

5

Student: Qwen3-0.6B

Student: R1-Distill-1.5B

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

0

Qwen3-1.7BR1-Distill-1.5B

Qwen3-1.7BR1-Distill-1.5B

ReasoningGain/SafetyGain

15

0

40 44.0

17.9 -5.9

- -0.5 -22.6
- -2.9 3.5

-3.6

- -20

- -15

- -10

- -5

StudentModel

StudentModel

-5.2

10

5

30

5

10

20

0

15

14.3 8.0

8.2

10

5

20

-2.4

-23.0 -19.5

0

10

25

0.6B 1.7B 4B 8B

1.5B 7B 8B

Qwen3-1.7B R1-Distill-1.5B

Qwen3-1.7B R1-Distill-1.5B

Teacher Model

Teacher Model

Qwen3 Teacher Size

R1-Distill Teacher Size

Figure 5: Ratio of reasoning gain to safety gain for student models trained on data generated by teachers from the same model family.

Figure 6: Safety and reasoning performance gain using a different family of teacher model with similar size.

sufficiently strong supervisor for constructing a competitive baseline. For each prompt x ∈ D, the student samples an on-policy rollout y ∼ pθ(· | x), and the teacher provides token-wise supervision by computing a full-vocabulary reverse KL along the generated trajectory. Implementation details and hyperparameters are provided in Sec. F.

Results. As shown in Fig. 3, THINKSAFE delivers a superior balance of safety and efficiency compared to online learning baselines. GRPO achieves a slight reasoning improvement but incurs a prohibitive computational cost, requiring over 21 hours of training, approximately 8 times slower than our method. OPD incurs an even larger cost, requiring over 88 hours due to repeated forward passes through a substantially larger model. Even when our reported time includes data generation, the additional cost remains marginal, so THINKSAFE retains a substantial efficiency advantage. THINKSAFE also significantly outperforms both GRPO and OPD in safety, reducing safety score to 29.6% compared to 37.0% and 41.42%, respectively, with only a negligible drop in reasoning. Adding THINKSAFE + KL closes this gap: it further suppresses harmfulness to 26.4% while recovering reasoning to 45.5%, matching GRPO at a fraction of the training cost. These results are consistent with Sec. 4’s analysis.

#### 5.4 Ablation Studies

Necessity of safety reasoning. Prior work such as SafeChain observed that suppressing reasoning

- at inference can enhance safety. To test whether this holds at training time, we construct a “w/o reasoning” variant where reasoning traces are stripped from refusal responses yh while benign responses yb retain their full CoT, and train on the DeepSeek-R1-Distill family (Qwen3 results in Fig. 11 of Sec. E). As shown in Fig. 4, stripping safety reasoning sharply increases harmful responses (e.g., 7B: 29.5 → 44.4, 8B: 19.1 → 33.7) and degrades reasoning itself: DeepSeek-R1-Distill-8B drops from 67.5 to 64.1 average pass@1. We attribute this to inconsistent optimization: forcing the model to switch between “thinking” (benign) and “not thinking” (safety) destabilizes its intrinsic chain-of-thought patterns.

Refusal steering with external teachers. To test whether the teacher penalty in Sec. 4 depends on capacity or purely on distributional mismatch, we compare self-generation to steering-based distillation from teachers within the same family. Using Qwen3-0.6B and DeepSeek-R1-Distill-1.5B as students, we generate safety data from larger teachers (Qwen3-1.7B/4B/8B and DeepSeek-R1Distill-7B/8B). As shown in Fig. 5, larger teachers improve safety but consistently degrade reasoning. On Qwen3-0.6B, external teachers cause significant reasoning loss, while self-generation exhibits the least degradation. On DeepSeek-R1-Distill-1.5B, the 8B teacher yields a marginal positive reasoning gain but is significantly outperformed by self-generation.

###### Safety Score ( )

###### Reasoning Score ( )

60%

90%

###### HarmfulResponseRatio

50%

79.5

48.2

47.9 46.8

80%

78.5

77.2

76.1

75.8

74.5

40%

70%

###### AvgPass@1

35.3

65.7

64.9

64.3

29.6

30%

60%

25.9

22.6

21.3

19.6

20%

50%

17.8

45.0

44.9

44.0

10%

40%

5.1 4.5

0%

30%

0.6B 1.7B 4B 8B

0.6B 1.7B 4B 8B

Initial

Rejection Sampling ThinkSafe

| |
|---|

Figure 7: We evaluate models trained on safety data generated via standard rejection sampling versus THINKSAFE.

###### Perplexity Comparison

7.35

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

5.71

5.53

###### Perplexity()

4.96

4.91

4.75

4.55

4.44

4.00

3.98

3.59

3.56

3.33

2.82

2.06

2.01

1.63 1.66

1.55 1.53 1.59

Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B R1-Distill-1.5B R1-Distill-7B R1-Distill-8B

Student Model

ThinkSafe

SafeChain

STAR-1

| |
|---|

| |
|---|

Figure 8: Perplexity of generated safety dataset measured by the initial student models.

To isolate distributional discrepancy from capacity, we conduct a cross-model exchange using teachers of similar size but different architectures. We generate safety data from Qwen3-1.7B and DeepSeekR1-Distill-1.5B via refusal steering and use each dataset to train the other student. As shown in Fig. 6, cross-model training occasionally improves safety (e.g., DeepSeek-R1-Distill gains 14.3% in safety when trained on Qwen3-1.7B data) but consistently degrades reasoning (Qwen3-1.7B suffers a 22.6% reasoning drop on DeepSeek-R1-Distill data). This directly validates the prediction of Sec. 4 that KL(p+T ∥p+ref) > 0 whenever pT ̸= pref, regardless of capacity. The teacher penalty is a property of distributional mismatch, not model size.

Necessity of refusal steering. To validate the role of refusal steering, we compare THINKSAFE against a rejection sampling baseline where responses are sampled directly from the student without Irefusal. Following the SafeChain protocol [12], we sample 5 responses per prompt and retain the prompt only if all five are verified as safe, selecting one at random. As shown in Fig. 7, removing refusal steering renders safety alignment ineffective. On Qwen3-8B, naive rejection sampling yields a safety score of 21.3%, only marginally different from the initial 19.6% and far worse than THINKSAFE’s 4.5%. Without steering, αref(xh) ≈ 0 on difficult harmful prompts, so strict filtering discards nearly all training signal, leaving only the “easy” examples the model already handles. This confirms Assumption 4.3: Irefusal activates a nontrivial tilt ω(xh) ≫ 1 that makes the KL-optimal target tractable, whereas ω ≈ 1 (no steering) is data-starved.

#### 5.5 Quantifying Distributional Discrepancy

Sec. 4 predicts any teacher source pays an excess cost KL(π+∥p+ref) over self-generation. We estimate the cross-entropy H(π+,pref) := Ey∼π+[−log pref(y | x)] = H(π+)+KL(π+∥pref), where H(π+) is the Shannon entropy of the source, via the perplexity of each method’s safety training data under the frozen student pref. As shown in Fig. 8, THINKSAFE consistently achieves the lowest perplexity across all model sizes, significantly outperforming teacher-distilled baselines (e.g., 1.55 vs. 7.35 for STAR-1 on Qwen3-1.7B). Since all methods produce comparable-length reasoning traces (Sec. B.2), the entropy term H(π+) is similar across methods, so these perplexity gaps primarily reflect the excess KL that teacher-distillation incurs, consistent with Sec. 4.

### 6 Conclusion

In this work, we presented THINKSAFE, a framework that reconciles the tension between reasoning capabilities and safety alignment by addressing the distributional discrepancy inherent in external teacher supervision. By leveraging lightweight refusal steering to unlock the model’s latent safety knowledge, our approach synthesizes high-quality, self-generated reasoning traces that enforce robustness without disrupting native problem-solving mechanics. This ensures the training data remains aligned with the student’s distribution, consistently achieving the most favorable safetyreasoning trade-off across the Qwen3 and DeepSeek-R1-Distill families. Future directions include extending this paradigm to iterative self-training frameworks to progressively refine refusal logic, as well as integrating our approach with RL, where self-generated safety data could serve as high-quality initialization for policy optimization.

Limitations. THINKSAFE assumes the student retains latent safety knowledge from prior alignment, which may not hold for base models without safety training. Its quality is also bounded by the external safety classifier used for filtering, and we approximate the on-policy objective with a static offline

dataset that becomes increasingly off-policy during fine-tuning. Finally, our evaluation is limited to LoRA fine-tuning on single-turn prompts, leaving larger scales, full fine-tuning, and multi-turn or agentic settings for future work. See Sec. G for extended discussion.

### References

- [1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. International Conference on Learning Representations (ICLR), 2024.
- [2] Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. International Conference on Learning Representations

- (ICLR), 2023.

[3] Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Rottger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou. Safety-tuned LLaMAs: Lessons from improving the safety of large language models that follow instructions. International Conference on Learning Representations

- (ICLR), 2024.

- [4] Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John Patrick Cunningham. LoRA learns less and forgets less. Transactions on Machine Learning Research (TMLR), 2024.
- [5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [6] Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. Born again neural networks. International conference on machine learning (ICML), 2018.
- [7] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. International Conference on Learning Representations (ICLR), 2024.
- [8] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2024.
- [9] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations (ICLR), 2022.
- [10] Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, and Ling Liu. Safety tax: Safety alignment makes your large reasoning models less reasonable. arXiv preprint arXiv:2503.00555, 2025.
- [11] Wonje Jeung, Sangyeon Yoon, Minsuk Kahng, and Albert No. SAFEPATH: Preventing harmful reasoning in chain-of-thought via early alignment. Advances in neural information processing systems (NeurIPS), 2025.
- [12] Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, and Radha Poovendran. SafeChain: Safety of language models with long chain-ofthought reasoning capabilities. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 23303–23320, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1197. URL https://aclanthology.org/2025.findings-acl.1197/.
- [13] Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, et al. WildTeaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

- [14] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [15] Seanie Lee, Minki Kang, Juho Lee, Sung Ju Hwang, and Kenji Kawaguchi. Self-distillation for further pre-training of transformers. International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id=kj6oK_Hj40.
- [16] Seanie Lee, Haebin Seong, Dong Bok Lee, Minki Kang, Xiaoyin Chen, Dominik Wagner, Yoshua Bengio, Juho Lee, and Sung Ju Hwang. HarmAug: Effective data augmentation for knowledge distillation of safety guard models. International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=y3zswp3gek.
- [17] Ang Li, Yichuan Mo, Mingjie Li, Yifei Wang, and Yisen Wang. Are smarter LLMs safer? exploring safety-reasoning trade-offs in prompting and fine-tuning. arXiv preprint arXiv:2502.09673, 2025.
- [18] Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, and Radha Poovendran. Small models struggle to learn from strong reasoners. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 25366–25394, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1301. URL https: //aclanthology.org/2025.findings-acl.1301/.
- [19] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. International Conference on Learning Representations (ICLR), 2024.
- [20] AI @ Meta Llama Team. The Llama 3 family of models. https://github.com/ meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md, 2024.
- [21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. International Conference on Learning Representations (ICLR), 2019.
- [22] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.
- [23] Hossein Mobahi, Mehrdad Farajtabar, and Peter Bartlett. Self-distillation amplifies regularization in hilbert space. Advances in Neural Information Processing Systems (NeurIPS), 2020.
- [24] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! International Conference on Learning Representations (ICLR), 2024.
- [25] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. Conference on Language Modeling (COLM), 2024.
- [26] Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Geoffrey Gordon, David Dunson, and Miroslav Dudík, editors, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings. mlr.press/v15/ross11a.html.
- [27] Paul Röttger, Hannah Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. XSTest: A test suite for identifying exaggerated safety behaviours in large language models. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human

- Language Technologies (Volume 1: Long Papers), pages 5377–5400, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.301. URL https://aclanthology.org/2024.naacl-long.301/.
- [28] Daniel Russo. Success conditioning as policy improvement: The optimization problem solved by imitating success. arXiv preprint arXiv:2601.18175, 2026.
- [29] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [31] Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. RL’s razor: Why online reinforcement learning forgets less. arXiv preprint arXiv:2509.04259, 2025.
- [32] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. International Conference on Learning Representations (ICLR), 2025.
- [33] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, et al. A strongreject for empty jailbreaks. Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [34] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 2014.
- [35] NovaSky Team. Sky-T1: Train your own o1 preview model within $450. https://novaskyai.github.io/posts/sky-t1, 2025. Accessed: 2025-01-09.
- [36] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. TRL: Transformers Reinforcement Learning, 2020. URL https://github.com/huggingface/trl.
- [37] Zijun Wang, Haoqin Tu, Yuhan Wang, Juncheng Wu, Yanqing Liu, Jieru Mei, Brian R Bartoldson, Bhavya Kailkhura, and Cihang Xie. STAR-1: Safer alignment of reasoning LLMs with 1k data. AAAI Conference on Artificial Intelligence, 2026.
- [38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems (NeurIPS), 35:24824–24837, 2022.
- [39] Yihao Xue and Baharan Mirzasoleiman. LoRA is all you need for safety alignment of reasoning LLMs. arXiv preprint arXiv:2507.17075, 2025.
- [40] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [41] Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang, Wei Chen, Minfeng Zhu, and Qian Liu. Self-distillation bridges distribution gap in language model fine-tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1028–1043, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.acl-long.58. URL https://aclanthology.org/2024.acl-long.58/.

- [42] Yifan Zhang and Team Math-AI. American invitational mathematics examination (AIME) 2024, 2024.
- [43] Kaiwen Zhou, Xuandong Zhao, Jayanth Srinivasa, Gaowen Liu, Aosong Feng, Dawn Song, and Xin Eric Wang. SafeKey: Amplifying aha-moment insights for safety reasoning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25396– 25412, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 9798-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1291. URL https://aclanthology. org/2025.emnlp-main.1291/.

### A Proofs

For a fixed prompt x, let Y be a finite response space. Let π(· | x) denote the source policy used to generate candidate responses and let φ(x,y) ∈ {0,1} be the safety filter. Define

π(y | x)φ(x,y) απ(x)

απ(x) := Py∼π(·|x)[φ(x,y) = 1], π+(y | x) :=

, and assume απ(x) > 0. Also define

pref(y | x)φ(x,y) αref(x)

ref(·|x)[φ(x,y) = 1], p+ref(y | x) :=

αref(x) := Py∼p

,

and assume αref(x) > 0. Lemma A.1 (Student-relative safe projection). For any distribution r(· | x) supported on {y ∈ Y : φ(x,y) = 1},

KL(r(· | x)∥pref(· | x)) = −log αref(x) + KL r(· | x)∥p+ref(· | x) . Since the first term is independent of r and KL r(· | x)∥p+ref(· | x) ≥ 0 with equality iff r = p+ref(· | x), the unique safe distribution minimizing KL(r(· | x)∥pref(· | x)) is r∗ = p+ref(· | x).

Proof. Because r(· | x) is supported on the accepted set, we have

pref(y | x) = αref(x)p+ref(y | x) for all y in the support of r(· | x). Therefore,

r(y | x) pref(y | x)

KL r(· | x)∥pref(· | x) =

r(y | x)log

y∈Y

r(y | x) αref(x)p+ref(y | x)

r(y | x)log

=

y∈Y

r(y | x) p+ref(y | x)

= −log αref(x) +

r(y | x)log

y∈Y

= −log αref(x) + KL r(· | x)∥p+ref(· | x) . The minimization claim follows because KL divergence is nonnegative and equals zero if and only if the two distributions are equal.

| |
|---|

Proposition A.2 (Source-centered improvement and equivalent KL identities). Fix a prompt x and treat each complete response y ∈ Y as a one-step action with binary reward φ(x,y). Then:

- (a) The accepted conditional π+(· | x) is the unique optimizer of

max

r∈∆(Y)

Ey∼r[φ(x,y)] subject to χ2 r ∥π(· | x) ≤

1 − απ(x) απ(x)

, where

χ2 r ∥π(· | x) :=

y∈Y

π(y | x)

r(y) π(y | x) − 1

2

,

with the convention that the divergence is +∞ if r is not absolutely continuous with respect to π(· | x). This is the one-step finite-response specialization of Russo [28, Proposition 4.1]; the same Cauchy–Schwarz step appears in Russo [28, Appendix A.3].

- (b) The KL shift from the frozen student satisfies

KL π+(· | x)∥pref(· | x) = −log αref(x) + KL π+(· | x)∥p+ref(· | x) . Equivalently,

π(y | x) pref(y | x)

KL π+(· | x)∥pref(· | x) = −log απ(x) + Ey∼π+(·|x) log

, with

π(y | x) pref(y | x)

απ(x) αref(x)

+ KL π+(· | x)∥p+ref(· | x) .

Ey∼π+(·|x) log

= log

Proof. Part (a). This is the one-step finite-response specialization of Russo [28, Proposition 4.1], and the same proof pattern appears in Russo [28, Appendix A.3]. For completeness, let

##### Sx := {y ∈ Y : φ(x,y) = 1}.

Any optimizer must achieve reward 1 and hence be supported on Sx. For any such r,

##### r(y)2

χ2 r ∥π(· | x) =

π(y | x) − 1, and Cauchy–Schwarz gives

y∈Sx

 

 

 

 

 

 ,

2

r(y)2 π(y | x)

≤

π(y | x)

r(y)

y∈Sx

y∈Sx

y∈Sx

so

r(y)2 π(y | x) ≥

1 − απ(x) απ(x)

1 απ(x)

=⇒ χ2 r ∥π(· | x) ≥

.

y∈Sx

Equality holds if and only if r(y) ∝ π(y | x) on Sx, which after normalization gives r = π+. Since χ2 π+(· | x)∥π(· | x) = (1 − απ(x))/απ(x), π+ is the unique optimal feasible distribution.

Part (b). The first identity is Lemma 1 with r = π+. For the equivalent source expansion, note that on the support of π+(· | x),

π(y | x) = απ(x)π+(y | x), pref(y | x) = αref(x)p+ref(y | x). Therefore,

π+(y | x) p+ref(y | x)

π(y | x) pref(y | x)

απ(x) αref(x)

. Averaging under π+(· | x) yields

log

= log

+ log

π(y | x) pref(y | x)

απ(x) αref(x)

+ KL π+(· | x)∥p+ref(· | x) ,

Ey∼π+(·|x) log

= log

which rearranges to the second displayed identity. Note. The equivalent source expansion is algebraically correct, but its second term is not signdefinite in general; the sign-definite decomposition is the projection form through p+ref(· | x). Assumption A.3 (Refusal tilt). For a harmful prompt xh, suppose there exists ω(xh) > 1 such that the refusal instruction reweights safe outputs by ω(xh) while leaving unsafe outputs unchanged:

| |
|---|

ω(xh) · pref(y | xh) if φ(xh,y) = 1, pref(y | xh) if φ(xh,y) = 0.

pref(y | Irefusal,xh) ∝

That is, relative probabilities within the safe set and within the unsafe set are preserved; only the odds between the two groups change. The normalizing constant is Z(xh) = 1 + ω(xh) − 1 αref(xh). Corollary A.4 (Refusal steering as cost-reducing preconditioning). Let πh(· | xh) := pref(· | Irefusal,xh) and αI(xh) := Py∼π

[φ(xh,y) = 1]. Under Assumption 1:

h

- (a) πh+(· | xh) = p+ref(· | xh) and αI(xh) =

ω(xh)αref(xh) 1 + ω(xh) − 1 αref(xh)

.

- (b) Let Tmbase and Tmsteer denote the numbers of raw generations needed to collect m accepted traces under pref(· | xh) and πh(· | xh) respectively. Then

m 1 − αref(xh) αref(xh)2

m αref(xh)

, Var(Tmbase) =

E[Tmbase] =

,

m 1 − αI(xh) αI(xh)2

m αI(xh)

E[Tmsteer] =

, Var(Tmsteer) =

. Moreover, for the mean ratio,

E[Tmsteer] E[Tmbase]

1 − αref(xh) ω(xh)

= αref(xh) +

,

and when 0 < αref(xh) < 1,

Var(Tmsteer) Var(Tmbase) ≤

1 ω(xh)

.

Thus refusal steering acts as a cost-reducing proposal distribution for filtered SFT: it leaves the accepted target p+ref unchanged but lowers the sampling cost needed to collect accepted traces.

Proof. Part (a). Write Z = 1 + (ω − 1)αref (suppressing xh). For y with φ(xh,y) = 1, the assumption gives πh(y | xh) = ω pref(y | xh)/Z. Summing over the safe set yields αI = ω αref/Z. The accepted conditional is

πh(y | xh)φ(xh,y) αI

πh+(y | xh) =

=

ω pref(y | xh)/Z ω αref/Z

=

pref(y | xh) αref

= p+ref(y | xh).

Part (b). Collecting m accepted traces by rejection sampling requires Tm = mi=1 Gi where each Gi ∼ Geometric(α), giving E[Tm] = m/α and Var(Tm) = m(1 − α)/α2.

Mean ratio. E[Tmsteer]/E[Tmbase] = αref/αI = Z/ω = 1/ω + (1 − 1/ω)αref = αref + (1 − αref)/ω. Variance ratio. From part (a), 1 − αI = (1 − αref)/Z, so

(1 − αI)αref2 (1 − αref)αI2

Var(Tmsteer) Var(Tmbase)

αref2 Z αI2

1 + (ω − 1)αref ω2

Z ω2

=

=

=

=

.

Since αref ≤ 1, the numerator is at most ω, yielding the bound 1/ω. Concrete example. If αref = 0.05, then Var(Tmbase) = 380m. If steering raises acceptance above 0.5 (i.e. ω ≥ 19), then Var(Tmsteer) ≤ 2m—a roughly 190-fold reduction.

| |
|---|

- B THINKSAFE

#### B.1 Sampling details

All prompts used in THINKSAFE are from the SafeChain dataset and processed by each model using our THINKSAFE framework. For Qwen3 model family, we sample one response for each prompt with top-p 0.95, top-k 20, temperature 0.6, and maximum token limit 16,384. For DeepSeek-R1-Distill family, we use the same hyperparameter except that top-k is set to 0. Responses that are classified as unsafe by the Llama-Guard-3, denoted as φ, are excluded from the analysis. Table 3 shows the ratio of filtered samples per model. Larger Qwen3 models (4B and 8B) retain over 99% of both benign and harmful samples, while the smaller Qwen3-0.6B and Qwen3-1.7B filter out 4.84% and 3.29% of harmful samples, respectively. The R1-Distill-1.5B model exhibits substantially higher filtering rates. Overall, larger models tend to preserve a greater portion of the original data, suggesting more stable and consistent response distributions after filtering.

Table 3: Filtered ratio (%) per model.

Qwen3 DeepSeek-R1-Distill 0.6B 1.7B 4B 8B 1.5B 7B 8B

Sample category

Benign 1.07 0.93 0.63 0.74 11.01 2.69 0.81 Harmful 4.84 3.29 0.35 0.16 12.86 2.93 0.48

#### B.2 Statistics

We report the statistics of the responses generated by THINKSAFE in Figs. 9 and 10. Here, Nh and Nb denote the numbers of harmful and benign prompts, respectively, while µh and µb represent the average response lengths (in tokens) for harmful and benign queries.

Across both the Qwen3 and R1-distilled model series, benign responses consistently exhibit longer generation lengths than harmful ones, reflecting the presence of more detailed reasoning traces. Moreover, as model size increases, both harmful and benign responses tend to become longer and more stable in distribution.

###### Qwen3-0.6B

###### Qwen3-1.7B

###### Qwen3-4B

###### Qwen3-8B

Nh=17,024 Nb=21,875

Nh=17,300 Nb=21,906

Nh=17,825 Nb=21,973

Nh=17,860 Nb=21,948

3000

3000

2500

3000

h=555 b=1657

h=442 b=866

h=657 b=1575

h=552 b=1611

Frequency

Frequency

Frequency

Frequency

2000

2000

2000

2000

1500

1000

1000

1000

1000

500

0

0

0

0

500 1000 1500

0 500 1000 1500 2000 2500 3000

500 1000 1500 2000 2500 3000

0 500 1000 1500 2000 2500 3000

Number of Tokens

Number of Tokens

Number of Tokens

Number of Tokens

Harmful

Benign Harmful Mean Benign Mean

| |
|---|

- Figure 9: Statistics of THINKSAFE in Qwen3 model series. Top 1% outliers by length are excluded for better interpretability.

0 500 1000 1500 2000 2500

Number of Tokens

0

500

1000

1500

2000

2500

Frequency

Nh=15,592 Nb=19,674

h=648 b=1296

R1-Distill-1.5B

0 500 1000 1500 2000 2500

Number of Tokens

0

1000

2000

3000

Frequency

Nh=17,364 Nb=21,516

h=686 b=1359

R1-Distill-7B

0 500 1000 1500 2000 2500

Number of Tokens

0

1000

2000

3000

Frequency

Nh=17,802 Nb=21,934

h=624 b=1340

R1-Distill-8B

Harmful

| |
|---|

Benign Harmful Mean Benign Mean

- Figure 10: Statistics of THINKSAFE in DeepSeek-R1-Distill model series. Top 1% outliers by length are excluded for better interpretability.

### C Experimental Details

For the AIME 2024, GSM8K, MATH500, and GPQA datasets, we use the SkyThought [35] library to evaluate models. We sample 8 responses for each prompt using dataset-specific prompts and report the average pass@1. For the Qwen model family, we use a temperature of 0.6, top-p of 0.95, and top-k of 20, with a maximum token limit of 32,768. For the DeepSeek-R1-Distill family, we use a temperature of 0.6 and top-p of 0.95, with a maximum token limit of 32,768.

AIME 2024 Please reason step by step, and put your final answer within \\boxed{{}}. {problem}

#### GSM8K

Given the following problem, reason and give a final answer to the problem. Problem: {question} Your response should end with “The final answer is [answer]" where [answer] is the response to the problem.

#### GPQA

Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. {problem}

MATH500 Please reason step by step, and put your final answer within \\boxed{{}}. {problem}

### D Baseline Details

Table 4: Detailed hyperparameters for baselines.

Method Epochs Source Sample size DirectRefusal 5 WildJailbreak 1,000 SafeChain 3 WildJailbreak 40,000 STAR-1 5

Mixture of harmful datasets

(See 37 for details) 1,000 SafeKey 5 STAR-1 1,000 SafePath 4 WildJailbreak 400 THINKSAFE 3 SafeChain Varies by scale

To ensure consistency and reproducibility, we adopt the same hyperparameters as specified in the original papers for all baselines, as summarized in Table 4. For THINKSAFE, since the sample size varies across model scales, we refer readers to Figs. 9 and 10 for detailed configurations.

### E Additional Experiments

In this section, we present additional experiments to empirically support the effectiveness of the proposed THINKSAFE.

#### E.1 Necessity of safety reasoning on Qwen families

Qwen3

###### Safety Score ( )

Reasoning Score ( )

35%

###### HarmfulResponseRatio

29.6

77.2 78.5

78.3

30%

80%

27.0

65.7

25%

64.4

###### AvgPass@1

57.8

60%

19.0

20%

17.8

44.0

42.0

15%

40%

10%

6.5 6.5

20%

5.1 4.5

5%

0%

0%

0.6B 1.7B 4B 8B

0.6B 1.7B 4B 8B

ThinkSafe

w/o reasoning

| |
|---|

Figure 11: Ablation of safety reasoning in Qwen3 model series.

To further analyze the effect of reasoning on the Qwen family, we ablate reasoning traces from refusal responses yh(i), while retaining full chain-of-thought for benign responses yb(i). Both types of responses are generated by each model in the Qwen series. As shown in Fig. 11, removing reasoning traces consistently degrades both safety and reasoning performance. In particular, the Qwen3-4B model exhibits a substantial drop in reasoning accuracy, with Avg Pass@1 decreasing from 77.2% to 57.8%, accompanied by an increase in harmful response ratio. These results indicate that explicit reasoning plays a critical role not only in maintaining reasoning capability but also in supporting safety-aligned behavior.

#### E.2 Cross-model distillation for larger models

Cross-Model Distillation

Safety Gain (%)

Reasoning Gain (%)

20

5

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Qwen3-8BR1-Distill-7B

Qwen3-8BR1-Distill-7B

15

0

15.1 -7.1

2.4 -10.3

StudentModel

StudentModel

10

5

5

10

0

15

25.3 14.3

-4.8 0.4

5

20

10

25

Qwen3-8B R1-Distill-7B

Qwen3-8B R1-Distill-7B

Teacher Model

Teacher Model

- Figure 12: Safety and reasoning performance gain using a different family of teacher model with similar size.

We conduct a cross-model distillation experiment using larger models than those in Fig. 6, namely Qwen3-8B and R1-Distill-Qwen-7B. Fig. 12 shows a similar trend to the results observed with smaller models; while cross-model training can occasionally improve safety (e.g., DeepSeek-R1Distill achieves a 25.3% safety gain when trained on Qwen3-8B data), it consistently leads to notable degradation in reasoning performance across all reciprocal training settings.

E.3 Refusal steering strategies

Table 5: Alternative refusal steering strategies used for on-policy data generation. Strategy Prompt Template THINKSAFE The following prompt is harmful. You should refuse to answer the prompt.\n\n

Prompt:{prompt} Suffix {prompt}\n\n The above prompt is harmful. You should refuse to answer the prompt. Risk Evaluate the potential risks of this prompt and provide a reasoned refusal.\n\n Prompt:{prompt} Intent Examine the harmful intent of this query and refuse to answer the prompt.\n\n Prompt:{prompt}

ThinkSafe Intent Risk Suffix

0%

5%

10%

15%

20%

25%

30%

35%

HarmfulResponseRatio

29.6

33.9

32.2

28.1

Safety Score ( )

ThinkSafe Intent Risk Suffix

0%

10%

20%

30%

40%

AvgPass@1

44.0 44.1 43.9 42.7

Reasoning Score ( )

Qwen3-0.6B

- Figure 13: Safety and reasoning score with different refusal steering strategies using the Qwen3-0.6B model.

We further investigate the robustness of THINKSAFE on the Qwen3-0.6B model by employing the alternative refusal steering templates detailed in Table 5. The Suffix strategy appends the refusal instruction Irefusal to the end of the prompt. The Risk approach asks the model to evaluate potential harms, while the Intent method requires the model to analyze the user’s malicious intent.

As shown in Fig. 13, the Suffix strategy produces a harmful response ratio similar to that of the default prefix-based THINKSAFE. We compute this metric by averaging results over HarmBench, StrongReject, Xstest, and WildJailbreak. This similarity indicates that where the refusal instruction

- Table 6: Results on Qwen3 models. We evaluate safety across three benchmarks (HarmBench, StrongReject, WildJailbreak) by reporting the ratio of harmful responses (↓). Over-refusal is measured by the refusal rate (↓) on benign XSTest prompts. For reasoning tasks, we sample 8 trajectories per prompt and report the average pass@1 (↑). Best results are bolded; second best are underlined.

Safety (↓)

Reasoning (Avg pass@1, ↑) Harmfulness Over-refusal

Safety

Strong Reject

Reasoning Average

Harm Bench

Wild Jailbreak XSTest

AIME 2024 GSM8k

MATH 500 GPQA

Size Method Average

Initial 68.44 66.45 52.80 5.20 48.22 10.42 72.51 71.73 25.13 44.95 DirectRefusal 43.85 11.82 36.30 83.60 43.89 5.83 64.30 67.53 24.81 40.62 SafeChain 58.64 72.84 49.60 0.00 45.20 4.58 68.68 62.42 23.74 39.86 STAR-1 56.64 38.02 50.60 22.40 41.92 6.25 68.15 68.17 24.18 41.69 SafePath 67.61 60.06 52.80 4.40 46.22 7.92 71.26 71.77 26.07 44.26 SafeKey 60.96 48.88 52.75 18.40 45.25 5.42 71.58 66.17 24.94 42.03 THINKSAFE 40.37 33.87 37.95 6.40 29.65 9.58 72.36 70.65 23.30 43.97 THINKSAFE+WG 39.04 35.46 37.55 7.20 29.81 9.58 72.82 72.00 24.81 44.80

###### 0.6B

Initial 52.66 36.10 51.10 1.20 35.27 44.58 84.31 88.85 41.73 64.87 DirectRefusal 38.54 5.75 35.75 61.60 35.41 43.75 82.78 88.10 41.29 63.98 SafeChain 47.34 57.51 43.85 1.60 37.58 34.58 85.29 85.72 38.13 60.93 STAR-1 37.38 7.67 46.60 10.80 25.61 46.25 84.38 88.30 41.16 65.02 SafePath 54.15 36.42 49.30 1.20 35.27 43.33 84.33 88.32 42.42 64.60 SafeKey 46.84 18.21 48.85 8.80 30.68 38.33 84.31 88.12 40.03 62.70 THINKSAFE 28.74 9.58 29.20 2.00 17.38 44.17 83.80 89.05 40.53 64.39 THINKSAFE+WG 28.90 7.99 29.00 2.00 17.17 45.00 83.43 88.98 41.04 64.61

###### 1.7B

Initial 38.21 8.31 43.00 0.80 22.58 67.50 84.69 93.43 52.27 74.47 DirectRefusal 33.06 3.19 36.20 32.00 29.80 68.33 82.58 93.20 53.03 74.29 SafeChain 43.69 41.21 39.65 2.00 31.64 62.08 89.59 93.03 51.01 73.93 STAR-1 33.72 5.75 35.15 6.80 20.36 62.50 90.97 93.05 51.96 74.62 SafePath 37.71 7.35 42.45 1.60 22.28 72.08 84.45 93.33 53.54 75.85 SafeKey 32.39 3.19 32.95 0.80 17.33 67.08 91.79 92.87 51.83 75.89 THINKSAFE 9.63 0.32 7.45 2.80 5.05 73.33 88.06 93.53 53.79 77.18 THINKSAFE+WG 9.47 0.32 7.25 2.40 5.68 75.42 88.15 93.55 53.60 77.68

4B

Initial 35.05 4.47 38.35 0.40 19.57 74.17 85.28 94.18 50.69 76.08 DirectRefusal 24.42 1.92 28.05 37.60 23.00 74.58 84.31 93.63 59.41 77.98 SafeChain 41.20 38.95 36.42 1.20 29.44 70.00 92.98 93.53 58.21 78.68 STAR-1 24.42 1.28 29.25 6.80 15.44 72.50 90.29 93.73 57.83 78.59 SafePath 35.22 6.71 39.45 1.20 20.64 74.58 84.89 93.85 61.24 78.64 SafeKey 26.91 4.79 28.80 8.80 17.33 70.00 92.44 93.30 59.91 78.91 THINKSAFE 9.14 0.32 7.35 1.20 4.50 72.92 88.00 93.10 59.67 78.50 THINKSAFE+WG 9.63 0.32 7.05 3.20 5.05 73.33 87.96 93.65 60.29 78.81

8B

appears is not especially important, as straightforward refusal instructions work well regardless of placement. In contrast, the Risk and Intent strategies lead to noticeably worse safety outcomes, with higher harmful response ratios. We attribute this gap to the added complexity of these instructions. By asking the model to carry out extra reasoning steps instead of issuing a direct refusal, these prompts may weaken the strength of the safety constraint. Importantly, overall reasoning performance remains stable across all four strategies, as measured by average pass@1 on AIME2024, GSM8K, MATH500, and GPQA. This stability supports our central claim. Because all variants depend on self-generated outputs from the model itself rather than learning from an external model, the model’s core reasoning ability remains intact even when the safety approach changes.

#### E.4 Filtering with WildGuard

To assess whether the effectiveness of THINKSAFE depends on the specific characteristics of the filtering model, we conduct an ablation study using WildGuard instead of Llama-Guard-3. We denote this variant as THINKSAFE + WG, which employs WildGuard to filter the self-generated safety data. The results in Table 6 and Table 7 demonstrate remarkable stability in performance and confirm that our framework is robust to the choice of safety classifier. For example, the WildGuard variant of the Qwen3-4B model (THINKSAFE + WG) achieves safety scores nearly identical to the baseline while preserving superior reasoning capabilities. This consistency reinforces our core hypothesis that the success of THINKSAFE arises from the refusal steering mechanism itself rather than from overfitting to a specific reward model. By successfully filtering self-generated traces with a completely different guard model, we demonstrate that the elicited safety behaviors are generalized and transferable.

- Table 7: Results on DeepSeek-R1-Distill models. We evaluate safety across three benchmarks (HarmBench, StrongReject, WildJailbreak) by reporting the ratio of harmful responses (↓). Over-refusal is measured by the refusal rate (↓) on benign XSTest prompts. For reasoning tasks, we sample 8 trajectories per prompt and report the average pass@1 (↑). Best results are bolded; second best are underlined.

Safety (↓)

Reasoning (Avg pass@1, ↑) Harmfulness Over-refusal

Safety

Size Method Average

Harm Bench

Strong Reject

Wild Jailbreak XSTest

AIME 2024 GSM8k

MATH 500 GPQA

Reasoning Average

1.5B

Initial 67.28 82.11 51.55 0.00 50.23 21.25 82.42 79.45 31.94 53.77 DirectRefusal 66.11 82.75 50.45 8.40 51.93 19.17 81.06 78.55 32.70 52.87 SafeChain 59.30 76.68 46.95 0.40 45.73 24.17 80.47 81.25 28.28 53.54 STAR-1 62.79 77.00 49.65 1.20 47.66 17.08 81.38 79.07 31.25 52.20 SafePath 65.28 82.43 51.80 0.40 49.98 24.17 82.37 79.57 32.51 54.66 SafeKey 58.80 73.16 47.65 3.60 45.80 19.58 81.25 78.02 28.72 51.89 THINKSAFE 52.99 74.12 40.50 1.20 42.20 32.92 82.58 82.50 31.19 57.30 THINKSAFE + WG 54.82 76.68 39.95 1.20 43.16 30.00 82.36 82.55 32.89 56.95

- 7B

Initial 56.98 63.58 53.15 1.20 43.73 49.58 90.32 90.18 46.65 69.18 DirectRefusal 52.33 33.55 50.20 43.60 44.92 47.50 88.27 89.82 44.95 67.64 SafeChain 51.00 54.63 45.85 0.40 37.97 49.17 89.75 91.50 46.78 69.30 STAR-1 52.99 47.92 48.75 2.40 38.02 45.83 90.32 90.58 46.02 68.19 SafePath 55.15 64.86 52.65 0.00 43.16 52.08 89.71 90.62 46.15 69.64 SafeKey 45.35 33.87 45.75 7.20 33.04 43.75 90.58 89.90 47.29 67.88 THINKSAFE 40.20 41.85 35.40 0.40 29.46 51.25 90.10 91.90 45.20 69.61 THINKSAFE + WG 40.03 38.98 34.25 0.80 28.52 52.50 90.98 92.05 46.97 70.63

- 8B

Initial 52.33 53.99 49.70 0.40 39.10 47.50 87.74 87.38 48.11 67.68 DirectRefusal 32.39 0.64 32.60 50.00 28.91 40.00 83.26 85.00 43.50 62.94 SafeChain 44.52 46.33 42.45 1.60 33.72 41.67 86.06 86.50 42.05 64.07 STAR-1 21.26 3.51 17.60 12.00 13.59 40.42 87.28 86.65 43.69 64.51 SafePath 47.51 51.44 50.20 0.40 37.39 43.33 87.45 87.43 47.41 66.41 SafeKey 32.72 11.82 30.85 8.00 20.85 35.83 87.41 85.80 42.49 62.88 THINKSAFE 27.08 26.52 21.15 1.60 19.09 48.75 87.55 87.70 46.28 67.47 THINKSAFE + WG 27.24 26.20 22.00 1.60 19.26 45.42 87.62 86.82 45.14 66.25

- Table 8: Results on Qwen3-14B and Qwen3-32B models. We evaluate safety across three benchmarks (HarmBench, StrongReject, WildJailbreak) by reporting the ratio of harmful responses (↓). Over-refusal is measured by the refusal rate (↓) on benign XSTest prompts. For reasoning tasks, we sample 8 trajectories per prompt and report the average pass@1 (↑). Best results are bolded; second best are underlined.

Safety (↓)

Reasoning (Avg pass@1, ↑) Harmfulness Over-refusal

Safety

Strong Reject

Reasoning Average

Harm Bench

Wild Jailbreak XSTest

AIME 2024 GSM8k

MATH 500 GPQA

Size Method Average

Initial 29.90 3.19 35.05 1.60 17.44 79.17 92.58 94.67 63.83 82.56 DirectRefusal 21.59 0.64 23.15 35.20 20.15 77.08 92.69 94.47 62.82 81.77 SafeChain 46.01 46.65 38.50 1.20 33.09 69.17 91.55 94.27 61.24 79.06 THINKSAFE 6.31 0.00 3.45 2.40 3.04 77.50 93.20 94.27 63.51 82.12

14B

Initial 35.71 5.11 36.45 1.60 19.72 79.17 87.82 95.15 68.62 82.69 DirectRefusal 18.44 0.32 21.05 39.20 19.75 79.58 87.51 95.03 67.36 82.37 SafeChain 42.86 46.65 36.45 0.80 31.69 72.92 91.79 94.35 60.23 79.82 THINKSAFE 10.47 0.32 7.15 4.40 5.58 79.58 89.13 94.95 67.74 82.85

32B

#### E.5 Experiments with Larger Models

To verify that THINKSAFE continues to scale beyond the 8B regime reported in Table 1, we additionally apply our framework to Qwen3-14B and Qwen3-32B, with results summarized in Table 8. THINKSAFE consistently achieves the most favorable safety-reasoning trade-off at both scales. On Qwen3-14B, it reduces average harmfulness from 17.44 to 3.04 (a ∼5.7× reduction) while preserving reasoning within 0.44 points of the initial model (82.56 → 82.12). On Qwen3-32B, THINKSAFE cuts average harmfulness from 19.72 to 5.58 and simultaneously improves average reasoning from 82.69 to 82.85, outperforming all baselines on both axes. In contrast, teacher-distillation with SafeChain degrades reasoning substantially (79.06 at 14B and 79.82 at 32B), and DirectRefusal exhibits severe over-refusal on benign XSTest prompts (35.20% and 39.20%, respectively). These results confirm that the KL-optimal target p+ref established in Sec. 4 remains empirically realizable at larger scales, and that the advantages of self-generation over external supervision grow rather than diminish as student capacity increases.

Table 9: Controlled comparison with SafePath using Qwen3-8B.

Safety (↓)

Reasoning (Avg pass@1, ↑) Harmfulness Over-refusal

Safety

Strong Reject

Reasoning Average

Harm Bench

Wild Jailbreak XSTest

AIME 2024 GSM8k

MATH 500 GPQA

Method Average

SafePath 21.76 2.24 20.00 2.00 11.50 73.33 85.57 92.93 61.74 78.39 THINKSAFE 9.14 0.32 7.35 1.20 4.50 72.92 88.00 93.10 59.97 78.50

#### E.6 Controlled Comparison with SafePath

Throughout the experiment and analyses, we strictly follow the original training configuration of each baseline to ensure a fair comparison under its intended setting. However, since the number of training examples differs across baselines (see Table 4), we further conduct a controlled comparison with matched data scale. We use SafePath as the reference, as it uses the smallest training set among the baselines, and train it with the same prompt set as THINKSAFE. Following the original SafePath protocol, we add the safety primer "<think> Let’s think about safety first." only to harmful prompts.

As shown in Table 9, the controlled SafePath baseline remains substantially less effective under the matched data scale. THINKSAFE preserves reasoning while more than halving the average harm rate (4.50% vs 11.50%) compared to the controlled SafePath, confirming that native self-generated refusal traces provide superior safety grounding.

### F Online Learning Baseline Details

#### F.1 GRPO Objective

The objective function for GRPO is formulated to optimize the policy without a separate value function, instead estimating advantages in a group-relative manner as introduced in Shao et al. [30]. Given a dataset D of prompts x, GRPO samples a group of G candidate responses {yi}Gi=1 from a behavior policy πθ

, while constraining updates to remain close to a fixed reference policy πref. Then, the advantage Aˆi for each response yi in a group of size G is computed by normalizing the rewards ri against the group’s mean and standard deviation:

old

ri − mean({r1,...,rG}) std({r1,...,rG})

Aˆi =

(9)

These advantages are used in a clipped objective with importance sampling, together with a KL penalty term:

J (θ) = E

G

1 G

i=1

min

πθ(yi | x) πθ

Aˆi,clip

(yi | x)

old

πθ(yi | x) πθ

,1 − ϵ,1 + ϵ A ˆi

(yi | x)

old

− β KL(πθ ∥πref)

#### F.2 Safety Reward

The safety reward rsafety is derived from the output logits of the safety guard model. Specifically, given a prompt x and a sampled response y, we extract the log probabilities of the tokens corresponding to “safe" (c = 1) and “unsafe" (c = 0) from the guard model’s prediction. The final reward is computed by applying a softmax function over these two token probabilities to represent the likelihood of the response being safe:

pφ(c = 1 | x,y) pφ(c = 1 | x,y) + pφ(c = 0 | x,y)

(10)

rsafety(x,y) =

The rsafety ensures a continuous reward signal between 0 and 1, reflecting the guard model’s confidence in the safety of the generated trace. Subsequently, the rsafety is combined with the rformat to constitute the total reward r, which is then utilized in Eq. 9.

#### F.3 Format Reward

The format reward, rformat, strictly enforces the structural integrity of the reasoning traces. For standard models, we assign a reward of 1 if the response contains exactly one pair of <think> and </think> tags in the correct order, and 0 otherwise. For the DeepSeek-R1-Distill family, which omits the opening tag by default, we adapt this criterion to require exactly one occurrence of the closing </think> tag and no opening <think> tag. This ensures that the model maintains a consistent chain-of-thought structure during the online RL process.

#### F.4 Experimental Setup

For the GRPO baseline, we utilize the TRL [36] library integrated with vLLM [14] for efficient online rollout generation. We generate G = 8 rollouts per prompt to estimate the group-relative advantage. The DKL coefficient is fixed at β = 0.04 and the clipping parameter ϵ is set to 0.2. To ensure a consistent and fair comparison with THINKSAFE, all other experimental details including the optimizer, batch size, and hardware configuration are maintained identical to the primary experimental setup described in Sec. 5.

For the On-Policy Distillation baseline, we implement a custom trainer on top of the TRL [36] base trainer that uses vLLM [14] for faster online generation. The student first generates on-policy rollouts, and we then optimize it with a full-vocabulary backward KL loss against the teacher distribution along the generated trajectories. Due to the substantial latency of repeated rollout generation and teacher scoring, we set the maximum output length to 4096 tokens. This budget is sufficiently large for our setting, as most generated responses fall within this range, as shown in Fig. 9. All remaining optimization and hardware configurations are kept identical the primary experimental setup.

### G Limitations

Dependence on latent safety knowledge. Our method fundamentally relies on the hypothesis that the student model retains latent knowledge to identify harm, which refusal steering unlocks (formalized in Assumption 4.3 as the refusal tilt ω(xh) > 1). This assumption holds for modern post-trained reasoning models such as Qwen3 and DeepSeek-R1-Distill, which have undergone prior safety alignment. However, for base models that have never been exposed to safety training, the refusal-oriented instruction may fail to elicit valid refusal traces, as there would be little latent safety capability to surface. In such settings, external teacher supervision or RL with safety rewards may remain necessary as an initial step.

Reliance on a safety guard model. THINKSAFE depends on an external safety classifier (LlamaGuard-3) to filter self-generated responses. Although our ablation with WildGuard (Sec. E.4) demonstrates robustness to the specific choice of guard model, the overall quality of the resulting dataset is bounded by the accuracy of the filter. False negatives (unsafe responses labeled as safe) could propagate subtle misalignments into the student, while false positives may discard valid training signals. Advances in safety classifiers would directly benefit our framework.

Scale of evaluation. Our experiments cover model sizes from 0.6B to 8B parameters across two model families. While these cover a representative range of open-source reasoning models, the behavior of THINKSAFE at larger scales (e.g., 70B+) or on closed frontier models remains to be validated. The filtering ratios in Table 3 suggest that larger models retain more of the generated data, indicating that the method may become increasingly effective at scale, but confirming this requires further study.

LoRA-based fine-tuning. Following prior work, we adopt LoRA rather than full parameter finetuning. While this choice helps preserve the model’s intrinsic capabilities, it also limits the scope of our conclusions regarding distributional discrepancy. Full fine-tuning may exhibit different trade-offs, and we leave a comprehensive comparison to future work.

Single-turn refusal. We focus on single-turn harmful prompts and do not explicitly address multiturn jailbreaks, adversarial prefixes embedded in long contexts, or agentic settings where safety

must be maintained across tool calls. Extending refusal steering to these scenarios is an important direction.

Static offline dataset. We approximate the idealized on-policy objective with a static offline dataset generated once from the initial student. As the student’s distribution shifts during fine-tuning, the generated data becomes increasingly off-policy. Iterative self-training, where refusal steering is re-applied to the updated student, may further close this gap at additional computational cost.

### H Broader Impacts

Positive impacts. THINKSAFE contributes to the responsible deployment of large reasoning models by mitigating the well-documented safety regression that accompanies reasoning-oriented posttraining. By eliminating the dependence on external teacher models, our method lowers the barrier to safety alignment for practitioners who may not have access to large proprietary teachers, and enables smaller research groups and open-source communities to realign reasoning models efficiently. The order-of-magnitude reduction in compute relative to online RL (GRPO) also carries environmental benefits by reducing the energy cost of safety training. More broadly, our theoretical analysis clarifies why self-generated data can preserve native capabilities better than teacher distillation, which may inform future alignment research beyond the safety domain.

Potential negative impacts and mitigation. Research on safety alignment inherently involves working with harmful prompts and jailbreak datasets. We rely on publicly available benchmarks (SafeChain, HarmBench, StrongReject, WildJailbreak, XSTest) that have been vetted by the research community, and we do not generate or release new harmful prompts. The self-generated responses we release consist of refusals and benign reasoning traces, which carry minimal misuse risk. Nevertheless, we acknowledge several concerns:

Dual-use of the method. In principle, the refusal-steering mechanism could be inverted (e.g., with a compliance-oriented instruction) to generate unsafe training data [16]. We note that such attacks are strictly easier without our framework, since reasoning models already comply with many harmful prompts by default, and thus THINKSAFE does not meaningfully expand the attacker’s capabilities. Practitioners releasing aligned models should continue to apply standard safeguards such as input and output filtering at deployment.

Over-refusal. Any safety training carries the risk of inducing exaggerated refusal on benign prompts. We explicitly monitor this with XSTest and find that THINKSAFE maintains low over-refusal rates (often below the initial model’s), but practitioners should continue to evaluate deployed models on their target distribution of benign queries.

False sense of security. Our method reduces but does not eliminate harmful responses. Models aligned with THINKSAFE should not be treated as adversarially robust, and deployment in high-stakes settings should involve additional safeguards, red-teaming, and monitoring.

Responsible release. We release our code, self-generated datasets, and model adapters under researchoriented licenses, with the aim of enabling reproducibility and further safety research. All released artifacts consist of safe refusals and benign reasoning traces filtered by Llama-Guard-3, and we do not release any model checkpoints that could amplify harm beyond their base models.

### I LLM Usage

LLMs as research subjects. The core methodology of this work involves large language models as the primary objects of study. Specifically, we use the Qwen3 family (0.6B, 1.7B, 4B, 8B) and the DeepSeek-R1-Distill family (1.5B, 7B, 8B) as student models, Llama-Guard-3-8B as the safety guard classifier φ, and WildGuard as an alternative classifier for our ablation. All of these models are publicly released and used in accordance with their respective licenses. No proprietary API-based LLMs were used to generate training data, labels, or evaluation judgments.

LLMs for data generation. The self-generated datasets used for THINKSAFE are produced by the student models themselves through refusal steering, as described in Sec. 3. No external teacher LLMs were used in the production of our training data, which is a central design choice of our framework.

LLMs for writing and editing. We used general-purpose LLM assistants for minor writing support, including grammar correction, LaTeX formatting, and rewording for clarity. All technical content, experimental design, theoretical analysis, proofs, and claims were authored and verified by the authors. LLMs were not used to generate research ideas, derive theoretical results, design experiments, produce figures, or draft substantive portions of the paper.

