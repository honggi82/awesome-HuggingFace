# arXiv:2606.18216v1[cs.CL]16Jun2026

[Figure 1]

2026-6-17

## Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients

Byung-Kwan Lee†, Ximing Lu, Shizhe Diao, Minki Kang, Saurav Muralidharan, Karan Sapra, Andrew Tao, Pavlo Molchanov, Yejin Choi, Yu-Chiang Frank Wang, Ryo Hachiuma

NVIDIA

| |
|---|

Figure 1 | Average accuracy gain (Δ, pp) from Qwen3.5 Family on 10 LLM (left), 16 VLM (middle), and 5 Video benchmarks (right). All methods are trained on the same multimodal dataset for vision-language models (VLMs) and evaluated on various benchmarks for generalization. † denotes augmentation with the prompt replay buffer.

### Abstract

Knowledge distillation transfers a teacher’s competence to a small student but is brittle in the small-student regime: forcing the student to imitate logits from a much larger teacher concentrates it on the teacher’s sharpest modes, hurting generalization on benchmark families beyond the training corpus. Reinforcement learning (RL) avoids logit imitation by training on the student’s own rollouts. However, on questions where every rollout fails—yielding zero advantage and being silently discarded—injecting a stronger teacher’s response into the policy gradient breaks the on-policy assumption and induces drift. We introduce Zone of Proximal Policy Optimization (ZPPO), inspired by Vygotsky’s zone of proximal development, which keeps the teacher inside the prompt rather than the policy gradient. On hard questions, ZPPO constructs two reformulated prompts: a Binary Candidate-included Question (BCQ) pairs one correct teacher response with one incorrect student response as anonymized candidates the student must discriminate, and a Negative Candidate-included Question (NCQ) aggregates the student’s wrong rollouts into a single prompt to surface their shared failure modes. A prompt replay buffer recirculates each hard question until it either graduates – the student’s mean rollout accuracy on it reaches half – or is FIFO-evicted under finite capacity, amplifying BCQ and NCQ inside the student’s current zone of proximal development. On the Qwen3.5 family at four student scales (0.8B–9B) with a 27B teacher, post-trained as vision–language models and evaluated on a 31-benchmark suite (16 VLM, 10 LLM, 5 Video), ZPPO outperforms off/on-policy distillation and GRPO, with the largest gains at the smallest scale. Our project page is available at link.

†Project Lead, byungkwanl@nvidia.com © 2026 NVIDIA. All rights reserved.

[Figure 2]

- Figure 2 | (a) Two failure modes that motivate ZPPO. In the small-student regime, fitting the student to a much larger teacher’s logits is brittle; on the RL side, injecting a teacher response into the policy gradient as if it were on-policy introduces drift. (b) BCQ recovers a learning signal on hard questions by pairing a correct teacher response with a wrong student response as anonymized in-prompt candidates, asking the student to reason about which one is correct. (c) NCQ aggregates the student’s own wrong rollouts into a single prompt, surfacing shared error patterns.

### 1. Introduction

Large language models (LLMs) and vision–language models (VLMs) have made rapid progress [1–5] on broad benchmarks, with reinforcement learning post-training [5–10]. The cost of those gains, however, has scaled with the cost of the model: tens of billions of activated parameters. Nearly all of the gains we see at the frontier do not transport to the regime that matters for deployment on mobile phones, AR/VR glasses, or embodied robots [11–16], where compute and memory budgets are much smaller.

Closing this deployment gap requires small models that generalize broadly, not narrow specialists. The standard knowledge-transfer recipe is knowledge distillation: a small student is trained to imitate the logits, hidden states, or sampled outputs of a strong teacher [17–21]. Despite its effectiveness, distillation carries a structural limitation that becomes severe when the student is much smaller than the teacher. Logit imitation is, at its core, supervised fitting on the teacher’s distribution: with limited training data, students memorize teacher answers [22–24], and the limited capacity (e.g., “mode-seeking bias”) concentrates the student on the teacher’s sharpest peaks [20, 25]. The result is a student whose generalization on benchmark families beyond the training corpus is brittle, and whose responses rarely venture beyond the paths the teacher has already taken (Tab. 2).

A line of work tries to alleviate these issues by using the student’s own rollouts to conduct distillation from the teacher to the student (on-policy distillation [25–31]) or by removing the external teacher and letting the student teach itself (self -distillation [32–34, 22, 35]). These reduce the teacher–student capacity gap, but they all share the same fundamental constraint: the training signal is still a logit distribution that the student must match. For 0.8B or 2B students, model capacity is too limited to absorb such a distribution broadly, and the resulting models lose generalization on benchmarks beyond the narrow training data [22–24].

A natural alternative is reinforcement learning (RL) [5, 36, 37, 8, 7, 9, 10, 38–41], which avoids logit imitation entirely and trains directly on the student’s own rollouts. However, RL has its own pathology in the small-model regime: hard questions with zero successful rollouts produce zero

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

###### Question: {Question}

###### Question: {Question}

Rollouts

[Figure 9]

Here are two candidate responses In <candidate> </candidate> tags to the question above. One is correct and another is wrong.

The following answers are all WRONG: {Parsed Answer}. Below are all incorrect reasoning processes in <candidate> </candidate> tags.

✓

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

###### Batch

[Figure 14]

Teacher

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

###### ×

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

New Questions

<candidate> {One student’s wrong response} </candidate>

[Figure 26]

[Figure 27]

Database

[Figure 28]

Sample

×

[Figure 29]

[Figure 30]

[Figure 31]

Question

[Figure 32]

<candidate> {One teacher’s correct response} </candidate>

[Figure 33]

[Figure 34]

✓

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

|[Figure 40]<br><br>Replayed<br><br>questions|
|---|

[Figure 41]

[Figure 42]

× <candidate> {One student’s wrong response} </candidate>

[Figure 43]

[Figure 44]

[Figure 45]

Replay

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

×

Buffer

[Figure 51]

[Figure 52]

[Figure 53]

✓

Student

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

<candidate> {One student’s wrong response} </candidate>

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

###### ×

×

[Figure 63]

[Figure 64]

<candidate> {One student’s wrong response} </candidate>

[Figure 65]

[Figure 66]

×

[Figure 67]

[Figure 68]

[Figure 69]

###### If#ofCorrect responses<#ofWrongResponses, × Question will be admitted into Replay Buffer.

[Figure 70]

[Figure 71]

###### Assistant:

###### Assistant:

(a) Rollouts and Hard Question Admission to Replay Buffer

(b) Binary Candidate-included Question (BCQ)

(c) Negative Candidate-included Question (NCQ)

[Figure 72]

###### Batch

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

RL Recipe

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- 1. Higher Clip
- 2. Token-level loss
- 3. No KL Penalty
- 4. Lower Iteration
- 5. Batch-level Norm (w/o Zero Advantage)

[Figure 83]

✓

[Figure 84]

[Figure 85]

[Figure 86]

New Questions

[Figure 87]

Database

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Sample

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Question Student ✓

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Policy Gradient Loss

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

× ×

[Figure 117]

|[Figure 118]<br><br>Replayed questions|
|---|

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Replay Buffer

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

BCQ NCQ

[Figure 136]

[Figure 137]

Training

[Figure 138]

: Trainable

(d) Overview of ZPPO

- Figure 3 | Overview of ZPPO. (a) Hard questions (mean rollout accuracy below half) are admitted to the prompt replay buffer. (b) BCQ pairs one correct teacher response with one wrong student response. (c) NCQ aggregates the student’s wrong rollouts into a single prompt. (d) Integrated batch drives the policy gradient update with RL Recipe.

advantage and are silently discarded, so the student never learns from the questions where it most needs help. A simple fix – using a stronger teacher’s correct response as if it were the student’s – breaks the on-policy assumption and induces policy drift, because the teacher’s response lies far outside the student’s rollout distribution. Fig. 2a summarizes these two failure modes.

In this paper, we ask a single question: how can we transfer the teacher’s knowledge to the student without ever imitating the teacher’s logits or injecting the teacher’s response directly into the student’s policy gradient? Our answer, which we call Zone of Proximal Policy Optimization (ZPPO), borrows its name from Vygotsky’s zone of proximal development [42] – the band of tasks a learner cannot yet solve alone but can solve with a small amount of guidance from a more capable peer. Concretely, ZPPO keeps the teacher only inside the prompt, never inside the student’s policy gradient (Fig. 2b,c and Fig. 3):

- • For each hard question (defined as one where the student’s mean rollout accuracy is below half), we construct a reformulated prompt – a Binary Candidate-included Question (BCQ) – that appends one correct teacher response and one wrong student response as shuffled, anonymized candidates, and asks the student to solve the problem afresh while consulting them. Since the student generates its own response, its policy gradient remains on-policy at the response level.
- • In parallel, we construct a Negative Candidate-included Question (NCQ) that appends all of the student’s wrong responses, explicitly shows the student that all of them are wrong, and asks for a correct answer. Within independent rollout groups, rollouts cannot “see” each other’s wrong answers; inside our training loop, NCQ is the first place at which a student’s failed attempts become collectively visible to the student, turning independent negatives into a shared signal.
- • Hard questions are also pushed into a prompt replay buffer (no responses are stored). They are

16 VLM Benchmarks AI2D BabyV CharXiv DynaM EmbSp InfoVQA MVerse MVision MVista MMMUPro MM-Vet OCREN OCRZH VisP VBlind WeMath Avg Qwen3.5-0.8B 65.6 6.7 54.3 17.8 67.9 68.6 43.5 16.4 60.7 26.8 53.2 40.0 17.0 20.5 42.8 54.4 41.0 + Off-Distill 66.3 6.7 55.1 19.4 65.8 66.0 45.3 17.3 62.0 27.6 51.5 38.9 16.2 20.0 41.8 55.9 41.0 + Off-Distill† 66.7 6.7 56.0 20.1 65.8 66.8 45.8 17.9 62.2 29.0 51.8 39.5 15.9 20.8 42.2 56.4 41.5 + On-Distill 67.6 7.5 56.0 20.0 66.7 67.1 45.8 19.4 62.7 28.2 52.7 39.6 16.7 21.2 41.5 57.7 41.9 + On-Distill† 68.6 7.8 56.4 21.1 67.1 67.6 47.4 18.7 63.6 28.8 52.5 39.1 16.7 20.9 42.3 58.6 42.3 + GRPO 69.5 8.6 57.0 21.3 69.2 71.0 47.7 20.1 65.2 29.9 55.6 40.9 17.0 24.2 43.2 60.6 43.8 + GRPO† 71.2 9.8 59.9 23.6 69.4 72.4 51.1 20.9 68.3 30.5 57.5 41.3 17.5 27.8 43.6 62.5 45.4 + ZPPO 76.5 13.9 63.9 31.1 71.5 75.3 59.3 29.2 73.2 37.6 59.9 42.5 18.7 35.0 44.7 71.7 50.3 Δ +5.3 +4.1 +4.0 +7.5 +2.1 +2.9 +8.2 +8.3 +4.9 +7.1 +2.4 +1.2 +1.2 +7.2 +1.1 +9.2 +4.9

Qwen3.5-2B 81.9 11.6 71.6 41.1 78.2 81.2 69.7 38.4 78.6 46.2 69.7 44.7 24.0 38.3 55.2 77.9 56.8 + Off-Distill 82.5 11.9 71.0 43.2 77.2 80.2 70.8 40.4 77.9 47.4 69.2 44.4 23.0 38.4 53.7 79.5 56.9 + Off-Distill† 82.9 12.5 71.4 43.7 77.4 80.4 71.4 41.8 77.8 47.9 69.2 44.5 23.7 37.8 54.6 79.3 57.3 + On-Distill 82.6 13.1 71.7 44.5 77.6 81.1 71.9 41.9 78.2 48.8 69.2 45.3 23.9 38.8 54.9 79.3 57.7 + On-Distill† 83.4 12.8 71.7 44.9 77.5 80.9 72.0 42.7 77.9 49.3 69.4 45.5 24.5 38.8 55.6 80.6 58.0 + GRPO 83.5 13.7 72.8 45.6 78.3 82.4 72.3 43.1 79.0 49.3 72.3 46.1 24.9 39.2 56.5 80.4 58.7 + GRPO† 83.8 14.4 73.0 45.9 78.7 83.1 72.8 43.4 79.3 49.6 74.0 46.4 25.2 40.3 57.0 80.7 59.2 + ZPPO 85.3 18.6 73.9 52.7 79.5 84.6 76.0 50.5 80.5 53.2 77.1 48.8 26.0 42.0 60.5 82.6 62.0 Δ +1.5 +4.2 +0.9 +6.8 +0.8 +1.5 +3.2 +7.1 +1.2 +3.6 +3.1 +2.4 +0.8 +1.7 +3.5 +1.9 +2.8

- Table 1 | Main results on the 16 VLM benchmarks at 0.8B and 2B; † denotes augmentation with the prompt replay buffer, and Δ reports ZPPO minus the strongest non-ZPPO method in each column on the same model size. 4B and 9B results: Tab. 10. Benchmark abbreviations: Appendix B.2.

10 LLM Benchmarks 5 Video Benchmarks

Method AIME25 AIME26 CEval GPQA-D HLE IMO-AB MMLU MMLU-Pro MMLU-Rd MultiCh Avg MMVU MVBench VMME VMMES VMMMU Avg Qwen3.5-0.8B 3.3 0.0 52.3 12.1 2.2 3.3 57.7 41.7 62.5 16.6 25.2 39.0 51.6 50.3 51.6 48.8 48.3 + Off-Distill† 0.0 0.0 47.9 10.1 2.1 2.9 52.6 38.3 56.9 14.7 22.5 35.4 48.9 46.8 48.5 45.4 45.0 + On-Distill† 0.0 0.0 49.2 10.6 2.8 4.0 53.8 38.8 58.2 14.9 23.2 36.8 49.2 47.8 49.6 45.7 45.8 + GRPO† 3.3 0.0 54.5 25.5 3.5 7.0 59.9 47.1 65.1 20.8 28.7 41.3 53.5 52.2 53.0 52.5 50.5 + ZPPO 3.3 0.0 57.6 42.4 5.0 10.5 63.0 51.7 69.0 28.6 33.1 42.5 56.6 53.9 54.9 56.4 52.8 Δ 0.0 0.0 +3.1 +16.9 +1.5 +3.5 +3.1 +4.6 +3.9 +7.8 +4.4 +1.2 +3.1 +1.7 +1.9 +3.9 +2.3

Qwen3.5-2B 26.7 30.0 71.1 53.5 3.5 11.3 74.6 66.2 80.4 35.8 45.3 53.7 60.7 59.9 61.2 67.6 60.6 + Off-Distill† 26.7 26.7 68.2 50.3 2.0 10.9 71.2 63.8 77.5 34.0 43.1 51.5 59.2 57.9 59.0 65.4 58.6 + On-Distill† 23.3 30.0 69.3 51.9 3.8 11.5 72.2 63.8 77.9 34.0 43.7 52.4 59.9 58.6 59.4 65.5 59.2 + GRPO† 26.7 33.3 72.4 56.5 4.2 19.3 75.4 67.3 81.4 37.3 47.3 56.7 61.7 61.2 62.6 67.3 61.9 + ZPPO 26.7 40.0 74.0 60.6 5.1 29.5 76.4 68.2 82.9 40.6 50.4 59.0 62.7 62.4 63.9 68.1 63.2 Δ 0.0 +6.7 +1.6 +4.1 +0.9 +10.2 +1.0 +0.9 +1.5 +3.3 +3.1 +2.3 +1.0 +1.2 +1.3 +0.5 +1.3

- Table 2 | Evaluation on the 10 LLM and 5 Video benchmarks (probing generalization beyond the multimodal training data). 4B and 9B results: Tab. 11.

repeatedly sampled until they either graduate (rollout accuracy reaches half, i.e. 𝑟¯𝑥≥0.5) or are FIFO-evicted once the buffer reaches capacity (Sec. 3.3, Appendix F.2). This focuses repeated practice precisely on prompts inside the student’s zone of proximal development.

These three components (BCQ, NCQ, and prompt replay buffer) are layered on top of GRPO [36] augmented with DAPO’s clip-higher and token-level policy gradient loss [8]. We also adopt two simple but easily mishandled recipe choices that materially affect a small model’s training dynamics: iterations per step and batch-level advantage normalization of REINFORCE++ [37], with zeroadvantage groups excluded from the batch statistics. Both are low-cost, yet ablating them accounts for a sizeable fraction of the performance drop we observe, despite their simplicity.

We instantiate ZPPO on the Qwen3.5 family at four student scales – 0.8B, 2B, 4B, and 9B – with a 27B teacher [4], post-train each student as a VLM on a multimodal RL dataset of image–question pairs, and evaluate on a 31-benchmark suite spanning 16 VLM, 10 LLM, and 5 Video benchmarks. At every scale, ZPPO outperforms (i) off- and on-policy distillation and (ii) GRPO (Fig. 1), and ablating any single component – BCQ, NCQ, or the prompt replay buffer – strictly degrades the recipe at every scale (Tab. 3, Tab. 12). The improvements are largest precisely where the small-model gap is widest – a 0.8B student gains +9.3 pp on the VLM benchmarks, while a 9B student gains +2.8 pp – and, crucially, ZPPO also improves on the LLM and Video benchmark families that lie beyond the training corpus, the very families where distillation hurts.

Entry Rollout Accuracy at Admission: 0% Entry Rollout Accuracy at Admission: 12.5% Entry Rollout Accuracy at Admission: 25% Entry Rollout Accuracy at Admission: 37.5%

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

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

- Figure 4 | Cumulative graduate counts (graduated/admitted = ratio) for ZPPO vs. GRPO† at 2B by entry rollout accuracy at admission; † denotes augmentation with the prompt replay buffer.

Distillation hurts generalization beyond the training corpus; ZPPO does not

16 VLM benchmarks (training-aligned): Distillation lifts the student only marginally – averaging Off- and On-policy variants gives +0.9/+0.9/+0.7/+0.6 pp at 0.8B/2B/4B/9B (Tab. 1, Tab. 10), while ZPPO gains +9.3/+5.2/+4.0/+2.8 pp.

10 LLM and 5 Video benchmarks (beyond the training corpus): Distillation degrades the same students by −2.5/−1.8/−0.9/−0.3 pp (Tab. 2, Tab. 11), whereas ZPPO improves them by +6.8/+4.3/+2.7/+2.7 pp.

### 2. Related Work

Three lines of work bear on ZPPO. Knowledge distillation—whether off-policy, on-policy, or selfdistillation [18, 21, 25, 26, 43, 32]—shares a single training signal, a logit distribution the student must match, which recent analyses show becomes brittle when the student is much smaller than the teacher, inducing memorization and mode-seeking [22–24]. RL post-training with rule-based rewards on PPO and group-relative variants [44, 36, 8, 5] has advanced reasoning, but leaves the small-student failure mode where every rollout fails and the group advantage is zero [8] unaddressed; hybrid RL+distillation fixes [45, 46] target this mode, but route teacher responses into the student’s policy gradient through an auxiliary distillation term, which can re-introduce drift. Prompt replay, curriculum, and hint scaffolding [47–52] re-prioritize or scaffold hard questions, but each path compromises a different channel: prompt replay and curriculum [47–49] re-expose the same hard prompt without adding any new signal about why the previous attempt failed; hint methods [52] append a directional cue (without the answer) that the student tends to copy as a shortcut without necessarily internalizing why the direction is correct; and prefix methods [50, 51] insert part of the teacher’s response as a prefix into the student’s rollout: the student is trained on off-policy trajectories conditioned on teacher response. ZPPO differs from all three by moving teacher knowledge from the gradient to the prompt via two reformulations (BCQ and NCQ), so every response is sampled from the student’s own policy, with the prompt replay buffer amplifying inside the student’s zone of proximal development [42]. (see Appendix A for full version)

### 3. Zone of Proximal Policy Optimization

ZPPO is built on top of GRPO [36]. We first set up notation and identify the precise failure mode that motivates ZPPO (Sec. 3.1); we then describe the two prompt reformulations BCQ and NCQ that recover a learning signal on hard questions (Sec. 3.2) and the prompt replay buffer that amplifies them (Sec. 3.3). How the reformulated rollouts plug into the training loop, together with two recipe-level choices on the backbone, is described at Sec. 4.1. The full ZPPO training step is summarized as Algorithm 1 in Appendix C.

#### 3.1. Preliminaries: GRPO’s Failure Mode

Let 𝑥 denote a question and 𝑦S a response sampled from a student policy 𝜋𝜃. For each 𝑥 we draw a group of 𝐺S student rollouts {𝑦S(𝑔)}𝐺𝑔=1S ∼ 𝜋𝜃(· | 𝑥) and assign each an outcome reward 𝑟(𝑥,𝑦S(𝑔)) ∈ {0,1} that signals whether the final answer is correct. Let 𝑟¯𝑥 and std𝑥 denote the withingroup mean and standard deviation of {𝑟(𝑥,𝑦S(𝑔))}𝐺𝑔=1S . The standard group-relative advantage [36, 8] is

𝑟(𝑥,𝑦S(𝑔)) − 𝑟¯𝑥 std𝑥 + 𝜖

𝐴(𝑔) =

. (1)

Eq. (1) is the textbook group-relative advantage that ZPPO conceptually builds on; the exact estimator used in our experiments is the two-step REINFORCE++ variant of Hu [37] (Step 1 subtracts the per-group mean; Step 2 batch-normalizes across the non-trivial groups), restated in our notation as Eqs. (2)–(3) in Appendix C. The student update applies the PPO surrogate [44] on top of 𝐴(𝑔). Either form leaves a blind spot for small students. Whenever a rollout group is all-wrong (¯𝑟𝑥=0) or all-correct (¯𝑟𝑥=1), every advantage in the group is exactly zero, so the question contributes no gradient signal at all. For a small student, the all-wrong case is exactly the set of questions that could be solved with teacher guidance. ZPPO’s goal is to recover a learning signal on those hard questions without ever placing a teacher response in the student’s gradient. We call 𝑥 a hard question when 𝑟¯𝑥 < 0.5 and use this single threshold throughout; the cutoff is not arbitrary, since under {0,1} rewards std𝑥 is maximized at 𝑟¯𝑥=0.5, where the group-relative advantage carries the strongest learning signal [7].

#### 3.2. Prompt Reformulation: BCQ and NCQ

Both BCQ and NCQ start from a hard question 𝑥 on which we have already drawn 𝐺S student rollouts {𝑦S(𝑔)}𝐺𝑔=1S . In parallel, we sample 𝐺T teacher rollouts on 𝑥 from a frozen teacher policy 𝜋T, score them with the same outcome reward, and keep the correct ones as the pool {𝑦T(+)} from which BCQ draws candidates. We use on-policy throughout in the response-level sense: every gradientcounted token is sampled from the current student. BCQ/NCQ prompts do contain teacher-derived text (correct and wrong candidate traces and, for NCQ, the parsed wrong-answer list), but this text is part of the input prompt and never enters the policy gradient as a target. Because we re-sample teacher rollouts every time 𝑥 is seen – whether new or replayed from ℬ – the candidates that BCQ uses change on every visit.

- (i) Candidate compression: Before any candidate enters a prompt, the frozen teacher rewrites it into a short reasoning trace under a shared compression prompt and a shared token cap (Appendix D); the same prompt and cap are applied to teacher-correct and student-wrong traces. The rewritten text still appears only inside the prompt.
- (ii) Binary Candidate-included Question (BCQ): For each hard question that admits at least

one correct teacher response, BCQ uniformly samples one 𝑦T(+) and one wrong student rollout 𝑦S(−), teacher-compresses both responses, anonymizes them inside identical <candidate> tags, randomly

shuffles the order, and appends the result to 𝑥 together with a single instruction (the verbatim code-side template, including the per-candidate <candidate> blocks, is reproduced in Appendix D.1):

Here are two candidate responses in <candidate> </candidate> tags to the question above. One is correct and another is wrong.

The student then samples a new rollout group {𝑦BCQ(𝑔) }𝐺𝑔=1S ∼ 𝜋𝜃(· | 𝑥BCQ) from the reformulated prompt; every response token is generated by the current student, so the policy gradient remains

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

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
| | | | | | | |

Zone of Proximal Policy Optimization

Easy BCQ NCQ

Grad Evict

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

(a) BCQ and NCQ Fraction Ratio in A Single Rollout Batch (b) Graduation and Eviction Ratio in Prompt Replay Buffer

- Figure 5 | (a) Composition of a single training rollout batch over the full run into Easy (¯𝑟𝑥 ≥ 0.5), BCQ, and NCQ shares, per student scale. (b) Cumulative graduation vs. FIFO-eviction ratio of the prompt replay buffer.

on-policy. The pedagogical effect comes from placing a correct teacher response and a wrong student response side by side – anonymized, shuffled, and presented without a correctness label – so the student is trained to select and reason between the two candidates without any explicitly labeled target entering the gradient.

#### (iii) Negative Candidate-included Question (NCQ): For each hard question, NCQ collects

every wrong student rollout 𝑦S(−) in the current group, parses out each rollout’s final answer and lists the parsed answers explicitly inside the prompt, and appends each teacher-compressed reasoning

trace as a <candidate> block, with the instruction (verbatim code-side template in Appendix D.1):

The following answers are all WRONG: ⟨ parsed answer ⟩. Below are the incorrect reasoning processes in <candidate> </candidate> tags.

The student then samples a new rollout group {𝑦NCQ(𝑔) }𝐺𝑔=1S ∼ 𝜋𝜃(· | 𝑥NCQ). As with BCQ, every 𝑦NCQ(𝑔) is entirely student-generated, so the policy gradient is on-policy at the response level. The pedagogical role of NCQ, however, is structurally different. In a standard rollout group, each wrong rollout contributes its own advantage to the student’s policy gradient independently, and – within independent rollout groups – the student has no way to see patterns across its failures. Within our training loop, NCQ is the first place at which independently sampled wrong rollouts on the same question converge into a single prompt: confronted with its own failed attempts, the student is cued to recognize the shared error patterns and avoid them.

#### 3.3. Integration with Prompt Replay Buffer

The prompt replay buffer ℬ exists solely to amplify BCQ and NCQ on questions the student has not yet mastered. It stores only the question 𝑥 (image and text), never any rollout responses.

- (i) Admission and graduation: After each training step we update ℬ from the current rollout batch: a question 𝑥 is admitted if its 𝑟¯𝑥 <0.5, and an already-admitted 𝑥 is graduated (removed) on any later step where 𝑟¯𝑥 reaches half (¯𝑟𝑥≥0.5). Because BCQ and NCQ are constructed only on hard questions, every replayed question is eligible for one or both reformulations on its next visit. The buffer therefore always tracks the student’s current zone of proximal development.
- (ii) Sampling and capacity: Each rollout batch combines new questions from the data loader

with replay samples drawn uniformly from ℬ, where the replay count is a fixed fraction 𝜌replay of the new-question count. From the union of new and replayed questions, BCQ and NCQ are constructed on the hardest ones first (ranked by ascending 𝑟¯𝑥); the combined BCQ+NCQ count per rollout step is then capped at a fraction 𝜌aug of the new-question count (Appendix C). The buffer therefore

16 VLM Benchmarks AI2D BabyV CharXiv DynaM EmbSp InfoVQA MVerse MVision MVista MMMUPro MM-Vet OCREN OCRZH VisP VBlind WeMath Avg Qwen3.5-0.8B 65.6 6.7 54.3 17.8 67.9 68.6 43.5 16.4 60.7 26.8 53.2 40.0 17.0 20.5 42.8 54.4 41.0 + GRPO 69.5 8.6 57.0 21.3 69.2 71.0 47.7 20.1 65.2 29.9 55.6 40.9 17.0 24.2 43.2 60.6 43.8 + GRPO† 71.2 9.8 59.9 23.6 69.4 72.4 51.1 20.9 68.3 30.5 57.5 41.3 17.5 27.8 43.6 62.5 45.4 + GRPO + Both 71.7 9.8 58.7 23.9 69.1 72.2 50.7 21.3 67.1 31.7 56.6 40.1 17.1 26.6 43.6 63.3 45.2 + GRPO† + BCQ 75.2 12.3 61.8 28.2 71.1 74.5 55.9 26.4 71.7 35.2 59.1 42.2 18.4 31.8 44.3 69.6 48.6 + GRPO† + NCQ 72.0 10.2 60.3 24.3 70.0 73.0 51.5 22.7 68.7 32.1 57.9 41.6 18.1 28.2 44.0 64.6 46.2 + ZPPO 76.5 13.9 63.9 31.1 71.5 75.3 59.3 29.2 73.2 37.6 59.9 42.5 18.7 35.0 44.7 71.7 50.3 Qwen3.5-2B 81.9 11.6 71.6 41.1 78.2 81.2 69.7 38.4 78.6 46.2 69.7 44.7 24.0 38.3 55.2 77.9 56.8 + GRPO 83.5 13.7 72.8 45.6 78.3 82.4 72.3 43.1 79.0 49.3 72.3 46.1 24.9 39.2 56.5 80.4 58.7 + GRPO† 83.8 14.4 73.0 45.9 78.7 83.1 72.8 43.4 79.3 49.6 74.0 46.4 25.2 40.3 57.0 80.7 59.2 + GRPO + Both 83.9 14.3 72.7 46.0 78.1 82.8 72.4 42.7 79.4 49.8 73.2 45.4 24.5 39.6 57.7 79.8 58.9 + GRPO† + BCQ 84.7 16.7 73.7 49.6 79.2 84.0 74.3 47.3 80.1 51.4 75.8 48.1 25.6 41.1 59.1 81.8 60.8 + GRPO† + NCQ 84.3 15.8 73.4 48.1 79.0 83.6 73.5 45.7 79.7 50.5 74.9 47.6 25.5 40.9 58.4 81.2 60.1 + ZPPO 85.3 18.6 73.9 52.7 79.5 84.6 76.0 50.5 80.5 53.2 77.1 48.8 26.0 42.0 60.5 82.6 62.0

- Table 3 | Component ablation of ZPPO on the 16 VLM benchmarks at 0.8B and 2B. Each row adds one component (BCQ, NCQ, or † = prompt replay buffer); the highlighted last row of each block is ZPPO. 4B and 9B: Tab. 12.

re-exposes each hard question many times – with freshly sampled BCQ/NCQ candidates on every visit – until it either graduates or is FIFO-evicted once |ℬ| exceeds |ℬ|max. (see Appendix C for ZPPO algorithm)

- 4. Experiments

#### 4.1. Setup

Students and teacher. We instantiate ZPPO on Qwen3.5 [4] at four student scales (0.8B, 2B, 4B, 9B) with a 27B teacher, post-training each student as a VLM on the ZPPO-77K multimodal RL dataset (Appendix B).

Training pipeline. ZPPO follows GRPO [36] with three DAPO ingredients [8]: clip-higher, token-level policy gradient loss, and removal of the KL penalty against a reference policy. We also use 𝐼 =4 iterations per step (vs. the standard 𝐼 =16) and batch-level advantage normalization [37] with zero-advantage groups excluded from the batch statistics.

Baselines. We compare ZPPO against four families from Sec. 2. (i) Off-policy distillation (Algorithm 2): imitate fresh teacher rollouts drawn online and filtered by correctness on every step. (ii) On-policy distillation (Algorithm 3): the student samples its own response, the teacher re-scores it for target logits, and the student imitates them. (iii) GRPO (Algorithm 4): ZPPO’s pipeline without BCQ, NCQ, and the replay buffer. (iv) GRPO†: ZPPO’s pipeline without BCQ, NCQ († marks prompt replay buffer). Details in Appendix D.2.

#### 4.2. Main results

Tab. 1 (16 VLM benchmarks) and Tab. 2 (10 LLM and 5 Video benchmarks) report the headline comparison; 4B/9B blocks are in Tab. 10 and Tab. 11. Benchmark details are in Appendix B.2.

- (i) Distillation hurts generalization beyond the training corpus. The best distillation variant stays within ∼1 pp of the original on VLM, while Off- and On-Policy distillation lose accuracy on Video and degrade the LLM–Video average at every scale (Tab. 2, Tab. 11), sharpest at 0.8B.
- (ii) Replay alone is insufficient. Adding the buffer to either distillation or GRPO yields only marginal Avg gains in Tab. 1 (+0.5/+0.4/+1.6 pp at 0.8B for Off-Distill, On-Distill, GRPO): re-exposure without a new learning signal does not recover the hard question loss.

I

I

I

I

I

I

- Figure 6 | RL recipe ablation at 2B on the 10 LLM (a, d), 16 VLM (b, e), and 5 Video (c, f) benchmarks. (Top, a–c) Iterations per step 𝐼 ∈{1,4,8,16}. (Bottom, d–f) Batch-level advantage normalization: No norm, Norm w/o Zero (ZPPO default), and Norm w/ Zero. Dashed gray line marks original 2B.

- (iii) ZPPO improves the macro average at every scale and benchmark family. The Avg Δ row is positive across all blocks of Tab. 1 and Tab. 2, with the largest gains where the small-student gap is widest. The buffer-isolated ZPPO−GRPO† macro-average gain is robust to benchmark resampling, as quantified by the cluster bootstrap of Appendix F.6 (Tab. 14); this CI measures benchmark-selection robustness, not per-example or run-to-run uncertainty. Crucially, ZPPO improves generalization where distillation degrades it.

#### 4.3. Inside the Zone: Admission to Graduation

Fig. 4 reports cumulative graduations at 2B by entry rollout accuracy, and Fig. 5 reports batch composition and graduation across all four scales.

- (i) BCQ and NCQ provide a usable learning signal on the hardest questions (Fig. 4). On entries admitted at 0% rollout accuracy – where ordinary RL has zero gradient signal – ZPPO graduates 28% (432/1568) vs. GRPO†’s 4% (73/2035); on the next-hardest, 54% vs. 14%. The student absorbs the reformulated signals over training.
- (ii) Batch and graduation dynamics with scale (Fig. 5). Fig. 5a shows pre-cap BCQ/NCQ eligibility (Algorithm 1). BCQ requires the teacher to succeed where the student fails; as students scale up, the remaining hard questions are increasingly ones the teacher also fails on, so BCQ candidates dry up (Fig. 5a), thinning the recovery signal. In the replay buffer, small scales mostly evict before graduating, large scales mostly graduate before eviction (Fig. 5b). ZPPO’s gain is steepest where the zone of proximal development is widest. At 0.8B and 2B, hard questions persist long enough for BCQ and NCQ to revisit them repeatedly, compounding into the largest accumulated improvements; at 4B and 9B the same questions either graduate or rotate out before the reformulations can stack.

0.8B ZPPO 0.8B GRPO†+NCQ

2B ZPPO 2B GRPO†+NCQ

| |
|---|

| |
|---|

- 2

- 3

- 4

- 5

- 6

- 7

- 8

+7.2

AvgΔ(pp)

+5.1

+4.4

+4.3

+3.1

+2.8

4B 27B Teacher size

9B

Figure 7 | All-averaged gain Δ (pp) across teacher size

#### 4.4. Component ablation

Tab. 3 isolates each ZPPO component on the 16 VLM benchmarks (4B and 9B in Tab. 12). Starting from GRPO, we incrementally add the buffer (†), both reformulations together (Both = BCQ+NCQ), and each reformulation paired with the buffer.

- (i) Each isolated component is modest. Replay alone (GRPO†) and reformulation alone (GRPO+Both) move only a small margin over GRPO at every scale – useful, but not sufficient.
- (ii) Replay × reformulation is super-additive. Pairing BCQ or NCQ with the buffer compounds far beyond the sum of isolated effects, because the buffer keeps re-exposing each hard question to a fresh candidate set per visit.
- (iii) The full recipe is strongest at every scale. ZPPO ties or achieves the column max on every benchmark of Tab. 3 and Tab. 12. Within the recipe, BCQ’s contribution shrinks with scale and NCQ’s grows – mirroring Fig. 5a – so weaker students lean on the teacher candidate while stronger students lean on collective failure analysis. The same pattern holds for generalization (Appendix F.4, Tab. 13).

#### 4.5. Additional RL recipe choices

Beyond BCQ, NCQ, and the buffer, two seemingly minor recipe choices materially shape smallstudent RL. We isolate them at 2B across all three benchmark groups in Fig. 6. (i) Iterations per step trade off-policy drift against rollout cost; 𝐼 = 4 is the sweet spot. Larger 𝐼 grows in-step drift: each minibatch pushes the gradient further from the current policy. GRPO’s default 𝐼 =16 [36, 8] runs 4× as many updates as 𝐼 =4 but yields only a marginal gain over the original 2B base (Fig. 6a–c); 𝐼 =1 removes drift but undertrains at matched step count. 𝐼 =4 pays a small drift cost but performs four updates per step – the trade that peaks on all three (Fig. 6a–c). (ii) Excluding zero-advantage groups from batch normalization (Norm w/o Zero) is critical. Norm w/o Zero (ZPPO default) consistently beats No norm, while Norm w/ Zero clearly degrades below both (Fig. 6d–f). Both ‘Norm’ variants share REINFORCE++ two-step form [37]; they differ only in whether zero-advantage groups are used (Appendix C, Eqs. 2–3). Trivial groups (all-correct/all-wrong) carry no learning signal, but their zero advantage values shrink the batch standard deviation that normalizes every other advantage, causing inflated advantages.

Qwen3.5-0.8B Qwen3.5-2B Method LLM VLM Video LLM VLM Video

Base 25.2 41.0 48.3 45.3 56.8 60.6 +GRPO† 28.7 45.4 50.5 47.3 59.2 61.9 +GRPO†+Hint 28.8 47.2 50.0 47.6 60.4 61.6 +GRPO†+Prefix 27.1 45.5 49.1 46.3 59.5 61.0 +GRPO†+BCQ 31.5 48.6 52.0 49.1 60.8 62.6 +ZPPO 33.1 50.3 52.8 50.4 62.0 63.2

Table 4 | Comparison with Hint and Prefix methods

#### 4.6. Discussion and Analysis

- (i) Teacher size matters: A larger teacher widens ZPPO’s gain (Fig. 7); shrinking the teacher to 4B/9B collapses the gain toward the GRPO†+NCQ level, since BCQ requires the teacher to succeed where the student fails, and that pool of teacher-correct rollouts dries up as the teacher gets smaller. The drop is steepest at 0.8B, where ZPPO’s gain is most BCQ-dominated (Sec. 4.4).
- (ii) Hint, Prefix vs. BCQ: We compare BCQ against two prompt-side teacher-guidance families: Hint appends an answer-free hint to the prompt; and Prefix forces a teacher response prefix into the student and lets the student generate the rest (Appendix F.5). Tab. 4 shows that Hint (Algorithm 5) helps VLM but transfers less to the LLM/Video evaluation suite, since teacher hints act as a directional shortcut without why the direction is correct; Prefix (Algorithm 6) transfers even less because the teacher prefix is off-policy, so drift accumulates over training. BCQ alone beats both because the student is trained to select between two anonymized candidates while reasoning, whereas Hint conditions the prompt on a labeled teacher-derived hint and Prefix routes a teacher-generated prefix into the response itself; full ZPPO opens the widest gap across all.
- (iii) Further evidence: buffer dynamics (App. F.1), ZPPO vs. GRPO† dynamics (App. F.2), multimetric dynamics across scales (App. F.3), the component ablation on the LLM/Video benchmarks (App. F.4), Hint/Prefix setup (App. F.5), cluster-bootstrap CIs (App. F.6), BCQ and NCQ dynamics (App. F.7), teacher capability (App. G), and compute cost (Tab. 9) are deferred to the appendix.

### 5. Conclusion

Distillation directly imitates teacher logits and teacher prefix-injection RL splices teacher tokens into the policy gradient, breaking the on-policy assumption and weakening generalization at small student scales. ZPPO answers our research question by keeping the teacher only inside the prompt, so the student’s gradient stays on its own rollouts while still benefiting from teacher knowledge. The flip side is a structural ceiling: questions on which both teacher and student fail admit no BCQ candidate and leave only NCQ, whose contribution alone is limited – the zone collapses to the NCQ-only regime. Extending the student’s zone beyond what the current teacher covers is, in our view, an important open problem this work leaves behind.

### Limitations

Teacher-bounded zone. The most fundamental limitation of ZPPO is the one we identify in Sec. 5: BCQ requires the teacher to succeed on a hard question, so questions on which both teacher and student fail admit no BCQ candidate and leave only NCQ, whose contribution alone is more limited (Tab. 3, Sec. 4.4). At the upper end of the student-capability range, BCQ’s contribution shrinks accordingly and the available zone effectively collapses to the NCQ-only regime. Extending the zone beyond what the current teacher covers – via curated synthetic prompts, ensembles of teachers, or curriculum-aware question selection – is, in our view, the most important open problem this work leaves behind.

Tension with dynamic sampling. Dynamic sampling [8] simply deletes any question where the student got every rollout correct or every rollout wrong. On the other hand, ZPPO does the opposite: those all-wrong questions are exactly the ones we store in the replay buffer, so that BCQ and NCQ can rewrite the prompt and give the student a fresh chance. Hence, naively combining ZPPO and dynamic sampling is not trivial. A natural fix is to run them in sequence: first identify hard questions and only call dynamic sampling if their BCQ- and NCQ-reformulated rollouts still come back all-correct or all-wrong – that is, apply dynamic sampling on top of BCQ and NCQ. We leave this hybrid algorithm to future work.

Scope beyond reasoning correctness. ZPPO post-trains an already-built compact VLM and is orthogonal to how such small models are first constructed – via architectural designs that compress visual–linguistic computation into small backbones [53–57] or via large-to-small distillation of strong teachers [58–60] – so it could in principle be stacked on any of these as a later post-training stage. Several other axes are deliberately out of scope. Extending the recipe to multi-step and agentic multimodal reasoning [61–63] is open; we neither evaluate nor train for robustness, which carries its own substantial literature on robust and causal training [64–68]; we do not consider inferencetime efficiency techniques such as visual-token pruning [69] that are orthogonal to the training recipe; we do not target conversational or social competence [70]; we do not touch the upstream perception stacks [71] that compact models ultimately depend on for embodied deployment; and our evaluation targets single-turn reasoning accuracy rather than complementary axes such as multi-sensor perception [72], multi-turn dialogue [73], or iterative self-refinement [74]. Combining ZPPO with these complementary directions is left to future work.

### Ethical Considerations

ZPPO is a post-training recipe that builds on the publicly released Qwen3.5 family (open-weight VLMs at 0.8B–27B scale) [4]. We do not modify or filter the base models’ pretraining data; any social, demographic, linguistic, or modality biases present in the upstream pretraining of Qwen3.5 are inherited by the post-trained students. Our reward signal (rule-based exact-match grader plus LLM-as-a-judge for free-form questions) targets answer correctness on reasoning benchmarks, not safety, fairness, or harm reduction, so ZPPO is not a substitute for dedicated safety alignment such as RLHF, Constitutional AI, or red-teaming. Practitioners deploying ZPPO-trained checkpoints should pair them with an upstream safety alignment stage and downstream content moderation appropriate to the deployment domain. Beyond inherited pretraining bias, dataset-level bias and spurious visual–linguistic correlations are a distinct concern for multimodal models, addressed by debiasing and causal approaches [75, 76]; ZPPO’s correctness-only reward neither introduces nor corrects such biases, so these directions remain complementary to our recipe.

### References

- [1] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [2] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [3] Anthropic. The claude 3 model family: Opus, sonnet, haiku. https: //www.anthropic.com, 2024. URL https://www-cdn.anthropic.com/ de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf.
- [4] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.
- [5] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [6] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [7] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-Like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [8] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [9] Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, et al. JustRL: Scaling a 1.5B LLM with a simple RL recipe. arXiv preprint arXiv:2512.16649, 2025.
- [10] Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S Dhillon, David Brandfonbrener, and Rishabh Agarwal. The art of scaling reinforcement learning compute for LLMs. arXiv preprint arXiv:2510.13786, 2025.
- [11] Byung-Kwan Lee, Ryo Hachiuma, Yu-Chiang Frank Wang, Yong Man Ro, and Yueh-Hua Wu. Vlsi: Verbalized layers-to-interactions from large to small vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29545–29557, June 2025.
- [12] Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Frank Wang, and Yueh-Hua Wu. Unified reinforcement and imitation learning for vision-language models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 156508–156534. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper_files/paper/2025/ file/e58497367bc8730f61a87d37800c0a06-Paper-Conference.pdf.

- [13] Byung-Kwan Lee, Yu-Chiang Frank Wang, and Ryo Hachiuma. Masking teacher and reinforcing student for distilling vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10126–10141, June 2026.
- [14] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.
- [15] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, et al. Fastvlm: Efficient vision encoding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19769–19780, 2025.
- [16] Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin Huang. Mobile edge intelligence for large language models: A contemporary survey. IEEE Communications Surveys & Tutorials, 2025.
- [17] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
- [18] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International journal of computer vision, 129(6):1789–1819, 2021.
- [19] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.
- [20] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024.
- [21] Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards streamlined distillation for large language models. arXiv preprint arXiv:2402.03898, 2024.
- [22] Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of llms? arXiv preprint arXiv:2603.24472, 2026.
- [23] Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026.
- [24] Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626, 2026.
- [25] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, 2024.
- [26] Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/onpolicy-distillation.

- [27] Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562, 2026.
- [28] Walid Bousselham, Hilde Kuehne, and Cordelia Schmid. Vold: Reasoning transfer from llms to vision-language models via on-policy distillation. arXiv preprint arXiv:2510.23497, 2025.
- [29] Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026.
- [30] Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler II, Qian Qian, Gregory D. Lyng, Sanjit Singh Batra, and Robert E. Tillman. Fast and effective on-policy distillation from reasoning prefixes. arXiv preprint arXiv:2602.15260, 2026.
- [31] Yecheng Wu, Song Han, and Han Cai. Lightning OPD: Efficient post-training for large reasoning models with offline on-policy distillation. arXiv preprint arXiv:2604.13010, 2026.
- [32] Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897, 2026.
- [33] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.
- [34] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.
- [35] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026.
- [36] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [37] Jian Hu. REINFORCE++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.
- [38] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.
- [39] Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Harchaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun Yang, Jan Kautz, et al. BroRL: Scaling reinforcement learning via broadened exploration. arXiv preprint arXiv:2510.01180, 2025.
- [40] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. Dler: Doing length penalty right - incentivizing more intelligence per token via reinforcement learning. arXiv preprint arXiv:2510.15110, 2025.

- [41] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. GDPO: Group rewarddecoupled normalization policy optimization for multi-reward RL optimization. arXiv preprint arXiv:2601.05242, 2026.
- [42] Lev Semenovich Vygotsky and Michael Cole. Mind in society: Development of higher psychological processes. Harvard university press, 1978.
- [43] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. Star: Bootstrapping reasoning with reasoning. arXiv preprint arXiv:2203.14465, 2022.
- [44] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [45] Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, and Fei Mi. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning. arXiv preprint arXiv:2506.02208, 2025.
- [46] Shicheng Xu, Liang Pang, Yunchang Zhu, Jia Gu, Zihao Wei, Jingcheng Deng, Feiyang Pan, Huawei Shen, and Xueqi Cheng. RLKD: Distilling LLMs’ reasoning via reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.
- [47] Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, and Yu Cheng. Exgrpo: Learning to reason from experience. arXiv preprint arXiv:2510.02245, 2025.
- [48] Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Rlep: Reinforcement learning with experience replay for llm reasoning. arXiv preprint arXiv:2507.07451, 2025.
- [49] Shijie Zhang, Guohao Sun, Kevin Zhang, Xiang Guo, and Rujun Guo. Clpo: Curriculum learning meets policy optimization for llm reasoning. arXiv preprint arXiv:2509.25004, 2025.
- [50] Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. StepHint: Multi-level stepwise hints enhance reinforcement learning to reason. arXiv preprint arXiv:2507.02841, 2025.
- [51] Xuechen Zhang, Zijian Huang, Yingcong Li, Chenshun Ni, Jiasi Chen, and Samet Oymak. BREAD: Branched rollouts from expert anchors bridge SFT & RL for reasoning. arXiv preprint arXiv:2506.17211, 2025.
- [52] Ziheng Li, Zexu Sun, Jinman Zhao, Erxue Min, Yongcheng Zeng, Hui Wu, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen, et al. Staying in the sweet spot: Responsive reasoning evolution via capability-adaptive hint scaffolding. arXiv preprint arXiv:2509.06923, 2025.
- [53] Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. CoLLaVO: Crayon large language and vision mOdel. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 1121–1138, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.findings-acl.66. URL https://aclanthology.org/2024.findings-acl.66/.
- [54] Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. Moai: Mixture of all intelligence for large language and vision models. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 273–302, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-72967-6.

- [55] Byung-Kwan Lee, Chae Won Kim, Beomchan Park, and Yong Man Ro. Meteor: Mamba-based traversal of rationale for large language and vision models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 40278–40315. Curran Associates, Inc., 2024. doi: 10.52202/079017-1274. URL https://proceedings.neurips.cc/paper_files/paper/ 2024/file/473a9a75edc46eff5ff224d53d5f7294-Paper-Conference.pdf.
- [56] Byung-Kwan Lee, Sangyun Chung, Chae Won Kim, Beomchan Park, and Yong Man Ro. TroL: Traversal of layers for large language and vision models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11314–11342, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.633. URL https://aclanthology.org/2024.emnlp-main.633/.
- [57] Byung-Kwan Lee, Sangyun Chung, Chae Won Kim, Beomchan Park, and Yong Man Ro. Phantom of latent for large language and vision models. arXiv preprint arXiv:2409.14713, 2024.
- [58] Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Yu-Chiang Frank Wang, and Yueh-Hua Wu. Genrecal: Generation after recalibration from large to small vision-language models. arXiv preprint arXiv:2506.15681, 2025.
- [59] Byung-Kwan Lee. Building high-performing, efficient-size vision language models: merge, modify, and distill. 2025.
- [60] Seonghoon Yu, Dongjun Nam, Byung-Kwan Lee, and Jeany Son. Hide to see: Reasoning-prefix masking for visual-anchored thinking in vlm distillation. arXiv preprint arXiv:2605.11651, 2026.
- [61] Byung-Kwan Lee, Youngchae Chee, and Yong Man Ro. Recursive think-answer process for llms and vlms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 9608–9621, June 2026.
- [62] Minki Kang, Shizhe Diao, Ryo Hachiuma, Sung Ju Hwang, Pavlo Molchanov, Yu-Chiang Frank Wang, and Byung-Kwan Lee. Agent explorative policy optimization for multimodal agentic reasoning. arXiv preprint arXiv:2605.28774, 2026.
- [63] Seokju Cho, Ryo Hachiuma, Abhishek Badki, Hang Su, Byung-Kwan Lee, Chan Hee Song, Sifei Liu, Subhashree Radhakrishnan, Seungryong Kim, Yu-Chiang Frank Wang, and Min-Hung Chen. Spatialclaw: Rethinking action interface for agentic spatial reasoning, 2026. URL https://arxiv.org/abs/2606.13673.
- [64] Byung-Kwan Lee, Youngjoon Yu, and Yong Man Ro. Towards adversarial robustness of bayesian neural network through hierarchical variational inference, 2021. URL https:// openreview.net/forum?id=Cue2ZEBf12.
- [65] Junho Kim, Byung-Kwan Lee, and Yong Man Ro. Distilling robust and non-robust features in adversarial examples by information bottleneck. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 17148–17159. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/ 8e5e15c4e6d09c8333a17843461041a9-Paper.pdf.

- [66] Byung-Kwan Lee, Junho Kim, and Yong Man Ro. Masking adversarial damage: Finding adversarial saliency for robust and sparse network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15126–15136, June 2022.
- [67] Junho Kim, Byung-Kwan Lee, and Yong Man Ro. Demystifying causal features on adversarial examples and causal inoculation for robust network by adversarial instrumental variable regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12302–12312, June 2023.
- [68] Byung-Kwan Lee, Junho Kim, and Yong Man Ro. Mitigating adversarial vulnerability through causal parameter estimation by adversarial double machine learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4499–4509, October 2023.
- [69] Jiwan Kim, Kibum Kim, Wonjoong Kim, Byung-Kwan Lee, and Chanyoung Park. Why and when visual token pruning fails? a study on relevant visual information shift in mllms decoding. arXiv preprint arXiv:2604.12358, 2026.
- [70] Young-Jun Lee, Byung-Kwan Lee, Dokyong Lee, Kyeong-Jin Oh, Yechan Hwang, Ho-Jin Choi, et al. Enhancing conversational agents with skill-of-mind-infused large language model.
- [71] Byung-Kwan Lee. Training encoder-attention through fully-connected crfs for efficient end-toend lane detection model. 2020.
- [72] Youngjoon Yu, Sangyun Chung, Byung-Kwan Lee, and Yong Man Ro. Spark: Multi-vision sensor perception and reasoning benchmark for large-scale vision-language models. arXiv preprint arXiv:2408.12114, 2024.
- [73] Young-Jun Lee, Byung-Kwan Lee, Jianshu Zhang, Yechan Hwang, Byungsoo Ko, Han-Gyu Kim, Dongyu Yao, Xuankun Rong, Eojin Joo, Seung-Ho Han, Bowon Ko, and Ho-Jin Choi. Multiverse: A multi-turn conversation benchmark for evaluating large vision and language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 708–719, October 2025.
- [74] Young-Jun Lee, Seungone Kim, Byung-Kwan Lee, Minkyeong Moon, Yechan Hwang, Jong Myoung Kim, Graham Neubig, Sean Welleck, and Ho-Jin Choi. Refinebench: Evaluating refinement capability of language models via checklists. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=GYJFJz9Dy5.
- [75] Yeonju Kim, Junho Kim, Byung-Kwan Lee, Sebin Shin, and Yong Man Ro. Mitigating dataset bias in image captioning through clip confounder-free captioning network. In 2023 IEEE International Conference on Image Processing (ICIP), pages 1720–1724, 2023. doi: 10.1109/ICIP49359.2023.10222502.
- [76] Junho Kim, Byung-Kwan Lee, and Yong Man Ro. Causal unsupervised semantic segmentation. Pattern Recognition, 171:112173, 2026. ISSN 0031-3203. doi: https://doi.org/10. 1016/j.patcog.2025.112173. URL https://www.sciencedirect.com/science/article/pii/ S0031320325008349.
- [77] Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

- [78] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V,

2025. Accessed: 2025-02-02.

- [79] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-Onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [80] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. OpenVLThinker: Complex vision-language reasoning via iterative SFT-RL cycles. arXiv preprint arXiv:2503.17352, 2025.
- [81] Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1zero’s" aha moment" in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.
- [82] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-RFT: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [83] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025.
- [84] Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, and Minghui Qiu. Reinforced MLLM: A survey on RL-based reasoning in multimodal large language models. arXiv preprint arXiv:2504.21277, 2025.
- [85] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.
- [86] Chen Wang, Zhaochun Li, Jionghao Bai, Hexuan Deng, Ge Lan, and Yue Wang. SCOPERL: Stable and quantitative control of policy entropy in RL post-training. arXiv preprint arXiv:2510.08141, 2025.
- [87] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? arXiv preprint arXiv:2504.13837, 2025.
- [88] Siheng Li, Zhanhui Zhou, Wai Lam, Chao Yang, and Chaochao Lu. Repo: Replay-enhanced policy optimization. arXiv preprint arXiv:2506.09340, 2025.
- [89] Long Li, Zhijian Zhou, Tianyi Wang, Weidi Xu, Zuming Huang, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, and Yuan Qi. Dyjr: Preserving diversity in reinforcement learning with verifiable rewards via dynamic jensen-shannon replay. arXiv preprint arXiv:2603.16157, 2026.
- [90] Andrei Baroian and Rutger Berger. Prompt replay: Speeding up GRPO with on-policy reuse of high-signal prompts. arXiv preprint arXiv:2603.21177, 2026.
- [91] Ilia Mahrooghi, Aryo Lotfi, and Emmanuel Abbe. Goldilocks rl: Tuning task difficulty to escape sparse rewards for reasoning. arXiv preprint arXiv:2602.14868, 2026.

- [92] Zhenghao Liu, Zhuoyang Wu, Xinze Li, Yukun Yan, Shuo Wang, Zulong Chen, Yu Gu, Ge Yu, and Maosong Sun. Long-chain reasoning distillation via adaptive prefix alignment. arXiv preprint arXiv:2601.10064, 2026.
- [93] Vaskar Nath, Elaine Lau, Anisha Gunjal, Manasi Sharma, Nikhil Baharte, and Sean Hendryx. Adaptive guidance accelerates reinforcement learning of reasoning models. arXiv preprint arXiv:2506.13923, 2025.
- [94] Gabriel Sarch, Linrong Cai, Qunzhong Wang, Haoyang Wu, Danqi Chen, and Zhuang Liu. Vero: An open rl recipe for general visual reasoning. arXiv preprint arXiv:2604.04917, 2026.
- [95] Honglin Lin, Zheng Liu, Yun Zhu, Chonghan Qin, Juekai Lin, Xiaoran Shang, Conghui He, Wentao Zhang, and Lijun Wu. Mmfinereason: Closing the multimodal reasoning gap via open data-centric methods. arXiv preprint arXiv:2601.21821, 2026.
- [96] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [97] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-LLaVA: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.
- [98] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. GeomVerse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023.
- [99] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th international conference on computational linguistics, pages 1511–1520, 2022.
- [100] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. MM-Eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.
- [101] Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, and Christopher Clark. Scaling text-rich image understanding via code-guided synthetic multimodal data generation. arXiv preprint arXiv:2502.14846, 2025.
- [102] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.
- [103] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 4999–5007, 2017.
- [104] Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, and Wenhu Chen. VisualWebInstruct: Scaling up multimodal instruction data through web search. arXiv preprint arXiv:2503.10582, 2025.

- [105] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [106] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal ArXiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231, 2024.
- [107] Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang, Hang Zhang, Xin Li, et al. MMR1: Enhancing multimodal reasoning with variance-aware sampling and open resources. arXiv preprint arXiv:2509.21268, 2025.
- [108] Jiarui Zhang, Ollie Liu, Tianyu Yu, Jinyi Hu, and Willie Neiswanger. Euclid: Supercharging multimodal LLMs with synthetic high-fidelity visual descriptions. arXiv preprint arXiv:2412.08737, 2024.
- [109] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021.
- [110] Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, Lei Bai, Tao Ji, Tao Gui, Qi Zhang, Philip Torr, and Xuanjing Huang. BMMR: A large-scale bilingual multimodal multi-discipline reasoning dataset. arXiv preprint arXiv:2507.03483, 2025.
- [111] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521, 2022.
- [112] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. VLRethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.
- [113] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. PathVQA: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.
- [114] Jason J. Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific Data, 5(1): 180251, 2018.
- [115] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. RAVEN: A dataset for relational and analogical visual rEasoNing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- [116] Muye Huang, Han Lai, Xinyu Zhang, Wenjun Wu, Jie Ma, Lingling Zhang, and Jun Liu. EvoChart: A benchmark and a self-training approach towards real-world chart understanding. arXiv preprint arXiv:2409.01577, 2024.
- [117] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.

- [118] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A benchmark for visual question answering using world knowledge. arXiv preprint arXiv:2206.01718, 2022.
- [119] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [120] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021.
- [121] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [122] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7W: Grounded question answering in images. arXiv preprint arXiv:1511.03416, 2016.
- [123] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019.
- [124] Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel Reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025.
- [125] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and PixMo: Open weights and open data for state-of-the-art vision-language models. arXiv preprint arXiv:2409.17146, 2024.
- [126] Yichen Feng, Zhangchen Xu, Fengqing Jiang, Yuetai Li, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. VisualSphinx: Large-scale synthetic vision logic puzzles for RL. arXiv preprint arXiv:2505.23977, 2025.
- [127] Yew Ken Chia, Vernon Toh, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. PuzzleVQA: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. In Findings of the Association for Computational Linguistics: ACL 2024, pages 16259–16273, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.findings-acl.962. URL https://aclanthology.org/2024.findings-acl.962/.
- [128] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. LLaVA-CoT: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.
- [129] Ang Li, Charles Wang, Deqing Fu, Kaiyu Yue, Zikui Cai, Wang Bill Zhu, Ollie Liu, Peng Guo, Willie Neiswanger, Furong Huang, Tom Goldstein, and Micah Goldblum. Zebra-CoT: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746, 2025.
- [130] Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Y. Charles, Yiping Bao, Yuantao Fan, Guopeng Li, Haiyang Shen, Xuanzhong Chen, et al. Babyvision: Visual reasoning beyond language. arXiv preprint arXiv:2601.06521, 2026.

- [131] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. arXiv preprint arXiv:2406.18521, 2024.
- [132] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.
- [133] Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. arXiv preprint arXiv:2406.05756, 2024.
- [134] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.
- [135] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804, 2024.
- [136] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [137] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmupro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [138] Weihao Yu, Zhengyuan Yang, Lingfeng Ren, Linjie Li, Jianfeng Wang, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Lijuan Wang, and Xinchao Wang. MM-Vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities. arXiv preprint arXiv:2408.00765, 2024.
- [139] Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.
- [140] Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. arXiv preprint arXiv:2504.10342, 2025.
- [141] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind: Failing to translate detailed visual features into words. arXiv preprint arXiv:2407.06581, 2024.
- [142] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.
- [143] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

- [144] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.
- [145] Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, et al. Towards robust mathematical reasoning. arXiv preprint arXiv:2511.01846, 2025.
- [146] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [147] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.
- [148] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? arXiv preprint arXiv:2406.04127, 2024.
- [149] Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, Ed-Yeremai Cardona, Dean Lee, Jeremy Kritz, Willow Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. arXiv preprint arXiv:2501.17399, 2025.
- [150] Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, et al. Mmvu: Measuring expert-level multi-discipline video understanding. arXiv preprint arXiv:2501.12380, 2025.
- [151] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv:2311.17005, 2024.
- [152] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [153] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.
- [154] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

### A. Full Related Work

This appendix expands the condensed discussion in Sec. 2 into the three full paragraphs that motivate ZPPO.

Knowledge distillation for LLMs and VLMs. The dominant recipe for transferring a stronger model’s competence to a smaller one is knowledge distillation, which trains the student to imitate the teacher’s logits, hidden states, or sampled outputs [18]. In off-policy distillation, the target sequence is fixed – a ground-truth response [17, 19] or a teacher rollout – and the student is trained to match the teacher’s per-token logits, with recent variants adaptively scheduling student-generated outputs to balance the training-inference mismatch [21]. In on-policy distillation, the student instead samples its own response during training, that sequence is fed back into the teacher to obtain target logits, and the student is trained to match them [20, 25–31]. Self-distillation removes the external teacher, instead using the student itself as the imitation target – either via best-of-𝑁 rejection sampling on its own rollouts [43, 77], or by conditioning the same model on privileged information (e.g., demonstrations, verified traces, or environment feedback) to act as an in-context self-teacher [32–35, 22]. All three families lean on the same training signal, a logit distribution the student must match, and a recent line of analyses shows that this signal turns brittle once the student is much smaller than the teacher: the student memorizes, mode-seeks, or loses general capability beyond the training data [22–24]. ZPPO leaves this regime altogether. The teacher never enters the student’s policy gradient and shows up only as a candidate inside the prompt.

Reinforcement learning post-training for reasoning. RL post-training built on PPO [44] and group-relative variants such as GRPO [36] and DAPO [8] has, with rule-based rewards, pushed reasoning on math, science, and broad knowledge tasks [5–7, 9, 10, 38, 39, 37, 40, 41], and the same recipe extends to multimodal reasoning on image–question pairs [78–84]. A complementary line scales RL compute either by prolonging training steps or by broadening rollouts per prompt to extend exploration [38, 39], regulates entropy to prevent the policy from collapsing onto its current peaks [85, 86], and probes whether RL extends a model’s reasoning frontier or merely sharpens what the model already knows [87]. However, these advances do not solve the small student failure mode on questions where the student’s every rollout still fails, the group advantage stays zero, and the question is silently filtered out as a zero-advantage sample [8]. The natural fix – replacing a failed student rollout with a teacher’s correct response inside the policy gradient – breaks the on-policy assumption and induces severe policy drift; recent hybrid RL+distillation lines [45, 46] keep the teacher in the policy gradient and therefore inherit the same brittleness. ZPPO instead moves teacher knowledge from the gradient to the prompt: BCQ and NCQ inject teacher and self-generated candidates as reformulated prompts on those same hard questions, so every response token that enters the policy gradient is student-generated (on-policy at the response level).

Prompt replay, curriculum, and hint scaffolding for RL. A growing body of work reintroduces previously seen questions or rollouts in the spirit of classical experience replay [47, 48, 88–90], or arranges questions by difficulty from easy to hard [49, 91]. Both lines re-prioritize hard questions but neither provides the guidance that distinguishes Vygotsky’s zone of proximal development [42] from mere difficulty ordering. A complementary line scaffolds the student through difficult problems by injecting hints: prefix-based methods seed a partial reasoning trace into the response so the rollout is part teacher and the policy gradient is no longer strictly on-policy [50, 51, 92], while prompt-injected hints add pre-authored guidance to the prompt only, keeping the response on-policy and optionally correcting the resulting off-policy bias by importance sampling [52, 93].

ZPPO is also a prompt-side method on the same hard questions, but its in-prompt content is BCQ and NCQ candidates assembled at training time from the current rollout batch – not pre-authored hint text – so candidates change dynamically as fresh rollouts are sampled throughout training, with the prompt replay buffer existing solely to amplify these reformulated prompts on questions inside the student’s current zone of proximal development.

### B. Training Dataset and Benchmarks

#### B.1. ZPPO-77K training corpus

ZPPO-77K, the multimodal RL training corpus used throughout the paper, contains ∼77K (𝑥image,𝑥text,𝑎⋆) triples — where 𝑥image is the input image, 𝑥text is the text question, and 𝑎⋆ is the short gold answer used as the rule-based reward target — assembled from two publicly released sources:

- • zlab-princeton/Vero-600k [94]1: 34 sub-datasets covering STEM (math, science, geometry diagrams), chart/OCR (chart, table, diagram, infographic), knowledge/recognition (general VQA), and counting/grounding/search.
- • OpenDataArena/MMFineReason-SFT-586K [95]2: a 586K-sample chain-of-thought VLM corpus annotated with a per-example pass_rate computed by Qwen3-VL-4B-Thinking (the smaller checkpoint released alongside the 235B-distilled corpus). The dataset name retains the 235B teacher tag because the corpus is distilled from the 235B model; the pass_rate column itself, which we use for our hard-example filter below, is the 4B model’s empirical success rate on each prompt.

We split sub-datasets into two tiers by their relevance to challenging multimodal reasoning:

- • Tier 1 (caps at 2,800 samples per sub-dataset): STEM and Chart/OCR sub-datasets that are directly aligned with the reasoning content the student is asked to learn (ai2d [96], geo170k [97], geomverse [98], geoqa_plus [99], mmk12 [100], cosyn_math/chart/diagram/table [101], wemath [102], tqa [103], visualwebinstruct [104], chartqa [105], arxivqa [106]; on the MMFineReason side, MMR1 [107], Euclid [108], Geo170K, Geo3K/Geometry3K [109], mm-openr1, WeMath, VisualWebInstruct, BMMR [110], TQA, AI2D, ScienceQA [111], ViRL39K [112]).
- • Tier 2 (caps at 1,400 samples per sub-dataset): auxiliary VQA, knowledge, and counting/grounding sub-datasets (pathvqa [113], vqarad [114], raven [115], evochart [116], infographic_vqa [117], aokvqa [118], gqa [119], iconqa [120], vqav2 [121], visual7w [122], tallyqa [123], pixelreasoner [124], multihop, pixmo [125], visual_probe; MMFineReason Raven, VisualSphinx [126], PuzzleQA [127], LLaVA-CoT [128], Zebra-CoT [129]).

Cross-source duplicates (e.g. Vero’s stem-ai2d_merged and MMFineReason’s AI2D) are merged via a canonical-name map and the priority-1 source (Vero) wins. Within MMFineReason we additionally drop every example whose Qwen3-VL-4B-Thinking pass_rate> 0.5 – i.e. questions the 4B annotator solves more often than not – so that the corpus is biased toward genuinely hard examples. Per-sample filters: answer length ≤ 512 characters, image ≥ 100 pixels in both dimensions (i.e., min(ℎ,𝑤)≥100).

#### B.2. Evaluation benchmarks

The 31-benchmark evaluation suite reported in the main paper consists of 16 VLM benchmarks, 10 LLM benchmarks, and 5 Video benchmarks. Tab. 5, Tab. 6, and Tab. 7 list the abbreviation used throughout the main-paper tables, the full benchmark name, the scoring protocol, and the HuggingFace data source. In the scoring column, Exact+Judge first applies exact/rule matching and falls back to the shared LLM-as-a-judge only on unresolved rows; Rule+Judge uses a benchmark-specific deterministic parser before the same fallback; Det. uses a deterministic official metric with no LLM-as-a-judge; and BenchJudge uses a benchmark-specific official judge prompt.

- 1https://huggingface.co/datasets/zlab-princeton/Vero-600k
- 2https://huggingface.co/datasets/OpenDataArena/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking

Abbreviation Full name Scoring Data source (HF) AI2D AI2 Diagrams [96] Exact+Judge lmms-lab/ai2d BabyV BabyVision [130] Exact+Judge UnipatAI/BabyVision CharXiv CharXiv (chart understanding) [131] BenchJudge princeton-nlp/CharXiv DynaM DynaMath (501 seeds × 10 variants) [132] Exact+Judge DynaMath/DynaMath_Sample EmbSp EmbSpatial-Bench [133] Exact+Judge FlagEval/EmbSpatial-Bench InfoVQA InfographicVQA [117] Det. lmms-lab/DocVQA (InfographicVQA) MVerse MathVerse (testmini) [134] Exact+Judge CaraJ/MathVerse-lmmseval MVision MathVision [135] Exact+Judge MathLLMs/MathVision MVista MathVista (testmini) [136] Rule+Judge AI4Math/MathVista MMMUPro MMMU-Pro (all 3 configs) [137] Exact+Judge MMMU/MMMU_Pro MM-Vet MM-Vet [138] BenchJudge lmms-lab/MMVet OCREN OCRBench v2 (English subset) [139] Det. ling99/OCRBench_v2 OCRZH OCRBench v2 (Chinese subset) [139] Det. ling99/OCRBench_v2 VisP VisualPuzzles [140] Exact+Judge neulab/VisualPuzzles VBlind VLMs-are-Blind [141] Exact+Judge XAI/vlmsareblind WeMath WeMath (testmini) [102] Exact+Judge We-Math/We-Math

- Table 5 | 16 VLM benchmarks (Tab. 1 and Tab. 3).

Abbreviation Full name Scoring Data source (HF)

- AIME25 AIME 2025 (AIME2025-I/II, 30 problems) Exact+Judge opencompass/AIME2025
- AIME26 AIME 2026 Exact+Judge MathArena/aime_2026 CEval C-Eval (val, all 52 subjects) [142] Exact+Judge ceval/ceval-exam GPQA-D GPQA-Diamond (198 rows, gated) [143] Exact+Judge Idavidrein/gpqa HLE Humanity’s Last Exam [144] (mostly text; small image-bearing subset) BenchJudge cais/hle IMO-AB IMO-AnswerBench [145] Exact+Judge OpenEvals/IMO-AnswerBench MMLU MMLU (test, all subjects) [146] Exact+Judge cais/mmlu MMLU-Pro MMLU-Pro [147] Exact+Judge TIGER-Lab/MMLU-Pro MMLU-Rd MMLU-Redux 2.0 (all 57 subjects) [148] Exact+Judge edinburgh-dawg/mmlu-redux-2.0 MultiCh MultiChallenge [149] BenchJudge ScaleAI/MultiChallenge

- Table 6 | 10 LLM benchmarks (Tab. 2, LLM block).

Abbreviation Full name Scoring Data source (HF) MMVU MMVU (validation, 1,000 rows) [150] BenchJudge yale-nlp/MMVU MVBench MVBench (20 task configs) [151] Rule+Judge OpenGVLab/MVBench VMME Video-MME (w/o subtitles) [152] Rule+Judge lmms-lab/Video-MME VMMES Video-MME (with subtitles) [152] Rule+Judge lmms-lab/Video-MME VMMMU Video-MMMU [153] Rule+Judge lmms-lab/VideoMMMU

- Table 7 | 5 Video benchmarks (Tab. 2, Video block).

Decoding configuration. Training and evaluation deliberately use different decoding settings. Training-time student and teacher rollouts (Appendix D.1, Tab. 8) sample at temperature=1.0, top-p=1.0 with no top-𝑘/penalty terms, so the policy gradient sees a high-entropy distribution and explores. Evaluation, by contrast, uses a single more deterministic configuration shared across every benchmark and every model checkpoint – temperature=0.6, top-p=0.95, top-k=20, min-p=0, presence-penalty=1.5, repetition-penalty=1.0, max-new-tokens=12288, max-model-len=262144, min-pixels=256×32×32, max-pixels=1280×32×32 – so that any difference between methods at the same student scale is attributable to training, not to evaluation hyperparameters.

Prompt and judge templates. For every benchmark we strip all reasoning- and answer-format directives from the upstream prompt (e.g. “Think step by step” or “put your final answer within \boxed{}”) and rely on a single shared RL closer that the student is also trained against (Fig. 8): This guarantees that the answer-extraction format used at evaluation time is the same one the policy

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.

Figure 8 | RL closer enforcing the think/boxed answer format, applied identically at training and evaluation.

gradient was optimized against.

For benchmarks where exact-match parsing is unsafe, we use the shared LLM-as-judge only after deterministic parsing fails (e.g. math derivations and open-ended VLM/video questions), and use benchmark-specific judge prompts for official judge-based tasks (CharXiv, MM-Vet, HLE, MultiChallenge, and MMVU). The judge model itself is the same Qwen3.5-27B-FP8 used as the teacher in training (Tab. 8, Appendix G), running on a dedicated sidecar pool at temperature=0.0, top-p=1.0, max-new-tokens=512, and parsing only a strict JSON {"verdict": "correct" | "wrong"} from the response (any malformed or out-of-vocabulary verdict is conservatively scored as 0, i.e. wrong). The same judge model, prompt, and fallback policy – including the look-inside-reasoning fallback used when a strict \boxed{} cannot be parsed – are applied identically to every method in the paper at every evaluation step, so any cross-method delta is attributable to method behaviour rather than to evaluation-protocol differences. The shared judge uses a single Jinja2 template, reproduced verbatim in Fig. 9: The same judge configuration is used for every method in Tab. 1, Tab. 2, Tab. 3, Tab. 4, and the appendix tables.

Benchmark-specific judge prompts. Five of our 31 benchmarks ship their own official judge prompt that we reproduce verbatim from the upstream evaluators so our reported numbers match the published leaderboards: MM-Vet, MultiChallenge, MMVU, HLE, and CharXiv (which uses two separate templates for its Reasoning and Descriptive question types). Placeholders in {{ ... }} are filled in at judge call time.

MM-Vet (Fig. 10) – ported verbatim from the official MM-Vet evaluator; absolute [0,1] correctness with the dataset’s own <AND>/<OR> tags and 15 paper-frozen few-shot exemplars (truncated below to the first 6 rows that drive the rubric; the full table is shipped with the codebase):

MultiChallenge (Fig. 11) – official YES/NO verifier (Scale AI, 2501.17399); the criterion is hidden from the student and revealed only to the judge:

MMVU (Fig. 12) – two-mode official template (Yale-NLP, 2501.12380); the open-ended branch enforces the official “exact same technique or concept” criterion:

You are an expert judge evaluating whether a model’s response is correct according to the ground truth. ## Your Role (READ FIRST) You are a COMPARATOR, not a solver. Your ONLY job is to decide how well the model’s response matches the provided GROUND TRUTH.

- • Do NOT attempt to solve the problem yourself.
- • Do NOT infer or derive what the “real” answer should be.
- • Do NOT second-guess the ground truth, even if it seems wrong.
- • The “Model Response” section below MAY be either (a) the model’s full trajectory including any <think>...</think> reasoning and a concluding \boxed{...} answer, OR (b) a short EXTRACTED snippet that was already pulled out as the candidate final answer (e.g. a bare letter like B, a number like 42, an expression like x = 5/3). Treat both forms uniformly: the snippet (when given) IS the model’s committed final answer, and a full trajectory’s final answer is the content inside the last \boxed{...} or the concluding sentence AFTER any </think> tag.
- • If the final answer is missing due to answer format error, truncation, or other reasons, look inside thinking or reasoning.
- • You should OUTPUT JSON with the format {"verdict": "correct" or "wrong"} based on the model’s response and the ground truth.

## Question {{ question }} ## Answer Options (rendered only when non-empty) {{ options }}

## Model Response {{ model_response }} ## Ground Truth Answer

{{ ground_truth }} Output ONLY a valid JSON object with exactly this format (no prose, no code fences, no trailing commentary): {"verdict": "correct" or "wrong"}

Figure 9 | Shared LLM-as-a-judge template used across every method at every evaluation step.

Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1,0.2,...,0.9, or 1.0 (totally right). Just complete the last space of the correctness score. Question | Ground truth | Prediction | Correctness

— | — | — | What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0 What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5 What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5 What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5 What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0

... (9 additional MM-Vet exemplars covering meme-explanation and free-form rows, kept verbatim) {{ question }} | {{ ground_truth }} | {{ prediction }} |

Figure 10 | MM-Vet official judge prompt (verbatim from upstream evaluator).

You are tasked with evaluating a model response to see if it meets a specific criteria. The criteria will always be YES/NO evaluation. The model response is as follows: <MODEL_RESPONSE> {{ response }} </MODEL_RESPONSE> The criteria that the model response must meet is as follows. Be VERY STRICT!: <CRITERIA> {{ target_question }} </CRITERIA> Print your reasoning followed by your verdict, either “YES” or “NO”.

Figure 11 | MultiChallenge YES/NO verifier prompt (Scale AI official).

Evaluate whether the model’s final answer is correct by comparing it to the ground-truth answer provided for the given question. You should first extract the final answer from the model’s response, and then compare the extracted answer with the ground-truth answer to determine its accuracy. (Open-ended branch only) The final answer generated by the model does not need to match the ground-truth answer word-for-word. However, it should ONLY be considered correct if it demonstrates the exact same technique or concept explicitly and unambiguously equivalent to the ground-truth answer. ## Question {{ question }} ## Options (MCQ only) {{ options }} ## Ground Truth Answer {{ ground_truth }} ## Model’s Full Response {{ model_response }} Output ONLY a valid JSON object with exactly this format (no prose, no code fences, no trailing commentary): {"verdict": "correct" or "wrong"}

Figure 12 | MMVU two-mode official judge prompt (Yale-NLP).

HLE (Humanity’s Last Exam) (Fig. 13) – official CAIS judge; the judge inspects the full response (including <think>) and emits four fields with correct ∈ {yes,no}:

Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below. [question]: {{ question }} [response]: {{ response }} Your judgement must be in the format and criteria specified below: extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as ‘None’ if there is no exact, final answer to extract from the response. [correct_answer]: {{ correct_answer }}

reasoning: Explain why the extracted answer is correct or incorrect based on [correct_answer], focusing only on whether there are meaningful differences. Do not solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer ‘yes’ if extracted_final_answer matches [correct_answer] (or is within a small margin of error for numerical problems), ‘no’ otherwise. confidence: Confidence score ∈ [0%,100%] extracted from [response] (default 100). Respond with ONLY a single JSON object using exactly these keys: extracted_final_answer, reasoning, correct ("yes"/"no"), confidence (integer 0–100).

Figure 13 | HLE (Humanity’s Last Exam) official CAIS judge prompt.

CharXiv – Reasoning (Fig. 14; official CharXiv Reasoning rubric; the rubric body is one of 4 categories: text-in-chart, text-in-general, number-in-chart, number-in-general):

You will be given a question, a ground truth answer and a model response. You need to extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0. Your response must follow JSON format with keys [extracted_answer, score] where score∈ {0,1}. You must follow the scoring rules: {{ rules }} ### Your Turn ###

- * Question: {{ question }}
- * Ground Truth: {{ ground_truth }}
- * Response: {{ response }} Respond with ONLY a single JSON object using exactly these keys: extracted_answer (string), score (integer 0 or 1).

Figure 14 | CharXiv Reasoning official judge prompt (4 rubric variants).

CharXiv – Descriptive (Fig. 15; official CharXiv Descriptive rubric; the rubric body is one of 7 classes: title, ocr, quant, bool, enum, trend, layout):

You will be given a pair of ground truth answer and model response under an overarching question. You need to extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0. Your response must follow JSON format with keys [extracted_answer, score] where score∈ {0,1}. Overarching Question: {{ overarching_question }} {{ rubric }} ### Your Turn ### Response: {{ response }} Ground Truth: {{ ground_truth }} Respond with ONLY a single JSON object using exactly these keys: extracted_answer (string), score (integer 0 or 1).

Figure 15 | CharXiv Descriptive official judge prompt (7 rubric classes).

### C. ZPPO algorithm

Algorithm 1 summarizes one ZPPO training step. The key invariant is that every response used in the policy gradient is generated by the current student. The frozen teacher is queried only to provide prompt-side candidate text for BCQ (and to compress candidates under a shared compression prompt and token cap), so teacher-generated tokens never enter the gradient as response tokens.

ZPPO inherits its RL backbone from three lines of recent work and adds four novel components. Inherited from GRPO [36]: (G1) group-relative advantage formulation; (G2) PPO-style clipped surrogate objective. Inherited from DAPO [8]: (Da1) asymmetric clip-higher with (𝜖low,𝜖high)= (0.20,0.28); (Da2) token-level loss aggregation; (Da3) no KL penalty against a reference policy. Inherited from REINFORCE++ [37]: (R1) two-step advantage estimator (Eqs. 2–3). Our recipe choice on top of REINFORCE++: (Z0) zero-advantage-group exclusion from the batch-statistics computation in Step 2, ablated in Sec. 4.5(ii) and derived in closed form in Eqs. 4–6. ZPPO’s own contributions: (Z1) BCQ – the anonymized binary-candidate prompt reformulation that turns a hard question into a discriminative one-of-two judgment between a correct teacher trace and a wrong student trace, both compressed by the same teacher under a shared compression prompt and token cap to mitigate obvious surface cues in the discrimination; (Z2) NCQ – the collectivenegative-candidate prompt reformulation that surfaces all of the student’s wrong rollouts on the same question, together with their parsed final answers, so the student must avoid the wrong rollouts it has just produced; (Z3) prompt replay buffer with hard-question admission, graduation by mean rollout accuracy, and FIFO eviction; (Z4) the super-additive combination of (Z3) with (Z1)+(Z2) that is the central empirical claim of the paper (Sec. 4.4). Algorithm 1 annotates each step with the tags above.

A few practical notes are worth pointing out. First, BCQ requires both a correct teacher rollout and a wrong student rollout to construct a candidate pair; questions on which the teacher itself fails on every 𝑦T skip BCQ for that visit and only contribute NCQ (or, if all student rollouts succeed on the second try, neither). Second, candidate compression runs in parallel with Stage 1 scoring, so it does not stall the rollout loop. Third, the teacher rollouts {𝑦T} on a replayed question are re-sampled on every visit – the BCQ candidate seen by the student on visit 𝑘 is not the same as the one seen on visit 𝑘−1, even when the underlying question is identical. This freshness is what allows the buffer to avoid degenerating into a single fixed exemplar per hard question. Fourth, only the plain student rollouts on the original 𝑥 feed buffer admission and graduation in Stage 6: the BCQ/NCQ rollouts contribute to the gradient but are skipped when computing the per-question accuracy used to decide buffer membership, so a question is judged “mastered” only when the student can solve it without candidate references. Fifth, the three groups (plain, BCQ, NCQ) on the same hard 𝑥 each have their own group identifier and are advantage-normalized independently; the plain group on 𝑥 and the reformulated groups on 𝑥BCQ, 𝑥NCQ are sampled from different prompt distributions and never share a baseline.

What counts as a “rollout step,” a “gradient update,” and a “token.” Because 𝜌replay and 𝜌aug are both expressed as a fraction of rollout_batch_size, the prompt counts seen by the gradient at each rollout step are additive rather than fractions of a fixed budget. One rollout step contains: (i) 𝑋new new prompts drawn from 𝒟 (over the rollout-step budget in Tab. 8, 𝑋new alone consumes the entire ZPPO-77K corpus in a single pass; 𝑋replay does not count against the dataset budget); (ii) 𝑋replay prompts drawn from ℬ on top of 𝑋new with |𝑋replay|=𝜌replay|𝑋new|; and (iii) at most 𝜌aug|𝑋new| BCQ/NCQ instances after the combined post-cap of Step 5. The cap is enforced in two stages. (a) Pre-cap on base questions (Step 3 of Alg. 1): once teacher rollouts have been drawn on 𝑋hard in Step 2, we keep the top 𝜌aug|𝑋new| base questions by ascending 𝑟¯𝑥 that admit at

Algorithm 1: ZPPO training step. Tags: G1–G2 from GRPO [36], Da1–Da3 from DAPO [8], R1 from REINFORCE++ [37]; Z0 marks our recipe choice on top of REINFORCE++ (zeroadvantage-group exclusion); Z1–Z4 are ZPPO’s contributions (BCQ, NCQ, prompt replay buffer, and their super-additive combination).

Input: Student 𝜋𝜃, teacher 𝜋T, dataset 𝒟, buffer ℬ; 𝐺S,𝐺T,𝜌replay,𝜌aug,|ℬ|max,𝐼,𝜏 =0.5. Output: Updated 𝜋𝜃, ℬ.

- 1 // 1. Plain rollouts. [G1, Z3]
- 2 𝑋 ←𝑋new∪𝑋replay with 𝑋new∼𝒟, 𝑋replay∼ℬ, |𝑋replay|=𝜌replay|𝑋new|. For each 𝑥∈𝑋, draw {𝑦S(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥), score, compute 𝑟¯𝑥, and collect wrong subset {𝑦S(−)(𝑥)}.
- 3 // 2. Teacher rollouts on hard questions. [Z1]
- 4 𝑋hard←{𝑥 : 𝑟¯𝑥<𝜏}. For each 𝑥∈𝑋hard, draw {𝑦T(𝑔)(𝑥)}𝐺𝑔=1T ∼𝜋T(·|𝑥), score, and keep the correct subset {𝑦T(+)(𝑥)}.
- 5 // 3. Pre-cap on base questions. [Z1, Z2]
- 6 Keep the top 𝜌aug|𝑋new| of 𝑋hard by ascending 𝑟¯𝑥 as 𝑋augpre (every hard 𝑥 has {𝑦S(−)}̸=∅ by 𝑟¯𝑥<𝜏 =0.5; the per-instance BCQ/NCQ admissibility check happens in Step 4).
- 7 // 4. Build BCQ/NCQ prompts. [Z1, Z2]
- 8 𝒜←∅. foreach 𝑥 ∈ 𝑋augpre do

- 9 if {𝑦T(+)},{𝑦S(−)}̸=∅ then teacher-compress one 𝑦T(+) and one 𝑦S(−) under the shared compression prompt and token cap, shuffle, form 𝑥BCQ; add (𝑥BCQ,uidBCQ,𝑥) to 𝒜
- 10 if {𝑦S(−)}̸=∅ then teacher-rewrite all wrong rollouts, list their parsed answers, form 𝑥NCQ; add (𝑥NCQ,uidNCQ,𝑥) to 𝒜
- 11 end
- 12 // 5. Post-cap and reformulated rollouts. [Z1, Z2]
- 13 if |𝒜|>𝜌aug|𝑋new| then

- 14 Build 𝒜′←∅ by iterating 𝑥∈𝑋augpre in ascending 𝑟¯𝑥; for each 𝑥, append its BCQ instance (if present) then its NCQ instance (if present) to 𝒜′. If appending the next instance would

push |𝒜′| above 𝜌aug|𝑋new|, skip that single instance (so when only one slot remains in a question that has both, BCQ is kept and NCQ is dropped) and stop. Set 𝒜←𝒜′.

- 15 end
- 16 For each (𝑥′,uid′,𝑥)∈𝒜, draw {𝑦(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥′) under uid′ and score.
- 17 // 6. Update and buffer refresh. [G2, Da1–Da3, R1, Z0, Z3, Z4]
- 18 Treat plain, BCQ, NCQ rollouts on 𝑥 as three separate groups (group key = uid), each of size 𝐺S. Apply Eqs. (2)–(3) (Step 1: subtract group mean; Step 2: batch-normalize only over the non-trivial subset 𝒢⋆={𝑔:std𝑥>0}, leaving trivial groups at 𝐴(𝑔)=0); update 𝜋𝜃 for 𝐼 iterations.
- 19 Compute 𝑟¯𝑥 from plain rollouts only; admit {𝑥 : 𝑟¯𝑥<𝜏} to ℬ, graduate the rest, FIFO-evict until |ℬ|≤|ℬ|max.
- 20 return 𝜋𝜃, ℬ.

least one of the BCQ or NCQ branches. (b) Post-cap on combined BCQ+NCQ instances (Step 5 of Alg. 1): because each 𝑥∈𝑋augpre may contribute one BCQ instance (when {𝑦T(+)}̸=∅ and {𝑦S(−)}̸=∅) and one NCQ instance (when {𝑦S(−)}̸=∅), the pre-instance count |𝒜| can be up to twice the pre-cap; the aug_max_ratio clause in the code then prunes 𝒜 as a single combined BCQ+NCQ pool (not separately per type) down to at most 𝜌aug|𝑋new|. Each surviving prompt – plain new, replayed, BCQ, or NCQ – carries 𝐺S rollouts, and the actual per-step prompt count enters the gradient via gradient accumulation; FLOPs in Tab. 9 are accumulated over all gradient-counted tokens (plain + replay + BCQ + NCQ, with the BCQ+NCQ portion respecting the augmentation cap above), with the 𝐼-update multiplier applied to each.

Two-step advantage estimator (REINFORCE++) used by ZPPO. Eq. (1) in Sec. 3.1 is the textbook GRPO advantage that ZPPO builds on. We adopt the two-step variant of Hu [37] (their REINFORCE++, Eqs. 6–7), which decouples within-group centering from across-group normalization; for clarity, we restate it in our notation and make explicit which sub-population enters the batch statistics in Step 2. Let 𝒢 denote the set of groups in a mini-batch (each group 𝑔 has 𝐺S rollouts sharing a uid), and let 𝒢⋆⊆𝒢 be the non-trivial subset, defined as those groups with std𝑥>0 (i.e. neither all-correct nor all-wrong). Step 1 subtracts the group mean,

𝐴′𝑥,𝑖(𝑔) = 𝑟(𝑥,𝑦(𝑔,𝑖)) − 𝑟¯𝑥, (2) which produces 𝐴′=0 for every rollout in a trivial (zero-advantage) group by construction. Step 2 batch-normalizes across a sub-population 𝒮 ⊆𝒢 of groups:

𝐴′𝑥,𝑖(𝑔) − 𝜇𝒮 𝜎𝒮 + 𝜖

𝐴(𝑥,𝑖𝑔) =

,

𝜇𝒮 = mean(𝑔,𝑖):𝑔∈𝒮 𝐴′𝑥,𝑖(𝑔), 𝜎𝒮 = std(𝑔,𝑖):𝑔∈𝒮 𝐴′𝑥,𝑖(𝑔). (3)

The three settings compared in Sec. 4.5(ii) (No norm, Norm w/o Zero, Norm w/ Zero) all perform Step 1 and differ only in 𝒮 and in whether Step 2 is applied:

- • No norm: Step 2 is skipped entirely, 𝐴(𝑔)=𝐴′(𝑔) (within-group centering only); equivalent to the standard group-relative advantage of Eq. (1) up to the std𝑥 rescaling.
- • Norm w/o Zero (ZPPO default): 𝒮 =𝒢⋆. The batch statistics in Step 2 are computed over the non-trivial groups only, and trivial groups keep 𝐴(𝑔)=0 (since their 𝐴′=0 is left untouched).
- • Norm w/ Zero: 𝒮 =𝒢. The batch statistics in Step 2 include every group, even the trivial ones whose 𝐴′ =0. We make the consequence quantitative. Because Step 1 centers each group to a sum of zero,

∑︁𝐺S

∑︁𝐺S

(︀𝑟(𝑔,𝑖) − 𝑟¯𝑔)︀

𝐴′(𝑔,𝑖) =

= 0 for every 𝑔 ∈ 𝒢, (4)

𝑖=1

𝑖=1

the batch mean is identically zero, 𝜇𝒢 ≡ 0, irrespective of how many trivial groups the batch happens to contain. The batch standard deviation, on the other hand, is depressed by the trivial groups’ zeros. Writing 𝑓nt=|𝒢⋆|/|𝒢| and using equal group size 𝐺S,

1 |𝒢|𝐺S ∑︁

∑︁

(︀𝐴′(𝑔,𝑖))︀2

𝜎𝒢2 =

𝑔∈𝒢

𝑖

1 |𝒢⋆|𝐺S ∑︁

∑︁

= |𝒢⋆| |𝒢|

(︀𝐴′(𝑔,𝑖))︀2

·

𝑔∈𝒢⋆

𝑖

= 𝑓nt 𝜎𝒢2⋆, (5)

so 𝜎𝒢 = √𝑓nt 𝜎𝒢⋆. Substituting into Eq. (3) gives a closed form for the resulting advantages:

𝑟(𝑔,𝑖) − 𝑟¯𝑔 √𝑓nt 𝜎𝒢⋆ + 𝜖

𝐴(w/Zero𝑔,𝑖) =

1 √𝑓nt · 𝐴(ZPPO𝑔,𝑖) (𝑔 ∈ 𝒢⋆), 𝐴(w/Zero𝑔,𝑖) = 0 (𝑔 ∈/ 𝒢⋆). (6)

=

Two things follow. First, trivial groups stay at 𝐴(𝑔,𝑖)=0 exactly as under ZPPO – they receive neither a positive nor a negative gradient signal. Second, every non-trivial advantage is uniformly amplified by 1/√𝑓nt relative to ZPPO. When 𝑓nt becomes small – early in training, or when the student is too weak/too strong for most questions – this amplification more often pushes the importance ratio outside PPO’s [1−𝜖low,1+𝜖high] window and produces the visibly worse training curves reported in Sec. 4.5(ii). The role of Norm w/o Zero is therefore not to alter any group’s centered advantage 𝐴′, but to estimate the normalizing 𝜎 on the sub-population 𝒢⋆ that actually carries learning signal, so that the scale of the non-trivial advantages does not depend on the trivial fraction 1−𝑓nt.

The implementation (Sec. D.1, advantage estimator row) registers Eqs. (2)–(3) with 𝒮 =𝒢⋆ as the ZPPO default, so all main-paper numbers use Norm w/o Zero. The change from Norm w/ Zero to Norm w/o Zero is a one-line gating on std𝑥 >0 in the batch-statistics computation, with zero compute overhead.

### D. Hyperparameters and baseline implementations

#### D.1. ZPPO hyperparameters

Tab. 8 lists every hyperparameter used to produce the ZPPO numbers reported in the main paper, including the prompt replay buffer ℬ and its capacity |ℬ|max. All ZPPO and baseline runs share the same hardware pool: 64 NVIDIA H100-80GB GPUs, organized as 8 nodes of 8 GPUs each; on every node 6 GPUs serve student rollouts and the gradient update, and 2 GPUs serve frozen teacher rollouts, teacher-side candidate compression, and the LLM-as-a-judge, with teacher generation overlapped with student rollouts so the wall-clock per step is dominated by the student-side cost. All RL rows – GRPO, GRPO†, GRPO+Both (BCQ+NCQ without the buffer), GRPO†+BCQ, GRPO†+NCQ, the Hint and Prefix baselines, and ZPPO (which is GRPO†+Both) – inherit every entry below; the only differences live in which student rollout source enters the gradient (Sec. D.2).

Group Hyperparameter Value

Student initialization Qwen/Qwen3.5-{0.8,2,4,9}B (HF), base checkpoints post-trained as a VLM in this paper Teacher Qwen/Qwen3.5-27B-FP8 (HF), base post-trained as a VLM and then frozen during ZPPO Backbone family naming every model uses “Qwen3.5” GPUs per node (student / teacher) 6/2 Total GPUs 64 H100-80GB (8 nodes × 8)

Student / teacher

Optimizer AdamW (AnyPrecisionAdamW, bf16 optimizer state) Learning rate 1.0×10−6 (constant for the full run) LR scheduler constant, no warmup, no decay (lr_warmup_ratio=0) AdamW (𝛽1,𝛽2) (0.9, 0.999) AdamW 𝜖 10−8 Weight decay 0.1 Gradient clipping (max grad norm) 1.0 PPO inner epochs (over a global batch) 1 (no inner re-iteration of the global batch) Mixed precision bfloat16 FSDP sharding full-shard, bfloat16, rank-0 init Fused LM-head kernel on, PyTorch backend KL penalty against reference policy 0 (following DAPO) Reference reset count 0 Online filtering disabled

Optimization

Student group size 𝐺S 8 Teacher group size 𝐺T (BCQ pool) 4 Tensor-parallel size (student / teacher) 1/1 Sampling temperature (student / teacher) 1.0/1.0 (training-time rollouts) Top-𝑝 (student / teacher) 1.0/1.0 (no nucleus truncation during training) Max prompt length 4,096 tokens (plain rollouts only; BCQ/NCQ rollouts are bounded only by the policy’s 262K context window) Max response length (student / teacher) 12,288 / 12,288 tokens

Rollouts

Reward rule-based binary ∈ {0,1} (math-aware boxed grader; details below) Advantage estimator two-step (REINFORCE++ [37]), zero-advantage groups excluded from batch stats Clip-lower 𝜖low 0.20 Clip-higher 𝜖high 0.28 (DAPO clip-higher) Dual clip ratio 10.0 Loss aggregation token-level (DAPO)

GRPO backbone

Iterations per step 𝐼 4 gradient updates per rollout step (Sec. 4.5) Batch advantage normalization on, with zero-advantage groups excluded from the batch statistics

Recipe choices

Hard-question threshold 𝜏 0.5 (mean rollout accuracy) Replay fraction 𝜌replay 0.25 Prompt replay buffer capacity |ℬ|max 10,000 prompts Augmentation fraction 𝜌aug 0.25 Eviction policy FIFO once |ℬ|>|ℬ|max Candidate compression on, max 512 tokens

ZPPO-specific

Data sampling single pass over the entire ZPPO-77K corpus Micro-batch size per device (update / exp.) 4/4 Global batch size 4×micro×student-GPUs×nodes/𝐺S = 96 prompts (one gradient update) Rollout batch size 𝐼·global = 384 prompts (one rollout step = 𝐼 gradient updates) Gradient accumulation per update 4 micro-batches per device (96×𝐺S/(micro×student-GPUs×nodes)) Total rollout steps per run 200 Total gradient updates per run 800 (𝐼·200) Padding-free training on

Schedule & batching

Table 8 | Training hyperparameters for ZPPO. Values shared with GRPO/GRPO† are unchanged across all RL rows of Tab. 1, Tab. 2, Tab. 3, Tab. 4, and the appendix tables.

Rule-based reward grader. The binary reward in the GRPO-backbone row of Tab. 8 is computed by a math-aware boxed grader from the mathruler.grader library. extract_boxed_content first

pulls the content inside the last \boxed{...} span of the response; grade_answer then compares that content to the gold short answer 𝑎⋆, treating LaTeX/numerical equivalences (e.g. 1/2 vs. 0.5, \frac{1}{2} vs. 0.5) as matches. If this strict comparison fails but the response did contain a boxed span, both sides are stripped of degree (∘, \circ, \deg) and percent (%, \%) annotations and re-graded once; the reward is 1 iff either pass succeeds and 0 otherwise. Free-form questions (math derivations, OCR, open-ended VLM) instead route to the LLM-as-a-judge described next; this routing is identical across every method in Tab. 1–Tab. 4.

Parallel teacher generation. Teacher rollouts {𝑦T} for the BCQ pool are generated by a colocated inference engine on 2 of every 8 per-node GPUs; their wall-clock overlaps with the student rollout phase, so they do not appear on the critical path of any training step.

LLM-as-a-judge for free-form rewards. Whenever a question’s gold answer is free-form (math derivations, OCR, open-ended VLM questions where exact-match parsing is unsafe), the binary reward is decided by the same LLM-as-a-judge configuration [154] used at evaluation (Appendix B.2). The judge runs in parallel with student rollout scoring on a sidecar inference pool and never touches the gradient. Numbers in Tab. 1–Tab. 4 use this judge identically across every method.

Candidate compression. The frozen teacher rewrites candidate responses into concise reasoning traces whose final answers are preserved verbatim. The cap of 512 tokens per candidate is chosen so that even an NCQ prompt carrying every wrong rollout in a group fits well below the policy’s 262K context window, and so that BCQ candidates from the teacher and the student have comparable length. This compression runs in parallel with rollout scoring on the same teacher-side inference pool and is therefore off the gradient path. The exact teacher-side prompt is shown in Fig. 16:

Compress the response below into a summary (in 5 lines max). Rules:

- • Response is in <response>...</response> tags
- • Summary should be in <summary>...</summary> tags and should be in 5 lines max
- • Keep ONLY the essential reasoning steps and the final answer
- • Remove ALL exploratory text, self-corrections, retries, and filler
- • Do NOT re-derive or add new information
- • End with the final answer in \boxed{} format <response>

⟨candidate response to be compressed – 𝑦T(+) for BCQ, 𝑦S(−) for NCQ⟩ </response>

Figure 16 | Teacher-side candidate compression prompt (off the gradient path).

Reformulated prompts (BCQ and NCQ). The body of the paper (Sec. 3.2) already shows the BCQ and NCQ instructions in compact form. For full reproducibility, the strings below are the exact templates appended to the original question 𝑥 – preserving the literal string concatenation, including the leading blank line and the per-candidate <candidate>...</candidate> blocks. ⟨candidate𝑖⟩ placeholders are filled at construction time with the teacher-compressed traces produced by the prompt above.

BCQ (Fig. 17). For each hard question with at least one correct teacher rollout, we draw one 𝑦T(+) and one 𝑦S(−), teacher-compress both, randomly shuffle the order, and append:

⟨original question 𝑥⟩ Here are two candidate responses in <candidate> </candidate> tags to the question above. One is correct and another is wrong. Use these as references to help you solve the problem. <candidate>

- ⟨candidate1 – compressed 𝑦T(+) or 𝑦S(−), shuffled⟩ </candidate> <candidate>
- ⟨candidate2 – the other one⟩ </candidate>

Figure 17 | BCQ reformulated-prompt template (one correct teacher candidate paired with one wrong student candidate, shuffled).

NCQ (Fig. 18). For each hard question with at least one wrong student rollout, we collect every wrong rollout in the current group, parse each rollout’s final boxed answer (deduplicated and joined with “, ”), teacher-compress every wrong rollout, and append: The shared RL closer of

⟨original question 𝑥⟩ The following answers are all WRONG: \boxed{𝑎1}, \boxed{𝑎2}, ..., \boxed{𝑎𝐾}. Below are the incorrect reasoning processes in <candidate> </candidate> tags. <candidate>

- ⟨compressed wrong rollout 1⟩ </candidate> <candidate>
- ⟨compressed wrong rollout 2⟩ </candidate>

... <candidate> ⟨compressed wrong rollout 𝐾⟩ </candidate>

Figure 18 | NCQ reformulated-prompt template (aggregates all wrong student rollouts in the group).

Appendix B.2 (“You FIRST think about the reasoning process ...put the final answer in \boxed{}”) is then applied on top of 𝑥BCQ and 𝑥NCQ, identical to new questions; the student therefore samples 𝑦BCQ,𝑦NCQ∼𝜋𝜃(·|𝑥BCQ/NCQ) in the same think-then-boxed format used everywhere else.

Prompt length budget and the role of dataset filtering. Plain student rollouts use a 4,096-token max prompt budget (Tab. 8, Rollouts row). This budget is enforced only at dataset construction time: ZPPO-77K is filtered so that every base prompt 𝑥 – the original question text plus its post-tokenization image tokens, plus the shared RL closer – fits comfortably inside 4,096 tokens (details in Appendix B). At training time, BCQ and NCQ append additional text on top of 𝑥: (i) the per-block instruction string and <candidate>...</candidate> tags, (ii) the parsed wrong-answer list (NCQ only), and (iii) the teacher-compressed candidate traces (each individually capped at 512 tokens by candidate compression; see the previous paragraph). BCQ and NCQ rollouts do not apply the 4,096-token plain-rollout cap and are bounded only by the policy’s 262K architectural context window (the Qwen3.5 long-context backbone), so the per-block instruction, candidate traces, and parsed wrong-answer list are passed to the student without truncation.

#### D.2. Baseline implementations

This subsection describes the implementation of every baseline reported in Tab. 1, Tab. 2, Tab. 3, Tab. 13, and Tab. 4. All baselines share the same student initialization, the same dataset (ZPPO77K), the same evaluation pipeline, and the same rollout-side hyperparameters (rollout batch size, 𝐺S/𝐺T, sequence-length budget, optimizer, schedule). The gradient-side differs by loss family: RL methods (GRPO, GRPO†, ZPPO, Hint, Prefix) split the per-step rollout batch into 𝐼 equal-sized mini-batches and apply 𝐼 gradient updates on the clipped surrogate (with PPO inner-epoch = 1 so no rollout token is re-iterated), while distillation methods (Off-Distill, On-Distill) apply a single gradient update on the entire per-step rollout batch under the JSD imitation loss. Each rollout token therefore enters the gradient exactly once per rollout step under both families.

Off-policy distillation. We instantiate the standard teacher-trajectory imitation paradigm of Hinton-style knowledge distillation [17] as adapted to sequence models by Sanh et al. [19], with two modifications motivated by the RL post-training setting: (O1) we filter the teacher’s trajectories by correctness using the same rule-based reward as ZPPO before they enter the imitation loss, instead of imitating the teacher unconditionally; (O2) we draw teacher trajectories online on every rollout step, with no precomputed pool, so that the buffer ablation in Tab. 3 is not confounded by a one-time amortized teacher pre-compute. Concretely, on every rollout step, for every prompt 𝑥 in the rollout batch – both new prompts 𝑥∈𝑋new and (under the † variant) replayed prompts 𝑥∈𝑋replay

– we draw 𝐺T = 4 teacher rollouts {𝑦T(𝑔)(𝑥)}𝐺𝑔=1T ∼ 𝜋T(·|𝑥) online from the frozen teacher, grade

them with the same rule-based reward as ZPPO, and keep only the correct subset {𝑦T(+,𝑛)(𝑥)}𝑁𝑛=1𝑥 (0≤𝑁𝑥 ≤𝐺T). The teacher samples for a given 𝑥 on visit 𝑘 are statistically independent of those

on visit 𝑘−1 (whether or not 𝑥 is replayed). In parallel we draw 𝐺S=8 student rollouts {𝑦S(𝑔)(𝑥)} that are used for hard-prompt bookkeeping (and, under †, for buffer admission/graduation), but

the student rollouts never enter the imitation gradient. When 𝑁𝑥 > 0, we minimize the average over the 𝑁𝑥 correct teacher targets of the per-token Jensen–Shannon divergence (JSD) between 𝜋𝜃(·|𝑥,𝑦T(+,𝑛)) and the teacher’s per-token distribution at the same prefix [17, 19, 21] – this is the standard sequence-distillation imitation loss specialized to the teacher’s correct trajectories; when 𝑁𝑥=0 (the teacher fails on 𝑥 at this step), the question contributes no imitation loss. Optimization shares the same AdamW configuration as ZPPO. The † variant additionally maintains the prompt replay buffer ([O3]: our extension) with identical admission/graduation/eviction policies as ZPPO; because both new and replayed prompts re-draw teacher rollouts every step, Off-Distill† matches ZPPO’s teacher-freshness symmetrically on every visit. Compared to a hypothetical offline-pool variant that caches the first batch of 𝐺T teacher rollouts per 𝑥 and reuses them on every revisit, the fully online setup we adopt costs proportionally more teacher-side compute (linear in the number of rollout steps a prompt participates in), but it removes the cache-staleness confound when interpreting the buffer ablation in Tab. 3 – any gain ZPPO shows over Off-Distill† cannot be attributed to ZPPO seeing fresher teacher samples on the same hard prompt.

On-policy distillation. We follow the on-policy distillation paradigm of Agarwal et al. [25], with a per-token Jensen–Shannon divergence (JSD) between student and teacher as the imitation loss. Specifically: (P1) the student samples its own trajectories rather than imitating teacher trajectories (the defining property of on-policy distillation); (P2) the frozen teacher rescores the student’s trajectories to produce per-token target distributions; (P3) the loss is the per-token JSD between student and teacher at the same prefix. Our only setup-specific extension is the optional buffer ([O4]) for the † variant, which is the same prompt replay buffer used by ZPPO and Off-Distill†. Concretely,

for each question 𝑥, the student first samples the full group of 𝐺S=8 responses {𝑦S(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥),

Algorithm 2: Off-policy distillation training step (with optional † buffer variant). Each step is tagged with the Hinton [17] (H1–H2) or Sanh [19] (S1–S2) component it inherits; O1–O3 mark our setup-specific extensions. Fully online: 𝐺T=4 teacher rollouts are drawn on every prompt on every step, with no precomputed pool and no cross-step caching, so the same 𝑥 visited on different steps consumes independent teacher samples.

Input: Student 𝜋𝜃, frozen teacher 𝜋T, dataset 𝒟, buffer ℬ (only for †); teacher group size

𝐺T=4, student group size 𝐺S=8, 𝜌replay,|ℬ|max,𝜏 (only for †).

- 1 // Step tags annotate the component inherited from Hinton-style distillation [17] (H1–H2) or Sanh-style sequence-model distillation [19] (S1–S2); O1–O3 mark our setup-specific extensions (correct-only filtering, fully online teacher sampling, optional buffer).
- 2 Sample 𝑋new∼𝒟 and (for †) 𝑋replay∼ℬ with |𝑋replay|=𝜌replay|𝑋new|; set 𝑋 ←𝑋new∪𝑋replay.
- 3 foreach 𝑥 ∈ 𝑋 do

- 4 Draw 𝐺S student rollouts {𝑦S(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥) and score them for hard-prompt bookkeeping only (not used in the imitation gradient).
- 5 Draw 𝐺T fresh teacher rollouts {𝑦T(𝑔)(𝑥)}𝐺𝑔=1T ∼𝜋T(·|𝑥) online on this step. // [H1, H2] teacher imitation targets; [O2] drawn online each step (no cached pool).
- 6 Grade {𝑦T(𝑔)(𝑥)} with the rule-based reward and keep the correct subset {𝑦T(+,𝑛)(𝑥)}𝑁𝑛=1𝑥 . // [S2, O1] correctness filtering of teacher outputs.
- 7 if 𝑁𝑥=0 (teacher fails on 𝑥 at this step) then

- 8 Skip the imitation gradient for this 𝑥 (its student rollouts still feed hard-prompt bookkeeping / buffer admission).
- 9 else

- 10 Compute the average over 𝑛=1,...,𝑁𝑥 of the per-token Jensen–Shannon divergence

between 𝜋𝜃(·|𝑥,𝑦T(+,𝑛)) and 𝜋T at the same prefix. // [S1] per-token JSD on teacher trajectories.

- 11 end
- 12 end
- 13 Update 𝜋𝜃 on the aggregated JSD loss (AdamW; same schedule as ZPPO).
- 14 († only) Use the student rollout scores to admit {𝑥 : 𝑟¯𝑥 < 𝜏} to ℬ, graduate the rest, FIFO-evict until |ℬ|≤|ℬ|max. // [O3] our extension: optional prompt replay buffer, matching ZPPO’s admission/graduation/FIFO policy.
- 15 return 𝜋𝜃 (and ℬ for †).

we then forward all student responses through the frozen teacher to obtain target logits, and we minimize the per-token JSD between the student and teacher distributions on the student’s own rollouts [25, 21]. The † variant adds the prompt replay buffer in the same way as Off-Distill†.

Algorithm 3: On-policy distillation training step (with optional † buffer variant). Each step is tagged with the Agarwal [25] (P1–P3) component it inherits; O4 marks our setup-specific extension.

Input: Student 𝜋𝜃, frozen teacher 𝜋T, dataset 𝒟, buffer ℬ (only for †); hyperparameters

𝐺S,𝜌replay,|ℬ|max,𝜏 (only for †).

- 1 // Step tags annotate the components inherited from on-policy distillation [25] (P1–P3); O4 marks our setup-specific extension (optional buffer).
- 2 Sample 𝑋new∼𝒟 and (for †) 𝑋replay∼ℬ with |𝑋replay|=𝜌replay|𝑋new|; set 𝑋 ←𝑋new∪𝑋replay.
- 3 foreach 𝑥 ∈ 𝑋 do

- 4 Draw 𝐺S student rollouts {𝑦S(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥). // [P1] student samples its own trajectories.
- 5 Forward every 𝑦S(𝑔) through 𝜋T to obtain teacher per-token target distributions on the student’s own responses. // [P2] teacher rescores the student’s trajectories.
- 6 Compute the per-token Jensen–Shannon divergence loss between 𝜋𝜃(·|𝑥,𝑦S(𝑔)) and these teacher targets for all 𝑔=1,...,𝐺S. // [P3] per-token JSD on the student’s own samples.
- 7 end
- 8 Update 𝜋𝜃 on the aggregated on-policy JSD loss.
- 9 († only) Score {𝑦S(𝑔)}𝐺𝑔=1S with the binary reward to obtain 𝑟¯𝑥; update ℬ with the same admission/graduation/FIFO policy as ZPPO. // [O4] our extension: optional prompt replay buffer, matching ZPPO’s admission/graduation/FIFO policy.
- 10 return 𝜋𝜃 (and ℬ for †).

GRPO and GRPO†. GRPO inherits three families of components from the recent RL posttraining literature, and our setup adds three minor changes. Adopted from GRPO [36]: (G1) the group-relative advantage formulation that draws 𝐺S rollouts per prompt and centers their rewards within the per-prompt group; (G2) the PPO-style clipped surrogate objective on the centered advantages. Adopted from DAPO [8]: (Da1) asymmetric clip-higher with (𝜖low,𝜖high)=(0.20,0.28); (Da2) token-level loss aggregation rather than sequence-level; (Da3) no KL penalty against a reference policy (kl_coef=0). Adopted from REINFORCE++ [37]: (R1) the two-step advantage estimator (within-group centering in Step 1 and across-group batch normalization in Step 2; see Eqs. 2–3). Our setup-specific recipe choices on top: (OG1) lower inner-iteration count 𝐼 = 4 instead of the standard 𝐼 =16 (Sec. 4.5); (OG2) optional prompt replay buffer in the † variant; (OG3) exclusion of zero-advantage (all-correct or all-wrong) groups from the batch-statistics computation in Step 2 of the two-step estimator (Eqs. 2–3; ablated in Sec. 4.5(ii)). GRPO is identical to ZPPO with the BCQ, NCQ, and replay-buffer branches all disabled; GRPO† adds the buffer alone (𝜌replay=0.25, same admission/graduation/eviction policy as ZPPO).

Hint baseline (Tab. 4). On each hard question, we follow Guide-GRPO [93] (with SEELE [52] as another member of the same prompt-injected hint family). As with Off-Distill above, teacher rollouts are drawn online on every step – there is no precomputed teacher pool – but only on the 𝑋aug subset of hard prompts that the augmentation branch actually consumes (a small fraction 𝜌aug=0.25 of 𝑋new), which keeps the teacher-side cost of Hint substantially below that of Off-Distill. Adopted from Guide-GRPO [93]: (A1) appending a natural-language hint to the prompt as the

Algorithm 4: GRPO training step (with optional † buffer variant; matches Algorithm 1 with the BCQ/NCQ branches removed). Each step is tagged with the GRPO [36] (G1–G2), DAPO [8] (Da1–Da3), or REINFORCE++ [37] (R1) component it inherits; OG1–OG3 mark our recipe-side choices, including the zero-advantage-group exclusion from batch statistics.

Input: Student 𝜋𝜃, dataset 𝒟, buffer ℬ (only for †); hyperparameters 𝐺S,𝜌replay,|ℬ|max,𝐼,𝜏.

- 1 // Step tags annotate the component inherited from GRPO [36] (G1–G2), DAPO [8] (Da1–Da3), or REINFORCE++ [37] (R1); OG1–OG3 mark our recipe-side choices.
- 2 Sample 𝑋new∼𝒟 and (for †) 𝑋replay∼ℬ with |𝑋replay|=𝜌replay|𝑋new|; set 𝑋 ←𝑋new∪𝑋replay.
- 3 foreach 𝑥 ∈ 𝑋 do

- 4 Draw 𝐺S student rollouts {𝑦S(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥). // [G1] group-relative GRPO: 𝐺S rollouts per prompt to form the per-prompt group.
- 5 Score each rollout with the binary reward and compute group-relative advantages via the two-step estimator (within-group centering, then cross-group normalization over the non-trivial subset; Eqs. 2–3). // [G1, R1, OG3] group-relative two-step advantages with zero-advantage-group exclusion.
- 6 end
- 7 Update 𝜋𝜃 for 𝐼 iterations on all student rollouts (no BCQ/NCQ branch is constructed), using the asymmetric clip-higher PPO surrogate at the token level and no reference-KL penalty. // [G2] clipped surrogate; [Da1] clip-higher (0.20,0.28); [Da2] token-level loss; [Da3] no reference-KL; [OG1] 𝐼 =4.
- 8 († only) Compute mean rollout accuracy 𝑟¯𝑥; admit {𝑥 : 𝑟¯𝑥 < 𝜏} to ℬ, graduate the rest, FIFO-evict until |ℬ| ≤ |ℬ|max. // [OG2] our extension: optional prompt replay buffer with admission/graduation/FIFO eviction.
- 9 return 𝜋𝜃 (and ℬ for †).

augmentation mechanism, before any student rollout; (A2) using a frozen teacher to generate that hint from a correct teacher trace; (A3) keeping the student rollout from the hint-augmented prompt on-policy at the response-token level – the gradient is computed under 𝜋𝜃(·|𝑥Hint) with every response token sampled from the current student, identical to Guide-GRPO’s guided-rollout formulation modulo the IS ratio it adds on top; (A4) applying the hint augmentation only on hard prompts where the unguided student rollouts fail, rather than uniformly over the dataset.

Adopted from SEELE [52]: (B1) sourcing the hint material from a teacher-derived correct trace rather than from a hand-written rule or a separate hint generator (SEELE uses partial solutions; we use an answer-free summary of the same correct teacher rollout, see the hint-generation prompt below); (B2) one hint per question per rollout step, i.e. a single shared hint string for the augmented group of 𝐺S student rollouts (SEELE generates one hint per training sample per step); (B3) the hint-augmented branch is applied as an additional augmentation group on top of plain student rollouts, rather than replacing them, mirroring SEELE’s augmented-sample construction.

The goal of this baseline is to capture the prompt-side hint mechanism of Guide-GRPO (A1–A4) and SEELE (B1–B3). Algorithm 5 annotates each step with the component it inherits. Hard questions on which the teacher itself never succeeds on the 𝐺T = 4 online rollouts at this step (i.e. 𝑁𝑥 = 0) contribute no hint group for that step: their plain student rollouts still feed buffer admission/graduation, but Algorithm 5 skips the hint-augmented branch entirely (the empiricallyobserved fraction of such “teacher-fails” hard questions is small at every scale, since the 27B teacher is much stronger than the ≤ 9B students at exactly the hard prompts the student is failing on). The hint-generation prompt is shown in Fig. 19: The generated hint ℎT is then appended to the

Read the solved response below and write a concise hint that helps solve the original question. Do NOT reveal the final answer, do NOT include any \boxed{} expression, and do NOT copy the full solution. Keep only high-level guidance or the key intermediate idea. <response>

𝑦T(+) </response>

Figure 19 | Hint-generation prompt for the Hint baseline (answer-free guidance from a correct teacher trace).

question inside <hint>··· </hint> tags, and the student samples a new rollout group from this hintaugmented prompt. Because every response token is sampled by the current student, the gradient remains on-policy at the response-token level (the prompt is augmented with teacher-derived hint text). Unlike BCQ, however, Hint gives one-sided teacher guidance without forcing the student to discriminate between a correct and an incorrect candidate; unlike NCQ, it never exposes the student’s own collective negatives.

Prefix baseline (Tab. 4). On each hard question, we follow BREAD [51] (with StepHint [50] as another member of the same prefix-injection family). For implementation simplicity we omit BREAD’s adaptive Episode Anchor Search (EAS) and fix the prefix at a single token fraction 𝛼=0.4 shared across all hard questions and all rollout steps; every other ingredient of BREAD (expert-trace anchor, branched rollouts, group-relative advantages, failure-conditioned application) is adopted as-is. As in Off-Distill and Hint, teacher rollouts are drawn online on every step – there is no precomputed teacher pool – and the draw happens only on the 𝑋aug subset of hard prompts that the prefix branch actually consumes.

Adopted from BREAD [51]: (E1) the teacher-forced prefix anchor as the start of the student rollout, drawn from a correct teacher rollout sampled online on this step (BREAD’s “expert trace anchor”);

Algorithm 5: Hint training step. Each step is tagged with the Guide-GRPO [93] (A1–A4) and SEELE [52] (B1–B3) component it inherits. Teacher rollouts are drawn online on this step, on the 𝑋aug subset only, with no precomputed pool. Hint augmentation requires a correct teacher rollout at this step; hard questions without one (i.e. teacher fails on all 𝐺T rollouts) fall back to the plain student rollout only.

Input: Student 𝜋𝜃, frozen teacher 𝜋T, dataset 𝒟, buffer ℬ; teacher group size 𝐺T=4,

hyperparameters 𝐺S,𝜌replay,𝜌aug,|ℬ|max,𝐼,𝜏.

- 1 // Step tags annotate the component inherited from Guide-GRPO (A1–A4) or SEELE (B1–B3). Teacher rollouts are drawn online on this step, on 𝑋aug only, with no precomputed pool.
- 2 Sample 𝑋new∼𝒟 and 𝑋replay∼ℬ with |𝑋replay|=𝜌replay|𝑋new|; set 𝑋 ←𝑋new∪𝑋replay.
- 3 foreach 𝑥 ∈ 𝑋 do

- 4 Draw 𝐺S student rollouts {𝑦S(𝑔)}∼𝜋𝜃(·|𝑥) and compute 𝑟¯𝑥.
- 5 end
- 6 𝑋hard←{𝑥∈𝑋 : 𝑟¯𝑥<𝜏}; rank by ascending 𝑟¯𝑥 and keep the top 𝜌aug|𝑋new| as 𝑋aug. // [A4] Guide-GRPO’s hard-prompt-selective application of the hint augmentation.
- 7 foreach 𝑥 ∈ 𝑋aug do

- 8 Draw 𝐺T fresh teacher rollouts {𝑦T(𝑔)(𝑥)}𝐺𝑔=1T ∼𝜋T(·|𝑥) online on this step, grade them, and keep the correct subset {𝑦T(+,𝑛)(𝑥)}𝑛𝑁=1𝑥 . // online; no precomputed pool.
- 9 if 𝑁𝑥=0 (the teacher has no correct rollout on 𝑥 at this step) then

- 10 Skip the hint branch for this 𝑥: no hint group is constructed, and 𝑥 contributes only its plain {𝑦S(𝑔)} to the gradient and to buffer bookkeeping.
- 11 else

- 12 Pick one 𝑦T(+)∈{𝑦T(+,𝑛)(𝑥)}𝑁𝑛=1𝑥 (uniform random over the correct rollouts drawn at this step). // [B1] SEELE-style hint material sourced from a teacher-derived correct trace.
- 13 Query 𝜋T with the hint-generation prompt above, instantiated on 𝑦T(+), to obtain an answer-free hint ℎT. // [A2] frozen teacher generates the hint (Guide-GRPO-style).
- 14 Form 𝑥Hint←𝑥‖<hint>ℎT </hint>. // [A1, B2] one shared hint string appended to the prompt.
- 15 Draw 𝐺S student rollouts {𝑦Hint(𝑔) }𝑔𝐺=1S ∼𝜋𝜃(·|𝑥Hint) as a fresh group with its own group identifier uidHint (advantages computed within this 𝐺S-sized group, identical to ZPPO’s BCQ/NCQ groups). // [A3, B3] on-policy augmentation group on top of plain rollouts.
- 16 end
- 17 end
- 18 Update 𝜋𝜃 for 𝐼 iterations on plain {𝑦S(𝑔)} and hint-augmented {𝑦Hint(𝑔) } rollouts under the same group-relative advantages and batch normalization as ZPPO.
- 19 Update ℬ with the same admission/graduation/FIFO policy as ZPPO.
- 20 return 𝜋𝜃, ℬ.

(E2) the branched-rollout structure that draws 𝐺S=8 student continuations from the same shared anchor (𝑥, 𝑝), so all 𝐺S rollouts in the augmented group condition on the identical prefix (BREAD’s defining “branched rollouts” design); (E3) GRPO-style group-relative advantages computed over those 𝐺S branched continuations (BREAD scores the branched group exactly this way, including the same group-relative normalization that we share with ZPPO/BCQ/NCQ); (E4) failure-conditioned application – BREAD invokes prefix injection on hard prompts where the student fails unaided, and we similarly route Prefix only through the 𝑋aug subset of hard questions selected by our 𝜌aug rule; (E5) teacher-prefix masking from the policy loss, i.e. the per-token policy-gradient summation runs over the student-continuation tokens 𝑐(𝑔) only and the teacher-forced prefix tokens 𝑝 are masked out so they cannot inherit the continuation’s advantage (this matches BREAD’s Eq. 1 in [51], whose objective sums 𝑡=1...|𝑐(𝑔)| with 𝑝 acting purely as conditioning context). Omitted for simplicity: BREAD’s Episode Anchor Search (EAS), the per-question binary search over an episode-split expert trace that picks the shortest sufficient prefix; we replace it with the fixed 𝛼 = 0.4 token fraction above.

Adopted from StepHint [50]: (F1) using a strong frozen teacher (rather than human annotations or a curated SFT dataset) as the source of the reasoning trace that the prefix is cut from (StepHint draws traces from DeepSeek-R1-class teachers, we draw from our 27B Qwen3.5 teacher); (F2) truncating the trace at the initial portion of the solution and feeding only that prefix to the student (StepHint’s “initial few steps as hints”), in our case at the fixed token fraction 𝛼=0.4; (F3) selective application of the prefix-injection to a subset of training prompts rather than to the entire dataset (StepHint’s selective hinting policy).

The goal of this baseline is to capture the response-prefix mechanism of BREAD (E1–E4) and StepHint (F1–F3). Algorithm 6 annotates each step with the component it inherits. Hard questions on which the teacher itself never succeeds on the 𝐺T=4 online rollouts at this step (i.e. 𝑁𝑥=0) fall back to the plain student rollout only, exactly as in Algorithm 5. All other hyperparameters (replay buffer, 𝐺S, optimizer) are identical to ZPPO.

#### D.3. Compute cost per run

Tab. 9 reports per-run wall-clock time and total training FLOPs for the headline methods of the main paper across all four student scales {0.8B,2B,4B,9B}. Wall-clock time is the measured per-run cost under the shared hardware setup (64×H100-80GB, 8 nodes ×8 GPUs, per-node 6:2 student/teacher split), averaged across the per-scale runs to the nearest hour. FLOPs are estimated from the algorithms in Appendix D under the standard token-level convention – 2𝑁 per token for KV-cached forwards (student rollouts, teacher rollouts, teacher KL target forwards, candidate compression) and 6𝑁 per token for forward+backward – with 𝐿gen=12,288 (the configured maximum response length, Tab. 8) used as a uniform upper bound across every method. RL methods (GRPO, GRPO†, ZPPO) apply the 𝐼 =4 inner-iteration multiplier on every rollout token that enters the gradient – including, for ZPPO, the BCQ and NCQ groups in addition to plain and replay rollouts. Distillation methods (Off-Distill, On-Distill) perform a single gradient update per rollout step (no PPO inner loop) but incur a teacher KL forward at 2𝑁T per target token. Crucially, Off-Distill is fully online in our setup (Algorithm 2): no precomputed teacher pool is materialized in advance, and 𝐺T =4 fresh teacher rollouts are drawn on every prompt on every step, then graded and reduced to their correct subset before entering the imitation gradient. The teacher-side FLOPs of Off-Distill therefore scale as 𝐺T·(total prompts the gradient sees across all steps) rather than as a one-time amortized precompute; Off-Distill† pays the same per-step teacher cost on both new and replayed prompts, so by construction it matches ZPPO’s teacher-freshness symmetrically on every visit. The Hint and Prefix baselines (Algorithms 5 and 6) likewise draw 𝐺T=4 teacher rollouts online on every step, but

Algorithm 6: Prefix training step. Steps are tagged with the BREAD [51] (E1–E5) and StepHint [50] (F1–F3) component each inherits, including BREAD-style teacher-prefix masking from the policy loss (E5). EAS is omitted and the prefix fraction is fixed at 𝛼=0.4. Teacher rollouts are drawn online on 𝑋aug only, and each prefix-augmented question contributes a 𝐺S-sized group sharing the same prefix, scored with group-relative advantages identical to ZPPO/BCQ/NCQ.

Input: Student 𝜋𝜃, frozen teacher 𝜋T, dataset 𝒟, buffer ℬ; teacher group size 𝐺T=4, prefix

fraction 𝛼=0.4, hyperparameters 𝐺S,𝜌replay,𝜌aug,|ℬ|max,𝐼,𝜏.

- 1 // Step tags annotate the component inherited from BREAD (E1–E5) or StepHint (F1–F3).

EAS omitted; fixed 𝛼=0.4 used instead. Teacher rollouts drawn online on this step on 𝑋aug only, no precomputed pool. Prefix tokens 𝑝 are masked from the policy loss (BREAD-style); only student-continuation tokens 𝑐(𝑔) enter the gradient.

- 2 Sample 𝑋new∼𝒟 and 𝑋replay∼ℬ with |𝑋replay|=𝜌replay|𝑋new|; set 𝑋 ←𝑋new∪𝑋replay.
- 3 foreach 𝑥 ∈ 𝑋 do

- 4 Draw 𝐺S student rollouts and compute 𝑟¯𝑥 as in GRPO.
- 5 end
- 6 Select 𝑋aug as the top 𝜌aug|𝑋new| hardest questions (¯𝑟𝑥<𝜏). // [E4, F3] failure-conditioned selective prefix injection.
- 7 foreach 𝑥 ∈ 𝑋aug do

- 8 Draw 𝐺T fresh teacher rollouts {𝑦T(𝑔)(𝑥)}𝐺𝑔=1T ∼𝜋T(·|𝑥) online on this step, grade them, and keep the correct subset {𝑦T(+,𝑛)(𝑥)}𝑛𝑁=1𝑥 . // online; no precomputed pool.
- 9 if 𝑁𝑥=0 then

- 10 Skip the prefix branch; 𝑥 contributes only its plain {𝑦S(𝑔)}.
- 11 else

- 12 Pick one 𝑦T(+)=(𝑡1,...,𝑡𝐿)∈{𝑦T(+,𝑛)(𝑥)}𝑁𝑛=1𝑥 (uniform random). // [E1, F1] expert-trace anchor from a strong online teacher.
- 13 Form the shared forced prefix 𝑝←(𝑡1,...,𝑡⌊𝛼𝐿⌋). // [F2] fixed token-fraction cut (𝛼=0.4); EAS omitted.
- 14 Draw 𝐺S student continuations {𝑐(𝑔)}𝐺𝑔=1S ∼𝜋𝜃(·|𝑥,𝑝) autoregressively from the same shared (𝑥,𝑝). // [E2] BREAD branched rollouts under one shared anchor.
- 15 Assemble the prefix-augmented group {𝑦Prefix(𝑔) =𝑝‖𝑐(𝑔)}𝐺𝑔=1S under a fresh uidPrefix; compute group-relative advantages over the 𝐺S continuations with the same REINFORCE++ estimator and zero-advantage-group exclusion as ZPPO. // [E3] BREAD group-relative advantages.
- 16 Score the augmented group in the PPO surrogate under 𝜋𝜃(·|𝑥,(𝑝‖𝑐(𝑔))<𝑡), with the policy-gradient summation restricted to the student-continuation tokens 𝑐(𝑔) and prefix tokens 𝑝 masked out. // [E5] BREAD-style prefix masking.
- 17 end
- 18 end
- 19 Update 𝜋𝜃 for 𝐼 iterations on plain {𝑦S(𝑔)} and prefix-augmented {𝑦Prefix(𝑔) } rollouts.
- 20 Update ℬ with the same admission/graduation/FIFO policy as ZPPO.
- 21 return 𝜋𝜃, ℬ.

only on the 𝑋aug subset of hard prompts (𝜌aug=0.25of |𝑋new|), which keeps their teacher-side cost substantially below Off-Distill’s. On-Distill instead forwards every 𝐺S=8 student rollout through the 27B teacher for KL targets, which dominates its FLOPs column.

Wall-clock note. Each ZPPO run terminates after the rollout-step budget in Tab. 8 (one rollout batch per step, 𝐼 gradient updates per rollout step over equal-sized mini-batches, 𝐼-times-the-rolloutstep-budget gradient updates per run in total). The reported wall-clock is student-side compute bound: on every step the per-node 6:2 student/teacher split runs teacher generation, teacher-side candidate compression, and the sidecar judge in parallel with the student rollout, and the student rollout phase (longer responses, larger group size 𝐺S=8) is always slower than the teacher phase (𝐺T = 4), so the teacher branch fully overlaps and does not extend the critical path. FLOPs in Tab. 9 are upper-bound estimates that count every gradient-counted and generated token at the uniform configured cap 𝐿gen=12,288 tokens with the standard 6𝑁/2𝑁 token-level convention, so per-method FLOPs track the algorithm-level cost (rollouts + gradient passes) and are insensitive to per-method variations in average actual response length.

0.8B 2B 4B 9B

Method Time (h) FLOPs Time (h) FLOPs Time (h) FLOPs Time (h) FLOPs Off-Distill 85 4.7·1020 91 5.2·1020 95 5.9·1020 104 8.0·1020 Off-Distill† 88 5.1·1020 96 5.7·1020 100 6.8·1020 108 9.2·1020 On-Distill 75 6.8·1020 78 7.4·1020 82 9.2·1020 88 1.4·1021 On-Distill† 78 8.0·1020 82 9.2·1020 83 1.2·1021 90 1.7·1021 GRPO 61 2.3·1020 69 5.6·1020 75 1.1·1021 90 2.5·1021 GRPO† 61 2.8·1020 69 6.8·1020 76 1.4·1021 93 3.1·1021 ZPPO 68 4.9·1020 79 9.8·1020 92 1.8·1021 110 3.9·1021

- Table 9 | Per-run training cost for the headline methods of the main paper. Time (h) is measured wall-clock on the shared cluster (64×H100-80GB), averaged across the per-scale runs to the nearest hour; FLOPs follow

the standard token-level 6𝑁/2𝑁 convention. The protocol {𝐺S,𝐺T,𝐼,𝜌aug,𝜌replay}={8,4,4,0.25,0.25} runs over the ZPPO-77K corpus. Compute-counting conventions, per-method derivations, and the wall-clock decomposition are described in Appendix D.3. Hint, Prefix, and the +BCQ/+NCQ/+Both componentisolation rows of Tab. 4–Tab. 3 are not listed individually: their per-step rollout/gradient budgets match GRPO† up to the small 𝜌aug = 0.25 augmentation branch, so their wall-clock and FLOPs fall within the GRPO†–ZPPO range at every scale. † denotes augmentation with the prompt replay buffer.

Other rows follow from removing or replacing the BCQ/NCQ rollout and the policy gradient pass. (i) GRPO differs from ZPPO only in disabling BCQ/NCQ; in our parallel implementation teacher generation is fully overlapped with student rollout, so the wall-clock difference between ZPPO and GRPO is essentially the cumulative cost of the BCQ/NCQ rollout phase. (ii) GRPO†, On-Distill†, Off-Distill† add the prompt replay buffer on top of their base method; the buffer contributes a small additional overhead coming from buffer admission/graduation bookkeeping plus the longer responses that hard prompts elicit. (iii) Off-policy distillation draws 𝐺S=8 student rollouts and 𝐺T=4 online teacher rollouts on every prompt every step (Algorithm 2); the teacher rollouts are graded by the rule-based reward and only their correct subset enters the imitation JSD. There is no precomputed pool to amortize this cost, so Off-Distill’s teacher-side FLOPs accumulate linearly with the number of rollout steps. The † variant inherits exactly the same per-step teacher cost on the additional 𝜌replay|𝑋new| replayed prompts, so on a replayed visit Off-Distill† and ZPPO see independent, equally fresh teacher samples by construction. Teacher work is overlapped with student rollouts on the teacher-side 2-GPU-per-node pool and therefore mostly stays off the critical path, but it contributes additively to the FLOPs column; the relative ordering of Off-Distill and

GRPO in Tab. 9 reflects this scaling: at the smallest student (0.8B) the teacher-side cost dominates and Off-Distill sits above GRPO, whereas at larger students the GRPO column grows faster because RL methods apply the 𝐼 =4 inner-iteration multiplier on top of the Θ(𝑁) student-side per-token cost. (iv) On-policy distillation samples 𝐺S =8 student rollouts per question and forwards all eight through the 27B teacher to form per-token KL targets; this makes its FLOPs substantially larger than GRPO, whose repeated logit computation is student-side only.

### E. Extended main-paper tables (4B and 9B)

The 4B and 9B blocks deferred from the main paper for space reasons follow the same row recipes and the same per-cell formatting rules (per-column max bolded, gray-shaded Avg). The main and generalization tables additionally carry a Δ=ZPPO−maxother row coloured green/red, whereas the component-ablation tables omit the Δ row because each non-ZPPO recipe row in those tables uses a different “other” as its natural reference. Tab. 10 extends Tab. 1, Tab. 11 extends Tab. 2, and

- Tab. 12 extends Tab. 3.

16 VLM Benchmarks AI2D BabyV CharXiv DynaM EmbSp InfoVQA MVerse MVision MVista MMMUPro MM-Vet OCREN OCRZH VisP VBlind WeMath Avg Qwen3.5-4B 88.7 16.0 78.7 59.3 82.1 88.5 83.1 64.3 83.9 64.0 78.4 50.8 28.9 47.4 66.4 89.2 66.9 + Off-Distill 87.2 17.5 80.3 60.9 80.6 86.4 83.7 67.0 83.0 64.8 76.8 51.0 29.4 47.6 66.9 90.2 67.1 + Off-Distill† 87.0 17.7 80.0 60.7 81.1 86.8 83.7 67.5 82.9 65.1 77.7 52.0 28.3 48.8 67.4 90.6 67.3 + On-Distill 87.1 17.5 80.7 61.5 81.0 87.2 83.9 67.8 84.0 65.6 78.2 51.3 29.5 48.2 67.8 90.6 67.6

- + On-Distill† 88.2 17.4 80.5 61.6 80.5 87.7 84.5 67.6 83.8 65.6 78.2 52.0 30.0 48.7 67.8 91.4 67.8

- + GRPO 88.4 18.9 80.5 61.5 81.3 88.0 84.6 68.3 84.1 66.0 78.7 52.2 30.6 49.2 68.4 90.8 68.2

+ GRPO† 88.5 20.9 81.4 61.4 82.0 88.1 84.5 68.3 85.3 65.5 78.4 52.4 30.4 51.0 69.4 91.8 68.7

- + ZPPO 89.6 27.8 82.8 63.9 82.4 88.8 85.7 72.2 85.4 67.1 79.8 54.6 33.0 54.5 74.9 92.6 70.9 Δ +0.9 +6.9 +1.4 +2.3 +0.3 +0.3 +1.1 +3.9 +0.1 +1.1 +1.1 +2.2 +2.4 +3.5 +5.5 +0.8 +2.2 Qwen3.5-9B 90.2 22.9 82.9 62.5 82.1 87.8 86.9 70.8 86.3 67.7 83.0 50.0 31.7 55.3 71.7 90.9 70.2

+ Off-Distill 89.1 23.0 82.0 64.7 80.9 88.1 85.7 73.2 85.5 67.9 81.4 51.1 32.4 54.4 73.4 91.0 70.2 + Off-Distill† 89.2 23.4 82.6 64.8 80.7 88.6 86.1 72.8 85.5 68.6 81.1 51.6 32.2 55.2 73.7 91.5 70.5 + On-Distill 89.5 24.3 82.2 64.6 81.0 89.5 86.5 72.9 86.4 69.1 82.8 51.6 32.6 54.1 73.3 91.5 70.7 + On-Distill† 89.1 24.0 83.0 64.6 80.9 89.5 86.4 73.5 86.4 69.6 82.9 52.0 32.5 55.6 73.6 91.8 71.0 + GRPO 89.8 24.7 83.4 65.3 80.4 89.6 87.4 74.3 86.0 69.1 83.2 52.2 32.9 55.9 73.3 91.8 71.2 + GRPO† 90.2 24.0 83.2 65.1 81.4 89.6 87.9 75.0 87.3 68.9 83.0 52.2 32.9 55.7 74.3 92.2 71.4

- + ZPPO 90.8 26.3 83.7 66.7 82.1 91.4 88.1 75.9 87.5 70.2 83.9 56.7 35.6 56.4 79.4 92.9 73.0 Δ +0.6 +1.6 +0.3 +1.4 0.0 +1.8 +0.2 +0.9 +0.2 +0.6 +0.7 +4.5 +2.7 +0.5 +5.1 +0.7 +1.6

##### Table 10 | Main results on the 16 VLM benchmarks at 4B and 9B; † denotes augmentation with the prompt replay buffer. Benchmark abbreviations: Appendix B.2.

10 LLM Benchmarks 5 Video Benchmarks

Method AIME25 AIME26 CEval GPQA-D HLE IMO-AB MMLU MMLU-Pro MMLU-Rd MultiCh Avg MMVU MVBench VMME VMMES VMMMU Avg Qwen3.5-4B 53.3 63.3 83.6 69.7 7.8 35.5 85.3 79.4 89.5 48.9 61.6 69.2 70.2 68.1 69.9 80.3 71.5 + Off-Distill† 53.3 63.3 83.0 68.2 6.3 35.7 83.7 77.3 88.1 48.1 60.7 67.9 67.4 66.4 68.8 78.2 69.7 + On-Distill† 53.3 63.3 82.6 69.3 7.8 35.8 85.0 77.8 88.6 48.2 61.2 68.7 67.8 67.0 69.3 78.9 70.3 + GRPO† 53.3 63.3 84.6 73.5 7.1 40.6 85.9 79.2 88.9 50.7 62.7 69.2 69.1 68.2 68.7 79.4 70.9 + ZPPO 56.7 66.7 86.1 77.3 8.3 48.0 86.0 79.7 90.1 56.0 65.5 70.6 69.6 68.5 70.2 80.2 71.8 Δ +3.4 +3.4 +1.5 +3.8 +0.5 +7.4 +0.1 +0.3 +0.6 +5.3 +2.8 +1.4 −0.6 +0.3 +0.3 −0.1 +0.3

Qwen3.5-9B 60.0 70.0 84.0 77.3 8.9 38.8 87.3 81.1 90.5 50.8 64.9 73.8 71.4 70.5 72.4 80.9 73.8 + Off-Distill† 60.0 70.0 84.5 76.8 8.4 39.3 87.5 79.9 89.5 51.0 64.7 72.5 69.6 68.5 71.5 80.8 72.6 + On-Distill† 60.0 70.0 84.2 77.4 9.2 39.4 87.2 81.2 90.5 52.1 65.1 73.0 70.4 69.6 71.6 80.7 73.1 + GRPO† 63.3 70.0 85.3 77.8 8.8 43.6 87.4 81.3 90.9 53.0 66.2 73.1 71.5 69.3 71.2 82.0 73.4 + ZPPO 70.0 73.3 87.2 78.8 9.8 49.8 88.1 82.0 91.7 56.8 68.8 74.2 71.7 70.2 72.8 82.1 74.2 Δ +6.7 +3.3 +1.9 +1.0 +0.6 +6.2 +0.6 +0.7 +0.8 +3.8 +2.6 +0.4 +0.2 −0.3 +0.4 +0.1 +0.4

##### Table 11 | Evaluation on the 10 LLM and 5 Video benchmarks (probing generalization beyond the multimodal training data); † denotes augmentation with the prompt replay buffer. Benchmark abbreviations: Appendix B.2.

16 VLM Benchmarks AI2D BabyV CharXiv DynaM EmbSp InfoVQA MVerse MVision MVista MMMUPro MM-Vet OCREN OCRZH VisP VBlind WeMath Avg Qwen3.5-4B 88.7 16.0 78.7 59.3 82.1 88.5 83.1 64.3 83.9 64.0 78.4 50.8 28.9 47.4 66.4 89.2 66.9

- + GRPO 88.4 18.9 80.5 61.5 81.3 88.0 84.6 68.3 84.1 66.0 78.7 52.2 30.6 49.2 68.4 90.8 68.2

+ GRPO† 88.5 20.9 81.4 61.4 82.0 88.1 84.5 68.3 85.3 65.5 78.4 52.4 30.4 51.0 69.4 91.8 68.7

- + GRPO + Both 89.0 20.7 81.3 61.0 82.1 87.8 84.2 67.0 84.7 65.9 79.0 51.2 30.6 50.5 69.5 91.3 68.5

- + GRPO† + BCQ 89.5 24.3 81.8 61.9 81.9 88.2 85.1 69.7 85.2 65.9 79.3 53.9 31.8 52.3 72.3 91.0 69.6

- + GRPO† + NCQ 89.3 24.9 81.4 62.7 81.7 88.7 85.2 70.2 84.9 66.3 79.4 54.2 32.1 52.2 72.8 92.1 69.9

- + ZPPO 89.6 27.8 82.8 63.9 82.4 88.8 85.7 72.2 85.4 67.1 79.8 54.6 33.0 54.5 74.9 92.6 70.9 Qwen3.5-9B 90.2 22.9 82.9 62.5 82.1 87.8 86.9 70.8 86.3 67.7 83.0 50.0 31.7 55.3 71.7 90.9 70.2

+ GRPO 89.8 24.7 83.4 65.3 80.4 89.6 87.4 74.3 86.0 69.1 83.2 52.2 32.9 55.9 73.3 91.8 71.2 + GRPO† 90.2 24.0 83.2 65.1 81.4 89.6 87.9 75.0 87.3 68.9 83.0 52.2 32.9 55.7 74.3 92.2 71.4 + GRPO + Both 90.3 24.3 83.3 64.5 80.9 89.8 87.7 73.5 86.0 69.3 83.6 52.7 32.5 56.0 74.5 92.2 71.3 + GRPO† + BCQ 90.3 25.3 83.3 65.4 82.0 90.6 87.3 74.6 86.5 70.1 83.5 55.0 34.0 56.0 76.2 92.4 72.0 + GRPO† + NCQ 90.7 26.0 82.9 66.4 81.6 90.7 87.9 75.4 86.9 70.0 83.3 56.2 35.2 56.2 78.4 92.8 72.5

- + ZPPO 90.8 26.3 83.7 66.7 82.1 91.4 88.1 75.9 87.5 70.2 83.9 56.7 35.6 56.4 79.4 92.9 73.0

- Table 12 | Component ablation of ZPPO on the 16 VLM benchmarks at 4B and 9B. Each row adds one component (BCQ, NCQ, or † = prompt replay buffer) to the GRPO backbone; the highlighted last row of each block is the full ZPPO recipe. Distillation baselines are in Tab. 10; benchmark abbreviations: Appendix B.2.

### F. Additional empirical results

#### F.1. Buffer dynamics across student scales

This subsection complements Sec. 4.3 with two views of how the prompt replay buffer behaves over training at every student scale, all measured on ZPPO runs with the hyperparameters listed in Tab. 8 (Appendix D).

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
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Entry Rollout Accuracy

0% 12.5% 25% 37.5%

- Figure 20 | Replay-buffer composition by entry rollout accuracy at admission, over training, for ZPPO at every student scale. Each panel stacks the four admission bins so the total band height is the buffer occupancy.

- (1) The buffer saturates earlier at small scales. In Fig. 20 the buffer hits its capacity ceiling within ∼70 steps at 0.8B and within ∼100 steps at 2B; at 4B and 9B it only approaches the cap toward step 200. The reason is simple. Weak students fail more questions, so admission outruns graduation and the buffer fills almost immediately, whereas stronger students graduate at close to the admission rate and fill it more slowly. Either way, every scale ends training near capacity, which is the trend Fig. 5b already hinted at in the main paper.

Figure 21 | (left) Total number of graduations per scale, broken down by entry rollout accuracy at admission (0%, 12.5%, 25%, 37.5%). (right) Per-scale composition of graduates. Higher bars on the leftmost colors at larger scales mean “ZPPO graduates a larger absolute number of the hardest entries (admitted at 0%) at larger scales”, not that hardest entries become a smaller share of the buffer.

- (2) Even the hardest admission bin graduates, with the absolute count growing through 4B and the relative share rising monotonically up to 4B before saturating. Fig. 21 (left) shows that the count of graduations from the 0% admission bin (the bin where ordinary RL has zero gradient signal because every rollout fails) follows 293→432→562→514 across 0.8/2/4/9B, peaking

at 4B and dipping slightly at 9B as the saturated student needs fewer hard-question graduations to begin with. The 0% share of the per-scale graduate composition (right panel) tracks the same shape, climbing 13% → 16% → 21% and then holding flat at 21% from 4B to 9B. In short, BCQ and NCQ supply usable learning signal on exactly the questions where ordinary RL stalls, which is the mechanism the main paper invokes for Fig. 4.

#### F.2. ZPPO vs. GRPO† training dynamics at 2B

This subsection isolates the source of ZPPO’s advantage over GRPO† at 2B by tracking the optimization-side and buffer-side dynamics of the two runs side-by-side.

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

- Figure 22 | Training dynamics of Qwen3.5-2B + ZPPO vs. Qwen3.5-2B + GRPO†. Left: student rollout accuracy. Center: fraction of student rollouts whose group-relative advantage is positive. Right: actor Shannon entropy of the policy distribution.

- (1) ZPPO sustains a higher fraction of positive advantages. Fig. 22 (center) shows the fraction of student rollouts with positive group-relative advantage staying ∼3pp above GRPO† throughout training. The mechanism is direct: BCQ turns an all-wrong group (zero gradient signal) into a mixed group as soon as one BCQ rollout is correct, and NCQ exposes the diversity inside an all-wrong group once at least two distinct failure modes appear. ZPPO therefore earns positive advantages on a subset of groups that would give none under GRPO†.
- (2) The student stays exploratory for longer. Fig. 22 (right) shows actor entropy under ZPPO climbing to 1.2 nats by step 200 vs. GRPO†’s ∼0.95, and rising rather than collapsing. We attribute this to the buffer: its continual re-admission of hard questions keeps slowing the policy’s concentration onto the easy distribution that GRPO† converges toward.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

Figure 23 | Replay-buffer fate of completed entries (graduated vs. FIFO-evicted) at 2B for ZPPO and GRPO†. ZPPO almost doubles the graduation rate (26.6% vs. 14.3%) and proportionally cuts evictions, on the same buffer with the same admission/graduation thresholds.

- (3) ZPPO almost doubles the graduation rate at matched buffer policy. Fig. 23 compares the fate of every entry that has left the buffer by the end of training, at 2B and at matched buffer policy. ZPPO graduates 2,627 entries and evicts 7,251 (26.6% / 73.4%); GRPO† graduates 1,440 and evicts 8,647 (14.3% / 85.7%). The buffer policy (admission threshold, graduation threshold, capacity, FIFO eviction rule) is identical in both runs; the admitted cohorts themselves differ per

- run, since each run only admits questions on which its own plain rollouts fail. The comparison is therefore “under matched buffer policy, what fraction of the admitted cohort ultimately graduates,” not “on the same set of questions, who graduates more.” Read that way, the ∼1.9× graduation rate points squarely at the reformulation channel (BCQ/NCQ) as the thing that separates the two runs.
- Figure 24 | (left two panels) Replay graduations partitioned by entry rollout accuracy at admission, for GRPO† vs. ZPPO at 2B (counts and per-model composition). (right two panels) Per-bucket fate decomposition by number of re-sample attempts: ZPPO converts the “improved (non-grad)” (light green) bucket into “graduated” (dark green) much more aggressively than GRPO†.

- (4) ZPPO’s extra graduations concentrate in the hardest admission bins. Fig. 24 (left two panels) shows that under matched buffer policy, ZPPO produces 432 vs. 73 graduations from the 0% admission bin and 510 vs. 169 from the 12.5% bin (each run measured over its own admitted cohort, as in finding (3)). The right two panels further decompose, by number of re-sample attempts, where in the buffer entries end up: under ZPPO the “never improved” (gray) fraction shrinks monotonically with attempts and the “graduated” (dark green) fraction climbs to ∼56% by attempt 6+, while under GRPO† a stubborn gray ceiling persists. So BCQ and NCQ are turning visit time into graduations rather than into wasted re-rollouts.

F.3. Multi-metric dynamics across scales

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

- Figure 25 | ZPPO training dynamics across all four student scales. (top row) policy gradient loss, BCQ-prompt accuracy, NCQ-prompt accuracy. (bottom row) fraction of positive advantages, student rollout accuracy, actor Shannon entropy.

Fig. 25 aggregates six per-step metrics for the ZPPO runs across 0.8/2/4/9B student scales. Three observations link these dynamics back to the main-paper claims.

#### (1) BCQ accuracy rises with training at every scale. BCQ accuracy starts low at every scale

- Table 13 | Component ablation of ZPPO on the 10 LLM and 5 Video benchmarks; same row structure as Tab. 3. Distillation baselines are in Tab. 2 and Tab. 11; benchmark abbreviations: Appendix B.2.

10 LLM Benchmarks 5 Video Benchmarks

Method AIME25 AIME26 CEval GPQA-D HLE IMO-AB MMLU MMLU-Pro MMLU-Rd MultiCh Avg MMVU MVBench VMME VMMES VMMMU Avg Qwen3.5-0.8B 3.3 0.0 52.3 12.1 2.2 3.3 57.7 41.7 62.5 16.6 25.2 39.0 51.6 50.3 51.6 48.8 48.3 + GRPO 0.0 0.0 53.9 20.0 3.2 5.1 59.4 44.8 64.5 19.7 27.1 39.9 53.2 51.4 52.7 50.8 49.6 + GRPO† 3.3 0.0 54.5 25.5 3.5 7.0 59.9 47.1 65.1 20.8 28.7 41.3 53.5 52.2 53.0 52.5 50.5 + GRPO + Both 3.3 0.0 54.8 25.2 2.6 5.8 60.4 45.9 64.2 20.7 28.3 40.3 53.5 51.0 51.8 51.3 49.6 + GRPO† + BCQ 3.3 0.0 56.7 35.8 4.4 8.9 62.1 50.0 67.9 25.9 31.5 41.8 55.8 53.3 54.4 54.8 52.0 + GRPO† + NCQ 3.3 0.0 55.2 27.0 4.1 7.4 60.6 47.5 66.0 22.5 29.4 41.6 54.3 52.5 53.6 52.9 51.0 + ZPPO 3.3 0.0 57.6 42.4 5.0 10.5 63.0 51.7 69.0 28.6 33.1 42.5 56.6 53.9 54.9 56.4 52.8 Qwen3.5-2B 26.7 30.0 71.1 53.5 3.5 11.3 74.6 66.2 80.4 35.8 45.3 53.7 60.7 59.9 61.2 67.6 60.6 + GRPO 26.7 33.3 72.0 55.3 3.8 15.8 74.9 66.8 81.1 37.0 46.7 55.1 61.3 60.7 62.1 66.8 61.2 + GRPO† 26.7 33.3 72.4 56.5 4.2 19.3 75.4 67.3 81.4 37.3 47.3 56.7 61.7 61.2 62.6 67.3 61.9 + GRPO + Both 26.7 33.3 72.7 56.1 4.3 18.3 75.4 66.6 80.4 36.7 47.0 55.7 61.0 60.3 61.6 67.3 61.2 + GRPO† + BCQ 26.7 36.7 73.4 58.7 4.9 24.6 76.0 68.0 82.3 39.3 49.1 57.6 62.3 61.9 63.4 67.9 62.6 + GRPO† + NCQ 26.7 36.7 73.0 57.8 4.6 22.3 75.8 67.8 81.9 38.7 48.5 56.9 62.0 61.8 63.1 67.5 62.3 + ZPPO 26.7 40.0 74.0 60.6 5.1 29.5 76.4 68.2 82.9 40.6 50.4 59.0 62.7 62.4 63.9 68.1 63.2 Qwen3.5-4B 53.3 63.3 83.6 69.7 7.8 35.5 85.3 79.4 89.5 48.9 61.6 69.2 70.2 68.1 69.9 80.3 71.5 + GRPO 53.3 63.3 84.1 70.9 7.5 38.9 85.3 78.1 88.6 50.5 62.1 69.1 68.3 66.8 69.2 79.6 70.6 + GRPO† 53.3 63.3 84.6 73.5 7.1 40.6 85.9 79.2 88.9 50.7 62.7 69.2 69.1 68.2 68.7 79.4 70.9 + GRPO + Both 53.3 63.3 84.2 72.7 8.2 39.6 85.5 79.1 89.7 51.5 62.8 70.2 68.1 67.7 69.2 79.5 70.9 + GRPO† + BCQ 53.3 66.7 85.2 74.9 8.2 44.4 85.4 78.9 89.4 54.3 64.1 70.3 69.0 68.3 69.3 79.4 71.3 + GRPO† + NCQ 56.7 66.7 85.5 75.4 8.2 44.5 85.6 79.5 89.7 54.2 64.6 70.2 69.2 68.4 70.0 80.0 71.6 + ZPPO 56.7 66.7 86.1 77.3 8.3 48.0 86.0 79.7 90.1 56.0 65.5 70.6 69.6 68.5 70.2 80.2 71.8 Qwen3.5-9B 60.0 70.0 84.0 77.3 8.9 38.8 87.3 81.1 90.5 50.8 64.9 73.8 71.4 70.5 72.4 80.9 73.8 + GRPO 63.3 70.0 84.3 77.4 8.9 41.5 87.5 80.6 90.9 52.2 65.7 73.5 70.9 69.2 71.6 80.9 73.2 + GRPO† 63.3 70.0 85.3 77.8 8.8 43.6 87.4 81.3 90.9 53.0 66.2 73.1 71.5 69.3 71.2 82.0 73.4 + GRPO + Both 63.3 70.0 85.3 77.9 9.2 42.6 87.6 81.0 90.8 52.9 66.0 73.1 71.1 69.5 72.1 81.4 73.4 + GRPO† + BCQ 66.7 73.3 86.7 77.8 9.0 45.8 87.9 81.8 91.0 54.5 67.5 73.8 71.5 68.8 72.6 81.5 73.6 + GRPO† + NCQ 70.0 73.3 86.9 78.1 9.5 48.3 87.8 81.7 91.6 55.6 68.3 73.9 71.6 69.7 72.5 81.7 73.9 + ZPPO 70.0 73.3 87.2 78.8 9.8 49.8 88.1 82.0 91.7 56.8 68.8 74.2 71.7 70.2 72.8 82.1 74.2

and climbs to a scale-dependent convergence-regime value (Sec. F.7). That rise is exactly what we would expect if the student is learning to single out the correct anonymized candidate: candidate compression matches the length and format of the two options, so obvious surface tells (preferring the longer trace, say) are gone.

- (2) NCQ accuracy rises at 2/4/9B but is essentially flat at 0.8B. On NCQ, the student must reason about why a set of its own previous attempts were all wrong; the 0.8B model has limited capacity to discriminate among its own failure modes and therefore gains little from being shown them collectively. This is the underlying mechanism of the NCQ-at-0.8B exception in Tab. 3 (Sec. 4.4, takeaway (iii)).
- (3) Entropy diverges with scale. The 0.8B and 2B policies climb to ∼1.0/1.2 nats by step 200 – they keep expanding their candidate distribution – while the 4B and 9B policies plateau at lower entropy after a brief rise. This matches the intuition that strong students concentrate quickly on a near-optimal solution, while weak students keep gaining from the extended exploration that BCQ/NCQ opens up.

#### F.4. Component ablation on the LLM and Video benchmarks

- Tab. 13 mirrors the row recipe of Tab. 3 but evaluates on the 10 LLM and 5 Video benchmarks rather than on the 16 VLM benchmarks. Two observations carry over from the ablation. (1) Replay × reformulation is super-additive on generalization too. Pairing BCQ or NCQ with the prompt replay buffer compounds far beyond the sum of the isolated effects on both LLM and Video, reproducing the pattern – the buffer’s role of re-exposing each hard question to new BCQ/NCQ candidates is not a VLM-specific phenomenon. (2) The BCQ→NCQ shift with scale tracks on generalization. The marginal contribution of BCQ to ZPPO’s generalization gain shrinks with scale while NCQ’s grows, mirroring the ablation in Tab. 3. Concretely, at 0.8B the dominant

generalization gain over GRPO† comes from BCQ, while at 9B NCQ overtakes BCQ, and the full ZPPO recipe is the column-max on the macro-average of every (scale,benchmark family) block (the original Qwen3.5 base narrowly retains the per-column lead on a handful of Video benchmarks at 4B/9B – MVBench and VMMMU at 4B, and VMME at 9B in Tab. 13).

Beyond replicating the VLM ablation pattern, Tab. 13 reveals a scale-dependent LLM-vs-VLM effect for the headline buffer-isolated comparison ZPPO−GRPO† – the same comparison that Appendix F.6 bootstraps. From 2B onward this ZPPO−GRPO† macro-average gap on LLM exceeds that on VLM (+3.1/+2.8/+2.6 pp on LLM vs. +2.8/+2.2/+1.6 pp on VLM at 2/4/9B), so the generalization benefit of moving the teacher into the prompt grows with student capacity. At 0.8B both gaps are large and comparable (+4.4 pp on LLM vs. +4.9 pp on VLM), since a multimodally-trained 0.8B student has substantial headroom on LLM as well as VLM.

#### F.5. Hint and Prefix experimental setup and results

This subsection details the implementation of the Hint and Prefix baselines reported in Tab. 4 and discusses the numerical pattern that the inline table summarizes. Hint follows the Guide-GRPO [93] / SEELE [52] family: the teacher generates an answer-free hint from a correct teacher trace, the hint is appended to the prompt inside a tag, and the student samples a fully on-policy rollout from the hint-augmented prompt. Prefix follows BREAD [51] (with StepHint [50] as another member of the same family): a correct teacher rollout is truncated to a token-fraction prefix and forced as a shared anchor for a branched group of 𝐺S student continuations, with group-relative advantages computed over the branched group. For implementation simplicity we omit BREAD’s adaptive Episode Anchor Search (EAS) and fix the prefix fraction at 𝛼=0.4. Per-step algorithm annotations are in Appendix D.2 (Algorithms 5 and 6).

Setup. Both baselines share the prompt replay buffer (admission, graduation, and capacity identical to ZPPO), the same student initialization, the same training dataset, and the same shared training pipeline as ZPPO. Per-method compute is bounded by the GRPO†–ZPPO range reported in Tab. 9 (within the same training-budget convention), rather than listed as separate rows. Each baseline differs from ZPPO only in how it uses a teacher-correct response on hard questions:

- • Hint: a single correct teacher rollout 𝑦T(+) is converted by the teacher into an answer-free hint and appended to the prompt inside an explicit <hint>··· </hint> tag. The student then samples its rollout from this hint-augmented prompt; the rollout itself is fully student-generated and the gradient is on-policy at the response-token level (the prompt is augmented with teacher-derived hint text). (Sec. D.2, “Hint baseline” for the exact prompt template.)
- • Prefix: a single correct teacher rollout 𝑦T(+) is truncated to its first 𝛼=0.4 token-fraction and forced as a starting prefix to the student’s rollout; the student then samples its continuation autoregressively. The teacher-forced prefix tokens are masked from the policy-gradient summation following BREAD (E5 in Appendix D.2), so only the student-continuation tokens enter the gradient as targets. The remaining distinction from ZPPO is the content of the training signal: Prefix supplies only a positive continuation cue (continue a correct teacher trace), without a paired wrong candidate to discriminate against (as BCQ provides) or the student’s own collective wrong rollouts to avoid (as NCQ provides).

Neither baseline runs BCQ or NCQ. All other hyperparameters (Tab. 8) are identical to ZPPO.

Result pattern. Tab. 4 surfaces three things at 2B that the appendix lets us interpret precisely.

- (i) Hint helps a moderate amount on VLM but plateaus. GRPO†+Hint gains +1.2 pp on VLM over GRPO†, but is still 0.4 pp below GRPO†+BCQ and 1.6 pp below ZPPO. The student learns from teacher-side guidance, but the signal is one-sided: it never has to compare a correct teacher candidate against its own wrong candidate, and it never sees its own wrong rollouts collected

- as negatives. The gain therefore saturates below BCQ and the full BCQ+NCQ recipe.

- (ii) Hint barely transfers to LLM or Video. GRPO†+Hint comes in at 47.6/61.6 on LLM /Video, marginal versus GRPO†’s 47.3/61.9, because the hint acts as a directional shortcut that the student copies without internalizing why the direction is correct, and exercises neither the contrastive discrimination nor the collective-negative avoidance that BCQ and NCQ supply.
- (iii) Prefix is worse on generalization. GRPO†+Prefix loses −1.0 pp on LLM and −0.9 pp on Video relative to GRPO†, while gaining only +0.3 pp on VLM. With teacher-prefix tokens masked from the policy loss following BREAD (E5), the gradient consumes only student-generated continuation tokens, so the gap is not a gradient-side issue. The issue is what the training signal contains: Prefix supplies only a single positive continuation cue per hard question – a correct teacher trace the student copies and extends as a continuation shortcut – without any paired wrong candidate to discriminate against (as BCQ provides) or any aggregation of the student’s own collective wrong rollouts to avoid (as NCQ provides). The gain on LLM and Video benchmarks therefore lags BCQ/NCQ, since those held-out suites stress discrimination and self-correction behaviors rather than continuation from a known-good anchor.

It all comes down to the gap between the best prompt-injection baseline (Hint, 47.6/60.4/61.6) and full ZPPO (50.4/62.0/63.2): +2.8/ + 1.6/ + 1.6 pp. We read that gap as the empirical price of any prompt-side method that drops (i) the discriminative two-candidate construction of BCQ, where one teacher-correct trace and one student-wrong trace are anonymized and shuffled with no correctness label, and (ii) the collective-negative aggregation of NCQ, where every wrong student rollout on the same question is surfaced together. Hint gives only an answer-free guidance signal with no negative side, and Prefix forces a continuation from a single teacher trace and surfaces no negatives. Neither comes close to BCQ or NCQ on either axis.

#### F.6. Benchmark-selection robustness via cluster bootstrap

We complement the training runs with a benchmark-level cluster bootstrap covering every pairwise comparison reported in the main paper (Tabs. 1, 10, 2, 11, 3, 12, 13), not just the headline ZPPO−GRPO† delta. The procedure quantifies how robust each macro-level delta is to which exact benchmarks are included in the family. For each (scale,benchmark family) block we resample the benchmarks with replacement 10,000 times and re-compute the macro-average delta on each resample, then report the macro-average Δ, its 95% percentile CI, and the win-rate (number of benchmarks where ZPPO > baseline). The reported Δ is this benchmark-level macro-average (the mean of the per-benchmark deltas computed from unrounded scores), so it can differ by ≤0.1 pp from the Avg-row Δ printed in the main tables, which is the difference of the rounded column averages. Within each block we draw a single shared benchmark resample per iteration that is reused across all columns and fix the RNG seed, so two columns that encode the identical comparison receive identical intervals and the cross-column deltas stay coherent. The Best non-ZPPO baseline is the strongest non-ZPPO row selected on the full sample and then bootstrapped under the same shared resample (it may be the untrained Base, e.g. on 4B/9B Video). Tab. 14 reports CIs for the main-result comparisons; Tab. 15 reports CIs for the ablation comparisons.

Scope of the CIs. Because the resampling unit is the benchmark (not the training run or the stochastic decoding seed), these CIs quantify robustness to which exact benchmarks enter the macro-

average, not run-to-run training variance or per-decode sampling variance. Statements like “the CI excludes zero” below should be read as “the macro-average Δ is robust to dropping or duplicating any single benchmark in the family,” which is the appropriate stress test for benchmark-selection bias but is weaker than a statistical significance claim against training-stochasticity null.

Findings. (1) The buffer-isolated comparison ZPPO−GRPO† has a 95% CI that excludes zero in every one of the twelve (scale,benchmark family) cells, with win-rate ≥8/10 in every cell and ≥9/10 in every cell except 0.8B LLM – the pattern we would expect if the prompt-side recipe is what drives the headline gain once all other RL knobs are held fixed. (2) The two component-isolated comparisons

– ZPPO−GRPO†+BCQ (the NCQ marginal contribution) and ZPPO−GRPO†+NCQ (the BCQ marginal contribution) – also have CIs that exclude zero in every cell, so neither reformulation channel can be dropped without giving up an empirically detectable share of the gain at the macro level under benchmark resampling. (3) The ZPPO−Best-non-ZPPO comparison (the column-wise Δ rendered in the main and generalization tables, and the same per-cell quantity for the ablation tables even though those tables omit the Δ row) excludes zero everywhere except on Video at 4B and 9B in the main-result table, where the CIs are [−0.24,+0.90] and [−0.02,+0.86] pp respectively. The strongest column-wise baseline on those rows is the untrained base Qwen3.5 model itself, which has already saturated several Video benchmarks (ZPPO still wins on 3/5 and 4/5 Video benchmarks

- at 4B and 9B); ZPPO’s edge over that saturated base therefore collapses into the within-run noise on those rows, matching the few small negative entries in the Δ row of Tab. 11. Crucially, the buffer-isolated and component-isolated comparisons remain robust under benchmark resampling on these same cells, so the limitation is specifically against the saturated base Video performance rather than against any RL or distillation baseline.

Scale Family ZPPO − Off-Distill† ZPPO − On-Distill† ZPPO − GRPO† ZPPO − Best non-ZPPO

Δ [95% CI] (Win) Δ [95% CI] (Win) Δ [95% CI] (Win) Δ [95% CI] (Win) 0.8B VLM +8.78 [+6.87, +10.61] (16/16) +7.93 [+6.18, +9.63] (16/16) +4.79 [+3.45, +6.12] (16/16) +4.79 [+3.45, +6.12] (16/16) 0.8B LLM +10.56 [+5.87, +16.47] (9/10) +9.88 [+5.30, +15.87] (9/10) +4.44 [+2.02, +7.77] (8/10) +4.44 [+2.02, +7.77] (8/10) 0.8B Video +7.86 [+6.80, +9.44] (5/5) +7.04 [+5.62, +8.96] (5/5) +2.36 [+1.54, +3.30] (5/5) +2.36 [+1.54, +3.30] (5/5) 2B VLM +4.72 [+3.69, +5.83] (16/16) +4.02 [+3.03, +5.09] (16/16) +2.76 [+1.89, +3.74] (16/16) +2.76 [+1.89, +3.74] (16/16) 2B LLM +7.27 [+4.40, +10.85] (9/10) +6.63 [+4.24, +9.82] (10/10) +3.02 [+1.36, +5.19] (9/10) +3.02 [+1.36, +5.19] (9/10) 2B Video +4.62 [+3.34, +6.18] (5/5) +4.06 [+2.92, +5.42] (5/5) +1.32 [+0.96, +1.84] (5/5) +1.32 [+0.96, +1.84] (5/5)

4B VLM +3.61 [+2.60, +4.84] (16/16) +3.10 [+2.00, +4.45] (16/16) +2.24 [+1.41, +3.19] (16/16) +2.24 [+1.41, +3.19] (16/16) 4B LLM +4.79 [+2.86, +7.11] (10/10) +4.32 [+2.28, +6.74] (10/10) +2.78 [+1.52, +4.23] (10/10) +2.78 [+1.52, +4.23] (10/10) 4B Video +2.08 [+1.70, +2.44] (5/5) +1.48 [+1.16, +1.78] (5/5) +0.90 [+0.48, +1.32] (5/5) +0.28 [-0.24, +0.90] (3/5)

9B VLM +2.50 [+1.90, +3.18] (16/16) +2.01 [+1.39, +2.77] (16/16) +1.54 [+0.92, +2.29] (16/16) +1.54 [+0.92, +2.29] (16/16) 9B LLM +4.06 [+2.16, +6.25] (10/10) +3.63 [+1.65, +5.91] (10/10) +2.61 [+1.36, +3.99] (10/10) +2.61 [+1.36, +3.99] (10/10) 9B Video +1.62 [+1.38, +1.86] (5/5) +1.14 [+0.86, +1.32] (5/5) +0.78 [+0.30, +1.26] (5/5) +0.40 [-0.02, +0.86] (4/5)

- Table 14 | Cluster bootstrap 95% CI on the paired ZPPO−baseline macro-average delta (pp) for every main-result comparison (Tabs. 1, 10, 2, 11). Within each (scale, family) block we draw 10,000 benchmark resamples (with replacement); a single shared resample is reused across all columns in each iteration (fixed seed), so identical comparisons yield identical intervals. Intervals are 95% percentile CIs and the reported Δ is the full-sample macro-average. Win is the number of benchmarks (out of the block size) on which ZPPO > baseline. Best non-ZPPO is the strongest non-ZPPO row in the corresponding main table (selected on the full sample, so it equals the GRPO† column except on 4B/9B Video, where the untrained Base is the strongest non-ZPPO); it matches the Δ row of the main tables.

#### F.7. BCQ and NCQ candidate-match audit

This subsection audits two distinct concerns: (a) BCQ answer-in-prompt match, that the labeled positive candidate in the BCQ prompt lets the student read off the correct answer rather than discriminate; and (b) NCQ rewriter behavior and triviality, that the rewriter quietly turns the listed wrong-student rollouts in the NCQ prompt into teacher-style correct traces, or that the student

Scale Family ZPPO − GRPO† ZPPO − GRPO†+BCQ ZPPO − GRPO†+NCQ ZPPO − Best non-ZPPO

Δ [95% CI] (Win) Δ [95% CI] (Win) Δ [95% CI] (Win) Δ [95% CI] (Win) 0.8B VLM +4.79 [+3.45, +6.12] (16/16) +1.64 [+1.13, +2.16] (16/16) +4.05 [+2.84, +5.22] (16/16) +1.64 [+1.13, +2.16] (16/16) 0.8B LLM +4.44 [+2.02, +7.77] (8/10) +1.61 [+0.69, +2.92] (8/10) +3.75 [+1.60, +6.80] (8/10) +1.61 [+0.69, +2.92] (8/10) 0.8B Video +2.36 [+1.54, +3.30] (5/5) +0.84 [+0.58, +1.22] (5/5) +1.88 [+1.16, +2.78] (5/5) +0.84 [+0.58, +1.22] (5/5) 2B VLM +2.76 [+1.89, +3.74] (16/16) +1.21 [+0.79, +1.67] (16/16) +1.86 [+1.26, +2.52] (16/16) +1.21 [+0.79, +1.67] (16/16) 2B LLM +3.02 [+1.36, +5.19] (9/10) +1.34 [+0.52, +2.42] (9/10) +1.87 [+0.81, +3.35] (9/10) +1.34 [+0.52, +2.42] (9/10) 2B Video +1.32 [+0.96, +1.84] (5/5) +0.60 [+0.32, +1.02] (5/5) +0.96 [+0.62, +1.54] (5/5) +0.60 [+0.32, +1.02] (5/5) 4B VLM +2.24 [+1.41, +3.19] (16/16) +1.31 [+0.86, +1.80] (16/16) +1.06 [+0.69, +1.47] (16/16) +1.06 [+0.69, +1.47] (16/16) 4B LLM +2.78 [+1.52, +4.23] (10/10) +1.42 [+0.71, +2.22] (9/10) +0.89 [+0.29, +1.63] (8/10) +0.89 [+0.29, +1.63] (8/10) 4B Video +0.90 [+0.48, +1.32] (5/5) +0.56 [+0.32, +0.80] (5/5) +0.26 [+0.16, +0.36] (5/5) +0.26 [+0.16, +0.36] (5/5) 9B VLM +1.54 [+0.92, +2.29] (16/16) +0.94 [+0.62, +1.35] (16/16) +0.44 [+0.31, +0.56] (16/16) +0.44 [+0.31, +0.56] (16/16) 9B LLM +2.61 [+1.36, +3.99] (10/10) +1.30 [+0.54, +2.16] (9/10) +0.47 [+0.20, +0.79] (8/10) +0.47 [+0.20, +0.79] (8/10) 9B Video +0.78 [+0.30, +1.26] (5/5) +0.56 [+0.24, +1.00] (5/5) +0.32 [+0.20, +0.44] (5/5) +0.32 [+0.20, +0.44] (5/5)

- Table 15 | Cluster bootstrap 95% CI on the paired ZPPO−baseline macro-average delta (pp) for every ablation comparison (Tabs. 3, 12, 13). Within each (scale, family) block we draw 10,000 benchmark resamples (with replacement) using a single shared resample reused across all columns per iteration (fixed seed), so identical comparisons yield identical 95% percentile intervals. ZPPO − GRPO† is the full BCQ+NCQ contribution; ZPPO − GRPO†+BCQ isolates the marginal contribution of NCQ; ZPPO − GRPO†+NCQ isolates the marginal contribution of BCQ. Win is the number of benchmarks (out of the block size) on which ZPPO >

baseline. Best non-ZPPO reports ZPPO − maxother recipe at each (scale, family) block, selected on the full sample (GRPO†+BCQ at 0.8B/2B, GRPO†+NCQ at 4B/9B); the ablation tables themselves omit a Δ row.

trivially picks a non-listed answer without learning to discriminate against its own negatives. We rely on the per-step BCQ/NCQ debug logs each ZPPO run writes (bcq_debug/ and ncq_debug/); the three convergence-regime snapshots per scale (the last three BCQ/NCQ-instance dumps recorded near the end of training, one per scale per branch) cover 1,528 BCQ rollouts and 7,720 NCQ rollouts across the four student scales. All comparisons are on the final ∖boxed string, normalized by lowercasing, whitespace-collapsing, and stripping ∖text{...} and similar surface wrappers. Before the aggregate numbers, Figs. 26–30 walk five worked BCQ/NCQ pairs from the same logs across two outcome regimes (recovery, BCQ-only).

Setup of the worked examples. Figs. 26–30 walk five hard ZPPO-77K questions on which the student fails every plain rollout (¯𝑟𝑥=0, triggering both branches of Algorithm 1). For each question we show three things: (i) the original image, question, reference answer, and parsed plain-rollout answers; (ii) the BCQ-reformulated prompt, in which the teacher-correct trace and one student-wrong trace are teacher-compressed under a shared compression prompt and token cap, randomly shuffled, and presented as two anonymized options, plus one of the student’s 𝐺S=8 BCQ rollouts; (iii) the NCQ-reformulated prompt, in which all student-wrong rollouts are teacher-rewritten and listed with their parsed final answers and the student is told the answer is not one of the listed values, plus one of the student’s 𝐺S =8 NCQ rollouts. In both branches the gradient sees only the student’s own response tokens; the teacher-derived material lives entirely inside the prompt. Reasoning excerpts in the figures are shortened and lightly edited for length and readability (—omitted— marks removed reasoning); only the final ∖boxed answers are reproduced exactly. The five figures together illustrate the mechanism behind the 0%-accuracy graduation result in Sec. 4.3 (Fig. 4), and two of the regimes the audit numbers in Tabs. 16–17 summarize – the recovery regime and the BCQ-only regime (0.8B NCQ floor) – spanning all four student scales.

Recovery regime – both branches succeed (Figs. 26–29). Four of the five figures cover questions on which both BCQ and NCQ recover a correct rollout despite 0/8 plain accuracy. Fig. 26

(visual counting, 2B): the plain rollouts all overcount a banana bunch as 3; BCQ contrasts the uniform “each of three” trace against a per-position decomposition of which bananas are stickered, and NCQ tells the student that 3 is wrong, prompting it to name the over-counted position rather than to eliminate from a closed answer set. Fig. 27 (visual counting, 4B): plain rollouts undercount a four-spine bookshelf as three; BCQ shows the undercount against a richer colour-pattern (White, Dark, White, Dark) observation. Fig. 28 (scene QA, 2B): plain rollouts misidentify the salient attachment as either a pocket flap or a button; NCQ confronts the student with both wrong modes simultaneously. Fig. 29 (chart-reading, 9B): plain rollouts unanimously report slope 400 mph by snapping the unlabelled half-step at 𝑥=1 down to the labelled 400-row; BCQ contrasts that 400-row chain against one that anchors on two exact-intersection grid points and the line’s terminating arrowhead, and NCQ – where the listed wrong-rollout answers are open-ended numeric and so cannot be ruled out by elimination from a closed set – prompts the student to name the down-snapping failure mode and re-derive 500 from positional grounding. In every case in this regime the gradient sees a positive-advantage rollout on a question that would otherwise contribute zero gradient under ordinary RL.

BCQ-only regime – the 0.8B NCQ floor (Fig. 30). Fig. 30 is the cleanest qualitative instance of the scale-dependent NCQ floor that the aggregate numbers in Tab. 17 report (match-neg 82.7% at 0.8B). The 0.8B student’s plain rollouts overlook a cluster of straws on the right edge of a pizza-meal photo and unanimously answer 0. Under BCQ the student is presented with one compressed candidate that localizes “three straws on the right edge” and another that concludes “no straws visible”; it commits to the positionally-grounded candidate on falsifiability grounds and recovers 3. Under NCQ – where the teacher trace is removed and the student is shown eight copies of its own “no straws” answer with the instruction that 0 is wrong – the 0.8B student fails to override its own consensus and repeats

on all eight rollouts. This is the qualitative shape of the match-neg behaviour we report aggregate at the 0.8B scale: the smallest student lacks the capacity to argue against its own collective negatives without a teacher-positive anchor in the prompt, so the NCQ-only signal degenerates. BCQ continues to work because the teacher trace is still inside the BCQ prompt.

|0|
|---|

BCQ audit. For each BCQ rollout we check whether the student’s normalized ∖boxed string matches any positive candidate (teacher_pos_compressed), any negative candidate (student_neg_compressed), or neither. Tab. 16 breaks the result down by judge correctness. We use the table to compare two readings:

- • Under a trivial answer-in-prompt match pattern, where the candidate format exposes which one is the teacher-correct trace, the student should pick the positive candidate on essentially every rollout and BCQ accuracy should approach 100%.
- • Under genuine binary discrimination, where the anonymized, shuffled candidates carry no obvious surface tell after compression under the shared prompt and token cap, the student should match the positive candidate on the rollouts it gets right and match a negative candidate on the rollouts it gets wrong, and BCQ accuracy should sit well below 100%.

At every scale the audit favors the second reading over the first. The most direct evidence is that BCQ accuracy in the audited snapshots is 36–69%, not ∼100%, so the anonymization + shared-compression pipeline does not appear to leave a trivially separable surface signal that the student can read off. The within-subset breakdown tells the same story: Correct rollouts match the positive candidate 78–91% of the time and Wrong rollouts match the negative candidate 77–97% of the time, i.e. the student commits to one of the two candidates on essentially every rollout (the residual other mass is ≤22.6%, mostly different surface text). match-pos in the Wrong subset is

[Figure 139]

How many bananas have stickers on them? Reference answer: 2 (Student’s plain rollouts: all eight answer 3, counting every banana in the bunch.)

- (a) BCQ reformulated prompt. The teacher’s correct trace and one of the student’s wrong traces are teacher-compressed under a shared prompt and token cap, randomly shuffled, and inserted into a binary-discrimination template. The student then

rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

- Candidate A (anonymized; here, the student-derived wrong trace – not labelled as such to the student):

Visual inspection of the bananas in the bottom right corner reveals small rectangular stickers on their skins. Each of the three bananas in the bunch displays a distinct sticker, likely indicating weight or size. Close examination confirms the presence of these labels on the top, middle, and bottom bananas.

–-omitted–- ∖boxed{3}

- Candidate B (anonymized; here, the teacher-derived correct trace – not labelled as such to the student):

Visual inspection of the banana bunch identifies the front-facing bananas. A sticker is clearly visible on the top banana in the bunch. A second sticker is clearly visible on the bottom banana in the bunch. The middle banana and the side banana do not display any visible stickers. –-omitted–- ∖boxed{2}

Student’s BCQ rollout (one of 𝐺S =8; this one is judged correct): <think> The two candidates are anonymized, so I cannot tell which is teacher-correct from the label. Candidate A asserts a uniform “each of the three” without per-position evidence; Candidate B distinguishes the top and the bottom from the middle and the side. The per-position decomposition is concrete and falsifiable in a way a blanket “three” is not, so I commit to Candidate B on reasoning quality alone: top and bottom stickered, middle and side bare. –-omitted–- Answer: 2. </think> ∖boxed{2} ✓

- (b) NCQ reformulated prompt. All of the student’s wrong rollouts on this question are collected, teacher-compressed under the same shared prompt and token cap, and concatenated with their parsed final answers. The teacher’s correct trace is not

inserted. The student then rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

Listed wrong candidates (all eight of the student’s previous attempts on this question, each teacher-rewritten; only two are shown; remaining six follow the same pattern):

<candidate> The bunch contains three distinct bananas. Each banana displays a small rectangular adhesive label. –-omitted–- ∖boxed{3} </candidate>

<candidate> Three individual bananas are observed within the bunch. Each of these three bananas displays a small rectangular sticker. –-omitted–- ∖boxed{3} </candidate>

Parsed final answers from the listed candidates: [3, 3, 3, 3, 3, 3, 3, 3] Instruction to the student: the answer is not one of the listed values.

Student’s NCQ rollout (one of 𝐺S =8; this one is judged correct): <think> All eight listed candidates report 3 with the same generic “each banana has a sticker” justification; none localises which banana is stickered. The prompt rules 3 out, which means the consensus over-counts a specific position rather than the count being arbitrarily off. The position most consistent with being silently absorbed by such a generic sweep is the middle one, so correcting this specific failure mode – not eliminating from a closed answer set – yields 2. –-omitted–- Answer: 2. </think> ∖boxed{2} ✓

- Figure 26 | Visual counting, 2B student. Worked BCQ/NCQ example on a hard ZPPO-77K question that

the 2B student fails on every plain rollout (¯𝑟𝑥 =0). (a) BCQ: one teacher-correct and one student-wrong trace, teacher-rewritten to length parity, anonymized, and randomly shuffled. (b) NCQ: all of the student’s wrong rollouts listed as negatives, with the teacher trace excluded from the prompt. Both branches recover a correct rollout. Figs. 27–30 extend the same template to four other (scale, domain, outcome) cells.

Original question (a 4B student gets 0/8 plain rollouts correct; 𝑟¯𝑥 =0, triggering both the BCQ and the NCQ branch in Algorithm 1).

[Figure 140]

How many books are standing upright on the bookshelf next to the floor lamp on the right? Reference answer: 4 (Student’s plain rollouts: all eight answer 3, undercounting one spine in the alternating sequence.)

- (a) BCQ reformulated prompt. The teacher’s correct trace and one of the student’s wrong traces are teacher-compressed under a shared prompt and token cap, randomly shuffled, and inserted into a binary-discrimination template. The student then

rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

- Candidate A (anonymized; here, the student-derived wrong trace – not labelled as such to the student):

The bookshelf next to the floor lamp on the right contains a section with thin vertical items resembling book spines. Counting the distinct upright spines reveals three books standing upright. –-omitted–-

- ∖boxed{3}

Candidate B (anonymized; here, the teacher-derived correct trace – not labelled as such to the student):

The bookshelf is located to the right of the floor lamp. On the shelf below the one with white spheres, books stand upright. Counting the distinct vertical spines from left to right – White, Dark, White, Dark

– indicates four books. The other shelves contain decorative objects, not upright books. –-omitted–-

- ∖boxed{4}

Student’s BCQ rollout (one of 𝐺S =8; this one is judged correct): <think> Anonymized side by side, Candidate A reports three spines without further structure, while Candidate B commits to an explicit colour sequence (White, Dark, White, Dark). The four-element alternation is a structural commitment: if only three spines existed, the sequence would not balance. I commit to Candidate B because its reasoning survives this internal consistency check, not because of any label. –-omitted–- Answer: 4. </think> ∖boxed{4} ✓

- (b) NCQ reformulated prompt. All of the student’s wrong rollouts on this question are collected, teacher-compressed under the same shared prompt and token cap, and concatenated with their parsed final answers. The teacher’s correct trace is not

inserted. The student then rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

Listed wrong candidates (all eight of the student’s previous attempts on this question, each teacher-rewritten; only two are shown; remaining six follow the same pattern):

<candidate> The bookshelf is on the right, next to the floor lamp. On a middle shelf, a row of thin vertical items appears to be books. Counting the distinct spines reveals three items. –-omitted–-

- ∖boxed{3} </candidate>

<candidate> Visual analysis of the bookshelf next to the right floor lamp identifies a section of slim, vertical book spines. A careful count of distinct vertical edges yields a total of three upright volumes.

–-omitted–- ∖boxed{3} </candidate>

Parsed final answers from the listed candidates: [3, 3, 3, 3, 3, 3, 3, 3] Instruction to the student: the answer is not one of the listed values.

Student’s NCQ rollout (one of 𝐺S =8; this one is judged correct): <think> All eight listed candidates report 3 with the same “row of three” justification, and none mentions a colour or position pattern. The consensus failure mode is therefore collapsing two adjacent spines into a single one, consistent with the absence of any colour-by-colour decomposition in the listed traces. Correcting this off-by-one error yields 4 – arrived at by naming the failure mode, not by exclusion from a closed answer set. –-omitted–- Answer: 4. </think>

- ∖boxed{4} ✓

- Figure 27 | Visual counting, 4B student. Worked BCQ/NCQ example on a hard ZPPO-77K free-form

counting question that the 4B student fails on every plain rollout (¯𝑟𝑥 =0). (a) BCQ contrasts the undercounting trace against one that reads the alternating colour pattern; (b) NCQ excludes the teacher trace and tells the student that the consensus answer (3) is wrong. Both branches recover the correct count of 4.

[Figure 141]

What’s attached to the coat? A. belt B. button C. pocket D. rope Reference answer: D (i.e., rope) (Student’s plain rollouts: six answer C and two answer B, latching onto a pocket flap or a button rather than the braided cord.)

- (a) BCQ reformulated prompt. The teacher’s correct trace and one of the student’s wrong traces are teacher-compressed under a shared prompt and token cap, randomly shuffled, and inserted into a binary-discrimination template. The student then

rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

- Candidate A (anonymized; here, the student-derived wrong trace – not labelled as such to the student):

Visual inspection of the yellow coat reveals distinct small circular shapes at the center front area near the neckline. These round objects are identified as metallic buttons used for fastening the garment. The location and appearance confirm functional fasteners typical for this coat style. –-omitted–∖boxed{B}

- Candidate B (anonymized; here, the teacher-derived correct trace – not labelled as such to the student):

A distinct white object on the yellow coat is identified as a braided cord tied in a knot at the chest level. The feature rules out belts and standard pockets; the white braided cord is determined to be a rope, serving as a decorative fastener. –-omitted–- ∖boxed{D}

Student’s BCQ rollout (one of 𝐺S =8; this one is judged correct): <think> Anonymized, Candidate A reads small circular shapes as buttons – a generic feature with no count or location – while Candidate B names a singular feature with a specific texture (braiding), colour (white), and configuration (a knot at the chest). A specific, harder-to-fabricate commitment outranks a generic reading on reasoning quality alone. –-omitted–- Answer: D (rope). </think> ∖boxed{D} ✓

- (b) NCQ reformulated prompt. All of the student’s wrong rollouts on this question are collected, teacher-compressed under the same shared prompt and token cap, and concatenated with their parsed final answers. The teacher’s correct trace is not

inserted. The student then rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

Listed wrong candidates (all eight of the student’s previous attempts on this question, each teacher-rewritten; only two are shown; remaining six follow the same pattern):

<candidate> Small circular shapes at the center front are identified as metallic buttons used for fastening. The features match buttons rather than belts, pockets, or ropes. –-omitted–- ∖boxed{B} </candidate>

<candidate> A rectangular stitched shape on the front resembles a pocket flap; the pocket design is the distinct attachment. –-omitted–- ∖boxed{C} </candidate>

Parsed final answers from the listed candidates: [C, C, B, C, C, C, B, C] Instruction to the student: the answer is not one of the listed values.

Student’s NCQ rollout (one of 𝐺S =8; this one is judged correct): <think> The prompt rules out B (button) and C (pocket); A (belt) and D (rope) remain. The decision between the two is not by exclusion alone: a belt is a long horizontal strap running across the waist, whereas the listed wrong candidates’ own descriptions consistently localise the attachment at the chest with a knot-like configuration. The remaining option that matches this named configuration is rope.

–-omitted–- Answer: D (rope). </think> ∖boxed{D} ✓

- Figure 28 | Scene QA, 2B student. Worked BCQ/NCQ example on a hard ZPPO-77K question that the

2B student fails on every plain rollout (¯𝑟𝑥=0) by misidentifying the salient attachment (six rollouts pick the pocket flap, two pick buttons). (a) BCQ contrasts the misidentification trace against the braided-cord trace; (b) NCQ confronts the student with both of its own wrong answer modes (B and C); after ruling them out, the student matches the chest-knot configuration described in the listed traces to the remaining option D. Both branches recover the correct answer.

Original question (a 9B student gets 0/8 plain rollouts correct; 𝑟¯𝑥 =0, triggering both the BCQ and the NCQ branch in Algorithm 1).

[Figure 142]

The graph shows the relationship between the distance an airplane has covered (y-axis, in miles) and the amount of time it has been flying (x-axis, in hours). What is the speed of the airplane in miles per hour?

Reference answer: 500 (Student’s plain rollouts: all eight answer 400, misreading the line at 𝑥=1 as 𝑦=400 rather than 𝑦=500.)

- (a) BCQ reformulated prompt. The teacher’s correct trace and one of the student’s wrong traces are teacher-compressed under a shared prompt and token cap, randomly shuffled, and inserted into a binary-discrimination template. The student then

rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

- Candidate A (anonymized; here, the student-derived wrong trace – not labelled as such to the student):

The speed equals the slope of the distance-time line. Identifying clear grid points, the line passes through (0, 0) and (4, 1600). Dividing 1600 by 4 gives a constant speed of 400 mph, consistent with (1, 400).

–-omitted–- ∖boxed{400}

- Candidate B (anonymized; here, the teacher-derived correct trace – not labelled as such to the student):

The speed corresponds to the slope of the line on the distance-time graph. Clear points on the line are (0, 0) and (2, 1000). The speed is (1000 − 0)/(2 − 0) = 500 mph, consistent with (1, 500) and the endpoint near (4, 2000). –-omitted–- ∖boxed{500}

Student’s BCQ rollout (one of 𝐺S =8; this one is judged correct): <think> Both candidates assert specific grid readings, so the choice cannot be made by length or order.

- Candidate A’s endpoint (4, 1600) would leave the line two labelled rows below the top of the 𝑦-axis.
- Candidate B’s chain (2, 1000) and (4, 2000) lies on exact labelled intersections, with (4, 2000) coinciding with the arrowhead’s terminating position. Only Candidate B’s chain is consistent on falsifiability grounds. –-omitted–- Answer: 500. </think> ∖boxed{500} ✓

- (b) NCQ reformulated prompt. All of the student’s wrong rollouts on this question are collected, teacher-compressed under the same shared prompt and token cap, and concatenated with their parsed final answers. The teacher’s correct trace is not

inserted. The student then rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

Listed wrong candidates (all eight of the student’s previous attempts on this question, each teacher-rewritten; only two are shown; remaining six follow the same pattern):

<candidate> The speed is the slope of the line. Identifying points (0, 0) and (2, 800), the slope is 800/2 = 400 mph, also consistent with (4, 1600) and (1, 400). –-omitted–- ∖boxed{400} </candidate>

<candidate> The line is linear through the origin. At 1 hour the distance reads 400 miles, and at 4 hours the distance reads 1600 miles. Dividing gives a rate of 400 miles per hour. –-omitted–- ∖boxed{400} </candidate>

Parsed final answers from the listed candidates: [400, 400, 400, 400, 400, 400, 400, 400] Instruction to the student: the answer is not one of the listed values.

Student’s NCQ rollout (one of 𝐺S =8; this one is judged correct): <think> All listed candidates commit to slope 400 via (1, 400), (2, 800), (4, 1600). The answer is open-ended numeric, so elimination from a closed set is unavailable – I have to re-derive. Re-anchoring on two exact-intersection points – (2, 1000) and the arrowhead terminating near (4, 2000) – gives slope 500. The consensus failed by snapping the unlabelled 𝑦 = 500 at 𝑥 = 1 down to the labelled 400-row. –-omitted–Answer: 500. </think> ∖boxed{500} ✓

- Figure 29 | Chart-reading, 9B student. Worked BCQ/NCQ example on an open-ended numeric ZPPO-77K

question that the 9B student plain-fails (¯𝑟𝑥 = 0) by down-snapping the unlabelled 𝑦 = 500 at 𝑥 = 1 to the labelled 400-row. (a) BCQ: contrasts that 400-row chain against one anchored on exact-intersection grid points and the arrowhead’s terminating position. (b) NCQ: with the teacher trace excluded, the student re-derives 500 by positional grounding – elimination is unavailable on an open-ended numeric answer space.

Original question (a 0.8B student gets 0/8 plain rollouts correct; 𝑟¯𝑥 =0, triggering both the BCQ and the NCQ branch in Algorithm 1).

[Figure 143]

How many straws are in the photo? Reference answer: 3 (Student’s plain rollouts: all eight answer 0, concluding that the meal scene contains no drinking straws.)

- (a) BCQ reformulated prompt. The teacher’s correct trace and one of the student’s wrong traces are teacher-compressed under a shared prompt and token cap, randomly shuffled, and inserted into a binary-discrimination template. The student then

rolls out 𝐺S =8 times on the new prompt; one correct rollout is shown below.

- Candidate A (anonymized; here, the student-derived wrong trace – not labelled as such to the student):

The image depicts a meal scene with pizza, a fork, and a knife. A thorough visual inspection reveals no beverages or drinking straws present in the frame. The count of straws is zero. –-omitted–- ∖boxed{0}

- Candidate B (anonymized; here, the teacher-derived correct trace – not labelled as such to the student):

Scanning the image reveals a cluster of white cylindrical objects on the right edge, identified as a bundle of straws. Detailed inspection shows three distinct straws stacked vertically: one at the top, one in the middle, and one at the bottom. A brownish object in the top-left corner is a pen, not a straw.

- –-omitted–- ∖boxed{3}

Student’s BCQ rollout (one of 𝐺S =8; this one is judged correct): <think> Candidate A performs a global “no straws in the meal scene” sweep without naming a location, while Candidate B commits to a specific region (the right edge of the frame), a specific shape (cylindrical), a specific colour (white), and a specific count of three. A positionally-grounded specific commitment is harder to fabricate than a blanket negative, so I commit to Candidate B on falsifiability, not the label.

- –-omitted–- Answer: 3. </think> ∖boxed{3} ✓

- (b) NCQ reformulated prompt. All of the student’s wrong rollouts on this question are collected, teacher-compressed under the same shared prompt and token cap, and concatenated with their parsed final answers. The teacher’s correct trace is not

inserted. The student then rolls out 𝐺S =8 times on the new prompt; a wrong rollout is shown below (all 𝐺S =8 NCQ rollouts on this question are judged wrong).

Listed wrong candidates (all eight of the student’s previous attempts on this question, each teacher-rewritten; only two are shown; remaining six follow the same pattern):

<candidate> The image depicts a scene of people eating pizza with cutlery; a thorough visual inspection reveals no beverages or drinking straws present. The count is zero. –-omitted–- ∖boxed{0} </candidate>

<candidate> Straws are designed for drinking and are typically cylindrical, distinct from the utensils present. A careful scan reveals no instances of straws anywhere in the scene. –-omitted–- ∖boxed{0} </candidate>

Parsed final answers from the listed candidates: [0, 0, 0, 0, 0, 0, 0, 0] Instruction to the student: the answer is not one of the listed values.

Student’s NCQ rollout (one of 𝐺S =8; this one is judged wrong – representative of the entire 0/8 NCQ group): <think> All eight listed candidates conclude 0 with the same global “no straws in the scene” reasoning, and none describes a region or shape that the consensus might have missed. The prompt rules 0 out, but without a single localised cue in the listed traces I have no foothold for identifying where a missed cluster would be. –-omitted–- I revert to the consensus. Answer: 0. </think> ∖boxed{0} ×

- Figure 30 | Visual counting, 0.8B student – BCQ✓ / NCQ×. The 0.8B student plain-fails (¯𝑟𝑥 =0) by overlooking a cluster of straws on the right edge. (a) BCQ: presented with a positionally-grounded compressed alternative (right-edge cluster, cylindrical/white, count of three), the student commits to it because that localized claim is checkable, and recovers 3. (b) NCQ: with the teacher trace removed and shown 8 instances of its own “no straws” answer plus the instruction that 0 is wrong, the student fails to

override its own consensus and repeats

on every rollout – the qualitative shape of the 0.8B match-neg row in Tab. 17.

|0|
|---|

≤2.7% at every scale, but we note the limitation that a student that trivially aligned to the positive would by definition not appear in the Wrong subset, so this row is a weak rather than decisive signal on its own.

NCQ audit. The NCQ prompt is constructed differently from the BCQ prompt: only the rewritten wrong student rollouts (student_neg_compressed) appear in the candidate block, together with the listed parsed final answers; the rewritten teacher rollout (teacher_pos_compressed) is logged for diagnostic purposes but never inserted into the NCQ prompt. We verified this directly on every NCQ prompt in the snapshots: 8/8 student_neg_compressed traces appear verbatim in the corresponding prompt, 0/4 teacher_pos_compressed traces appear in any prompt. The audit table for NCQ (Tab. 17) therefore changes meaning: match-neg is the rate at which the student repeats one of its own listed wrong answers (the NCQ failure mode the prompt asks it to avoid), and match-pos is the rate at which the student’s boxed string matches the rewritten teacher’s boxed string even though the teacher trace was not shown to it (the student arriving at the correct answer independently). The reading of the table is now:

- • Under a trivial avoidance pattern (e.g. the rewriter compressing the listed negatives so coarsely that the student can avoid them by emitting any non-listed string), NCQ accuracy should be high but match-pos should be near zero.
- • Under a rewriter-correctness pattern (the rewriter turning student_neg_compressed into a teacher-style correct trace), the prompt would now contain the answer, match-neg should drop sharply across all scales, and NCQ accuracy should be high at every scale.

The numbers are not consistent with either pattern. NCQ accuracy is intermediate and scaledependent (1.2% at 0.8B, 22%/35%/31% at 2/4/9B). match-neg at 0.8B is 82.7% – i.e. when the smallest student is given its own collected wrong rollouts and told to avoid them, it repeats one of them as its final answer most of the time. Only as the student scales up does match-neg collapse (2.2% at 2B, 0.7%/0.2% at 4/9B), in lockstep with NCQ accuracy growing. This shape is what one would expect from a student that gradually learns to discriminate against its own previously listed wrong answers; it is harder to reconcile with a static avoidance pattern, where one would expect match-neg to be uniformly small from the first scale.

Caveats on the audit. Three caveats apply to both tables. First, the comparison uses exact normalized ∖boxed-string match, so a student rollout that arrives at the correct answer through different surface text (e.g. “C” versus the prose description spelled out by the rewritten teacher) is counted as other rather than as a match. Second, the snapshots cover the convergence regime rather than the entire training trajectory; convergence-regime numbers do not by themselves rule out a different match pattern earlier in training, although they are the regime in which any persistent match behaviour would have had the most time to be exploited. Third, on MCQ-style benchmarks with small answer spaces, NCQ’s collective negative listing can in principle help by elimination as well as by discrimination; the audit table cannot disentangle these two contributions, and the open-ended worked examples in Figs. 26, 27, 29, and 30 (free-form counting and chart-reading, all with open-ended numeric answer spaces) are where elimination is least likely to substitute for discrimination.

Scale Subset 𝑁 match-pos match-neg other BCQ acc

Correct 450 90.7% 3.3% 6.0%

0.8B

68.6% Wrong 206 0.0% 90.3% 9.7%

Correct 271 79.7% 0.0% 20.3%

2B

60.5% Wrong 177 0.0% 77.4% 22.6%

Correct 97 78.4% 0.0% 21.6%

4B

39.1% Wrong 151 0.0% 97.4% 2.6%

Correct 64 84.4% 0.0% 15.6%

9B

36.4% Wrong 112 2.7% 86.6% 10.7%

- Table 16 | BCQ candidate-match audit on the convergence-regime snapshots (three BCQ-instance dumps per scale recorded near the end of training, 1,528 rollouts in total). For each BCQ rollout we extract the final ∖boxed string and check whether the normalized string matches any positive candidate (teacher_pos_compressed), any negative candidate (student_neg_compressed), or neither (other; includes the ≤ 1% of rollouts with no parseable ∖boxed). Rows are split by judge correctness. The reading we emphasize in Sec. F.7 is that BCQ accuracy plateaus at 36–69% (not ∼100%), so the anonymized candidates – compressed under a shared prompt and token cap – do not appear to expose a trivially separable surface signal; the within-subset rows (Correct→match-pos 78–91%, Wrong→match-neg 77–97%) show the student committing to one of the two candidates rather than to a third option.

Scale 𝑁 match-neg (failure) match-pos other NCQ acc

0.8B 1,656 82.7% 0.4% 16.9% 1.2% 2B 1,872 2.2% 4.8% 92.9% 22.1% 4B 2,056 0.7% 3.7% 95.5% 35.2% 9B 2,136 0.2% 1.8% 98.0% 30.5%

- Table 17 | NCQ candidate-match audit on the convergence-regime snapshots (three NCQ-instance dumps per scale recorded near the end of training, 7,720 rollouts in total). match-neg is the fraction whose normalized ∖boxed answer matches one of the listed wrong-rollout candidates in the prompt (the failure mode that NCQ trains the student to avoid); match-pos is the fraction whose answer matches the rewritten teacher’s boxed answer (teacher_pos_compressed), which is not included in the NCQ prompt at all and therefore represents the student arriving at the correct boxed string independently. Other covers everything else (different surface text; correct or incorrect under the judge). The NCQ accuracy column is the judge-side accuracy on the same rollouts. The scale-dependent collapse of match-neg (82.7% at 0.8B → 0.2% at 9B) is consistent with the student learning to discriminate against its own listed wrong answers as scale grows; the audit cannot separate this from elimination on MCQ-style answer spaces (see Sec. F.7 caveats).

### G. Teacher capability on the same benchmarks

This appendix focuses solely on the 27B teacher used throughout the paper. We report its perbenchmark accuracy on the exact 10 LLM, 16 VLM, and 5 Video benchmarks used throughout the paper (Appendix B.2), evaluated with the same decoding settings, judges, and benchmark harness as the main-paper tables (Tab. 19–Tab. 21). The cross-teacher comparison (4B/9B/27B teachers in Fig. 7) is presented in the body (Sec. 4.6); this appendix does not expand it. The teacher-size sweep in Fig. 7 keeps the student initialization, the ZPPO-77K training data, the rollout-step budget, the candidate-compression prompt, and every hyperparameter in Tab. 8 identical to the 27B-teacher headline run; only the teacher checkpoint changes between the 4B/9B/27B variants.

#### G.1. ZPPO students vs. the 27B teacher

Tab. 18 pairs every base student in the main paper with its ZPPO-trained counterpart and lines them up against the 27B teacher; for each benchmark family we report Base, ZPPO, and the macro-average delta Δ=ZPPO−Base.

LLM Avg VLM Avg Video Avg Model Base ZPPO Δ Base ZPPO Δ Base ZPPO Δ

Qwen3.5-0.8B 25.2 33.1 +7.9 41.0 50.3 +9.3 48.3 52.8 +4.5 Qwen3.5-2B 45.3 50.4 +5.1 56.8 62.0 +5.2 60.6 63.2 +2.6 Qwen3.5-4B 61.6 65.5 +3.9 66.9 70.9 +4.0 71.5 71.8 +0.3 Qwen3.5-9B 64.9 68.8 +3.9 70.2 73.0 +2.8 73.8 74.2 +0.4

Qwen3.5-27B 71.8 75.5 77.4

Table 18 | ZPPO-trained students at four scales vs. the 27B teacher, on the shared 10 LLM, 16 VLM, 5 Video benchmark suite. Δ=ZPPO−Base. Two patterns: (i) Δ is largest at 0.8B and 2B, where the teacher–student gap is widest and BCQ has the most room to inject useful candidates; (ii) at every scale Δ is smaller on Video than on LLM/VLM, which fits the fact that VLM is the training data (Appendix B.1) while Video is reached only by transfer rather than by direct optimization.

#### G.2. Per-benchmark results: ZPPO students vs. the 27B teacher

The next three tables expand both the 27B row and the four ZPPO rows of Tab. 18 into perbenchmark numbers. All values are PASS-AT-1 accuracy under the shared evaluation harness of Appendix B.2, so the per-cell comparison between a ZPPO student and the 27B teacher is directly meaningful.

Model AIME25 AIME26 CEval GPQA-D HLE IMO-AB MMLU MMLU-Pro MMLU-Rd MultiCh Avg Qwen3.5-0.8B +ZPPO 3.3 0.0 57.6 42.4 5.0 10.5 63.0 51.7 69.0 28.6 33.1 Qwen3.5-2B +ZPPO 26.7 40.0 74.0 60.6 5.1 29.5 76.4 68.2 82.9 40.6 50.4 Qwen3.5-4B +ZPPO 56.7 66.7 86.1 77.3 8.3 48.0 86.0 79.7 90.1 56.0 65.5 Qwen3.5-9B +ZPPO 70.0 73.3 87.2 78.8 9.8 49.8 88.1 82.0 91.7 56.8 68.8 Qwen3.5-27B 70.0 76.7 91.2 83.3 16.0 51.3 91.6 86.3 94.3 57.5 71.8

- Table 19 | Per-benchmark accuracy on the 10 LLM benchmarks: ZPPO students at four scales vs. the 27B teacher used throughout the paper. The 9B ZPPO row approaches the 27B teacher on AIME25 (70.0 vs. 70.0) and on the MMLU family (within ∼4pp on MMLU, MMLU-Pro, MMLU-Rd) while never seeing teacher logits in the gradient.

Model AI2D BabyV CharXiv DynaM EmbSp InfoVQA MVerse MVision MVista MMMUPro MM-Vet OCREN OCRZH VisP VBlind WeMath Avg Qwen3.5-0.8B +ZPPO 76.5 13.9 63.9 31.1 71.5 75.3 59.3 29.2 73.2 37.6 59.9 42.5 18.7 35.0 44.7 71.7 50.3 Qwen3.5-2B +ZPPO 85.3 18.6 73.9 52.7 79.5 84.6 76.0 50.5 80.5 53.2 77.1 48.8 26.0 42.0 60.5 82.6 62.0 Qwen3.5-4B +ZPPO 89.6 27.8 82.8 63.9 82.4 88.8 85.7 72.2 85.4 67.1 79.8 54.6 33.0 54.5 74.9 92.6 70.9 Qwen3.5-9B +ZPPO 90.8 26.3 83.7 66.7 82.1 91.4 88.1 75.9 87.5 70.2 83.9 56.7 35.6 56.4 79.4 92.9 73.0 Qwen3.5-27B 92.1 30.4 86.0 70.5 84.2 91.5 89.1 80.0 88.2 74.0 84.9 55.7 33.5 61.9 92.1 93.8 75.5

- Table 20 | Per-benchmark accuracy on the 16 VLM benchmarks: ZPPO students at four scales vs. the 27B teacher. The 9B ZPPO row approaches the 27B teacher within ∼1pp on OCREN (56.7 vs. 55.7), InfoVQA, MVerse, MVista, MM-Vet, and WeMath; the largest remaining teacher–student gaps are on VBlind, VisP, and BabyV. Benchmark abbreviations: Appendix B.2.

Model MMVU MVBench VMME VMMES VMMMU Avg Qwen3.5-0.8B +ZPPO 42.5 56.6 53.9 54.9 56.4 52.8 Qwen3.5-2B +ZPPO 59.0 62.7 62.4 63.9 68.1 63.2 Qwen3.5-4B +ZPPO 70.6 69.6 68.5 70.2 80.2 71.8 Qwen3.5-9B +ZPPO 74.2 71.7 70.2 72.8 82.1 74.2 Qwen3.5-27B 75.7 74.6 76.1 76.0 84.7 77.4

- Table 21 | Per-benchmark accuracy on the 5 Video benchmarks: ZPPO students at four scales vs. the 27B teacher. The teacher–student gap is the narrowest of the three benchmark families (e.g. at 9B, every Video benchmark sits within a few pp of the 27B teacher, with the largest residual on VMME at 5.9pp), even though the student is trained as a VLM and Video is purely a generalization eval.

#### G.3. Discussion

Two observations on the 27B teacher link Tab. 18 and Tab. 19–Tab. 21 back to the per-benchmark structure of ZPPO’s gains.

- (1) The 27B teacher’s accuracy varies sharply across benchmarks. Even at 27B, the teacher is far from saturated on the hard math/reasoning slices: HLE (16.0), OCRZH (33.5), BabyV (30.4), and OCREN (55.7) all sit well below the macro-averages. BCQ candidates can therefore exist on hard benchmarks (HLE, BabyV, ...) because the teacher still occasionally solves what the student cannot, but the candidate pool is intrinsically thinner there than on saturated benchmarks (AI2D, MMLU-Rd, ...), and ZPPO’s gains correspondingly correlate with how often the teacher–student gap on a benchmark is non-trivial but non-empty.
- (2) A ZPPO-trained 9B student approaches the 27B teacher on several benchmarks. Per-benchmark, the 9B+ZPPO row of Tab. 19–Tab. 21 sits within ≤1.0 pp of the 27B teacher on AIME25 (70.0 vs. 70.0), OCREN (56.7 vs. 55.7), InfoVQA, MultiCh, MVerse, MVista, MM-Vet, and WeMath. We do not interpret these proximate values as the student strictly surpassing the teacher: the per-run noise band on single ZPPO runs is comparable to the residual gaps. The largest remaining gaps (HLE, VBlind, VMME) line up with benchmarks where the 27B teacher itself is least saturated – exactly the benchmarks on which BCQ has the least leverage because the teacher cannot supply many extra solved instances the student is missing.

Putting the two together, the BCQ-pool argument can be read straight off Tab. 18–Tab. 21. Under the 27B teacher we use throughout, ZPPO’s per-benchmark gains follow wherever the teacher–student capability gap is non-trivial but non-empty, and by 9B the student has closed in on the teacher across a good part of the suite.

