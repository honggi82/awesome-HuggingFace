# arXiv:2605.11609v1[cs.LG]12May2026

## Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information

Guobin Shen1 Xiang Cheng1 Chenxiao Zhao1 Lei Huang1 Jindong Li2 Dongcheng Zhao2 Xing Yu1∗

1Xiaohongshu Inc. 2Institute of Automation, Chinese Academy of Sciences

#### Abstract

On-policy self-distillation, where a student is pulled toward a copy of itself conditioned on privileged context (e.g., a verified solution or feedback), offers a promising direction for advancing reasoning capability without a stronger external teacher. Yet in math reasoning the gains are inconsistent, even when the same approach succeeds elsewhere. A pointwise mutual information analysis traces the failure to the privileged context itself: it inflates the teacher’s confidence on tokens already implied by the solution (structural connectives, verifiable claims) and deflates it on deliberation tokens (Wait, Let, Maybe) that drive multi-step search. We propose Anti-Self-Distillation (AntiSD), which ascends a divergence between student and teacher rather than descending it: this reverses the per-token sign and yields a naturally bounded advantage in one step. An entropy-triggered gate disables the term once the teacher entropy collapses, completing a drop-in replacement for default self-distillation. Across five models from 4B to 30B parameters on math reasoning benchmarks, AntiSD reaches the GRPO baseline’s accuracy in 2 to 10× fewer training steps and improves final accuracy by up to 11.5 points. AntiSD opens a path to scalable self-improvement, where a language model bootstraps its own reasoning through its training signal.

github.com/FloyedShen/AntiSD wandb.ai/brain-cog/AntiSD

#### 1 Introduction

Reinforcement learning has become a primary axis of post-training progress for reasoning tasks, with reinforcement learning from verifiable rewards (RLVR; 23; 32; 5; 9) emerging as the dominant paradigm. The reward signal in RLVR, however, is typically a sparse, trajectory-level scalar: a single bit per rollout that does not indicate which intermediate step was responsible, leaving credit assignment to individual reasoning steps as an open problem. To address this, two main directions have emerged: training a separate process reward model (PRM) to score intermediate steps [14; 26; 18], or applying on-policy distillation (OPD) to provide a token-level imitation signal from a stronger teacher [1; 4; 17]. Both, however, depend on an external model. Can the model itself supply this credit?

On-policy self-distillation answers this in the affirmative. It specializes OPD by taking the teacher to be the student itself, conditioned on privileged context: typically a verified solution and any feedback from the environment. The token-level signal is then produced by the model’s own forward pass under richer conditioning, requiring neither an external teacher nor a separate reward model. A series of recent works [36; 7; 31; 21] has developed this idea along several axes, connecting back to the older framework of learning under privileged information [25; 16].

On math reasoning, however, the picture is more mixed. Diagnostic studies report that on-policy selfdistillation can improve instruction-following, scientific QA, and tool-use tasks [7], while delivering

∗Correspondence to: yuanshan2@xiaohongshu.com, floyed_shen@outlook.com.

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

Problem Solve

Since ﬁts,

[Figure 4]

[Figure 5]

Solution Try :

Privileged context (teacher: factoring)

Default self-distillation

###### Anti-self-distillation:

parrots the answer

ﬁnds its own path

Qwen3-8B — HMMT 2025

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | |6.3× faster to 0.40<br><br>+1| | | |6| |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.5

###### pp

0.4

avg@32

0.3

0.2

0.1

0.0

0 50 100 150 200

Training step

GRPO SD AntiSD

- Figure 1: (a) An oracle-conditioned teacher biases the student toward a single root; reversing the signal preserves deliberation and recovers the rest. (b) Qwen3-8B on HMMT 2025: AntiSD reaches GRPO’s peak in ∼1/5 the steps and ends +15pp higher; default self-distillation underperforms GRPO.

only modest or inconsistent gains on more challenging mathematical problems [8]. We observe the same pattern across model families ranging from 4B to 30B parameters: on math reasoning benchmarks such as AIME 2024 and 2025, default self-distillation typically fails to outperform a strong GRPO baseline (Figure 1(b) shows one representative case; full sweep in Section 4.1).

To understand the cause, we inspect the per-token signal that default self-distillation produces (Figure 2). The pattern points to the privileged context itself: conditioning the teacher on a verified solution effectively turns it into an oracle, leaving it confident on tokens that follow once the answer is known, such as structural connectives and verifiable-claim words, and unsure on deliberation tokens like Wait, Let, and Maybe that the student emits when re-examining alternatives. Standard self-distillation pulls the student toward this oracle teacher, reinforcing tokens that track the known solution and weakening tokens that drive deliberation, as shown in Figure 1(a).

This motivates a simple fix: invert the gradient direction. We propose Anti-Self-Distillation (AntiSD), which ascends a divergence between student and teacher rather than descending it, reversing the per-token sign and yielding a naturally bounded advantage in one step. An entropy-triggered gate disables the term once the teacher’s per-token entropy collapses, completing a drop-in replacement for default self-distillation. Across five models from 4B to 30B parameters on math reasoning benchmarks, AntiSD reaches the GRPO baseline’s accuracy in 2 to 10× fewer training steps and improves final accuracy by up to 11.5 points.

Our contributions are summarized as follows:

- • We expose a structural shortcut bias in standard self-distillation, where the per-token signal rewards tokens the privileged context already implies and suppresses deliberation tokens, and ground this observation in a conditional pointwise mutual information identity (Section 3.1).
- • We propose Anti-Self-Distillation (AntiSD), which reverses the per-token signal by ascending Jensen-Shannon divergence between student and teacher; the JSD shape provides automatic bounding, leaving an entropy-triggered gate as the only practical stabilizer. AntiSD is a drop-in replacement for default self-distillation with no additional cost.
- • Across five 4B–30B models on math and coding tasks, AntiSD matches the GRPO baseline in 2 to 10× fewer steps and adds up to 11.5 points of final accuracy over both GRPO and default self-distillation.

#### 2 Preliminaries

Setup. We work with an autoregressive language model πθ that, given a problem x, samples a trajectory y = (y1,...,yT). RLVR provides a scalar verifiable reward R(x,y) scoring the final answer. Following GRPO [23], we sample a group of G rollouts per prompt and use the groupnormalized sequence-level advantage Aseqi := (Ri − µR)/σR as the policy-gradient signal for the i-th rollout, where µR and σR are the within-group mean and standard deviation.

On-policy self-distillation. On-policy self-distillation augments the GRPO objective with a pertoken signal derived from a self-teacher. Let c denote privileged context (a verified solution and any environment feedback) provided at training time but not at inference. The same network πθ

holds

boxed

1.0

Q: Sixteen chairs are arranged in a row. Eight people each select a chair in which to sit so that no person sits next to two other people. Let $N$ be the number of subsets of $16$ chairs that could be selected. Find the remainder when $N$ is divided by $1000$.

2.0

solve succeeds

[Figure 6]

Assign

combination

1.5

minimal

0.8

avoid

p=π(y∣x,y,c)tt<t

1.0

A: We are given :\n\n - 1 6 chairs in a row .\n - 8 people each

Given

logπ−logπts

0.5

0.6

require

select a chair to sit in .\n - The condition : ** ... two other people **

Understand

0.0

.\n\n Wait — this wording is ambiguous . Let 's parse it carefully .\n\n > " Eight people each select a chair in which to sit so that no person sits next to two other people ... $\n\n ---\n\n ### Final Answer :\n\n $$ \n \ boxed { 9 0 7 }\n $$

0.4

−0.5

Let

First

suppose

−1.0

Alternatively

0.2

−1.5

seems

look

Wait

maybe

−2.0

0.0

Another

0.0 0.2 0.4 0.6 0.8 1.0

ps = π(yt ∣ x, y<t)

- Figure 2: Per-token signal ut = tt − st on Qwen3-4B-IT-2507 at AIME-25. (a) Single-rollout trace. (b) (πS,πT) heatmap. Blue marks deliberation tokens (ut ≪ 0); red marks shortcut tokens (ut ≫ 0).

plays two roles: the student πS(· | x,y<t) := πθ(· | x,y<t) generates the rollout, while the teacher πT(· | x,y<t) := πθ(· | x,c,y<t) scores it under richer conditioning (we suppress c from the teacher’s left-hand-side conditioning as a notational shorthand, since c is fixed throughout each training step). With sg[·] denoting stop-gradient, the standard self-distillation loss is the per-token KL,

LSD(θ) = Ex, y∼π

S(·|x)

T

t=1

DKL πS(· | x,y<t) sg[πT(· | x,y<t)] , (1)

in addition to the GRPO objective. More generally, LSD is one member of a family of per-token f-divergences between student and teacher; the choice of f shapes the resulting per-token advantage and we revisit it in Section 3.2. The basic on-policy distillation formulation drops the GRPO term (Aseqi ≡ 0) and uses this per-token signal alone [1; 4; 36]; recent reasoning RL methods [7; 12; 27] instead combine it with the trajectory-level reward Aseqi through various forms (additive, multiplicative, or sample-level routing). We adopt the additive form:

Ai,t = Aseqi + λ · δt, (2)

where δt is the per-token contribution of −∇θLSD written in policy-gradient form (closed form in Section 3.1) and λ > 0 is a mixing weight.

- 3 Anti-Self-Distillation

Section 3.1 identifies the per-token signal δt from Equation (2) with conditional pointwise mutual information and shows, in conjunction with Figure 2, that it carries a structural shortcut bias. Section 3.2 responds with Anti-Self-Distillation (AntiSD), which ascends Jensen-Shannon divergence between student and teacher under a single entropy-triggered gate.

##### 3.1 Per-token reward as conditional PMI

We abbreviate st := log πS(yt | x,y<t), tt := log πT(yt | x,y<t), and ut := tt −st. To get a closed form for δt in Equation (2), differentiate the per-token KL summand Ev∼π

[log πS(v) − log πT(v)] in Equation (1) with respect to θ. The constant-coefficient term Ev[∇θ log πS(v)] vanishes by the score-function identity v ∇θπS(v) = 0, the teacher gradient is killed by the stop-gradient, and only a weighted score-function term survives (full proof in Appendix A, Lemma 1):

S

S(·|x,y<t) uv · ∇θ log πS(v | x,y<t) , (3)

∇θ DKL πS(· | x,y<t) πT(· | x,y<t) = −Ev∼π

with uv := log πT(v | x,y<t) − log πS(v | x,y<t). The combined advantage in Equation (2) therefore uses δt = +ut. Following standard policy-gradient practice for distillation, we treat the outer rollout expectation Ey∼π

as a sample-mean estimator with stop-gradient on the trajectory distribution; the trajectory-level REINFORCE term that would otherwise arise from differentiating πS(y | x) is dropped, since trajectory-level credit assignment is handled separately by the GRPO term Aseqi .

S

ut as conditional PMI. Under the self-distillation setup, πS and πT share parameters, so ut admits a closed-form interpretation:

πθ(yt | x,c,y<t) πθ(yt | x,y<t)

= PMI(yt ; c | x,y<t), (4)

ut = log

the conditional pointwise mutual information between the next token yt and the privileged context c. The sign of ut records whether c raises (ut > 0) or lowers (ut < 0) πθ(yt). The default per-token reward δt = +ut therefore rewards tokens whose probability is raised by c and penalizes those it lowers; Figure 2 makes this concrete on real data.

We compute ut on student rollouts from Qwen3-4B-IT-2507 at AIME-25, with c from our selfdistillation pipeline (Appendix C). The teacher reward splits tokens into two informative regimes.

Shortcut tokens (ut ≫ 0, deep red) – Given, Assign, succeeds, holds – are strongly rewarded once the answer is known. Deliberation tokens (ut ≪ 0, deep blue) – Wait, Let, Maybe, Alternatively – are strongly penalized, since c has committed to a solution and the teacher down-weights tokens that re-examine alternatives. Generic tokens along the diagonal and answer-template tokens near (πS,πT) ≈ (1,1) carry no signal. Figure 2(a) traces these regimes alternating along a single rollout, and (b) aggregates them into a (πS,πT) heatmap with two off-diagonal lobes of opposite ut sign.

Default self-distillation thus rewards shortcut tokens and penalizes deliberation tokens. This is consistent with a phenomenon repeatedly observed under on-policy self-distillation – responses shorten as training proceeds [7; 8; 21] – but recasts it as a structural shortcut rather than benign compression, with the suppression concentrated on the deliberation steps that drive search rather than on redundant filler. The polarity is not specific to reverse KL: for any convex f in the family from Section 2, descent on Df(πS∥πT) has per-token advantage monotonically increasing in ut and inherits the same shortcut/deliberation split.

Two empirical observations from this analysis will drive the method. (O1) Wrong polarity for reasoning: the per-token reward δt = +ut has the wrong sign – rewarding shortcut tokens and penalizing the deliberation tokens that drive search. (O2) Asymmetric distribution: because rollouts come from πS, tokens with πS > πT are over-sampled in the batch – visible in Figure 2(b) as the heavier deliberation lobe (ut < 0), with individual tokens in the tail reaching ut ≤ −20 (Figure 2(a)).

##### 3.2 Ascent on Jensen-Shannon divergence

AntiSD has three components. From (O1), we reverse the gradient direction (descent → ascent), flipping the per-token reward at the source. From (O2), we ascend Jensen-Shannon divergence rather than reverse KL: JSD’s f-divergence-derived advantage is asymmetrically bounded (capped on the over-sampled deliberation side and linear on the under-sampled shortcut side), directly counterbalancing the empirical asymmetry. The third component, an entropy-triggered gate, follows from the first two: once we ascend a divergence, the policy gradient is no longer self-terminating, so we need a signal-quality criterion to disable the term once πT’s information about πS degenerates. We make each concrete below.

JSD ascent. Writing DJSD(πS∥πT) = Eπ

[f(πS/πT)] for the corresponding f-divergence generator, the score-function trick (analogous to Equation (3)) gives

T

f′ π

S(v)

πT(v) ∇θ log πS(v) . (5)

∇θDJSD(πS∥πT) = Ev∼π

S

Substituting πS/πT = e−u identifies f′(πS/πT) = −φ(u) (full simplification in Appendix A), so ascending JSD via policy gradient has per-token advantage

AAntiSDt = −φ(ut), φ(u) := 21(softplus(u) − log 2). (6) The shape φ is the f-divergence derivative for DJSD, so its monotonicity and sign-preservation follow from JSD’s convexity. At small |u|, φ′(0) = 12σ(0) = 14 gives −φ(u) ≈ −14u, which recovers ascent on reverse KL up to a positive scalar; the two choices diverge in the tails, where φ(u) ≥ −12 log 2 globally (proof in Appendix A) caps the AntiSD advantage on the deliberation side at 12 log 2. This is exactly the side that (O2) flagged as both over-sampled and heavy-tailed: the cap absorbs the ut ≤ −20 spikes and rebalances per-token gradient contributions against the lighter, under-sampled shortcut side, while the shortcut side keeps its linear penalty since extreme shortcut tokens are precisely the ones AntiSD should suppress proportionally. We ablate the divergence choice in Section 4.3.

Entropy-triggered gate. The JSD ascent direction is not self-terminating, so we need a criterion to disable the term once ut stops carrying useful conditional information. The teacher’s per-token entropy aggregated over the batch, H := mediani,t H[πT(· | xi,yi,<t)], provides this signal. The log-ratio ut = log(πT/πS) is well-conditioned only as long as πT retains substantial entropy: when

πT collapses to a near-deterministic mode (low H), most tokens lie at floor probability under πT and uv becomes dominated by numerical floor rather than conditional information. We disable the AntiSD term when H falls below an auto-calibrated threshold τdown, and re-enable it once H recovers to its pre-collapse baseline Hwarm (a Schmitt trigger to avoid chatter):

 

1 if g = 0 and H ≥ Hwarm, 0 if g = 1 and H < τdown, g otherwise,

λ = g · λmax. (7)

g ←



τdown is auto-calibrated from W warmup steps at λ = 0 (concrete values in Section 4 Setup). Algorithm 1 (Appendix B) summarizes the resulting update.

#### 4 Experiments

Setup. We train five language models from the Qwen3 [29] and Olmo-3 [20] families (4B–30B parameters) on DAPO-Math-17k [32] for 200 on-policy steps, comparing four conditions per model: the un-trained base, +GRPO (Equation (2) with λ = 0), +SD (default self-distillation, δt = +ut), and +AntiSD (Algorithm 1). The privileged context c is a verified solution sampled from the rollout group when at least one rollout is correct, else from the dataset, concatenated with a binary correctness feedback string. AntiSD’s gate is auto-calibrated from the first 5 training steps (run at λ = 0): we record the median teacher entropy Hwarm and set τdown = 0.93Hwarm, with the gate re-enabling once H recovers to Hwarm. The 0.93 multiplier is shared across all model families, requiring no per-model tuning. Held-out evaluation reports avg@32 on AIME 2024 [33] / 2025 [34] / 2026 [35] and HMMT 2025 [3], and avg@4 on MinervaMath [11]. Full model list, sampling settings, gate-calibration details, and example teacher prompts are in Appendix B and C.

##### 4.1 Main results

Table 1: Main results (accuracy %). AIME24/25/26 and HMMT25: avg@32; Minerva: avg@4. Subscript on Avg = peak-mean step; Speedup = GRPO’s best-Avg step / this row’s first-reach step (×: never reached). Bold = column best within each model block; highlighted row is canonical AntiSD.

Method AIME24 AIME25 AIME26 HMMT25 Minerva Average Speedup Qwen3-8B 25.5 21.3 17.0 10.8 39.0 22.7 –

+GRPO 73.5 65.2 64.2 39.2 45.1 57.4@200 1.0×

+SD 40.1 30.5 26.9 14.9 40.7 30.6@200 ×

+AntiSD 78.4 73.4 73.7 54.4 48.5 65.7@180 5.0× Qwen3-4B-IT-2507 62.1 48.2 53.8 30.4 43.4 47.6 –

+GRPO 67.8 57.7 63.5 34.1 33.2 51.3@200 1.0×

+SD 59.8 45.8 52.0 28.8 43.0 45.9@10 ×

+AntiSD 76.6 70.2 74.4 46.7 46.4 62.8@100 10.0× Olmo3-7B-IT 53.3 39.7 44.4 25.9 38.7 40.4 –

+GRPO 57.0 45.3 52.1 31.2 29.1 43.0@190 1.0×

+SD 54.5 41.8 46.6 24.4 38.5 41.1@10 ×

+AntiSD 62.4 49.1 55.2 32.3 42.4 48.3@200 9.5× Olmo3-7B-TK 74.6 68.7 73.3 48.2 45.2 62.0 –

+GRPO 76.5 71.7 75.3 50.5 46.4 64.1@80 1.0×

+SD 77.2 68.5 74.0 48.3 44.8 62.6@40 ×

+AntiSD 77.6 75.3 76.1 56.2 45.8 66.2@40 2.0× Qwen3-30B-A3B 28.1 22.3 21.8 11.7 43.5 25.5 –

+GRPO 74.1 66.8 65.5 42.2 47.1 59.1@150 1.0×

+SD 40.9 32.5 34.0 20.9 44.1 34.5@60 ×

+AntiSD 79.4 75.6 74.1 55.2 49.7 66.8@140 2.9×

Table 1 reports avg@32 at each (model, method)’s best-Avg checkpoint. Three patterns hold: (i) AntiSD reaches GRPO’s accuracy in a fraction of the steps, with a speedup of 2–10× across all five models. The largest speedups appear on the smaller models with weaker GRPO baselines (Qwen34B-IT-2507 10×, Olmo3-7B-IT 9.5×, Qwen3-8B 5×); the speedup shrinks but stays positive on the two strongest baselines (Olmo3-7B-TK 2×, where GRPO already sits at 64.1; Qwen3-30B-A3B 2.9×, the 30B mixture-of-experts model). This early ignition is consistent with the diagnosis in

Section 3.1: the per-token reward −φ(ut) is informative from the first step, so credit-assignment does not have to wait for sparse trajectory-level reward to propagate through the policy. (ii) AntiSD’s final mean accuracy exceeds GRPO’s on every model, by +2.1 to +11.5 points (Avg). The gap is widest on the weaker baselines (+5.3 to +11.5 on Qwen3-8B, Qwen3-4B-IT-2507, Olmo3-7B-IT), still substantial at scale (+7.7 on Qwen3-30B-A3B), and narrowest on the strongest GRPO baseline Olmo3-7B-TK (+2.1), where GRPO at 64.1 and the un-trained base at 62.0 leave little headroom on DAPO-Math-17k. Per-benchmark, 4 of 5 models win on every individual benchmark; the lone near-tie is a −0.6pp gap on MinervaMath for Olmo3-7B-TK, ruling out the explanation that one easy benchmark inflates the mean. The gain matches our prediction that biasing optimization toward deliberation tokens unlocks problems that GRPO’s sparse signal cannot reach. (iii) Default selfdistillation underperforms the GRPO baseline on every model, often by a wide margin (Qwen3-8B Avg: 30.6 vs 57.4). The mechanism behind this collapse, and the entropy dynamics that distinguish it from AntiSD, are examined in Section 4.2.

Qwen3-8B

Qwen3-4B-IT-0527

Olmo3-7B-IT

Olmo3-7B-TK

Qwen3-30B-A3B

1.0

| |GRPO| | | | | | |
|---|---|---|---|---|---|---|---|
| |SD<br><br>AntiSD| | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

kHMMT25pass@

0.8

0.6

0.4

0.2

0.0

1 2 4 8 16 32

1 2 4 8 16 32

1 2 4 8 16 32

1 2 4 8 16 32

1 2 4 8 16 32

k (in pass@k)

k (in pass@k)

k (in pass@k)

k (in pass@k)

k (in pass@k)

- Figure 3: HMMT25 pass@k at each row’s peak-mean snapshot. AntiSD’s lead over GRPO is sustained across k – the gain reflects expanded coverage, not just variance reduction.

A natural concern is whether AntiSD’s gain comes from better single-rollout accuracy or from concentrating probability mass on already-correct rollouts at the cost of generation diversity. Figure 3 plots pass@k on HMMT 2025 (the hardest of the five benchmarks) to disentangle these. AntiSD’s lead over GRPO is sustained across k: on Qwen3-8B the gap is ∼ 13 points at k = 1 and remains ∼ 7–10 points at k = 32. The non-converging curves at high k indicate that AntiSD genuinely solves problems that GRPO cannot reach even with 32 attempts and preserves the rollout diversity needed to do so, rather than trading diversity for single-rollout consistency.

Table 2: Code reasoning on Qwen3-8B (avg@10). Bold marks the best per column.

Method HumanEval+ MBPP+

+GRPO 40.4 61.0 +AntiSD 41.6 63.3

Code reasoning. To probe whether AntiSD generalises beyond math, we run the same on-policy self-distillation setup on the Dolci-RLZero code RL dataset [20] and evaluate on HumanEval+ and MBPP+ [15] (Table 2). On Qwen3-8B, AntiSD improves over the GRPO baseline by +1.2 points on HumanEval+ and +2.3 on MBPP+; the gains are smaller than on math reasoning but consistent in direction, indicating that the per-token mechanism transfers to a setting where the trajectory-level reward is itself denser.

4.2 Training dynamics

- Figure 4 traces six training-time signals through the run. AntiSD ignites earliest: truncation-corrected train reward climbs from ∼ 0.5 to ∼ 0.95 within ∼ 30 steps on Qwen3-8B and Qwen3-4B-IT-2507, a regime GRPO reaches only after ∼ 150 steps and SD never reaches, with HMMT25 and AIME25 moving in lockstep. The Qwen3-4B-IT-2507 plateau sits near 0.95 rather than 1.0 and drifts slightly late in training; held-out accuracy does not drop, so this is saturation against the DAPO-Math problem distribution – once almost every sampled problem is solved, the surviving gradient signal is noise – rather than overfitting.

Default self-distillation diverges in opposite directions across model families. Both AntiSD and SD couple the student and teacher distributions, but their entropy traces tell different stories: AntiSD remains in a stable middle band on all three models, while SD’s teacher and actor entropy collapse toward ∼ 0.1 nats per token on Qwen3-4B-IT-2507 (over-confident on the shortcut answer template) and inflate past 1 nat per token on Olmo3-7B-IT (drift away from useful tokens). This is exactly the bidirectional failure mode that the sign reversal in Section 3.2 addresses; the same shortcut bias that explains SD’s gap to GRPO in Table 1 is what is visibly amplifying or eroding teacher entropy

GRPO SD AntiSD

HMMT25 avg@32

AIME26 avg@32

Train reward

Teacher entropy

Actor entropy

Response length

1.0

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.6

0.5

0.75

Qwen3-8B

10000

avg@32

avg@32

nats/tok

nats/tok

reward

tokens

0.4

0.50

0.3

0.5

5000

0.2

0.25

0.3

0.2

0.0

0.00

0.0

Qwen3-4B-IT-0527

0.7

1.0

0.7

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

0.6

0.75

15000

0.5

0.5

avg@32

avg@32

nats/tok

nats/tok

reward

tokens

0.3

0.4

0.50

0.3

10000

0.5

0.2

0.2

0.2

0.25

5000

0.1

0.1

0.0

0.00

0.0

- 1

0.5

0.7

- 2
- 3

1.0

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0.6

8000

0.75

Olmo3-7B-IT

1

avg@32

avg@32

nats/tok

nats/tok

reward

tokens

0.4

6000

0.50

0.7

0.5

0.5

0.2

4000

0.25

0.3

0.0

0.00

0.0

0 100 200

0 100 200

0 100 200

0 100 200

0 100 200

0 100 200

training step

training step

training step

training step

training step

training step

- Figure 4: Training dynamics on three models (rows) along six axes (columns). Faded traces are raw values; bold traces are 20-step rolling means.

here. Its sharpest expression is the Qwen3-4B-IT-2507 collapse around step 80: train reward to zero, response length pinned at the 32K cap, and both entropies spiking, all within a single step window before the run terminates.

4.3 Ablations

AntiSD adds three components on top of the GRPO advantage: sign-reversed reward −φ(ut), the JSD/softplus shape, and an entropy-triggered gate. Sign reversal is the dominant lever and was already established in Table 1: removing it (default SD) drops Qwen3-8B Avg from 65.7 to 30.6. We focus the remaining ablations on the other two components and on the privileged context itself, reporting both training-curve health (does the run survive?) and held-out accuracy on Qwen3-4B-IT-2507, the most failure-prone model in our suite.

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.00

0.25

0.50

0.75

1.00

reward(completed)

Qwen3-8B

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.00

0.25

0.50

0.75

1.00

Qwen3-4B-IT-0527

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.00

0.25

0.50

0.75

1.00

Olmo3-7B-IT

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0 25 50 75 100 125 150 175 200

training step

6

12

length(ktokens)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0 25 50 75 100 125 150 175 200

training step

6

12

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0 25 50 75 100 125 150 175 200

training step

4

8

12

AntiSD GRPO SD No-teacher No-gate

- Figure 5: Mechanism ablations: failure modes. Top: reward over non-truncated rollouts. Bottom: mean response length. Line truncation indicates run termination after collapse.

No-teacher: self-reinforcement collapse. Removing the teacher entirely – so the per-token signal becomes a function of the student’s log-probability alone, with no teacher–student differential – collapses on all three models within ∼ 70 training steps (Figure 5, orange). Without external information from the privileged context, the per-token term degenerates into a function of the student’s own probability, producing a positive-feedback signal that reinforces whatever the policy already emits; this is a textbook self-reinforcement collapse and is the strongest evidence in our suite that AntiSD’s gain depends on the privileged information identity from Section 3.1, not on a generic shaping of student log-probabilities. This contrasts with recent self-reward methods [37; 6], which keep an external signal (typically majority-vote agreement across rollouts) rather than removing all conditioning. Our No-teacher variant strips the privileged context entirely, leaving only πS in the loss, and that is precisely the configuration that fails to learn. AntiSD’s privileged-context conditioning

instead preserves rollout diversity (Figure 3: sustained pass@k lead over GRPO out to k = 32), ruling out the mode-collapse-onto-majority failure that self-reward methods often incur.

No-gate: stabilization is model-conditional. Removing the entropy gate (always-on PRM) is the most striking model-dependent failure (Figure 5, brown). On Qwen3-8B and Qwen3-4B-IT-2507 the No-gate run actually ignites faster than canonical AntiSD – reward peaks ∼ 0.97 around step 40 – before collapsing near step 90 as the teacher’s per-token entropy crosses zero, at which point the PRM signal degenerates because the teacher has absorbed the answer template. On Olmo3-7B-IT the same configuration survives the full 200 steps and even attains the highest plateau on this model. The asymmetry tracks initial teacher entropy: Qwen models start at ≈ 0.4 nats per token (close to the absorption threshold) while Olmo starts higher, leaving headroom that the gate would have monitored. Read together with Section 3.2, the gate acts as a cross-model insurance policy rather than a per-model necessity; it is essential for Qwen and inert for Olmo.

- Table 3: Component sensitivity on Qwen3-4B-IT-2507. Single-knob deviations from canonical AntiSD (highlighted). Speedup as in Table 1. Bold = column best among +AntiSD rows.

Method Div τdown Compose AIME24 AIME25 AIME26 HMMT25 Minerva Average Speedup GRPO – – – 67.8 57.7 63.5 34.1 33.2 51.3@200 1.0×

+AntiSD

rev. KL 0.93 add. 64.1 49.0 58.9 32.0 43.7 49.5@10 ×

JSD none add. 72.4 66.5 72.6 44.9 46.9 60.6@30 10.0× JSD 0.90 add. 68.8 56.5 63.1 39.6 44.6 54.5@20 10.0×

JSD 0.95 add. 77.6 69.4 63.4 44.7 46.2 60.3@110 6.7× JSD 0.93 mult. 70.8 61.5 67.6 36.6 46.1 56.5@170 5.0×

JSD 0.93 add.∗ 74.8 65.3 70.1 39.4 51.8 60.3@20∗ 10.0× JSD 0.93 add. 76.6 70.2 74.4 46.7 46.4 62.8@100 10.0×

∗ Gate signal swapped from teacher- to student-perplexity.

Component sensitivity. Table 3 factorises the remaining design choices on Qwen3-4B-IT-2507 along three knobs: Div (JSD vs reverse-KL ascent), the gate deactivation threshold τdown, and Compose (additive vs multiplicative shaping of the GRPO advantage). Threshold sensitivity is model-conditional. Loosening τdown from 0.93 to 0.90 drops Q4 Avg by 8.3 points (62.8 → 54.5); on Qwen3-8B the same loosening leaves Avg essentially unchanged (65.9 vs canonical 65.7; Appendix D.1). The canonical 0.93 is not a per-model sweet spot but the value that transfers across the models we evaluate. Additive composition outperforms multiplicative. Replacing additive with multiplicative drops Avg by 6.3 points (62.8 → 56.5) and halves the speedup (10× → 5×): when Aseq is small, the multiplicative form scales the AntiSD term down toward zero, removing the deliberation push exactly when GRPO’s ORM signal is uninformative. The gate-off row’s 60.6 peak at step 30 is a transient pre-collapse value (Figure 5, brown), not a sustainable plateau.

4.4 Beyond GRPO saturation

A practical question is whether AntiSD must be trained from the base model, or whether the advantage shaping can be applied as a refinement on top of an existing GRPO checkpoint. We resume the Qwen3-8B GRPO run at step 200 – the saturation point in Table 1 – and continue with the canonical AntiSD configuration for 50 further steps. Optimizer state, dataloader index, and reference policy are inherited from the GRPO run via the standard verl [24] resume path; the gate threshold is recalibrated against the new Hwarm to account for the shifted teacher-entropy distribution of the saturated policy.

- Table 4: Continual AntiSD on Qwen3-8B. +AntiSD† resumes from GRPO@200; Steps counts post-resume only. Speedup as in Table 1. Bold = column best.

Method Steps AIME24 AIME25 AIME26 HMMT25 Minerva Average Speedup

GRPO 200 73.5 65.2 64.2 39.2 45.1 57.4@200 1.0× +AntiSD 200 78.4 73.4 73.7 54.4 48.5 65.7@180 5.0× +AntiSD† +50 74.9 72.4 73.0 54.0 50.6 65.0@30 5.0×

- Table 4 contrasts three policies: the saturated GRPO baseline at step 200, AntiSD trained from base, and continual AntiSD that initialises from the GRPO checkpoint. Continual AntiSD essentially matches the from-base peak (65.0 vs 65.7 Avg) using only 30 post-resume steps – 6× fewer than the

180 steps from-base AntiSD takes to reach its own peak – demonstrating that AntiSD’s advantage stacks on top of GRPO rather than replacing it: the deliberation-token reward signal remains informative even at the saturation point that GRPO’s trajectory-level reward cannot push past. The same experiment on Qwen3-4B-IT-2507 (Appendix D.2) comes within ∼ 1pp of the from-base peak before settling slightly lower, suggesting GRPO’s basin admits most but not all of AntiSD’s deliberation pressure.

#### 5 Related Work

On-policy self-distillation. A series of recent works develops on-policy self-distillation along parallel axes [36; 7; 31; 21; 30], all using a teacher–student log-ratio in which the teacher is the student conditioned on privileged context (a verified solution and any environment feedback). These methods build on on-policy distillation [1; 4] (student-sampled rollouts with an external teacher) and learning under privileged information [25; 16], and share the same gradient direction. AntiSD inverts that direction from the start of training, replacing descent on reverse KL with bounded ascent on Jensen–Shannon divergence; an auto-calibrated entropy-triggered gate then disables the term once the teacher’s per-token entropy collapses.

Diagnoses of self-distillation. Self-distillation has been diagnosed as degrading reasoning capability both directly [8] and implicitly via its framing as a response-compression tool [21]. Existing selfdistillation methods [36; 7; 31; 30] report mainly on simpler benchmarks, not the AIME / HMMT-class problems where we observe the clearest failure. A broader line of on-policy distillation diagnostics [4; 13; 12; 28] documents teacher–student capability gaps and distribution mismatch without isolating self-distillation from external-teacher OPD. We confirm the same symptom across model families from 4B to 30B parameters (Section 4), but trace it to a structural property of the per-token signal itself: under privileged context it is conditional pointwise mutual information between the next token and that context (Section 3.1), which biases credit toward tokens the context already implies and away from deliberation tokens. AntiSD acts on this mechanism by reversing the per-token sign rather than by reweighting samples or filtering teacher confidence.

Process reward models and reward shaping. To address sparse credit assignment in RLVR, a separate line of work trains process reward models that score intermediate reasoning steps [14; 26; 18; 22; 10], either from human annotations or from Monte Carlo rollout estimates of step value, while implicit-reward methods such as PRIME [2] derive process rewards from preference signals jointly with the policy. AntiSD’s per-token signal is structurally a PRM, but a training-free one: it is the difference Vt − Vt−1 of the model’s own log-posterior Vt = log P(c | x,y≤t) for the privileged context c. This places the signal in the framework of potential-based reward shaping [19]: the pertoken contributions telescope over a trajectory to the trajectory-level pointwise mutual information log P(c | x,y) − log P(c | x), so the shaping term leaves the set of optimal policies invariant. The PRM is therefore obtained without auxiliary annotation, learned step-value heads, or Monte Carlo rollouts. Our contribution is to identify the conditional PMI bias of this shaping signal (inflating shortcut tokens, suppressing deliberation tokens) and correct it through gradient-direction inversion rather than as an additional reward channel.

#### 6 Conclusion

We identified the per-token signal of default on-policy self-distillation as conditional pointwise mutual information between the next token and the privileged context, exposing a structural shortcut bias that rewards tokens the context already implies and penalises the deliberation tokens that drive search. Anti-Self-Distillation responds by inverting the gradient direction from the first training step, replacing reverse-KL descent with bounded Jensen–Shannon ascent; a single auto-calibrated, entropytriggered gate then disables the term once the teacher’s per-token entropy collapses, preventing the run-away drift that pure ascent would otherwise incur. Across five language models from 4B to 30B parameters, AntiSD reaches a strong GRPO baseline’s accuracy in 2 to 10× fewer training steps and improves final accuracy by up to 11.5 points; the gap between AntiSD and default self-distillation in Table 1 is consistent with sign reversal carrying most of the gain on top of an already-bounded shape. The PMI characterisation describes individual gradient contributions rather than the global optimum of the combined objective, and our evaluation focuses on math reasoning, leaving extensions to multi-turn agentic settings and broader coding benchmarks as natural next directions. A fuller discussion of limitations and broader impacts is in Appendix E.

#### References

- [1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations, 2024.
- [2] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- [3] Jasper Dekoninck, Nikola Jovanovi´c, Tim Gehrunger, Kári Rögnvalddson, Ivo Petrov, Chenhao Sun, and Martin Vechev. Beyond benchmarks: MathArena as an evaluation platform for mathematics with LLMs. arXiv preprint arXiv:2605.00674, 2026.
- [4] Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes, 2026.
- [5] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [6] Bingxiang He, Yuxin Zuo, Zeyuan Liu, Shangziqi Zhao, Zixuan Fu, Junlin Yang, Cheng Qian, Kaiyan Zhang, Yuchen Fan, Ganqu Cui, et al. How far can unsupervised rlvr scale llm training? arXiv preprint arXiv:2603.08660, 2026.
- [7] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.
- [8] Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of llms? arXiv preprint arXiv:2603.24472, 2026.
- [9] Kimi Team, Angang Du, Bofei Gao, Bowen Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1.5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.
- [10] Nakyung Lee, Sangwoo Hong, and Jungwoo Lee. Efficient process reward modeling via contrastive mutual information. arXiv preprint arXiv:2604.10660, 2026.
- [11] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.
- [12] Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan Zhang, Jinqiao Wang, and Tat-Seng Chua. Unifying group-relative and self-distillation policy optimization via sample routing. arXiv preprint arXiv:2604.02288, 2026.
- [13] Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026.
- [14] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.
- [15] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in neural information processing systems, 36:21558–21572, 2023.

- [16] David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, and Vladimir Vapnik. Unifying distillation and privileged information. arXiv preprint arXiv:1511.03643, 2015.
- [17] Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policydistillation.
- [18] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.
- [19] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the International Conference on Machine Learning (ICML), volume 99, pages 278–287. Citeseer, 1999.
- [20] Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025. URL https: //arxiv.org/abs/2512.13961.
- [21] Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. On-policy self-distillation for reasoning compression. arXiv e-prints, pages arXiv–2603, 2026.
- [22] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for LLM reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=A6Y7AqlzLW.
- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [24] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.
- [25] Vladimir Vapnik and Akshay Vashist. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544–557, 2009.
- [26] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024.
- [27] Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.
- [28] Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. Paced: Distillation and onpolicy self-distillation at the frontier of student competence. arXiv preprint arXiv:2603.11178, 2026.

- [29] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [30] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026.
- [31] Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275, 2026.
- [32] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [33] Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024,

- 2024.

[34] Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025,

- 2025.

[35] Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2026,

- 2026.

- [36] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.
- [37] Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

### Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information Supplementary Material

##### Table of Contents

- A Proofs and Deferred Statements. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .13
- B Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C Self-Teacher Context Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- D Additional Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- D.1 Component sensitivity on Qwen3-8B. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .16
- D.2 Continual AntiSD on Qwen3-4B-IT-2507. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .17

- E Limitations and Broader Impacts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

#### A Proofs and Deferred Statements

This appendix collects the derivations summarised in Section 3: the reverse-KL gradient identity (Equation (3), Lemma 1); the PMI characterization of ut under self-distillation (Equation (4), Lemma 2) and its trajectory-level telescope into a potential-based shaping term (Lemma 3); the JSD f-divergence shape used in Equation (6) (Lemma 4); and the properties of φ relied on in Section 3.2 (Lemma 5). All distributions are over the next-token vocabulary; we suppress the conditioning (x,y<t) where it is fixed.

- Lemma 1 (Reverse-KL gradient identity, Equation (3)). Let πS(·) = πθ(· | x,y<t) and πT(·) = πθ(· | x,c,y<t) with stop-gradient on πT, and write uv := log πT(v) − log πS(v). Then

∇θ DKL(πS∥πT) = −Ev∼π

S

[uv · ∇θ log πS(v)].

Proof. Expanding D := DKL(πS∥πT) = v πS(v)(log πS(v) − log πT(v)) and applying the product rule,

∇θD =

v

∇θπS(v)(log πS(v) − log πT(v))

(I)

+

v

πS(v)∇θ(log πS(v) − log πT(v))

(II)

.

Term (II) vanishes: ∇θ log πT(v) = 0 by stop-gradient, and v πS(v)∇θ log πS(v) =

v ∇θπS(v) = ∇θ v πS(v) = ∇θ1 = 0. For term (I), the score-function identity ∇θπS(v) = πS(v)∇θ log πS(v) gives

(I) = Ev∼π

S

[(log πS(v) − log πT(v))∇θ log πS(v)] = −Ev∼π

S

[uv∇θ log πS(v)], where log πS − log πT = −u. Combining (I) + (II) proves the claim.

| |
|---|

- Lemma 2 (PMI characterization, Equation (4)). Under the self-distillation parameter sharing πS(·) = πθ(· | x,y<t) and πT(·) = πθ(· | x,c,y<t),

πθ(yt | x,c,y<t) πθ(yt | x,y<t)

= PMI(yt ; c | x,y<t).

ut = log

Proof. Bayes’ rule applied to the joint of (yt,c) given (x,y<t) gives

πθ(yt | x,c,y<t) =

πθ(c | x,y≤t)πθ(yt | x,y<t) πθ(c | x,y<t)

,

θ(yt|x,c,y<t) πθ(yt|x,y<t) = ππθ(c|x,y≤t)

so π

θ(c|x,y<t), and taking logs yields the conditional pointwise mutual information PMI(yt;c | x,y<t).

| |
|---|

- Lemma 3 (Trajectory-level potential shaping). Summing ut over a complete trajectory telescopes to a sequence-level pointwise mutual information:

T

t=1

ut = log πθ(c | x,y) − log πθ(c | x) = PMI(y ; c | x).

Hence the per-token contributions {ut} are the increments of a potential Φt := log πθ(c | x,y≤t), and the augmented advantage in Equation (2) is a potential-based reward shaping in the sense of Ng et al. [19]: it leaves the set of optimal policies invariant for any underlying scalar reward.

Proof. By Lemma 2, ut = log πθ(c | x,y≤t)−log πθ(c | x,y<t) = Φt −Φt−1. The sum telescopes to ΦT − Φ0 = log πθ(c | x,y) − log πθ(c | x). The potential-based shaping invariance result follows directly [19].

| |
|---|

- Lemma 4 (JSD f-divergence shape, Equation (6)). Write the symmetric Jensen–Shannon diver-

gence in f-divergence form DJSD(πS∥πT) = Eπ

T

[f(πS/πT)] with generator f(r) = 12r log 1+2rr +

- 1

- 2 log 1+2r. Then f′(r) = 12 log 1+2rr and, for r = πS/πT = e−u,

f′ π

S

πT = 12(log 2 − softplus(u)) = −φ(u),

recovering the AntiSD advantage AAntiSDt = −φ(ut) via the score-function identity in Equation (5).

Proof. Differentiating f term by term, with g(r) := log 1+2rr = log 2 + log r − log(1 + r) and g′(r) = 1r − 1+1r = r(1+1 r), the product rule gives

f′(r) = 12g(r) + 21r g′(r) − 12 · 1+1r = 21 g(r) + 1+1r − 1+1r = 12 log 1+2rr.

Substituting r = e−u gives 1+2rr = eu2+1, so f′(e−u) = 12(log 2 − log(1 + eu)) = 12(log 2 − softplus(u)) = −φ(u).

| |
|---|

- Lemma 5 (Properties of φ). The shape φ(u) := 12(softplus(u) − log 2) from Equation (6) satisfies:

- (i) Strict monotonicity: φ′(u) = 12σ(u) > 0, where σ(u) = (1 + e−u)−1 is the logistic function.

- (ii) Sign preservation: φ(0) = 0 and sign(φ(u)) = sign(u).
- (iii) One-sided bound: φ(u) ≥ −12 log 2 for all u ∈ R, with equality attained as u → −∞; φ is unbounded above.

u

Proof. (i) dud softplus(u) = e

1+eu = σ(u), so φ′(u) = 12σ(u) > 0 for all u. (ii) softplus(0) = log 2, so φ(0) = 0; combined with (i), φ is strictly increasing through 0. (iii) softplus(u) = log(1+eu) > 0 for all u, with infu softplus(u) = 0 attained as u → −∞, hence φ(u) ≥ −12 log 2 with equality in the limit. As u → ∞, softplus(u) ∼ u → ∞, so φ has no upper bound.

| |
|---|

Properties (i)–(ii) ensure the shape inherits the per-token sign structure of Equations (3)–(4): −φ flips that sign at the source, so deliberation tokens (ut < 0) receive positive advantage and shortcut tokens (ut > 0) receive negative advantage. Property (iii) bounds the AntiSD advantage at 12 log 2 on the deliberation side – the side that observation (O2) flagged as both over-sampled and heavy-tailed – so the cap re-balances per-token gradient contributions against the lighter shortcut side. The entropy gate then disables the term entirely once ut degenerates into floor-level noise.

Algorithm 1 Anti-Self-Distillation (AntiSD) – one training step. Require: Policy πθ, batch {(xi,yi,ci)}Bi=1 with sequence-level GRPO advantage Aseqi ; hyperpa-

rameter λmax; gate state g and calibrated threshold τdown (warmup baseline Hwarm).

- 1: for each rollout i and token t ∈ {1,...,Ti} do
- 2: si,t ← log πθ(yi,t | xi,yi,<t) ▷ student log-prob
- 3: ti,t ← stopgrad(log πθ(yi,t | xi,ci,yi,<t)) ▷ teacher log-prob
- 4: ui,t ← ti,t − si,t ▷ = PMI(yi,t; ci | xi,yi,<t), see Equation (4)
- 5: φi,t ← 21(softplus(ui,t) − log 2) ▷ JSD f-divergence advantage; see Equation (6)

- 6: end for
- 7: H ← mediani,t H[πT(· | xi,yi,<t)] ▷ teacher entropy
- 8: Update gate: g ← 1 if H ≥ Hwarm, g ← 0 if H < τdown, else unchanged.
- 9: λ ← g · λmax
- 10: Ai,t ← Aseqi − λ · stopgrad(φi,t) ▷ ascent on DJSD(πS∥πT); advantage is treated as a constant weight (cf. Eq. (5))
- 11: Update θ via standard policy gradient using {Ai,t}.

#### B Hyperparameters

We evaluate five language models: Qwen3-8B, Qwen3-4B-Instruct-2507, Olmo-3-7B-Instruct, Olmo-3-7B-Think, and Qwen3-30B-A3B. All models share the configuration below, with training at 32K maximum sequence length; the only model-conditional knob is the evaluation-time maximum sequence length, which is doubled for the thinking-model variant (Olmo-3-7B-Think) to accommodate longer chains-of-thought. AntiSD adds the gate parameters in the bottom block; the auto-calibration of Hwarm and τdown is described in Section 4 (Setup).

Table 5: Training and evaluation hyperparameters. AntiSD shares the GRPO/SD configuration and adds only the gate parameters in the bottom block.

Hyperparameter Value

Optimizer AdamW Learning rate 1 × 10−6 Training steps 200 Batch size (problems) 32 Rollouts per problem (group size) 8 Max sequence length (training) 32K (all models) Max sequence length (evaluation) 32K (Qwen, OLMo-Instruct), 64K (OLMo-Think) GRPO clip ratio 0.2 KL penalty coefficient 0 (no reference-policy KL term, following [32]) Reference model πref frozen base model Training rollout sampling temperature 1.0, top-p=1.0 Evaluation rollout sampling temperature 0.7, top-p=0.95 Evaluation rollouts per problem 32 (AIME, HMMT) / 4 (MinervaMath) Hardware 8 NVIDIA H20 GPUs per node (multi-node for Qwen3-30B-A3B) Framework verl [24]

AntiSD mixing weight λmax 0.5 AntiSD warmup length W 5 steps AntiSD deactivation threshold τdown 0.93 · Hwarm AntiSD reactivation threshold Hwarm

#### C Self-Teacher Context Examples

We show the context that the self-teacher πT(· | x,y<t,c) sees when re-evaluating the student’s response. The teacher’s input concatenates the original prompt, a verified solution (sampled from a successful rollout in the same batch when available, else from the dataset), and a binary feedback string indicating correctness, followed by an instruction to re-solve the problem. The student’s original response y is placed in the assistant role; the teacher then re-evaluates y’s log-probabilities under this enriched context. Templates follow [7; 36] and are identical for math and code; only the

verified-solution and feedback strings differ by task. Colors: original prompt, feedback string (a sub-component of c), student response y (re-evaluated by teacher).

Reading the template. The block under Your previous attempt: carries the verified solution (a peer rollout or a dataset reference); the Previous assessment: line is the binary correctness feedback for the student’s actual rollout y that follows in the assistant turn, not for the verified solution shown above. The two slots therefore play complementary roles: the verified solution narrows the teacher’s posterior toward the correct answer, while the assessment string indicates that the trajectory the teacher will now re-evaluate is the student’s own (which may be wrong). The deliberate asymmetry between “correct reference” and “incorrect attempt” gives the teacher a contrastive signal, and matches the prompt structure used in prior on-policy self-distillation work [7; 36].

Math (AIME / HMMT / MinervaMath) — self-teacher context [User] Solve the following math problem. Place the final answer in \boxed{}. Find the number of integer pairs (a, b) with 1 ≤ a, b ≤ 100 such that gcd(a, b) + lcm(a, b) = a + b + 50. Your previous attempt:

(a successful rollout from the same batch, or a reference solution from the dataset) Let d = gcd(a, b), a = dm, b = dn...so d(1 + mn) = dm + dn + 50... \boxed{25}

Previous assessment: Your answer is incorrect. Now solve this problem step by step.

[Assistant] (student’s original response y, re-evaluated by teacher) We start by writing a = dm, b = dn with gcd(m, n) = 1... \boxed{30}

###### Code (LiveCodeBench v6 / Dolci-RLZero) — self-teacher context

[User] You are a coding expert. Write a correct Python function that solves the following problem; place the final code in a \texttt{“‘python} block. Given a list of integers nums, partition it into the minimum number of strictly increasing subsequences and return that number.

Your previous attempt: (a successful rollout from the same batch, or a reference implementation) from bisect import bisect_left def partitions(nums): tails = []; for x in nums: ...; return len(tails) Previous assessment: Your code passes 7 of 12 test cases. Now solve this problem step by step.

[Assistant] (student’s original response y, re-evaluated by teacher) We track the tail of each subsequence in sorted order... def partitions(nums): ...

The feedback string is the only task-conditional piece. For math we use a binary form (“Your answer is correct.” / “Your answer is incorrect.”); for code, we use a continuous form (“Your code passes N of M test cases.”) matching the per-test fraction returned by the reward function. Both forms parallel the task’s underlying score: math is exact-match boolean, while code’s score is a fraction over executed test cases.

#### D Additional Experiments

This section gathers experimental results referenced from the main paper but deferred to the appendix for space. Each subsection corresponds to one of the experimental probes whose narrative is summarised in Section 4.

##### D.1 Component sensitivity on Qwen3-8B

- Table 6 mirrors the Qwen3-4B-IT-2507 ablation in Section 4.3 on Qwen3-8B. Two patterns from the main analysis carry over: rev. KL ascent collapses (30.6 Avg, −35.1 pp from canonical) and the

- Table 6: Component sensitivity on Qwen3-8B. Same format as Table 3. Speedup uses GRPO’s best-Avg step (200). Bold = column best among +AntiSD rows.

Method Div τdown Compose AIME24 AIME25 AIME26 HMMT25 Minerva Average Speedup GRPO – – – 73.5 65.2 64.2 39.2 45.1 57.4@200 1.0×

+AntiSD

rev. KL 0.93 add. 40.1 30.5 26.9 14.9 40.7 30.6@200 ×

JSD none add. 75.5 68.7 69.2 45.9 48.7 61.6@40† 6.7× JSD 0.90 add. 77.8 73.2 73.2 57.0 48.2 65.9@100 6.7× JSD 0.95 add. 76.3 73.4 75.8 51.9 49.7 65.4@180 1.4×

JSD 0.93 mult. 73.1 60.6 61.7 38.5 44.6 55.7@130 ×

JSD 0.93 add.∗ 78.1 72.7 73.2 52.3 50.9 65.4@40 6.7× JSD 0.93 add. 78.4 73.4 73.7 54.4 48.5 65.7@180 5.0×

† The no-gate run on Qwen3-8B collapses by step ∼ 90 (cf. Section 4.2); the reported peak is from 8 pre-collapse checkpoints. ∗ Gate signal swapped from teacher- to student-perplexity.

gate is necessary on this model family (the no-gate run shows a transient peak at step ∼ 40 before collapsing by step ∼ 90, echoing the dynamics in Section 4.2). The threshold-sensitivity story differs in direction: loosening from 0.93 to 0.90 slightly improves Avg on Qwen3-8B (65.7 → 65.9), in stark contrast to the −8.3pp drop on Qwen3-4B-IT-2507. Tightening to 0.95 slightly drops the peak (65.7 → 65.4) and slows ignition by ∼ 4×. The canonical 0.93 is therefore not the per-model optimum on Qwen3-8B but the value that transfers across all models we evaluate without per-model retuning.

D.2 Continual AntiSD on Qwen3-4B-IT-2507

We repeat the continual experiment from Section 4.4 on Qwen3-4B-IT-2507. Unlike Qwen3-8B, where continual AntiSD essentially closes the gap to from-base AntiSD (65.0 vs 65.7 Avg), continual on the smaller model peaks at 61.9 briefly at step +20 before drifting to a plateau of ≈ 60.5; the plateau is 2.3pp short of the from-base peak (62.8).

- Table 7: Continual AntiSD on Qwen3-4B-IT-2507. Same setup as Table 4 on the smaller model. Bold = column best.

Method Steps AIME24 AIME25 AIME26 HMMT25 Minerva Average Speedup GRPO 200 67.8 57.7 63.5 34.1 33.2 51.3@200 1.0×

+AntiSD 200 76.6 70.2 74.4 46.7 46.4 62.8@100 10.0× +AntiSD† +50 76.2 69.7 73.3 43.2 47.0 61.9@20 10.0×

The plateau is consistent with two interpretations: (i) GRPO’s basin admits some, but not all, of the deliberation pressure that AntiSD applies from base; (ii) the auto-recalibrated gate threshold derived from the saturated policy’s Hwarm is mildly conservative on this model, leaving residual AntiSD gain unrealised. We do not attempt to disentangle these here; the practical takeaway is that continual AntiSD provides a strong fraction of the from-base improvement at a fraction of the cost, with a model-conditional ceiling.

#### E Limitations and Broader Impacts

Scope and extensions. The conditional-PMI account in Section 3.1 is a local, per-step characterization of the per-token signal rather than a global-optimum statement about the combined objective; understanding the long-horizon dynamics under the full ascent + gate update is itself an interesting question. Our evaluation spans five language models from the Qwen3 and Olmo-3 families (4B–30B parameters) on mathematical reasoning, with an initial probe on code reasoning (Section 4.1, Table 2). Several natural extensions follow the same algorithmic skeleton (Algorithm 1) without modifying the AntiSD update: (i) multi-turn agentic settings, where the reward depends on a sequence of tool calls rather than a single rollout, with the privileged context spanning the full interaction trace; (ii) broader code-reasoning benchmarks such as LiveCodeBench v6 with longer per-problem horizons and richer

test-case structure; and (iii) richer privileged-context content – process-level critiques, partial-credit annotations, and rationale-comparison rankings – replacing the binary or continuous correctness feedback used here. Larger model scales beyond 30B and multimodal conditioning are also natural settings to test whether the conditional-PMI characterization remains the dominant credit-assignment signal.

Broader impacts. AntiSD is a post-training method that improves credit assignment in RLVR. Positive impacts: stronger open-weight reasoning models, lower training cost (2 to 10× fewer steps to reach a given accuracy), and a clearer theoretical handle on why default self-distillation underperforms on math reasoning, which may inform future training-free PRM designs. Negative impacts: as with any improvement to LLM reasoning, gains are dual-use; a stronger reasoning model can be applied to adversarial or harmful tasks. AntiSD does not introduce a new attack surface beyond the pre-existing dual-use profile of large language models, and we do not release new high-risk model artifacts. We see no specific path to fairness, privacy, or safety harms beyond those already attaching to the underlying open-weight base models.

