## arXiv:2605.19436v1[cs.LG]19May2026

### CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization

Ahmed Heakl1 Abdelrahman M. Shaker1 Youssef Mohamed1 Rania Elbadry1 Omar Fetouh1 Fahad Shahbaz Khan1,2 Salman Khan1,3 1MBZUAI 2Linköping University 3Australian National University

#### Abstract

When a model produces a correct solution under reinforcement learning with verifiable rewards (RLVR), every token receives the same reward signal regardless of whether it was a decisive reasoning step or a grammatical filler. A natural fix is to condition the model on the correct answer as a teacher, identifying tokens it would have generated differently had it known the answer. Prior work shows this either corrupts training by leaking the answer into the gradient, or produces a weak signal that cannot distinguish decisive steps from filler, since both look equally surprising relative to the model’s baseline. We propose Contrastive Evidence Policy Optimization (CEPO), which asks a sharper question at every token: not just “does the correct answer favor this token?” but “does the correct answer favor it while the wrong answer disfavors it?” A token satisfying both is a genuine reasoning step; one satisfying neither is filler. The wrong-answer teacher is constructed from rejected rollouts already in the training batch, incurring no additional sampling cost. We prove CEPO inherits all structural safety guarantees of the prior state of the art while strictly sharpening credit at decisive tokens, with the improvement vanishing exactly at filler positions. Empirically, CEPO achieves 43.43% and 60.56% average accuracy across five multimodal mathematical reasoning benchmarks at 2B and 4B scale, respectively, versus 41.17% and 57.43% for GRPO under identical training budgets. Distribution-matching self-distillation methods (OPSD, SDPO) fall below the untrained baseline, empirically confirming the information leakage our theory predicts. Our code is available at https://github.com/ahmedheakl/CEPO.

#### 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has become the dominant paradigm for post-training large language models to reason [17, 4, 20]. The core loop is simple: sample rollouts from the current policy, score them against a verifier, and update the policy to increase the probability of correct trajectories. Group Relative Policy Optimization (GRPO) [17] operationalizes this at scale by eliminating the value network entirely, normalizing rewards within groups of sampled responses to obtain sequence-level advantages. Yet the simplicity that makes GRPO practical also makes it blunt: every token in a correct trajectory receives the same positive advantage,

0.40

CEPO GRPO RLSD

0.38

Accuracy

0.36

0.34

0 5 10 15 20 25 30 35 40 45 50

Training Step

Figure 1: Accuracy over 50 training steps. CEPO improves faster than GRPO and RLSD, reaching its largest gap around step 40 before partially converging by the final checkpoint.

Preprint.

and every token in a wrong one receives the same negative signal. The credit assignment problem, which tokens actually mattered?, is left entirely unresolved.

This is not a minor inefficiency. In mathematical reasoning, a single arithmetic error or a single correct inferential step can determine the outcome of an entire chain-of-thought [7, 5]. Uniform credit assignment wastes gradient signal on filler tokens (connectives, formatting, boilerplate) while underweighting the few decisive tokens that distinguish correct from incorrect reasoning. The result is slow convergence, noisy updates, and poor sample efficiency, problems that worsen as reasoning chains grow longer and sparser in decision-relevant content [24]. Figure 1 illustrates this empirically, with CEPO improving faster than GRPO and RLSD early in training.

A natural fix is to condition the model on the correct answer r+ as its own teacher, using the resulting distribution PT+(· | x,r+) as a dense, token-level training signal. On-policy self-distillation methods [26, 6, 12] pursue exactly this, minimizing a per-token divergence between PT+ and the student PS over on-policy rollouts. [21] showed this is structurally unsafe: the gradient of any divergence objective decomposes into a benign component and a harmful deviation with variance proportional to I(Yt;R+ | X). As training progresses the benign signal vanishes and the deviation dominates, driving the model to encode spurious x→r+ correlations, a pathology termed information leakage that is irreducible regardless of implementation details.

RLSD [21] resolved leakage by evaluating the evidence ratio PT+(yt)/PS(yt) only at the sampled token, under a stop-gradient, using it solely to modulate the magnitude of the GRPO advantage while

keeping its sign anchored to the verifier. No vocabulary-wide sum over r-conditioned weights appears in the gradient, so privileged information cannot redirect gradient flow. This is a sound structural recipe for safe self-distillation, but structural safety is not the same as signal quality. We identify three specific limitations of RLSD’s evidence ratio. The denominator PS(yt) reflects base-rate fluency, not semantic relevance, so a common token suppresses the ratio regardless of how strongly r+ favors it (fluency confound). For wrong trajectories, the signal penalizes tokens that r+ would have supported, indirect, with no explicit grounding in what r− predicts (asymmetric negative). Most critically, PT+/PS cannot distinguish a filler token that both the correct and wrong answers support equally from a decisive reasoning step that r+ supports while r− actively disfavors; both receive identical weight (one-sided evidence).

We propose Contrastive Evidence Policy Optimization (CEPO), which replaces PT+/PS with the contrastive ratio PT+/PT−(yt), where PT− is the model conditioned on a wrong answer drawn from rejected rollouts already in the training batch. The student prior PS cancels entirely, eliminating the fluency confound by construction. The contrastive ratio admits a clean Bayesian interpretation as the differential belief update: how much token yt simultaneously raises posterior belief in r+ and lowers it for r−. Decisive reasoning steps score high; filler tokens score near unity.

We prove CEPO preserves all structural safety guarantees of RLSD: direction anchoring (sign(Aˆt) = sign(A) for all tokens) and leakage-free gradients (no vocabulary-wide r-conditioned sum). When

PT−(yt) = PS(yt), CEPO reduces exactly to RLSD, making RLSD a limiting case when the wronganswer teacher carries no information. Beyond these guarantees, Proposition 1 gives exact necessary

and sufficient conditions for CEPO to assign strictly sharper credit than RLSD at any token: for correct trajectories, sharpness holds precisely when PT−(yt) < PS(yt), a condition we validate empirically concentrates at arithmetically and inferentially decisive positions rather than at filler.

##### Contributions.

- 1. We identify three concrete limitations of RLSD’s evidence ratio: the fluency confound, asymmetric negative signal, and one-sided evidence.
- 2. We propose CEPO, replacing PT+/PS with PT+/PT−, with a Bayesian interpretation as the differential belief update which inherits all structural safety guarantees of RLSD while strictly generalizing it.
- 3. We derive exact conditions under which CEPO sharpens credit relative to RLSD and validate empirically that these concentrate at semantically decisive token positions.
- 4. We demonstrate accuracy improvements of 3.7% and 2.2% over base at 2B and 4B scale across five multimodal mathematical reasoning benchmarks.

#### 2 Related Work

RLVR and the credit assignment bottleneck. Reinforcement learning with verifiable rewards trains language models by scoring sampled rollouts against a deterministic verifier [4]. GRPO [17] eliminates the value network by normalizing rewards within a rollout group, and extensions such as DAPO [22] improve exploration stability. All methods in this family assign uniform sequence-level advantages: every token in a correct trajectory receives the same signal regardless of its contribution. Token-level methods address this gap either through Monte Carlo re-simulation,

- as in VinePPO [7] and SPO [5], or through a separately trained process reward model (PRM; [9, 16]). Both families appear in the top block of Table 1: they improve credit assignment without privileged information but either require expensive re-simulation or an auxiliary network.

Table 1: Comparison of credit assignment methods. Priv.: uses privileged info at training time. Leak-free: no vocabulary-wide r-conditioned gradient. Contr.: uses both positive and negative references. No Aux.: requires no auxiliary network.

Leakfree Contr.

No Aux.

Method Priv.

GRPO [17] ✗ — ✗ ✓ PPO [15] ✗ — ✗ ✗ VinePPO [7] ✗ — ✗ ✓ SPO [5] ✗ — ✗ ✓ PRM [9] ✗ — ✗ ✗ DPO/cDPO [14, 2] — — ✓ ✓

OPSD [26] ✓ ✗ ✗ ✓ SDPO [6] ✓ ✗ ✗ ✓ HDPO [3] ✓ ✗ ✗ ✓ RLSD [21] ✓ ✓ ✗ ✓

CEPO (ours) ✓ ✓ ✓ ✓

On-policy self-distillation with privileged information. A natural alternative is to condition the model on the correct answer r+ as its own teacher, producing a dense token-level signal at no auxiliary network cost. OPSD [26] minimizes the per-token KL divergence between the privileged teacher PT+ and the student; SDPO [6] extends this with Jensen-Shannon divergence and EMA teacher stabilization; and HDPO [3] applies the same recipe specifically to prompts where all rollouts fail. As shown by [21], any method that uses PT+ as a distributional target produces gradients containing a vocabulary-wide sum of r-conditioned weights, a structural source of information leakage whose variance is irreducible regardless of implementation. These methods are marked Priv. but not Leak-free in Table 1, and we confirm their degradation empirically in §5. The closest work to the contrastive direction within the DPO family [14] is cDPO [2], which identifies critical tokens via contrastive estimation, but it operates offline on fixed response pairs under a sequence-level implicit reward rather than within the RLVR loop.

RLSD [21] resolves leakage by evaluating the teacher signal only at the sampled token under a stop-gradient, using the evidence ratio PT+(yt)/PS(yt) solely to modulate the magnitude of the GRPO advantage while anchoring its direction to the verifier. This makes RLSD both Priv. and Leak-free, which no prior method achieves. However, the denominator PS(yt) conflates reasoning importance with base-rate fluency, the negative signal for wrong trajectories is indirect, and the ratio cannot distinguish a decisive reasoning step from filler when both have the same PT+/PS value.

#### 3 Method

##### 3.1 Preliminaries

Let πθ be an autoregressive language model with parameters θ and vocabulary V, trained on S = {(xi,ri+)}Ni=1 where ri+ is a verifiable correct answer. A deterministic verifier R : X × Y → {0,1} scores responses. GRPO [17] samples G rollouts per question and computes a normalized sequencelevel advantage:

R(x,y(i)) − µG σG

A(i) =

, (1)

partitioning rollouts into correct (G+) and wrong (G−) subsets. We define three next-token distributions sharing parameters θ but differing in context:

- PS(yt) ≜ πθ(yt | x,y<t), PT+(yt) ≜ πθ(yt | x,r+,y<t), PT−(yt) ≜ πθ(yt | x,r−,y<t), (2)

denoting the student, correct teacher, and wrong teacher respectively. We write sg(·) for the stopgradient operator.

[Figure 1]

##### Figure 2: CEPO training pipeline and its relationship to GRPO and RLSD. Given a question x,

the policy πθ produces G rollouts that are partitioned into correct (G+) and wrong (G−) sets by a verifiable reward. CEPO conditions two frozen teachers on a sampled correct rationale r+ ∈ G+ and

rejected rationale r− ∈ G−, and defines a per-token contrastive evidence delta ∆CEt that amplifies advantage at decisive tokens (large |∆CEt |) while leaving filler tokens near unit weight. Then, the token-level modulated advantage is plugged into a standard PPO-clipped surrogate.

##### 3.2 Background: Leakage in Self-Distillation and the RLSD Fix

Methods such as OPSD [26] and SDPO [6] minimize per-token KL divergence between a privileged teacher PT+ and the student, producing a gradient of the form:

PT+(v | r+)∇θ log PS(v). (3)

∇θLOPSD = −

v∈V

This vocabulary-wide sum encodes r+ directly into every gradient direction. [21] showed this produces a harmful deviation δ(θ;r+) with variance ∝ I(Yt;R+ | X) that dominates as training progresses, a pathology termed information leakage that is irreducible regardless of implementation. Our results confirm it empirically: OPSD and SDPO fall below the untrained baseline on four of five benchmarks (§5).

RLSD [21] resolves leakage by evaluating the teacher signal only at the sampled token yt under stop-gradient, using the evidence ratio PT+(yt)/PS(yt) solely to modulate the magnitude of the GRPO advantage:

wtRLSD = exp sign(A) · sg(log PT+(yt) − log PS(yt)) , Aˆ(ti) = A(i) · (1 − λ) + λ · clip wtRLSD, 1−ϵw, 1+ϵw .

(4)

Because Aˆt is θ-constant via sg, no vocabulary-wide sum appears in the gradient and the update direction is anchored to the verifier.

##### 3.3 Limitations of Single-Reference Evidence

Despite its safety guarantees, RLSD’s ratio PT+(yt)/PS(yt) has three signal quality limitations. (1) Fluency confound: the denominator PS(yt) reflects base-rate corpus frequency, not semantic relevance, suppressing the ratio at common tokens regardless of the numerator. (2) Asymmetric

negative signal: for wrong trajectories, the weight PS/PT+ penalizes tokens that r+ would have supported, indirect, with no grounding in what r− predicts. (3) One-sided evidence: PT+/PS cannot distinguish a filler token (supported equally by both r+ and r−) from a decisive reasoning step (r+ supports it, r− disfavors it); both receive identical weight if their PT+/PS ratio coincides.

Algorithm 1 Contrastive Evidence Policy Optimization (CEPO) Require: Policy πθ, dataset S, verifier R, group size G, λ schedule, clip bounds ϵw, ϵ

- 1: for each training iteration do
- 2: for each (x,r+) in batch do
- 3: Sample {y(i)}Gi=1 ∼ πθ(· | x); compute A(i) via Eq. (1)
- 4: r− ← answer arg minj∈G− R(y(j)) ; if G− = ∅ set PT− ← PS
- 5: for each trajectory i, each position t do
- 6: ∆t ← sg(log PT+(yt) − log PT−(yt))
- 7: Aˆ(ti) ← A(i) · [(1−λ) + λ · clip(esign(A

(i))∆t, 1−ϵw, 1+ϵw)]

- 8: end for
- 9: Update θ via PPO clipped surrogate with Aˆ(ti)
- 10: end for
- 11: end for

- 3.4 Contrastive Evidence Policy Optimization Contrastive evidence delta. We replace PT+(yt)/PS(yt) with the contrastive ratio

- PT+(yt)/PT−(yt), where r− is the final answer of the lowest-reward rejected rollout in G−, available at no additional inference cost. The student prior PS cancels entirely, eliminating the fluency confound by construction. The contrastive evidence delta is:

PT+(yt) PT−(yt)

∆CEt = sg log

. (5)

Bayesian interpretation. Applying Theorem 4 of [21] to both teachers and subtracting, PS cancels and we obtain:

P(r− | x,y≤t) P(r− | x,y<t)

P(r+ | x,y≤t) P(r+ | x,y<t)

∆CEt = log

. (6)

−log

belief update for r+

belief update for r−

Thus ∆CEt is the differential belief update: how much token yt simultaneously strengthens posterior belief in r+ and weakens it for r−. Decisive steps receive large positive ∆CEt ; filler tokens receive ∆CEt ≈ 0.

Token-level advantage and update. The contrastive weight and clipped token-level advantage are:

sign(A)

PT+(yt) PT−(yt)

wtCE = exp sign(A) · ∆CEt =

,

(7)

Aˆ(ti) = A(i) · (1−λ) + λ · clip(wtCE, 1−ϵw, 1+ϵw) .

where λ decays linearly from λ0 to 0 over Twarm steps. The policy is updated by maximizing the standard PPO-style clipped surrogate objective [15] with Aˆ(ti) in place of A(i). When G− = ∅, we set PT− = PS, recovering RLSD exactly. CEPO adds one teacher forward pass over RLSD per trajectory, the same marginal overhead as RLSD over GRPO, with no additional sampling cost. Algorithm 1 summarizes the full procedure.

Theoretical guarantees. We establish three formal properties of CEPO (proofs in Appendix A). Theorem 1 (CEPO Properties). For λ ∈ [0,1] and ϵw ∈ (0,1), CEPO satisfies:

- (i) Direction anchoring. sign(Aˆt) = sign(A) for all t, privileged information cannot flip any token’s update direction.
- (ii) Leakage-free gradient. ∇θLCEPO contains no vocabulary-wide r-conditioned sum; r+ and r− enter only as stop-gradiented scalars at the sampled token.
- (iii) RLSD containment. Setting PT− = PS recovers RLSD exactly; RLSD is the degenerate case where the wrong-answer teacher carries no information.

Beyond safety, we characterize when CEPO strictly improves over RLSD.

Table 2: Results on five multimodal mathematical reasoning benchmarks. All methods trained 50 steps on Geo3k, λ0=0.5, ϵw=0.5. OPSD and SDPO degradation below baseline is consistent with the information leakage identified in §3.2.

DynaMath LogicVista MathVis.m MMMU WeMath Average Acc. Qwen3-VL-2B-Instruct

Base 50.08±0.7 32.81±2.2 19.41±2.3 44.11±1.7 52.24±1.2 39.73±0.8 +GRPO [17] 50.36±0.7 37.50±2.3 21.05±2.3 42.33±1.6 54.60±1.2 41.17±0.8 +OPSD [26] 46.85±0.7 28.79±2.1 14.14±2.0 43.78±1.7 41.26±1.2 34.96±0.7 +SDPO [6] 46.65±0.7 29.46±2.2 15.46±2.1 43.00±1.7 43.91±1.2 35.70±0.8 +RLSD [21] 50.36±0.7 36.38±2.3 23.39±2.3 39.44±1.6 55.26±1.2 40.05±0.8 +CEPO (Ours) 51.44±0.7 37.72±2.3 25.99±2.5 45.78±1.7 56.21±1.2 43.43±0.8

Qwen3-VL-4B-Instruct

Base 64.59±0.7 54.91±2.4 44.41±2.8 53.56±1.7 74.31±1.0 58.36±0.8 +GRPO [17] 63.97±0.7 54.98±2.4 42.76±2.8 52.34±1.7 73.10±1.1 57.43±0.9 +OPSD [26] 61.80±0.7 55.58±2.3 44.41±2.8 47.00±1.7 72.36±1.1 56.23±0.8 +SDPO [6] 61.58±0.7 52.01±2.4 43.42±2.8 48.11±1.7 73.62±1.1 55.75±0.9 +RLSD [21] 65.07±0.7 56.92±2.3 44.08±2.8 53.22±1.7 73.28±1.1 58.51±0.8 +CEPO (Ours) 65.37±0.7 61.16±2.3 47.37±2.9 54.11±1.7 74.77±1.0 60.56±0.9

Proposition 1 (Discriminative sharpness). For a correct trajectory: wtCE > wtRLSD if and only if PT−(yt) < PS(yt), precisely when the wrong-answer teacher disfavors this token relative to the student prior. The symmetric condition holds for wrong trajectories. At filler tokens, PT− and PT+ both track PS closely, so wtCE ≈ wtRLSD ≈ 1: CEPO introduces no spurious signal where none is warranted.

This concentration property is the crux of CEPO’s design. RLSD’s denominator PS(yt) is blind to r−, so it cannot distinguish a decisive reasoning step from a fluent filler token when both happen

to have the same PT+/PS ratio. CEPO’s denominator PT− breaks this tie: a token the wrong answer actively disfavors receives a smaller denominator and strictly higher credit, exactly at positions where

the gradient signal is semantically meaningful. The filler-token neutrality is therefore not a limitation but a correctness criterion, amplifying filler gradients would introduce noise, not signal. We validate the sharpness conditions empirically via token-weight analysis in §5.2.1

#### 4 Experiments

Models and training. We train Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct [1] using the EasyR1 [28] framework with FSDP [27] and vLLM [8]-accelerated inference. All models are finetuned with LoRA (rank 16) for 50 steps on Geo3k [11], a geometry question-answering dataset of 3,000 training problems with verifiable numeric answers. We use AdamW [10] with lr 10−6 (CEPO 5×10−6), batch size 32, rollout group size G = 8, and maximum sequence length 2,048 tokens. For all CEPO runs, λ0 = 0.5 with linear decay to 0 over Twarm = 25 steps and ϵw = 0.5 unless otherwise stated. The negative reference r− is the final answer extracted from the lowest-reward rejected rollout in the current group. The teacher is the same as the actor. All experiments run on NVIDIA RTX6000 Pro Blackwell 100GBs GPUs. Table 3 reports wall-clock training times; CEPO’s two teacher forward passes add 36 minutes over GRPO, comparable to that of RLSD/SDPO over GRPO.

Table 3: Wall-clock training time for 50 steps on Geo3k.

Method Time

GRPO 5h 58m SDPO 6h 14m RLSD 6h 15m CEPO 6h 34m

Baselines. We compare against four baselines under identical training budgets: GRPO [17], the sequence-level RL baseline; OPSD [26], which minimizes per-token KL divergence to a correctanswer teacher; SDPO [6], which extends OPSD with Jensen-Shannon divergence and EMA teacher stabilization; and RLSD [21], the direct predecessor of CEPO. All baselines use the same LoRA rank, group size, and training steps as CEPO. Other training hyperparameters are detailed in Appendix B.

1CEPO is not equivalent to a contrastive KL objective: the gradient of DKL(PT+∥PS) − DKL(PT−∥PS) produces a vocabulary-wide sum − v[PT+(v) − PT−(v)]∇θ log PS(v), structurally identical to OPSD’s leakage flaw (Eq. 3).

- Table 5: Feedback source ablation. Average accuracy on Qwen3-VL-2B after 50 Geo3k training steps; ∆ is relative to GRPO.

Positive r+ Negative r− Avg. Acc ∆GRPO GRPO baseline, no teacher 41.17±0.8 0.00

Peer rollout (prefix) Peer rollout (prefix) 40.47±0.8 −0.70 Peer rollout (suffix) Peer rollout (suffix) 40.60±0.8 −0.57 Peer rollout (full) Peer rollout (full) 41.99±0.8 +0.82 Ground truth answer Peer rollout (full) 42.74±0.8 +1.57 Ground truth answer Peer rollout (answer only) 43.43±0.8 +2.26

Evaluation. We report accuracy on five held-out multimodal mathematical reasoning benchmarks: DynaMath [29],LogicVista [19], MathVisionmini [18], MMMU [23], and WeMath [13]. All models are evaluated using lmms-eval [25] with sampling (temperature 1.0, top-p 1.0, top-k 40, presence penalty 2.0, maximum 32,000 tokens).

#### 5 Results

Table 2 reports results on both model scales. On Qwen3-VL-2B, CEPO achieves 43.43% average accuracy, compared to 41.17% for GRPO (+2.26pp), 34.96% for OPSD, and 35.70% for SDPO. On Qwen3-VL-4B, CEPO achieves 60.56%, versus 57.43% for GRPO (+3.13pp) and 56.23% for OPSD. Gains are most pronounced on LogicVista (+6.18pp over GRPO on 4B) and MathVisionmini (+4.94pp over GRPO on 2B), benchmarks that reward fine-grained multi-step reasoning over short, pattern-matchable answers. MMMU, which is primarily a multiple-choice knowledge retrieval benchmark with limited reasoning chains, shows the smallest gain (+1.67pp on 2B), consistent with the expectation that CEPO’s contrastive signal provides less leverage when reasoning traces are short.

OPSD and SDPO degradation. A notable finding is that both OPSD and SDPO fall below the untrained base model on 2B (34.96% and 35.70% vs. 39.73%). This is consistent with the information leakage analysis in §3.2: as training progresses, the vocabulary-wide r-conditioned gradient deviation δ(θ;r+) dominates the benign signal, driving the model to encode spurious x → r+ correlations that degrade generalization. The same pattern appears at 4B (56.23% for OPSD vs. 58.36% base), confirming that the leakage pathology is not an artifact of model scale. CEPO avoids this entirely: its gradient contains no vocabulary-wide r-conditioned term by construction (Theorem 1(ii)).

##### 5.1 Ablations

Teacher source (Table 4). We compare three teacher sources: a fixed reference policy, a periodically synced teacher, and the actor policy itself. The actor-policy teacher performs best, reaching 43.43%, a +2.26pp improvement over GRPO. This indicates that, in our setting, the most useful teacher is the one aligned with the current on-policy rollout distribution, even if its token distribution remains close to the student. Crucially, sharing weights with the actor requires no separate parameter copy, reducing memory overhead. The fixed reference policy improves over GRPO but reaches only 42.18%, suggesting that a frozen teacher provides a useful but increasingly stale contrastive signal as the policy changes. Synchronizing the teacher with the actor every 25 steps improves performance to 42.74%, narrowing the gap to the actor-policy teacher by keeping the teacher fresher while still partially decoupling it from the student. Overall, these results suggest that teacher freshness and on-policy alignment are more important than maintaining a large teacher-student distribution gap for CEPO.

Table 4: Teacher source ablation. ∆ is relative to GRPO.

Teacher Source Avg. Acc ∆GRPO GRPO baseline 41.17±0.8 0.00

Reference policy 42.18±0.8 +1.01 Sync every 25 steps 42.74±0.8 +1.57 Actor policy 43.43±0.8 +2.26

Feedback source (Table 5). We ablate the construction of r+ and r− across five configurations. The main CEPO setting, ground truth final answer as r+ and peer answer only as r−, performs best

- at 43.43%, improving over GRPO by +2.26pp. Using the full peer rollout as the negative reference

(a) Constant schedule

(b) Linear decay schedule ( 0 = 1.0)

(c) Evidence clip bound

Averageaccuracy(%)

GRPO (41.17%)

43.0

43.0

43.0

CEPO

42.0

42.0

42.0

41.0

41.0

41.0

40.0

40.0

40.0

GRPO (41.17%)

GRPO (41.17%)

0.1 0.5 1.0

10 25 35 50

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Clip bound w

Fixed value

Warmup steps Twarm (decay to 0)

- Figure 3: Hyperparameter sensitivity averaged across 5 reasoning benchmarks). (a) Constant λ schedule: λ = 0.5 peaks at 41.40%, outperforming GRPO (41.17%); sustained high-λ training (λ = 1.0) introduces noise that offsets the credit-assignment benefit. (b) Linear-decay schedule from

λ0 = 1.0: a 25-step warmup matches the constant-λ peak (41.25%). (c) Evidence clip bound εw: performance peaks in [0.4,0.5] at 42.7% (+1.5pp over GRPO) and degrades at both extremes, small

εw collapses CEPO toward GRPO, large εw destabilizes the modulated advantage.

also improves performance, reaching 42.74%, while full peer rollout conditioning on both sides reaches 41.99%. Partial peer context performs worse. Prefix only and suffix only conditioning reach 40.47% and 40.60%, both below GRPO, suggesting that truncated reasoning traces provide a noisy contrastive signal. Overall, the strongest ablation result comes from using the verified final answer as the positive reference and a compact rejected answer as the negative reference.

Hyperparameter sensitivity (Figure 3). Evidence clip bound ϵw. Performance peaks at ϵw ∈ [0.4,0.5] and degrades toward both extremes. At ϵw = 0.1, the clip is too tight and the method effectively reduces to GRPO. At ϵw ≥ 0.8, unconstrained weights introduce variance that destabilizes advantage estimation. We recommend ϵw = 0.5 as the default. λ schedule. A constant λ = 0.5 and a 25-step linear decay both outperform GRPO, while λ = 1.0 (constant maximum) performs worse despite the highest integrated CEPO pressure (50 units vs. 25 for λ = 0.5). A 10-step fast decay achieves comparable performance to the 25-step schedule, suggesting that the benefit of contrastive credit assignment is front-loaded: the first 10–25 steps drive the bulk of the improvement. Extending the schedule beyond 25 steps introduces noise that offsets the signal.

##### 5.2 Analysis

Contrastive delta fractions. Figure 4 tracks the sign structure of CEPO’s raw contrastive delta ∆CEt = log PT+(yt) − log PT−(yt) before clipping or reweighting. A positive delta means the token is more supported by the correct-answer teacher, so CEPO amplifies credit on positive-advantage rollouts; a negative delta means the token is more supported by the wrong-answer teacher, so CEPO assigns stronger blame on negative-advantage rollouts. The positive-delta fraction rises during early updates, indicating CEPO increasingly identifies tokens supporting correct reasoning, while the negative-delta fraction declines, suggesting the model produces fewer tokens compatible with the rejected-answer teacher. This confirms CEPO’s intended behavior: training shifts evidence toward correct-answer support rather than uniformly increasing weights.

0.023

0.28

Positive delta

Negative delta

0.020

0.26

Negativedelta

Positivedelta

0.018

0.015

0.24

0.013

0.010

0.22

0.007

0.20

1 5 10 15 20 25

Training Step

Figure 4: Contrastive delta fractions during CEPO training. We track the fraction of tokens assigned positive versus negative contrastive evidence. Positive-delta mass increases early, while negative-delta mass decreases.

Token weight heatmap. Figure 5 compares token-level credit assignment between RLSD and CEPO on the same geometry trajectory. In the top (RLSD) panel, credit is spread broadly across fluent setup prose and connective tokens, with no strong concentration on decisive steps. In the bottom (CEPO) panel, the contrastive signal sharpens credit onto the critical algebraic derivation and final answer tokens, while suppressing filler to near-unity weight. The lower CEPO clip rate (49.5% vs. 71.3%) confirms that CEPO operates over a wider effective dynamic range than RLSD.

[Figure 2]

- 1

3

- 2

[Figure 3]

RLSD

[Figure 4]

[Figure 5]

CEPO

Figure 5: Token-level credit assignment on a parallelogram problem. Green/red/white denote high, low, and neutral token weights. Numbered regions illustrate three claims: ① RLSD over-credits fluent setup prose, while CEPO suppresses it; ② CEPO localizes blame to the misapplied angle-equality inference instead of diffusing penalties; ③ CEPO sharpens credit on the decisive algebraic derivation (x+4 = 3x−6, isolation steps, final answer). The lower CEPO clip rate (49.5% vs. 71.3%) indicates a wider effective dynamic range, consistent with Proposition 1.

This qualitative pattern is consistent with Proposition 1 and provides interpretable evidence that the contrastive denominator PT− successfully distinguishes decisive reasoning steps from fluent but uninformative tokens.

#### 6 Conclusion

We presented CEPO, a token-level credit assignment method for RLVR that replaces RLSD’s singlereference evidence ratio with a contrastive ratio between correct- and wrong-answer teachers drawn from rejected rollouts in the training batch. We proved this preserves all structural safety guarantees of RLSD (direction anchoring and leakage-free gradients) while strictly sharpening credit at decisive tokens and leaving filler unchanged (Theorem 1, Proposition 1). CEPO outperforms GRPO, OPSD, and SDPO across five multimodal mathematical reasoning benchmarks at 2B and 4B scale; the collapse of OPSD and SDPO below the untrained baseline confirms that structural safety is a practical prerequisite, not a theoretical nicety. These results are validated on Qwen3-VL trained on Geo3k, and extending CEPO to larger models, text-only reasoning, and code generation is a natural next step. We hope CEPO offers a principled and practical building block for the next generation of credit-aware RLVR training.

#### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [2] Meng Cao, Lei Shu, Lei Yu, Yun Zhu, Nevan Wichers, Yinxiao Liu, and Lei Meng. Enhancing reinforcement learning with dense rewards from language model critic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.
- [3] Ken Ding. Hdpo: Hybrid distillation policy optimization via privileged self-distillation. arXiv preprint arXiv:2603.23871, 2026.
- [4] Daya Guo et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [5] Yiran Guo, Lijie Xu, Jie Liu, Dan Ye, and Shuang Qiu. Segment policy optimization: Effective segment-level credit assignment in rl for large language models. arXiv preprint arXiv:2505.23564, 2025.
- [6] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.
- [7] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Refining credit assignment in rl training of llms, 2025.
- [8] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, 2023.
- [9] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.
- [10] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [11] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021.
- [12] Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models. arXiv preprint arXiv:2602.04942, 2026.
- [13] Runqi Qiao et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.
- [14] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 2023.
- [15] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [16] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146, 2024.

- [17] Zhihong Shao et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [18] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 2024.
- [19] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.
- [20] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [21] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026.
- [22] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [23] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024.
- [24] Chenchen Zhang. From reasoning to agentic: Credit assignment in reinforcement learning for large language models. arXiv preprint arXiv:2604.09459, 2026.
- [25] Kaichen Zhang et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, 2025.
- [26] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.
- [27] Yanli Zhao et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.
- [28] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, and Richong Zhang. EasyR1: An efficient, scalable, multi-modality RL training framework. https://github.com/hiyouga/EasyR1, 2025.
- [29] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.

# Appendix

#### A Proofs

- A.1 Proof of Theorem 1

- (i) Direction anchoring. Since exp(·) > 0, we have wtCE > 0 unconditionally. Because ϵw ∈ (0,1), both clip bounds 1 ± ϵw are positive, so clip(wtCE,1−ϵw,1+ϵw) > 0. For any λ ∈ [0,1]:

(1 − λ) + λ · clip(wtCE,1−ϵw,1+ϵw) > 0,

since it is a convex combination of 1 and a positive quantity. Therefore Aˆt = A · [positive], giving sign(Aˆt) = sign(A) unconditionally. □

- (ii) Leakage-free gradient. The stop-gradient on ∆CEt renders Aˆ(ti) θ-constant within each update step. Therefore:

∇θLCEPO = E

1 G i

1 |y(i)| t

Aˆ(ti) · ∇θ log πθ(yt(i) | x,y<t(i)) ,

where Aˆ(ti) is constant. The gradient acts only at the sampled token yt(i); no vocabulary-wide sum v∈V appears, and r+,r− enter only through the θ-constant scalar Aˆ(ti). □

- (iii) RLSD containment. When PT−(yt) = PS(yt) for all t: ∆CEt = log PT+(yt) − log PT−(yt) = log PT+(yt) − log PS(yt) = ∆RLSDt .

Hence wtCE = wtRLSD and AˆCEPOt = AˆRLSDt for all t. □

- A.2 Proof of Proposition 1 From Eqs. (7) and (4):

PT+(yt) PT−(yt)

PT+(yt) PS(yt)

wtCE A>0 =

, wtRLSD A>0 =

.

- Case 1 (A > 0). wtCE > wtRLSD iff PT+(yt)/PT−(yt) > PT+(yt)/PS(yt). Since PT+(yt) > 0, this reduces to 1/PT−(yt) > 1/PS(yt), i.e. PS(yt) > PT−(yt). Under the joint condition PT+(yt) ≥ PS(yt), RLSD already assigns above-baseline credit (wtRLSD ≥ 1) and CEPO strictly amplifies it. □
- Case 2 (A < 0). The weights are wtCE = PT−(yt)/PT+(yt) and wtRLSD = PS(yt)/PT+(yt). By the

same argument, wtCE > wtRLSD iff PT−(yt) > PS(yt). Under the joint condition PT+(yt) ≤ PS(yt), CEPO assigns strictly stronger blame to a token RLSD already penalizes. □

- Case 3 (filler). When PT+(yt) ≈ PT−(yt) ≈ PS(yt), all ratios are near 1, so ∆CEt ≈ 0 and wtCE ≈ wtRLSD ≈ 1. Neither method discriminates at informationally neutral positions. □

#### B Baseline Hyperparameter Details

All methods in our experiments share a common training infrastructure based on Qwen3-VL-{2B,4B}Instruct fine-tuned with LoRA via the EasyR1 [28] framework with FSDP and vLLM-accelerated rollout generation. Table 6 reports the shared infrastructure hyperparameters and Table 7 reports the method-specific hyperparameters.

- Table 6: Shared training hyperparameters. Identical across all five methods in our experimental setup.

Hyperparameter Value

Base models Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct Training dataset Geo3k (3,000 geometry problems) Training steps 50 Optimizer AdamW Learning rate 1 × 10−6 (5 × 10−6 for CEPO) LR schedule Cosine decay, 5-step linear warmup Batch size (prompts) 32 Rollout group size G 8 Rollout temperature 1.0 LoRA rank / α 16 / 32 LoRA dropout 0.0 Max sequence length 2,048 tokens

PPO clip bound high ϵhigh 0.28 PPO clip bound low ϵlow 0.20 KL penalty None Entropy regularization None Reward model Rule-based verifier (numeric match) Framework EasyR1 + FSDP + vLLM

- Table 7: Method-specific hyperparameters used in our reimplementation. “—” denotes parameters not applicable to a given method.

Hyperparameter GRPO OPSD SDPO RLSD CEPO

Distillation objective — DKL(PT+∥PS) DJS(PT+∥PS) — Distillation weight α — 1.0 1.0 — —

EMA teacher decay β — — 0.999 — EMA update frequency — — every step — Evidence weight λ0 — — — 0.5 0.5 λ decay schedule — — — linear → 0 linear → 0

Warmup steps Twarm — — — 25 25 Evidence clip ϵw — — — 0.5 0.5 Positive reference r+ — ground truth answer ground truth answer ground truth answer ground truth answer

Negative reference r− — — — — rejected rollout (answer only) Teacher source — actor EMA of actor actor actor Teacher sync frequency — every step EMA every 50 steps every step Stop-gradient on teacher — Yes Yes Yes Yes

Training prompts. The prompt used for training the student is as follows:

{{ problem }} Solve the problem step by step, keeping reasoning brief. Put ONLY the final answer inside \boxed{}.

The prompt for computing the log probabilities through the teacher is as follows:

{{ problem }} Solve the problem step by step, keeping reasoning brief. Put ONLY the final answer inside \boxed{}. Here is a sample answer: {{ ground_truth_answer }} {{ student_answer }}

