# arXiv:2604.14084v4[cs.LG]21May2026

## TIP: Token Importance in On-Policy Distillation

Yuanda Xu∗† Hejian Sang∗ Zhengze Zhou∗ Ran He∗ Zhipeng Wang Alborz Geramifard

### Abstract

On-policy knowledge distillation (OPD) trains a student on its own rollouts under token-level supervision from a teacher, but not all token positions matter equally, and existing views of token importance are incomplete. We ask: which tokens carry the most useful learning signal in OPD? Our answer is that informative tokens come from two regions: positions with high student entropy, and positions with low student entropy plus high teacher–student divergence—where the student is overconfident and wrong. Empirically, student entropy is an effective but structurally incomplete proxy. Retaining 50% of tokens with entropy-based sampling matches or exceeds all-token training while cutting end-to-end peak training memory by up to 47%; under more aggressive retention, memory savings reach up to 58%. But entropy alone misses a second important region. When we isolate low-entropy, high-divergence tokens, training on fewer than 10% of all tokens nearly matches full-token baselines, showing that overconfident tokens carry dense corrective signal despite being nearly invisible to entropy-only rules. We organize these findings with TIP(Token Importance in on-Policy distillation), a two-axis taxonomy over student entropy and teacher–student divergence that explains why entropy is useful yet structurally incomplete and motivates type-aware selection rules that combine uncertainty and disagreement. We validate this picture across three teacher–student pairs spanning Qwen3, Llama, and Qwen2.5 on MATH-500 and AIME 2024/2025, and on the DeepPlanning benchmark for long-horizon agentic planning, where Q3-only training with 20% of tokens surpasses full-token OPD. Our experiments are implemented by extending the open-source OPD repository https://github.com/HJSang/OPSD_OnPolicyDistillation, which provides the practical training base for reproducing this work and supports memoryefficient distillation of larger models under limited GPU budgets.

### 1 Introduction

Knowledge distillation transfers capability from a large teacher to a smaller student by training on the teacher’s output distributions [Hinton et al., 2015], and is a primary driver of the rapid growth in small-model capacity [Xiao et al., 2025]. In on-policy distillation (OPD), the student generates its own responses and learns from the teacher’s corrections at each token [Agarwal et al., 2024, Gu et al., 2025]. Since the student generates the context, token importance is a property of the student–teacher state at each position. This raises a direct question: which tokens carry the most useful learning signal?

Our central claim is simple. In OPD, informative tokens come from two regions of the token state space: (1) positions with high student entropy, where the student is uncertain and still forming its prediction; and (2) positions with low student entropy but high teacher–student divergence, where the student is confident but misaligned with the teacher. The first region is easy to detect with entropy alone: retaining 50% of tokens by entropy-based sampling already matches or improves all-token training while substantially reducing end-to-end training memory. The second is easy to miss: under more aggressive retention, entropy-only selection discards overconfident tokens—positions where the

∗Equal contribution. †Correspondence to yuanda@math.princeton.edu

Preprint.

Baseline (100%)

Entropy 50%

Entropy 20%

Soft-OR 50%

Soft-OR 20%

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### MATH-500

###### AIME'24

###### AIME'25

DeepPlanning

11.6 12.0

12.2 12.6 12.1 12.6 12.3

###### 18.5

70.0 69.2

67.6 69.2 67.2

17.5

10.8

16.8

| |
|---|

| |
|---|

15.3

15.3

9.7

8.8

Base.Ent.50%

Ent.20%SO50%SO20%

Base.Ent.50%Ent.20%SO50%SO20%

Base.Ent.50%Ent.20%SO50%SO20%

Base.Ent.50%

Ent.20%SO50%SO20%

- Figure 1: Cross-task summary: average accuracy by selection method. Each panel shows one benchmark; bar height is the mean accuracy (mean@16) averaged across three teacher–student pairs for mathematical reasoning (Qwen3-8B→4B, Llama-70B→8B, Qwen2.5-14B→1.5B) and across two teacher sizes (14B, 32B) for DeepPlanning. Methods: Base. = all-token OPD (100%); Ent. 50%/20% = entropy-based token selection at the stated retention ratio; SO 50%/20% = Soft-OR (Eq. 7) top-k selection. The dashed line marks the all-token baseline. Soft-OR consistently improves over entropy-only selection on the mathematical reasoning benchmarks and remains competitive on DeepPlanning, confirming that augmenting entropy with divergence recovers the Q3 blind spot without sacrificing Q1/Q2 coverage.

student is sharply peaked on a continuation that the teacher strongly disfavors—because their low entropy makes them indistinguishable from correctly solved tokens.

We organize this picture with TIP, a two-axis taxonomy crossing student entropy and teacher–student divergence into four quadrants (Section 4). Conceptually, entropy is an effective but structurally incomplete proxy: it must conflate “confident and correct” with “confident and wrong,” and a parameter-free Soft-OR score fixes this blind spot (Section 5). Experimentally, the combined score consistently improves over entropy-only selection on mathematical reasoning and remains competitive on agentic planning, where Q3-only selection is strongest (Section 7).

#### Contributions.

- 1. We propose TIP, a two-axis taxonomy that organizes token importance by student entropy and teacher–student divergence, requiring no verification labels and no extra computation beyond the standard OPD loss.
- 2. We show that entropy is an effective but structurally incomplete proxy and prove that any entropyonly score is structurally blind to overconfident tokens, while a parameter-free Soft-OR score fixes this blind spot (Propositions 1–2, Remark 1).
- 3. We validate the taxonomy across several datasets and model families, and show that Soft-OR consistently outperforms entropy-only selection on mathematical reasoning while remaining competitive on long-horizon agentic planning in DeepPlanning [Zhang et al., 2026], where Q3-only selection is strongest.

### 2 Related Work

Curriculum learning and importance sampling. The idea that not all training examples contribute equally dates to curriculum learning [Bengio et al., 2009] and self-paced learning [Kumar et al., 2010], which order or weight samples by difficulty. Importance sampling extends this to gradient estimation: Katharopoulos and Fleuret [2018] select mini-batch elements by gradient norm, and Ren et al. [2018] learn per-example weights via meta-gradients. These methods operate at the example level. Our work pushes the granularity to individual tokens within a sequence, where the relevant axes are student uncertainty and teacher–student disagreement rather than a scalar difficulty score.

Off-policy vs. on-policy distillation. Classical sequence-level KD [Kim and Rush, 2016] trains the student on teacher-generated sequences (off-policy). On-policy distillation [Agarwal et al., 2024, Gu et al., 2025] instead lets the student generate its own rollouts and applies teacher supervision token-by-token, avoiding the train–test distribution mismatch inherent in off-policy data. Sang et al. [2026] further show that on-policy reverse KL self-distillation can compress lengthy reasoning chains

into shorter ones. Distillation has proven effective across diverse settings—from pretraining to extreme compression to expanding reasoning capabilities beyond what RL alone achieves [Gu et al.,

- 2024, Xu et al., 2024, Yue et al., 2025]. Because token importance in OPD is determined by the student’s own distribution at each position, it cannot be pre-computed from teacher outputs—it must be assessed online. This makes the choice of which tokens to train on a fundamentally different problem from off-policy sample selection.

Response-level selection. Several methods operate at the sequence level: PACED [Xu et al., 2026b] utilizes a Beta-kernel weighting function based on the student model’s pass rate to adaptively prioritize samples of intermediate difficulty, filtering out both trivial and overly complex data to enhance distillation efficiency, and LION [Jiang et al., 2023] uses quality signals. These approaches select rollouts to train on but treat all tokens within a response uniformly. A complementary questionwhich we address—is which token within a response carry the most signal.

Token-level importance in distillation and RL. In RL, Wang et al. [2025c] showed that highentropy “forking tokens” drive most gradient signal, Cui et al. [2025] further revealed that the covariance between token log-probability and advantage drives entropy collapse during policy optimization, SPINE [Wu et al., 2025] extends this idea to test-time RL by updating only decisioncritical branch points with entropy-band regularization, and Xu et al. [2026a] identified overconfident errors as a critical failure mode. In distillation, AdaSwitch [Peng et al., 2025] switches between teacher and student guidance based on divergence, Entropy-Aware OPD [Jin et al., 2026] adapts the loss based on teacher entropy, SelecTKD [Huang et al., 2025] lets the teacher verify student-proposed tokens via a propose-and-verify procedure and masks or down-weights rejected positions, LeaF [Guo et al., 2025] uses gradient-guided comparison with a teacher to identify and prune confounding tokens during distillation, and AdaKD [Xie et al., 2026] combines a divergence-based token selector (LATF, top-r% by Hellinger distance) with token-level temperature scaling (IDTS). Beyond fine-tuning, EntroDrop [Wang et al., 2025a] shows that dropping low-entropy tokens during pretraining improves generalization under multi-epoch training, providing independent evidence that high-entropy positions carry most learning signal. EDIS [Zhu et al., 2026] further demonstrates that the temporal dynamics of token entropy—not just its magnitude—can diagnose correct vs. incorrect reasoning trajectories.

Several concurrent works also explore token-level weighting, pruning, or compression across distillation and fine-tuning [Wang et al., 2020, Tavor et al., 2026, Kim and Baek, 2026, Wang et al., 2025b]. Most closely related is AdaKD [Xie et al., 2026], whose LATF module performs hard top-r% selection by teacher–student Hellinger distance—a divergence-only view we ablate in Appendix B.2. Our Q3 specification is not a relabeling of “large-divergence tokens”: it is defined by the conjunction of low student entropy and high disagreement, and the two axes induce different selections (Table 3 vs. Table 8). The AdaKD ablation itself supports this: LATF alone yields only +0.04 average ROUGE-L on Qwen2-1.5B (their Table 3a), with AdaKD’s full gain coming from an orthogonal temperature-scaling module (IDTS); our budget-matched comparison (Appendix B.2) reaches the same conclusion. Beyond this, our work proves that any entropy-only rule is structurally blind to low-entropy, high-divergence tokens (Proposition 2), and proposes a parameter-free Soft-OR score that explicitly recovers this region, validated on mathematical reasoning and long-horizon agentic planning.

### 3 Setup

Let T denote a frozen teacher and Sθ a trainable student over vocabulary V . A prompt x ∼ D is drawn, the student generates a rollout y = (y1,...,ym) ∼ Sθ(· | x), and the teacher scores each position. The context at position t is ct = (x,y<t). The standard on-policy distillation loss is:

m

1 m

DKL(PS(· | ct)∥PT(· | ct)). (1) We characterize each token position by two quantities, both already computed during training: Student entropy.

L =

t=1

H PS(· | ct) log |V |

∈ [0,1]. (2)

ht =

Table 1: Token taxonomy. Classification by student entropy ht and teacher–student divergence δt.

Quadrant ht δt Learning role

- Q1: High entropy, high divergence High High Correct errors or consolidate fragile knowledge
- Q2: High entropy, low divergence High Low Stabilize underconfident predictions
- Q3: Overconfident Low High Break systematic confident biases
- Q4: Solved Low Low Negligible signal

Q3 blind spot Q1 + Q2 student-entropy-visible

disagrees

| ||Q3<br><br>Confident + disagrees Overconfident errors<br><br>missed by student-entropy-only|
|---|
| |Q1<br><br>Uncertain + disagrees Densest signal<br><br>student-entropy-visible| |
|---|---|---|---|---|
| |Q4<br><br>Confident + agrees Solved tokens| |Q2<br><br>Uncertain + agrees Stabilization signal| |
| | | | | |

Teacher-studentdivergencet

mid

agrees

confident mid uncertain

Student entropy ht

- Figure 2: TIP taxonomy as a two-axis map. Entropy determines whether the student is uncertain or confident; divergence determines whether the teacher agrees or disagrees. Q1 and Q2 are visible to entropy-based methods, while Q3 is the low-entropy blind spot that requires divergence to detect. High ht means the student is uncertain; low ht means it is confident. Teacher–student divergence.

δt = DKL(PS(· | ct)∥PT(· | ct)). (3)

High δt means the teacher disagrees with the student. This is the per-token loss itself—no extra computation.

These two quantities define the plane in which we study token importance. The empirical question of this paper is whether useful training signal concentrates in particular regions of the (ht,δt) plane.

### 4 TIP Taxonomy: A Two-Axis View of Token Importance

We organize token importance along two axes already computed during standard OPD training: student entropy ht and teacher–student divergence δt. Crossing them yields four quadrants (Table 1, Figure 2). The quadrants are highly imbalanced: Q4 accounts for roughly 40–47% of all tokens, Q1 and Q2 together make up 40–52%, and Q3 constitutes only 3–15% across model families and datasets in the experimental setup, yet carries disproportionate corrective signal (Section 7.3; Appendix B.5 gives representative token-level examples, especially for Q1 and Q3).

### 5 Theoretical Analysis

The taxonomy suggests three predictions: high-entropy tokens should carry substantial learning signal relative to solved tokens (Q1/Q2 ≫ Q4); entropy-only selection should miss a specific class of tokens (Q3); and adding divergence should recover them. We formalize these below and test each one experimentally in Section 7. Specifically, we prove: (1) an oracle token weight suppresses

solved tokens while allowing positive weight on both high-entropy and overconfident corrective tokens (Proposition 1); (2) entropy-only scores are structurally blind to Q3 (Proposition 2); and (3) augmenting entropy with divergence restores coverage of all informative quadrants (Remark 1).

#### 5.1 Oracle Token Weight

We want to identify which tokens most accelerate training. We formalize this as: what per-token weights {wt} minimize the expected loss after one gradient step?

Let gt = ∇θℓt be the per-token gradient, µ¯t = E[gt], and define ϕ¯t = ⟨∇L,µ¯t⟩ and M¯t = E[∥gt∥2]. Under β-smoothness and a token-separable approximation that neglects cross-token covariance terms

- (Appendix A.2), a weighted step gˆ = t wtgt satisfies the surrogate bound:

m

η2β 2

−η wt ϕ¯t +

E[L(θ − ηgˆ)] − L(θ) ≲

wt2 M¯t . (4)

t=1

- Proposition 1 (Oracle token weight). The bound is minimized at wt∗ = ϕ¯t/(ηβM¯t), with per-token descent ∆∗t = −ϕ¯2t/(2βM¯t).

2β 2 wt2M¯t

Indeed, the bound is separable across tokens, so each coordinate minimizes −ηwtϕ¯t + η

independently. Differentiating gives −ηϕ¯t + η2βwtM¯t = 0, hence wt∗ = ϕ¯t/(ηβM¯t). Substituting back yields ∆∗t = −ϕ¯2t/(2βM¯t).

This is an oracle quantity (it depends on the population gradient), but it gives a clear interpretation: informative tokens have gradients that align well with descent without excessive energy. Across the four quadrants:

- • Q1: Large ϕ¯t (strong correction under uncertainty) ⇒ high oracle weight when the gradient is not too noisy.
- • Q2: Moderate ϕ¯t (stabilizing underconfident predictions) ⇒ non-negligible oracle weight.
- • Q3: Positive, sometimes large ϕ¯t (real corrective signal despite low entropy) ⇒ non-negligible oracle weight that entropy-only rules miss.
- • Q4: Near-zero ϕ¯t ⇒ negligible wt∗.

Thus the robust conclusion is not a fixed total ordering among Q1, Q2, and Q3; it is the separation between informative regions with positive descent signal and Q4, whose signal is near zero.

#### 5.2 A Signal-to-Curvature View

The oracle bound above can be read alongside a simpler local diagnostic picture. For a single token, consider the forward-KL/cross-entropy diagnostic loss as a function of the student’s logits zt, with gradient gt = ∇zt

ℓt and Hessian Ht = ∇2ztℓt. A weighted step gives the second-order approximation ∆ℓt(wt) ≈ −ηwt∥gt∥2 +

η2wt2 2

gt⊤Htgt, (5) so the local optimum satisfies

wtlocal = ∥gt∥2 η g⊤ t Htgt

. (6)

Thus token importance is a signal-to-curvature tradeoff: the numerator measures first-order corrective signal, while the denominator measures the local second-order cost of moving on that token.

This view clarifies why entropy alone is incomplete. High-entropy tokens often have useful corrective signal, explaining why entropy is an effective but structurally incomplete proxy. But Q3 tokens can also have a large numerator because the teacher strongly disagrees with a confident student prediction, while their low-entropy student distribution yields small softmax curvature (Appendix A.1). Q4 tokens share the same low-curvature regime, but their teacher–student disagreement is small, so the numerator is near zero. The distinction between Q3 and Q4 is therefore a signal distinction, not an entropy distinction.

In practice, wt∗ is unavailable because it depends on population-level quantities. A natural proxy is student entropy ht, but any such score is structurally blind to Q3:

- Proposition 2 (Blind spot). Let wˆ(ht) = f(ht) be any non-decreasing score with f(0) = 0 (e.g.,

f(h) = h or f(h) = [h ≥ τ] ). Then Q3 tokens—which may have wt∗ > 0—receive wˆ(ht) ≈ 0. Entropy alone cannot distinguish “confident and correct” (Q4) from “confident and wrong” (Q3).

Appendix B.5 illustrates this concretely: Examples 1, 3, and 4 show Q3 tokens with ht < 0.4 that an entropy-only rule would discard, while Examples 2 and 5 show the contrasting high-entropy Q1 cases that entropy-based rules do capture.

Since divergence δt is already computed as part of the loss, the natural fix is a score that is nonzero whenever either axis is active. We define the Soft-OR score with min-max normalized inputs

hˆt,δˆt ∈ [0,1]:

st = hˆt + δˆt − hˆt · δˆt = 1 − (1 − hˆt)(1 − δˆt). (7)

This is parameter-free: st is nonzero whenever either entropy or divergence is nonzero, without a tuning coefficient.

Remark 1 (Soft-OR fixes the blind spot). For any Q3 token with hˆt ≈ 0 and δˆt > 0, the entropy proxy gives wˆ0(ht) ≈ 0 (Proposition 2), but st ≈ δˆt > 0. Simultaneously, Q4 tokens (hˆt ≈ 0, δˆt ≈ 0) remain suppressed: st ≈ 0. Q1 tokens retain the highest scores because both hˆt and δˆt are large (st ≈ 1). The Soft-OR score therefore preserves coverage of the high-entropy regions while recovering Q3 and suppressing Q4, without requiring ϕ¯t or M¯t.

Empirical predictions. Table 2 maps each theoretical result to its experimental test.

#### Table 2: Theoretical predictions and experimental tests.

Result Prediction Tested in

- Proposition 1 Q1/Q2 carry the most signal; removing Q4 improves efficiency Section 7.2
- Proposition 2 Entropy-only selection misses Q3 tokens Section 7.3 Remark 1 Combined score recovers Q3 and outperforms entropy-only Section 7.4

### 6 Method: Type-Aware Token Selection

Given a retention ratio ρ ∈ (0,1], we retain the top-ρ fraction of tokens by the Soft-OR score st = hˆt + δˆt − hˆt · δˆt (Equation 7):

T = TopK {st}mt=1, ⌊ρm⌋ . (8) The training loss is:

1 |T | t∈T

DKL(PS(· | ct)∥PT(· | ct)). (9)

LTIP =

Setting δˆt = 0 recovers entropy-only selection; including δˆt additionally promotes Q3 tokens. In practice, before computing st, we clip the top 2% of entropy values within each batch and then apply min-max normalization to obtain hˆt, which suppresses rare outliers and stabilizes token ranking across batches. The score is parameter-free and both ht and δt are already computed during standard distillation, so the only extra cost is this clipped min-max normalization and the top-k sort—O(mlog m) per rollout, negligible compared with forward and backward passes.

### 7 Experiments

We now validate each prediction of the taxonomy. Table 2 maps each theoretical result to its experimental test; we proceed from the strongest signal (high-entropy tokens, Section 7.2) to the blind spot (Q3, Section 7.3) to the combined score (Section 7.4).

#### 7.1 Experimental Setup

Models. Three teacher–student pairs across three model families for mathematical reasoning, plus one pair for agentic planning:

- Table 3: Entropy sampling across model pairs. Accuracy (%, mean@16 ± std). Sampling selects

tokens with probability pt ∝ ht. Peak Mem reports end-to-end training peak GPU memory. Bold marks the best per benchmark.

Model pair Benchmark 100% 50% 20% 10%

MATH-500 76.7 ± 0.7 78.6 ± 0.6 74.1 ± 0.9 70.8 ± 1.0

- AIME’24 21.9 ± 1.2 23.8 ± 1.3 22.5 ± 1.1 21.1 ± 1.2
- AIME’25 19.4 ± 1.1 20.7 ± 1.3 21.5 ± 1.2 19.2 ± 1.0

Qwen3-8B (GRPO) → 4B

MATH-500 71.0 ± 0.7 74.0 ± 0.8 73.6 ± 0.7 72.5 ± 0.8

- AIME’24 21.5 ± 1.1 25.3 ± 1.5 18.8 ± 1.3 20.6 ± 1.4
- AIME’25 4.9 ± 0.9 7.5 ± 1.1 10.0 ± 1.2 9.3 ± 1.0

Llama-70B → 8B

MATH-500 55.1 ± 0.9 54.9 ± 0.9 54.0 ± 0.9 55.1 ± 0.9

- AIME’24 2.4 ± 0.7 3.3 ± 1.4 4.6 ± 1.3 2.7 ± 1.1
- AIME’25 2.1 ± 0.9 1.0 ± 0.5 1.0 ± 0.6 1.3 ± 0.8

Qwen2.5-14B → 1.5B

Peak Mem (GB, Qwen3) 72.0 38.1 35.5 35.3 Peak Mem (GB, Llama) 41.6 32.1 31.8 31.9 Peak Mem (GB, Qwen2.5) 35.8 19.7 15.0 14.9

- • Qwen3 Small: Qwen3-8B (GRPO) → Qwen3-4B [Yang et al., 2025]
- • Llama: Llama-3.3-70B-Instruct → Llama-3.1-8B-Instruct [Grattafiori et al., 2024]
- • Qwen2.5: Qwen2.5-14B-Instruct-thinking → Qwen2.5-1.5B-Instruct [Qwen et al., 2025] (∼9× capacity gap, reasoning teacher)
- • Qwen3 Agentic: Qwen3-{14B, 32B} → Qwen3-1.7B [Yang et al., 2025] (all with thinking enabled, trained on agentic planning data)

Data and evaluation. For mathematical reasoning, training prompts are from DAPO [Yu et al.,

- 2025], with evaluation on MATH-500 [Hendrycks et al., 2021] (500 problems) and AIME 2024/2025 (30 each). For agentic planning, training data is from DeepPlanning [Zhang et al., 2026], a benchmark featuring multi-day travel and multi-product shopping tasks that require proactive information acquisition, local constrained reasoning, and global constrained optimization; the Qwen3 Agentic pair is trained for 15 epochs. All models are trained with AdamW, a cosine schedule, and reverse KL on student-generated rollouts (lr = 1 × 10−6 for Qwen3 and Qwen2.5; lr = 3 × 10−7 for Llama).

#### 7.2 High-Entropy Tokens (Q1/Q2)

We begin with the simplest test of the taxonomy: if Q1/Q2 tokens dominate learning signal while Q4 tokens are negligible, then selecting by student entropy should preserve most of the benefit of OPD.

- Table 3 (and Figure 3 in the Appendix) confirm this prediction. Across all three model pairs, retaining 50% of tokens with entropy-based sampling matches or outperforms the all-token baseline on most benchmarks, while total peak training memory drops substantially. For Qwen3 Small, MATH improves from 76.7 to 78.6; for Llama, from 71.0 to 74.0. This indicates that many low-entropy tokens are effectively solved (Q4) and mainly dilute the gradient; Appendix B.5 gives token-level intuition for the contrasting high-entropy cases that do carry corrective signal (Examples 2 and 5).

At the same time, entropy alone is incomplete. As the retention ratio becomes more aggressive, performance often drops below the full-token baseline, suggesting that useful signal remains in the discarded low-entropy region as Proposition 2 indicates. We test this hypothesis directly in the next section by isolating the low-entropy, high-divergence tokens that entropy-only selection discards.

#### 7.3 Overconfident Tokens (Q3)

We next test the blind-spot prediction directly by constructing the opposite of entropy-based selection: a selector that prioritizes tokens with low entropy but high teacher–student divergence—targeting the Q3 regime in the taxonomy.

Q3 selection procedure. We isolate overconfident tokens using a confidence-weighted divergence score:

- 1. Compute per-token forward KL: δtfwd = DKL(PT(· | ct)∥PS(· | ct)).

- Table 4: Training on Q3 (overconfident) tokens only. Accuracy (%, mean@16 ± std) when training exclusively on low-entropy, high-divergence tokens. Q3-only training with <10% of all tokens can nearly match the all-token baseline across model pairs. Peak Mem reports end-to-end training peak GPU memory.

Model pair Benchmark Baseline (100%) Q3 20% Q3 10%

MATH-500 76.7 ± 0.7 75.7 ± 0.8 76.1 ± 0.9

- AIME’24 21.9 ± 1.2 20.2 ± 1.1 21.5 ± 1.3
- AIME’25 19.4 ± 1.1 19.2 ± 1.1 17.1 ± 1.0

Qwen3-8B → 4B

MATH-500 71.0 ± 0.7 71.8 ± 1.0 70.4 ± 1.1

- AIME’24 21.5 ± 1.1 20.2 ± 1.2 20.8 ± 1.1
- AIME’25 4.9 ± 0.9 4.5 ± 0.7 4.1 ± 0.7

Llama-70B → 8B

MATH-500 55.1 ± 0.9 54.6 ± 0.9 54.1 ± 0.9

- AIME’24 2.4 ± 0.7 2.5 ± 0.9 3.3 ± 1.4
- AIME’25 2.1 ± 0.9 3.3 ± 1.4 0.8 ± 0.4

Qwen2.5-14B → 1.5B

Peak Mem (GB, Qwen3) 72.0 36.4 36.2 Peak Mem (GB, Llama) 41.6 32.5 32.4 Peak Mem (GB, Qwen2.5) 35.8 19.7 19.6

- 2. Compute per-token student entropy ht (Equation 2), clip at the 98th batch percentile ht ← min(ht,h(0.98)), and min-max normalize to [0,1]: hˆt = (ht − hmin)/(hmax − hmin).
- 3. Define confidence as conft = 1 − hˆt (low entropy ⇒ high confidence).
- 4. Compute the Q3 score: wtQ3 = δtfwd · conft.

Tokens with high wtQ3 are precisely the positions where the student is highly confident while the teacher strongly disagrees.

It is important to distinguish the roles of the two KL directions in this experiment. Forward KL is used solely to rank tokens for selection; the training loss on the selected subset remains the standard reverse KL of Equation 1, DKL(PS∥PT). The two choices serve different purposes. Why reverse KL for the loss: Reverse KL is mode-seeking—it drives the student to concentrate on the teacher’s dominant mode rather than spread mass across all modes—which is the standard and numerically stable objective for on-policy distillation; it is also already computed during the forward pass, so restricting the loss to a token subset introduces no new computation. Why forward KL for ranking: When the student is near-deterministic on v∗, reverse KL is dominated by the teacher probability assigned to the student’s chosen token and has limited sensitivity to teacher-preferred alternatives that the student nearly ignores. Forward KL, v PT(v)log(PT(v)/PS(v)), directly penalizes missing teacher mass and therefore produces a sharper ranking signal for Q3-type overconfidence. We do not use forward KL as the training loss because its mode-covering bias would encourage the student to spread probability over multiple teacher-supported continuations, whereas the reverse-KL OPD baseline is the objective being compared. The conceptual definition of Q3—low student entropy plus high teacher–student disagreement—does not depend on the KL direction; forward KL is an operational detector for this diagnostic Q3-only experiment.

- Table 4 confirms that the Q3 region carries real corrective signal. For Qwen3, training on only 5.7K overconfident tokens (<10% of all tokens) reaches 76.1 on MATH, versus 76.7 for the full-token baseline. For Qwen2.5, Q3-only training matches or exceeds the baseline on several benchmarks. These results validate the taxonomy’s prediction that Q3 tokens are informative despite having near-zero entropy. Appendix B.5 provides concrete examples: a student that repeats a generic variable instead of substituting a concrete value (Ex. 1), an arithmetic computation error (Ex. 3), and a confident variable-level misstep in a derivation (Ex. 4)—all near-zero entropy, all strongly corrected by the teacher.

- Q3 is not just “large divergence”. Q3 tokens are by construction high-δ, but the selector is

structurally different from a pure divergence view. Ranking by δt alone at a budget matched to Q3-only underperforms the all-token baseline and only catches up when given 5× more tokens

- (Appendix B.2, Table 8); it is the low-entropy conjunct that makes selection performant under tight budgets. Conversely, high student entropy is not a proxy for high divergence—the two axes induce different selections and different performance curves (Table 3 vs. Table 8), and entropy-only rules

- Table 5: Main results: Baseline vs. Entropy-only vs. Soft-OR. Accuracy (%, mean@16 ± std). Soft-OR uses st = hˆt +δˆt −hˆt ·δˆt (Eq. 7) with Top-K selection. Bold marks the best per benchmark.

Baseline Entropy-only Soft-OR Model pair Benchmark 100% 50% 20% 50% 20%

Qwen3-8B → 4B

MATH-500 76.7 ± 0.7 78.6 ± 0.6 74.1 ± 0.9 79.1 ± 0.8 77.6 ± 0.7 AIME’24 21.9 ± 1.2 23.8 ± 1.3 22.5 ± 1.1 25.7 ± 1.4 24.5 ± 1.2 AIME’25 19.4 ± 1.1 20.7 ± 1.3 21.5 ± 1.2 21.9 ± 1.2 23.2 ± 1.2

Llama-70B → 8B

MATH-500 71.0 ± 0.7 74.0 ± 0.8 73.6 ± 0.7 74.7 ± 1.0 74.2 ± 0.7 AIME’24 21.5 ± 1.1 25.3 ± 1.5 18.8 ± 1.3 26.0 ± 1.4 21.0 ± 1.5 AIME’25 4.9 ± 0.9 7.5 ± 1.1 10.0 ± 1.2 11.5 ± 1.1 10.9 ± 1.4

Qwen2.5-14B → 1.5B

MATH-500 55.1 ± 0.9 54.9 ± 0.9 54.0 ± 0.9 56.2 ± 1.2 55.8 ± 0.9 AIME’24 2.4 ± 0.7 3.3 ± 1.4 4.6 ± 1.3 3.8 ± 1.2 5.0 ± 1.3 AIME’25 2.1 ± 0.9 1.0 ± 0.5 1.0 ± 0.6 1.5 ± 0.7 1.8 ± 0.6

- Table 6: Top 50% vs. bottom 50% by Soft-OR score. Accuracy (%, mean@16 ± std). “Top” trains

on the highest-scoring half by st; “Bot.” trains on the lowest-scoring half. The bottom tokens carry substantially less signal.

Qwen3-8B → 4B Llama-70B → 8B Qwen2.5-14B → 1.5B Benchmark Top 50% Bot. 50% Top 50% Bot. 50% Top 50% Bot. 50% MATH-500 79.1 ± 0.8 72.3 ± 0.9 74.7 ± 1.0 67.4 ± 1.1 56.2 ± 1.2 50.3 ± 1.0

- AIME’24 25.7 ± 1.4 15.5 ± 1.3 26.0 ± 1.4 17.2 ± 1.3 3.8 ± 1.2 1.5 ± 0.7
- AIME’25 21.9 ± 1.2 12.8 ± 1.2 11.5 ± 1.1 2.9 ± 0.8 1.5 ± 0.7 0.8 ± 0.5

are provably blind to Q3 (Proposition 2). A related concern is that high-δ tokens already yield the largest per-token gradients and thus should dominate the update; but in full-token training the aggregate update sums over all positions—mostly low-/moderate-δ ones—and the operationally relevant question is which tokens are worth keeping under a budget, which Table 8 answers in favor of the low-entropy + high-divergence selector.

#### 7.4 Type-Aware Selection (TIP)

The prediction of Remark 1 is that combining entropy with divergence should outperform entropy-only selection by recovering Q3 without sacrificing Q1/Q2. Table 5 and Figure 1 present the comparison, and Appendix B.5 provides concrete token-level examples of the Q1/Q3 behaviors that the combined score is designed to retain.

Top vs. bottom tokens by Soft-OR score. A natural sanity check is the complementary experiment: instead of training on the top 50% tokens by Soft-OR score, train on the bottom 50%. If the taxonomy is correct, these tokens should be predominantly Q4 (solved) and carry negligible learning signal.

Teacher entropy is uninformative. Teacher entropy is nearly constant across positions and therefore provides little discriminative signal for token selection; details are deferred to Appendix B.1. The useful axes are the student’s state (ht) and the student–teacher gap (δt), not the teacher’s uncertainty.

#### 7.5 Beyond Mathematical Reasoning: Agentic Planning

The preceding experiments focus on mathematical reasoning. To test whether TIP generalizes beyond this domain, we apply it to the DeepPlanning benchmark [Zhang et al., 2026] (Section 7.1).

Setup. Using the Qwen3 Agentic pair described in Section 7.1, we train on 80% of the DeepPlanning Travel Planning tasks and evaluate on the remaining 20%. We report the average accuracy (Avg@16) over 16 samples. Following DeepPlanning, we score each plan by the fraction of personalized hard constraints satisfied; these scores are lower than commonsense scores because personalized requirements (e.g., budget limits, dietary restrictions) are more demanding.

- Table 7: Agentic planning on DeepPlanning (Qwen3-1.7B student, thinking-enabled, Avg@16 %). Left: reference and entropy-only methods. Right: Q3-only and Soft-OR. Q3-only 20% surpasses full-token OPD; Soft-OR matches or exceeds entropy-only.

Method Teacher 14B Teacher 32B

OPD, all tokens (100%) 11.7±0.07 12.8±0.07 + Entropy-only 50% 12.1±0.06 13.1±0.07 + Entropy-only 20% 11.6±0.07 12.7±0.06

Method Teacher 14B Teacher 32B OPD + Q3-only 20% 12.6±0.07 13.6±0.07 OPD + Soft-OR 50% 12.0±0.06 13.1±0.08 OPD + Soft-OR 20% 12.1±0.06 12.6±0.07

- Table 7 confirms that TIP generalizes to a fundamentally different domain, with entropy-based selection at 50% retention matching or exceeding full-token OPD (12.1 vs. 11.7 for 14B; 13.1 vs. 12.8 for 32B). The most striking finding is Q3: training on only 20% of overconfident tokens surpasses full-token OPD for both teachers (12.6 vs. 11.7; 13.6 vs. 12.8), confirming that entropy discards exactly the tokens with the densest corrective signal. This aligns with the structure of agentic tasks: a single wrong but confident commitment—booking a closed venue, violating a budget constraint—can invalidate an entire plan, making Q3 corrections especially concentrated. Appendix B.3 provides Best@16 results, confirming the same pattern.

- 8 Discussion and Conclusion

TIP establishes that token importance in OPD is governed by two axes—student entropy and teacher– student divergence—and that both are necessary. All three theoretical predictions (Propositions 1–2, Remark 1) are supported across three model families and two task domains (Tables 3–7), with the clearest result being the Q3 blind spot: entropy-only rules provably cannot distinguish “confident and correct” from “confident and wrong,” yet fewer than 10% of tokens selected precisely for overconfidence nearly matches full-training performance. This picture is sharpest on DeepPlanning, where Q3-only training with 20% of tokens surpasses full OPD (12.6 vs. 11.7 Avg@16), suggesting that the value of catching overconfident errors grows when more downstream computation depends on a single committed step. More broadly, the two-axis framing applies to other settings where on-policy token-level supervision is used, including process reward fine-tuning and speculative decoding.

Limitations. Our largest teacher is 70B and rollouts reach 16K tokens; whether the quadrant structure and Q3 concentration persist at trillion-parameter scale or with the extremely long rollouts typical of agentic tool-calling pipelines remains an open question.

### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations (ICLR), 2024. URL https: //arxiv.org/abs/2306.13649.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. ICML, 2009.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. The llama 3 herd of models. 2024. URL https://arxiv.org/abs/2407.21783.

Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. MiniPLM: Knowledge distillation for pre-training language models. arXiv preprint arXiv:2410.17215, 2024.

Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. MiniPLM: Knowledge distillation for pre-training language models. 2025. URL https://openreview.net/forum? id=tJHDw8XfeC.

Yiju Guo, Wenkai Yang, Zexu Sun, Ning Ding, Zhiyuan Liu, and Yankai Lin. Learning to focus: Causal attention distillation via gradient-guided token pruning. In Advances in Neural Information Processing Systems, volume 38, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. NeurIPS, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Haiduo Huang, Jiangcheng Song, Yadong Zhang, and Pengju Ren. SelecTKD: Selective tokenweighted knowledge distillation for LLMs. arXiv preprint arXiv:2510.24021, 2025.

Yuxin Jiang, Chunkit Chan, Mingyang Chen, and Wei Wang. Lion: Adversarial distillation of proprietary large language models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. URL https://arxiv.org/abs/2305.12870.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026.

Angelos Katharopoulos and François Fleuret. Not all samples are created equal: Deep learning with importance sampling. In International Conference on Machine Learning (ICML), 2018.

Minsang Kim and Seung Jun Baek. Explain in your own words: Improving reasoning via tokenselective dual knowledge distillation. arXiv preprint arXiv:2603.13260, 2026.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2016. URL https: //arxiv.org/abs/1606.07947.

M Pawan Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable models. NeurIPS, 2010.

Zhen Peng et al. Adaswitch: Adaptive switching between teacher and student for on-policy distillation. arXiv preprint, 2025.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. 2025. URL https://arxiv.org/abs/2412.15115.

Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust deep learning. In International Conference on Machine Learning (ICML), 2018.

Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. CRISP:

Compressed reasoning via iterative self-policy distillation. arXiv preprint arXiv:2603.05433, 2026. Almog Tavor, Itay Ebenspanger, Neil Cnaan, and Mor Geva. Rethinking selective knowledge

distillation. arXiv preprint arXiv:2602.01395, 2026.

Jiapeng Wang, Yiwen Hu, Yanzipeng Gao, Haoyu Wang, Shuo Wang, Hongyu Lu, Jiaxin Mao, Wayne Xin Zhao, Junyi Li, and Ji-Rong Wen. Entropy-guided token dropout: Training autoregressive language models with limited domain data. arXiv preprint arXiv:2512.23422, 2025a.

Shaobo Wang, Jiaming Wang, Jiajun Zhang, Cong Wang, Yue Min, Zichen Wen, Xingzhang Ren, Fei Huang, Huiqiang Jiang, Junyang Lin, Dayiheng Liu, and Linfeng Zhang. Winning the pruning gamble: A unified approach to joint sample and token pruning for efficient supervised fine-tuning. arXiv preprint arXiv:2509.23873, 2025b. URL https://arxiv.org/abs/2509.23873.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. Advances in Neural Information Processing Systems, 38, 2025c.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2002. 10957.

Jianghao Wu, Yasmeen George, Jin Ye, Yicheng Wu, Daniel F. Schmidt, and Jianfei Cai. SPINE: Token-selective test-time reinforcement learning with entropy-band regularization. arXiv preprint arXiv:2511.17938, 2025.

Chaojun Xiao, Jie Cai, Weilin Zhao, Guoyang Zeng, Biyuan Lin, Jie Zhou, Zhi Zheng, Xu Han, Zhiyuan Liu, and Maosong Sun. Densing law of LLMs. Nature Machine Intelligence, 2025.

Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, and Jiangning Zhang. LLM-oriented token-adaptive knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 34070–34078, 2026.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. Overconfident errors need stronger correction: Asymmetric confidence penalties for reinforcement learning. arXiv preprint arXiv:2602.21420, 2026a.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. PACED: Distillation and self-distillation at the frontier of student competence. arXiv preprint, 2026b.

Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, and Wanxiang Che. OneBit: Towards extremely low-bit large language models. In Advances in Neural Information Processing Systems, volume 37, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, WeiYing Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source llm reinforcement learning system. arXiv preprint arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems, volume 38, 2025. Oral.

Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv, and Junyang Lin. DeepPlanning: Benchmarking long-horizon agentic planning with verifiable constraints. arXiv preprint arXiv:2601.18137, 2026.

Chenghua Zhu, Siyan Wu, Xiangkang Zeng, Zishan Xu, Zhaolu Kang, Yifu Guo, Yuquan Lu, Junduan Huang, Guojing Zhou, et al. EDIS: Diagnosing LLM reasoning via entropy dynamics. arXiv preprint arXiv:2602.01288, 2026.

### A Supplementary Theory

#### A.1 Logit-Level Geometry of Forward KL

This appendix gives the local diagnostic calculation behind the signal-to-curvature view in Section 5; it is not a replacement for the reverse-KL OPD objective used for training. For a token position, write the student distribution as PS = softmax(z) and treat the teacher distribution PT as fixed. The forward KL is

|V |

PT(k) PS(k)

. (10)

DKL(PT∥PS) =

PT(k)log

k=1

The softmax Jacobian is

∂PS(k) ∂zj

= PS(k) {j = k} − P S(j) . (11) Therefore the gradient with respect to the student logits is

|V |

∂ ∂zj

DKL(PT∥PS) = −

k=1

PT(k) {j = k} − P S(j) = PS(j) − PT(j), (12)

or, in vector form, ∇zDKL(PT∥PS) = PS − PT. Differentiating once more gives the softmax Hessian

H = ∇2zDKL(PT∥PS) = diag(PS) − PSPS⊤. (13) Its trace is

|V |

|V |

PS(i)2. (14)

PS(i)(1 − PS(i)) = 1 −

tr(H) =

i=1

i=1

Thus the curvature term is small for near-deterministic student distributions and larger for diffuse student distributions. This trace is not Shannon entropy, but it is a curvature-side proxy for the spread of the student distribution.

For a weighted local update on token t, the second-order approximation is

η2wt2 2

gt⊤Htgt, (15) which is minimized at

∆ℓt(wt) ≈ −ηwt∥gt∥2 +

wtlocal = ∥gt∥2 η g⊤ t Htgt

. (16) Moreover, by the Rayleigh quotient,

gt⊤Htgt ≤ λmax(Ht)∥gt∥2 ≤ tr(Ht)∥gt∥2. (17) This bound should not be read as an entropy-only weighting rule: small curvature is valuable only when the numerator ∥gt∥2 is non-negligible. Q3 and Q4 can both have low entropy and low curvature, but Q3 has large teacher–student disagreement while Q4 does not.

#### A.2 Derivation of the Descent Bound

- Assumption 1 (Smoothness). L(θ′) ≤ L(θ) + ⟨∇L(θ),θ′ − θ⟩ + β2∥θ′ − θ∥2.

- Assumption 2 (Token-separable approximation). For tractability, we neglect off-diagonal gradient interactions across token positions. Concretely, for t ̸= s we treat the centered cross-token covariance

E[(gt − µ¯t)(gs − µ¯s)⊤]

as lower-order, so that the quadratic term admits a token-separable approximation.

Derivation. Expand L(θ − ηgˆ) via smoothness where gˆ = t wtgt:

η2β 2 ∥gˆ∥2. (18)

L(θ − ηgˆ) ≤ L(θ) − η⟨∇L,gˆ⟩ +

Using linearity of expectation and gˆ = t wtgt,

wt ϕ¯t. (19) For the quadratic term,

E[−η⟨∇L,gˆ⟩] = −η

wt E[⟨∇L,gt⟩] = −η

wt ⟨∇L,µ¯t⟩ = −η

t

t

t

2

wt2E[∥gt∥2] +

E[∥gˆ∥2] = E

wtws E[⟨gt,gs⟩]. (20)

=

wtgt

t

t

t̸=s

Write gt = µ¯t + (gt − µ¯t). Under Assumption 2, we drop the off-diagonal covariance contribution and treat the remaining mean interaction terms as lower-order; absorbing these into the ≲ notation gives the token-separable approximation

E[∥gˆ∥2] ≲

wt2M¯t. (21) Combining the two displays gives

t

η2β 2

−ηwtϕ¯t +

E[L(θ − ηgˆ)] − L(θ) ≲

wt2M¯t . (22) The right-hand side is separable in t, so minimizing each term gives

t

η2β 2

∂ ∂wt −ηwtϕ¯t +

wt2M¯t = −ηϕ¯t + η2βwtM¯t = 0, (23) which yields wt∗ = ϕ¯t/(ηβM¯t).

| |
|---|

- A.3 Entropy-Weighted Sampling: Coverage and Variance Sampling tokens with pt ∝ ht and using the importance-weighted estimator gˆIS = m1 t∈S gt/pt is unbiased whenever pt > 0, with variance

Var(ˆgIS) =

1 m2

m

t=1

1 − pt pt

E[∥gt∥2]. (24)

This makes the tradeoff transparent: entropy sampling preserves nonzero Q3 coverage, but the variance cost grows as 1/pt for low-entropy tokens.

- A.4 Why Adding Divergence Improves the Ranking

Under entropy-only scoring wˆ0 = ht, Q3 and Q4 are both pushed to the low-score end. The Soft-OR score st = hˆt+δˆt−hˆt·δˆt separates them: Q3 receives positive score from divergence (st ≈ δˆt) while

- Q4 remains near zero (both axes small). Because the product term hˆt · δˆt prevents double-counting, the high-entropy ranking is preserved while Q3 is separated from Q4 without any tuning parameter.

### B Supplementary Experiments

MATH-500

AIME'24

AIME'25

25

20

75

20

Accuracy(%)

Accuracy(%)

Accuracy(%)

15

70

Qwen3-8B 4B

15

Llama-70B 8B

65

10

Qwen2.5-14B 1.5B

10

60

5

5

55

0

1.0 0.5 0.2 0.1

1.0 0.5 0.2 0.1

1.0 0.5 0.2 0.1

Retention Ratio

Retention Ratio

Retention Ratio

- Figure 3: Entropy sampling across retention ratios. Accuracy (mean@16) on three benchmarks as a function of retention ratio. Retaining 50% of tokens with entropy-based sampling matches or outperforms the all-token baseline across model pairs. At very low retention, entropy-only selection begins to plateau or degrade.

#### B.1 Teacher Entropy Is Uninformative

Teacher entropy is near-zero everywhere (mean 0.031, std 0.055 for Qwen3; mean 0.067, std 0.164 for Llama; median token probability ≥0.79), so any scheme that conditions on teacher entropywhether for token selection, loss weighting, or sampling—receives an almost constant input and therefore adds little discriminative information. In practice, the useful axes are the student’s state (ht) and the student–teacher gap (δt), not the teacher’s uncertainty.

#### B.2 Isolating the Entropy Axis: Comparison with Divergence-Only Selection

Section 7.3 shows that low-entropy, high-divergence (Q3) tokens carry dense corrective signal. A natural follow-up question is whether the low-entropy conjunct matters at all, or whether ranking by divergence δt alone—ignoring student entropy—would suffice. This is also the most direct comparison to the LATF module of AdaKD [Xie et al., 2026], which selects the top-r% of tokens by teacher–student Hellinger distance: our Div-only top-k selector is the same idea instantiated with reverse KL. We deliberately give Div-only a much larger budget (top 50%) than Q3-only (top 10%): if the entropy axis were redundant, Div-only with 5× more tokens should clearly dominate.

External reference point. The published AdaKD ablation provides an independent data point for divergence-only selection. On Qwen2-7B→1.5B (instruction-following, RKD baseline), their Table 3a reports that adding LATF alone changes ROUGE-L average from 37.03 to 37.07 (+0.04); the full +1.98 improvement of AdaKD comes from their orthogonal temperature-scaling module (IDTS), not from the token selector. This is consistent with our Proposition 2 and motivates the present ablation in our setting.

Setup. On Qwen3-8B (GRPO) → 4B with the standard mathematical-reasoning protocol (Section 7.1), we compare:

- • Baseline: all tokens (100%).
- • Div-only 50% (LATF-style hard top-k): top 50% of tokens by δt alone—no entropy term.
- • Div-only 10%: top 10% of tokens by δt alone—budget-matched to Q3-only.
- • Q3-only 10%: top 10% of tokens by δtfwd · (1 − hˆt) (Section 7.3)—explicitly requires both low entropy and high divergence.

This design isolates two questions: (i) at 5× fewer tokens, does adding the low-entropy filter beat divergence alone (Div-only 50% vs Q3-only 10%)? and (ii) at equal budget, does the low-entropy filter beat pure divergence ranking (Div-only 10% vs Q3-only 10%)?

- Table 8: Isolating the entropy axis: Div-only vs Q3-only on Qwen3-8B (GRPO) → 4B (mean@16 %, ± std). Div-only is a hard top-k instantiation of LATF-style divergence selection [Xie et al., 2026]. We compare Q3-only at 10% against Div-only at both 5× the budget (50%) and matched budget (10%). Bold marks the best per benchmark.

Baseline Div-only Div-only Q3-only Benchmark 100% 50% 10% 10%

MATH-500 76.7 ± 0.7 76.3 ± 0.8 74.3 ± 0.8 76.1 ± 0.9

- AIME’24 21.9 ± 1.2 21.5 ± 1.1 19.2 ± 1.2 21.5 ± 1.3
- AIME’25 19.4 ± 1.1 21.1 ± 1.2 15.3 ± 1.1 17.1 ± 1.0

Accuracy

Results. Table 8 resolves both questions and sharpens what the entropy axis contributes.

- (i) Equal budget (Div-only 10% vs Q3-only 10%). At the same 10% retention, Q3-only beats Div-only on every benchmark: MATH-500 76.1 vs 74.3 (+1.8), AIME’24 21.5 vs 19.2 (+2.3), AIME’25 17.1 vs 15.3 (+1.8). Div-only at 10% drops well below the all-token baseline (−2.4,−2.7,−4.1), while Q3-only stays within 0.6 of the baseline on MATH/AIME’24. With the divergence ranking and

budget held fixed, simply requiring tokens to also be low-entropy—i.e., re-weighting by (1 − hˆt)—is what closes the gap to the baseline.

- (ii) 5× budget gap (Div-only 50% vs Q3-only 10%). Once Div-only is given 5× the tokens (50%), it largely catches up: MATH-500 76.3 vs Q3-only’s 76.1 (+0.2), AIME’24 a tie at 21.5, AIME’25

21.1 vs 17.1 (+4.0 in Div-only’s favor). This is consistent with the fact that, as r grows, the highdivergence tail necessarily includes most Q3 tokens as a side effect. The headline is per-token efficiency: Q3-only matches Div-only on two of three benchmarks while using 5× fewer tokens.

What this comparison does and does not say. We are not claiming that Q3 is numerically disjoint from a “large-divergence” view. Q3 is, by construction, a high-δ region; under a sharp teacher distribution it can also overlap with high-entropy student tokens. The point is structural rather than set-theoretic. Prior work on token-level importance has mostly interpreted “which tokens matter” through the lens of student uncertainty—high entropy is taken as the proxy for an informative position—and entropy-based rules are provably blind to confident-but-wrong tokens (Proposition 2). What our taxonomy adds is a more fine-grained decomposition: informative tokens do not come from a single homogeneous source, and in particular there is a distinct low-entropy, high-disagreement region (Q3) that entropy-based views fail to isolate. Table 8 makes this concrete from two directions. Large-divergence ranking on its own is not sufficient: Div-only at a budget matched to Q3-only is consistently weaker than the joint score, and only recovers when given a 5× larger budget—i.e., when it also sweeps up much of the low-entropy tail. Conversely, large student entropy is not the same selector as large divergence: high-h tokens need not be high-δ, and—as the entropy sweeps in Table 3 show—the two selectors trace different performance curves across retention ratios. The “low entropy + high disagreement” specification is therefore not a redundant restatement of either axis; it is a separate, structurally identifiable region.

Why “large gradient” is not enough. A natural objection is that high-divergence tokens already produce the largest pointwise gradients, so they must be the dominant source of useful signal in the full-token recipe. That intuition does not survive scrutiny under a budget. In standard fulltoken training, updates are taken over all token positions; the vast majority typically have low or moderate divergence, with a much smaller subset of high-divergence positions. A larger gradient on an individual high-divergence token does not by itself imply that the accumulated contribution from such tokens dominates the overall update, nor that those gradients carry the most useful signal for optimization. More importantly, the question addressed here is not whether a token has a large gradient, but whether that gradient is informative and worth keeping under a limited budget. The Div-only 10% column is exactly this stress test: when forced to commit the entire budget to the highest-divergence tail, the resulting model underperforms even the all-token baseline. The structured comparison—separating large-divergence-only from low-entropy-and-large-divergence—turns the coarse intuition that “large-divergence tokens matter” into a more specific and testable claim about which structurally distinct token types are actually informative under budgeted training, and identifies Q3 as the region where this distinction matters most.

Connecting back to AdaKD. The Div-only 50% column is the within-our-setup analogue of AdaKD’s published LATF-only ablation [Xie et al., 2026]: a hard top-r% divergence selector with no entropy interaction. Their Table 3a reports +0.04 ROUGE-L for LATF alone on Qwen2-1.5B; we observe a similarly small effect on average (−0.4 on MATH-500, −0.4 on AIME’24, +1.7 on AIME’25 relative to the all-token baseline). The takeaway is the same in both settings: divergence ranking on its own is roughly baseline-equivalent at large budgets and clearly worse at small budgets, while a low-entropy + high-divergence selector is competitive at both—consistent with Proposition 2 and Remark 1.

#### B.3 Agentic Planning: Held-Out Queries

[Figure 1]

Figure 4: Token selection for agentic OPD on 20% held-out travel-planning queries. Top row: Avg@16; Bottom row: Best@16 (Pass@16). Within each row the left panel uses the 14B teacher and the right panel uses the 32B teacher. Q3-only 20% matches or exceeds the full-token baseline in every setting, consistent with Table 7. Best@16 results show the same pattern: overconfident-token training improves the upper tail of performance, not just the mean.

Figure 4 complements Table 7 with a finer-grained view. The Avg@16 panels confirm the maintext findings: Q3-only 20% leads for both teacher sizes (12.6 and 13.6 vs. baselines of 11.7 and 12.8), and entropy-only 50% improves over full-token OPD. The Best@16 (Pass@16) panels show the same pattern: Soft-OR 20% achieves the highest Best@16 with the 14B teacher (20.3 vs. 18.9 baseline), while Q3-only 20% leads with the 32B teacher (20.1 vs. 19.7). This indicates that correcting overconfident tokens improves not just average performance but also the upper tail.

#### B.4 Hyperparameters

- Table 9 summarizes all training hyperparameters.

Table 9: Training hyperparameters across model pairs. Qwen3 (8B→4B) Llama (70B→8B) Qwen2.5 (14B→1.5B)

Optimizer AdamW AdamW AdamW Learning rate 1 × 10−6 3 × 10−7 1 × 10−6 Batch size (rollouts) 8 8 8 Rollouts per prompt 16 16 16 Max response length 8192 8192 8192 Max prompt length 2048 2048 2048 OPD chunk size 512 512 512 OPD max length 16,384 16,384 16,384 Tensor parallel size 2 2 2 Generation temperature 1.0 1.0 1.0 Top-p (generation) 1.0 1.0 1.0 GPUs 8×H200 8×H200 4×H200

For entropy sampling experiments, tokens are sampled with probability pt ∝ ht, with the retention ratio determining how many tokens are kept. For Top-K experiments, the top-ρ fraction of tokens by score is selected deterministically. All evaluations use mean@16 (average accuracy over 16 independent samples per problem) with temperature 1.0.

#### B.5 Qualitative Examples Across Quadrants

We present five representative tokens from training on Qwen2.5-14B → 1.5B, spanning different quadrants of the taxonomy. Each example shows the problem, the student’s response with the

target token highlighted , and the student and teacher top-5 distributions. The entropy values reported in this appendix are raw entropies in nats.

#### Example 1: Generic variable vs. concrete substitution (Q3: low entropy, high divergence).

- Student entropy: 0.02 (extremely confident). Forward KL: 5.27. Overconfidence score: 5.24. Problem

Let P be a set of monic polynomials with integer coefficients of the least degree, with root k · cos 47π , as k spans over the positive integers. Let P(x) ∈ P be the polynomial so that |P(1)| is minimized. Find the remainder when P(2017) is divided by 1000.

Student response (excerpt)

...The polynomial with roots of the form k · cos 47π can be constructed by considering the minimal polynomial of cos 47π ...

Token distributions

Student top-5: k (99.8%), x (0.1%), e (0.0%), a (0.0%), z (0.0%) Teacher top-5: 2 (49.9%), k (30.3%), x (8.7%), e (6.8%), z (2.8%)

Analysis. The student assigns 99.8% to k, mechanically repeating the generic variable from the problem statement, while the teacher places 49.9% on the concrete value 2—indicating that the next reasoning step should substitute a specific integer rather than restate the variable. An entropy-only rule would assign near-zero weight to this position because ht = 0.02, yet it carries one of the densest corrective signals in the batch (overconfidence score = 5.24).

Example 2: Reasoning fork—restating vs. advancing (Q1: high entropy, high divergence).

- Student entropy: 1.82 (uncertain). Forward KL: 5.31. Problem

An arithmetic sequence of positive integers has n ≥ 3 terms, initial term a, and common difference d > 1. Carl wrote down all the terms correctly except for one term, which was off by 1. The sum of the terms he wrote was 222. What is a + d + n?

Student response (excerpt)

...Given ai = a + k where k = 1 if the term a + kd (for some integer k) was off and 1 if the term a + (k + 1)d was off....

Token distributions

Student top-5: off (54.4%), missing (15.6%), the (9.5%), not (4.5%), replaced (3.1%) Teacher top-5: written (40.3%), increased (35.6%), off (7.9%), decreased (2.9%), incremented (2.0%)

Analysis. The student favors “off” (54.4%), restating the problem, while the teacher prefers “written” (40.3%) or “increased” (35.6%)—words that advance the solution by characterizing the error direction. The teacher’s distribution pushes toward more precise mathematical reasoning, while the student’s choice leads to circular rephrasing. This is a classic Q1 token: the student is uncertain (entropy = 1.82) and the teacher strongly disagrees—entropy-based selection would catch this token.

- Example 3: Arithmetic computation error (Q3: low entropy, high divergence). Student entropy: 0.40. Forward KL: 3.54. Overconfidence score: 3.27.

Problem Find the sum of the real roots of f(x) = x4 + 9x3 + 18x2 + 18x + 4.

Student response (excerpt)

...f(2) = 24 + 9 · 23 + 18 · 22 + 18 · 2 + 4 = 16 + 72 + 72 + 36 + 4 = 19 0 ̸= 0...

Token distributions

Student top-5: 0 (91.2%), 8 (4.0%), 2 (3.5%), 4 (0.7%), 6 (0.5%) Teacher top-5: 6 (60.0%), 2 (25.0%), 8 (6.3%), 9 (3.8%), 0 (3.0%)

Analysis. The student confidently writes f(2) = 190 with 91.2% on digit 0, committing an arithmetic error in summing 16 + 72 + 72 + 36 + 4. The teacher distributes probability across 6 (60.0%) and 2 (25.0%), clearly disagreeing with 0. Like Example 1, this is a Q3 token: the student is fairly confident (ht = 0.40), yet the teacher strongly disagrees—entropy-based selection would under-weight this position.

- Example 4: Confident on the wrong variable (Q3: low entropy, high divergence). Student

- entropy: 0.12 (very confident). Forward KL: 5.58. Overconfidence score: 5.44. Problem

If the three interior angles A,B,C of triangle △ABC have cotangents cotA,cotB,cotC that form an arithmetic sequence, and the maximum value of angle B can be expressed as

mπ

n , find the value of m + n.

Student response (excerpt)

...Next, consider the cotangent of each angle cot B = sin

B

cosB , based on the arithmetic sequence property that cot B − cot A and cot B − cot C are equal...

Token distributions

Student top-5: B (98.2%), (B (1.0%), \’ (0.2%), ˆ (0.2%), C (0.1%) Teacher top-5: A (52.4%), (A (11.7%), C (11.7%), B (6.3%), ( (6.3%)

Analysis. The student assigns 98.2% to B, while the teacher prefers A (52.4%). The student writes cotB = cossinBB, which is mathematically incorrect (cotB = cossinBB); the teacher’s preferred continuation reflects a different and more correct reasoning path. Another clear Q3 token: ht = 0.12 would be invisible to entropy-only selection.

Note. The original rollout is in Chinese; the problem and response excerpts above are translated for readability. The token identifiers (B, A, etc.) are unchanged from the raw log, as mathematical symbols are language-invariant.

Example 5: Wrong mathematical symbol (Q1: moderate entropy, high divergence). Student

- entropy: 1.38. Forward KL: 4.27.

Problem (Same as Example 4)

Student response (excerpt)

...Assume the common difference between two adjacent terms is equal, i.e., b − a = \ Delta (where ∆ is a constant)...

##### Token distributions

Student top-5: text (53.5%), Delta (25.3%), frac (13.5%), lambda (1.4%), pm (1.0%) Teacher top-5: cot (91.7%), frac (4.6%), text (1.1%), delta (1.0%), Delta (0.8%)

Analysis. The teacher assigns 91.7% to cot—the mathematically relevant function for this problemwhile the student splits probability across irrelevant symbols (text, Delta, frac). This is a Q1 token (moderate entropy, high divergence): unlike Q3 tokens, entropy-based selection would catch it, but the teacher’s near-deterministic preference for cot makes it an especially high-value training signal. Note that Examples 4 and 5 come from the same student rollout at different token positions (same Chinese-language response as Ex. 4; translated above), illustrating how a single response can contain both Q3 and Q1 tokens.

Summary. Examples 1, 3, and 4 illustrate why divergence is needed: the student’s entropy is low, so any entropy-only rule would skip these tokens, yet the teacher strongly disagrees—a generic variable where a concrete value is needed (Ex. 1), an arithmetic error (Ex. 3), or a confident variable-level misstep in a derivation (Ex. 4). Examples 2 and 5 show Q1 tokens that entropy-based selection handles well—the student is already uncertain, and the teacher provides a clear corrective signal. The taxonomy’s value is precisely that it identifies both regions as informative while distinguishing them from Q4 (solved) tokens where both entropy and divergence are low.

