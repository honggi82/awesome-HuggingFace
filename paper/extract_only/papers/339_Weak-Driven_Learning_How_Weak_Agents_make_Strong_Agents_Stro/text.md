# arXiv:2602.08222v2[cs.AI]8Jun2026

## Weak-Driven Learning: How Weak Agents Make Strong Agents Stronger

Zehao Chen1,2,∗ Gongxun Li1,2,∗ Tianxiang Ai2,∗ Yifei Li1,2 Zixuan Huang1 Wang Zhou2 Tao Huang2 Fuzhen Zhuang1 Xianglong Liu1 Jianxin Li1 Deqing Wang1,† Yikun Ban1,† 1Beihang University 2China Telecom eSurfing Cloud ∗Equal contribution †Corresponding authors gyy_chenzehao@chinatelecom.cn zehaochenacid@buaa.edu.cn yikunb@buaa.edu.cn

### Abstract

As supervised fine-tuning (SFT) becomes central to post-training large language models, target-only training can yield diminishing returns while hard samples remain under-learned. While existing methods continue to reinforce target predictions, informative corrective signal can remain latent in models’ own historical weak states. Motivated by this observation, we study Weak-Driven Learning, a post-training direction that repurposes weak historical checkpoints as corrective signal for continued optimization. We instantiate this direction with WMSS (Weak Agents Can Make Strong Agents Stronger), a training framework that combines Weak-Driven Discrepancy Selection (WD-DS) with Weak-Driven Joint Training (WD-JT). By identifying recoverable learning gaps via entropy dynamics and reinforcing them through weak-driven logit mixing, WMSS provides one practical way to reuse historical weak states during training. Experiments on mathematical reasoning, code generation, and logical reasoning across four 3B–8B base models (Qwen3-4B/8B, Qwen2.5-3B, Gemma-3-4B-PT) show gains over competitive post-training baselines, while incurring zero additional deployment-time inference cost.

### 1 Introduction

Supervised Fine-Tuning (SFT) Ouyang et al. [2022], Touvron et al. [2023], Zou et al. [2025], Chen et al. [2025b] has become the standard post-training paradigm for improving large language models, translating pretrained capability into usable reasoning and instruction-following performance. Meanwhile, Knowledge Distillation (KD) Hinton et al. [2015], Gou et al. [2021], Agarwal et al. [2023] has emerged as a particularly effective approach for training small and medium-sized LLMs by matching the outputs or trajectories of stronger models. These imitation-based strategies are powerful when high-quality supervision is available, but such supervision is often tied to proprietary, expensive, or domain-specific teacher models that are not always accessible. This motivates post-training methods that can obtain useful learning signals without relying on an external stronger teacher.

A growing line of self-improvement methods follows this direction by constructing supervision from the model itself. However, in most such methods, the model-derived signal must first be converted into explicit supervision through additional computation, such as sampling self-generated responses for preference construction [Chen et al., 2024] or running auxiliary contexts to form teacher targets [Mitra and Ulukus, 2025]. Rather than generating new supervision, we look back at a source already created by ordinary training: the weak agents produced along the training trajectory. The intuition comes from human collaboration: a strong problem solver can improve by observing and correcting the mistakes of a weaker collaborator, because those mistakes reveal concrete errors that must be ruled out. We therefore ask: Can the weak agents already produced during standard training be reused directly as corrective signal?

Preprint.

[Figure 1]

Distillation-Based Learning

teacher signals

[Figure 2]

[Figure 3]

[Figure 4]

Teacher Student Improved Student

Weak-Driven Learning

corrective signals

[Figure 5]

[Figure 6]

[Figure 7]

Weak(Avg. 47.4) Strong (Avg. 61.0) Stronger (Avg. 69.1 ↑8.1)

Figure 1: Paradigm Comparison: Distillation-Based Learning vs. Weak-Driven Learning.

We answer this question by exploring the reverse direction of knowledge distillation and challenging the assumption that weaker models are useful only as students. We study this idea as Weak-Driven Learning and instantiate it as WMSS (Weak agents Make Strong agents Stronger). As illustrated in Figure 1, weak agents are not useful because their entire output distributions should be imitated. On routine tokens, weak and strong agents may behave similarly and provide little additional signal. Their value appears at key decision tokens, where the weak agent often remains uncertain or assigns mass to weak-revealed hard negatives: incorrect non-target tokens that mark unresolved boundaries. These token-level weak–strong discrepancies expose where the strong agent still needs to sharpen its decision boundary. WMSS therefore does not imitate the weak distribution; instead, it injects weak logits into a supervised mixed-logit objective where the ground-truth token remains the target, turning weak-agent mistakes into token-level corrective signal. Thus, otherwise discarded weak agents can become reusable corrective signal during training without deployment-time overhead.

Our concrete implementation is WMSS, a practical post-training framework built around a core mixed-logit mechanism and an auxiliary discrepancy-selection module. Weak-Driven Joint-Logit Training (WD-JT) forwards the weak and strong agents on the same batch, fuses their logits, and computes cross-entropy against the ground-truth token to optimize the training pair while retaining the strong branch as the output model. Because weak logits assign probability mass to weak-revealed hard negatives, this objective turns weak-model errors into token-level corrective signal for the strong branch. Weak-Driven Discrepancy Selection (WD-DS) then improves the efficiency of WD-JT by selecting where this weak–strong computation should be spent: it forwards both checkpoints on the training set, computes token-averaged entropy over supervised completion tokens, and resamples an active distribution using weak-model difficulty and weak–strong entropy changes. In this way, WD-DS prioritizes samples that are historically hard, brittle, or forgotten by the current strong model, where weak-model confusion is more likely to provide useful corrective signal for WD-JT. We further formalize weak-driven learning on hard samples and provide a conditional gradient-level analysis of the mixed-logit mechanism in this regime.

More broadly, our findings suggest that continued post-training need not rely exclusively on stronger external teachers or newly curated annotations: historical training states can serve as reusable supervision. Our contributions are summarized as follows:

- • Weak-driven post-training. We introduce Weak-Driven Learning, a new post-training direction in which weak historical checkpoints are used as corrective signal for strengthening a model on hard samples.
- • Training framework. We instantiate this direction with WMSS, a practical post-training framework that combines WD-DS for entropy-discrepancy sample selection and WD-JT for weak-driven logit mixing. Together, these two modules operationalize weak-driven learning without additional inference overhead.
- • Theoretical analysis. We provide a conditional gradient-level analysis of the mixed-logit mechanism, showing how incorporating weak-model logits can increase negative-token gradients under explicit margin assumptions.

- • Empirical performance. Using SFT as the anchor, WMSS improves Math-Avg by up to

+5.9 points and Code-Avg by up to +4.4 points across the reported backbones. Compared with the strongest baseline, WMSS nearly doubles the average gain over SFT on the main math/code settings, while incurring no additional deployment-time inference cost.

### 2 Related Work

Post-training: distillation and perturbation-based regularization. Post-training commonly starts from supervised fine-tuning (SFT) Ouyang et al. [2022], Touvron et al. [2023], Yang et al. [2026] and often uses knowledge distillation from a larger teacher Hinton et al. [2015], Agarwal et al. [2023], Gu et al. [2024] or smoothed self-predictions Zhang et al. [2019], Xu et al. [2023]. Such objectives transfer knowledge effectively, but can leave hard or high-discrepancy samples insufficiently corrected once aggregate SFT gains saturate. Perturbation regularizers such as NEFTune Jain et al. [2024b] instead inject noise for regularization, rather than using the model’s historical error patterns as corrective signals.

Self-improvement and weak-to-strong supervision. Another line of work improves models using weaker or alternate-context variants, including weak-to-strong generalisation Burns et al.

- [2023], Gulcehre et al. [2023], Somerstep et al. [2025], weak-supervision reasoning Yang et al.
- [2024], Yuan et al. [2026], preference optimisation Zhu et al. [2025], and self-improvement methods such as SPIN Chen et al. [2024], SSB Mitra and Ulukus [2025], and LightReasoner Wang et al.
- [2025]. Nearby work also studies zero-to-strong elicitation Liu et al. [2025], weak-to-strong test-time search Zhou et al. [2024], tool generalisation He et al. [2025], adaptive curricula Chen et al. [2025a], differential-entropy data selection Su et al. [2026], and capability drift Thede et al. [2026].

Difference from prior post-training methods. The above methods often rely on inference-time steering, rollout curricula, or selection-only entropy scores; in contrast, WMSS reuses historical model states as corrective signals during post-training and avoids deployment-time search.

### 3 Weak-Driven Learning

#### 3.1 Preliminaries

We consider an autoregressive language model Mθ that maps a context sequence x to logits zt ∈ R|V| over a vocabulary V, inducing Pθ(· | x) = Softmax(zt). Supervised fine-tuning. In standard supervised fine-tuning (SFT), we minimize the negative log-likelihood over a dataset D:

LSFT(θ) = −E(x,y)∼D[log Pθ(y | x)], (1) where y denotes the ground-truth next token under context x (we omit the time index when clear).

Predictive entropy. To quantify uncertainty and weak–strong discrepancy, we monitor the predictive entropy H(Pθ(· | x)) = − v∈V Pθ(v | x)log Pθ(v | x). Let ℓ(x,y;θ) = −log Pθ(y | x) be the per-token cross-entropy loss. Its gradient w.r.t. the logit of any token k ∈ V is ∂ℓ/∂zt[k] = Pθ(k | x) − I[k = y]. In particular, for any negative token k ̸= y, |∂ℓ/∂zt[k]| = Pθ(k | x), i.e., the gradient magnitude on a negative class is proportional to its assigned probability.

#### 3.2 Problem: Weak-Driven Learning

We define Weak-Driven Learning as a class of post-training paradigms in which the improvement of a strong agent is driven by its systematic discrepancies with a weaker agent, rather than by imitating a stronger teacher or directly imitating the weak agent. Formally, let D = {(xi,yi)}Ni=1 be a posttraining dataset, let Mstrong denote the current strong agent, and let Mweak denote a weaker agent, such as a historical checkpoint from the same training trajectory or a related lower-capability model. For an input x, let

ostrong(x) = Mstrong(x), oweak(x) = Mweak(x)

denote their outputs, which may take the form of logits, probabilities, hidden representations, generated responses, rewards, uncertainty estimates, or other model-derived signals.

The goal of weak-driven learning is to construct a discrepancy-aware training signal

SWD = Φ(ostrong(x),oweak(x),y), (x,y) ∈ D, (2) where Φ transforms weak–strong discrepancies into corrective supervision for the strong agent while preserving the task target y as the optimization anchor. The corresponding post-training objective can be written abstractly as

M+strong,Mweak = arg min

LWD (Mstrong,Mweak;SWD(x,y)), (3)

Mstrong, Mweak

(x,y)∈D

where both the strong and weak agents are optimized during post-training, but only the updated strong agent M+strong is retained for downstream use.

In this problem, the weak agent is not treated as a teacher to imitate. Instead, its errors, uncertainty, and disagreement with the strong agent reveal residual weaknesses, unstable predictions, or underoptimized regions of the strong agent. Thus, weak-driven learning converts weaker agents from discarded training artifacts into useful sources of corrective post-training signal.

### 4 Method: WMSS

WMSS is our practical instantiation of weak-driven learning. It consists of two modules: WeakDriven Joint-Logit Training (WD-JT), the core mechanism that converts weak-model confusion into gradient signal through logit mixing, and Weak-Driven Discrepancy Selection (WD-DS), an auxiliary module that uses weak–strong entropy discrepancies to identify hard or high-discrepancy samples for continued learning.

In this section, we describe how WMSS operationalizes weak-driven learning to further strengthen a strong agent.

Weak and strong agent initialization. In WMSS, the weak agent Mweak is initialized from a historical checkpoint and provides weak logits during WD-JT. We begin with Phase 1 (Initialization):

starting from a base model M0, we perform standard SFT to obtain M1, and set

Mweak ← M0, Mstrong ← M1. (4)

The weak agent provides a corrective signal via its logits zweak(x) ∈ R|V|, which preserve a softer decision boundary and highlight weak-revealed hard negatives, stabilising continued optimisation. For the iterative variant (K>1 in Algorithm 1), the weak agent for each outer iteration is initialized from the previous iteration’s output checkpoint.

#### 4.1 WD-JT: Weak-Driven Joint-Logit Training

The core weak-driven operation in WMSS is WD-JT: given a weak agent Mweak and a strong agent Mstrong, we optimize the paired agents through a weak-conditioned mixed-logit objective during training, converting weak-model confusion into additional optimization signal. The strong branch is retained as the final improved model. This module is the primary mechanism by which WMSS instantiates weak-driven learning.

Weak-conditioned logit mixing. For a training pair (x,y), let zstrong(x),zweak(x) ∈ R|V| denote the next-token logits produced by Mstrong and Mweak, respectively. We construct a mixed probability map by linearly mixing the logits:

zmix(x) = λzstrong(x) + (1 − λ)zweak(x), (5)

where λ ∈ [0,1]. We then optimize the training pair using the mixed-logit distribution Pmix(· | x) := Softmax(zmix(x)):

Lmix = −E(x,y)∼D [log Pmix(y | x)] (6)

This mixed-logit objective couples the weak agent’s predictive distribution with the paired training update. Rather than treating the weak agent as a teacher, WD-JT uses its output distribution as tokenlevel corrective signal. Because the weak agent retains relatively higher probability on weak-revealed hard negatives than the strong agent, the fused distribution exposes non-target tokens that standard SFT may have driven toward negligible probability. Computing ground-truth cross-entropy on this fused distribution converts these hard negatives into corrective signal for hard samples.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### 1 Initialization

###### 2 WD-DS

###### 3 WD-JT

###### 4 Output

SFT creates paired historical states

Pick samples for weak-logit training

Mix weak/strong logits with CE signal

Final strong model

H(M

;x

)

ΔH

###### =H

###### -H

Batch (x,y)

s

i

i

s

w

[Figure 12]

Dataset

current difficulty

entropy shift

z

D

weak

ACTIVE SAMPLING

[Figure 13]

p

∝α[-ΔH

]

+βH

+γ[ΔH

]

###### LOGIT MIXER

i

i

+

s,i

i

+

z

=λz

+ (1-λ)z

Weak

mix

strong

weak

α consolidate

β hard now

γ repair

weak harder before

strong uncertain

strong regressed

SFT

[Figure 14]

[Figure 15]

[Figure 16]

z

strong

skip easy/similar cases; keep hard or regressed ones

Final

###### CE(z

,y)

mix

###### M

+

###### Active set

strong

###### M

###### M

###### D

active

weak

strong

Strong

[Figure 17]

###### LOGIT-SPACE EVIDENCE

###### Correction Mechanism

[Figure 18]

[Figure 19]

###### Strong logits

###### Weak logits

###### Stronger logits

[Figure 20]

###### "Definitely A"

###### "Is it B?"

The stronger agent corrects the

The weak agent is

uncertain. corrective signals

weak agent’s error. Label

Label

Label

Weak

[Figure 21]

[Figure 22]

[Figure 23]

###### "It is A."

The strong agent is fairly confident.

Stronger A B C

A B C

A B C

Strong

increase the target-token score while suppressing error-prone tokens

scores spread across tokens

target score dominates

The strong agent becomes stronger by suppressingerrors exposed by the weak agent.

Weak logits expose hard negatives, and mixed-logit cross-entropy converts them into corrective signals that refine the strong model.

- Figure 2: Weak-Driven Learning instantiated by WMSS. Overview of the WMSS framework. It has three phases: (1) initialization, (2) Weak-Driven Discrepancy Selection (WD-DS), and (3) Weak-Driven Joint-Logit Training (WD-JT); the right panel visualizes the weak-driven mixed-logit principle through logit mixing and gradient amplification.

#### 4.2 WD-DS: Weak-Driven Discrepancy Selection

WD-DS is a lightweight selector for WD-JT. While WD-JT specifies how weak agents provide corrective signal through logit mixing, WD-DS decides where this weak–strong computation is most useful. It prioritizes samples that remain learnable and expose weak–strong discrepancies, and downplays samples already confidently handled by both agents or samples where the two models behave too similarly to reveal weak-model mistakes.

Entropy-based selection. For each supervised sequence example xi = (ui,yi), let Ti denote the target-token positions included in the training loss and let ci,t = (ui,yi,<t) be the prompt concatenated with the ground-truth target prefix before position t. We use the length-normalized predictive entropy

1 |Ti| t∈T

H(M;xi) = −

The weak–strong discrepancy is

PM(v | ci,t)log PM(v | ci,t). (7)

i v∈V

∆Hi = H(Mstrong;xi) − H(Mweak;xi). (8) WD-DS samples examples according to

pi ∝ α[−∆Hi]+ + βH(Mstrong;xi) + γ[∆Hi]+, (9)

where [u]+ := max(u,0) and pi is normalized over D. The three terms select consolidation samples, currently difficult samples, and recoverable regressions. Given pi, WD-DS draws |D| examples with replacement and removes duplicate sampled indices to form Dactive; WD-JT then trains on Dactive with the same mixed-logit objective in Eq. (6).

##### 4.3 Training Pipeline Algorithm 1 summarizes WMSS and the roles of its two modules.

- Phase 1 (Initialization) trains M0 with SFT to obtain Mstrong and initializes Mweak ← M0,Mstrong ← M1.
- Phase 2 (WD-DS) constructs an active training distribution by computing token-averaged sequence entropy between Mweak and the current Mstrong, yielding a selected active dataset Dactive via

Algorithm 1 Weak Agents make Strong Agents Stronger Require: Dataset D, Base Model M0 Require: Iterations K, Hyperparams α,β,γ,λ

- 1: Phase 1: Initialization
- 2: M1 ← Train(M0,D,LSFT)
- 3: Phase 2: Iterative Training Loop
- 4: for t = 1 to K do
- 5: // Step A: Weak-Driven Discrepancy Selection
- 6: Compute token-averaged sequence entropies as defined above
- 7: ∆H ← H(Mt) − H(Mt−1)
- 8: Calculate p via Eq. 9
- 9: Sample and deduplicate to form Dactive
- 10: // Step B: Weak-Driven Learning
- 11: Initialize (M(weakt) ,M(strongt) ) ← (Mt−1,Mt)
- 12: for batch (x,y) ∈ Dactive do
- 13: zweak ← Forward(M(weakt) ,x)
- 14: zstrong ← Forward(M(strongt) ,x)
- 15: zmix ← λzstrong + (1 − λ)zweak
- 16: Update (M(weakt) ,M(strongt) ) on CE(zmix,y)
- 17: end for
- 18: Mt+1 ← M(strongt)
- 19: end for
- 20: return MK+1

Eq. (9).

- Phase 3 (WD-JT) then performs weak-driven mixed-logit training on Dactive by mixing logits as in Eq. (5) and optimizing the mixed-logit cross-entropy loss in Eq. (6). This allows the training pair

to learn from weak-revealed hard negatives on hard samples, producing M+strong from the retained strong branch.

### 5 Mechanistic Analysis of Weak-Driven Learning

We next provide a conditional gradient-level explanation for why weak-driven logit mixing can help: mixing strong logits with a weak agent can amplify gradient signal on weak-revealed hard negatives under the margin conditions stated below. The analysis is local and mechanism-oriented: it explains one sufficient regime in which WD-JT increases probability mass on non-target tokens in H, thereby converting weak-agent confusion into corrective signal for the hard-sample regime targeted by WD-DS.

Negative-token gradient setup. Let ℓ(x,y;θ) = −log Pθ(y | x). Its gradient with respect to any logit is ∂ℓ/∂zt[k] = Pθ(k | x) − I[k = y], so for every negative token k ̸= y the gradient magnitude equals Pθ(k | x). On hard samples, target-only SFT can assign very small probability to weak-revealed hard negatives before they are fully resolved, leaving limited gradient signal on the corresponding negative tokens. WMSS exploits this by increasing probability mass on these hard negatives through logit mixing.

Consider a weak agent Mweak initialized from a historical checkpoint and a strong agent Mstrong that, for a context x, produce logits zweak(x),zstrong(x) ∈ R|V|, and the mixed logits

zmix(x) = (1 − λ)zweak(x) + λzstrong(x), λ ∈ [0,1]. (10)

Operationally, zmix(x) is the fused logit map used for the paired training update, injecting the weak agent’s uncertainty while preserving the strong agent’s target direction. Let y denote the target

index, ey the one-hot vector, and define Pmix(· | x) = Softmax(zmix(x)) and Pstrong(· | x) = Softmax(zstrong(x)). More generally, for any logit map z(x), let Pz(· | x) = Softmax(z(x)). The cross-entropy gradient on fused logits is

gmix = ∇zmix(x)L = Pmix(· | x) − ey, (11)

For any negative token k ̸= y, gmix[k] = Pmix(k | x), so increasing negative probability mass directly increases the gradient magnitude on that token. Standard SFT on the strong agent gives

gsft = Pstrong(· | x) − ey and provides little signal on a hard negative once Pstrong(k | x) is already small.

Margins and hard negatives. Define the target margin for any negative token k ̸= y as mk(z(x)) = z(x)[y] − z(x)[k]. Smaller margins mean higher confusion. Define the hard-negative set

H = {k ̸= y : mk(zweak(x)) < mk(zstrong(x))}, (12)

i.e., tokens where the weak agent is less separated than the strong agent, highlighting unresolved boundaries. The mixed margin is a convex combination mk(zmix(x)) = (1 − λ)mk(zweak(x)) + λmk(zstrong(x)), so mixing shrinks margins toward the weak agent on H and raises the relative probability of hard negatives.

Theorem 1 (Total negative-mass increase under uniform margin shrinkage). If mk(zweak(x)) ≤ mk(zstrong(x)) for all k ̸= y, then

Pmix(y | x) ≤ Pstrong(y | x),

Pstrong(k | x). (13)

Pmix(k | x) ≥

k̸=y

k̸=y

This sufficient condition shows that, when the weak agent is uniformly more uncertain than the strong agent, mixed-logit training shifts probability mass from the target to negative tokens. In that regime, gradient magnitude on negative classes can be amplified on hard samples.

Corollary 2 (Per-token amplification on hard negatives). For any k ∈ H, the mixed negative gradient satisfies

Pmix(k | x) ≥ Pstrong(k | x) whenever Pmix(y | x)

Pstrong(y | x) ≥ exp(−(1 − λ)∆mk), (14) where ∆mk = mk(zstrong(x)) − mk(zweak(x)) > 0.

For hard negatives, mixed-logit training increases their assigned probability—and thus their gradient magnitude—under a mild, explicit condition on the relative target probabilities.

Proposition 3 (Branch-level logit updates on negative and target tokens). Under a local diagonalkernel approximation to the mixed-logit paired update in Algorithm 1, for branch i ∈ {weak,strong} with sweak = 1 − λ and sstrong = λ,

∆zi,k ≈ −ηsi Pmix(k | x) (k ̸= y), ∆zi,y ≈ ηsi (1 − Pmix(y | x)). (15)

Any increase in Pmix(k | x) amplifies the local gradient signal on hard negatives in the paired objective, while a decrease in Pmix(y | x) strengthens the upward push on the target logit. The full parameter-space update passes through the Jacobian Gram matrix and is derived in Appendix D; the simplified form above states the token-level direction of the cross-entropy signal. These dynamics are consistent with the empirical reduction of the non-target logit mean observed in our logit statistics.

Mechanistic interpretation. The result explains the “why” of WD-JT: weak logits can expose weak-revealed hard negatives inside a supervised CE objective, and the resulting larger negativetoken probabilities produce stronger corrective updates against those tokens. Appendix D provides additional dynamics diagnostics for branch sensitivity, confident-regime shielding, and mean-logit drift, while Appendix D.6 discusses the validity of the margin conditions and multi-step dynamics.

### 6 Experiments

This section presents the setup, main comparisons, module ablations, convergence dynamics, and WD-DS step-reduction analysis. Additional diagnostics are provided in Appendix B.

Table 1: Main results. pass@1 accuracy (%) under greedy decoding. All methods are trained for two epochs; WMSS uses the same two-epoch total budget. Additional results on different domains (logic) and model families (Gemma) are reported in Appendix B.3 and Appendix B.2.

Math Code AIME2025 MATH500 AMC23 AQUA GSM8K MAWPS SVAMP Avg. HE+ MBPP+ BCB LCB Avg.

Methods

Qwen3-4B-Base as base model.

SFT 13.3 66.2 47.5 61.8 83.9 91.2 85.5 64.2 73.8 64.0 38.5 17.8 48.5 UNDIAL 13.3 63.2 40.0 55.1 82.9 90.8 84.1 61.3 ↓2.9 72.6 66.8 39.6 17.5 49.1 ↑0.6 NEFTune 16.7 68.2 42.5 59.1 86.7 93.7 87.7 64.9 ↑0.7 75.0 66.9 40.3 16.0 49.6 ↑1.1 SPIN 20.0 70.8 47.5 66.9 85.8 87.4 87.6 66.6 ↑2.4 76.2 66.7 39.1 18.8 50.2 ↑1.7 SSB 10.0 70.6 45.0 66.1 87.1 95.4 88.8 66.1 ↑1.9 78.0 67.4 42.5 18.0 51.5 ↑3.0 WMSS 20.0 71.4 50.0 67.7 88.5 96.2 90.3 69.1 ↑4.9 78.7 68.2 43.5 21.0 52.9 ↑4.4

Qwen3-8B-Base as base model.

SFT 16.7 72.2 45.0 63.0 87.5 95.0 88.6 66.9 80.5 69.2 44.5 21.0 53.8 UNDIAL 10.0 71.6 47.5 66.9 89.9 96.6 91.2 67.6 ↑0.7 79.9 69.4 45.7 21.5 54.1 ↑0.3 NEFTune 16.7 73.8 40.0 71.3 90.3 97.5 91.5 68.7 ↑1.8 81.1 70.4 45.9 21.0 54.6 ↑0.8 SPIN 16.7 72.4 47.5 78.3 88.8 95.4 92.7 70.3 ↑3.4 81.7 70.0 46.3 21.8 55.0 ↑1.2 SSB 16.7 74.6 52.5 76.4 89.3 93.3 93.0 70.8 ↑3.9 81.1 66.9 46.0 20.8 53.7 ↓0.1 WMSS 20.0 75.4 52.5 77.2 92.5 97.9 94.0 72.8 ↑5.9 84.1 71.4 47.1 22.3 56.2 ↑2.4

Qwen2.5-3B as base model.

SFT 3.3 47.0 20.0 51.2 76.6 91.6 84.4 53.4 58.5 59.0 27.5 11.0 39.0 UNDIAL 3.3 50.8 17.5 49.2 77.4 91.2 83.2 53.2 ↓0.2 56.1 59.5 28.9 12.5 39.3 ↑0.3 NEFTune 3.3 51.2 22.5 47.2 78.1 90.3 84.7 53.9 ↑0.5 56.1 59.5 29.6 11.5 39.2 ↑0.2 SPIN 3.3 51.6 20.0 49.6 77.4 91.2 86.1 54.1 ↑0.7 59.8 60.6 31.4 12.0 41.0 ↑2.0 SSB 3.3 51.4 22.5 52.0 79.9 92.0 84.9 55.1 ↑1.7 54.3 52.9 30.4 9.3 36.7 ↓2.3 WMSS 6.7 52.0 22.5 53.9 80.9 92.4 86.0 56.3 ↑2.9 62.2 61.9 32.7 12.5 42.3 ↑3.3

#### 6.1 Experimental setup

We evaluate WMSS on mathematical reasoning, code generation, and logical reasoning against standard SFT, UNDIAL [Dong et al., 2024], NEFTune [Jain et al., 2024b], SPIN [Chen et al., 2024], and SSB [Mitra and Ulukus, 2025]. All compared models are trained for the same twoepoch budget. Because WMSS is an offline/off-policy post-training method, we focus the main comparison on offline or off-policy baselines for a fair comparison. These methods use fixed supervision, alternate-context targets, or responses generated before the current update, rather than on-policy RL rollouts whose results depend strongly on online sampling and reward-model budgets. More details are provided in Appendix E.

#### 6.2 Main results

- (1) WMSS gives a larger gain over the same two-epoch SFT baseline. We use the two-epoch SFT row as the reference and report improvements over it. On the three Qwen math blocks in

- Table 1, non-WMSS post-training baselines improve Math Avg by +1.2 points on average over SFT, whereas WMSS improves by +4.6 points on average, more than 3× the mean competing gain. The same comparison is visible per backbone: Qwen3-4B-Base improves from 64.2% to 69.1% (+4.9), Qwen3-8B-Base from 66.9% to 72.8% (+5.9), and Qwen2.5-3B from 53.4% to 56.3% (+2.9). Code and logic show the same direction: Qwen3-4B-Base gains +4.4 Code Avg and +4.0 Logic Avg, while Qwen3-8B-Base gains +2.4 Code Avg and +7.4 Logic Avg. Per-base logic results (LogiQA 2.0, ReClor) are tabulated in Table 5 of Appendix B.3.

- (2) WMSS extends across the tested 3B–8B backbones and Qwen / Gemma families. The Math-Avg gain over the two-epoch SFT baseline is positive on every tested base: Qwen3-4B-Base

+4.9, Qwen3-8B-Base +5.9, and Qwen2.5-3B +2.9 (Table 1); Gemma-3-4B-PT +4.4 (Table 4 of Appendix B.2). In these single-run comparisons, WMSS obtains the strongest Math Avg among the listed baselines, suggesting that weak-driven logit mixing continues to surface useful corrective signal within this scale range. Appendix B.5 also identifies a practical condition: the weak agent should retain sufficient entropy separation from the strong agent, since too-close pairings collapse toward averaging similar distributions.

|w/ WD-DS| | | | |
|---|---|---|---|---|
|w/o WD-DS| | | | |
|SFT| | | | |
| | | | | |
| | | | | |
| | | | | |

79.5

100

Avg.acc.

80

77.5

###### Accuracy(%)

60

75.5

40

73.5

20

AIME2025 (Hard)

GSM8K SVAMP MAWPS (Easy)

0 50 123 218 steps

MATH500

Convergence Zone

amc2023

AQUA

0

0 1 2 3 4

Training Epochs

- Figure 3: Convergence of WMSS over four epochs. The trajectory is reported as an extended diagnostic beyond the fixed two-epoch budget used for the main comparison.

Figure 4: Faster rise with WD-DS. Avg. of MATH500 and GSM8K after the shared initialization; w/ and w/o denote with and without WD-DS.

#### 6.3 Module ablation: do WD-DS and WD-JT both matter?

- Table 2 isolates the contribution of the two modules on Qwen3-4B-Base. WD-DS alone improves the three-benchmark average from 54.4% to 56.3%, showing that discrepancy-based selection already identifies more useful training examples. WD-JT alone reaches 58.2% by converting weak-model discrepancies into corrective logit-mixing gradients. Combining WD-DS with WD-JT gives the full WMSS result of 59.9% and improves AIME2025 from 13.3% to 20.0%, indicating that WD-JT provides the main corrective training signal while WD-DS improves where that signal is spent.

Method AIME MATH GSM8K Avg. Baseline 13.3 66.1 83.9 54.4 + WD-DS 13.3 69.4 86.2 56.3 + WD-JT 16.7 70.2 87.6 58.2 WMSS 20.0 71.3 88.5 59.9

Table 2: Ablation of individual modules of WMSS (Qwen3-4B-Base).

#### 6.4 Convergence dynamics: when does WMSS converge?

Figure 3 visualizes an extended trajectory across seven math datasets beyond the two-epoch main comparison. The curves show rapid acquisition followed by stabilization: gains accumulate early, then marginal utility diminishes. The final stage exhibits over-optimisation: AMC23 regresses after Epoch 3 and GSM8K shows late-stage volatility, while other datasets plateau. This pattern is compatible with the late-stage drift mechanism in Section 5, but is not used as causal evidence or as a per-benchmark early-stopping rule.

#### 6.5 Does WD-DS reduce redundant weak-driven steps?

Weak-driven training adds a weak-model view of each update, so the key question is whether this signal is spent where it matters. WD-DS selects examples where weak–strong disagreement can produce corrective gradients and removes duplicate resampled indices so WD-JT runs on a shorter active schedule. Figure 4 starts from the shared initialization checkpoint (MATH500 / GSM8K = 0.658/0.852). The w/o WD-DS schedule uses 218 steps and reaches 0.694/0.873; with WD-DS, the active schedule reaches 0.714/0.876 with 123 steps, about 56% of the full budget, and also surpasses w/o WD-DS at early checkpoints.

### 7 Discussion

We introduced Weak-Driven Learning, turning weak historical states into corrective training signal rather than obsolete checkpoints. Weak agents can still expose plausible but wrong alternatives on hard tokens, while the ground-truth cross-entropy target keeps optimization anchored to the correct answer. WMSS combines weak-driven logit mixing with discrepancy-aware selection, so extra weak-agent computation is spent where weak–strong differences are most informative.

This makes WMSS a practical post-training mechanism: it reuses artifacts already produced during training and requires no online rollouts, external strong teachers, or deployment-time inference changes. Our experiments show consistent gains across math, code, and logic, with WD-JT providing the main corrective signal and WD-DS improving where that signal is applied. Broader transfer and limitations are discussed in Appendix A.

### References

Rishabh Agarwal, Nino Vieillard, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. Gkd: Generalized knowledge distillation for auto-regressive sequence models. CoRR, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025a.

Zehao Chen, Tianxiang Ai, Yifei Li, Gongxun Li, Yuyang Wei, Wang Zhou, Guanghui Li, Bin Yu, Zhijun Chen, Hailong Sun, et al. Llmboost: Make large language models stronger with boosting. arXiv preprint arXiv:2512.22309, 2025b.

Zixiang Chen, Yihe Deng, Huizhu Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramon Huerta, and Ivan Vuli´c. Undial: Selfdistillation with adjusted logits for robust unlearning in large language models, 2024. URL https://arxiv.org/abs/2402.10052.

Gemma Team and Google DeepMind. Gemma 3 technical report, 2025. URL https://arxiv.

org/abs/2503.19786.

Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129(6):1789–1819, Mar 2021. ISSN 1573-1405. doi: 10.1007/s11263-021-01453-z. URL http://dx.doi.org/10.1007/s11263-021-01453-z.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: On-policy distillation of large language models. In International Conference on Learning Representations (ICLR), 2024.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (rest) for language modeling, 2023. URL https://arxiv.org/abs/2308.08998.

Zexue He, Tianxiang Sun, Hongjin Zhang, Zhongkun Liu, Yiheng Zhang, Zhenyu Zheng, Xue Jiang, Baotian Zhang, Haizhou Li, Jun Wang, Zheng Li, Wenhu Chen, Bin Li, Jun Yan, Yun Chen, Luo Si, and Min Zhang. Gentool: Enhancing tool generalization in language models through zero-to-one and weak-to-strong simulation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 1097–1122, 2025.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024a.

Neel Jain, Ping yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Neftune: Noisy embeddings improve instruction finetuning. In International Conference on Learning Representations (ICLR), 2024b. URL https://arxiv. org/abs/2310.05914.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACLHLT), pages 1152–1157, 2016. doi: 10.18653/v1/N16-1136.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. arXiv preprint arXiv:1705.04146, 2017.

Chaoqun Liu, Qin Chao, Wenxuan Zhang, Xiaobao Wu, Boyang Li, Anh Tuan Luu, and Lidong Bing. Zero-to-strong generalization: Eliciting strong capabilities of large language models iteratively without gold labels. In Proceedings of the 31st International Conference on Computational Linguistics, pages 3716–3731, 2025.

Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI), pages 3622–3628, 2020.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Purbesh Mitra and Sennur Ulukus. Semantic soft bootstrapping: Long context reasoning in LLMs without reinforcement learning. arXiv preprint arXiv:2512.05105, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191, 2021.

Seamus Somerstep, Felipe Maia Polo, Moulinath Banerjee, Ya’acov Ritov, Mikhail Yurochkin, and Yuekai Sun. A transfer learning framework for weak-to-strong generalization. In International Conference on Learning Representations (ICLR), 2025.

Junyou Su, He Zhu, Xiao Luo, Liyu Zhang, Hong-Yu Zhou, Yun Chen, Peng Li, Yang Liu, and Guanhua Chen. Instructdiff: Domain-adaptive data selection via differential entropy for efficient llm fine-tuning, 2026. URL https://arxiv.org/abs/2601.23006.

Lukas Thede, Stefan Winzeck, Zeynep Akata, and Jonathan Richard Schwarz. Captrack: Multifaceted evaluation of forgetting in llm post-training. arXiv preprint arXiv:2603.06610, 2026.

Hugo Touvron et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouedec. TRL: Transformer reinforcement learning. https://github.com/huggingface/trl, 2022.

Jingyuan Wang, Yankai Chen, Zhonghang Li, and Chao Huang. Lightreasoner: Can small language models teach large language models reasoning? arXiv preprint arXiv:2510.07962, 2025.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, 2020.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Fengkai Yang, Zherui Chen, Xiaohan Wang, Xiaodong Lu, Jiajun Chai, Guojun Yin, Wei Lin, Shuai Ma, Fuzhen Zhuang, Deqing Wang, Yaodong Yang, Jianxin Li, and Yikun Ban. Your group-relative advantage is biased, 2026. URL https://arxiv.org/abs/2601.08521.

Yuqing Yang, Yan Ma, and Pengfei Liu. Weak-to-strong reasoning. arXiv preprint arXiv:2407.13647, 2024.

Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. Reclor: A reading comprehension dataset requiring logical reasoning. In International Conference on Learning Representations (ICLR), 2020.

Yige Yuan, Teng Xiao, Shuchang Tao, Xue Wang, Jinyang Gao, Bolin Ding, and Bingbing Xu. Incentivizing strong reasoning from weak supervision. In Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2026.

Linfeng Zhang, Jiebo Song, Anni Gao, Jingwei Chen, Chenglong Bao, and Kaisheng Ma. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. doi: 10.1109/ICCV.2019.00381. URL https://arxiv.org/abs/1905.08094.

Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, Yunjie Ji, and Xiangang Li. 1.4 million open-source distilled reasoning dataset to empower large language model training, 2025. URL https://arxiv.org/abs/2503.19633.

Zhanhui Zhou, Zhixuan Liu, Jie Liu, Zhichen Dong, Chao Yang, and Yu Qiao. Weak-to-strong search: Align large language models via searching over small language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Wenhong Zhu, Zhiwei He, Xiaofeng Wang, Pengfei Liu, and Rui Wang. Weak-to-strong preference optimization: Stealing reward from weak aligned model. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2410.18640. arXiv:2410.18640.

Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, Simon Brunner, Chen Gong, Thong Hoang, Armel Randy Zebaze, Xiaoheng Hong, Wen-Ding Li, Jean Kaddour, Ming Xu, Zhihan Zhang, Prateek Yadav, Naman Jain, Alex Gu, Zhoujun Cheng, Jiawei Liu, Qian Liu, Zijian Wang, David Lo, Binyuan Hui, Niklas Muennighoff, Daniel Fried, Xiaoning Du, Harm de Vries, and Leandro Von Werra. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. In International Conference on Learning Representations (ICLR), 2025.

##### Jiaru Zou, Yikun Ban, Zihao Li, Yunzhe Qi, Ruizhong Qiu, Ling Yang, and Jingrui He. Transformer copilot: Learning from the mistake log in llm fine-tuning. arXiv preprint arXiv:2505.16270, 2025.

### Appendix Contents

- A. Limitations and Broader Implications 15
- B. Additional Experiments 16

- B.1 Freezing the Weak Agent during Weak-Driven Training 16
- B.2 Results on Gemma-3-4B-PT 16
- B.3 Logic-Reasoning Results 16
- B.4 Logit Statistics 17
- B.5 Weak–Strong Entropy Separation 17
- B.6 Hyperparameter Sensitivity 18
- B.7 Pass@k Comparison with DPO 20
- B.8 Training Cost and WD-DS Runtime 21

- C. Supplementary Theory: Gradient Amplification 22
- D. Additional Dynamics Diagnostics 25
- E. Data Construction and Training Details 28

### A Limitations and Broader Implications

Limitations. While WMSS introduces no additional inference-time overhead, it uses an additional weak-agent forward pass during training. This makes the method most attractive in settings where modest extra training-time computation is acceptable in exchange for stronger post-training performance at deployment. The method is also designed for regimes where weak and strong agents retain useful token-level discrepancies; if the two agents become nearly identical, the additional corrective signal naturally diminishes. Our experiments focus on structured reasoning and coding tasks, where sharpening decisions against hard negatives is directly aligned with the evaluation objective. Extending the same principle to more open-ended generation settings may require task-adaptive mixing or softer objectives, which we leave to future work.

Broader implications. This work studies a post-training alternative in which useful supervision need not always originate from a stronger external source. In structured reasoning settings, our results suggest that weak agents produced during ordinary training can be reused as corrective signal when target-only SFT yields diminishing returns on hard samples. This provides a practical way to extract more value from existing training trajectories without introducing additional deployment-time inference mechanisms or requiring new external teachers. As with other methods that improve reasoning and code-generation ability, WMSS should be deployed with the same safeguards appropriate for capable language models.

### B Additional Experiments

#### B.1 Freezing the Weak Agent during Weak-Driven Training

Table 3 isolates WD-JT without the WD-DS selection module and compares it with a variant that freezes the base weak-side agent. The frozen-base variant performs substantially worse, suggesting that WD-JT benefits from a weak–strong pair produced along the same training trajectory rather than a fixed base reference. When the weak side is jointly trained as a learner, it can also lose mathematical competence, weakening the corrective contrast that WD-JT relies on.

- Table 3: Effect of freezing the weak-side agent. Results are MATH500/GSM8K pass@1 accuracy (%) at relative checkpoints after the shared initialization.

ck-rel WD-JT only (w/o WD-DS) Frozen base

50 66.6 / 86.2 62.0 / 79.6 100 69.0 / 86.4 62.2 / 82.9 150 68.0 / 86.6 63.2 / 85.4 200 69.4 / 87.6 63.8 / 84.9 final 68.8 / 86.4 63.2 / 86.1

B.2 Results on Gemma-3-4B-PT

To test the weak-driven mixed-logit mechanism beyond Qwen-series models, we extend the mainpaper benchmark (Table 1) to Gemma-3-4B-PT and report results in Table 4. WMSS attains the best Math-Avg (+4.4 over SFT) and Code-Avg (+3.9 over SFT).

- Table 4: Results on Gemma-3-4B-PT. We extend the main-paper benchmark (Table 1) to a different model family. All numbers are pass@1 accuracy (%) under greedy decoding. Methods

Math Code MATH500 AMC23 AQUA GSM8K MAWPS SVAMP Avg. HE+ MBPP+ BCB LCB Avg.

SFT 28.0 17.5 46.5 54.0 83.6 68.3 49.7 36.6 43.1 18.5 7.5 26.4 UNDIAL 31.8 20.0 43.7 58.8 86.6 70.9 52.0 36.6 47.1 19.2 7.8 27.7 NEFTune 32.6 17.5 46.1 59.2 86.6 70.8 52.1 37.8 46.3 20.1 8.0 28.1 SPIN 30.6 7.5 48.0 58.6 84.5 66.1 49.2 37.2 34.7 19.6 7.5 24.8 SSB 31.2 5.0 43.7 51.1 69.3 58.7 43.2 39.0 39.7 19.0 7.3 26.3 WMSS 33.2 22.5 51.2 60.5 87.4 69.5 54.1 41.5 48.9 21.8 9.0 30.3

B.3 Logic-Reasoning Results

To keep the main-paper Table 1 focused on math and code, we report the logic-reasoning benchmarks (LogiQA 2.0 and ReClor) separately in Table 5. Across all four base models, WMSS attains the best LogiQA 2.0, ReClor, and logic-average accuracy, suggesting that the weak-driven mixed-logit mechanism extends beyond numerical and coding domains within structured reasoning tasks.

- Table 5: Logic-reasoning results across base models. We evaluate WMSS against SFT, UNDIAL, NEFTune, SPIN, and SSB on LogiQA 2.0 and ReClor. All numbers are pass@1 accuracy (%) under greedy decoding. Bold marks the best value per column; underline marks the second best.

Qwen3-4B-Base Qwen3-8B-Base Qwen2.5-3B Gemma-3-4B-PT LogiQA2 ReClor Avg. LogiQA2 ReClor Avg. LogiQA2 ReClor Avg. LogiQA2 ReClor Avg.

Methods

SFT 66.1 74.6 70.4 69.2 79.8 74.5 48.5 52.0 50.3 55.4 63.3 59.4 UNDIAL 69.1 75.4 72.3 71.3 81.8 76.6 54.8 60.0 57.4 55.7 67.4 61.6 NEFTune 68.2 73.0 70.6 72.3 83.8 78.1 54.1 59.0 56.6 56.1 68.2 62.2 SPIN 68.3 76.8 72.6 71.3 80.6 76.0 51.5 58.8 55.2 58.1 71.0 64.6 SSB 67.5 74.6 71.1 69.0 81.0 75.0 50.8 53.8 52.3 57.1 70.8 64.0 WMSS 69.5 79.2 74.4 74.6 89.2 81.9 58.0 67.6 62.8 60.5 75.6 68.1

- Table 7: Extended Logit Statistics Before vs. After WD-JT (Qwen3-4B-Base). The weak column reports the weak-side initialization used to form mixed logits; pre/post statistics are reported for the retained strong branch. ∆ denotes strong Post−Pre (percent change in parentheses). Centered norm values are taken directly from the logit analysis report.

Metric Weak init Strong Pre Strong Post ∆ Mean logit zmean 3.65 2.87 0.97 -1.90 (-66.2%) Std σ 3.06 3.18 3.16 -0.02 (-0.6%) Centered norm ∥z˜∥2 1191.33 1240.10 1229.79 -10.31 (-0.8%) Max logit zmax 43.25 48.75 60.50 11.75 (+24.1%) Min logit zmin -24.00 -26.63 -35.75 -9.12 (+34.2%) L2 norm ∥z∥2 1978.92 1865.96 1688.59 -177.37 (-9.5%) Entropy H 0.52 0.44 0.48 0.04 (+9.1%) Max prob pmax 0.83 0.85 0.83 -0.02 (-2.4%)

#### B.4 Logit statistics: does WD-JT suppress hard negatives?

- Table 6: Logit Dynamics Analysis (Qwen3-4B-Base). Comparison of logit statistics at Epoch 3. Values in parentheses indicate the relative change of WMSS vs. SFT.

Metric SFT WMSS

Target Strength (ztarget) 35.88 36.10 (+0.6%) Distractors Mean (zbg) 2.09 0.90 (-56.9%)

Target-to-Background Gap (∆gap) 33.79 35.20 (+4.2%) Logit Std. (σ) 2.93 3.45 (+17.7%)

Table 6 provides a logit-space diagnostic consistent with the mechanism in Section 5. After standard SFT, target logits are already large, so further improvement is unlikely to come mainly from pushing the correct token higher. Instead, WMSS reduces the non-target logit mean from 2.09 to 0.90 (−56.9%), while the target logit changes only slightly (35.88 → 36.10). This suppression-dominant behavior expands the target-to-background gap by +1.41 logit points; we treat it as mechanistic evidence consistent with WD-JT increasing gradient signal on weak-revealed hard negatives, rather than as standalone causal proof.

To provide a more complete view of the same dynamics, Table 7 reports additional statistics from the logit analysis report. Beyond the non-target logit mean reduction above, the extended metrics compare the frozen weak reference with the strong branch before and after WD-JT; only the strong branch is updated.

Implementation details. All logit statistics are computed from 200 randomly sampled examples drawn from the AM-1.4M training dataset used in our experiments. We run forward passes for the frozen weak reference and for the strong pre-/post-WD-JT checkpoints on these examples and aggregate statistics across the sampled set. The centered-norm definition used in the table follows Appendix D (Eq. (48)).

#### B.5 Weak–strong entropy separation

We fix the strong branch to ck-218 and vary the weak-side checkpoint m1 to diagnose entropy separation. When ∆H is small, the two branches have similar uncertainty patterns and logit mixing adds little corrective signal; ck-200 drops to 0.484/0.742. Once the gap is sufficient, the weak branch can supply uncertainty without destabilizing training because CE still anchors the target. The close results of base and ck-50 suggest a threshold effect: sufficient separation matters, but larger gaps do not continuously improve fusion.

0.11

0.09

Relativegap

0.06

0.03

0

###### ∆H = H(m1) − H(ck-218)

Weak–strong entropy gap

Accuracy(%)

strong

base 50 100 150 200 218

m1 checkpoint

MATH500 GSM8K

Fusion with fixed ck-218

90 100 50 base

|150| | | |
|---|---|---|---|
|200| | | |
| | | | |
| | | | |

75

60

45

0.02 0.05 0.08 0.11

∆H to ck-218

Figure 5: Diagnostic of weak–strong entropy separation. Small-gap pairings degrade, while base and ck-50 remain close, suggesting a threshold rather than monotonic gains from larger ∆H.

#### B.6 Fixed hyperparameters and post-hoc sensitivity

To avoid benchmark-level model selection, the headline results in Table 1 use a single fixed WMSS setting across all reported models and domains: α = 0.1, β = 0.8, γ = 0.1, and λ = 0.5. These values are treated as method defaults rather than per-benchmark tuned hyperparameters. In particular, we do not replace the main results with the best configuration found in the sweeps below; the λ sweep later peaks at λ = 0.42, while the headline runs remain at the fixed default λ = 0.5. Tables 8 and 9 are therefore intended as post-hoc sensitivity analyses that characterize robustness around the default operating point, not as a model-selection procedure for the systems reported in Table 1. Three observations support this interpretation:

- (i) The default operating point is not at a sharp peak. The λ sweep (Table 9) shows a broad inverted-U with a plateau over λ ∈ [0.42,0.50], and the (α,β,γ) sensitivity (Table 8) places the default configuration within ∼2pp Math-Avg of the neighboring configurations tested. Points inside the plateau yield results close to the fixed-default headline.
- (ii) Non-default hyperparameter choices remain competitive. The worst-performing configuration in our λ sweep (λ=0.9, Math Avg. 67.6% on Qwen3-4B-Base) still exceeds the strongest non-WMSS baseline in Table 1 (SSB, 66.1%) by ∼1.5 pp and exceeds plain SFT (64.2%) by >3pp. This suggests that the observed gain is not solely a product of a narrow hyperparameter peak.
- (iii) The high-performing λ region is consistent with the theory. The centered-norm crossover formula in Appendix D (Eq. (17)) predicts a high-performing mixing region near λ ≈ 0.49 from the empirical centered-norm ratio of the two agents, in close agreement with the observed plateau around λ ∈ [0.42,0.50]. This is a mechanism-level consistency check, not a selection rule for the headline models.

#### B.7 Sensitivity of Difficulty, Consolidation, and Repair Coefficients.

We further investigate the impact of the consolidation coefficient α, the difficulty coefficient β, and the regression-repair coefficient γ. Table 8 reports a post-hoc sensitivity check on Qwen3-4B-Base and Qwen2.5-3B. On Qwen3-4B-Base we observe a distinct trade-off between standard mathematical proficiency (MATH 500) and complex reasoning capability (AIME 2025).

Configuration C (α = 0.1,β = 0.9,γ = 0), which disables the regression-repair signal, achieves the highest accuracy on MATH 500 (70.2%). However, its performance on the more challenging AIME benchmark drops significantly to 10.3%. This suggests that while stronger emphasis on base difficulty (β = 0.9) helps with standard problems, it can still lead to optimization stagnation on harder tasks without regression repair.

In contrast, the fixed default configuration (α = 0.1,β = 0.8,γ = 0.1) introduces a controlled regression-repair emphasis. Although this slightly reduces MATH performance (70.2% → 68.2%), it yields a notable +6.4% gain on AIME (10.3% → 16.7%). This suggests that the additional signal induced by γ is useful for harder reasoning cases and that the default setting is not an isolated peak chosen solely from the aggregate benchmark score.

- Table 8: Post-hoc hyperparameter sensitivity analysis. We evaluate the impact of the mixing coefficients α (consolidation), β (base difficulty), and γ (regression repair) on mathematical reasoning across two base models. These sweeps are diagnostic and are not used to select the headline models in Table 1. All numbers are pass@1 accuracy (%) under greedy decoding. Bold marks the best value per metric within each base-model column group. “-” indicates that the base model produces no correct answer under any configuration.

Coefficients Qwen3-4B-Base Qwen2.5-3B

Setup

α β γ AIME 2025 MATH 500 AIME 2025 MATH 500

- Config A 0.2 0.7 0.1 10.0 68.2 - 44.6 Default 0.1 0.8 0.1 16.7 68.2 - 48.2

- Config B 0.2 0.6 0.2 10.7 67.8 - 42.0
- Config C 0.1 0.9 0.0 10.3 70.2 - 47.6
- Config D 0.0 1.0 0.0 9.7 68.3 - 47.0
- Config E 0.05 0.85 0.1 14.0 69.5 - 47.2

Qwen2.5-3B. On the smaller Qwen2.5-3B base, this post-hoc sweep does not register any correct AIME 2025 answer across the six tested α/β/γ variants (shown as “-” in Table 8), so the variants cannot be discriminated on this benchmark. The MATH 500 column nevertheless shows that the fixed default (α = 0.1,β = 0.8,γ = 0.1) remains competitive and attains the best score (48.2%), outperforming the regression-repair-disabled Config C (47.6%) and the difficulty-only Config D (47.0%). The relative gap between the default and Config C is smaller on 3B than on 4B (+0.6 vs. −2.0 on MATH 500), consistent with the interpretation that the regression-repair term is most useful when the base model can recover hard-task signal in the first place.

#### B.8 Sensitivity Analysis on Mixing Coefficient λ

The mixing coefficient λ explicitly controls the relative weight between the weak and strong logits in mixed-logit training, with fused logits zmix = (1 − λ)z1 + λz2. Smaller λ assigns more influence to the weaker model, while larger λ emphasizes the stronger model’s direct fit to the targets.

We sweep λ over [0.1,0.9] with a finer grid in [0.4,0.6] after the headline experiments have been fixed. Table 9 shows a broad inverted U-shape: the best average performance occurs at λ = 0.42 (Avg. 75.5%), and a strong plateau persists in λ ∈ [0.42,0.50], which includes the fixed default λ = 0.5. All benchmarks are evaluated with three-run averages; minor fluctuations on small sets (e.g., AIME25, 30 problems) are still expected, but the high-performing region remains stable.

At the extremes, the behavior aligns with the mechanism. As λ → 1, zmix ≈ z2 and the mixed-logit objective reduces to relying on the strong model, so the compensatory interaction weakens and accuracy drops (e.g., Avg. 67.6% at λ = 0.9). As λ → 0, the weak model dominates the fused logits, reducing effective target learning and risking underfitting; this is especially pronounced for weaker base models (e.g., at λ = 0.3, a weaker model attains only 13% accuracy on MATH500).

Why does the optimum sit near λ ≈ 0.42 rather than λ → 1? A natural intuition is that more weight on the strong model (λ → 1) should be better, since the strong model alone is already correct on most tokens. But WMSS’s benefit comes from the weak agent’s contribution to the fused distribution: the weak agent’s softer decision boundary preserves probability mass on hard negatives, which is exactly the signal needed on hard samples (Theorem 1). At λ → 1 this weak–strong interaction vanishes and the loss reverts to standard target-only SFT. The optimum at λ ≈ 0.42 therefore reflects a balance between two distinct roles: the strong agent contributes the target direction (we still want the loss to push toward y), while the weak agent contributes uncertainty over hard negatives. WMSS is not equivalent to a regulariser that shrinks logits toward zero: the weak agent supplies a structurally informed direction, not isotropic noise, which is consistent with our gradient-share crossover prediction near λ ≈ 0.49 (Eq. (17)).

Numerical consistency with the gradient-share crossover. Following the theoretical analysis in Appendix D, we can estimate the effective sensitivity ratio using the centered-norm proxy:

α ≈

∥z˜2∥2 ∥z˜1∥2

2

. (16)

- Table 9: Post-hoc sensitivity analysis of mixing coefficient λ. We report average greedy pass@1 accuracy (%) over three independent training runs per λ on Qwen3-4B-Base. These sweeps are diagnostic and are not used to select the headline models in Table 1, which use the fixed default λ = 0.5. The curve exhibits a broad inverted U-shape, with the strongest performance typically in the λ ∈ [0.42,0.50] range (peak at λ = 0.42 in Avg.), reflecting a balanced trade-off between learning new features and preserving historical constraints.

Dataset

Mixing Coefficient λ 0.1 0.2 0.3 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.7 0.8 0.9

AIME25 7.8 10.0 12.2 16.7 20.0 20.0 16.7 20.0 20.0 21.1 16.7 16.7 16.7 10.0 20.0 20.0 12.2 MATH500 71.4 67.1 70.8 73.1 74.9 74.5 73.2 73.3 73.3 70.7 74.0 73.0 73.3 68.9 71.7 70.9 66.1 AQUA 77.2 72.8 74.7 75.2 73.9 73.1 72.4 71.8 73.9 71.0 71.3 71.9 70.2 64.8 66.9 61.4 60.8 GSM8K 88.2 89.1 89.4 91.9 91.8 91.7 91.0 91.4 91.0 91.2 91.4 90.6 90.9 89.2 89.3 88.1 85.7 MAWPS 94.9 96.6 95.5 97.8 98.2 97.8 97.8 98.0 95.7 96.5 97.2 96.8 97.5 94.7 96.2 95.5 93.6 SVAMP 90.4 91.8 93.3 94.9 94.1 93.6 93.5 92.9 92.6 92.4 93.5 92.8 92.9 90.8 91.9 89.6 87.4

Avg. 71.7 71.2 72.7 74.9 75.5 75.1 74.1 74.6 74.4 73.8 74.0 73.6 73.6 69.7 72.7 70.9 67.6

From the logits evaluation report, the strong pre-update centered norm is ∥z˜2∥2 ≈ 1240.10 and the weak-side initialization centered norm is ∥z˜1∥2 ≈ 1191.33, giving α ≈ 1.08. Plugging into the crossover formula in Eq. (63),

λcross ≈

1

1 + √α ≈ 0.490, (17) which lies close to the empirically strongest region (λ ∈ [0.42,0.50]). We stress that this is a heuristic consistency check: α is phase-dependent and the linearization is local, so the theory predicts a broad optimum region rather than a sharp inversion point.

B.9 Pass@k Comparison with DPO

To compare WMSS against a preference-based baseline under multi-sample evaluation, we evaluate both WMSS (Qwen3-4B-Base) and a DPO baseline trained from the same SFT warm-start on the same math data. The DPO preference pairs are constructed by sampling 4 rollouts per prompt at T=0.8 from the SFT-warm-start policy and using math_verify against the ground-truth boxed answer to label each rollout as correct or incorrect. We use the first verified correct rollout in the fixed sampling order as the chosen response and the first verified incorrect rollout as the rejected response; prompts with no valid pair are skipped. No external reward model or process-reward signal is used. For evaluation, we draw n = 64 rollouts per problem at T = 0.7, top-p = 0.95, and report

the unbiased pass@k estimator of Chen et al. [2021], pass@k = Eproblem[1 − n−k c / nk ] (where c is the number of correct rollouts), for k ∈ {1,4,8,16,32,64} on AIME 2025, AMC23, and MATH500.

- Table 10: Pass@k evaluation of WMSS vs. DPO on Qwen3-4B-Base. Sampling at T = 0.7, top-p = 0.95, n = 64 samples per problem; the k=1 entry is equivalent to mean@64. Bold marks the higher value in each (benchmark, k) cell.

AIME 2025 AMC23 MATH500 WMSS DPO WMSS DPO WMSS DPO

k

1 17.97 14.20 49.49 42.80 75.32 70.90 4 24.69 22.40 63.25 56.70 84.36 84.00 8 26.59 24.90 68.06 61.00 87.20 87.40

16 28.57 27.30 71.79 65.20 89.48 89.90 32 30.83 29.10 75.13 69.80 91.49 91.80 64 33.33 30.00 77.50 75.00 93.20 93.40

WMSS outperforms DPO on hard reasoning benchmarks across k. On AIME 2025 and AMC23 WMSS beats DPO at every k: +3.8pp at pass@1 (17.97 vs. 14.20) and +6.7pp at pass@1 (49.49 vs. 42.80) respectively, with the lead remaining positive at the diversity ceiling k=64 (+3.3 on AIME, +2.5 on AMC23). On the easier MATH500, WMSS leads at pass@1 / pass@4 (+4.4 / +0.4) but DPO catches up for k ≥ 8 within 0.4pp as both methods approach the 90% ceiling. This pattern is consistent with WMSS helping most where hard-reasoning gradients remain informative, while preference-based diversity can close the gap on easier benchmarks at larger k.

The advantage is sharpest at low k. The relative gap between WMSS and DPO is largest at small k and shrinks as k grows: on AIME 2025 the ratio (WMSS/DPO) is 1.27× at k=1 and 1.11× at k=64; on AMC23 it shrinks from 1.16× to 1.03×; on MATH500 from 1.06× to 1.00×. This suggests that WMSS improves the single-decode regime, which is relevant when best-of-N rejection sampling is expensive. This is consistent with the gradient-amplification analysis in Section 5: by increasing gradient signal on weak-revealed hard negatives, WMSS may sharpen the per-token decision boundary and improve top-1 accuracy.

Diversity gain (pass@64 – pass@1). The pass@k growth ∆ = pass@64 − pass@1 measures how much additional problem coverage the model finds via diverse sampling. WMSS grows by ∆AIME=+15.4, ∆AMC23=+28.0, ∆MATH500=+17.9pp; DPO grows by ∆AIME=+15.8, ∆AMC23=+32.2, ∆MATH500=+22.5. DPO has higher diversity gains across the board, while WMSS starts from a stronger pass@1 baseline on AIME and AMC23; as a result, DPO’s additional sampling does not close the absolute gap on those two hard benchmarks by pass@64.

#### B.10 Training Cost and WD-DS Runtime

- Table 11 reports measured wall-clock time for timed post-training runs on Qwen3-4B-Base after the same SFT warm-start. These numbers are implementation-level measurements, not a FLOPnormalized comparison: rollout-heavy methods include their own generation stages, while WMSS includes the WD-DS active schedule and dual forward pass during WD-JT. The WMSS entry uses the deduplicated WD-DS schedule reported in Section 6.5, replacing the earlier full-schedule runtime with the measured ck-123 runtime.

- Table 11: Measured training time on Qwen3-4B-Base. Wall-clock reports total measured training time after the shared SFT warm-start; it is not FLOP-normalized across methods with different rollout or dual-forward costs. Benefit is the Math-Avg pass@1 gain over SFT in Table 1.

Method Total training time Math-Avg ∆ SFT 4h30m – NEFTune 4h30m +0.7 UNDIAL 4h31m −2.9 SSB ∼18h +1.9 SPIN ∼7h53m +2.4 WMSS (dedup WD-DS) 4h49min +4.9

### C Supplementary Theory: Gradient Amplification under Weak-Driven Logit Mixing

Notation. Fix a context x and a target token index y, and analyze a single mixed-logit training step; we omit the conditioning on x for readability. The weak and strong agents Mweak and Mstrong produce logits zweak(x),zstrong(x) ∈ R|V|, and the mixed logits are

zmix = (1 − λ)zweak + λzstrong, λ ∈ [0,1]. (18)

For any logit map z(x), define Pz(· | x) = Softmax(z(x)) and use the shorthand Pz(k); let Pmix = Pz

and ey denote the one-hot target vector. For any negative token k ̸= y, the target margin is mk(z) = z[y] − z[k].

mix

- C.1 Setup and Baseline We compare the mixed-logit loss with standard SFT on the strong agent:

Lmix = −log Pmix(y), LSFT = −log Pstrong(y), (19) where Pstrong = Pz

strong

. The gradient with respect to the fused logits is

gmix = ∇zmixLmix = Pmix(·) − ey. (20) In standard SFT on the strong agent alone, the corresponding gradient is

gsft = ∇zstrongLSFT = Pstrong(·) − ey. (21) By the chain rule, the branch-level logit gradients induced by the mixed objective are

∇zweakLmix = (1 − λ)gmix, ∇zstrongLmix = λgmix. (22)

- C.2 Log-Odds and Margin Contraction

- Definition 1 (Target margin). For any negative token k ̸= y, define mk(z) = z[y] − z[k]. (23)

- Lemma 4 (Softmax log-odds). For any k ̸= y, the log-odds under logits z satisfy

log

Pz(k) Pz(y)

= −mk(z). (24)

Proof. By definition, Pz(k)/Pz(y) = exp(z[k] − z[y]) = exp(−mk(z)).

| |
|---|

- Lemma 5 (Margin mixing). For any k ̸= y, the mixed margin equals the convex combination mk(zmix) = (1 − λ)mk(zweak) + λmk(zstrong). (25)

Proof. Using (18) and (23), we expand mk(zmix) = zmix[y] − zmix[k]

= (1 − λ)zweak[y] + λzstrong[y] − (1 − λ)zweak[k] + λzstrong[k]

= (1 − λ)(zweak[y] − zweak[k]) + λ(zstrong[y] − zstrong[k])

= (1 − λ)mk(zweak) + λmk(zstrong). (26)

| |
|---|

#### C.3 Negative-Gradient Amplification

- Definition 2 (Hard-negative set). Define the hard-negative set H = {k ̸= y : mk(zweak) < mk(zstrong)}, (27)

i.e., tokens for which the weak agent has a smaller target-vs-negative margin.

- Lemma 6 (Relative probability increase on hard negatives). For any k ∈ H,

Pstrong(k) Pstrong(y)

Pmix(k) Pmix(y)

. (28)

>

Proof. By (25), mk(zmix) < mk(zstrong) for k ∈ H. Applying (24) yields a larger log-odds ratio for the mixed logits.

| |
|---|

Corollary 7 (Sufficient condition for per-token amplification). For any k ∈ H, the negative-token gradient on fused logits satisfies

Pmix(k) ≥ Pstrong(k) whenever

where ∆mk = mk(zstrong) − mk(zweak) > 0. Proof. Using log-odds,

Pmix(y) Pstrong(y) ≥ exp(−(1 − λ)∆mk), (29)

Pmix(y)exp(−mk(zmix)) Pstrong(y)exp(−mk(zstrong))

Pmix(k) Pstrong(k)

=

Pmix(y) Pstrong(y)

exp(mk(zstrong) − mk(zmix)). (30) By (25),

=

mk(zstrong) − mk(zmix) = (1 − λ) mk(zstrong) − mk(zweak)

= (1 − λ)∆mk. (31) Thus

Pmix(k) Pstrong(k)

Pmix(y) Pstrong(y)

exp((1 − λ)∆mk). (32)

=

Solving for Pmix(k) ≥ Pstrong(k) gives (29). Theorem 8 (Total negative-mass increase under uniform margin shrinkage). If mk(zweak) ≤ mk(zstrong) for all k ̸= y, then

| |
|---|

Pstrong(k). (33)

Pmix(y) ≤ Pstrong(y),

Pmix(k) ≥

k̸=y

k̸=y

Proof. By (25) and the assumption,

mk(zmix) = (1 − λ)mk(zweak) + λmk(zstrong) ≤ mk(zstrong) ∀k ̸= y. (34) Applying (24) yields

exp − mk(zmix) ≥ exp − mk(zstrong) ∀k ̸= y. (35) Summing over k ̸= y gives

exp − mk(zstrong) . (36)

exp − mk(zmix) ≥

k̸=y

k̸=y

Since

 

 1 +

−1

, (37)

exp − mk(z)

Pz(y) =

k̸=y

the denominator for Pmix(y) is no smaller than that for Pstrong(y), and therefore Pmix(y) ≤ Pstrong(y). The second inequality follows from probabilities summing to one:

Pmix(k) = 1 − Pmix(y) ≥ 1 − Pstrong(y) =

k̸=y

Pstrong(k). (38)

k̸=y

| |
|---|

Proposition 9 (Logit updates emphasize negative suppression). For any negative token k ̸= y, the local logit update for branch i ∈ {weak,strong} under the mixed objective is

∆zi[k] ≈ −ηsiPmix(k), sweak = 1 − λ, sstrong = λ, (39) while for the target token,

∆zi[y] ≈ ηsi(1 − Pmix(y)). (40)

Thus any increase in Pmix(k) directly amplifies suppression of hard negatives, while a decrease in Pmix(y) strengthens the upward push on the target logit.

Proof. From (22), the branch-i logit gradient satisfies

∇ziLmix = sigmix, (41) where gmix = Pmix(·) − ey. For a negative token k ̸= y,

∂Lmix ∂zi[k]

= siPmix(k), (42) and for the target token,

∂Lmix ∂zi[y]

= si(Pmix(y) − 1). (43) Under a first-order update ∆zi ≈ −η∇ziLmix, we obtain (39) and (40).

| |
|---|

- Remark 1 (Consistency with logit statistics). The mechanism above provides a local explanation for the logit statistics reported in Table 6 and Table 7: increased negative probability mass amplifies downward updates on incorrect tokens, while a lower mixed target probability strengthens the upward update on the correct token. Mean shifts can further accumulate along shift-invariant directions, as analyzed in Appendix D.

### D Additional Dynamics Diagnostics for Mixed-Logit Training

#### D.1 Problem Setting and Notation

Notation. Fix a context x and a target index y, and analyze a single WD-JT mixed-logit update; we omit the conditioning on x for readability. The weak and strong agents form a training-time pair and produce logits zweak(x),zstrong(x) ∈ R|V|. We write branch-level derivatives with respect to both logit maps because the mixed objective backpropagates through the paired logits, while the strong branch is the model retained for evaluation and deployment. The mixed logits are

zmix = (1 − λ)zweak + λzstrong, λ ∈ [0,1]. (44)

For any logit map z(x), define Pz(· | x) = Softmax(z(x)) and use the shorthand Pz(k); let Pmix = Pz

and ey denote the one-hot target. We use a subscript i ∈ {weak,strong} to index the two models in generic expressions. Let 1 be the all-ones vector in R|V|. We use the mixed-logit loss

mix

Lmix = −log Pmix(y), (45) and the residual

g = ∇zmixLmix = Pmix(·) − ey. (46) For any negative token k ̸= y, the margin is mk(z) = z[y] − z[k]. We use η to denote the learning rate. For each model i ∈ {weak,strong}, let Ji = ∂zi/∂θi and define the Jacobian Gram matrix

Ki = JiJi⊤. (47)

Definition 3 (Centered logits and centered norm). Let z¯ = |V|1 1⊤z be the logit mean. The centered logits and centered norm are

z˜ = z − z¯1, ∥z˜∥2 =

|V|

(z[k] − z¯)2. (48)

k=1

Since ∥z˜∥2 = |V| · Std(z), the centered norm is a shift-invariant measure of sharpness.

#### D.2 First-Order Gradients and Centered Linearized Dynamics By the chain rule,

∇zweakLmix = (1 − λ)g, (49) ∇zstrongLmix = λg. (50)

Thus the mixed objective exposes branch-level logit derivatives in the same direction but with different magnitudes. The following branch-sensitivity calculation is a local diagnostic of how the jointly optimized weak–strong pair responds under the mixed objective. Under a first-order (local) linearization, we apply a Taylor expansion of logits around θi:

zi(θi + ∆θi) ≈ zi(θi) + Ji∆θi, i ∈ {weak,strong}. (51) With ∆θi = −η∇θiLmix, this yields

∆zi ≈ −ηJiJi⊤g = −ηKig, (52) and incorporating the mixing weights in (49)–(50) yields

∆zi ≈ −ηsiKig, sweak = 1 − λ, sstrong = λ. (53)

To remove mean-shift effects, let Π = I − |V|1 11⊤ be the centering projector. Since 1⊤g = 0, we have Πg = g and define the centered kernel

K˜i = ΠKiΠ. (54) Then the centered logit dynamics are

∆˜zi ≈ −ηsiK˜ig. (55)

Lemma 10 (PSD of centered kernel). For each model i ∈ {weak,strong}, K˜i is positive semidefinite. Proof. For any x ∈ R|V|, x⊤K˜ix = x⊤ΠJiJi⊤Πx = ∥Ji⊤Πx∥22 ≥ 0. For the jointly optimized training pair, the first-order fused-logit change is

| |
|---|

∆˜zmix = (1 − λ)∆˜zweak + λ∆˜zstrong

= −η (1 − λ)2K˜weak + λ2K˜strong g. (56)

The corresponding diagnostic first-order loss decrease follows from a Taylor expansion of Lmix at zmix:

∆Lmix ≈ ∇zmixL⊤mix∆˜zmix = g⊤∆˜zmix. (57) Substituting (56) gives

∆Lmix ≈ g⊤∆˜zmix = −η (1 − λ)2g⊤K˜weakg + λ2g⊤K˜strongg . (58)

#### D.3 Stage I: Hard-Negative Amplification and Strong-Model Dominance

Early in mixed-logit training, the weak model is more confused, so for many hard negatives k we have mk(zweak) < mk(zstrong). By the logit-mixing analysis in Section 5 of the main paper (Theorem 1), mixing shrinks these margins and increases total negative probability mass. Consequently, the residual g grows in magnitude and is biased toward hard negatives: the weak model acts as a gradient amplifier, not a competitor in final accuracy.

In this diagnostic decomposition, the effective per-step loss decrease attributable to model i ∈ {weak,strong} is

Ei ≜ ηs2ig⊤K˜ig, sweak = 1 − λ, sstrong = λ. (59) Dominance corresponds to Estrong > Eweak. Define the directional sensitivity κi = g⊤K˜ig/∥g∥22.

Assumption 11 (Sensitivity advantage of the strong model). Because the strong model is sharper (larger centered norm / lower entropy), its centered kernel responds more strongly along the residual direction:

g⊤K˜strongg ≈ α g⊤K˜weakg, α > 1. (60) Substituting (60) into (58) yields the total effective rate

S(λ) = (1 − λ)2 + αλ2, (61) and the strong model dominates the effective mixed-logit update when

λ2α > (1 − λ)2. (62) Solving (62) gives the gradient-share crossover

1 1 + √α

. (63)

λcross =

- Remark 2 (Crossover vs. accuracy). Equation (63) characterizes a local crossover of branch sensitivity under a one-step linearization. It does not predict an accuracy inversion between the two agents. In practice, g, K˜weak, and K˜strong evolve with λ and training time, while optimizer dynamics smooth the trajectory, yielding a broad optimum region rather than a sharp transition.
- Remark 3 (Softmax amplification). Because Pmix(k) ∝ exp(zmix[k]), modest logit differences can disproportionately tilt Pmix and g. This amplifies hard-negative gradients and reinforces the early dominance in (62).

#### D.4 Stage II: Gradient Shielding via Hessian Contraction

We next analyze why the weak-induced interaction becomes increasingly shielded once the strong branch becomes confident.

#### D.4.1 Softmax Jacobian and Loss Hessian

Recall Pmix = Softmax(zmix). The Jacobian of Softmax is

∂Pmix ∂zmix

= diag(Pmix) − PmixPmix⊤ . (64) Since ey is constant, the Hessian of the cross-entropy loss with respect to zmix is

∂(Pmix − ey) ∂zmix

= diag(Pmix) − PmixPmix⊤ . (65) Lemma 12 (PSD of the loss Hessian). HL is positive semidefinite. Proof. For any v ∈ R|V|,

HL = ∇2zmixLmix =

 

 

2

v⊤HLv =

Pmix[j]vj2 −

(v) ≥ 0. (66)

Pmix[j]vj

= VarP

mix

j

j

| |
|---|

#### D.4.2 Interaction Hessian Between Models

The mixed logits depend on both branches, so the cross-Hessian is useful as a sensitivity diagnostic: it captures how strong-logit changes contract the weak-side derivative, although the weak agent itself is not updated:

∂ ∂zstrong

[∇zweakLmix]

Hws =

∂(Pmix − ey) ∂zmix

∂zmix ∂zstrong

= (1 − λ)

= λ(1 − λ)HL. (67)

#### D.4.3 Shielding in the Confident Regime

Assume the target index is y and the strong model drives the mixed prediction to Pmix(y) → 1. Then Pmix(k) → 0 for all k ̸= y, which implies

HL = 0. (68) Consequently,

lim

Pmix→ey

Hws = 0, (69)

lim

Pmix→ey

and the cross-branch curvature vanishes. This is the mathematical form of gradient shielding: once the strong branch dominates and the fused prediction becomes highly concentrated, both the firstorder residual g and its local sensitivity collapse, making weak-induced corrections fade in the mixed objective.

#### D.5 Stage III: Null-Space Drift and Mean Shift

We now explain why global logit shifts can appear in post-WD-JT diagnostics without necessarily changing centered sharpness.

#### D.5.1 Shift Invariance of the Mixed Softmax

Softmax is invariant to global shifts. For any scalar c,

Softmax(z + c1) = Softmax(z). (70) Applying this to the fused logits, let zweak′ = zweak + c1. Then

zmix′ = (1 − λ)zweak′ + λzstrong

= zmix + (1 − λ)c1, (71) and by (70) the predictive distribution is unchanged. Hence, the loss is flat along the mean-shift direction of each model.

#### D.5.2 Zero-Eigenvalue Direction of the Hessian Let 1 denote the all-ones vector. From (65),

HL1 = diag(Pmix)1 − Pmix(Pmix⊤ 1) = Pmix − Pmix = 0. (72) Thus 1 is a zero-eigenvalue direction of the Hessian, confirming that the loss has no curvature along global shifts.

#### D.5.3 Why Drift Accumulates in the Null Space

For cross-entropy, the gradient along a global-shift direction is exactly zero because 1⊤g = 0. In any trainable branch, stochastic optimizer noise and optimizer regularization can still move parameters in directions that change raw logit means; a simplified stochastic component can be written as

θ+ = θ − ηϵ, ϵ ∼ N(0,Σ), (73) which is a random walk. The logit space decomposes as

z = z¯1 + z,˜ z˜⊤1 = 0, (74)

where the null space is span{1} and the active space is its orthogonal complement. In the active space, even small gradients can weakly restore the distribution shape. In the null space, there is no CE restoring force; stochasticity can increase the variance of z¯, while optimizer details such as weight decay can determine the observed direction of the mean shift. This explains why raw logit means can drift without a corresponding increase in centered norm.

#### D.6 Discussion: regime of validity, multi-step behaviour, and failure modes

Regime of validity. The uniform margin shrinkage condition mk(zweak) ≤ mk(zstrong) for all k ̸= y (Theorem 1 of the main paper) is sufficient but not necessary. In practice, on a small fraction of tokens the strong agent may be less confident than the weak agent (e.g., a token where the strong model has not yet committed). For such tokens, mixing shrinks negative mass instead of expanding it, and the global benefit of WMSS relies on the much larger pool of tokens where the assumption holds. Since zmix is a convex combination, the effect on a violating token is bounded by the gap between Pstrong and Pweak on that token, but the intended amplification can be reduced or reversed locally.

From single-step to multi-step training. The analysis in Section 5 of the main paper is a singlestep linearisation: it characterises the immediate effect of one gradient update under fused logits. Across many steps, the cumulative behaviour is consistent with what we observe empirically. Stages I and II describe the typical regime in early to mid training, where amplification dominates and gradients on hard negatives keep the strong agent moving. Stage III corresponds to late-training over-optimisation: as Pmix becomes nearly one-hot, gradients shrink on every token and optimizer dynamics can accumulate in the shift-invariant null space, producing raw mean-logit drift without comparable centered-norm growth. The convergence trajectories in Figure 3 are consistent with this picture, but are used as diagnostics rather than as benchmark-specific early-stopping evidence.

When does suppression hurt? Theorem 1 and Stage I together predict stronger suppression of non-target probability mass, consistent with the empirical zbg : 2.09 → 0.90 in Section B.4. On tasks where the correct answer is single-mode—competition math, code, structured logic—this suppression is desirable. On tasks where multiple valid completions coexist (open-ended generation, dialogue with multiple acceptable continuations, tasks requiring calibrated hedging), aggressive suppression of hard negatives may be over-confident. Our experiments cover only single-mode benchmarks; we leave hedging-sensitive evaluation to future work and flag this as a limitation in Appendix A.

### E Data Construction and Training Details

#### E.1 Evaluation Benchmarks and Baselines

Tasks and evaluation. For mathematical reasoning, we report performance on AIME2025, MATH500, AMC23, AQuA Ling et al. [2017], GSM8K Cobbe et al. [2021], MAWPS KoncelKedziorski et al. [2016], and SVAMP Patel et al. [2021]. For code generation, we evaluate on

HumanEval+ (HE+) and MBPP+, the augmented test-case versions of HumanEval Chen et al. [2021] and MBPP Austin et al. [2021] introduced by EvalPlus Liu et al. [2023], together with BigCodeBench (BCB) Zhuo et al. [2025] and LiveCodeBench (LCB) Jain et al. [2024a] for broader, contaminationresistant coverage. For logical reasoning, we evaluate on LogiQA 2.0 Liu et al. [2020] and ReClor Yu et al. [2020].

Baselines. Standard SFT optimizes the conventional next-token prediction objective without discrepancy-based sample selection or logit-space perturbation. UNDIAL [Dong et al., 2024] suppresses target-token logits using stochastic penalties to mitigate over-confident predictions. NEFTune [Jain et al., 2024b] injects random noise into embeddings during training to improve generalization. SPIN [Chen et al., 2024] employs a self-play mechanism where the model iteratively distinguishes its own generated responses from human-annotated data, progressively converting a weak policy into a stronger one without external reward models. SSB [Mitra and Ulukus, 2025] (Semantic Soft Bootstrapping) is an RL-free self-distillation method that uses the same base model as both teacher and student under different contexts: the teacher, conditioned on a hint trace, generates soft logit targets over the answer, which the student (without hints) learns to match via KL divergence, eliminating the need for preference data or RL training.

#### E.2 Training Data Sources

Our training mixture covers three domains: mathematical reasoning, code generation, and logical reasoning. All samples use a chat message format (user/assistant turns), with assistant completions retaining the full <think>-wrapped chain of thought and a structured final answer. We use the full set of verified samples in each domain (Table 12); the math pool is by far the largest (∼5.7× code, ∼7.5× logic).

Table 12: Training data across the three domains used in our experiments. Each domain uses the full set of verified samples from the corresponding source.

Domain #Samples Source Math 111,657 Full AM-DeepSeek-R1-1.4M Zhao et al. [2025] math subset,

verified by math_verify Code 19,457 AM-DeepSeek-R1-1.4M code subset, execution-verified against the provided unit tests Logic 14,882 LogiQA 2.0 and ReClor distilled CoT, retained after answermatch verification (Section E.3)

Mathematical reasoning. Sourced from the AM-DeepSeek-R1-1.4M math subset (∼111 K verified examples spanning algebra, geometry, number theory, combinatorics, and competition problems). We retain DeepSeek-R1’s full <think> CoT together with the final \boxed{X} answer and discard samples that fail math_verify parsing, leaving 111,657 training examples. SSB and SPIN are exceptions and run on a 20K subsample of this pool due to the high cost of their rollout phases.

Code generation. Sourced from the same corpus, filtered by executing the produced code against the provided unit tests. We retain the <think> reasoning block; this yields 19,457 training examples after execution verification.

##### E.3 Logic Data Synthesis The Logic training set is synthesized in two steps from public benchmarks.

- Step 1: merging raw questions. We combine the raw splits of LogiQA 2.0 Liu et al. [2020] (MRC version, train/dev/test) and ReClor Yu et al. [2020] (train/val) into a unified JSON format (∼17,205 items). Before any CoT distillation, the LogiQA 2.0 test split (1,572) and the ReClor val split (500) are removed from the training-candidate pool and held out for evaluation; we use ReClor val because the public ReClor test set has no released labels. The remaining training-candidate pool contains 15,133 items.

- Step 2: CoT distillation with answer verification. We prompt Qwen3-32B in its thinking mode on each training-candidate question to obtain a long-form chain-of-thought answer. A sample is retained only if the extracted prediction matches the ground-truth option. This yields 14,882 verified Logic training samples, corresponding to a 98.3% retention rate after the held-out evaluation splits have already been removed.

#### E.4 Gemma-3-4B-PT: Tag-Free Preprocessing

In preliminary experiments, Gemma-3-4B-PT did not reliably follow the <think>/<answer> tag convention during training. We therefore adopt a tag-free variant only for Gemma.

Tag stripping. Starting from the original CoT outputs, we remove the literal <think>, </think>, <answer>, and </answer> markers while retaining the reasoning text and the inner answer payload. Thus Gemma still trains on tag-free step-by-step solutions ending with \boxed{X} for math, The answer is (X) for logic, or Python code blocks for code.

Fallback chat template. Gemma-3-4B-PT ships without a chat template (only the IT variant has one). We inject a Gemma-3-IT-style fallback in which the assistant role is mapped to model, a leading system message is prepended to the first user turn (rather than forming its own turn), and each turn is wrapped with <start_of_turn>/<end_of_turn>. These special tokens are already present in the PT vocabulary.

Tagged vs. tag-free system prompts. Qwen-series bases use a tag-style system prompt that instructs the model to emit <think>···</think> followed by <answer>···</answer>. Gemma uses a tag-free prompt that ends with “Solve the problem step by step. End your solution with the final answer enclosed in \boxed{}” (math), “End your response with ‘The answer is (X).’” (logic), or “Provide clean, correct Python code” (code). The SPIN and SSB rollout/teacher phases are aligned to the base-appropriate prompt via their –system-prompt / –student-system-prompt flags so that the self-generated trajectories match the same formatting that the student is trained to emit.

vLLM multimodal wrap for evaluation. Training saves a Gemma3ForCausalLM (text-only) checkpoint, while vLLM v0.11 loads Gemma-3 only as Gemma3ForConditionalGeneration (multimodal). To make the saved checkpoint loadable for evaluation, we (i) re-key every weight with a language_model. prefix and (ii) inject the base model’s vision_tower.* and multi_modal_projector.* weights unchanged. The wrapped checkpoint is functionally identical for text-only decoding but conforms to vLLM’s expected module layout.

#### E.5 Training Configuration

Implementation details. We use Qwen3-4B-Base, Qwen3-8B-Base Yang et al. [2025], Qwen2.53B, and Gemma-3-4B-PT [Gemma Team and Google DeepMind, 2025] as backbone models, and fine-tune them on the AM-1.4M dataset Zhao et al. [2025] for math and code, together with the Logic training set synthesised in Appendix E.3. All experiments are conducted on eight NVIDIA H800 80GB GPUs. Our implementation is based on the TRL library [von Werra et al., 2022] with the Qwen3 / Gemma architectures from transformers [Wolf et al., 2020]. We use full-parameter fine-tuning for SFT and WMSS unless a baseline explicitly states LoRA, AdamW optimization, a global learning rate of 1 × 10−5, maximum sequence length 4096, global batch size 512, gradient clipping at 1.0, and a 10% warm-up ratio. For WMSS, we set α = 0.1, β = 0.8, γ = 0.1, and use λ = 0.5 for the main runs unless otherwise stated; Appendix B.6 reports a finer sensitivity sweep around this region.

Per-domain training budget. All methods reported in Table 1 use the same base models and per-domain training budget unless an exception is stated below for rollout-heavy baselines. The total training budget is fixed to 2 epochs for the main SFT-style comparisons, including the SFT baseline and WMSS.

Training schedule and early stopping. The SFT baseline is trained for 2 epochs. WMSS uses the same 2-epoch total budget: 1 SFT warm-up epoch producing M1, followed by 1 weak-driven epoch

using WD-DS and WD-JT. This fixed budget is used consistently for the reported main comparisons rather than selected separately per benchmark. The convergence trajectories in Figure 3 are reported as extended diagnostics beyond the main comparison setting: pushing further can introduce overoptimisation on small benchmarks (e.g., AMC23) without improving the math average, consistent with the Stage III drift in our theoretical analysis.

Exceptions specific to individual baselines are noted in the corresponding paragraph below.

#### E.6 Baseline Hyperparameters

We report the key hyperparameters we used for each baseline; all runs share the same base models and training data as WMSS, and follow the per-domain epoch budget described in Appendix E.5 unless stated otherwise.

UNDIAL. We follow the original recipe and set the logit-smoothing scale σ = 1.0. Following common practice but not stated in the original paper, we normalize the injected noise by 1/

√

L (where L is the sequence length) to avoid amplification on long sequences, so that noise = σ ·ε/

√

L

with ε ∼ N(0,I). NEFTune. We adopt the paper-recommended embedding-noise magnitude α = 5.

SPIN. SPIN is DPO-based self-play. Per the 2-epoch allocation in Appendix E.5, we initialise from the SFT-epoch-1 checkpoint and then run 1 epoch of DPO on SPIN-generated preference pairs. Hyperparameters were tuned on a small grid and we ended up using: β = 0.1 (recommended by SPIN/TRL’s DPO and empirically preferred for iterative self-play, as low β allows the student to deviate further from the reference), learning rate 5 × 10−7 (DPO is highly learning-rate sensitive—

1×10−6 was unstable in our runs), sampling temperature 1.0 (following SPIN §4 for rejected-sample diversity), gradient-norm clip 0.5 (tighter than the TRL default of 1.0, which we found noticeably improves DPO stability) and a 10% warm-up ratio to prevent early learning-rate shocks to the reference model. Following the original SPIN protocol, each iteration trains for exactly 1 epoch before resampling.

SSB. SSB is a self-bootstrapping distillation baseline. For every prompt we first draw 4 rollouts from the current student m2 (sampling temperature 0.8, top-k 50) and retain one correct and one incorrect trajectory. A teacher (the same m2 conditioned on a hint-augmented prompt, temperature 0.4, max_model_len= 16384) then produces a corrective CoT. During optimisation we store the top-20 logits of each teacher token, which covers >95% of the softmax mass, and train the student with a LoRA adapter (r = 32, α = 32) using an equal mixture of KD (T = 1.5) and cross-entropy loss. Training uses lr = 1 × 10−4, 2 epochs, and a global batch size of 256 (1 × 32 × 8). For math, SSB’s rollout and teacher-generation phases are substantially more expensive than the other baselines, so we subsample to 20K examples while keeping the 2-epoch budget; this subsampling applies only to the SSB math setting.

SSB on Gemma-3-4B-PT. Because the Gemma pretrained model alone cannot produce coherent rollouts for the self-play stage, we additionally initialise the SSB student from a 1-epoch SFT checkpoint and then run one further SSB epoch on top (total budget still = 2 epochs). The Qwenseries bases (Qwen2.5-3B, Qwen3-4B-Base, Qwen3-8B-Base) do not require this warm-up and run SSB directly from the base model.

