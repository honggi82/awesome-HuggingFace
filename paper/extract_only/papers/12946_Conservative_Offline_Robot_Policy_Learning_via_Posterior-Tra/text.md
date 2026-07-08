# arXiv:2603.16542v1[cs.RO]17Mar2026

∝ ∝

∝

∝BeingBeyond

BeingBeyond

智在⽆界

智在⽆界

## Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting

### Wanpeng Zhang1,3 Hao Luo1,3 Sipeng Zheng3 Yicheng Feng1,3 Haiweng Xu1,3 Ziheng Xi2,3 Chaoyi Xu1,3 Haoqi Yuan1,3 Zongqing Lu1,3,†

1Peking University 2Tsinghua University 3BeingBeyond https://research.beingbeyond.com/ptr

### Abstract

Offline post-training adapts a pretrained robot policy to a target dataset by supervised regression on recorded actions. In practice, robot datasets are heterogeneous: they mix embodiments, camera setups, and demonstrations of varying quality, so many trajectories reflect recovery behavior, inconsistent operator skill, or weakly informative supervision. Uniform post-training gives equal credit to all samples and can therefore average over conflicting or low-attribution data. We propose Posterior-Transition Reweighting (PTR), a reward-free and conservative post-training method that decides how much each training sample should influence the supervised update. For each sample, PTR encodes the observed post-action consequence as a latent target, inserts it into a candidate pool of mismatched targets, and uses a separate transition scorer to estimate a softmax identification posterior over target indices. The posterior-to-uniform ratio defines the PTR score, which is converted into a clipped-and-mixed weight and applied to the original action objective through self-normalized weighted regression. This construction requires no tractable policy likelihood and is compatible with both diffusion and flow-matching action heads. Rather than uniformly trusting all recorded supervision, PTR reallocates credit according to how attributable each sample’s post-action consequence is under the current representation, improving conservative offline adaptation to heterogeneous robot data.

Date: Mar 17, 2026

### 1 Introduction

Pretrained vision-language-action (VLA) policies [1–3] provide a practical foundation for robot learning. Large-scale pretraining [4–8] encodes broad robot priors into a shared backbone, and supervised post-training adapts the policy to a target setting. This pipeline stays purely offline and keeps deployment simple.

Data heterogeneity is the core challenge. Large robot collections mix trajectories from different embodiments, camera viewpoints, control delays, and diverse teleoperators [9–11]. Even within one embodiment, operator skill varies: some demonstrations are near-optimal, while others contain recovery behaviors or hesitations. Across embodiments, similar images can correspond to different kinematic solutions. Logged action chunks are therefore multi-modal, with uneven quality and substantial suboptimal supervision.

†Correspondence to Zongqing Lu <lu@beingbeyond.com>.

Cross-embodiment mixtures also carry a latent positive-transfer potential. Different robots can demonstrate the same high-level skill and provide additional coverage of task-relevant progress, even when their low-level action chunks differ. Recent VLAs such as Being-H0.5 [8] enable this by mapping heterogeneous robots into a unified action space. The difficulty is to exploit the signal selectively without incurring negative transfer from embodiment-specific artifacts.

This paper proposes a simple idea: use observed post-action consequences as a reward-free signal for deciding which recorded chunks deserve more credit. Offline datasets record not only an action chunk but also what happens after it. PTR turns this observation into an identification test. Given the current policy representation and the recorded chunk, can the matched post-action consequence be identified among a pool of mismatched alternatives? Concentrated identification posteriors indicate attributable, high-quality chunks that receive more weight. Diffuse posteriors indicate ambiguous or suboptimal samples that are down-weighted. Conservative clipping and mixture constraints keep the induced distribution shift bounded.

When demonstrations are already consistent, PTR weights stay close to uniform and the method reduces to standard post-training. The gains come from reallocating credit along two axes: PTR raises the performance floor by suppressing suboptimal and conflicting supervision, and it raises the ceiling by selectively leveraging cross-embodiment coverage when post-action consequences align across sources.

Our contributions are threefold:

- • A reward-free sample scoring mechanism that converts post-action consequences into an identification posterior, whose log ratio to the uniform baseline measures how attributable each recorded chunk is to the current policy context.
- • A conservative weight mapping that bounds the induced distribution shift while preserving the original supervised action objective, with formal guarantees connecting the score to KL divergence and the weight to bounded density ratios.
- • Empirical validation on simulation benchmarks and 12 real-robot tasks across three embodiments, demonstrating the general effectiveness of PTR.

### 2 Related Work

Vision-language-action models. Vision-language-action (VLA) models unify vision encoders [12–14], language models [15, 16], and action decoders into end-to-end robot policies [17, 18]. Early systems such as RT-1 [1] and RT-2 [2] demonstrated that transformer-based architectures can learn generalizable robot control from large datasets. PaLM-E [19] showed that multimodal language models can ground in embodied tasks. Open-source generalist policies including OpenVLA [3] and Octo [20] have made VLA pretraining broadly accessible. Autoregressive VLAs tokenize actions and predict them sequentially [3, 21], while a growing family of models generates action chunks via continuous generative processes. π0 [4] and π0.5 [5] use flow matching [22, 23] as the action head. GR00T N1 [6] adopts a dual-system architecture with a DiT-based action generator. Diffusion Policy [24] applies denoising diffusion [25] to visuomotor control. Being-H0.5 [8] combines a Mixture-of-Transformers backbone with flow matching and introduces a unified action space that maps heterogeneous robots to shared semantic slots, enabling cross-embodiment pretraining. Large-scale cross-embodiment datasets [9–11] provide the data substrate for these models but also introduce the heterogeneity and suboptimal demonstrations that motivate PTR. PTR operates at the post-training stage of such systems and is compatible with both autoregressive and generative action heads.

Offline policy improvement and data reweighting. Standard behavioral cloning treats all demonstrations equally. Dataset composition and demonstration quality significantly affect imitation learning performance [26– 28]. Weakly supervised quality estimators [29], representation modulation [30], and mutual-information-based data curation [31] attempt to address this. A classical alternative is advantage-weighted regression (AWR) [32], which casts policy improvement as supervised learning with exponential weights exp(A/β). Reward-weighted regression [33], REPS [34], and MPO [35] share this exponential-weight structure. Reward-conditioned policies [36] and Decision Transformer [37] condition on returns rather than reweighting. PTR adopts the same exponential weight form as AWR but replaces reward-based advantages with a reward-free identification

reweight

PTR Core

Query Head

f

ot+!

query

q(x)

Belief Tokenizer

B

[Figure 1]

pD(x)

Target Encoder

zt → zt+1

Lωact

| | |
|---|---|
|y+<br><br>|y→|
|---|
<br><br>|y→|
|---|
<br><br>y→<br><br>y→| |

<belief>

at aˆt

q(x) → pD(x)w

VLM Backbone Action Expert

ht

T

#### Y

Candidate Pool

Posterior-Transition Reweighting

<proprio> Noise

<text> <vision>

In-batch Cross-rank

VLA Policy

Figure 1: Overview of PTR. Left: the standard policy stack (backbone + action expert) is augmented with a lightweight scorer and a BeliefTokenizer. Right: for each training chunk, the scorer identifies the matched post-action target among mismatched candidates; the resulting identification posterior is converted into a conservative weight that rescales the supervised action loss. No reward labels or policy likelihoods are needed.

score derived from post-action consequences. A growing line of work [38–40] applies reinforcement learning to VLA fine-tuning [41–43]. These methods require reward signals or online interaction [44–48]. PTR uses no reward, no value function, and no policy gradient; its connection to this literature is structural (the exponential weight form from KL-regularized optimization [49]) rather than algorithmic. The identification posterior builds on InfoNCE [50, 51] and causality assignment methods [52, 53]. The conservative constraints (clipping, mixture, self-normalization) mirror truncated importance weighting [54, 55] and self-normalized estimators [56] from the off-policy and offline RL literature [57–59]. PTR adapts these principles to reward-free supervised post-training with an identification-based score.

- 3 Preliminaries and Notation Robot dataset and training tuples.

Each sample from an offline dataset D is a five-tuple (ot,st,l,at:t+L−1,ot+∆). It contains visual observation ot, state st ∈ Rd

a, and future observation ot+∆. Here L is fixed, while ∆ may vary across samples. Only (ot,st,l) is used at inference; ot+∆ serves exclusively as a training-time target for the identification test.

s, instruction l, action chunk at:t+L−1 ∈ RL×d

VLA backbone and unified action space. Let fϕ denote a transformer backbone that maps (ot,l) to hidden states Ht and a pooled context ht. Being-H0.5 [8] maps heterogeneous robots into a shared 200-dimensional action space with sparse semantic slot assignments, so that similar motor components always occupy the same dimensions regardless of embodiment. PTR inherits this representation.

###### Action heads and post-training objective. The action head maps (ht,st) to aˆt:t+L−1 ∈ RL×d

a. For flow-matching heads [22, 23], the per-sample loss is

ℓact(ϕ;ht,st,at:t+L−1) = vϕ σat:t+L−1 + (1−σ)ϵ, σ, ht, st − (at:t+L−1 − ϵ) 2, (1)

where ϵ ∼ N(0,I) and σ ∼ p(σ). Diffusion heads [24, 25] admit a similar form. Uniform post-training minimizes

min

t,st,l,at:t+L−1)∼D ℓact(ϕ;ht,st,at:t+L−1) . (2)

E(o

ϕ

### 4 Posterior-Transition Reweighting

PTR overlays standard offline post-training with a conservative reweighting mechanism. A lightweight consequence encoder and transition scorer produce per-sample weights from observed post-action consequences, without requiring reward labels or a tractable policy likelihood. The section is organized along the data flow: we first describe the belief proxy tokens that summarize interaction history (Section 4.1), then the identification scorer that converts post-action consequences into a per-sample quality signal (Section 4.2), followed by the theoretical foundations that justify reading this signal as a density ratio and KL divergence (Section 4.3). We then present the conservative weight mapping that bounds distribution shift (Section 4.4), the adaptive controller that keeps the scorer in a stable operating range (Section 4.5), and the practical training pipeline (Section 4.6).

##### 4.1 BeliefTokenizer

PTR maintains M compact belief proxy tokens zt ∈ RM×d that are appended to the backbone input. These tokens summarize pre-action interaction history and help define what counts as a similar context under partial observability. For a segment starting at t = t0, the initial tokens zt

= zinit are learned. At each chunk index t ≥ t0, the forward pass produces:

0

(Ht,ht) = fϕ(ot,l,zt), (3)

Et = gϕ(ht,st,at:t+L−1) ∈ RL×d, et = pool(Et), (4) zt+1 = sg(B(Ht,Et)). (5)

Here Ht denotes token-level backbone hidden states, ht is a pooled context representation, Et is the sequence of action-channel tokens, and et is its pooled summary used by the scorer. The BeliefTokenizer B compresses current-step features into next-step tokens via soft causal assignments. The stop-gradient on zt+1 blocks gradients through time; the tokenizer learns from current-step losses only. An adaptive scale controller monitors identification statistics and adjusts the scorer temperature τscore, the advantage scaling β, and the hard-negative ratio within fixed bounds to keep training stable (Section 4.5).

Soft causal tokenization. For a chunk t, let Ct ∈ RL×d denote per-step context features and At ∈ RL×d the corresponding action-channel features. In code, Ct corresponds to transformer hidden states on the action-token positions and At to the action embeddings used by the action head; d is the hidden size of that action-channel representation. The tokenizer compresses these L per-step features into M belief proxy tokens (L = 16, M = 4). It first fuses the two streams:

ct,i = tanh Wf[Ct,i;At,i] ∈ Rd, i = 1,...,L, (6) then computes assignment logits for M slots, normalized over time:

exp(ξt,i,m/τtok)

. (7)

ξt,i,m = (Wact,i)m, πt,i,m =

L j=1 exp(ξt,j,m/τtok)

The merged belief tokens are weighted averages:

zt+1,m =

We also reconstruct per-step features as

L

πt,i,m ct,i, m = 1,...,M. (8)

i=1

M

c˜t,i =

πt,i,m zt+1,m. (9)

m=1

The recursion in Eq (3) passes zt+1 to the next chunk with stop-gradient.

Tokenizer regularizers. Two auxiliary losses prevent degenerate tokenizer behavior. An entropy term encourages each slot to attend decisively rather than spreading weight uniformly. Let Πt ∈ RL×M stack πt,i,m. The average entropy of each slot’s distribution over time is

1 M

M

L

Htok = −

πt,i,m log(πt,i,m). (10)

m=1

i=1

Adding Htok to the loss with a positive coefficient encourages the tokenizer to form more decisive groupings. To prevent collapse where multiple slots attend to the same subset of time steps, we penalize the slot Gram matrix:

Dtok = Π⊤t Πt − I 2F . (11) The combined tokenizer loss is Ltok = λentHtok + λdivDtok.

##### 4.2 Posterior transition score

With belief proxy tokens providing a compact summary of history, PTR builds a reward-free quality signal from an identification posterior over post-action consequences. The posterior here refers to a softmax distribution over candidate targets in a finite pool, not a Bayesian posterior or a predictive dynamics model.

Post-action targets. PTR encodes the observed post-action observation into the matched latent target yt,+0 = sg(g(ot+∆)), where g is a momentum (EMA) target encoder (distinct from the action-channel head gϕ in Eq (3)). The motivation is to trace the causal effect of actions from future consequences [52, 53]. PTR works in a latent target space rather than raw pixels. Reweighting only needs a compact representation that makes consequences distinguishable for identification. We reuse an intermediate layer of the policy’s own vision tower and maintain it with EMA, following momentum encoders in contrastive learning [60, 61]. A frozen target space becomes misaligned as the policy representation evolves; a fully online target is unstable. EMA is a stable compromise.

Concretely, the target encoder g extracts features from vision layer 12 of InternViT-300M-448px and is updated via exponential moving average with decay µ = 0.999: g ← µg + (1 − µ)v. All target features are L2-normalized before entering the candidate pool. Post-action targets are always stop-gradient features; the future observation is never fed back into the action policy as an additional input, keeping PTR in the offline post-training regime.

Candidate pool. PTR forms an ordered candidate set Yt = {yt,+0} ∪ Yt− with Yt− = {yt,−1,...,yt,K− }, where yt,−1,...,yt,K− are mismatched targets from other samples. These are target replacements, not trajectory splicing.

For each minibatch, we compute matched targets yt,+0 for valid samples and draw mismatched targets from three sources: (i) in-batch targets from other samples, (ii) cross-rank gathered targets from other GPUs, and (iii) a FIFO queue storing targets from previous iterations. All targets are treated as constants for the current update; the queue and gather are non-differentiable and exist only to enlarge the candidate pool. Negatives are formed after removing the current sample’s matched target, so the scorer must identify the correct post-action consequence against genuinely mismatched alternatives. When the refiner increases the hard-negative ratio, harder samples are mixed into the same pool rather than handled by a separate objective. In the default configuration, the FIFO queue holds 1024 entries, each minibatch draws up to 64 queue negatives per sample, and targets are gathered across all 8 GPUs via a non-differentiable all_gather before pool construction.

When a chunk lacks a valid post-action observation, we omit it from the scorer-side losses and use the conservative fallback wt = 1, so the sample contributes exactly as in uniform post-training.

Identification posterior. The scorer forms a query embedding ut = f(ht,et) from the current representation (ht,et) in Eq (3), using a lightweight projection head f (distinct from the backbone fϕ). It computes a cosinesimilarity logit against each candidate, dt,i := ⟨norm(ut), norm(yt,i)⟩/τscore, and defines the identification posterior, where It ∈ {0,...,K} indexes the candidate believed to be the matched target:

exp(dt,0) K j=0 exp(dt,j)

. (12)

pˆ(It = 0 | ht,et,Yt) =

This posterior has the same form as InfoNCE identification objectives [50, 51]. In our implementation, the action head already computes Et and we set et = pool(Et); the figure shows at:t+L−1 for clarity.

The scorer conditions on an explicit action channel. In code, et is a pooled representation of the action-channel tokens already used by the action head in the same forward pass, so PTR does not introduce a second action encoder. The notation gϕ in Eq (3) should be read purely as the action-channel projector used by the action head; it is unrelated to the EMA target encoder g.

To prevent the scorer from collapsing into a context-only shortcut, we add an action-sensitivity regularizer. The projection from the pooled action summary into the scorer’s query space is a two-layer MLP with Xavier initialization on both layers. Let u+t = f(ht,et) and u−t = f(ht,e˜t), where e˜t is obtained by permuting action features within the minibatch. Let d+t,0 and d−t,0 denote the matched-target logits computed with u+t and u−t respectively. The ranking loss is

Lrank(θ) = −Et log σ d+t,0 − d−t,0 . (13)

PTR score. We define the posterior-to-uniform ratio as

pˆ(It = 0 | ht,et,Yt) 1/|Yt|

. (14)

Tt ≜ log

If the posterior is uniform over the candidate pool, then Tt = 0 and the sample falls back to uniform supervision. If the posterior is concentrated on the matched target, then Tt > 0. Because the score is produced by a separate scorer, PTR does not require the policy itself to expose a likelihood and remains compatible with flow-matching action heads.

Natural suppression of suboptimal demonstrations. Robot datasets inevitably contain suboptimal trajectories: recovery behaviors, hesitant motions, or demonstrations from less-skilled operators. For such samples the post-action observation ot+∆ is often less distinctive under the pre-action context (ht,et), so the consequence becomes harder to attribute to the recorded chunk. The identification posterior therefore spreads across the candidate pool, yielding a low or negative PTR score: ambiguous samples stay near uniform weight, while clearly counter-evidential ones are down-weighted. In contrast, high-quality demonstrations produce distinctive post-action consequences that concentrate the posterior, resulting in Tt ≫ 0 and higher credit. This mechanism provides a floor: PTR removes extra emphasis from suboptimal data and down-weights it whenever the matched target becomes less likely than the pool-average alternative.

##### 4.3 Theoretical foundations

The PTR score defined above is an empirical quantity computed from a finite candidate pool. This subsection establishes its theoretical grounding by following the mathematical dependency chain: we first formalize the candidate-set identification model, show that Bayes-optimal logits recover a density ratio (Proposition 1), use this to connect the PTR score to a KL divergence (Proposition 2), derive the exponential weight form from a KL-regularized objective, and analyze how tilting reallocates weight across data sources (Proposition 3). Formal proofs of all three propositions are collected in Appendix A.

Candidate-set model. All theoretical results rest on a common probabilistic model of the identification task. The model is standard in the contrastive learning literature [50, 51] and is included here to fix notation.

Fix a context representation (h,e) and a baseline target distribution p−(y | h) = pN(y | h). Assume the positive distribution p+(· | h,e) is absolutely continuous with respect to p−(· | h) on the support induced by the candidate pool, so the density ratio r(y) = p+(y | h,e)/p−(y | h) is well-defined there. First draw a candidate position I uniformly from {0,...,K}. Then sample the matched candidate YI from p+(y | h,e) := p(y | h,e) and sample every mismatched candidate Yj (j ̸= I) independently from p−(y | h). The ordered training view used by PTR is obtained by conditioning on I = 0, so the matched target sits at index 0 and the remaining K entries act as negatives. The distribution p− is the population counterpart of the practical pool-construction rule described in Section 4.2; it can absorb any fixed mixture of in-batch, cross-rank, queued, or harder same-task negatives.

A scorer produces logits s(h,e,y) and induces the identification posterior

exps(h,e,Y0) K j=0 exps(h,e,Yj)

pˆ(I = 0 | h,e,Y ) =

. (15)

Density-ratio form of optimal logits. The Bayes-optimal scorer recovers a log density ratio between the action-conditioned and baseline target distributions. This result underpins the KL and entropy interpretations and clarifies why the PTR score can serve as a meaningful quality signal even though it is computed from a finite candidate set.

Proposition 1 Bayes-optimal logits recover a density ratio

Under the candidate-set model above, Bayes-optimal shared per-candidate logits for the identification task in Eq (15) can be written as

p(y | h,e) pN(y | h)

s⋆(h,e,y) = log

+ b(h,e), (16) where b(h,e) does not depend on y.

This proposition has two practical consequences. First, the identification scorer is not learning an arbitrary discriminative function: at optimality, the logits recover a principled statistical quantity (the log density ratio) that measures how much the action changes the distribution over future observations. Second, the additive constant b(h,e) cancels in the softmax posterior of Eq (15), so the PTR score depends only on the density ratio induced by the chosen baseline pool and not on any candidate-independent offset.

KL and entropy views of the PTR score. With the density-ratio form in hand, we can relate the population PTR score to a KL divergence. For fixed (h,e), define p+(y) = p(y | h,e), p−(y) = pN(y | h), and r(y) = p+(y)/p−(y). By Proposition 1, the Bayes-optimal identification posterior takes the form

r(Y0) K j=0 r(Yj)

p⋆(I = 0 | h,e,Y ) =

. (17)

Under the bounded-ratio regularity condition stated below, the law of large numbers drives the denominator toward its expectation (which equals one under p−), so the score converges pointwise to log r(Y0). Taking expectations over Y0 ∼ p+ then recovers KL(p+∥p−), which is the content of the following Proposition 2.

- Proposition 2 Large-candidate limit yields a KL score

Let p+(y) = p(y | h,e) and p−(y) = pN(y | h) denote the positive (action-conditioned) and negative (baseline, drawn from the candidate pool) target distributions, and let T⋆ be the PTR score under the Bayes-optimal identification posterior. Assume in addition that on the common support there exist constants 0 < c ≤ C < ∞ such that c ≤ r(y) = p+(y)/p−(y) ≤ C. As the candidate-set size K → ∞,

E[T⋆ | h,e] −→ KL p+(y)∥p−(y) . (18)

Samples whose action makes the next observation highly distinguishable from random alternatives have large

- KL(p+∥p−) and receive high PTR scores. Suboptimal or noisy actions blur this distinction, yielding low or negative scores. Proposition 2 is a population lens for a fixed baseline p−. In practice, finite candidate sets, cross-rank reuse, and hard-negative mining replace p− by an empirical mixture that varies slowly over training. The equality in Eq (18) is therefore not asserted sample-by-sample; it explains what quantity the learned score approaches when the pool is large and the sampling rule is stable.

How the score behaves in practice. Because Tt is centered against the uniform posterior over the realized candidate pool, ambiguous samples concentrate near 0. Samples whose matched target is systematically less supported than the pool average can have Tt < 0 and, after exponentiation and clipping, receive weights below one. Operationally, PTR performs a conservative reallocation of credit: clear samples are amplified, ambiguous samples revert toward uniform, and strongly counter-evidential samples are suppressed.

Entropy view under a uniform baseline. If targets are indexed by a finite universe Ω and the target embedding is a lossless encoding of the latent target identity U, then, in the same large-candidate limit, choosing a uniform baseline pN(U | h) = Unif(Ω) gives

E[T⋆ | h,e] −→ log |Ω| − H(U | h,e).

This recovers the entropy interpretation: samples with low conditional entropy (highly predictable consequences) receive high PTR scores.

Compatibility with generative action heads. The density-ratio and entropy/KL lenses apply to the score estimator (the identification classifier), not to the policy. They do not require the policy to provide a likelihood over actions, which is why PTR is compatible with diffusion and flow-matching action heads.

KL-regularized tilting yields exponential weights. PTR’s exponential weight wt = exp(Tt/β) arises as the solution to a KL-regularized score-maximization problem (Eq (21)). This is a classical result in the policy search literature [33–35]; we reproduce the short derivation here to make the paper self-contained and to clarify the role of the temperature β.

Consider a base distribution pD(x) and a measurable score function J(x) such that the partition function Zβ := Ep

[exp(J(x)/β)] is finite. The KL-regularized tilting objective is max

D

Ex∼q[J(x)] − β KL q∥pD . (19)

q : q≪pD

To see that the optimizer has the form q⋆(x) ∝ pD(x)exp(J(x)/β), write the Lagrangian with multiplier λ for the normalization constraint:

q(x) pD(x)

L(q,λ) = q(x)J(x)dx − β q(x)log

dx + λ 1 − q(x)dx .

Taking the functional derivative with respect to q(x) and setting it to zero:

q(x) pD(x) − λ = 0.

J(x) − β 1 + log

Rearranging gives log q(x) = log pD(x) + J(βx) + c, where c collects constants. Exponentiating yields q⋆(x) ∝ pD(x)exp(J(x)/β). Instantiating J(xt) = Tt for PTR recovers the weight wt from Eq (21).

The temperature β controls the trade-off between score maximization and proximity to the original data distribution. As β → ∞, the KL penalty dominates and q⋆ → pD (uniform weighting, equivalent to standard SFT). As β → 0+, the tilted distribution concentrates on the highest-scoring samples. PTR uses this exponential form as the raw score-to-weight map and then applies clipping and mixture as in Eq (21). Within the unclipped regime this preserves the ordering induced by Tt; outside it, extreme ratios are intentionally flattened to enforce conservatism. PTR operates in an intermediate regime where β is adapted online by the adaptive scale controller (Section 4.5) to maintain a meaningful but bounded weight range.

Selective transfer from cross-embodiment data. The exponential tilting derived above also induces a principled reallocation of effective weight across data sources. Cross-embodiment and multi-operator datasets are naturally modeled as mixtures of source-specific distributions. PTR provides both a floor and a ceiling for cross-embodiment learning. The floor suppresses mismatched or suboptimal pooled data: when a helper embodiment’s trajectories produce post-action observations that are uninformative for the target embodiment, the posterior stays diffuse and the sample loses extra emphasis, reverting toward uniform weighting and sometimes below it after exponentiation and clipping. The ceiling amplifies useful transfer: when helper data cover task-relevant regions that the target embodiment lacks, the posterior sharpens and PTR allocates more credit there. The following result, proved in Appendix A, formalizes this selective pooling at the source level.

Proposition 3 Source reweighting under exponential tilting Let pD(x) = m πmpm(x) be a mixture over M sources. Assume also that Ep

[exp(J(x)/β)] < ∞ for every source m. Tilting by exp J(x)/β yields

m

exp(J(x)/β) j πj Ep

πm Ep

. (20)

q⋆(m) =

m

exp(J(x)/β)

j

Sources whose samples consistently receive high PTR scores gain effective weight in the induced training distribution; sources with low scores are suppressed. The unified action space of Being-H0.5 [8] ensures that cross-embodiment actions share a common representation, making the identification signal meaningful across embodiment boundaries.

##### 4.4 Conservative reweighting and the induced training distribution

Conservative weight mapping. PTR maps the score to a conservative per-sample weight via exponentiation, clipping, and a convex mixture:

wt := 1 + α clip[w

min,wmax] exp(Tt/β) − 1 , α ∈ [0,1]. (21)

The exponential form is the optimizer of the KL-regularized tilting objective derived in Section 4.3. Clipping and mixture control variance and bound distribution shift, similar in spirit to truncated importance weighting [54, 55].

Density ratio bound. Define the implicit distribution q(x) ∝ pD(x)w(x). If w(x) ∈ [w,w] for all x, then the density ratio is bounded:

q(x) pD(x) ≤

w Ep

w Ep

. (22)

[w] ≤

[w]

D

D

Under the mixture form in Eq (21) with α ∈ [0,1] and clipping w ∈ [wmin,wmax], the effective bounds are w = 1 + α(wmin − 1) and w = 1 + α(wmax − 1). Setting rmax = w/w, the KL divergence satisfies

- KL(q∥pD) ≤ log rmax. With the default PTR parameters (wmin = 0.25, wmax = 4.0, α = 1), this gives KL(q∥pD) ≤ log 16 ≈ 2.77 nats. This bound is the formal justification for calling PTR “conservative”: the induced training distribution can never drift far from the original data distribution, regardless of the score function.

Clipping and mixture bound: detailed derivation. Define the implicit distribution q(x) ∝ pD(x)w(x) from Eq (28), where w(x) is the clipped-and-mixed weight. If w(x) ∈ [w,w] for all x, then the density ratio is bounded:

q(x) pD(x) ≤

w Ep

w Ep

. (23) Under the mixture form in Eq (21) with α ∈ [0,1] and clipping w ∈ [wmin,wmax], the effective bounds are

[w] ≤

[w]

D

D

w = 1 + α(wmin − 1), w = 1 + α(wmax − 1). (24) In the minibatch implementation, Ep

[w] is replaced by the empirical denominator t wt from Eq (27); the same pointwise bounds hold once all per-sample weights lie in the prescribed interval. By definition, q(x) = pD(x)w(x)/Z with normalizer Z = Ep

D

[w]. Dividing both sides by pD(x): q(x) pD(x)

D

w(x) Z

=

. (25)

Since w(x) ∈ [w,w] for all x, and Z is a fixed positive constant, the ratio is bounded by w/Z from below and w/Z from above, yielding Eq (23).

KL bound. Setting rmax = w/w, the pointwise bound implies

q(x) pD(x) ≤ log rmax = log

w w

KL(q∥pD) = Eq log

. (26)

With the default PTR parameters (wmin = 0.25, wmax = 4.0, α = 1), we obtain KL(q∥pD) ≤ log 16 ≈ 2.77 nats. When α = 1, the mixture form reduces to pure clipping, so w = wmin and w = wmax. Smaller α tightens the bound further toward zero. The general α ∈ [0,1] formulation in Eq (24) is retained because it clarifies the continuum between uniform SFT (α = 0) and full PTR reweighting (α = 1), and because practitioners working with smaller or noisier datasets may benefit from intermediate values.

The clipping range [wmin,wmax] and mixture coefficient α are the two user-facing knobs that control the conservatism level. In the limit α → 0, PTR recovers uniform SFT with q = pD.

Self-normalized weighted regression. PTR applies the weight to the original supervised action loss:

sg(wt) ℓact(ϕ;ht,st,at:t+L−1) t sg(wt)

L⋆act = t

. (27)

The stop-gradient on wt blocks the path where the policy could increase its own weights by manipulating the scorer. The scorer still learns from its own identification loss. Eq (27) also gives a direct distribution view. Let xt denote the full chunk-level sample excluding the post-action observation, e.g.xt = (ot,st,l,at:t+L−1). Self-normalized weighting makes the gradient update equivalent to sampling from an implicit distribution

q(x) ∝ pD(x)w(x). (28)

Clipping and mixture bound how far q can deviate from pD (Eq (22)). This is the operational meaning of conservative in PTR.

Mixture lens for selective pooling. Proposition 3 formalizes how tilting reallocates weight across sources: setting J(xt) = Tt (the PTR score), the ratio q⋆(m)/πm is proportional to the moment-generating function Ep

[exp(Tt/β)] of the score distribution within source m. Sources whose samples consistently yield high PTR scores have large moment-generating values and gain effective weight. Sources dominated by ambiguous samples remain closer to their original proportion, while sources with persistently low or negative scores are suppressed. This provides a formal mechanism for the “floor and ceiling” described above: the floor suppresses low-quality or mismatched sources, while the ceiling amplifies sources that provide informative coverage for the target embodiment.

m

Conservative bounds at the source level. In practice, PTR applies clipping and mixture (Eq (21)), so the effective per-sample weight satisfies w(x) ∈ [w,w]. The induced source proportion becomes

πm Z

q(m) =

[w(x)], (29)

Ex∼p

m

where Z = Ep

[w]. Because every per-sample weight lies in [w,w], the ratio q(m)/πm = Ep

[w]/Z is bounded:

D

m

q(m) πm ≤

w w ≤

w w

. (30)

With the default PTR parameters (wmin = 0.25, wmax = 4.0, α = 1), this gives q(m)/πm ∈ [1/16,16], and the mixture coefficient α < 1 tightens this range further toward unity. No source can be amplified or suppressed by more than a bounded factor, regardless of how extreme its score distribution is.

Implementation considerations. Real datasets can record post-action observations at different offsets ∆. Large offsets can make identification harder and inject additional ambiguity. To remain conservative across offsets, we apply a discount γ∆

to the identification loss and, optionally, to the resulting weights, where ∆′ is a normalized offset and γ ∈ (0,1]. In large heterogeneous robot corpora, some mismatched targets can still be semantically close to the positive, especially under cross-embodiment pooling or repeated task motifs. PTR treats these cases conservatively: overlap lowers the identification margin and pushes the posterior toward uniform rather than creating spuriously large weights. Together with the fallback wt = 1 for missing targets, uncertainty primarily removes extra emphasis instead of fabricating it.

′

##### 4.5 Adaptive scale control

The identification posterior is informative only when it is neither saturated nor completely flat. The adaptive scale controller is a multi-system training feedback mechanism [62] that monitors identification statistics at each logging step and adjusts three parameters within fixed bounds to keep training in a stable operating range.

Monitored statistics. The refiner tracks exponential moving averages (EMA, decay 0.98) of four quantities: (i) identification accuracy (nce_acc), the fraction of samples where the matched target receives the highest logit; (ii) score margin (nce_margin), the gap between the matched logit and the best mismatched logit; (iii) mean PTR score (nce_adv); and (iv) the valid-target ratio, the fraction of samples with a usable post-action observation.

Adapted parameters. Based on these statistics, the refiner adjusts:

- • Scorer temperature τscore: raised when nce_acc falls below a lower threshold (to keep an immature scorer conservative), and lowered when it exceeds an upper threshold (to sharpen a mature scorer). Bounded in [τmin,τmax] = [0.03, 0.20].
- • Advantage scaling β: adjusted to keep the effective weight range meaningful. Larger β makes weights more conservative by compressing them toward one, while smaller β allows stronger differentiation once the scorer is reliable. Bounded in [βmin,βmax] = [0.5, 3.0].
- • Hard-negative ratio: the fraction of candidate-pool negatives drawn from semantically similar samples (same task family or similar proprioceptive state) rather than uniformly at random. Increased as the scorer matures (rising nce_acc) to maintain a challenging identification task. Bounded in [0, 0.5].

Update rule. Let T¯ denote the EMA-smoothed identification accuracy. The refiner applies two conditional branches at each logging step. If T¯ < 0.05 (scorer too weak), both τscore and β are multiplied by 1.01 to keep the posterior and weights conservative while the scorer is still organizing the candidate space. If T¯ > 0.35 and the score margin exceeds 0.10 (scorer confident and well-separated), both parameters are multiplied by 0.995 to sharpen the posterior and allow stronger differentiation between informative and ambiguous samples. The hard-negative ratio is set by linear interpolation of T¯ within [0.10, 0.50]: when T¯ ≤ 0.10 the ratio is 0.0 (all random negatives), and when T¯ ≥ 0.50 the ratio saturates at 0.5. All updates are clipped to the parameter’s allowed range after each step. These adjustments change only scalar sharpness and sampling parameters; for a fixed candidate set they can widen or narrow the posterior and the induced weight spread, but they do not alter which target is designated as matched.

Empirical behavior. Training dynamics of the belief tokenizer and adaptive scale controller are visualized in Figures 8–10. Two observations from the controller logs are worth noting. First, τscore decreases from its initial value to approximately 0.03 as the contrastive signal strengthens, sharpening the posterior. Second, the hard-negative ratio increases from 0.0 to 0.50 over training, indicating that the refiner progressively enriches the candidate pool with challenging negatives as the scorer matures.

##### 4.6 Practical pipeline and gradient routing

Modules. PTR augments a standard post-training stack with four lightweight components: an EMA target encoder g for post-action targets; a query head f; a candidate pool built from in-batch targets, cross-rank targets, and a FIFO queue; and a BeliefTokenizer B that produces belief proxy tokens.

Estimator losses. PTR trains the scorer with the identification loss

Lid(θ) = −Et log pˆ(It = 0 | ht,et,Yt) , (31)

and an action-sensitivity regularizer (the ranking loss Lrank defined in Section 4.2). The BeliefTokenizer uses entropy and diversity regularizers on its soft assignments (Section 4.1).

###### One-step optimization with routed gradients. The total objective is

Ltotal = L⋆act + λidLid + λrankLrank + Ltok. (32)

All trainable components are updated jointly. Stop-gradient markers block specific paths (Figure 1): (i) sg(wt) blocks the policy→scorer manipulation path, but the scorer still learns from Lid and Lrank; (ii) sg(g(·)) makes the post-action target encoder an EMA teacher rather than an online head; (iii) sg(zt+1) blocks gradients through time, while B still learns from current-step losses; (iv) the candidate queue and cross-rank gather are non-differentiable, but the scorer still receives gradients on the current query and the current matched target. Conservative weights wt are applied only to the action loss; the remaining terms receive uniform gradients.

Since wt is treated as a constant, PTR does not alter attention computations. It only rescales per-sample gradients. Let w¯t = sg(wt)/ j sg(wj). Then for any shared parameter block ϕ in the backbone or action head,

∇ϕ L⋆act =

w¯t ∇ϕℓact(ϕ;ht,st,at:t+L−1). (33)

t

Transformer view. PTR assumes the backbone is a transformer that consumes a token sequence under a fixed attention mask (causal on language tokens, bidirectional within vision tokens). Belief proxy tokens are inserted as ordinary context tokens and follow the same mask semantics. PTR does not modify the mask; its only interaction with the transformer update is the gradient scaling in Eq (33).

Algorithmic summary. Algorithm 1 summarizes one PTR training segment. The outer loop iterates over chunks within a trajectory segment; the inner computation at each chunk produces a conservative weight wt that rescales the action loss gradient. All modules (scorer, BeliefTokenizer, action head) share a single forward/backward pass, so the algorithm adds no extra environment interaction or second-stage optimization.

Algorithm 1 Posterior-Transition Reweighting (PTR) for one segment

- 1: Initialize belief tokens zt

0 ← zinit

- 2: for t = t0,...,t0 + T − 1 do
- 3: Compute (Ht,ht) = fϕ(ot,l,zt)
- 4: Compute action loss ℓact(ϕ;ht,st,at:t+L−1) and action tokens Et = gϕ(ht,st,at:t+L−1); set et = pool(Et)
- 5: Compute post-action target yt,+0 = sg(g(ot+∆))
- 6: Build candidate set Yt = {yt,+0} ∪ Yt− from in-batch / cross-rank / queue
- 7: Compute query ut = f(ht,et) and posterior pˆ(It = 0 | ht,et,Yt)
- 8: Compute score Tt = log pˆ(It=0|ht,et,Yt) 1/|Yt|

- 9: Compute conservative weight wt = 1 + α(clip(eT

t/β) − 1)

- 10: Update belief tokens zt+1 = sg(B(Ht,Et))
- 11: end for
- 12: Update parameters by minimizing Ltotal in Eq (32) with sg(wt)

### 5 Experiments

##### 5.1 Setup and baselines

All methods build on Being-H0.5 [8], a state-of-the-art VLA whose 200-dimensional unified action space maps heterogeneous robots to shared semantic slots, making it a natural testbed for cross-embodiment posttraining. All methods share the same backbone, action head, and training budget (60k steps, batch size 128, cosine learning-rate schedule; full hyperparameter listing in Appendix B). Three configurations are compared throughout: (i) SFT sets α=0 in Eq (21), recovering uniform supervised post-training; (ii) SFT+Belief adds belief proxy tokens but keeps uniform weighting, isolating the effect of richer context from the effect of reweighting; (iii) PTR enables the full pipeline with α=1.

Two simulation benchmarks complement the real-robot evaluation. LIBERO [63] provides four task suites (Spatial, Object, Goal, Long-Horizon), each containing 10 tasks evaluated over 50 episodes per task (500 per suite, 2000 total). RoboCasa [64] provides 24 kitchen tasks evaluated over 50 trials across 5 scene layouts per task (1200 total).

Table 1: Standard simulation results (success rate %) on LIBERO and RoboCasa. LIBERO reports per-suite averages over 500 episodes each; RoboCasa reports category averages over 1200 total trials. Bold: best; Underlined: second best.

###### LIBERO RoboCasa

Method Spatial Object Goal Long Avg Pick&Pl. Door/Dr. Others Avg SFT 98.8 99.0 98.6 96.8 98.3 36.0 71.3 55.3 54.2 SFT+Belief 98.2 98.4 98.0 95.6 97.6 36.7 71.5 55.0 54.4 PTR 98.0 99.2 97.6 97.0 97.8 38.3 73.0 55.5 55.6

To stress-test robustness, we introduce four training-data corruptions applied before post-training: (i) Action Noise Injection (ANI) adds Gaussian noise ϵ ∼ N(0,σ2I) with σ=0.1 to 30% of trajectories, simulating teleoperation jitter; (ii) Trajectory Truncation (TT) randomly truncates 25% of trajectories to 40–70% of their original length, simulating incomplete demonstrations; (iii) Label Noise (LN) randomly reassigns language instructions for 20% of trajectories, simulating annotation errors; (iv) Combined applies all three simultaneously. All corruptions are applied only to the post-training data; the evaluation environments and their success criteria are left unchanged.

##### 5.2 Simulation benchmarks

###### (a) LIBERO Average

###### (b) RoboCasa Average

###### Success Rate (%)

###### Success Rate (%)

100

60

−0.5

+1.4

98.3

97.8

+1.4

55.6

+2.2

+4.0

96.2

54.2

+5.2

55

95.4

+2.4

52.6

+6.2

94.8

52

95

94

50.4

93.2

50

48.6

+5.6

+9.4

91.6

46.8

91

45.8

44.2

90

45

40

85.4

85

36.4

35

80

30

Standard Action Noise

Truncation Label Noise

Combined

Standard Action Noise

Truncation Label Noise

Combined

SFT PTR

SFT PTR

- Figure 2: Robustness under corrupted training data (success rate %). Colored badges show PTR−SFT deltas. PTR maintains higher performance across all corruption types.

Standard evaluation. Table 1 reports results on curated (uncorrupted) training data. On LIBERO, all three methods remain near ceiling (97.6–98.3%), indicating that PTR stays competitive when the training data are already clean. PTR leads on Object (99.2) and Long-Horizon (97.0), the two suites where additional selectivity appears most helpful.

On RoboCasa, where demonstrations span diverse kitchen scenes and object configurations, PTR outperforms SFT by 1.4 pp (55.6 vs. 54.2), with gains across all three task categories: Pick&Place (38.3 vs. 36.0), Door/Drawer (73.0 vs. 71.3), and Others (55.5 vs. 55.3).

SFT+Belief falls between the two on RoboCasa (54.4), indicating that belief tokens improve context quality but the reweighting signal adds further value on heterogeneous benchmarks. The gap between SFT+Belief and PTR (1.2 pp) isolates the contribution of the identification-based reweighting beyond what richer context alone provides.

[Figure 2]

[Figure 3]

[Figure 4]

- (a) Unitree G1 & LinkerBot O6 (b) PND Adam-U (c) FR3 & Inspire Hand

Figure 3: Real-robot platforms used for evaluation. (a) Unitree G1 with LinkerHand O6 dexterous hands.

- (b) PND Adam-U with bimanual dexterous manipulation and a movable head. (c) FR3 single-arm with Inspire dexterous hand.

##### 5.3 Robustness under corrupted training data

Figure 2 compares PTR and SFT under the four corruption protocols. Under Action Noise Injection, SFT drops by 5.1 pp on LIBERO while PTR drops by only 2.4 pp, because noisy action chunks produce atypical post-action consequences that the identification posterior tends to de-emphasize.

Trajectory Truncation and Label Noise show a similar pattern. On LIBERO, SFT loses 6.7 pp and 3.5 pp, while PTR loses 3.8 pp and 1.6 pp. On RoboCasa, the corresponding drops are 10.0 pp and 5.6 pp for SFT versus 5.2 pp and 3.0 pp for PTR. The combined corruption is the most revealing: SFT loses 12.9 pp on LIBERO and 17.8 pp on RoboCasa relative to the clean baseline, whereas PTR loses 6.8 pp and 9.8 pp, yielding absolute gains of +5.6 pp and +9.4 pp respectively.

The pattern is consistent across both benchmarks: corrupted samples typically produce less concentrated identification posteriors and therefore lose relative emphasis, while clean samples with distinctive consequences are amplified. RoboCasa shows larger absolute gaps than LIBERO under all corruption types, reflecting its greater inherent heterogeneity in kitchen scenes and object configurations.

Table 2: Robot platform specifications for the three real-robot embodiments.

Platform DoF DoF breakdown Morphology Camera View Hz

Unitree G1 + LinkerHand O6 26 7×2 arms + 6×2 hands Bimanual, dexterous D435 Fixed ego 20 PND Adam-U 31 3 waist + 2 head + 7×2 arms + 6×2 hands Bimanual + head + waist ZED Mini Movable ego (dual) 20 FR3 + Inspire Hand 13 7 arm + 6 hand Single-arm, dexterous 2×D435 Fixed 3rd-person 50

###### 5.4 Real-robot evaluation and cross-embodiment transfer Three real-robot platforms span a range of embodiment complexity. Table 2 summarizes their specifications.

Unitree G1 + LinkerHand O6. A 26-DoF bimanual humanoid upper-body setup: 7 joints per arm (×2) and 6 joints per dexterous hand (×2), with a single fixed egocentric Intel RealSense D435 camera running at 20Hz. Its tasks emphasize dual-hand coordination, regrasping, and larger workspace transfers.

PND Adam-U. A 31-DoF bimanual platform: 3 waist joints, 2 head joints, 7 joints per arm (×2), and

- 6 joints per dexterous hand (×2), with dual ego views from a movable head-mounted ZED Mini stereo rig running at 20Hz. Compared with G1, it introduces stronger viewpoint drift due to head motion and longer multistage sequences enabled by the additional waist and head.

FR3 + Inspire Hand. A 13-DoF single-arm platform: 7 arm joints and 6 hand joints, with two fixed third-person Intel RealSense D435 cameras running at 50Hz. It stresses precise single-arm grasping, placement, and contact-sensitive manipulation under tighter kinematic constraints.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

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

[Figure 20]

Figure 4: Representative real-robot tasks across three platforms and four capability suites.

Table 3: Specialist and generalist training results (success rate %). Generalist trains a single checkpoint on all five sources (three real-robot sources plus LIBERO and RoboCasa); Specialist trains one checkpoint per source. “Overall” is the mean of the LIBERO average, the RoboCasa average, and the real-robot overall average. Bold: best; Underlined: second best per column.

Simulation Real Robot Method LIBERO RoboCasa Bimanual Long-Hor. Spatial Robust Overall

SFT-Specialist 98.3 54.2 55.0 63.3 75.0 50.0 71.1 PTR-Specialist 97.8 55.6 66.7 61.7 78.3 61.7 73.5 SFT-Generalist 96.2 50.8 45.0 51.7 63.3 40.0 65.7 PTR-Generalist 97.4 55.0 60.0 65.0 73.3 56.7 72.1

These three embodiments create the heterogeneity that PTR is designed to handle: identical semantic progress can arise from very different camera trajectories, contact patterns, and kinematic solutions.

Unified action space slot assignment. All three platforms share the 200-dimensional unified action space of Being-H0.5 [8], but each embodiment activates only the semantic slots corresponding to its available motor components. All platforms use the shared right-arm and right-hand semantic groups. G1 additionally activates the mirrored left-arm and left-hand groups, yielding 26 active dimensions in total. Adam-U further activates head and waist groups on top of the bimanual groups, yielding 31 active dimensions. FR3 uses only the single-arm plus dexterous-hand groups, yielding 13 active dimensions. All remaining slots are masked to zero during both training and inference, so the flow-matching head operates only in the embodiment-relevant subspace while the shared semantic layout still supports cross-embodiment transfer.

Evaluation protocol. Table 3 consolidates all results across simulation and real-robot benchmarks under both specialist (per-embodiment) and generalist (single-checkpoint) training. Each real-robot task is evaluated over 20 trials. 12 tasks across three platforms are grouped into four capability-oriented suites, each containing three tasks (60 trials per suite, 240 total): Bimanual (dual-arm coordination), Long-Horizon (multi-step sequential manipulation), Spatial (precise placement and arrangement), and Robust (generalization under scene variation and contact uncertainty). Figure 4 illustrates representative tasks from each suite. Suite labels follow the evaluation protocol rather than the task title alone, so semantically related tasks can appear in different suites across embodiments when their dominant difficulty differs.

Specialist results. Under per-embodiment training, PTR outperforms SFT on three of four real-robot suites. The largest suite-level gains appear on Bimanual and Robust, both at +11.7 pp (66.7 vs. 55.0 and 61.7 vs. 50.0 respectively), where operator variability and partial completions leave more room for the identification posterior to discriminate between informative and ambiguous chunks. Spatial improves by +3.3 pp (78.3 vs. 75.0). On Long-Horizon, SFT retains a slight edge (63.3 vs. 61.7), likely because sequential tasks exhibit lower operator variability and the identification signal provides less additional discrimination. Across all 12 tasks, PTR averages 67.1% versus 60.8% for SFT (+6.3 pp).

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 5: Cross-embodiment task correspondence. Different robot platforms (Adam-U & FR3) execute semantically similar manipulation tasks, illustrating the shared post-action structure that enables PTR to selectively transfer useful knowledge.

Cross-embodiment generalist training. Generalist training pools all five sources into a single checkpoint. Under SFT, this degrades the real-robot average by 10.8 pp due to embodiment conflicts. PTR-Generalist limits the drop to 3.3 pp, widening the PTR-vs-SFT gap from +6.3 pp (specialist) to +13.8 pp (generalist) on real robots. Two results highlight selective transfer. First, PTR-Generalist (65.0) surpasses SFT-Specialist (63.3) on Long-Horizon, showing that cross-embodiment data provide useful coverage for multi-step tasks when filtered by PTR. Second, on RoboCasa PTR-Generalist (55.0) approaches PTR-Specialist (55.6) with only a 0.6 pp drop, versus 3.4 pp for SFT. On LIBERO, PTR-Generalist (97.4) nearly matches SFT-Specialist (98.3), confirming that PTR extracts useful signal from cross-embodiment data even when the specialist already saturates. These results align with Proposition 3: PTR concentrates effective weight on samples whose post-action consequences overlap with the target embodiment, suppressing mismatched sources while amplifying informative ones. Figure 5 visualizes this overlap across platforms.

The specialist-versus-generalist comparison reveals two patterns. Even without cross-embodiment pooling, the specialist comparison still favors PTR on average, confirming that identification-based reweighting provides value beyond cross-embodiment transfer by suppressing ambiguous or low-quality samples within a source’s own data distribution. The PTR–SFT improvement is consistently larger in the generalist setting, where

- Proposition 3 provides explanatory power: helper samples from other sources receive high PTR scores only when their post-action consequences are identifiable under the target’s context representation.

Per-task breakdown. Table 4 breaks down the real-robot results into individual tasks. PTR improves over SFT on 8 of 12 tasks in the specialist setting, ties on 3, and trails on 1, confirming that the aggregate gain is broad rather than driven by a single outlier task. In the generalist setting, PTR improves over SFT on all 12 tasks. The generalist PTR checkpoint exceeds the specialist SFT checkpoint on 7 of 12 tasks and matches it on 4 more, demonstrating that selective cross-embodiment pooling can partially compensate for the dilution of multi-source training.

Task details and qualitative rollouts. Figure 6 shows example PTR rollouts for all 12 official real-robot tasks. The platform-specific task names are: Adam-U – Pour Water, Clear Desk, Arrange Flower, Handover, Drawer Organization, Wipe Board; G1 – Put & Close Box, Scan Package; FR3 – Drawer Organization, Water Plant, Wipe Board, Stack Bowl. Each row displays eight keyframes from a single successful trial under the generalist PTR checkpoint, with tasks grouped by platform.

The Bimanual suite tests dual-hand coordination. Pour Water on Adam-U requires one hand to hold the cup steady while the other tilts and pours, demanding precise bimanual force coordination to avoid spilling. Handover on Adam-U demands coordinated timing and grip force during object transfer, where the object must be securely grasped before the transfer hand releases. Put & Close Box on G1 requires both dexterous hands to place objects into a box and then close the lid, a sequence where the post-action observation changes substantially between the placing and closing stages.

Table 4: Per-task real-robot success rates (%).

Suite / Task Platform SFT-Spec PTR-Spec SFT-Gen PTR-Gen Bimanual

Pour Water Adam-U 60.0 70.0 50.0 65.0 Handover Adam-U 50.0 65.0 40.0 55.0 Put & Close Box G1 55.0 65.0 45.0 60.0 Suite average 55.0 66.7 45.0 60.0

Long-Horizon

Clear Desk Adam-U 70.0 65.0 60.0 70.0 Drawer Organization Adam-U 60.0 60.0 45.0 65.0 Scan Package G1 60.0 60.0 50.0 60.0 Suite average 63.3 61.7 51.7 65.0

Spatial

Arrange Flower Adam-U 80.0 85.0 70.0 80.0 Stack Bowl FR3 70.0 75.0 60.0 70.0 Water Plant FR3 75.0 75.0 60.0 70.0 Suite average 75.0 78.3 63.3 73.3

Robust

Wipe Board Adam-U 55.0 70.0 45.0 60.0 Drawer Organization FR3 50.0 60.0 40.0 55.0 Wipe Board FR3 45.0 55.0 35.0 55.0 Suite average 50.0 61.7 40.0 56.7

Overall average 60.8 67.1 50.0 63.8

The Long-Horizon suite chains multiple subgoals. Clear Desk on Adam-U involves sequential grasp-transportplace for all clutter objects, where the visual scene changes after each object removal and the chunk-level post-action signal can be harder to interpret than in single-stage placement tasks. Drawer Organization on Adam-U requires pulling open a drawer, placing objects inside, and closing it, a sequence where the post-action observation changes dramatically at each stage. Scan Package on G1 involves flipping a package to expose its barcode and then scanning it, requiring precise reorientation under dexterous hand control.

The Spatial suite tests placement accuracy. Arrange Flower on Adam-U demands accurate insertion of flowers into a narrow vase aperture, where even small positional errors cause the stems to miss. Stack Bowl on FR3 tests vertical alignment accuracy when stacking bowls onto a plate; the motion is relatively stereotyped across demonstrations, leaving less room for large gains from reweighting than in the more variable contact-rich tasks. Water Plant on FR3 requires grasping a sprinkling can and tilting it at the correct angle over the plant, where the pour trajectory must be spatially precise.

The Robust suite randomizes scene conditions. Wipe Board on Adam-U varies writing patterns and rag starting positions across resets; PTR shows one of the largest specialist per-task improvements here (+15% over SFT), consistent with its tendency to avoid over-emphasizing training chunks whose post-action consequences are ambiguous due to visual variability. Drawer Organization on FR3 shares the same semantic goal as its Adam-U counterpart but uses a different embodiment with different kinematics and camera viewpoints, testing cross-embodiment robustness directly. Wipe Board on FR3 similarly mirrors the Adam-U wiping task under a different morphology.

Two task titles appear on both Adam-U and FR3 (Wipe Board and Drawer Organization), directly testing cross-embodiment robustness under shared semantics. The semantics are shared but the raw trajectories are not, which is the regime PTR is explicitly designed to target. Suite membership follows the official evaluation protocol rather than task title alone, which is why semantically related tasks can contribute to different suites across embodiments.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- #3. Arrange Flower (Adam-U): Arrange the flowers neatly into the vase.

- #8. Scan Package (G1): Flip the package then scan the barcode with the scanner.

- #10. Water Plant (FR3): Use your hand to hold the sprinkling can and water the plants.

- #1. Pour Water (Adam-U): Pour the water from the orange cup into the blue cup.
- #2. Clear Desk (Adam-U): Put all the clutter on the desk into the basket.

- #4. Handover (Adam-U): Hand over the objects on the table into the box on the other side.
- #5. Drawer Organization (Adam-U): Pull open the drawer, put the objects inside, and then close it.
- #6. Wipe Board (Adam-U): Wipe the writing off the whiteboard with a rag.

- #11. Wipe Board (FR3): Wipe the writing off the whiteboard with a rag.

#7. Put & Close Box (G1): Put the two dexterous hands into the box and close the lid.

#9. Drawer Organization (FR3): Pull open the drawer, put the objects inside, and then close it.

- #12. Stack Bowl (FR3): Stack three bowls on the plate.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

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

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

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

###### Figure 6: PTR rollouts on all 12 real-robot tasks.

Unseen Obj/BG Flickering Lights

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Transparent Bottle

Figure 7: Illustrative hard out-of-distribution (OOD) evaluation with PTR. The scene contains unseen objects, a novel background, flickering overhead lighting, and a transparent bottle that challenges depth perception.

- Figure 7 presents a deliberately challenging OOD scene beyond the standard evaluation protocol, combining unseen objects and backgrounds, flickering overhead lighting, and a transparent bottle. The PTR rollout still completes the intended task sequence. This behavior is consistent with PTR’s design: samples with ambiguous or noisy post-action consequences receive less relative emphasis during training, so learning is biased toward demonstrations where the action-to-outcome link is clearer.

##### 5.5 Ablations

Table 5 isolates the contribution of each PTR component on the standard (uncorrupted) benchmarks. Weight clipping is the most critical component: removing it drops LIBERO by 2.3 pp and RoboCasa by 6.5 pp, confirming that bounding extreme weights is essential for stability. The EMA target encoder ranks second (−2.5/−5.4 pp), because a frozen target space cannot track the evolving policy representation. The adaptive scale controller contributes 0.7/3.7 pp by adjusting τscore and β online (Section 4.5). Cross-rank negative gathering and belief proxy tokens each provide moderate gains (0.8/1.3 pp and 0.4/1.6 pp) by enriching candidate-pool diversity and pre-action context quality. Setting α=0 recovers SFT+Belief, which trails PTR by 0.3/1.3 pp, consistent with Table 1.

The ablations separate three roles. Clipping and β control the scale of the gradient reallocation. The EMA target encoder keeps the scorer’s target space aligned with the evolving backbone. Belief tokens and richer negative pools improve the quality of the identification problem itself. The strong drops from removing clipping or EMA are consistent with PTR’s design: without bounded weights or a stable target space, the score-to-weight map becomes either stale or overly volatile.

##### 5.6 Training analysis

Training dynamics. Figure 8 tracks three core PTR quantities across six embodiment settings during 60k training steps. Two convergence groups emerge in identification accuracy (a): Cross-Emb., Adam-U, FR3, and G1 improve markedly earlier, while LIBERO and RoboCasa rise more gradually and reach their plateau later in training.

Table 5: Ablation study on standard benchmarks (success %). Bold: best; Underlined: second best.

###### LIBERO RoboCasa

Configuration Spatial Object Goal Long Avg Pick&Pl. Door/Dr. Others Avg PTR (full) 98.0 98.8 97.4 97.0 97.8 38.3 73.0 55.5 55.6

w/o belief tokens 97.6 98.4 97.6 95.8 97.4 36.8 71.0 54.2 54.0 w/o cross-rank gather 97.2 98.2 97.0 95.4 97.0 35.7 71.7 55.5 54.3 w/o EMA (frozen enc.) 95.8 96.6 95.2 93.4 95.3 33.0 67.3 50.5 50.2 w/o refiner 97.8 97.8 96.4 96.2 97.1 34.7 69.3 52.0 51.9 w/o clipping 96.2 97.0 95.6 93.0 95.5 32.3 66.0 49.3 49.1 α=0 (SFT+Belief) 97.4 98.6 98.0 96.0 97.5 36.7 71.5 55.0 54.3

The PTR score Tt (b) shows a similar pattern: the earlier-converging settings stabilize around 3.2–3.8, while the slower group reaches roughly 2.8–3.1. The cross-embodiment joint run achieves the highest final score (∼3.8), consistent with the richer candidate pool from five data sources. The action loss ℓact (c) decreases by over an order of magnitude for all settings, from 0.2–0.4 down to 0.005–0.012.

(b) PTR Score Tt

(c) Action Loss act

(a) Identification Accuracy

- 0

- 1

- 2

- 3

- 4

1.0

0.8

Accuracy

10 1

0.6

act

Tt

0.4

0.2

10 2

0.0

0 20 40 60

0 20 40 60

0 20 40 60

Training Steps (k)

Training Steps (k)

Training Steps (k)

Cross-Emb. LIBERO RoboCasa Adam-U FR3 G1

- Figure 8: PTR training dynamics across six embodiment settings (60k steps). (a) Identification accuracy shows two convergence groups; the legend applies to all three subplots. (b) PTR score Tt stabilizes at embodiment-dependent levels. (c) Action loss ℓact decreases steadily.

Belief tokens and identification margin. Figure 9 provides two complementary views. Belief token entropy Htok (a) drops steadily from roughly 0.3–0.5 toward a low plateau below 0.1, indicating that the soft causal tokenizer converges to compact slot assignments regardless of embodiment. The identification margin (b), the gap between the matched-target logit and the hardest negative, grows from near-zero values to approximately 1.9–7.5 depending on the setting. The cross-embodiment run achieves the highest final margin (∼7.5), consistent with diverse data providing more distinctive post-action signatures. LIBERO exhibits the lowest margin (∼1.9) despite reasonable accuracy, suggesting that its homogeneous task structure produces less separable target embeddings. Accuracy saturates once the positive is usually top-1 in the candidate set, whereas the margin can continue to grow as hard negatives become better separated.

Hyperparameter sensitivity. Figure 10 sweeps the three key hyperparameters (τ0, β0, wmax) on the cross-embodiment LIBERO setting. The scorer temperature τ0 (a) has a clear sweet spot near 0.12: smaller values make the posterior too sharp and reduce stability, while larger values over-flatten the posterior and weaken the score signal. The advantage scaling β0 (b) controls weight spread directly: very small values amplify score differences into unstable weights, while very large values compress weights toward uniformity and effectively disable reweighting. The clipping bound wmax (c) provides a complementary safety mechanism: wmax = 4.0 gives the best balance in this sweep, whereas larger values allow less controlled gradients from extreme weights. The default configuration (τ0=0.12, β0=1.5, wmax=4.0) achieves the best overall balance across the three sweeps shown here.

(a) Belief Token Entropy Htok

(b) Identification Margin

8

0.5

6

0.4

Margin

0.3

4

Htok

0.2

2

0.1

0

0.0

0 10 20 30 40 50 60

0 10 20 30 40 50 60

Training Steps (k)

Training Steps (k)

Cross-Emb. LIBERO RoboCasa Adam-U FR3 G1

- Figure 9: Belief token and identification margin dynamics. (a) Belief token entropy Htok converges to near-zero across all settings. (b) Identification margin grows steadily, with cross-embodiment training achieving the highest discriminability.

0.0 0.1 0.2 0.3

0

0.0

0.2

0.4

0.6

0.8

1.0

NormalizedMetric

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

(a) 0 sweep

1 2 3 4 5

0

0.0

0.2

0.4

0.6

0.8

1.0

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(b) 0 sweep

2 4 6 8

wmax

0.0

0.2

0.4

0.6

0.8

1.0

| |
|---|

| |
|---|

| |
|---|

(c) wmax sweep

Success Rate PTR Score Tt Stability (1/Std)

- Figure 10: Hyperparameter sensitivity on cross-embodiment LIBERO. Each subplot sweeps one parameter

with others fixed at defaults, tracking three normalized diagnostics: success rate, final PTR score Tt, and weight stability 1/Std(w). Dashed lines mark the default values. Legend is shared across all three subplots.

Weight evolution and loss reduction. Figure 11 visualizes how PTR’s sample weights evolve during training. Subplots (a)–(c) overlay the weight distribution at two snapshots: early stage (3k steps, grey) when the NCE scorer has just finished warmup and weights remain near-uniform around w=1, versus late stage (60k steps, colored) after the scorer has fully matured.

LIBERO (a), with relatively homogeneous simulation data, concentrates much of its mass near the clipping bound wmax=4.0 and retains only a thin suppressed tail near wmin. RoboCasa Human 50 shots (b), with more diverse scenes and noisier demonstrations, shows a visibly broader suppressed region. The real-robot generalist setting (c) exhibits the widest spread, reflecting cross-embodiment heterogeneity where kinematically mismatched demonstrations are less likely to receive large identification scores.

Subplot (d) shows the relative loss reduction (ℓSFT − ℓPTR)/ℓSFT over training. All three benchmarks show near-zero reduction during the 3000-step warmup and then diverge as the scorer activates. RoboCasa reaches the largest late-stage reduction, consistent with a noisier data distribution where PTR has more room to suppress uninformative samples. The real-robot generalist setting shows earlier onset of improvement, which is consistent with the richer candidate pool from cross-embodiment data.

(a) LIBERO

(b) RoboCasa

(c) Real-Robot

(d) Relative Loss Reduction

60

Early (3k) Late (60k)

Early (3k) Late (60k)

Early (3k) Late (60k)

Warmup

LossReduction(%)

- 0

- 1

- 2

- 3

- 0

- 1

- 2

- 3

- 0

- 1

- 2

- 3

| |
|---|

| |
|---|

| |
|---|

40

Density

20

LIBERO

RoboCasa

Real-Robot

0

0 2 4

0 2 4

0 2 4

0 10 20 30 40 50 60

Training Steps (k)

Weight w

Weight w

Weight w

- Figure 11: PTR weight evolution across benchmarks. (a)–(c) Sample weight distribution at early stage (3k steps, grey) vs. late stage (60k steps, colored) for LIBERO, RoboCasa, and real-robot generalist training. (d) Relative loss reduction (ℓSFT − ℓPTR)/ℓSFT over training.

Adaptive controller dynamics over training. The refiner’s controller signals reveal a consistent two-phase pattern. During the first ∼5k steps (including the 3000-step NCE warmup), τscore stays near its initial value of 0.12 and the hard-negative ratio remains at 0.0. Once the identification head begins producing meaningful gradients, τscore decreases steadily to approximately 0.03, reflecting a sharpening posterior. Concurrently, the hard-negative ratio ramps from 0.0 to approximately 0.50, and β shows a mild downward trend from 1.5 to approximately 1.2, consistent with the refiner allowing stronger weight differentiation as the scorer matures.

Ablation training curves. The variants without clipping (w/o clip) and without EMA (w/o EMA) exhibit noticeably less stable dynamics. Without clipping, the weight distribution develops heavy tails early in training: a small fraction of samples receive weights above 10, causing gradient spikes that manifest as periodic loss oscillations. Without EMA, the target space drifts with backbone updates, creating a moving-target problem for the scorer; identification accuracy fluctuates rather than converging monotonically, and the resulting weight signal is noisy. Both failure modes are consistent with the design rationale in Section 4.4.

### 6 Conclusion and Limitations

PTR is a conservative offline post-training method that reallocates credit across heterogeneous robot demonstrations without reward labels or a tractable policy likelihood. By scoring each sample through a candidate-set identification posterior over post-action consequences, PTR provides two complementary benefits: a floor that removes extra emphasis from ambiguous or noisy demonstrations and suppresses clearly counter-evidential ones, and a ceiling that amplifies useful cross-embodiment transfer when pooled data cover task-relevant regions.

The exponential weight mapping, clipping, and self-normalization keep the induced training distribution close to the original data, making the method conservative by construction.

Three limitations remain. PTR is most informative when post-action observations are available; chunks without a usable future observation revert to uniform weighting, and purely real-time streaming scenarios remain outside the method’s scope. The identification signal depends on the quality of the learned representation: a poorly pretrained backbone limits the scorer’s discriminability, and PTR falls back to near-uniform weighting. Finally, PTR improves the effective training distribution but does not directly optimize task success; it is a data-curation mechanism rather than a policy-optimization algorithm.

### References

- [1] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.
- [2] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.
- [3] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy

Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

- [5] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.
- [6] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [7] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597, 2025.
- [8] Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, et al. Being-h0.5: Scaling human-centric robot learning for cross-embodiment generalization. arXiv preprint arXiv:2601.12993, 2026.
- [9] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.
- [10] OX-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 1(2), 2023.
- [11] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.
- [12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [13] Yicheng Feng, Yijiang Li, Wanpeng Zhang, Sipeng Zheng, Hao Luo, Zihao Yue, and Zongqing Lu. Videoorion: Tokenizing object dynamics in videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20401–20412, 2025.
- [14] Hao Luo, Zihao Yue, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Deheng Ye, and Zongqing Lu. Openmmego: Enhancing egocentric understanding for lmms with open weights and data. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [15] Wanpeng Zhang, Zilong Xie, Yicheng Feng, Yijiang Li, Xingrun Xing, Sipeng Zheng, and Zongqing Lu. From pixels to tokens: Byte-pair encoding on quantized visual modalities. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=3TnLGGHhNx.
- [16] Wanpeng Zhang, Yicheng Feng, Hao Luo, Yijiang Li, Zihao Yue, Sipeng Zheng, and Zongqing Lu. Unified multimodal understanding via byte-pair visual encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12976–12986, 2025.

- [17] Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, and Zongqing Lu. Joint-aligned latent action: Towards scalable vla pretraining in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
- [18] Yicheng Feng, Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Sipeng Zheng, and Zongqing Lu. Spatial-aware vla pretraining through visual-physical alignment from human videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
- [19] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
- [20] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.
- [21] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.
- [22] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [23] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [24] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [26] Suneel Belkhale, Yuchen Cui, and Dorsa Sadigh. Data quality in imitation learning. Advances in neural information processing systems, 36:80375–80395, 2023.
- [27] Joey Hejna, Chethan Bhateja, Yichen Jiang, Karl Pertsch, and Dorsa Sadigh. Re-mix: Optimizing data mixtures for large scale imitation learning. arXiv preprint arXiv:2408.14037, 2024.
- [28] Ye Wang, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Haoqi Yuan, Chaoyi Xu, Haiweng Xu, Yicheng Feng, Mingyang Yu, Zhiyu Kang, et al. Rethinking visual-language-action model scaling: Alignment, mixture, and regularization. arXiv preprint arXiv:2602.09722, 2026.
- [29] Sachit Kuhar, Shuo Cheng, Shivang Chopra, Matthew Bronars, and Danfei Xu. Learning to discern: Imitating heterogeneous human demonstrations with preference and representation learning. In Conference on Robot Learning, pages 1437–1449. PMLR, 2023.
- [30] Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Yicheng Feng, Sipeng Zheng, Qin Jin, and Zongqing Lu. Dig-flow: Discrepancy-guided flow matching for robust vla models. arXiv preprint arXiv:2512.01715, 2025.
- [31] Joey Hejna, Suvir Mirchandani, Ashwin Balakrishna, Annie Xie, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Dhruv Shah, Coline Devin, and Dorsa Sadigh. Robot data curation with mutual information estimators. arXiv preprint arXiv:2502.08623, 2025.
- [32] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.
- [33] Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pages 745–750, 2007.
- [34] Jan Peters, Katharina Mulling, and Yasemin Altun. Relative entropy policy search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pages 1607–1612, 2010.
- [35] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. arXiv preprint arXiv:1806.06920, 2018.

- [36] Aviral Kumar, Xue Bin Peng, and Sergey Levine. Reward-conditioned policies. arXiv preprint arXiv:1912.13465, 2019.
- [37] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:15084–15097, 2021.
- [38] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025.
- [39] Yanjiang Guo, Jianke Zhang, Xiaoyu Chen, Xiang Ji, Yen-Jen Wang, Yucheng Hu, and Jianyu Chen. Improving vision-language-action model with online reinforcement learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15665–15672. IEEE, 2025.
- [40] Dongchi Huang, Zhirui Fang, Tianle Zhang, Yihang Li, Lin Zhao, and Chunhe Xia. Co-rft: Efficient fine-tuning of vision-language-action models through chunked offline reinforcement learning. arXiv preprint arXiv:2508.02219, 2025.
- [41] Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, et al. πrl: Online rl fine-tuning for flow-based vision-language-action models. arXiv preprint arXiv:2510.25889, 2025.
- [42] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.
- [43] Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What can rl bring to vla generalization? an empirical study. arXiv preprint arXiv:2505.19789, 2025.
- [44] Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, and Jiangmiao Pang. A vision-language-action-critic model for robotic real-world reinforcement learning. arXiv preprint arXiv:2509.15937, 2025.
- [45] Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Siwei Han, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. Grape: Generalizing robot policy via preference alignment. arXiv preprint arXiv:2411.19309, 2024.
- [46] Kevin Frans, Seohong Park, Pieter Abbeel, and Sergey Levine. Diffusion guidance is a controllable policy improvement operator. arXiv preprint arXiv:2505.23458, 2025.
- [47] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025.
- [48] Hongyin Zhang, Shuo Zhang, Junxi Jin, Qixin Zeng, Runze Li, and Donglin Wang. Robustvla: Robustness-aware reinforcement post-training for vision-language-action models. arXiv preprint arXiv:2511.01331, 2025.
- [49] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018.
- [50] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 297–304. JMLR Workshop and Conference Proceedings, 2010.
- [51] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
- [52] Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Schölkopf. Causal discovery from heterogeneous/nonstationary data. Journal of Machine Learning Research, 21

(89):1–53, 2020.

- [53] Wanpeng Zhang, Yilin Li, Boyu Yang, and Zongqing Lu. Tackling non-stationarity in reinforcement learning via causal-origin representation. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 59264–59288. PMLR, 2024.

- [54] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [55] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International conference on machine learning, pages 1407–1416. PMLR, 2018.
- [56] Adith Swaminathan and Thorsten Joachims. The self-normalized estimator for counterfactual learning. advances in neural information processing systems, 28, 2015.
- [57] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.
- [58] Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International conference on machine learning, pages 2052–2062. PMLR, 2019.
- [59] Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. Advances in neural information processing systems, 32, 2019.
- [60] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.
- [61] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.
- [62] Wanpeng Zhang and Zongqing Lu. Adarefiner: Refining decisions of language models with adaptive feedback. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 782–799, 2024.
- [63] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.
- [64] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

### A Proofs

This appendix collects the formal proofs of the three propositions stated in the main text.

##### A.1 Proof of Proposition 1 (Density-ratio form)

Proof. Let p+(y) := p(y | h,e) and p−(y) := pN(y | h). Under the symmetric candidate-set model from Section 4.3, conditioned on (h,e) and the full candidate set Y = (Y0,...,YK), the Bayes posterior for candidate position j is

p+(Yj) ℓ̸=j p−(Yℓ)

p⋆(I = j | h,e,Y ) =

K m=0 p+(Ym) ℓ̸=m p−(Yℓ)

p+(Yj) p−(Yj)

=

, (34)

p+(Ym) p−(Ym)

K m=0

where the second line divides numerator and denominator by Kℓ=0 p−(Yℓ). Taking posterior odds between any two candidate positions j and m gives

p+(Yj) p−(Yj) − log

p+(Ym) p−(Ym)

p⋆(I = j | h,e,Y ) p⋆(I = m | h,e,Y )

= log

. (35)

log

- A (K+1)-way softmax classifier represents exactly such pairwise log-odds via differences of logits. Therefore the shared scoring function must agree with log r(y) up to a candidate-independent additive constant: fixing any reference value yref on the common support gives s⋆(h,e,y) − s⋆(h,e,yref) = log r(y) − log r(yref), so

+(y)

s⋆(h,e,y) = log p

p−(y) + b(h,e) for some b(h,e) independent of y. Substituting back p+(y) = p(y | h,e) and p−(y) = pN(y | h) yields Eq (16).

| |
|---|

##### A.2 Proof of Proposition 2 (KL lens)

- Proof of Proposition 2 under the bounded-ratio regularity condition. By Proposition 1, Bayes-optimal shared per-candidate logits equal log r(y) up to an additive constant that cancels in the softmax. The corresponding identification posterior therefore takes the form

r(Y0) K j=0 r(Yj)

p⋆(I = 0 | h,e,Y ) =

. (36)

Define the empirical average

1 K+1

K

r(Yj).

AK :=

j=0

Substituting Eq (36) into the definition of the PTR score gives

p⋆(I = 0 | h,e,Y ) 1/(K+1)

T⋆ = log

= log r(Y0) − log AK. (37) Because each negative target Yj (j ≥ 1) is drawn from p−, the density ratio has baseline mean one:

p+(y) p−(y)

[r(Y )] =

p−(y)dy

EY ∼p

−

= p+(y)dy = 1. (38)

By the strong law of large numbers, the empirical average of the negative density ratios converges almost surely:

1 K

K

r(Yj) −→ 1 almost surely as K → ∞. (39)

j=1

The single matched term contributes at most C/(K+1), so AK → 1 almost surely; equivalently,

1 K+1

K

r(Yj) −→ 1 almost surely. (40)

j=0

Plugging Eq (40) into Eq (37) yields the pointwise limit T⋆ −→ log r(Y0) almost surely. (41)

Under 0 < c ≤ r(y) ≤ C < ∞, we have AK ∈ [c,C] for every K, hence |T⋆| ≤ log(C/c). Dominated convergence therefore applies, and taking expectations over Y0 ∼ p+ gives

E[T⋆ | h,e] −→ EY

0∼p+[log r(Y0)]

p+(y) p−(y)

dy = KL(p+∥p−), (42)

= p+(y)log

which is Eq (18).

| |
|---|

##### A.3 Proof of Proposition 3 (Source reweighting)

- Proof of Proposition 3. Assume the dataset is a mixture

pD(x) =

m

πmpm(x), (43)

where m indexes sources (embodiments, operators, or domains) and πm is the prior proportion of source m. Assume also that Ep

[exp(J(x)/β)] < ∞ for every source m, so the tilted source marginals are well-defined. Formally, let M be an observed source label and write the joint distribution as pD(x,m) = πmpm(x). Tilting by the score J(x) with temperature β produces the joint induced distribution q⋆(x,m) ∝ πmpm(x)exp(J(x)/β), whose source marginal is q⋆(m) = q⋆(x,m)dx. Let Z denote the normalizing constant. Using the joint form q⋆(x,m) = Z1 πmpm(x)exp(J(x)/β), the source marginal is

m

q⋆(m) = q⋆(x,m)dx

1 Z

πmpm(x)exp J(x)/β dx. (44)

=

Pulling out πm and rewriting the integral as an expectation gives q⋆(m) =

πm Z

pm(x)exp J(x)/β dx

exp(J(x)/β) Z

πm Ep

. (45)

=

m

By construction, Z = j πj Ep

[exp(J(x)/β)]. Substituting this expression for Z yields the closed-form expression

j

exp(J(x)/β) j πj Ep

πm Ep

, (46)

q⋆(m) =

m

exp(J(x)/β)

j

which is Eq (20).

| |
|---|

### B Training Configuration

Table 6 lists the complete hyperparameter configuration used for all PTR experiments. All experiments use these defaults unless a parameter is explicitly varied in a sensitivity sweep (Section 5.6).

Table 6: Complete hyperparameter listing for PTR post-training.

Parameter Symbol Value Optimization Optimizer – AdamW Learning rate η 10−4 Weight decay – 0.01 Schedule – Cosine with 2000-step linear warmup Total steps – 60k Global batch size – 128 Base policy Unified action dimension – 200 Action chunk length L 16 Flow-matching time-step sampling – Beta(1.5, 1.0) Inference denoising steps – 4 PTR scorer Identification loss weight λid 0.05 NCE warmup steps – 3000 Scorer temperature (init) τscore 0.12 Advantage scaling (init) β 1.5 Logit clamp – 20.0 Clipping lower bound wmin 0.25 Clipping upper bound wmax 4.0 Mixture coefficient α 1 Candidate pool FIFO queue size – 1024 Max queue negatives per sample – 64 Cross-rank gather – Enabled (8 GPUs) EMA target encoder EMA decay µ 0.999 Source layer – Layer 12 of InternViT-300M L2 normalization eps – 10−6 Belief tokenizer Number of belief tokens M 4 Tokenizer temperature τtok 1.0 Entropy regularizer λent 10−3 Diversity regularizer λdiv 10−3 Action sensitivity Ranking loss weight λrank 0.25 Adaptive scale control EMA momentum – 0.98 τscore bounds [τmin, τmax] [0.03, 0.20]

β bounds [βmin, βmax] [0.5, 3.0] Hard-negative ratio bounds – [0.0, 0.5]

