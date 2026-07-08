# arXiv:2606.02398v1[cs.LG]1Jun2026

## A Local Perturbation Theory for Cross-Domain Interference and Recovery in Multi-Domain RL

Lei Yang1 Siyu Ding2∗ Deyi Xiong1† 1TJUNLP Lab, College of Intelligence and Computing, Tianjin University, Tianjin, China 2Baidu Inc., Beijing, China yanglei_9@tju.edu.cn

### Abstract

Reinforcement learning (RL) post-training improves large language models (LLMs) on individual domains such as mathematical reasoning, code generation, question answering, and creative writing (CW), but training on one domain often degrades performance on others. Existing explanations based on catastrophic forgetting or global gradient conflict are incomplete: substantial interference can occur even when full-model gradients are nearly orthogonal. We show that single-domain RL produces sparse, small-magnitude parameter edits with weak overlap among topchanged neurons, while different domains still share substantial active computation routes on which update directions determine whether they act synergistically or conflict. Guided by this observation, we prove under a local perturbation model of multi-domain RL that later-domain training harms an earlier domain mainly through a second-order damage term, which under the observed sparse route structure concentrates in a low-dimensional shared conflict subspace. Moreover, a short domain refresh contracts the harmful component on this subspace, enabling selective recovery with limited collateral damage. Consistent with the theory, a brief Re-Math refresh after Code → Math → QA → CW recovers Math from 57.66 to 66.04 while largely preserving performance on the other domains, yielding the best average score of 66.39. Beyond refresh, a training-free rollback on a sparse proxy conflict coordinate set for the Math-QA pair partially restores Math, providing direct proxy-level evidence for localized damage. These results provide a localized mechanistic account of interference and recovery in multi-domain RL.

### 1 Introduction

RL has become a primary technique for improving large language models across reasoning, coding, question answering, and open-ended generation. In real-world settings, post-training should improve a model across multiple heterogeneous domains rather than produce single-domain experts. A natural approach is sequential multi-domain RL, where the model is trained on one domain after another.

However, a simple sequential curriculum already exposes a puzzling behavior. As shown in Table 1, following the Omni-Thinker [17] order, Code → Math → QA → CW, Math performance rises to 66.49 after Math training but drops to 57.66 after subsequent QA and CW training, while Code and QA remain relatively stable. This suggests that cross-domain interference is not uniform forgetting, but selective and asymmetric degradation. Existing explanations such as catastrophic forgetting [14, 28] or global gradient conflict [23, 18] are too coarse for this behavior. In our experiments, full-model gradient cosines can be close to zero even when substantial degradation occurs. Interference can

∗Project leader †Corresponding author

Preprint.

therefore remain invisible at the whole-model level. This leads to our central question: where does cross-domain interference reside if not in global gradient antagonism?

We answer this question by shifting the analysis from full-model objective conflict to the local active routes through which domain updates are expressed. This route-level view links three factors: where RL changes the model, which routes different domains activate at inference time, and whether update directions agree or conflict on shared routes. Although full-model gradient cosines are nearly orthogonal, layer- and modulelevel analysis reveals localized conflict and synergy. Domain RL induces sparse, small-magnitude edits with weak top-changed neuron overlap, ruling out the idea that domains mainly interfere by rewriting the same neurons. Yet reasoning-oriented domains still share substantial active-neuron overlap, so even sparse edits can interact through shared routes. On these shared edited routes, update direction separates positive transfer from interference.

Table 1: A motivating example under the OmniThinker curriculum. Sequential RL improves the current domain, but later stages selectively damage Math.

Codeo Matho QAo CWo Code 52.67 50.69 50.99 50.47 Math 59.63 66.49 59.90 57.66 QA 60.89 60.52 62.34 62.34 CW 82.40 81.44 81.79 86.52

Guided by these observations, we develop a local perturbation model of multi-domain RL. After training on a domain A, the selected checkpoint is approximately stationary for the local objective. Therefore, the leading effect of a later-domain update is a second-order damage term determined by whether the update enters curvature-sensitive directions of the earlier-domain objective. Together with the empirically observed sparse parameter updates and highly overlapping active routes, this suggests that degradation is largely concentrated in a sensitive, low-dimensional shared active conflict subspace. Therefore, near-orthogonal full-model gradients, sparse parameter changes, and low cross-domain edit overlap are insufficient to prevent cross-domain interference.

Based on this theory, we further show that a short refresh on domain A geometrically contracts the harmful component in this shared conflict subspace, thereby restoring the damaged domain while keeping collateral damage to other domains bounded. We validate this prediction by retraining Math from CWo: Math performance recovers to 66.04, close to the Math-domain expert, while Code, QA, and CW are largely preserved, leading to the best overall average score of 66.39. We further probe the localization claim by a training-free rollback on a sparse coordinate proxy for the conflict subspace in the Math–QA pair.

Our contributions are threefold:

- • We show that cross-domain degradation is not explained by global gradient conflict or direct overlap among edited neurons, but by sparse RL edits interacting along shared active computation routes.
- • We formalize this mechanism by showing that degradation is driven by second-order damage on a low-dimensional shared conflict subspace, and that short domain refresh yields selective recovery with limited damage.
- • We provide two complementary validations: short-refresh recovery at the task level, and a training-free targeted rollback on proxy conflict coordinates for a pairwise interaction.

### 2 Related Work

RL post-training and multi-domain RL mainly optimize aggregate performance, leaving cross-domain interactions inside the model poorly understood. Early RLHF systems build on human-preference RL [5], summarization with feedback [29], and PPO-style optimization [22]. Later work extends this setting to process supervision [19], mathematical reasoning [24], and large-scale reasoning RL [8, 31]. Recent multi-domain RL methods include Omni-Thinker [17] and CGPO [18]. The present work instead asks where cross-domain interference resides, why it is selective, and why a short domain refresh restores performance.

Cross-domain degradation is often discussed in terms of continual learning (CL), forgetting, or gradient conflict. Recent studies show that apparent forgetting can reflect shifts in alignment or policy behavior [40]. SFT–RL comparisons likewise suggest that RL may yield milder drift [6, 15, 25, 3],

Global Gradients: Math vs QA

0.02

Cosine

0.00

0.02

0 100 200 300 Training Step

(a) Global gradient cosine.

layers.30.attn QA vs Math mean=-0.0136 conflict=65%

0.10

0.05

Cosine

0.00

0.05

0.10

0 100 200 300

Step

(b) Most conflicting module.

layers.7.attn Code vs Math mean=0.0326 conflict=37%

0.3

0.2

Cosine

0.1

0.0

0.1

0 100 200 300

Step

(c) Most synergistic module.

Figure 1: Gradient relations between Math and QA at the global, attention and MLP levels.

with additional evidence for reasoning transfer [11]. Multi-task optimization offers a complementary global view through gradient balancing [4, 23] and gradient surgery [39, 20]. Our results suggest that full-model gradients can appear nearly orthogonal even when localized conflict harms earlier domains.

Task-vector and mechanistic work suggests that model changes can often be localized. In weight space, model soups [36] and TIES-style merging [38] attribute transfer and interference to redundant or sign-conflicting updates [21]. Mechanistic studies localize knowledge and skills to compact subsets of parameters [2] or neurons [7, 34, 16]. Recent feature-level analyses suggest that RL mostly preserves the base representation while strengthening compact task-relevant features [27]. We build on this locality view to study changed neurons, active routes, and direction-dependent interference.

### 3 Empirical Study Setup

To investigate where cross-domain interference arises and how recovery later occurs, we construct a controlled multi-domain RL setting. We use Qwen3-4B-Thinking-2507 [33] as the initial model and consider four domains: mathematical reasoning, code generation, question answering, and creative writing, denoted as Math, Code, QA, and CW, respectively.

#### 3.1 Data Construction

The Math training data is randomly sampled from OpenR1-math [12]. The Code training data is from KlearReasoner-CodeSub-15K [30]. The QA training data is sampled from SuperGPQA [32] by subfield and difficulty. The CW training data is from the crownelius/Creative-Writing series.3 Detailed data construction is described in Appendix A.1.

#### 3.2 Training Setup

All training runs are based on GRPO [24] implemented in VeRL [26]. Except for the training domain and the initialization checkpoint, all experiments share the same hyperparameters, reported in Appendix A.2. In addition, distinct reward functions are used for each domain, with detailed specifications in Appendix A.3.

We first train four single-domain expert models from the base model, denoted as Maths, Codes, QAs, and CWs. We then perform sequential multi-domain training. Following the Omni-Thinker curriculum [17], we use the fixed order Code → Math → QA → CW. The resulting checkpoints after the four stages are denoted as Codeo, Matho, QAo, and CWo, respectively. In sequential training, each stage continues from the checkpoint obtained in the previous stage.

In addition, we include two baselines for four-domain mixed training in Section 6, namely JT and CGPO [18]. For JT, we use naive joint training, in which each batch contains equal amounts of data from all domains and parameters are updated in the standard way. In contrast, CGPO follows the official implementation: it computes domain-wise updates within each batch and then updates the affected parameters with a specific learning rate.

3https://huggingface.co/Crownelius

Element-Level Absolute Change Distribution Per Task

###### Math Absolute Change

###### Code Absolute Change

< 1e-07: 79.3% of params

< 1e-07: 77.6% of params

ProportionofParameters

ProportionofParameters

Global

Global

0.08

0.08

ATTN

ATTN

MLP

MLP

0.06

0.06

NORM

NORM

0.04

0.04

0.02

0.02

0.00

0.00

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

###### QA Absolute Change

###### CW Absolute Change

< 1e-07: 77.1% of params

< 1e-07: 89.1% of params

ProportionofParameters

ProportionofParameters

Global

Global

0.08

0.04

ATTN

ATTN

MLP

MLP

0.06

0.03

NORM

NORM

0.04

0.02

0.02

0.01

0.00

0.00

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

(a) Absolute parameter changes.

Element-Level Relative Change Distribution Per Task

###### Math Relative Change

###### Code Relative Change

0.05

< 1e-03: 79.3% of params

< 1e-03: 77.6% of params

ProportionofParameters

ProportionofParameters

Global

Global

0.04

ATTN

ATTN

0.04

MLP

MLP

0.03

NORM

NORM

0.03

0.02

0.02

0.01

0.01

0.00

0.00

10 3 10 2 10 1 100 101 | W| / |W_base|

10 3 10 2 10 1 100 101 | W| / |W_base|

###### QA Relative Change

###### CW Relative Change

0.05

0.025

< 1e-03: 77.1% of params

< 1e-03: 89.1% of params

ProportionofParameters

ProportionofParameters

Global

Global

ATTN

ATTN

0.04

0.020

MLP

MLP

NORM

NORM

0.03

0.015

0.02

0.010

0.01

0.005

0.00

0.000

10 3 10 2 10 1 100 101 | W| / |W_base|

10 3 10 2 10 1 100 101 | W| / |W_base|

(b) Relative parameter changes.

Figure 2: Parameter-change distributions of the four single-domain experts relative to the base model.

#### 3.3 Checkpoint Selection and Evaluation

For each training stage, we train until convergence and select the checkpoint with the best validation performance on the current training domain. Final evaluation is conducted on independent benchmarks. Math is evaluated on AIME24/25/26 [1], OlympiadBench [10], and HMMT [9]. Code is evaluated on LiveCodeBench-v6 [13]. QA is evaluated on SuperGPQA-test [32] and MMLU-Pro [35]. CW is evaluated on WritingBench [37]. See Appendix A.1 for details.

### 4 Structural Evidence for Localized Cross-Domain Interference

This section identifies where cross-domain interference arises in the model. Full-model gradients give only a coarse picture; the analysis then focuses on sparse parameter edits, shared active routes, and directional alignment along those routes.

#### 4.1 Global Gradients

A natural first question is whether cross-domain interference in domain RL can be explained by global gradient conflict. During JT training, domain-specific gradients are periodically computed

⊤ digdj

) = g

and the cosine similarity between domain pairs is measured, cos(gd

∥gdi∥2 ∥gdj∥2 , where gd is the gradient for domain d. Positive, negative, and near-zero values indicate alignment, conflict, and approximate orthogonality, respectively.

,gd

i

j

Figure 1a shows that the global gradient cosine between Math and QA stays close to zero. This means that the later drop in Math after QA training cannot be attributed to strong full-model gradient antagonism alone. However, near-orthogonality at the global level does not mean interference is absent. Decomposing gradients by layer and module reveals a clear local structure: Figure 1b highlights the strongest conflict locations in attention and MLP modules, while Figure 1c highlights the strongest synergy locations. Thus, cross-domain interaction is not uniformly antagonistic across the whole model; it is localized, with both conflict and synergy appearing in different layers and modules. Complete results are provided in Appendix B.1.

#### 4.2 Sparse and Small-Magnitude Updates

The gradient analysis above shows that cross-domain interaction is localized. We next examine whether the updates actually written into the model by domain RL are also localized. Each singledomain expert is compared with the base model, with parameter changes measured at the element

level across the full model: ∆W = Wexpert − Wbase, r(W) = |W|∆W|

base|, where W denotes the model parameter, ∆W is the absolute change, and r(W) is the relative change.

- As shown in Figure 2, across the four single-domain experts, approximately 77%–89% of parameters have absolute changes below 10−7 and relative changes below 10−3. Moreover, even among

parameters with non-negligible changes, the update magnitudes remain small. These results suggest that domain-specific RL acts as a mild perturbation to the base model rather than a global parameter rewrite. The parameter analysis of sequential training is presented in Appendix B.2.

#### 4.3 Weak Edit Overlap

Since domain RL updates are sparse, a natural explanation for strong interference is direct co-editing: different domains may concentrate their large updates on the same set of functional units. We test this explanation by lifting the analysis from parameters to MLP neurons and measuring the overlap among the most strongly changed neurons across domain experts. We treat each MLP intermediate channel as a neuron and score its change by aggregating the parameter differences associated with its gate, up, and down projections. For each layer, we select the top 10% most changed neurons for each domain expert and compute the pairwise Jaccard overlap between these sets. Full definitions are provided in Appendix B.3.

[Figure 1]

0.6

Math

1.000 0.189 0.134 0.138

[Figure 2]

0.5

0.4

Code

0.189 1.000 0.120 0.124

0.3

QA

0.134 0.120 1.000 0.109

0.2

0.1

CW

0.138 0.124 0.109 1.000

0.0

Math Code QA CW

(a) Changed-neuron overlap.

- Figure 3a shows that top-changed neurons overlap only weakly across domains, with average Jaccard coefficients below 0.19 for all domain pairs. This suggests that domain RL updates target distinct local neuron subsets rather than a common large set of neurons. Therefore, cross-domain interference is unlikely to arise simply from large-scale coediting of the same neurons. This leads to a key question: if different domains edit largely non-overlapping neurons, why does later-domain training still produce clear cross-domain effects?

4.4 Act on Shared Computation Routes

The previous section shows weak overlap among top-changed neurons across domains. But low edit overlap does not imply functional independence, so top-active neurons are examined to test whether domains share active routes during inference. To test this possibility, we measure inference-time activation overlap. Using the same MLP-neuron definition as above, we rank neurons in each layer by their average activation magnitude on each domain and select the top 5% most active neurons. We then compute the pairwise Jaccard overlap of these active-neuron sets across domains. Full metric definitions are provided in Appendix B.4.

- Figure 3b shows that, among reasoning-oriented domains, active-neuron overlap is much higher than changed-neuron overlap. Math, Code, and QA exhibit higher mutual overlap and therefore share more active computation routes, whereas CW remains relatively independent. Thus, low edit overlap does not preclude functional coupling: sparse updates can still affect other domains through shared active routes.

[Figure 3]

0.6

Math

1.000 0.449 0.515 0.263

[Figure 4]

0.5

0.4

Code

0.449 1.000 0.440 0.267

0.3

QA

0.515 0.440 1.000 0.296

0.2

0.1

CW

0.263 0.267 0.296 1.000

0.0

Math Code QA CW

(b) Active-neuron overlap.

Figure 3: Neuron-overlap rates under different settings.

#### 4.5 Directionality on Shared Routes

The previous section shows that different domains often share highly active computation routes. But shared routes do not automatically cause interference. We therefore examine the directional alignment of edits on shared route components. For each pair of domain experts, we select the top 10% most changed neurons per domain, take their intersection, and compute the cosine similarity between the corresponding neuron-level update vectors. Full metric definitions are provided in Appendix B.5.

Figure 4 shows that domain pairs exhibit distinct directional patterns on shared neurons. Code-Math is predominantly aligned, with the average cosine staying clearly positive. By contrast, Math-QA is not globally antagonistic but splits by layer: the average cosine is negative in layers L3–6 and positive in layers L14–21.

Code vs CW

Code vs Math

0.03

Code vs QA CW vs Math CW vs QA

Math vs QA

0.02

MeanCosine

0.01

0.00

0.01

0.02

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35

Layer

Figure 4: Layer-wise average directional cosine on shared top-changed neurons across domain pairs.

Taken together, these analyses give a clear picture of cross-domain RL interference. Global gradients are nearly orthogonal, domain-specific RL updates are sparse and small, and different domains affect largely distinct neuron sets. However, sparse edits can produce cross-domain effects through these routes. The direction of updates on shared routes then determines whether the effect is synergistic or conflicting. These findings motivate the theory developed next: cross-domain interference is better modeled as a localized conflict on shared active computation routes. These signals also motivate a later direct rollback experiment based on shared activation, update magnitude, and directional conflict.

### 5 Theory: Local Recoverability under Sparse Low-Dimensional Interference

The structural evidence above suggests three constraints that any explanation of cross-domain interference should satisfy. First, interference is not well explained by full-model gradient conflict, since domain gradients can be nearly orthogonal. Second, domain RL writes sparse and smallmagnitude edits, and different domains modify weakly overlapping neuron sets. Third, sparse edits can still affect other domains because different domains reuse shared active routes, and the direction of edits on these routes determines synergy or conflict.

We formalize these observations with a local perturbation model. Later-domain training can damage an earlier domain by moving along curvature-sensitive conflict directions, while a short refresh contracts this harmful component and enables fast, selective recovery.

#### 5.1 Notation and Structural Assumptions

We write Ld(θ) as the local objective for domain d, with gradient gd(θ) and Hessian Hd(θ). Suppose training on domain A selects a checkpoint θA∗ , and later training on domain B induces a local update δB, producing θA∗ + δB. We measure interference from B to A by the increase in the earlier-domain objective: ∆A←B = LA(θA∗ + δB) − LA(θA∗ ). The goal is to characterize when this quantity is large, why it is selective across domains, and why a short refresh on A can reduce it.

Under standard local smoothness, we use three structural conditions motivated by Section 4: (i) the selected domain-A checkpoint is approximately stationary for LA; (ii) later-domain updates are local and effectively sparse; and (iii) the curvature-sensitive part of the later update is concentrated in a lowdimensional shared active conflict subspace SA,B. For the refresh analysis, we additionally assume positive curvature of LA restricted to SA,B and weak coupling from the orthogonal complement back into this subspace. Formal statements of these assumptions are provided in Appendix C.1.

#### 5.2 Second-Order Local Damage

We first explain why later-domain training can harm an earlier domain even without strong global gradient opposition. After training on domain A, the selected checkpoint θA∗ is approximately stationary for the local objective LA. Therefore, when a later domain B induces a local update δB, the first-order change in LA is small, and the leading effect is governed by second-order local curvature.

- Proposition 1. Under local smoothness and approximate stationarity of θA∗ , for a later-domain update δB with ∥δB∥2 ≤ r, the interference from B to A satisfies

∆A←B =

- 1

- 2

δB⊤HA(θA∗ )δB + O εA∥δB∥2 + ∥δB∥32 . (1)

This result shows that earlier-domain degradation depends mainly on whether the later-domain parameter update moves along high-curvature directions of the earlier-domain objective. In other words, a sparse and small-magnitude update can still cause substantial degradation if it moves along curvature-sensitive directions of LA. This explains why near-orthogonal full-model gradients do not rule out cross-domain interference: the harmful effect can appear as a localized second-order displacement rather than a global first-order gradient conflict. The full proof, along with the equivalent sensitivity-based form, is provided in Appendix C.2.

5.3 Low-Dimensional Conflict Subspace

- Proposition 1 shows that later-domain damage is governed by the local curvature of the earlier-domain objective. The next question is where these curvature-sensitive harmful directions lie. The structural evidence in Section 4 suggests that they are not spread across the full parameter space: domain RL updates are sparse, different domains edit weakly overlapping neuron sets, but they can still interact through shared active routes. We therefore model the harmful component of a later-domain update as concentrated in a low-dimensional shared active conflict subspace SA,B.
- Proposition 2. Let PS denote the projection onto the shared active conflict subspace SA,B. Under the local structural conditions above, interference from domain B to domain A satisfies

∆A←B =

- 1

- 2

(PSδB)⊤HA(θA∗ )(PSδB) + O εA∥δB∥2 + γA∥δB∥22 + ∥δB∥32 . (2)

This result localizes the second-order damage identified in Proposition 1. The leading harmful term depends only on the projection of the later-domain update onto SA,B, up to controlled residual terms. Thus, low edit overlap does not imply low interference: even if two domains modify largely different neuron sets, a later-domain update can still harm an earlier domain if its projection onto the shared active conflict subspace is large. This also explains why full-model near-orthogonality can coexist with selective degradation. When global first-order conflict is weak, the main damage can still arise from localized second-order displacement along shared curvature-sensitive directions. It also motivates the weaker empirical test used later in Section 6.2: if a sparse coordinate proxy captures a nontrivial portion of this harmful projection, then reverting the corresponding part of the later-domain update should partially reduce the damage, even when the proxy is only basis-aligned rather than a direct estimate of SA,B. We provide the full decomposition and proof in Appendix C.3.

5.4 Short Refresh Geometrically Contracts the Conflict Component

- Proposition 2 shows that earlier-domain degradation is mainly controlled by the projection of the

later-domain update onto the shared active conflict subspace SA,B. This suggests that recovery does not require undoing the entire later-domain update. Instead, a short refresh on domain A only needs

to reduce the harmful component inside SA,B.

Starting from the degraded checkpoint θ0 = θA∗ + δB, we consider a short refresh on domain A with gradient descent: θt+1 = θt −αgA(θt). Under positive curvature of LA restricted to SA,B and weak

coupling from SA,B⊥ back into SA,B, this component contracts geometrically. The formal assumption is provided in Appendix C.1.

Theorem 1. Starting from θ0 = θA∗ + δB, a short refresh on domain A satisfies

∥PS(θt − θA∗ )∥2 ≤ (1 − αµA)t∥PSδB∥2, (3) where µA > 0 is the local curvature lower bound on SA,B. See Appendix C.4 for the proof.

The theorem shows that the harmful component decays geometrically with the number of refresh steps. This gives a local explanation for fast recovery: early refresh steps can rapidly remove the

- Table 2: Performance of single-domain experts, sequential training, mixed-domain baselines, and Re-Math refresh.

Task Base Codes Maths QAs CWs Codeo Matho QAo CWo CGPO JT Re-Math

Math 43.19 59.63 66.84 55.31 39.78 59.63 66.49 59.90 57.66 61.93 64.80 66.04 Code 29.57 52.67 34.65 32.07 28.15 52.67 50.69 50.99 50.47 50.05 48.61 51.05 QA 60.64 60.89 60.76 63.31 60.50 60.89 60.52 62.34 62.34 62.48 62.11 62.49 CW 82.44 82.40 81.38 81.76 86.24 82.40 81.44 81.79 86.52 86.73 86.97 85.96

AVG 53.96 63.90 60.91 58.11 53.67 63.90 64.79 63.76 64.25 65.30 65.62 66.39

component responsible for earlier-domain degradation, without requiring full retraining or reversing the whole later-domain update. It also explains why recovery can be selective. Since the refresh mainly contracts the component to which domain A is sensitive, it can restore domain A while causing only bounded collateral damage to other domains under global near-orthogonality; the detailed proof is provided in Appendix C.5.

Taken together, Proposition 1, Proposition 2, and Theorem 1 give a local explanation for selective interference and recovery. Later-domain training harms an earlier domain mainly through a localized second-order displacement in shared conflict directions, while a short refresh recovers the domain by geometrically contracting this harmful component. This view yields two complementary empirical predictions: a short refresh should selectively recover the damaged domain, and a sparse rollback on a coordinate proxy for the harmful component should partially reduce the damage without full retraining. We further discuss an alternating-refresh extension in Appendix C.6. The extension shows that one cycle of alternating refresh is, to first order, equivalent to a descent step on a weighted multi-domain objective. Under the standard weighted-sum view of multi-objective optimization, this suggests that repeated refresh moves toward a local Pareto-stationary compromise among domains.

### 6 Task-Level Validation and Direct Intervention

This section validates the theory at two levels: task-level recovery under short refresh, and a direct weight-space rollback on a coordinate proxy for the conflict subspace in the fixed Math→QA pair.

#### 6.1 Recovery by Short Refresh

Main result. Table 2 shows the domain-level results of single-domain experts, sequential checkpoints, mixed-training baselines, and Re-Math, obtained by a short Math refresh from CWo. Full benchmark-level results are in Appendix D.1.

Selective interference. Sequential training does not cause uniform forgetting. After Code → Math, Matho reaches 66.49 on Math, close to the single-domain Math expert (66.84). After later QA and CW training, however, Math drops to 57.66, while the other domains remain largely stable or improve primarily during their own stages. This shows that interference is domain-specific rather than uniform.

Fast recovery. A short Math refresh from CWo raises Math from 57.66 to 66.04, recovering most of the loss from later-domain training and bringing performance close to both Matho and the single-domain math expert. This suggests that Re-Math acts as a local correction.

Limited side effects. While recovering Math, Re-Math leaves the other domains nearly unchanged: Code rises slightly to 51.05, while QA and CW remain essentially unchanged. This pattern also highlights asymmetric sensitivity: Math is more sensitive to the later QA/CW updates than the other domains are to the short Re-Math correction. As a result, Re-Math attains the best average score, 66.39. See Appendices D.2 and D.3 for further analysis and validation.

#### 6.2 Conflict Subspace Proxy

The refresh results above validate recoverability, but they do not by themselves directly probe the localization claim in Proposition 2, which attributes the dominant damage from a later domain to the projection of its update onto a shared active conflict subspace. We perform a training-free weight-space rollback on the checkpoint pair Matho → QAo to isolate the QA→Math interaction.

- Table 3: Selective rollback on a coordinate proxy for the conflict subspace from QAo toward Matho. The Matho row gives the pre-QA reference upper bound. Joint MLP+Attn includes attention layers; all other non-baseline selectors use MLP neurons only.

Selector Budget Math Avg ∆ vs QAo Recovery ∆ QA Avg

Matho – 66.49 +6.59 100.0% -1.81 QAo – 59.90 – 0.0% 0.00

Random 2% 59.49 -0.42 -6.3% +0.01 A×M 2% 60.53 +0.63 9.6% +0.04 A×C 2% 60.55 +0.65 9.9% -0.05 M×C 2% 61.11 +1.21 18.3% +0.04 A×M×C 2% 61.25 +1.35 20.4% -0.06

Joint MLP+Attn 32% 64.75 +4.85 73.6% -0.45

We first study an MLP-only coordinate proxy based on three signals: shared activation under Math and QA (A), the magnitude of the QA update (M), and directional conflict between the Math and QA task vectors (C). Treating checkpoints as parameter vectors, define the QA-induced displacement as δQ|M = QAo − Matho. We then revert only the QA increment on the selected neurons: θrev = QAo − PSˆδQ|M, where PSˆ denotes the selected neurons. This rollback removes a proxy for part of the harmful component in Proposition 2, without identifying the latent subspace.

Table 3 shows that the rollback is selective. Reverting only 2% of MLP neurons selected by the composite score A × M × C raises Math Avg from 59.90 to 61.25, recovering 20.4% of the QAinduced Math loss while changing QA Avg by only −0.06; by contrast, reverting the same number of randomly selected neurons decreases Math Avg to 59.49. The additional ablations in Appendix D.4 indicate partial redundancy among the localization signals. Under the fixed-budget protocol, A-only matches the full selector on Math recovery, whereas recomputing the layer budget lowers recovery; among the two-factor variants, M × C remains the closest approximation to the full selector. We also extend the intervention to attention coordinates. As Figure 13 shows, as the budget increases, the joint MLP+Attn selector recovers more damage beyond MLP coordinates, reaching 73.6% recovery at a 32% budget. Together, these results suggest that, for this fixed Math-QA interaction, the harmful QA-induced displacement is localized enough that a sparse coordinate-proxy rollback can recover a substantial portion of the Math damage without retraining.

Overall, the task-level and intervention results support our theory: multi-domain RL causes selective interference, short refresh largely reverses it with limited side effects, and a small targeted rollback on a coordinate proxy for the conflict subspace recovers a substantial portion of the QA-induced Math loss without retraining.

### 7 Conclusion

This work studies cross-domain interference in multi-domain RL and shows that it is not a full-model phenomenon: global domain gradients can be nearly orthogonal, yet conflict arises locally on shared active computation routes. Although RL updates are sparse, small, and have little overlap across domains, reasoning domains still reuse common routes, and directional misalignment on those routes determines whether later training helps or harms earlier-domain performance.

The theoretical analysis further shows that this damage is mainly a local second-order effect concentrated in a low-dimensional shared conflict subspace, which means a short refresh on the damaged domain can contract the harmful component while preserving other domains. Empirically, both short refresh and sparse proxy rollback can restore damaged performance while largely preserving other domains. These results suggest that controlling localized route-level interference is a promising direction toward more stable and scalable multi-domain RL.

### Limitations

This work has several limitations and also suggests natural future directions. First, although our results indicate a simple practical route for multi-domain RL, we have not yet developed it into an

automatic training algorithm. A key implication is that stable multi-domain improvement may not require delicate data-mixture tuning or hand-designed replay schedules: after sequential domain training, a short targeted refresh on the most degraded domains can quickly recover performance with limited side effects. Future work should automate the detection of degraded domains, the selection of refresh order and budget, and examine whether repeated targeted refresh consistently approaches a local Pareto-stationary compromise.

Second, our conflict-subspace intervention is still based on a coarse proxy. Although the 32% rollback budget is relatively large, it should be interpreted primarily as a causal sufficiency test rather than as an estimate of the minimal conflict subspace. Because the proxy coordinates are likely redundant, comparable recovery may be achievable with a smaller and more precisely identified intervention set. Moreover, the current proxy is basis-aligned and does not directly estimate a latent rotated subspace, nor does it cover all potentially relevant factors such as normalization parameters, residual-stream couplings, or higher-order cross-module interactions. A natural next step is to identify the conflict subspace more precisely and use it for projection-based training, constrained updates, or route-aware regularization, so that harmful cross-domain components can be suppressed with a smaller and less redundant intervention budget.

Finally, our analysis focuses on multi-domain RL, while similar training dynamics may also arise in other post-training paradigms. For example, on-policy distillation also involves evolving training distributions and policy-dependent data, where interference, instability, or route-level specialization may emerge over training. Extending our diagnostic framework to such settings could help explain broader post-training dynamics beyond the specific multi-domain RL setup studied here.

### References

- [1] Art of Problem Solving. Aime problems and solutions. https://artofproblemsolving.com/ wiki/index.php/AIME_Problems_and_Solutions, 2024a. Accessed: 2025-12-18.
- [2] A. Bau, Y. Belinkov, H. Sajjad, N. Durrani, F. Dalvi, and J. R. Glass. Identifying and controlling important neurons in neural machine translation. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.
- [3] H. Chen, N. Razin, K. Narasimhan, and D. Chen. Retaining by doing: The role of on-policy data in mitigating forgetting. CoRR, abs/2510.18874, 2025.
- [4] Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In J. G. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 793–802. PMLR, 2018.
- [5] P. F. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 4299–4307, 2017.
- [6] T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. CoRR, abs/2501.17161, 2025.
- [7] D. Dai, L. Dong, Y. Hao, Z. Sui, B. Chang, and F. Wei. Knowledge neurons in pretrained transformers. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 8493–8502. Association for Computational Linguistics, 2022.
- [8] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025.

- [9] J. Dekoninck, N. Jovanovi´c, T. Gehrunger, K. Rögnvalddson, I. Petrov, C. Sun, and M. Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. CoRR, abs/2605.00674, 2026.
- [10] C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In L. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 3828–3850. Association for Computational Linguistics, 2024.
- [11] M. Huan, Y. Li, T. Zheng, X. Xu, S. Kim, M. Du, R. Poovendran, G. Neubig, and X. Yue. Does math reasoning improve general LLM capabilities? understanding transferability of LLM reasoning. CoRR, abs/2507.00432, 2025.
- [12] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025.
- [13] N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and

I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

- [14] J. Kirkpatrick, R. Pascanu, N. C. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell. Overcoming catastrophic forgetting in neural networks. CoRR, abs/1612.00796, 2016.
- [15] S. Lai, H. Zhao, R. Feng, C. Ma, W. Liu, H. Zhao, X. Lin, D. Yi, M. Xie, Q. Zhang, H. Liu, G. Meng, and F. Zhu. Reinforcement fine-tuning naturally mitigates forgetting in continual post-training. CoRR, abs/2507.05386, 2025.
- [16] Y. Leng and D. Xiong. Towards understanding multi-task learning (generalization) of LLMs via detecting and exploring task-specific neurons. In O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa, B. D. Eugenio, and S. Schockaert, editors, Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 2969–2987. Association for Computational Linguistics, 2025.
- [17] D. Li, J. Zhou, A. Kazemi, Q. Sun, A. Ghaddar, M. A. Alomrani, L. Ma, Y. Luo, D. Li, F. Wen, J. Hao, M. Coates, and Y. Zhang. Omni-thinker: Scaling cross-domain generalization in llms via multi-task RL with hybrid rewards. CoRR, abs/2507.14783, 2025.
- [18] X. Liang, L. Yang, J. Wang, R. Liu, Y. Lu, J. Zeng, H. Chen, D. Li, and J. Hao. Boosting multi-domain reasoning of LLMs via curvature-guided policy optimization. In International Conference on Learning Representations, 2026.
- [19] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman,

I. Sutskever, and K. Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

- [20] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu. Conflict-averse gradient descent for multi-task learning. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 18878–18890, 2021.
- [21] M. Matena and C. Raffel. Merging models with fisher-weighted averaging. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
- [22] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.

- [23] O. Sener and V. Koltun. Multi-task learning as multi-objective optimization. In S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 525–536, 2018.
- [24] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024.
- [25] I. Shenfeld, J. Pari, and P. Agrawal. RL’s razor: Why online reinforcement learning forgets less. CoRR, abs/2509.04259, 2025.
- [26] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pages 1279–1297. ACM, 2025.
- [27] D. Shi, Z. Han, S. Ostermann, R. Jin, J. van Genabith, and D. Xiong. Why does reinforcement learning generalize? a feature-level mechanistic study of post-training in large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 2026.
- [28] H. Shi, Z. Xu, H. Wang, W. Qin, W. Wang, Y. Wang, and H. Wang. Continual learning of large language models: A comprehensive survey. CoRR, abs/2404.16789, 2024.
- [29] N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to summarize with human feedback. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
- [30] Z. Su, L. Pan, M. Lv, Y. Li, W. Hu, F. Zhang, K. Gai, and G. Zhou. Ce-gppo: Controlling entropy via gradient-preserving clipping policy optimization in reinforcement learning, 2025.
- [31] K. Team. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025.
- [32] M. Team. Supergpqa: Scaling LLM evaluation across 285 graduate disciplines. CoRR, abs/2502.14739, 2025.
- [33] Q. Team. Qwen3 technical report. CoRR, abs/2505.09388, 2025.
- [34] X. Wang, K. Wen, Z. Zhang, L. Hou, Z. Liu, and J. Li. Finding skill neurons in pre-trained transformer-based language models. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 11132–11152. Association for Computational Linguistics, 2022.
- [35] Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [36] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, and L. Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 23965–23998. PMLR, 2022.

- [37] Y. Wu, J. Mei, M. Yan, C. Li, S. Lai, Y. Ren, Z. Wang, J. Zhang, M. Wu, Q. Jin, and F. Huang. Writingbench: A comprehensive benchmark for generative writing. CoRR, abs/2503.05244, 2025.
- [38] P. Yadav, D. Tam, L. Choshen, C. A. Raffel, and M. Bansal. Ties-merging: Resolving interference when merging models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.
- [39] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn. Gradient surgery for multi-task learning. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
- [40] J. Zheng, X. Cai, S. Qiu, and Q. Ma. Spurious forgetting in continual learning of language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

Table 4: Training hyperparameters.

train batch size ppo mini batch size max prompt length max response length adv estimator filter overlong prompts

256 256 2048 16384 grpo True

lr use dynamic bsz use kl loss kl loss coef kl loss type clip ratio 1e-6 True False 0.0 low_var_kl 0.2

entropy coeff rollout n rollout temperature rollout top k rollout top p val top k

0 8 1.0 -1 1.0 -1 val top p val temperature val n val do sample use kl in reward

0.9 0.7 8 True False

### A Implementation Details

#### A.1 Data Construction Details

We ensure that the prompt length of all training data is within 2,048. Each domain uses 5,120 training examples. Validation sets are constructed from non-training examples in the corresponding data sources. For Math, we use AIME25 [1] as the validation set, which contains 30 problems. For Code, QA, and CW, each validation set contains 50 examples.

For CW, we sample 5,120 examples from the crownelius/Creative-Writing series.4 The data come from four subsets: Sonnet4.6-800x,5 Gemini3Pro-2700x,6 Reasoning-KimiK2.5-600x,7 and Qwen3.5Plus-2000x.8 For 2,560 of these examples, we resample the reference responses using Qwen3-235B-A22B-Instruct-2507 and use the newly sampled responses as references.

For evaluation, LiveCodeBench-v6 contains 175 problems spanning January 2025 to April 2025. For QA evaluation, after excluding the training and validation data from SuperGPQA, we sample 10% of the remaining data, resulting in 2,141 test examples. Similarly, we sample 10% of MMLU-Pro, resulting in 1,203 test examples.

#### A.2 Training Hyperparameters

All experiments use the same GRPO training configuration. The full hyperparameters are summarized in Table 4.

At each training stage, we train the model until convergence and select the checkpoint with the optimal validation performance in the current training domain. Specifically, single-domain training adopts the best-performing checkpoint within the single domain, while multi-domain mixed training selects the checkpoint with the optimal average performance across all domains.

#### A.3 Reward Functions

Math and QA use answer-correctness rewards. The model is required to put the final answer in boxed. We use math_verify to parse the answer and match it against the reference answer. The reward is binary: 1 for a correct answer and 0 otherwise.

Code uses execution-based rewards, computed from the fraction of test cases passed.

CW uses an LLM-as-a-judge preference reward. For crownelius/Creative-Writing series, given a prompt, a reference response, and a model response, the judge compares their quality. If the model response is better, the reward is 1; if the two responses are comparable, the reward is 0.5; if the reference response is better, the reward is 0. For each query of WritingBench, the model generates one response, which is scored along multiple WritingBench evaluation dimensions. We use Qwen3-235B-A22B-Instruct-2507 [33] as the judge for training and evaluation.

- 4https://huggingface.co/Crownelius
- 5https://huggingface.co/datasets/Crownelius/Creative-Writing-Sonnet4.6-800x
- 6https://huggingface.co/datasets/Crownelius/Creative-Writing-Gemini3Pro-2700x
- 7https://huggingface.co/datasets/Crownelius/Creative-Writing-Reasoning-KimiK2.5-600x
- 8https://huggingface.co/datasets/Crownelius/Creative-Writing-Qwen3.5Plus-2000x

Mean Cosine by Layer (ATTN) Red=Conflict(neg), Blue=Synergy(pos)

[Figure 5]

QA vs Code

[Figure 6]

0.010

QA vs CW

0.005

QA vs Math

0.000

Code vs CW

0.005

Code vs Math

0.010

CW vs Math

L0

L1

L2

L3

L4

L5

L6

L7

L8

L9

L10

L11

L12

L13

L14

L15

L16

L17

L18

L19

L20

L21

L22

L23

L24

L25

L26

L27

L28

L29

L30

L31

L32

L33

L34

L35

(a) Attention-module heatmap of pairwise gradient cosine.

Mean Cosine by Layer (MLP) Red=Conflict(neg), Blue=Synergy(pos)

[Figure 7]

QA vs Code

0.006

[Figure 8]

QA vs CW

0.004

0.002

QA vs Math

0.000

Code vs CW

0.002

Code vs Math

0.004

0.006

CW vs Math

L0

L1

L2

L3

L4

L5

L6

L7

L8

L9

L10

L11

L12

L13

L14

L15

L16

L17

L18

L19

L20

L21

L22

L23

L24

L25

L26

L27

L28

L29

L30

L31

L32

L33

L34

L35

(b) MLP-module heatmap of pairwise gradient cosine.

Figure 5: Layer-wise heatmaps of pairwise gradient cosine in attention and MLP modules.

Top 6 Conflict (ATTN_MLP) cosine over steps

layers.30.attn QA vs Math mean=-0.0136 conflict=65%

layers.2.mlp CW vs Math mean=-0.0103 conflict=55%

layers.9.attn QA vs Math mean=-0.0096 conflict=63%

0.10

0.1

0.05

0.0

Cosine

0.0

0.00

0.05

0.1

0.1

0.10

0 100 200 300

0 100 200 300

0 100 200 300

Step

Step

Step

layers.21.attn QA vs Math mean=-0.0094 conflict=60%

layers.31.attn CW vs Math mean=-0.0084 conflict=56%

layers.34.attn QA vs Math mean=-0.0079 conflict=55%

0.05

0.1

0.00

Cosine

0.0

0.0

0.05

0.2

0.10

0.1

0 100 200 300

0 100 200 300

0 100 200 300

Step

Step

Step

(a) Top six attention and MLP modules with the strongest conflict.

Top 6 Synergy (ATTN_MLP) cosine over steps

layers.7.attn Code vs Math mean=0.0326 conflict=37%

layers.32.attn QA vs Code mean=0.0155 conflict=41%

layers.1.attn Code vs Math mean=0.0124 conflict=45%

0.2

0.1

0.2

Cosine

0.0

0.0

0.0

0.2

0.1

0 100 200 300

0 100 200 300

0 100 200 300

Step

Step

Step

layers.32.attn Code vs Math mean=0.0118 conflict=44%

layers.3.attn Code vs Math mean=0.0101 conflict=44%

layers.10.attn Code vs Math mean=0.0085 conflict=41%

0.2

0.10

0.1

0.1

0.05

Cosine

0.0

0.0

0.00

0.1

0.05

0.1

0 100 200 300

0 100 200 300

0 100 200 300

Step

Step

Step

(b) Top six attention and MLP modules with the strongest synergy.

- Figure 6: Module-level conflict and synergy trends across the most prominent attention and MLP modules.

### B Additional Structural Analysis

#### B.1 Extended Module-Level Gradient Conflict and Synergy

Figure 5 shows the six attention and MLP modules with the most prominent conflict and synergy, respectively. Figure 6 presents the layer-wise heatmaps of pairwise gradient cosine in attention and MLP modules. These results show that conflicts are not uniformly distributed across all parameters but are concentrated in specific regions of the network. In particular, local conflicts are most significant for the Math-QA domain pair. In addition, certain local conflicts also exist between Code-CW. Nevertheless, despite relatively clear module interactions, the corresponding cosine values remain small and are largely orthogonal.

#### B.2 Sequential Parameter Changes Remain Sparse

In addition to single-domain experts, we also measure parameter changes along the sequential training chain: Code → Math → QA → CW. We analyze both cumulative changes relative to the base model and incremental changes relative to the previous checkpoint.

As shown in Figure 7, cumulative deviation from the base model gradually increases as more domains are added. The fraction of parameters with |∆W| < 10−7 and r(W) < 10−3 decreases from 77.6% after Code training to 73.4% after Code-Math-QA-CW training.

However, as shown in Figure 8, each individual stage remains sparse. When comparing each checkpoint with the immediately preceding one, later domain stages modify only a small fraction of parameters: 87.2%, 84.0%, and 88.3% of parameters remain below the 10−7 threshold for the Math, QA, and CW stages, respectively.

These results show that sequential domain RL accumulates parameter shifts over domains, but each stage still acts as a sparse incremental update rather than a global parameter rewrite.

#### B.3 Neuron-Level Edit Definition

For each MLP layer ℓ, we define the i-th intermediate channel as one neuron. Its associated parameters consist of the corresponding row of the gate projection, the corresponding row of the up projection, and the corresponding column of the down projection:

##### gi(ℓ) = Wgate(ℓ) [i,:], u(iℓ) = Wup(ℓ)[i,:], d(iℓ) = Wdown(ℓ) [:,i]. (4)

Absolute Change vs Base Sequential Chain (Approach A)

###### code base

###### code_math base

0.10

< 1e-07: 77.6% of params

< 1e-07: 76.2% of params

ProportionofParameters

ProportionofParameters

Global

Global

0.08

ATTN

ATTN

0.08

MLP

MLP

0.06

NORM

NORM

0.06

0.04

0.04

0.02

0.02

0.00

0.00

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

###### code_math_qa base

###### code_math_qa_cw base

0.10

0.10

< 1e-07: 74.0% of params

< 1e-07: 73.4% of params

ProportionofParameters

ProportionofParameters

Global

Global

ATTN

ATTN

0.08

0.08

MLP

MLP

NORM

NORM

0.06

0.06

0.04

0.04

0.02

0.02

0.00

0.00

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

(a) Absolute cumulative parameter changes relative to the base model.

Relative Change vs Base Sequential Chain (Approach A)

###### code base

###### code_math base

0.05

0.05

< 1e-03: 77.6% of params

< 1e-03: 76.2% of params

ProportionofParameters

ProportionofParameters

Global

Global

ATTN

ATTN

0.04

0.04

MLP

MLP

NORM

NORM

0.03

0.03

0.02

0.02

0.01

0.01

0.00

0.00

10 3 10 2 10 1 100 101 | W| / |W_before|

10 3 10 2 10 1 100 101 | W| / |W_before|

###### code_math_qa base

###### code_math_qa_cw base

0.05

0.05

< 1e-03: 74.0% of params

< 1e-03: 73.4% of params

ProportionofParameters

ProportionofParameters

Global

Global

ATTN

ATTN

0.04

0.04

MLP

MLP

NORM

NORM

0.03

0.03

0.02

0.02

0.01

0.01

0.00

0.00

10 3 10 2 10 1 100 101 | W| / |W_before|

10 3 10 2 10 1 100 101 | W| / |W_before|

(b) Relative cumulative parameter changes relative to the base model.

- Figure 7: Cumulative parameter-change distributions along the sequential domain RL chain relative to the base model.

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

0.00

0.02

0.04

0.06

0.08

ProportionofParameters

< 1e-07: 77.6% of params

code base

Global

ATTN

MLP

NORM

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

0.00

0.01

0.02

0.03

0.04

0.05

ProportionofParameters

< 1e-07: 87.2% of params

code_math code

Global

ATTN

MLP

NORM

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

0.00

0.02

0.04

0.06

ProportionofParameters

< 1e-07: 84.0% of params

code_math_qa code_math

Global

ATTN

MLP

NORM

10 7 10 6 10 5 10 4 10 3 10 2 10 1 | W|

0.00

0.01

0.02

0.03

0.04

0.05

ProportionofParameters

< 1e-07: 88.3% of params

code_math_qa_cw code_math_qa

Global

ATTN

MLP

NORM

Absolute Change Incremental Sequential Chain (Approach B)

(a) Absolute incremental parameter changes relative to the previous checkpoint.

10 3 10 2 10 1 100 101 | W| / |W_before|

0.00

0.01

0.02

0.03

0.04

0.05

ProportionofParameters

< 1e-03: 77.6% of params

code base

Global

ATTN

MLP

NORM

10 3 10 2 10 1 100 101 | W| / |W_before|

0.00

0.01

0.02

0.03

ProportionofParameters

< 1e-03: 87.2% of params

code_math code

Global

ATTN

MLP

NORM

10 3 10 2 10 1 100 101 | W| / |W_before|

0.00

0.01

0.02

0.03

ProportionofParameters

< 1e-03: 84.0% of params

code_math_qa code_math

Global

ATTN

MLP

NORM

10 3 10 2 10 1 100 101 | W| / |W_before|

0.000

0.005

0.010

0.015

0.020

0.025 ProportionofParameters

< 1e-03: 88.3% of params

code_math_qa_cw code_math_qa

Global

ATTN

MLP

NORM

Relative Change Incremental Sequential Chain (Approach B)

(b) Relative incremental parameter changes relative to the previous checkpoint.

- Figure 8: Incremental parameter-change distributions at each stage of the sequential domain RL chain.

For domain d, let ∆gd,i(ℓ), ∆u(d,iℓ), and ∆d(d,iℓ) denote the corresponding parameter changes between the domain expert and the base model. We measure the edit magnitude of neuron i in layer ℓ by aggregating the squared parameter changes over these three components:

2 2

2 2

2 2

s(d,iℓ) = ∆gd,i(ℓ)

+ ∆u(d,iℓ)

+ ∆d(d,iℓ)

. (5) For each layer and each domain, we select the top 10% neurons with the largest edit magnitude: Nd(ℓ) = Top10% s(d,iℓ)

. (6) The layer-wise overlap between two domains A and B is measured by the Jaccard coefficient:

i

J(ℓ)(A,B) = |NA(ℓ) ∩ NB(ℓ)| |NA(ℓ) ∪ NB(ℓ)|

. (7)

Finally, we report the average overlap across the L MLP layers:

1 L

J(A,B) =

L

J(ℓ)(A,B). (8)

ℓ=1

#### B.4 Active-Route Overlap Metric

Using the same MLP-neuron definition as in Appendix B.3, we measure whether two domains activate overlapping active routes during inference. For domain d and layer ℓ, let h(d,i,tℓ) (x) denote the activation of neuron i at token position t for sample x ∈ Dd, and let Tx be the number of tokens in x. We define the sample-level average activation magnitude as

Tx

1 Tx

h(d,i,tℓ) (x) . (9) The dataset-level activation score is

a¯(d,iℓ)(x) =

t=1

1 |Dd| x∈D

a¯(d,iℓ)(x). (10)

a(d,iℓ) =

d

For each layer and each domain, we select the top 5% neurons with the largest dataset-level activation score:

A(dℓ) = Top5% a(d,iℓ)

. (11)

i

The layer-wise active-route overlap between two domains A and B is measured by the Jaccard coefficient:

Jact(ℓ)(A,B) = |A(Aℓ) ∩ A(Bℓ)| |A(Aℓ) ∪ A(Bℓ)|

. (12) Finally, we report the average active-route overlap across the L MLP layers:

1 L

Jact(A,B) =

L

Jact(ℓ)(A,B). (13)

ℓ=1

#### B.5 Directional Alignment on Shared Edited Neurons

For each domain d, layer ℓ, and MLP neuron i, we define the neuron-level update vector by concatenating the parameter changes associated with its gate, up, and down projections:

∆vd,i(ℓ) = ∆gd,i(ℓ);∆u(d,iℓ);∆d(d,iℓ) . (14) For two domains A and B, we define the shared edited neuron set in layer ℓ as

Q(A,Bℓ) = NA(ℓ) ∩ NB(ℓ), (15)

where Nd(ℓ) is the Top-10% changed-neuron set defined in Appendix B.3. For each shared edited neuron, we compute the cosine similarity between the two domain update vectors:

c(A,B,iℓ) = ⟨∆vA,i(ℓ) ,∆vB,i(ℓ) ⟩ ∥∆vA,i(ℓ) ∥2∥∆vB,i(ℓ) ∥2

. (16)

The layer-wise average directional alignment on shared edited neurons is

1 |Q(A,Bℓ) | i∈Q(ℓ)

c(A,B,iℓ) . (17)

C(ℓ)(A,B) =

A,B

Positive values indicate that two domains edit their shared neurons in aligned directions on average, while negative values indicate conflicting directions.

### C Detailed Proof of Our Theory

#### C.1 Structural Assumptions

For readability, the main text only summarizes the structural conditions used by the local perturbation analysis. We state them formally here while directly connecting each condition to the empirical setting.

- Assumption 1: local smoothness. For each domain d, Ld is twice continuously differentiable in a local neighborhood B(θ¯,r) containing the sequential RL trajectory, with

∥Hd(θ)∥2 ≤ βd, ∀θ ∈ B(θ¯,r). (18) When a third-order remainder is needed, we further assume local Hessian Lipschitzness:

∥Hd(θ) − Hd(θ′)∥2 ≤ ρd∥θ − θ′∥2, ∀θ,θ′ ∈ B(θ¯,r). (19) This standard condition is appropriate because Section 4 shows that RL post-training behaves as a sequence of small incremental updates rather than a global parameter rewrite; bounded Hessians control second-order sensitivity, and Hessian Lipschitzness keeps the Taylor remainder cubic in the update magnitude.

- Assumption 2: approximate stationarity. After training on domain A, the selected checkpoint is approximately stationary for its objective:

∥gA(θA∗ )∥2 = ∥∇LA(θA∗ )∥2 ≤ εA. (20)

This does not require θA∗ to be a global optimum; it only requires the checkpoint to lie near a locally stable region for its own domain, matching our validation-based checkpoint selection protocol.

- Assumption 3: small and effectively sparse updates. Later-domain training induces a local perturbation ∥δB∥2 ≤ r, and there exists a low-dimensional update subspace UB ⊂ Rp such that

∥PU⊥

B

δB∥2 ≤ τB∥δB∥2, τB ≪ 1. (21) The norm bound formalizes locality around the previous checkpoint, while the effective sparsity condition is motivated by the parameter- and neuron-level evidence in Section 4: updates are not literally zero outside a tiny support, but their dominant mass is concentrated in relatively few directions.

- Assumption 4: low-dimensional shared conflict subspace. For a pair of domains A and B, there

exists a low-dimensional subspace SA,B ⊂ Rp with projection PS = PS

A,B

such that

δB⊤HA(θA∗ )δB − (PSδB)⊤HA(θA∗ )(PSδB) ≤ γA∥δB∥22. (22) This is the key structural abstraction of our empirical findings: different domains often edit different parameters, yet their active routes overlap, so SA,B captures the shared active directions where later-domain updates can affect the earlier-domain objective. The assumption does not claim that all interference lies exactly in SA,B, only that off-subspace contributions remain perturbative.

- Assumption 5: global near-orthogonality. Although local conflict may exist inside SA,B, full gradients remain nearly orthogonal at the global scale:

|⟨gA(θ),gB(θ)⟩| ≤ κA,B∥gA(θ)∥2∥gB(θ)∥2, κA,B ≪ 1. (23) This separates local conflict from full-model antagonism: as shown in Section 4, global gradient cosine can be close to zero even when localized conflict exists, so the theory treats interference as a structured local phenomenon inside a small shared subspace rather than as uniformly conflicting gradients over the whole parameter space.

- Assumption 6: positive curvature on SA,B and weak cross-subspace coupling. There is a local neighborhood NA of θA∗ containing the refresh trajectory such that, for all θ ∈ NA and v ∈ SA,B,

##### µA∥v∥22 ≤ v⊤HA(θ)v ≤ β¯A∥v∥22, 0 < µA ≤ β¯A. (24) Equivalently, the restriction HAS(θ) := PSHA(θ)PS|SA,B

satisfies

µAIS ⪯ HAS(θ) ⪯ β¯AIS. (25) In addition, the coupling from SA,B⊥ back into SA,B is weak throughout NA:

∥PSHA(θ)(I − PS)∥2 ≤ ξA, (26) where ξA is small. We choose the refresh step size such that

1 β¯A

. (27)

0 < α ≤

This assumption requires positive curvature only along the shared conflict directions, not local strong convexity in the full parameter space, and ensures that gradient descent on LA contracts the harmful component inside SA,B while the influence of off-subspace coordinates enters only as a controlled perturbation.

#### C.2 Proof of Proposition 1

We provide the full derivation for Proposition 1 and make explicit its sensitivity-based interpretation. Recall that training first reaches a domain-A checkpoint θA∗ and later domain-B training induces a local displacement δB, so the degraded checkpoint is θA∗ + δB. The interference from B to A is

∆A←B = LA(θA∗ + δB) − LA(θA∗ ). (28) By Taylor expansion of LA around θA∗ ,

- 1

- 2

δB⊤HA(θA∗ )δB + RA(δB). (29) Subtracting LA(θA∗ ) gives

LA(θA∗ + δB) = LA(θA∗ ) + gA(θA∗ )⊤δB +

- 1

- 2

∆A←B = gA(θA∗ )⊤δB +

δB⊤HA(θA∗ )δB + RA(δB). (30)

The first term is the possible first-order drift of the domain-A objective, while the second term is the local curvature cost incurred by moving from θA∗ in the direction δB. Here RA(δB) is the third-order Taylor remainder. Under Assumption 1, the Hessian of LA is locally Lipschitz in a neighborhood containing the segment from θA∗ to θA∗ + δB, so the standard remainder bound gives

ρA

6 ∥δB∥32. (31) Under Assumption 2, the checkpoint θA∗ is approximately stationary for domain A, which gives

|RA(δB)| ≤

|gA(θA∗ )⊤δB| ≤ ∥gA(θA∗ )∥2∥δB∥2 ≤ εA∥δB∥2. (32)

Substituting these two estimates into the expression for ∆A←B and applying the triangle inequality yields

- 1

- 2

ρA

δB⊤HA(θA∗ )δB ≤ εA∥δB∥2 +

6 ∥δB∥32, (33) which implies

∆A←B −

- 1

- 2

δB⊤HA(θA∗ )δB + O εA∥δB∥2 + ∥δB∥32 . (34)

∆A←B =

- This proves Proposition 1.

The statement can also be written in a form that separates update magnitude from directional sensitivity. For any nonzero direction v, define the local second-order sensitivity of domain A as

v⊤HA(θ)v ∥v∥22

SA(2)(θ;v) :=

. (35)

Taking v = δB yields the equivalent expression ∆A←B =

- 1

- 2∥δB∥22 SA(2)(θA∗ ;δB) + O εA∥δB∥2 + ∥δB∥32 . (36)

Thus, once θA∗ is approximately stationary and δB is local, forgetting is not determined by update size alone. A small or sparse later-domain update can still produce visible degradation if its direction

has large second-order sensitivity under LA, whereas a larger update in low-curvature directions may cause little damage. In particular, if εA is small enough that the linear term is negligible compared with the quadratic term, for example, when εA = o(∥δB∥2) as ∥δB∥2 → 0, then the dominant contribution to interference is the curvature term 12δB⊤HA(θA∗ )δB rather than first-order drift. This is the precise sense in which sequential updates induce second-order local damage.

#### C.3 Proof of Proposition 2

Proposition 2 sharpens Proposition 1 by showing that the dominant second-order damage is concentrated in the shared active conflict subspace SA,B rather than spread across the whole parameter space. Let PS be the projection onto SA,B. Starting from Proposition 1,

∆A←B =

- 1

- 2

δB⊤HA(θA∗ )δB + O εA∥δB∥2 + ∥δB∥32 . (37)

Under Assumption 4,

(δB)⊤HA(θA∗ )δB − (PSδB)⊤HA(θA∗ )(PSδB) ≤ γA∥δB∥22. (38) That is, the full quadratic term differs from its restriction to the shared conflict subspace by at most a quadratic residual. Multiplying this bound by 21 and combining it with Proposition 1 yields

- 1

- 2

(PSδB)⊤HA(θA∗ )(PSδB) + O εA∥δB∥2 + γA∥δB∥22 + ∥δB∥32 . (39) Equivalently,

∆A←B =

γA 2 ∥δB∥22 +

- 1

- 2

ρA 6 ∥δB∥32. (40)

(PSδB)⊤HA(θA∗ )(PSδB) ≤ εA∥δB∥2 +

∆A←B −

- This proves Proposition 2. In particular, when εA, γA, and ∥δB∥2 are sufficiently small, the leading contribution to cross-domain degradation is the component of the later-domain update inside the low-dimensional shared active conflict subspace. The decomposition underlying this bound makes the localization explicit. Write

##### δB = PSδB + (I − PS)δB. (41) Write

x := PSδB,y := (I − PS)δB, (42) so that δB = x + y. The quadratic term in Proposition 1 can then be expanded as

δB⊤HA(θA∗ )δB = (x + y)⊤HA(θA∗ )(x + y). (43) Using bilinearity of the quadratic form,

(x + y)⊤HA(θA∗ )(x + y) = x⊤HA(θA∗ )x + x⊤HA(θA∗ )y

(44)

+ y⊤HA(θA∗ )x + y⊤HA(θA∗ )y.

Since HA(θA∗ ) is the Hessian of a twice-differentiable scalar objective, it is symmetric, so

x⊤HA(θA∗ )y = y⊤HA(θA∗ )x. (45) Therefore the two cross terms combine, giving

δB⊤HA(θA∗ )δB = (PSδB)⊤HA(θA∗ )(PSδB)

+ 2(PSδB)⊤HA(θA∗ )(I − PS)δB

(46)

+ ((I − PS)δB)⊤HA(θA∗ )(I − PS)δB.

The first term is the in-subspace contribution isolated in Proposition 2. It measures the part of the second-order damage caused by the component of the later-domain update that lies directly inside SA,B. The second term measures leakage between the conflict subspace and its orthogonal complement: even if part of the update lies outside SA,B, curvature can couple that off-subspace motion back into the sensitive directions of domain A. The third term captures purely off-subspace curvature effects.

Assumption 4 is precisely what makes this decomposition useful. It states that the latter two terms are bounded by a quadratic residual, so the leading contribution to forgetting is carried by the projection PSδB. In this sense, Proposition 2 refines Proposition 1: Proposition 1 says that forgetting is controlled by a second-order quadratic form, while Proposition 2 identifies where the dominant part of that quadratic damage is located.

This decomposition clarifies an important point: low parameter-overlap across domains does not imply low interference. A later-domain update may modify only a small or apparently unrelated subset of parameters, yet still cause substantial forgetting if its projection enters the small set of directions along which the previous-domain objective is locally sensitive. Conversely, a larger update can remain relatively benign when most of it lies outside these sensitive shared directions.

This also connects back to the near-orthogonality result in Section 4. If global gradients are nearly orthogonal, then immediate first-order cross-domain conflict is weak, but localized second-order displacement inside SA,B can still dominate. Thus, full-model near-orthogonality, sparse edits, and low edit overlap are compatible with selective degradation, because the relevant conflict is carried by a small shared active subspace rather than by global gradient antagonism.

#### C.4 Proof of Theorem 1

Let

et = θt − θA∗ , eSt = PSet. (47) The refresh iteration is

θt+1 = θt − αgA(θt), (48) which implies

et+1 = et − αgA(θt). (49) Projecting onto the shared conflict subspace gives

eSt+1 = PSet − αPSgA(θt) = eSt − αPSgA(θt). (50)

By Assumption 1, the Hessian of LA is locally Lipschitz near θA∗ . Since Assumption 6 ensures that the refresh trajectory remains inside a local neighborhood NA of θA∗ , we may write, for each iterate along this trajectory,

gA(θt) = gA(θA∗ ) + HA(θA∗ )et + rt, (51)

where the remainder satisfies ∥rt∥2 ≤ cA∥et∥22 for some local constant cA > 0. Substituting this into the projected recursion gives

eSt+1 = eSt − αPSgA(θA∗ ) − αPSHA(θA∗ )et − αPSrt. (52) Decompose et = eSt + e⊥t with e⊥t = (I − PS)et. Then

PSHA(θA∗ )et = PSHA(θA∗ )PSet + PSHA(θA∗ )(I − PS)et. (53) By Assumption 6, the second term is controlled by weak cross-subspace coupling:

∥PSHA(θA∗ )(I − PS)et∥2 ≤ ξA∥e⊥t ∥2. (54) Thus the projected recursion becomes

eSt+1 = (I − αPSHA(θA∗ )PS)eSt − αPSgA(θA∗ ) + ζt, (55)

where ζt = −αPSHA(θA∗ )(I − PS)et − αPSrt collects off-subspace and higher-order local errors. Consequently,

∥ζt∥2 ≤ αξA∥e⊥t ∥2 + cAα∥et∥22 (56)

along the refresh trajectory. Under Assumption 6, µAIS ⪯ HAS(θ) ⪯ β¯AIS and 0 < α ≤ β¯1

for all θ ∈ NA. Hence every eigenvalue of I − αPSHA(θA∗ )PS on SA,B lies in [0,1 − αµA], so

A

∥I − αPSHA(θA∗ )PS∥2 ≤ 1 − αµA. (57) Moreover, by Assumption 2,

∥PSgA(θA∗ )∥2 ≤ ∥gA(θA∗ )∥2 ≤ εA. (58) Taking norms in the projected recursion yields

∥eSt+1∥2 ≤ (1 − αµA)∥eSt ∥2 + αεA + αξA∥e⊥t ∥2 + cAα∥et∥22. (59)

As long as refresh stays in the same local neighborhood, we do not need to derive a separate dynamical bound on e⊥t ; instead, we treat the off-subspace contribution as part of a local perturbation term and absorb

ηloc,t := ξA∥e⊥t ∥2 + cA∥et∥22 (60) into a local error sequence and write

∥eSt+1∥2 ≤ (1 − αµA)∥eSt ∥2 + α εA + ηloc,t . (61) Unrolling the recursion gives

∥eSt ∥2 ≤ (1 − αµA)t∥eS0 ∥2 +

t−1

(1 − αµA)t−1−k αεA + αηloc,k . (62)

k=0

###### Table 5: Full benchmark-level results. Step denotes the selected checkpoint step for the corresponding run. AVG is the average over the four evaluation domains: Math, Code, QA, and CW.

Math Code QA CW

Model Step

AVG AIME24 AIME25 AIME26 OlympiadBench HMMT Math LiveCodeBench-v6 SuperGPQA-test MMLU-Pro QA WritingBench

Base – 48.75 42.50 40.63 61.68 22.40 43.19 29.57 47.32 73.96 60.64 82.44 53.96 Maths 525 71.77 67.71 69.79 77.95 46.98 66.84 34.65 47.84 73.68 60.76 81.38 60.91 Codes 600 61.46 61.46 62.71 71.38 41.15 59.63 52.67 47.84 73.93 60.89 82.40 63.90 QAs 705 58.65 55.83 56.77 70.19 35.10 55.31 32.07 51.85 74.77 63.31 81.76 58.11 CWs 120 44.27 38.85 35.00 59.74 21.04 39.78 28.15 47.16 73.84 60.50 86.24 53.67 Codeo 600 61.46 61.46 62.71 71.38 41.15 59.63 52.67 47.84 73.93 60.89 82.40 63.90 Matho 180 73.33 66.56 70.83 77.57 44.17 66.49 50.69 47.53 73.51 60.52 81.44 64.79 QAo 300 64.38 59.17 61.98 72.73 41.25 59.90 50.99 49.98 74.69 62.34 81.79 63.76 CWo 135 62.29 56.04 58.65 70.89 40.42 57.66 50.47 49.91 74.77 62.34 86.52 64.25 CGPO 630 65.10 58.13 66.88 75.35 44.17 61.93 50.05 50.90 74.05 62.48 86.73 65.30 JT 1035 69.58 65.63 65.52 77.12 46.15 64.80 48.61 50.04 74.18 62.11 86.97 65.62 Re-Math 135 74.27 64.48 68.65 77.06 45.73 66.04 51.05 50.28 74.69 62.49 85.96 66.39 QA → Math 675 71.15 66.77 68.54 77.04 43.13 65.33 32.53 51.55 74.44 63.00 82.01 60.72 Math → CW 120 72.71 67.71 70.42 77.75 47.19 67.16 32.82 48.31 74.04 61.18 85.77 61.73

Since e0 = δB, we have eS0 = PSδB. If we define ηloc := supk ηloc,k over the local refresh trajectory, then

t−1

εA + ηloc µA

(1 − αµA)t−1−kα εA + ηloc,k ≤

. (63)

k=0

Therefore,

εA + ηloc µA

∥PS(θt − θA∗ )∥2 ≤ (1 − αµA)t∥PSδB∥2 + O

. (64)

Here ηloc summarizes bounded off-subspace and higher-order local errors along the refresh trajectory. In the ideal locally quadratic case with gA(θA∗ ) = 0 and ξA = 0, these error terms vanish, and the recursion reduces to

eSt+1 = (I − αPSHA(θA∗ )PS)eSt , (65) so that

∥PS(θt − θA∗ )∥2 ≤ (1 − αµA)t∥PSδB∥2. (66) This proves the theorem.

#### C.5 Stability of Other Domains under Near-Orthogonal Gradients

A desirable property of refresh is that it should recover the target domain without substantially damaging other domains. We now show that this follows from global near-orthogonality and small refresh steps.

Consider one refresh step on domain A:

θt+1 = θt − αgA(θt). (67) For another domain C ̸= A, local smoothness gives

βCα2

2 ∥gA(θt)∥22. (68) By Assumption 5,

LC(θt+1) ≤ LC(θt) − α⟨gC(θt),gA(θt)⟩ +

|⟨gC(θt),gA(θt)⟩| ≤ κA,C∥gC(θt)∥2∥gA(θt)∥2. (69) Therefore,

βCα2 2 ∥gA(θt)∥22. (70)

LC(θt+1) − LC(θt) ≤ ακA,C∥gC(θt)∥2∥gA(θt)∥2 +

Corollary 1. Under Assumptions 1 and 5, the one-step increase of another domain objective LC during refresh on domain A is bounded by

βCα2 2 ∥gA(θt)∥22. (71)

LC(θt+1) − LC(θt) ≤ ακA,C∥gC(θt)∥2∥gA(θt)∥2 +

Thus, when global gradients are nearly orthogonal, refresh steps are small, and the involved gradient norms are not too large, the one-step damage to other domains is limited.

Math

Code

###### QA

CW

0.53

1.00

0.64

0.53

0.52

Acc(mean@16)

0.52

0.52

0.99

0.62

0.52

0.51

0.60

0.99

0.52

0.51

0.52

0.58

0.50

0.99

0.51

0.49

0.56

0.51

0.99

0.49

0 25 50 75 100 125

0 25 50 75 100 125

0 25 50 75 100 125

0 25 50 75 100 125

Step

Step

Step

Step

Figure 9: Validation dynamics during Re-Math refresh from CWo across the four domains.

The same calculation also explains the instantaneous interference during sequential training. If one instead performs a single update on domain B, namely θ+ = θ − αgB(θ), then local smoothness of LA gives

βAα2

2 ∥gB(θ)∥22. (72) Under near-orthogonality, the first-order cross term is again small, so the immediate harm from a later-domain step is controlled mainly by higher-order curvature terms rather than by strong global gradient antagonism.

LA(θ+) − LA(θ) ≤ −α⟨gA(θ),gB(θ)⟩ +

For a short refresh trajectory of T steps, summing the one-step bound yields

T−1

βCα2 2 ∥gA(θt)∥22 . (73)

LC(θT) − LC(θ0) ≤

ακA,C∥gC(θt)∥2∥gA(θt)∥2 +

t=0

Hence, when refresh is short and gradients remain nearly orthogonal along the trajectory, cumulative degradation to another domain stays limited.

An equivalent total-displacement view is obtained by writing rT := θT − θ0. By local smoothness,

βC

LC(θT) − LC(θ0) ≤ gC(θ0)⊤rT +

2 ∥rT∥22. (74) If the starting point is also approximately stationary for domain C, so that ∥gC(θ0)∥2 ≤ εC, then

βC

2 ∥rT∥22. (75) Therefore, another domain remains stable whenever the total refresh displacement is small, even if we do not track every intermediate step in detail.

LC(θT) − LC(θ0) ≤ εC∥rT∥2 +

This corollary complements Theorem 1. Refresh can strongly reduce the target-domain conflict component because the target gradient is aligned with the local corrective direction in SA,B. At the same time, its effect on other domains is bounded because full-domain gradients are approximately orthogonal at the global scale. This separation between local conflict and global near-orthogonality explains how Re-Math can recover Math while largely preserving Code, QA, and CW.

#### C.6 Extension: Alternating Refresh as Local Multi-Objective Optimization

The same local view also explains why alternating refresh across domains can approach a local compromise among multiple objectives. The key point is that one refresh cycle is not best understood as “forgetting A and then forgetting B” again; to first order, it behaves like one descent step on a single weighted local objective. Consider two domains A and B. One cycle of alternating refresh first updates on A:

θ′ = θ − αAgA(θ), (76) and then updates on B:

θ′′ = θ′ − αBgB(θ′). (77) For small step sizes, local smoothness gives

gB(θ′) = gB(θ) + O(αA∥gA(θ)∥2). (78) Thus, one refresh cycle satisfies

θ′′ = θ − αAgA(θ) − αBgB(θ) + O(αAαB). (79)

To first order, alternating refresh therefore follows gradient descent on the weighted local objective

Φ(θ) = αALA(θ) + αBLB(θ). (80) More generally, alternating refresh over multiple domains approximates gradient descent on

Φ(θ) =

d∈D

A stationary point of this local objective satisfies

αdLd(θ). (81)

αdgd(θ†) ≈ 0. (82)

d∈D

For positive weights {αd}, this is the first-order stationarity condition of a weighted-sum scalarization, which is a standard local proxy for Pareto compromise under regularity assumptions. Therefore, while a single refresh targets recovery of one degraded domain, alternating refresh can be viewed as a local mechanism for approaching a Pareto-stationary compromise among domains under the weighted-sum scalarization view. In particular, it clarifies why repeated refresh can keep improving for several rounds without implying that all single-domain optima are simultaneously reachable.

### D Additional Validation Results

#### D.1 Full Task-Level Results

- Table 5 reports the full benchmark-level results for all model checkpoints used in the main task-level validation. For sequential and refresh runs, the step number in parentheses denotes the selected checkpoint step within the last training stage.

#### D.2 Refresh Dynamics

52.50

Code Math

66.50

- Figure 9 further confirms that Re-Math behaves as a short local correction rather than full retraining. Along the refresh trajectory from CWo, Math performance increases rapidly in the early steps and then approaches saturation, while Code, QA, and CW remain largely stable with only small fluctuations. This dynamic pattern supports the contraction view in Theorem 1: the refresh update mainly removes the Math-sensitive harmful displacement induced by later-domain training, instead of globally overwriting the policy or trading off other domains for Math recovery.
- Figure 10 provides a complementary view through the Code → Math → Re-Code trajectory, further clarifying our local recoverability result. In the early refresh stage, Code improves while Math changes only mildly, consistent with Theorem 1: a short refresh can contract the harmful displacement on the shared conflict directions and recover the target domain within a local neighborhood. However, as Re-Code training continues, the update is no longer well described as a small local correction, and the Math decline becomes more visible. This behavior is exactly what Proposition 1 predicts: once the accumulated displacement grows, second-order damage to the previously optimized domain can increase even without strong global gradient opposition. Figure 10 should therefore be interpreted as evidence for a local small-update regime rather than unrestricted continued specialization: refresh is effective when it quickly removes the target-domain conflict component, but prolonged single-domain training can again perturb directions to which the other domain is locally sensitive.

-0.24

| |
|---|

52.00

| |
|---|

66.00

51.50

| |
|---|

Code

Math

65.50

+0.84

51.00

| |
|---|

65.00

| |
|---|

50.50

| |
|---|

| |
|---|

64.50

0 15 30 45 60 75 90 105 120 Step

Figure 10: Math and Code performance changes during Re-Code on Matho.

#### D.3 Directional Asymmetry of Interference

We additionally compare QA → Math with the reverse Math → QA ordering. As shown in Figure 11, in QA → Math, Math improves steadily during the Math stage while QA remains largely stable, unlike the stronger Math degradation observed when QA is trained after Math.

This provides further evidence that crossdomain interference is directional rather than symmetric. This asymmetry is also consistent with our local perturbation analysis and Proposition 1: the damage to an earlier domain depends on whether the later update enters its curvature-sensitive shared directions. Here, the Math update appears to perturb QA-sensitive directions only weakly, whereas the QA update more strongly perturbs Math-sensitive directions in the reverse ordering.

- D.4 Direct Rollback on a Coordinate Proxy for the Conflict Subspace

72.0

57.0

69.0

54.0

66.0

51.0

Math

QA

63.0

48.0

60.0

45.0

Math

QA

57.0

QA avg: 50.0 42.0

0 100 200 300 400 500 600 700 800 Step

Figure 11: Validation dynamics for the reverse QA

→ Math ordering.

To more directly probe the localization claim behind Proposition 2, we intervene on the checkpoint pair Codeo → Matho → QAo. Treating checkpoints as their parameter vectors, define the QAinduced displacement as

δQ|M = QAo − Matho. (83) We restrict the intervention to MLP neurons, using the same neuron definition as in Appendix B.3. For neuron i in layer ℓ, let ∆vM,i(ℓ) and ∆vQ,i(ℓ) denote the Math and QA task vectors in the concatenated gate/up/down parameterization from Appendix B.5. We build a neuron-level proxy score

Si(ℓ) = A(iℓ)Mi(ℓ)Ci(ℓ), (84) where

A(iℓ) = min Pct(a(M,iℓ) ),Pct(a(Q,iℓ) ) , (85)

Mi(ℓ) = Pct ∥∆vQ,i(ℓ) ∥2 , (86) and

Ci(ℓ) = max 0, −cos ∆vM,i(ℓ) ,∆vQ,i(ℓ) . (87)

Here a(d,iℓ) is the dataset-level activation score defined in Appendix B.4, and Pct(·) denotes percentile rank within a layer.

For activation collection, we use 512 samples and first generate up to 4096 tokens, followed by masked teacher-forcing forward passes to record MLP intermediate activations. Unless otherwise stated, we use a total intervention budget

B = β × L × dint, (88) with β = 2%, L = 36, and dint = 9728, giving B = 7004 out of 350,208 total MLP neurons.

To distribute the global budget across layers, we first compute a layer score by averaging the top-ρ neuron scores within each layer, with ρ = 10%. The final layer budget is then allocated proportionally to

exp (S¯(ℓ))α , (89)

with α = 2.0, followed by rounding so that ℓ Bℓ = B. Given the selected neuron set Sˆ, we revert only the QA increment on its gate/up/down parameters:

θrev = QAo − PSˆδQ|M, (90)

where PSˆ is the coordinate projection induced by Sˆ. This intervention requires no additional optimization and provides a direct proxy-level test of whether a sparse subset of the QA displacement

carries measurable Math damage.

For single-factor selectors (A, M, or C), we report two variants. In the fixed-budget variant, we keep the layer allocation from the full A × M × C selector and only change the within-layer ranking. In

- Table 6: Additional ablations for the proxy conflict-subspace intervention at β = 2%. Fixedbudget variants keep the layer allocation of the full selector and change only within-layer ranking; recomputed-budget variants recompute both ranking and layer allocation.

Selector AIME24 AIME25 AIME26 OlyBench HMMT Math Avg ∆ vs QAo Recovery SuperGPQA MMLU-Pro QA Avg ∆ QA Avg Matho 73.33 66.56 70.83 77.57 44.17 66.49 +6.59 100.0% 47.53 73.51 60.52 −1.81 QAo 64.38 59.17 61.98 72.73 41.25 59.90 0.00 0.0% 49.98 74.69 62.33 0.00 Random 63.54 59.90 60.00 73.05 40.94 59.49 −0.42 −6.3% 49.97 74.71 62.34 +0.01 A×M×C (main) 66.25 60.21 63.65 73.42 42.71 61.25 +1.35 20.4% 50.11 74.43 62.27 −0.06 M×C 66.67 60.00 61.77 73.36 43.75 61.11 +1.21 18.3% 50.16 74.58 62.37 +0.04 A×C 65.00 59.48 62.81 73.50 41.98 60.55 +0.65 9.9% 49.92 74.65 62.29 −0.05 A×M 65.73 61.46 61.77 72.97 40.73 60.53 +0.63 9.6% 50.25 74.49 62.37 +0.04 M (fixed budget) 64.69 60.10 63.12 73.50 42.40 60.76 +0.86 13.1% 49.93 74.56 62.25 −0.09 A (fixed budget) 65.52 59.90 63.75 73.23 43.85 61.25 +1.35 20.4% 49.93 74.55 62.24 −0.09 C (fixed budget) 65.83 60.21 61.77 73.30 42.60 60.74 +0.84 12.7% 49.94 74.59 62.27 −0.07 M (recomputed) 65.21 59.38 63.33 73.47 42.71 60.82 +0.92 13.9% 50.14 74.69 62.41 +0.08 A (recomputed) 66.15 60.42 61.88 73.41 43.12 61.00 +1.09 16.6% 50.18 74.56 62.37 +0.04 C (recomputed) 65.52 61.35 62.71 73.23 42.08 60.98 +1.08 16.3% 50.15 74.57 62.36 +0.03

the recomputed-budget variant, we recompute the layer allocation from the corresponding selector itself.

- Table 6 shows that the full selector recovers 20.4% of the QA-induced Math loss, while the two-factor selector M ×C remains close at 18.3%. By contrast, removing either update magnitude or directional conflict reduces recovery to about half of the full result. Interestingly, A-only under the full selector’s layer budget matches the full selector, whereas A-only with its own recomputed budget drops to 16.6%. This indicates that shared activation alone is not sufficient to explain the best-performing intervention; part of its apparent strength comes from inheriting the conflict-aware layer allocation.

- Figure 12 shows that the depth profile of the intervention is not uniform. Once the conflict term is included, the layer budget becomes strongly U-shaped, concentrating more neurons in shallow and late layers; without the conflict term, the budget becomes much flatter. The selector M × C is the closest two-factor approximation to the full selector, whereas A-only can exhibit much smaller support overlap despite competitive recovery in the fixed-budget setting. This suggests that the proxy support is partly redundant: different sparse coordinate subsets can play similar functional roles as long as they target the harmful QA displacement.
- Figure 13 shows a clear dose-response pattern. Increasing the revert budget from 1% to 4% raises recovery from 2.0% to 29.4%, after which performance fluctuates around a saturation regime rather than improving monotonically. This suggests that a relatively small fraction of MLP neurons carries a substantial part of the harmful QA displacement.

The MLP-only results above show that a sparse rollback on proxy conflict coordinates can already recover a substantial part of the QA-induced Math loss, but the recovery saturates early and remains incomplete. To test whether the recoverable harmful displacement also extends beyond MLP coordinates, we extend the intervention to attention layers.

For layer ℓ, query head h, and row index r, we define a fine-grained attention unit

AttnUnit(ℓ,h,r) = {∆WQ(ℓ)[h,r,:], ∆WO(ℓ)[:,h,r], √1g∆WK(ℓ)[κ(h),r,:], √1g∆WV(ℓ)[κ(h),r,:]}. (91)

where κ(h) = ⌊h/g⌋ maps a query head to its shared KV head under grouped-query attention, and g = 4 is the group size. Since each layer has 32 query heads and head dimension 128, this gives 32 × 128 = 4096 attention units per layer, or 147,456 attention units across all 36 layers.

Unlike the MLP case, collecting per-unit activation statistics for attention would require decomposing attention outputs at the head-row level. We therefore use a conservative two-factor attention score

##### Sattn(ℓ,h,r) = Mattn(ℓ,h,r) Cattn(ℓ,h,r), (92)

where Mattn measures the QA update magnitude of the unit and Cattn measures directional conflict between the Math and QA task vectors. MLP neurons keep the original three-factor score

##### Smlp(ℓ,i) = A(iℓ)Mi(ℓ)Ci(ℓ). (93)

###### Layer Scores by Condition

1.0

0.9

LayerScore(normalized)

0.8

0.7

0.6

0.5

A×M×C

A (recomputed budget) M (recomputed budget) C (recomputed budget)

M×C A×C A×M

0.4

0.3

###### Layer Budget Allocation (total=7004)

A×M×C

A (recomputed budget) M (recomputed budget) C (recomputed budget) uniform (194.6)

500

M×C A×C A×M

LayerBudget(#neurons)

400

300

200

100

###### Per-Layer Overlap with A×M×C Set

70

M×C A×C A×M A (fixed budget)

M (fixed budget) C (fixed budget) A (recomputed budget)

M (recomputed budget) C (recomputed budget) Random

JaccardOverlapwithMain(%)

60

50

40

30

20

10

0

0 5 10 15 20 25 30 35

Layer

- Figure 12: Layer-wise neuron selection analysis. Top: normalized layer scores; middle: budget allocation; bottom: Jaccard overlap with the A × M × C set across 36 layers under different scoring criteria. The conflict factor C dominates the non-uniform layer budget distribution, and two-factor combinations containing C recover the highest overlap with the full composite set.

0 4 8 12 16 20 24 28 32 36

Neuron Budget (%)

0

10

20

30

40

50

60

70

80

MathRecovery(%)

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

MLP-only

MLP+Attn

Figure 13: Math recovery under different intervention budgets.

To keep the comparison with the MLP-only experiments fair, we define the total budget in the same units as before:

B = β × Nmlp, (94)

where Nmlp = 36 × 9728 = 350,208. We first compute a joint layer score by combining the top-ρ scores from MLP and attention units:

sℓ =

S¯mlpρ (ℓ)dint + S¯attnρ (ℓ)nattn dint + nattn

, (95)

with ρ = 0.1, dint = 9728, and nattn = 4096. The layer budget is then allocated by

bℓ = B ·

(sℓ)α ℓ′(sℓ′)α

, α = 2.0. (96)

Within each layer, we merge all MLP-neuron and attention-unit scores into a single ranked list and select the top-bℓ units regardless of type.

For selected MLP neurons, the revert operation is identical to the MLP-only setting. For a selected attention unit AttnUnit(ℓ,h,r), we fully revert the corresponding WQ and WO slices. For shared KV weights, if k out of the g query heads attached to the same KV head select row r, we revert the corresponding WK and WV rows by fraction k/g.

- Figure 13 shows the joint MLP+Attn sweep. The joint selector is not uniformly better at the smallest budget: at β = 2% it recovers 12.7% of the QA-induced Math loss, below the MLP-only result of

20.4%. However, from β = 4% onward it consistently surpasses the MLP-only sweep, reaching 36.1% recovery at 4% and 73.6% at 32%. This pattern indicates that the MLP-only proxy already captures the highest-yield recoverable component under extremely sparse rollback, while the joint intervention reveals additional recoverable damage outside MLP coordinates. Recovery still remains incomplete even at larger budgets, suggesting residual interference outside the reverted MLP+Attn coordinates or non-coordinate-aligned effects. Although the best recovery is obtained with a relatively large 32% budget, this is consistent with the redundancy observed above: multiple sparse coordinate subsets can have similar functional effects, and the current proxy selector is not expected to isolate the conflict subspace optimally. Improving the identification score to more precisely localize the shared conflict subspace is therefore a natural direction for future work.

