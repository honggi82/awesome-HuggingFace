## Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics

Ziwen Xu1,2, Chenyan Wu1, Hengyu Sun1, Haiwen Hong2*, Mengru Wang1, Yunzhi Yao1, Longtao Huang2, Hui Xue2, Shumin Deng1, Zhixuan Chu1, Huajun Chen1, Ningyu Zhang1* 1Zhejiang University, 2Alibaba Group

# arXiv:2602.02343v3[cs.CL]12Apr2026

Abstract

Methods for controlling large language models (LLMs), including local weight fine-tuning, LoRA-based adaptation, and activation-based interventions, are often studied in isolation, obscuring their connections and making comparison difficult. In this work, we present a unified view that frames these interventions as dynamic weight updates induced by a control signal, placing them within a single conceptual framework. Building on this view, we propose a unified preference-utility analysis. This analysis separates control effects into two components: preference, defined as the tendency toward a target concept, and utility, defined as coherent and task-valid generation. Both components are measured on a shared log-odds scale using polarity-paired contrastive examples. Across methods, we observe a consistent trade-off between preference and utility: stronger control increases preference while predictably reducing utility. We further explain this behavior through an activation manifold perspective, in which control shifts representations along target-concept directions to enhance preference, while utility declines primarily when interventions push representations off the model’s valid-generation manifold. Finally, we introduce a new steering approach SPLIT guided by this analysis that improves preference while better preserving utility1.

### 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities and are increasingly deployed in real-world applications (Zhao et al., 2023). Growing demands for safety, controllability, and personalization make reliable control over model behavior a central challenge. To address this, prior work has developed diverse

* Corresponding Author. 1https://github.com/zjunlp/EasyEdit/blob/main/

examples/SPLIT.md.

| | |b+<br><br>|
|---|---|---|
|W+<br><br>|m∆W| |

m

m∆b

Preference

b ∆ b

m

z

∆W

W

Local Weight

Local Weight LoRA Vector

b

b

m

Multiplier m

0

- WA
- WB

W+m∆W

W

Utility

Local Weight

LoRA

LoRA Vector

m

v

b+ m∆b

b

W

W

Vector

Multiplier m

0

Figure 1: The figure illustrates how different methods operate on the linear layers of the model. We present a unified view in which diverse large language model intervention methods are casted as dynamic weight updates. The right panel shows the changes in model utility and preference across different control methods under varying intervention multipliers. Further details are provided in Section 3.

paradigms for controlling LLMs, spanning trainingtime adaptations, such as local weight fine-tuning and parameter-efficient methods like LoRA (Hu et al., 2022b; Ding et al., 2023; Mao et al., 2025), and inference-time interventions, including activation-level steering via hidden-state manipulation (Rimsky et al., 2024; Han et al., 2024; Bartoszcze et al., 2025; Shu et al., 2025).

Despite their empirical success, these approaches are often studied in isolation, under different assumptions, objectives, and evaluation protocols. This fragmentation hinders rigorous comparison and obscures shared failure modes. In this work, as shown in Figure 1, we mathematically observe that local weight fine-tuning, LoRA, and activation-level steering can all be formulated as instances of a common dynamic weight update framework (Eq. 1). Building on this unified perspective, we introduce a preference–utility analysis and show that, across methods instantiated within this framework, both preference and utility exhibit consistent, predictable patterns as control strength varies.

hi+1 = (W + m1 ∆W)hi + (b + m2 ∆b). (1)

Note that a particular challenge in controlled text generation is the trade-off between enforcing the target concept and preserving task validity: as control strength increases, the target attribute is amplified, but undesirable side effects—such as incoherence, instruction violations, or context drift—also become more frequent, reducing overall task effectiveness. Moreover, because control quality is typically evaluated via realized outputs, degradation in task validity can confound assessments and obscure the intended concept signal. Guided by this mechanistic understanding, we propose a training objective that explicitly optimizes preference while preserving utility, and experimentally demonstrate that it achieves superior performance.

Our contributions are as follows:

- • Unified View. We propose a unified view of dynamic weight updates that casts local weight fine-tuning, parameter-efficient finetuning (e.g., LoRA), and activation interventions (steering) into a common intervention form. Building on this view, we introduce a unified preference–utility analysis and show that, across methods instantiated within the dynamic-update framework, both preference and utility exhibit consistent regularities as control strength varies.
- • Preference–Utility Analysis. We introduce an activation manifold hypothesis and analyze preference and utility under this assumption, suggesting that preference is jointly determined by (i) the projection onto a targetpreference direction and (ii) activation validity, which degrades as representations deviate from the manifold, while utility degradation is primarily driven by this off-manifold deviation and the resulting activation invalidation. We further derive two quantitative relationships between the preference log-odds and m, and between the utility log-odds and m, and validate them with high-R2 fits.
- • New Steering Method. Guided by this mechanism, we introduce SPLIT, a training objective that explicitly optimizes preference while preserving utility, and demonstrate that it achieves better overall performance.

### 2 Preliminary

#### 2.1 Intermediate Representations in LLMs

During the forward propagation of intermediate layers in LLMs, several key representations occur at specific points in the computation, such as FFN outputs, residual stream states, and linear projections within the attention mechanism (Query, Key, Value, and final output). These representations can be uniformly expressed as the output of an affine transformation:

hi+1 = Whi + b, (2)

where hi and hi+1 denote the input and output representations of a linear layer, and W, b are its weights and biases.

For example, in an FFN block, the up-projection is computed as hmid = Wuphin + bup, followed by a non-linear activation, hmid,act = σ(hmid), and then the down-projection is computed as hout = Wdown hmid,act + bdown. Similarly, the Q, K, V , and output projections in the attention module follow the same affine form as in Eq. 2.

#### 2.2 Parameter Update

We consider two parameter adaptation methods for large language models: Low-Rank Adaptation (LoRA) and local weight fine-tuning.

LoRA LoRA freezes the original weight matrix W and introduces a trainable low-rank update ∆W = BA, where A ∈ Rr×k, B ∈ Rd×r, and the rank r ≤ min(d,k). At inference, the adapted weights are given by W ← W + ∆W. In its canonical form, LoRA applies only to the weight matrix while keeping the bias term b fixed, although extensions exist that also adapt biases.

Local Weight Fine-tuning Local weight finetuning updates parameters within a restricted subset of the network, leaving all other parameters frozen. It can be applied to any layer or parameter type, with full-parameter training representing the special case where the subset covers the entire model. A generic update for the weight matrix W and bias vector b can be expressed as: (W,b) ← (W + ∆W, b + ∆b). In our experiments, parameter updates are applied only to the FFN down-projection layer.

#### 2.3 Activation Steering

Activation Steering Activation steering modifies intermediate representations during inference by

adding a steering vector to selected activations. Its mathematical form can be written as

##### hi+1 = Whi + b + mv, (3)

where v is a predetermined direction and m is a scalar coefficient controlling its magnitude. This approach builds on the linear representation hypothesis (Mikolov et al., 2013; Pennington et al., 2014; Nanda et al., 2023; Tigges et al., 2023; Park et al., 2024) that abstract concepts correspond approximately to linear subspaces of representation space.

The steering vector v can be equivalently expressed as a bias adjustment ∆b, yielding b ← b + m∆b. This formulation highlights activation steering as a special case of dynamic parameter update, closely related to methods such as LoRA and local weight fine-tuning.

From a unified perspective, both parameter updates and activation steering operate by injecting a change vector ∆h into intermediate representations during forward propagation, differing only in the mechanism by which ∆h is generated. More related works can be found in Appendix 6.

### 3 Unified View of Dynamic Weights in Inference

We present a unified framework for dynamic interventions during inference, as illustrated in Figure 1. Our unified view has three components: (i) a unified dynamic weights intervention view that expresses local weight updates, LoRA, and activation steering as dynamic weight updates, (ii) a unified analysis view based on preference/utility log-odds, and (iii) a unified dynamics observation showing consistent preference–utility response patterns across intervention forms.

#### 3.1 Unified Dynamic Weight Formulation

We propose a unified framework that encompasses both parameter update methods and activation steering methods, by viewing them as dynamic weight updates. Under this formulation, both can be expressed through a shared affine transformation view of intermediate representations; detailed derivations and formulations are provided in Section 2.

Formally, the dynamic modification of the weight matrix W and bias vector b during inference can be written as:

hi+1 = (W + m1 ∆W)hi + (b + m2 ∆b), (4)

where ∆W and ∆b are update terms, and m1,m2 are scalar scaling coefficients controlling their magnitudes (Fierro and Roger, 2025). In other words, the original parameters are updated to W′ = W + m1 ∆W and b′ = b + m2 ∆b before computing the next-layer activation.

When a model weight is modified, the effect can be equivalently interpreted from the activation perspective, as a change to the activation at the corresponding position. In this view, diverse intervention methods are unified as adding a change term to the activation:

##### ∆h = m1 ∆W hi + m2 ∆b. (5)

Under this unified view, local weight fine-tuning, LoRA, and activation steering are all specific instances, differing only in which components are updated: local weight fine-tuning modifies both W and b; LoRA modifies W via low-rank factors; activation steering modifies only b. Table 1 summarizes their affine forms, corresponding activation update, and parameter sizes.

Notably, introducing explicit scaling coefficients extends traditional formulations and enables continuous control over perturbation strength, a capability that plays a central role in our subsequent analysis.

3.2 Unified Analysis View: Preference and Utility Log-Odds

We analyze intervention effects along two complementary dimensions. Preference denotes the model’s internal inclination toward a target concept, independent of whether the model completion is well-formed. For the prompt “Write a short review for this restaurant”, generating “The food was excellent and the service was wonderful” indicates a positive preference, while “The food was terrible and the service was disappointing” indicates a negative preference. Utility denotes the model’s task competence that is independent of the target concept. It captures whether the model can produce a task-valid completion that is coherent, relevant to the prompt, and consistent with the requested format. For the same prompt, utility is high when the output is a readable restaurant review, regardless of polarity. Utility is low when the output is incoherent such as “food food wonderful ??? service 19% ##”, off-topic such as “Here is a Python script to scrape restaurant data...”, or instruction-violating even if polarity-bearing words appear.

In controlled generation, performance is typically evaluated from the realized outputs. When

###### Form Unified Affine Formula Activation Impact (∆h) Param. Size

Local Weight (W + m ∆W) hi + (b + m ∆b) m (∆W hi + ∆b) din × dout + dout LoRA (W + m BA) hi + b m (BA hi) din × r + r × dout Steering Vector W hi + (b + m ∆b) m ∆b dout

Table 1: All methods in our unified framework, expressed under the affine weight-update formulation and their corresponding activation changes ∆h. din and dout denote the input and output dimensions of the layer; r is the LoRA rank with r ≪ min(din,dout).

preference is increased at the expense of utility, completions often become incoherent or instruction-violating, reducing usability and obscuring the intended concept signal under outputbased evaluation. Therefore, effective model control should shift preference while preserving utility.

Notation. Given a query q, we construct a polarity pair of completions: a concept-positive answer Ap and a concept-negative answer An. We denote their conditional probabilities as P(Ap | q) and P(An | q), and define the corresponding crossentropy losses as Lp ≜ −log P(Ap | q) and Ln ≜ −log P(An| q). We further introduce latent preference probabilities P(pp | q) and P(pn | q), as well as a polarity-invariant task-success probability P(u | q).

Preference–Utility Factorization. Following prior work that assumes concept directions are mutually orthogonal (Bigelow et al., 2025), we likewise treat concept preference as independent from task utility for a given query q. Concretely, for a polarity pair (Ap,An), we decompose

P(Ap | q) = P(u | q)P(pp | q), P(An | q) = P(u | q)P(pn | q), (6)

where P(u | q) is shared across the pair and P(pp | q) + P(pn | q) = 1.

Preference Log-odds. The shared utility cancels in the likelihood ratio, yielding

P(pp | q) P(pn | q)

PrefOdds(q) ≜ log

= Ln − Lp. (7)

Utility Log-odds. The total probability mass assigned to the matched pair recovers utility, P(u | q) = P(Ap | q) + P(An | q); substituting P(A | q) = e−L gives

P(u | q) 1 − P(u | q)

UtilOdds(q) ≜ log

e−Lp + e−Ln 1 − e−Lp − e−Ln . (8)

= log

We use PrefOdds and UtilOdds throughout to track how interventions shift concept preference versus task utility on a common additive scale, with additional derivations in Appendix D.

#### 3.3 Unified Dynamics Observation

Experimental Setup. We evaluate dynamic interventions on two types of tasks: (i) a personality tendency classification task (Psychopathy), and (ii) open-ended generation using PowerSeeking and the top 10 concept subsets from AxBench. We run experiments on Gemma-2-9B-IT at layer 20 and Qwen-2.5-7B-Instruct at layer 14, following Bigelow et al. (2025), and consider three intervention types: local weight, low-rank adaptation LoRA, and vector. We train each type using both the SFT objective and the RePS objective. Additionally, for vector, we include a trainfree method called DiffMean (Marks and Tegmark, 2023). More details are provided in Appendix A.

Metrics. For each query q with matched answers (Ap,An), we compute preference and utility logodds in Eqs. (7) and (8). These metrics allow us to track how preference and utility evolve as we vary the intervention scale m.

Unified Dynamics. Experimental results show that, under the unified perspective framework, different intervention forms exhibit remarkably consistent dynamic patterns. As shown in Figure 2, localized weight updates, low-rank adaptation (LoRA), and vector-based interventions display highly similar overall curve shapes. Additional results are included in Appendix A.

For preference log-odds, all methods typically follow a three-stage pattern when plotted against the steering factor m: for small |m|, they enter a Linear Region, where log-odds grows approximately linearly with m (Bigelow et al., 2025); this is followed by a Transitional Region with a noticeable change in trend, and finally a Convergence Region where the curve flattens and stabilizes.

Utility log-odds, in contrast, generally peak near

[Figure 1]

- Figure 2: Unified preference and utility dynamics under steering. Solid lines represent preference log-odds, and dashed lines represent utility log-odds. The top panel shows steering with vector-form parameter modifications, and the bottom panel shows parametric interventions including LoRA and local weight updates. Results are shown for the Gemma-2-9B-IT model on the AxBench dataset, evaluated over its top 10 concept subsets. The horizontal axis corresponds to the steering factor.

m ≈ 0, and remain near their maximum within this narrow range. As |m| increases, utility gradually declines and eventually stabilizes.

These patterns reveal a unified steering response of preference and utility.

### 4 Capability Dynamics: Mechanism Analysis and Optimization

Motivated by the unified preference–utility dynamics observed across intervention forms (Figure 2), this section provides a mechanistic account and an empirical characterization. We take an activation-manifold perspective and introduce a simple validity-decay factor to capture the tendency for capability to degrade as steering pushes activations away from the activation manifold, without committing to a specific underlying geometry. On this basis, we express preference as the combined effect of (i) steering-induced preference projection

changes and (ii) validity decay, while utility is modeled as being dominated by the validity decay term. Finally, under this hypothesis we formalize how the steering factor m shapes both preference and utility log-odds, and show via curve-fitting that the resulting forms match the observed log-odds–m dynamics well across settings.

#### 4.1 Activation Manifold Hypothesis

Prior work suggests that model activations often concentrate on low-dimensional, manifold-like sets in representation space (Bricken et al., 2023; Wollschläger et al., 2025). Adopting this manifold perspective, we analyze additive steering as a translation of hidden states along an approximately fixed direction in activation space. Intuitively, small translations may adjust model behavior in a targeted way, whereas large translations may push representations away from the high-density region learned during training, increasing the risk of a

[Figure 2]

- Figure 3: Mechanism of projection gain and validity decay. Right: An activation manifold view illustrating Assumption 4.1. An activation P lies on or near the manifold. Steering using preference vector v with scaling

factors m+ and m− moves P to P1 and P2, corresponding to intersections with the manifold. Top-left: Projection gain. Projections onto the utility axis exhibit limited variation, whereas projections along the preference direction differ between P1 and P2, suggesting that steering primarily influences preference-related components. Bottom-left: Steering-induced validity decay. As assumed in Assumption 4.2, increasing steering factor increases off-manifold deviation, leading to a monotonic decrease in validity and degraded downstream decoding.

representation–decoder mismatch and thus degrading general capability.

We formalize this view with two assumptions.

Assumption 4.1 (Training-Induced Activation Manifold). Fix a layer l with hidden dimension dl. There exists a low-dimensional set (or its neighborhood) Ml ⊂ Rdl such that for inputs x drawn from a set of stably handled inputs Xstable, the corresponding activation hl(x) lies on or near Ml with high probability:

##### [d(hl(x),Ml) ≤ ϵ] ≥ 1 − δ, (9)

##### Pr

x∼Xstable

where d(·,Ml) denotes distance to Ml, ϵ > 0 is a neighborhood radius, and δ ∈ (0,1) is a small failure probability.

Assumption 4.1 asserts that pre-training induces a “typical” region of activation space where representations concentrate for stably handled inputs. We next introduce a generic notion of representation validity, which is high near Ml and decreases as hidden states move away from it. This abstraction avoids committing to a specific geometry for Ml while retaining the key implication: sufficiently off-manifold activations are more likely to

be decoded unreliably by the remaining network.

Assumption 4.2 (Steering-Induced Validity Decay). Let Fl→L denote the remainder of the model from layer l to the output logits. There exists a validity function Vl : Rdl → [0,1] that is monotonically nonincreasing in d(h,Ml), capturing how well Fl→L can stably decode an activation h.

For an additive steering intervention at layer l,

##### h˜l(m) = hl + m∆h, (10)

with steering direction ∆h and steering factor m ∈ R, define the average validity over stably handled inputs:

##### D(m) ≜ Ex∼Xstable Vl h ˜l(m) . (11)

We assume that D(m) ∈ [0,1] decreases with |m| (i.e., larger interventions induce larger off-manifold shifts on average), and that the resulting capability degradation is dominated by this validity decay.

To connect Assumptions 4.1–4.2 to a concrete functional form, we view steering as moving an activation along a one-dimensional line in representation space, h˜l(m) = hl + m∆h. Under the

manifold hypothesis, degradation is governed primarily by how far this line trajectory departs from the typical region near Ml, so it is natural to model D(m) as a smooth function of the (signed) distance along this line to the nearest “on-manifold” locations. In particular, as illustrated in Fig. 3, the steered trajectory may intersect the manifold neighborhood at one or more values {mi} (e.g., one for m > 0 and one for m < 0). We therefore model validity as being highest near these intersection points and decaying as |m − mi| grows.

A convenient choice that is positive, smooth, and exhibits heavy-tailed distance-based decay is the rational quadratic (RQ) form, widely used in kernel methods and Gaussian processes to model multi-scale, polynomial-rate attenuation with distance (Rasmussen, 2004). Prior research on controllability metrics has established that model steerability is often asymmetric (Miehling et al., 2025), exhibiting varying degrees of responsiveness along different directions of the same dimension. Motivated by this observation, we employ a piecewise parameterized model to quantify degradation:

 

−p+

2 L+

1 + (m−m+)

if m ≥ 0 1 + (m−m−)

D(m) =

−p−



2 L−

if m < 0

(12) where m± corresponds to the signed distance from the original activation point P to an on-manifold intersection point P± along the steering line (Fig. 3); L± sets the characteristic scale of decay and reflects how fast the distance-to-manifold grows along the steering direction (larger when the direction is locally parallel to the manifold and smaller when it cuts across it); and p± controls the decay rate (tail heaviness) as the trajectory moves away from the manifold neighborhood.

- 4.2 Preference Capability: Projection Gain With Decay

We study how additive steering changes a model’s preference through intermediate activations. An intervention at layer l updates the hidden state as h˜(m) = h + m∆h.

Prior work under LRH-style assumptions often models preference probability with a logistic form, P(pp | h) = σ − (ωTp h + bp) , where ωp is the preference vector. Separately, work on activation geometry suggests that after low-dimensional projection (e.g., PCA), opposite preference labels are often approximately linearly separable. Un-

der the activation-manifold view, this motivates a two-dimensional preference plane and a preference direction whose signed coordinate reflects preference intensity. Our contribution is to incorporate validity attenuation D(·) (Assumption 4.2) to account for off-manifold steering.

To model this, we write the steered preference probability as

P(pp | h˜(m)) = σ − ωTp h+αpm Dp(m)−bp , (13)

where αp ≜ ωTp ∆h measures how much the steering direction aligns with the preference vector: α

is large when ∆h is aligned with ωp, and αp = 0 when ∆h is orthogonal to ωp.

This implies the preference log-odds

P(pp | h˜(m)) 1 − P(pp | h˜(m))

= ωTp h+αpm Dp(m)+bp.

log

(14)

Key implication (linear regime → nonlinear collapse). From Eq. (14), the mdependence enters as αpmDp(m). When |m−m±| ≪ L±, Eq. (12) gives Dp(m) ≈ 1, hence preference log-odds is approximately linear in m with slope α (matching the near-linear regime in Bigelow et al. (2025)). As |m − m±| grows and becomes comparable to or larger than L±, Eq. (12) implies substantial decay in Dp(m), so attenuation dominates and the log-odds response becomes strongly nonlinear and can collapse off-manifold.

Fitting Form. We fit the measured preference log-odds as a function of m with

P(pp | h˜(m)) 1 − P(pp | h˜(m))

log

= (αpm+βp)Dp(m) + bp,

(15)

where βp = ωTp h is a per-example constant (since h is fixed for a given input), and bp is an offset.

Fit Results. Table 2 reports the fit quality of Eq. (15), with R2 values exceeding 0.95 across most settings. These results validate the model’s ability to accurately characterize the dynamics of preference log-odds. Details are in Appendix E.

4.3 Utility Capability: Only Validity Decay

Utility Log-odds Under Manifold-Validity Decay. Let h ∈ Rdl denote the activation at layer

l. We quantify utility capability by the log-odds of positive vs. negative utility outcomes (up/un). Similar to preference, we assume utility is also associated with a direction ωu in activation space. Under steering h˜(m) = h + m∆h,, we model

P(u | h˜(m)) 1 − P(u | h˜(m))

= ωTuhDu(m) + bu, (16)

log

where Du(m) follows the manifold-validity decay in Eq. (12) and decreases with |m|. Crucially, for preference steering directions we typically have ωTu∆h ≈ 0, so utility is affected primarily through validity decay rather than a direct projection term. Fitting form. Accordingly, we fit the measured utility log-odds with a pure decay curve:

P(u | h˜(m)) 1 − P(u | h˜(m))

= βu Du(m) + bu, (17)

log

where βu is the baseline log-odds and bu is an offset capturing residual bias.

Fit Results. Table 2 reports the fit quality of Eq. (17). Uniformly high R2 values (typically > 0.97) suggest utility variations under preference steering are well captured by the proposed formulation. Additional details are in Appendix E.

### 5 Method

#### 5.1 Preference–Utility Joint Optimization

Building on the preceding mechanistic analysis, we propose Steering with Preference–UtiLity IntervenTion (SPLIT), a training objective improving preference while delaying utility degradation.

Utility Loss. To preserve utility, we train on both the positive and negative samples for the same input using the language-modeling cross-entropy:

Lutil = λp Lp + λn Ln, (18)

where Lp and Ln are the token-level cross-entropy losses on positive and negative samples, respectively, and λp,λn control their relative weight.

Preference Loss. By Eq. (7), the loss gap Ln − Lp is exactly the preference log-odds. We therefore maximize this gap via a hinge-style margin loss:

Lpref = γ · σ(θ − (Ln − Lp)), (19)

where σ(·) is ReLU and θ is a margin threshold, and γ trades off preference improvement against utility preservation.

Preference R2 ↑ Utility R2 ↑

Form Method

PSY PWR AXB Avg PSY PWR AXB Avg

Gemma-2-9B-IT Weight SFT 0.97 0.98 0.99 0.98 0.98 0.99 0.98 0.98

RePS 0.99 0.99 0.99 0.99 0.96 0.99 0.99 0.98 LoRA SFT 0.92 0.99 0.98 0.96 0.98 0.99 0.99 0.99

RePS 0.83 0.99 0.99 0.94 0.99 0.99 0.99 0.99

- Vector DiffMean 0.97 0.99 0.99 0.98 0.97 0.99 0.98 0.98 SFT 0.93 0.97 0.98 0.96 0.99 0.99 0.99 0.99

- RePS 0.99 0.98 0.95 0.97 0.99 0.99 0.99 0.99

Qwen-2.5-7B-IT Weight SFT 0.99 0.99 0.99 0.99 0.99 0.99 0.98 0.99

- RePS 0.99 0.99 0.99 0.99 0.99 0.99 0.97 0.98

LoRA SFT 0.97 0.99 0.99 0.98 0.99 0.99 0.99 0.99 RePS 0.94 0.99 0.99 0.97 0.98 0.96 0.99 0.98

- Vector DiffMean 0.98 0.95 0.98 0.97 0.99 0.99 0.97 0.98 SFT 0.93 0.97 0.98 0.96 0.98 0.99 0.99 0.99 RePS 0.97 0.98 0.93 0.96 0.99 0.98 0.99 0.99

Table 2: Curve fitting performance. Results on Psychopathy (PSY), PowerSeeking (PWR), and AxBench (AXB). We report R2 (higher is better), measuring alignment between theoretical curves and empirical data. Color intensity indicates R2 values. Consistently dark shading shows high fidelity across settings (R2 > 0.95).

Final Objective. We combine the two components as

L = Lutil + Lpref. (20) 5.2 Experiment Results.

We evaluate the proposed preference-utility joint optimization method under three intervention forms: local weight update, low-rank adaptation (LoRA), and activation vector steering. As shown in Table 3, our approach consistently achieves higher scores compared with baseline methods across all three intervention types. These results demonstrate the robustness and generality of the proposed optimization strategy. See Appendix A and B for more experimental details and results.

### 6 Related Work

Mechanism. Most activation steering methods assume linear structure in activation space, controlling concepts by adding scaled direction vectors to hidden states (Mikolov et al., 2013; Pennington et al., 2014; Nanda et al., 2023; Tigges et al., 2023; Park et al., 2024; Wang et al., 2024; Yao et al., 2025; Zhang et al., 2026; Hu et al., 2026). Building on this view, Bigelow et al. (2025) show that

Psychopathy PowerSeeking AxBench Acc(%, 0–100) ↑ Concept(0–4) ↑ Concept(0–2) ↑ Harmonic(0–2) ↑

Model Form Method

Vanilla Vanilla 50.00 1.87 0.4750 0.4950

SFT 100.00 3.50 1.6625 1.4538 REPS 100.00 3.39 1.7750 1.6362 SPLIT (Ours) 100.00 3.59 1.8500 1.6225

Local Weight

SFT 100.00 3.41 1.7625 1.5188 REPS 99.00 3.44 1.7375 1.6525 SPLIT (Ours) 100.00 3.56 1.7750 1.6412

Gemma-2-9B-IT

LoRA

DiffMean 53.00 2.95 1.1625 1.0550 SFT 97.00 3.30 1.7000 1.4487 REPS 98.00 3.61 1.7000 1.5550 SPLIT (Ours) 99.00 3.62 1.8500 1.6475

Vector

Vanilla Vanilla 50.00 2.24 0.4500 0.4713

SFT 97.00 3.53 1.5375 1.1287 REPS 96.00 3.24 1.6875 1.4163 SPLIT (Ours) 98.00 3.66 1.7000 1.4325

Local Weight

SFT 99.00 3.05 1.4875 1.3175 REPS 100.00 3.34 1.4875 1.4013 SPLIT (Ours) 100.00 3.59 1.7375 1.6362

Qwen-2.5-7B-IT

LoRA

DiffMean 55.00 3.17 0.9500 0.9950 SFT 97.00 3.58 1.5750 1.5800 REPS 88.00 3.63 1.7375 1.6412 SPLIT (Ours) 98.00 3.65 1.8125 1.6500

Vector

Table 3: Main task performance of steering methods evaluated on three datasets. Psychopathy is reported with classification accuracy (Acc, %), PowerSeeking is evaluated using LLM-judge preference scores on a 0–4 scale, and AxBench reports the concept score and the harmonic mean (HM) over concept, instruction, and fluency scores, each on a 0–2 scale, as evaluated by an LLM judge. All methods perform inference-time interventions on hidden representations. Best and second-best results are highlighted within each model and intervention form.

steering yields an approximately linear trend in posterior odds, but mainly in the small-scale regime. Recent studies further report non-monotonic or adverse effects under stronger steering, challenging a naive global linearity assumption (Bricken

- et al., 2023; Wollschläger et al., 2025). Meanwhile, representation-manifold work provides a complementary geometric lens for understanding steering and its limitations (Modell et al., 2025; Li and He, 2025; Xie et al., 2025).

Activation Steering. Activation steering controls the behavior of LLMs by intervening in hidden states during forward propagation, using steering vectors to control single attributes as well as more complex behavioral targets (Turner et al., 2023; Rimsky et al., 2024; van der Weij et al., 2024; Rahn

- et al., 2024; Scalena et al., 2024; Tan et al., 2024; Bhattacharjee et al., 2024; Postmus and Abreu, 2024; Konen et al., 2024; Hazra et al., 2024; Han
- et al., 2025; Jiang et al., 2025). However, recent studies have shown that the coarse-grained nature of activation steering can lead to degradation in model utility (Wang et al., 2025; Wu et al., 2025a). Cao et al. (2024); Wu et al. (2025b) introduce explicit preference learning objectives to optimize activation steering, enabling more precise control.

Parameter-Efficient Fine-Tuning. ParameterEfficient Fine-Tuning (PEFT) methods, including adapters and LoRA, show that effective adaptation of LLMs does not require updating all parameters. LoRA achieves performance comparable to full fine-tuning, indicating that adaptation relies on structured low-rank weight updates rather than full parameter changes (Hu et al., 2022a; Zi et al., 2023; Zhang et al., 2023; Hayou et al., 2024; Kopiczko et al., 2024; Zhang et al., 2024; Chen et al., 2024). Local weight updates further reveal that LLM knowledge is highly localized, as modifying a small subset of parameters in specific layers suffices to change factual associations (Geva et al., 2021; Zaken et al., 2022; Ding et al., 2022; Chen, 2024; Yang et al., 2025).

### 7 Conclusion

We propose a unified dynamic weight update framework that incorporates parameter updates, LoRA, and activation interventions, revealing a consistent preference–utility decay pattern in the log-odds space. Building on this mechanistic insight, we design a joint optimization method that consistently improves preference while mitigating utility degradation across diverse intervention forms, demonstrating versatility and robustness.

### Limitations

While our unified dynamic weight update framework provides a coherent perspective on LLM control and enables predictable preference–utility trade-offs, several limitations remain. First, our analysis assumes that model representations lie near a well-structured activation manifold, which may not hold for extremely large or highly diverse models, potentially reducing the accuracy of our quantitative predictions. Second, our experiments focus primarily on attribute-level control (e.g., sentiment, style), leaving the applicability to complex multi-turn reasoning or safety-critical content largely unexplored. Third, while our proposed training objective mitigates the utility–preference trade-off, it does not guarantee complete avoidance of undesirable side effects such as subtle instruction violations or context drift under extreme control strengths. Finally, our study evaluates control under pre-defined intervention multipliers, and generalization to adaptive or dynamically varying control signals requires further investigation.

### Ethics Statement

Controlled LLM generation carries inherent ethical considerations. While our framework aims to improve controllability and preserve task validity, it could potentially be misused to manipulate user perception, amplify biased viewpoints, or generate persuasive yet misleading content. Our experiments are conducted on standard benchmark datasets and do not involve sensitive personal information. We emphasize that the proposed methods should be deployed with human oversight, adherence to fairness guidelines, and robust monitoring to prevent harm. By explicitly modeling preference–utility trade-offs, we aim to make LLM interventions more interpretable and safer, but responsible usage depends on context-aware implementation and alignment with societal norms.

### Acknowledgements

We would like to express our sincere gratitude to the anonymous reviewers for their thoughtful and constructive feedback. This work was supported by the National Natural Science Foundation of China (No. 62576307, No. NSFCU23B2055, No. NSFCU19B2027), the Fundamental Research Funds for the Central Universities (226-2023-00138), the Yongjiang Talent Introduction Programme (2021A156-G), and the Information Technology Center

and State Key Lab of CAD&CG, Zhejiang University. This work was supported by Alibaba Group through Alibaba Innovative Research Program.

### References

Lukasz Bartoszcze, Sarthak Munshi, Bryan Sukidi, Jennifer Yen, Zejia Yang, David Williams-King, Linh Le, Kosi Asuzu, and Carsten Maple. 2025. Representation engineering for large-language models: Survey and research challenges. CoRR, abs/2502.17601.

Amrita Bhattacharjee, Shaona Ghosh, Traian Rebedea, and Christopher Parisien. 2024. Towards inferencetime category-wise safety steering for large language models. CoRR, abs/2410.01174.

Eric J. Bigelow, Daniel Wurgaft, YingQiao Wang, Noah D. Goodman, Tomer D. Ullman, Hidenori Tanaka, and Ekdeep Singh Lubana. 2025. Belief dynamics reveal the dual nature of in-context learning and activation steering. CoRR, abs/2511.00617.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread. Https://transformercircuits.pub/2023/monosemanticfeatures/index.html.

Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. 2024. Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Huajun Chen. 2024. Large knowledge model: Perspectives and challenges. DATA INTELLIGENCE, 6(3):587–620.

Songlin Chen, Weicheng Wang, Xiaoliang Chen, Peng Lu, Zaiyan Yang, and Yajun Du. 2024. Llama-lora neural prompt engineering: A deep tuning framework for automatically generating chinese text logical reasoning thinking chains. DATA INTELLIGENCE, 6(2):375–408.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong Sun. 2022. Delta tuning: A comprehensive study of

parameter efficient methods for pre-trained language models. CoRR, abs/2203.06904.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. 2023. Parameter-efficient fine-tuning of large-scale pretrained language models. Nature machine intelligence, 5(3):220–235.

Constanza Fierro and Fabien Roger. 2025. Steering language models with weight arithmetic. CoRR, abs/2511.05408.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are keyvalue memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 5484–5495. Association for Computational Linguistics.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. 2024. Word embeddings are steers for language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16410–16430.

Peixuan Han, Cheng Qian, Xiusi Chen, Yuji Zhang, Denghui Zhang, and Heng Ji. 2025. Internal activation as the polar star for steering unsafe llm behavior. arXiv preprint arXiv:2502.01042.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. 2024. Lora+: Efficient low rank adaptation of large models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Rima Hazra, Sayan Layek, Somnath Banerjee, and Soujanya Poria. 2024. Safety arithmetic: A framework for test-time safety alignment of language models by steering parameters and activations. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 21759–21776. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022a. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022b. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Yi Hu, Jiaqi Gu, Ruxin Wang, Zijun Yao, Hao Peng, Xiaobao Wu, Jianhui Chen, Muhan Zhang, and Liangming Pan. 2026. Towards a mechanistic understanding of large reasoning models: A survey

of training, inference, and failures. arXiv preprint arXiv:2601.19928.

Houcheng Jiang, Junfeng Fang, Ningyu Zhang, Guojun Ma, Mingyang Wan, Xiang Wang, Xiangnan He, and Tat-seng Chua. 2025. Anyedit: Edit any knowledge encoded in language models. arXiv preprint arXiv:2502.05628.

Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, and Tobias Hecking. 2024. Style vectors for steering generative large language models. In Findings of the Association for Computational Linguistics: EACL 2024, St. Julian’s, Malta, March 17-22, 2024, pages 782–802. Association for Computational Linguistics.

Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M. Asano. 2024. Vera: Vector-based random matrix adaptation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Tianhong Li and Kaiming He. 2025. Back to basics: Let denoising generative models denoise. arXiv preprint https://arxiv.org/pdf/2511.13720.

Yuren Mao, Yuhang Ge, Yijiang Fan, Wenyi Xu, Yu Mi, Zhonghao Hu, and Yunjun Gao. 2025. A survey on lora of large language models. Frontiers of Computer Science, 19(7):197605.

Samuel Marks and Max Tegmark. 2023. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. CoRR, abs/2310.06824.

Erik Miehling, Michael Desmond, Karthikeyan Natesan Ramamurthy, Elizabeth M. Daly, Kush R. Varshney, Eitan Farchi, Pierre Dognin, Jesus Rios, Djallel Bouneffouf, Miao Liu, and Prasanna Sattigeri. 2025. Evaluating the prompt steerability of large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7874–7900, Albuquerque, New Mexico. Association for Computational Linguistics.

Tomás Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013. Linguistic regularities in continuous space word representations. In Human Language Technologies: Conference of the North American Chapter of the Association of Computational Linguistics, Proceedings, June 9-14, 2013, Westin Peachtree Plaza Hotel, Atlanta, Georgia, USA, pages 746–751. The Association for Computational Linguistics.

Alexander Modell, Patrick Rubin-Delanchy, and Nick Whiteley. 2025. The origins of representation manifolds in large language models. CoRR, abs/2505.18235.

Neel Nanda, Andrew Lee, and Martin Wattenberg. 2023. Emergent linear representations in world models

of self-supervised sequence models. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@EMNLP 2023, Singapore, December 7, 2023, pages 16–30. Association for Computational Linguistics.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL, pages 1532–1543. ACL.

Joris Postmus and Steven Abreu. 2024. Steering large language models using conceptors: Improving addition-based activation engineering. CoRR, abs/2410.16314.

Nate Rahn, Pierluca D’Oro, and Marc G. Bellemare. 2024. Controlling large language model agents with entropic activation steering. CoRR, abs/2406.00244.

Carl Edward Rasmussen. 2004. Gaussian Processes in Machine Learning, pages 63–71. Springer Berlin Heidelberg, Berlin, Heidelberg.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 15504–15522. Association for Computational Linguistics.

Daniel Scalena, Gabriele Sarti, and Malvina Nissim. 2024. Multi-property steering of large language models with dynamic activation composition. CoRR, abs/2406.17563.

Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai, Ziyu Yao, Ninghao Liu, and Mengnan Du. 2025. A survey on sparse autoencoders: Interpreting the internal mechanisms of large language models. CoRR, abs/2503.05613.

Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige, Adrià Garriga-Alonso, and Robert Kirk. 2024. Analyzing the generalization and reliability of steering vectors. CoRR, abs/2407.12404.

Curt Tigges, Curt Tigges, Oskar Hollinsworth, Curt Tigges, Atticus Geiger, Atticus Geiger, Oskar Hollinsworth, Neel Nanda, Neel Nanda, Atticus Geiger, and Neel Nanda. 2023. Linear representations of sentiment in large language models. http://arxiv.org/abs/2310.15154.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Activation addition: Steering language models without optimization. arXiv eprints, pages arXiv–2308.

Teun van der Weij, Massimo Poesio, and Nandi Schoots.

2024. Extending activation steering to broad skills and multiple behaviours. CoRR, abs/2403.05767.

Mengru Wang, Ziwen Xu, Shengyu Mao, Shumin Deng, Zhaopeng Tu, Huajun Chen, and Ningyu Zhang. 2025. Beyond prompt engineering: Robust behavior control in llms via steering target atoms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 23381–23399. Association for Computational Linguistics.

Mengru Wang, Yunzhi Yao, Ziwen Xu, Shuofei Qiao, Shumin Deng, Peng Wang, Xiang Chen, Jia-Chen Gu, Yong Jiang, Pengjun Xie, et al. 2024. Knowledge mechanisms in large language models: A survey and perspective. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7097–7135.

Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, and Johannes Gasteiger. 2025. The geometry of refusal in large language models: Concept cones and representational independence. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D Manning, and Christopher Potts. 2025a. Axbench: Steering llms? even simple baselines outperform sparse autoencoders. arXiv preprint arXiv:2501.17148.

Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, and Christopher Potts. 2025b. Improved representation steering for language models. CoRR, abs/2505.20809.

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang. 2025. mhc: Manifoldconstrained hyper-connections. arXiv preprint https://arxiv.org/pdf/2512.24880.

Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming Xu, Mengru Wang, Xinle Deng, Yunzhi Yao, Guozhou Zheng, Huajun Chen, and Ningyu Zhang. 2025. Easyedit2: An easy-to-use steering framework for editing large language models. CoRR, abs/2504.15133.

Wanli Yang, Fei Sun, Rui Tang, Hongyu Zang, Du Su, Qi Cao, Jingang Wang, Huawei Shen, and Xueqi

Cheng. 2025. Fine-tuning done right in model editing. CoRR, abs/2509.22072.

Yunzhi Yao, Jiaxin Qin, Ningyu Zhang, Haoming Xu, Yuqi Zhu, Zeping Yu, Mengru Wang, Yuqi Tang, Jia-Chen Gu, Shumin Deng, et al. 2025. Rethinking knowledge editing in reasoning era. Authorea Preprints.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. 2022. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2022, Dublin, Ireland, May 2227, 2022, pages 1–9. Association for Computational Linguistics.

Hengyuan Zhang, Zhihao Zhang, Mingyang Wang, Zunhai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan, Ercong Nie, Xufeng Duan, Qibo Xue, et al. 2026. Locate, steer, and improve: A practical survey of actionable mechanistic interpretability in large language models. arXiv preprint arXiv:2601.14004.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Ruiyi Zhang, Rushi Qiang, Sai Ashish Somayajula, and Pengtao Xie. 2024. Autolora: Automatically tuning matrix ranks in low-rank adaptation based on meta learning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 5048– 5060. Association for Computational Linguistics.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2).

Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, and Lei Zhang. 2023. Delta-lora: Fine-tuning high-rank parameters with the delta of low-rank matrices. CoRR, abs/2309.02411.

### A Experiment Details

Datasets. We evaluate our dynamic intervention methods on three datasets: (i) Psychopathy (personality tendency classification), (ii) PowerSeeking (open-ended generation), and (iii) the top-10 concept subsets from AxBench (open-ended generation). To support training and evaluation under our paired-setting and data availability constraints, we construct task-specific train/test splits as follows.

For Psychopathy, we sample 500 instances for training and 100 for testing. For PowerSeeking, we sample 500 instances for training and 200 for testing. For AxBench, since its original test set is randomly sampled from an instruction-following corpus and does not provide matched positive/negative answer pairs, we re-split the original 72 instances per concept for each of the top-10 concept subsets into 64 training instances and 8 test instances.

Evaluation and Metrics. For the experiments in §3.3, following Bigelow et al. (2025), we compute preference and utility log-odds (Eqs. (7) and (8)) for each query q with matched answers (Ap,An) on both training and test sets, and vary the intervention scale m to track their changes. For the final performance evaluation in §5.2, we adopt dataset-specific metrics. For Psychopathy, following Bigelow et al. (2025), we report classification accuracy (Acc). For PowerSeeking, following Cao et al. (2024), we use gpt-4.1-mini to score generations on the test set on a 0–4 scale. For AxBench, following Wu et al. (2025a), we use gpt-4.1-mini to evaluate concept score, instruction score, and fluency score on the test set, each on a 0–2 scale; we report the concept score and the harmonic mean over the three scores.

Baselines. We evaluate multiple methods under three intervention forms: local weight updates, LoRA, and vector interventions. For each form, we train interventions with either the SFT objective or the RePS objective (Wu et al., 2025a). For vector interventions, we additionally include a train-free baseline DiffMean (Marks and Tegmark, 2023). We also report Vanilla results without any steering.

Intervention Setup. We run experiments on Gemma-2-9B-IT at layer 20 and on Qwen-2.5-7B-Instruct at layer 14, following Bigelow et al. (2025). We consider three intervention forms: local weight updates, LoRA, and vector interventions. For local weight and LoRA, we train intervention parameters on the MLP down-projection matrix; for vector interventions, we apply the intervention directly to the residual stream. For hyperparameters, we largely follow the default settings in Wu et al. (2025a); Xu et al. (2025). We optimize with AdamW and a linear learning-rate scheduler. We also perform reasonable hyperparameter tuning to ensure stable and competitive performance.

Results: Unified Dynamics Observation. Figures 2 and 4 show the unified preference–utility dynamics of the Gemma-2-9B-IT and Qwen-2.5-7B-IT models on the AxBench dataset, evaluated over the top-10 concept subsets. And figure 5 shows the unified preference-utility dynamics on Power-seeking and Psychopathy datasets under different models. We observe that the utility can increase under slight perturbations of m in either the positive or negative direction. In some cases, this suggests that the origin may not lie exactly on the utility manifold, implying that the utility is not always strictly optimal at m = 0.

Results: Performance Comparison. Table 3 compares our method with various baselines under different intervention forms (local weight, LoRA, and vector) on two base models. Across intervention forms, our method remains competitive with strong baselines, and often improves concept metrics while maintaining comparable or higher harmonic scores. The gains are most consistent under LoRA and vector, where our approach typically strengthens concept control relative to both SFT- and RePS-trained variants, and achieves the best or near-best harmonic mean on AXBENCH in multiple settings. Under full weight updates, we observe smaller but still stable differences, with our method remaining comparable and without an apparent drop in utility. Overall, the results indicate that the proposed optimization transfers across different steering forms and can provide reliable, albeit sometimes incremental, improvements.

### B Results of DPO-based Method

To provide a more exhaustive analysis of preference-based objectives in the context of behavior steering, we implemented a DPO-based method following BiPO (Cao et al., 2024).

Form Psy. Pow. AXB.

Acc Concept Concept Harmonic (%)↑ (0–4)↑ (0–2)↑ (0–2)↑

Local Weight 91.00 1.91 0.525 0.566 LoRA 99.00 1.87 0.550 0.564 Vector 99.00 1.93 0.575 0.631

Table 4: Performance of the DPO-based method on Gemma-2-9B-IT across different adaptation forms. The evaluation encompasses Psychopathy (Psy.), PowerSeeking (Pow.), and AxBench (AXB.) metrics.

As shown in Table 4, the DPO-based method performs consistently with prior observations re-

Symbol Description Unified Analysis Framework m The steering scalar coefficient. PrefOdds(q) Preference log-odds (Ln − Lp).

(7) UtilOdds(q) Utility log-odds. (8) P(u|q) Latent utility probability. P(p±|q) Latent preference probability. P(•|h) Equivalent to P(•|q) as the

weights remain unchanged.

Lp,Ln Cross-entropy losses corresponding to Ap and An. Mechanistic Manifold Model Ml The activation manifold of stably

handled inputs at layer l.

D(m) Average validity decay function. m± Distance from P to P± along the

steering line in Fig 3 L± Characteristic scale of decay. p± Asymptotic decay rate. Joint Optimization

Lutil Utility loss component. Lpref Preference loss component.

Table 5: Notations for Key Concepts. A summary of the specialized symbols introduced for the unified analysis, mechanistic modeling, and optimization objective.

ported in RePS (Wu et al., 2025b). In contrast, our method explicitly models the preference–utility trade-off, achieving a more balanced improvement across dimensions, indicating performance beyond a standard DPO-style approach.

### C List of Mathematical Symbols

The Table 5 below lists the important symbols used in this paper.

### D Derivations and Implementation Details for Log-Odds

This appendix derives the loss-based forms of Eqs. (7)–(8) from the preference–utility independence assumption, and states how we compute the required sequence losses.

[Figure 3]

- Figure 4: Unified preference and utility dynamics under steering. Solid lines represent preference log-odds, and dashed lines represent utility log-odds. The top panel shows steering with vector-form parameter modifications, and the bottom panel shows parametric interventions including LoRA and local weight updates. Results are shown for the Qwen-2.5-7B-IT model on the AxBench dataset, evaluated over its top 10 concept subsets. The horizontal axis corresponds to the steering factor.

- D.1 From preference–utility independence to log-odds

For a query q and a polarity pair (Ap,An), we assume

P(Ap | q) = P(u | q)P(pp | q), P(An | q) = P(u | q)P(pn | q), (21)

with P(pp | q) + P(pn | q) = 1.

Preference log-odds. Taking the ratio of (21) cancels P(u | q):

P(Ap | q) P(An | q)

=

P(pp | q) P(pn | q)

. (22)

Applying log(·) gives PrefOdds(q) ≜ log

P(pp | q) P(pn | q)

P(Ap | q) P(An | q)

= log

.

(23)

Using the loss definition L ≜ −log P(A | q), we have P(A | q) = e−L, and thus

e−Lp e−Ln = Ln − Lp, (24)

PrefOdds(q) = log

which matches Eq. (7). Utility probability and log-odds. Summing (21) and using P(pp | q) + P(pn | q) = 1 yields

P(Ap | q) + P(An | q) = P(u | q) P(pp | q) + P(pn | q)

= P(u | q). (25)

[Figure 4]

- (a) Powerseeking Results

[Figure 5]

- (b) Psychopathy Results

- Figure 5: Unified preference and utility dynamics under steering. Solid lines represent preference log-odds, and dashed lines represent utility log-odds. Figure (a) shows the unified preference and utility dynamics of the power-seeking dataset under two different models, while Figure (b) shows the results for the psychopathy dataset. The horizontal axis corresponds to the steering factor.

Therefore,

P(u | q) 1 − P(u | q)

UtilOdds(q) ≜ log

P(Ap | q) + P(An | q) 1 − P(Ap | q) − P(An | q)

= log

.

(26)

Substituting P(A | q) = e−L(A|q) gives the loss form

e−Lp + e−Ln 1 − e−Lp − e−Ln , (27)

UtilOdds(q) = log

which matches Eq. (8). Note that since (Ap,An) are only two candidate continuations, we typically

have P(Ap | q) + P(An | q) < 1.

#### D.2 Computing sequence losses

Let A = (y1,...,yT) be a completion (excluding the query/prompt tokens). We compute the sequence negative log-likelihood (cross-entropy loss) under teacher forcing:

L(A | q) ≜ −log P(A | q)

T

log P(yt | q,y<t). (28)

= −

t=1

We then set Lp ≜ L(Ap | q) and Ln ≜ L(An | q) and plug them into Eqs. (24) and (27).

Length normalization (optional). When Ap and An have different lengths, we optionally use the mean loss L¯(A | q) ≜ L(A | q)/T in place of L(A | q) to reduce length effects. In that case, the corresponding quantities use e−L¯ instead of e−L.

#### D.3 Preference log-odds and Utility log-odds

Here we show how the preference and utility capability can be represented as (14) and (16). We take preference log-odds as example. First, the conditional probability before steering is given by:

P(pp | h) = σ − ωTp hDp(0) − bp ,

= σ(η) (29)

where η ≜ −ωTp hDp(0) − bp.

When an intervention at layer l updates the hiden state as h˜(m) = h + m∆h, we can get the steered preference probablity as (13) :

P(pp | h˜(m)) = σ −(ω⊤p h + αpm)Dp(m) − bp

Next, we can represent the initial preference logodds as −η:

P(pp | h) P(pn | h)

P(pp | h) 1 − P(pn | h)

log

= log

σ(η) 1 − σ(η)

= log

1/(1 + eη) 1 − 1/(1 + eη)

= log

1/(1 + eη) eη/(1 + eη)

= log

= log e−η

= −η

= ωTp hDp(0) + bp (30)

Finally, when we steering h by h˜(m), we can get preference log-odds by:

P(pp | h˜(m)) P(pn | h˜(m))

= −ηsteered

log

= (ω⊤p h + αpm)Dp(m) + bp

(31) For utility capability, we have:

P(u | h˜(m)) = σ − ωTuh˜(m)Du(m) − bu .

(32)

For preference steering directions, we typically have ωTu∆h ≈ 0. So we can quantify utility capability by:

P(u | h˜(m)) 1 − P(u | h˜(m))

= ω⊤u hDu(m) + bu

log

### E Fitting Experiment Details

#### E.1 Fitting Results on Test Set

To further validate our theoretical model, we performed parameterized fitting on the test set using the SLSQP algorithm, strictly enforcing continuity between positive and negative segments at the origin. As shown in Table 6, the direct fitting yielded high goodness-of-fit values (R2 > 0.95) for most methods. This confirms that the steering effect follows a deterministic trajectory predicted by our theory rather than random perturbations, thereby validating the proposed interaction mechanism.

#### E.2 Analysis of Generalization Ability

Following the validation of our theoretical mechanism, we conducted train-to-test transfer experiments to evaluate the extent to which different methods decouple "concepts" from specific inputs. Theoretical curve parameters were derived solely from training data and applied directly to the test set for prediction (Table 7).

Robust Generalization Overall, the fitted curves generalize well to held-out data, with vector-based interventions achieving consistently strong R2 across most settings. Input-dependent approaches such as LoRA- and local-weight-based methods also generalize well in many cases, but exhibit larger variance across datasets and occasional failures, suggesting that input-dependent updates can be more sensitive to the evaluation distribution.

Preference R2 ↑ Utility R2 ↑

Type Method

PSY PWR AXB Avg PSY PWR AXB Avg

Gemma-2-9B-IT Weight SFT 0.96 0.96 0.99 0.97 0.98 0.93 0.99 0.97

RePS 0.95 0.98 0.95 0.96 0.98 0.93 0.99 0.97 LoRA SFT 0.99 0.99 0.98 0.99 0.98 0.98 0.99 0.98 RePS 0.99 0.99 0.98 0.99 0.99 0.99 0.99 0.99

Vector DiffMean 0.89 0.99 0.99 0.96 0.94 0.99 0.98 0.97 SFT 0.90 0.97 0.97 0.95 0.98 0.99 0.99 0.99 RePS 0.96 0.98 0.96 0.97 0.96 0.99 0.99 0.98

Qwen-2.5-7B-IT Weight SFT 0.99 0.82 0.99 0.93 0.99 0.99 0.95 0.98

RePS 0.99 0.89 0.97 0.95 0.98 0.99 0.90 0.96 LoRA SFT 0.70 0.95 0.98 0.88 0.99 0.99 0.99 0.99 RePS 0.88 0.95 0.95 0.93 0.98 0.99 0.98 0.98

Vector DiffMean 0.99 0.99 0.98 0.99 0.97 0.94 0.98 0.96 SFT 0.99 0.99 0.97 0.98 0.97 0.95 0.99 0.97 RePS 0.99 0.98 0.93 0.97 0.96 0.96 0.98 0.97

- Table 6: Performance comparison of curve fitting quality on test sets. We evaluate the models on three datasets: Psychopathy (PSY), PowerSeeking (PWR), and AXBench (AXB).

Type Method

Preference R2 ↑ Utility R2 ↑

PSY PWR AXB Avg PSY PWR AXB Avg

Gemma-2-9B-IT Weight SFT 0.96 0.85 -4.25 -0.81 0.98 0.98 0.61 0.86

RePS 0.99 0.98 -1.16 0.27 0.96 0.98 0.73 0.89 LoRA SFT 0.92 0.99 -0.56 0.45 0.98 0.99 0.96 0.98 RePS 0.83 0.99 0.74 0.85 0.98 0.99 0.97 0.98

Vector DiffMean -0.14 0.99 0.75 0.53 0.97 0.99 0.97 0.98 SFT 0.90 0.91 0.74 0.85 0.98 0.99 0.99 0.99 RePS 0.98 0.89 0.65 0.84 0.99 0.99 0.99 0.99

Qwen-2.5-7B-IT Weight SFT 0.99 -0.32 -12.03 -3.79 0.99 -1.33 -3.07 -1.14

RePS 0.96 0.98 -3.82 -0.63 0.99 0.42 -1.15 0.09 LoRA SFT 0.97 0.98 -0.40 0.52 0.99 0.99 0.95 0.98 RePS 0.94 0.99 -0.13 0.60 0.98 0.96 0.96 0.97

Vector DiffMean 0.86 0.94 0.80 0.87 0.37 0.99 0.97 0.78 SFT 0.67 0.92 0.71 0.77 0.96 0.99 0.99 0.98 RePS 0.97 0.93 0.74 0.88 0.99 0.98 0.98 0.98

- Table 7: Generalization ability of curve fitting. The table reports the R2 scores where the curves are fitted on the training set and evaluated on the test set across three datasets: Psychopathy (PSY), PowerSeeking (PWR), and AXBench (AXB). Negative values imply that the fitted curves do not generalize well to unseen data.

