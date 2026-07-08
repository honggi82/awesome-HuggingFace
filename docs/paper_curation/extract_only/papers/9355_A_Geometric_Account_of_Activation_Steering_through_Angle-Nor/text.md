## A Geometric Account of Activation Steering through Angle–Norm Decomposition

Georgii Aparin Huawei Noah’s Ark Lab aparingm@gmail.com

Tatiana Gaintseva Queen Mary University of London t.gaintseva@qmul.ac.uk

# arXiv:2606.06735v2[cs.AI]8Jun2026

### Abstract

Linear activation steering has gained popularity as a simple and empirically effective way to control language model behavior. More recently, spherical steering paradigms have been proposed to address limitations of additive interventions, often motivated by the assumption that hidden-state norm does not carry conceptrelevant information. In this work, we revisit this assumption through a controlled empirical study designed to disentangle the roles of angular and radial components. We show that steering methods differ mainly in how they couple two geometric effects: changing a token’s angular alignment with a concept direction and changing its hidden-state norm. Across seven language models, we find that concepts are represented primarily in angular structure, supporting the motivation for spherical methods, but that norm remains important for the stability and downstream effects of steering. Our results explain why interventions with similar conceptlevel effects can behave differently, and suggest that activation steering should be parameterized by interpretable angular and radial components of the intervention, rather than by a single additive coefficient that entangles these two effects.

### 1 Introduction

Linear activation steering has become a widely used approach for controlling language model behavior through interventions on intermediate representations (Zou et al., 2023; Turner et al., 2023; Panickssery et al., 2023). Given a steering direction associated with a target concept, standard methods add this direction to hidden states with a scalar strength. These interventions are simple, trainingfree, and effective across behaviors such as truthfulness, sentiment, toxicity, and refusal (Zou et al., 2023; Turner et al., 2023; Panickssery et al., 2023; Li et al., 2023; Rimsky et al., 2024; Arditi et al., 2024). However, additive steering treats activation space as if concept control were naturally linear:

Accuracy change

Perplexity ratio

0.1

- =0.9

- =1.0 (S)

| | |
|---|---|
| | |

PPLratio(log)

- 100

- 101

- 102

taskmetric

- =1.1

- =1.2

0.0

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Concept score ( )

Concept score ( )

- Figure 1: Effect of norm scaling in SN. The left panel shows downstream task metric change, and the right panel shows perplexity ratio. Increasing β has little effect on the semantic task metric but substantially reduces perplexity at high γ, indicating that the norm primarily controls generation stability.

0.1 0.3 0.5 0.7

Concept score ( )

0.0

0.5

1.0

Fractionoffolds

Highest task metric

- =0.9

- =1.0 (S)

- =1.1

- =1.2

0.1 0.3 0.5 0.7

Concept score ( )

0.0

0.5

1.0

Lowest PPL ratio

- Figure 2: Fraction of folds in which each β value achieves the best perplexity or task metric. At γ = 0.7, β = 1.2 achieves the lowest perplexity in all folds in our evaluation, indicating that strict norm preservation is not always the most stable choice for high-strength spherical steering.

increasing the steering coefficient is assumed to move representations in a meaningful behavioral direction. This obscures the geometry of the intervention, since adding a vector changes both the direction and the norm of the hidden state (Park et al., 2024; Vu and Nguyen, 2025; You et al., 2026).

Recent angular and spherical steering methods offer an alternative: instead of translating activations, they rotate hidden states toward a concept direction, often while preserving norm (Vu and Nguyen, 2025; You et al., 2026). This is motivated by the hypothesis that concept information is primarily angular, while norm preservation maintains generation quality and input relevance. Although spherical methods can improve stability over naive

additive steering, their underlying assumptions remain insufficiently examined: do concepts mainly live in activation direction, and is strict norm preservation always the right steering constraint?

We study these questions through a controlled geometric comparison of activation steering methods. We decompose each hidden state into an angular component, which determines alignment with a concept direction, and a radial component, given by its norm. This lets us compare six steering approaches that differ in whether they enforce a target angular concept score, preserve the original norm, or allow the norm to change.

Our experiments show that the angular hypothesis is largely correct. Across seven language models and four concept datasets, probes trained on normalized hidden states closely match probes trained on raw hidden states, while norm-only probes remain near chance. Thus, for the concepts we study, concept-discriminative information is encoded primarily in activation direction rather than magnitude.

However, norm is not irrelevant. Although activation magnitude does not directly encode the target concept, it plays an important role in generation stability and capability preservation. At high angular strength, strict norm preservation can cause large perplexity increases and capability degradation. Conversely, methods that reach the same angular target while allowing a modest norm increase often better preserve fluency and downstream performance (Figs. 1, 2). This yields a more nuanced conclusion than either additive or spherical steering alone suggests: angular control explains semantic steering, but radial scaling can determine whether the intervention remains usable at high strength.

We hypothesize that hidden-state norm partly controls the effective representational capacity available at a token. Under strong steering, forcing a target concept into the original fixed radius may leave less capacity for other context-relevant information. A modest norm increase can relieve this pressure, allowing the model to express the desired concept direction while retaining enough representational scale for other features.

Overall, our findings suggest that activation steering should be viewed neither as a one-parameter additive intervention nor as a purely angular operation with fixed norm. Instead, steering is better understood as a two-parameter geometric intervention governed by both angle and radius: angle controls the intended semantic effect, while radius

influences generation stability, input relevance, and capability preservation. This perspective explains why methods with similar concept-level effects can behave differently and suggests a more interpretable design space for future steering methods.

Our contributions are as follows:

- • We formulate activation steering as a twocomponent geometric intervention that separates angular concept control from radial norm modification.
- • We compare six steering methods under a common framework, distinguishing whether they preserve norm and whether they enforce a per-token target concept score.
- • We empirically test the angular encoding hypothesis across seven language models and four concept datasets, finding that concept information is primarily encoded in activation direction.
- • We show that norm still plays a crucial role in steering stability: at high steering strengths, modest norm increases can reduce perplexity by up to 1.8× without substantially changing the semantic steering effect.

### 2 Related Work

Activation steering and representation engineering. Activation steering controls model behavior by modifying intermediate activations at inference time, without updating weights. Most methods identify a direction in hidden-state space associated with a target behavior and intervene along this direction during generation. ITI, ActAdd, CAA, and Representation Engineering have been used to affect truthfulness, sentiment, topic, refusal, toxicity, and other high-level attributes (Li et al., 2023; Turner et al., 2023; Panickssery et al., 2023; Zou et al., 2023; Rimsky et al., 2024; Arditi et al., 2024). These methods are simple and training-free, but their usual additive strength has unclear geometry: changing it alters both the hidden state’s alignment with the steering direction and its norm. Our work studies this ambiguity directly by decomposing steering into angular and radial components.

Linear concept representations. Activation steering is closely related to the hypothesis that highlevel model properties are represented linearly in activation space. Under this view, directions in

hidden-state space correspond to concepts or behaviors, and projections onto these directions can serve as concept scores (Park et al., 2024; Zou et al., 2023). This motivates contrastive direction extraction, probing, and direction-based interventions. However, identifying a useful concept direction does not determine how to intervene along it. Additive steering simultaneously changes angular alignment and representation norm, which may play different roles. We therefore separate two often conflated questions: whether concept information is encoded in activation direction, and how norm changes affect steering outcomes.

Angular and spherical steering. Recent work has proposed angular or spherical alternatives to additive steering. Angular Steering rotates activations in a behavior-related subspace (Vu and Nguyen, 2025), while Spherical Steering performs a norm-preserving geodesic rotation toward a target direction (You et al., 2026). These methods are motivated by the idea that concept information is primarily angular and that preserving activation norm helps maintain generation quality. Our work provides a controlled examination of these assumptions: we test whether concepts are indeed primarily encoded in direction, and whether strict norm preservation remains desirable once angular control is fixed. Unlike prior work focused on specific steering rules, we analyze additive, renormalized, matched, angular, spherical, and norm-scaled interventions in a single angular–radial framework.

Adaptive and token-wise steering. Several methods suggest that a single global steering coefficient is insufficient. ITI selects intervention sites at the attention-head level (Li et al., 2023); Representation Engineering studies control directions across layers and behaviors (Zou et al., 2023); and selective or adaptive steering methods vary interventions across layers, tokens, or examples to reduce side effects (Dang and Ngo, 2026). Our results clarify what such adaptation should control: the achieved angular concept score and the radial norm scale. This yields a more interpretable design space in which methods differ not only in average steering strength, but also in per-token angular precision and norm handling.

Our contribution. Prior work shows that activation directions can control behavior, and recent spherical methods show that norm-aware interventions can improve stability. We ask which geometric component is responsible for concept control,

and which affects stability. Our experiments indicate that the evaluated concepts are primarily encoded in activation direction, supporting the motivation for spherical steering. At the same time, norm is not merely nuisance variation: modest radial changes can substantially affect perplexity and capability preservation even when angular concept score is fixed. Thus, we reframe activation steering as a two-parameter intervention over angle and radius, rather than a one-dimensional choice of additive strength or a binary choice between additive and norm-preserving methods.

### 3 Methodology

We study activation steering as a geometric intervention on the residual-stream hidden state of a language model. Given a hidden state x ∈ Rd at a fixed transformer layer and a unit steering direction s, we decompose x into a radial component and an angular component:

x r

, (1)

r = ∥x∥, u =

u − cs ∥u − cs∥

. (2)

c = ⟨u,s⟩, v =

Here, r is the hidden-state norm, u is the corresponding unit vector, c is the angular concept score, and v is the unit residual direction orthogonal to s. Any unit vector in the two-dimensional subspace span(s,v) can be written as

γs + 1 − γ2 v, (3)

where γ ∈ [−1,1] is the target concept score. This decomposition lets us separate two aspects of steering that are entangled in standard additive interventions: the angular movement toward the concept direction and the change in hidden-state magnitude.

#### 3.1 Steering direction construction

For each model and dataset, we construct a concept direction using contrastive mean-difference. We sample N = 256 positive-negative completion pairs from a held-out direction split and extract residual-stream activations at the last prompt token. The steering direction is the unit-normalized difference between the mean positive activation and the mean negative activation:

µ+ − µ− ∥µ+ − µ−∥

. (4)

s =

The same direction s is used for all steering methods within a model-dataset-fold cell, ensuring that

Method Norm preserved Tokenwise γ CAA − − CAA-r + − CAA-m − + S + + AS + − SN − +

Table 1: Summary of steering methods by whether they preserve the original hidden-state norm and whether they enforce a fixed per-token concept score.

comparisons isolate the geometry of the intervention rather than differences in direction estimation.

#### 3.2 Steering methods

We compare six steering operations that differ in whether they preserve the original norm and whether they target the concept score independently for each token. Table 1 summarizes the geometric constraints imposed by each method. Below, we describe each of the methods in detail.

Concept Activation Addition (CAA). The standard additive baseline applies a fixed global perturbation:

y = x + αs. (5)

α is usually treated as a hyperparameter. CAA is neither norm-preserving nor per-token calibrated: it applies the same fixed addition during all generation steps. The achieved concept score varies across tokens depending on the initial norm and alignment of x.

Renormalized CAA (CAA-r). CAA-r applies the same fixed additive update and then projects the result back to the original norm:

x + αs ∥x + αs∥

. (6)

y = r

This isolates the effect of post-hoc norm preservation while retaining the fixed-strength nature of CAA. CAA-r preserves ∥y∥ = ∥x∥, but it does not enforce a target concept score for each token.

Matched CAA without renormalization (CAAm). CAA-m chooses a token-specific additive coefficient α so that the normalized output reaches a desired concept score γ:

y ∥y∥

,s = γ. (7)

y = x + αs,

We compute α using the formula derived in Appendix C. Unlike CAA-r, CAA-m does not renormalize the output. Thus, it exactly controls the

angular concept score while allowing the norm to change.

CAA-m and Spherical Steering can be compared in a shared geometric subspace because both operate inside span(s,v). Once CAA-m chooses α such that the normalized output has concept score γ, its direction lies on the same ray as the spherical target

γs + 1 − γ2 v. (8)

Therefore, CAA-m and Spherical Steering have the same angular component and differ only in their radial component. If the matched CAA output is additionally renormalized to the original norm, the resulting method, CAA-mr, is exactly equivalent to Spherical Steering. We therefore do not treat CAA-mr as a separate method.

Spherical Steering (S). Spherical Steering directly constructs the minimum-geodesic-distance unit direction with target score γ, then restores the original norm:

y = r γs + 1 − γ2 v . (9)

This method preserves ∥y∥ = ∥x∥ exactly and enforces ⟨y/∥y∥,s⟩ = γ independently for every token.

Additive Spherical (AS). Additive Spherical applies a fixed spherical displacement toward the concept direction. Let

θ = arccos(c), θ′ = max(θ − ∆θ,0). (10) The steered state is

y = r cosθ′ s + sinθ′ v . (11)

AS preserves the norm and the residual direction, but it does not target the same final concept score for every token. Instead, the resulting score depends on the token’s initial angle to s.

Spherical Steering with Norm Scaling (SN). Finally, we introduce an explicit radial parameter β on top of Spherical Steering:

y = βr γs + 1 − γ2 v . (12)

When β = 1, SN reduces exactly to S. For β ̸= 1, the angular component is unchanged while the norm is scaled by a fixed multiplicative factor. This lets us test whether the norm acts primarily as a stability parameter once semantic angular control is fixed.

Our experimental design isolates the roles of angular control and norm modification through four controlled experiments.

- 1. Hidden-state norm variation. First, we measure hidden-state norm variation across layers and token populations. For each model, we sample examples from multiple corpora and compute the coefficient of variation of ∥x∥ for the last prompt token, all prompt tokens, and generated tokens. This experiment characterizes the radial geometry of the representation space and determines whether norm preservation is a meaningful constraint.
- 2. Angular versus radial concept encoding. Second, we test whether concept information is encoded primarily in direction or magnitude. We train three linear probes: one on raw hidden states h, one on normalized hidden states h/∥h∥, and one on the scalar norm ∥h∥. If normalized probes match raw probes while norm-only probes remain near chance, this indicates that concept information is primarily angular rather than radial.
- 3. Steering at matched angular control. Third, we compare steering methods under matched angular control. For per-token methods, we set a target concept score γ. For fixed-strength methods, we calibrate the global strength parameter by binary search so that the mean achieved concept score on evaluation activations matches the desired target γ¯. We then compare downstream task performance, per-token concept-score variance, norm ratio ∥y∥/∥x∥, perplexity, and general capability metrics. This experiment distinguishes three possible explanations for steering behavior: per-token precision, angular displacement, and norm preservation.
- 4. Isolating the role of norm scaling. Fourth, we isolate the role of the norm using SN. Holding γ fixed, we vary only the multiplicative norm scale β. Because β changes only the radius and leaves the angular concept score fixed, this experiment directly tests whether modest norm changes improve generation stability without changing semantic control.

All steering directions are computed on held-out direction splits and evaluated on separate held-out examples. Where methods require calibration, we perform binary search over the steering parameter before measuring downstream behavior. For CAA and CAA-r, the searched parameter is α; for AS, it is ∆θ; for CAA-m, it is the token-specific α needed to achieve γ.

Together, these experiments provide a controlled comparison between interventions that alter direc-

tion, norm, or both. By matching either per-token concept score or mean concept score across methods, we can determine whether steering success is explained by angular movement alone, strict norm preservation, or a two-parameter interaction between angle and radius.

### 4 Experiments

#### 4.1 Evaluation setup

Models and steering layer. We evaluate all methods on seven transformer language models spanning 1B to 70B parameters: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Llama-3.18B, Llama-3.2-1B-Instruct, Qwen2.5-3B-Instruct, and Llama-3.1-70B-Instruct. For each model, steering is applied to the residual-stream output at 75% depth. This gives steering layers 24, 21, 31, 24, 12, 27, and 60 respectively. We use a single forward hook at this layer, replacing each hidden state x with a steered state y at every token position during generation.

Datasets and task metrics. We evaluate steering on four concept datasets: TruthfulQA for truthfulness, SST-2 for sentiment, CivilComments for toxicity, and IMDB for sentiment. For TruthfulQA, we use closed-form multiple-choice metrics, primarily MC1. For SST-2 and IMDB, we measure the positive rate of generated continuations. For CivilComments, we measure non-toxicity using a toxicity classifier. For generation-based evaluations, we sample 128 tokens using nucleus sampling with p = 0.95 and temperature T = 0.7. Dataset and benchmark details are provided in Appendix I.

Quality and capability metrics. To measure whether steering damages general language-model behavior, we compute perplexity on 200 held-out WikiText-103 passages with maximum length 512. We report perplexity as a ratio relative to the unsteered baseline for the same model, dataset, and fold. We also evaluate MMLU accuracy using logprobability ranking on a fixed subset of 300 items, providing an auxiliary measure of retained model capability.

Calibration protocol. For per-token methods, we sweep target concept scores γ ∈ {0.1,0.3,0.5,0.7}. For fixed-strength methods, we calibrate the global steering parameter so that the mean achieved concept score on evaluation activations matches the desired target γ¯. Specifically, we binary-search α for CAA and CAA-r,

and ∆θ for AS. For SN, we hold the angular target fixed and sweep β ∈ {0.9,1.0,1.1,1.2}. All reported comparisons use held-out direction splits and held-out evaluation examples. Unless otherwise stated, results are aggregated over seven models, four datasets, one seed, and two folds.

#### 4.2 Experimental results

Hidden-state norms vary across layers and architectures. We first examine whether activation norms can be treated as approximately constant during steering. Figure 3 reports the coefficient of variation of last-prompt-token hidden-state norms across layers, models, and corpora. The results show that norm concentration is architecturedependent: Llama and Qwen models generally have relatively concentrated norms at middle and later layers, while Gemma exhibits much larger norm variation across most layers. We hypothesize that this difference is largely due to Gemma’s postnorm architecture. Across models, the activations after the last transformer block consistently have the lowest coefficient of variation, indicating that norm concentration increases toward the final layers. This indicates that the radial component is not a universally negligible part of the representation space. Additional layer-wise norm statistics are reported in Appendix A.

Importantly, norm variation by itself does not determine whether a concept is encoded in the norm or in the direction. Rather, this experiment motivates treating the norm as a separate geometric degree of freedom: even when semantic information is primarily angular, preserving or modifying the radius may still affect generation stability.

Concept information is primarily angular. We next test whether concept-discriminative information is encoded in the direction or the magnitude of hidden states. As shown in Figure 4, for each model and dataset, we train linear probes on three representations: raw hidden states h, normalized hidden states h/∥h∥, and scalar norms ∥h∥. Across all models and concept datasets, normalized probes closely match raw probes, while norm-only probes remain near chance. This supports the central geometric assumption that concept information is primarily represented in angular directions rather than in hidden-state magnitudes. Additional directionalencoding results are provided in Appendix B.

Matched additive steering and spherical steering share the same angular target but differ

Llama-3.1-8B-Instruct

25

20

CV(%)

15

10

5

0

0 5 10 15 20 25 30

Layer

Gemma-2-9B-IT

100

80

CV(%)

60

40

20

0

0 10 20 30 40

Layer

Llama-3.2-1B-Instruct

20

15

CV(%)

10

5

0

0 2 4 6 8 10 12 14 16

Layer

Llama-3.1-70B-Instruct

40

30

CV(%)

20

10

0

0 10 20 30 40 50 60 70 80

Layer

Qwen2.5-7B-Instruct

OpenWebText

NatQuestions

Alpaca

CivilComments

30

arXiv

CNN/DailyMail

WritingPrompts

PubMedQA

TruthfulQA

CodeSearchNet

CV(%)

20

10

0

0 5 10 15 20 25

Layer

Llama-3.1-8B (base)

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

25

20

CV(%)

15

10

5

0

0 5 10 15 20 25 30

Layer

Qwen2.5-3B-Instruct

60

50

40

CV(%)

30

20

10

0

0 5 10 15 20 25 30 35

Layer

60

MeanCVacrosscorpora(%)

50

40

30

20

10

0

0 10 20 30 40 50 60 70 80

Layer

Figure 3: T1: CV of hidden-state norms vs. layer for all 7 models, 10 corpora. Grey dotted = L75 steering layer. Bottom right: combined mean CV across corpora.

in norm. We next compare CAA-m and S at matched per-token target γ. Both methods steer inside the same two-dimensional subspace span(s,v) and reach the same normalized concept direction. Their difference is radial: S restores the original norm, while CAA-m leaves the additive norm change intact. This comparison therefore isolates the effect of norm change while holding angular control fixed. Further comparisons are provided in Appendix D.

Figure 5 shows that CAA-m produces only mild norm inflation at low and moderate steering strengths, but the effect grows at high γ.

Despite matching the same angular target, S and CAA-m differ strongly in generation stability. At high γ, strict norm preservation can produce large perplexity penalties and substantial capability loss, whereas CAA-m often retains lower perplexity and higher MMLU accuracy. This shows that preserving the original norm is not always the most stable choice once the angular edit becomes large.

Norm preservation alone does not explain stability in fixed-strength steering. We next isolate the fixed-strength family: CAA, CAA-r, and AS. Unlike S and CAA-m, these methods do not enforce the same concept score independently for each token. Instead, each method uses a single

T2: Linear probe accuracy vs. layer (all datasets)

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

100

100

Probeaccuracy(%)

Probeaccuracy(%)

90

90

80

80

70

70

TruthfulQA

raw unit norm-only

SST-2

60

60

CivilComments

50

50

IMDB

0 5 10 15 20 25 30

0 5 10 15 20 25

Layer

Layer

Gemma-2-9B-IT

Llama-3.1-8B (base)

100

100

Probeaccuracy(%)

Probeaccuracy(%)

90

90

80

80

70

70

60

60

50

50

0 10 20 30 40

0 5 10 15 20 25 30

Layer

Layer

Llama-3.2-1B-Instruct

Qwen2.5-3B-Instruct

100

100

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

Probeaccuracy(%)

Probeaccuracy(%)

90

90

80

80

70

70

60

60

50

50

0 2 4 6 8 10 12 14

0 5 10 15 20 25 30 35

Layer

Layer

Llama-3.1-70B-Instruct

100

Probeaccuracy(%)

90

80

70

60

50

0 10 20 30 40 50 60 70 80

Layer

- Figure 4: Linear probe accuracy versus layer for all four concept datasets. Each dataset contains three probe variants: raw hidden states h, normalized hidden states h/∥h∥, and norm-only features ∥h∥. Raw and normalized curves nearly overlap, while norm-only probes remain close to chance, indicating that the evaluated concepts are encoded primarily in direction.

0.1 0.3 0.5 0.7

1.0

1.1

1.2

1.3

1.4

/yxCM

TQA

0.1 0.3 0.5 0.7

SST-2

0.1 0.3 0.5 0.7

Civil

0.1 0.3 0.5 0.7

IMDB

Llama-8B Qwen-7B Gemma-9B

Llama-8B-base

Llama-1B Qwen-3B Llama-70B

CAA-m norm ratio vs

- Figure 5: Norm ratio ∥y∥/∥x∥ for CAA-m at matched per-token target γ.

global steering parameter, calibrated so that the mean achieved concept score matches the target level. This comparison tests whether preserving the hidden-state norm is sufficient to explain downstream stability. Additional results are provided in Appendix E.

The first comparison is between CAA and CAAr. These methods have the same normalized output direction after the additive update; CAA-r only rescales the resulting vector back to the original norm. As a result, their downstream behavior is very similar across steering strengths. This shows that post-hoc renormalization is not, by itself, a reliable source of improved stability.

The second comparison is between CAA-r and AS. Both methods preserve the hidden-state norm, but they produce different token-level angular profiles. CAA-r applies a fixed additive perturbation before renormalization, so the resulting angular displacement depends on the token’s initial norm

Downstream metric under S vs CAA-m steering

TQA

###### SST-2

Civil

IMDB

100

100

positiverate(%)

positiverate(%)

non-toxicity(%)

80

MC1(%)

40

90

50

60

80

30

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Perplexity under S vs CAA-m steering

Perplexity(WikiText,logscale)

###### TQA

###### SST-2

Civil

IMDB

- 101
- 102
- 103

- 101

- 102

- 103

- 101
- 102
- 103
- 104

|| |
|---|
<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

|| |
|---|
<br><br>| | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

S

- 101
- 102
- 103
- 104

CAA-m

Llama-8B Qwen-7B Gemma-9B

| |
|---|

Llama-8B-base

| |
|---|

Llama-1B Qwen-3B

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

MMLU accuracy under S vs CAA-m steering

###### TQA

###### SST-2

Civil

IMDB

MMLUaccuracy(%)

75

50

| |
|---|

25

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Figure 6: Downstream task metric, WikiText-103 perplexity, and MMLU accuracy under S and CAA-m at matched per-token γ. The two methods implement nearly identical angular control, but they differ in radial behavior. At high steering strengths, S incurs much larger perplexity penalties, while CAA-m better preserves generation stability and general capability.

and alignment with the steering direction. AS applies a fixed spherical displacement, so its effect is distributed differently across tokens. The fact that these two norm-preserving methods behave differently shows that norm preservation alone cannot explain the steering quality trade-off. Instead, the per-token distribution of achieved concept scores is an important part of the geometry.

The Pareto frontier depends on both angular precision and radial behavior. We then compare all five main methods: CAA, CAA-r, CAA-m, S and AS. Fixed-strength methods are calibrated to matched mean concept score, while per-token methods directly enforce the target score for each token. We plot downstream task improvement against WikiText-103 perplexity ratio, so better methods move toward higher task improvement and lower perplexity. Figure 7 shows the Pareto comparison separately for each dataset.

As shown in Figure 7, these results suggest that steering should not be reduced to a binary choice between preserving and changing the norm. CAAm and S have the same angular target, yet CAA-m is much more stable at high γ. Conversely, CAAr and AS both preserve norm, but AS produces much higher perplexity at large steering strengths. Particular, S achieves the highest downstream task score even at γ = 0.5, showing that strict pertoken angular targeting can be highly effective for semantic control. The relevant design space is

TruthfulQA

###### SST-2

CivilComments

IMDB

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

CAA

- 100
- 101
- 102

PPLratio(log)

- 100

- 101

- 102

CAA-r

- 100

- 101

- 100
- 101

CAA-m

S

| |
|---|

AS

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

0.08 0.06 0.04 0.02 0.00

0.10 0.05 0.00 0.05 0.10

0.00 0.01 0.02 0.03 0.04 0.05 0.06

0.00 0.05 0.10 0.15 0.20

task metric

task metric

task metric

task metric

Figure 7: Per-dataset Pareto curves for all methods. The same qualitative pattern appears across datasets: CAA-m provides a strong high-control, low-perplexity trade-off, while S suffers a large perplexity increase at high steering strengths.

therefore two-dimensional: angular control determines the semantic effect of steering, while norm scale strongly influences whether the model can continue generating coherently.

Norm scaling acts as a stability lever. Finally, we test this interpretation directly by adding an explicit multiplicative norm scale β on top of S. This gives SN:

y = βr γs + 1 − γ2v .

Changing β leaves the angular concept score fixed while changing only the radius of the steered representation. Thus, within this controlled intervention family, differences across β values reflect the effect of radial scaling under a fixed semantic direction.

Figure 1 shows that β has only a small effect on the task metric but a large effect on perplexity at high γ. Moving from β = 1.0 to β = 1.2 improves perplexity by roughly 1.8× at γ = 0.7, while task metrics remain within about 2.5 percentage points across the tested β values. We hypothesize that this effect arises because the hidden-state norm partly determines the representational capacity available to the model at that token. When steering strongly toward a concept direction while strictly preserving the original norm, a large fraction of the fixedradius representation may be devoted to expressing the steered concept, leaving less effective capacity for other information needed to maintain fluent and contextually coherent generation. A modest norm increase may compensate for this by allowing the target concept to be expressed without compressing the remaining information into the same radius.

Overall, these experiments support a twoparameter view of activation steering. The angular component controls the intended concept, as shown by the probe results and by the matched behavior of S and CAA-m in concept space. The radial component controls stability: preserving the

original norm is sometimes useful, but at high steering strengths a modest norm increase can substantially reduce perplexity without materially changing the semantic steering effect. Additional betasweep results are provided in Appendix H.

### 5 Conclusion

We presented a geometric account of activation steering that separates two effects entangled by additive interventions: angular movement toward a concept direction and radial change in hidden-state norm. This explains why a single additive coefficient is hard to interpret: the same coefficient can induce different angular shifts and norm changes depending on each token’s initial geometry.

Across seven language models and four concept datasets, we find that the evaluated concepts are represented primarily in activation direction. Normalized probes closely match raw probes, while norm-only probes remain near chance, supporting the view that semantic control is largely angular.

At the same time, norm preservation is not always the right constraint. Even with fixed angular concept score, radial changes can strongly affect perplexity and capability preservation. Strict norm preservation can become unstable at high steering strengths, while modest norm increases reduce degradation without materially changing the semantic effect.

Overall, our findings reframe activation steering as a two-parameter intervention over angle and radius. Angle controls the intended concept, while radius controls intervention stability. This explains why methods with similar concept-level effects can behave differently and suggests a more interpretable basis for future token-wise steering methods. Effective steering requires choosing not only where to point a representation, but also how much representational scale to give it.

### 6 Limitations

Our study has several limitations. First, we apply steering at a single fixed layer, chosen at 75% depth for each model. Although this gives a controlled comparison across methods, the optimal angle–norm trade-off may vary across layers.

Second, our experiments cover a limited set of models and concepts. We evaluate Llama, Qwen, and Gemma models on truthfulness, sentiment, and toxicity-related steering, but other architectures or more complex behaviors may exhibit different geometry.

Third, all methods use the same contrastive mean-difference steering direction. This isolates the effect of the intervention geometry, but does not test whether the conclusions hold for other ways of estimating steering directions.

Finally, our norm-scaling experiments use a small discrete set of β values. The results show that the norm is an important stability parameter, but they do not provide an automatic rule for choosing the best norm scale for a new model, layer, or task.

### References

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2019. Nuanced metrics for measuring unintended bias with real data for text classification. arXiv preprint arXiv:1903.04561.

Center for AI Safety and Hugging Face Datasets Contributors. 2024. Mmlu dataset card. https:// huggingface.co/datasets/cais/mmlu. Lists the dataset distribution under MIT.

Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. 2018. A discourse-aware attention model for abstractive summarization of long documents. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 615–621. Association for Computational Linguistics.

Quy-Anh Dang and Chris Ngo. 2026. Selective steering: Norm-preserving control through discriminative layer selection. Preprint, arXiv:2601.19375.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898. Association for Computational Linguistics.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

GitHub and CodeSearchNet Contributors. 2019. Codesearchnet repository. https://github.com/ github/CodeSearchNet. Code and documentation are MIT; source-code examples include per-file upstream licenses.

Aaron Gokaslan and Vanya Cohen. 2019. Openwebtext corpus download page. https://skylion007. github.io/OpenWebTextCorpus/. Dataset packaging released under CC0; underlying web text not owned by dataset authors.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. 2019. Openwebtext corpus. https: //skylion007.github.io/OpenWebTextCorpus/.

Google. 2026. Gemma terms of use. https://ai. google.dev/gemma/terms. Last modified: April 1, 2026.

Google and Hugging Face Datasets Contributors. 2024. Civil comments dataset card. https://huggingface.co/datasets/google/ civil_comments. Lists the dataset distribution under CC0-1.0.

Google Research. 2019. Natural questions download page. https://ai.google.com/research/ NaturalQuestions/download. Lists Natural Questions under the Creative Commons Share-Alike 3.0 license.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Karl Moritz Hermann, Tomáš Koˇciský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, volume 28.

- Hugging Face Datasets Contributors. 2024a. Cnn/dailymail dataset card. https://huggingface. co/datasets/abisee/cnn_dailymail. Lists the dataset distribution under Apache-2.0.
- Hugging Face Datasets Contributors. 2024b. Scientific papers dataset card. https://huggingface.co/ datasets/armanc/scientific_papers. Dataset obtained from arXiv and PubMed OpenAccess sources; license should be checked against the selected source distribution.
- Hugging Face Datasets Contributors. 2024c. Truthfulqa dataset card. https://huggingface.co/ datasets/domenicrosati/TruthfulQA. Lists the dataset distribution under Apache-2.0.
- Hugging Face Datasets Contributors. 2024d. Writingprompts dataset card. https://huggingface.co/ datasets/euclaise/writingprompts. Lists the dataset distribution under MIT.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. 2019a. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 2567–2577. Association for Computational Linguistics.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. 2019b. Pubmedqa repository. https://github.com/pubmedqa/pubmedqa. Repository released under MIT.

Kaggle Dataset Contributors. 2024. Stanford sentiment treebank v2 (sst2) dataset. https: //www.kaggle.com/datasets/atulanandjha/ stanford-sentiment-treebank-v2-sst2. Lists the dataset distribution under CC0.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. In Transactions of the Association for Computational Linguistics, volume 7, pages 453–466.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. Preprint, arXiv:2306.03341.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics

(Volume 1: Long Papers), pages 3214–3252. Association for Computational Linguistics.

Andrew L. Maas. 2011. Large movie review dataset. https://ai.stanford.edu/~amaas/ data/sentiment/. Original Stanford dataset page.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150. Association for Computational Linguistics.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843.

- Meta AI. 2024a. Llama 3.1 community license agreement. https://huggingface.co/meta-llama/ Llama-3.1-8B-Instruct/blob/main/LICENSE. Version release date: July 23, 2024.
- Meta AI. 2024b. Llama 3.2 community license agreement. https://huggingface.co/meta-llama/ Llama-3.2-1B-Instruct. Version release date: September 25, 2024.

Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2023. Steering Llama 2 via contrastive activation addition. Preprint, arXiv:2312.06681.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 39643–39666. PMLR.

- Qwen Team. 2024a. Qwen research license agreement. https://huggingface.co/Qwen/Qwen2. 5-3B-Instruct. Qwen2.5-3B-Instruct license.
- Qwen Team. 2024b. Qwen2.5 model release and licensing. https://qwenlm.github.io/blog/qwen2. 5/. Qwen2.5-7B-Instruct is released under Apache2.0.

Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Salesforce and Hugging Face Datasets Contributors. 2024. Wikitext dataset card. https: //huggingface.co/datasets/Salesforce/ wikitext. Lists WikiText under a Creative Commons Attribution-ShareAlike license.

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get to the point: Summarization with pointergenerator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073– 1083. Association for Computational Linguistics.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642. Association for Computational Linguistics.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023a. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023b. Stanford alpaca repository. https://github.com/ tatsu-lab/stanford_alpaca. Dataset released under CC BY-NC 4.0 for research / non-commercial use.

Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Activation addition: Steering language models without optimization. Preprint, arXiv:2308.10248.

Hieu M. Vu and Tan M. Nguyen. 2025. Angular steering: Behavior control via rotation in activation space. Preprint, arXiv:2510.26243.

Zejia You, Chunyuan Deng, and Hanjie Chen. 2026. Spherical steering: Geometry-aware activation rotation for language models. Preprint, arXiv:2602.08169.

Zenodo Dataset Contributors. 2023. Binary stanford sentiment treebank 2 (sst-2). https://zenodo.org/ records/7555310. Lists the dataset distribution under CC-BY-4.0.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation engineering: A top-down approach to AI transparency. Preprint, arXiv:2310.01405.

### Contents

- 1 Introduction 1
- 2 Related Work 2
- 3 Methodology 3

- 3.1 Steering direction construction . . 3
- 3.2 Steering methods . . . . . . . . . 4

- 4 Experiments 5

- 4.1 Evaluation setup . . . . . . . . . . 5
- 4.2 Experimental results . . . . . . . 6

- 5 Conclusion 8
- 6 Limitations 9

- A Additional Norm-Variation Analysis 12
- B Additional Directional-Encoding Results 13
- C CAA-m Per-Token Matching Algorithm 14
- D Additional Fixed-Angle Steering Results 15
- E Additional Fixed-Strength Steering Results 15
- F Concept-Score Closure 17
- G Off-Arc Perturbations 17
- H Additional Norm-Scaling Results 18
- I Datasets and Data Sources 19
- J Models and Licenses 20 Acknowledgments

This work was supported by a Google DeepMind PhD Studentship, and the work utilized Queen Mary’s Andrena HPC facility, supported by QMUL Research-IT. This work was also supported by the Engineering and Physical Sciences Research Council [grant number EP/Y009800/1], through funding from Responsible Ai UK (KP0016).

### A Additional Norm-Variation Analysis

The main text reports the layerwise pattern of lastprompt-token norm variation. This appendix provides the supporting details. We first report percorpus CV at the 75%-depth layer, and then expand the analysis to prompt and generation po-

sitions. These per-position plots separate crosssample norm variation from position-dependent norm effects, which can be hidden by aggregate statistics.

Per-corpus norm variation. Table 2 reports the per-corpus CV of last-prompt-token hidden-state norms at the 75%-depth layer. The same qualitative pattern as in the main text holds across corpora: Llama and Qwen models usually have moderate CV, while Gemma has substantially larger variation because of its post-norm architecture.

Prompt-token positions. Figure 8 shows pointwise CV across prompt positions. The largest position-specific effect appears at the beginning of the prompt: in Llama and Qwen models, the first token behaves like an attention-sink position and has a distinct norm distribution. After the first few tokens, CV settles to a more stable plateau. Gemma remains different, with elevated variation across many layers because of its post-norm architecture.

Generation-token positions. Figure 9 shows the same analysis for generated tokens. Compared with prompt tokens, generation positions are more stable for most instruction-tuned models, which is the relevant regime for the steering hook during decoding. The Llama base model is less stable under unconstrained generation and shows larger CV at later layers.

Cumulative CV. Figures 10 and 11 show cumulative CV when positions are pooled from the start of the sequence. For prompt tokens, the early attention-sink positions strongly affect the pooled statistic; as more content positions are included, this effect is diluted. For generated tokens, the cumulative curves converge quickly, indicating that generation-time norm variation is not dominated by a few outlier positions.

Layerwise token-population comparison. Figure 12 compares the mean CV across corpora for three token populations: the last prompt token, all prompt tokens, and generated tokens. The allprompt-token curve is much larger because it pools positions with different typical norm scales. In contrast, generation-token CV is more stable for most instruction-tuned models, which supports the interpretation that decoding-time steering operates on a comparatively stable radial landscape.

Table 2: Per-corpus CV of last-prompt-token hidden-state norms at the 75%-depth layer.

Corpus Llama-8B Qwen-7B Gemma-9B Llama-8B (base) Llama-1B Qwen-3B Llama-70B OpenWebText 12.65% 8.99% 54.93% 14.60% 7.86% 9.80% 13.44% Alpaca 7.63% 8.43% 62.12% 7.89% 5.06% 8.08% 8.72% arXiv 6.68% 5.85% 88.08% 7.54% 5.42% 8.50% 6.72% WritingPrompts 10.26% 8.25% 50.77% 11.63% 7.89% 8.58% 12.10% TruthfulQA 5.98% 7.33% 51.48% 8.23% 4.86% 7.03% 8.85% Natural Questions 5.82% 5.73% 46.79% 5.93% 6.04% 7.06% 5.69% CivilComments 7.01% 8.00% 81.66% 7.30% 4.53% 7.68% 8.85% CNN/DailyMail 13.21% 8.53% 21.25% 14.00% 9.11% 8.87% 13.27% PubMedQA 5.16% 4.70% 17.45% 4.97% 4.46% 4.71% 6.29% CodeSearchNet 5.75% 6.05% 52.12% 5.80% 4.51% 6.51% 4.51%

Pointwise CV vs prompt token position

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

Gemma-2-9B-IT

Llama-3.1-8B (base)

| |L0 L6<br><br>L12 L18<br><br>L23 L29 L35 L41<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

L0 L4 L9 L13

L18 L22 L27 L31

L0 L4 L8 L12

L15 L19 L23 L27

L0 L4 L9 L13

L18 L22 L27 L31

400

125

20

20

PointwiseCV(%)

300

100

15

15

75

200

10

10

50

100

5

5

25

0

0

0

0

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Token position

Token position

Token position

Token position

Llama-3.2-1B-Instruct

Qwen2.5-3B-Instruct

Llama-3.1-70B-Instruct

120

L0 L2 L4 L6

L9

L0 L5

L20 L25 L30 L35

L0

L45 L56 L68 L79

60

L11 L13 L15

L11 L23 L34

100

400

PointwiseCV(%)

L10 L15

80

300

40

60

200

40

20

100

20

0

0

0

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Token position

Token position

Token position

- Figure 8: Pointwise CV of hidden-state norms across prompt-token positions. The first prompt positions, especially position 0, show strong architecture-dependent effects; later positions settle to a more stable plateau.

Mean norm profiles. Figures 13 and 14 report the corresponding mean norm profiles. Prompttoken norms show strong position effects, especially at the first token in Llama and Qwen architectures. In contrast, generation-token norms are nearly constant across positions at a fixed layer for most instruction-tuned models. This supports the main-text interpretation that prompt-token statistics can be strongly position-dependent, while decoding positions are more stable.

Overall, prompt-token norms are strongly affected by position, whereas generation-token norms are more stable across decoding steps. Thus, norm preservation in spherical steering should be understood as preserving each token’s own radius, not as forcing all activations onto a shared global radius.

Token-population summary. Table 3 summarizes norm CV at the 75%-depth layer for the three token populations used in the norm-variation analysis. The all-prompt-token statistic is much larger because it pools positions with very different typi-

cal norm scales. Generated tokens are more stable for most instruction-tuned models, which is the regime most relevant to steering during decoding.

### B Additional Directional-Encoding Results

The main text reports the layerwise probe curves showing that concept information is primarily encoded in activation direction. Here we provide the corresponding per-model and per-dataset probe accuracies. We compare linear probes trained on raw hidden states, unit-normalized hidden states, and norm-only features. Across all evaluated concepts, unit-normalized probes closely match raw probes, while norm-only probes remain near chance.

The table supports the directional-encoding claim used throughout the paper. Normalizing hidden states to unit length causes almost no loss in probe accuracy, indicating that the concepts remain linearly accessible after removing the radial component. In contrast, probes trained only on the activation norm are close to chance for all datasets and model families. This pattern also holds for

Pointwise CV vs generation token position

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

Gemma-2-9B-IT

Llama-3.1-8B (base)

100

| |L0 L4 L9 L13<br><br>L18 L22 L27 L31<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L4 L8 L12<br><br>L15 L19 L23 L27<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L6<br><br>L12 L18<br><br>L23 L29 L35 L41<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L4 L9 L13<br><br>L18 L22 L27 L31<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

30

50

25

25

PointwiseCV(%)

80

40

20

20

60

30

15

15

40

20

10

10

20

10

5

5

0

0

0

0

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Token position

Token position

Token position

Token position

Llama-3.2-1B-Instruct

Qwen2.5-3B-Instruct

Llama-3.1-70B-Instruct

| |L0 L2 L4 L6<br><br>L9<br><br>L11 L13 L15<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L5<br><br>L10 L15<br><br>L20 L25 L30 L35<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0<br><br>L11 L23 L34<br><br>L45 L56 L68 L79<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

50

50

200

PointwiseCV(%)

40

40

150

30

30

100

20

20

50

10

10

0

0

0

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Token position

Token position

Token position

- Figure 9: Pointwise CV of hidden-state norms across generation-token positions. Instruction-tuned models show relatively stable generation-token CV, while the Llama base model has elevated variation at later layers.

| |L0 L4 L9 L13<br><br>L18 L22 L27 L31<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

100

200

300

400

500

600

CumulativeCV(%)

Llama-3.1-8B-Instruct

| || |L0 L4 L8 L12<br><br>L15 L19 L23 L27<br><br>|
|---|---|
| | |
| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

200

400

600

800

1000

Qwen2.5-7B-Instruct

| |L0 L6<br><br>L12 L18<br><br>L23 L29 L35 L41<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

25

50

75

100

125

150

Gemma-2-9B-IT

| |L0 L4 L9 L13<br><br>L18 L22 L27 L31<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

100

200

300

400

500

600

Llama-3.1-8B (base)

| || |L0 L2 L4<br><br>L9<br><br>L11 L13<br><br>|
|---|---|
| |L6 L15<br><br>|
| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

200

400

600

800

CumulativeCV(%)

Llama-3.2-1B-Instruct

| |L0 L5<br><br>L10 L15<br><br>L20 L25 L30 L35<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

100

200

300

400

500

Qwen2.5-3B-Instruct

| |L0<br><br>L11 L23 L34<br><br>L45 L56 L68 L79<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500

Token position

0

200

400

600

Llama-3.1-70B-Instruct

Cumulative CV [0:i] vs prompt token position

- Figure 10: Cumulative CV over prompt-token positions. Pooling early attention-sink positions with later content positions produces large prompt-token CV, showing that aggregate prompt statistics are sensitive to positiondependent norm scale.

Gemma, where norm variation is much larger than in the Llama and Qwen models, showing that large radial variability does not imply that the concept itself is encoded in the norm.

### C CAA-m Per-Token Matching Algorithm

CAA-m chooses a separate additive coefficient for every token so that the normalized output reaches the requested concept score. Let

##### x = r(cs + 1 − c2 v),

where r = ∥x∥, c = ⟨x/∥x∥,s⟩, and v is the unit component of x/∥x∥ orthogonal to s. For y =

x + αs, the target constraint is

Since

y ∥y∥

,s = γ.

y = (rc + α)s + r 1 − c2 v, solving the constraint gives

√

1 − c2 1 − γ2

γ

− c . (13)

α = r

This expression is well-defined for γ ∈ (−1,1). When |γ| approaches 1, the required additive coefficient can become large, reflecting the fact that an almost perfectly aligned target direction may require a large displacement for tokens whose initial

Cumulative CV [0:i] vs generation token position

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

Gemma-2-9B-IT

Llama-3.1-8B (base)

25

| || |L0 L4 L9<br><br>L18 L22 L27<br><br>|
|---|---|
| |L13 L31<br><br>|
| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L4 L8 L12<br><br>L15 L19 L23 L27<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L6<br><br>L12 L18<br><br>L23 L29 L35 L41<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L4 L9 L13<br><br>L18 L22 L27 L31<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

30

40

500

CumulativeCV(%)

25

20

400

30

20

15

300

15

20

10

200

10

10

5

100

5

0

0

0

0

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Token position

Token position

Token position

Token position

Llama-3.2-1B-Instruct

Qwen2.5-3B-Instruct

Llama-3.1-70B-Instruct

| |L0 L2 L4 L6<br><br>L9<br><br>L11 L13 L15<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |L0 L5<br><br>L10 L15<br><br>L20 L25 L30 L35<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| || |L0 L45<br><br>|
|---|---|
| |L11 L23 L34<br><br>L56 L68 L79<br><br>|
| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

40

50

CumulativeCV(%)

30

40

30

30

20

20

20

10

10

10

0

0

0

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Token position

Token position

Token position

- Figure 11: Cumulative CV over generation-token positions. The curves converge quickly for most instruction-tuned models, indicating that generation-token norm variation is not dominated by a small number of outlier positions.

|Last prompt tok<br><br>All prompt toks<br><br>Generation toks| | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 10 20 30

Layer

- 100
- 101
- 102
- 103

MeanCVacrosscorpora(%)

Llama-3.1-8B-Instruct

| | | | | |
|---|---|---|---|---|
| | | | | |

0 10 20

Layer

Qwen2.5-7B-Instruct

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

0 10 20 30 40

Layer

Gemma-2-9B-IT

0 10 20 30

Layer

Llama-3.1-8B (base)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 5 10 15

Layer

- 100
- 101
- 102
- 103

MeanCVacrosscorpora(%)

Llama-3.2-1B-Instruct

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 10 20 30

Layer

Qwen2.5-3B-Instruct

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 20 40 60 80

Layer

Llama-3.1-70B-Instruct

- Figure 12: Mean CV across corpora for last prompt tokens, all prompt tokens, and generation tokens. Positiondependent norm variation, especially from early attention-sink positions, strongly inflates the all-prompt-token CV. Generation-token norms are more stable for most instruction-tuned models.

residual component orthogonal to s is large. Thus it controls the angular concept score while allowing the norm to change.

### D Additional Fixed-Angle Steering Results

This appendix provides additional results for the comparison between S and CAA-m at matched pertoken target γ. Since both methods reach the same normalized concept direction, their difference is radial: S preserves the original norm, while CAAm leaves the additive norm change intact. Here we show per-dataset gaps comparison and the full per-cell tables.

Per-dataset gaps. Figure 15 reports the difference between CAA-m and S across datasets and

models. At low γ, the two methods are close on all metrics. At larger γ, CAA-m opens a large stability gap: it usually has much lower perplexity and higher MMLU accuracy, while the downstream task metric remains comparable on average but varies more across models and datasets.

### E Additional Fixed-Strength Steering Results

This appendix provides additional results for the fixed-strength methods: CAA, CAA-r, and AS. Unlike S and CAA-m, these methods use a single global steering parameter and are calibrated to match the target mean concept score γ¯. This comparison isolates whether norm preservation alone explains downstream stability. CAA-r and AS both preserve the hidden-state norm, while CAA does

Mean norm vs position Prompt tokens

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

Gemma-2-9B-IT

Llama-3.1-8B (base)

15000

500

5000

L0 L4 L9 L13

L18 L22 L27 L31

L0 L4 L8 L12

L15 L19 L23 L27

L0 L6

L23 L29 L35 L41

L0 L4 L9 L13

L18 L22 L27 L31

500

12500

400

4000

L12 L18

400

10000

Meannorm

300

3000

300

7500

200

2000

200

5000

100

100

1000

2500

0

0

0

0

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Token position

Token position

Token position

Token position

Llama-3.2-1B-Instruct

Qwen2.5-3B-Instruct

Llama-3.1-70B-Instruct

800

L0 L2 L4 L6

L9

L0 L5

L20 L25 L30 L35

L0

L45 L56 L68 L79

L11 L13 L15

L11 L23 L34

600

3000

L10 L15

600

Meannorm

400

2000

400

200

1000

200

0

0

0

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Token position

Token position

Token position

- Figure 13: Mean hidden-state norm across prompt-token positions. Norms increase with layer depth, and the first prompt position can have a disproportionately large norm in Llama and Qwen architectures.

0 20 40 60 80 100 120

Token position

0

10

20

30

40

50

Meannorm

Llama-3.1-8B-Instruct

L0 L4 L9 L13

L18 L22 L27 L31

0 20 40 60 80 100 120

Token position

0

100

200

300

400

Qwen2.5-7B-Instruct

L0 L4 L8 L12

L15 L19 L23 L27

0 20 40 60 80 100 120

Token position

200

400

600

800

1000

Gemma-2-9B-IT

L0 L6

L12 L18

L23 L29 L35 L41

0 20 40 60 80 100 120

Token position

0

50

100

150

200

Llama-3.1-8B (base)

L0 L4 L9 L13

L18 L22 L27 L31

0 20 40 60 80 100 120

Token position

2.5

5.0

7.5

10.0

12.5

15.0

Meannorm

Llama-3.2-1B-Instruct

L0 L2 L4 L6

L9

L11 L13 L15

0 20 40 60 80 100 120

Token position

0

100

200

300

400

Qwen2.5-3B-Instruct

L0 L5

L10 L15

L20 L25 L30 L35

0 20 40 60 80 100 120

Token position

0

20

40

60

80

Llama-3.1-70B-Instruct

L0

L11 L23 L34

L45 L56 L68 L79

Mean norm vs position Generation tokens

- Figure 14: Mean hidden-state norm across generation-token positions. At each layer, generation-token norms are nearly constant across positions for most instruction-tuned models.

not; however, the results show that the token-level angular profile is more important than norm preservation alone.

Downstream trajectory. Figure 16 compares the downstream-metric trajectory of the three fixedstrength methods as the target mean concept score increases. The methods behave similarly at moderate targets, while AS becomes less stable at high γ¯.

CAA-r versus CAA. Figure 17 compares CAA-r and CAA at matched mean concept score. These methods have the same normalized output direction after the additive update; CAA-r only rescales the result back to the original norm. Consequently, their downstream and PPL curves stay close across targets. Figure 18 shows the same comparison as per-dataset gaps, confirming that post-hoc renor-

malization is not the main factor controlling stability in the fixed-strength setting.

CAA-r versus AS. Figure 19 compares CAA-r and AS at matched mean concept score. Both methods preserve ∥y∥ = ∥x∥, but they distribute the angular intervention differently across tokens. CAA-r inherits a token-dependent angular displacement from the additive update, whereas AS applies a fixed spherical displacement. Figure 20 shows that this difference produces a large PPL gap at high γ¯: AS becomes substantially less stable despite preserving the norm. Thus, norm preservation alone is not sufficient to explain the steering–quality tradeoff.

Calibration dose response. Figure 21 shows the calibration curves used to match the target mean concept score. For CAA-r, the required additive

Table 3: Norm CV at the 75%-depth layer for three token populations, averaged across corpora. Generation tokens are the positions directly modified by the steering hook during decoding.

Model Last prompt token All prompt tokens Generation tokens

Llama-3.1-8B-Instruct 8.0% 140.0% 12.5% Qwen2.5-7B-Instruct 7.2% 478.8% 11.0% Gemma-2-9B-it 52.7% 99.6% 21.1%

- Llama-3.1-8B 8.8% 126.1% 89.6%
- Llama-3.2-1B-Instruct 6.0% 365.3% 15.4% Qwen2.5-3B-Instruct 7.7% 220.6% 13.7% Llama-3.1-70B-Instruct 8.8% 149.6% 15.7%

strength is highly model-dependent because it depends on the scale of the residual stream. Gemma requires a much wider search range for α. In contrast, AS is calibrated by angular displacement and is therefore less sensitive to activation norm scale.

### F Concept-Score Closure

This appendix compares the steering methods by how tightly they close the gap to the requested concept score. The main experiments compare downstream behavior and perplexity; here we isolate the intervention itself by measuring the achieved pertoken concept score after steering. This diagnostic separates methods that enforce a target score tokenby-token from methods that only match a target score on average.

Per-token score variance. Figure 22 reports the standard deviation of achieved concept scores across tokens at the target level used in the closure diagnostic. S and CAA-m have near-zero spread because they explicitly solve for the target concept score for each token. In contrast, CAA, CAA-r, and AS use a single global steering strength. Even when their mean achieved score is calibrated to the target, individual tokens spread over a much wider range. This confirms that concept-score closure is a separate axis of method design: two methods can have the same average steering strength but very different token-level precision.

Concept-score distributions. Figure 23 shows the full achieved-score distributions on CivilComments. The distributional view makes the same point as the standard-deviation summary: targeted methods produce a sharp peak at the requested score, whereas fixed-strength methods produce broader token-level distributions. The difference between CAA-r and AS also becomes more pro-

nounced at higher target scores. Although both methods are calibrated to the same mean concept score and both preserve the hidden-state norm, their achieved-score distributions diverge at large γ because they induce different token-level angular profiles.

Overall, these results justify separating semantic strength from concept-score closure. Meanmatched fixed-strength methods can express the target concept on average, but they do not apply the same intervention to every token. Per-token targeted methods close the concept score much more precisely, which explains why they occupy a distinct part of the control–quality trade-off in the main experiments.

### G Off-Arc Perturbations

This appendix tests whether the great-circle arc used by S is empirically meaningful, not only geometrically minimal. Starting from the spherical solution, we perturb the residual component away from the arc while keeping both the hidden-state norm and the target concept score fixed. Thus, any degradation caused by the perturbation cannot be explained by weaker concept control or by a different norm; it must come from moving away from the task-relevant residual direction.

Using the notation from the main text, we perturb the residual direction by an angle δ toward a direction q orthogonal to both the concept direction and the original residual direction:

y(δ) = ∥x∥ γs + 1 − γ2 (cosδ v + sinδ q) .

All points on this sweep have the same norm and the same concept score. The arc solution is δ = 0. If the great-circle arc is the empirically relevant axis, then perturbing in either direction should de-

Downstream gap (CAA-m S)

TQA

###### SST-2

Civil

IMDB

metric(CAA-mS),pp

60

Llama-8B Qwen-7B Gemma-9B

40

20

Llama-8B-base

Llama-1B Qwen-3B Llama-70B

0

20

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Perplexity (CAA-m S)

PPL(CAA-mS,symlog)

TQA

###### SST-2

Civil

IMDB

0

101

102

103

104

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

MMLU (CAA-m S)

TQA

###### SST-2

Civil

IMDB

MMLU(CAA-mS),pp

50

40

30

20

10

0

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

- Figure 15: Per-dataset S vs. CAA-m gaps at matched per-token target γ. Top: downstream-metric gap, CAA-m − S, in percentage points. Middle: WikiText-103 perplexity gap, CAA-m − S, using a symlog scale. Bottom: MMLU gap, CAA-m − S, in percentage points. The dashed grey line marks zero gap. For downstream metrics and MMLU, positive values favour CAA-m. For perplexity, negative values favour CAA-m because lower perplexity is better.

grade perplexity, MMLU, or downstream task performance.

Aggregate off-arc degradation. Table 5 reports degradation relative to the arc solution. PPL increase away from δ = 0, while MMLU and downstream task metrics generally decrease. The effect is approximately symmetric and becomes stronger

- as |δ| increases.

Perturbation direction type. Table 6 breaks the PPL effect down by the type of off-arc direction. Random directions produce the mildest degradation, PCA directions produce the steepest valleys, and cross-dataset directions fall in between. This suggests that the most important residual directions are aligned with high-variance structure in the residual subspace, while concept axes from related datasets also overlap with task-relevant residual variation.

PPL is minimized at δ = 0 in almost all com-

pleted cells, and the few exceptions have negligible relative gaps. Overall, perturbing away from the spherical arc worsens model behavior even though the concept score and norm are held fixed. This supports the interpretation that the S direction is not merely the shortest geometric edit, but also the empirically task-relevant residual direction.

### H Additional Norm-Scaling Results

This appendix provides the detailed tables for the norm-scaling sweep on top of S. In this experiment, the angular component is held fixed while the norm is multiplied by β. Thus, changing β does not change the target concept score; it only changes the radius of the steered activation. This makes the sweep a direct test of whether the norm acts as an independent stability lever.

PPL summary. Table 7 summarizes the mean PPL ratio for each (γ,β) pair, together with fold-

Downstream metric change vs.\ target concept score

TruthfulQA

###### SST-2

CivilComments

IMDB

0.000

CAA

| | |
|---|---|
| | |

0.06

taskmetric

CAA-r

0.1

0.2

0.025

AS

0.04

0.0

0.050

0.1

0.02

0.1

0.075

0.0

0.00

0.100

0.2

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

- Figure 16: Mean downstream metric change, ∆ task, versus target mean concept score, averaged across models per dataset. CAA, CAA-r, and AS produce similar gains at moderate targets, while AS diverges at high γ¯ because its fixed spherical displacement causes larger token-level disruption.

0.1 0.3 0.5 0.7

30

35

40

45

MC1(%)

TQA

0.1 0.3 0.5 0.7

20

40

60

80

100

positiverate(%)

SST-2

0.1 0.3 0.5 0.7

80

85

90

95

100

non-toxicity(%)

Civil

0.1 0.3 0.5 0.7

40

60

80

positiverate(%)

IMDB

Downstream metric under CAA-r vs CAA steering

| | | | | |
|---|---|---|---|---|
| | | | | |

0.1 0.3 0.5 0.7

- 100
- 101
- 102

PPLratio(WikiText,log)

TQA

| | | | | |
|---|---|---|---|---|
| | | | | |

0.1 0.3 0.5 0.7

SST-2

|CAA-r<br><br>CAA<br><br>Llama-8B Qwen-7B Gemma-9B<br><br>Llama-8B-base<br><br>Llama-1B Qwen-3B<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

0.1 0.3 0.5 0.7

Civil

| | | | | |
|---|---|---|---|---|
| | | | | |

0.1 0.3 0.5 0.7

IMDB

Perplexity under CAA-r vs CAA steering

- Figure 17: CAA-r versus CAA at matched mean concept score. Top: downstream metric versus γ¯. Bottom: WikiText-103 PPL ratio. Since CAA-r only renormalizes the additive CAA output, the two methods remain close in downstream behavior.

level win counts. The monotone pattern at high γ is the key result: once the angular edit is large, increasing the norm reduces the PPL penalty.

Task-metric summary. Table 8 reports the corresponding downstream task-metric changes. Compared with PPL, task performance is much less sensitive to β: the spread across norm scales remains small at each γ. This supports the interpretation that β is primarily a stability knob, not a semanticcontrol knob.

Large-model sensitivity. Table 9 isolates the 70B model. The larger model is more sensitive to strong angular edits, producing larger PPL ratios

- at high γ, but the ordering over β remains the same. Larger norm scales still reduce PPL most strongly at high steering strengths.

Overall, the β sweep confirms that the radius is

not merely a passive quantity. Once the angular edit is fixed, changing the norm has little effect on the semantic task metric but a large effect on generation stability. This strengthens the paper’s two-parameter view of steering: γ controls the angular concept intervention, while β controls the radial stability of the resulting activation.

### I Datasets and Data Sources

This section summarizes the datasets used in our experiments. We use three groups of data: concept datasets for direction construction and downstream steering evaluation, auxiliary capability and language-modeling benchmarks, and unlabeled corpora for norm-variation diagnostics.

Concept datasets. We evaluate steering on four concept datasets. TruthfulQA is used for truth-

Downstream gap (CAA-r CAA)

metric(CAA-rCAA),pp

TQA

###### SST-2

Civil

IMDB

|Llama-8B| |
|---|---|
|Qwen-7B<br><br>Gemma-9B<br><br>Llama-8B-base<br><br>Llama-1B Qwen-3B Llama-70B<br><br>| |

0

10

20

30

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Perplexity (CAA-r CAA)

PPLratio(CAA-rCAA,symlog)

TQA

###### SST-2

Civil

IMDB

0

| |
|---|

101

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

- Figure 18: CAA-r − CAA gap per dataset, with one line per model. Top: downstream-metric difference in percentage points. Bottom: WikiText-103 PPL-ratio difference, shown on a symlog scale. The dashed grey line marks zero gap. The gaps remain small across most targets, showing that renormalizing CAA does not substantially change behavior in this fixed-strength regime.

fulness steering and closed-form multiple-choice evaluation (Lin et al., 2022). SST-2, derived from the Stanford Sentiment Treebank, is used for sentiment steering (Socher et al., 2013). CivilComments is used for toxicity and non-toxicity steering (Borkan et al., 2019). IMDB is used as a second sentiment dataset with longer movie-review inputs (Maas et al., 2011). These datasets define the contrastive concept directions and the task-specific downstream metrics reported in the main experiments.

Auxiliary evaluation datasets. To measure whether steering degrades general model behavior, we evaluate perplexity on WikiText-103 (Merity et al., 2016). We also evaluate general capability using MMLU, a multi-task benchmark covering broad factual and reasoning domains (Hendrycks et al., 2021). These auxiliary datasets are not used to construct steering directions; they are used only to measure quality and capability retention under intervention.

Norm-variation corpora. For the normvariation analysis, we use a heterogeneous set of corpora spanning web text, instruction data, scientific writing, stories, question answering, toxicity comments, news, biomedical text, and code. Specifically, we sample from OpenWebText (Gokaslan et al., 2019), Alpaca (Taori et al.,

- 2023a), arXiv scientific papers (Cohan et al., 2018), WritingPrompts (Fan et al., 2018), TruthfulQA (Lin et al., 2022), Natural Questions (Kwiatkowski et al., 2019), CivilComments (Borkan et al., 2019), CNN/DailyMail (Hermann et al., 2015; See et al., 2017), PubMedQA (Jin et al., 2019a), and CodeSearchNet (Husain et al., 2019). This mixture is intended to test whether the radial geometry of hidden states is stable across content domains rather than being an artifact of a single dataset.

Dataset licenses. Table 10 summarizes the licenses or usage terms associated with the dataset distributions used in this work. Licenses vary across datasets and, in some cases, across mirrors of the same dataset. We use all datasets only for research evaluation and do not redistribute the datasets. For datasets whose original source does not specify a clear open-data license, we report the relevant usage status conservatively and refer readers to the original source or distribution page.

J Models and Licenses

Table 11 summarizes the model checkpoints used in our experiments, together with their source families and licenses. We evaluate three model families: Llama (Dubey et al., 2024), Qwen2.5 (Qwen Team et al., 2024), and Gemma 2 (Gemma Team et al.,

- 2024). All models are used only for research evalu-

Downstream metric under CAA-r vs AS steering

###### TQA

###### SST-2

Civil

IMDB

100

100

positiverate(%)

positiverate(%)

non-toxicity(%)

75

80

95

40

MC1(%)

90

50

30

60

85

25

20

40

80

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Perplexity under CAA-r vs AS steering

###### TQA

###### SST-2

Civil

IMDB

PPLratio(WikiText,log)

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

|CAA-r<br><br>AS<br><br>Llama-8B Qwen-7B Gemma-9B<br><br>Llama-8B-base<br><br>Llama-1B Qwen-3B<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

- 100
- 101
- 102

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

- Figure 19: CAA-r versus AS at matched mean concept score. Top: downstream metric versus γ¯. Bottom: WikiText103 PPL ratio. Both methods preserve the hidden-state norm, but they induce different token-level angular profiles. AS becomes much more costly in PPL at high γ¯, showing that norm preservation alone is not sufficient for stable steering.

ation; we do not redistribute model weights.

The licenses differ in permissiveness. Qwen2.57B-Instruct is released under Apache-2.0, whereas Qwen2.5-3B-Instruct is governed by the Qwen Research License. The Llama checkpoints are released under Meta’s Llama Community License, with separate license versions for Llama 3.1 and Llama 3.2. Gemma-2-9B-it is distributed under Google’s Gemma Terms of Use. We report these licenses for transparency and refer readers to the original model cards and license documents for the full legal terms.

Downstream gap (CAA-r AS)

metric(CAA-rAS),pp

TQA

###### SST-2

Civil

IMDB

40

Llama-8B Qwen-7B Gemma-9B

20

Llama-8B-base

0

Llama-1B Qwen-3B Llama-70B

20

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

Perplexity (CAA-r AS)

PPLratio(CAA-rAS,symlog)

TQA

###### SST-2

Civil

IMDB

0

101

| |
|---|

102

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

0.1 0.3 0.5 0.7

- Figure 20: CAA-r − AS gap per dataset, with one line per model. Top: downstream-metric difference in percentage points. Bottom: WikiText-103 PPL-ratio difference, shown on a symlog scale. Negative PPL gaps mean CAA-r has lower perplexity than AS. Although both methods preserve norm, AS incurs much larger PPL degradation at high γ¯.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 101 102

CAA-r strength (log scale)

0.0

0.2

0.4

0.6

0.8

1.0

Achievedmeanconceptscorec

CAA-r dose response (calibrated per target )

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

AS rotation (radians)

0.0

0.2

0.4

0.6

0.8

1.0

Achievedmeanconceptscorec

AS dose response (calibrated per target )

Llama-3.1-8B-Inst Qwen-2.5-7B-Inst Gemma-2-9B-IT

- Llama-3.1-8B

- Llama-3.2-1B-Inst

Qwen-2.5-3B-Inst

Llama-3.1-70B-Inst

- Figure 21: Dose-response curves for fixed-strength calibration. Left: CAA-r mean concept score versus additive strength α on a log scale. Right: AS mean concept score versus angular displacement ∆θ. CAA-r calibration is sensitive to residual-stream norm scale, while AS calibration is norm-invariant.

Llama-8BQwen-7BGemma-9BLlama-8B-baseLlama-1BQwen-3BLlama-70B

10 5

10 4

10 3

10 2

10 1

Concept-scorestd(logscale)

TQA

S

CAA-m

CAA-r

AS

Llama-8BQwen-7BGemma-9BLlama-8B-baseLlama-1BQwen-3BLlama-70B

SST-2

Llama-8BQwen-7BGemma-9BLlama-8B-baseLlama-1BQwen-3BLlama-70B

Civil

Llama-8BQwen-7BGemma-9BLlama-8B-baseLlama-1BQwen-3BLlama-70B

IMDB

Per-token concept-score std at =0.3

- Figure 22: Per-token concept-score standard deviation at matched target score. Per-token targeted methods collapse tightly around the requested value, while fixed-strength methods have much larger spread. This shows that matching the mean concept score is not equivalent to closing the concept score for each token.

- Table 4: Linear-probe accuracies for raw, unit-normalized, and norm-only representations. Unit-normalized features retain almost all of the predictive information in raw hidden states, while norm-only features remain close to chance.

Model Dataset Raw h Unit h/∥h∥ Norm-only ∥h∥

Llama-3.1-8B-Instruct TruthfulQA 81.6% 81.1% 53.8% Llama-3.1-8B-Instruct SST-2 92.5% 92.4% 53.4% Llama-3.1-8B-Instruct CivilComments 84.0% 83.9% 53.1% Llama-3.1-8B-Instruct IMDB 88.5% 88.4% 47.9%

Qwen2.5-7B-Instruct TruthfulQA 70.1% 70.1% 57.9% Qwen2.5-7B-Instruct SST-2 91.5% 91.4% 51.0% Qwen2.5-7B-Instruct CivilComments 80.6% 80.2% 52.9% Qwen2.5-7B-Instruct IMDB 87.3% 86.9% 53.7%

Gemma-2-9B-it TruthfulQA 57.6% 58.4% 51.7% Gemma-2-9B-it SST-2 71.6% 71.0% 48.8% Gemma-2-9B-it CivilComments 65.9% 65.0% 46.6% Gemma-2-9B-it IMDB 75.4% 74.5% 51.1%

- Llama-3.1-8B TruthfulQA 80.4% 80.6% 51.5%

- Llama-3.1-8B SST-2 92.8% 93.0% 49.7%

- Llama-3.1-8B CivilComments 84.0% 84.0% 56.8%

- Llama-3.1-8B IMDB 88.9% 88.9% 53.9%

- Llama-3.2-1B-Instruct TruthfulQA 73.6% 74.0% 56.2%

- Llama-3.2-1B-Instruct SST-2 88.9% 88.9% 51.0%

- Llama-3.2-1B-Instruct CivilComments 78.6% 78.4% 48.1%

- Llama-3.2-1B-Instruct IMDB 83.6% 83.8% 51.0%

Qwen2.5-3B-Instruct TruthfulQA 66.2% 66.5% 56.1% Qwen2.5-3B-Instruct SST-2 87.5% 87.5% 52.9% Qwen2.5-3B-Instruct CivilComments 76.1% 76.5% 54.6% Qwen2.5-3B-Instruct IMDB 79.5% 79.9% 55.1%

Llama-3.1-70B-Instruct TruthfulQA 86.0% 85.5% 50.5% Llama-3.1-70B-Instruct SST-2 92.0% 92.5% 45.2% Llama-3.1-70B-Instruct CivilComments 83.9% 84.5% 47.1% Llama-3.1-70B-Instruct IMDB 91.2% 91.0% 52.4%

Concept-score distributions Civil (seed 0, fold 0)

= 0.1

= 0.3

= 0.5

= 0.7

15

10.0

8

S ( =0.0002)

S ( =0.0002)

S ( =0.0002)

S ( =0.0001)

7.5

CAA-r ( =0.0488)

CAA-r ( =0.0455)

CAA-r ( =0.0393)

CAA-r ( =0.0288)

Llama-8B

7.5

6

10

AS ( =0.0499)

AS ( =0.0478)

AS ( =0.0434)

AS ( =0.0358)

5.0

5.0

4

5

2.5

2.5

2

0

0.0

0.0

0

0.1 0.0 0.1 0.2 0.3

0.2 0.3 0.4 0.5

0.4 0.5 0.6 0.7

0.6 0.7 0.8

8

S ( =0.0005)

6 S ( =0.0004)

S ( =0.0003)

10.0 S ( =0.0002)

6

CAA-r ( =0.0613)

CAA-r ( =0.0613)

CAA-r ( =0.0487)

CAA-r ( =0.0322)

6

Qwen-7B

7.5

AS ( =0.0612)

AS ( =0.0612)

AS ( =0.0558)

AS ( =0.0461)

4

4

4

5.0

2

2

2

2.5

0.0

0

0

0

0.1 0.2 0.3 0.4 0.5

0.1 0.2 0.3 0.4 0.5

0.3 0.4 0.5 0.6

0.6 0.7 0.8

15 S ( =0.0001)

S ( =0.0002)

S ( =0.0002)

S ( =0.0001)

10

20

CAA-r ( =0.0555)

CAA-r ( =0.0504)

CAA-r ( =0.0312)

CAA-r ( =0.0536)

Gemma-9B

10

AS ( =0.0555)

AS ( =0.0551)

AS ( =0.0502)

AS ( =0.0415)

10

5

10

5

5

0

0

0

0

0.1 0.2 0.3 0.4 0.5

0.1 0.2 0.3 0.4 0.5

0.3 0.4 0.5 0.6 0.7

0.5 0.6 0.7 0.8

S ( =0.0002)

S ( =0.0001)

8 S ( =0.0001)

S ( =0.0001)

6

Llama-8B-base

6

CAA-r ( =0.0577)

CAA-r ( =0.0506)

CAA-r ( =0.0395)

CAA-r ( =0.0253)

10

6

AS ( =0.0581)

AS ( =0.0556)

AS ( =0.0504)

AS ( =0.0414)

4

4

4

5

2

2

2

0

0

0

0

0.1 0.0 0.1 0.2 0.3

0.1 0.2 0.3 0.4 0.5

0.3 0.4 0.5 0.6 0.7

0.6 0.7 0.8

6

8 S ( =0.0000)

6 S ( =0.0000)

S ( =0.0000)

S ( =0.0000)

CAA-r ( =0.0638)

CAA-r ( =0.0565)

CAA-r ( =0.0442)

CAA-r ( =0.0281)

10

6

Llama-1B

4

AS ( =0.0646)

AS ( =0.0617)

AS ( =0.0557)

AS ( =0.0456)

4

4

5

2

2

2

0

0

0

0

0.1 0.0 0.1 0.2 0.3 0.4

0.1 0.2 0.3 0.4 0.5

0.3 0.4 0.5 0.6 0.7

0.6 0.7 0.8 0.9

- 6

Qwen-3B

S ( =0.0004)

CAA-r ( =0.0594)

AS ( =0.0594)

0.1 0.2 0.3 0.4 0.5

0

2

4

6 S ( =0.0003)

CAA-r ( =0.0550)

AS ( =0.0574)

0.3 0.4 0.5 0.6 0.7

0

2

4

6

8 S ( =0.0002)

CAA-r ( =0.0449)

AS ( =0.0522)

0.6 0.7 0.8

0

5

10

S ( =0.0001)

CAA-r ( =0.0301)

AS ( =0.0432)

0.0 0.1 0.2 0.3

0.0

2.5

5.0

- 7.5

4

2

0

0.0 0.1 0.2 0.3

10.0

10.0 S ( =0.0000)

S ( =0.0000)

S ( =0.0000)

S ( =0.0000)

10

CAA-r ( =0.0407)

CAA-r ( =0.0440)

CAA-r ( =0.0440)

CAA-r ( =0.0359)

Llama-70B

7.5

10

AS ( =0.0397)

AS ( =0.0380)

AS ( =0.0344)

AS ( =0.0282)

5.0

5

5

2.5

0

0

0.0

0.2 0.3 0.4

0.4 0.5 0.6

0.6 0.7 0.8

- Figure 23: Achieved concept-score distributions on CivilComments. Each panel corresponds to one model and target score. Targeted methods collapse near the requested score, while fixed-strength methods spread across a wider interval even when calibrated to the same mean. At higher target scores, the CAA-r and AS distributions become more different, reflecting their distinct token-level angular profiles.

- Table 5: Aggregate degradation under off-arc perturbations. PPL is reported as ratios relative to δ = 0; MMLU and downstream metrics are reported as absolute changes relative to δ = 0. PPL ratio above 1 indicate degradation, while negative MMLU/downstream changes indicate degradation.

Metric δ = −0.2 δ = −0.1 δ = +0.1 δ = +0.2 PPL ratio (δ/0) ×1.22 ×1.05 ×1.06 ×1.23

∆MMLU −0.014 −0.003 −0.003 −0.012 ∆Downstream −0.005 +0.001 −0.004 −0.013

- Table 6: PPL ratio by off-arc perturbation direction type, averaged across completed cells. PCA directions produce the steepest degradation, consistent with the residual subspace containing task-relevant variation.

Direction type δ = −0.2 δ = −0.1 δ = +0.1 δ = +0.2

Random (n = 2) ×1.15 ×1.03 ×1.06 ×1.21 PCA (n = 2) ×1.43 ×1.09 ×1.07 ×1.32 Cross-dataset (n = 4) ×1.20 ×1.05 ×1.06 ×1.23

- Table 7: Mean PPL ratio under the β sweep. The best mean PPL for each γ is highlighted in bold. The lower block reports the number of folds in which each β achieves the lowest PPL.

γ β = 0.9 β = 1.0 (S) β = 1.1 β = 1.2 Best

0.1 1.10 1.10 1.11 1.12 β = 1.0 0.3 1.76 1.71 1.69 1.69 β = 1.2 0.5 9.82 7.98 6.88 6.21 β = 1.2 0.7 262.5 151.8 107.2 83.5 β = 1.2

Win counts: lowest PPL

0.1 26/70 25/70 9/70 10/70 0.3 5/70 10/70 15/70 40/70 0.5 2/70 0/70 6/70 62/70 0.7 0/70 0/70 0/70 70/70 —

- Table 8: Downstream task-metric change under the β sweep, in percentage points. “Spread” is the maximum minus minimum over β ∈ {0.9,1.0,1.1,1.2}.

γ β = 0.9 β = 1.0 (S) β = 1.1 β = 1.2 Spread 0.1 +2.03 +1.59 +1.15 +0.84 1.19pp 0.3 +6.53 +6.03 +6.20 +6.02 0.51pp 0.5 +6.65 +6.72 +7.10 +7.33 0.67pp 0.7 +2.34 +3.45 +3.94 +4.83 2.49pp

- Table 9: 70B-only PPL ratios under the β sweep. The 70B model amplifies the PPL gap at high γ, but the best β ordering is unchanged.

γ β = 0.9 β = 1.0 (S) β = 1.1 β = 1.2 Best

0.1 1.09 1.08 1.08 1.08 β = 1.1–1.2 0.3 2.13 1.96 1.85 1.78 β = 1.2 0.5 27.6 19.2 14.2 11.2 β = 1.2 0.7 585.8 453.1 354.6 281.6 β = 1.2

- Table 10: Dataset licenses or usage terms for the datasets used in our experiments. When a license depends on the distribution mirror, we report the license of the distribution we rely on or note that the license should be checked against the local source.

Dataset Use in this work License / usage terms Source TruthfulQA Concept / evaluation Apache-2.0 for common HF distributions; check source distribution (Lin et al., 2022; Hugging Face Datasets Contributors, 2024c) SST-2 Concept / evaluation Distribution-dependent; common public mirrors list CC0 or CC-BY-4.0 (Socher et al., 2013; Kaggle Dataset Contributors, 2024; Zenodo Dataset Contributors, 2023) CivilComments Concept / evaluation CC0-1.0 in the Hugging Face distribution (Borkan et al., 2019; Google and Hugging Face Datasets Contributors, 2024) IMDB Concept / evaluation Original Stanford release does not state a standard open-data license; used for research (Maas et al., 2011; Maas, 2011) WikiText-103 Perplexity evaluation CC BY-SA (Merity et al., 2016; Salesforce and Hugging Face Datasets Contributors, 2024) MMLU Capability evaluation MIT in common public distributions (Hendrycks et al., 2021; Center for AI Safety and Hugging Face Datasets Contributors, 2024) OpenWebText Norm-variation corpus CC0 for the dataset packaging; underlying web text not owned by dataset authors (Gokaslan et al., 2019; Gokaslan and Cohen, 2019) Alpaca Norm-variation corpus CC BY-NC 4.0; research / non-commercial use (Taori et al., 2023a,b) arXiv scientific papers Norm-variation corpus Distribution-dependent; license should be checked against the selected arXiv/paper source (Cohan et al., 2018; Hugging Face Datasets Contributors, 2024b) WritingPrompts Norm-variation corpus MIT for the Hugging Face distribution used by common loaders (Fan et al., 2018; Hugging Face Datasets Contributors, 2024d) Natural Questions Norm-variation corpus Creative Commons Share-Alike 3.0 on the Google download page (Kwiatkowski et al., 2019; Google Research, 2019) CNN/DailyMail Norm-variation corpus Apache-2.0 in the Hugging Face distribution (Hermann et al., 2015; See et al., 2017; Hugging Face Datasets Contributors, 2024a) PubMedQA Norm-variation corpus MIT (Jin et al., 2019a,b) CodeSearchNet Norm-variation corpus Code/documentation MIT; source-code examples retain per-file upstream licenses (Husain et al., 2019; GitHub and CodeSearchNet Contributors, 2019)

- Table 11: Model checkpoints used in the experiments. Licenses are reported according to the corresponding model cards or license pages.

Model checkpoint Provider / family Citation License

Llama-3.1-8B-Instruct Meta / Llama 3.1 Dubey et al. (2024) Llama 3.1 Community License (Meta AI, 2024a) Llama-3.1-8B Meta / Llama 3.1 Dubey et al. (2024) Llama 3.1 Community License (Meta AI, 2024a)

- Llama-3.1-70B-Instruct Meta / Llama 3.1 Dubey et al. (2024) Llama 3.1 Community License (Meta AI, 2024a)
- Llama-3.2-1B-Instruct Meta / Llama 3.2 Dubey et al. (2024) Llama 3.2 Community License (Meta AI, 2024b) Qwen2.5-7B-Instruct Alibaba / Qwen2.5 Qwen Team et al. (2024) Apache-2.0 (Qwen Team, 2024b) Qwen2.5-3B-Instruct Alibaba / Qwen2.5 Qwen Team et al. (2024) Qwen Research License (Qwen Team, 2024a) Gemma-2-9B-it Google / Gemma 2 Gemma Team et al. (2024) Gemma Terms of Use (Google, 2026)

