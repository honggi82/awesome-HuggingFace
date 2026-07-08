# arXiv:2602.18292v2[cs.LG]25Feb2026

### Decoding as Optimisation on the Probability Simplex:

From Top-K to Top-P (Nucleus) to Best-of-K Samplers

###### Xiaotong Ji1,†, Rasul Tutunov1, Matthieu Zimmer1, Haitham Bou-Ammar1,2,†

1 Huawei Noah’s Ark 2 AI Centre, Department of Computer Science, UCL † Equal contributions

Abstract: Decoding sitsbetween a language model and everything wedowithit, yet it is still treated as a heuristic knob-tuning exercise. We argue decoding should be understood as a principled optimisation layer: at each token, we solve a regularised problem over the probability simplex that trades off model score against structural preferences and constraints. This single template recovers greedy decoding, Softmax sampling, Top-K, Top-P, and Sparsemax-style sparsity as special cases, and explains their common structure through optimality conditions. More importantly, the framework makes it easy to invent new decoders without folklore. We demonstrate this by designing Best-of-K (BoK), a KL-anchored coverage objective aimed at multi-sample pipelines (self-consistency, reranking, verifier selection). BoK targets the probability of covering good alternatives within a fixed K-samplebudgetandimprovesempiricalperformance. Weshowthatsuchsamples can improve accuracy by, for example, +18.6% for Qwen2.5-Math-7B on MATH500 at high sampling temperatures.

Our message is simple: Decoding is not a hack; it is optimisation!

## 1 Introduction

Large language model (LLM) decoding is usually taught like a cookbook: Top-K [11, 31], temperature [14, 17], Top-P [15, 30], greedy [38, 35], beam search [42, 12]; a shelf of “tricks” you pick from depending on the application[16, 20, 24, 27, 40, 41, 48, 52], whether you want more determinism [13, 37], more diversity [43, 45, 50, 51], or fewer hallucinations [10, 44, 47]. The problem is that this framing makes decoding feel like a bag of heuristics: useful, but conceptually disconnected from the rest of machine learning.

This paper argues the opposite: many decoding strategies are not heuristics at all. They are solutions to explicit optimisation problems, often the same optimisation problem, with differentregularisersandconstraints. Onceyouseedecodingthroughthatlens, familiaralgorithms stop looking like folklore and start looking like principled design choices. Greedy decoding becomes a limiting case of an objective with no regularisation. Softmax sampling becomes the unique optimum of a score-maximisation problem regularised by (negative) Shannon entropy. Sparsity-inducing decoders arise from alternative convex penalties. In short: decoders differ less by “how they sample” and more by “what objective they are implicitly optimising.”

Our starting point is a small, but surprisingly powerful, shift in perspective. A decoder

does not have to immediately choose a token. At each step, it can first choose a distribution over tokens, and only then sample (or take the mode). This turns decoding into a clean optimisation problem over the probability simplex: pick a distribution that (i) puts mass on high-scoring tokens while (ii) satisfying desirable structural preferences such as smoothness, sparsity, or staying close to a reference distribution. This “distribution-first” view is general enough to cover deterministic and stochastic decoding in one line, and it exposes what decoding algorithms do: they are trading off score against regularisation under simplex constraints.

With that formulation in place, we do something that decoder discussions rarely do: we derivetheoptimalityconditionscarefully. Becausethevariableisaprobabilitydistribution, constraints are not decorative; they shape the solution. The simplex geometry introduces the familiar “active vs inactive” behaviour: tokens assigned a nonzero probability satisfy an equality condition, while tokens pushed to zero satisfy an inequality. These KKT-style conditions act like a master key: once derived, you can plug in a choice of regulariser and immediately recover the structure of the decoder it implies.

We then use this master key to re-derive canonical decoding rules as special cases. Setting the regulariser weight to zero collapses the optimisation to a linear objective on the simplex, recovering greedy decoding. Choosing negative entropy yields a smooth interior optimumandproducesthesoftmaxdistributionasaclosed-formsolution,wheretemperaturesamplingisnotahack; itistheoptimiserofamaximum-entropy-regularisedobjective. More broadly, this paper builds the intuition that “decoding methods” are best understood as regularisation families, rather than as unrelated procedures. The takeaway is simple: decoding is an optimisation layer sitting on top of the model’s scores. More precisely, decoding is a convex optimisation problem whose geometry is determined by the choice of regulariser. Once we treat it that way, we gain a principled vocabulary for designing new decoders: decide the behaviour (diversity, sparsity, conservatism, stability), encode it as a regulariser or constraint, and let the resulting optimisation problem define the algorithm.

Beyond theoretical unification, our framework serves as a generative tool for designing next-generation decoding objectives. We exemplify this by addressing the inefficiencies in current multi-sample pipelines through the design of Best-of-K (BoK). Unlike traditional methods that rely on empirical folklore, BoK utilises a KL-anchored coverage objectivetomaximisethelikelihoodofcapturinghigh-qualitycandidateswithinaconstrainedKsample budget. Our results demonstrate that by optimising for coverage, BoK not only provides a more principled decoding path but also achieves improved practical performance.

Across two 7B Qwen models (a math-specialised variant and a general-purpose variant) and three complementary benchmarks, MATH500, GPQA-diamond, and HumanEval, BoK samplers act as a practical decoding-time regulariser that improves multi-sample generation without any extra training or external verifiers. Sweeping temperature from neardeterministic(τ=0.1)tohighlystochastic(τ=0.9), BoKsamplersconsistentlymatchoroutperform standard sampling and Top-K, with the largest gains precisely where vanilla samplingismostdiversebutleastreliable. Forexample,onthemath-specialisedmodelatτ=0.9 on MATH500, BoK raises accuracy from 53.0% (Base) to 71.6% (+18.6%), exceeding TopK (56.2%) by +15.4%; similar high-temperature gains appear on GPQA (+6.06%) and HumanEval (+14.64%). These improvements are robust across a range of (β, λ) choices (coverage vs. KL anchoring), indicating a stable operating region rather than brittle tuning. Finally,BoK’sbenefitscomewithonlymodestcomputeoverhead: using5mirror-ascentsteps

per token adds about 1s on MATH500 (16.88s vs. 15.84s), while even 2 steps already yield a sizeable jump (64.4%→69.6%) with negligible runtime increase—suggesting fast solver convergence and making BoK viable as a lightweight drop-in for practical decoding.

In short, our contributions can be stated as: Summary of Contributions

- 1. Unified Theory of Decoding: We unify decoding strategies into a single framework, proving they are closed-form optima of objectives on the simplex.
- 2. Generative "Master Key" for Decoder Design: A general framework for automatically deriving iterative algorithms for any decoding behaviour expressed as a regulariser, enabling optimisation beyond closed-form solutions.
- 3. Case Study - Best-of-K (BoK) Decoding: We introduce Best-of-K (BoK), a coverage-based objective for self-consistency and reranking. It replaces heuristic sampling with a mathematically grounded approach that improves performance.

## 2 Decoding, Sampling and Optimisation

Our central claim is that decoding strategies are not heuristics, but rather implicit solutions to well-defined optimisation problems. Sampling-based methods do not introduce randomness arbitrarily; instead, they correspond to soft or regularised forms of optimisation. Conversely, deterministic methods such as greedy arise as limiting or approximate solutions to hard optimisation objectives. From this perspective, decoding algorithms differ not in kind, but in what they optimise and the constraints they impose.

###### MASTER OBJECTIVE

###### [⟨q, st⟩ − λΩ(q)]

max

q∈∆(V)

subject to: q ∈ Ct

| | | | |
|---|---|---|---|
| | | | |

###### MODE 01

###### MODE 02

###### MODE 03

###### MODE 04

###### MODE 05

λ = 0

Ω ∝ ∥q∥22

Ω = −H(q)

Support Ct

Ω(BoK)

###### Top-K/P

###### Greedy

###### Softmax

###### Sparsemax

###### BoK

closed form

closed form

closed form

closed form

mirror ascent

Figure 1. Framework of Decoding as Optimisation: The master objective generalises standard LLM decoding strategies. By choosing appropriate λ, Ω(q) and Ct, we can recover current decoding strategies as special cases.

The goal of this section is to make this connection explicit. We begin by formalising decoding as the problem of selecting a distribution over next tokens, rather than directly selecting a token. This shift allows us to express decoding as an optimisation problem over distributions, balancing model score and regularisation. We then show that common decoding strategies emerge naturally as special cases.

###### 2.1 Decoding as a Decision Over Distributions

To make the optimisation perspective precise, we start by formalising what a decoder actually does at a single generation step. Consider a language model with parameters θ. Given a prefix x<t = (x1, . . . , xt−1), the model produces a real-valued score for every token v in the vocabulary V. In practice, these scores are the logits or log-probabilities produced by the model. We denote them by: st(v) ∈ R, for v ∈ V.

At this point, most decoding descriptions implicitly assume that we must immediately choose a single token xt. Instead, in this work, we take a small but crucial conceptual step back. Rather than asking which token to select, we ask:

Which distribution over tokens should we use to make the selection?

Formally,weintroduceanauxiliarydistributionqt(·) ∈ ∆(V)with∆(V)denotingtheprobability simplex over the vocabulary V. Once qt(·) is chosen, decoding proceeds by either:

- • sampling xt ∼ qt(·), or
- • deterministically selecting xt ∈ argmaxv∈Vqt(v).

Importantly, this viewpoint is general enough to cover both stochastic and deterministic decoding, whereby i) greedy decoding corresponds to a degenerate distribution qt(·) that places all its mass on a single token 1, while ii) sampling-based decoding corresponds to a non-degenerate qt(·) with positive entropy. To illustrate this distinction, consider a simple example with a vocabulary of three tokens: V = {a, b, c}. Suppose that, for a given prefix xt, the language model scores each token as follows:

st(a) = 3, st(b) = 2, st(c) = 0. Now, greedy decoding selects the highest-scoring token:

a ∈ argmax

st(v).

v∈V

In our framework, this corresponds to choosing a degenerate distribution qt(·) that places all its probability mass on the maximiser of the score function:

qt(v) =

1, if v ∈ argmax

st(u), 0, otherwise.

u

In the example above, since st(a) > st(b) > st(c), this reduces to: qt(a) = 1, qt(b) = 0, qt(c) = 0.

1Please note we assume a rule that breaks ties.

By contrast, sampling-based decoding selects a non-degenerate distribution that assigns nonzero probability to multiple tokens, while still favouring higher-scoring ones. In our framework, such a distribution is also obtained as a function of the score vector st. Concretely, for a given temperature parameter τ > 0, sampling-based decoding chooses:

exp(st(v)/τ) ∑u∈Vexp(st(u)/τ)

,

qt(v) =

which assigns higher probability to tokens with larger scores, while maintaining positive entropy. In the example above, with scores st(a) = 3, st(b) = 2, and st(c) = 0, this construction yields a distribution of the form:

qt(a) ≈ 0.6, qt(b) ≈ 0.3, qt(c) ≈ 0,

for an appropriate choice of τ. As such, sampling from qt(·) will most often select the highest-scoring token a, but will occasionally produce b or c. This stochasticity allows the decoder to explore alternative continuations while remaining aligned with the model’s scoring function.

Up to this point, common decoding strategies such as greedy decoding and temperature-based sampling may still appear as largely heuristic choices: practical rules that users select and tune based on empirical behaviour or trial-and-error. From this perspective, the decoder is seen as a sampler whose properties, e.g., determinism, diversity, or randomness, are controlled by externally chosen parameters such as temperature. In the next section, we show that this view is incomplete if not misleading. In fact, these decoding strategies are not heuristic procedures, but rather exact solutions to well-defined optimisation problems. Greedy decoding and temperature-based sampling emerge naturally as optimisers of a common objective that trades off expected model score against regularisation. The apparent differences between decoding methods are therefore not algorithmic accidents, but reflect differences in the underlying objectives being optimised.

This perspective enables a shift in how decoding strategies are designed and understood. Ratherthantreatingsamplersasad-hocmechanisms,wecanviewthemassolutions to optimisation problems over distributions. Under this lens, designing a decoder with desired properties such as increased diversity, robustness, or constraint satisfaction reduces to specifying the appropriate optimisation objective and constraints. We now make this connection explicit.

###### 2.2 Decoding as Distributional Optimisation

The discussion so far suggests that decoding can be viewed as a decision over distributions rather than individual tokens. We now make this perspective explicit by introducing a general optimisation formulation that subsumes a wide range of decoding strategies.

Atagivendecodingstept,recallthatthemodelassignsascorest(v)toeachtokenv ∈ V. We seek to choose a distribution qt(·) ∈ ∆(V) from which the next token will be drawn (or whose mode will be selected). We define decoding as the solution to the following optimisation problem:

“MASTER PROBLEM”

|q∗t = arg max<br><br>q∈∆(V)<br><br>⟨q, st⟩ − λ Ω(q) s.t. q ∈ Ct.|
|---|

(1)

In the above equation, ⟨q, st⟩ = ∑v∈Vq(v) st(v), as the expected model’s score under the distribution q(·), and Ω(q) is a regularisation functional that encodes preferences such as diversity, sparsity, or stability. Moreover, λ ≥ 0 controls the strength of regularisation, and Ct denotes constraints on the support of q(·). This optimisation perspective is not restricted to token-level scores: decoding criteria can also be defined over latent spaces or task-structured representations [39]. Similar optimisation-based reinterpretations have also been developed for other core LLM mechanisms (e.g., attention) [23], reinforcing the viewthatstandardneuralprimitivescanoftenbederivedassolutionstoexplicitregularised objectives.

From a geometric perspective, each component of the formulation plays a structural role. The simplex constraints induce boundary behaviour: tokens may become active or inactive depending on whether the optimiser settles in the interior or on a face of the feasible set. Different choices of regulariser shape this geometry in different ways: some encourage interior solutions, while others allow the optimiser to concentrate mass on lowerdimensional faces. Likewise, hard support constraints restrict the feasible region itself, carvingout sub-simplicesbeforeoptimisation evenbegins. AswewillseeinSection3, classical decoding rules correspond to particular geometric choices within this framework.

InterpretingEquation(1). Theobjectiveaboveconsistsoftwocompetingterms. Thefirst term, ⟨q, st⟩, encourages the decoder to place probability mass on tokens with high model scores. If this term were optimised alone, the resulting distribution would collapse onto the highest-scoring token. The second term, Ω(q), acts as a regulariser that penalises certain properties of the distribution q(·). Its role is to control the shape of the decoding distribution, for example, by encouraging diversity or limiting deviation from a reference distribution. The scalar λ determines the trade-off between strictly optimising model score and satisfying these additional preferences. Finally, the constraint set Ct allows the decoder to enforce hard restrictions, such as limiting the support to a subset of tokens or excluding invalid continuations. Under this view, the design of a decoding strategy reduces to:

- 1. What notion of quality is being optimised (through st(v));
- 2. What properties are desired in the decoding distribution (through Ω(q)); and
- 3. What hard constraints must be respected (through Ct).

Deriving Solution Conditions for Equation (1). Deriving the closed-form solution conditions for our general optimisation problem is relatively involved. We present it in steps that make various assumptions about Ct, ∆(q) and Ω(q). In the first case, we consider the absence of Ct. Here, our optimisation problem becomes the following:

q⋆t = arg max

q∈∆(V)

(⟨q, st⟩ − λΩ(q)) , for λ ≥ 0, (2)

where ∆(V) is defined as: ∆(V) = q ∈ R|V| : q(v) ≥ 0, ∑v q(v) = 1 . It is easy to see that maximising the objective in Equation (2) can be equivalently rewritten as a minimisation problem, such that:

(λΩ(q) − ⟨q, st⟩). (3)

q⋆t = arg min

q∈∆(V)

Assuming that Ω(q) is convex2, the minimisation problem we just derived (Equation (3)) is simply a “convex - linear” minimisation objective for which clean optimality conditions are easily attainable. What makes solving the problem in Equation (3) hard is the existence of the constraint set ∆(V). If we were to ignore those constraints, we could just set the derivative to zero, like in unconstrained calculus, and proceed. However, the existence of those constraints means we must be more careful: we are minimising over a restricted set, so we should take those into account.

But what do those constraints actually mean?

Our constraints in ∆(V) are not fancy at all. They are pretty simple! All they do is make q(·) a valid probability distribution. In detail, when we say q ∈ V, we really mean the following two properties should hold:

- • ∑v q(v) = 1 → Forcing q(·) to be a probability distribution that sums to 1 across all realisations of the random variable; and
- • q(v) ≥ 0 for all v ∈ V→ You can’t have negative q(·) values on any v ∈ V.

- As is tradition in mathematical derivations, we begin by pretending the inconvenient part does not exist, ignore the second condition (q(v) ≥ 0), and come back to it later. So pretend for a moment that q(v) can be any real numbers as long as they sum to 1. Here, our optimisation problem becomes:

J(q) ≡ λΩ(q)−∑

q(v)st(v), s.t.∑

q(v) = 1.

min

q

v

v

Now, we really, really want to take derivatives and set them to zero. Sadly, there is a constraint! A standard trick is: add the constraint into the objective with a “price” η. This gives us what every constrained optimisation problem eventually produces, the Lagrangian:

(3)

L(q,η) = λΩ(q)(1) −∑

q(v)st(v)(2) +

##### η ∑

.

###### q(v) − 1

v

v

The Constraint Penalty

We now have an unconstrained objective, which allows us to deploy one of the great discoveries of the last century: the gradient. Let us do that! Let us take the partial derivatives of L(q, η) with respect to q(v). Doing so, term-by-term, we get:

∂ ∂q(v)

λΩ(q) = λ

(1) →

∂ ∂q(v)

∂

∂q(v) −∑

Ω(q)(v), (2) →

###### q(v)st(v) = −st(v),

v

∂ ∂q(v)

##### η(∑

(3) →

###### q(v) − 1) = η.

v

2The convexity of Ω(q) is typical. As we show later, many special cases of LLM decoding strategies stem from different convex choices of Ω(q).

So on the stationary point q⋆t , we get this condition: λ

∂ ∂q(v)

∂ ∂q(v)

Ω(q⋆t )(v) = η, ∀v ∈ V. (4) It is easy“ish” to interpret the condition in Equation (4). It simply says that for every token v, the quantity st(v) − λ∂q∂(v)Ω(q⋆t )(v) must be the same constant η. This is a “balancing” condition created by the fact that probabilities must sum to 1.

###### Ωq(v)(q⋆t )(v) − st(v) + η = 0 =⇒ st(v) − λ

Getting the Inequality Back. At this stage, we have derived a general condition, but we should not forget about our previous promise! We still have to enforce q(v) ≥ 0 because some solutions to Equation (4) can give negative q⋆t (v).

One Dimensional Example for Handling Inequality Constraints For now, let us forget about tokens and consider a single-variable objective:

min

f(x).

x≥0

An optimal x⋆ means that no feasible small move from x⋆ can reduce the function value. A move is feasible if it respects the constraint x ≥ 0. Thus: i) if x⋆ > 0 (an interior point), we can move a small amount both to the right and to the left, i.e., to x⋆ + δ and x⋆ − δ, for sufficiently small δ > 0; ii) if x⋆ = 0 (a boundary point), we can only move to the right (to 0 + δ), since 0 − δ would violate x ≥ 0. To make this precise, we use a first-order Taylor approximation. For small δ > 0,

###### f(x⋆ + δ) ≈ f(x⋆) + δf′(x⋆), f(x⋆ − δ) ≈ f(x⋆) − δf′(x⋆).

Since x⋆ is optimal, any feasible move must not decrease f, i.e.,

f(x⋆ + δ) − f(x⋆) ≥ 0 for all sufficiently small feasible δ > 0, and, when x⋆ > 0 (so that x⋆ − δ is also feasible),

f(x⋆ − δ) − f(x⋆) ≥ 0 for all sufficiently small δ > 0. Combining these inequalities with the Taylor expansions gives the two cases below:

- • Case I (Interior optimum x⋆ > 0): both x⋆ + δ and x⋆ − δ are feasible. Hence δf′(x⋆) ≥ 0 and − δf′(x⋆) ≥ 0 ∀δ > 0,

which implies f′(x⋆) = 0.

- • Case II (Boundary optimum x⋆ = 0): only 0 + δ is feasible. Hence δf′(0) ≥ 0 ∀δ > 0,

which implies f′(0) ≥ 0.

In other words, these conditions reflect whether x⋆ lies in the interior or on the boundary of the feasible set. If x⋆ > 0, we can perturb x⋆ both to the right and to the left while remaining feasible, and optimality therefore forces the slope to vanish, i.e., f′(x⋆) = 0. If x⋆ = 0, we can only move “into” the feasible region (to the right), sooptimalityonlyrequiresthatrightwardperturbationsdonotdecrease f(·), equivalently f′(0) ≥ 0 (a one-sided slope condition).

Havingunderstoodwhatconditionswewouldneedwhenhavinganon-negativityfeasibility set, we can now go back to our tokens and apply the same reasoning coordinate-wise to each component q(v) ≥ 0. In our problem, the role of f′(x⋆) is played by the partial derivative of the Lagrangian with respect to the coordinate q(v). Recall the Lagrangian:

L(q,η) = λΩ(q)−⟨q,st⟩+η ∑

q(u) − 1 ,

u∈V

where the scalar η enforces the normalisation constraint ∑u q(u) = 1. Taking the partial derivative with respect to q(v) yields:

∂ ∂q(v)

∂ ∂q(v)

###### Ω(q)(v) − st(v) + η.

L(q, η) = λ

By the same one-dimensional argument as above, optimality under the constraint q(v) ≥ 0 implies a two-case condition:

 

q⋆(v) > 0 ⇒ ∂q∂(v)L(q⋆, η) = 0, q⋆(v) = 0 ⇒ ∂q∂(v)L(q⋆, η) ≥ 0.



Substituting the expression for ∂q∂(v)L(·, ·) and rearranging gives us two conditions for the optimality of the solution, which we state below.

###### Take Home Message: KKT Optimality Conditions

From the above derivations, the take-home message is that we can characterise an optimal q⋆t using the following identities:

GLOBAL ∑

q⋆t (v) = 1, q⋆t (v) ≥ 0 ∀v ∈ V.

v∈V

∂ ∂q(v)

ACTIVE q⋆t (v) > 0 =⇒ st(v) − λ

Ω(q⋆t )(v) = η,

(5)

∂ ∂q(v)

INACTIVE q⋆t (v) = 0 =⇒ st(v) − λ

Ω(q⋆t )(v) ≤ η.

## 3 LLM Decoding Strategies are Different Regularisers

- At this point, the reader may reasonably wonder what all of the preceding mathematics was for. We now reap what we have sown (Galatians 6:7). In this section, we show that many widely used decoding strategies arise as simple special cases of our general formulation. Those will correspond to different choices of regularisation Ω(q), λ and Ct.

###### 3.1 Greedy Decoding: The Boring but Necessary Case

We begin with the simplest possible case, which serves mainly to reassure us that nothing has gone terribly wrong. To recover greedy decoding, we assume there is no regulariser by setting λ = 0. We also ignore Ct for the time being. As such, our “Master” objective from Equation (1) quietly collapses into a far more modest one:

∑

q⋆t = arg max

⟨q, st⟩ ≡ arg max

q(v)st(v).

q∈∆(V)

q∈∆(V)

v

We now “cash in” the take-home KKT-style conditions from Equation (5). In this greedy case, we have for some scalar η:

ACTIVE q⋆t (v) > 0 ⇒ st(v) = η, INACTIVE q⋆t (v) = 0 ⇒ st(v) ≤ η. Thesetwolinesalreadycharacterisethesolution: everytokenthatreceivesanonzeroprobability must have exactly the same score, and every token with zero probability must have a score no larger. Let M := maxu∈Vst(u) and define the argmax set

V⋆ := argmax u∈V

st(u) = {v ∈ V: st(v) = M}.

The inactive condition forces η ≥ st(v) for all v, hence η ≥ M. But the active condition says there exists at least one active token (since ∑v q⋆t (v) = 1), and any active token must satisfy st(v) = η, hence η must equal the maximum score: η = M. Therefore, the only tokens that can be active are those in V⋆, i.e.,

q⋆t (v) > 0 ⇒ v ∈ V⋆, equivalently supp(q⋆t ) ⊆ V⋆. In words: an optimal solution places all probability mass on the highest-scoring token(s). If the maximiser is unique, V⋆ = {v⋆}, then the unique optimum is the degenerate distribution

q⋆t (v) = δv⋆(v), where v⋆ ∈ argmax u∈V

st(u).

If there are ties (i.e., |V⋆| > 1), then any distribution supported on V⋆ is optimal. In other words, greedy decoding corresponds to selecting a vertex of this optimal face, i.e., picking one v⋆ ∈ V⋆ and using q⋆t = δv⋆.

###### 3.2 From Negative Entropy to Softmax: A Predictable Ending

Let us now consider the case where, in a move that will surprise no one, Ω(q) is chosen to be the negative entropy of q, such that: Ω(q) = ∑v∈Vq(v) log q(v). This will make our optimisation objective from Equation (1) look like the following:

q⋆t = arg max

q∈∆(V)

##### ∑

q(v)st(v)−λ∑

q(v) log q(v) .

v

v

Returning to the conditions in Equation (5), the suspense is, of course, which case applies: active or inactive? This hinges on a single question: does q⋆t (·) live comfortably in the interior of the simplex, or does it press up against the boundary?

###### The Key Mathematical Fact: The Derivative Blows Up at Zero

Tounderstandthis, letustakeastepbackandlookatthederivativeof f(x) = x log x. Why this f(x)? That is how the entropy looks! For x > 0, we can see that:

d dx

d dx

d dx

(x log x) =

x × log x +

log x × x

1 x × x

= 1 × log x +

= 1 + log x.

We now examine what happens as x approaches the mysterious value 0+: lim

1 + log x → −∞,

x→0+

because as x → 0+, log x → −∞, and thus 1 + log x → −∞. In other words, as our variable approaches 0+, the gradients blowup. Therefore, if we were optimising for x, every time we get closer to zero, the gradient politely but firmly tells us to move.

Equipped with this realisation, we return to our case that contains the negative entropy regulariserΩ(q),whichstronglydiscourageszeroprobabilities. Conveniently,thegradients make this preference explicit:

∂ ∂q(v)

Ω(q) = 1 + log q(v), (6)

which blows up as we q(v) approaches zero. Hence, our winner is ... the ACTIVE case:

∂ ∂q(v)

Ω(q⋆t )(v) = η.

ACTIVE q⋆t (v) > 0 =⇒ st(v) − λ

Let us do some algebra! Replacing the derivative from Equation 6 in our condition, we get: st(v) − λ(1 + log q⋆t (v)) − η = 0 =⇒ λ(1 + log q⋆t (v)) = st(v) − η

st(v) − η λ − 1

=⇒ log q⋆t (v) =

st(v) − η λ − 1

=⇒ q⋆t (v) = exp

st(v) λ

= C exp

,

with the constant C = e−1e−λη . We know that q⋆t (v) must be a valid probability distribution,

and valid probability distributions insist on summing to one. Let us indulge them:

st(u) λ

st(u) λ

#### q⋆t(u) = 1 =⇒ ∑

##### ∑

#### = 1 =⇒ C ∑

C exp

exp

= 1

u∈V

u∈V

u∈V

1 ∑u∈Vexp st(λu)

.

=⇒ C =

Therefore, our optimal distribution q⋆t (·) if we picked Ω(q) to be the negative entropy regulariser, ends up being:

exp (st(v)/λ) ∑u∈Vexp (st(u)/λ)

SOFTMAX DECODERS: q⋆t (v) =

.

We thus recover the classical softmax decoder directly from our master optimisation problem. Inparticular, choosingthe(negative)Shannonentropyastheregulariserproducesthe familiar temperature-controlled distribution, with λ playing exactly the role of the temperature τ in the LLM literature.

This demonstrates that softmax decoding is not an independent heuristic, but the optimiser of our abstract objective under an entropic geometry. In this sense, our framework elevates decoding to a higher-level problem definition: different decoders arise from different regularisers. Decoder design becomes regulariser design.

###### 3.3 Trimming the Vocabulary: Top-K Samplers

Fortunately, extending the results derived above to Top-K samplers requires only minimal additional work. Until now, we have been pretending that the constraint Ct in Equation 1 does not exist. For Top-K, this pretence is no longer sustainable. First, we define the indices of the k highest-scoring tokens. We define the set Vk ⊂ Vsuch that: i) |Vk| = k, and ii) ∀v ∈ Vk and ∀u ∈/ Vk, st(v) ≥ st(u). This allows the constraint set Ct(k) to be a subset of the simplex ∆(V) where tokens outside the top-k are forced to have zero probability:

Ct(k) = {q ∈ ∆(V) : q(v) = 0, ∀v ∈/ Vk}.

To ensure we end up with a Softmax over the Top-K candidates, as before, we use the

(negative) Shannon entropy as our regulariser Ω(q) = ∑v∈Vq(v) log q(v). With this, the objective becomes:

q(v)st(v)−λ ∑

##### ∑

q∗t = arg max q∈Ct(k)

q(v) log q(v) .

v∈V

v∈V

Because q(v) = 0 for all v ∈/ Vk per our constraint, the summation effectively collapses to the indices in Vk:

q(v)st(v)−λ ∑

##### ∑

q∗t = arg max

q(v) log q(v) .

q∈∆(Vk)

v∈Vk

v∈Vk

Following a similar derivation to that in Section 3.2, we arrive at Top-K samplers giving us:

 

exp(st(v)/λ) ∑u∈Vk exp(st(u)/λ) if v ∈ Vk, 0 otherwise.

TOP-K DECODERS: q⋆t (v) =



###### 3.4 Mind the Mass: Top-P Sampling

Top-P (nucleus) sampling adapts the size of the active support set based on the model’s confidence in the current context. When the model’s distribution is flat (high uncertainty), the nucleus expands to preserve diversity; when it is peaky (high confidence), the nucleus contracts, filtering out low-probability noise in the long tail.

In our optimisation view, Top-P is obtained by replacing the fixed-cardinality support constraint of Top-K with a cumulative-mass support constraint. Rather than restricting the decoder to exactly K tokens irrespective of confidence, Top-P restricts the decoder to the smallestsetoftokenswhosemodel-assignedprobabilitymass exceedsathreshold p ∈ (0,1].

Defining the nucleus. Let pt(·) denote the model distribution over next tokens induced by the scores st. Sort tokens in descending order of pt: pt(v(1)) ≥ pt(v(2)) ≥ · · · . Define m to be the smallest index such that the cumulative mass reaches p, i.e., ∑im=1 pt v(i) ≥ p, and set the nucleus to be3:

Vp := {v(1), . . . , v(m)}.

This induces the context-dependent constraint set Ct(p) := {q ∈ ∆(V) : q(v) = 0, ∀v ∈/ Vp}, which is a sub-simplex of ∆(V) supported on Vp. To recover the standard Top-P sampler, we again use the (negative) Shannon entropy regulariser, and solve the master objec-

tive restricted to Ct(p):

q⋆t = arg max

⟨q, st⟩ − λΩ(q) .

q∈Ct(p)

Because q(v) = 0 for all v ∈/ Vp, this optimisation is equivalent to optimising over the simplex ∆(Vp):

∑

q(v)st(v)−λ ∑

q⋆t = arg max

q(v) log q(v) .

q∈∆(Vp)

v∈Vp

v∈Vp

The derivation in Section 3.2 therefore applies verbatim, yielding a softmax distribution renormalised over the nucleus:

 

exp(st(v)/λ) ∑ u∈Vp

if v ∈ Vp,

exp(st(u)/λ)

TOP-P DECODERS: q⋆t (v) =



0 otherwise.

In words, Top-P sampling first selects a context-dependent nucleus Vp based on cumulative model probability mass, and then samples from a temperature-controlled softmax restricted to that nucleus.

###### 3.5 Letting Probabilities Go to Zero: Sparsemax Decoding

While standard Softmax sampling is the standard choice for text generation, it suffers from the “heavy tail” problem: because the derivative of Shannon entropy approaches −∞ as any probability q(v) approaches zero, the optimiser is strictly forbidden from reaching the

3If ties occur at the cutoff, we may break them arbitrarily.

boundary of the simplex. This forces the model to assign a non-zero, albeit small, probability to every single token in the vocabulary, which can lead to the “tail risk” of sampling nonsensical or hallucinatory tokens.

To solve this without the ad-hoc truncation rules of Top-K or Top-P, we look toward Sparsemax [25]. By replacing the logarithmic penalty of entropy with a quadratic penalty, we allow the optimisation to reach the simplex boundary, effectively performing an automated, adaptive truncation that assigns exactly zero probability to low-scoring candidates. We again consider a single decoding step with an empty Ct and solve the “Master problem” again. In this special case, we choose the quadratic regulariser:

- 1

- 2 ∑

- 1

- 2∥q∥22 = v∈V

λ 2 ||q||22 . (7)

q(v)2, ⇒ q⋆t = arg max

Ω(q) =

⟨q, st⟩ −

q∈∆(V)

Please note that in our problem, both conditions in Equation (5) need to be considered. This is the case since Ω(q) = 12||q||22 allows solutions that are positive but can also equate to zero. Of course, the latter condition is what enables sparsity. Let us first consider the active tokens case, i.e., when q⋆t (v) > 0. For a token v ∈ V, the active condition can be written as:

- 1

- 2 ∑

∂ ∂q(v)

∂ ∂q(v)

q⋆t (v)2 = η

Ω(q⋆t )(v) = η =⇒ st(v) − λ

ACTIVE st(v) − λ

v∈V

###### =⇒ st(v) − λq⋆t (v) = η.

Solvingforq⋆(v), weget: q⋆t (v) = λ1(st(v) − η). Alsonoticethatsinceweareintheq⋆t (v) > 0, we can, thus, clearly see that st(v) > η.

Moving on to the second condition, we observe the following:

INACTIVE st(v) − λ

∂ ∂q(v)

###### Ω(q⋆t )(v) ≤ η =⇒ st(v) − λq⋆t (v) ≤ η.

Sincethisconditionholdsforatokeninwhichq⋆t (v) = 0,thenwecanconcludethat: st(v) ≤ η. So, essentially, from those two conditions we learned that:

 

st(v) − η λ

if st(v) > η 0 if st(v) ≤ η.

1 λ

=⇒ q⋆t (v) =

q⋆t (v) =

###### [st(v) − η]+,



with [x]+ = max(0, x) = ReLU(x). The last remaining ingredient needed to finalise our derivation is determining η. To do so, we make use of the constraint that ∑v∈Vq⋆t (v) = 1:

1 λ

#### q⋆t(v) = 1 =⇒ ∑

##### ∑

#### [st(v)−η]+ = 1 =⇒ ∑

[st(v) − η]+ = λ

v∈V

v∈V

v∈V

=⇒ ∑

max(0, st − η) = λ.

v∈V

In other words, we need η such that the sum of the positive parts equals λ. As we see, it is challenging to acquire a closed-form solution for η. Therefore, we develop an algorithmic solution. The key idea is that we will assume we know which tokens are active by defining

an active set S(η) = {v ∈ V : st(v) > η}. Then, we have: i) If v ∈ S(η), then [st(v) − η]+ = st(v) − η; and ii) If v ∈/ S(η), then [st(v) − η]+ = 0. Therefore, those conditions over η simplify to a sum over the active set S(η), giving us: ∑v∈V(st(v) − η) = λ. Unfortunately, this result is still “implicit” because S(η) depends on η itself. So, we apply the standard trick of guessing the active set size k.

The key observation is that the active tokens are exactly those satisfying st(v) > η. Consequently, if a token with some score st(v) is active, then every token with a larger score must also be active. Therefore, the support of q⋆t (·) must be a Top-K set for some k.

Lets(t1) ≥ s(t2) ≥ · · · ≥ s(tn) denotethescoressortedindescendingorder,wheren := |V|. Assume that the active set has size k and is given by the top-k indices:

Sk(η) = {(1), (2), . . . , (k)}.

Under this hypothesis, the normalisation condition becomes:

k

1 λ

s(ti) − η = 1,

∑

##### ∑

q⋆t (v) =

i=1

v∈S(η)

since tokens outside Sk(η) contribute zero after the [·]+ clipping. Defining the prefix sum: Ak := ∑ik=1 s(ti), we obtain:

1 λ

Ak − λ k

Ak − kη =⇒ Ak − kη = λ =⇒ ηk :=

.

1 =

Importantly, ηk is not an arbitrary choice: it is the unique threshold value that would make ∑v q⋆t (v) = 1if theactivesetwereexactlythetopk tokens. Theremainingquestioniswhich k is self-consistent. The Top-K hypothesis holds precisely when the computed threshold ηk separates the top k scores from the rest, i.e., s(tk) > ηk and s(tk+1) ≤ ηk, with the convention s(tn+1) = −∞. We select any k⋆ satisfying the above inequalities, set η = ηk⋆, and then recover the optimal distribution via:

SPARSEMAX-STYLE DECODERS: q⋆t (v) =

1 λ

[st(v) − η]+.

## 4 Going Beyond Current Decoders

So far, our story has been pleasantly “closed-form”. Once we write decoding as a regularised optimisation problem on the simplex, the KKT conditions act like a master key, and classical decoders fall out as special cases. In a complementary direction, prior work has shown that search-based decoding methods (e.g., beam search) can also be reinterpreted as optimising an explicit regularised objective [26], and recent work on controlled andalignment-orienteddecodingalsoconstructsinference-timepoliciesbyoptimisingexplicit regularised objectives (often KL-anchored to the base model) rather than relying on heuristic sampling rules [28, 7]. But this view raises an immediate practical question: what do we do when the optimiser is no longer solvable in one line? The moment we introduce richerregularisers,couplingterms,coverageobjectives,ormorestructuredconstraints,the

distributionq⋆t (·)isstillwell-defined—butwecannotalwayswriteitdownanalytically. This section switches from deriving decoders to computing them. We introduce mirror descent

(and mirror ascent) as a principled method tailored to simplex geometry. It preserves nonnegativity and normalisation by construction and provides an algorithmic template that solves our master problem even when closed-form solutions are unavailable.

###### 4.1 Why not just run Vanilla (projected) Gradient Ascent on q(·)?

If we look back at our master problem, it is natural to try solving it with a standard tool from constrained optimisation: projected gradient ascent [3, 4]. After all, we are maximising a differentiable objective over a convex set (the simplex, possibly intersected with additional constraints). Onecouldtakeagradientstepandthenprojectbackonto∆(V),repeatinguntil convergence. In practice, this approach is indeed feasible, and it is often the first method people reach for. To make this explicit, we write the following optimisation problem and update rule4:

“MASTER PROBLEM” max

q∈∆(V)

“PROJECTED GRADIENTS”

f(q), with f(q) = ⟨q, st⟩ − λ Ω(q),

|qj+1 = ∏<br><br>∆(V)<br><br>qj + η∇f(qj) ,|
|---|

with ∏∆(V) denoting the projection onto the simplex.

What do projected gradients solve? We note that the projected ascent step above is also somewhat misguided as it implicitly equips the simplex with an L2 geometry that does not match the way probability distributions behave, especially near the boundary where many tokens should receive exactly zero (or near-zero) mass. This mismatch can lead to unstable updates, overly aggressive redistribution of mass, and “fightingthe constraints”at every iteration. This is not a philosophy! We can explicitly characterise this property by understanding that projected gradients are the solution to the following optimisation problem:

- 1

- 2η

q − qj 22 . (8)

qj+1 = arg max

⟨∇f(qj), q − qj⟩ −

q∈∆(V)

Let us see why! We begin by expanding the terms in Equation (8):

- 1

- 2η ||q||22 − 2⟨q, qj⟩ + ||qj||22

∇f(qj)T(q − qj) −

qj+1 = arg max

q∈∆(V)

- 1

- 2η ||q||22 − 2⟨q, qj⟩ + ||qj||22

∇f(qj)Tq − ∇f(qj)Tqj −

= arg max

q∈∆(V)

- 1

- 2η||qj||22 .

- 1

- 2η ||q||22 − 2⟨q, qj⟩ +

∇f(qj)Tq − ∇f(qj)Tqj −

= arg max

q∈∆(V)

4Pleasenoticethatwehaveignored Ct fornow. Wecaneasilyincorporateitagainbydefiningasub-simplex depending on the ∆(V) ⊆ V∩ Ct.

Notice from the above that ∇f(qj)Tqj and ||qj||22 are independent from the optimisation variable q. Removing those and multiplying by η, we get:

- 1

- 2||q||22 + ⟨q, qj⟩

η∇f(qj)Tq −

qj+1 = arg max

q∈∆(V)

- 1

- 2||q||22 .

= arg max

⟨q, qj + η∇f(qj)⟩ −

q∈∆(V)

If we call y = qj + η∇f(qj), we get the following optimisation problem:

- 1

- 2||q||22 − ⟨q, y⟩ .

qj+1 = arg min

q∈∆(V)

Since y is independent of q, we can add 12||y||22 to our minimisation problem to complete the squares, giving us:

- 1

- 2||y||22 = arg min

- 1

- 2||q||22 − ⟨q, y⟩ +

- 1

- 2 ||q − y||22 .

qj+1 = arg min

q∈∆(V)

q∈∆(V)

From the last line, we see that the projected-gradient update is exactly a Euclidean projection:

1 2||q − y||22.

##### qj+1 = ∏

(y), where ∏

(y) = arg min

q∈∆(V)

∆(V)

∆(V)

In other words, projected gradient ascent produces the next iterate by choosing the distribution q on the simplex that is closest to the unconstrained step y in squared L2 distance. This reveals an important (and often overlooked) fact: projected gradient ascent is not “geometry neutral”: it is the solution to a proximal subproblem with an L2 regulariser, i.e., it implicitly assumes Euclidean geometry. While this is a reasonable choice in Rn, it is a poor match for probability distributions on the simplex, which form a constrained manifold whose natural notion of distance is typically divergence-like (e.g., KL) rather than Euclidean. This mismatch is precisely what mirror descent corrects by replacing the squared L2 proximity term with a Bregman divergence that respects the simplex geometry.

###### 4.2 Bregman Divergences & Mirror Ascent

If decoding lives on the simplex, then using L2 geometry is like doing navigation on the Earth with a flat map: it works locally, but it distorts what “small” moves mean. Mirror ascent [2, 29, 34] replaces this distortion with the right geometry. Instead of measuring steps by squared distance, it measures them by a Bregman divergence Dψ(q, qj), induced by a convex potential ψ [5]. The Bregman divergence is defined as:

Dψ(q, qj) = ψ(q) − ψ(qj) − ⟨∇ψ(qj), q − qj, ⟩ forastrictlyconvexanddifferentiablefunctionψ : ∆(V) → R, whichiscalledthedistancegenerating (or potential) function. Given Dψ(q, qj), we rewrite the problem in (8) as a Bregman regularised one:

1 η

Dψ(q, qj) . (9)

qj+1 = arg max

⟨∇f(qj), q − qj⟩ −

q∈∆(V)

The L2 Special Case It is interesting to see that if we choose ψ(q) to be the Euclidean norm, i.e., ψ(q) =

- 1

- 2||q||22, we would recover Equation (8) as a special case of the generalised problem in Equation (9). To show that, we begin by understanding the Bregman divergence:

- 1

- 2||q||22 −

1 2||qj||22 − ⟨qj, q − qj⟩ =

1 2||q − qj||22.

Dψ(q, qj) =

If we replace this last result into Equation (9), we exactly obtain Equation (8), which we have shown corresponds to the optimisation problem solved with projected gradient ascent. This goes to say that the problem in Equation (8) allows us to use a more general notion of divergences defined through ψ, giving us the ability to consider manifolds (e.g., simplexes) that go beyond Euclidean spaces.

Entropic Distance-Generating Functions over the Simplex. When optimising over a simplex [19], the standard choice for the distance-generating function is the entropic potential: ψ(q) = ∑in=1 q(i) log q(i). In here, the resulting Bregman divergence becomes the Kullback-Leibler (KL) divergence. Therefore, the problem in Equation (9) becomes:

η⟨∇f(qj), q⟩ − KL(q||qj) . (10)

qj+1 = arg max

q∈∆(V)

Now, let us solve the problem in Equation (10) to get the final form of our update. Noticing that we have a constraint ∑in=1 q(i) = 1 and ignoring the positivity constraint 5 on q(i), the Lagrangian is defined as:

n

##### ∑

L(q, ν) = η

i=1

∂f(qj) ∂q(i)

n

q(i) q(ji)

q(i) −

q(i) log

##### ∑

i=1

n

q(i) ,

##### ∑

+ ν 1 −

i=1

with ν being the Lagrange multipliers. Consider the partial derivative of L(q, ν) for a single component q(i) and set it to zero:

∂f(q) ∂q(i) − (log q(i) + 1 − log q(ji)) − ν = 0

∂L ∂q(i)

= η

  = η

 q(i)

∂f(qj) ∂q(i) − (1 + ν)

=⇒ log

q(ji)

q(i) q(ji)

###### ∂f(qj) ∂q(i) × exp (−(1 + ν))

= exp η

=⇒

∂f(qj) ∂q(i)

=⇒ q(i) = Cq(ji) exp η

,

5The constraint has been ignored since the log function naturally enforces positivity.

with C = exp (−(1 + ν)). To find this constant, we use the constraint that the sum ∑in=1 q(i) = 1:

n

q(ji) exp η

##### ∑

C

i=1

∂f(qj) ∂q(i)

1 ∑nl=1 q(jl) exp η ∂∂fq((ql)j)

.

= 1 =⇒ C =

Hence, the overall update for the ith component of q at round j + 1 amounts to:

q(ji) exp η ∂∂fq((qi)j) ∑nl=1 q(jl) exp η ∂∂fq((ql)j)

q(j+i)1 =

. (11)

To write in a vectorised form that is amenable to implementation, we define the following

- as the vector of partial derivatives:

gj =

∂f(qj) ∂q(n)

∂f(qj) ∂q(1)

, . . . ,

T

. (12)

The numerator of Equation (11) is thus:

Numerator = qj ⊙ exp(ηgj),

with ⊙ being the Hadamard or element-wise product. The denominator is simply the L1 norm of the numerator: Denominator = ||qj ⊙ exp(ηgj)||1. Therefore:

Numerator Denominator

qj ⊙ exp(ηgj) ||qj ⊙ exp(ηgj)||1

. (13)

“MIRROR ASCENT UPDATE” qj+1 =

=

In practice, computing exp(η∂∂q(fi)) directly can lead to numerical overflow if the gradients are large. To prevent this, we use the Log-Sum-Exp trick by subtracting the maximum value M = max(ηgj)fromtheexponents. Becausethesumnormalisestheupdate,thisshift cancels out mathematically but ensures the computer always handles values ≤ 1.

###### 4.3 Use Case: Best-of-K Samplers

We have done a lot of groundwork: we reframed decoding as optimisation on the simplex, re-derived several classical decoders as closed-form special cases, and introduced mirror ascent as a practical solver when closed forms are unavailable. The point of all this was not just unification for its own sake. It was to argue that an optimisation view gives us a principled design language for decoding: samplers are best understood as regularisers (and constraints), and new decoding behaviour should be obtained by writing down the objective we actually want to optimise. To make good on this claim, we now present a concrete use case. We introduce a new regulariser, Best-of-K (BoK), designed for settings where we draw multiple samples and care about coverage of high-quality alternatives, not just the properties of a single draw. BoK drops into our master problem as a plug-in term, and mirror ascent gives an immediate, implementable algorithm.

The motivation is simple: many modern pipelines do not sample once; they sample K times (self-consistency, rejection sampling, reranking, verifier-based selection, etc.). In that regime, the decoder is no longer judged by the quality of a single draw, but by what the set of K draws contains. Standard decoders were not designed for this: they often waste budget by repeatedly sampling the same high-probability continuation, while allocating too little mass to plausible alternatives that are individually unlikely but collectively valuablewhenwehavemultipletries. Whatwewantinsteadisadecodingdistributionq⋆t (·)that makes good options show up at least once within K samples, i.e., that explicitly trades off model score against multi-sample coverage. This is exactly the behaviour our framework is built to express: we define a coverage utility for K draws, convert it into a regulariser, and plug it into the master objective to obtain the Best-of-K (BoK) sampler.

The hit probability. Fix a token v ∈ Vand suppose we draw K i.i.d. samples from a decoding distribution q(·). The probability that we never sample v is (1 − q(v))K. Therefore, the probability that v appears at least once among the K samples is:

Pr[v appears at least once in K samples] = 1 − (1 − q(v))K. (14)

This quantity captures what single-sample decoding objectives ignore: in a multi-sample regime, even moderately small probabilities can become valuable if they increase the chance of seeing a useful alternative at least once.

Summing Equation (14) over all tokens yields a notion of total coverage. However, we do not want to reward covering junk tokens equally with high-quality ones, so we introduce nonnegative importance weights wt(v) ≥ 0 and define the weighted K-coverage:

wt(v) 1 − (1 − q(v))K . (15)

##### UK,t(q) := ∑

v∈V

Intuitively, UK,t(q) is large when: i) We allocate mass to many tokens that we care about (large wt(v)), and ii) this mass is sufficient for them to be hit at least once within K samples. In practice, wt(v) can be any nonnegative proxy for “how much we would like to see token v among K draws” (e.g., a monotone function of the model score st(v), a Top-M indicator, or a softened rank-based weight).

A useful property of the hit probability 1 − (1 − q(v))K is that it exhibits diminishing returns: its marginal gain decreases as q(v) grows. Indeed:

∂ ∂q(v)

###### 1 − (1 − q(v))K = K(1 − q(v))K−1,

which is strictly decreasing in q(v) for K > 1. Thus, BoK has an inherent “anti-collapse” bias: itismorerewardingtoallocateprobabilitymasstounder-coveredbutvaluabletokens than to keep increasing the mass of tokens that are already likely to appear. This is exactly the multi-sample behaviour we want.

BoK as a regulariser. Maximising UK,t(q) alone would encourage spreading probability mass too widely, including onto implausible tokens. To keep the decoder anchored to the

model, wecombinecoveragewithaKLtrustregionaroundthemodeldistribution pt(·)and define the BoK regulariser

Ω(tBoK)(q) := KL(q∥pt) − β UK,t(q), (16) where β ≥ 0 controls how strongly we reward coverage.

Plugging Equation (16) into our master problem yields the BoK decoding objective:

q⋆t = arg max

q∈∆(V)

= arg max

q∈∆(V)

with β¯ := λβ for convenience.

⟨q, st⟩ − λ Ω(tBoK)(q)

⟨q, st⟩ − λ KL(q∥pt) + β¯ UK,t(q) , (17)

Mirror-Ascent Update for Ω(BoK). Unfortunately, exact closed-forms are hard to come by for Ω(BoK). Thus, we make use of our mirror ascent update rule in Equation 13. For a single decoding step, BoK-regularised objective is defined as:

f(q) = ⟨q, st⟩ − λ KL(q∥pt) + β¯ UK,t(q), (18) where q ∈ ∆(V), pt ∈ ∆(V) is the reference/model distribution, and

wt(v(i)) 1 − (1 − q(i))K . (19)

##### UK,t(q) = ∑

v∈V

Define the vector of partial derivatives at iterate qj as

###### gj =

∂f(qj) ∂q(|V|)

∂f(qj) ∂q(1)

, . . . ,

###### T

. (20)

For each coordinate i ∈ {1, . . . , n}, the partial derivative admits the closed form (we denote wt(i) = wt(v(i)))

######  log

  + β¯ wt(i) K (1 − q(ji))K−1. (21)

q(ji) p(ti)

∂f(qj) ∂q(i)

###### = s(ti) − λ

###### + 1

Substituting Equation (21) into the mirror-ascent update from Equation (11) yields the explicit BoK update:

(i) j

q(ji) exp η s(ti) − λ log q

+ 1 + β¯ wt(i)K(1 − q(ji))K−1

p(ti)

q(j+i)1 =

. (22)

(l) j

∑nl=1 q(jl) exp η s(tl) − λ log q

+ 1 + β¯ wt(l)K(1 − q(jl))K−1

p(tl)

Algorithm 1 BoK Decoder via Mirror Ascent (one decoding step) Require: scores st ∈ Rn, reference distribution pt ∈ ∆n, weights wt ∈ Rn≥0 Require: hyperparameters K ∈ N, λ ≥ 0, β¯ ≥ 0, stepsize η > 0, iterations J ∈ N

- 1: Initialise q0 ← pt
- 2: for j = 0,1, . . . , J − 1 do
- 3: for i = 1,2, . . . , n do
- 4: g(ji) ← s(ti) − λ log q

(i) j

p(ti)

+ 1 + β¯ wt(i)K(1 − q(ji))K−1

- 5: end for
- 6: M ← max(ηgj) ▷ Log-Sum-Exp stabilisation
- 7: q˜j+1 ← qj ⊙ exp(ηgj − M1)
- 8: qj+1 ← q˜j+1/∥q˜j+1∥1
- 9: end for
- 10: return qJ

Of course, with gj as in Equation (20), the update can be written compactly as:

qj ⊙ exp(ηgj) ∥qj ⊙ exp(ηgj)∥1

. (23)

qj+1 =

Algorithm 1 summarises how to compute the BoK decoding distribution at a single time step. We initialise the iterate with the model distribution q0 = pt (a natural warm-start), then perform J mirror-ascent steps. Each step computes the vector of partial derivatives gj using the closed form in Equation (21), and applies the multiplicative update in Equation (23), and renormalises to stay on the simplex. In practice, we implement the update with a Log-Sum-Exp stabilisation by subtracting M = max(ηgj) before exponentiating.

###### 4.4 Use Case Evaluation: How good are BoK Samplers?

We evaluatethe BoKSampler as a practical, decoding-time regulariser forimproving multisample generation without additional training or external verifiers. The evaluation is designed to answer three questions: (i) whether BoK improves solution quality over standard sampling baselines; (ii) whether the gains are robust across decoding temperatures and different hyperparameter choices in the underlying optimisation; and (iii) what compute overhead is incurred when solving the BoK objective with mirror ascent. Our main finding is that BoK consistently improves or matches standard decoding across tasks and models, with the largest gains appearing in higher-temperature regimes where vanilla sampling is more diverse but less reliable. BoK leverages the regularised objective to retain diversity while increasing the chance of sampling high-quality alternatives with only a small computational overhead.

Experimental setup. We evaluate BoK on a math-specialised model Qwen2.5-Math-7B and a general-purpose model Qwen2.5-7B, across three complementary benchmarks spanning math, QA, and code: MATH500 [22], GPQA-diamond [32], and HumanEval [8]. We compare BoK against standard autoregressive sampling (Base) from the model distribution

Method τ=0.10 τ=0.25 τ=0.50 τ=0.70 τ=0.90 MATH500 (Qwen2.5-Math-7B)

Base 72.2 72.6 69.4 64.4 53.0 Top-K 72.8 73.4 69.4 65.0 56.2

- BoK (Ours) β=0.01, λ=0.1 74.2 72.8 71.0 73.0 71.2
- BoK (Ours) β=0.02, λ=0.2 72.6 71.8 72.2 72.4 71.6 BoK (Ours) β=0.05, λ=0.5 72.8 71.6 72.8 72.8 70.8

###### GPQA (Qwen2.5-Math-7B)

Base 32.32 33.84 26.26 31.31 30.30 Top-K 33.84 35.35 28.79 31.31 31.82

- BoK (Ours) β=0.01, λ=0.1 30.81 33.84 31.82 33.33 36.36
- BoK (Ours) β=0.02, λ=0.2 31.31 36.36 33.33 34.85 31.82 BoK (Ours) β=0.05, λ=0.5 33.84 32.83 35.35 31.82 31.82

###### HumanEval (Qwen2.5-Math-7B)

Base 56.71 53.05 48.78 49.39 32.93 Top-K 54.88 51.83 51.83 46.34 37.80

- BoK (Ours) β=0.01, λ=0.1 54.27 54.88 50.0 54.27 52.44
- BoK (Ours) β=0.02, λ=0.2 54.88 54.88 51.83 53.05 52.44 BoK (Ours) β=0.05, λ=0.5 56.01 51.83 51.83 52.44 51.22

- Table 1. Accuracy across temperatures for Qwen2.5-Math-7B on MATH500, GPQA, and HumanEval. We report Base, Top-K (K=50), and BoK with three representative (β, λ) settings, illustrating robustness across both temperature and hyperparameters.

- at temperature τ, and Top-K, sampling restricted to the top-K tokens per step with renormalisation(wefix K=50acrosstasksandtemperatures). Allmethodsusethesameprompts (Qwen default prompts [49]) and evaluation scripts, the same maximum generation length Tmax=3072 with early stopping on EOS, and the same random-seed protocol.

Results and Analysis. We implement the BoK sampler following Algorithm 1, using mirror-ascent updates to compute a per-step sampling distribution q⋆t (·) that increases exploration via the coverage utility while remaining anchored to the base model through KL regularisation (Eq. (17)). In particular, β controls the strength of the coverage reward, and λ controls the strength of KL anchoring toward the reference distribution pt. Across tasks and models, BoK matches or improves upon Base and Top-K sampling in most settings, as shown in Tables 1 and 2. The gains are most pronounced at higher temperatures, where vanilla sampling becomes more diverse but less reliable. ForQwen2.5-Math-7BonMATH500at τ=0.9, BoK increases accuracy from 53.0% to 71.6% (+18.6%) and exceeds Top-K at 56.2% by +15.4%. We observe asimilarpatternonGPQAandHumanEval, whereBoKimprovesoverthebaselinesat τ=0.9 by +6.06% and +14.64%. Similar improvements hold for Qwen2.5-7B, where BoK substan-

Method τ=0.10 τ=0.25 τ=0.50 τ=0.70 τ=0.90 MATH500 (Qwen2.5-7B)

Base 57.6 60.4 56.6 50.2 44.2 Top-K 59.1 59.4 52.8 51.6 41.0 BoK (Ours) β=0.01, λ=0.1 57.6 59.6 60.6 60.0 58.0 BoK (Ours) β=0.02, λ=0.2 58.8 59.2 61.8 60.4 60.2 BoK (Ours) β=0.05, λ=0.5 59.1 61.0 59.4 59.4 59.4

###### GPQA (Qwen2.5-7B)

Base 32.83 27.78 21.72 25.76 24.24 Top-K 28.28 29.80 27.78 29.80 32.32 BoK (Ours) β=0.01, λ=0.1 31.82 29.80 29.80 29.29 32.32 BoK (Ours) β=0.02, λ=0.2 31.82 29.29 30.30 27.78 29.29 BoK (Ours) β=0.05, λ=0.5 30.30 30.30 29.80 29.29 24.24

###### HumanEval (Qwen2.5-7B)

Base 70.13 67.68 71.34 71.95 45.12 Top-K 71.95 70.73 71.95 65.24 57.93 BoK (Ours) β=0.01, λ=0.1 72.56 71.95 71.95 72.56 71.34 BoK (Ours) β=0.02, λ=0.2 71.59 73.17 74.78 71.51 69.51 BoK (Ours) β=0.05, λ=0.5 72.56 72.56 69.51 69.51 66.46

- Table 2. Accuracy across temperatures for Qwen2.5-7B on MATH500, GPQA, and HumanEval (same baselines and BoK settings as Table 1).

tially mitigates the accuracy drop at high temperatures. At lower temperatures, improvements are naturally smaller; in a few near-deterministic regimes, Base or Top-K sampling can remain slightly stronger or comparable to BoK, which is consistent with the reduced need for exploration when the model distribution is already sharply peaked. The results also indicate that BoK is not overly sensitive to hyperparameter choice: multiple (β, λ) settings yield competitive performance across temperatures, with the best pair varying by task and regime. Overall, the consistent improvements across all three tested configurations suggest a stable operating region. BoK thus provides a practical and robust way to trade off diversity and reliability at decoding time without exhaustive hyperparameter search, with particularly strong benefits in high-entropy sampling settings.

Efficiency and practical deployment. In practice, BoK introduces only a small runtime overhead because computing the BoK-optimal distribution q⋆t requires only a few mirrorascent updates per token. All BoK results in Tables 1 and 2 use 5 mirror-ascent steps per token. Acrossthethreebenchmarks,theadditionalcostremainsmodestrelativetobasedecoding: on MATH500, BoK runs in 16.88s vs. 15.84s; on GPQA, 17.60s vs. 15.43s; and on HumanEval, BoK is slightly faster in our implementation (8.65s vs. 9.74s), which we attribute to shorter generations under BoK. This suggests that the mirror-ascent solver converges quickly, so only a small number of optimisation steps are required to obtain an effective q⋆t .

Gradient steps Base 2 5 10 15 20 MATH500 acc. (%) 64.4 69.6 73.0 71.6 71.2 72.8 Runtime (s) 15.84 15.87 16.88 17.70 17.91 18.26

- Table 3. Effect of the number of mirror-gradient steps per token on MATH500 accuracy for Qwen2.5-MATH-7B (BoK) using τ = 0.7, β = 0.01 and λ = 0.1.

To make this explicit, we further study the effect of the number of mirror-gradient steps per token on the MATH500 task, as shown in Table 3. There is an improvement compared with thebasedecoding, evenwithasmallnumberofsteps: usingonly2stepsimprovesaccuracy from 64.4% to 69.6% with a negligible runtime increase (15.87s), and 5 steps reach 73.0% with a still modest overhead (16.88s). Increasing the number of steps beyond 5 yields only marginal changes in accuracy while steadily increasing runtime. Overall, these results indicate that BoK remains effective with a small, fixed number of mirror-ascent steps, making it suitable for practical decoding-time use.

## 5 Conclusion and Future Work

We argued that decoding should not be viewed as a collection of disconnected heuristics. Instead, many widely used decoding rules arise as exact solutions to a single optimisation template on the probability simplex, where different decoding behaviours correspond to different regularisers and feasible sets. This perspective unifies classical procedures (e.g., greedy, Softmax, Top-K/Top-P, and sparse decoders) under one “Master” problem, and makes explicit the role of optimality conditions in determining which tokens become active or inactive. When closed-form solutions are unavailable, we showed that mirror ascent provides a principled, simplex-native solver whose updates preserve valid distributions by construction. Finally, to demonstrate that the framework is not merely explanatory, we introduced a concrete use case, the Best-of-K (BoK) regulariser, which directly targets multisample coverage and can be implemented with the same mirror-ascent machinery.

The optimisation view suggests several promising directions. First, while we focused on per-step decoding, it would be natural to extend the framework to sequence-level objectives that couple decisions across time [17, 18], e.g., enforcing coverage, length, or style constraints globally rather than locally. Second, BoK illustrates how multi-sample utilities can be encoded as regularisers; a broader family of compute-aware objectives could be explored, including utilities that model downstream reranking, verifier selection, or selfconsistency more explicitly [1, 9, 46, 53]. Third, mirror ascent opens the door to richer constraint sets beyond the simplex, such as structured sparsity, group constraints, or dynamic support sets that depend on external tools or retrieval modules [6, 21, 33, 36]. We hope this work helps shift decoder design from folklore to first-principles objective design.

###### In short: Decoding is not a hack; it is optimisation!

## References

- [1] Pranjal Aggarwal et al. Let’s Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs. 2023. arXiv: 2305.11860 [cs.CL].
- [2] Amir Beck and Marc Teboulle. “Mirror descent and nonlinear projected subgradient methods for convex optimization”. In: Operations Research Letters 31.3 (2003), pp. 167–175.
- [3] Dimitri P. Bertsekas. Nonlinear Programming. 2nd. Belmont, MA: Athena Scientific, 1999.
- [4] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.
- [5] Lev M Bregman. “The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming”. In: USSR computational mathematics and mathematical physics 7.3 (1967), pp. 200–217.
- [6] Andrew Brown, Muhammad Roman, and Barry Devereux. A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges. 2025. arXiv: 2508.06401 [cs.DL].
- [7] Souradip Chakraborty et al. “Transfer q-star: Principled decoding for llm alignment”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 101725–101761.
- [8] Mark Chen et al. “Evaluating Large Language Models Trained on Code”. In: arXiv preprint arXiv:2107.03374 (2021).
- [9] Xinyun Chen et al. Universal Self-Consistency for Large Language Model Generation.

2023. arXiv: 2311.17311 [cs.CL].

- [10] Yung-Sung Chuang et al. DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models. 2024. arXiv: 2309.03883 [cs.CL].
- [11] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical Neural Story Generation.

2018. arXiv: 1805.04833 [cs.CL].

- [12] GiorgioFranceschelliandMircoMusolesi.CreativeBeamSearch:LLM-as-a-JudgeFor Improving Response Generation. 2024. arXiv: 2405.00099 [cs.AI].
- [13] Raja Gond et al. LLM-42: Enabling Determinism in LLM Inference with Verified Speculation. 2026. arXiv: 2601.17768 [cs.LG].
- [14] Chuan Guo et al. On Calibration of Modern Neural Networks. 2017. arXiv: 1706 . 04599 [cs.LG].
- [15] Ari Holtzman et al. The Curious Case of Neural Text Degeneration. 2020. arXiv: 1904. 09751 [cs.CL].
- [16] Xiaotong Ji et al. On Almost Surely Safe Alignment of Large Language Models at Inference-Time. 2025. arXiv: 2502.01208 [cs.LG].
- [17] Xiaotong Ji et al. Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening. 2026. arXiv: 2601.21590 [cs.LG].

- [18] Aayush Karan and Yilun Du. Reasoning with Sampling: Your Base Model is Smarter Than You Think. 2025. arXiv: 2510.14901 [cs.LG].
- [19] Jyrki Kivinen and Manfred K Warmuth. “Exponentiated gradient versus gradient descent for linear predictors”. In: Information and computation 132.1 (1997), pp. 1–63.
- [20] Ehsan Latif, Ramviyas Parasuraman, and Xiaoming Zhai. “PhysicsAssistant: An LLMPowered Interactive Learning Robot for Physics Lab Investigations”. In: 2024 33rd IEEEInternationalConferenceonRobotandHumanInteractiveCommunication(ROMAN). 2024, pp. 864–871. DOI: 10.1109/RO-MAN60168.2024.10731312.
- [21] Patrick Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2021. arXiv: 2005.11401 [cs.CL].
- [22] Hunter Lightman et al. “Let’s Verify Step by Step”. In: arXiv preprint arXiv:2305.20050

(2023).

- [23] Elon Litman. “Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport”. In: arXiv preprint arXiv:2508.08369 (2025).
- [24] Fang Liu et al. Beyond Functional Correctness: Exploring Hallucinations in LLMGenerated Code. 2026. arXiv: 2404.00971 [cs.SE].
- [25] Andre Martins and Ramon Astudillo. “From softmax to sparsemax: A sparse model of attention and multi-label classification”. In: International conference on machine learning. PMLR. 2016, pp. 1614–1623.
- [26] Clara Meister, Ryan Cotterell, and Tim Vieira. “If beam search is the answer, what was thequestion?”In:Proceedingsofthe2020ConferenceonEmpiricalMethodsinNatural Language Processing (EMNLP). 2020, pp. 2173–2185.
- [27] Ludovico Mitchener et al. BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology. 2025. arXiv: 2503.00096 [q-bio.QM].
- [28] Sidharth Mudgal et al. “Controlled decoding from language models”. In: arXiv preprint arXiv:2310.17022 (2023).
- [29] Arkadij Semenovič Nemirovskij and David Borisovich Yudin. “Problem complexity and method efficiency in optimization”. In: (1983).
- [30] Minh Nhat Nguyen et al. Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs. 2025. arXiv: 2407.01082 [cs.CL].
- [31] Georgy Noarov et al. Foundations of Top-k Decoding For Language Models. 2025. arXiv: 2505.19371 [cs.AI].
- [32] DavidReinetal.“Gpqa:Agraduate-levelgoogle-proofq&abenchmark”.In:FirstConference on Language Modeling. 2024.
- [33] Timo Schick et al. Toolformer: Language Models Can Teach Themselves to Use Tools.

2023. arXiv: 2302.04761 [cs.CL].

- [34] Shai Shalev-Shwartz. “Online Learning and Online Convex Optimization”. In: Foundations and Trends® in Machine Learning 4.2 (2012), pp. 107–194.
- [35] Chufan Shi et al. A Thorough Examination of Decoding Methods in the Era of LLMs.

2024. arXiv: 2402.06925 [cs.CL].

- [36] Zhengliang Shi et al. Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents. 2025. arXiv: 2405.16533 [cs.CL].
- [37] Tarun Suresh et al. BEAVER: An Efficient Deterministic LLM Verifier. 2025. arXiv: 2512.05439 [cs.AI].
- [38] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. “Sequence to Sequence Learning with Neural Networks”. In: Advances in Neural Information Processing Systems. Ed. by Z. Ghahramani et al. Vol. 27. Curran Associates, Inc., 2014.
- [39] Tim Tomov, Dominik Fuchsgruber, and Stephan Günnemann. “Task-Awareness Improves LLM Generations and Uncertainty”. In: arXiv preprint arXiv:2601.21500

(2026).

- [40] Rasul Tutunov et al. Model-Based and Sample-Efficient AI-Assisted Math Discovery in Sphere Packing. 2025. arXiv: 2512.04829 [cs.AI].
- [41] Rasul Tutunov et al. Why Can Large Language Models Generate Correct Chain-ofThoughts? 2024. arXiv: 2310.13571 [cs.CL].
- [42] Ashwin K Vijayakumar et al. Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models. 2018. arXiv: 1610.02424 [cs.AI].
- [43] Luke Vilnis et al. Arithmetic Sampling: Parallel Diverse Decoding for Large Language Models. 2023. arXiv: 2210.15458 [cs.CL].
- [44] Chenxi Wang et al. MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation. 2025. arXiv: 2410.11779 [cs.CL].
- [45] Tianchun Wang et al. On the Effect of Sampling Diversity in Scaling LLM Inference.

2025. arXiv: 2502.11027 [cs.LG].

- [46] Xuezhi Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. 2023. arXiv: 2203.11171 [cs.CL].
- [47] Jiaheng Wei et al. Measuring and Reducing LLM Hallucination without GoldStandard Answers. 2024. arXiv: 2402.10412 [cs.CL].
- [48] DavidP.Woodruffetal.AcceleratingScientificResearchwithGemini:CaseStudiesand Common Techniques. 2026. arXiv: 2602.03837 [cs.CL].
- [49] An Yang et al. “Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement”. In: arXiv preprint arXiv:2409.12122 (2024).
- [50] Jiayi Zhang et al. Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity. 2025. arXiv: 2510.01171 [cs.CL].
- [51] Yuxuan Zhou, Margret Keuper, and Mario Fritz. Balancing Diversity and Risk in LLM Sampling:HowtoSelectYourMethodandParameterforOpen-EndedTextGeneration.

2025. arXiv: 2408.13586 [cs.CL].

- [52] Matthieu Zimmer et al. Bourbaki: Self-Generated and Goal-Conditioned MDPs for Theorem Proving. 2025. arXiv: 2507.02726 [cs.AI].
- [53] Matthieu Zimmer et al. Rethinking Large Language Model Distillation: A Constrained Markov Decision Process Perspective. 2025. arXiv: 2509.22921 [cs.LG].

