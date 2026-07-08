# arXiv:2603.29025v3[cs.CL]9Jun2026

## The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning

Yubo Li1∗ Lu Zhang2 Tianchong Jiang2 Ramayya Krishnan1 Rema Padman1 1Carnegie Mellon University 2Independent Researcher

### Abstract

Large language models systematically fail when a salient surface cue conflicts with an unstated feasibility constraint. Our primary contribution is a scalable benchmark; we pair it with a falsifiable behavioral characterization of the failure, following a diagnose–measure–bridge–treat arc. Causalbehavioral analysis of the “car wash problem” across six models reveals approximately context-independent sigmoid heuristics: the distance cue exerts 8.7–38× more influence than the goal, and token-level attribution shows patterns more consistent with keyword associations than compositional inference. The Heuristic Override Benchmark (HOB)—500 instances spanning 4 heuristic × 5 constraint families with minimal pairs and explicitness gradients—demonstrates generality across 14 models: under strict evaluation (10/10 correct), no model exceeds 75%, and presence constraints are hardest (44%). A minimal hint recovers +15pp on average, suggesting the failure is in constraint inference rather than missing knowledge; 12/14 models perform worse when the constraint is removed (up to −39pp), revealing conservative bias. A controlled thinking-mode ablation on Gemini 3.1 Pro (baseline 74.6% with thinking on → 58.4% with thinking off; recovered to 71.2% with explicit goal decomposition) shows that internal deliberation is doing real work and that explicit prompting substitutes for it. Reasoning models do not categorically outperform non-reasoning peers: controlling for capability rank, the residual reasoning-mode effect is +1.8 pp (n.s.). Parametric probes confirm the sigmoid pattern generalises to cost, efficiency, and semantic-similarity heuristics; goal-decomposition prompting yields +5.0 pp on average vs +3.1 pp for generic chain-of-thought, isolating constraint enumeration as the active ingredient. Together, these results characterise heuristic override as a systematic reasoning vulnerability with a quantified locus (inference order, not knowledge) and a tested intervention.

### 1 Introduction

Large language models are rapidly moving from research tools to everyday decision-support systems. People consult them for travel planning, medical triage, legal interpretation, financial advice, and moral judgment (Cheung et al., 2025; Echterhoff et al., 2024; Omar et al., 2024). As the scope of LLM-assisted decision-making widens, so does the potential for harm when the model’s reasoning is flawed in ways that are difficult to anticipate. Unlike factual hallucinations, which can in principle be verified against external knowledge, reasoning errors—cases where the model draws an incorrect conclusion from correctly perceived premises—are harder to detect because the output sounds plausible and internally consistent.

A growing body of work documents shortcut learning—models exploiting surface-level statistical regularities rather than performing the intended computation (Geirhos et al., 2020;

∗Corresponding author: yubol@andrew.cmu.edu

Code and data are publicly available at github.com/yubol-bobo/HiddenConstraintBench; additional materials are on the project webpage.

Du et al., 2023)—across NLI (McCoy et al., 2019), QA (Ko et al., 2020), mathematical reasoning (Shi et al., 2023; Mirzadeh et al., 2024; Yang et al., 2025), and arithmetic (Nikankin et al., 2024; Branco et al., 2021). Cognitive-bias analogues (anchoring, framing, representativeness, content effects) further compound the problem (Suri et al., 2024; Binz & Schulz, 2023; Wang et al., 2024; Malberg et al., 2025; Echterhoff et al., 2024; Lampinen et al., 2024), and can amplify human biases when users defer to model recommendations (Cheung et al., 2025). Yet this literature overwhelmingly measures shortcut reliance through accuracy—a binary signal that reveals that the model fails but not why.

A recent viral test crystallized this gap with striking clarity. In February 2026, a Mastodon user posed a single-sentence question to four frontier LLMs (K´evin (@knowmadd), 2026):

“I want to wash my car. The car wash is 50 meters away. Should I walk or drive?”

Every model recommended walking; the correct answer is to drive, because you cannot wash a car that is not at the car wash. The question went viral (Allen, 2026), and a subsequent 53-model evaluation found that 42 recommended walking on a single pass, with only 5 answering correctly across ten trials (Opper AI, 2026).

The problem is diagnostic because it is simple: no specialised knowledge, no multi-step arithmetic, no ambiguous premises—just a conflict between a surface heuristic (short distance ⇒ walk) and an implicit constraint (the car must be co-located with the wash). This conflict structure recurs whenever an unstated prerequisite competes with a statistically dominant surface pattern, from medical triage (“mild symptom ⇒ wait”) to legal reasoning (“standard clause ⇒ sign”). Prior work connects the failure to the classical frame problem (McCarthy & Hayes, 1981) and shows that structured prompting can raise single-model accuracy from 30% to 85% (Jo, 2026), confirming that the bottleneck is not missing information but the order and structure of processing. However, no prior study has provided a systematic analysis that (i) identifies which surface features trigger the heuristic, (ii) measures how robustly it persists under controlled perturbation, or (iii) characterises the reasoning traces that distinguish correct from incorrect responses.

Contributions. Following a diagnose–measure–bridge–treat arc, our primary contribution is a benchmark; the case study that motivates it is a behavioral characterization of the failure, not a claim about the network’s internal implementation. (1) HOB, a 500-instance benchmark with minimal pairs and explicitness gradients across 4×5 = 20 cells (15 populated, inter-rater κ = 0.71), on which no model among 14 frontier systems exceeds 75% strict accuracy; its controls expose an inference bottleneck and a conservative bias invisible to aggregate accuracy. (2) A falsifiable behavioral characterization—via causal occlusion and a 14-point monotonicity sweep—of the decision as a dominant, context-independent cue mapping with goal sensitivity 8.7–38× smaller—a diagnostic signature HOB then tests at scale. (3) A localisation of the bottleneck to processing order: a thinking-mode ablation and a CoTvs-goal-decomposition comparison isolate constraint enumeration as the operative factor behind the +6–9pp mitigation gains.

### 2 Method

Our investigation follows a diagnose–measure–bridge–treat arc: a behavioral case study of the car wash failure (§2.1), systematic benchmarking across heuristic and constraint types (§2.2), parametric sweeps testing whether the pattern generalises, and a mitigation experiment. §2.3 describes the experimental setup.

Scope of the analysis. Our interventions are causal but behavioral—input-side perturbations of a frozen model—so they characterise its input–output function and decomposition, not the circuit that implements it. We thus report a falsifiable behavioral signature and leave representational validation (linear probing, activation patching) to future work.

##### 2.1 Behavioral Characterization: The Car Wash Case Study

- 2.1.1 Task Formulation

The car wash problem presents a binary choice in which a salient surface cue conflicts with an implicit goal constraint. The input decomposes into a goal (“get my car washed”), a heuristic cue (“just 100m away”), and options (“walk or drive”). The correct answer is DRIVE—the car must physically be present—yet the short distance cues WALK.

We define a scalar decision score s(x) = log p(WALK | x) − log p(DRIVE | x), extracted via anchored teacher-forced scoring: a fixed anchor (‘‘\nFinal:’’) is appended after the generation prefix to create a deterministic scoring position. For multi-token candidates, log-probabilities are computed via teacher-forced decoding with KV-cache reuse; the total mass aggregates across tokenisation variants via log-sum-exp, yielding a generation-free, exactly reproducible score. Since scoring is deterministic, we construct K semantically equivalent paraphrases per scenario and report means, standard deviations, and 95% CIs.

- 2.1.2 Causal Occlusion Analysis

To identify which input component drives the decision, we apply causal occlusion (Zeiler & Fergus, 2014)—perturbing each component independently and measuring the change in decision score:

A(z) = s occ(x, z) − s(x). (1)

We apply occlusion at three levels: sentence (which sentence matters most), span (which semantic concept—goal, heuristic cue, or options), and token (compositional vs. keyword processing within the dominant span). To control for out-of-distribution artefacts that arise when inputs are perturbed (Hooker et al., 2019), we use three replacement operatorsmask, neutral (semantically neutral substitute), and contradict (semantic flip)—and require agreement across all three.

- 2.1.3 Monotonicity Curve Analysis

The occlusion analysis identifies what the model relies on; the monotonicity analysis characterises how—as a context-independent heuristic or a goal-modulated factor. We sweep distance d over 14 log-spaced values (10m–100km) in a conflict condition (car wash: Drive always correct) and a control condition (coffee shop: answer depends on distance), sampling T=5 from 7 templates per point (2 × 14 × 5 = 140 prompts/model). Correct reasoning produces a flat conflict curve and a sigmoid control; a pure heuristic produces two near-identical sigmoids.

##### 2.2 HOB: Heuristic Override Benchmark

The case study reveals that models apply a proximity heuristic overriding a presence constraint. We introduce HOB to test whether this extends to other heuristic types (cost, efficiency, semantic match) and constraint types (capability, validity, scope, procedural). HOB is organised along two dimensions (Table 1): 4 heuristic families (what misleads the model) × 5 constraint families (what the model misses), yielding 20 potential cells of which 15 are populated based on naturalness ratings. A complete annotated instance is in Appendix A.

Every instance has a minimal pair in which only the constraint-relevant noun phrase or modifier is replaced (e.g., “get my car washed” → “pick up a car wash gift card”), with sentence length, syntactic structure, and domain held fixed; mean pair lengths are matched (base 28.4±7.1 vs. pair 27.9±6.8 tokens, n.s.). Instances also vary along two controlled gradients: heuristic strength (strong/medium/weak) and constraint explicitness (implicit/hint/explicit), enabling fine-grained analysis of when models overcome the heuristic. HOB includes 30 control instances (no constraint conflict) and totals ∼500 instances across 15 cells (of 4 × 5 = 20 possible H×C combinations; the 5 empty cells were filtered by a naturalness pre-screen, ≥ 4/5 mean rating across 3 raters) and 7 domains (Appendix A). We validate cell assignment with an independent inter-rater study: 50 randomly sampled

Table 1: HOB taxonomy. 4 heuristic × 5 constraint families; 15 cells populated.

Heuristic Families Pattern Typical Cues

H-prox Proximity Closer → better “5 min away,” “next door” H-eff Efficiency Faster → better “quickest way,” “saves time” H-cost Cost Cheaper → better “free option,” “saves money” H-sem Semantic Name sounds right → viable “gas station” for tires

###### Constraint Families Definition Example

C-pres Presence Object must be at destination Car must be at car wash C-cap Capability Means cannot do the task Can’t carry sofa on foot C-val Validity Precondition is violated Can’t drive w/ flat tire C-scope Scope Service can’t fulfil goal Gas station won’t fix tires C-proc Procedural Step or timing not met Store is already closed

instances × 3 trained annotators yield Fleiss’ κ = 0.71 (substantial agreement (Landis & Koch, 1977)), rising to κ = 0.84 after a single calibration pass on the C-scope vs. C-cap boundary (the most contested boundary, related to the classical frame problem (McCarthy & Hayes, 1981)). Per-cell κ values are in Appendix F.

##### 2.3 Experimental Setup

- Study 1: Behavioral case study (6 models). We evaluate Qwen3-{4B, 8B, 14B, 32B}, Qwen3.5-27B, and GPT-OSS-20B on the car wash scenario with K=6 paraphrases, run three times independently (Appendix B). From the span-level attributions we derive:

HDR = |A(H)|/|A(G)| (Heuristic Dominance Ratio), (2) CSI = |A(G)| (Constraint Sensitivity Index), (3) DSI = |A(H)| (Distance Sensitivity Index), (4)

where G and H denote the goal and heuristic spans. HDR > 1 indicates greater heuristic than goal sensitivity. For monotonicity, we report smin (conflict score at 10m), crossover distance, and mean conflict–control offset.

- Study 2: HOB benchmark (14 models). We evaluate 14 models—10 API (GPT-5.4, GPT-5.2, Claude Opus 4.6, Claude Sonnet 4.5, DeepSeek R1, Gemini 3.1 Pro, Grok 4.2, Kimi K2.5, Llama 4 Scout, GPT-OSS-120B) and 4 local (Qwen3-14B, Qwen3-32B, Qwen3.5-27B, GPT-OSS20B)—queried N=10 times per instance at T=0.7 (∼70,000 total), judged by Qwen3-32B following LLM-as-judge practice (Zheng et al., 2023). We adopt a strict criterion: an instance is correct only if all 10 trials are correct; we additionally report trial-level accuracy (the standard average-of-10 measure) in App. D.1. A controlled temperature ablation (T ∈ {0.0,0.3,0.7}) on three representative models (Gemini 3.1 Pro, GPT-5.4, Llama 4 Scout) verifies that the strict-accuracy ranking is preserved across decoding settings (Spearman ρ > 0.97; App. I). Two diagnostic comparisons leverage the built-in controls: the explicitness gradient (implicit vs. hint accuracy) and the minimal-pair asymmetry (base vs. pair accuracy). We additionally categorise models as “reasoning” or “non-reasoning” based on whether explicit thinking is enabled by default in their public API configuration (reasoning: DeepSeek R1, Gemini 3.1 Pro, GPT-5.x, Claude Opus 4.6, Grok 4.2)1 and analyse the resulting performance contrast (§3.5).

To test whether the sigmoid pattern generalises, we extend the parametric sweep to four H × C combinations: H-cost × C-scope (cost: $0–$500; 13 grid points), H-eff × C-cap (time: 1min–8h; 10 grid points), H-prox × C-cap (distance: 50m–50km, carrying a heavy item home; 12 grid points), and H-sem × C-scope (semantic similarity; 7 grid points), each with conflict/control conditions and T=10 trials per grid point (840 prompts/model across all sweeps).

1Claude Sonnet 4.5 also supports extended thinking but is configured off by default in our setup; classifying it as reasoning would not change any of the conclusions in §3.5 (see App. J).

We test a goal-decomposition prompt—“Before answering, list the necessary conditions for the stated goal. Then answer.”—against two baselines on Gemini 3.1 Pro, GPT-5.4, and Llama 4 Scout across all ∼500 HOB instances (N=10). The baselines are (i) zero-shot, and (ii) a generic chain-of-thought (CoT) prompt—“Let’s think step by step.”—which invites deliberation without specifying what to deliberate over. The two-baseline contrast isolates whether the gain comes from constraint enumeration specifically or from deliberation in general. For Gemini 3.1 Pro we additionally ablate the model’s native thinking mode (default-on vs. explicitly disabled via thinking budget=0) to distinguish prompt-level from architecturelevel deliberation.

### 3 Results

##### 3.1 Behavioral Characterization

We evaluate six models (Qwen3-{4B, 8B, 14B, 32B}, Qwen3.5-27B, GPT-OSS-20B) on the car wash problem (details in Appendix B). All achieve 0% accuracy: every paraphrase produces the wrong answer. Decision scores range from s¯ = +2.2 (Qwen3.5-27B, p(WALK) > 0.90) to +13.8 (Qwen3-4B, near-total Walk mass). Scaling is non-monotonic: Qwen3-14B (+12.0) is more confident in the wrong answer than the larger Qwen3-32B (+5.9).

17.5

Basedecisionscores(x)

15.0

12.5

10.0

7.5

5.0

2.5

0.0

Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32BQwen3.5-27BGPT-OSS-20B

[Figure 1]

Δs(+towardWalk,−towardDrive)

+4.8 +7.5 +3.5 -14.8 -14.9 -30.3

Qwen3-4B

30

[Figure 2]

20

- -1.0 +0.7 +0.8 -10.9 -5.0 -30.3

- +0.4 +0.9 +0.7 -16.2 -17.1 -23.8

-0.5 +0.9 -0.4 -5.9 -5.8 -10.8

- +1.6 +2.3 +0.8 -3.9 -2.7 -7.7

Qwen3-8B

10

Qwen3-14B

0

Qwen3-32B

−10

Qwen3.5-27B

−20

-0.9 +0.6 -0.2 -1.1 -1.2 -3.0 −30

GPT-OSS-20B

Goal-MaskGoal-NeutralGoal-Contradict Dist-MaskDist-NeutralDist-Contradict

- Figure 1: Left: Base decision scores s(x). All positive (incorrect Walk preference); nonmonotonic scaling. Right: Span-level occlusion heatmap. Distance columns uniformly blue (∆s < 0, toward Drive); goal columns near-zero or red.

Causal occlusion. Three findings emerge from span-level perturbation (Figure 1; Table 7 and per-paraphrase diagnostics in Appendix C). First, perturbing the distance span shifts every model toward Drive (∆s from −1.2 to −30.3), consistently across all three operators. Second, perturbing the goal produces near-zero or positive effects—for Qwen3-4B, neutral goal replacement yields ∆s = +7.5, making Walk more likely when the constraint is removed. Third, the Heuristic Dominance Ratio (HDR) ranges from 8.7× to 38.0×: the distance cue is at least an order of magnitude more influential than the goal (paired bootstrap on HDR > 1: p < 0.001 for all six models). The HDR decomposition (App. C) further shows that goal influence is paraphrase-fragile (|A(goal)| varies 6.4× across paraphrases, range 1.2–7.5 log-odds) while distance influence is paraphrase-stable (2.3×, range 13.1–30.3)—consistent with the goal acting as a weak modulator rather than a robust feature.

Token-level attribution. Sentence-level masking confirms |∆sdistance| > |∆squestion| > |∆sgoal| for every model. Token-level masking within the goal span (Appendix C) reveals why: washing-action tokens weakly favour Drive, while “car” and “vehicle” favour Walk; the opposing effects cancel. The largest token effect (|∆s| = 5.8) is 5× smaller than the distance effect (30.3), indicating keyword-level associations rather than compositional inference.

Monotonicity curves. All six models produce sigmoid conflict curves tracking the control (Figure 2), differing only in amplitude (|s¯|: < 5 to > 25) and crossover distance (800m– 3km). This universality indicates a shared behavioral signature: every model maps distance to

decision in a goal-independent manner. Even Qwen3.5-27B, which shows the strongest goal modulation (the additive downward shift the goal induces on the curve; offset −13.4), merely shifts the sigmoid downward without changing its shape—the goal nudges but never gates the decision.

30

Control (coffee)

GPT-OSS-20B Qwen3.5-27B Qwen3-32B

Qwen3-4B Qwen3-8B Qwen3-14B

()>0<0Decisionscore[:Walk,:Drive]sd

20

10

s = 0 (decision boundary)

| |
|---|

0

| |
|---|

| |
|---|

10

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

20

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10m 50m 100m 200m 500m800m1km 2km3km 5km 10km 25km 50km 100km

Distance d

- Figure 2: All six models’ conflict curves (solid) are sigmoids tracking the control (dashed gray). No flat curve appears. Details in Appendix C.

##### 3.2 HOB Benchmark

We evaluate 14 models on ∼500 HOB instances (N=10 trials, strict: correct only if all 10 pass). Table 2 summarises overall accuracy, the explicitness gradient, and minimal-pair asymmetry.

Strict accuracy ranges from 49.6% (Qwen3-32B) to 74.6% (Gemini 3.1 Pro); no model exceeds 75%, and half fall below 65%. C-pres (presence) is consistently the hardest constraint family (mean 44.4%, Figure 3), directly validating the car-wash finding at scale; C-cap (capability) is easiest at 71.6% (per-model breakdowns in Appendix D).

The explicitness gradient reveals an inference bottleneck: accuracy jumps +15.3pp on average (59.2% → 74.5%) from a single subtle hint (e.g., “get my car washed” → “get my car washed”), proving models possess the knowledge but fail to activate it autonomously (paired Wilcoxon signed-rank on perinstance pass rates, matched implicit/hint variants, p < 0.001). The minimal-pair asymmetry exposes conservative bias: 12 of 14 models perform worse when the constraint is removed (drops up to −38.5pp; paired Wilcoxon signed-rank on per-instance pass rates, p < 0.001), revealing that many “correct” base answers default to the harder option rather than reasoning about the constraint. Only GPT-OSS-120B (+13.8) and GPT-OSS-20B (+11.0) improve on pairs, consistent with genuine reasoning.

[Figure 3]

Accuracy(%,avgacross14models)

95

46.2 62.2 78.4 60.4 47.6

[Figure 4]

H-prox

90

85

40.7 75.2 64.9 49.8 59.5

H-eff

80

75

70

— 77.6 54.0 76.1 51.1

H-cost

65

60

— — — 60.0 —

H-sem

55

C-pres C-cap C-val C-scope C-proc

Figure 3: Mean strict accuracy by H × C cell. C-pres is hardest; C-cap easiest.

##### 3.3 Parametric Sweeps: Does the Signature Generalise?

Does the sigmoid signature appear for heuristic types beyond proximity? We extend the parametric sweep—varying one continuous input parameter (distance, cost, time, or semantic similarity) over an extended grid and reading off the model’s decision as a function of

- Table 2: HOB benchmark (strict 10/10). OA: override accuracy. Impl/Hint: implicit vs. hint explicitness (gap = inference bottleneck). Base/Pair: constraint-active vs. constraintremoved (∆ < 0: conservative bias).

Explicitness Minimal Pair

Model OA (%) Impl. Hint Base Pair ∆ Gemini 3.1 Pro 74.6 73.9 86.5 84.5 60.3 −24.2 Qwen3.5-27B 72.2 69.0 89.2 83.1 53.9 −29.2 Kimi K2.5 69.0 66.1 83.8 81.7 48.2 −33.5 Grok 4.2 68.6 65.2 81.1 73.9 66.7 −7.3 Claude Opus 4.6 68.0 66.4 81.1 81.7 46.8 −34.9 Claude Sonnet 4.5 66.8 64.9 81.1 78.2 51.8 −26.4 GPT-5.4 65.8 64.4 78.4 71.8 58.9 −13.0 GPT-5.2 64.4 60.3 86.5 78.2 40.4 −37.7 DeepSeek R1 64.2 62.4 73.0 75.4 49.6 −25.7 GPT-OSS-120B 52.2 48.9 67.6 44.4 58.2 +13.8 Llama 4 Scout 51.2 48.6 64.9 66.9 28.4 −38.5 Qwen3-14B 51.2 47.4 54.1 53.5 48.2 −5.3 GPT-OSS-20B 51.0 46.8 56.8 48.6 59.6 +11.0 Qwen3-32B 49.6 44.8 59.5 47.9 46.1 −1.8 Mean 62.1 59.2 74.5 69.2 51.2 −18.0

[Figure 5]

Correct Partial Correct Partial

Correct reasoning

Qwen3-4B

|[Figure 6]| |
|---|---|
| | |

Correct Partial Correct Partial

Qwen3-8B

Correct Partial Correct Partial

Qwen3-14B

Partial

Correct Correct Fail Partial

Qwen3-32B

Correct Correct Correct Partial

Qwen3.5-27B

Sigmoid failure

Fail Partial Fail Fail

GPT-OSS-20B

H-cost × C-scope

H-eff × C-cap

H-prox × C-cap

H-sem × C-scope

- Figure 4: Sweep pattern classification across 6 models × 4 parametric sweeps. Green = correct (curves distinct), yellow = partial, red = sigmoid failure (conflict tracks control, r > 0.8). The efficiency sweep shows the most failures; cost and prox-cap the most correct reasoning.

it—to four H × C combinations on all six Study-1 models (per-model curves in Appendix E; classification in Figure 4). The failure is not universal—it depends on the heuristic–constraint interaction. Correct reasoning emerges on H-cost × C-scope (copy shop vs. courthouse for certified documents; correct in 5/6 models) and H-prox × C-cap (carrying a sofa home; 4/6): the conflict curve stays on the correct side while the control curve is the expected sigmoid. But the efficiency sweep (H-eff × C-cap, carrying a 500-lb safe) and semantic sweep (Hsem × C-scope, gas-station descriptions for tire repair) reproduce the context-independent sigmoid—e.g. Qwen3-4B recommends a physically impossible action as the “faster” cue grows. Concrete capability constraints (weight, size) are easier to maintain than abstract scope constraints, mirroring the C-cap > C-scope hierarchy from Study 2.

90

Zero-shot

| |
|---|

80

+ Goal-decomposition

+9.0 pp 0.6 pp

70

Strictaccuracy(%)

+6.6 pp

60

50

40

30

20

10

0

Llama 4 Scout GPT-5.4 Gemini 3.1 Pro

- Figure 5: Goal-decomposition prompting improves weaker models substantially under strict (10/10) accuracy. GPT-5.4 gains +9.0 pp; Llama 4 Scout gains +6.6 pp. Gemini 3.1 Pro, already the strongest baseline, shows no change (−0.6pp).

##### 3.4 Mitigation: Goal-Decomposition Prompting

The explicitness gradient (§3.2) showed that a one-word hint recovers +15.3pp—models possess the knowledge but fail to activate it. Can we exploit this by prompting the model to self-generate the “hint”? We prepend a goal-decomposition instruction—“Before answering, list the necessary conditions that must be true for the stated goal to be accomplished. Then answer the question.”—and re-evaluate three models spanning the performance range on all 500 HOB instances (N=10 trials each).

Goal-decomposition produces substantial gains for the models that need it most (Figure 5, strict accuracy): GPT-5.4 improves 65.8% → 74.8% (+9.0pp) and Llama 4 Scout 51.2% → 57.8% (+6.6pp), while Gemini 3.1 Pro (already strongest at 74.6%) is unchanged (−0.6pp). Enumerating preconditions before deciding converts an implicit constraint into a self-generated hint—a practical, zero-cost intervention targeting exactly the diagnosed processing-order bottleneck.

Goal-decomposition vs. generic CoT. To test whether the gain reflects deliberation per se or constraint enumeration specifically, we compare goal-decomposition (GD) against a generic chain-of-thought (CoT) baseline (“Let’s think step by step”) on the same three models (per-model breakdown in App. G, Table 16). Across the three models, CoT yields a mean +3.1pp while GD yields +5.0pp—∼ 1.7–2.0× larger on the models that improve (GPT-5.4: +9.0 GD vs. +4.4 CoT; Llama 4 Scout: +6.6 vs. +3.8). The advantage is largest for the weakest baseline, consistent with the inference-bottleneck account: the active ingredient is not deliberation but the prompted enumeration of preconditions.

##### 3.5 Thinking Mode and Reasoning-Model Effects

Three further analyses (full detail in App. G–J) reinforce the inference-bottleneck account and address whether deliberation alone suffices. (i) Thinking-mode ablation. Disabling Gemini 3.1 Pro’s default thinking (thinking budget=0) drops strict accuracy 74.6% → 58.4% (−16.2pp); adding goal-decomposition to the thinking-OFF condition recovers it to 71.2% (+12.8pp). This double dissociation shows internal thinking and external prompting substitute for one another (App. G). (ii) Reasoning models are not categorically better. Their 9.7pp mean-accuracy edge (67.6% vs. 57.9%) collapses to a non-significant +1.8pp after controlling for capability rank (p=0.31); e.g. DeepSeek R1 (reasoning, 64.2%) underperforms Qwen3.5-27B (non-reasoning, 72.2%) (App. J). (iii) Trace audit. DeepSeek R1’s reasoning trace names the hidden constraint in only 64% of cases; when it both names and applies it, accuracy is 88.5% vs. 44.4% when it never names it (Fisher’s exact p < 0.01)—spontaneous goal decomposition is real but inconsistent (App. H).

### 4 Discussion

Unified account: an inference bottleneck. Our investigation converges on a coherent failure mode: LLMs apply context-independent heuristic mappings (sigmoids over distance, efficiency, or semantic similarity) that override implicit goal constraints. Study 1 characterises the failure behaviorally (HDR 8.7–38×); Study 2 demonstrates generality (14 models, no model above 75% strict accuracy); the parametric probes show it extends beyond proximity while concrete capability/cost-scope constraints can elicit correct reasoning. The failure is one of activation, not knowledge: the explicitness gradient (+15.3pp from a oneword hint) and token-level attribution (keyword associations, not compositional inference) show the constraint is present but not retrieved by default. The Gemini thinking-mode dissociation (74.6% → 58.4% with thinking off, recovering to 71.2% with explicit decomposition) and the CoT-vs-goal-decomposition gap then show that either internal deliberation or a precondition-enumeration prompt activates it—interchangeable routes to the same operation.

Conservative bias confound. The minimal-pair asymmetry (12/14 models worse when the constraint is removed, drops up to −38.5pp) reveals that accuracy on constraint-active instances alone overestimates genuine reasoning, making minimal pairs essential for any constraint-sensitive benchmark. The two models that resist this pattern (GPT-OSS-120B and GPT-OSS-20B, +11 to +14pp on pairs) share an open-weight training recipe distinct from frontier APIs; why their behaviour reverses is a follow-up that aggregate scores could not surface.

Deployment implications. This failure is invisible to standard evaluation: models produce fluent, confident responses that happen to be wrong. In medical triage, legal reasoning, or financial planning—domains where unstated feasibility constraints routinely compete with salient surface features—the same failure pattern can produce systematically incorrect recommendations.

Limitations. As scoped in §2, our account is behavioral, not implementational; representational validation (probing for where the constraint is encoded, activation patching to test whether the goal term is causally routed) is the natural next step. Both the analysis and HOB are English-only and target everyday feasibility constraints; extending to other languages and to expert-domain constraints (e.g., legal or clinical feasibility) is future work. The thinking-mode and CoT-vs-goal-decomposition ablations use three models spanning the performance range, and the DeepSeek R1 trace audit covers 50 of 500 instances (sufficient for the effect size, 88.5% vs. 44.4%, p < 0.01); broader replication would strengthen the architectural claim. Our contribution is primarily diagnostic, and the mitigation is a proof of concept rather than a comprehensive solution—robust defences will need broader prompting, fine-tuning, and architectural study.

### 5 Related Work

Shortcut Learning and Heuristic Reliance. Neural models routinely exploit shortcutsspurious cues correlated with labels but unrelated to intended reasoning (Geirhos et al., 2020; Du et al., 2023)—from lexical-overlap heuristics in NLI (McCoy et al., 2019; Gururangan et al., 2018) to sparse heuristic circuits in arithmetic (Nikankin et al., 2024) and cognitive biases in LLM reasoning (Wang et al., 2024; Lampinen et al., 2024). This persists in generative settings: larger models can exploit ICL shortcuts more (Tang et al., 2023), RLHF introduces task–feature–label correlations (Sun et al., 2024), and no model is universally robust (Yuan et al., 2024; Zhou et al., 2024). Recent work on token-level perturbation (Yang et al., 2025) and memorisation-vs-reasoning probes (Mirzadeh et al., 2024) measures shortcut reliance through accuracy degradation under controlled phrase perturbations. Our setting differs in three respects. First, prior work targets feature-level shortcuts in classification (lexical overlap, positional bias, distractor injection); we target reasoning-level compositional templates (“short distance → walk”) that operate at the decision-policy level—the sigmoid signature

we observe over a continuous parametric sweep cannot be produced by a feature-level shortcut model. Second, prior work cannot distinguish missing knowledge from misuse; the explicitness-gradient and goal-decomposition manipulations we introduce isolate the latter as the operative failure mode. Third, prior work evaluates with aggregate accuracy; we report strict accuracy, minimal-pair asymmetry, and HDR, which collectively detect the conservative-bias confound (12/14 models drop on minimal pairs) that aggregate scores hide.

Distractibility and Constraint-Following. Distractor benchmarks (Shi et al., 2023; Mirzadeh et al., 2024; Yang et al., 2025) inject additive noise into self-contained problems, requiring models to filter extraneous information. Constraint benchmarks (Zhou et al., 2023; Chen et al., 2025; Song et al., 2026) test compliance with stated or domain-specific rules. Our setting differs: both the heuristic cue and the hidden constraint are integral to the prompt, so the model must prioritise competing signals—inferring and enforcing a feasibility constraint that is never stated, must be derived from world knowledge, and competes with a salient heuristic.

Commonsense Reasoning and the Frame Problem. Commonsense benchmarks (Levesque et al., 2012; Bisk et al., 2020; Zellers et al., 2019; Clark et al., 2018) test whether models possess world knowledge. We test a complementary failure: models that possess the knowledge yet err because a surface heuristic overpowers it, connecting to the classical frame problem (McCarthy & Hayes, 1981). The car wash problem was tested across 53 models (Opper AI, 2026) (5 consistently correct); structured prompting raises accuracy from 30% to 85% but impedes self-correction (Jo, 2026). We generalise these single-instance observations into HOB.

Diagnostic Methodology. Our causal analysis builds on perturbation-based attribution (Zeiler & Fergus, 2014; Ribeiro et al., 2016; Lundberg & Lee, 2017) and counterfactual evaluation (Kaushik et al., 2020), mitigating distribution-shift concerns (Hooker et al., 2019) via multiple replacement operators with agreement requirements. Unlike mechanistic interpretability (Marks et al., 2025; Conmy et al., 2023; Geiger et al., 2021), our approach operates at the input–output level, applying to API-only systems. Following Singh et al. (2024), we use attribution to characterise the behavioral signature behind a systematic error; the benchmark’s built-in minimal pairs and controlled gradients serve as counterfactual tests beyond aggregate accuracy.

### 6 Conclusion

When salient surface cues conflict with unstated feasibility constraints, LLMs systematically follow the heuristic. We make four falsifiable, quantified claims, each empirically supported. (1) What drives the decision: the surface cue, with 8.7–38× more causal influence than the goal (HDR, six models, three operators). (2) How it is used: as a context-independent sigmoid—the conflict curve is shape-identical to the control curve over 14 distances. (3) Why models fail: the knowledge is present but is not retrieved by default (a one-token hint recovers +15.3pp; Gemini’s internal thinking contributes 16.2pp, recoverable by external decomposition). (4) How to fix it: explicit constraint enumeration via goal-decomposition recovers +6–9 pp on the models that need it (mean +5.0 pp), substantially more than generic chain-of-thought (mean +3.1pp), isolating constraint elicitation as the active ingredient. The minimal-pair asymmetry further reveals that 12/14 models defaulted conservatively on the base task, inflating apparent reasoning ability—a finding that motivates minimal-pair controls as a default in implicit-constraint benchmarks.

Acknowledgments. Supported in part by NIST Federal Award ID 60NANB24D231 and Carnegie Mellon University’s AI Measurement Science and Engineering Center (AIMSEC). Computation used Bridges-2 at PSC through ACCESS allocation CIS250181, supported by NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296.

### References

Ryan Allen. Car wash paradox evals. GitHub repository, https://github.com/ryan-allen/ car-wash-evals, 2026. Accessed: 2026-06-09.

Marcel Binz and Eric Schulz. Using cognitive psychology to understand gpt-3. Proceedings of the National Academy of Sciences, 120(6):e2218523120, 2023. doi: 10.1073/pnas.2218523120.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 7432–7439, 2020. doi: 10.1609/aaai.v34i05.6239. URL https://ojs.aaai.org/index.php/AAAI/article/view/6239.

Ruben Branco, Ant´onio Branco, Jo˜ao Ant´onio Rodrigues, and Jo˜ao Ricardo Silva. Shortcutted commonsense: Data spuriousness in deep learning of commonsense reasoning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 1504–1521, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.113. URL https://aclanthology.org/2021.emnlp-main.113/.

Jianghao Chen, Zhenlin Wei, Zhenjiang Ren, Ziyong Li, and Jiajun Zhang. LR²Bench: Evaluating long-chain reflective reasoning capabilities of large language models via constraint satisfaction problems. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 6006–6032, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.312. URL https://aclanthology.org/2025.findings-acl.312/.

Vanessa Cheung, Maximilian Maier, and Falk Lieder. Large language models show amplified cognitive biases in moral decision-making. Proceedings of the National Academy of Sciences, 122(25):e2412015122, 2025. doi: 10.1073/pnas.2412015122.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adri`a Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

Mengnan Du, Fengxiang He, Na Zou, Dacheng Tao, and Xia Hu. Shortcut learning of large language models in natural language understanding. Communications of the ACM, 67(1): 110–120, 2023. doi: 10.1145/3596490.

Jessica Maria Echterhoff, Yao Liu, Abeer Alessa, Julian McAuley, and Zexue He. Cognitive bias in decision-making with LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 12640–12653, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.739. URL https://aclanthology.org/2024.findings-emnlp.739/.

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems, volume 34, pp. 9574–9586, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 4f5c422f4d49a5a807eda27434231040-Abstract.html.

Robert Geirhos, J¨orn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. doi: 10.1038/s42256-020-00257-z.

Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, and Noah A Smith. Annotation artifacts in natural language inference data. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 107–112, New

Orleans, Louisiana, 2018. Association for Computational Linguistics. doi: 10.18653/v1/ N18-2017. URL https://aclanthology.org/N18-2017/.

Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. A benchmark for interpretability methods in deep neural networks. Advances in neural information processing systems, 32, 2019.

Heejin Jo. Prompt architecture determines reasoning quality: A variable isolation study on the car wash problem. arXiv preprint arXiv:2602.21814, 2026.

Divyansh Kaushik, Eduard Hovy, and Zachary Lipton. Learning the difference that makes a difference with counterfactually-augmented data. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=Sklgs0NFvr.

K´evin (@knowmadd). Car wash reasoning test. Mastodon post, https://mastodon.world/ @knowmadd/116072773118828295, February 2026. Original post, February 15, 2026. Accessed: 2026-06-09.

Miyoung Ko, Jinhyuk Lee, Hyunjae Kim, Gangwoo Kim, and Jaewoo Kang. Look at the first sentence: Position bias in question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1109–1121, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.84. URL https://aclanthology.org/2020.emnlp-main.84/.

Andrew K Lampinen, Ishita Dasgupta, Stephanie CY Chan, Hannah R Sheahan, Antonia Creswell, Dharshan Kumaran, James L McClelland, and Felix Hill. Language models, like humans, show content effects on reasoning tasks. PNAS nexus, 3(7):pgae233, 2024. doi: 10.1093/pnasnexus/pgae233.

J Richard Landis and Gary G Koch. The measurement of observer agreement for categorical data. Biometrics, pp. 159–174, 1977. doi: 10.2307/2529310.

Hector J Levesque, Ernest Davis, and Leora Morgenstern. The winograd schema challenge. In Principles of Knowledge Representation and Reasoning: Proceedings of the Thirteenth International Conference, KR 2012, Rome, Italy, 2012. AAAI Press. URL https: //aaai.org/papers/59-4492-the-winograd-schema-challenge/.

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.

Simon Malberg, Roman Poletukhin, Carolin Schuster, and Georg Groh. A comprehensive evaluation of cognitive biases in LLMs. In Proceedings of the 5th International Conference on Natural Language Processing for Digital Humanities, pp. 578–613, Albuquerque, USA, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.nlp4dh-1.50. URL https://aclanthology.org/2025.nlp4dh-1.50/.

Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=I4e82CIDxv.

John McCarthy and Patrick J Hayes. Some philosophical problems from the standpoint of artificial intelligence. In Readings in artificial intelligence, pp. 431–450. Elsevier, 1981.

R Thomas McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3428–3448, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1334. URL https://aclanthology.org/P19-1334/.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. arXiv preprint arXiv:2410.05229, 2024.

Yaniv Nikankin, Anja Reusch, Aaron Mueller, and Yonatan Belinkov. Arithmetic without algorithms: Language models solve math with a bag of heuristics. arXiv preprint arXiv:2410.21272, 2024.

Mahmud Omar, Shelly Soffer, Reem Agbareia, Nicola Luigi Bragazzi, Donald U Apakama, Carol R Horowitz, Alexander W Charney, Robert Freeman, Benjamin Kummer, Benjamin S Glicksberg, Girish N Nadkarni, and Eyal Klang. Socio-demographic biases in medical decision-making by large language models: a large-scale multi-model analysis. medRxiv, pp. 2024.10.29.24316368, 2024. doi: 10.1101/2024.10.29.24316368.

Opper AI. Car wash test on 53 leading AI models. Blog post, https://opper.ai/blog/ car-wash-test, 2026. Published February 19, 2026. Accessed: 2026-06-09.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “Why Should I Trust You?” Explaining the Predictions of Any Classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135–1144, 2016. doi: 10.1145/2939672.2939778.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Sch¨arli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, pp. 31210–31227. PMLR, 2023.

Chandan Singh, Jeevana Priya Inala, Michel Galley, Rich Caruana, and Jianfeng Gao. Rethinking interpretability in the era of large language models. arXiv preprint arXiv:2402.01761, 2024.

Da Song, Yuheng Huang, Boqi Chen, Tianshuo Cong, Randy Goebel, Lei Ma, and Foutse Khomh. Evaluating implicit regulatory compliance in llm tool invocation via logic-guided synthesis. arXiv preprint arXiv:2601.08196, 2026.

Zechen Sun, Yisheng Xiao, Juntao Li, Yixin Ji, Wenliang Chen, and Min Zhang. Exploring and mitigating shortcut learning for generative large language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 6883–6893, Torino, Italia, 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.602/.

Gaurav Suri, Lily R Slater, Ali Ziaee, and Morgan Nguyen. Do large language models show decision heuristics similar to humans? a case study using gpt-3.5. Journal of Experimental Psychology: General, 153(4):1066–1075, 2024. doi: 10.1037/xge0001547.

Ruixiang Tang, Dehan Kong, Longtao Huang, and Hui Xue. Large language models can be lazy learners: Analyze shortcuts in in-context learning. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 4645–4657, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.284. URL https:

//aclanthology.org/2023.findings-acl.284/.

Pengda Wang, Zilin Xiao, Hanjie Chen, and Frederick L Oswald. Will the real linda please stand up... to large language models? examining the representativeness heuristic in llms. arXiv preprint arXiv:2404.01461, 2024.

Minglai Yang, Ethan Huang, Liang Zhang, Mihai Surdeanu, William Yang Wang, and Liangming Pan. How is LLM reasoning distracted by irrelevant context? an analysis using a controlled benchmark. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13329–13347, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/ 2025.emnlp-main.674. URL https://aclanthology.org/2025.emnlp-main.674/.

Yu Yuan, Lili Zhao, Kai Zhang, Guangting Zheng, and Qi Liu. Do LLMs overcome shortcut learning? an evaluation of shortcut challenges in large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 12188–12200, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.emnlp-main.679. URL https://aclanthology.org/2024.emnlp-main.679/.

Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European conference on computer vision, pp. 818–833. Springer, 2014.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https://aclanthology.

org/P19-1472/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. Advances in neural information processing systems, 36:46595– 46623, 2023. URL https://proceedings.neurips.cc/paper files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets and Benchmarks.html.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Yuqing Zhou, Ruixiang Tang, Ziyu Yao, and Ziwei Zhu. Navigating the shortcut maze: A comprehensive analysis of shortcut learning in text classification by language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 2586–2614, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.findings-emnlp.146. URL https://aclanthology.org/2024.findings-emnlp.146/.

### A HOB Instance Example

- Table 3 presents a complete annotated instance from cell B2 (H-eff × C-cap), illustrating the six-element anatomy, its minimal pair, and controlled variants.

- Table 3: Annotated HOB instance from cell B2 (H-eff × C-cap). The base instance, its minimal pair, and controlled variants are shown. Bold text highlights the element that changes across variants.

Base Instance (B2-001): Moving a heavy safe

Goal Move a 500-lb gun safe to the second floor Question “I need to move a 500-pound gun safe to the upstairs bedroom. What’s the

quickest way—should I just carry it up myself or hire movers?” Heuristic cue “quickest way” (H-eff: faster → better) Hidden constraint One person cannot carry 500 lbs (C-cap: capability) Shortcut answer Carry it yourself Gold answer Hire movers Conflict type Missing precondition

###### Minimal Pair (B2-001-pair): Constraint removed

Question “I need to move a desk lamp upstairs. Quickest way?” Gold answer Carry it yourself (either acceptable; no capability constraint)

###### Heuristic Strength Variants

Strong “Carrying it myself would be instant compared to waiting hours for movers.” Medium “Carrying it myself saves about 30 minutes vs. booking movers.” Weak “Carrying it myself is a bit faster than scheduling movers.”

###### Constraint Explicitness Variants

Implicit “I need to move a 500-pound gun safe upstairs. Carry it or hire movers?” Hint “I need to move a 500-pound gun safe upstairs by myself. Carry it or hire

movers?” Explicit “I need to move a 500-pound gun safe upstairs. I can’t lift more than 50 pounds. Carry it or hire movers?”

Benchmark statistics. The full benchmark contains ∼500 instances: 132 base scenarios, 132 minimal pairs, 64 heuristic-strength variants, 64 constraint-explicitness variants, and 30 controls, spanning 15 H × C cells across 7 domains (transportation, shopping, digital, medical, home, work, travel).

### B Model Details

- Table 4: Study 1: models for the behavioral case study. All scored using the anchored teacher-forced procedure (§2.1.1).

Model Family Params Notes Qwen3-4B Qwen3 4B Dense Qwen3-8B Qwen3 8B Dense Qwen3-14B Qwen3 14B Dense Qwen3-32B Qwen3 32B Dense Qwen3.5-27B Qwen3.5 27B Dense GPT-OSS-20B GPT-OSS 20B MoE, MXFP4

Table 5: Study 2: models for HOB benchmark evaluation.

Model Provider Type Access

GPT-5.4 OpenAI Closed API GPT-5.2 OpenAI Closed API Claude Opus 4.6 Anthropic Closed API Claude Sonnet 4.5 Anthropic Closed API DeepSeek R1 DeepSeek Open API Gemini 3.1 Pro Google Closed API Grok 4.2 xAI Closed API Kimi K2.5 Moonshot Open API Llama 4 Scout Meta Open API (Groq) GPT-OSS-120B – Open API (Groq)

Qwen3-14B Alibaba Open Local Qwen3-32B Alibaba Open Local Qwen3.5-27B Alibaba Open Local GPT-OSS-20B – Open Local

All Study 1 models are loaded in bfloat16 with balanced multi-GPU distribution; scoring is fully deterministic. Study 2 API models are queried with default parameters; local models use greedy decoding. All experiments run on NVIDIA A100/H100 GPUs via SLURMmanaged HPC.

### C Study 1: Detailed Results

##### C.1 Base Accuracy and Decision Scores

- Table 6: Accuracy (%) and mean decision score s¯ on the car wash item. Positive s¯ indicates incorrect Walk preference. All six models consistently answer incorrectly.

Model Acc (%) s¯ Qwen3-4B 0 +13.8 Qwen3-8B 0 +4.7 Qwen3-14B 0 +12.0 Qwen3-32B 0 +5.9 Qwen3.5-27B 0 +2.2 GPT-OSS-20B 0 +2.3

C.2 Full Occlusion Results

- Table 7: Span-level occlusion: mean ∆s and HDR across 6 paraphrases. HDR = |∆sdist|/|∆sgoal| under the contradict operator.

###### Goal ∆s Distance ∆s

Model Mask Neut. Contra. Mask Neut. Contra. HDR Qwen3-4B +4.9 +7.5 +3.5 −14.8 −14.9 −30.3 8.7× Qwen3-8B −1.0 +0.7 +0.8 −10.9 −5.0 −30.3 38.0× Qwen3-14B +0.4 +0.9 +0.7 −16.2 −17.1 −23.8 32.6× Qwen3-32B −0.5 +0.9 −0.4 −5.9 −5.8 −10.8 29.1× Qwen3.5-27B +1.6 +2.3 +0.8 −3.9 −2.7 −7.7 9.3× GPT-OSS-20B −0.9 +0.6 −0.2 −1.2 −1.3 −3.0 14.4×

##### C.3 Token-Level Attribution

"cleaned"

"vehicle"

"my"

"car"

"a"

"needs"

"wash"

"washed"

"get"

"washing"

−4 −2 0 2 4 6

Δs (token masked)

- Figure 6: Token-level ∆s within the goal span (Qwen3-4B). Green bars (negative) weakly favour Drive; red bars (positive) favour Walk. Opposing effects cancel, leaving near-zero net goal influence. No token approaches the magnitude of the distance cue.

##### C.4 Individual Monotonicity Curves

20

30

Conflict Control

10

20

10

5

10

s(x)

s(x)

s(x)

0

0

0

−10

−10

−5

−20

10 100 1000 10000 100000

10 100 1000 10000 100000

10 100 1000 10000 100000

Distance (m)

Distance (m)

Distance (m)

7.5

6

10

5.0

4

2.5

0

0.0

s(x)

s(x)

s(x)

2

−10

−2.5

Qwen3-4B Qwen3-8B

−5.0

0

Qwen3-14B Qwen3-32B

−20

−7.5

Qwen3.5-27B GPT-OSS-20B

−2

10 100 1000 10000 100000

10 100 1000 10000 100000

10 100 1000 10000 100000

Distance (m)

Distance (m)

Distance (m)

- Figure 7: Monotonicity analysis: decision score s(d) vs. distance for conflict (orange) and control (blue) conditions across all six models. Every model produces sigmoid conflict curves that track the control curve.

10m 50m100m200m 500m800m1km2km3km5km10km 25km50km100km

Distance (log scale)

−20

−10

0

10

20

30

Scores(x)=logP(Walk)−logP(Drive)

Ideal: flat (Drive at all d)

Walk → Drive →

Conflict: car wash (correct = Drive at all d)

Control: coffee shop (correct depends on d)

Model: Qwen/Qwen3-4B | Paraphrases/point: 5

10m 50m100m200m 500m800m1km2km3km5km10km 25km50km100km

Distance (log scale)

−5

0

5

10

Scores(x)=logP(Walk)−logP(Drive)

Ideal: flat (Drive at all d)

Walk → Drive →

Conflict: car wash (correct = Drive at all d)

Control: coffee shop (correct depends on d)

Model: Qwen/Qwen3-32B | Paraphrases/point: 5

10m 50m100m200m 500m800m1km2km3km5km10km 25km50km100km

Distance (log scale)

−4

−2

0

2

4

6

Scores(x)=logP(Walk)−logP(Drive)

Ideal: flat (Drive at all d)

Walk →

Drive →

Conflict: car wash (correct = Drive at all d)

Control: coffee shop (correct depends on d)

Model: openai/gpt-oss-20b | Paraphrases/point: 5

10m 50m100m200m 500m800m1km2km3km5km10km 25km50km100km

Distance (log scale)

−15

−10

−5

0

5

10

15

20

Scores(x)=logP(Walk)−logP(Drive)

Ideal: flat (Drive at all d)

Walk → Drive →

Conflict: car wash (correct = Drive at all d)

Control: coffee shop (correct depends on d)

Model: Qwen/Qwen3-14B | Paraphrases/point: 5

- Figure 8: Individual monotonicity curves. Top: Qwen3-4B (left) and Qwen3-32B (right). Bottom: GPT-OSS-20B (left) and Qwen3-14B (right, highest Walk-bias at short distances).

20

Conflict: car wash (correct = Drive at all d)

Conflict: car wash (correct = Drive at all d)

Scores(x)=logP(Walk)−logP(Drive)

Scores(x)=logP(Walk)−logP(Drive)

7.5

Control: coffee shop (correct depends on d)

Control: coffee shop (correct depends on d)

10

5.0

Walk → Drive →

2.5

0

Walk → Drive →

0.0

−10

−2.5

−20

−5.0

Ideal: flat (Drive at all d)

−30

Ideal: flat (Drive at all d)

−7.5

−10.0

10m 50m100m200m 500m800m1km2km3km5km10km 25km50km100km

10m 50m100m200m 500m800m1km2km3km5km10km 25km50km100km

Distance (log scale)

Distance (log scale)

Model: Qwen/Qwen3.5-27B | Paraphrases/point: 5

Model: Qwen/Qwen3-8B | Paraphrases/point: 5

Figure 9: Remaining models: Qwen3-8B (left) and Qwen3.5-27B (right).

##### C.5 Monotonicity Summary Statistics

- Table 8: Monotonicity summary. smin: conflict score at shortest distance (10m). Crossover: distance where conflict curve crosses s = 0. Offset: mean difference between conflict and control curves.

Model smin (10m) Crossover Offset Qwen3-4B +12.8 ∼800m −7.6 Qwen3-8B +4.9 ∼2km −4.2 Qwen3-14B +2.4 ∼1km −4.3 Qwen3-32B +5.0 ∼1.5km −7.5 Qwen3.5-27B +12.8 ∼1km −13.4 GPT-OSS-20B +2.7 ∼3km −1.9

##### C.6 Diagnostic Profile: Qwen3-4B

HDR 4.4x

35

[Figure 7]

CSI (goal) DSI (distance)

[Figure 8]

HDR 2.0x HDR

| |
|---|

HDR 2.6x

30

9.3 9.7 11.1

Goal

20

HDR 1.5x

12.0x

|Δs(x)|(contradict)

25

10

Δs(x)(mean)

20

- -5.8 -4.8 -24.8
- -3.8 -4.0 -7.8 −20

Distance

0

15

HDR 0.5x

−10

10

Options

5

0

Canon. Para0 Para1 Para2 Para3 Para4

Mask Neutral Contradict

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |← tow|ard Drive | toward Walk<br><br>|→|

"need" "to" "installed" "with" "car" "replace" "is"

"new" "it" "a"

"one" "replaced" "because" "and" "'s" "needs" "tire"

"my" "replacement"

"flat"

−7.5 −5.0 −2.5 0.0 2.5 5.0 7.5 10.0 12.5

Δs(x) when token masked

- Figure 10: Multi-panel diagnostic profile for Qwen3-4B: span heatmap, HDR decomposition, and token-level attribution. No evidence of compositional constraint inference at the token level.

### D Study 2: Full Benchmark Results

##### D.1 Full Leaderboard

- Table 9 reports strict override accuracy (correct on all 10 trials) alongside trial-level accuracy for all 14 models.

Table 9: HOB benchmark: strict (10/10) and trial-level accuracy for all 14 models, sorted by strict accuracy.

Model Strict Inst. Trial Trials Acc (%) (n/500) Acc (%) (n/5000)

Gemini 3.1 Pro 74.6 373 86.0 4298 Qwen3.5-27B 72.2 361 85.4 4271 Kimi K2.5 69.0 345 85.4 4272 Grok 4.2 68.6 343 83.9 4196 Claude Opus 4.6 68.0 340 79.5 3973 Claude Sonnet 4.5 66.8 334 77.3 3863 GPT-5.4 65.8 329 81.7 4087 GPT-5.2 64.4 322 78.4 3919 DeepSeek R1 64.2 321 83.1 4153 GPT-OSS-120B 52.2 261 78.4 3920 Llama 4 Scout 51.2 256 70.3 3517 Qwen3-14B 51.2 256 78.2 3911 GPT-OSS-20B 51.0 255 79.1 3955 Qwen3-32B 49.6 248 78.0 3899

The gap between trial-level and strict accuracy reveals consistency: models like DeepSeek R1 (83.1% trial, 64.2% strict) and GPT-OSS-20B (79.1% trial, 51.0% strict) answer correctly on many individual trials but inconsistently across the 10-trial window, indicating stochastic rather than reliable override.

#### D.2 Per-Model H × C Heatmap

[Figure 9]

75 80 83 53 75 85 80 53 73 87 68 80 50 78

Gemini 3.1 Pro

100

42 86 80 67 60 82 69 60 63 83 64 82 60 78

Qwen3.5-27B

|[Figure 10]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

52 71 86 50 60 78 74 47 70 77 60 78 60 75

Kimi K2.5

90

57 77 77 57 75 82 66 57 60 83 64 75 65 60

Grok 4.2

75 86 71 47 40 80 71 70 70 80 56 78 60 55

Claude Opus 4.6

80

62 57 83 67 40 78 66 60 67 77 60 80 70 62

Claude Sonnet 4.5

Accuracy(%)

60 89 80 53 35 78 63 57 50 93 52 78 40 57

GPT-5.4

70

55 80 80 53 25 78 69 47 57 77 56 80 55 62

GPT-5.2

60

35 77 77 40 45 70 66 57 70 87 60 85 50 60

DeepSeek R1

22 31 91 33 35 70 60 37 57 53 40 68 40 55

GPT-OSS-120B

50

28 37 77 50 15 60 51 50 53 63 48 75 35 52

Llama 4 Scout

25 34 71 30 10 70 60 40 47 73 36 75 55 52

Qwen3-14B

40

28 26 66 40 35 78 60 43 57 70 40 65 45 45

GPT-OSS-20B

30 40 74 27 20 65 54 20 40 83 52 68 30 48

Qwen3-32B

A1 A2 A3 A5 B1 B2 B3 B4 B5 C2 C3 C4 C5 D4

H×C Cell

- Figure 11: Strict accuracy across H × C cells for all 14 models. Cells A1 (H-prox × C-pres) and B1 (H-eff × C-pres) are consistently the hardest. Several models fall below 30% on these cells.

##### D.3 Accuracy by Constraint Family

100

90

71.6%

67.0% 62.6%

80

Accuracy(%)

44.4%

70

52.9%

60

50

40

Presence (C-pres)

Capability (C-cap)

Validity (C-val)

Scope (C-scope)

Procedural (C-proc)

- Figure 12: Strict accuracy by constraint family (mean ± range across 14 models). C-pres (presence) is hardest (mean: 44.4%), followed by C-proc (procedural, 52.9%). C-cap (capability, 71.6%) is easiest.

The constraint hierarchy (Table 10) is consistent across models. C-pres instances require inferring that an object must be physically co-located with a service—the same pattern identified in Study 1. C-proc instances require inferring temporal or procedural prerequisites (e.g., a store being closed, needing an appointment), which are similarly unstated. C-cap instances (e.g., cannot carry a sofa on foot) involve more concrete, visualisable constraints, which models appear to handle better.

Table 10: Strict accuracy by constraint family: mean, min, and max across 14 models.

Constraint Mean Min Max C-pres (Presence) 44.4% 20.0% 75.0% C-proc (Procedural) 52.9% 32.5% 67.5% C-scope (Scope) 62.6% 46.2% 77.7% C-val (Validity) 67.0% 56.8% 77.9% C-cap (Capability) 71.6% 52.4% 85.7%

##### D.4 Accuracy by Heuristic Family

Table 11: Strict accuracy by heuristic family: mean, min, and max across 14 models.

Heuristic Mean Min Max H-cost (Cost) 68.1% 54.1% 76.2% H-eff (Efficiency) 61.4% 45.7% 74.7% H-prox (Proximity) 59.1% 39.2% 74.3% H-sem (Semantic) 59.0% 46.7% 80.0%

Cost-based heuristics (H-cost) are the easiest to override, while proximity (H-prox) and semantic-match (H-sem) cues are the hardest. Proximity cues may be harder because distance-to-decision mappings are highly frequent in training data (as demonstrated by the sigmoid heuristic in Study 1). Semantic-match cues exploit category-level associations (e.g., “gas station” sounds car-related, so it should fix car problems), which are similarly deeply embedded in language model representations.

##### D.5 Heuristic Strength Analysis

Contrary to expectation, stronger heuristic cues do not reliably produce lower accuracy (Table 12). Mean strict accuracy is 62.8% for strong cues, 56.2% for medium, and 59.6% for weak—a non-monotonic pattern. This suggests that the failure is not simply a matter of being “overwhelmed” by a strong signal; even weak heuristic cues are sufficient to override constraint inference. The bottleneck appears to be in activating the constraint reasoning pathway, not in the competition between heuristic and constraint signals.

- Table 12: Strict accuracy by heuristic strength. No consistent gradient: even weak cues trigger override failures.

Strength Mean Min Max Strong 62.8% 49.4% 75.3% Medium 56.2% 42.3% 69.2% Weak 59.6% 30.8% 80.8%

D.6 Accuracy by Domain

- Table 13: Strict accuracy by scenario domain. Travel and medical scenarios are substantially harder, likely due to specialised procedural constraints.

Domain Mean Min Max Home 74.5% 61.1% 81.1% Digital 68.0% 54.8% 83.3% Work 66.1% 49.4% 78.7% Transportation 58.7% 41.4% 78.2% Medical 56.0% 23.3% 69.8% Shopping 55.4% 34.2% 68.4% Travel 41.4% 25.0% 62.5%

The domain breakdown reveals that scenarios involving specialised procedural knowledge (travel: visa requirements, booking prerequisites; medical: prescription requirements, appointment systems) are substantially harder than everyday scenarios (home, digital). The 33-point gap between the easiest (home, 74.5%) and hardest (travel, 41.4%) domain underscores that constraint inference difficulty increases with domain specificity.

### E Parametric Sweep Details

##### E.1 Per-Sweep Curves (Qwen3-4B)

20

| |H-cost × C-scope (Qwen3-4B)<br><br>Conflict Control<br><br>| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

H-eff × C-cap (Qwen3-4B)

25

10

20

s[>0:Shortcut]

s[>0:Shortcut]

0

15

10

−10

5

−20

0

−30

1minute5minutes15minutes30minutes 1hour 2hours 3hours 4hours 6hours 8hours

$0(free) $2 $5 $10 $20 $35 $50 $75 $100 $150 $200 $300 $500

H-prox × C-cap (Qwen3-4B)

H-sem × C-scope (Qwen3-4B)

20

10

s[>0:Shortcut]

s[>0:Shortcut]

10

0

0

−10

−10

−20

asmallconveniencestorearoadsideshop afuelstationagasstationthatsellscaraccessoriesagasstationagasstationwithanautosuppliessectionafull-servicegasstationwithacarcarecenter

50m 100m 200m 500m 800m 1km 2km 3km 5km 10km 25km 50km

- Figure 13: Parametric sweeps across four H × C combinations (Qwen3-4B). Orange: conflict; blue: control. Top-left: H-cost × C-scope—correct reasoning (curves distinct). Top-right: H-eff × C-cap—sigmoid failure (curves track). Bottom-left: H-prox × C-cap—correct reasoning. Bottom-right: H-sem × C-scope—semantic sigmoid.

##### E.2 Efficiency Sweep: Cross-Model Overlay

Qwen3-4B Qwen3-8B Qwen3-14B

Qwen3-32B

20

Conflictscores[>0:CarryMyself]

Qwen3.5-27B GPT-OSS-20B

15

10

5

0

−5

1minute 5minutes 15minutes 30minutes 1hour 2hours 3hours 4hours 6hours 8hours

Time saved by doing it yourself

Figure 14: H-eff × C-cap conflict curves for all six models. Qwen3-4B stays strongly positive (sigmoid failure); larger models (Qwen3-32B, Qwen3.5-27B) correctly shift negative. GPTOSS-20B hovers near zero.

##### E.3 Semantic Sweep: Cross-Model Overlay

Conflictscores[>0:GasStation]

Qwen3-4B Qwen3-8B Qwen3-14B

Qwen3-32B

10

Qwen3.5-27B GPT-OSS-20B

5

0

−5

−10

gasst. withanautosuppliessection afull-servicegasstationwithacarcarecenter

asmallconveni... aroadsideshop... afuelstation... agasstation gasst. thatsellscaraccessories

Gas station description (increasing car-relatedness →)

- Figure 15: H-sem × C-scope conflict curves for all six models. As the gas station description becomes more “car-related” (left to right), most models shift toward incorrectly recommending it for tire repair. Qwen3-4B shows the strongest semantic sigmoid; Qwen3.5-27B and Qwen3-32B remain closer to the decision boundary.

### F Inter-Rater Agreement on Cell Assignment

We sample 50 random HOB instances from the 500-instance set (stratified by cell to ensure coverage of all 15 populated cells). Three annotators (PhD students unaffiliated with the project, trained on a 1-page taxonomy reference) independently assign each instance to one of the 20 H×C cells (or mark “unclear / off-taxonomy”). Annotators worked blind to original cell labels and the paper’s claims.

Aggregate agreement. Fleiss’ κ across all 50 instances is 0.71 (substantial agreement on the Landis–Koch scale). After a single 30-minute calibration discussion focused on the C-scope vs. C-cap boundary (the most contested one, where service-offering vs. physical-means distinctions blur for borderline cases), a second-pass annotation on the same 50 items yields κ = 0.84. Of 50 instances, 38 received unanimous cell agreement; 9 received 2-of-3 majority agreement (resolved by majority); 3 were re-classified after calibration.

Per-cell breakdown. Cell-by-cell κ ranges from 0.55 (C-proc vs. C-val: preconditionviolation ambiguity) to 0.88 (H-cost × C-cap: cost cues for physical capability constraints are visually distinctive). The C-scope vs. C-cap boundary, which the reviewer flagged using a gas-station-tire example, had pre-calibration κ = 0.62 (6/50 disagreements) and post-calibration κ = 0.83 (1/50 disagreement). The disambiguating rule, codified in the second-pass annotation guide: C-scope if the service exists but its offering does not include the goal (gas station can deliver fuel but not tire repair); C-cap if the means itself cannot accomplish the goal (carrying a sofa on foot).

Implications. Inter-rater agreement of κ = 0.71 is in the substantial-to-strong range typical of cognitive-science taxonomies (Landis & Koch 1977); κ = 0.84 post-calibration is comparable to gold-standard NLP benchmarks (e.g., Penn Treebank constituent annotation at κ ≈ 0.83). The taxonomy is reproducible by independent annotators with a brief calibration step.

Table 14: Per-cell inter-rater agreement (Fleiss’ κ), pre- and post-calibration.

Cell pair Pre-cal κ Post-cal κ

C-scope vs. C-cap 0.62 0.83 C-proc vs. C-val 0.55 0.74 H-eff vs. H-cost 0.68 0.81 C-pres vs. C-cap 0.79 0.85 H-prox vs. H-eff 0.74 0.82

###### Overall (all 20 cells) 0.71 0.84

### G Gemini Thinking-Mode Ablation and CoT Baseline

##### G.1 Thinking-Mode Ablation on Gemini 3.1 Pro

The Gemini 3.1 Pro API enables an internal thinking mode by default. To test whether this internal deliberation explains the null effect of explicit goal-decomposition prompting, we re-run all 500 HOB instances (N=10 trials each) with thinking budget=0 (thinking explicitly disabled).

- Table 15: Gemini 3.1 Pro: thinking-mode ablation. Internal thinking accounts for 16.2pp; goal-decomposition recovers nearly all of it.

Configuration Strict Acc (%) ∆ Thinking ON, zero-shot 74.6 Thinking ON, + goal-decomposition 74.0 −0.6

Thinking OFF, zero-shot 58.4 −16.2 Thinking OFF, + goal-decomposition 71.2 +12.8

The pattern provides a clean double dissociation: (i) explicit prompting and internal thinking are substitutable routes to the same effect; (ii) neither is additive on top of the other (74.6 thinking-ON → 74.0 with extra GD, n.s.); (iii) removing one and adding the other approximately preserves performance. This is the strongest direct evidence for the inferencebottleneck account: the underlying knowledge is available, but some form of deliberationinternal or external—is required to activate it.

G.2 Per-Model CoT vs. Goal-Decomposition

- Table 16 reports the full comparison.

- Table 16: Per-model breakdown of zero-shot, CoT, and goal-decomposition (GD). GD outperforms generic CoT on every model except Gemini, whose internal thinking already performs the equivalent operation.

Model Zero-shot + CoT + GD ∆CoT ∆GD Gemini 3.1 Pro 74.6 75.8 74.0 +1.2 −0.6 GPT-5.4 65.8 70.2 74.8 +4.4 +9.0 Llama 4 Scout 51.2 55.0 57.8 +3.8 +6.6 Mean 63.9 67.0 68.9 +3.1 +5.0

### H DeepSeek R1 Thinking-Trace Audit

We re-collect DeepSeek R1 responses on a stratified sample of 50 HOB instances (10 per constraint family) using a modified collection script that preserves both reasoning content and content fields. Two annotators independently rate each trace on three binary dimensions:

- 1. Mentions constraint: the trace explicitly names the hidden feasibility constraint (e.g., “the car needs to be present”, “I can’t carry 500 lbs alone”).
- 2. Applies constraint: the trace uses the constraint to derive the answer (not merely mentions and ignores it).
- 3. Correct final answer: the trial’s verdict from the LLM judge.

Inter-annotator agreement on each dimension is κ ≥ 0.85.

- Table 17: DeepSeek R1 trace audit (50 instances). Spontaneous constraint mention strongly predicts correctness, but the model fails to mention the constraint in 36% of cases.

Trace pattern N Correct (%) Mentions and applies constraint 26 88.5 (23/26) Mentions but does not apply 6 16.7 (1/6) Does not mention constraint 18 44.4 (8/18) Total 50 64.0 (32/50)

The mention-and-apply vs. never-mention difference (88.5% vs. 44.4%) is significant by a two-sided Fisher’s exact test (p < 0.01); we use Fisher’s exact rather than McNemar here because the comparison is between disjoint groups of instances (those whose traces mention the constraint vs. those that do not), not a paired within-item comparison. The pattern is consistent with the inference-bottleneck account: (i) when the constraint is enumerated and used, correctness is high (88.5%); (ii) when not mentioned, the model relies on the heuristic and is correct only when the heuristic accidentally aligns with the constraint (44.4%, near chance for the C-pres-heavy subset); (iii) reasoning models do not reliably perform spontaneous constraint enumeration—the trace omits the constraint 36% of the time. Goal-decomposition prompting raises (iii) toward 100% by external scaffolding.

### I Temperature Ablation

To verify that the strict-accuracy ranking is not an artifact of stochastic decoding, we reevaluate three representative models—Gemini 3.1 Pro, GPT-5.4, and Llama 4 Scout—at three temperatures (T ∈ {0.0,0.3,0.7}) on a 100-instance random sample of HOB (N=10 trials each).

- Table 18: Strict accuracy across temperatures. Ranking is preserved (Spearman ρ > 0.97 between any two temperatures); strict accuracy increases slightly at lower temperatures, as expected with reduced sampling variance.

Model T = 0.0 T = 0.3 T = 0.7 ρ(0.0, 0.7) Gemini 3.1 Pro 76.2 75.4 74.6 0.98 GPT-5.4 67.1 66.2 65.8 0.97 Llama 4 Scout 52.8 51.9 51.2 0.98

Two observations: (i) Strict-accuracy ranking is invariant across temperatures (Spearman ρ > 0.97 for all pairwise comparisons across the three models). (ii) The absolute strict-accuracy values shift by at most 1.6pp from T = 0.0 to T = 0.7, indicating that the 10/10 criterion captures genuine reasoning reliability rather than sampling stability artifacts. These results confirm that the strict-accuracy metric is robust to decoding choices and that the crossmodel comparisons reported in the main text would hold under any reasonable temperature setting.

### J Reasoning vs. Non-Reasoning Model Breakdown

We classify the 14 evaluated models by whether they invoke explicit thinking by default: Reasoning models (n=6): DeepSeek R1, Gemini 3.1 Pro, GPT-5.4, GPT-5.2, Claude Opus 4.6, Grok 4.2 (reasoning). Non-reasoning models (n=8): Claude Sonnet 4.5, Kimi K2.5, Qwen3.527B, Llama 4 Scout, GPT-OSS-120B, GPT-OSS-20B, Qwen3-14B, Qwen3-32B.

Table 19: Aggregate comparison of reasoning vs. non-reasoning models on HOB.

Reasoning (n = 6) Non-reasoning (n = 8) ∆

Mean strict accuracy 67.6% 57.9% +9.7 Median strict accuracy 66.9% 51.7% +15.2 Mean minimal-pair gap −23.8 pp −13.7 pp −10.1 Mean MP gap (excl. GPT-OSS) −23.8 pp −22.5 pp −1.3 Mean explicitness gradient 15.7 pp 15.1 pp +0.6

The 9.7pp aggregate gap is largely confounded with general model capability: reasoning models in our sample tend to be more recent and higher-tier on capability rankings. A partial regression of strict accuracy on (Chatbot Arena Elo rank, reasoning-mode indicator) yields residual reasoning-mode effects of β = 1.8pp (p = 0.31, not significant) after controlling for Elo. Two direct same-tier contrasts further refute a pure reasoning-mode explanation: DeepSeek R1 (reasoning, 64.2%) underperforms Qwen3.5-27B (non-reasoning, 72.2%); Claude Opus 4.6 (reasoning, 68.0%) is matched by Kimi K2.5 (non-reasoning, 69.0%).

The minimal-pair gap does not separate cleanly by reasoning class once outliers are inspected. The raw non-reasoning mean (−13.7pp) is pulled toward zero by two modelsGPT-OSS-120B (+13.8) and GPT-OSS-20B (+11.0)—which are the only two of fourteen with a positive minimal-pair gap. Excluding these two outliers, the non-reasoning mean is −22.5 pp, statistically indistinguishable from the reasoning mean of −23.8 pp (Welch’s t-test, p = 0.78). We therefore make no claim that reasoning-mode reduces conservative bias. The GPT-OSS family’s atypical minimal-pair behaviour (positive gap; low base accuracy with notably higher pair accuracy) is worth separate investigation but lies outside the scope of this paper.

### K Statistical Testing

We match each test to the structure of its outcome variable, and avoid running tests on the binarised strict verdict (which collapses ten trial responses into one bit).

Continuous scores (Study 1). For the decision score s(x) we report paraphrase-level 95% bootstrap CIs and a paired bootstrap test of HDR > 1 (p < 0.001 for all six models).

Matched conditions (Study 2). The explicitness gradient (implicit vs. hint variants of a scenario) and the minimal-pair asymmetry (base vs. pair) are matched within scenario. We analyse them at three levels of granularity and find the same result throughout (Table 20): (i) a trial-level logistic regression on the individual binary responses with condition as a fixed effect and cluster-robust standard errors (clustered by model × scenario, absorbing the within-instance correlation of the ten trials and the base/pair pairing); (ii) a paired Wilcoxon signed-rank test on the per-instance pass rate (k/10); (iii) McNemar’s test on the binarised strict verdict, reported only for reference. We adopt (i)/(ii) as the reported tests; strict accuracy is used purely as a descriptive reliability metric.

Table 20: Matched-comparison significance at three levels of granularity (14 evaluated models). The effect is significant under all three; no conclusion relies on the binary collapse.

Test (outcome granularity) Minimal-pair Explicitness

Trial-level logistic (10 responses) OR = 0.53, p≈10−19 OR = 3.75, p≈10−16 Wilcoxon signed-rank (k/10) p≈2.5 × 10−19 p≈6.1 × 10−14 McNemar (strict 10/10, ref. only) p≈2 × 10−30 p≈2 × 10−11

matched units / trials 1,974 / 39,480 238 / 4,760

Between-group comparison (trace audit). The DeepSeek R1 trace audit compares correctness across disjoint groups of instances (traces that mention the constraint vs. those that do not), so it is unpaired; we use a two-sided Fisher’s exact test (p < 0.01), not McNemar.

