# arXiv:2602.12395v1[cs.CV]12Feb2026

What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis

Xirui Li1,∗, Ming Li1,∗, Tianyi Zhou2 1University of Maryland, 2Mohamed bin Zayed University of Artificial Intelligence

∗Co-first Author

Reinforcement learning (RL) with verifiable rewards has become a standard post-training stage for boosting visual reasoning in vision-language models, yet it remains unclear what capabilities RL actually improves compared with supervised fine-tuning as cold-start initialization (IN). End-to-end benchmark gains conflate multiple factors, making it difficult to attribute improvements to specific skills. To bridge the gap, we propose a Frankenstein-style analysis framework, including: (i) functional localization via causal probing; (ii) update characterization via parameter comparison; and (iii) transferability test via model merging. Instead, RL induces a consistent inference-time shift primarily in mid-to-late layers, and these mid-to-late refinements are both transferable (via merging) and necessary (via freezing) for RL gains. Overall, our results suggest that RL’s reliable contribution in visual reasoning is not a uniform enhancement of visual perception, but a systematic refinement of mid-to-late transformer computation that improves vision-to-reasoning alignment and reasoning performance, highlighting the limitations of benchmark-only evaluation for understanding multimodal reasoning improvements.

Date: February 16, 2026 Correspondence: Last authors at tianyi.david.zhou@gmail.com Project Page: https://github.com/tianyi-lab/Frankenstein

1 Introduction

Recent progress in large language models (LLMs) post-training for reasoning tasks has been driven by a two-stage paradigm: supervised finetuning (SFT) as an initialization stage (IN) followed by reinforcement learning (RL) with verifiable rewards (Guo et al., 2025; Zhang et al., 2025d; Wen et al., 2025). Compared to SFT-only post-training, RL leverages sparse reward feedback and delayed credit assignment, and has been shown to elicit strong reasoning behaviors with minimal supervision in language-only settings (Guo et al., 2025; Zhang et al., 2025d; Wen et al., 2025). Motivated by these successes, the same IN+RL1 paradigm has been increasingly adopted for vision-language models (VLMs) (Zhou et al., 2025a; Li et al., 2025a), where multiple works report substantial gains on visual reasoning benchmarks across spatial reasoning (Meng et al., 2025; Huang et al., 2025) and multimodal understanding (Deng et al., 2025b; Liu et al., 2025c).

Despite these gains, it remains unclear what is actually improved by RL in visual reasoning. Prior studies show that vision does not reliably benefit from increased inference length alone (Tian et al., 2025; Rahmanzadehgervi et al., 2024; Fu et al., 2025; Qin et al., 2025), and verbose reasoning can even exacerbate visually grounded errors by amplifying reliance on language priors (Chu et al., 2025; Fan et al., 2025; Liu et al., 2025a). Our experiments reinforce this uncertainty by showing that the end-to-end benchmark accuracy alone cannot distinguish improvements in vision, vision-to-reasoning alignment, and pure reasoning. At the same time, we observe the consistently increased attention from reasoning tokens to vision tokens under RL across different training recipes, suggesting a shared RL training effect whose functional role remains unclear. These two observations motivate us to a more concise question: What is consistently improved by RL for visual reasoning across training recipes?

1Unless otherwise specified, Base model, IN model, and RL model respectively refer to the baseline, supervised initialization, and reinforcement-learning models within a two-stage post-training pipeline.

|IN| |
|---|---|
| | |
|RL| |

###### Evaluation

RL:RL:IN

IN:RL:RL

Model Merging

IN:IN:RL

RL:IN:IN

IN:RL:RL

[Figure 3]

Update Norm

IN:RL:IN

RL:IN:RL

Parameter Analysis

Update Direction

Base

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Functional Localization via Causal Probing Update Characterization via Parameter Comparison

Late Layer Region

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

How many positive whole-number divisors does 196 have?

Update Energy

|Reasoning|
|---|

[Figure 17]

IN: more energy on Late RL: more energy on Mid

[Figure 18]

Mid Layer Region

First prime factorize $196=2^2\cdot7^2$...

[Figure 19]

[Figure 20]

IN

[Figure 21]

[Figure 22]

OCR, Counting, Grounding

Is there a cat in the image?

Update

IN: steeper rank spectrum on Early RL: steeper rank spectrum on Mid-Late

[Figure 23]

Early Layer Region

|Recognition|
|---|

Diversity

Yes

RL

Base

[Figure 24]

[Figure 25]

[Figure 26]

Transferability Test via Model Merging

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

RL demonstrates transferable

improvements on Mid-Late

IN RL IN RL IN:IN:RL IN:RL:RL IN:RL:IN RL:RL:IN RL:IN:IN RL:IN:RL

- Figure 1 Frankenstein-style Analysis Framework. The framework proceeds through three components: (1) functional localization via causal probing across transformer depth, (2) update characterization via parameter comparison to identify region-wise update pattern in post-training , and (3) transferability test via model merging, assessing whether the localized functionalities are preserved in layers.

###### Evaluation

IN:RL:RL

[Figure 32]

IN

Model Merging

IN:RL:IN

RL:IN:RL

IN:IN:RL

RL:IN:IN

IN:RL:RL

RL

We hereby adopt a Frankenstein-style analysis framework as shown in Figure 1 to trace where RL alters VLMs and what is altered across different training recipes. By analogy to Frankenstein’s construction, we analyze VLMs by disassembling them into functional regions, intervening on these components, and reassembling them to test the causal contribution of RL-induced changes. First, we perform Functional Localization via Causal Probing, localizing vision- and reasoning-related computations along transformer depth and establishing a coarse Early/Mid/Late functional regions. Second, we conduct Update Characterization via Parameter Comparison, showing that IN and RL differ systematically in both update magnitude and update geometry, with RL exhibiting refinements concentrated in mid-late layers. Finally, we run the Transferability Test via Model Merging: transplanting RL-refined regions of layers into IN models yields consistent improvements, primarily in vision-to-reasoning alignment and reasoning, indicating that the associated functional behaviors are modular and transferable.

In addition, we further perform Necessity Validation via Model Freezing during RL training. Freezing the parameters in mid or late regions largely removes RL gains, whereas freezing those in early regions has a much weaker effect, suggesting that mid-late refinement is a critical driver of RL improvements. Taken together, our Frankenstein-style analysis clarifies what is consistently learned under RL across training recipes and highlights the limitations of benchmark-only evaluation for understanding visual reasoning improvements in VLMs.

Key Findings:

- 1. Despite apparent gains on end-to-end visual reasoning benchmarks, fine-grained evaluation reveals that vision ability, language-side reasoning ability, and vision-to-reasoning alignment do not improve monotonically from the Base model to the IN model and then to the RL model.
- 2. Across diverse training recipes, RL consistently induces a shift in inference behavior, characterized by increased attention from reasoning tokens to visual tokens, primarily in mid-late transformer layers.
- 3. At the parameter level, RL exhibits a consistent and structured update pattern across recipes: refinements concentrated in mid-late layers. These refinements are transferable and contribute primarily to improved vision-to-reasoning alignment and reasoning performance.

- 2 Motivation

- 2.1 The Ambiguity on Visual Reasoning Improvements

The ambiguity of RL improvements is rooted in a fundamental asymmetry between reasoning and vision in current VLMs. Reasoning ability in language models is known to improve with extended inference, trajectory exploration, and delayed credit assignment, as demonstrated by chain-of-thought prompting and reinforcement learning for reasoning (Wei et al., 2022; Wang et al., 2022, 2025d; Zhou et al., 2025b).

100

###### OpenMMReasoner MMR1 Revisual

84.0

80.0

78.0 78.0

75

65.0 63.0

61.0 63.0

60.0 63.0

Score

59.2

57.0

55.1 57.0

56.0

55.0

53.8

53.0

53.0

52.0

50.0

50

49.0 45.9

45.9 47.2

47.0

46.0

46.0 46.0

46.0

45.9

44.0

42.0

38.0

38.0

38.0

25

0

BaseINRLBaseINRLBaseINRLBaseINRL BaseINRLBaseINRLBaseINRLBaseINRL BaseINRLBaseINRLBaseINRLBaseINRL

Average Benchmark Accuracy

Vision

Vision-to-Reasoning

Reasoning

Metric:

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 2 Average Benchmark Accuracy versus Fine-Grained Abilities (Vision, Reasoning, and Vision-to-Reasoning Alignment). The green arrows denote model post-training pipelines (Base Model → IN Model → RL Model) that exhibit monotonic performance gains, whereas the purple arrows indicate model groups that do not. Despite apparent improvements on visual reasoning benchmarks, fine-grained evaluation metrics reveal that vision ability and reasoning ability do not improve monotonically from the base model to the IN model and then to the RL model.

- Table 1 Fine-grained evaluation metrics targeting vision, vision-to-reasoning alignment, and pure reasoning ability. We let f(·) denote the VLM’s predicted answer given its inputs. And notation i refers to the original image, b to a black image of the same size, d to the textual description of the original image, and p to the prompt.

Metric Task Definition

Mvis General VQA N1 Nn=1 I f(in, pn) = yn ∧ f(bn, pn) ̸= yn Mv2r Math VQA N1 Nn=1 I f(in, pn) = yn ∧ f(bn, dn, pn) = yn Mrea Textual Math N1 Nn=1 I f(pn) = yn

Vision, by contrast, does not reliably benefit from extended inference alone. Additional tokens do not introduce new visual evidence (Tian et al., 2025), nor do they systematically resolve perceptual errors such as mislocalized objects, missing attributes, or incorrect counts (Rahmanzadehgervi et al., 2024; Fu et al., 2025; Qin et al., 2025). Moreover, extended language-side reasoning may even exacerbate visually grounded errors by amplifying reliance on linguistic priors when visual representations are imperfect (Chu et al., 2025; Fan et al., 2025; Liu et al., 2025a).

As a result, improvements on visual reasoning benchmarks alone do not reveal whether a model’s vision or reasoning capability has actually improved. Aggregate accuracy cannot distinguish among different sources of improvement: the same performance increase may arise from stronger reasoning over unchanged visual inputs, from changes in how reasoning attends to existing visual evidence, or from genuine improvements in visual representations.

- 2.2 Inconsistent Fine-grained Improvements

To analyze this ambiguity, we introduce three fine-grained metrics that disentangle visual reasoning: (i) visual perception, (ii) vision-to-reasoning alignment, and (iii) language-side reasoning (Table 1). This design helps separate functional gains from confounds such as hallucinated answers (Fan et al., 2025) and spurious reliance on visual inputs (Brown et al., 2025):

- • Vision (Mvis). We measure whether the image provides necessary information beyond language priors. Specifically, Mvis counts an example as correct only if the model answers correctly with the real image (i,p), but fails when the image is replaced by a black image (b,p). This criterion filters out cases that can be solved without vision and focuses on instances where visual evidence is causally used.
- • Vision-to-Reasoning alignment (Mv2r). To test whether perceptual evidence is consistently incorporated into reasoning, we compare two semantically matched inputs: the original image (i,p) and a textualized

version of the same visual content paired with a black image (b,d,p). Mv2r requires the model to answer correctly in both settings, indicating that the model can preserve the correct reasoning outcome when visual evidence is replaced by an equivalent description, rather than relying on image-specific shortcuts.

- • Reasoning (Mrea). Finally, Mrea isolates language-side reasoning by evaluating performance on text-only problems p without any visual input. This provides a control for improvements arising solely from strengthened reasoning.

We conduct evaluation on OpenMMReasoner (Zhang et al., 2025c), MMR1 (Leng et al., 2025), and Revisual (Chen et al., 2025c) families, as they include publicly available training recipes that (1) follow the dominant RL-based post-training paradigm and (2) provide reproducible checkpoints across training stages, enabling mechanistic comparison. For end-to-end evaluation, we also report average benchmark accuracy across MathVista (Lu et al., 2023), MathVerse (Zhang et al., 2024), MathVision (Wang et al., 2024), and LogicVista (Xiao et al., 2024).

- As shown in Figure 2, average benchmark accuracy exhibits a monotonic increase from Base models to IN models and then to RL models. However, these gains do not translate into consistent improvements in the

fine-grained abilities isolated by our metrics: neither vision (Mvis) nor standalone reasoning (Mrea) improves monotonically across training stages. This discrepancy highlights a key limitation of end-to-end evaluation: they conflate qualitatively different internal changes and cannot reveal which functional components are actually strengthened by RL.

Take-away: Despite apparent improvements on visual reasoning benchmarks, fine-grained evaluation reveals that vision and reasoning ability do not improve monotonically from the Base model to the IN model and then to the RL model.

2.3 Consistent Attention Patterns

0 5 10 15 20 25

Layer Index

0.02

0.04

0.06

0.08

0.10

AggregatedAttention

OpenMMReasoner

IN model

RL model

0 5 10 15 20 25

Layer Index

0.02

0.04

0.06

0.08

0.10

AggregatedAttention

MMR1

IN model

RL model

0 5 10 15 20 25

Layer Index

0.01

0.02

0.03

0.04

0.05

0.06

0.07

0.08

0.09

AggregatedAttention

Revisual

IN model

RL model

Figure 3 Aggregated Attention from Reasoning Tokens to Vision Tokens. Compared to IN models, there is more attention from reasoning tokens to vision tokens in RL models’ inference. The pattern is concentrated in later layers across training recipes, while absent in earlier layers.

In contrast to the inconsistent improvements observed on fine-grained metrics, we identify a consistent inference-time change induced by RL: increased attention from reasoning tokens to visual tokens. Formally, for transformer layer ℓ and attention head h, let A(ℓ,h) denotes the corresponding self-attention matrix, where each row sums to one. Given a set of reasoning tokens R and visual tokens V, we define the attention mass from reasoning to vision as

A(ℓ)(R→V) =

1 |H||R||V| h∈H i∈R j∈V

A(ijℓ,h). (1)

- As shown in Figure 3, RL modifies attention in a highly consistent manner across training recipes: attention from reasoning tokens to visual tokens increases in mid-late layers, while attention patterns in early layers remain largely unchanged.

Take-away: Compared to IN models, RL models exhibit stronger attention from reasoning tokens to vision tokens. The effect is concentrated in later layers across training recipes, while absent in earlier layers.

- 3 Frankenstein-Style Analysis

The contrast between inconsistent improvements on fine-grained metrics and consistent attention patterns across training recipes raises a key question: What is consistently improved by RL in visual reasoning across training recipes? We answer the question by a Frankenstein-style analysis framework that locates RL consistent

change at the granularity of transformer layers. Specifically, the framework consists of three components: (1) establishing a functional layer region partition that localizes vision- and reasoning-related computations (Section 3.1), (2) identify distinct and systematic patterns of RL updates on those layers (Section 3.2) and (3) validate the transferability of those behaviors through layer-wise model merging (Section 3.3).

- 3.1 Functional Localization via Causal Probing

recognition

GSM8K

grounding

MATH500

ocr

100

100

counting

Error Rate = (1 - Accuracy) (%)

80

80

Change Rate (%)

60

60

40

40

20

20

0

0

counting

MATH500

ocr

0

0

Dataset

5

Task

5

grounding

10

10

15

15

Layer

Layer

GSM8K

20

20

recognition

25

25

(a) Vision localization via vision-token swapping.

(b) Reasoning localization via layer skipping.

- Figure 4 Layer-wise functional localization of vision and reasoning. Both plots indicate the relative importance of each layer for the evaluated task. Vision-related functionalities are primarily associated with Early and Mid transformer layers, whereas reasoning-related computations are concentrated in Late layers.

To understand how RL improves visual reasoning, we begin by testing whether vision and reasoning can be functionally localized to distinct regions within VLMs, enabling the subsequent region-wise analyses. Specifically, we analyze the baseline model prior to any IN and RL training and apply minimal, targeted interventions to localize vision- and reasoning-related functionalities across transformer layers. This localization establishes a layer-wise functional reference frame that serves as the foundation for our analyses of RL training effects.

- 3.1.1 Localizing vision regions

To identify where visual information is functionally processed in the base model, we adopt a targeted visiontoken intervention strategy inspired by Shi et al. (2025). Rather than preserving in-distribution hidden states, our goal is to causally probe each layer’s contribution to the final prediction through controlled interventions. Following the practice in Shi et al. (2025), we construct paired images that differ in exactly one visual attribute (e.g., text difference in OCR task), while all other visual factors are held constant. Concrete examples are shown in Appendix 7. For each image pair (i,i′), we replace the visual tokens at a chosen transformer layer ℓ with those from the paired image. If swapping visual tokens at layer ℓ leads to a change in the model’s prediction, this indicates that layer ℓ functionally consumes visual information relevant to the swapped attribute. We quantify attribute sensitivity at layer ℓ using the change rate, defined as

1 N

Change Rate(ℓ) =

N

I f(i(nℓ),pn) ̸= f(i′n(ℓ),pn) , (2)

n=1

where f(·) denotes the model’s predicted answer, pn is the prompt, and i(nℓ) and i′n(ℓ) denote inputs in which visual tokens at layer ℓ are sourced from the original and paired images, respectively. Importantly, we interpret change rates comparatively across layers and visual attributes, rather than as absolute measures of robustness. Under this interpretation, structured peaks in the change rate indicate layers at which the model actively processes the corresponding visual information.

###### OpenMMReasoner

###### MMR1

###### Revisual

0.001225

0.00082

0.00065

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |IN<br><br>|Update| | | | |
| |RL|Update| | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.034

IN Update

IN Update

0.036

RL Update

RL Update

0.001200

0.11

0.00080

0.032

0.034

0.00060

0.001175

0.00078

###### RLUpdateNorm

###### RLUpdateNorm

###### RLUpdateNorm

INUpdateNorm

INUpdateNorm

INUpdateNorm

0.030

0.10

0.032

0.001150

0.00076

0.00055

0.028

0.030

0.001125

0.00074

0.09

0.028

0.026

0.00072

0.001100

0.00050

0.026

0.00070

0.001075

0.08

0.024

0.024

0.00068

0.001050

0.00045

0.022

0.07

0.022

0.00066

0.001025

0 5 10 15 20 25

0 5 10 15 20 25

0 5 10 15 20 25

Layer Index

Layer Index

Layer Index

- Figure 5 Layer-wise parameter update norms comparison between IN and RL. Per-layer Frobenius norms of parameter updates for IN (solid) and RL (dashed). Both training stages concentrate on optimization in the Mid layers, while RL exhibits a distinct redistribution of update magnitude compared to IN.

Based on the results demonstrated in Figure 4, we summarize the localized functional patterns using a coarse partition of the transformer layers into Early, Mid, and Late layers, each comprising one third of the layers, following prior region-wise analyses of transformer representations (e.g., Gurnee et al. (2025)). This abstraction allows us to focus on relative functionality rather than precise layer indices. Under this partition, vision functionality exhibits a clear stratification, consistent with prior work (Chen et al., 2025b), suggesting that vision and language-based reasoning can be decoupled within VLMs. Simple vision processing, such as recognition, is concentrated primarily in the Early layer. Vision functionality that requires more processing, including OCR, grounding, and counting, needs contributions from Early to Mid layers.

- 3.1.2 Localizing reasoning regions

Unlike vision, reasoning computations do not correspond to a distinct or explicitly tagged input that can be directly manipulated. As a result, token-level swapping is not applicable for localizing reasoning, since the contribution of individual reasoning tokens to the final answer is distributed and context-dependent. We therefore employ a layer-wise causal ablation strategy, following prior analyses of mathematical reasoning in language models (Nepal et al., 2025). Rather than perturbing the input, this intervention directly removes the contribution of a specific transformer layer while preserving the computations of all remaining layers. Specifically, for each transformer layer, we perform layer skipping by setting transformer layer’s input as output, allowing us to assess its causal contribution to reasoning performance. We evaluate these interventions on textual reasoning-heavy benchmarks, including mathematical reasoning tasks such as GSM8K (Cobbe et al., 2021) and MATH500 (Lightman et al., 2023). For each ablated layer, we measure the resulting error rate, defined as 1 − Acc, to quantify its contribution to the reasoning process. A substantial increase in error rate indicates that the ablated layer plays a critical role in reasoning computation. Additional analyses and examples of reasoning functionality localization are provided in Appendix D.3.

In contrast to vision functionality located in earlier layers, reasoning functionality localization reveals a complementary pattern: layer-wise skipping ablation shows that the majority of the reasoning process is concentrated in the Late layers, with comparatively minor dependence on early layers.

We have localized vision and reasoning functionality at the granularity of layers, but emphasize that these regions only serve as a functional reference frame rather than precise architectural boundaries. All subsequent analyses of decoupling and RL-based post-training effects are interpreted with respect to this coarse EarlyMid-Late functionality partition.

Take-away: Early layers primarily support simple visual processing, Mid layers handle higher-level visual information, and Late layers are more heavily involved in reasoning computations.

- 3.2 Update Characterization via Parameter Comparison

Having established a functional reference frame that localizes vision and reasoning behaviors, we next examine whether RL updates exhibit distinct and systematic patterns within these localized regions. Recent studies

OpenMMReasoner MMR1 Revisual

###### IN update

###### IN update

###### IN update

Rank 10 trend Rank 20 trend Rank 30 trend Rank 40 trend

Rank 10 trend Rank 20 trend Rank 30 trend Rank 40 trend

Rank 10 trend Rank 20 trend Rank 30 trend Rank 40 trend

0.1

0.1

0.0

0.0

0.0

Log

Log

Log

0.1

0.1

0.2

(Norm. SV)

(Norm. SV)

(Norm. SV)

0.2

0.2

0.4

0.3

0.3

0.6

0.4

0.4

0.5

0.5

0.8

0

0

0

0

0

0

5

5

5

10

10

10

SingularValueRank

SingularValueRank

SingularValueRank

10

10

10

20

20

20

Layer

Layer

Layer

15

15

15

30

30

30

20

20

20

40

40

40

25

25

25

50

50

50

###### RL update

###### RL update

###### RL update

Rank 10 trend Rank 20 trend Rank 30 trend Rank 40 trend

Rank 10 trend Rank 20 trend Rank 30 trend Rank 40 trend

Rank 10 trend Rank 20 trend Rank 30 trend Rank 40 trend

0.0

0.0

0.0

Log

Log

Log

0.2

0.2

0.2

(Norm. SV)

(Norm. SV)

(Norm. SV)

0.4

0.4

0.4

0.6

0.6

0.6

0.8

0.8

- 0 5

10

15

20

25

Layer

- 1.0

0.8

- 0 5

10

15

20

25

Layer

- 1.0

0

0

0

0

5

10

10

10

SingularValueRank

SingularValueRank

SingularValueRank

10

20

20

20

Layer

15

30

30

30

20

40

40

40

25

50

50

50

- Figure 6 Singular value spectra of parameter updates. Each panel visualizes the log-normalized singular value spectrum of the layer-wise update matrix. Color intensity reflects the steepness of the spectral decay. Compared to IN, RL updates exhibit a consistently steeper spectral decay in the Mid-Late layers, indicating that optimization is concentrated in a smaller number of dominant directions rather than being diffusely distributed.

have shown that RL can affect reasoning behavior in ways that differ qualitatively from its IN (Zhu et al., 2025). Motivated by this observation, we characterize the post-training changes through the geometry of parameter updates, focusing on their magnitude and structure across layers.

For each training recipe, we compute layer-wise parameter updates ∆W(ℓ) = Wtrained(ℓ) − Wbase(ℓ) for each transformer layer ℓ. We characterize these updates using two complementary geometric measures: update

energy and diversity.

- 3.2.1 Update Energy

To quantify update energy, we compute the Frobenius norm ∥∆W(ℓ)∥F for each layer, which measures the total optimization magnitude accumulated during post-training. Figure 5 shows that both IN and RL concentrate the majority of their update energy in the Mid layers, consistent with their role in bridging visual perception and high-level reasoning.

- 3.2.2 Update Diversity To observe the update diversity, we also analyze the singular value spectrum of ∆W(ℓ),

∆W(ℓ) = Udiag(σ1,...,σr)V⊤, (3)

which characterizes how optimization energy is distributed across update directions. For visualization, we normalize the singular values by the largest singular value of each layer and plot their logarithm, log(σi/σ1). This normalization maps the dominant update direction to 0 and measures all other directions relative to it on a log scale, removing scale differences across layers while preserving comparative decay structure. Under this representation, a steeper decay in the spectrum indicates that optimization is concentrated in a small number of dominant directions, corresponding to more focused, low-dimensional parameter refinements, whereas a flatter spectrum reflects more diffuse updates spread across many directions. As shown in Figure 6, each

- Table 2 Frankenstein-style model merging. We report vision, reasoning, and vision-to-reasoning alignment scores for Base, IN, and RL checkpoints, as well as hybrid models constructed by transferring training-induced changes across Early, Mid, and Late transformer layers. We denote the performance gap between merged models and IN with green on positive values and purple on the rest. Across training recipes, when RL and RL are kept, merged models demonstrate a consistent pattern of performance improvement, indicating RL consistent refinements in the Mid-Late layers.

Region-wise Merged Models

Ability Base IN IN : RL : RL IN : IN : RL RL : RL : IN RL : IN : IN IN : RL : IN RL : IN : RL RL

Training Recipe – OpenMMReasoner (Zhang et al., 2025c) Vision 38.0 47.0 42.0 (-5.0) 48.0 (+1.0) 45.0 (-2.0) 45.0 (-2.0) 44.0 (-3.0) 40.0 (-7.0) 42.0

- Vision-to-Reasoning 46.0 55.0 59.0 (+4.0) 58.0 (+3.0) 63.0 (+8.0) 57.0 (+2.0) 58.0 (+4.0) 55.0 (-0.0) 61.0 Reasoning 63.0 78.0 81.0 (+3.0) 73.0 (-5.0) 78.0 (-0.0) 73.0 (-5.0) 80.0 (+2.0) 73.0 (-5.0) 78.0

Training Recipe – MMR1 Leng et al. (2025)

Vision 38.0 44.0 37.0 (-7.0) 41.0 (-3.0) 45.0 (+1.0) 45.0 (+1.0) 41.0 (-3.0) 43.0 (-1.0) 50.0 V-to-R 46.0 46.0 51.0 (+5.0) 50.0 (+4.0) 50.0 (+4.0) 56.0 (+10.0) 52.0 (+6.0) 57.0 (+13.0) 54.0 Reasoning 63.0 60.0 61.0 (+1.0) 64.0 (+4.0) 64.0 (+4.0) 79.0 (+19.0) 56.0 (-4.0) 72.0 (+12.0) 62.0

Training Recipe – Revisual Chen et al. (2025c) Vision 38.0 40.0 42.0 (+2.0) 42.0 (+2.0) 41.0 (+1.0) 37.0 (-3.0) 40.0 (-0.0) 46.0 (+6.0) 35.0

- Vision-to-Reasoning 46.0 56.0 59.0 (+3.0) 58.0 (+2.0) 58.0 (+2.0) 56.0 (-0.0) 60.0 (+4.0) 51.0 (-5.0) 59.0 Reasoning 63.0 81.0 85.0 (+4.0) 84.0 (+3.0) 84.0 (+3.0) 85.0 (+4.0) 86.0 (+5.0) 88.0 (+7.0) 88.0

panel visualizes the log-normalized singular value spectrum of the per-layer update matrix. To demonstrate robustness across the spectrum, we additionally plot spectra at every tenth rank, which exhibit consistent qualitative trends across both layers and ranks. Compared to IN, RL updates display a markedly steeper spectral decay in the Mid-Late layers, indicating that optimization energy is concentrated in a smaller number of dominant directions within these regions.

Take-away: Both IN and RL impose high norm update in Mid layers. However, IN and RL impose complementary optimization diversity: RL applies less diverse refinements to Mid-Late layers.

- 3.3 Transferability Test via Model Merging

The functional localization and update analysis above establish where vision and reasoning are implemented and how RL modifies different layers of the model. However, these analyses alone cannot determine whether the observed RL effects reflect region-specific functional changes that only emerge in an end-to-end setting. We hereby ask whether the improvements in vision, reasoning, and vision-to-reasoning alignment associated with RL are transferable, i.e., whether they can be preserved when parameters are updated in specific regions transplanted into models without such training.

To answer this question, we employ model merging as a controlled intervention. Following the functional partition identified in Section 3.1, we perform model merging at the granularity of the Early, Mid, and Late regions. Hybrid models are constructed by enumerating all region-wise combinations of Early, Mid, and Late regions drawn from the IN and RL checkpoints, and by directly copying the complete parameter state of all transformer layers within each selected region (see Appendix E for implementation details). This operation transfers all parameters associated with the selected transformer layers, including any architecturally integrated vision encoder and projection layers, while leaving all other components unchanged.

Across all evaluated training recipes in Table 2, model merging reveals a consistent and interpretable pattern in how RL-induced behaviors transfer across models. Hybrid models that preserve RL-refined layers in the Mid-Late layers (especially merged model IN:RL:RL) consistently retain vision-to-reasoning alignment and reasoning improvements relative to the IN model. In contrast, alternative region combinations do not exhibit consistent or uniform improvements across training recipes, and several configurations lead to degraded performance.

Together, these results indicate that RL-induced refinements in the Mid-Late layers encode transferable functional behaviors that can be preserved through model merging. Notably, the retained gains primarily manifest as improved vision-to-reasoning alignment and reasoning, consistent with the observation of concentrated refinements in Mid-Late layers discussed in Section 3.2.

- Table 3 Parameter freezing during RL training. We compare fine-grained and benchmark metrics for standard RL training and RL training with parameters in one transformer region frozen. Freezing the Late layers leads to a pronounced drop in reasoning performance and vision-to-reasoning alignment, whereas freezing earlier layers has a smaller effect.

Fine-grained Metrics Benchmark Metrics

Vision (Mvis) Vision-to-Reasoning (Mv2r) Reasoning (Mrea) MathVista MathVision MathVerse

Original Training Recipe

IN Model (Zhang et al., 2025c) 34.0 21.0 26.0 46.5 18.4 37.0 RL Model (Zhang et al., 2025c) 33.0 29.0 34.0 48.1 14.1 37.8

Training with Frozen Blocks

RL Model - Frozen Early Block 35.0 31.0 36.0 48.2 21.0 34.5 RL Model - Frozen Mid Block 25.0 29.0 38.0 46.5 15.5 35.7 RL Model - Frozen Late Block 30.0 27.0 34.0 47.9 16.8 35.0

Take-away: Model merging confirms consistent RL refinements in Mid-Late layers, which consistently improve vision-to-reasoning alignment and reasoning capabilities.

- 3.4 Summary of Region-wise RL Effects

Through our proposed Frankenstein-style framework, we find that RL-induced changes in VLMs are localized to the Mid-Late transformer layers and exhibit concentrated refinements. Region-wise model merging further shows that preserving these RL-refined layers consistently retains improvements in vision-to-reasoning alignment and reasoning across training recipes. Together, these results across training recipes indicate that RLs consistently contribute to Mid-Late layers in vision-to-reasoning alignment and reasoning behaviors.

- 4 Necessity Validation via Model Freezing

The Frankenstein-style analysis in Section 3 shows that across different training recipes, RL consistently demonstrates concentrated refinements on Mid-Late layers on vision-to-reasoning and reasoning capabilities. However, region-wise transferability alone does not establish whether refinement of these layers is required for RL gains to emerge during training.

To validate the region-wise RL effects identified above, we perform targeted training-time interventions via region-wise parameter freezing during RL training (see Appendix F for training details). Following the partition identified in Section 3.1, we freeze one region of layers at a time while allowing the remaining to be optimized, keeping all other training settings identical. This design enables a necessary-condition test: if RL improvements are causally mediated by refinement on Mid-Late layers, then preventing RL from updating the Mid-Late layers should mitigate these gains, whereas constraining earlier layers should have a markedly smaller effect.

As demonstrated in Table 3, freezing the Late layers during RL training results in substantial degradation across both fine-grained and benchmark metrics. In contrast, freezing the Early and leaving Mid-Late layers intact yields better performance than the rest of the models, including IN and RL. These results demonstrate that Mid-Late refinement is not merely correlated with RL improvements, but it is a necessary component for achieving them.

Take-away: RL gains depend causally on Mid-Late refinement: Preventing RL from updating the Late region during training eliminates improvements in both fine-grained and benchmark metrics.

- 5 Related Work

- 5.1 RL and Its Analysis

RL has emerged as a powerful post-training paradigm for improving the reasoning capabilities of large language models. GRPO, introduced in DeepSeekMath (Shao et al., 2024) and later scaled in DeepSeek-R1 (Guo et al.,

2025), forms the basics of many recent advances. Subsequent variants, including DAPO (Yu et al., 2025b), GSPO (Zheng et al., 2025), GFPO (Shrivastava et al., 2025), and GDPO (Liu et al., 2026), further refine the stability and efficiency of RL training.

Most prior analyses of RL focus on end-to-end performance metrics such as pass@k or accuracy improvements (Yue et al., 2025; Wen et al., 2025), treating the model as a black box. In contrast, only a limited number of works (Mukherjee et al., 2025; Zhu et al., 2025) investigate RL from a parameter-space perspective, examining where updates are localized and how their geometric structure differs from supervised finetuning. Our work builds on this emerging line of inquiry by explicitly analyzing the location and geometry of RL updates in VLMs.

- 5.2 Improvement of Visual Reasoning in VLMs

Early work demonstrated that visual reasoning in VLMs can be enhanced through Chain-of-Thought finetuning (Zhang et al., 2023; Wei et al., 2022). More recent R1-style training pipelines show that such reasoning behaviors can be further amplified using RL; we refer readers to the Appendix B for a comprehensive discussion of this line of work. Alternative approaches achieve similar gains by model composition (Chen et al., 2025b), vision tokens (Bigverdi et al., 2025), explicit grounding (Sarch et al., 2025; Zhang et al., 2025a), or multi-agent framework (Jia et al., 2025).

In parallel, several studies examine how post-training affects visual processing itself. Notably, Song et al. (2025) shows that RL can improve the vision encoder in VLMs more effectively than SFT, suggesting that RL updates may not be confined to purely linguistic or reasoning components.

### 6 Conclusion

In this work, we address the ambiguity underlying RL-based improvements in visual reasoning by asking what is consistently changed by RL across training recipes. Rather than relying solely on end-to-end benchmarks, we adopt a Frankenstein-style analysis framework that decomposes VLMs at the granularity of transformer layers and probes their functional roles. Through causal localization, update characterization, and region-wise model merging, we show that RL does not uniformly improve visual perception or standalone reasoning. Instead, RL is consistently associated with structured refinements in middle and late transformer layers, reflected behaviorally in improved vision-to-reasoning alignment and reasoning capabilities. This perspective clarifies how RL alters VLM behavior beyond what aggregate accuracy reveals and provides a principled framework for diagnosing improvements in visual reasoning across training recipes.

References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025a.

Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231, 2025b.

Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3836–3845, 2025.

Ellis Brown, Jihan Yang, Shusheng Yang, Rob Fergus, and Saining Xie. Benchmark designers should "train on the test set" to expose exploitable non-visual shortcuts. arXiv preprint arXiv:2511.04655, 2025.

Liang Chen, Hongcheng Gao, Tianyu Liu, Zhiqi Huang, Flood Sung, Xinyu Zhou, Yuxin Wu, and Baobao Chang. G1: Bootstrapping perception and reasoning abilities of vision-language model via reinforcement learning. arXiv preprint arXiv:2505.13426, 2025a.

Shiqi Chen, Jinghan Zhang, Tongyao Zhu, Wei Liu, Siyang Gao, Miao Xiong, Manling Li, and Junxian He. Bring reason to vision: Understanding perception and reasoning through model merging. arXiv preprint arXiv:2505.05464, 2025b.

Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025c.

Zhangquan Chen, Xufang Luo, and Dongsheng Li. Visrl: Intention-driven visual perception via reinforced reasoning. arXiv preprint arXiv:2503.07523, 2025d.

Xu Chu, Xinrong Chen, Guanyu Wang, Zhijie Tan, Kui Huang, Wenyu Lv, Tong Mo, and Weiping Li. Qwen look again: Guiding vision-language reasoning models to re-attention visual information. arXiv preprint arXiv:2505.23558, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, and Yu Kang. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065, 2025a.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint arXiv:2503.17352, 2025b.

Chenrui Fan, Ming Li, Lichao Sun, and Tianyi Zhou. Missing premise exacerbates overthinking: Are reasoning models losing critical thinking skill? arXiv preprint arXiv:2504.06514, 2025.

Stephanie Fu, Tyler Bonnen, Devin Guillory, and Trevor Darrell. Hidden in plain sight: Vlms overlook their visual representations. arXiv preprint arXiv:2506.08008, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Wes Gurnee, Emmanuel Ameisen, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, and Joshua Batson. When models manipulate manifolds: The geometry of a counting task. Transformer Circuits Thread, 2025. https://transformer-circuits.pub/2025/linebreaks/index.html.

Minjie Hong, Zirun Guo, Yan Xia, Zehan Wang, Ziang Zhang, Tao Jin, and Zhou Zhao. Apo: Enhancing reasoning ability of mllms via asymmetric policy optimization. arXiv preprint arXiv:2506.21655, 2025.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Hongrui Jia, Chaoya Jiang, Shikun Zhang, and Wei Ye. Decoupling reasoning and perception: An llm-lmm framework for faithful visual reasoning. arXiv preprint arXiv:2509.23322, 2025.

Qing Jiang, Xingyu Chen, Zhaoyang Zeng, Junzhi Yu, and Lei Zhang. Rex-thinker: Grounded object referring via chain-of-thought reasoning. arXiv preprint arXiv:2506.04034, 2025.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.

Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang, Hang Zhang, Xin Li, Lidong Bing, et al. Mmr1: Enhancing multimodal reasoning with variance-aware sampling and open resources. arXiv preprint arXiv:2509.21268, 2025.

Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, et al. Perception, reason, think, and plan: A survey on large multimodal reasoning models. arXiv preprint arXiv:2505.04921, 2025a.

Zongzhao Li, Zongyang Ma, Mingze Li, Songyou Li, Yu Rong, Tingyang Xu, Ziqi Zhang, Deli Zhao, and Wenbing Huang. Star-r1: Spatial transformation reasoning by reinforcing multimodal llms. arXiv preprint arXiv:2505.15804, 2025b.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. arXiv preprint arXiv:2505.21523, 2025a.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, YuChiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026.

Zhiyuan Liu, Yuting Zhang, Feng Liu, Changwang Zhang, Ying Sun, and Jun Wang. Othink-mr1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning. arXiv preprint arXiv:2503.16081, 2025b.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025c.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models. arXiv preprint arXiv:2505.11711, 2025.

Aadim Nepal, Safal Shrestha, Anubhav Shrestha, Minwu Kim, Jalal Naghiyev, Ravid Shwartz-Ziv, and Keith Ross. Layer importance for mathematical reasoning is forged in pre-training and invariant after post-training. arXiv preprint arXiv:2506.22638, 2025.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, and Xudong Wang. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025.

Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Proceedings of the Asian Conference on Computer Vision, pages 18–34, 2024.

Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning. arXiv preprint arXiv:2505.23678, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Cheng Shi, Yizhou Yu, and Sibei Yang. Vision function layer in multimodal llms. arXiv preprint arXiv:2509.24791, 2025.

Vaishnavi Shrivastava, Ahmed Awadallah, Vidhisha Balachandran, Shivam Garg, Harkirat Behl, and Dimitris Papailiopoulos. Sample more to think less: Group filtered policy optimization for concise reasoning. arXiv preprint arXiv:2508.09726, 2025.

Junha Song, Sangdoo Yun, Dongyoon Han, Jaegul Choo, and Byeongho Heo. Rl makes mllms see better than sft. arXiv preprint arXiv:2510.16333, 2025.

Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.

Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, Fabian Waschkowski, Lukas Wesemann, Peter Tu, and Jing Zhang. More thought, less accuracy? on the dual nature of reasoning in vision-language models. arXiv preprint arXiv:2509.25848, 2025.

Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, et al. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025a.

Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025b.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

Peiyu Wang, Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang, Yunzhuo Hao, et al. Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning. arXiv preprint arXiv:2504.16656, 2025c.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025d.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025e.

Zhiqiang Wang, Pengbin Feng, Yanbin Lin, Shuzhang Cai, Zongao Bian, Jinghua Yan, and Xingquan Zhu. Crowdvlm-r1: Expanding r1 ability to vision language model for crowd counting using fuzzy group relative policy reward. arXiv preprint arXiv:2504.03724, 2025f.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, et al. R1-sharevl: Incentivizing reasoning capability of multimodal large language models via share-grpo. arXiv preprint arXiv:2505.16673, 2025.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954,

- 2025a.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476,

- 2025b.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Chi Zhang, Haibo Qiu, Qiming Zhang, Yufei Xu, Zhixiong Zeng, Siqi Yang, Peng Shi, Lin Ma, and Jing Zhang. Perceptual-evidence anchored reinforced learning for multimodal reasoning. arXiv preprint arXiv:2511.18437, 2025a.

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025b.

Kaichen Zhang, Keming Wu, Zuhao Yang, Bo Li, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, and Lidong Bing. Openmmreasoner: Pushing the frontiers for multimodal reasoning with an open and general recipe. arXiv preprint arXiv:2511.16334, 2025c.

Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025d.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, and Minghui Qiu. Reinforced mllm: A survey on rl-based reasoning in multimodal large language models. arXiv preprint arXiv:2504.21277, 2025a.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s" aha moment" in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025b.

Hanqing Zhu, Zhenyu Zhang, Hanxian Huang, DiJia Su, Zechun Liu, Jiawei Zhao, Igor Fedorov, Hamed Pirsiavash, Zhizhou Sha, Jinwon Lee, et al. The path not taken: Rlvr provably learns off the principals. arXiv preprint arXiv:2511.08567, 2025.

## Appendix

### A Limitations

Our study has several limitations that point to directions for future work. First, we focus on the dominant IN+RL-style post-training paradigm rather than direct RL from scratch. While representative of current practice, our findings may not directly generalize to alternative RL training recipes. Second, our analysis is limited to VLMs aim to improve visual reasoning. The functional decomposition identified here may not transfer to other task domains or modalities. Finally, all experiments are conducted on models from the Qwen series (Bai et al., 2025a) (to the best of our knowledge, most training recipes are all based on the Qwen series). Although the proposed framework is model-agnostic in principle, its applicability to models without clearly separable functional regions remains to be explored.

- Table 4 Training Recipes and Checkpoint Availability of RL-Based Visual Reasoning Models. We compare recently proposed RL-based post-training methods along key dimensions, including base model, training pipeline, reinforcement learning algorithm, training domain, claimed ability gains, and checkpoint availability. Although many methods share highly similar training recipes, only a small subset of these works sufficiently full-fill checkpoints to support controlled, stage-wise and block-level analysis. We therefore focus on the gray-highlighted models in this work, which follow the dominant RL training recipe while enabling reproducible and mechanistic comparison across training stages.

Name Date Base Model Pipeline RL Algorithm Training Domain Claimed Gains All CKPT Availability

Visual-RFT (Liu et al., 2025c) 25.03 Qwen-VL IN + RL GRPO Vision Task Vision ✘ MM-Eureka (Meng et al., 2025) 25.03 Intern-VL RL GRPO Math Visual Reasoning ✘ Vision-R1 (Huang et al., 2025) 25.03 Qwen-VL IN + RL GRPO Math Visual Reasoning ✘ VisualThinker (Zhou et al., 2025b) 25.03 Qwen-VL IN + RL GRPO Vision Task Vision ✘ Curr-ReFT (Deng et al., 2025a) 25.03 Qwen-VL IN + RL GRPO Vision Task + Math Visual Reasoning ✘ VisRL (Chen et al., 2025d) 25.03 Qwen-VL, LLaVA IN + RL DPO Vision Task Vision ✘ LMM-R1 (Peng et al., 2025) 25.03 Qwen-VL RL PPO Math Visual Reasoning ✘ R1-Onevision (Yang et al., 2025) 25.03 Qwen-VL IN + RL GRPO Math Visual Reasoning ✘ Skywork R1V (Wang et al., 2025c) 25.03 QwQ RL MPO + GRPO Math Visual Reasoning ✘ R1-VL (Zhang et al., 2025b) 25.03 Qwen-VL RL GRPO Math Visual Reasoning ✘ OThink-MR1 (Liu et al., 2025b) 25.03 Qwen-VL RL GRPO-D Vision Task + Math Visual Reasoning ✘ OpenVLThinker (Deng et al., 2025b) 25.03 Qwen-VL IN + RL (iterative) GRPO Vision Task + Math Visual Reasoning ✘ Reason-RFT (Tan et al., 2025) 25.03 Qwen-VL IN + RL GRPO Vision Task Vision ✔ CrowdVLM-R1 (Wang et al., 2025f) 25.04 Qwen-VL RL GRPO Vision Task Vision ✘ VLM-R1 (Shen et al., 2025) 25.04 Qwen-VL IN + RL GRPO Vision Task Vision ✘ Perception-R1 (Yu et al., 2025a) 25.04 Qwen-VL RL GRPO Vision Task Vision ✔ VLAA-Thinker (Yu et al., 2025a) 25.04 Qwen-VL IN + RL GRPO Vision Task + Math Visual Reasoning ✘ VL-Rethinker (Wang et al., 2025a) 25.04 Qwen-VL IN + RL GRPO Vision Task + Math Visual Reasoning ✘ G1 (Chen et al., 2025a) 25.05 Qwen-VL RL GRPO Vision Task Vision ✘ UniVG-R1 (Bai et al., 2025b) 25.05 Qwen-VL RL IN + GRPO Vision Task Vision ✘ STAR-R1 (Li et al., 2025b) 25.05 Qwen-VL RL GRPO Vision Task Vision ✘ Pixel Reasoner (Wang et al., 2025b) 25.05 Qwen-VL IN + RL GRPO Vision Task Vision ✔ R1-ShareVL (Yao et al., 2025) 25.05 Qwen-VL RL Share-GRPO Vision Task + Math Visual Reasoning ✘ SRPO (Wan et al., 2025) 25.06 Qwen-VL IN + RL SRPO Vision Task + Math Visual Reasoning ✘ Rex-Thinker (Jiang et al., 2025) 25.06 Qwen-VL IN + RL GRPO Vision Task Vision ✘ ReVisual-R1 (Chen et al., 2025c) 25.06 Qwen-VL IN + RL GRPO Vision Task + Math Visual Reasoning ✔ APO (Hong et al., 2025) 25.06 Qwen-VL RL APO Vision Task + Math Visual Reasoning ✔ PAPO (Wang et al., 2025e) 25.07 Qwen-VL RL PAPO Vision Task + Math Visual Reasoning ✔ MMR1 (Leng et al., 2025) 25.09 Qwen-VL IN + RL GRPO Vision Task + Math Visual Reasoning ✔ OpenMMReasoner (Zhang et al., 2025c) 25.11 Qwen-VL IN + RL GRPO Vision Task + Math Visual Reasoning ✔

### B More Related Work

- B.1 RL-based Visual Reasoning Improvements

Recent work has proposed a large number of RL-based post-training methods for improving visual reasoning in VLMs. As summarized in Table 4, however, this apparent diversity masks a highly concentrated design space. Most methods adopt the same underlying base model, follow a similar two-stage pipeline consisting of instruction tuning followed by reinforcement learning (IN + RL), and rely on closely related policy optimization algorithms such as GRPO. Differences across methods are therefore largely confined to task mixtures, reward designs, or training heuristics, rather than to fundamentally distinct training paradigms.

For the purposes of this work, our goal is not to compare end-to-end benchmark performance across all proposed methods, but to understand what internal changes are induced by RL-based post-training. Achieving this requires controlling for the training recipe itself: we fix a common and widely adopted training pipeline so

that observed differences can be attributed to RL-induced updates rather than to heterogeneous optimization schemes or architectural choices. In addition, such analysis fundamentally depends on access to complete and well-aligned checkpoints across training stages, which are necessary for stage-wise comparison and layer-level interventions.

We therefore focus on three representative models that are highlighted in Table 4: ReVisual-R1, MMR1, and OpenMMReasoner. First, they follow the dominant RL training recipe and target visual reasoning using mixed vision and math supervision, making them representative of current practice. Second, they provide sufficiently complete checkpoint releases to enable reproducible, stage-wise comparison. By restricting our analysis to these models, we can isolate RL-induced effects while avoiding confounding factors arising from architectural differences or incomplete training artifacts.

### C Fine-grained Metrics

This appendix details the fine-grained evaluation metrics used in Table 1. We decompose visual reasoning into three abilities: Vision, Vision-to-Reasoning Alignment, and Reasoning, and define each metric using paired input settings that selectively control the availability of visual evidence.

- C.1 Taxonomy

We decompose visual reasoning into three complementary abilities that capture distinct aspects of multimodal behavior.

Vision. Vision measures whether a model can extract task-relevant information from visual input in a way that changes the final prediction. A model with strong vision should produce correct answers in cases where language-only inference fails, indicating that visual evidence contributes non-trivially to decision making.

Vision-to-Reasoning. Vision-to-Reasoning measures whether perceptual evidence is consistently and correctly incorporated into downstream reasoning. This ability captures alignment between visual perception and reasoning by assessing whether the model arrives at correct answers under both real-image input and textualized visual input representing the same underlying information.

Reasoning. Reasoning measures language-side inference ability independent of visual input. It reflects the model’s capacity for multi-step reasoning when no visual evidence is available.

- C.2 Metric Definitions

We use the following notation, consistent with Table 1. Let i denote the original image, b a black image of the same size, d a textual description of the original image, and p the task prompt. Let f(·) denote the model prediction and y the ground-truth answer. All metrics are computed over N evaluation samples.

Vision. Vision is evaluated on General VQA tasks using an unconditional paired metric:

N

1 N

I[f(in,pn) = yn ∧ f(bn,pn) ̸= yn]. (4)

Mvis =

n=1

This metric measures the fraction of instances whose correct prediction is enabled by visual input. Because it uses the full evaluation set as the denominator, VisionScore supports direct comparison across checkpoints.

Vision-to-Reasoning. Vision-to-Reasoning is evaluated on Math VQA tasks using a joint correctness criterion:

N

1 N

I[f(in,pn) = yn ∧ f(bn,dn,pn) = yn]. (5)

Mv2r =

n=1

Here, b,d,p replaces the image with a textual description of the same visual content. A higher score indicates stronger vision-to-reasoning alignment, as the model produces correct and consistent reasoning outcomes under both visual and textualized-visual inputs. Using the full dataset as the denominator avoids model-dependent conditional subsets and enables cross-checkpoint comparison.

Reasoning. Reasoning is evaluated on textual math tasks as:

N

1 N

Mrea =

n=1

This setting isolates language-side reasoning ability.

I[f(pn) = yn]. (6)

- C.3 Evaluation Tasks and Datasets We instantiate the above metrics using the following datasets:

- • Vision (General VQA). AI4Math/MathVista (Lu et al., 2023) testmini split with metadata[’category’] = general-vqa,
- • Vision-to-Reasoning (Math VQA). AI4Math/MathVista (Lu et al., 2023) testmini split with metadata[’category’] = math-targeted-vqa,
- • Reasoning (Textual Math). HuggingFaceH4/MATH-500 (Lightman et al., 2023) test split.

- D Functional Region Localization This appendix provides full experimental details for the functional region localization experiments described in

- Section 3.2. These experiments are designed as causal probes to identify where visual and reasoning signals are functionally consumed in the base model. They serve to establish a functional reference frame for interpreting the effects of RL-based post-training.

- D.1 Design Principles

All functional localization experiments follow a shared design principle: minimal, localized intervention under fixed inference-time computation. Any observed output change is therefore attributable to the targeted intervention rather than differences in inference depth or search.

All experiments are conducted on the base checkpoint (in our case Qwen2.5-VL-7B-Instruct (Bai et al., 2025a)) to ensure that the resulting functional regions reflect intrinsic properties of the pretrained model rather than effects introduced by post-training. The base checkpoint consists of a vision encoder followed by a 28-layer transformer language backbone. Interventions are applied to individual transformer layers indexed from early to late depth.

- D.2 Vision Functionality Localization

- D.2.1 Layer-Wise Vision Token Swapping

Vision functional localization is performed via vision token swapping. Given a pair of images, we replace the visual token sequence produced by one image (source) with that of the other image (target) at a designated transformer layer, while preserving all textual tokens and the decoding procedure.

Formally, let v(s) and v(t) denote the visual token sequences corresponding to a source and target image. At layer k, the hidden states associated with v(t) are replaced by those of v(s), while all other hidden states remain unchanged. The forward computation then proceeds normally.

This intervention preserves architectural structure and hidden-state scale, ensuring that output changes reflect altered visual evidence rather than numerical instability. A concrete demonstration could be found in Figure 7.

|Layer0|
|---|

|Layer1|
|---|

|Layeri|
|---|

###### Source Image

|Target Answer|
|---|

|2 cubes|
|---|

[Figure 33]

[Figure 34]

Layer0

Layer1

Layeri

Change Rate

|Model Response|
|---|

|3 cubes|
|---|

|Prompt|
|---|

|How many cubes are<br><br>there in the image?|
|---|

Vision Token Swapping

|Layeri|
|---|

|Layer26|
|---|

|Layer27|
|---|

|Layer0|
|---|

|Layer1|
|---|

|Target Image|
|---|

Layer26

Layer27

Layer0

Layer1

Layeri

[Figure 35]

[Figure 36]

- Figure 7 Illustration for vision token swapping to measure layer i’s contribution to model’s response.

- D.2.2 Paired Image Dataset Construction

To isolate individual visual functions, we construct paired image datasets in which each image pair differs in exactly one visual attribute. This enables functional attribution by ensuring that any output change can be causally linked to the perturbed attribute.

We consider four visual functions:

Optical Character Recognition (OCR). OCR pairs consist of images containing different words rendered on visually uniform blank backgrounds. Words are sampled from a deduplicated arXiv corpus. Queries ask the model to read and report the text content. The change rate is defined as whether the generated text differs.

Object Counting. Counting pairs are adapted from the CLEVR (Johnson et al., 2017) dataset. Each pair differs only in the number of instances of a target object category, with object appearance and background held fixed. The change rate is defined as whether the predicted count changes.

Object Grounding. Grounding pairs consist of identical objects placed at different random spatial locations on otherwise clean backgrounds. Queries request a bounding box for the target object. Change is measured by whether the Intersection-over-Union (IoU) between the predicted bounding box and the swapped ground-truth bounding box exceeds 0.5.

Object Recognition. Recognition pairs are drawn from COCO (Lin et al., 2014) images containing a target object and paired with visually blank canvases. Queries ask whether the target object is present. The change rate is defined as the proportion of “No” predictions following token swapping.

- D.2.3 Change Rate Computation

For each visual function and each transformer layer, we compute the change rate as the proportion of test instances for which the model’s output differs between the original and swapped conditions. Aggregating change rates across layers yields a layer-wise sensitivity profile, where a higher change rate indicates stronger functional reliance on visual information at that layer.

- D.2.4 Examples We provide an example for each task in Figure 8.

- D.3 Reasoning Functionality Localization

- D.3.1 Layer-wise Skipping

#### To identify layers that are causally necessary for reasoning, we perform layer-wise zero ablation following prior analyses of mathematical reasoning in transformer models. For a given layer k, all trainable parameters

|OCR|
|---|

|Object Grounding|
|---|

|Object Counting|
|---|

|Recognition|
|---|

|Read the text in the image. Answer with the exact word only.|
|---|

|Locate the object: red circle. Return bounding box as<br><br>x1,y1,x2,y2.|
|---|

|How many cubes are there in the image?|
|---|

|Is there a toaster in the image? Answer Yes or No only.|
|---|

Source Image Target Image

Source Image Target Image

Source Image Target Image

Source Image Target Image

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

- Figure 8 Examples of paired image datasets constructed to isolate visual functions.

|Source Image|
|---|

- Table 5 Inference hyperparameters for functional localization experiment.

[Figure 53]

[Figure 54]

Parameter Setting

| | |
|---|---|
|max_new_tokens temperature do_sample<br><br>|128 0.0 Disabled<br><br>|

[Figure 55]

[Figure 56]

in both the multi-head self-attention and feed-forward submodules are set to zero, while residual connections and normalization layers are preserved.

This intervention reduces the ablated layer to an identity transformation while leaving the model architecture and decoding process unchanged. Each layer is ablated independently, and no gradients are computed during evaluation. A concrete demonstration is provided in Figure 9.

- D.3.2 Reasoning Benchmarks

Reasoning sensitivity is evaluated on reasoning-dominant, text-only benchmarks that require multi-step inference. We focus on GSM8k (Cobbe et al., 2021) and MATH500 (Lightman et al., 2023), a curated subset of competition-level mathematical problems requiring structured derivation and symbolic manipulation. These tasks do not depend on perceptual inputs and therefore isolate reasoning computation.

- D.3.3 Layer-wise Contribution Profile

Aggregating accuracy drops across evaluation samples yields a layer-wise reasoning sensitivity profile. Layers whose ablation causes substantial performance degradation are interpreted as being functionally critical for reasoning, while layers with minor effects are considered less essential. This profile defines a reasoning functional region rather than a single exclusive locus of reasoning computation.

- D.4 Inference Settings

For model inference, we summarize the experimental configurations used in the functional localization studies in Table 5.

- D.5 Interpretation and Limitations

The functional localization experiments in this appendix are designed as causal probes rather than explanatory models of internal computation. High sensitivity indicates necessity under intervention, not sufficiency or exclusive responsibility.

Vision token swapping identifies where visual evidence is actively consumed, while layer-wise zero ablation identifies layers whose computation is critical for multi-step reasoning. Together, these functional regions provide a unified reference frame for interpreting decoupling analyses and RL-based post-training effects in the main text.

- E Model Merging

This section details the implementation of the region-wise model merging experiments used throughout the paper.

Layer Skipping

|Ground Truth|
|---|

|9|
|---|

|Layer0|
|---|

|Layer1|
|---|

|Layeri-1|
|---|

|Layeri|
|---|

|Layeri+1|
|---|

|Layer26|
|---|

|Layer27|
|---|

Change Rate

|Prompt|
|---|

Layeri+1

Layeri-1

Layer26

Layer27

Layer0

Layer1

Layeri

|Response|
|---|

|8|
|---|

|How many positive<br><br>whole-number divisors<br><br>does 196 have?|
|---|

- Figure 9 Illustration for layer skipping to measure layer i’s contribution to model’s response.

Layer region partition. Following the functional localization results in Section 3.1, we partition the transformer backbone into three contiguous regions of equal depth. For the 28-layer transformer used in our experiments, the partition is defined as:

- • Early region: layers 0 to 9,
- • Mid region: layers 10 to 18,
- • Late region: layers 19 to 27.

This 1/3-1/3-1/3 partition provides a coarse, depth-based functional reference frame, enabling consistent comparison of region-wise effects across training recipes. We emphasize that these regions are not intended as precise architectural boundaries, but as an abstraction for analyzing relative functional roles across model depth.

Model merging procedure. Given an IN checkpoint and a RL checkpoint from the same training recipe, we construct hybrid models by selectively transferring layers between checkpoints. For each hybrid configuration, layers are sourced either from the IN or RL checkpoint according to the specified region assignment, while all remaining components are left unchanged.

For each selected layer, we directly copy the full parameter state of all layers within that layer, including self-attention, feed-forward, and layer normalization parameters. No additional fine-tuning, re-normalization, or calibration is performed after merging.

- F Model Training with Region-wise Freezing This section describes the training configuration used for the region-wise parameter freezing experiments in

- Section 4.

- F.1 Implementation.

All region-wise freezing experiments are implemented using the official OpenMMReasoner training codebase2. We follow the standard GRPO-based RL post-training pipeline provided by the repository and modify it by freezing selected transformer layers during training. Specifically, parameters belonging to the frozen layers are excluded from optimization, while all remaining parameters are updated normally. No other changes are made to the training procedure.

- F.2 Hardware and training setup.

All experiments are conducted using 2× NVIDIA H200-SXM GPUs. Each RL run is trained for 2000 steps. To ensure stable training under region-wise freezing and limited GPU memory, we adopt reduced batch sizes and sequence lengths while keeping all other settings identical across freezing conditions.

2https://github.com/EvolvingLMMs-Lab/OpenMMReasoner

- Table 6 Training hyperparameters for region-wise freezing experiments.

Parameter Setting

| | |
|---|---|
|Training steps GPUs train_batch_size ppo_mini_batch_size ppo_micro_batch_size_per_gpu n_rollout_per_prompt max_response_length Frozen parameters<br><br>|2000 2× H200-SXM<br><br>1<br><br>2 1 4 2048 Selected transformer regions<br><br><br>|

- F.3 Batching and rollout configuration.

We use a per-device training batch size of 1. PPO minibatching is applied across GPUs, and for each prompt the model generates multiple candidate responses to support reward-based optimization. The full batching and rollout configuration used for all region-wise freezing experiments is summarized in Table 6.

