# arXiv:2605.09266v2[cs.AI]12May2026

## SEEPHYS PRO: Diagnosing Modality Transfer and Blind-Training Effects in Multimodal RLVR for Physics Reasoning

Kun Xiang1, Terry Jingchen Zhang2, Zirong Liu1, Bokai Zhou1, Yueling Tang1, Junjie Yu1, Jiacong Lu1, Shangrui Huang1, Heng Li1, Likui Zhang1, Kunkun Liu4, Changzheng Zhang4, Yangle Fang4, Boqiang Guo4, Hui-Ling Zhen4, Dandan Tu4,∗, Yinya Huang2,3, Xiaodan Liang1,∗

1Sun Yat-sen University 2ETH Zurich 3ETH AI Center 4Huawei Technologies Ltd. ∗Corresponding authors.

### Abstract

We introduce SEEPHYS PRO, a fine-grained modality transfer benchmark that studies whether models preserve the same reasoning capability when critical information is progressively transferred from text to image. Unlike standard visionessential benchmarks that evaluate a single input form, SEEPHYS PRO features four semantically aligned variants for each problem with progressively increasing visual elements. Our evaluation shows that current frontier models are far from representation-invariant reasoners: performance degrades on average as information moves from language to diagrams, with visual variable grounding as the most critical bottleneck. Motivated by this inference-time fragility, we further develop large training corpora for multimodal RLVR and use blind training as a diagnostic control, finding that RL with all training images masked can still improve performance on unmasked validation sets. To analyze this effect, text-deletion, image-mask-rate, and format-saturation controls suggest that such gains can arise from residual textual and distributional cues rather than valid visual evidence. Our results highlight the need to evaluate multimodal reasoning not only by final-answer accuracy, but also by robustness under modality transfer and by diagnostics that test whether improvements rely on task-critical visual evidence.

Project Page: https://seephyspro.github.io. Challenge: https://www.codabench.org/competitions/16010/. GitHub: https://github.com/AI4Phys/SeePhy-Pro.

### 1 Introduction

A key challenge for multimodal AI is modality consistency, namely whether a model preserves the same reasoning behavior when equivalent information is expressed in different forms [35, 14, 33, 29]. This gap is easy to miss when benchmarks evaluate a single input format, and improvements in final-answer accuracy do not necessarily imply representation-invariant reasoning. Physics provides a particularly sharp testbed, since a diagram can define the physical system itself rather than merely illustrate the text [13, 28, 23, 36, 4]. Here, structure refers to the schema of the system, such as circuit connectivity in a circuit diagram, the contact graph and force directions in a mechanics sketch, or the topology of optical elements in a ray diagram. Variables refer to the labeled quantities tied to specific entities or relations, such as voltages and currents attached to particular nodes and branches, masses tied to specific blocks, or angles tied to specific rays. As information shifts from text into vision, the model must perform grounding and semantic binding, not just generic perception.

Preprint.

To address this gap, we introduce SEEPHYS PRO, a fine-grained modality-transfer benchmark built on the principle of same physics, different representation. Each problem has four aligned variants that progressively move task-critical information from language to vision: (L1) text-only, (L2) structurein-image, (L3) structure+variables-in-image, and (L4) fully rendered problem image. This setup decomposes performance degradation into structural transfer, variable grounding, and full-rendering effects. Across a wide range of MLLMs, average performance drops as information is transferred from text to vision, with the largest degradation often occurring when variables must be grounded from the image. We have also released SEEPHYS PRO as Challenge 31 in the 3rd AI for Math Workshop at ICML 2026.2

To study training-time behavior, we further build two large-scale multimodal RLVR corpora, PhysRL38K and PhysRL-8K. While recent multimodal RLVR studies improve visual reasoning performance [26, 11, 17, 31], outcome-only rewards may still encourage shortcuts that do not depend on valid visual evidence [27, 10, 34]. We therefore include a blind-training control that masks all training images, making each training instance visually unsolvable. Surprisingly, this blind-training RL still improves accuracy on unmasked validation sets, showing that models can infer or reconstruct useful reasoning paths from unsolvable text-only inputs. Further text-deletion and mask-rate controls suggest that these gains are likely driven by residual language, problem templates, and dataset-level statistical regularities rather than effective visual evidence. Taken together, the training-time and test-time results highlight the need for future multimodal reasoning research to look beyond absolute accuracy gains and examine whether such gains come from task-critical visual evidence or from shortcuts in textual structure.

Our contributions are summarized as follows:

- • We introduce SEEPHYS PRO, a progressive modality-transfer benchmark grounded in multimodal physics reasoning, together with metrics that decompose performance into structure recognition, variable grounding, modality gap, and representation consistency.
- • We evaluate a wide range of closed- and open-weight MLLMs and find that even frontier models remain fragile under modality transfer, with the largest degradation often occurring when variables must be extracted from image(s).
- • We release PhysRL-38K and PhysRL-8K as source-matched, test-disjoint physics RL training corpora, and use blind training (masked-image RL) as a negative-control setting; we find it can still improve unmasked test accuracy without reliably closing modality-transfer gaps, showing that accuracy gains do not necessarily imply visually grounded learning.

### 2 Related Work

Multimodal physics reasoning. General science and expert reasoning benchmarks such as ScienceQA [13], MMMU [33], and OlympiadBench [8] show that multimodal scientific problem solving remains difficult. More specialized physics benchmarks, including SeePhys [28], PhyX [23], PhysReason [36], PhysicsArena [4], and QuantiPhy [21], further highlight the difficulty of diagram-based physical reasoning. However, most evaluate each problem in a fixed input form. SEEPHYS PRO instead studies a controlled modality-transfer setting where the underlying physics is fixed while structure, variables, and the full statement are progressively moved from text into vision.

Vision grounding ability in reasoning. Several benchmarks test whether MLLMs truly use visual evidence during reasoning. MathVista [14], MathVerse [35], and CrossMath [29] use visual mathematical problems or information-controlled variants to expose modality gaps. Our setting is stricter: physics diagrams often define the system itself (e.g., topology and variable-to-entity bindings), and SEEPHYS PRO separates structural grounding, variable grounding, and full-rendering effects across aligned levels.

Reinforcement learning with verifiable rewards. Reinforcement learning with verifiable rewards (RLVR) is widely used to post-train reasoning models [22, 32, 37], and has also been explored in multimodal settings [26, 25, 11, 17, 31]. However, outcome-only rewards may encourage shortcuts that do not depend on valid visual evidence, and recent analyses suggest that reasoning-style training can amplify ungrounded behavior or improve behaviors already latent in the base model [27, 10, 34].

- 1https://www.codabench.org/competitions/16010/.
- 2https://ai4math2026.github.io/.

[Figure 1]

- Figure 1: Overview of the four modality-transfer levels in SEEPHYS PRO. Each seed problem is transformed into four semantically aligned variants that progressively move problem-critical information from language to vision: Level 1 is text-only, Level 2 moves structure into the image, Level 3 further moves variables and labels into the image, and Level 4 renders the full problem into a single visual input.

We therefore use blind training (masking all training images) as a negative control: if RL still improves unmasked test performance, the gain is not fully attributable to better visual grounding.

### 3 SEEPHYS PRO: A Fine-Grained Benchmark for Modality-Transfer

Design principle. SEEPHYS PRO is built around the diagnostic principle of same physics, different representation. For each seed problem, the physical system, required laws, answer, and reasoning target are fixed, while problem-critical information is progressively moved across modalities. This design turns a single physics question into a controlled probe of whether MLLMs reason over stable physical semantics or over surface-level input formats, following controlled-modality diagnostics from visual mathematics and modality-gap evaluation [35, 29]. It also addresses an ambiguity in ordinary vision-essential physics evaluation [28, 23, 36]: when a model succeeds or fails on a visual physics question, final accuracy alone cannot tell whether the decisive factor is diagram perception, variable reading, physical abstraction, OCR, or downstream symbolic reasoning.

Four-level modality transfer. Figure 1 illustrates the four-level transformation with an example. Each seed problem is converted into four aligned variants with identical physical semantics. The variants are not independently written questions: annotators manually redraw and edit the same problem so that the physical system, queried quantity, variables, constraints, solution path, and gold answer remain unchanged. Only the carrier of information changes across levels. Level 1 is text-only: all structural relations, variables, and numerical quantities are described in language, providing a reference point for text-based physics reasoning. Level 2 adds structured visual information by moving only the physical structure into the image while keeping variables in text, testing visual structural understanding such as circuit topology, force configuration, graph layout, or pulley connection. Level

- 3 further overlays variables and labels onto the same diagram, testing whether models can read quantities and bind them to the correct physical entities. Level 4 builds upon Level 3 by converting the problem statement into handwritten text and rendering it alongside the diagram into a single image. This forces the model to simultaneously process handwritten formulas, complex layouts, and physical reasoning within a unified visual context. This controlled construction makes Level 1–4 a modality-transfer probe rather than a collection of different problems.

Benchmark data collection. We collect seed problems from heterogeneous physics sources rather than directly reusing existing fixed-form physics benchmarks [28, 36, 8, 23], including public datasets, textbooks and problem books, PhD qualifying and entrance examinations, olympiad archives, and school or university exam papers. The source pool contains over 5,000 PDF pages, which are processed with Mathpix OCR [15] and then curated by 10 engineering-trained annotators, including 7 bachelor’s-level and 3 PhD-level annotators. Each accepted problem is assigned a three-level taxonomy covering discipline, field, and domain. During curation, we filter invalid samples, normalize notation and answer formats, and remove near-duplicates using script-based text matching, manual review, and GPT-5-mini [19] for LLM-assisted checks. During transformation, annotators rewrite accepted seeds into four aligned variants while preserving the physical system, target quantity, and gold answer. Diagrams are then redrawn and separated into structure and variable layers, enabling Level 2 to test structure grounding and Level 3 to test variable grounding. We filter out problems with uncertain answers, incomplete statements, or insufficient solution conditions, and verify that

##### Figure 2: 3-levels taxonomy.

##### Table 1: Dataset statistics.

##### Table 2: Level-1-conditioned transfer accuracy.

###### Statistics Number

###### Model A1|1 A2|1 A3|1 A4|1

Seed Questions 1,000 Total Questions 4,000

GPT-5.4 100.0 79.6 68.8 64.8 GPT-5 100.0 66.0 45.4 44.7 Gemini-3.1-Proa 100.0 88.7 81.0 77.5 Claude-4.7-Opusa 100.0 81.8 68.2 57.4 Qwen3.5-27B 100.0 55.6 46.4 41.0 Qwen-3.5-9B 100.0 75.2 64.1 62.0 Gemma-4-26B-A4B-it 100.0 54.1 44.9 37.3 Gemma-4-31B-it 100.0 60.1 41.5 38.9

5.7%

Disciplines 6 Domains 38 Fields 104

5.7%

45.6%

###### Categories

Answer Type Value 46.6% Option 46.0% Equation 6.2% Expression 1.2%

38.9%

Note. Aℓ|1 = 100|C1 ∩ Cℓ|/|C1|. Row sizes are 539/334/142/148/360/242/292/311. a Evaluated on testmini.

Mechanics (456) Electromagnetism (389) Optics (57)

Thermodynamics (57) Waves/Acoustics (23) Modern Physics (18)

each four-level group preserves the same physical system, variables, answer, and reasoning path. The full construction workflow is described in Appendix A.2.

Taxonomy and metadata. SEEPHYS PRO is annotated with physics domains, visual information types, and reasoning skills. The domain taxonomy covers major areas such as mechanics, electricity and magnetism, optics, thermodynamics, waves, and modern physics, with a long tail of more specialized topics, as summarized in Figure 2; Table 1 reports the corresponding dataset scale and answer-type distribution. Visual evidence is categorized by the type of information required for solution, including structure/topology, variable labels, directions and vectors, graphs and curves, geometric relations, and symbolic diagrams. Reasoning metadata further records skills such as conservation-law reasoning, force analysis, circuit reduction, graph-to-equation conversion, geometric optics, unit reasoning, and multi-step numerical derivation. These annotations support fine-grained analysis of whether failures arise from structural perception, variable grounding, or downstream physical reasoning.

Diagnostic metrics. For a model f and an aligned test set D = {(x(1)i ,x(2)i ,x(3)i ,x(4)i ,yi)}Ni=1, where x(iℓ) is the Level-ℓ representation of the same seed problem, we define the Level-ℓ accuracy as

N

100 N

I g f(x(iℓ)) = yi , (1)

Aℓ(f) =

i=1

where g(·) denotes answer extraction and normalization. The modality-transfer gaps are signed differences in percentage points:

∆S(f) = A1(f) − A2(f), ∆V(f) = A2(f) − A3(f), ∆R(f) = A3(f) − A4(f), ∆T(f) = A1(f) − A4(f). (2)

Here ∆S measures the cost of transferring structural information into vision, ∆V measures the additional cost of visually grounding variables, ∆R measures the cost of full visual rendering, and ∆T = ∆S + ∆V + ∆R is the total transfer gap. Positive values indicate degradation under a more visual representation, while negative values indicate that the visually richer variant is solved more accurately. We also report four-way representation consistency,

N

100 N

I[ˆyi,L1 = yi ∧ yˆi,L2 = yi ∧ yˆi,L3 = yi ∧ yˆi,L4 = yi], (3)

Cons4 =

i=1

which measures the percentage of seed problems answered correctly across all four aligned representations. Together, these metrics separate absolute problem-solving ability from robustness to visual modality transfer.

### 4 Test-Time Modality Transfer

This section focuses on the first research question of SEEPHYS PRO: Can MLLMs maintain their performance when the same problem is expressed through progressively more visual and less textual representations? We first outline our evaluation setup and present our main results.

Table 3: Main results on SEEPHYS PRO. We report accuracy (%) on four controlled modalitytransfer levels, four-way representation consistency, and signed transfer gaps in percentage points. ∆S, ∆V, and ∆R correspond to Level 1→2 structural transfer, Level 2→3 variable grounding, and Level 3→4 rendering, respectively; ∆T = A1 − A4 is the total gap. Cons4 denotes the percentage of test problems answered correctly at all four levels. Larger positive gaps indicate stronger representation sensitivity.

Model Accuracy / consistency Transfer gap

L1 L2 L3 L4 Cons4 ↑ ∆S ↓ ∆V ↓ ∆R ↓ ∆T ↓ Human Performance 54.0 58.5 59.5 56.0 49.0 -4.5 -1.0 3.5 -2.0 Closed-weight Frontier Models

GPT-5.4 67.4 64.1 55.8 53.0 32.6 3.3 8.3 2.8 14.4 GPT-5 41.8 32.9 23.8 23.2 8.9 8.9 9.1 0.5 18.5 Gemini-3.1-Proa 71.0 72.0 66.5 66.5 47.0 -1.0 5.5 0.0 4.5 Gemini-3-Pro 58.9 51.2 46.2 45.5 43.0 7.7 5.0 0.7 13.4 Claude-4.7-Opusa 74.0 67.0 56.5 46.5 33.5 7.0 10.5 10 27.5 Claude-4.6-Opus 58.5 55.3 45.4 30.0 19.0 3.2 9.9 15.4 28.5 Kimi K2.5a 52.0 48.5 46.0 42.0 26.0 3.5 2.5 4.0 10.0 Qwen-3.6-flash 61.4 59.3 49.9 48.4 29.9 2.1 9.4 1.5 13.0 Qwen3.5-122B-A10Ba 47.5 48.0 39.5 42.0 25.5 -0.5 8.5 -2.5 5.5 SuperNovaa 25.5 27.5 20.0 22.0 11.0 -2.0 7.5 -2.0 3.5

###### Open-weight Models

Qwen3.5-27B 45.0 34.8 28.0 25.6 9.9 10.3 6.8 2.4 19.4 Qwen-3.5-9B 30.3 47.8 37.8 35.4 12.8 -17.5 10.0 2.4 -5.1 P1-VL-30B-A3B 29.0 20.9 15.1 14.9 4.3 8.1 5.8 0.2 14.1 Gemma-4-26B-A4B-it 36.5 29.5 26.1 19.4 9.0 7.0 3.4 6.7 17.1 Gemma-4-31B-it 38.9 33.5 23.9 22.0 8.9 5.4 9.6 1.9 16.9

Average 49.2 46.1 38.7 35.8 21.4 3.0 7.4 2.9 13.4

a Evaluated on the 200-sample testmini subset due to API-budget constraints.

Models. We evaluate 10 closed-weight and 5 open-weight MLLMs. The closed-weight set includes GPT-5.4 and GPT-5 [19], Gemini-3.1-Pro and Gemini-3-Pro [6], Claude-4.7-Opus and Claude-4.6Opus [1], Kimi K2.5 [18], Qwen-3.6-flash and Qwen3.5-122B-A10B [30, 2], and SuperNova [24]. The open-weight set includes Qwen3.5-27B and Qwen-3.5-9B [30, 2], P1-VL-30B-A3B [20], and Gemma-4-26B-A4B-it and Gemma-4-31B-it [7].

Test sets. For efficient development and API-cost control, each level is split into an 800-example test set and a 200-example testmini set with an 8:2 ratio. Unless otherwise stated, reported results use test; models marked in Table 3 are evaluated on testmini. We evaluate the same seed problems across Level 1–4, enabling direct measurement of representation sensitivity under controlled modality transfer.

Judging. Following the evaluation practice of SeePhys [28] and the LMMS-Eval toolkit [12], we implement a composite answer judge. It first applies deterministic extraction and matching rules, including boxed-answer parsing, multiple-choice option matching, numerical tolerance, symbolic normalization, and unit-aware comparison. For outputs not resolved by these rules, we use DeepSeekV3.2 [5] as a more robust LLM judge. Closed-weight models are evaluated with a 32K context window and temperature 0, except GPT-family models where the official API constraints require temperature 1. Open-weight models are evaluated with a 16K context window. Additional benchmark and judging details are given in Appendix A.

Main Results Across all evaluated models, average accuracy decreases from 49.2% at Level 1 to 35.8% at Level 4, yielding an average total modality-transfer gap of 13.4 points. The degradation is not limited to weaker models: GPT-5.4 drops from 67.4% to 53.0%, and Claude-4.7-Opus drops from 74.0% to 46.5%. Gemini-3.1-Pro is the strongest model on Level 4, but still does not match its own Level-1 performance. As a human reference, 100 Chinese high-school students achieve 54.0%, 58.5%, 59.5%, and 56.0% on the testmini subset from Level 1 to Level 4. Several frontier

models exceed this reference in marginal accuracy, but none matches the human group’s four-way consistency.

Variable grounding is the dominant bottleneck. The staged gaps in Table 3 show that moving only structure into images causes a smaller model-average gap (∆S = 3.0), while moving variables and labels into images causes the largest drop (∆V = 7.4). The final rendering stage adds a smaller but non-negligible gap (∆R = 2.9), reflecting the additional burden of OCR, formula recognition, and layout understanding. Thus, the central failure mode is often not simply recognizing the diagram, but reading the right visual quantities and binding them to the correct physical entities.

Marginal accuracy overestimates cross-representation stability. For example, Claude-4.7-Opus achieves the highest Level-1 accuracy but only 33.5% four-way consistency, and GPT-5.4 has 32.6% consistency despite 67.4% Level-1 accuracy.

The marginal gap ∆T = A1 − A4 mixes two effects: whether the model can solve the underlying physics at all, and whether it can preserve that solution when information is moved into vision. Table 2 removes the first factor by conditioning on problems that each model already solves at Level 1. The remaining drops are still large: among Level-1-correct problems, GPT-5.4 retains 64.8% accuracy at Level 4 and Claude-4.7-Opus retains 57.4%. These conditioned results show that the modality-transfer gap is not merely a consequence of difficult physics questions; models often lose an already-demonstrated solution when the same information is represented visually.

We further analyze performance by physics discipline in Appendix A.4, and present representative error clusters and case studies in Appendices C and D. Across these analyses, the dominant trend remains the same: visually grounded variable use is difficult beyond any single physics category.

### 5 Training-Time Diagnostic: Can RL Help Close the Modality Gap?

Section 4 shows an inference-time failure: models lose accuracy when the same physics is represented more visually. This section asks a different but directly connected question: if we train on visionnecessary multimodal data, does RL actually close the modality-transfer gaps defined by SEEPHYS PRO? We therefore evaluate training by both accuracy gain and gap dynamics. A visually grounded improvement should not merely raise final-answer accuracy; it should preferentially improve the visually demanding levels and reduce gaps such as ∆V and ∆T.

##### 5.1 Diagnostic Setup

In addition to the benchmark itself, we construct and release two physics RL training corpora, PhysRL-38K and PhysRL-8K. This is motivated by a practical gap: multimodal physics datasets for RLVR remain scarce compared with visual math [14, 35, 26], even though physics entails rich multimodal representation structure, from circuit analysis to Feynman diagrams. PhysRL-38K is an approximately 38K-example physics VQA collection built from the same source pool and data engine as SEEPHYS PRO, covering public datasets, textbooks, olympiad archives, and exam-style problems. The training corpora are source-matched to SEEPHYS PRO but instance-disjoint from all benchmark test sets. Unlike the benchmark split, PhysRL-38K is designed for scalable training rather than controlled evaluation: it does not require the full manual redrawing, four-level alignment, and fine-grained modality-transfer annotation used by SEEPHYS PRO. We further obtain PhysRL-8K by filtering PhysRL-38K with GPT-5-mini [19] to retain approximately 8K vision-necessary examples. Both PhysRL-38K and PhysRL-8K will be released, and Appendix B.1 validates the larger pool through RL runs that improve multiple held-out physics benchmarks.

We fine-tune Qwen2.5-VL-7B-Instruct [3] and Qwen3-VL-4B-Instruct [2] with outcome-supervised RL on physics and math vision-necessary corpora. For physics, the main training corpus is PhysRL-

- 8K. For math, we construct ViRL39K-VN, a 22K-example vision-necessary subset selected from ViRL39K [26, 25] with GPT-5-mini filtering. We use matched validation suites for the two domains. For physics, we evaluate on unmasked SEEPHYS PRO Level 1–4 and on held-out physics benchmarks including SeePhys [28], PhysReason [36], OlympiadBench [8], and PhyX [23]. For math, following the PAPO evaluation setting [26], we evaluate on the vision-dependent split of MathVerse [35], referred to simply as MathVerse, and on MMK12 Test [16]. Unless otherwise stated, all validation images are kept unmasked, so blind-training performance is measured on normal visual inputs.

All runs use a math-style final-answer prompt and an answer-verification reward. For the main runs, we use GSPO-style token-level policy optimization [37] with four rollouts per prompt, rollout temperature 1.0, top-p = 1.0, maximum prompt and response lengths of 4096 tokens, and AdamW

Normal RL Blind RL

(a) L1: text

(b) L2: structure

(c) L3: variables

(d) L4: rendered

22%

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

18%

accuracy

14%

10%

6%

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

step

step

step

step

(e) Total transfer gap: A1 − A4

(f) Variable gap: A2 − A3

- 3%
- 4%
- 5%
- 6%
- 7%
- 8%
- 9%
- 10%

- 2%

- 3%

- 4%

- 4%

- 5%

- 5%

- 6%

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| |gap| | | | | | | | |
| | | | | | | | | | |
| |accuracy| | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

accuracygap

accuracygap

2%

0 25 50 75 100 125 150 175

0 25 50 75 100 125 150 175

step

step

- Figure 3: RL does not appear to close modality gaps. Qwen3-VL-4B is trained on source-matched but test-disjoint vision-necessary physics data and evaluated on unmasked SEEPHYS PRO Level 1–4. The top row tracks validation accuracy on each modality-transfer level after normal and blind RL updates; the bottom row tracks the total transfer gap ∆T = A1 − A4 and variable-grounding gap ∆V = A2 − A3. Both normal and blind RL improve validation accuracy, but the visually induced gaps remain large.

with learning rate 10−6 and weight decay 0.01. We use one PPO epoch per update, bfloat16 FSDP training, vLLM rollout serving [9], and evaluate every five training iterations. Qwen3-VL-4B runs use rollout batch size 256 and validation batch size 1024; Qwen2.5-VL-7B uses a larger rollout batch size of 512. Exact launch scripts and dataset variants are provided in the supplementary material.

Normal vs. blind RL. We compare standard RL with original images (normal RL) against a matched blind-training control in which all training images are replaced with black images (blind RL), while the train/test splits, reward function, and all other training settings remain unchanged. We report normal and blind gains as follows,

Gainnormal = Anormal − A0, Gainblind = Ablind − A0, (4) and define the visually grounded residual and blind-gain ratio as

Gainblind Gainnormal

. (5) We also track gap closure,

GroundedResidual = Anormal − Ablind, ρblind =

∆0 − ∆after ∆0

, (6)

Closure(∆) =

where ∆ is one of the modality-transfer gaps from Section 3. Positive closure means RL reduces the benchmark-defined failure mode.

##### 5.2 SeePhys Pro Reveals Accuracy Gains without Gap Closure

- Figure 3 shows the central observation. Both normal and blind RL improve all four levels of SEEPHYS PRO. For Qwen3-VL-4B, Level-1 accuracy increases from 9.9% to 18.3% under normal RL and 20.9% under blind RL; Level-4 accuracy increases from 6.4% to 10.8% and 13.0%, respectively.

Normal RL Blind RL

Qwen3-VL-4B physics

Qwen2.5-VL-7B physics

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | | | | |ma|th|
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

- SeePhys Pro L1
- SeePhys Pro L2
- SeePhys Pro L3
- SeePhys Pro L4

SeePhys

PhyX-MCQ

PhyX-OE

SeePhys

PhyX-MCQ

OlympiadBench

PhyX-OE

OlympiadBench

MathVerse

math

MathVerse

MMK12

MMK12

0% 5% 10% 15% 20% 25% 30% 35%

0% 5% 10% 15% 20% 25% 30% 35%

peak gain

peak gain

- Figure 4: Blind gains are not unique to SEEPHYS PRO. Peak validation gains for normal and blind RL on external physics and math benchmarks. Blind RL often recovers a substantial fraction of normal RL gains, and on several math settings it matches or exceeds normal RL.

The gap dynamics tell a different story. The total transfer gap ∆T widens from 3.5 percentage points before training to 7.5 points after normal RL and 7.9 points after blind RL. The variable-grounding

gap ∆V also does not show stable closure: the normal and blind curves in Figure 3(f) remain close and even cross during training, rather than exhibiting a clear monotonic separation. The close trajectories of normal and blind RL indicate that valid training images are not sufficient by themselves to produce stable gap closure in this setting. Most of the gain appears to come from improvements in general physics problem solving, such as better physical-law selection, equation manipulation, numerical calculation, adaptation to common problem templates, and more reliable answer generation. These abilities can raise accuracy on all four levels, because the same underlying physics still has to be solved in every representation. By contrast, an improvement in visual grounding should make the model more reliable specifically when task-critical information is moved from text into the image, which would appear as larger gains on the more visual levels and a reduction in the modality gaps.

##### 5.3 Cross-Benchmark Negative Controls

- Figure 4 shows that blind-training gains are not unique to SEEPHYS PRO. The same negative control also produces gains on external math and physics benchmarks. The pattern is strongest on the math suite. On MathVerse and MMK12, blind RL often recovers a large fraction of the normal-RL gain, and in several settings reaches comparable or slightly higher peak gains. The physics suite is more heterogeneous. Normal RL is clearly stronger for Qwen3-VL-4B on benchmarks such as PhyX and OlympiadBench, suggesting that valid images can provide task-relevant signal in visually demanding physics problems. At the same time, the persistent blind gains across multiple physics evaluations show that outcome-RL on vision-necessary data can improve benchmark accuracy through non-visual adaptation; accuracy gains alone are therefore insufficient evidence of improved visual grounding. Additional details are reported in Appendix B.

### 6 Mechanism Analysis: What Does Blind RL Learn?

Blind RL improving on vision-necessary benchmarks is surprising, but it is not paradoxical. The key distinction is between instance-level vision necessity and distribution-level learnability. A single problem may be underdetermined without its image, yet a collection of such problems can still contain exploitable regularities in text, options, formula templates, answer ranges, and task style. We therefore analyze which non-visual signals remain available under blind training and which simple alternative explanations are insufficient.

Residual language is the main source of blind gains. Figure 5(a) shows that blind gains shrink as training text is deleted. For Qwen3-VL-4B, the blind peak gain on MathVerse [35] drops from about 26.6 points at 25% deletion to nearly zero at 100% deletion; on MMK12 [16], it drops from about 19.0 points to below one point. Physics is noisier but shows the same boundary condition: complete text deletion reduces blind gains on PhysReason [36] and SeePhys Pro to near zero. Thus blind RL

(a) Quantitative text deletion

(b) Targeted text deletion

[Figure 2]

[Figure 3]

27 28 26 26 24 25

7B-MMK12

25%

25%

7B-MathVerse

blindpeakgain

26 22 24 24 23 24

7B-MathVerse

20%

20%

4B-MathVerse

18 18 16 18 12 18

4B-MMK12

15%

15%

7B-MMK12

26 27 25 25 19 26

4B-MathVerse

10%

4B-MMK12

10%

SeePhys Pro

6 6 6 4 5 5

PhysReason

5%

5%

PhysReason

8 9 10 6 8 10

SeePhys Pro

0%

0%

bg know.formulanumber option query

0.25 0.50 0.75 1.00

deleted sentence ratio

(c) Image mask-rate ablation

(d) After format saturation

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 4%

- 5%

- 6%

- 7%

SeePhys Pro PhysReason

20%

additionalacc.gain

peakacc.gain

15%

10%

| |
|---|

5%

| |
|---|

| |
|---|

0%

10 25 50 75 90

MMK12 normal

MMK12 blind

MathVerse normal

MathVerse blind

image mask rate (%)

- Figure 5: Mechanism controls for blind-training gains. Text deletion, targeted deletion, maskrate ablation, and post-format-saturation controls show that blind gains depend on residual textual/distributional cues rather than valid visual evidence. All gains are computed on unmasked validation sets.

relies on residual language, problem style, answer priors, and generic reasoning practice rather than valid visual evidence.

The shortcuts are distributed rather than single-field artifacts. Figure 5(b) removes background, knowledge statements, formulas, numbers, options, and question clauses. In math, answer options are a visible shortcut source, especially for Qwen3-VL-4B on MMK12 and MathVerse. In physics, no single span explains the effect: deleting numbers, options, or formulas can reduce gains, but blind gains persist under most single-category deletions. The shortcut signal is therefore distributed across weak textual cues and dataset regularities.

Two auxiliary controls rule out simpler explanations. Figure 5(c) varies the training-image mask rate from 10% to 90%. If blind gains came from a special all-black-image artifact, gains should change monotonically as masking increases. Instead, early peak gains remain positive and non-monotonic. Figure 5(d) also weakens the format-only explanation: after format reward reaches 90%, Qwen2.5-VL-7B still gains about 21.2/19.1 points on MMK12 under normal/blind RL and

- 9.6/3.9 points on MathVerse.

Together, these controls show that blind-training gains are non-visual gains induced by outcomeonly RL on residual textual and structural regularities, explaining why Section 5 observes accuracy improvement without proportional modality-gap closure. This is a credit-assignment failure in outcome-only multimodal RL [26, 25]: final-answer rewards do not identify whether answers came from visual evidence, text priors, or shortcuts.

Limitations. SEEPHYS PRO is not intended to be the most difficult possible physics benchmark; its goal is controlled diagnosis of modality-transfer robustness rather than maximizing raw task difficulty. The paper also identifies inconsistency under modality transfer and non-visual gains under blind RL, but does not propose or validate a complete mitigation strategy. Promising directions include counterfactual image-text pairs, black-image unanswerability, and process-level rewards.

### 7 Conclusion

We introduced SEEPHYS PRO, a progressive modality-transfer benchmark for multimodal physics reasoning. By constructing four semantically aligned variants for each problem, SEEPHYS PRO diagnoses whether models preserve physical reasoning when information moves from text to visual structure, visual variables, and fully rendered diagrams. Our results show that current MLLMs remain fragile under modality transfer, with visual variable grounding emerging as a key bottleneck. We further developed PhysRL, a physics reasoning VQA training set for RL. Using them with SEEPHYS PRO as a diagnostic target, we find that blind training can improve validation accuracy even when training images contain no valid visual information. Together, these results motivate evaluating

multimodal reasoning through both inference-time modality robustness and training-time grounding diagnostics, rather than relying only on final-answer accuracy.

### References

- [1] Anthropic. Introducing Claude 4. https://www.anthropic.com/news/claude-4, 2025. Accessed 2026-05-05.
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/abs/2511.21631.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. doi: 10.48550/arXiv.2502.13923. URL https://arxiv.org/abs/2502.13923.
- [4] Song Dai, Yibo Yan, Jiamin Su, Dongfang Zihao, Yubo Gao, et al. PhysicsArena: The first multimodal physics reasoning benchmark exploring variable, process, and solution dimensions. arXiv preprint arXiv:2505.15472, 2025. doi: 10.48550/arXiv.2505.15472. URL https: //arxiv.org/abs/2505.15472.
- [5] DeepSeek-AI. DeepSeek-V3.2. https://arxiv.org/abs/2512.02556, 2025. Accessed 2026-05-05.
- [6] Gemini Team. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. doi: 10.48550/arXiv.2312.11805. URL https://arxiv.org/abs/ 2312.11805.
- [7] Google DeepMind. Gemma open models. https://deepmind.google/models/gemma/,

2026. Accessed 2026-05-05.

- [8] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, et al. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand, August

2024. Association for Computational Linguistics. URL https://aclanthology.org/2024. acl-long.211/.

- [9] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, et al. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.
- [10] Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, et al. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. arXiv preprint arXiv:2505.21523, 2025. doi: 10.48550/arXiv.2505.21523. URL https://arxiv.org/abs/ 2505.21523.
- [11] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, et al. Visual-RFT: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. doi: 10.48550/arXiv.2503.

01785. URL https://arxiv.org/abs/2503.01785.

- [12] LMMS-Eval Contributors. LMMS-Eval: Evaluation Suite for Large Multimodal Models. https://github.com/EvolvingLMMs-Lab/lmms-eval, 2024. Accessed 2026-05-05.
- [13] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, et al. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, 2022.
- [14] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, et al. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In Proceedings of the International Conference on Learning Representations, 2024.
- [15] Mathpix. Mathpix: Document conversion done right. https://mathpix.com/, 2026. Accessed 2026-05-07.

- [16] Fanqing Meng. MMK12-Test: A multimodal K-12 mathematics evaluation set. https:// huggingface.co/datasets/FanqingM/MMK12, 2024. Hugging Face dataset card, accessed 2026-05-05.
- [17] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, et al. MM-Eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. doi: 10.48550/arXiv.2503.07365. URL https://arxiv. org/abs/2503.07365.
- [18] Moonshot AI. Kimi K2.5. https://www.kimi.com/ai-models/kimi-k2-5, 2026. Accessed 2026-05-05.
- [19] OpenAI. GPT-5 system card. https://cdn.openai.com/gpt-5-system-card.pdf, 2025. Accessed 2026-05-05.
- [20] P1 Team. P1-VL. https://arxiv.org/abs/2602.09443, 2026. arXiv preprint arXiv:2602.09443, accessed 2026-05-05.
- [21] Li Puyin, Tiange Xiang, Ella Mao, Shirley Wei, Xinye Chen, et al. QuantiPhy: A quantitative benchmark evaluating physical reasoning abilities of vision-language models. arXiv preprint arXiv:2512.19526, 2025. doi: 10.48550/arXiv.2512.19526. URL https://arxiv.org/abs/ 2512.19526.
- [22] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, et al. DeepSeekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.
- [23] Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, et al. PhyX: Does your model have the “Wits” for physical reasoning? arXiv preprint arXiv:2505.15929, 2025. URL https://arxiv.org/abs/2505.15929.
- [24] StepFun. SuperNova. https://platform.stepfun.com/, 2025. Model page, accessed 2026-05-05.
- [25] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, et al. VL-Rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. doi: 10.48550/arXiv.2504.08837. URL https://arxiv. org/abs/2504.08837.
- [26] Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, et al. PAPO: Reinforcement learning for advanced perception and reasoning in vision-language models. arXiv preprint arXiv:2507.06448, 2025. doi: 10.48550/arXiv.2507.06448. URL https: //arxiv.org/abs/2507.06448.
- [27] Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, et al. Grounded chainof-thought for multimodal large language models. arXiv preprint arXiv:2503.12799, 2025. doi: 10.48550/arXiv.2503.12799. URL https://arxiv.org/abs/2503.12799.
- [28] Kun Xiang, Heng Li, Terry Jingchen Zhang, Yinya Huang, Zirong Liu, et al. SeePhys: Does seeing help thinking? – benchmarking vision-based physics reasoning. arXiv preprint arXiv:2505.19099, 2025. doi: 10.48550/arXiv.2505.19099. URL https://arxiv.org/abs/ 2505.19099.
- [29] Yige Xu, Yongjie Wang, Zizhuo Wu, Kaisong Song, Jun Lin, et al. Do vision-language models truly perform vision reasoning? A rigorous study of the modality gap. arXiv preprint arXiv:2604.16256, 2026. URL https://arxiv.org/abs/2604.16256.
- [30] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/arXiv.2505.09388. URL https://arxiv.org/abs/2505.09388.
- [31] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, et al. R1-Onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. doi: 10.48550/arXiv.2503.10615. URL https://arxiv.org/abs/ 2503.10615.

- [32] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, et al. DAPO: An opensource LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. doi: 10.48550/arXiv.2503.14476. URL https://arxiv.org/abs/2503.14476.
- [33] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [34] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, et al. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? arXiv preprint arXiv:2504.13837, 2025. doi: 10.48550/arXiv.2504.13837. URL https://arxiv.org/abs/ 2504.13837.
- [35] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, et al. MathVerse: Does your multi-modal LLM truly see the diagrams in visual math problems? In Proceedings of the European Conference on Computer Vision, pages 169–186, 2024.
- [36] Xinyu Zhang, Yuxuan Dong, Yanrui Wu, Jiaxing Huang, Chengyou Jia, et al. PhysReason: A comprehensive benchmark towards physics-based reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16593–16615, Vienna, Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.811. URL https://aclanthology.org/2025.acl-long. 811/.
- [37] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. doi: 10.48550/arXiv.2507.18071. URL https://arxiv.org/abs/2507.18071.

### A Additional Evaluation Results

##### A.1 Evaluation Protocol Details

All models are evaluated with the same answer-oriented prompt template. For Chinese problems, the instruction is the Chinese equivalent of asking the model to produce its reasoning as an internal monologue wrapped by <thinking> and </thinking> and then place the final answer in \boxed{}. For English problems, we use the following instruction:

{{ content | trim }} You first think through the reasoning process as an internal monologue, enclosed within <thinking> </thinking> tags. Then, provide your final answer enclosed within \boxed{}.

For inference, closed-weight models use a 32K context window and temperature 0 when supported. GPT-family models are evaluated with temperature 1 due to official API constraints. Open-weight models use a 16K context window, with otherwise default greedy or deterministic decoding when available. Our judge follows the composite-evaluation style used in SeePhys [28] and LMMSEval [12]. We first extract candidate final answers from \boxed{} spans, final-answer markers, option letters, and trailing short answers. Deterministic rules then check multiple-choice options, normalized strings, symbolic expressions, numerical values under tolerance, and unit-aware equivalence. For unresolved or ambiguous cases, we call DeepSeek-V3.2 [5] as an LLM judge with the problem, gold answer, and model prediction, and use it only to decide answer equivalence rather than to rescore reasoning quality.

Hardware. Local inference for open-weight models is run on NVIDIA GH200 GPUs. API-based closed-weight models are evaluated through their hosted providers. All RL experiments are conducted on 16 NVIDIA GH200 GPUs with bfloat16 FSDP training and vLLM rollout serving [9].

##### A.2 Benchmark Construction Details

We organize the benchmark construction pipeline into four logical stages: source collection, curation, transformation, and sketching, as summarized in Figure 6. We first collect candidate problems from heterogeneous physics sources, including public datasets, textbooks and problem books (e.g., University Physics and College Physics), PhD qualifying and entrance examination collections, olympiad archives such as IPhO and CPhO, and school or university exam papers such as Cambridge IGCSE and AS/A-Level physics. The source pool contains over 5,000 PDF pages. Source PDFs are converted into structured text with Mathpix OCR [15], after which the curation stage performs deduplication, filtering, standardization, and labeling through text normalization, formula cleanup, and near-duplicate removal using exact-match checks, normalized edit-distance thresholds, and n-gram overlap signals, with manual review for formula-heavy borderline cases. In stages that require model assistance, including OCR cleanup, candidate normalization, and LLM-based verification of borderline cases, we use GPT-5-mini [19]. The curated pool is then labeled by a team of ten annotators with engineering training (7 bachelor’s-level and 3 PhD-level annotators). During this stage, each problem receives a three-level taxonomy covering discipline, field, and domain, together with auxiliary tags for required visual evidence and reasoning skills. The broad discipline layer includes major areas such as Classical Mechanics, Electromagnetism and Optics, and Statistical Mechanics and Thermodynamics, while lower field/domain layers capture more specific settings such as circuit analysis, rigid-body equilibrium, geometric optics, wave phenomena, and thermal processes.

After curation, accepted seeds are transformed into aligned four-level variants under the principle of same physics, different representation. This transformation stage includes scenario reframing, variable substitution, objective shift, and structure transformation, while preserving the underlying physical system, target quantity, and gold answer. In the sketching stage, each problem is first hand-redrawn into a clean raw diagram that standardizes topology, geometry, arrows, and object boundaries. The visual content is then explicitly separated into a structure layer and a variable layer: the structure-only version retains entities and relations but removes symbolic quantities, whereas the variable layer overlays labels, values, and other quantity tokens onto the same diagram. This separation directly supports Level 2 and Level 3. Finally, the complete statement, formulas, and diagram are jointly rendered into a single image for Level 4, while keeping the semantic content aligned with the text-only Level 1 variant.

[Figure 4]

- Figure 6: Data pipeline for constructing SEEPHYS PRO. We describe the construction process as four logical stages: source collection, curation, transformation, and controlled sketching into raw, structural, and variable-level variants.

Table 4: Level-3 accuracy by physics discipline.

Model Mech. EM Thermo. Optics Waves Mod. Phys. # Examples 456 389 57 57 23 18

GPT-5.4 60.5 50.8 53.1 47.9 93.3 38.5 GPT-5 23.2 23.6 24.5 25.0 40.0 15.4 Claude-Opus-4.7a 56.4 59.2 50.0 44.4 62.5 40.0 Gemini-3.1-Proa 69.1 67.1 62.5 44.4 62.5 60.0 Qwen3.6-Flash 55.8 46.3 40.8 29.2 86.7 38.5

Qwen-3.5-9B 43.1 34.2 26.5 22.9 73.3 30.8 Qwen-3.5-27B 34.8 20.8 28.6 14.6 60.0 23.1 Gemma-4-26B-It 29.3 23.0 22.5 22.9 40.0 23.1 Gemma-4-31B-It 28.7 19.8 26.5 14.6 20.0 15.4

Mech.: Mechanics; EM: Electromagnetism; Thermo.: Thermodynamics; Optics: Optics; Waves: Waves/Acoustics; Mod. Phys.: Modern Physics. a Evaluated on the 200-problem testmini subset.

- A.3 Benchmark Embeddings

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Text Embedding Multimodal Embedding

Text Embedding Multimodal Embedding

Text Embedding Multimodal Embedding

Text Embedding Multimodal Embedding

6

- 4

2

0

- -2
- -4

-7.5 -5.0 -2.5 0.0 2.5 5.0 7.5 10.0

6

4

2

0

- -2
- -4

-7.5 -5.0 -2.5 0.0 2.5 5.0 7.5 10.0

12 10

8 6 4 2 0

- -2
- -4
- -7.5 -5.0 -2.5 0.0 2.5 5.0 7.5 10.0 12.5

12 10

8 6 4 2 0

- -2
- -4
- -7.5 -5.0 -2.5 0.0 2.5 5.0 7.5 10.0 12.5

(a) SeePhys Pro Level2 (b) SeePhys Pro Level3 (c) MathVerse Vision Intensive (d) Mathverse Vision Dominant

- Figure 7: SEEPHYS PRO and MathVerse Data Embeddings Panels (a, b) are the embeddings of text and multimodal inputs at Level 2 and Level 3 of SEEPHYS PRO. Panels (c, d) are the embeddings of text and multimodal inputs for the Vision Intensive and Vision Dominant subsets of MathVerse.

- Figure 7 provides a qualitative embedding visualization. In this view, the text-only and multimodal inputs in SEEPHYS PRO appear more separated at Level 3 than at Level 2, while the two MathVerse subsets appear broadly comparable. We use this figure as an illustrative distributional snapshot rather than as a quantitative metric.

##### A.4 Discipline-Level Results

Table 4 reports Level-3 accuracy across physics disciplines. Since Level 3 places both structural information and variables in the image, this setting directly evaluates visually grounded physics reasoning within each domain. The results should be interpreted primarily as a diagnostic breakdown rather than a fine-grained leaderboard. Mechanics and Electromagnetism dominate this split with 456 and 389 examples, respectively, while the remaining disciplines contain substantially fewer samples. Therefore, performance on Thermodynamics, Optics, Waves/Acoustics, and Modern Physics should be interpreted with caution.

### B Additional Training Diagnostic Results

This appendix reports additional visualizations generated from the SwanLab exports. Main-paper claims use smoothed curves for readability, but all summaries are computed from the raw exported validation records. Unless otherwise stated, RL runs use 16 NVIDIA GH200 GPUs. The appendix includes PhysRL-38K validation (Figure 8), post-format-saturation gains (Figure 9), full image-maskrate curves (Figure 10), and expanded cross-benchmark peak-gain plots (Figure 11).

##### B.1 Physics Training Pool Validation

PhysRL-38K is the larger source-matched physics training pool used to derive PhysRL-8K. The training examples are instance-disjoint from all benchmark test sets. As a quality check, we train Qwen3-VL-4B with GSPO on PhysRL-38K and evaluate on held-out physics validation sets, including SeePhys Pro Level 3/4 and PhysReason. Figure 8 shows two context-length settings. Both runs improve validation accuracy on SeePhys Pro and PhysReason while also increasing training reward accuracy, indicating that PhysRL-38K contains transferable physics reasoning signal rather than only SEEPHYS PRO-specific artifacts. Because some SwanLab exports contain resume-induced duplicate metric columns or repeated steps, we average duplicated values at the same step and apply light moving-average smoothing only for visualization. We include additional cross-benchmark diagnostic plots below; all summaries are computed from raw SwanLab validation exports.

seq. len 4096 seq. len 8192

(a) SeePhys Pro

(b) SeePhys Pro

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

L3accuracy

L4accuracy

15%

15%

10%

10%

5%

5%

0 100 200 300 400 500

0 100 200 300 400 500

step

step

(c) PhysReason

(d) PhysRL-38K

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

trainrewardaccuracy

50%

40%

accuracy

40%

35%

30%

30%

0 100 200 300 400 500

0 100 200 300 400 500

step

step

- Figure 8: PhysRL-38K training-pool validation with Qwen3-VL-4B. We train with GSPO on PhysRL-38K using two sequence-length settings and evaluate on held-out, unmasked validation sets. Curves show that validation accuracy improves on SeePhys Pro Level 3/4 and PhysReason, while training reward accuracy also increases. Duplicate values from interrupted/resumed runs are averaged by step before smoothing for display.

Accuracy keeps improving after format reward reaches 90%

22%

| |Normal RL Blind RL<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

20%

post-format-saturation

18%

peakaccuracygain

15%

12%

10%

8%

5%

2%

0%

Qwen2.5-VL-7BMathVerse Qwen3-VL-4BMMK12 Qwen3-VL-4BMathVerse Qwen3-VL-4B

Qwen2.5-VL-7B MMK12

SeePhysPro

- Figure 9: Post-format-saturation gains. After the format reward crosses 90%, answer accuracy can still increase substantially, especially in the Qwen2.5-VL-7B math runs highlighted in the main paper. This helps rule out a purely format-compliance explanation for blind gains.

4%

6%

8%

10%

12%

reward

SeePhys Pro: accuracy

20%

40%

60%

80%

100%

SeePhys Pro: format

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0 50 100 150 200 250 300

step

- 32%
- 33%
- 34%
- 35%
- 36%
- 37%
- 38%

reward

PhysReason: accuracy

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0 50 100 150 200 250 300

step

50%

60%

70%

80%

90%

100%

PhysReason: format

Image mask rate

10% 25% 50% 75% 90%

- Figure 10: Full image-mask-rate ablation curves. We plot accuracy and format reward for Qwen3-VL-4B trained with different image mask rates and evaluated on unmasked SeePhys Pro and PhysReason. Main-paper Figure 5 summarizes the peak gains within the first 200 updates.

Math cross-benchmark gains

| | | | | | |
|---|---|---|---|---|---|
| |Normal RL Blind RL<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

30%

25%

peakaccuracygain

Physics cross-benchmark gains

20%

Normal RL Blind RL

35%

30%

15%

peakaccuracygain

25%

10%

20%

15%

5%

10%

0%

5%

Qwen2.5-VL-7BMathVerse Qwen3-VL-4BMMK12 Qwen3-VL-4BMathVerse

Qwen2.5-VL-7B MMK12

0%

Qwen2.5-VL-7B OlympiadBench Qwen2.5-VL-7BPhyX-MCQQwen2.5-VL-7BPhyX-OEQwen2.5-VL-7BSeePhysQwen3-VL-4B OlympiadBench Qwen3-VL-4BPhyX-MCQQwen3-VL-4BPhyX-OEQwen3-VL-4BSeePhysQwen3-VL-4B

SeePhysProL1 Qwen3-VL-4B SeePhysProL2

Qwen3-VL-4B SeePhysProL3

Qwen3-VL-4B SeePhysProL4

Figure 11: Separate cross-benchmark peak-gain plots. These are expanded views of Figure 4.

### C Error-Type Clustering

Before presenting individual case studies, we summarize the manually annotated error clusters across modality-transfer levels and representative frontier models. Figure 12 reports the distribution of error types for GPT-5.4, Gemini-3.1-Pro, and Claude-Opus-4.7 from Level 1 to Level 4. The cluster distribution shows that text-level failures are dominated by physics modeling and reasoning errors, while visually richer settings introduce more structural figure reading, numerical figure reading, and rendered-text reading errors.

Level 1

###### Level 2

###### Level 3

###### Level 4

8%

14%

16%

14%

35%

GPT-5.4

23%

25%

24%

n=52

n=63

n=84

n=86

20%

65%

29% 38%

22% 9%

29%

###### Gemini-3.1-Pro

12%

21%

14%

24%

15%

26%

43%

n=58

n=56

n=67

###### n=67

24%

22% 30%

67%

22% 12%

38%

###### Claude-Opus-4.7

7%

9%

21%

11%

37%

26%

42%

42%

n=52

n=66

n=87

n=107

50%

24%

19%

30%

24%

24%

Structural figure reading Numerical figure reading

Rendered-text reading Physics modeling

Formula misuse Reasoning

Calculation Other

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

Figure 12: Error-type clustering across modality-transfer levels and models. Donut charts show the distribution of manually annotated error types for GPT-5.4, Gemini-3.1Pro, and Claude-Opus-4.7 across Level 1–4. The center value n denotes the number of analyzed errors for each model-level cell.

### D Case Study Examples

The case studies below are selected independently for interpretability and are not restricted to the same fixed model set used in Figure 12.

###### Question:

Question:

As shown in the figure, h=1.8 m, M=3 kg, m=1 kg. The coefficients of kinetic friction both between the sliding board and the small block and between the sliding board and the ground are μ=0.4. F=136 N, the horizontal distance is s=1.35 m, and g=10 m/s²; the pushing force F=136 N is removed immediately when the skateboard hits the obstacle, and the skateboard rebounds with the speed before the collision. The small block leaves the skateboard from point A and falls freely to the ground. Find the horizontal distance from the landing point of the small block to end B of the board when the small block lands, i.e. the horizontal spacing between the landing point and end B of the board. (unit: m)

At the left and right corners of the roof of a cart, one point is fixed at each corner, and each is connected by a light string to the same object at one point inside the cart.2 is in the lower left corner, and 1 is in the upper right corner.String 1 makes an angle of 30° with the horizontal roof, and string 2 makes an angle of 60° with the horizontal roof; both strings are taut. The mass of the suspended object is m. When the cart moves with a uniformly accelerated straight-line motion with acceleration magnitude 2g (g is the acceleration due to gravity), which of the following statements about the magnitudes of the tensions in string 1 and string 2 is correct? ( )

5 mg.

|[Figure 9]|
|---|

5 mg.

5 mg.

- A. When the cart accelerates uniformly to the left, the tension in string 1 is
- B. When the cart accelerates uniformly to the left, the tension in string 2 is
- C. When the cart accelerates uniformly to the right, the tension in string 1 is
- D. When the cart accelerates uniformly to the right, the tension in string 2 is 0

5 mg. 5 mg.

5 mg.

###### Answer: 10.08 m Difficulty: Level 2 Field: Friction

Answer: BCD Difficulty: Level 1 Field: Tension force

###### Response:

###### Response:

Let the small block + skateboard move together before the collision. The total mass is 4 kg, so the ground friction is f = μ(M + m)g = 16 N and the acceleration is a = (136 - 16) / 4 = 30 m/s². Using s = 1.35 m, we get v = 9 m/s. So just before collision, both the skateboard and the small block have speed 9 m/s to the right.

Take the cart’s frame. The mass is treated as in static equilibrium under gravity, the pseudo-force, and the two string tensions. The effective force makes an angle about 26.6° with the horizontal.

... Since 26.6°< 30°, the model concludes that both strings cannot remain taut at the same time. It therefore assumes one string must go slack and the other alone supports the mass. So the tension in the taut string is taken as: T =

... The skateboard rebounds with the same speed, while the small block keeps speed 9 m/s to the right.The small block leaves from point A and then falls freely.

... Initially, point C is 1.35 m to the right of the vertical through A.

|5 mg.|
|---|

... From the usual labeling in the figure, string 1 is the left string, so: = This matches A.

... s = 𝑥𝐷 − 𝑥𝐶= 5.4 - (-3.33) = 8.73

|5 mg.|
|---|

|𝑇1|
|---|

Error: Structural information misreading: it incorrectly assumes joint motion before collision and misinterprets the 1.35 m label. As a result, both the motion model and final distance are wrong.

Error: Oversimplified force model: a two-string equilibrium is reduced to a singlestring case, ignoring force balance.

Figure 13: Oversimplification of physical modelling. Models oversimplified or misread the motion structure.

#### Question:

#### Question:

|[Figure 10]|
|---|

|[Figure 11]|
|---|

A particle enters the magnetic field region with velocity horizontally to the right, perpendicular to the uniform magnetic field B and perpendicular to the magnetic field boundary. Find the magnitude of the particle's velocity (unit: m/s).

- As shown in the figure.

|2 3𝑞𝐵𝑑 3𝑚<br><br>|
|---|

Response:

Inside the magnetic field, the particle moves along a circular arc.

...

- At the right boundary, the velocity makes a 60°angle with the vertical boundary, so it makes 30° with the horizontal. The model then assumes that the particle has swept out a central angle of 30°. So it uses: d = r sin 30°= Hence, r = 2d.

###### Answer: 450 m/s

###### Answer: Difficulty: Level 3 Field: Charged-particle motion

Difficulty: Level 4 Field: Hydrogen atom

###### Response:

Treat the rotating circle as a drum of radius R = 1 m.

... The drum rotates with angular speed ω = 150 rad/s, so during the atoms’ flight, the deposit appears shifted from G to E. The arc shift is given as s = 3π cm = 0.03π m.

... Since t = s / (Rω)

|𝑟 2.|
|---|

... They go from the bottom opening to the opposite side of the drum, so the distance is the diameter: d = 2R = 2 m.

... Substituting this into the circular-motion relation, the model gets: v = So the final answer is:

|2𝑞𝐵𝑑 𝑚|
|---|

...

v ≈ 3180m/s

|2𝑞𝐵𝑑 𝑚|
|---|

v =

Error: Key information misreading: the radius and arc mark are interpreted incorrectly, which propagates to the flight-time, travel-distance, and final-speed calculations.

Error: Numeric information misreading: the labeled angle is mapped to the wrong geometric relation, which distorts the subsequent geometric setup.

Figure 14: Visual geometry grounding errors. Models misground geometric cues such as angles, radii, and arc marks.

[Figure 12]

Gemini 3.1 Pro

Question:

A light inextensible rope passes over two fixed pulleys mounted at the upper left and upper right corners of a cart, with both ends hanging vertically and each suspending a heavy object; Objects will not detach the rope remains horizontal between the two pulleys. The masses of the left and right heavy objects are both m₁=m₂=2 kg, and they always slide up and down along the outer side walls of the cart. The coefficient of sliding friction between the heavy objects and the cart side walls is μ=0.2. Neglect pulley friction. The cart moves to the right on a horizontal ground with acceleration a=g. Take g=10 m/s². Find the magnitude of the rope tension T.

###### Response:

To find the magnitude of the rope tension T, we analyze the system in the non-inertial frame of the cart. … Because the problem states that both ends of the rope hang vertically and the objects always slide up and down along the outer side walls, the objects are constrained to move exactly vertically relative to the cart. … For the object sliding upwards: T − mg − f = 𝑚𝑎𝑦 For the object sliding downwards: mg − T − f = 𝑚𝑎𝑦 Notice that the friction forces cancel out. Thus:

Constraint description misread

Over-simplified motion assumption

T = mg = 20 N.

###### Answer: 26.1 N Difficulty: Level 1 Field: Tension force

[Figure 13]

[Figure 14]

ChatGPT 5.4

Claude Opus 4.6

###### Response:

###### Response:

Over-simplified constraint

Treat the motion in the cart’s frame. … The rope length is constant, so if one mass moves down, the other moves up with the same speed/acceleration magnitude. Let the left mass have downward acceleration +α, then the right mass has downward acceleration −α. … For the left mass: mg − T − f = mα. For the right mass: mg − T + f = −mα. Adding the two equations gives 2mg − 2T = 0. So:

I need to find the tension in the rope when the cart accelerates to the right with acceleration a = g. … For both masses, N = mg, so the friction force is: f = μmg = 4 N. … Let’s assume the masses are in equilibrium in the cart’s frame. … By symmetry, both should remain stationary relative to the cart. For vertical equilibrium, T + f = mg so, T = mg(1 − μ) = 16 N. Therefore, 16 N.

False equilibrium assumption

Incorrect motion assumption

T = mg = 20 N.

Over-simplified symmetry assumption

Figure 15: Constraint and equilibrium failures. Models show constraint oversimplification, false equilibrium assumptions, and incorrect motion assumptions.

[Figure 15]

Gemini 3.1 Pro

|Question:|[Figure 16]|
|---|---|
|As shown in the figure, the correct statement is ()<br><br>(A). The maximum rate of change of the magnetic flux through the iron core is 0.2 Wb/s<br>(B). The frequency of the alternating current is 50 Hz<br>(C). The ammeter A reads 0.4 2 A<br><br>(D). The input power of the transformer is 16 W.<br><br><br>Choice:| |
| |Answer: D Difficulty: Level 3 Field: Inductive coupling|

|Response:<br><br>Based on the provided figure and information, let’s analyze each option step by step. … The input voltage is given by u = 200 2 sin(100πt) V. … The angular frequency is ω = 100π rad/s. The frequency of the alternating current is 50 Hz. This makes Option B correct. … This makes Option D correct. … Both statements B and D are factually correct.<br><br>Misread numerical value<br><br>Frequency derived from misread data|
|---|

[Figure 17]

[Figure 18]

Claude Opus 4.6

ChatGPT 5.4

|Response:<br><br>Let’s analyze the transformer step by step. … Primary voltage: u₁ = 200 2 sin(100πt) V … The angular frequency is ω = 100π, thus f = 50 Hz. So B is correct. … For an ideal transformer, the input power equals the output power, so D is correct. … Therefore the correct statements are B and D.<br><br>Misread numerical value<br><br>Frequency derived from misread data<br><br>|
|---|

| |Response:<br><br>I need to analyze this transformer circuit with the given information. … Given information: Primary coil: n₁ = 100 turns Primary voltage: u = 20 2 sin(100πt) V … The frequency is 50 Hz. So Option B is correct. … Option D: The input power is 16 W. … <answer>BD</answer><br><br>Frequency derived from misread data<br><br>Misread numerical value<br><br>|
|---|---|
| | |

Figure 16: Transformer numerical misreading. Models misread numerical values in the visual input, which propagates into incorrect frequency and option judgments.

Response:

Question:

|[Figure 19]|
|---|

|For a smooth contact, the reaction is perpendicular to the beam. So the forces on the beam are: Weight W= 200 N, acting at the midpoint of 𝐴𝐵. Floor reaction at 𝐵, vertical upward . Reaction at 𝐶, call it 𝑅𝐶 ,perpendicular to the beam . Reaction at 𝐷, call it 𝑅𝐷 ,also perpendicular to the beam … Therefore, the horizontal components of the reactions at 𝐶and 𝐷must cancel:<br><br>𝑹𝑪 𝒄𝒐𝒔 𝟑𝟎∘ = 𝑹𝑫 𝒄𝒐𝒔 𝟑𝟎∘ So, 𝑅𝐶 = 𝑅𝐷<br><br>… From the geometry, 𝑅𝐶 and 𝑅𝐷 act in opposite senses about 𝐵, so:<br><br>𝑹𝑫 𝟏.𝟎 − 𝑹𝑪 𝟎.𝟓 − 𝟏𝟓𝟎 = 𝟎 Using 𝑅𝐶 = 𝑅𝐷 = 𝑅, 𝑅 1.0 − 𝑅 0.5 = 150 Hence,<br><br>𝑅𝐶 = 300 N<br><br>Correct horizontalforce relation<br><br>Correct moment balance|
|---|

As shown in the figure, Ignoring frictional force，the uniform beam AB weighs 200 N and makes an angle of 60°with the floor. Given that AB=3 m, CB=0.5 m, and BD=1 m, find the reaction force at point C (unit: N).

Difficulty:

- Level 2

Difficulty:

- Level 3

Response:

|[Figure 20]|
|---|

|Let the beam be in equilibrium under these forces: its weight 𝑊 = 200 N, acting at the midpoint of 𝐴𝐵, the reaction at 𝐶, call it 𝑅𝐶 ,the reaction at 𝐷, call it 𝑅𝐷 ,the floor reaction at 𝐵. … At 𝐶, the reaction is perpendicular to the beam, so its moment arm about 𝐵is simply<br><br>𝐵𝐶 = 0.5 m. Also, since the beam is at 60∘ ,the reaction 𝑅𝐶 has horizontal component<br><br>𝑅𝐶 cos30∘ = 𝑅𝐶<br><br>3 2<br><br>= 0.866𝑅𝐶. Because 𝑅𝐷is horizontal, horizontal force balance gives<br><br>𝑹𝑫 = 𝟎.𝟖𝟔𝟔𝑹𝑪. …<br><br>So, 0.866𝑅D − 0.5𝑅C − 200 0.75 = 0. Substitute 𝑅𝐷 = 0.866𝑅𝐶: 𝟎. 𝟖𝟔𝟔 𝟎.𝟖𝟔𝟔𝐑𝐂 − 𝟎. 𝟓𝐑𝐂 − 𝟏𝟓𝟎 = 𝟎 So the reaction force at point 𝐶 is:<br><br>600 N<br><br>Wrong reaction-direction assumption<br><br>Wrong moment equation|
|---|

Question:

As shown in the figure.Ignoring frictional force,find the reaction force at point C (unit: N).

Answer: 300 N

Figure 17: Reaction-force reasoning. The examples contrast correct force-balance reasoning with wrong reaction-direction and moment-equation assumptions.

Response:

Question:

Treat plate A as connected to ground through two capacitors in parallel:

|[Figure 21]|
|---|

As shown in the figure, the areas are all S=2.0×10⁻² m², the distance

- between A and B: C₍AB₎ = ε₀S / d₁
- between A and C: C₍AC₎ = ε₀S / d₂ … Q₍AC₎ = 2 / 3 Q = (2 / 3)(3.0 × 10⁻⁷ C)

Correct read numeric value

Result: 300 N

- between plates A and B is d₁=4.0×10⁻³ m, the distance
- between plates A and C is d₂=2 mm, and both plates B and C are grounded. If plate A is positively charged with Q=3.0×10⁻⁷ C, find: how much induced negative charge is on plate C?

= 2.0 × 10⁻⁷ C This is the positive charge on the A-side facing C, so the induced charge on plate C is equal in magnitude and negative: q꜀ = -2.0 × 10⁻⁷ C

So the amount of induced negative charge on plate C is: 2.0 × 10⁻⁷ C

Result: 2.0 × 10⁻⁷ C

Difficulty:

- Level 2

Difficulty:

- Level 3

Response:

Treat the middle plate A and the two grounded plates C and B as two capacitors in parallel: capacitor between A and C: C₁ = ε₀S / d₁ capacitor between A and B: C₂ = ε₀S / d₂ … q₁ = Q × (1 / d₁) / (1 / d₁ + 1 / d₂) = Q × d₂ / (d₁ + d₂) Given: d₁ = 4 × 10⁻³ m, d₂ = 2 × 10⁻³ m, Q = 3 × 10⁻⁸ C q₁ = 3 × 10⁻⁸ × (2 × 10⁻³) / (4 × 10⁻³ + 2 × 10⁻³)

|[Figure 22]|
|---|

Question:

Result: 600 N

As shown in the figure, how much negative induced charge is on plate C?

Misread numeric value

= 3 × 10⁻⁸ × 2 / 6

= 1 × 10⁻⁸ C This is the magnitude of charge induced on the side facing C. Since the induced charge on grounded plate C is negative,

-1.0 × 10⁻⁸ C

Result: -1.0 × 10⁻⁸ C

Answer: 2.0 × 10⁻⁷ C

Figure 18: Induced-charge calculation. The examples show how correct physical grounding can be disrupted by numerical misreading.

