# arXiv:2505.18129v3[cs.CV]16Apr2026

[Figure 1]

Full author list in Contributions1

Reinforcement learning (RL) is becoming an important direction for post-training vision-language models (VLMs), but public training methodologies for unified multimodal RL remain much less mature, especially for heterogeneous reasoning and perception-heavy tasks. We propose V-Triune, a Visual Triple Unified Reinforcement Learning methodology for unified multimodal RL. It organizes training around three coordinated abstractions: Sample-Level Reward Routing, Verifier-Level Outcome Verification, and Source-Level Diagnostics. Within this methodology, Dynamic IoU provides localization-specific reward shaping that avoids reward ambiguity under loose thresholds and reward sparsity under strict ones. Built on V-Triune, we develop Orsta (7B, 32B), a family of models jointly trained on eight reasoning and perception tasks. Under matched budgets, unified training matches or outperforms specialist mixtures. The final Orsta models improve over their backbones on MEGA-Bench, compare favorably with strong multi-task RL-VLM baselines, and transfer these gains to a broad set of downstream benchmarks. These results show that unified RL can improve both reasoning and perception within a single VLM RL pipeline.

## 1. Introduction

Reinforcement learning (RL) is becoming an important direction for post-training foundation models (Liu et al., 2025a; Xiao et al., 2026). Compared with text-only settings, however, public and reproducible training methodologies for multimodal RL remain much less mature (Kimi et al., 2026; Qwen Team, 2026). Although more works have begun to explore RL post-training in visual settings (Liu et al., 2025b,c,d; Ma et al., 2025a; Shen et al., 2025; Tan et al., 2025; Wang et al., 2025b; Yu et al., 2025), a stable methodology for scaling RL across heterogeneous visual tasks remains an important open problem for vision-language models (VLMs).

For VLMs, this challenge lies in handling both capability heterogeneity and verification heterogeneity within one pipeline. Post-training must cover both reasoning-heavy and perception-heavy tasks, while their outcomes may require symbolic checking or continuous spatial feedback such as IoU. A unified multimodal RL methodology must therefore support heterogeneous reward structures and verification regimes within the same optimization loop. We study unified RL post-training for visual tasks with verifiable outcomes spanning both reasoning-heavy and perception-heavy settings.

Existing work usually addresses only part of this problem, focusing either on visual reasoning (Huang et al., 2025; Meng et al., 2025; Wang et al., 2025a; Yang et al., 2025) or on perception tasks such as detection and grounding (Liu et al., 2025b,c,d; Ma et al., 2025a; Shen et al., 2025; Tan et al., 2025; Wang et al., 2025b; Yu et al., 2025). A clear and stable RL framework for jointly improving a single VLM on both high-level reasoning and fine-grained perception remains missing.

1Please send correspondence to model@minimaxi.com.

© 2026 MiniMax. All rights reserved

In practice, unified multimodal RL breaks down for three concrete reasons. First, different tasks, and even different samples, can require different reward compositions and verifiers, so hard-coding this logic into the trainer quickly becomes brittle and difficult to scale. Second, for localization tasks such as detection and grounding, fixed IoU thresholds create two failure modes: loose thresholds blur reward differences between coarse predictions, whereas strict thresholds make rewards too sparse for stable learning. Third, in joint training, aggregate metrics often hide source-specific instability, reward degradation, or other failures, making them difficult to detect and diagnose in time.

To address these blockers, we introduce Visual Triple Unified Reinforcement Learning (V-Triune), a training methodology for unified multimodal RL over heterogeneous visual tasks. V-Triune organizes unified training around three coordinated abstractions: sample-level reward routing (§3.1), which decouples reward composition and verifier choice from the trainer core; verifier-level outcome verification (§3.2), which provides a common interface for heterogeneous rewarding; and sourcelevel diagnostics (§3.3), which expose source-specific failures hidden by aggregated metrics. Within this methodology, Dynamic IoU Reward (§3.2.1) serves as a localization-specific reward-shaping mechanism by progressively increasing the IoU requirement, avoiding both reward ambiguity under loose thresholds and reward sparsity under strict ones.

Built on V-Triune, we develop the Orsta (One RL to See Them All) model family at both 7B and 32B scales and jointly train these models on eight representative tasks under a single RL pipeline, including reasoning tasks such as mathematics, science, chart, and puzzle, as well as perception tasks such as detection, grounding, OCR, and counting. Under matched budgets, unified training matches or outperforms specialist mixtures while remaining strong across both reasoning and perception within a single RL pipeline. Orsta improves over its base models on MEGA-Bench (Chen et al., 2024), a comprehensive benchmark spanning over 440 diverse visual tasks, and these gains further transfer to downstream benchmarks. Taken together, these results suggest that V-Triune provides an effective and practical training methodology for joint RL post-training of reasoning and perception in open VLMs.

Our main contributions are threefold. First, we formulate unified VLM post-training as a heterogeneous multimodal RL problem and identify three practical blockers: rigid reward interfaces, the ambiguity-versus-sparsity trade-off in localization rewards, and lack of source-level observability during joint training. Second, we propose V-Triune, a training methodology built on sample-level reward routing, verifier-level outcome verification, and source-level diagnostics, together with Dynamic IoU reward shaping for localization tasks. Third, we instantiate this methodology in the Orsta family and validate it through eight-task unified training, matched-budget specialist comparisons, MEGA-Bench evaluation, broader reasoning and perception benchmarks, and extension to a new GUI task domain.

## 2. Related Work

RL post-training is becoming important for foundation models and is extending from language models to multimodal models (Kimi et al., 2026; Liu et al., 2025a; Meituan LongCat et al., 2025; Qwen Team, 2026; Xiao et al., 2026). However, public multimodal RL recipes remain much less mature than their text-only counterparts, especially for setups that span heterogeneous tasks and reward regimes. As a result, most existing VLM-RL work is still organized around particular capability families or relatively homogeneous task settings, rather than a unified recipe for jointly training reasoning-heavy and perception-heavy tasks.

One major line of work focuses on multimodal reasoning. Vision-R1 (Huang et al., 2025), R1OneVision (Yang et al., 2025), MM-Eureka (Meng et al., 2025), VL-Rethinker (Wang et al., 2025a), and GThinker (Zhan et al., 2025) mainly target mathematics, science, and related reasoning tasks,

###### 2. Verifier-Level Outcome Verification

###### 1. Sample-Level Reward Routing

###### 3. Source-Level Diagnostics

Data Source C

Data Source B

Diverse Visual Datasets

Data Source A

Verifier Router

###### Training Reward

Response Behavior

Reasoning Perception

|Correct Avg. Len|
|---|

[Figure 2]

0-1 Rule-based Accuracy Reward

[Figure 3]

share same verifier interface

|Wrong Avg. Len|
|---|

Dynamic Threshold IoU Reward

|Average Length|
|---|

x Replicas

|Truncated Rate|
|---|

|Format Reward|
|---|

...

...

Verifier Verifier Verifier

Auxiliary Metric

Reflection

Data Schema

Fixed threshold IoU @(50, 75, 95, 99)

Routing based on the Key

Reflection Ratio

[Figure 4]

|ViT/LLM Gradient|
|---|

Mixed-Task Data Sample

Correct Ratio in Reflection Resp.

Detection Verifier accuracy format

MathVerify Verifier accuracy

Response Examples

image: ... prompt: ... data source: ... accuracy ratio: float format ratio: float verifier: MathVerify / Detection

metadata

[Figure 5]

[Figure 6]

.....

[Figure 7]

[Figure 8]

.....

[Figure 9]

[Figure 10]

IoU Threshold

2.1. Dynamic IoU for Precision Enhancement

|Grouped by<br><br>key|
|---|

[Figure 11]

Reward Values

Mixed-Task Data Batch with Responses

Mixed-Task Data Batch with Verified Rewards

IoU Threshold

- Figure 1 | Overview of V-Triune. The methodology organizes unified multimodal RL around three training abstractions: sample-level reward routing, verifier-level outcome verification, and source-level diagnostics. They address rigid reward interfaces, localization reward ambiguity versus sparsity, and lack of source-level observability in mixed-task training.

typically using rule-based rewards and R1-style post-training to improve reasoning. These works have substantially advanced multimodal reasoning RL, but their task composition and reward design remain centered on reasoning-heavy problems. Another line focuses on perception-heavy or relatively homogeneous visual tasks. Visual-RFT (Liu et al., 2025d), DeepPerception (Ma et al., 2025a), Reason-RFT (Tan et al., 2025), Perception-R1 (Yu et al., 2025), VLM-R1 (Shen et al., 2025), and VisionReasoner (Liu et al., 2025b) show that RL can be effective for detection, grounding, OCR, counting, segmentation, and related tasks, often with task-specific rewards such as IoU or mAP. Taken together, these two lines do not directly address how reasoning-heavy and perception-heavy tasks can stably coexist within the same RL training framework.

A smaller set of recent work moves toward broader multimodal RL settings. Mixed-R1 (Xu et al., 2025) studies multimodal RL with mixed answer types and reward forms, including open-ended text rewards, while OneThinker (Feng et al., 2025) addresses heterogeneous image-video training through optimizer-side reward normalization and a GRPO (Shao et al., 2024) variant. These works move toward broader multimodal RL settings, but they address different problems from ours. Our work instead studies a public methodology for jointly training open VLMs on reasoning-heavy and perception-heavy tasks, where capability heterogeneity, verifier heterogeneity, localization-specific reward shaping, and source-level observability must all be handled within one pipeline.

## 3. V-Triune: Visual Triple Unified Reinforcement Learning

This section presents V-Triune, our training methodology for unified multimodal RL in vision-language models. The central problem is how to stably train reasoning-heavy and perception-heavy tasks within a single RL pipeline despite their different reward interfaces, outcome verification regimes, and observability requirements.

In practice, this setting breaks down in three recurring ways. First, heterogeneous tasks do not fit a rigid shared reward interface, and hard-coded task-specific reward branches make the trainer brittle and difficult to extend. Second, localization-centric tasks face an ambiguity-versus-sparsity trade-off under fixed IoU thresholds. Third, aggregate-only monitoring can hide source-specific failure modes, making training problems difficult to localize.

V-Triune addresses these issues through three coordinated abstractions, shown in Fig. 1. Samplelevel reward routing defines a clean boundary between the trainer and task-specific reward logic. Verifier-level outcome verification provides a common interface for heterogeneous reward computation and incorporates Dynamic IoU for localization tasks. Source-level diagnostics restore the observability that aggregate-only monitoring cannot provide. Together, these abstractions make heterogeneous rewards, verification, and failure signals manageable within one pipeline, supporting stable unified training.

#### 3.1. Sample-Level Reward Routing

Unified multimodal RL needs an extensibility boundary between the trainer and task-specific reward logic. Without such a boundary, adding new task families means inserting special-case reward branches into the training loop, which quickly becomes brittle and hard to maintain. V-Triune makes this boundary explicit through sample-level reward routing.

Each sample carries a compact routing specification that determines which reward components should be used, how they are weighted, and which verifier should evaluate the rollout. The trainer consumes only this shared routing interface, while task-specific verification logic stays outside the update loop. This turns heterogeneous reward handling from trainer-side branching into a unified sample-side interface.

In practice, the routing metadata includes component weights, a verifier key, and the source identifier later used for diagnostics. In this work, sample-level routing makes standard mixed-task RL explicit rather than introducing per-sample reward tuning. The metadata only selects among a small set of verifier-defined reward templates, making explicit the routing that heterogeneous verifiablereward training already requires. The full schema is shown in Fig. 8. What matters methodologically is that the trainer no longer needs task-specific reward code paths in order to mix reasoning and perception tasks in one run.

This design defines a clean extensibility boundary: adding a new task requires preparing data that matches an existing verification regime, or registering a new verifier, without rewriting the trainer itself. The same routing metadata also supports source-aware diagnosis by allowing rewards and behaviors to be regrouped by source after verification.

#### 3.2. Verifier-Level Outcome Verification

Sample-level routing determines which verification path a rollout should follow, but unified training still needs a common interface for heterogeneous outcome evaluation. Some tasks can be checked with deterministic rules after parsing the model output, whereas others require continuous spatial

scoring. V-Triune handles this at the verifier level: the trainer sends predictions and references to the designated verifier, which performs parsing, verification, and reward computation through a shared interface. This abstraction lets the trainer interact with heterogeneous reward logic in one way while preserving task-specific verification inside each verifier. In this work we primarily instantiate two verifiers, corresponding to two verification regimes:

#### 3.2.0.1 MathVerifyVerifier: Rule-verifiable outcomes

This verifier handles tasks whose outputs can be parsed and deterministically checked, including mathematics, puzzles, science, chart reasoning, OCR, and counting. For these tasks, we parse the model output into a normalized answer and verify it against the reference using math_verify (Kydlíček, 2025). The resulting accuracy reward follows the standard 0–1 rule-based form:

𝑅acc(𝑎,ˆ 𝑎) = 𝕀(verify(parse(𝑎ˆ),parse(𝑎))) (1)

where 𝑎ˆ denotes the predicted answer and 𝑎 the ground-truth answer. In our setup, model responses are instructed to place the final answer inside \boxed{}.

#### 3.2.0.2 DetectionVerifier: Localization-centric outcomes

This verifier handles tasks such as detection and grounding, where outputs must satisfy both a structural format and a spatial accuracy criterion. We therefore compute a composite reward with separate format and accuracy terms. To enforce the required output structure, we define a format reward as

∑︁4

𝑅format(𝑜𝑞) = 0.25

𝕀(count(𝑜𝑞, 𝑠𝑖) = 1) (2)

𝑖=1

where 𝑜𝑞 is the model response to query 𝑞, and the four format tags are {𝑠𝑖}4𝑖=1 ={<think>, </think>, <answer>, </answer>}.

For the spatial component, we use IoU-based accuracy:

IoU(𝑎,ˆ 𝑎), if IoU(𝑎,ˆ 𝑎) ≥ 𝜖 0, else

Area(𝑎ˆ ∩ 𝑎) Area(𝑎ˆ ∪ 𝑎)

, where IoU(𝑎,ˆ 𝑎) =

𝑅acc(𝑎,ˆ 𝑎) =

(3)

where 𝑎ˆ is the predicted box and 𝑎 is the ground-truth box. The final reward combines the two parts as 𝛼acc · 𝑅acc + 𝛼format · 𝑅format, where 𝛼acc and 𝛼format are specified by the sample-level routing metadata.

Verifier-level computation gives different task families a clear boundary for reward logic while keeping the trainer unchanged. For new task domains that can reuse an existing output structure and verification regime, only the data routing needs to be updated. We verify this point with the GUI-domain extension experiment in Sec. 4.7.

#### 3.2.1. Dynamic IoU for Precision Enhancement

Within DetectionVerifier, the main optimization challenge is to construct a localization reward that is informative without becoming too sparse. IoU is the most direct spatial signal for detection and grounding, but a fixed threshold creates a practical dilemma.

With a loose threshold such as IoU@50 (Yu et al., 2025), reward density is high but reward ambiguity remains severe: multiple coarse boxes can receive nearly identical rewards, leaving limited pressure for finer localization. With a highly strict threshold such as near-exact matching, the objective becomes clearer but early rollouts receive almost no positive feedback, creating a cold-start problem. This issue is amplified in unified training because localization rewards must coexist with sharper rule-based rewards from reasoning tasks.

To address this trade-off, we introduce Dynamic IoU, a dense-to-strict reward shaping mechanism. Rather than fixing one threshold, we use a simple three-stage schedule that begins with denser supervision, then progressively raises the IoU requirement, and finally enforces near-exact localization later in training. This staging preserves learnability in early optimization while still imposing highprecision supervision in the late phase.

Dynamic IoU makes high-precision localization learnable under joint training by combining dense early supervision with stricter late-stage refinement. We provide the exact thresholds and stage boundaries in the implementation details, and compare this design against fixed thresholds and alternative schedules in the experiments and appendix.

- 3.3. Source-Level Diagnostics

Even with sample-level routing and verifier-level computation, unified RL can still fail in ways that aggregate metrics do not expose. Global averages may remain seemingly stable while individual sources undergo reward collapse, format drift, abnormal token generation, local task suppression, or optimization instability. V-Triune therefore tracks verified samples at the source level using the same data_source key introduced in the routing interface.

We organize these signals into four groups: training reward, response behavior, reflection-related metrics, and auxiliary diagnostics. Training reward metrics provide a source-level breakdown of rule-based accuracy, format reward, and IoU-based reward. Response behavior metrics include average length, lengths of correct and incorrect responses, and truncation rates, which help reveal verbosity drift or collapse. Reflection metrics provide a lightweight online proxy for response strategy by tracking whether reflective cues appear in responses; the precise definition is given in Sec. I. Auxiliary diagnostics include fixed-threshold IoU, ViT/LLM gradients, and response examples for qualitative analysis.

In our experiments, these diagnostics directly informed two training decisions in the final recipe: freezing the vision encoder after observing ViT gradient explosion, and filtering leaked image special tokens before reward recomputation (Sec. B). They also reveal divergent response-length and reflection patterns across task families and help detect when one source is being overshadowed by stronger training signals elsewhere. Source-level diagnostics are therefore part of the training methodology rather than a dashboard add-on: they make unified training actionable and diagnosable.

- 4. Experiment

- 4.1. Experimental Setup

We evaluate V-Triune at two levels. First, under a fixed budget, we compare a unified mixture against two specialist mixtures: Reason-only and Perception-only. Second, we report the final Orsta models under a longer training horizon to evaluate overall competitiveness on broad and task-specific benchmarks.

We use Qwen2.5-VL-7B and 32B (Bai et al., 2025) as backbones. We study unified RL post-training

- Table 1 | Fair-budget evidence under a fixed training budget. Left: MEGA-Bench task-composition curves for the 7B off-policy setting. Right: benchmark breakdown at the same 60-step budget. Unified

= Reason+Perception. MME-R = MME-Reasoning (Yuan et al., 2025). CharXiv(RQ) = CharXiv Reasoning Question. COCO (S|M) reports single-object and multi-object mAP in COCO val-2017. COCO mAP uses the standard cocoapi; see details in Sec. E. Best results are bolded and second-best results are underlined.

Ablation on Training Tasks

Reasoning Benchmarks Mixture MMMU MathVista MathVision MME-R CharXiv (RQ)

37.5

37.0

Unified 56.56 71.40 30.51 28.20 44.50 Reason-only 54.89 71.70 31.24 28.03 44.00 Perception-only 56.67 68.20 29.59 26.26 41.00

36.5

| |
|---|

Score

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

36.0

| |
|---|

Perception Benchmarks Mixture HrBench4K VStar COCO (S|M) OCRBenchV2 ScreenSpot-Pro

35.5

| |
|---|

Reason+Perception

Reason

35.0

Perception

Unified 73.75 82.20 80.69 | 38.62 55.87 23.91 Reason-only 73.25 78.53 78.36 | 33.87 55.77 23.97 Perception-only 76.00 80.10 80.78 | 37.37 55.33 23.78

0 20 40 60

Step (a)

on strong public VLM backbones when the target outcomes are verifiable. RL post-training uses a 47.7K-sample VQA training set with verifiable answers from 18 sources, covering four reasoning tasks (math, puzzle, science, and chart) and four perception tasks (detection, grounding, counting, and OCR). Detailed data construction is given in Sec. A and Tab. 5. Appendix Sec. F.1 further reports a matched-count curated-versus-random control to separate data quality from the training methodology. We use GRPO (Shao et al., 2024) in all main experiments, with a rollout batch size of 1024 and 8 sampled responses per prompt, and implement the unified RL pipeline on top of Verl (Sheng et al., 2024). Controlled comparisons use a fixed training budget, whereas final-model results use 3 training epochs. For stability, we freeze the vision encoder and update only the LLM layers. Rollouts are generated with vLLM using temperature=1.0, top-p=1.0, and max_length=2048, and localization tasks use a three-stage Dynamic IoU schedule.

#### 4.2. Does Unified Training Help Under Fair Budgets?

We first study unified training under a fixed training budget. Unless otherwise stated, all fair-budget comparisons use the same 7B off-policy with 8 optimization steps per rollout. We compare the unified mixture against two specialist mixtures under the same budget.

The left panel of Tab. 1 provides the first evidence through the MEGA-Bench task-composition curves. Under the same 60-step budget, Reason+Perception follows the strongest or tied-strongest trajectory throughout training. This suggests that joint training does not introduce a clear reasoningversus-perception trade-off on MEGA-Bench.

We further compare the same three mixtures on a 10-benchmark suite spanning both reasoning and perception, and report the fixed-budget benchmark breakdown in Tab. 1. The benchmark list and evaluation protocols are given in Sec. D. The unified mixture remains stronger or comparable on both sides and reaches the best result on multiple benchmarks. Across this suite, the unified model matches or exceeds the specialist baselines on 5/10 benchmarks and stays close on the rest.

This is consistent with improved budget efficiency under unified training, since the unified model remains competitive despite receiving less task-specific exposure per task. One plausible explanation is mild cross-task regularization, where exposure to both reasoning and perception tasks improves

- Table 2 | Performance on MEGA-Bench.

Model Score 7B Models

Qwen2.5-VL-7B 35.07 MM-Eureka-7B 35.96 VL-Rethinker-7B 37.25 Orsta-7B 38.31 Δ (Ours - Backbone) +3.24

32B Models (0321)

Qwen2.5-VL-32B-0321 11.87 MM-Eureka-32B 18.57 VL-Rethinker-32B 19.41 Orsta-32B-0321 25.94 Δ (Ours - Backbone) +14.07

32B Models (0326)

Qwen2.5-VL-32B-0326 43.67 Gemma3-27B 41.82 InternVL-3-38B 46.69 Orsta-32B-0326 45.77 Δ (Ours - Backbone) +2.10

##### Table 3 | Comparison against strong multi-task RL-VLM baselines on reasoning and perception benchmarks.

Reasoning Benchmarks Model MMMU MathVista MathVision MME-R CharXiv (RQ)

Orsta-7B 57.10 72.50 31.73 31.14 48.40 MM-Eureka-7B 55.33 74.10 30.84 28.45 42.10 VL-Rethinker-7B 56.70 75.40 32.46 29.38 44.00 VisionReasoner-7B 56.56 69.70 29.20 25.84 41.20

Perception Benchmarks Model HrBench4K VStar COCO (S|M) OCRBenchV2 ScreenSpot-Pro

Orsta-7B 77.25 81.68 80.73 | 41.41 56.05 23.91 MM-Eureka-7B 59.62 57.07 79.73 | 35.84 53.38 24.23 VL-Rethinker-7B 65.12 68.60 72.50 | 31.54 55.70 24.48 VisionReasoner-7B 74.38 80.63 80.22 | 36.58 55.44 24.23

##### Table 4 | Extension to a new GUI task domain with 3K ShowUI samples under the same 60-step fixed-compute budget.

Mixture ScreenSpot-Pro OCRBenchV2

Unified 23.91 55.87 Perception-only 23.78 55.33 Perception-only + GUI-3K 29.85 55.96 Unified + GUI-3K 31.68 56.09

shared learning signals beyond specialist training alone. We leave a deeper mechanistic analysis of this effect to future work.

#### 4.3. General Performance on MEGA-Bench

The fixed-budget results in Sec. 4.2 show that unified RL training matches or outperforms specialist mixtures under matched budgets. We now turn to the final Orsta models and report their overall performance on MEGA-Bench, which spans over 440 diverse tasks for general VLM capability. For each backbone, we train both on-policy and off-policy variants for 3 epochs, designate one model as Orsta based on its MEGA-Bench performance, and evaluate that same selected model on all subsequent benchmarks. We provide the full MEGA-Bench evaluation curves in Sec. K.

Overall, Orsta consistently improves the MEGA-Bench performance of its base model at both 7B and 32B scales. For the 7B model, unified RL post-training yields a clear overall gain. The same pattern holds at 32B, but with two different gain profiles in Tab. 2: for the 0321 checkpoint,2 RL post-training brings a much larger improvement; for the stronger 0326 checkpoint, V-Triune still delivers stable and notable gains.

#### 4.4. Comparison with Strong Multi-task RL-VLM Baselines

Beyond the overall MEGA-Bench gains in Sec. 4.3, we compare Orsta-7B against three strong multi-task RL-VLM baselines on the same 10-benchmark suite as in Sec. 4.2. MM-Eureka-7B (Meng et al., 2025)

20321 and 0326 follow the release dates of the public HuggingFace checkpoints. The former shows noticeably weaker perception and formatting abilities, while the latter is a stronger later release.

COCO Val (Multi-Objects)

COCO Val (Multi-Objects)

Source-level Training Analysis

COCO Val (Multi-Objects)

Gradient Norm

800

0.5

- 0

- 1

- 2

- 3

- 4

- 5

0.56

Length(Puzzle:PuzzleVQA)

ViT+LLM

ViT+LLM

0.56

Length(Detection:V3Det)

ViT

ViT

Reflection Ratio(Puzzle:PuzzleVQA)

| | | | |
|---|---|---|---|
| | | | |

LLM

LLM

700

| | | |
|---|---|---|
| | | |

Reflection Ratio(Detection:V3Det)

0.4

| | |
|---|---|
| | |

0.55

| |
|---|

| |
|---|

| |
|---|

0.55

ResponseLength

| |
|---|

| |
|---|

ReflectionRatio

| |
|---|

| | |
|---|---|

| |
|---|

| | |
|---|---|

| | |
|---|---|
| | |

| |
|---|

IoU@50Score

IoU@50Score

| |
|---|

600

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|

| |
|---|

| |
|---|

| |
|---|

0.54

| |
|---|

| |
|---|

0.3

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

| | |
|---|---|

0.54

| |
|---|

| |
|---|

| |
|---|

500

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0.53

| | |
|---|---|
| | |
| | |

| |
|---|

0.2

| |
|---|

| |
|---|

400

0.53

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| | |
|---|---|

0.52

| |
|---|

| |
|---|

| |
|---|

0.1

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

300

| |
|---|

| |
|---|

IoU@Dynamic

IoU@Dynamic(our,0.85 0.95 0.99)

0.52

| |
|---|

0.51

IoU@99 IoU@50

- IoU@Dynamic(var1,0.7 0.8 0.85)

- IoU@Dynamic(var2,0.9 0.95 0.99)

| |
|---|

| |
|---|

200

0.0

0 20 40 60 80

0 20 40 60 80

0 25 50 75 100 125

0 50 100

0 50 100

(a) (b) (c)

- Figure 2 | Dynamic IoU ablations and source-level diagnostics. (a) Dynamic IoU ablations on COCO multi-object using only the detection and grounding subsets for training. The left plot compares fixed IoU@50, fixed IoU@99, and the main Dynamic IoU schedule; the right plot compares alternative staged schedules. Fixed IoU@50 learns quickly but degrades later, whereas fixed IoU@99 is more stable but learns more slowly; the main Dynamic IoU schedule gives the best balance. (b) Source-level logs from full 8-task Orsta-32B unified RL training show that the reasoning-heavy Puzzle source has increasing response length and reflection ratio, whereas the localization-centric Detection source remains short and nearly reflection-free. (c) In a full 8-task 7B training ablation that updates the ViT only, the LLM only, or both, continuing to update the vision encoder hurts COCO performance and increases gradient norms, motivating the decision to freeze the vision encoder and connector in the main experiments. More detailed failure analyses are provided in Sec. B.

and VL-Rethinker-7B (Wang et al., 2025a) are more reasoning-oriented, whereas VisionReasoner7B (Liu et al., 2025b) is more perception-centric.

As shown in Tab. 3, Orsta-7B achieves the best score on 7/10 benchmarks. On the reasoning side, it wins MMMU, MME-Reasoning, and CharXiv(RQ), while staying competitive on MathVista and MathVision. On the perception side, it reaches the top score on 4/5 benchmarks, with the clearest margin on the challenging COCO multi-object setting. This complements the fair-budget evidence in Sec. 4.2: the benefit of unified training remains visible in the final model and is not confined to either reasoning or perception alone.

#### 4.5. Why is Dynamic IoU Necessary for Localization Tasks?

We next examine why Dynamic IoU is necessary for localization-centric tasks and why a staged schedule is effective. Unless otherwise stated, the ablations in this subsection use only the detection and grounding subsets of the training data in the same 7B off-policy setup. The main schedule uses IoU thresholds 0.85, 0.95, and 0.99 over the first 10%, the next 15%, and the remaining training steps.

On COCO multi-object, the fixed-threshold comparison in Fig. 2(a) shows the two failure modes directly. IoU@50 learns quickly but degrades later, which we attribute to reward ambiguity under a loose threshold. Qualitative cases in Sec. L support this interpretation: late-stage predictions often drift among multiple coarse boxes that remain similarly rewarded under IoU@50, rather than continuing to sharpen around the ground-truth box. IoU@99 is more precise but learns much more slowly because the reward is too sparse early on. Dynamic IoU avoids both issues by remaining learnable early while enforcing higher precision later. We further compare alternative staged schedules in Fig. 2(a). The loose variant (0.7 → 0.8 → 0.85) plateaus because its final threshold remains too permissive, whereas the strict variant (0.9 → 0.95 → 0.99) slows early learning by making the initial reward too sparse. Our main schedule (0.85 → 0.95 → 0.99) gives the best balance.

The same advantage pattern also appears on the OVDEval negation subset; full curves are given

in Sec. M. We also implemented an adaptive scheduler based on the batch-level bbox success rate, but it does not yield a clear advantage over the fixed three-stage schedule in our setting; full results are given in Sec. J.

#### 4.6. What Do Source-Level Diagnostics Reveal?

These diagnostics reveal signals aggregate metrics can hide: global averages may look stable while source-specific response behavior and failure modes diverge during training (Fig. 2).

On the behavior side, Fig. 2(b) uses Puzzle and Detection as two representative sources and shows unified training does not collapse them into the same response pattern. For the reasoning-heavy Puzzle source, both response length and reflection ratio increase over training. For the localizationcentric Detection source, responses remain short and direct, and reflection stays low. This suggests unified training preserves task-appropriate response patterns across task families. Combined with the fair-budget results, these diagnostics suggest that joint training improves robustness across task families without forcing them into a uniform response pattern. More detailed multi-source dynamics are provided in Sec. C.

On the stability side, Fig. 2(c) shows that continuing to update the vision encoder quickly hurts COCO performance and increases total gradient norms. Sec. B further shows that this instability appears as gradient explosion in the ViT and soon hurts detection and grounding performance. The same source-level logs also exposed leaked image special tokens in model responses, which can cause the number of visual features to no longer match the input sequence and directly lead to training failure. Based on these findings, we freeze the vision encoder and connector in the main experiments and filter leaked image special tokens before reward recomputation; more detailed layer-wise gradients, image-token failure cases, and stability analysis are provided in Sec. B.

#### 4.7. Can V-Triune Extend to New Task Domains?

Finally, we test whether V-Triune can absorb a new task domain beyond the original training mix, and whether the benefit of unified training remains after the new domain is added. As a concrete case, we add about 3K GUI grounding samples from ShowUI (Lin et al., 2025), a domain not covered in the original dataset. We treat GUI grounding as a new domain to test the scalability of our framework, which can be added by reusing the existing localization-style verifier regime.

As shown in Tab. 4, adding ShowUI data substantially improves ScreenSpot-Pro, with the unified model rising from 23.91 to 31.68. Notably, after adding the same GUI data, Unified + GUI still outperforms Perception + GUI by +1.83 points on ScreenSpot-Pro. Reasoning-heavy data therefore does not block learning in the new GUI domain, and unified training continues to show positive transfer once the model has basic in-domain visual support.

These results suggest that V-Triune can extend beyond the original training domains while preserving the benefit of unified training. In the ShowUI case, this extension is straightforward because the GUI task can reuse the existing localization-style verifier regime.

## 5. Discussion & Future Work

We presented V-Triune, a training methodology for unified multimodal RL over reasoning-heavy and perception-heavy VLM tasks. By organizing training around reward routing, verifier-level outcome verification, and source-level diagnostics, together with Dynamic IoU for localization-centric tasks, V-Triune addresses rigid reward interfaces, localization reward ambiguity versus sparsity, and lack

of observability in mixed-task RL. Under matched budgets, unified training matches or outperforms specialist mixtures, and the final Orsta models improve over their backbones on MEGA-Bench and a broad set of downstream benchmarks, while the same recipe also extends to a new GUI domain.

Two future directions seem especially important. One is to extend unified RL from static benchmark settings to multimodal agentic tool-use tasks. The other is to generalize this training recipe beyond vision, toward joint RL training across speech, video, and text.

## Contributions

#### Core Contributions

Yan Ma∗1,4, Linge Du∗1,3, Xuyang Shen∗†1, Junjie Yan1 Contributions

Shaoxiang Chen1, Pengfei Li1, Qibing Ren1,2 Advisor

Junjie Yan1, Pengfei Liu2,4, Yuchao Dai3, Lizhuang Ma2 Affiliation

- 1 MiniMax
- 2 Shanghai Jiao Tong University
- 3 Northwestern Polytechnical University
- 4 Generative Artificial Intelligence Lab (GAIR)

∗ Equal Contribution; † Project Lead; Corresponding Author

## References

Apache Software Foundation. Apache parquet documentation. https://parquet.apache.org/ docs/, 2025. Accessed: 2025-05-20.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Ekaterina Borisova and Georg Rehm. Scivqa: Scientific visual question answering. SDProc 2025,

### 2025. URL https://sdproc.org/2025/scivqa.html.

Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. Mega-bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024.

Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. arXiv preprint arXiv:2403.13315, 2024.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024.

Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025.

Ling Fu, Biao Yang, Zhebin Kuang, Jiajun Song, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, Mingxin Huang, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.

Deepanway Ghosal, Vernon Toh Yan Han, Yew Ken Chia, and Soujanya Poria. Are language models puzzle prodigies? algorithmic puzzles unveil serious challenges in multimodal reasoning. arXiv preprint arXiv:2403.03864, 2024.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.

Yoonsik Kim, Moonbin Yim, and Ka Yeon Song. Tablevqa-bench: A visual question answering benchmark on multiple table domains. arXiv preprint arXiv:2404.19205, 2024.

Team Kimi, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, Guokun Lai, Han Zhu, Hao Ding, Hao

Hu, Hao Yang, Hao Zhang, Haoning Wu, Haotian Yao, Haoyu Lu, Heng Wang, Hongcheng Gao, Huabin Zheng, Jiaming Li, Jianlin Su, Jianzhou Wang, Jiaqi Deng, Jiezhong Qiu, Jin Xie, Jinhong Wang, Jingyuan Liu, Junjie Yan, Kun Ouyang, Liang Chen, Lin Sui, Longhui Yu, Mengfan Dong, Mengnan Dong, Nuo Xu, Pengyu Cheng, Qizheng Gu, Runjie Zhou, Shaowei Liu, Sihan Cao, Tao Yu, Tianhui Song, Tongtong Bai, Wei Song, Weiran He, Weixiao Huang, Weixin Xu, Xiaokun Yuan, Xingcheng Yao, Xingzhe Wu, Xinxing Zu, Xinyu Zhou, Xinyuan Wang, Y. Charles, Yan Zhong, Yang Li, Yangyang Hu, Yanru Chen, Yejie Wang, Yibo Liu, Yibo Miao, Yidao Qin, Yimin Chen, Yiping Bao, Yiqin Wang, Yongsheng Kang, Yuanxin Liu, Yulun Du, Yuxin Wu, Yuzhi Wang, Yuzi Yan, Zaida Zhou, Zhaowei Li, Zhejun Jiang, Zheng Zhang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Zijia Zhao, and Ziwei Chen. Kimi-VL technical report, 2025. URL https://arxiv.org/abs/2504.07491.

Team Kimi, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Hynek Kydlíček. Math-verify: A library for rule-based verification of mathematical answers, 2025.

### URL https://github.com/huggingface/Math-Verify.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981, 2025.

Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Stan Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19498–19508, 2025.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, Bohao Peng, Shu Liu, Bei Yu, and Jiaya Jia. Visionreasoner: Unified visual perception and reasoning via reinforcement learning. arXiv preprint arXiv:2505.12081, 2025b.

Zhiyuan Liu, Yuting Zhang, Feng Liu, Changwang Zhang, Ying Sun, and Jun Wang. Othink-mr1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning. arXiv preprint arXiv:2503.16081, 2025c.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025d.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. CoRR, 2023.

Xinyu Ma, Ziyang Ding, Zhicong Luo, Chi Chen, Zonghao Guo, Derek F Wong, Xiaoyi Feng, and Maosong Sun. Deepperception: Advancing r1-like cognitive visual perception in mllms for knowledge-intensive visual grounding. arXiv preprint arXiv:2503.12797, 2025a.

Yan Ma, Steffi Chern, Xuyang Shen, Yiran Zhong, and Pengfei Liu. Rethinking rl scaling for vision language models: A transparent, from-scratch framework and comprehensive evaluation scheme. arXiv preprint arXiv:2504.02587, 2025b.

Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, et al. Chartqapro: A more diverse and challenging benchmark for chart question answering. arXiv preprint arXiv:2504.05506, 2025.

Team Meituan LongCat, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, et al. Longcat-flash technical report. arXiv preprint arXiv:2509.01322, 2025.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. arXiv preprint arXiv:2504.10342, 2025.

Kai Sun, Yushi Bai, Ji Qi, Lei Hou, and Juanzi Li. Mm-math: Advancing multimodal math evaluation with process evaluation and fine-grained classification. arXiv preprint arXiv:2404.05091, 2024.

Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint

- arXiv:2504.08837, 2025a.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint

- arXiv:2504.08837, 2025b.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirtyeight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https://openreview.net/forum?id=QWTCcxMpPA.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. arXiv preprint, 2024b. URL https://arxiv.org/abs/2408.15556.

Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10126–10135, 2020.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024c.

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Peng Ye, Min Dou, Botian Shi, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. arXiv preprint arXiv:2402.12185, 2024.

Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.

Chi Xie, Zhao Zhang, Yixuan Wu, Feng Zhu, Rui Zhao, and Shuang Liang. Described object detection: Liberating object detection with flexible expressions. Advances in Neural Information Processing Systems, 36:79095–79107, 2023.

Shilin Xu, Yanwei Li, Rui Yang, Tao Zhang, Yueyi Sun, Wei Chow, Linfeng Li, Hang Song, Qi Xu, Yunhai Tong, et al. Mixed-r1: Unified reward perspective for reasoning capability in multimodal large language models. arXiv preprint arXiv:2505.24164, 2025.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

Yiyang Yao, Peng Liu, Tiancheng Zhao, Qianqian Zhang, Jiajia Liao, Chunxin Fang, Kyusong Lee, and Qing Wang. How to evaluate the generalization of detection? a benchmark for comprehensive open-vocabulary detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 6630–6638, 2024.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025.

Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, et al. Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327, 2025.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

Yufei Zhan, Ziheng Wu, Yousong Zhu, Rongkun Xue, Ruipu Luo, Zhenghao Chen, Can Zhang, Yifan Li, Zhentao He, Zheming Yang, et al. Gthinker: Towards general multimodal reasoning via cue-guided rethinking. arXiv preprint arXiv:2506.01078, 2025.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024. URL https://arxiv.org/abs/2407.12772.

## A. Data Curation

We select four reasoning tasks—Math, Puzzle, Science, and Chart—for their varied reasoning demands, and four perception tasks—Detection, Grounding, Counting, and OCR—for their broad coverage of visual understanding. Data sources for each task are listed below:

- • For the Math task, mm_math (Sun et al., 2024), geometry3k (Lu et al., 2021), and mmk12 (Meng et al., 2025) are chosen.
- • For the Puzzle task, PuzzleVQA(Chia et al., 2024) and AlgoPuzzleVQA(Ghosal et al., 2024) are merged due to their shared origin, and VisualPuzzles (Song et al., 2025) is additionally included.
- • For the Science task, ScienceQA (Lu et al., 2022), SciVQA (Borisova and Rehm, 2025), and the “Broader STEM Topics” and “(GradeSchool) Science” categories from ViRL39K (Wang et al., 2025b) are used.
- • For the Chart task, ChartQAPro(Masry et al.,2025), ChartX(Xia et al.,2024), Table-VQA(Kim et al., 2024), and the Tables/Diagrams/Charts categories from ViRL39K (Wang et al., 2025b) are used.
- • For the Detection task, V3Det (Xie et al., 2023) and Object365 (Shao et al., 2019) are chosen.
- • For the Grounding task, 𝐷3 (Xie et al., 2023) is used.
- • For the Counting task, CLEVR (Johnson et al., 2017; Tan et al., 2025) is used.
- • For the OCR task, English OCR questions are extracted from LLaVA-OV Data (Li et al., 2024) and EST-VQA (Wang et al., 2020).

To reduce noise, we apply a two-stage data filtering process (Figure Fig. 3): (1) rule-based filtering and (2) difficulty-based filtering. This yields 47.7K high-quality samples across 18 datasets and 8 tasks. To mitigate dataset bias, puzzle data is duplicated to ensure sufficient coverage. The final corpus includes approximately 20.6K perception and 27.1K reasoning samples, primarily consisting of single-image, single-turn conversations.

Rule-based Filter

###### Difficulty Filter Data Curation

Visual Reasoning Data

[Figure 12]

Multi-Choice and True/False Filter Symbol-containing Answer Filter Overlong and Complex Answer Filter Overlong Query Filter (Image + Text) Bounding Box Size and Number Filter Category Balance Filter Verifiable Answer Filter

Reasoning

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Visual Perception Data

[Figure 19]

[Figure 20]

[Figure 21]

Perception

[Figure 22]

[Figure 23]

pass@16IoUReward

- Figure 3 | Data Curation Process. First, visual reasoning and visual perception data pass through a rule-based filter, which removes samples that do not meet preset criteria. Subsequently, the data enters a difficulty filter, which removes samples that are too easy or too hard based on model performance, ultimately producing the Curated Dataset.

- A.0.0.1 First Stage: Rule-based Filter For four visual reasoning tasks, the following filters are applied:

- • Multiple-choice and true/false questions that are prone to hacking are discarded. (Kimi et al., 2025)

- • Answers containing symbols such as “=”, “[”, “]”, “(”, “)”, and “;” are removed, as the absence of these symbols may cause answer mismatches even if the numeric values are correct.
- • Answers longer than 20 characters are discarded to avoid overly complex answers.

The filtering process for visual perception tasks involves additional complexity:

- • Detection: Following Qwen2.5-VL (Bai et al., 2025), data is converted to relative coordinates. Single-box samples contain one box per category, while multi-box samples retain original annotations. Samples with over 10 boxes per category or boxes exceeding 50% of the image are removed. A 1:2 single-to-multi-box ratio is enforced, and category-level long tails are avoided.
- • Grounding: Data is processed into relative coordinates, and data with a box size greater than 50% of the image is discarded. Complex phrase labels are filtered out.
- • Counting: Data is balanced per category and only English data is retained.
- • OCR: Only English OCR data is retained, and final labels must be verifiable by math_verify (Kydlíček, 2025). Since no verifiable reward model (RM) is designed, the OCR task data must pass this validation.

#### A.0.0.2 Second Stage: Difficulty-based Filter

To remove low-value samples, easy questions already solvable by the base model are filtered out.

For reasoning tasks, we use Qwen2.5-VL-32B-0321 to compute pass@8, retaining only samples with 0 < pass@8 < 100%. For perception tasks, specifically detection and grounding, pass@16 is computed using Qwen2.5-VL-7B with a 0.5 IoU threshold, and samples with cumulative IoU rewards between 2 and 10 are selected. This split keeps reasoning filtering based on a stronger reasoningoriented model, while perception filtering follows the reward setting used for localization-style supervision.

All curated data is stored in Parquet format (Apache Software Foundation, 2025) and uniformly mixed for training without online filtering or curriculum scheduling.

- Table 5 | Data source composition and curation. The curated corpus contains 27,133 reasoning samples (Math, Puzzle, Science, Chart; 56.8%) and 20,633 perception samples (Detection, Grounding, Counting, OCR; 43.2%), for a total of 47,766 examples (47.7K).

Count (Proportion)

Task

Data source name After curation Original count Notes

mm math 3,539 5,901 geometry3k 2,539 3,002 mmk12 5,732 15,616

Math

11,810 (24.72%)

PuzzleVQA + AlgoPuzzleVQA

Puzzle data are duplicated because the original puzzle data size is relatively small.

2,648 × 2 3,800 × 2

Puzzle

5,980 (12.52%)

Puzzle data are duplicated because the original puzzle data size is relatively small.

VisualPuzzles 342 × 2 1,168 × 2

ScienceQA 536 4,114 SciVQA 1,264 15,120 ViRL39K (“STEM” & “Science”)

Science

4,339 (9.08%)

2,539 4,431

ChartQAPro 498 1,948 ChartX 2,353 4,848 Table-VQA-Bench 496 1,500 ViRL39K (Tables / Diagrams / Charts)

Chart

5,004 (10.48%)

1,657 6,189

We randomly sample a 15k subset from 183,354 images; after filtering we obtain 6,287 samples and then randomly select 4k.

V3Det 4,000 15,000

Detection 8,000 (16.75%)

We randomly sample a 15k subset from 1.74M images; after filtering we obtain 8,889 samples and then randomly select 4k.

Object365 4,000 15,000

4,870 (10.20%)

D3 4,870 20,278

Grounding

1,725 (3.61%)

CLEVR 1,725 4,000 Wefromsamplethefulla4k35ksubset

Count

dataset.

We sample 8k images from the 56,613 images in the ocr_vqa category of LLaVA-OneVision-MidData.

LLaVA-OneVision (OCR-en)

3,092 8,000

OCR

6,038 (12.64%)

We sample 7k images from the 17,047 images in the EST-VQA training set.

EST-VQA 2,946 8,000

## B. Insights from Source-Level Monitoring

Source-level monitoring was essential for turning unified RL into a stable training pipeline. It exposed several concrete issues that were difficult to see from aggregate metrics alone, especially vision-encoder instability and leaked image special tokens. This section summarizes the corresponding adjustments used in our final recipe.

#### B.1. Stabilizing Training by Freezing the Vision Encoder

COCO Testset (Multi-ojbects) Performance

Sum of Gradient Norms

Layer Gradient Norm in ViT+LLM Training

- 0

- 1

- 2

- 3

- 4

- 5

0.56

Train ViT+LLM

Train ViT+LLM

ViT Grad (Avg. over Steps)

0.200

Train ViT

Train ViT

LLM Grad (Avg. over Steps)

Train LLM

Train LLM

0.175

0.55

| | |
|---|---|

0.150

| | |
|---|---|
| | |

IoUScore

0.54

0.125

Value

Value

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |
| | |

0.53

0.100

0.075

0.52

| |
|---|

| |
|---|

| |
|---|

0.050

0.51

0.025

0 50 100

0 50 100

30 20 10 0

Step (a)

Step (b)

Layer Num (c)

- Figure 4 | Analysis of vision-encoder training instability. (a) COCO performance under different training schemes. (b) Sum of gradient norms under different training schemes. (c) Layer-wise gradient norms of the vision encoder and LLM during full-parameter training. Updating the vision encoder leads to a performance drop and much less stable gradients, while the LLM remains comparatively stable.

In initial experiments, we performed full-parameter training by jointly optimizing the vision encoder and LLM. However, detection performance consistently collapsed after several dozen steps, regardless of hyperparameter settings. Log analysis revealed unusually large and spiking gradient norms, suggesting instability originating from the vision encoder. To verify this, we compared three training configurations: (1) LLM-only, (2) vision encoder-only, and (3) full-parameter training, all under identical RL settings on Orsta-7B with mixed-task data. We monitored COCO performance, total gradient norm, and layer-wise gradient trends during full-parameter training.

As shown in Fig. 4a, joint training leads to a performance drop, whereas LLM-only training maintains stable gains. Vision encoder-only training yields minimal improvement, indicating that the main RL gains do not come from updating the vision encoder. Fig. 4b shows that training the vision encoder produces much larger gradient norms than LLM-only training.

Layer-wise analysis in Fig. 4c confirms this pattern: LLM gradients remain relatively stable across layers, while vision-encoder gradients amplify during backpropagation. This gradient explosion destabilizes training and undermines visual performance. We therefore freeze the vision encoder and connector in the main experiments.

The precise cause of this instability remains open. For the present work, the practical conclusion is that continuing to update the vision encoder is not beneficial under our unified RL setup, whereas freezing the vision encoder and connector gives substantially more stable training.

#### B.2. Mitigating Leaked Image Special Tokens

Sample

[Figure 24]

#### Figure 5 | An Example of Leaked Image Tokens in VLM Outputs.

To enable accurate advantage estimation, logits for both the query and the generated response are recomputed, as those returned by the inference engine may be imprecise. During the forward pass, image placeholders (highlighted in the red box in Fig. 5, appearing before the “vision_end” token) are replaced with visual features extracted by the vision encoder and connector. However, the model may mistakenly generate special tokens (highlighted in the blue box in Fig. 5), such as image or video placeholders, that lack corresponding features. These leaked tokens cause the number of visual features to no longer match the input sequence and can directly lead to training failure. We therefore filter all such special tokens from the rollout sequence before reward recomputation.

## C. Training Dynamics Analysis

Math(MMK12)

Puzzle(PuzzleVQA)

Detection(V3Det)

OCR(estvqa)

400

Response Length

Response Length

Response Length

Response Length

320

700

850

375

300

650

350

280

800

325

600

260

750

300

240

550

0 50 100

0 50 100

0 50 100

0 50 100

Math(MMK12)

Puzzle(PuzzleVQA)

Detection(V3Det)

OCR(estvqa)

0.10

0.10

0.30

Reflection Ratio

Reflection Ratio

Reflection Ratio

Reflection Ratio

0.12

0.08

0.08

0.25

0.10

0.06

0.06

0.20

0.04

0.04

0.08

0.15

0.02

0.02

0.06

0.10

0.00

0.00

0 50 100

0 50 100

0 50 100

0 50 100

Math(MMK12)

Puzzle(PuzzleVQA)

Detection(V3Det)

OCR(estvqa)

0.7

1.0

Correct Ratio in Reflection Responses

Correct Ratio in Reflection Responses

Correct Ratio in Reflection Responses

Correct Ratio in Reflection Responses

0.7

0.04

0.8

0.6

0.6

0.02

0.6

0.5

0.00

0.5

0.4

0.02

0.4

0.4

0.2

0.04

0.3

0.3

0.0

0 50 100

0 50 100

0 50 100

0 50 100

Step

Step

Step

Step

- Figure 6 | Training dynamics of response length (top row), reflection ratio (middle row), and correct ratio in reflection responses (bottom row) during training steps for Math (MMK12), Puzzle (PuzzleVQA), Detection (V3Det), and OCR (estvqa) tasks using the Orsta-32B-0321 off-policy setting. Each column corresponds to a different task, and each row represents a distinct metric.

This appendix provides supplementary source-level dynamics for four representative tasks: Math (MMK12), Puzzle (PuzzleVQA), Detection (V3Det), and OCR (estvqa), all drawn from Orsta-32B-0321 off-policy training logs. We report three metrics defined in Sec. 3.3: response length, reflection ratio, and correctness of reflection responses. A more detailed explanation of the reflection-related metrics is provided in Sec. I.

As shown in Fig. 6, response behavior varies substantially across tasks. Reasoning-oriented tasks such as Math and Puzzle exhibit increasing response length and reflection usage over training, whereas Detection remains shorter and shows near-zero reflection. OCR displays a different trajectory from Detection, underscoring that perception tasks are not behaviorally uniform.

The bottom row further shows that reflection quality also differs across tasks. Math, Puzzle and OCR exhibit improving correctness in reflection responses, and Detection stays near zero throughout. We include these plots as supplementary evidence for the source-level behavior divergence discussed in the main text.

## D. Benchmark Details

We evaluate general VLM capability with MEGA-Bench (Chen et al., 2024) and report its scores using the official evaluation code. For the 10-benchmark suite used in the fixed-budget comparison and the strong-baseline comparison, we evaluate MMMU (Yue et al., 2024), MathVista (Lu et al., 2023), MathVision (Wang et al., 2024a), MME-Reasoning (Yuan et al., 2025), CharXiv (RQ) (Wang et al., 2024c), HrBench4K (Wang et al., 2024b), VStar (Wu and Xie, 2023), COCO (Lin et al., 2014), OCRBenchV2 (Fu et al., 2024), and ScreenSpot-Pro (Li et al., 2025). Among them, OCRBenchV2 is evaluated with Lmms-eval (Zhang et al., 2024), and the other eight benchmarks except COCO are evaluated with VLMEvalKit (Duan et al., 2024). For COCO, we use the official cocoapi; the detailed evaluation procedure is given in Sec. E. All bounding boxes and keypoints are represented using coordinate values relative to the original input image dimensions.

## E. Evaluation on COCO

We conduct our evaluation on the COCO val-2017 dataset (Lin et al., 2014), which contains 4,952 images with 36,781 ground-truth bounding boxes. The dataset includes 593 images with a single object (593 boxes) and 4,359 images with multiple objects (36,188 boxes). For the experiment, we use the instruction shown in Fig. 7 to prompt the model to generate a list of all target detections for each of the 4,952 images. The model operates at a temperature of 0 and outputs all bboxes in the format: [{‘bbox_2d’: [x1,y1,x2,y2],‘label’: label_name} ...] at one time.

The model’s output boxes are parsed into the COCO format, and we use the official cocoapi to calculate the mean Average Precision (mAP). The mAP computation requires a confidence score for each prediction to rank them. We use the predicted box’s area relative to the total image area as a pseudo-confidence score. The score is calculated as follows:

score = (𝑥2 − 𝑥1) × (𝑦2 − 𝑦1) image_width × image_height

To validate the robustness of our evaluation, we also conducted an ablation study on the choice of the pseudo-confidence function. We implemented and compared several alternative heuristics, including methods based on object position (center_bias) and shape (aspect_ratio), alongside fixed and random baselines. As shown in Tab. 6, the mAP scores are remarkably stable across all deterministic heuristics, with a total spread of around 0.5 mAP. This stability suggests that our evaluation results are not sensitive to the specific choice of the ranking method. Therefore, we adopt the simple and interpretable area_ratio method for all main experiments reported in this paper.

- Table 6 | Ablation study on pseudo-confidence scoring methods for Qwen2.5-VL-7B-Instruct on the full COCO val-2017 dataset.

Scoring Method mAP@50:95

area_ratio (our choice) 33.63 center_bias 33.60 aspect_ratio 33.08 fixed (1.0) 33.07

random (baseline) 33.05

## F. Additional Controls

#### F.1. Matched-Count Curated-vs.-Random Control

To isolate the effect of data quality from the rest of the training pipeline, we compare the main Orsta-7B model trained on the curated 47.7K corpus against a matched-count random control. The random control uses the exact same backbone, RL recipe, and training budget as Orsta-7B; the only change is that each data source is replaced by a stratified random subset from the corresponding raw pool, while preserving the same post-curation sample count. This keeps the task and source distribution fixed and varies only the data quality.

Table 7 | Matched-count curated-vs.-random control on the 10-benchmark suite. Curated-47.7K consistently improves over the random control on complex reasoning and fine-grained perception benchmarks.

Model MMMU MathVista MathVision MME-R Charxiv (RQ) HrBench4K VStar COCO (M) OCRBenchV2 ScreenSpot Pro

Orsta-7B-Random 56.33 72.20 31.57 28.62 46.40 74.38 83.25 40.38 55.83 23.85 Orsta-7B-Curated 57.10 72.50 31.73 31.14 48.40 77.25 81.68 41.41 56.05 23.91 Gain +0.77 +0.30 +0.16 +2.52 +2.00 +2.87 -1.57 +1.03 +0.22 +0.06

As shown in Tab. 7, the random control is already competitive, which indicates that the unified V-Triune pipeline remains effective even without curation. At the same time, the curated corpus yields additional gains on more demanding reasoning and fine-grained perception benchmarks, suggesting that reward-aware filtering improves data efficiency beyond simply increasing sample count.

#### F.2. Detection Fast Path and Latency Profiling

We also profile the end-to-end inference cost of the detection prompt used in the main experiments and compare it against a shorter direct-mode prompt that removes the explicit CoT trigger. Both measurements are conducted on COCO val-2017 with Orsta-7B, using a single H200 GPU and vLLM v0.11.0 under greedy decoding.

- Table 8 | Direct-mode fast path for detection on COCO val-2017. Removing the explicit CoT trigger shortens responses, improves throughput, and slightly improves mAP.

Mode Avg. Tokens FPS mAP@50:95

Standard CoT Prompt 208.0 25 33.63 Direct Prompt 93.5 30 34.40

The direct-mode result complements the source-level behavior analysis in the main text. For detection, the model does not require a long reasoning-style response at inference time: removing the explicit CoT trigger reduces output length, improves throughput, and slightly improves COCO performance.

## G. Query Example of Detection and Grounding

Query Example of Detection and Grounding

|Please detect all instances of the following category within the image: {LABEL}.<br><br>Let’s think step by step and output the final answer in <answer> and </answer> tags. For example: Your detailed reasoning process here. <answer> [{’bbox_2d’: [x1,y1,x2,y2],’label’: label_name}] </answer><br><br>|
|---|

- Figure 7 | Example query format for detection and grounding tasks. The query instructs VLMs to identify instances of a given object and format the output in a specific reasoning-answer format.

Data Format

|{<br><br>"data_source": Value(dtype="string"), "images": Sequence(feature=Image(mode=None, decode=True)), "prompt": [<br><br>{<br><br>"content": Value(dtype="string"), "role": Value(dtype="string")<br><br>}<br><br>], "ability": Value(dtype="string"), "reward_model": {<br><br>"answer": Value(dtype="string"), "ground_truth": Value(dtype="string"), "accuracy_ratio": Value(dtype="float32"), "format_ratio": Value(dtype="float32"), "verifier": Value(dtype="string"), "verifier_parm": Value(dtype="dict")<br><br>}, "extra_info": {<br><br>"id": Value(dtype="string"), "image_path": Value(dtype="string")<br><br>} }<br><br>|
|---|

- Figure 8 | Sample-level Data Scheme for Unified Training. This format, implemented using HuggingFace datasets, allows fine-grained control over reward computation by defining reward_model (including reward types, weights like accuracy/format_ratio) and verifier specifications at the individual sample level. This enables flexible and scalable handling of diverse multimodal tasks.

- H. Sample-level Data Scheme for Unified Training
- I. Detailed Explanation of Reflection Metrics

This appendix provides a detailed breakdown of the reflection metrics used in our source-level metric monitoring. These metrics are designed to quantitatively assess the model’s self-correction and reasoning processes.

#### I.0.0.1 Reflective Word Set

Following Ma et al. (2025b), we track a curated list of 15 English words and phrases that indicate a reflective or self-correcting thought process. A response is considered "reflective" if it contains one or more of the following terms:

- • re-check, re-evaluate, re-examine, re-think
- • recheck, reevaluate, reexamine, rethink
- • reevaluation
- • check again, think again, try again
- • verify, wait, yet

- I.0.0.2 Metric Definitions Based on this word set, we define two metrics:

- 1. Reflection Ratio (𝑅reflect): This metric measures the overall frequency of reflective responses. It is defined as the total number of responses containing at least one reflective word (𝑁reflect) divided by the total number of all responses (𝑁total).

𝑅reflect =

𝑁reflect 𝑁total

(4)

- 2. Correctness Rate within Reflection (𝐶reflect): This metric assesses the effectiveness of the model’s reflective reasoning. It is defined as the number of reflective responses that are also correct (𝑁correct_reflect) divided by the total number of reflective responses (𝑁reflect).

𝑁correct_reflect 𝑁reflect

𝐶reflect =

(5)

We emphasize that this keyword-based approach serves as a rough proxy for reflective behavior, not a precise measurement. It is intended for lightweight, online monitoring to gauge general trends in the model’s reasoning process, rather than for a formal or rigorous evaluation of its reflection capabilities.

## J. Adaptive IoU Threshold Scheduling

To compare our fixed three-stage Dynamic IoU schedule against an adaptive alternative, we implement a controller that automatically adjusts the IoU threshold according to the model’s current batch-level success rate. All runs in this section are trained on the detection and grounding subsets of our data, and we monitor IoU@50 on the OVDEval negation subset throughout training.

We define a discrete set of thresholds S = {0.5, 0.55, 0.6, . . . , 0.95, 0.99} and initialize training at 𝑇0 = 0.5. At each training step 𝑡, we compute the Batch Success Rate (BSR), defined as the proportion of predicted boxes in the current batch whose IoU exceeds the current threshold 𝑇𝑡. The threshold for the next step, 𝑇𝑡+1, is updated according to a target-success hyperparameter 𝜏:

 

next(𝑇𝑡, S) if BSR > 𝜏 prev(𝑇𝑡, S) if BSR < 𝜏 𝑇𝑡 otherwise

(6)

𝑇𝑡+1 =

 

We sweep 𝜏 ∈ {0.1, 0.3, 0.5, 0.7, 0.9} to cover controllers that are respectively more aggressive or more conservative in tightening the threshold. The resulting training dynamics are shown in Fig. 9.

#### J.0.0.1 Results.

At high target success (𝜏 = 0.9), the threshold remains relatively stable but tends to plateau around 0.85, never reaching the strict high-precision regime. At lower target success (𝜏 ≤ 0.7), the threshold rises to 0.99 too early, which introduces severe reward sparsity and unstable training dynamics. In other words, the adaptive controller does not remove the schedule-design problem, but shifts it to the choice of 𝜏.

#### J.0.0.2 Conclusion.

While the adaptive scheduler is flexible, it introduces an additional control parameter that is not straightforward to calibrate. In our setting, the fixed three-stage schedule remains the more stable and interpretable choice for driving training toward a high-precision regime.

OVDEval (Negation subset)

Success Rate

IOU Threshold

1.0

1.0

=0.1 =0.3 =0.5 =0.7 =0.9

0.58

0.56

0.9

0.8

0.54

IoU@50Values

IOUThreshold

0.8

0.6

SuccessRate

0.52

0.50

0.7

0.4

0.48

=0.1 =0.3 =0.5 =0.7 =0.9

=0.1 =0.3 =0.5 =0.7 =0.9

0.6

0.2

0.46

0.44

0.0

0.5

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Step (a)

Step (b)

Step (c)

- Figure 9 | Training dynamics of adaptive IoU scheduling. We vary the target success rate 𝜏 ∈ {0.1, . . . , 0.9}. (a) Validation performance (IoU@50) on OVDEval. (b) Batch Success Rate stability. (c) Evolution of the IoU threshold. The results highlight the trade-off between threshold stagnation at high 𝜏 and premature saturation at low 𝜏.

## K. MEGA-Bench Evaluation Curves

We provide the full MEGA-Bench evaluation curves for the intermediate checkpoints of the 7B, 32B-0321, and 32B-0326 training runs. The fourth panel further shows the task-level trajectory of Orsta-32B-0321 on MEGA-Bench.

###### Orsta-32B-0321 Task-level MEGA-Bench Performance

Orsta-7B Performance

Orsta-32B-0321 Performance

Orsta-32B-0326 Performance

26

35

| |
|---|

38.36

on-policy off-policy

on-policy off-policy

on-policy off-policy

| |
|---|

| |
|---|

Perc.

38.0

45.5

25.94

24

30

38.27

| |
|---|

Sci.

| | | | |
|---|---|---|---|
| | | | |

45.77

| |
|---|

| |
|---|

37.5

22

| |
|---|

Math

45.0

Info.Ex.

25

Core Know.

| |
|---|

| |
|---|

| | |
|---|---|

| |
|---|

37.0

20

18.90

Score

Score

Score

Score

45.21

| |
|---|

Metrics

| |
|---|

44.5

20

| |
|---|

| |
|---|

| | |
|---|---|
| | |

36.5

18

| |
|---|

Plan

| | |
|---|---|

Code

| |
|---|

| |
|---|

| | |
|---|---|
| | |
| | |

| |
|---|

| |
|---|

15

44.0

| |
|---|

| | |
|---|---|

| |
|---|

16

36.0

| |
|---|

| |
|---|

| |
|---|

10

| |
|---|

43.5

14

35.5

Metrics

Info.Ex.

Math

43.67

| |
|---|

Sci.

Code Perc.

Know.

35.07

| |
|---|

11.87

5

| |
|---|

Plan

Core

12

35.0

| |
|---|

0 50 100

0 25 50 75 100 125

0 50 100

0 50 100

Step (a)

Step (b)

Step (c)

Step (d)

- Figure 10 | MEGA-Bench evaluation curves across training. The first three panels compare on-policy and off-policy variants for the 7B, 32B-0321, and 32B-0326 backbones. Models are evaluated every 5 generation steps from 0 to 135. The fourth panel shows the task-level trajectory of Orsta-32B-0321 on MEGA-Bench.

- L. Qualitative Cases for Dynamic IoU We provide qualitative cases to illustrate the late-stage drift discussed in Sec. 4.5.

[Figure 25]

- Figure 11 | Qualitative cases from the fixed low-threshold ablation curve (IoU@50) shown in Fig. 2(a). Each row shows one COCO multi-object sample, where we select the largest ground-truth box in the image for visualization. Each column corresponds to an intermediate checkpoint along the IoU@50 training trajectory. In these cases, predictions become more accurate in the middle stage but later drift among multiple coarse boxes around the target, rather than continuing to sharpen around the ground-truth box. This pattern is consistent with the reward ambiguity induced by a loose threshold such as IoU@50, under which multiple coarse boxes can receive similarly high rewards.

These cases help explain the late-stage degradation of the IoU@50 curve in Fig. 2(a). Although predictions become more accurate in the middle stage, later checkpoints often drift among several coarse boxes around the target instead of continuing to sharpen around the ground-truth box. A loose threshold such as IoU@50 still provides non-zero reward through the IoU value itself, but it also creates a broad region in which multiple coarse predictions can receive similarly high rewards. Once training enters this regime, the marginal reward difference between these boxes becomes weak, so optimization no longer strongly favors continued refinement toward the ground-truth box. This qualitative pattern is consistent with the reward-ambiguity explanation used in the main text.

## M. OVDEval Curves for Dynamic IoU

We provide the OVDEval negation-subset curves referenced in Sec. 4.5. The comparison follows the same setup as the COCO ablations in the main text and includes fixed IoU@99 together with the three staged Dynamic IoU schedules.

OVDEval (Negation subset)

0.575

0.550

IoU@50Values

0.525

0.500

0.475

0.450

IoU@Dynamic(our, 0.85 0.95 0.99)

0.425

IoU@99

- IoU@Dynamic(var1, 0.7 0.8 0.85)

- IoU@Dynamic(var2, 0.9 0.95 0.99)

0.400

0 20 40 60 80

Step

- Figure 12 | Dynamic IoU ablations on the OVDEval negation subset (Yao et al., 2024). The main schedule (0.85 → 0.95 → 0.99) outperforms fixed IoU@99 as well as the looser and stricter staged variants, matching the same overall pattern observed on COCO multi-object.

