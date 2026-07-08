# arXiv:2606.04513v2[cs.AI]16Jun2026

## MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation

Deguo Xia∗

Zihan Li∗

Haochen Zhao∗

xdg23@mails.tsinghua.edu.cn Tsinghua University Baidu Beijing, China

mc45085@um.edu.mo University of Macau Macao, China

zhaohaochen@iie.ac.cn Institute of Information Engineering, Chinese Academy of Sciences Beijing, China

Dong Xie∗

xiedong04@baidu.com Baidu Beijing, China

Yuyao Kong

kongyuyao@iie.ac.cn Institute of Information Engineering, Chinese Academy of Sciences Beijing, China

Xiyan Liu

liuxiyan@baidu.com Baidu Beijing, China

Jizhou Huang†

huangjizhou01@baidu.com Baidu Beijing, China

Mengmeng Yang†

yangmm_qh@tsinghua.edu.cn School of Vehicle and Mobility State Key Laboratory of Intelligent Green Vehicle and Mobility Tsinghua University Beijing, China

Diange Yang

ydg@mail.tsinghua.edu.cn School of Vehicle and Mobility State Key Laboratory of Intelligent Green Vehicle and Mobility Tsinghua University Beijing, China

### Abstract

Lane-level maps are critical infrastructure for autonomous driving and lane-level navigation, yet constructing and maintaining standardized lane networks for hundreds of cities remains highly labor-intensive. Recent end-to-end vectorized mapping methods can predict lane geometry and topology directly from sensor data, but they typically treat mapping specifications and traffic regulations as implicit, dataset-dependent supervision. Moreover, in complex scenes (e.g., worn or missing markings and occlusions), correct lane configurations are often under-determined by visual evidence alone, making specification violations a major source of human post-editing. We propose MapAgent, an industrial-grade agentic architecture that augments a vectorization backbone for specification-compliant lane-map production. Rather than merely adding an agent loop to map prediction, MapAgent couples backbone perception with explicit specification verification, constraintaware reasoning, and deterministic map editing under a bounded, verification-driven Judge–Planner–Worker loop. A vision–language Judge diagnoses errors by jointly inspecting visual evidence and draft vectors, while a tool-calling Planner generates minimal corrective edits with post-edit re-validation. To remain scalable for city-scale production, MapAgent is selectively triggered only on tiles with low backbone confidence, adding modest overhead while

∗These authors contributed equally to this work. †Corresponding authors.

This work is licensed under a Creative Commons Attribution 4.0 International License. KDD ’26, Jeju Island, Republic of Korea

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2259-2/2026/08 https://doi.org/10.1145/3770855.3818443

preserving throughput. Experiments on real-world datasets show consistent gains over strong production baselines, especially in complex and long-tail scenarios. Additionally, MapAgent has been integrated into Baidu Maps, supporting lane-level map generation for over 360 cities nationwide and elevating the overall production automation to over 95%, demonstrating MapAgent’s practicality and effectiveness for large-scale lane-level map generation.

### CCS Concepts

• Applied computing → Transportation.

### Keywords

Map Generation; Map Agent; Vision-Language Models

ACM Reference Format:

Deguo Xia, Zihan Li, Haochen Zhao, Dong Xie, Yuyao Kong, Xiyan Liu, Jizhou Huang, Mengmeng Yang, and Diange Yang. 2026. MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26), August 09–13, 2026, Jeju Island, Republic of Korea. ACM, New York, NY, USA, 14 pages. https: //doi.org/10.1145/3770855.3818443

### 1 Introduction

Lane-level maps have become core infrastructure for autonomous driving, advanced driver assistance, and lane-level navigation. They provide centimeter-level priors on road geometry, lane topology, and traffic control, enabling robust planning and decision-making beyond the sensing horizon. However, constructing and maintaining lane-level maps for hundreds of cities at nationwide scale remains extremely expensive. Traditional pipelines rely heavily on trained experts to interpret mapping rules and standards and to

conduct labor-intensive annotation and updating, which limits nationwide coverage and update cycles [12, 22, 23]. Recent end-to-end HD map learning has pushed automation from research to industry: methods such as HDMapNet [12], VectorMapNet [15], MapTR [13] and MapTRv2 [14] convert multi-sensor inputs into bird’s-eye-view (BEV) features and directly decode vectorized polylines or topology, replacing much of the manual mapping pipeline while achieving strong performance. To meet nationwide requirements on scale, efficiency, and quality, in our prior work, we developed DuMapNet [22], an industrial vectorization system for city-scale lane-level map generation in Baidu Maps (deployed since 2023), and subsequently developed LDMapNet-U [23] for map updating conditioned on historical maps (deployed since April 2024); together, they support over 360 cities and substantially reduce production cost and update latency.

explicitly represent such knowledge nor offer a principled way to reason about per-scene ambiguities or to decide when and how to invoke geometric editing tools.

Motivated by this gap, we propose a shift from one-pass end-toend vectorization to an industrial-grade paradigm of agent-based refinement on top of a frozen backbone, where the backbone generates drafts and an agentic layer enforces specifications via verification and deterministic edits. In this setting, a BEV vectorization backbone remains essential for scalability and visual performance, but it is treated as a draft generator rather than the sole component responsible for satisfying cartographic and regulatory constraints. We therefore refine frozen backbone outputs with a controllable workflow that combines grounded diagnosis, specification-aware verification, and deterministic tool-based edits. To realize this, we introduce MapAgent, a refinement-on-top-of-backbone framework that selectively processes only hard tiles through a bounded iterative loop. A lightweight Quality Agent performs early acceptance using backbone confidence and cheap consistency checks; for the remaining tiles, MapAgent runs a Judge–Planner–Worker loop where a vision–language Judge produces structured error evidence by verifying geometry, topology, and specification compliance, a Planner converts evidence into a tool-grounded edit plan under capability constraints, and deterministic Workers execute edits with re-validation until success or budget exhaustion. By coupling end-to-end perception with explicit verification and rule-grounded editing under strict safety constraints (closed tools, bounded budget, best-state fallback), MapAgent enables scalable, production-quality lane-level maps aligned with domain standards.

- A. Baseline System

- B. Our MapAgent System

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Vectorized Map (manual labels)

Human Annotation

BEV Image

GeMap/DuMapNet Prediction Human Annotation

Multi-view Camera

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Refine

Reason

Judge Agent

###### BEV Image

Verify & Evaluate

Vectorized Map (Production)

Multi-view Camera

[Figure 17]

High Quality Verified Output Iterative Refinement Cycle

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Concretely, this paper makes the following key contributions to both the research and industrial communities:

[Figure 25]

[Figure 26]

[Figure 27]

Step

###### Vectorized Map (Draft)

Planner Agent

Worker Agent

[Figure 28]

- • Potential impact: We introduce MapAgent as an industrialgrade agentic refinement layer for city-scale lane-level map generation and updating. MapAgent has been integrated into Baidu Maps, supporting lane-level map generation and updating for over 360 cities nationwide, and elevating the overall production automation to over 95%.
- • Novelty: MapAgent introduces an industrial-grade agentic paradigm for lane-level mapping, designed for specificationcompliant production. By coupling a frozen BEV vectorization backbone with a bounded, verification-driven loop (Quality Agent + Judge–Planner–Worker), it enforces hard validity checks for geometric, topological, and specification constraints.
- • Technical quality: Extensive experiments on large-scale real-world datasets demonstrate consistent gains over strong baselines, especially in complex and long-tail scenarios, while substantially reducing manual post-editing effort. The successful production integration of MapAgent further validates its robustness and scalability for city-scale deployment.

Single-pass No Feedback

- Figure 1: Framework Comparison. Top: A one-pass pipeline vectorizes lanes from BEV images and relies on human postediting. Bottom: MapAgent performs agent-based refinement with a Judge–Planner–Worker loop to automatically verify errors, apply corrections, and produce a high-quality map.

Despite these advances, a crucial gap remains between endto-end vectorization and fully automated, specification-compliant lane-level map production. Existing systems mainly learn what is visible—lane boundaries, crosswalks, stop lines, and local topologyfrom supervised labels [12, 13, 15], whereas industrial maps must also satisfy cartographic standards and traffic regulations, requiring consistent organization of lane groups, geometry, attributes, and topology. In long-tail real-world scenes (e.g., degraded/missing markings, wide unstructured pavements, adverse lighting, and occlusions), the lane configuration is frequently under-determined by visual evidence alone and instead depends on specification- and rule-based priors, leading supervised models to exhibit geometric artifacts and semantic misclassifications that still require substantial human post-editing, even in city-scale deployments such as DuMapNet and LDMapNet-U. As a result, commercial pipelines still depend on expert editors who apply codified specifications via interactive tools to repair topology and ensure compliancecapabilities that current one-pass models lack, since they neither

### 2 MapAgent Framework

Compared to conventional one-pass pipelines (BEV observation → vector decoding → human annotation → final map), MapAgent reconceptualizes lane-level map generation as a controllable refinement process: the backbone output is treated as a mutable map state rather than a final result, as shown in Figure 1. The system

Core MapAgent System SolidArrow= MainFlow Dashed Arrow = Feedback/Fast Track Blue = Validated/Pass Gray = Agent Workflow

##### Input Stage

##### Output Stage

[Figure 29]

###### Quality Agent

[Figure 30]

###### Early Accept

Confidence-based Early Filtering

[Figure 31]

Judge Agent

###### Iteration Control

[Figure 32]

###### Need Refinement

[Figure 33]

Max Iterations=3

[Figure 34]

Constraint Verification

Iteration Status

⚠️ Category ⚠️Extra Lane ⚠Geometry ⚠Structural

✅ Correct

Correct

[Figure 35]

[Figure 36]

###### BEV Image

###### Validation Passed

Evidence Generation

Multi-view Camera BEV

Re-validation

Error Report

iter = 3

Suggestions

Worker Agent

Vectorized Map (Production)

###### Best State

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Category Worker

Delete Worker

Planner Agent

Selection High Quality

Type Correction

Removal

[Figure 41]

Verified Output

Constraint Reasoning Plan Generation (JSON) Task Dispatching

category_modification

delete_lane

###### Initial Prediction

Ordered Action Plan(1.2.3.4)

[Figure 42]

[Figure 43]

Smooth Worker Regenerate Worker

Updated State

Geometry Refine

Reconstruction

Backbone Output (Draft Map)

skeletonize_and_smooth

regenerate

Query & Access

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Knowledge & Capability Sources

Memory Rule Tool Model

[Figure 49]

Historical Patterns

Mapping Specifications

Editing Toolset

VLM Reasoning

- Figure 2: Overall MapAgent system. Given a BEV image and a draft vector map from a backbone, a Quality Agent first performs confidence-based early filtering. High-confidence tiles are directly accepted, while the rest enter a bounded Judge–Planner– Worker refinement loop. The Judge detects rule violations and produces structured reports, the Planner generates an ordered correction plan under constraints, and the Worker applies deterministic edits such as removal, type correction, smoothing, or regeneration. Each iteration is re-validated, and after at most three rounds, the system outputs the validated refined map or the best intermediate result.

verifies the draft map using a structured Judge, plans minimal corrective actions, and executes deterministic edits via tools, making the pipeline efficient, interpretable, and production-ready. In the following subsections, we first formalize MapAgent as a constrained iterative refinement process. We then introduce the overall system architecture and refinement workflow, followed by detailed descriptions of the Judge Agent for constraint verification, the Planner Agent for tool-based plan generation, and the deterministic Worker agents for map editing.

### 2.1 Problem Formulation

Following prior works on lane-level map generation and updating [22, 23], the general task is to convert a BEV observation into a standardized vectorized lane map. Concretely, given a BEV observation 𝐼 collected from vehicle-mounted sensors, a map generation or updating system aims to predict a structured vectorized map 𝑉, where each map element is represented in a unified vectorized form with its geometry and attributes. In this paper, we formulate MapAgent as a constrained map refinement problem built on top of a frozen BEV vectorization backbone. Given a BEV observation 𝐼 and an initial draft map 𝑉0 produced by the backbone, MapAgent iteratively edits 𝑉𝑡 into a specification-compliant map.

At refinement step𝑡, we treat𝑠𝑡 = (𝐼,𝑉𝑡) ∈ S as the environment state. Let L denote the set of low confidence lane instances selected by the Quality Agent for refinement.

The Judge J𝜙 produces a lane-wise structured diagnosis for each 𝑙 ∈ L,

Sjudge(𝑙) = J𝜙 𝐼,𝑉𝑡,𝑙 , (1) and we aggregate them into a step-level diagnosis

𝐸𝑡 = {Sjudge(𝑙) | 𝑙 ∈ L}. (2)

Conditioned on (𝑉𝑡,𝐸𝑡), the Planner is implemented as a rule-based module 𝑔(·) that generates a tool-grounded refinement plan:

P𝑡 = 𝑔(𝑉𝑡,𝐸𝑡), P𝑡 ∈ A∗. (3) Concretely, P𝑡 is an ordered sequence of actions

P𝑡 = (𝑎𝑡,1,𝑎𝑡,2, . . .,𝑎𝑡,𝐾𝑡 ), 𝑎𝑡,𝑘 ∈ A. (4) Each action follows a fixed schema (consistent with Worker tools), e.g., 𝑎𝑡,𝑘 = (tool, lane_id, params), and the plan is executed deterministically to update the map via

𝑉𝑡+1 = T (𝑉𝑡, P𝑡). (5)

All edits must satisfy an immutable feasibility gate Ω (geometric and topological validity, mapping specifications); edits that lead to Ω(𝑉𝑡+1) = 0 are rejected by design (see App. A). Overall, MapAgent is driven by a learned Judge J𝜙 that produces structured diagnoses 𝐸𝑡 (trained via SFT and RL; see Sec. 2.3), and a rule-based Planner 𝑔(·) that generates minimal valid plans under Ω using a closed action set defined by Worker tools.

###### Part 1: Training Pipeline (SFT → GRPO → Deployment)

[Figure 50]

Training Data Input Annotated Dataset

Training Output

Judge Training Module (VLM-based)

Core Innovation: Rule-Guided Training

[Figure 51]

Trained VLM Judge Capabilities: Priority Reasoning,

[Figure 52]

SFT Training Stage

Output: Base Model (Logic & Format learned ✓)

- •Pred Draft Map Image
- •GT Draft Map Image
- •Metadata (bbox, category)
- •Oracle Label (CoT, Error Type) Supervised Training Data
- •Priority Rules
- •Format Constraints
- •Evidence Policy
- •Tool Mapping Structured Knowledge

Annotated Data VLM Backbone Supervised Loss

Structured Output

[Figure 53]

[Figure 54]

GRPO Training Stage

Reward Formula R R=w1⋅Raccuracy +w2⋅Rrule +w3⋅Rexec

Performance Metrics

Sample Multiple Outputs (Group)

Grouprelative Advantage Estimation

Policy Update (GRPO

Rule Library

Rule-based Reward Computation

Base SFT Model

[Figure 55]

[Figure 56]

[Figure 57]

√ Raccuracy Error-Type Correctness

√ √

Type Acc: 86%+ Format Acc: 99%+

Rrule Rule-consistent Reasoning

Rexec Planner Executability

Loss) :

[Figure 58]

Hyper-params (Temp, Top-p)

Prompt Engineering (Few-shot, Templates)

Performance Check (Format, Acc)

Model Adaptation & Optimization

###### Part 2: Inference Stage (Priority-based Error Detection & Suggestion) Model Deployment: Training produces the model deployed for real-time inference.

[Figure 59]

[Figure 60]

Judge Inference (Priority Short-Circuit Reasoning)

Inference Output

Inference input

[Figure 61]

Error Report

[Figure 62]

[Figure 63]

Priority-based Short-Circuit Detector

[Figure 64]

Pred Draft Map Image

Structured Output Generator

[Figure 65]

- • Error Type
- • 4-Sentence Reason
- • Confidence
- • Summary

Extra Lane Error

[Figure 66]

Extra Lane Check "On real marking?"

Semi-transparent Mask over BEV Image

Real-time input

4-Sentence Reason

Error Report

NO NO YES YES

[Figure 67]

Extra Category Geometry Structure Correct

Category Check "Category match?"

[Figure 68]

Category Error

- S1 [Check 1]

- S2 [Check 2]

- S3 [Evidence]

- S4 [Conclusion]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Metadata

[Figure 73]

Geometry Check "Local noise?"

Geometry Error

[Figure 74]

- •pred_bbox
- •pred_category

Legend Main Flow Short-Circuit/Rule Flow Priority Skip Core Innovation

Structure Error

[Figure 75]

Structure Check "Major misalignment?"

[Figure 76]

√ Concise Evidence Summary

Correct "No Error Detected"

[Figure 77]

[Figure 78]

- Figure 3: Architecture of the VLM-based Judge Agent. The framework consists of two phases. Top: A rule-guided training pipeline that aligns the VLM via SFT followed by GRPO, using a composite reward to enforce rule compliance, reasoning consistency, and executability. The training leverages annotated data and structured rule libraries to learn priority reasoning and structured outputs. Bottom: At inference, the Judge takes a BEV image and a draft map, applies a priority-based short-circuit mechanism to check errors in order, and generates structured, evidence-backed reports. The output includes error type, concise reasoning, and confidence, which are used by the Planner for downstream refinement.

### 2.2 Overall Architecture

As illustrated in Figure 2, MapAgent is an agent-based refinement framework on top of a frozen backbone. It builds upon city-scale BEV vectorization backbones to retain strong perception performanceandscalability,whileintroducing an explicit agentic verificationand-editing loop that enforces lane-level specifications. The Quality Agent applies an early-acceptance fast track to tiles whose confidence score exceeds a fixed threshold 𝛿 = 0.7, while only tiles with confidence below 𝛿 are forwarded to a bounded refinement loop. This selective routing focuses computation on difficult regions while preserving the throughput of the backbone. In practice, most tiles are directly accepted, and only a minority of hard tiles require iterative refinement. The separation between fast-track acceptance and iterative correction also simplifies system integration, as the backbone remains unchanged.

For each forwarded tile, MapAgent runs an iterative Judge– Planner–Worker workflow with re-validation. We explicitly bound the outer refinement budget by a small constant, three rounds in our system, to ensure predictable latency and prevent over-editing. This bounded design provides stable runtime behavior and avoids cascading modifications. At each round, the system evaluates the current map state before applying the next plan, ensuring that only validated edits are retained. If the refinement converges earlier, the loop terminates immediately. When the budget is exhausted without further admissible improvement, the system outputs the best validated map state.

### 2.3 Judge Agent: Constraint Verification and Evidence Generation

As shown in Figure 3, the Judge Agent conducts geometry and topology verification, enforces specification compliance, and produces structured evidence for downstream refinement.

For each inspected lane𝑙, the Judge produces a structured, scalarsized diagnostic entry

𝑆judge(𝑙) = 𝑙, 𝑦ˆ𝑙, 𝑐ˆ𝑙,𝑒 , (6)

where 𝑦ˆ𝑙 ∈ C is the predicted error category drawn from the predefined error set C, 𝑐ˆ𝑙 ∈ [0, 1] is the associated confidence, and 𝑒 is a concise evidence summary. The taxonomy C is derived from largescale historical post-editing logs; newly observed anomalies are reviewed through a human-in-the-loop process, clustered, and used to update mapping specifications and subsequent Judge retraining data. This fixed schema 𝑆judge(𝑙) is the sole interface exposed to the Planner: internal artifacts (e.g., chain-of-thought traces) are retained for training and debugging only and are not directly consumed by downstream modules. Applying the Judge to all 𝑙 ∈ L yields lane-wise diagnoses {𝑆judge(𝑙)}, which are aggregated as 𝐸𝑡 and passed to the Planner at round 𝑡.

To create supervised training instances, we sample individual lanes from the backbone draft map and construct model inputs by overlaying the target lane 𝑙 onto the BEV observation 𝐼 as a semitransparent mask to focus attention on local evidence. Each instance is paired with a ground-truth error label 𝑦𝑙 ∈ C and, optionally,

an explanatory reference trace used for SFT. We denote a single supervised trajectory for lane 𝑙 as 𝜏𝑙 = (R1, . . ., R𝑀,𝑦𝑙), where R𝑚 denotes the 𝑚-th intermediate reasoning rationale, and the supervised corpus as D = {𝜏𝑙}. Note that the lane-masking strategy is a training-time input construction intended to focus attention on local evidence while preserving the global context through the unmasked portions of 𝐼. Thus, the Judge is lane-centered but not lane-isolated: it still conditions on the full BEV observation and current map context, allowing it to reason about consistency with adjacent lanes.

The JudgeAgentis implementedasan autoregressive multimodal policy 𝜋𝜙 that conditions on the BEV 𝐼, the current map context 𝑉𝑡, the selected lane 𝑙, and previously generated tokens to produce a short reasoning sequence followed by the final structured prediction 𝑆judge(𝑙).

We first perform supervised fine-tuning on D by maximizing the likelihood of the reference traces and final labels. To maintain stability and avoid catastrophic changes to the pretrained backbone, only lightweight adapter parameters are updated during SFT while the backbone weights remain frozen.

After SFT, we further apply GRPO to align the Judge with the downstream Planner–Worker objective. Given a lane-level input 𝑥 = (𝐼,𝑉𝑡,𝑙), the Judge policy generates an output 𝑦 consisting of a concise reasoning trace, a final error type, a confidence score, and an evidence summary. For each input 𝑥, GRPO samples a group of candidate outputs from the old policy,

{𝑦1, . . .,𝑦𝐺} ∼ 𝜋𝜙old(·|𝑥), (7) and computes a scalar reward 𝑅𝑖 for each candidate 𝑦𝑖. Unlike PPO, GRPO does not require a separate value model; instead, it estimates the advantage by normalizing rewards within the sampled group:

𝑅𝑖 − mean({𝑅𝑗}𝐺𝑗=1) std({𝑅𝑗}𝐺𝑗=1) + 𝜖

. (8)

𝐴𝑖 =

This group-relative normalization reduces memory cost during VLM fine-tuning and naturally fits our setting, where multiple candidate diagnoses for the same lane can be directly compared.

We optimize the clipped surrogate 𝐿𝑖(𝜙) = min (𝜌𝑖𝐴𝑖, 𝜌¯𝑖𝐴𝑖) ,

𝜋𝜙(𝑦𝑖|𝑥) 𝜋𝜙old(𝑦𝑖|𝑥)

(9)

𝜌𝑖 =

, 𝜌¯𝑖 = clip 𝜌𝑖, 1 − 𝜖clip, 1 + 𝜖clip .

The GRPO objective is then defined as

∑︁𝐺

1 𝐺

JGRPO(𝜙) = E𝑥,{𝑦𝑖}𝐺

𝐿𝑖(𝜙) − 𝛽𝐷KL 𝜋𝜙(·|𝑥)∥𝜋ref(·|𝑥) ,

𝑖=1

𝑖=1

(10)

where 𝜋ref is the frozen SFT policy. The KL penalty prevents the updated Judge from drifting away from the structured output behavior learned during SFT.

For each candidate output, the reward is computed as

𝑅𝑖 = 𝑅acc(𝑦𝑖) + 0.5𝑅rule(𝑦𝑖) + 𝑅exec(𝑦𝑖). (11) We compute the terms in an executability-first order. 𝑅exec acts as a hard gate: if the response cannot be parsed as JSON, the reward computation stops with 𝑅exec = −2.0; if the JSON misses required

fields or the evidence mentions lane identifiers outside the valid metadata set, we also assign a −2.0 executability penalty. A valid schema with no hallucinated lane identifier receives a small format reward 𝑅exec = 0.2. 𝑅acc is +1.0 when the predicted error_type matches the oracle label and −1.0 otherwise. 𝑅rule starts from zero and checks the reasoning trace: it subtracts 0.5 if the trace does not contain exactly four sentences, subtracts 0.5 if it mentions any lower-priority error type than the final prediction, and subtracts 0.3 if required higher-priority exclusions are missing. If none of these rule violations occurs,𝑅rule = 0.5. During GRPO training, we update only the LoRA adapter parameters and keep the pretrained VLM backbone frozen, preserving visual grounding while improving Judge reliability on hard cases under industrial map-production constraints.

We do not train the full Judge–Planner–Worker stack end-toend. This is intentional: the Planner encodes immutable mapping specifications and the Workers are deterministic executors, so exposing them to unconstrained policy optimization would weaken safety guarantees. The learned component is therefore restricted to the Judge, while Planner/Worker behavior remains auditable and version-controlled.

### 2.4 Planner Agent: Tool-Grounded Plan Generation

The Planner Agent is a decision-making module that converts lanelevel quality assessments produced by the Judge Agent into executable refinement plans under strict mapping constraints.

At refinement step 𝑡, the Planner receives the structured Judge outputs for candidate lane lines and consumes the pair (𝑉𝑡,𝐸𝑡) to generate the corrective plan P𝑡. When required for consistency, the Planner may additionally query the historical map state 𝑉𝑡−1 to reason about past modifications. This query acts as an antioscillation filter and is triggered only when the same lane is edited in consecutive iterations; in production logs, this occurs in less than 10% of refinement cases. No perceptual features beyond these structured inputs are available to the Planner.

The Planner outputs a structured refinement plan P𝑡 following a fixed schema. A plan consists of an ordered sequence of actions as defined in Eq. 4, where each action 𝑎𝑡,𝑘 is defined by a tool type, a target lane identifier, and tool-specific parameters. If all lanes are judged as correct, the Planner outputs an empty plan P𝑡 = ∅.

All planning decisions are constrained by externally defined and immutable mapping rules that encode lane-level specifications and safety constraints. These rules prohibit the creation of new lane lines, cross–lane-group modifications, and large-scale structural changes. Every generated plan must satisfy both schema validity and rule compliance before execution; plans that fail validation are rejected and treated as empty plans.

The Planner maintains a lightweight memory of past refinement actions and their outcomes, which is used to filter redundant or conflicting decisions. If no valid plan can be constructed under the imposed constraints and current map state, the Planner outputs an empty plan and terminates refinement. This conservative fallback mechanism prevents cascading errors and reduces the risk of degrading the map when corrective actions are uncertain or inadmissible.

### 2.5 Worker Agent: Deterministic Tool-Based Editing

MapAgent operates under a bounded refinement budget and a closed, specification-verified tool set. The Worker tools are deterministic lane-line editors that support only local, auditable modifications. They cannot create new lane instances or perform crosslane-group/non-local topology modifications. Deletion is allowed only for lanes diagnosed as redundant or spurious and is accepted only after passing the feasibility gate Ω. Therefore, the Workers are designed to avoid unsafe global topology changes while enabling local, auditable correction.

The Worker Agent is responsible for carrying out the refinement plans generated by the Planner Agent. MapAgent exposes a fixed and limited set of Worker tools, each designed for a specific type of lane-level refinement: (i) a Category Worker for semantic label correction, (ii) a Delete Worker for removing redundant or spurious lane lines, (iii) a Smooth Worker for local geometric smoothing, and (iv) a Regenerate Worker for local geometry repair using a trained model (SAM3) [5]. Among the available tools, the Regenerate Worker is the only one that leverages a learned model. It is designed for local geometry repair within the spatial support of an existing lane line, and its behavior is bounded by Planner-specified constraints, targeting local inconsistencies such as broken segments or misaligned geometry. Refinement actions are executed sequentially according to the order specified in the plan P𝑡.

3 Experiments

In this section, we describe the experimental setup, dataset, and evaluation protocol for MapAgent, followed by quantitative results and analysis. We also include ablations and case studies to better understand the refinement behavior under challenging scenes.

### 3.1 Experimental Settings

Dataset Construction. Following the DuLD dataset construction protocol in DuMapNet [22], we build large-scale lane-level vector maps from the Baidu Map Database by rendering a high-quality offline BEV image 𝐼 via multi-trajectory aggregation of camera– LiDAR fusion signals and assembling the vectorized ground truth 𝑉GT from lane-related instances and attributes within each region (grouped by lane-group IDs, transformed into the local BEV coordinate system, and filtered with basic consistency operations such as removing invalid geometries and normalizing attributes). On top of this base dataset, we further curate a challenging hard subset using a backbone-agnostic difficulty criterion (e.g., high junction complexity and occlusion proxies derived from map topology / scene metadata), and use the same subset consistently across all backbones to ensure fairness. The training split contains 3,712 BEV images with 59,434 ground-truth lane instances; DuMapNet and GeMap produce 59,928 and 50,263 predicted lane instances, respectively. The test split contains 656 images with 10,254 ground-truth lanes; DuMapNet and GeMap produce 10,340 and 8,734 predicted lane instances, respectively.

Metrics. We measure lane-level correction quality by matching each predicted lane to the most appropriate ground-truth lane and then computing metrics on the resulting assignments. Accuracy,

Precision, Recall, and F1 are defined on fully correct lanes: a prediction is counted as a true positive only if it can be matched to a ground-truth lane and is correct as a whole, i.e., it satisfies the matching criterion and has the correct lane category. Predictions without a valid match are treated as false positives, and unmatched ground-truth lanes are treated as false negatives. In contrast, BBox IoU and Mask IoU quantify geometric overlap, and Cls Acc measures category correctness, all computed only over matched lane pairs.

Backbones and Protocol. We evaluate MapAgent as a post-hoc refinement module on top of two representative BEV vectorization backbones, GeMap and DuMapNet. Unless otherwise noted, backbone predictors are frozen and MapAgent is applied without retraining. For the VLM-based Judge Agent, we compare different base models, including Qwen3-VL-Instruct (8B), Qwen3-VL-Thinking (8B), and InternVL-3.5-8B, under an identical prompting and inference protocol (same prompt template, decoding strategy, and refinement budget). All hyperparameters related to refinement (e.g., maximum number of tool calls and termination criteria) are fixed across experiments.

### 3.2 Implementation Details

Hardware and Software. All experiments were run on a single server equipped with 8× NVIDIA A800 80GB GPUs. The software stack is PyTorch 2.6.0 with CUDA 12.4.

VLM, SFT and GRPO. The Judge Agent module is based on the model Qwen3-VL-8B-Thinking (approximately 8B parameters).

The model is initialized from a publicly available pretrained checkpoint and further adapted using parameter-efficient finetuning. Specifically, we first apply supervised fine-tuning (SFT) with LoRA using a learning rate of 1e−4, batch size of 32, LoRA rank 8, and 2 epochs. On top of the SFT checkpoint, we perform a lightweight GRPO stage to better align Judge outputs with downstream map-refinement objectives. GRPO updates only the LoRA parameters with learning rate 1e−5, rollout batch size 16, GRPO clip 𝜖 = 0.2, KL coefficient 0.1, and 3 epochs per update.

Runtime Measurements. After 100 warm-up tiles, we measure runtime over 1,000 randomly sampled validation tiles (including data I/O). The full MapAgent pipeline achieves a mean latency of 420ms/tile, median 380ms, p95 920ms, and p99 1.6s. Module-level averages are 230ms per tile for the Judge Agent (p95 540ms) and 140ms/tile for the Worker (SAM + regenerate, p95 600ms). Peak GPU memory usage is approximately 19GB per A800. MapAgent is triggered on about 30% of tiles in the test set.

### 3.3 Main Results

Table 1 details the performance of VLM-based Judges on lanequality inspection. SFT yields 58.23% for InternVL-3.5-8B and 70.16% for Qwen3-VL-8B, while subsequent GRPO alignment yields further targeted gains. For instance, GRPO improves the overall accuracy of Qwen3-VL-8B-Thinking from 83.55% to 86.01% and improves most class-wise precision/recall values. However, the Structure Error category shows a slight drop, suggesting a trade-off between overall decision accuracy and this minority/harder error type.

#### Table 1: Impact of VLM Judges and Fine-Tuning on Lane Quality Inspection.

Precision / Recall (%)

Judge Model Accuracy (%)

No Error Extra Lane Line Category Error Geometry Error Structure Error

InternVL-3.5-8B (SFT) 58.23 65.00 / 82.80 80.00 / 49.38 66.67 / 54.05 57.14 / 40.82 31.87 / 55.43 Qwen3-VL-8B (SFT) 70.16 87.50 / 89.17 93.33 / 86.42 88.24 / 67.57 83.33 / 51.02 18.18 / 32.61 Qwen3-VL-8B-Thinking (SFT) 83.55 84.39 / 92.99 91.67 / 81.48 88.04 / 72.97 81.25 / 79.59 70.43 / 88.04 Qwen3-VL-8B-Thinking (GRPO) 86.01 92.31 / 94.90 96.15 / 85.80 93.33 / 81.08 87.10 / 82.65 66.67 / 82.61

#### Table 2: Compact ablation of MapAgent under unified evaluation metrics.

Variant Accuracy ↑ Precision ↑ Recall ↑ F1-score ↑ BBox IoU ↑ Mask IoU ↑ Cls Acc ↑ Base Predictor (w/o MapAgent) 52.5 71.7 66.5 68.9 70.4 35.0 90.0 w/o Reason (Judge predicts error type only) 58.4 77.1 70.9 73.7 71.2 35.5 94.8 Max Correction Rounds (𝑇)

- 𝑇=1 58.3 76.9 70.8 73.6 71.2 35.6 94.5
- 𝑇=2 60.3 78.7 72.2 75.2 71.7 35.7 97.5

- 𝑇=3 62.6 80.7 73.9 77.0 71.8 36.0 98.0

The comparison also shows that reasoning-oriented VLMs are more suitable for priority-based quality inspection. Qwen3-VL-8BThinking substantially outperforms the non-thinking Qwen3-VL-8B under SFT, indicating that explicit reasoning helps the Judge distinguish visually similar error types and follow the predefined shortcircuit order. After GRPO, the model further improves on no_error, extra_lane_line, category_error, and geometry_error, suggesting that reward-based alignment makes the Judge outputs more consistent with executable downstream refinement.

Table 3 evaluates MapAgent as a frozen post-hoc refinement layer for GeMap and DuMapNet. MapAgent consistently improves lane-level correctness, scaling with Judge capability. On GeMap, Qwen3-VL-Thinking lifts Accuracy from 52.8 to 61.3 and F1 from 69.1 to 76.0. DuMapNet sees even larger gains (Accuracy 52.2→63.9, F1 68.6→78.0). Stable improvements across various models suggest MapAgent benefits systematically from stronger visual–language judgments rather than specific architectures.

Notably, gains are concentrated in Precision, Recall, and classification correctness, while geometric metrics (e.g., IoU) remain stable. This aligns with MapAgent’s role as a specification-aware editor: it corrects spurious lanes, category mismatches, and match failures caused by local geometry/category errors rather than aggressively altering lane geometry. The pronounced improvements on DuMapNet further indicate the refinement loop is particularly effective at rectifying errors in initial draft maps.

### 3.4 Ablation Study

To better understand which parts of MapAgent drive the gains, we run compact ablations under the same unified evaluation protocol as Table 3. For readability, Table 2 reports the averaged results over GeMap and DuMapNet, and we only show the final map quality after post-hoc refinement. These ablations focus on two key factors:

whether the Judge provides structured evidence beyond error labels, and how much benefit can be obtained from iterative correction.

We first examine what happens when the Judge is reduced to a pure error-type classifier. When we remove the explicit reasoning/evidence generation and let the Judge output the error type only (w/o Reason), the system still improves noticeably over the frozen base predictors, moving Accuracy from 52.5 to 58.4 and F1 from 68.9 to 73.7, with Cls Acc rising from 90.0 to 94.8. However, the gap to the full MapAgent remains clear: with the complete Judge– Planner–Worker loop, F1 reaches 77.0 and Cls Acc reaches 98.0. This difference suggests that, beyond recognizing what is wrong, the Judge’s structured reasoning is important for producing actionable, localized evidence that the Planner can reliably translate into safe, tool-grounded edits. In other words, reasoning improves not only interpretability, but also the executability of subsequent correction.

We then vary the maximum correction budget𝑇 in the bounded retry loop. A single round already captures a large portion of the benefit (𝑇=1: 58.3 Accuracy and 73.6 F1), indicating many errors can be fixed with one pass of diagnosis and execution. Allowing a second iteration brings a further jump (𝑇=2: 60.3 Accuracy and 75.2 F1), and a third round continues to help, though with diminishing returns (𝑇=3: 62.6 Accuracy and 77.0 F1). This trend shows that iterative refinement is useful, but most easy-to-correct cases are resolved in the first few rounds. Across these settings, BBox/Mask IoU change only slightly (e.g., 70.4/35.0 at baseline to 71.8/36.0 at 𝑇=3), which aligns with MapAgent’s conservative design: most gains come from resolving discrete false positives/negatives and type mistakes rather than aggressively deforming lane geometry or altering topology.

### 3.5 Case Study

Figure 4 shows several examples from the hard subset. As scenes become more challenging—e.g., with faint, missing, or occluded

Table 3: Unified comparison of MapAgent as a post-hoc editing module across different map prediction backbones and VLM bases. MapAgent is applied to fixed backbone predictions without retraining. Accuracy, Precision, Recall, F1-score, bounding box IoU (BBox IoU), mask IoU, and classification accuracy (Cls Acc) are reported (%).

Map Backbone MapAgent Base Accuracy Precision Recall F1-score BBox IoU Mask IoU Cls Acc

Original Prediction 52.8 75.1 64.0 69.1 69.3 32.9 91.9 InternVL-3.5-8B 54.9 77.1 65.6 70.8 69.7 33.3 96.5 Qwen3-VL-Instruct (8B) 56.5 78.6 66.8 72.2 69.8 33.5 96.8 Qwen3-VL-Thinking (8B) 61.3 82.9 70.1 76.0 70.7 34.2 98.1

GeMap

Original Prediction 52.2 68.3 68.9 68.6 71.4 37.1 88.0 InternVL-3.5-8B 55.0 70.9 71.1 71.0 71.9 37.4 94.6 Qwen3-VL-Instruct (8B) 57.1 72.7 72.7 72.7 72.1 37.4 95.3 Qwen3-VL-Thinking (8B) 63.9 78.4 77.6 78.0 72.8 37.7 97.8

DuMapNet

lane markings—both GeMap and DuMapNet often produce messy predictions, including spurious lanes, fragmented segments, and inconsistent topology. This failure mode is not specific to a particular backbone, but reflects a general limitation of feed-forward map prediction under weak or ambiguous visual evidence.

MapAgent effectively refines these noisy predictions into cleaner and more structured maps. The corrected results contain fewer spurious segments and exhibit more consistent global topology. In practice, the refinement focuses on obvious structural issues—removing hallucinated lanes, suppressing fragments, and restoring coherent layouts—while keeping geometry conservative. This qualitative trend matches the quantitative gains in lane-level correctness and demonstrates the method’s scalability.

### 4 Discussion

MapAgent bridges the gap between BEV vectorization backbones and production-grade,specification-compliant lane maps. Integrated into Baidu Maps for city-scale lane-level map generation and updating, it supports 360+ cities and elevates automation to over 95%. Here, the automation rate is defined as the ratio of lane-level mileage completed fully automatically without human intervention to the total lane-level mileage over a fixed production period. Instead of one-shot prediction followed by heavy human post-editing, MapAgent adds a constrained Judge–Planner–Worker refinement layer: violations are diagnosed under factorized constraints, corrected via a closed set of deterministic tools, and re-validated within a bounded budget. By triggering refinement only on hard tiles, it preserves throughput while reducing manual correction workload with predictable latency.

Despite these advances, several challenges remain worth exploring to further enhance autonomy. First, extreme visual ambiguity can make lane addition and topology modification under-determined from visual evidence, requiring stronger priors and principled uncertainty handling to avoid unsafe edits. In the current production setting, MapAgent therefore prioritizes the primary manual post-editing categories in our pipeline that can be handled safely with deterministic, local tools while deferring lane addition and non-local topology modification to future work. Second, to meet large-scale engineering requirements, MapAgent currently refines outputs from an existing backbone; an important direction is a

unified agentic framework capable of autonomously scheduling and orchestrating different specialized perception backbones (e.g., specialized backbones for junctions, occlusions, or rare topologies) and integrating their complementary predictions into a single constrained generation-and-updating pipeline, thereby maximizing flexibility and throughput.

5 Related Work

- 5.1 Map Construction

High-definition map construction has evolved from multi-stage segmentation pipelines to end-to-end vectorized regression [12]. VectorMapNet [15] and MapTR [13] advanced direct polyline decoding with autoregressive and permutation-invariant transformers, respectively, while recent systems further improve geometric consistency and structured modeling, e.g., GeMap [29] and HiMap [30]. Beyond purely visual cues, a growing line of work leverages map priors to stabilize matching and extend range, including Neural Map Prior [24], P-MapNet [9], and PriorMapNet [20], which incorporate SD maps or historical/outdated priors as additional conditioning signals. More recently, SDTagNet [8] exploits text-annotated SD maps to enhance far-range online HD map construction. In parallel, mixture-of-experts and interaction-based designs (e.g., MapExpert [27] and InteractionMap [21]) improve long-tail element modeling via expert routing and structured temporal–spatial interactions. Concurrently, VLM-based efforts such as MapGPT [28] and MAPLM [4] suggest that multimodal reasoning may help interpret maps and traffic scenes. However, these methods largely remain monolithic predictors that optimize geometric/semantic metrics and treat cartographic standards and traffic regulations as implicit supervision, leaving specification violations to manual post-editing. MapAgent addresses this gap by serving as a specification-aware refinement layer on top of strong BEV backbones.

- 5.2 Agentic System

LLM/VLM-based agents have rapidly advanced in tool use and selfcorrection. ReAct [26] interleaves reasoning and acting for interactive problem solving, Toolformer [16] learns to decide when/how to call tools, and Reflexion [17] improves agents via feedback-driven memory. Recent work further strengthens intrinsic self-correction

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

(a) Origin (b) Ground Truth (c) GeMap (d) DuMapNet (e) Corrected (Ours)

- Figure 4: Case study of lane-level map refinement under challenging scenarios. We compare the original input, ground-truth annotations, predictions from GeMap and DuMapNet, and the refined results produced by our correction framework. Best viewed in color and zoomed in.

via multi-turn RL (e.g., SCoRe [11]). Complementary lines study modular tool routing and grounded execution (e.g., MRKL [10], SayCan [2], ReWOO [25], Voyager [19]). Meanwhile, general-purpose multimodal foundations (e.g., GPT-4 [1], Qwen-VL [3], InternVL [6], PaLM-E [7]) provide strong visual grounding and verification capabilities. While agentic workflows have been applied to driving planning (e.g., DriveLM [18]), they have not been adapted for the strict safety constraints of map production. MapAgent pioneers this paradigm in mapping by combining VLM-based specification verification with deterministic tool execution, ensuring industrial-grade compliance.

### 6 Conclusions

In this paper, we present MapAgent, an industrial-grade agentic framework that bridges end-to-end vectorization and specificationcompliant map production. Motivated by the limitations of purely data-driven backbones in complex scenes, we formulate map refinement as a bounded, verification-driven iterative refinement process and implement MapAgent as refinement-on-top-of-backbone. A lightweight Quality Agent selects hard tiles, and a bounded Judge– Planner–Worker loop refines them: a structured vision–language Judge provides grounded diagnosis, a constrained Planner generates tool-grounded edits, and deterministic Workers execute them safely

to enforce geometric/topological validity and traffic standards. Experiments on large-scale real-world datasets show consistent gains over strong production baselines, especially in long-tail scenarios. Furthermore, MapAgent has been successfully integrated into the production pipeline of Baidu Maps. Supporting lane-level map generation and updating for over 360 cities nationwide, it has elevated the overall production automation to over 95%, proving that agentbased refinement is a viable and efficient paradigm for large-scale autonomous driving infrastructure. These results highlight the effectiveness of structured agentic refinement for improving map quality under challenging conditions. We believe this paradigm provides a practical and scalable direction for future large-scale mapping systems.

### 7 Acknowledgments

This work was supported by Beijing Natural Science Foundation (L231008, L243008), National Natural Science Foundation of China (52472449, 52402499), Independent Research Project of the State Key Laboratory of Intelligent Green Vehicle and Mobility, Tsinghua University (No. ZZ-PY-20250408), the Tsinghua University-Toyota Joint Center, and Tsinghua University–SAIC GM Wuling Joint Research Center.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. 2022. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691 (2022).
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966 (2023).
- [4] Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M Rehg, et al. 2024. Maplm: A real-world largescale vision-language benchmark for map and traffic scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21819–21830.
- [5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025).
- [6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 24185–24198.
- [7] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. 2023. Palm-e: An embodied multimodal language model. (2023).
- [8] Fabian Immel, Jan-Hendrik Pauls, Richard Fehler, Frank Bieder, Jonas Merkert, and Christoph Stiller. 2025. SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction. In Advances in Neural Information Processing Systems, Vol. 38.
- [9] Zhou Jiang, Zhenxin Zhu, Pengfei Li, Huan-ang Gao, Tianyuan Yuan, Yongliang Shi, Hang Zhao, and Hao Zhao. 2024. P-mapnet: Far-seeing map generator enhanced by both sdmap and hdmap priors. IEEE Robotics and Automation Letters

(2024).

- [10] Ehud Karpas, Omri Abend, Yonatan Belinkov, Barak Lenz, Opher Lieber, Nir Ratner, Yoav Shoham, Hofit Bata, Yoav Levine, Kevin Leyton-Brown, et al. 2022. MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arXiv preprint arXiv:2205.00445 (2022).
- [11] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. 2024. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917 (2024).
- [12] Qi Li, Yue Wang, Yilun Wang, and Hang Zhao. 2022. Hdmapnet: An online hd map construction and evaluation framework. In 2022 International Conference on Robotics and Automation (ICRA). IEEE, 4628–4634.
- [13] Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, and Chang Huang. 2022. Maptr: Structured modeling and learning for online vectorized hd map construction. arXiv preprint arXiv:2208.14437 (2022).
- [14] Bencheng Liao, Shaoyu Chen, Yunchi Zhang, Bo Jiang, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. 2023. Maptrv2: An end-to-end framework for online vectorized hd map construction. arXiv preprint arXiv:2308.05736 (2023).

- [15] Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, and Hang Zhao. 2023. Vectormapnet: End-to-end vectorized hd map learning. InInternational Conference on Machine Learning. PMLR, 22352–22369.
- [16] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems 36 (2023), 68539–68551.
- [17] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems 36 (2023), 8634–8652.
- [18] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. 2024. Drivelm: Driving with graph visual question answering. InEuropean conference on computer vision. Springer, 256–274.
- [19] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023).
- [20] Rongxuan Wang, Xin Lu, Xiaoyang Liu, Xiaoyi Zou, Tongyi Cao, and Ying Li.

2024. Priormapnet: Enhancing online vectorized hd map construction with priors. arXiv preprint arXiv:2408.08802 (2024).

- [21] Kuang Wu, Chuan Yang, and Zhanbin Li. 2025. InteractionMap: Improving Online Vectorized HDMap Construction with Interaction. In Proceedings of the Computer Vision and Pattern Recognition Conference. 17176–17186.
- [22] Deguo Xia, Weiming Zhang, Xiyan Liu, Wei Zhang, Chenting Gong, Jizhou Huang, Mengmeng Yang, and Diange Yang. 2024. DuMapNet: An End-to-End Vectorization System for City-Scale Lane-Level Map Generation. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 6015–6024.
- [23] Deguo Xia, Weiming Zhang, Xiyan Liu, Wei Zhang, Chenting Gong, Xiao Tan, Jizhou Huang, Mengmeng Yang, and Diange Yang. 2025. LDMapNet-U: An Endto-End System for City-Scale Lane-Level Map Updating. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2693–2702.
- [24] Xuan Xiong, Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, and Hang Zhao. 2023. Neural map prior for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17535–17544.
- [25] Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, and Dongkuan Xu. 2023. Rewoo: Decoupling reasoning from observations for efficient augmented language models. arXiv preprint arXiv:2305.18323 (2023).
- [26] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations.
- [27] Dapeng Zhang, Dayu Chen, Peng Zhi, Yinda Chen, Zhenlong Yuan, Chenyang Li, Rui Zhou, Qingguo Zhou, et al. 2025. Mapexpert: Online hd map construction with simple and efficient sparse map element expert. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 14745–14753.
- [28] Yifan Zhang, Zhengting He, Jingxuan Li, Jianfeng Lin, Qingfeng Guan, and Wenhao Yu. 2024. MapGPT: an autonomous framework for mapping by integrating large language model and cartographic tools. Cartography and Geographic Information Science 51, 6 (2024), 717–743.
- [29] Zhixin Zhang, Yiyuan Zhang, Xiaohan Ding, Fusheng Jin, and Xiangyu Yue.

2023. Online Vectorized HD Map Construction using Geometry. arXiv preprint arXiv:2312.03341 (2023).

- [30] Yi Zhou, Hui Zhang, Jiaqian Yu, Yifan Yang, Sangil Jung, Seung-In Park, and ByungIn Yoo. 2024. Himap: Hybrid representation learning for end-to-end vectorized hd map construction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 15396–15406.

### A Feasibility Check and Specification Library

In MapAgent problem formulation, we implement feasibility as a boolean quality gate. Let Ω : V → {0, 1} denote a QC function induced by a versioned library of hard (boolean) specification pred-

icates R(𝑣) = {𝑟𝑘}𝑘𝐾=1, derived from industry standards and internal cartographic specifications. We define

Ω(𝑉) ≜ GeoValid(𝑉) ∧ TopoValid(𝑉) ∧ SpecValid(𝑉),

𝐾

SpecValid(𝑉) ≜

𝑟𝑘(𝑉),

𝑘=1

(12) where GeoValid and TopoValid are lightweight geometric / topological sanity checks (e.g., no self-intersection, bounded curvature/length, lane-group consistency), and each 𝑟𝑖(𝑉) ∈ {0, 1} encodes a non-negotiable cartographic / traffic constraint.

Given a deterministic tool transition 𝑉𝑡+1 = T (𝑉𝑡,𝑎𝑡), each action must satisfy ActionValid(𝑎𝑡,𝑉𝑡) (schema / parameter validity, bounded edit magnitude, lane-group scope), and an updated state is accepted only if it passes the QC gate, i.e., Ω(𝑉𝑡+1) = 1.

### B SAM3 Fine-tuning for Lane Detection

This part describes the fine-tuning strategy of SAM3 for lane detection. Figure B.1 illustrates the overall architecture and the progressive fine-tuning design, where different stages gradually relax the optimization constraints on the pretrained components. We provide detailed training configurations and the rationale behind each stage, followed by a quantitative comparison in Table B.1. The results demonstrate that progressively unfreezing the backbones leads to consistent improvements across detection, segmentation, and classification metrics. Related training configurations and examples have been released at: https://github.com/eadst/KDD-2026-MapAgent.

Naïve: All Unfrozen

Proposed: Progressive (Stage 1→2→3)

[Figure 99]

[Figure 100]

[Figure 101]

Unfreeze Vision

Full Joint Optimization

[Figure 102]

[Figure 103]

[Figure 104]

###### Frozen

[Figure 105]

[Figure 106]

Stage 3: Full Joint Optimization

###### Stage 2: Vision Unfreezing

###### Stage 1: Transformer Tuning

- (a) Vision Backbone
- (b) Language Backbone
- (c) Transformer & Heads

Vision Backbone

Vision Backbone

Vision Backbone

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Language Backbone

Language Backbone

Language Backbone

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Transformer & Heads

Transformer & Heads

Transformer & Heads

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Primary Target

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

= Frozen = Unfrozen = Primary Target

- Figure B.1: Architecture and fine-tuning strategy of SAM3 for lane detection.

### B.1 Training setup

All stages are initialized from the same pretrained SAM3 checkpoint (sam3.pt) and trained on the lane dataset specified by train.json and val.json. We enable segmentation supervision throughout (enable_segmentation=True) and load RLE masks during batching (with_seg_masks=True). Input images are resized to a square resolution of 1008 with a minimum scale of 480, followed by padding

to the same size and normalization with mean and std both set to (0.5, 0.5, 0.5). During training, we perturb input boxes with Gaussian noise (std 0.1, capped at 20 pixels) and filter samples with empty targets; we also cap the maximum number of lane instances per image to 30.

We optimize with AdamWandmixed precision enabled (bfloat16), using gradient clipping with max norm 1.0. The learning rate is scheduled by an inverse square root scheduler with 2000 warmup steps and timescale 10000. We use a base learning rate of 2 × 10−5 for transformer layers, and apply smaller learning rates for backbones when they are unfrozen: 6.25 × 10−6 for the vision backbone and 1.25 × 10−6 for the language backbone. Stage 1 updates only the transformer layers, Stage 2 additionally unfreezes the vision backbone with layer-wise decay 0.7, and Stage 3 further unfreezes the language backbone with the same conservative scaling. Weight decay is set to 0.05, while biases and LayerNorm parameters are excluded from decay. We train for 120 epochs with a fixed random seed of 123.

For matching and loss computation, we use a binary Hungarian matcher with focal loss (𝛼 = 0.3,𝛾 = 2.0), augmented by an auxiliary one-to-many branch with weight 2.0, top-𝑘 = 4, and threshold 0.4. The loss weights are set to 3.0 for Lbbox and 1.5 for Lgiou, 15.0 for classification and presence losses, and we strengthen mask supervision with Lmask weight 8.0 and Ldice weight 12.0. To better fit thin lane structures, mask loss is computed with point sampling (12544 points, oversample ratio 3, importance sampling ratio 0.75), focusing updates on uncertain boundary regions. All experiments are trained with distributed data parallel using NCCL on 8 GPUs per node, batch size 2 per GPU (global batch 16), and we keep all remaining settings fixed across stages to ensure fair comparison.

### B.2 Progressive Fine-tuning Strategies

To adapt SAM3 to the lane detection task, we investigate three fine-tuning strategies that differ in how much of the pretrained model is allowed to adapt. Rather than treating fine-tuning as a single configuration choice, we gradually relax the optimization constraints to balance representation stability and task-specific adaptability.

- Stage 1. We begin with a conservative setting in which both the vision and language backbones are kept fixed, and only the transformer layers are optimized. In this configuration, the model primarily leverages pretrained visual representations while learning to reorganize cross-modal interactions for lane detection. This setting emphasizes stability and provides a reliable reference, revealing the extent to which SAM3 can model thin and elongated lane structures without modifying its backbone features.
- Stage 2. We then allow limited adaptation of the vision backbone while keeping the language backbone frozen. Specifically, the vision backbone is trained with a smaller learning rate and layer-wise decay, such that deeper layers receive progressively weaker updates. This design acknowledges that lane detection introduces domainspecific geometric patterns, including strong perspective effects and long-range continuity, which benefit from moderate refinement of visual features without disrupting low-level representations.
- Stage 3. Finally, we consider a more flexible fine-tuning strategy in which all major components are updated with carefully scaled learning rates. The transformer remains the primary optimization

Table B.1: Lane detection performance of SAM3 under the three progressive fine-tuning stages (Stage 1–3). Stage 1 updates only the transformer layers, Stage 2 additionally adapts the vision backbone with conservative optimization, and Stage 3 enables carefully scaled updates for all major components.

Model Accuracy (%) Precision (%) Recall (%) F1-score (%) BBox IoU (%) Mask IoU (%) Cls Acc (%)

- SAM3 (Stage 1) 50.7 70.2 60.1 64.8 67.2 30.8 92.9
- SAM3 (Stage 2) 54.1 74.5 63.4 68.5 68.4 31.9 94.8
- SAM3 (Stage 3) 58.4 80.1 67.0 73.0 69.5 33.0 96.6

Pred Image GT Image

Overall, these strategies form a progressive fine-tuning spectrum that balances stability and adaptability. In practice, this design leads to stable optimization and provides insights into the contribution of different model components to lane detection performance.

[Figure 126]

[Figure 127]

### C Chain-of-Thought Data Generation with GPT-5.2

To supervise the Judge Agent with structured and priority-aware reasoning traces, we construct a chain-of-thought (CoT) dataset using GPT 5.2 as an offline data generator. Rather than inferring labels, the model is tasked with producing concise visual rationales that justify a fixed oracle outcome. This design ensures that the generated CoT reflects the Judge’s diagnostic logic without introducing label ambiguity or decision leakage.

Prompt:

Explain why the predicted lane line is a structure_error.

[Figure 128]

ChatGPT 5.2 Output:

The predicted lane line lies along a visible lightcolored road marking/edge feature in the highlighted region rather than being placed on plain unmarked asphalt.

[Check Existence]

### C.1 Problem Setup

That marking appears as a continuous boundary-like feature, which is consistent with a virtual boundary line rather than a dashed or double-line pattern.

[Check Category]

Each training instance corresponds to a single predicted lane line produced by the frozen backbone model. The input to GPT 5.2 includes: (i) a Pred image containing exactly one highlighted predicted lane line, (ii) a GT image used for supervision only, and (iii) structured metadata, including the predicted category, a tight bounding box around the predicted line, and an oracle annotation specifying correctness and error type.

However, the predicted line drifts off the center of the visible boundary and becomes wavy, and it also starts and ends partway along the visible feature instead of covering it continuously through the region.

[Check Geometry]

The path and coverage do not follow the same marking overall, therefore this is a structure_error.

Importantly,theoracleannotationis treated as immutable. GPT 5.2

[Check Structure]

is explicitly instructed not to question, revise, or infer the outcome, but only to generate a textual explanation that is consistent with the given oracle decision.

- Figure C.1: Example of priority-structured chain-of-thought generated by GPT 5.2 for Judge supervision. Given a predicted lane line (red overlay), the model produces a four-sentence explanation that follows a fixed elimination order: existence, category, geometry, and structure. Once a higher-priority condition is satisfied, the reasoning short-circuits, yielding a final structure_error conclusion. For clarity, the prompt shown in the figure is a simplified illustrative version; the actual training prompt enforces the same reasoning structure and constraints.

### C.2 Prompt Design Principles

A single unified system prompt is used to ensure consistency and controllability across all generated CoT samples. The prompt enforces the following core principles.

Pred-only visual grounding. Although both Pred and GT images are available to the model to understand the scene, the generated explanation is constrained to rely only on observable cues in the Pred image. Explicit prohibitions prevent any reference to ground truth, labels, or cross-image comparison, ensuring that the resulting CoT is visually grounded and safe for deployment.

target, while the vision and language backbones are adjusted more conservatively. This asymmetric optimization reflects their different roles in the task. The vision backbone focuses on refining geometric perception, whereas limited adaptation of the language backbone improves alignment between textual queries and fine-grained visual evidence in complex scenes. Despite its increased flexibility, this strategy remains well regularized and avoids excessive drift from pretrained semantics.

Priority-based short-circuit reasoning. The reasoning process follows a strict, ordered error taxonomy:

- (1) extra_lane_line
- (2) category_error
- (3) geometry_error
- (4) structure_error

(5) no_error

Once a higher-priority error is identified, all lower-priority checks must be skipped and must not be mentioned. This short-circuit structure mirrors the Judge Agent’s inference-time behavior and prevents mixed or internally inconsistent explanations.

Fixed-length structured rationale. Each CoT explanation is constrained to exactly four sentences. The initial sentences rule out higher-priority error types using visible evidence, while the final sentence explicitly states the oracle error type as a conclusion. This fixed structure simplifies downstream parsing and stabilizes supervised training.

Concrete and verifiable language. Abstract or hedging expressions are disallowed. Instead, the model is required to describe concrete, image-verifiable phenomena such as early termination, over-extension, curvature deviation, gaps, misalignment, or partial coverage of visible markings.

### C.3 Prompt Instantiation

Given the system prompt, each instance is instantiated with a lightweight task prompt that injects per-sample metadata, including the predicted category name, bounding box, and oracle error type. Oracle-related fields are marked as do not question and do not mention, reinforcing that the model’s role is explanation rather than decision-making.

The output is constrained to a single JSON object of the form:

{"reason" : string},

where the value contains exactly four sentences.

### C.4 Qualitative Example

Figure C.1 illustrates a representative CoT example generated by GPT 5.2. The explanation follows the prescribed priority order, first ruling out existence, category, and local geometry issues, and then short-circuiting at a structure_error due to global path and coverage inconsistency. The rationale is strictly grounded in observable cues from the Pred image and does not reference any ground-truth annotations.

### C.5 Resulting CoT Dataset

The resulting CoT dataset consists of concise and deterministic rationales that: (i) are strictly grounded in the Pred image, (ii) align exactly with the oracle error labels, and (iii) follow the same prioritystructured decision logic as the Judge Agent.

During training, these CoT traces are used only as supervised reasoning targets. They are not exposed to downstream Planner or

Worker modules at inference time, preserving a clean separation between interpretable diagnosis and executable system actions.

### D Bad Cases

Although MapAgent achieves strong overall performance, some challenging cases remain difficult. Figure D.1 shows representative failure cases where the scene contains weak lane evidence, severe shadows, local occlusions, or highly ambiguous topology. In such situations, the backbone predictions may already deviate substantially from the underlying lane layout, making reliable refinement more difficult. Since MapAgent is designed to apply conservative and tool-grounded edits, it tends to prioritize precision and structural consistency over aggressive topology rewriting. As a result, when the visual evidence is extremely limited or the initial prediction is severely corrupted, certain missing, shifted, or structurally inconsistent lane segments may still remain unresolved. These cases highlight an inherent limitation of refinement-based pipelines: the final quality still depends on both the recoverability of visual cues and the quality of the initial backbone prediction.

### E Training-Time Prompt for Judge Supervision

To train the Judge Agent to produce priority-consistent quality assessments, we supervise it with the CoT dataset described in Section C. The generated rationales are used as reference reasoning traces, while the oracle error type and category provide structured supervision targets.

The training prompt presents each predicted lane line as a red semi-transparent mask overlaid on the BEV image. The model is instructedtotreatthe maskasaprediction hypothesis and to base its judgment on observable road evidence, including painted markings, curbs, asphalt boundaries, lane continuity, and local topology. This design encourages the Judge to verify whether the predicted lane is visually and structurally supported, rather than relying on the mask appearance alone.

The prompt follows the same priority-constrained evaluation rule used by the Judge Agent. Once a higher-priority error is confirmed, all lower-priority checks are skipped. This short-circuit mechanism prevents mixed diagnoses and keeps the training target consistent with inference-time decision logic. For virtual lines, the prompt allows predictions that are topologically justified even without visible painted markings, and treats them as extra_lane_line only when no such justification exists.

The model is required to output a reasoning trace enclosed within <think> tags, followed by one final error type and a concise evidence summary. The reasoning trace is supervised by the generated CoT rationales, whereas the final error type and evidence summary are supervised by oracle annotations and reference explanations.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

(a) Origin (b) Ground Truth (c) GeMap (d) Corrected (Ours)

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

(a) Origin (b) Ground Truth (c) DuMapNet (d) Corrected (Ours)

- Figure D.1: Representative bad cases under challenging road conditions. The examples illustrate scenarios with weak visual evidence, ambiguous lane topology, and large deviations in the initial predictions, where refinement remains difficult. Such cases reveal the practical limits of conservative correction and suggest directions for improving robustness in future work.

