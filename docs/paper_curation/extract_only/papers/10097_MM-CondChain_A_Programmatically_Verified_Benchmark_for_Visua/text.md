# arXiv:2603.12266v1[cs.CV]12Mar2026

[Figure 1]

[Figure 2]

## MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning

Haozhan Shen1,2∗ Shilin Yan1† Hongwei Xue1‡ Shuaiqi Lu1 Xiaojun Tang1 Guannan Zhang1 Tiancheng Zhao3‡ Jianwei Yin2

1Accio Team, Alibaba Group 2Zhejiang University 3ZJU-BJ {tattoo.ysl,xuehongwe}@gmail.com

† Project Leader ‡ Corresponding Author

Abstract Multimodal Large Language Models (MLLMs) are increasingly used to carry out visual workflows such as navigating GUIs, where the next step depends on verified visual compositional conditions (e.g., “if a permission dialog appears and the color of the interface is green, click Allow”) and the process may branch or terminate early. Yet this capability remains under-evaluated: existing benchmarks focus on shallow-compositions or independent-constraints rather than deeply chained compositional conditionals. In this paper, we introduce MM-CondChain, a benchmark for visually grounded deep compositional reasoning. Each benchmark instance is organized as a multi-layer reasoning chain, where every layer contains a non-trivial compositional condition grounded in visual evidence and built from multiple objects, attributes, or relations. To answer correctly, an MLLM must perceive the image in detail, reason over multiple visual elements at each step, and follow the resulting execution path to the final outcome. To scalably construct such workflow-style data, we propose an agentic synthesis pipeline: a Planner orchestrates layer-by-layer generation of compositional conditions, while a Verifiable Programmatic Intermediate Representation (VPIR) ensures each layer’s condition is mechanically verifiable. A Composer then assembles these verified layers into complete instructions. Using this pipeline, we construct benchmarks across three visual domains: natural images, data charts, and GUI trajectories. Experiments on a range of MLLMs show that even the strongest model attains only 53.33 Path F1, with sharp drops on hard negatives and as depth or predicate complexity grows, confirming that deep compositional reasoning remains a fundamental challenge.

Project Page: https://accio-lab.github.io/MM-CondChain

Github Repo: https://github.com/Accio-Lab/MM-CondChain

HuggingFace: https://huggingface.co/datasets/Accio-Lab/MM-CondChain

### 1 Introduction

As Large Language Models (LLMs) Abdin et al. [2024], Achiam et al. [2023], Anthropic, Yang et al. [2025a], Jiang et al. [2025], Qwen Team [2026], Google DeepMind [d], Li et al. [2025], Grattafiori et al. [2024], Liu et al. [2024] and Multimodal Large Language Models (MLLMs) Achiam et al. [2023], OpenAI, Google DeepMind [d,c,b,a], Qwen Team [2026], Bai et al. [2025], Yan et al. [2025], Anthropic, Hong et al. [2025] grow more capable, they are increasingly expected to go beyond simple visual question answering and tackle complex visual workflows where the correct action depends on

∗This work was done during an internship at Accio Team, Alibaba Group.

Input Image Other Benchmarks

[Figure 3]

single-layer attribute compositions (MMCompostion, Winoground etc)

independent constraints (MIA-Bench etc)

Is anyone wearing a blue top, meanwhile holding a phone?

Describe what is happening in the image in exactly two sentences. Your description must mention the bicycles in the background and the phones held by people in the foreground. Use present tense.

single-layer hard negative replacement (VL-CheckList, ARO etc)

Is the left person wearing bluetop? Is the left person wearing browntop?

MM-CondChain

① Intra-layer Compositional Complexity

|Qwen3.5397B-A17B|
|---|

|GPT-4o-1120|
|---|

|Gemini-3-Flash|
|---|

If the man in the center holding a folded paper either has brown hair and is wearing a t-shirt, or he is partially blocked by another object, while he also carries a mobile phone and is not sitting down, then continue; otherwise answer [Based on the foliage visible in the background, which season is depicted?] (A1. Summer A2. Spring A3. Winter A4. Autumn);

|Qwen3-VL-235BA22B-Thinking|
|---|

|GPT-5-0807|
|---|

|Gemini-3-Pro|
|---|

100

|Kimi-K2.5|
|---|

②Inter-layerDeepNestedCondition

Given the preceding conditions hold, if his upper garment is either a blue t-shirt or is currently folded, and it is also completely unobstructed with a white printed design on the center chest, then continue; otherwise answer [What type of bag is the man in the center wearing?] (B1. A messenger bag B2. A backpack B3. A duffel bag B4. A tote bag);

53.33

60

50.34

48.31

45.90 46.83

③ Hard Negative in the Whole Chain left

45.25

Given the preceding conditions hold, if the woman with a ponytail is either positioned on the right side of the image facing left or stands alone, and she is not wearing a hat while either wearing plastic sunglasses or sitting down, then continue; otherwise answer [What type of event does the equipment in the background indicate?] (C1. .. ;)

20.06

False Path: One condition is replaced by a minimally perturbed counterfactual. MLLMs are expected to exit early at this layer and answer the auxiliary question.

Given the preceding conditions hold, if her glasses are either brown and currently worn or they are lying on a table, while also featuring tinted lenses and not being positioned anywhere other than on the face, then continue; otherwise answer [What does the clothing of the three individuals in the foreground indicate about their relationship?] (D1...);

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Given the preceding conditions hold, if the electronic device associated with the man in the center is either a single silver object or is cylindrical in shape, and it is fully visible while being either held or positioned on the table, then continue; otherwise answer [What is the likely functional purpose of the yellow canopy structure in the background?] (E1...);

[Figure 11]

###### Advanced MLLMs

[Figure 12]

[Figure 13]

[Figure 14]

Perplexity ↑ Performance↓

[Figure 15]

Given all preceding conditions hold, please answer [Considering the folded map or guide held in the man's other hand, for what functional purpose is this device likely being used?] (F1. Watching a streaming movie F2. Assisting with navigation or coordinating a location-based activity F3....)

True Path: All condition holds. MLLMs are expected to reach the end of the condition chain and answer the final question.

- Figure 1: MM-CondChain targets visually grounded deep conditional reasoning beyond prior benchmarks. Top: existing benchmarks typically evaluate either shallow single-layer visual compositions or independent instruction constraints. Bottom left: MM-CondChain introduces nested, inter-layer conditional chains with rich intra-layer compositional predicates, where a minimally perturbed condition can create a hard negative that changes the execution path and causes early termination. Bottom right: experiments show that even advanced MLLMs achieve limited performance on this benchmark, highlighting visually grounded deep compositional reasoning as a fundamental challenge.

a chain of visual checks (e.g., if a dialog appears, verify it requests location access; if so and the app is trusted, click Allow; otherwise...). These tasks require visually grounded deep compositional reasoning: at each step, the model must verify a multi-factor visual condition, and then determines whether the workflow continues or terminates early. Thus, a natural question arises: can current advanced MLLMs reliably follow deeply compositional condition instructions that require verification against visual input at every step?

Answering this question requires a benchmark that systematically probes such capabilities. However, existing benchmarks fall short in two key respects. First, in compositional depth. Prior visual reasoning benchmarks Hsieh et al. [2023], Johnson et al. [2017], Hudson and Manning [2019], Hua

- et al. [2024] typically evaluate single-layer compositions (e.g., “Is the object red and large?”), while instruction-following benchmarks Zhou et al. [2023], Jiang et al. [2024b], Qian et al., Wen et al.

- [2024], Pyatkin et al. [2025], Ding et al. [2025] focus on independent constraints. Neither requires models to perform deep compositional reasoning across layers. In these tasks, the model must verify a multi-factor visual condition at each step, and the outcome of each step then determines the subsequent reasoning path. Second, in the difficulty of hard negatives. Some prior benchmarks include contrastive pairs for compositional understanding Thrush et al. [2022], Yuksekgonul et al. [2023], Zhao et al. [2022a,b], but these are usually limited to a single-layer change, such as replacing one attribute or relation.

To address these gaps, we introduce MM-CondChain, a benchmark for visually grounded deep compositional reasoning in MLLMs. Unlike prior benchmarks that test shallow-compositions or independent-constraints, MM-CondChain requires models to follow multi-layer control flow where

Table 1: Comparison with existing benchmarks. Compose: intra-layer multi-attribute composition; Nested: inter-layer chained conditions; Visual: conditions grounded in visual input; Hard Neg.: contrastive pairs with minimal perturbation; Prog. Verif.: ground truth verified via code execution; Determ.: deterministic evaluation without LLM-as-judge; Auto.: automated data construction.

Benchmark Compose Nested Visual Hard Neg. Prog. Verif. Determ. Auto. Visual Reasoning

SugarCrepe Hsieh et al. [2023] ✓ ✗ ✓ ✓ ✗ ✓ ✓ Winoground Thrush et al. [2022] ✓ ✗ ✓ ✓ ✗ ✓ ✗ ARO Yuksekgonul et al. [2023] ✓ ✗ ✓ ✓ ✗ ✓ ✗ MMComposition Hua et al. [2024] ✓ ✗ ✓ ✗ ✗ ✓ ✗

###### Instruction Following

IFEval Zhou et al. [2023] ✗ ✗ ✗ ✗ ✓ ✓ ✗ FollowBench Jiang et al. [2024b] ✗ ✗ ✗ ✗ ✗ ✗ ✗ MIA-Bench Qian et al. ✓ ✗ ✓ ✗ ✗ ✗ ✗ ComplexBench Wen et al. [2024] ✓ ✓ ✗ ✗ ✗ ✗ ✗ MM-IFEval Ding et al. [2025] ✓ ✗ ✓ ✗ ✗ ✗ ✓

MM-CondChain ✓ ✓ ✓ ✓ ✓ ✓ ✓

each decision is gated by a compositional condition that must be verified against the visual input, and where the execution may branch or terminate early.

However, building this kind of benchmark at scale is challenging. If we directly ask an MLLM agent to generate long, multi-layer visual reasoning chains, the results often contain logical conflicts, unclear visual references, or statements that cannot be reliably determined from the visual input. To address this, we decouple logical construction from natural-language writing through the proposed Verifiable Programmatic Intermediate Representation (VPIR). Instead of generating the final instruction directly, we first represent each layer as an executable, Python-like predicate and mechanically verify whether it is true or false against structured visual facts, and only then translate the verified logic into natural language. This makes the benchmark construction process reliable, controllable, and grounded in verifiable visual evidence.

Building on VPIR, we further develop an agentic synthesis pipeline that incrementally constructs each benchmark instance, as illustrated in Figure 1. At each layer, the pipeline generates a visually grounded compositional condition, verifies it mechanically against structured visual facts, and only then extends the reasoning chain. VPIR explicitly represents both the verified condition and its minimally perturbed counterfactual at each layer, which naturally enables chained hard negatives. As shown in Figure 1, flipping a single predicate can change the execution path while keeping the overall instruction nearly unchanged, thereby forcing the model to accurately verify every condition along the way. Compared with prior benchmarks, which mainly test shallow compositions or independent constraints, our benchmark targets deep, multi-layer reasoning with chained hard negatives. Table 1 summarizes the differences between MM-CondChain and existing benchmarks.

Using this pipeline, we instantiate MM-CondChain across three visual domains: natural images, data charts, and GUI trajectories. Experiments on a range of state-of-the-art MLLMs show that visually grounded deep compositional reasoning remains highly challenging: even the strongest model achieves only 53.33 average Path F1, performance drops sharply on False-path hard negatives, and accuracy further degrades as reasoning depth and predicate complexity increase.

Our contributions are summarized as follows:

- • We introduce MM-CondChain, the first benchmark for visually grounded deep compositional reasoning, featuring multi-layer control flow with chained hard negatives.
- • We propose a VPIR-based agentic synthesis pipeline that decouples logical construction from language rendering, enabling scalable benchmark construction with mechanical verifiability.
- • We instantiate the framework across three visual domains and evaluate ten MLLMs, showing that even state-of-the-art models struggle with fine-grained verification of compositional visual conditions, especially on hard-negative instances and under greater depth or predicate complexity.

### 2 Related Work

Programmatically Verifiable Evaluation. IFEval Zhou et al. [2023] introduced verifiable instructions whose compliance can be checked by simple Python functions, focusing on surface-level constraints. IFBENCH Pyatkin et al. [2025] extended this with out-of-domain constraints and used programmatic verification as reinforcement learning rewards. In both cases, verification occurs post-hoc: code checks whether model outputs satisfy prescribed format rules. Our approach differs fundamentally: we apply programmatic verification during benchmark construction, not evaluation. Rather than checking output formats, we verify the semantic correctness of generated conditions by executing predicates against extracted visual facts. This ensures benchmark data is logically sound by design, eliminating contradictions that arise when LLMs directly generate complex instructions. In short, prior work uses code to judge outputs; we use code to guarantee data quality.

Compositional and Logical Visual Reasoning. Recent advancements evaluate MLLMs beyond basic perception by targeting compositional relations, spatial intelligence, and logic Zerroug et al. [2022], Zhang et al. [2019], Jiang et al. [2024a], Yang et al., 2026]. Frameworks such as VisuLogic Xu et al. [2025b], VER-Bench Qiang et al. [2025], and LogicVista Xiao et al. [2024] challenge models with visual-centric puzzles that demand fine-grained evidence extraction to preclude text-only shortcuts. Concurrently, multi-step capabilities and rigorous analytical deductions are assessed through sequential reasoning tasks Lu et al. [2024], Masry et al. [2022], Zhang et al. [2024b], Qian et al. [2025]. Our approach differs in structure: while existing frameworks predominantly evaluate singlelayer compositions, isolated visual relations, or sequential reasoning without verified branching, MM-CondChain targets visually grounded deep compositional reasoning under multi-layer control flow. At each step, the model must verify a compositional visual condition, and the outcome of one step determines the next reasoning path.

Complex Visual Instruction Following. The evaluation of instruction following has recently transitioned from purely textual constraints to multi-modal and cross-contextual environments. Benchmarks like MIA-Bench Qian et al., VC-IFEval He et al. [2026], and MC-Bench Xu et al.

- [2025a] test the strict adherence of MLLMs to layered, visual-centric directives. To navigate these complex tasks, models increasingly leverage structured inference paradigms such as Visual Chain-ofThought (VCoT), Visual-Interleaved CoT, and step-by-step curriculum learning Chen et al., Thawakar et al. [2025], Shao et al. [2024], Wu et al. [2025]. Our approach differs structurally: prior visual instruction datasets usually present flat, additive constraints, where missing one visual detail mainly reduces an overall compliance score. In contrast, MM-CondChain organizes instructions as multilayer chains of compositional visual conditions, so that failing one condition changes the downstream execution path. Moreover, VPIR allows us to pair each verified chain with a minimally perturbed counterfactual, producing mechanically verified hard negatives that are nearly identical in wording but differ in execution outcome.

### 3 VPIR-based Agentic Benchmark Construction

#### 3.1 Overview

Directly prompting an MLLM agent to generate long, multi-layer compositional reasoning chains often leads to logical inconsistencies and unverifiable claims. To address this, we propose a VPIRbased agentic benchmark construction pipeline that decouples logical construction from language rendering. The core idea is to first construct a Verifiable Programmatic Intermediate Representation (VPIR), which is executable Python-like predicates whose truth values can be mechanically verified against visual facts. We then render the verified logic into natural language. Figure 2 illustrates the overall pipeline.

Given a multimodal input (e.g., a natural image, a chart, or a GUI trajectory), the pipeline iteratively builds a multi-layer reasoning chain. At each layer, it selects a visually grounded subject, extracts structured facts, generates an executable VPIR predicate, and renders the verified predicate into natural language (Sec. 3.2). Each layer must pass verification before the chain can extend further.

###### Input Data Domain

Chart Image + metajson GUI Trajectory + Annotation

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

###### Composer

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Natural Image Only

[Figure 25]

First, let a QA generator to generate an auxiliary QA pair for each layer, and an additional final QA for end layer. All QAs are formated as MCQ

(Lebanon: 57) (Venezuela: 47)

(Action 1: Click) (Target: Google App)

...

...

###### Now, we have

True Condition text 1

False Condition text 1

Aux Quesiton 1

Not at the layer limit; more visual information remains. EXTEND Assign Relation

Not at the layer limit; more visual information remains. EXTEND Assign Relation

[Figure 26]

[Figure 27]

[Figure 28]

Layer limit reached, or visual cues exhausted.

True Condition text 2 False Condition text 2

Aux Question 2

Planner

FINISH

True Condition text T False Condition text T

Aux Question T

Seed Relationship: Choose a main instance

Final Question

Carried Item Focus

Spatial Shift

FINISH

###### True Path:

St

###### St

[Figure 29]

[Figure 30]

[Figure 31]

If

white surfboard held by the Stormtrooper

, then continue else answer

Stormtrooper action figure

True Condition text 1

Aux Quesiton 1

t=1

t=2

Fact Extractor

[Figure 32]

[Figure 33]

[Figure 34]

Verifiable in Python Env

decouples logic from language

If True Condition text T , then continue else answer Aux Question T ; answer Final Question

[Figure 35]

True Condition text 1

###### Ft

((any(i.get('name') == 'surfboard' and i.get('orientation') == 'vertical' for i in carried_items) and spatial_relation == 'on the sand next to a female figurine') or pose == 'sitting') and (not (action == 'swimming') and 'orange' in colors)

either the figure carries a vertical surfboard and is located on the sand next to a female figurine, or it is currently sitting, and furthermore, it is not swimming and features the color orange

"

[Figure 36]

"colors": ["white", "black","orange"], "material": ["plastic" ], "shape": "humanoid", "action": "standing", "pose": "standing upright", "position": "foreground right", "spatial_relation": "on the sand next to a female figurine", "orientation": "facing forward", "carried_items": [

CheckList

[Figure 37]

###### Repace

###### False Path:

≥4 logical operators

[Figure 38]

[Figure 39]

If True Condition text 1 , then continue

True Condition text 2

≥4 attribute keys

((any(i.get('name') == 'surfboard' and i.get('orientation') == 'vertical' for i in carried_items) and spatial_relation == 'on the sand next to a female figurine') or pose == 'sitting') and (not (action == 'standing') and 'orange' in colors)

[Figure 40]

[Figure 41]

≥2 nested group

If False Condition text i

Translator

, then continue else answer

{

"name": "surfboard", "has_text": true

t=1

Aux Question i

...

False Condition text i

[Figure 42]

VPIR Generator

Hard Negative

- Figure 2: Overview of the MM-CondChain agentic synthesis pipeline. Given a multimodal input, the Planner iteratively extends a conditional chain: at each layer, structured facts are extracted, a VPIR predicate pair is generated and verified via code execution, and the logic is rendered into natural language. The Composer then compiles the verified chain into paired True-path and Falsepath instances for evaluation.

To coordinate chain construction, a Planner (Sec. 3.4) decides whether to extend, terminate, or rollback the chain, working together with a Verifier (Sec. 3.3) that performs quality control. Finally, a Composer (Sec. 3.5) compiles each verified chain into paired benchmark instances: a True-path where all conditions hold, and a False-path where one condition is replaced by a minimally perturbed counterfactual. This near-isomorphic design yields hard negatives that require both precise visual grounding and deep compositional reasoning

#### 3.2 Layer-wise VPIR Synthesis: Facts, Strategy, and Programmatic Logic

We construct a deep control-flow chain iteratively, where each layer depends on the successful verification of its predecessors. At each layer t, the pipeline synthesizes verifiable layer logic through a four-stage workflow: (1) selecting a relational strategy rt that constrains subject transition, (2) extracting structured facts Ft grounded in visual evidence, (3) generating the programmatic predicate pair (pt,p˜t), and (4) rendering executable logic into natural language. This decoupling of logic formation from language rendering ensures that truth values are mechanically computable before any linguistic expression.

#### 3.2.1 Step 1: Relational Strategy & Subject Selection

At each layer t, we choose a relational strategy rt ∈ R, where R is a discrete taxonomy of inter-layer relations (e.g., Deepening vs. Transition). Intuitively, Deepening continues reasoning about the same subject by zooming into its parts or new attribute dimensions, while Transition moves to a distinct but related entity via spatial/semantic relations.

Given the input sample x and the execution-ordered chain history Ht−1, we instantiate rt as a subject filter and construct a feasible set of visually grounded candidates:

Ωt ≜ Ω(x,Ht−1,rt). (1)

We use Ωt to constrain the extractor in Step 2, which selects the subject and extracts facts jointly. Here Ht−1 summarizes previous layers in execution order, including their selected subjects and verification outcomes, since the control flow is evaluated sequentially along the chain.

#### 3.2.2 Step 2: Structured Fact Extraction

To prevent hallucination during logic synthesis, the pipeline grounds generation in a structured, domain-agnostic factual representation. Conditioned on rt (and thus Ωt) and history Ht−1, the extractor jointly selects a grounded subject St ∈ Ωt and produces the subject–fact pair:

(St,Ft) = E(x,rt,Ht−1). (2) For the seed layer (t = 1), H0 = ∅ and r1 is a foundational seed strategy. The extracted facts Ft constitute a typed key-value mapping {(k,vk)}2, where each key k denotes a visual attribute dimension (e.g., color, spatial_relation, count, gui_state) and vk is a typed observation (e.g., red, left-of, 50, list-layout).

We enforce two critical design principles:

- • Object-Centric Grounding: The subject St must be uniquely localizable in the visual input, ensuring conditions are rooted in visual evidence.
- • Structure-First Representation: By representing Ft as a JSON dictionary (rather than free-form

text), we define a programmatic namespace Vt ≜ keys(Ft), enabling mechanical verification via executable semantics.

#### 3.2.3 Step 3: VPIR Generation

With the fact space Ft and variable namespace Vt established, the pipeline synthesizes the Verifiable Programmatic Intermediate Representation (VPIR). We define the VPIR at layer t as a pair of

executable predicate programs: the true-logic pt and the counterfactual false-logic p˜t.

To formally verify these predicates, we evaluate VPIR in a sandboxed execution environment Env(Ft). This environment exposes only whitelisted built-in operators B (e.g., len, set, all, any) and binds

each fact key k ∈ Vt to its extracted value Ft[k]. The semantics of a VPIR predicate is then defined by its deterministic boolean output:

p (Ft) ≜ Exec(p;Env(Ft)) ∈ {0,1}. (3)

This programmatic formulation guarantees absolute verifiability, the generated predicates are accepted only by mechanical execution against Ft:

pt (Ft) = 1, p ˜t (Ft) = 0. (4)

Furthermore, through prompt-based constraints, we encourage (i) non-trivial predicate complexity (e.g., multi-clause boolean compositions with nested structure and multiple fact keys) and (ii) minimal counterfactual perturbations in p˜t relative to pt, so that True/False instances remain nearly isomorphic in surface form and cannot be distinguished by shallow textual cues.

#### 3.2.4 Step 4: Logic Rendering

Once the VPIR pair (pt,p˜t) passes programmatic verification, an LLM-based Translator renders the executable logic into natural language: a true condition text ct and a counterfactual condition text c˜t (rendered from p˜t). Here c˜t is retained for downstream paired-path compilation (Sec. 3.5), where it will be substituted at a single layer to trigger early termination in the False-path instance.

Crucially, truth values are anchored in code execution; language is merely a surface rendering for evaluation. We then apply expression-level verification (Sec. 3.3) to ensure the rendering is fluent, unambiguous, and faithful to the verified VPIR semantics.

Tiny Example. Consider a red car parked left of a blue truck. At layer t, the Planner selects rt = Transition; the extractor produces St = “the car” with Ft = {color : “red”,position : “left”}; the pipeline generates pt: color == "red" and position == "left" and its minimal perturbation p˜t: color == "blue" and ...; finally, the Translator renders ct = “the car is red and on the left”. Mechanical execution confirms pt = 1, p ˜t = 0.

2“Typed” means values in Ft use JSON-compatible types (e.g., str/int/float/bool, list/dict) and are exposed as variables for VPIR execution. VPIR only permits whitelisted primitives (e.g., len, any/all, min/max/sum) on these types, ensuring deterministic verifiability.

#### 3.3 Dedicated Verifier

We employ a dedicated MLLM-based Verifier for centralized quality control throughout chain construction.

At layer t, a candidate is a bundle Bt = (St,Ft,pt,p˜t,ct,c˜t). The Verifier returns a structured verdict v = {passed,reasons,fix_hint}. Verification proceeds in two stages:

- Stage I: Fact and Subject Verification. Stage I validates the grounded materials (St,Ft) before any language rendering occurs. It checks:

- • Visual Grounded: St must be uniquely localizable in the input x;
- • Non-Repetition: the subject and extracted facts must not duplicate those in Ht−1;
- • Relational Compliance: the selection must satisfy the chosen strategy rt;
- • Schema & Consistency: Ft must conform to the domain schema with coherent cross-attribute values.

- Stage II: Language Realization Verification. Stage II validates the rendered natural-language conditions (ct,c˜t) against the verified VPIR predicates (pt,p˜t). It checks:

- • Semantic Fidelity: the natural language must preserve the VPIR logic without residual code artifacts;
- • Unambiguous Reference: each clause must explicitly name its subject, avoiding coreference ambiguity;
- • Counterfactual Quality: c˜t must faithfully reflect p˜t while remaining minimally perturbed from ct.

Feedback-Driven Regeneration. Verification is stage-aware: failures in Stage I trigger regeneration of (St,Ft), while failures in Stage II retain the verified (St,Ft,pt,p˜t) and only re-render (ct,c˜t).

#### 3.4 Planner: Verification-Aware Chain Control

We introduce a verification-aware Planner that governs chain-level control flow. This dynamic interplay between the MLLM-based Planner and the Verifier constitutes the agentic core of our pipeline: the Planner proposes actions, the Verifier provides feedback, and the Planner adapts accordingly.

At each layer t, the Planner outputs a decision (at,rt) = π(Ht−1), where at is an action and rt ∈ R is a relational strategy (Sec. 3.2, Step 1). The action space consists of three options:

- • EXTEND: synthesize a new layer under the proposed strategy rt;
- • FINISH: terminate the chain and proceed to composition;
- • ROLLBACK: discard the most recent non-seed layer and resume from a verified prefix.

- 3.4.1 Hybrid Depth Control

The Planner combines hard-coded rules with an MLLM-driven policy. Given a target depth interval [dmin,dmax]:

- • If depth(Ht−1) < dmin: force at = EXTEND;
- • If depth(Ht−1) ≥ dmax: force at = FINISH;
- • Otherwise: delegate to at = πMLLM(Ht−1), an MLLM-based policy that decides based on chain coherence and remaining synthesis potential.

#### 3.4.2 Verification-Aware Backtracking

The Planner is tightly coupled with the Verifier (Sec. 3.3). When repeated verification failures occur at the current frontier (e.g., persistent subject repetition or unsatisfiable relational constraints), the Planner triggers ROLLBACK, pruning the failing layer and resuming synthesis from the last verified prefix. This feedback loop prevents the pipeline from getting stuck in unrecoverable states.

Once the Planner emits FINISH, the chain is finalized and forwarded to the Composer (Sec. 3.5).

#### 3.5 Composition: Paired-Path Instruction Compilation

After the Planner emits FINISH, we obtain a verified control-flow skeleton comprising T layers, where each layer t provides a grounded subject St and its true/counterfactual conditions (ct,c˜t).

Since the control flow may terminate at any layer, we attach a question to each possible exit point: a final question qfin for the terminal layer, and an auxiliary question qtaux for each intermediate layer. All questions are multiple-choice with deterministic answers. Unlike prior complex-instruction benchmarks that depend on LLM-as-judge for open-ended evaluation Zhang et al. [2025], Yang et al. [2025b], Deshpande et al. [2025], Zou et al. [2025], Yao et al. [2023], Wen et al. [2024], Qian et al., our design enables fully reproducible and objective scoring. The Composer compiles this skeleton into evaluation-ready instances through two steps.

- Step 1: Subject De-leakage. A subject description may inadvertently reveal its associated condition. For example, if a condition tests “whether the car is red,” describing the subject as “the red car” would

leak the answer. To prevent this, an MLLM-based rewriter rephrases each St into a safe subject S¯t by removing condition-revealing attributes and substituting alternative visually grounded descriptors

(e.g., spatial location) when needed. The core constraint is that S¯t must remain uniquely referential, i.e., it should still unambiguously identify the same target object St in the visual input.

- Step 2: Paired-Path Instantiation. From each skeleton, we compile two nearly isomorphic evaluation instances:

- • True-path: All conditions {ct}Tt=1 hold, so the control flow reaches the terminal layer and the correct answer corresponds to qfin.
- • False-path: We uniformly sample a divergence layer j ∈ {1,...,T−1} and swap cj ← c˜j. Since p ˜j (Fj) = 0, the flow terminates early at layer j, and the correct answer becomes qjaux.

Finally, we merge each (S¯t,ct) into a fluent natural-language if-clause to produce the final nested instruction. This paired compilation creates a hard negative: the two paths share identical structure and nearly identical wording, differing only in a single subtly perturbed condition hidden among multiple true ones. Distinguishing them thus requires fine-grained reasoning over each condition rather than superficial pattern matching.

#### 3.6 Domain-Specific Instantiation

The VPIR synthesis pipeline is domain-agnostic at its core; domain-specific adaptations are confined to input preprocessing and fact extraction. We instantiate the framework across three visual domains (natural images, data charts, and GUI trajectories), each requiring different input normalization before entering the unified engine (Table 2).

Natural Images. No preprocessing is required; the MLLM directly extracts open-schema visual attributes (e.g., color, spatial relations) from the raw image.

Charts. ChartQA annotations often exhibit x/y length mismatches and zero-placeholder artifacts (missing data points marked by null bounding boxes). We apply deterministic CSV alignment to fix length inconsistencies, and LLM-based value extraction to repair missing entries, producing clean meta_json before invoking the engine.

#### Table 2: Domain-specific adaptations within the unified VPIR framework.

##### Aspect Natural Chart GUI

Input Single image Image + metadata Image seq. + annotation Preprocessing None CSV align + LLM repair Completeness + CoAT parse Fact Focus Visual attributes Numerical stats Temporal actions

GUI Trajectories. We verify trajectory completeness (ensuring screenshot count matches annotation length), parse CoAT action descriptions into structured fields per step (action type, target element, location, etc.), and pass the multi-image sequence to the engine.

Crucially, the core components (VPIR predicate generation, two-stage verification, and Planner backtracking) remain entirely domain-agnostic. Domain-specific code is isolated to input adapters, fact builders, and strategy registries. This demonstrates that the VPIR abstraction generalizes across visual modalities, from unconstrained natural scenes to structured data visualizations and interactive interface trajectories. Full preprocessing details are provided in Appendix.

### 4 Evaluation

#### 4.1 Evaluation setup

Data Statistics. We construct MM-CondChain from three visual domains using publicly available datasets. The Natural domain comprises 398 images drawn from SAM Kirillov et al. [2023] (204) and GQA Hudson and Manning [2019] (194). The Chart domain includes 200 chart images from ChartQA Masry et al. [2022], spanning bar, line, and pie charts with structured numerical annotations. The GUI domain contains 377 interaction trajectories (3,421 screenshots in total, averaging 9.07 frames per trajectory) sourced from AITZ Zhang et al. [2024a], which provides fine-grained reasoning annotations over AITW Rawles et al. [2023]. This results in 975 evaluation samples in total, each containing a paired True-path and False-path instance.

Extracted Facts and VPIR Variable Statistics. Figure 3 shows the attribute distributions in extracted facts and the variables used in VPIR predicates across domains. We observe clear domainspecific patterns: Natural instances mainly rely on object attributes and spatial relations, Chart instances concentrate on numerical and structural statistics, and GUI instances emphasize action, state, and trajectory-level metadata. We also find that the VPIR variable distributions do not simply mirror the full extracted fact distributions. Instead, VPIR selectively reuses subsets of extracted attributes to compose executable predicates, indicating that benchmark difficulty is driven by structured compositional reasoning over grounded visual facts rather than by raw attribute frequency alone.

Logical Pattern Statistics. Figure 4 shows that VPIR expressions in MM-CondChain exhibit substantial structural diversity. Although several pattern families appear more frequently than others, the benchmark is not dominated by one or two simple templates: the top-20 templates cover only 50.07% of all expressions, and 128 unique templates are needed to reach 80% coverage. This indicates that the benchmark contains a broad range of compositional logic structures rather than a small set of repeated forms. Moreover, the dominant templates themselves are already structurally complex. As illustrated by the example on the right, a single VPIR template can involve multiple predicates, nested logical operators, executable program form, and its corresponding natural-language rendering. As a result, correctly solving these instances requires not only visual grounding of the relevant objects, attributes, and relations, but also compositional reasoning over how these visual factors jointly determine whether the condition holds.

Benchmark Generation. We employ Gemini-3-Pro Google DeepMind [d], currently among the strongest MLLMs in comprehensive reasoning capabilities, to instantiate all MLLM and LLM agents in our synthesis pipeline, including the Planner, Verifier, Fact Extractor, and Translator.

Top 20 Attribute Frequencies in Extracted Facts of Natural Domain

position

| |[Figure 43]<br><br>191<br>192<br><br><br>210<br><br>222<br><br>272<br><br>333<br><br>424<br><br>434<br><br>656<br><br>900<br><br>1002<br><br>1036<br><br>1184<br><br>1354<br><br>1517<br><br>1635<br><br>1722<br><br>1869<br><br>2039<br><br>2087<br><br>Color/Material<br><br>Shape/Size Appearance State<br><br>Spatial<br><br>Action/Pose<br><br>Clothing<br><br>Body Features<br><br>Text<br><br>Count Stats<br><br>Other| | | | |
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
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

spatial_relat...

colors

state

shape

material

parts

is_occluded

size

is_cropped

orientation

pattern

action

has_text

visible_text

pose

clothing_items

count

age_appearance

gender

0 500 1000 1500 2000

Frequency

(a)

Top 20 Variables Used in VPIR of Natural Domain

state

| |[Figure 44]<br><br>100<br><br>106<br><br>109<br><br>121<br><br>172<br><br>180<br><br>195<br><br>216<br><br>280<br><br>306<br><br>501<br><br>503<br><br>554<br><br>595<br><br>804<br><br>835<br><br>924<br><br>1142<br><br>1177<br><br>1248<br><br>Color/Material<br><br>Shape/Size Appearance State<br><br>Spatial<br><br>Action/Pose<br><br>Clothing<br><br>Body Features<br><br>Surface/Markings<br><br>Text<br><br>Count Stats<br><br>Other| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

parts

colors

is_occluded

position

spatial_relat...

orientation

material

shape

is_cropped

action

pattern

visible_text

clothing_items

has_text

pose

surface_marki...

count

body_features

printed_desig...

0 200 400 600 800 1000 1200 1400

Frequency

###### (b)

Top 20 Attribute Frequencies in Extracted Facts of Chart Domain

metric_name

| |[Figure 45]<br><br>339<br><br>339<br><br>339<br><br>339<br><br>339<br><br>339<br><br>339<br><br>370<br><br>370<br><br>583<br><br>583<br><br>771<br><br>922<br><br>922<br><br>1005<br><br>1005<br><br>1005<br><br>1005<br><br>1040<br><br>1040<br><br>Chart Meta<br><br>Central Stats<br><br>Range Stats Count Stats Coordinates Point Labels Delta<br><br>Trend| | | | | |
|---|---|---|---|---|---|---|
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
| | | | | | | |

chart_type

series_name

mean_y

min_y

max_y

median_y

y_range

num_points

max_label

min_label

delta

abs_delta

x1

x2

y1

y2

x1_index

x2_index

direction

0 200 400 600 800 1000

Frequency

(c)

Top 20 Variables Used in VPIR of Chart Domain

mean_y

| |[Figure 46]<br><br>102<br><br>105<br><br>112<br><br>114<br><br>119<br><br>126<br>127<br><br><br>141<br><br>209<br><br>211<br><br>215<br><br>227<br><br>230<br><br>268<br><br>326<br><br>329<br><br>352<br><br>451<br><br>496<br><br>692<br><br>Central Stats<br><br>Range Stats Count Stats Coordinates Delta<br><br>Rank Stats<br><br>Dispersion Gap Stats Series Values<br><br>Threshold Count<br><br>Other| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

median_y

max_y

min_y

y1

y2

y_range

num_points

points

y1_rank

abs_delta

y2_rank

gap_to_max

y

rank_in_series

std_y

y_a

y_b

iqr

count_ge_mean

0 100 200 300 400 500 600 700

Frequency

###### (d)

Top 20 Attribute Frequencies in Extracted Facts of GUI domain

instruction

| |[Figure 47]<br><br>389<br><br>389<br><br>454<br><br>454<br><br>454<br><br>454<br><br>454<br><br>454<br><br>454<br><br>454<br><br>454<br><br>454<br>455<br><br><br>687<br><br>687<br><br>687<br><br>687<br><br>687<br><br>1501<br><br>1501<br><br>Trajectory Meta<br><br>Action Type<br><br>VLM Summary| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

episode_length

step_id

vlm_screen_la...

vlm_primary_c...

vlm_has_navig...

vlm_has_statu...

vlm_visual_su...

vlm_starting_...

vlm_ending_sc...

vlm_most_comm...

vlm_visited_l...

vlm_layout_ch...

vlm_has_navig...

vlm_has_dialo...

vlm_has_scrol...

vlm_scroll_di...

vlm_is_contin...

action_type

is_scroll

0 200 400 600 800 1000 1200 1400 1600

Frequency

Top 20 Variables Used in VPIR of GUI domain

vlm_screen_la...

| |[Figure 48]<br><br>168<br><br>173<br>174<br><br><br>178<br><br>183<br><br>188<br><br>188<br><br>192<br><br>204<br><br>222<br>223<br><br><br>249<br><br>251<br><br>263<br><br>291<br><br>293<br><br>309<br><br>323<br><br>367<br><br>385<br><br>Trajectory Meta<br><br>Action Type<br><br>Screen Basic<br><br>VLM Summary Element Count<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

episode_length

vlm_visited_l...

num_icon_elem...

num_text_elem...

num_clicks

new_text_count

num_elements_...

num_scrolls

shared_text_c...

vlm_layout_ch...

vlm_transitio...

vlm_visual_hi...

vlm_starting_...

num_elements_1

vlm_visible_c...

ad_is_click

num_elements_2

vlm_layout_ch...

vlm_ending_sc...

0 50 100 150 200 250 300 350 400

Frequency

(e)

(f)

- Figure 3: Top attribute frequencies in extracted facts and VPIR variables across domains. (a,c,e) show the top 20 attributes in extracted facts for the Natural, Chart, and GUI domains, respectively; (b,d,f) show the top 20 variables used in VPIR predicates for the corresponding domains.

Evaluated Models. We evaluate a range of MLLMs spanning both open-source and proprietary families. Open-source models include the Qwen3-VL series Bai et al. [2025], Qwen3.5 series Qwen Team [2026], GLM-4.6V series Team et al. [2025], Kimi-K2.5 Team et al. [2026], InternVL3 series Zhu et al. [2025] and InternVL3.5-8B Wang et al. [2025]. Proprietary models include GPT4o-1120 Achiam et al. [2023], GPT-5-0807 OpenAI, Gemini-2.5-Flash Google DeepMind [a], Gemini-2.5-Pro Google DeepMind [b], Gemini-3-Flash Google DeepMind [c], Gemini-3-Pro Google DeepMind [d], Qwen3-VL-Flash, and Qwen3-VL-Plus.

Evaluation Metrics. We report three metrics for each domain: (1) True-path Accuracy: the percentage of True-path instances where the model correctly follows all conditions and selects the answer corresponding to the final question qfin; (2) False-path Accuracy: the percentage of False-path instances where the model correctly identifies the early-termination point and selects the auxiliary answer qjaux; (3) Path F1: the harmonic mean of True-path and False-path accuracy, measuring balanced performance across both paths. We also report Avg(F1), the arithmetic mean of Path F1 across three domains, as the overall score.

###### Overall Logic Pattern Composition and Dominant Templates of VPIR

###### Top-20 Dominant VPIR Logic Templates

Mixed-NoNeg Conj-Negation Disj-Negation

[Figure 49]

(A & B) & ((C & D) | (E & F)) ((A & B) | (C & D)) & (E & F)

323

###### Overall Logic Composition

307

((A & B) & C) & ((D & E) | (F & G)) ((A & B) | C) & (D & !E)

296

229

###### Example Instantiation of VPIR

(A | B) & (C & !D) ((A & B) | !C) & (D & E) ((A & B) | C) & (!D & E)

144

Domain: natural Template: ((A B) C) (D ¬E) Predicate mapping:

134

132

((A & B) | (C & D)) & E ((A & B) & (C & D)) & ((E & F) | (G & H))

117

- A = len(colors) >= 2

- B = 'purple' in colors

- C = shape != 'round'

- D = state == 'whole'

- E = is_occluded VPIR code: ((len(colors) >= 2 and 'purple' in colors) or shape != 'round') and (state == 'whole' and not is_occluded) Rendering Language: either it displays at least two colors including purple or is not round in shape, while also being whole and unobstructed

84

Total 4634

((A & B) | C) & (D & E) (A & B) & ((C & D) | !E)

80

65

((A & B) | (C & D)) & (E & !F) ((A & !B) | C) & (D & E) (A & !B) & ((C & D) | E)

61

57

47

(A & B) | (C & D) (A & B) | (C & !D)

46

42

((A & B) & ((C & D) | (E & F))) & (G & H)

40

(A | B) & (C & D) A & ((B & C) | (D & E))

39

Top-20 covers 50.0% of all expressions 80% coverage needs top 128 Unique templates: 678

38

Conj-Negation (2065, 44.6%)

Mixed-NoNeg (1953, 42.1%)

((A & B) | (C & D)) & ((E & F) & G)

34

Disj-Negation (409, 8.8%)

0 50 100 150 200 250 300 350 400

Rare-Other (109, 2.4%)

Count

Terminal-Negation (98, 2.1%)

- Figure 4: Logic pattern composition of VPIR expressions. Left: overall distribution of high-level VPIR logic families. Middle: top-20 dominant concrete VPIR templates. Right: an example showing how a VPIR template is instantiated into executable predicates and natural-language conditions.

Implementation Details of Evaluation. All models are evaluated in a zero-shot setting using each provider’s default API parameters (temperature, max tokens, etc.). Each instance is presented as a multiple-choice question with a specified output format. Answers are extracted by prioritizing the last \boxed{...} match, with fallback to standalone option patterns; unparseable outputs are marked incorrect.

#### 4.2 Main Results

Main Results. The main results are summarized in Table 3. Overall, current MLLMs still struggle on MM-CondChain. Among all evaluated models, Gemini-3-Pro achieves the best overall result with 53.33 average Path F1, followed by GPT-5-0807 at 50.34. Even the strongest model remains only slightly above 50 F1, indicating that visually grounded deep compositional reasoning under multi-layer control flow is still highly challenging for current MLLMs.

True vs. False Paths. A clear pattern is that many models perform substantially better on the True-path than on the False-path. For example, GPT-4o-1120 scores 83.92 vs. 12.81 on Natural, Qwen3.5-4B scores 88.92 vs. 15.37 on Natural, and Qwen3.5-9B scores 91.69 vs. 13.10 on Natural. This gap suggests that under complex multi-layer conditions, models tend to over-assume that the conditions hold and thus favor the “continue” branch. Such a bias can be risky in real visual workflows, where failing to detect a violated condition may cause the model to proceed when it should stop, switch branches, or reject the action.

Model Comparisons. Proprietary models generally outperform open-source ones in overall performance, with Gemini-3-Pro and GPT-5-0807 ranking first and second, respectively. At the same time, open-source models remain competitive in specific settings: notably, Qwen3.5-397B-A17B achieves the best score on GUI (F1=40.19), surpassing all proprietary models on that domain. We also observe that Thinking models generally outperform their Instruct counterparts, suggesting that explicit reasoning-oriented models are better suited for this complex benchmark.

Domain-wise Difficulty. We observe clear domain-dependent difficulty. GUI is the most challenging domain overall: its best F1 is only 40.19, lower than the best results on Natural (55.91) and Chart (66.04). This is likely because GUI instances require reasoning over multi-frame trajectories, user actions, and interface state transitions, whereas many Chart conditions reduce to deterministic numerical comparisons once the relevant values are grounded.

- Table 3: Main results on MM-CondChain across domains. All numbers are percentages (%). Path F1 is the harmonic mean of True- and False-path accuracy. Avg(F1) is the mean of the three domain F1 scores. Rows are sorted by Avg(F1) in ascending order within each category.

Natural Chart GUI Avg Model True False F1 True False F1 True False F1 F1

Open-Source MLLMs

Qwen3.5-0.8B 33.17 2.26 4.23 31.50 3.00 5.48 33.95 1.86 3.52 4.41 GLM-4.6V-Flash 83.92 9.55 17.14 81.91 5.53 10.36 87.53 0.53 1.05 9.52 InternVL3-8B 65.33 8.29 14.72 47.50 8.50 14.42 63.66 5.31 9.79 12.98 InternVL3.5-8B 82.41 10.30 18.31 76.00 19.50 31.04 82.23 1.33 2.61 17.32 InternVL3-14B 76.38 13.57 23.04 43.00 21.00 28.22 84.62 2.39 4.64 18.63 Qwen3.5-4B 88.92 15.37 26.20 86.50 20.00 32.49 65.78 7.69 13.77 24.15 Qwen3.5-35B-A3B 93.43 11.62 20.66 88.50 17.00 28.52 74.27 14.32 24.02 24.40 Qwen3-VL-30B-A3B-Instruct 27.64 27.14 27.38 44.00 35.50 39.30 73.67 7.98 14.40 27.03 InternVL3-38B 73.62 20.60 32.20 31.00 31.50 31.25 57.03 12.47 20.46 27.97 Qwen3.5-9B 91.69 13.10 22.92 86.50 28.50 42.87 71.62 11.67 20.07 28.62 Qwen3-VL-8B-Instruct 47.98 30.81 37.52 39.78 39.78 39.78 58.67 12.53 20.65 32.65 GLM-4.6V 73.37 26.13 38.54 66.00 34.50 45.31 30.50 24.40 27.11 36.99 Qwen3-VL-8B-Thinking 60.71 30.48 40.58 49.50 37.00 42.35 37.14 27.85 31.83 38.25 Qwen3.5-122B-A10B 95.48 20.85 34.23 84.50 37.50 51.95 65.78 23.08 34.17 40.12 Qwen3-VL-30B-A3B-Thinking 30.90 31.16 31.03 58.00 56.50 57.24 40.53 27.73 32.93 40.40 Kimi-K2.5 75.57 41.06 53.21 46.00 52.00 48.82 50.93 25.20 33.72 45.25 Qwen3-VL-235B-A22B-Instruct 62.12 43.94 51.47 55.00 61.00 57.84 62.60 17.24 27.04 45.45 Qwen3.5-397B-A17B 52.01 31.16 38.97 67.00 52.00 58.55 40.05 40.32 40.19 45.90 Qwen3-VL-235B-A22B-Thinking 65.49 39.55 49.31 61.50 58.50 59.96 28.91 33.95 31.23 46.83

Proprietary MLLMs

- GPT-4o-1120 83.92 12.81 22.23 17.00 18.00 17.49 63.40 12.20 20.46 20.06 Gemini-2.5-Flash 29.40 48.24 36.53 35.50 47.00 40.45 6.90 44.83 11.95 29.64 Qwen3-VL-Flash 61.56 29.65 40.02 59.50 47.50 52.83 58.62 10.61 17.97 36.94 Gemini-2.5-Pro 38.94 55.28 45.70 55.50 64.50 59.66 10.34 54.38 17.38 40.91 Qwen3-VL-Plus 67.59 32.16 43.58 56.00 54.50 55.24 34.75 38.20 36.39 45.07 Gemini-3-Flash 54.77 41.46 47.19 60.50 63.50 61.96 36.87 34.75 35.78 48.31

- GPT-5-0807 80.65 33.67 47.51 63.50 67.50 65.44 30.77 49.87 38.06 50.34 Gemini-3-Pro 73.87 44.97 55.91 70.00 62.50 66.04 32.63 45.62 38.05 53.33

- Table 4: Effect of chain depth and predicate complexity on Path F1 (%). Left: Performance degrades as chain depth increases, with ∼30% relative drop from D=2 to D=6. Right: Increasing intra-layer predicate complexity (SIMPLE vs. COMPLEX) causes 28–36% degradation at fixed depth.

Model D=2 D=4 D=6 ∆2→6

Model SIMP. COMP. ∆

Gemini-3-Flash 70.68 53.85 47.19 −33.2% Qwen3-VL-Plus 61.51 52.56 43.58 −29.1% GPT-4o-1120 31.39 27.67 22.23 −29.2%

Gemini-3-Flash 65.26 47.19 −27.7% Qwen3-VL-Plus 62.91 43.58 −30.7% GPT-4o-1120 34.75 22.23 −36.0%

- 4.3 Design Ablations

- 4.3.1 Effect of Chain Depth.

To investigate how chain depth affects model performance, we construct ablation instances with controlled maximum depths of 2, 4, and 6 layers on the Natural domain and evaluate three representative models. As shown in Table 4 Left, all models exhibit consistent performance degradation as chain depth increases. From depth 2 to depth 6, Path F1 drops by approximately 29–33% in relative terms across all tested models. Notably, this degradation is not uniform: Gemini-3-Flash suffers the largest relative drop (−33.2%), despite achieving the highest absolute scores, suggesting that even strong models struggle to maintain accuracy as the number of sequential verification steps grows.

These results confirm that tracking multi-layer conditional logic poses a fundamental challenge for current MLLMs. The near-linear degradation with depth indicates that errors compound across layers, rather than being isolated to specific conditions. This underscores the value of MM-CondChain’s configurable depth design for probing the limits of sequential visual reasoning.

#### 4.3.2 Effect of Predicate Complexity.

Beyond chain depth, we examine how intra-layer predicate complexity affects model performance. We contrast two VPIR generation settings: SIMPLE predicates (at most 2 logical operators, at least 2 attribute keys, no nesting requirement) versus COMPLEX predicates (at least 4 logical operators, 4 attribute keys, and 2 nested groups). Both settings share the same chain depth to isolate the effect of compositional logic. As shown in Table 4 Right, increasing predicate complexity leads to substantial performance drops across all models, with relative degradation ranging from 27.7% to 36.0%. Notably, GPT-4o-1120 suffers the largest relative decline (−36.0%), suggesting that models with weaker baseline performance are disproportionately affected by compositional complexity.

These results reveal that current MLLMs struggle not only with sequential reasoning across layers (as shown in the depth ablation), but also with compositional reasoning within a single predicate. The two dimensions—chain depth and predicate complexity—jointly define the difficulty landscape of MM-CondChain, enabling fine-grained diagnosis of model capabilities.

#### 4.3.3 Summary.

The above ablations reveal two orthogonal axes of difficulty in MM-CondChain: vertical complexity (chain depth) and horizontal complexity (intra-layer predicate composition). Increasing either dimension leads to consistent and substantial performance degradation across all tested models, confirming that both sequential reasoning and compositional reasoning remain fundamental bottlenecks for current MLLMs. Crucially, these two axes are independently controllable in our VPIR-based synthesis pipeline, enabling fine-grained difficulty calibration. This design allows MM-CondChain to serve not only as an evaluation benchmark, but also as a diagnostic tool for pinpointing where and why models fail in visually grounded conditional reasoning.

### 5 Conclusion

In this paper, we introduce MM-CondChain, a benchmark for evaluating visually grounded deep conditional reasoning in MLLMs. Unlike prior benchmarks that test shallow compositions or independent constraints, MM-CondChain requires tracking multi-layer control flow where each decision is gated by a visually verifiable condition. To enable scalable construction with guaranteed correctness, we proposed an agentic synthesis pipeline centered on Verifiable Programmatic Intermediate Representation (VPIR), which decouples logic formation from language rendering and produces benchmark instances with deterministic ground truth and near-isomorphic hard negatives. Experiments across three visual domains and a range of MLLMs reveal that visually grounded conditional reasoning remains a fundamental bottleneck: even state-of-the-art models struggle as chain depth or predicate complexity increases. We believe MM-CondChain will serve as a valuable resource for diagnosing model weaknesses and driving future research toward more robust multimodal reasoning.

### References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Anthropic. System card: Claude opus 4.6. URL https://www-cdn.anthropic.com/4263b940cabb546aa

0e3283f35b686f4f3b2ff47.pdf.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. Mint-cot: Enabling interleaved visual tokens in mathematical chain-of-thought reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Kaustubh Deshpande, Ved Sirdeshmukh, Johannes Baptist Mols, Lifeng Jin, Ed-Yeremai Hernandez-Cardona, Dean Lee, Jeremy Kritz, Willow E Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18632–18702, 2025.

Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Mm-ifengine: Towards multimodal instruction following. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1099–1109, 2025.

Google DeepMind. Gemini 2.5 flash model card. https://storage.googleapis.com/deepmind-media

/Model-Cards/Gemini-2-5-Flash-Model-Card.pdf, a. PDF. Accessed: 2026-03-05.

Google DeepMind. Gemini 2.5 pro model card. https://storage.googleapis.com/deepmind-media/M

odel-Cards/Gemini-2-5-Pro-Model-Card.pdf, b. PDF. Accessed: 2026-03-05.

Google DeepMind. Gemini 3 flash model card. https://deepmind.google/models/model-cards/gemin

i-3-flash/, c. Accessed: 2026-03-05.

###### Google DeepMind. Gemini 3 pro model card. https://deepmind.google/models/model-cards/gemin

i-3-pro/, d. Accessed: 2026-03-05.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Weilei He, Feng Ju, Zhiyuan Fan, Rui Min, Minhao Cheng, and Yi R Fung. Empowering reliable visual-centric instruction following in mllms. arXiv preprint arXiv:2601.03198, 2026.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in neural information processing systems, 36:31096–31116, 2023.

Hang Hua, Yunlong Tang, Ziyun Zeng, Liangliang Cao, Zhengyuan Yang, Hangfeng He, Chenliang Xu, and Jiebo Luo. Mmcomposition: Revisiting the compositionality of pre-trained vision-language models. arXiv preprint arXiv:2410.09733, 2024.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

Yifan Jiang, Jiarui Zhang, Kexuan Sun, Zhivar Sourati, Kian Ahrabian, Kaixin Ma, Filip Ilievski, and Jay Pujara. Marvel: Multidimensional abstraction and reasoning through visual evaluation and learning. In Advances in Neural Information Processing Systems, volume 37, 2024a.

Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, and Xu Yang. d2cache: Accelerating diffusion-based llms via dual adaptive caching. arXiv preprint arXiv:2509.23094, 2025.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. FollowBench: A multi-level fine-grained constraints following benchmark for large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, Bangkok, Thailand, August 2024b. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.257.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

Pengxiang Li, Shilin Yan, Joey Tsai, Renrui Zhang, Ruichuan An, Ziyu Guo, and Xiaowei Gao. Adaptive classifier-free guidance via dynamic low-confidence masking. arXiv preprint arXiv:2505.20199, 2025.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Pan Lu et al. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In

International Conference on Learning Representations, 2024. Ahmed Masry et al. Chartqa: A benchmark for visual question answering on charts. In ACL, 2022. OpenAI. Gpt-5 system card. https://openai.com/index/gpt-5-system-card/. Accessed: 2026-03-05. Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lam-

bert, and Hannaneh Hajishirzi. Generalizing verifiable instruction following. arXiv preprint arXiv:2507.02833, 2025.

Yusu Qian, Hanrong Ye, Jean-Philippe Fauconnier, Peter Grasch, Yinfei Yang, and Zhe Gan. Mia-bench: Towards better instruction following evaluation of multimodal llms. In The Thirteenth International Conference on Learning Representations.

Yusu Qian, Cheng Wan, Chao Jia, Yinfei Yang, Qingyu Zhao, and Zhe Gan. Prism-bench: A benchmark of puzzle-based visual tasks with cot error detection. arXiv preprint arXiv:2510.23594, 2025.

Chenhui Qiang, Zhaoyang Wei, Xumeng Han, Zipeng Wang, Siyao Li, Xiangyuan Lan, Jianbin Jiao, and Zhenjun Han. Ver-bench: Evaluating mllms on reasoning with fine-grained visual evidence. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12698–12705, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?i

###### d=qwen3.5.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36: 59708–59728, 2023.

Hao Shao et al. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In NeurIPS, 2024.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv.org/abs/2507.01006.

Omkar Thawakar, Dinura Dissanayake, Ketan Pravin More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Ilmuz Zaman Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamav-o1: Rethinking step-by-step visual reasoning in llms. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24290–24315, 2025.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality, 2022. URL https://arxiv.org/abs/2204.03162.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxing Xu, et al. Benchmarking complex instruction-following with multiple constraints composition. Advances in Neural Information Processing Systems, 37:137610–137645, 2024.

Xuecheng Wu, Jiaxing Liu, Danlei Huang, Yifan Wang, Yunyun Shi, Kedi Chen, Junxiao Xue, Yang Liu, Chunlin Chen, Hairong Dong, et al. Vic-bench: Benchmarking visual-interleaved chain-of-thought capability in mllms with free-style intermediate state representations. arXiv preprint arXiv:2505.14404, 2025.

Yijia Xiao, Edward Sun, Tianyu Liu, et al. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Runsen Xu et al. Mc-bench: A benchmark for multi-context visual grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025a.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025b.

Shilin Yan, Jiaming Han, Joey Tsai, Hongwei Xue, Rongyao Fang, Lingyi Hong, Ziyu Guo, and Ray Zhang. Crosslmm: Decoupling long video sequences from lmms via dual cross-attention mechanisms. arXiv preprint arXiv:2505.17020, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Chenghao Yang, Yinbo Luo, Zhoufutu Wen, Qi Chu, Tao Gong, Longxiang Liu, Kaiyuan Zhang, Jianpeng Jiao, Ge Zhang, Wenhao Huang, et al. Mars-bench: A multi-turn athletic real-world scenario benchmark for dialogue evaluation. arXiv preprint arXiv:2505.23810, 2025b.

Sihan Yang, Runsen Xu, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. In International Conference on Learning Representations, 2026.

Xuyou Yang, Yucheng Zhao, Wenxuan Zhang, and Immanuel Koh. Space-eval: A benchmark for real-world multi-modal reasoning. In The Fourteenth International Conference on Learning Representations.

Shunyu Yao, Howard Chen, Austin W Hanjie, Runzhe Yang, and Karthik Narasimhan. Collie: Systematic construction of constrained text generation tasks. arXiv preprint arXiv:2307.08689, 2023.

M Yuksekgonul, F Bianchi, P Kalluri, D Jurafsky, J Zou, et al. When and why vision-language models behave like bags-of-words, and what to do about it? In 11th International Conference on Learning Representations, ICLR 2023. International Conference on Learning Representations, ICLR, 2023.

Aimen Zerroug, Mohit Vaishnav, Julien Colin, Sebastian Musslick, and Thomas Serre. A benchmark for compositional visual reasoning. In Advances in Neural Information Processing Systems, volume 35, pages 21551–21565, 2022.

Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5317–5327, 2019.

Jiwen Zhang, Jihao Wu, Teng Yihua, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12016–12031, 2024a.

Qinyan Zhang, Xinping Lei, Ruijie Miao, Yu Fu, Haojie Fan, Le Chang, Jiafan Hou, Dingling Zhang, Zhongfei Hou, Ziqiang Yang, et al. Inverse ifeval: Can llms unlearn stubborn training conventions to follow real instructions? arXiv preprint arXiv:2509.04292, 2025.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024b.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. Vl-checklist: Evaluating pre-trained vision-language models with objects, attributes and relations, 2022a. URL https://arxiv.org/abs/2207.00221.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. An explainable toolbox for evaluating pre-trained vision-language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 30–37, 2022b.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

Tao Zou, Xinghua Zhang, Haiyang Yu, Minzheng Wang, Fei Huang, and Yongbin Li. Eifbench: Extremely complex instruction following benchmark for large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20941–20964, 2025.

