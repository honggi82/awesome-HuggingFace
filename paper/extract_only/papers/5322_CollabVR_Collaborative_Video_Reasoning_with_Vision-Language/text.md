# arXiv:2605.08735v1[cs.CV]9May2026

## CollabVR: Collaborative Video Reasoning with Vision-Language and Video Generation Models

#### Joowon Kim1∗ Seungho Shin2∗ Joonhyung Park1 Eunho Yang1,3† 1KAIST 2Kyung Hee University 3AITRICS

{kjwispro, deepjoon, eunhoy}@kaist.ac.kr ssh9918@khu.ac.kr

[Figure 1]

Figure 1: VLM as planner, VGM as simulator. A VLM is strong at reasoning but weak at visual simulation, while a VGM simulates short clips but lacks reasoning, causing long-horizon drift and mid-clip simulation errors. CollabVR couples them in a closed loop where the VLM plans progressively and diagnoses each generated clip, turning failures into correctable signals.

### Abstract

Recent Thinking with Video approaches use Video Generation Models (VGMs) for visual reasoning by producing temporally coherent Chain-of-Frames as reasoning artifacts. Even strong VGMs, however, exhibit two recurring failure modes on goaldirected tasks: long-horizon drift on multi-step tasks and mid-clip simulation errors that compound. Both stem from the absence of explicit reasoning built upon the VGM’s short-horizon visual prior, a role naturally filled by Vision-Language Models (VLMs), but where to place the VLM is non-trivial: upfront plans commit before any frame is generated and post-hoc critiques over whole videos intervene too late. We propose VLM-VGM Collaborative Video Reasoning (CollabVR), a closedloop framework that couples the VLM with the VGM at step-level granularity: the VLM plans the immediate next action, inspects the clip the VGM generates, and folds the verifier’s diagnosis directly into the next action prompt to repair detected failures. On Gen-ViRe and VBVR-Bench, CollabVR improves both open-source and closed-source VGMs over single-inference, Pass@k, and prior test-time scaling baselines at matched compute, with the largest gains on the hardest tasks. It also yields further improvements on top of a reasoning-fine-tuned VGM, indicating that step-level VLM supervision is orthogonal to and stackable with reasoning-oriented fine-tuning. We provide video samples and additional qualitative results at our project page: https://joow0n-kim.github.io/collabvr-project-page.

∗Equal contribution. †Corresponding author.

### 1 Introduction

Recent progress in visual reasoning has largely centered on the Thinking with Images paradigm, in which VisionLanguage Models (VLMs) reason through visual intermediate steps rather than purely textual ones [29, 11, 23, 15]. While promising, static images impose fundamental physical and cognitive limits on reasoning about dynamic processes. A single frame captures one instant in time, and is therefore inherently constrained in representing temporal evolution, causal sequences, or physical interactions that unfold over time.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

To overcome these static limitations, a new paradigm has emerged: Thinking with Video. Departing from symbolic, text-based Chain-of-Thought, this line of work leverages Video Generation Models (VGMs) to produce a Chain-ofFrames: a form of generative visual reasoning in which the reasoning trajectory is realized as a temporally coherent video rather than a token sequence. Alongside this shift, a growing suite of benchmarks now evaluates the zero-shot reasoning capabilities of VGMs [9, 20, 4, 21, 32, 36]. Recent studies report that VGMs exhibit promising reasoning behavior within short sequences, covering physical dynamics, spatial consistency, fine-grained visual detail, and geometric tracing, in some cases matching or surpassing VLMs on targeted video reasoning benchmarks [34, 30].

Figure 2: Performance–Cost tradeoff on Gen-ViRe [20]. Pass@k resampling plateaus quickly with cost and VideoTPO [4] trades extra budget for modest improvement, while CollabVR reaches markedly higher score at lower budget on both models [32, 7].

VLMs and VGMs nonetheless exhibit complementary strengths and weaknesses (Figure 1). VLMs excel at logical reasoning: they decompose complex problems, formulate multi-step plans, and draw abstract inferences with high fidelity, but their capacity to directly perceive and simulate the visual world remains limited. VGMs, conversely, are strong at perceiving visual detail, preserving physical coherence, and performing short-horizon simulation, yet weak at abstract logical reasoning and longrange causal consistency. This disparity in VGMs aligns with a broader pattern in recent evaluations: modern VGMs are trained to optimize perceptual quality, not task-level reasoning correctness [9, 3].

As a direct consequence, VGMs exhibit two recurring failure modes in goal-directed generative visual reasoning. (i) Overloaded-prompt failure: when a single prompt specifies a long-horizon task, the VGM collapses it into one short-horizon rollout that deviates from the intended trajectory, since it lacks the planning capacity to decompose the task into coherent sub-goals. (ii) Execution failure: even within a single short clip, the VGM commits localized errors mid-clip (e.g., an agent crossing a wall, an object losing its identity after an interaction, or a sub-action stopping before completion) that, once introduced, propagate through subsequent frames and contaminate the entire trajectory. Both failures can be traced to a common root cause: the absence of an explicit, corrigible reasoning process, on top of the VGM’s strong yet short-horizon visual prior.

In light of this, the VGM needs a reasoning supervisor that can plan beyond a single short clip and verify what was just produced, capabilities at which VLMs are already strong. A straightforward way to leverage VLMs is to select the most perceptually plausible one among k samples, as in existing test-time scaling approaches [19, 10, 5]. On video reasoning tasks, however, valid outputs are task-specific and tightly defined, unlike general text-to-video which admits a wide range of realizations. More importantly, the correct trajectory often lies outside the generator’s distribution and thus is difficult to reach by simply drawing more plausible samples (Figure 2). Therefore, in video reasoning, VLM supervision has to serve as a natural complement to the VGM’s short-horizon visual prior, most effectively when applied to progressively construct the correct trajectory through the task.

We propose VLM-VGM Collaborative Video Reasoning (CollabVR), a closed-loop framework that couples the VLM with the VGM at step-level granularity. The VLM plans only the immediate next action, inspects the clip the VGM produces, and decides on the spot how to proceed, matching the intervention to the diagnosed state rather than applying one fixed recovery across whole videos. This

step-level coupling catches each failure at the moment it occurs, before it contaminates an entire trajectory. CollabVR consists of two tightly coupled modules, each targeting one of the failures above.

VLM-Driven Progressive Planning addresses (i) by letting the VLM adaptively decide the step count and plan only the immediate next action conditioned on previously generated frames, mitigating longhorizon drift without a fixed upfront decomposition; VLM-VGM Collaborative Reasoning addresses

- (ii) by having the VLM verify each clip, diagnose the failure, and revise the next action prompt to repair local errors before they compound. Although the VLM is not infallible, the step-level design contains any single error to one clip, and our human-annotated reliability benchmark (Section 4.4) confirms that VLM-predicted failure localization and step counts are reliable at the granularity the framework requires. We summarize our contributions as follows.

- • Progressive Planning against long-horizon drift. An adaptive planning module where the VLM decides the step count on the fly and emits only the immediate next action, conditioned on previously generated frames.
- • Collaborative Reasoning against execution failure. A failure-aware intervention module where the VLM verifies each VGM clip and folds the diagnosed failure back into the next action prompt for repair.
- • Consistent gains at matched compute. CollabVR improves both open- and closed-source VGMs over single-inference, Pass@k, and VideoTPO on Gen-ViRe and VBVR-Bench, with further gains on reasoning-fine-tuned VGMs [32]. A human-annotated benchmark confirms that VLM-predicted task complexity and failure localization align with expert judgments.

### 2 Related Work

#### 2.1 Thinking with Video: Video Reasoning

Using generated or retrieved perceptual artifacts as intermediate reasoning steps originated in the Thinking with Images line of work on VLMs [29, 11, 23, 15, 28, 8], where sketches, diagrams, and sub-images scaffold multi-step visual inference, but static images cannot capture dynamic processes or causal unfolding over time. The emergence of high-fidelity Video Generation Models such as Sora [1], Veo [7], and Wan [31], among many others [25, 40, 24, 38], gives rise to the Thinking with Video paradigm [9, 34, 30], in which a generated video itself serves as the reasoning artifact: a Chain-of-Frames whose temporal trajectory embodies the solution. A rapidly growing body of benchmarks [9, 30, 21, 36, 20, 4, 32, 17, 3] probes this regime, consistently demonstrating that modern VGMs excel at short-horizon visual simulation but remain weak at long-horizon planning, strict geometric and logical constraints, global state consistency, and process-level faithfulness, the diagnosis that motivates our framework.

#### 2.2 Test-Time Scaling for Video Generation

Test-time scaling (TTS) for Large Language Models [27, 2] and diffusion models [22] substantially improves output quality with additional inference compute, and video-specific extensions [19, 10, 5, 16, 13] apply this to the temporal axis through frame-level search, evolutionary sampling, and selfrefinement. These methods optimize visual quality rather than task correctness, but reasoning failures are systematic (wrong solution paths, skipped sub-goals, incorrect physical outcomes) and cannot be averaged out by sampling more, requiring instead diagnosis and repair by a VLM. Among reasoningtargeting approaches, VideoTPO [4] uses LLM critique to iteratively re-prompt the generator, but its single mechanism of whole-video prompt refinement leaves task-decomposition failures unaddressed. We instead operate at sub-action granularity, with the VLM verifying each clip and folding its diagnosis directly into the next action prompt for repair.

#### 2.3 Iterative Refinement and VLM-Guided Generation

Iterative refinement and LLM/VLM-guided generation cast an LLM as verifier or planner in a closed loop with the generator, originating in the image domain [14, 37] and extending to video [18, 39, 35,

- Module 1 : VLM-Driven Progressive Planning

(VLM decides step count N and plans each action on-the-fly)

Failure Diagnosis : Wall crossed at τ.

- Module 2 : VLM-VGM Collaborative Reasoning (VLM refines prompt on failure, up to M retries)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

…

Step 1 Step 2

Step N

[Figure 8]

[Figure 9]

[Figure 10]

###### VLM VGM

[Figure 11]

1 2

N

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Retry 1 / M

Output (𝑽)

###### Input

Evolve

Plan

Plan

Plan

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Move right to the first corner.

Move down until you reach the wall.

Move down slowly, avoid walls.

Reach the green square

𝑪𝟏 𝑪𝟐 𝑪𝑵

|[Figure 32]|
|---|

[Figure 33]

|[Figure 34]|
|---|

VLM interprets task.

[Figure 35]

[Figure 36]

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

###### Image

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

|[Figure 52]<br><br>|
|---|

τ

Move the red circle to the

Generate

Generate

Generate𝑪𝟐

Generate

𝑪𝟏

𝑪𝟐

𝑪𝑵

|[Figure 53]<br><br>|
|---|

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

green square.

|[Figure 58]<br><br>|
|---|

|[Figure 59]<br><br>|
|---|

|[Figure 60]<br><br>|
|---|

|[Figure 61]|
|---|

|[Figure 62]<br><br>|
|---|

|[Figure 63]<br><br>|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Prompt

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Verify .

Verify .

Verify .

Verify .

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Figure 3: Overall pipeline of CollabVR. A persistent VLM plans one action at a time and, after observing each generated clip, decides whether to accept, re-generate, or re-plan. Module 1 adaptively determines the step count, and Module 2 verifies each clip and folds the verifier’s diagnosis into the next action prompt to repair the failure.

33, 12], but these systems optimize visual or physical quality, treat the video as an indivisible unit, and lack a mechanism to diagnose specific failures or localize corrections. CollabVR closes these gaps in a single training-free loop that plans progressively, verifies and recovers at each step based on an explicit failure diagnosis, and works with any off-the-shelf VGM.

### 3 CollabVR: Closed-Loop Step-Level Video Reasoning

We present CollabVR, a closed-loop test-time framework that treats video reasoning as a construction problem: the correct trajectory is assembled stepwise through VLM planning and VGM execution, rather than sampled from the generator’s output distribution. Figure 3 illustrates the overall pipeline, whose two core modules target the two failure modes identified in Section 1. VLM-Driven Progressive Planning

- (Section 3.2) addresses overloaded-prompt failure by letting the VLM plan adaptively, one step at a time. VLM-VGM Collaborative Reasoning (Section 3.3) addresses execution failure by letting the VLM verify each clip and revise the action prompt with the diagnosis.

The reliability of this VLM-as-supervisor design (whether plan-depth, verification, and prompt evolution genuinely align with human judgment rather than amplifying VLM hallucinations) is empirically validated in Section 4.4 on a dedicated human-annotated benchmark.

##### Algorithm 1 CollabVR (I0,q,Nmax,M).

Require: input image I0, task prompt q; max planning steps

Nmax; per-step attempt budget M Ensure: generated video V

- 1: H ← ∅, f ← I0
- 2: for t = 1, . . . , Nmax do
- 3: at ← πplan(I0, q, H) {(Section 3.2)}
- 4: for j = 1, . . . , M do
- 5: ct ← g(f, at)
- 6: (v, d) ← πverify(I0, q, H, ct) {(Section 3.3)}
- 7: if v = accept then
- 8: H ← H ∪ {ct}
- 9: f ← last frame of ct
- 10: if task complete then
- 11: return V = c1 ⊕ · · · ⊕ ct
- 12: end if
- 13: break
- 14: else
- 15: at ← evolve(at, d) {prompt evolution}
- 16: end if
- 17: end for
- 18: end for
- 19: return V = c1 ⊕ · · · ⊕ c|H|

#### 3.1 Problem Formulation

A video reasoning task is specified by an input image I0 and a task prompt q that describes the desired visual process or transformation, and the goal is to produce a video V whose trajectory realizes the

reasoning demanded by (I0,q). Our framework has two actors: a VLM-based planner/verifier π (queried in two roles, πplan and πverify) and an image-to-video generator g that maps a conditioning frame f and an action prompt at to a short clip ct. Throughout the loop we maintain f as the latest conditioning frame (initially I0) and H as the history of accepted clips. The full closed loop is given in Algorithm 1, and the output is the concatenation of accepted clips V = c1 ⊕ ··· ⊕ cN; the rest of

Plan 1

Plan 2

Plan N

Pre-Planning

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

- Plan 1
- Plan 2

[Figure 95]

[Figure 96]

| | |
|---|---|
| | |

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

| | |
|---|---|
| | |

[Figure 105]

…

[Figure 106]

[Figure 107]

VGM

VGM

###### VGM

[Figure 108]

Input Image

[Figure 109]

[Figure 110]

…

[Figure 111]

VLM

Plan N

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Last Frame

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Progressive Planning (Ours)

Plan 1

[Figure 128]

[Figure 129]

VLM Last Plan

VLM

[Figure 130]

[Figure 131]

Next Plan

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Plan 1

[Figure 138]

[Figure 139]

| | |
|---|---|
| | |

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

| | |
|---|---|
| | |

[Figure 148]

??

###### VGM …

VGM

VGM

[Figure 149]

[Figure 150]

[Figure 151]

Input Image

[Figure 152]

[Figure 153]

…

[Figure 154]

VLM

??

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Last Frame

(a) Pre-planning vs. Progressive planning pipeline.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | || |
|---|
| | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

(b) Performance–cost on Gen-ViRe.

- Figure 4: Pre-planning vs. Progressive planning on Gen-ViRe with VBVR-Wan2.2 (Module 1 only). Progressive planning achieves a +13% relative gain over pre-planning at matched cost.

this section motivates its two design choices, progressive planning (Section 3.2) and collaborative reasoning (Section 3.3).

#### 3.2 VLM-Driven Progressive Planning

A naive extension of Chain-of-Thought to video is pre-planning [12] (Figure 4a, top): the VLM decomposes q into N milestone prompts upfront, and the VGM sequentially generates a clip per milestone. This reduces drift on long-horizon tasks but commits the plan before any VGM output exists, so it cannot adapt to the realized generation, and N itself is difficult to fix from the prompt alone. We therefore adopt progressive planning (Figure 4a, bottom; line 4 of Algorithm 1): the VLM plans only the immediate next action and inspects the realized clip before deciding whether to continue, so both subsequent steps and N adapt to what the generator actually produces, with N capped at a hyperparameter Nmax. At matched generation budgets, this yields a substantially better performance–cost trade-off than pre-planning (Figure 4b).

#### 3.3 VLM-VGM Collaborative Reasoning

For each generated clip ct, the VLM verifier πverify produces a structured judgment (v,d), where v ∈ {accept,reject} and d packages a textual reason and an actionable suggestion for repair (line 6 of Algorithm 1). The verifier judges whether the planned action was executed, flagging wrong direction, wrong target, or scene collapse but not partial-but-correct progress, which is the planner’s concern. This single-clip judgment stays within VLMs’ strong perception regime, so d is specific enough to be actionable. The simplest recovery folds d’s suggestion (e.g., “circle the red square, not the yellow”) back into the action prompt, at ← evolve(at,d), and re-samples the VGM with the plan fixed, reusing the verifier’s output without an extra VLM call.

### 4 Experiments

We evaluate CollabVR on Gen-ViRe and VBVR-Bench, two complementary video reasoning benchmarks. The section presents the experimental setup (Section 4.1), main accuracy comparisons under matched compute (Section 4.2), per-module ablations (Section 4.3), and an analysis covering percategory module effectiveness and the human-annotated reliability of the VLM-as-supervisor design

- (Section 4.4). Complementary evidence in the appendix includes a blind user study (Appendix B.1) and a cost decomposition validating the matched-compute framing (Appendix C.3).

#### 4.1 Implementation Details

Benchmarks and evaluation. We evaluate CollabVR on two complementary video-reasoning benchmarks. Gen-ViRe [20] contains 72 samples across 6 categories and uses a rubric-based VLM judge (Gemini 2.5 Pro) that scores each generated video against per-task criteria targeting task correctness rather than visual quality.3 VBVR-Bench [32] focuses on synthetic visual reasoning tasks

3We report Gen-ViRe scores averaged over three runs to account for judge stochasticity.

- Table 1: Benchmarking results on Gen-ViRe. All scores are 3-run VLM evaluation averages (Gemini 2.5 Pro) for stability. VGM Cost is the average total VGM generation seconds per sample (all steps and re-generations included). Higher is better. Bold: best within each model; Underlined: second best.

Category Method

VGM Cost (s) Avg. Abst. Algo. Analog. Perc. Plan. Spat.

| | | | |
|---|---|---|---|
|Open-source Video Models VBVR-Wan2.2 VBVR-Wan2.2 + Pass@2 VBVR-Wan2.2 + Pass@4 VBVR-Wan2.2 + VideoTPO [4] VBVR-Wan2.2 + CollabVR<br><br>|6.0 12.0 24.0 30.0 17.8<br><br>|0.391 0.398 0.438 0.488 0.531<br><br>|0.479 0.415 0.250 0.261 0.554 0.387 0.576 0.437 0.278 0.257 0.481 0.357 0.622 0.418 0.250 0.275 0.604 0.462 0.535 0.443 0.417 0.313 0.671 0.552 0.569 0.606 0.333 0.367 0.821 0.488<br><br>|
|Closed-source Video Models Veo 3.1 Veo 3.1 + Pass@2 Veo 3.1 + Pass@4 Veo 3.1 + CollabVR|8.0 16.0 32.0 21.4<br><br>|0.481 0.491 0.509 0.550<br><br>|0.420 0.512 0.361 0.274 0.744 0.573 0.458 0.587 0.389 0.242 0.721 0.571 0.425 0.573 0.417 0.296 0.726 0.646 0.434 0.641 0.472 0.325 0.768 0.657<br><br>|

- Table 2: Benchmarking results on VBVR-Bench. Overall In-Domain (ID) and Out-of-Domain (OOD) scores are reported alongside category-wise performance. Higher is better. Bold: best in group; Underlined: second best.

In-Domain by Category Out-of-Domain by Category Models

VGM Cost (s) Overall Avg. Abst. Know. Perc. Spat. Trans. Avg. Abst. Know. Perc. Spat. Trans.

VBVR-Wan2.2 3.70 0.671 0.762 0.701 0.746 0.802 0.793 0.803 0.577 0.674 0.674 0.503 0.528 0.633 VBVR-Wan2.2 + Pass@2 7.40 0.694 0.783 0.791 0.742 0.795 0.774 0.812 0.602 0.728 0.617 0.494 0.532 0.701 VBVR-Wan2.2 + Pass@4 14.80 0.707 0.789 0.751 0.734 0.826 0.805 0.841 0.622 0.785 0.660 0.535 0.577 0.683 VBVR-Wan2.2 + VideoTPO [4] 11.10 0.650 0.717 0.723 0.698 0.641 0.744 0.816 0.582 0.767 0.619 0.513 0.540 0.572 VBVR-Wan2.2 + CollabVR 10.91 0.757 0.819 0.828 0.784 0.805 0.828 0.852 0.696 0.884 0.634 0.641 0.608 0.720 Cosmos-Predict2.5 3.70 0.308 0.312 0.272 0.327 0.355 0.227 0.390 0.304 0.368 0.169 0.309 0.377 0.274 Cosmos-Predict2.5 + CollabVR 10.91 0.403 0.406 0.404 0.431 0.411 0.301 0.482 0.400 0.481 0.286 0.400 0.471 0.346

with controlled ground-truth references. It consists of a 500-sample test set spanning 5 reasoning categories with In-Domain and Out-of-Domain splits, and adopts a rule-based, judge-free protocol for comparing generated videos against ground-truth references.

Video generation models. We apply CollabVR to two main VGMs: the closed-source API model4 Veo 3.1 [7], and VBVR-Wan2.2 [31, 32], a 14B open-source image-to-video model that is the strongest open baseline on VBVR-Bench. We additionally report on Cosmos-Predict-2.5 [25] as a second open-source VGM.

Baselines. We compare CollabVR against three baselines: (i) Single Inference, one-shot generation from (I0,q); (ii) Pass@k, k ∈ {2,4} independent generations from which a VLM selects the one that best achieves the task; and (iii) VideoTPO [4], a TTS-for-video-reasoning baseline that iteratively rewrites prompts based on full-video critiques. Both (ii) and (iii) use Gemini 2.5 Pro, the same model used as our step-level verifier; full configuration is in Appendix A.1. Since VLM compute is negligible relative to VGM compute, we report Cost throughout as the total seconds of video generated by the VGM per sample.

Our framework. We use Gemini 2.5 Pro as the default VLM for both planning and verification, and evaluate alternative VLMs in Section 4.3. Unless stated otherwise, we set Nmax=3 and the per-step attempt budget to M=3; the resulting total generation budget is comparable to Pass@2–Pass@4 for a single VGM.

#### 4.2 Main Results

Tables 1 and 2 report our main results on Gen-ViRe and VBVR-Bench. On Gen-ViRe, CollabVR delivers consistent improvements over single-inference on both the open-source VBVR-Wan2.2 (Pass@1 0.391 → 0.531) and the closed-source Veo 3.1 (Pass@1 0.481 → 0.550) (Table 1), with the largest margins over baselines on Planning and Algorithmic categories where complex long-horizon reasoning is required. On VBVR-Bench, CollabVR consistently surpasses baselines on both opensource models, VBVR-Wan2.2 and Cosmos-Predict-2.5 (Table 2), with gains most pronounced on categories requiring multi-step spatial or transformation reasoning. Compared to VideoTPO and

4We exclude Sora 2 from the main results due to insufficient first-frame fidelity for step-by-step clip concatenation, and defer its analysis to Appendix A.4.

“Starting from the green cell, the agent moves one step at a time in four directions on the grid.

It must follow the shortest path that visits all blue blocks before reaching the red goal cell.”

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

VBVR-Wan2.2

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

+CollabVR

“A person opened a can of food. (Fixed camera angle)”

VBVR-Wan2.2

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

+CollabVR

First Frame (Input Image)

Last Frame

- Figure 5: Qualitative comparison on various visual reasoning tasks from Gen-ViRe and VBVR-Bench. We compare VBVR-Wan2.2 with CollabVR on diverse tasks that require step-by-step reasoning.

Pass@k, CollabVR achieves higher accuracy at lower per-sample generation cost, supporting our claim that adaptive progressive planning coupled with failure-aware recovery is a more effective test-time scaling axis than full-video resampling. Notably, CollabVR also yields further gains on top of VBVR-Wan2.2, a VGM already fine-tuned on reasoning data, demonstrating that test-time reasoning supervision is orthogonal to and stacks on reasoning-oriented fine-tuning.

Table 3: Per-module ablation on GenViRe and VBVR-Bench. ∆: gain over the baseline (no module).

Qualitative results. Figure 5 shows representative examples where CollabVR contributes over the baseline. On a long-horizon task (top), VBVR-Wan2.2 alone cannot solve in a single shot, whereas CollabVR succeeds by decomposing the plan to visit the goal cells starting from the right. The framework also extends to real-world scenarios (bottom): for “opened a can of food”, VBVRWan2.2 alone bypasses the can opener and pries the lid by hand, whereas CollabVR’s planner emits explicit tool-use sub-actions and the VGM produces a faithful execution. Beyond automated metrics, we further confirm via a user study that human annotators prefer CollabVR’s outputs (73.8%) over Pass@4 (19.7%) and Pass@1 (6.5%) on a blind side-by-side comparison (Appendix B.1).

M1 M2 Cost (s) Overall ∆ Gen-ViRe

- ✗ ✗ 6.0 0.391 –

✓ ✗ 10.9 0.511 +0.120

- ✗ ✓ 9.9 0.436 +0.045

✓ ✓ 17.8 0.531 +0.140 VBVR-Bench

- ✗ ✗ 3.70 0.671 –

✓ ✗ 6.19 0.706 +0.035

- ✗ ✓ 6.03 0.734 +0.063

###### ✓ ✓ 10.91 0.757 +0.086

Gen-ViRe

VBVR-Bench

#### 4.3 Ablation Study

17%

21%

46%

Module 1 vs. Module 2 vs. Combined. We compare three configurations: (M1) progressive planning only (no verification or recovery), (M2) verification and failureaware recovery only (Nmax=1, no progressive planning), and (M1+M2) the full pipeline. Figure 6 shows the humanannotated distribution of step counts N for the two benchmarks; this distribution reflects each benchmark’s mix of task complexity and type, with Gen-ViRe dominated by multi-step tasks and VBVR-Bench dominated by single-step tasks. Table 3 reports the corresponding per-module gains: on Gen-ViRe, M1’s progressive planning is the larger contributor (+0.120 vs. +0.045 for M2), whereas on VBVR-Bench,

54%

29%

33%

N = 1

N = 2

N = 3

| |
|---|

| |
|---|

| |
|---|

Figure 6: Human-annotated distribution of step counts N for the benchmarks.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Nmax

Figure 7: Effect of maximum planning steps Nmax on Gen-ViRe.

Table 4: Comparison of test-time scaling methods and VLM choices across benchmarks.

Method VLM Gen-ViRe VBVR-Bench

- Pass@1 (baseline) – 0.391 0.671
- Pass@2 Gemini 2.5 Pro 0.398 0.694 Pass@4 Gemini 2.5 Pro 0.438 0.707 VideoTPO [4] Gemini 2.5 Pro 0.488 0.650

Qwen3.5-9B 0.514 0.710 Qwen3.5-27B 0.510 0.717 Gemini 2.5 Pro 0.531 0.757

CollabVR

M2 is the larger contributor (+0.063 vs. +0.035 for M1). On Gen-ViRe, CollabVR frequently improves performance by adaptively decomposing complex tasks into sub-steps the VGM can satisfy individually. On VBVR-Bench, M2 corrects single-clip execution failures without unnecessarily splitting tasks that are already solvable in a single action. That the dominant module shifts with the benchmark’s N profile indicates the framework adapts to task character rather than relying on a single fixed mechanism.

Effect of step count Nmax. We sweep the planning step count Nmax ∈ {1,...,5} on Gen-ViRe with VBVR-Wan2.2 and observe a non-monotonic relationship: scores rise as Nmax increases up to the level required by the task, then plateau or degrade as further splitting introduces step-boundary artifacts on already-simple sub-actions. This empirically motivates the adaptive N selection in Section 3.2.

VLM choice. We replace the default Gemini 2.5 Pro planner/verifier with open-source alternatives (Qwen3.5-27B, Qwen3.5-9B [26]) and report the final task accuracy on each benchmark (Table 4). Performance degrades gracefully with weaker VLMs, and notably even the smallest model we test, Qwen3.5-9B paired with CollabVR, surpasses every Pass@k and VideoTPO baseline that uses the proprietary Gemini 2.5 Pro on both benchmarks, confirming that the framework is not tied to a single proprietary model. We further examine in Section 4.4 how these end-to-end gaps are consistent with each VLM’s per-axis supervisor quality (plan-depth, verification, evolution) on the human-annotated benchmark.

- 4.4 Analysis

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

[Figure 191]

Category-wise module effectiveness and limitation. CollabVR does not behave uniformly across reasoning types. Section 4.3 attributed module dominance to a benchmark’s N profile. Here we examine the finer-grained role of each reasoning category (Figure 8). For instance, a Planning prompt such as “use the kettle and teabag to make a cup of tea in the mug” packs a chain of distinct physical actions into a deceptively short instruction; a single-shot VGM compresses these into one ambiguous clip and typically resolves only the first action, whereas the progressive planner exposes the chain as separate sub-goals that the VGM can satisfy individually (M1 alone +0.165). Conversely, an Analogy prompt such as “generate the missing object in the lower right region and solve the visual analogy” is a single atomic transformation: verifier-driven re-sampling alone is sufficient (M2 alone +0.139). Crucially, the full M1+M2 pipeline yields a positive gain in every category (+0.083 to +0.267), and only the combination meaningfully improves long-horizon Spatial tasks, indicating decomposition and recovery are complementary

Figure 8: Per-category ∆ over Pass@1 on GenViRe (VBVR-Wan2.2). Module configurations follow Section 4.3.

Yet the framework cannot manufacture capabilities the VGM lacks: the smallest gains concentrate on categories whose target transformations are symbolic rather than physical (Analogy ∆+0.083, Abstract ∆+0.090), where decomposition has nothing to decompose into and verifier-driven resampling can only redraw from the VGM’s existing distribution. On Analogy in particular, M1+M2 even underperforms either module alone (both +0.139), since forcing decomposition on an atomic transformation yields contrived intermediates that the verifier then rejects. This residual gap is

| |
|---|

###### N

- Figure 9: Human-annotated analysis of planning, verification, and evolution. (a) Distribution of human-annotated step counts N. (b) Plan-depth match per VLM: exact-match accuracy (left axis) and mean absolute error (MAE) (right axis). (c) Verification agreement: F1 score on a balanced 1:1 split. (d) Evolution quality: mean human rating on a three-point scale.

orthogonal to test-time orchestration, and we view reasoning-oriented VGM training (e.g., physicsaware fine-tuning, symbolic-transformation pretraining) as a complementary future direction.

Human-annotated benchmark and VLM supervision. To externally check the VLM-assupervisor assumption underlying the closed loop, we construct a human-annotated benchmark on VBVR-Wan2.2 output videos, with each item annotated along three axes that mirror the framework’s decision points. (i) Plan depth. From the input image and prompt alone, annotators decide the appropriate step count N for solving the task, which reflects the task’s reasoning complexity, compared against the VLM’s adaptive N from Module 1. (ii) Clip-level verification. For 250 clips drawn at the verifier’s decision points, annotators independently judge whether the clip realizes the intended sub-action; this is compared against the VLM verifier’s accept/reject output from Module 2.

- (iii) Evolution scoring. For 240 rejected clips, annotators rate the suitability of the verifier’s suggested repair (used by prompt evolution) on a three-point scale (1: poor, 2: moderate, 3: well-suited).

For reliability and fairness, the accept/reject ratio is held to 1:1 (preventing class-imbalance inflation of verifier accuracy) and the N distribution is balanced across reasoning categories. Each item is independently annotated by AI experts from two different affiliations, with disagreements resolved through cross-validation, providing a non-trivial inter-annotator baseline against which VLM agreement can be calibrated.

With Gemini 2.5 Pro [6] as the planner/verifier, the VLM aligns most closely with human annotators on all three axes (Figure 9), supporting its role as the default planner/verifier in our pipeline. The same axes are used in the VLM ablation (Section 4.3) to compare alternative open-source VLMs (Qwen3.527B, Qwen3.5-9B [26]) along an axis decoupled from final task accuracy. Detailed statistics, sample annotations, and per-VLM breakdowns are deferred to the Appendix.

### 5 Conclusion

We presented VLM-VGM Collaborative Video Reasoning (CollabVR), a closed-loop framework that pairs a VLM with a video generation model at step-level granularity: the VLM plans one sub-action at a time, inspects the clip the VGM produces, and adaptively chooses among accepting, regenerating, or further decomposing the action. This step-level coupling redirects test-time compute from sampling more videos toward refining the one being constructed, and it consistently improves both open-source (VBVR-Wan2.2) and closed-source (Veo 3.1) generators over single-inference, Pass@k, and prior test-time scaling baselines on Gen-ViRe and VBVR-Bench. A human-annotated benchmark further confirms that the VLM’s plan-depth, verification, and evolution decisions align with expert annotators, supporting the use of a single VLM as an end-to-end supervisor for the loop.

Limitations. Test-time orchestration cannot overcome a VGM that lacks the underlying capability: abstract or symbolic transformations stay hard because the generator never approximates them (Section 4.4), and our gains diminish on lower-capability VGMs whose weak per-step instructionfollowing compounds errors across sub-clips faster than re-generation can repair (Section B.6). The verifier is also imperfect, allowing a fraction of failed clips to propagate downstream (Section 4.4). Future directions include reasoning-oriented VGM training and finer-grained failure localization, which can be orthogonally integrated into our test-time loop.

### References

- [1] Tim Brooks, Bill Peebles, et al. Video generation models as world simulators. https:// openai.com/index/video-generation-models-as-world-simulators/, 2024. OpenAI Technical Report.
- [2] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.
- [3] Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, et al. MMGR: Multi-modal generative reasoning. arXiv preprint arXiv:2512.14691, 2025.
- [4] Harold Haodong Chen, Disen Lan, Wen-Jie Shu, Qingyang Liu, Zihan Wang, et al. TiViBench: Benchmarking think-in-video reasoning for video generative models. In Computer Vision and Pattern Recognition (CVPR), 2026.
- [5] Wenyan Cong, Hanqing Zhu, Peihao Wang, et al. Can test-time scaling improve world foundation model? In Conference on Language Modeling (COLM), 2025.
- [6] Gemini Team, Google DeepMind. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [7] Google DeepMind. Veo 3.1. Technical report, Google DeepMind, January

2026. URL https://blog.google/innovation-and-ai/technology/ai/ veo-3-1-ingredients-to-video/. Released January 13, 2026.

- [8] Jiawei Gu, Yunzhuo Hao, Huichen Will Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. ThinkMorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.
- [9] Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, et al. Are video models ready as zero-shot reasoners? an empirical study with the MME-CoF benchmark. arXiv preprint arXiv:2510.26802, 2025.
- [10] Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Ling Pan. Scaling image and video generation via test-time evolutionary search. arXiv preprint arXiv:2505.17618, 2025.
- [11] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [12] Ziqi Huang, Ning Yu, Gordon Chen, et al. VChain: Chain-of-visual-thought for reasoning in video generation. arXiv preprint arXiv:2510.05094, 2025.
- [13] Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Saining Xie, Jaehong Yoon, and Sung Ju Hwang. Self-refining video sampling. arXiv preprint arXiv:2601.18577, 2026.
- [14] Tsung-Wei Ke, Fahim Tajwar, et al. Self-correcting LLM-controlled diffusion models. In Computer Vision and Pattern Recognition (CVPR), 2024.
- [15] Chengzu Li, Wenshan Wu, Huanyu Zhang, et al. Imagine while reasoning in space: Multimodal visualization-of-thought. In International Conference on Machine Learning (ICML), 2025.
- [16] Chengzu Li, Zanyi Wang, Jiaang Li, Yi Xu, Han Zhou, et al. Thinking in frames: How visual context and test-time scaling empower video reasoning. arXiv preprint arXiv:2601.21037, 2026.
- [17] Yifan Li, Yukai Gu, Yingqian Min, Zikang Liu, Yifan Du, Kun Zhou, Min Yang, Wayne Xin Zhao, and Minghui Qiu. Beyond the last frame: Process-aware evaluation for generative video reasoning. arXiv preprint arXiv:2512.24952, 2026.

- [18] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. VideoDirectorGPT: Consistent multiscene video generation via LLM-guided planning. In Conference on Language Modeling (COLM), 2024.
- [19] Fangfu Liu, Hanyang Wang, Yimo Cai, Kaiyan Zhang, Xiaohang Zhan, and Yueqi Duan. VideoT1: Test-time scaling for video generation. In International Conference on Computer Vision (ICCV), 2025.
- [20] Xinxin Liu, Zhaopan Xu, Ming Li, Kai Wang, Yong Jae Lee, and Yuzhang Shang. Can world simulators reason? Gen-ViRe: A generative visual reasoning benchmark. arXiv preprint arXiv:2511.13853, 2025.
- [21] Yang Luo, Xuanlei Zhao, Baijiong Lin, Lingting Zhu, Liyao Tang, Yuqi Liu, Ying-Cong Chen, Shengju Qian, Xin Wang, and Yang You. V-ReasonBench: Toward unified reasoning benchmark suite for video generation models. arXiv preprint arXiv:2511.16668, 2025.
- [22] Nanye Ma, Shangyuan Tong, Haolin Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. In Computer Vision and Pattern Recognition (CVPR), 2025.
- [23] Sachit Menon, Richard Zemel, and Carl Vondrick. Whiteboard-of-thought: Thinking stepby-step across modalities. In Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [24] Meta AI. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- [25] NVIDIA Cosmos Team. World simulation with video foundation models for physical ai, 2025. URL https://arxiv.org/abs/2511.00062.
- [26] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.
- [27] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
- [28] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, et al. OpenThinkIMG: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025.
- [29] Zhaochen Su, Peng Xia, Hangyu Guo, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.
- [30] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. In Computer Vision and Pattern Recognition (CVPR), 2026.
- [31] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [32] Maijunxian Wang, Ruisi Wang, Juyi Lin, et al. A very big video reasoning suite. arXiv preprint arXiv:2602.20159, 2026.
- [33] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. VideoAgent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision (ECCV), 2024.
- [34] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.
- [35] Qiyao Xue, Xiangyu Yin, Boyuan Yang, et al. PhyT2V: LLM-guided iterative self-refinement for physics-grounded text-to-video generation. In Computer Vision and Pattern Recognition (CVPR), 2025.

- [36] Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, et al. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks. arXiv preprint arXiv:2511.15065, 2025.
- [37] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal LLMs. In International Conference on Machine Learning (ICML), 2024.
- [38] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, et al. UniSim: Learning interactive real-world simulators. In International Conference on Learning Representations (ICLR), 2024.
- [39] Xindi Yang, Baolu Li, Yiming Zhang, et al. VLIPP: Towards physically plausible video generation with vision and language informed physical prior. In International Conference on Computer Vision (ICCV), 2025.
- [40] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations (ICLR), 2025.

## Appendix

### A Implementation Details

#### A.1 Hyperparameters

Pipeline budget. We set Nmax=3 planning steps and M=3 per-step generation attempts as the default CollabVR configuration. The step-level verifier receives one frame per second uniformly sampled from each candidate clip. For Gemini 2.5 Pro we instead pass the raw video as input and let the model perform its own frame sampling internally.

Vision-Language Models (VLMs). All VLM calls (planner, step-level verifier, and failure router) use Gemini 2.5 Pro (gemini-2.5-pro) with decoding temperature 0.2. The same model identity also serves as the output reward model for Pass@k selection and as the critic for the VideoTPO baseline, keeping comparisons matched in VLM compute. Prompt templates are listed in Appendix A.2.

Video Generation Models (VGMs). For VBVR-Wan2.2, we use the released 14B image-to-video checkpoint (Wan2.2 fine-tuned on VBVR-Bench reasoning data), generating at a maximum area of 832 × 480 with aspect ratio preserved, at 16 fps, with 20 sampling steps and CFG scale 5.0. On Gen-ViRe, the first step produces a 6s clip (96 frames) and subsequent steps produce 3s clips (48 frames). On VBVR-Bench, we follow the official setup and match each clip to the duration of the corresponding ground-truth video. Inference runs on a single A100 GPU. For Veo 3.1, we use the veo-3.1-fast-generate-preview API at native resolution (16:9 with letterboxing for non-matching inputs) and 24 fps. The first step generates 8s of video and subsequent steps generate 4s. Due to API cost, we evaluate Veo 3.1 only on the smaller Gen-ViRe benchmark. For CosmosPredict-2.5, we use the released 14B post-trained checkpoint at 832 × 480 resolution and 16 fps, generating 5s clips per step. All other settings (sampling steps, guidance scale, negative prompt, scheduler, precision) follow the upstream default configuration of the released checkpoint.

Baselines. Single Inference generates one video per sample using the same clip length as the N=1 branch of CollabVR. Pass@k with k ∈ {2,4} runs k independent generations with seeds {1,...,k} under identical inference settings. Gemini 2.5 Pro is shown all k candidates in a single call and selects the best, acting as an output reward model. We also tested scoring each seed independently and choosing the highest scorer, but the single-call joint selection gave both higher accuracy and stronger alignment with human judgments. We expect independent scoring to become more competitive as k grows. VideoTPO follows the official setup with two prompt-rewrite iterations, two seeds per iteration, and a final iteration using only seed 1, yielding five total generations per sample. The LLM optimizer and the critic VLM are both Gemini 2.5 Pro to match CollabVR.

Evaluation. Gen-ViRe is judged by Gemini 2.5 Pro using the rubric prompts from the official repository, with each sample scored across three independent runs to absorb judge stochasticity. VBVR-Bench uses a deterministic rule-based protocol that compares each generated video against ground-truth references, so we report a single run. We report Cost as the total VGM-generated seconds per sample, since VLM compute is negligible relative to VGM compute.

#### A.2 Prompt Templates

We provide the verbatim prompt templates used by the VLM in our pipeline: (i) the progressive planner, which emits the next action prompt and a task-complete flag; and (ii) the step verifier, which returns an accept/reject judgment with a structured diagnosis d (textual reason, actionable suggestion, and a good_fraction estimate of how much of the clip executed correctly). Each VLM call additionally receives the input image and the history of accepted clip frames; the verifier additionally receives the candidate clip. Prompt evolution itself does not require a separate VLM call: the verifier’s suggestion field is folded directly into the next action prompt by evolve(at,d).

#### Progressive planner (step_planner)

You are a visual reasoning expert that plans video generation steps for an

→ image-to-video model.

Given an input image and a task prompt, you must break the task into sequential action steps that a video generation model can execute one at a time. Each step should describe a short, concrete visual action (4-8 seconds of video).

→ →

Inputs

- - TASK_PROMPT: The full task description
- - CURRENT_IMAGE: The current state (input image or last frame of previous clip)
- - COMPLETED_STEPS: What has already been done (empty if first step)
- - STEP_NUMBER: Which step we are planning (1-indexed) Rules

- 1. Each step must describe ONE clear visual action that a video generation model

→ can simulate in 6 seconds.

- 2. DO NOT plan the entire task at once. Only plan the NEXT IMMEDIATE step.
- 3. Describe the action in terms of VISIBLE MOTION and CHANGE -- what should

→ move, where, and how.

- 4. Include the EXACT target state: what the frame should look like when this

→ step is done.

- 5. If the task appears to be already complete based on the current image, set

→ "task_complete" to true.

Output (strict JSON): {

"observation": "Brief description of what you see in the current image", "remaining_goal": "What still needs to happen to complete the task", "task_complete": false, "instruction": "Detailed video generation prompt for the next step.", "target_state": "Visual description of what the last frame should look like

→ after this step",

"estimated_steps_remaining": 2 }

##### Step verifier (step_verifier)

You are a video quality judge evaluating whether a generated video clip executed

→ its intended action correctly. Inputs: TASK_PROMPT, PLANNED_ACTION, TARGET_STATE, VIDEO. Critical Distinction: judge whether the ACTION was executed, NOT whether the

→ full task is complete. Evaluation Criteria

- 1. Did the intended motion/transformation START to happen in the correct

→ direction?

- 2. Is the result CONSISTENT with the planned action (even if incomplete)?
- 3. Were there FUNDAMENTAL errors (wrong direction, wrong object, completely

→ wrong action, scene collapse)? What is NOT a rejection reason:

- - Action happened but didn't fully complete (partial progress is fine)
- - Minor rendering artifacts or small imprecisions
- - The final task goal is not yet reached (planner's job)

On Rejection -- estimate "good_fraction" (0.0-1.0): the fraction of the video that was correct before the error occurs. We use this to do partial re-generation from the good portion.

→ →

###### Output (strict JSON):

If accept: {

"verdict": "accept", "action_executed": true, "progress": "partial or complete", "confidence": "high", "reason": "...", "suggestion": ""

} If reject: {

"verdict": "reject", "action_executed": false, "progress": "none", "good_fraction": 0.3, "confidence": "high", "reason": "...", "suggestion": "..."

}

Step 1

|[Figure 192]<br><br>|[Figure 193]|
|---|
<br><br>|[Figure 194]|
|---|
<br><br>|[Figure 195]|
|---|
|
|---|

[Figure 196]

|[Figure 197]|
|---|

[Figure 198]

The two shortest navy

books lift up, move to

VLM

VGM Generation

the left, and place themselves on the shelf immediately to the right of the last green book in the second cluster.

Planning

Prompt

Input Image

[Figure 199]

𝑪𝟏

[Figure 200]

Verify .

[Figure 201]

###### Self Evolving

|[Figure 202]<br><br>|[Figure 203]|
|---|
<br><br>|[Figure 204]|
|---|
<br><br>|[Figure 205]|
|---|
|
|---|

[Figure 206]

[Figure 207]

[Figure 208]

The two shortest navy books lift up, move to the left, and place themselves side by side in the first empty gap within the second cluster of green books.

VLM

VGM Generation

Evolving

Prompt

Input Image

𝑪𝟏

[Figure 209]

[Figure 210]

Verify .

[Figure 211]

Step 2

|[Figure 212]<br><br>|[Figure 213]|
|---|
<br><br>|[Figure 214]|
|---|
<br><br>|[Figure 215]|
|---|
|
|---|

[Figure 216]

|[Figure 217]|
|---|

[Figure 218]

The remaining navy book lifts up, moves to the left, and places itself in the

VLM

VGM

###### only remaining empty

Generation

Planning

gap within the second cluster of green books.

Prompt

Input Image

𝑪𝟐

[Figure 219]

[Figure 220]

Verify .

[Figure 221]

- Figure 10: Trace through one full CollabVR loop on a multi-step bookshelf task (VBVR-Wan2.2).

Within-step prompt evolution (M2) corrects C1 in Step 1, after which across-step progressive planning (M1) advances to Step 2, sharing the same per-clip verifier.

#### A.3 Verifier Output Examples

- Figure 10 instantiates one full execution of Algorithm 1 on a multi-step bookshelf task, exercising both branches of the inner loop in a single trace.

- Step 1: the reject-and-evolve branch. The planner emits a1 (“...place the two navy books on the shelf immediately to the right of the last green book...”) and the VGM produces c1. The verifier rejects: the two navy books land flush against the cluster’s right edge rather than inside an empty gap. Its diagnosis d1, packaging a textual reason and a concrete suggestion, is folded into the next prompt

- as a′1 = evolve(a1,d1) (“...first empty gap within the second cluster...”, the Self-Evolving panel). The VGM regenerates c1, and the verifier now accepts; the clip is committed to H and its last frame becomes the conditioning frame for Step 2.

- Step 2: the accept-on-first-attempt branch. Conditioned on H = {c1}, the planner emits a2 for the remaining navy book. The VGM produces c2, the verifier accepts immediately, and the outer loop terminates at N=2 without ever entering the evolve branch.

Coverage of the (v,d) interface. Together, the three verifier calls cover both routes the per-clip (v,d) interface can take: reject-with-suggestion drives within-step prompt rewriting (M2), and accept commits the clip and advances the across-step plan (M1). The auxiliary failure router (Appendix A.5) is reached only when within-step recovery is exhausted, which does not happen here.

#### A.4 First-Frame Fidelity for VGM Selection

First-frame fidelity, which measures how closely the first frame of a generated clip matches the conditioning image, is a prerequisite for the step-by-step clip concatenation in our pipeline (each step’s first frame must faithfully resume from the previous step’s final frame) as well as for the partial re-generation auxiliary that re-enters the VGM at a mid-clip frame (Appendix D.5); we therefore use it as a key selection criterion for candidate VGMs. We compute the SSIM between the input image and the first frame of the generated video on Gen-ViRe.

Table 5: First-frame fidelity (SSIM) on Gen-ViRe. VGM SSIM (mean ± std)

VBVR-Wan2.2 0.970 ± 0.043 Cosmos-Predict 2.5 0.971 ± 0.037 Veo 3.1 0.977 ± 0.035 Sora 2 0.818 ± 0.137

VBVR-Wan2.2, Veo 3.1, and Cosmos-Predict 2.5 all achieve sufficient first-frame fidelity for our pipeline. Sora 2 uses an input_reference field rather than strict first-frame conditioning, so its first frame drifts from the input. We therefore exclude Sora 2 from main CollabVR results since downstream step concatenation cannot preserve information across clips, and Sora 2’s upcoming service deprecation further reduced the value of additional engineering against it.

#### A.5 Auxiliary Failure Router

Our implementation extends the per-step prompt evolution of Section 3.3 with an auxiliary VLM call, the failure_router, that takes over when evolution alone cannot produce an accepted clip. After all M within-step evolution attempts return reject, the router examines the failed clip together with the verifier’s diagnosis and chooses one of three follow-up actions. The simplest is regen, a single-shot retry that starts from the first failing frame whenever the verifier’s good_fraction is moderate-to-high, so the correctly executed prefix is preserved (Appendix D.5). When the failure is structural rather than a one-off slip, the router triggers split, which decomposes the residual task into additional sub-steps and re-enters the progressive planner. Finally, when prior decomposition has produced step-boundary artifacts on what is fundamentally a single-shot transformation, fallback collapses to a single-inference run with N=1.

Veo 3.1’s strong single-shot prior often suffices on its own. We therefore invoke the same router at the sample level before any per-step planning, choosing between accepting the single-shot baseline and proceeding with multi-step orchestration. The full prompt template is shown below.

##### Failure router (failure_router)

You are a visual reasoning expert deciding how to recover from a FAILED

→ single-shot video generation. Choose among three recovery strategies. Inputs: TASK_PROMPT, INPUT_IMAGE, FAILED_VIDEO, REJECT_REASON, SUGGESTION,

→ GOOD_FRACTION.

Strategies:

- "regen": single-shot retry. Choose when the failure was an execution slip on a single coherent transformation; good_fraction moderate-to-high; partial regen will likely fix it.

→ →

- "split": decompose into multiple steps. Choose when the failure is structural

-- the task fundamentally needs intermediate states (multi-action procedures, navigation with turns, assembly, drawing multiple objects), or good_fraction is very low and the failure indicates conflated steps.

→ → →

- "fallback": collapse to single-inference (N=1). Choose when prior decomposition introduced step-boundary artifacts and the residual task is simple enough to be generated in a single clip.

→ →

Output (strict JSON): {"action": "regen", "suggestion": "...", "reason": "..."} or {"action": "split", "estimated_steps": 3, "suggestion": "...", "reason": "..."} or {"action": "fallback", "suggestion": "...", "reason": "..."}

### B Additional Quantitative Results

#### B.1 User Study

We additionally conduct a blind, side-by-side human study comparing CollabVR against Pass@1 and Pass@4 at matched VGM (VBVR-Wan2.2). The study is hosted on Prolific, and the per-trial UI is shown in Figure 11.

[Figure 222]

- Figure 11: Per-trial user-study UI. The participant sees the task prompt and the input image, watches three blinded videos, and answers a forced-choice preference and a confidence rating. The condition→label mapping is randomized per task per participant.

Sample selection. We curate 16 tasks spanning Gen-ViRe and VBVR-Bench (one task per GenViRe category and a balanced In-Domain / Out-of-Domain split on VBVR). Within each category we pick the task whose per-sample score gap between CollabVR and Pass@1/Pass@4 is largest (top-|∆|), excluding cases where all three methods score 0 or 1 (no signal). For each task, three videos (Pass@1, Pass@4, CollabVR) are mapped to the labels A/B/C by a fresh random permutation per task per participant.

Procedure. On each task page, the participant sees the input image, the task description, and three blinded videos, and answers two questions: “which video best executes the task?” (A/B/C/equal) and a 1–5 confidence Likert.

Results. We collect n=40 valid submissions across the 16 tasks. Equal responses comprise 17.7% of all trials; aggregating the remaining decisive preferences: In head-to-head comparisons (excluding Equal votes), CollabVR is preferred over Pass@1 in 91.7% and over Pass@4 in 78.9% (Figure 12). These margins corroborate the automated-metric gains in Section 4: human raters on a blinded side-byside comparison consistently prefer CollabVR outputs over both single-shot and best-of-4 baselines. Inter-rater agreement is moderate: average pairwise raw agreement is 66.3% (decisive-only) and Gwet’s AC1 is 0.575.

| |
|---|

| |
|---|

| |
|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- Figure 12: Human preference share on the user study (n=40 participants, 16 tasks; Equal responses excluded). Each row is a 100%-stacked bar restricted to the listed conditions. CollabVR is the dominant choice in all three views.

#### B.2 Effect of Per-Step Attempt Budget M

Effect of attempt budget M. We sweep the per-step attempt budget M ∈ {1,...,5} on VBVR-Bench with VBVR-Wan2.2 under the

- M2-only configuration (Figure 13). The first two budget increments deliver most of the gain (+4.89% from M=1 to M=2, +1.59% from

- M=2 to M=3), after which each additional attempt yields below 1% (+0.97%, +0.51%) while cost continues to grow at a near-constant per-step rate. Beyond M=3, the cost-quality profile thus collapses to that of plain Pass@k resampling: extra compute buys little more than an additional independent draw. We therefore set
- M=3 as the default, retaining the cost-efficient regime where verifier-guided re-generation still provides meaningful repair.

0.76

0.745

0.741

0.734

0.74

+0.004 (+0.51%)

0.722

+0.007 (+0.97%)

0.72

+0.011 (+1.59%)

Score

0.70

+0.034 (+4.89%)

0.688

0.68

0.66

M = 1 (4.5s)

M = 4 (8.6s)

M = 5 (9.7s)

M = 2 (6.1s)

M = 3 (7.5s)

Per-Step Attempt Budget M

Figure 13: Per-step attempt budget M on VBVRBench (VBVR-Wan2.2, M2-only). Score grows monotonically with M, but per-step gains drop below 1% beyond M=3 while cost continues to scale nearly linearly.

#### B.3 Per-VLM Human-Alignment Breakdown

We provide the per-VLM, per-axis numbers underlying Section 4.4 in Table 6, pooling annotations from Gen-ViRe and VBVR-Bench (the latter contributed by collaborators). Three axes are evaluated: D1 plan-depth (N prediction), D2 verifier accept/reject agreement on step clips, and D3 evolution quality of the verifier’s suggested repair (1=poor, 2=moderate, 3=well).

Table 6: Overall human-eval across 3 VLMs. D1 plan-depth on VBVR (n=100, GT video shown) and Gen-ViRe (n=72, image only). D2 verifier agreement on a balanced 125 accept + 125 reject sub-sample. D3 evolution-suggestion quality on n=80 reject clips per VLM.

Gemini 2.5 Pro Qwen3.5-27B Qwen3.5-9B D1 plan-depth (exact-match)

VBVR-Bench 64.0% 64.2% 53.0% Gen-ViRe 73.6% 55.6% 61.1% Overall 68.0% 58.7% 56.4% MAE (parseable) 0.366 0.491 0.484

D2 verifier agreement

accept-recall 84.8% 82.4% 77.6% reject-recall (failure detection) 65.6% 44.8% 40.8% Overall 75.2% 63.6% 59.2% Cohen’s κ (Gen-ViRe) 0.676 0.432 0.304

D3 evolution quality (mean 1–3)

#### Overall mean 2.61 2.55 2.35

≥ 2 (moderate or better) 93.8% 95.0% 86.3% = 3 (top) 67.5% 60.0% 48.8%

Gemini 2.5 Pro leads on every axis. The closed-vs-open gap is largest on the verifier (D2 reject-recall: 65.6% vs. 40–45%, a ∼ 21-point gap; per-verdict confusion matrices in Table 7), suggesting the gap is in training data and objective rather than scale alone (the size lift from 9B to 27B contributes only +4 points on this axis). On the evolution axis (D3) the closed-vs-open gap shrinks to ∆ = 0.06 (Gemini–27B), within noise. D1 ranking flips between setups: 9B over-counts steps when given the GT video on VBVR-Bench (N=3 collapses to 2/20 correct) but matches Gemini’s plan-depth distribution on image-only Gen-ViRe.

Table 7: Verdict confusion matrices, balanced 125:125 sub-sample. Gemini 2.5 Pro Qwen3.5-27B Qwen3.5-9B H=acc H=rej H=acc H=rej H=acc H=rej

pred=accept 106 43 103 69 97 74 pred=reject 19 82 22 56 28 51

#### B.4 Per-Category ∆ Heatmap on VBVR-Bench

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

We mirror the per-category ∆ analysis of Section 4.4 on VBVR-Bench (Figure 14). Module dominance flips relative to Gen-ViRe: M2 alone is now the larger single-module contributor (Table 3: ∆M2 +0.063 vs. ∆M1 +0.035), driven primarily by Abstract (M2 alone +0.131) where atomic perceptual targets are recoverable through verifier-driven re-sampling but admit no genuine sub-goals to expose. The two modules dominate different categories: M1 alone is actually the larger contributor on Spatial (+0.061 vs. M2 +0.019), where decomposition exposes navigation-style sub-goals that the VGM can satisfy individually, while Transformation is the clearest case in which neither mod-

[Figure 223]

Figure 14: Per-category ∆ over Pass@1 on VBVR-Bench (VBVR-Wan2.2). Module configurations follow Section 4.3; per-category overall is the sample-count-weighted mean of the In-Domain and Out-of-Domain entries (categories carry 115, 70, 150, 65, 100 samples with different ID/OOD splits).

ule alone is sufficient (∆M1 −0.040, ∆M2 +0.032) yet the combination recovers a clean positive gain (∆M1+M2 +0.075). With both modules enabled, all five categories turn positive (+0.010 to +0.159), reproducing the per-instance adaptive activation observed on Gen-ViRe.

Yet the symbolic limit identified in Section 4.4 surfaces here as well: Knowledge remains marginal across all three configurations (∆+0.010 for M1+M2, an order of magnitude below the other categories), since neither decomposing nor re-sampling supplies the missing world knowledge—there is no sub-step to expose, and the verifier can only redraw from the VGM’s existing distribution. This parallels the small Analogy (+0.083) and Abstract (+0.090) gains on Gen-ViRe, again pointing to knowledge-grounded VGM training as a complementary direction rather than additional planner or verifier modules.

#### B.5 Veo 3.1 Module Ablation

We ablate the same M1 / M2 modules on Gen-ViRe with Veo 3.1, mirroring Table 3 for VBVRWan2.2.

- Table 8: Per-module ablation on Gen-ViRe with Veo 3.1. ∆ over Pass@1. M1 / M2 columns indicate which module is enabled. The last row is an additional ablation that fixes N=3 for every sample, bypassing the planner’s per-state adaptive N selection.

M1 M2 Cost (s) Overall ∆

- ✗ ✗ 8.0 0.481 –

✓ ✗ 10.1 0.446 −0.035

- ✗ ✓ 12.5 0.527 +0.046

✓ ✓ 21.4 0.550 +0.069 Ablation: fixed N=3 (bypass progressive N selection)

✓ ✓ 24.5 0.450 −0.031

The verifier-and-regen module M2 alone has a single-shot ceiling. With N=1 and M=2 (no decomposition, regen only), Veo 3.1 reaches 0.527, a +0.046 gain over Pass@1 that is almost identical to the corresponding +0.045 gain on VBVR-Wan2.2 (Table 3). However, this number reflects only the tasks that Veo 3.1 can complete in a single clip; tasks requiring multiple sequential sub-actions remain out of reach for any number of regenerations.

- M1 unlocks the tasks where single-shot is insufficient. The full M1+M2 configuration (max N=3, max M=2) reaches 0.550, a further +0.023 over the M2-only ceiling. The gain comes from samples on which the single-shot stream cannot complete the task: there, the planner emits a multistep decomposition (N>1) that breaks the task into sub-actions Veo 3.1 can satisfy individually. M1’s decomposition is therefore the active ingredient: it unlocks tasks that no amount of regeneration on a single clip could solve. Even with the maximum decomposition forced (N=3 for every sample, last row of Table 8), Veo 3.1 drops to 0.450, confirming that adaptive N selection, not just the existence of decomposition, is what makes M1 productive.

Why M1 alone helps VBVR-Wan2.2 but actually hurts Veo 3.1. M1 alone delivers most of the total gain on VBVR-Wan2.2 (+0.120 of +0.140; Table 3), yet on Veo 3.1 it falls 0.035 below Pass@1. The asymmetry follows from the two VGMs’ design priors. VBVR-Wan2.2 is fine-tuned for visual reasoning, so the artificial intermediate states our planner emits (partial rotations, half-drawn lines, incremental fills) are inside its training distribution; each sub-step executes literally, leaving the next step’s conditioning aligned with the planner’s target. Veo 3.1’s strong end-to-end motion prior gives it a higher single-shot Pass@1 (0.481 vs. 0.391 for VBVR-Wan2.2), but reinterprets these intermediate-state requests into outputs the prior considers natural, so with M=0 the deviation feeds the next step’s conditioning unchecked and errors compound across the chain. M1 alone therefore hurts Veo 3.1 most on Planning (−0.069) and Spatial (−0.067), the categories where Veo 3.1’s single-shot capacity was highest to begin with. M2 reconciles Veo 3.1’s generalist prior with the planner’s task-specific structure: the verifier rejects mid-state deviations before they propagate, and the regen loop pulls Veo 3.1’s output back onto the trajectory. This is why M1+M2 reaches 0.550 even though M1 alone underperforms, and the same dynamic explains the Spatial-category synergy in Section 4.4: long-horizon simulation needs both modules, since M1 alone accumulates deviation while M2 alone hits the single-shot ceiling.

#### B.6 Cosmos-Predict-2.5 Detailed Results

CollabVR improves Cosmos-Predict 2.5 (14B) on VBVR-Bench (Table 2, 0.308 → 0.403) but degrades on Gen-ViRe (0.287 → 0.182, ∆ = −0.105 vs. Pass@4; Table 9). This contrast follows each benchmark’s task profile: VBVR-Bench is dominated by single-step tasks, where CollabVR mostly invokes single-clip verification with at most a few re-generations, both of which Cosmos handles competently. Gen-ViRe is dominated by multi-step reasoning tasks (Section 4.3), so CollabVR forces Cosmos to execute decomposed sub-actions across N≤3 clips. Here Cosmos’s weaker per-step instruction-following becomes the bottleneck and decomposition compounds rather than absorbs the error across clips.

- Table 9: Cosmos-Predict 2.5 per-category on Gen-ViRe, fair intersection of CollabVR and Pass@k. Method Abstract Algo Analogy Perc. Planning Spatial Avg

Pass@1 (baseline) 0.204 0.346 0.042 0.305 0.352 0.200 0.246 Pass@4 0.278 0.432 0.042 0.288 0.444 0.217 0.287 CollabVR (N=3) 0.009 0.249 0.375 0.000 0.368 0.131 0.182

The single Gen-ViRe category where CollabVR still helps Cosmos is Analogy (0.042 → 0.375, ∆ = +0.333): Pass@1 is essentially zero there, so any successful sub-step accumulates to a positive end-state. This delineates the regime where decomposition still helps a weak VGM: when single-shot performance is so low that any partial progress beats the baseline.

Limitation. CollabVR’s gain on a given VGM is bounded below by the VGM’s per-step instructionfollowing reliability. We frame CollabVR as orthogonal to, rather than a substitute for, training stronger VGMs: the framework requires a generator that meets a minimum per-step instructionfollowing floor before decomposition becomes profitable.

C Module Diagnostics

- C.1 Pipeline Statistics

We aggregate the runtime behaviour of CollabVR+VBVR-Wan2.2 on Gen-ViRe and VBVR-Bench under the default configuration (Nmax=3, M=3, i.e. up to two re-generations per step).

Per-sample summary. Each Gen-ViRe sample takes on average 2.56 planning steps, 1.46 regenerations, 4.01 generated clips, and 6.79 VLM calls; on VBVR-Bench the same quantities drop to 1.48, 0.90, 2.38, and 3.85 respectively (Table 10), tracking the shorter trajectories on VBVR-Bench.

- Table 10: Per-sample runtime summary on Gen-ViRe (72 samples) and VBVR-Bench (500 samples), under default config.

Quantity Benchmark Mean Std Median Range Steps taken

Gen-ViRe 2.56 0.76 3 [1, 3] VBVR-Bench 1.48 0.77 1 [1, 3]

Gen-ViRe 1.46 1.72 1 [0, 6] VBVR-Bench 0.90 1.26 0 [0, 6]

Re-generations

Gen-ViRe 4.01 2.18 3 [1, 9]

Generated clips

- VBVR-Bench 2.38 1.96 1 [1, 9]

VLM calls

Gen-ViRe 6.79 2.49 6 [2, 12]

- VBVR-Bench 3.85 2.69 2 [2, 12]

Step trajectory length. The planner terminates early at N=1 for 16.7% of Gen-ViRe samples (single-shot accept via task-complete signal), at N=2 for 11.1%, and reaches the cap N=3 for the remaining 72.2%. On VBVR-Bench the trajectory is much shorter (mean N=1.48 vs. 2.56 on Gen-ViRe; Table 11), consistent with VBVR-Bench’s reasoning-heavy but visually constrained tasks where many problems admit a single closed-form action.

Table 11: Step trajectory length distribution on Gen-ViRe and VBVR-Bench.

N (steps used) 1 2 3 (cap) Mean

Gen-ViRe share 16.7% 11.1% 72.2% 2.56 VBVR-Bench share 69.8% 12.8% 17.4% 1.48

Re-generation distribution. Table 12 reports the per-sample re-generation count distribution on both benchmarks. On Gen-ViRe, the distribution decays gradually from 0 to the cap of 6 regens (mean 1.46). On VBVR-Bench, the distribution is bimodal: a primary mode at 0 regens (58.6%) from samples the planner finishes in a single shot, and a secondary mode at 2 regens (26.4%) from multi-step trajectories that fail mid-way (mean 0.90, contributing 451 extra clips overall).

Table 12: Re-generation distribution per sample on Gen-ViRe and VBVR-Bench. Each row reports the share of samples whose total re-generation count across the trajectory falls in the given bin.

Re-gens / sample 0 1 2 3 4 5 6 Mean

Gen-ViRe (n=72) 48.6% 9.7% 12.5% 13.9% 9.7% 2.8% 2.8% 1.46 VBVR-Bench (n=500) 58.6% 7.6% 26.4% 2.0% 4.2% 0.2% 1.0% 0.90

Per-category breakdown on Gen-ViRe. Algorithmic samples consume the most compute per task (5.0 clips on average) due to a higher re-generation rate (2.50/sample), while Analogy and Planning converge fastest (Table 13).

Table 13: Per-category compute breakdown on Gen-ViRe.

Category Avg N Avg re-gens Avg clips Avg VLM calls

Abstract 2.25 1.75 4.00 6.67 Algorithmic 2.50 2.50 5.00 7.83 Analogy 2.00 1.08 3.08 5.67 Perceptual 3.00 1.50 4.50 7.50 Planning 2.75 0.67 3.42 6.17 Spatial 2.83 1.25 4.08 6.92

- C.2 Verifier Run-time Behavior We characterize the verifier on the same Gen-ViRe run.

Step-level outcomes. The verifier accepts 49.5% of clips on the first attempt and recovers another 14.7% through re-generation, for a 64.1% final acceptance rate at an average retry depth of 0.57 per step. The remaining 35.9% are rejected even after the maximum two re-generations and propagated downstream as the best-of-attempts clip.

Per-step rejection trend. Final-reject rate is comparable at the first two steps but rises sharply

- at the deepest step (Table 14). The step-3 spike reflects cumulative visual drift: each step’s first frame is the previous step’s last frame, so any mild deformation accepted earlier makes the next step harder to satisfy. The same pattern shows up in the diminishing returns of N beyond 3 on Gen-ViRe (Section 4.3) and even more sharply on Cosmos-Predict 2.5 (Appendix B.6). Overall, the verifier is exercised aggressively (about 37% of steps trigger at least one re-generation) without saturating the budget, supporting our default choice of M=3.

Table 14: Verifier final-reject rate by step index on Gen-ViRe.

Step index Step 1 Step 2 Step 3 Final reject rate 31.9% 31.7% 46.2%

#### C.3 Cost Decomposition

We verify the “VLM compute is negligible relative to VGM compute” claim in Section 4.1 along two axes: GPU wall-clock time on the open-source backbone (VBVR-Wan2.2, single A100) and API $ cost on the closed-source backbone (Veo 3.1).

VLM per-call profile. We replayed representative Gemini 2.5 Pro calls on Gen-ViRe (planner: 1 image + task description; verifier: 1 instruction + 1 video clip) and recorded latency and token counts, averaged over 3 trials per call type.

Call type Latency Input tok Output tok

Planner 10.15s 800 186 Step verifier 10.60s 2,724 117

- Table 15: Per-call Gemini 2.5 Pro profile. The verifier’s 2.7K-token input is dominated by the inline video; the planner’s 800-token input is the prompt template, the task description, and a single image.

Component Per-sample wall-clock Share VBVR-Wan2.2 VGM (A100) 979.6s (16.3min) 93.5% Gemini 2.5 Pro VLM (planner + verifier) 68.5s (1.14min) 6.5% Total 1048.1s (17.5min) 100% VGM / VLM ratio ≈ 14×

- Table 16: Per-sample wall-clock decomposition (VBVR-Wan2.2+CollabVR on a single A100, averaged over Gen-ViRe).

Closed-source: API cost. Veo 3.1 Fast is priced at $0.15/s of generated video, so the per-sample 21.4s reported in Table 1 costs $3.21, compared to $0.026 for the VLM, a ∼ 125× ratio (Table 17).

Component Per-sample $ Share Veo 3.1 VGM $3.210 99.2% Gemini 2.5 Pro VLM (planner + verifier) $0.026 0.8% Total $3.236 100% VGM / VLM ratio ≈ 125×

Table 17: Per-sample API cost decomposition (Veo 3.1+CollabVR on Gen-ViRe).

Per-sample VLM aggregate. With 2.56 planner and 4.01 verifier calls per sample (Section C.1), each Gen-ViRe sample triggers ∼ 6.6 VLM calls totalling ∼ 68.5s of API wall-clock and ∼ 13K input tokens. At Gemini 2.5 Pro pricing ($1.25/M input, $10/M output), this works out to $0.026 per sample.

Open-source: GPU wall-clock time. On a single A100 at 480p with 20 inference steps, VBVRWan2.2 generates the per-sample 17.8s of video in approximately 980s of wall-clock, while the VLM contributes only ∼ 68.5s, a ∼ 14× ratio (Table 16).

On both axes the VGM dominates by an order of magnitude or more, justifying the convention in Section 4.1 of reporting Cost as VGM-generated seconds per sample as a faithful proxy for total compute on either deployment regime.

Three circles are … from left to right … mark the second largest circle with

Your task is to identify all hollow points. Circle each hollow point with

Show the black ball eating all red balls in a greedy largest-first

Show circles … horizontal bar, sorted by their circumference in

Animal faces … Sort them by size from smallest to largest and align

Prompt

Predict the next color in the sequence.

a red outline.

a red ring.

sequence.

descending order.

them.

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

InputImage

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

GT(LastFrame)

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

VBVR-Wan2.2

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

+CollabVR

N = 1 N = 2 N = 3

- Figure 15: Qualitative gains scale with planner-predicted step count N on VBVR-Bench. For each task we show the input image, GT last frame, single-shot VBVR-Wan2.2 output, and VBVRWan2.2+CollabVR output, with two representative tasks grouped under each of N=1, N=2, and N=3.

### D Additional Qualitative Results

#### D.1 Examples by Step Count N

We organize Figure 15 by the planner’s predicted step count to make the per-bin behavior of CollabVR visible.

- N=1: matched outputs, incidental fixes. When the planner deems the task atomic, both methods produce visually similar outputs and CollabVR adds little structural change. Its contribution is restricted to suppressing incidental errors, e.g. on “predict the next color in the sequence” the baseline fills the next slot with red while CollabVR correctly emits green.

- N=2–3: decomposition recovers the GT trajectory. At higher step counts, the failure modes of single-shot generation become structural rather than incidental. On “show circles ...sorted by their circumference in descending order”, single-shot VBVR-Wan2.2 drops most of the circles before any sorting occurs; CollabVR instead lays out all circles on the bar in a first sub-step and only then performs the sort, recovering the GT layout. The same pattern appears on “sort animal faces by size and align them”: the baseline misorders or substitutes faces, while progressive planning produces the correct ascending arrangement.

A group of shapes are arranged in a 'large-medium-small-large-medium' pattern, draw the next shape in the designated area.

Cosmos-Predict2.5

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

###### +CollabVR

Predict the next color in the sequence.

Cosmos-Predict2.5

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

###### +CollabVR

A group of shapes with consistent size, circle the only one that is different.

Cosmos-Predict2.5

|[Figure 272]|
|---|

|[Figure 273]|
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

###### +CollabVR

Create a complex wave with multiple peaks and mark each maximum point. The animation should clearly show the wave's oscillation and highlight all local maxima using red hollow circles with solid red center dots.

Cosmos-Predict2.5

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

###### +CollabVR

First Frame Last Frame

- Figure 16: CollabVR generalizes to Cosmos-Predict-2.5 on VBVR-Bench. Each two-row block contrasts Cosmos-Predict-2.5 alone (top) with Cosmos-Predict-2.5+CollabVR (bottom) over a sixframe sequence (first → last frame).

A dashcam perspective of a vehicle in 'stop-and-go' highway traffic. The video shows the vehicle smoothly following the car in front (a white SUV). The lead SUV travels at a steady speed, then suddenly decelerates and comes to a complete stop.

VBVR-Wan2.2

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

###### +CollabVR

Instantly reflect this pattern along the central vertical axis, while keeping the existing colored pattern unchanged. Static camera perspective, no zoom or pan.

VBVR-Wan2.2

|[Figure 308]|
|---|

|[Figure 309]|
|---|

|[Figure 310]|
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

|[Figure 317]|
|---|

|[Figure 318]|
|---|

|[Figure 319]|
|---|

+CollabVR

“The task is to get a bottle of Coke from the refrigerator. The video must show the entire process of completing the task in a physically realistic and continuous sequence of actions.”

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

Veo3.1

|[Figure 326]|
|---|

|[Figure 327]|
|---|

|[Figure 328]|
|---|

|[Figure 329]|
|---|

|[Figure 330]|
|---|

|[Figure 331]|
|---|

###### +CollabVR

This is a Raven's Progressive Matrix reasoning task.Analyze the visual patterns across all rows and columns in this 3x3 grid. Deduce the logical rule that governs the entire matrix.Then, generate the single, correct shape in the empty bottom-right square that perfectly completes this pattern.

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

Veo3.1

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

###### +CollabVR

First Frame Last Frame

- Figure 17: CollabVR works across open- and closed-source VGMs on Gen-ViRe. Each two-row block contrasts the base VGM (top) with +CollabVR (bottom); the upper two tasks use VBVRWan2.2 and the lower two use Veo 3.1.

- D.2 Examples by VGM CollabVR is VGM-agnostic and applies to a range of generators beyond VBVR-Wan2.2.

Generalization to another open-source VGM. Figure 16 shows that the same gains carry over to Cosmos-Predict-2.5: although the alone model exhibits its own characteristic failures (e.g., hallucinating an external hand or pen instead of acting on the canvas), pairing it with CollabVR consistently drives the generator toward the intended in-canvas action across all four tasks.

Open- and closed-source on Gen-ViRe. Figure 17 broadens the comparison to Gen-ViRe with an open-source (VBVR-Wan2.2) and a closed-source (Veo 3.1) generator, spanning real-world (dashcam, refrigerator) and symbolic-pattern tasks (mirror reflection, Raven’s matrices). +CollabVR produces visibly more faithful executions than the base VGM in every case, supporting the results in Tables 1 and 2.

###### VLM Detection Failure

[Figure 344]

[Figure 345]

[Figure 346]

Find the pentagon among the polygons and draw a red circle around it, starting small and expanding to encircle the shape.

Verify .

[Figure 347]

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

|[Figure 353]|
|---|

Qwen3.5-9B

###### VLM Partial Detection

[Figure 354]

[Figure 355]

[Figure 356]

Find the pentagon among the polygons and draw a red circle around it, starting small and expanding to encircle the shape.

###### Verify .

[Figure 357]

|[Figure 358]|
|---|

|[Figure 359]|
|---|

|[Figure 360]|
|---|

|[Figure 361]|
|---|

|[Figure 362]|
|---|

|[Figure 363]|
|---|

Qwen3.5-9B

Find the pentagon among the polygons and draw a red circle around the green pentagon on the right side of the scene,

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

Evolving Verify .

[Figure 368]

starting small and expanding until it fully encircles the shape.

|[Figure 369]|
|---|

|[Figure 370]|
|---|

|[Figure 371]|
|---|

|[Figure 372]|
|---|

|[Figure 373]|
|---|

|[Figure 374]|
|---|

Qwen3.5-27B

VLM Detection Success

[Figure 375]

Find the pentagon among the polygons and draw a red circle around it, starting small and expanding to encircle the shape.

[Figure 376]

[Figure 377]

###### Verify .

[Figure 378]

|[Figure 379]|
|---|

|[Figure 380]|
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

Gemini2.5Pro

Find the pentagon among the polygons and draw a red circle around the green pentagon on the right side of the scene, not the yellow diamond in the center, starting small and expanding until it fully encircles the shape.

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Evolving Verify .

[Figure 389]

|[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

|[Figure 395]|
|---|

Gemini2.5Pro

- Figure 18: Verifier VLM choice shapes the recovery loop on a single VBVR-Bench trace (VBVRWan2.2 as the VGM). The same prompt is verified by Qwen3.5-9B (top, false-accept), Qwen3.5-27B (middle, evolution with a coarse positional cue), and Gemini 2.5 Pro (bottom, evolution that explicitly excludes the distractor).

'Large-MediumSmall' pattern, draw the next shape in the designated area.

Please identify the triple intersection area in this Venn diagram and color it red.

Find the pentagon among the polygons and draw a red circle around it.

The left ... higher density liquid, the right ... lower density ... add a red rectangular ... higher-density liquid.

Mirror the left half across the … to complete the symmetric incrementing pattern.

Prompt

Please mark the second largest circle with a red outline.

|[Figure 396]|
|---|

|[Figure 397]|
|---|

|[Figure 398]|
|---|

|[Figure 399]|
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

InputImage

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

|[Figure 407]|
|---|

GT(LastFrame)

|[Figure 408]|
|---|

|[Figure 409]|
|---|

|[Figure 410]|
|---|

|[Figure 411]|
|---|

|[Figure 412]|
|---|

|[Figure 413]|
|---|

Qwen3.5-9B

|[Figure 414]|
|---|

|[Figure 415]|
|---|

|[Figure 416]|
|---|

|[Figure 417]|
|---|

|[Figure 418]|
|---|

|[Figure 419]|
|---|

Qwen3.5-27B

|[Figure 420]|
|---|

|[Figure 421]|
|---|

|[Figure 422]|
|---|

|[Figure 423]|
|---|

|[Figure 424]|
|---|

|[Figure 425]|
|---|

Gemini2.5Pro

- Figure 19: Final +CollabVR outputs track verifier capability across VBVR-Bench tasks (VBVRWan2.2 as the VGM). Each row shows the last frame produced when the verifier is Qwen3.5-9B, Qwen3.5-27B, or Gemini 2.5 Pro, with the input image and GT last frame at the top.

#### D.3 Examples by VLM

The verifier-quality gaps quantified in Section 4.4 translate into qualitatively distinct recovery behaviors that propagate to the final output.

A single trace under three verifiers. Figure 18 traces one regeneration loop on the prompt “find the pentagon among the polygons and draw a red circle around it”. Qwen3.5-9B false-accepts a mis-localized target, so no recovery is triggered. Qwen3.5-27B rejects but evolves the prompt with only a coarse positional cue (“on the right side”), and the regenerated clip is only partially correct. Gemini 2.5 Pro both rejects and articulates the precise distractor (“not the yellow diamond in the center”), and the regenerated clip recovers the correct red circle around the pentagon.

Aggregate behavior across tasks. Figure 19 confirms this is not a single-trace artifact: across multiple VBVR-Bench tasks, proximity of the final +CollabVR output to the GT last frame increases monotonically with verifier capability, mirroring the score ordering in Table 4.

###### Case 1 : VLM Failure → VGM Misguidance

[Figure 426]

[Figure 427]

Verify .

There are multiple shapes with the same form but different colors. Circle the leftmost one.

[Figure 428]

|[Figure 429]|
|---|

|[Figure 430]|
|---|

|[Figure 431]|
|---|

|[Figure 432]|
|---|

|[Figure 433]|
|---|

|[Figure 434]|
|---|

GT

|[Figure 435]|
|---|

|[Figure 436]|
|---|

|[Figure 437]|
|---|

|[Figure 438]|
|---|

|[Figure 439]|
|---|

|[Figure 440]|
|---|

CollabVR

[Figure 441]

[Figure 442]

[Figure 443]

Observe this circular arrangement of arrows and circle the one that points in a different direction from the rest. Verify .

|[Figure 444]|
|---|

|[Figure 445]|
|---|

|[Figure 446]|
|---|

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

GT

|[Figure 450]|
|---|

|[Figure 451]|
|---|

|[Figure 452]|
|---|

|[Figure 453]|
|---|

|[Figure 454]|
|---|

|[Figure 455]|
|---|

CollabVR

The visual distinction between the objects is subtle, which makes it challenging for the VLM to reliably recognize the correct target.

###### Case 2 : VLM Success → VGM Failure

The scene shows a 10x10 grid with a green start point, a red end point, and yellow cells marked with numbers 1, 2, and 3. An orange circular agent is positioned at the green start point. The agent can move to adjacent cells (up, down, left, right). Starting from the green start point, the agent must visit the numbered yellow cells in numerical order (1, then 2, then 3), taking the shortest path between each consecutive pair of numbered cells. After visiting all numbered cells in sequence, the agent must reach the red end point, also following the shortest path.

|[Figure 456]|
|---|

|[Figure 457]|
|---|

|[Figure 458]|
|---|

|[Figure 459]|
|---|

|[Figure 460]|
|---|

|[Figure 461]|
|---|

GT

[Figure 462]

[Figure 463]

- Step 1

The orange agent moves from the green start cell one cell to the left, and then six cells down, stopping on the yellow cell labeled '1'.

[Figure 464]

[Figure 465]

- Step 2

The orange agent moves from its current position on the yellow cell '1’ five cells to the right and then one cell up, stopping on the yellow cell labeled '2'.

[Figure 466]

[Figure 467]

- Step 3

[Figure 468]

[Figure 469]

[Figure 470]

Verify .

[Figure 471]

[Figure 472]

Evolving The agent should move five cells to the right, then one cell up. Do not move six cells to the right.

The orange agent moves from its current position on the yellow square at the bottom right of the

grid, four cells to the left and three cells up, stopping on the red square.

|[Figure 473]|
|---|

|[Figure 474]|
|---|

|[Figure 475]|
|---|

|[Figure 476]|
|---|

|[Figure 477]|
|---|

|[Figure 478]|
|---|

CollabVR

Execution Error : Still move six cells.

The agent can only move along edges in the direction they point moving from one node to an adjacent node each step. Move the blue triangular agent from the green starting node to the red ending node

along the path with the minimum number of steps.

|[Figure 479]|
|---|

|[Figure 480]|
|---|

|[Figure 481]|
|---|

|[Figure 482]|
|---|

|[Figure 483]|
|---|

|[Figure 484]|
|---|

GT

[Figure 485]

[Figure 486]

Evolving The blue agent should move only to nodes connected by valid directed edges in the correct direction.

|[Figure 487]|
|---|

|[Figure 488]|
|---|

|[Figure 489]|
|---|

|[Figure 490]|
|---|

|[Figure 491]|
|---|

|[Figure 492]|
|---|

CollabVR

Even with a successful VLM plan and evolve, the VGM may fail to execute visually fine-grained operations that require long-horizon control.

- Figure 20: Two distinct ceilings limit full CollabVR on VBVR-Bench. Case 1: the VLM verifier fails to detect the issue, so no recovery is triggered. Case 2: the VLM diagnoses the failure and evolves the prompt correctly, but the VGM cannot execute the fine-grained operation.

#### D.4 Failure Cases

- Figure 20 characterizes the residual failures of the full CollabVR pipeline, decomposing them into two complementary causes that test-time orchestration alone cannot resolve.

- Case 1: VLM detection failure. The verifier itself fails to identify the issue, so no recovery is triggered and the incorrect clip is finalized. CollabVR circles a yellow square instead of the leftmost light-blue triangle, and circles an in-distribution arrow instead of the only outward-pointing one—both failures pass the verifier silently.
- Case 2: VGM execution failure. The verifier correctly diagnoses the failure and the prompt evolves with the right semantic content (“Do not move six cells to the right.”, “valid directed edges in the correct direction”), yet the VGM still misexecutes the fine-grained operation. Here the bottleneck has moved from the supervisor to the generator, and additional retries cannot help.

These two ceilings map onto the small-∆ categories of Figure 8: the symbolic-category gap (Knowledge, Analogy) is largely a Case 1 issue, while the residual Spatial / Transformation gap reflects a Case 2 ceiling. They point respectively to stronger VLM grounding and reasoning-oriented VGM training as complementary directions.

D.5 Partial vs. Full Re-generation: Maze Case Study

|Prefix Partial Regeneration Partial Regeneration Start Point<br><br>|
|---|

|[Figure 493]<br><br>01|[Figure 494]<br><br>02|
|---|---|
|[Figure 495]<br><br>03|[Figure 496]<br><br>04|

|[Figure 497]|
|---|

Full Regeneration Partial Regeneration

Figure 21: Partial re-generation from fτ outperforms full re-generation on a maze task. Full regeneration is run for four independent attempts (left); partial re-generation reuses the correct prefix up to the first failing frame (right).

We additionally explore a recovery design specific to navigation-style tasks: on rejection, the VGM is re-invoked from the first failing frame fτ rather than re-rolling from scratch, so that the prefix already produced correctly is preserved across attempts.

- Figure 21 compares the two modes on a maze instance. Full re-generation fails to reach the goal across four independent attempts because each resampled trajectory discards previously accumulated progress. Partial re-generation

from fτ reuses the correct prefix and successfully converges on the goal cell, illustrating that test-time compute is much more effective when it targets the failed suffix rather than the entire trajectory.

We position partial re-generation as an auxiliary mode applicable when the prefix carries useful information toward the goal, with precise failure detection as a promising future direction.

### E Broader Impact

Positive impacts. Video reasoning is becoming a central AI capability for dynamic, temporallygrounded processes that static-image or pure-text reasoning cannot express, with applications spanning educational and scientific demonstrations, procedural walkthroughs, navigation in synthetic environments, and embodied-agent simulation. CollabVR contributes to this direction by showing that strong general-purpose VLMs and VGMs, while individually imperfect for goal-directed video generation, can be composed at step-level granularity into a closed-loop reasoning system that improves task fidelity without any additional training, and that the resulting trajectories are more interpretable and auditable than single-shot VGM outputs because each step’s verification decision and prompt evolution are exposed as discrete artifacts. We see the next frontier of video reasoning coming less from a single ever-larger end-to-end generator than from richer inference-time collaborations between strong specialised models such as VLMs as step-level supervisors, VGMs as visual simulators, and other modality-specific reasoners, with CollabVR as one concrete instance of this design pattern.

Potential negative impacts. The framework inherits the misuse risks of the underlying generative models, including higher-fidelity synthetic video that could be used for deceptive or non-consensual content. CollabVR adds no new generative capability beyond the base VGM; it composes existing models for closed-loop reasoning rather than training new generators.

Mitigations. We use only off-the-shelf VGMs (Wan2.2, Cosmos-Predict-2.5, Veo 3.1) and VLMs (Gemini 2.5 Pro, Qwen3.5) under their original licenses and existing safeguards. The released artifacts (orchestration code and the human-annotated reliability benchmark) carry no novel high-risk generative capability beyond what is already publicly available.

