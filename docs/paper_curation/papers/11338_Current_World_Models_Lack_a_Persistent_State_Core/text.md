## arXiv:2606.20545v1[cs.CV]18Jun2026

[Figure 1]

[Figure 2]

[Figure 3]

June 2026

[Figure 4]

# Current World Models Lack a Persistent State Core

Jinpeng Lu1,2,∗ Dexu Zhu2,3,∗ Haoyuan Shi1,2 Linghan Cai5 Guo Tang6 Yinda Chen1,2 Jie Cao3 Duyu Tang4

Yi Zhang2 Yong Dai2 Xiaozhu Ju2

1 University of Science and Technology of China 2 Beijing Innovation Center of Humanoid Robotics (X-Humanoid) 3 NLPR, Institute of Automation, Chinese Academy of Sciences 4 Independent Researcher 5 Dresden University of Technology 6 Peking University

“I like to think the moon is there even if I am not looking at it.” —Albert Einstein World models are increasingly regarded as a decisive step toward artificial general intelligence, yet modeling the physical world demands more than rendering convincing frames on demand: it requires an internal world state that keeps evolving over time, decoupled from observation, so that objects endure and events run to their conclusions whether or not a camera is watching, much as the moon holds to its orbit when no one is looking. This requirement is a blind spot of existing benchmarks, which reward surface properties such as fidelity, motion, and camera controllability while never asking whether a generated world keeps evolving once it is unobserved. We introduce WRBench, the first systematic diagnostic benchmark that treats camera motion as an intervention on observability and resolves evaluation into a human-calibrated chain that asks whether the camera executes the requested interaction, whether the scene stays continuous and identifiable while in view, and whether a returning target remains consistent with the event that was set in motion. Across 9,600 videos from 23 models spanning four control paradigms, one finding proves stubborn: current systems maintain the observed world as a tracking shot, resuming a returning target in the state at which it was abandoned rather than advancing the event while it went unseen. Because this failure recurs across control paradigms, model families, and increments of scale, robust world-state evolution does not follow from cleaner imagery, tighter control, richer geometric priors, or sheer parameter count; scaling Wan from 1.3B to 14B parameters even lowers re-observed state from 0.66 to 0.62, and Lingbot-World, while attaining the strongest visible spatial and state consistency among controllable models at 0.874 and 0.719, surrenders control with a requested-camera precision of only 0.468, the lowest of any controllable model. We therefore argue that the stability of the physical state kernel and the consistency of worldlines under viewpoint intervention should become first-class objectives of world-model design, so that a world model captures how the world will unfold rather than how the next frame appears.

Project Page: https://jinplu.github.io/WRBench Code: https://github.com/JinPLu/WRBench Correspondence: jinplu@mail.ustc.edu.cn, vito.dai@x-humanoid.com, juxzhu@googlemail.com

∗ Equal contribution. Work done during the internship at X-Humanoid.

### 1 Introduction

World models are increasingly seen as a key step toward AGI. Video generation has progressed rapidly, and a growing body of work positions generators as visual world models, from interactive environment generation [1] and physical-AI platforms [2] to camera- and 4D-aware generation systems [3–7] and memory-based long-horizon generators [8]. Yet claiming to be a world model demands far more than producing realistic frames. At minimum, a generator must maintain

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### No Robust Evolution

###### Model Input

###### Interactive World

[Figure 10]

[Figure 11]

[Figure 12]

(when camera moves away)

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Prompt

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

In the bedroom, a cat jumps onto the bed.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Expected World State

[Figure 41]

[Figure 42]

Start

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Is the cat still correct after the camera moves away and returns?

First Frame

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Dragged by Camera Never leaves view

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

No Returned Consistency

###### Expected Video

[Figure 66]

[Figure 67]

[Figure 68]

(when camera returns)

[Figure 69]

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

###### Camera Action

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Frozen/Reset Vanished

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

↑

w

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

← ↓ →

a s d

OutofView

[Figure 117]

[Figure 118]

[Figure 119]

Position Jump Object Split

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

- Figure 1 WRBench Teaser. A shared scene, event, and viewpoint intervention test whether visible and returned evidence support the same evolving world state. The returned target must preserve the event endpoint, not merely reappear plausibly.

a world state that evolves continuously over time, independent of observation, not just render plausible frames when watched. The moon orbits the Earth whether or not anyone is looking at it. This is the blind spot of existing benchmarks. Quality-decomposition benchmarks [9–11], temporal and physical benchmarks [12–19], and recent world-modeling benchmarks [20–27] all evaluate fidelity, motion, and camera controllability, but none asks whether the generated dynamic world keeps evolving when the camera looks away.

How can this capability be tested? Viewpoint change offers a natural probe: it alters only the angle of observation, not what is happening in the world. As illustrated in Figure 1, the prompt specifies “in the bedroom, a cat jumps onto the bed,” with the cat on the floor in the starting frame. The camera then turns away and later returns. If the world state continues to evolve while unobserved, the cat should be on the bed when the camera comes back. In practice, generators exhibit diverse failures (Figure 1, right): the cat may be dragged along with the camera and never leave the view; upon return, it may still be on the floor, may have vanished, may appear in an unexpected position, or may split into duplicates. The difficulty is that these failures stem from entirely different causes: perhaps the camera never turned away, or perhaps it did but the world stopped evolving while unobserved. A single score cannot tell these apart. We call this the attribution problem of viewpoint intervention. The capability it tests, that a target’s state remains consistent with the event endpoint after leaving and re-entering the view, is what we term world-state persistence under viewpoint intervention.

We introduce WRBench, a diagnostic benchmark built around this evidence-attribution problem. WRBench is grounded in Natural-25, a prompt suite of 25 scene families crossed with a four-level event design that factors spatial displacement against state change. Rather than treating camera control as a rendering command, WRBench treats it as an intervention on observability: each test case specifies an initial scene, event, and viewpoint intervention, then asks what claims about the evolving world state are actually supported by the generated video.

Evaluation in WRBench follows a hierarchical diagnostic chain with six dimensions: (i) requestedcamera precision, (ii) prompt-camera alignment, (iii) visual integrity, (iv) visible spatial and state consistency, (v) re-observation support, and (vi) re-observed spatial and state consistency. These dimensions progress from control execution to state closure—was the requested trajectory executed, did a prompt-only system follow the intended camera intent, is the frame evidence intact, are

spatial relations and action states correct while visible, does the target re-enter the view in judgeable form, and does the returned target preserve the event endpoint. Dimensions (iv) and (vi) each report separate spatial and state scores, but they remain single diagnostic dimensions with shared denominators. All dimensions are reported on their respective denominators; crucially, reobserved-consistency scores in (vi) are computed only on the subset that passes the re-observation gate in (v), so each of the four situations above receives its own diagnostic conclusion rather than being merged into one score.

To enable fair comparison across generators with different architectures and control paradigms, we introduce WRBenchLib, which makes viewpoint-control provenance explicit. It systematically records what form of camera instruction each model received and what condition it actually delivered, supporting condition-level diagnostics across source-video, geometry-cache, modelinferred, and prompt-only controllability paradigms. We also ground the benchmark with 2,547 deduplicated human annotator verdicts that calibrate the automatic evaluation at the same granularity as the benchmark outputs, rather than reducing WRBench to a learned holistic preference score.

We evaluate 23 video generators on 9,600 videos. Analyzing the results along evaluation dimensions, controllability paradigms, and within-family increments, we identify a recurring preservation–access–re-observed-consistency gap: visible quality and re-observation access largely determine whether the return test is posed, while re-observed-state correctness is a separate capability that is not predicted by model scale among the evaluated systems. The failure concentrates on in-place state change rather than relocation. This diagnosis is consistent with the hypothesis that progress requires not more pixels, but a what-memory that records hidden change and a training objective that supervises endpoint persistence.

The contributions are:

- 1. Viewpoint-conditioned world-state benchmark. We introduce WRBench to evaluate whether dynamic event state remains attributable under camera-induced changes in observability.
- 2. Open-source, unified, human-validated toolkit. (a) We release WRBenchLib, which records delivered viewpoint-control conditions and re-observation judgeability boundaries so that heterogeneous generators are compared by the evidence they actually provide; (b) we release 2,547 deduplicated human annotator verdicts that calibrate and bound the automatic evaluators at the same granularity as the benchmark outputs.
- 3. Directional diagnosis. Evaluating 23 models, we show that re-observed-state consistency is a capability distinct from visible fidelity and re-observation access; current scale and architectures do not bind it within the evaluated scope. The diagnosis points toward a what-memory and an endpoint-persistence objective as a direction for future work.

### 2 Related Work and Positioning

Decomposed video evaluation. WRBench is closest to benchmark suites that turn video generation from a single preference problem into explicit evaluation dimensions. FETV [28] and EvalCrafter [29] organize video quality, motion, and text–video alignment into interpretable measurements. VBench establishes the most relevant community-infrastructure pattern: define targeted measurement components, attach tailored evaluators, validate them against human judgments, and release reusable prompts, metrics, generations, and annotations [9]. VBench++ [10] and VBench-2.0 [11] extend this pattern toward image-to-video generation, trustworthiness, controllability, physics, commonsense, and intrinsic faithfulness. WRBench follows the same decomposed, human-calibrated evaluation principle, but changes the measured object: whether world-state evidence remains attributable when viewpoint intervention changes what is visible.

World-generation and physical evaluation. A second line evaluates generated videos or scenes as candidate worlds. T2V-CompBench [18], TC-Bench [12], ChronoMagic-Bench [17], StoryEval [19], VideoPhy [13], VideoPhy-2 [14], PhyGenBench [15], and T2VPhysBench [16] probe prompt composition, event order, metamorphic structure, physical commonsense, and physical-law adherence. WorldModelBench studies world-modeling violations such as instruction following and physics adherence [20], while WorldScore is especially relevant because it treats camera trajectory and layout specification as part of the world-generation task [21]. WRBench is complementary: it treats camera control not only as an output target or generation condition, but as an evidence intervention. The central question is whether the same event-induced state remains supported as evidence is visible, temporarily unobserved, and returned to a judgeable view.

State, memory, and out-of-sight dynamics. The closest conceptual neighbors study continuity under interrupted observation, memory, interaction, or revisiting. STEVO-Bench [25] tests whether naturally evolving processes continue when observation is interrupted; LiveWorld [8] studies out-ofsight dynamics and event permanence; MBench [26] and MIND [27] evaluate entity, environment, causal, or action-conditioned memory; and WorldMark [22], WBench [23], and iWorld-Bench [24] study interactive or multi-turn video world models. WRBench shares the motivation that hidden or revisited evidence is a critical world-model test, but it is not a return-only memory benchmark. It separates visible evolution, re-observation support, and re-observed-state consistency. Missing judgeable return evidence is insufficient support for a hidden-state claim; incorrect spatial or event-state evidence after a judgeable return is a re-observed-consistency failure.

Heterogeneous viewpoint condition types. Recent video generators expose heterogeneous viewpoint interfaces: explicit camera trajectories in MotionCtrl [3] and CameraCtrl [4], novelview or multi-view generation in ViewCrafter [30], CAT4D [31], and GEN3C [5], source-video transformation in ReCamMaster [32], spatial memory in Spatia [6], 4D geometric control in VerseCrafter [7], and out-of-sight memory in LiveWorld [8]. This heterogeneity is itself an evaluation problem. A model that receives a dense pose trajectory, a model that receives a source video, and a model that receives only a natural-language camera request should not be interpreted as if they were given the same evidence. WRBench therefore records the intended intervention, delivered condition, and generated evidence, so results can be read by viewpoint condition type rather than collapsed into one scalar leaderboard.

Table 1 summarizes this positioning. The comparison does not rank the overall value of prior benchmarks; each prior suite covers an important slice of video or world-model evaluation. The table instead records which benchmark targets are explicit enough to enable dynamic world-state evaluation under viewpoint intervention. Here, state robustness means viewpoint/visibility-aware state evidence with separate control, re-observation, and error attribution, and Pathway Diagnostics means metric-, condition-type-, and feature-readout diagnostics.

WRBench is therefore not a survey-style ranking over all video or world-model capabilities. It inherits the human-calibrated evaluation discipline of VBench [9] and the control-aware worldgeneration framing of WorldScore [21], but targets a different claim: dynamic world-state attribution under viewpoint intervention. Natural-25, the WRBenchLib generation-provenance layer, judgeability criteria, human calibration, and diagnostic slices across events, cameras, and viewpoint condition types produce bounded evidence for preservation, access, and re-observed-state consistency rather than a single scalar leaderboard.

### 3 WRBench Suite

WRBench is a diagnostic benchmark for dynamic world-state evaluation under viewpoint intervention. Following the decomposed-suite design VBench [9] established for video benchmarks, it is organized in four layers: (1) a Natural-25 prompt suite that defines the scene and event substrate; (2) WRBenchLib, a unified toolkit that delivers test conditions to each generator and records

- Table 1 Representative benchmark coverage. Rows summarize whether prior benchmarks explicitly target dynamic world-state evaluation capabilities.

Pathway Diagnostics

World Dynamics

Unified Control

Visual Quality

State Robustness

Evolution Consistency

Method

VBench [9] ✗ ✓ ✗ ✗ ✗ VBench-2.0 [11] ✓ ✗ ✗ ✗ WorldModelBench [20] ✓ ✗ ✗ ✗ ✗ WorldScore [21] ✓ ✓ ✗ ✗ ✗ WorldMark [22] ✓ ✓ ✓ ✗ ✗ ✗ WBench [23] ✓ ✓ ✓ ✗ ✗ ✗ iWorld-Bench [24] ✓ ✓ ✓ ✗ ✗ ✗ MIND [27] ✓ ✓ ✗ ✗ ✗ STEVO-Bench [25] ✓ ✗ ✗ LiveBench [8] ✓ ✗ MBench [26] ✓ ✗ ✗ WRBench (Ours) ✓ ✓ ✓ ✓ ✓ ✓

Note. ✓ indicates full coverage, partial coverage, and ✗ no coverage; symbols do not rank overall benchmark value.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Event Scene

Camera

Input

###### WRBench Metric

State

State lie down

Full jump up

|[Figure 133]|
|---|

w

↑

[Figure 134]

[Figure 135]

Visible Spatial/State Consistency

[Figure 136]

None stand, wait

Spatial approach

← ↓ →

a s d

Qwen 3.5

Returned Spatial/State Consistency

Sptial

[Figure 137]

[Figure 138]

[Figure 139]

Unified Camera Control

Qwen 3 VL

Out-of-View Applicability Gate

[Figure 140]

[Figure 141]

[Figure 142]

Explicit Camera Matrices & Number

[Figure 143]

[Figure 144]

DINO v2

Visual Integrity

Native Action Signals

[Figure 145]

Rendered Geometric Conditions

[Figure 146]

Unified OpenCV C2W Sidecar

[Figure 147]

VGGT-𝛀

Camera Precision/Alignment

Camera Prompt for API Models

Return

GO

- Figure 2 WRBench method overview. Natural-25 event-view records supply the scene, event, and viewpoint intervention; WRBenchLib translates each record into a generated video and provenance record per model; the evaluation suite scores six diagnostic dimensions from control execution to re-observed-state consistency; and human preference annotation calibrates each dimension independently.

what was actually received and produced; (3) a six-dimensional evaluation suite that scores each generated video from control execution through to re-observed-state consistency; and (4) human preference annotation that calibrates the automatic evaluators axis by axis. Figure 2 summarizes the pipeline.

- 3.1 Natural-25 Prompt Suite and Event-View Records This section defines the basic unit of evaluation. WRBench does not score a generated video

- as a standalone visual-quality sample; it treats the video as evidence for a specific event under a specific viewpoint intervention. Natural-25 supplies the scene and event substrate: 25 scene families crossed with a four-level event design that factors spatial displacement against state change (details in Section 4). Each test fixes the same initial observation and event, then varies the viewpoint condition so that the relevant state evidence may remain visible, become temporarily hidden, or return to a judgeable view. Prompts specify the initial scene, event, and viewpoint request without revealing the returned endpoint state, so the generated video itself must carry the evidence for any re-observed-state claim.

The evaluated unit is an event-view record, not a prompt or generated clip alone:

ri = x0i,ei,τi,νi,πi . (1)

Here x0i is the initial observation, ei the specified event, τi the intended viewpoint intervention, νi the visibility regime, and πi the prompt or interface variant. Spatial evidence covers positions, paths, contacts, supports, and anchors; state evidence covers actions, poses, object states, and events.

#### 3.2 WRBenchLib: A Unified Toolkit for Scenarios, Events, and Camera Control

With the test unit defined, the next question is how different generators receive it. Some models accept explicit camera trajectories, others take a source video, and others receive only a naturallanguage prompt. Forcing all models into one interface would make the comparison unfair. WRBenchLib addresses this by translating each event-view record into the form a given generator can consume and recording what was actually delivered, so heterogeneous models are compared on the evidence they received rather than on a single control interface.

WRBenchLib supplies the generation-provenance map for each model m:

zi,m = Φm(ri) = (ui,m,di,m,vi,m,ηi,m), (2)

where ui,m is the model-specific input (trajectory, source video, geometric condition, or prompt), di,m the condition that was actually delivered, vi,m the generated video, and ηi,m a provenance record that logs the full pipeline for reproducibility. WRBenchLib records all four components before any evaluation dimension is scored.

#### 3.3 Evaluation Dimension Suite

With the test unit and delivery layer in place, WRBench scores each generated video along six diagnostic dimensions, matching the chain introduced in Section 1:

- (i) Requested-camera precision. For models that receive an explicit trajectory, did the generated video follow the requested viewpoint path?
- (ii) Prompt-camera alignment. For prompt-only or API models that do not receive a trajectory, did the video follow the intended camera direction (e.g. common-yaw, static-hold)?
- (iii) Visual integrity. Is the frame evidence structurally intact and readable, or are there artifacts such as cuts, disappearance, identity drift, or structural collapse that would make downstream judgments unreliable?
- (iv) Visible spatial and state consistency. While the target remains in view, are its spatial relations (position, contact, support) and action/event states correct? Reported as separate spatial and state scores.
- (v) Re-observation support. After a nontrivial hidden interval, does the target return to the field of view in a form clear enough to be scored? This dimension decides whether the re-observed-state test is posed at all.
- (vi) Re-observed spatial and state consistency. On the subset where (v) is satisfied, does the returned target preserve the expected spatial relation and event endpoint? Again reported as separate spatial and state scores.

Dimensions (i)–(v) are scored over their respective full denominators; dimension (vi) is conditional on (v), so that absence of judgeable return evidence is recorded as insufficient support rather than

- as a passed or failed state test.

#### 3.4 Evaluation Method Suite

The previous section defines what WRBench measures; this section describes how each dimension is scored, in the same order.

Camera control (i–ii). Dimension (i) targets models that receive explicit trajectories: it compares the requested viewpoint path against the trajectory recovered from the generated video using VGGT-Ω [33], built on VGGT [34]. Dimension (ii) targets prompt-only and API models that do not receive trajectories: it uses CamAlign and static-hold diagnostics to test whether the video follows the intended camera direction (common yaw) rather than full pose recovery. Because the two dimensions address different control interfaces, they are never merged into one control score.

Visual integrity (iii). Before scoring any world-state dimension, WRBench checks whether the frame evidence is trustworthy. Visual integrity uses a fixed DINOv2 feature proxy [35]:

Ivis(v) = min (sglobal(v),slocal(v)). (3)

A low value marks unreliable visual evidence such as cuts, disappearance, identity drift, or local structural collapse. Implementation details are in Appendix D.1.

Visible consistency (iv). While the target evidence remains in view, WRBench probes whether spatial relations and action states match the specified event. The probes are prompt-conditioned yes/no questions scored by Qwen-3.5-9B. The context c bundles the scene, event, target object or relation, camera condition, visibility regime, and a dimension-specific rubric. Probe templates and their polarity are fixed before model measurement. For video v, context c, and question q, the evaluator produces a yes probability from yes/no token scores:

expLyes(q,v,c) expLyes(q,v,c) + expLno(q,v,c)

, (4)

pyes(q | v,c) =

where Lyes and Lno are log-sum-exp scores over allowed yes/no answer tokens. Positive probes Pa+ ask whether intended evidence is present; negative probes Pa− ask whether counterevidence is absent:

ea(q,v,c) =   

pyes(q | v,c), q ∈ Pa+, 1 − pyes(q | v,c), q ∈ Pa−.

(5)

Visible spatial and visible state measurements in dimension (iv) average the polarity-adjusted probes:

1 |Pa+| + |Pa−|

Ma(v,c) =

ea(q,v,c). (6)

q∈Pa+∪Pa−

Thus dimension (iv) asks whether the event is wrong while evidence remains observable.

Re-observation support and re-observed consistency (v–vi). Dimensions (v) and (vi) address evidence that leaves the field of view and later returns — the distinctive design target of WRBench. Re-observation support (v) records whether the re-observed-state test is even applicable: the target evidence must become temporarily hidden or unjudgeable, later return to the observable field, and be identifiable enough for re-observed scoring. Re-observed spatial and re-observed state consistency in dimension (vi) reuse the same probe form as (iv), but only after judgeable re-observation is established. Let

Sreturn = {arsp,arst} and Ra(v,c) = H(v,c) ∧ U(v,c) ∧ Ja(v,c), (7)

where H marks a nontrivial hidden or unjudgeable interval, U marks return to the observable field, and Ja marks enough identifiable returned evidence for metric a. These judgeability predicates are evaluated by a separate VLM gate (Qwen-3-VL-Instruct-8B), kept distinct from the Qwen-3.5-9B scoring evaluator so that re-observation applicability is established before any re-observed-consistency score is computed. The re-observed measurements are then

Ma(v,c) =   

1

|Pa+|+|Pa−| q∈Pa+∪Pa− ea(q,v,c), Ra(v,c) = 1, NA, Ra(v,c) = 0,

a ∈ Sreturn. (8)

Dimension (vi) therefore asks whether the endpoint remains coherent after temporary nonobservation; dimension (v) decides whether that question can be asked at all.

Aggregation. The six dimensions are kept as separate diagnostic outputs rather than collapsed into a single leaderboard score. For model m and metric a, let Sa(vi,m,ci) denote whether evidence supports measuring that metric. Camera precision (i), prompt-camera alignment (ii), visual integrity (iii), and visible metrics (iv) use their full applicable denominators, with visual integrity reported separately as an evidence-quality floor. Re-observation support (v) is reported as a support rate over all generated outputs; re-observed metrics in (vi) set Sa = Ra and are averaged only on judgeable rows. WRBench therefore reports both support rates and conditional metric values:

Dm,a = {i : Sa(vi,m,ci) = 1}, ρm,a = |Dm,a| |Ia|

,

(9)

M¯m,a =   

1

|Dm,a| i∈Dm,a Ma(vi,m,ci), |Dm,a| > 0, NA, |Dm,a| = 0.

The resulting diagnostic profile therefore keeps view failure, unreadable evidence, visible event error, missing re-observation support, and re-observed-state error as distinct, interpretable outcomes rather than averaging them away.

#### 3.5 Human Preference Annotation

Automatic metrics measure observable evidence but cannot serve as ground truth without human calibration. Rather than collecting a single holistic preference per pair, WRBench asks annotators to judge each of the six diagnostic dimensions independently: requested-view execution, promptcamera alignment, visual integrity, visible world-state consistency, re-observation support, and re-observed consistency. Each comparison asks which video better satisfies one claim on one axis. For re-observed dimensions, annotators first decide whether hidden-and-returned evidence is judgeable; re-observed-consistency labels are interpreted only on that subset. WRBench reports prevalence-robust AC1 [36] for human agreement and keeps nominal κ and ordinal/Krippendorffstyle α [37] in Appendix G.

For a pair p = (vpA,vpB) and metric a, let yp,a ∈ {−1,0,1} denote the ordered human label. With Ca denoting pairs for which both human and automatic evidence are defined, WRBench reports rank alignment and a thresholded decision check:

∆p,a = Ma(vpA,cp) − Ma(vpB,cp), ρa = Spearman ({yp,a}p∈Ca,{∆p,a}p∈Ca),

 

1, ∆p,a > τa, 0, |∆p,a| ≤ τa, −1, ∆p,a < −τa.

yˆp,a(τa) =



(10)

Industrial indoor

Transit platform

0.822

0.812

Farmyard

0.999

0.781

0.977

Construction site

0.640

0.664

0.904

Domestic indoor

0.725†

7†

Medical indoor

0.

Camera

0.094

Re-ObservedState

Precision CameraAlignment

Source-video Geometry-cache Model-inferred Prompt-only API

0.870

| |
|---|

44%Domestic

0.681

###### WRBench

| |
|---|

Office indoor

Re-Observed Spatial

###### WRBench

0.864

0.734

Integrity

| |
|---|

Camera · OOV · World-State Closure

0.846

0.771†

| |
|---|

Natural-25

0.824

0.855†

Visible State

OOV Ratio

0.558

Education indoor

Visible Spatial

73.0%

0.668

62.3%

39.6%

Domestic outdoor

12.7%

0.735

0.731 0.821

Retail checkout

0.759

0.876

0.891

Hotel lobby

Residential

shared Commercial

dining

- Figure 3 Benchmark coverage and diagnostic frontiers. Left panel shows Natural-25 scene coverage; right panel shows best observed diagnostic values by viewpoint condition type. Re-observed metrics are conditional on judgeable re-observation, separating access from re-observed correctness.

The thresholds τa are fixed before reporting and create a tie region around small measurement differences; they are not tuned per model or per result table.

- 4 Experiment

WRBench evaluates 23 models over 9,600 generated videos as a diagnostic profile for dynamic world-state evaluation under viewpoint changes. The profile separates preservation, access, and re-observed consistency. Visual integrity and visible spatial/state consistency are measured over the generated-video evidence where the relevant content is observable. Requested-camera precision is measured on 7,500 certified local requested-control rows, while prompt-camera alignment is a separate common-yaw diagnostic. Re-observed spatial and state consistency are conditionally aggregated over the 2,073 outputs with judgeable hidden-and-returned evidence.

- Figure 3 summarizes the benchmark. The left panel charts Natural-25 coverage as a balanced

factorial grid rather than a naturalistic frequency sample. It spans 25 scene families drawn from 19 distinct venues, ranging over indoor and outdoor settings and over human and animal actors, and it crosses every family with a four-level event design that factors spatial change against state change. Domestic indoor scenes form the largest single group, at 8 of the 25 families, yet no family dominates the aggregate, since each contributes an equal share. The suite therefore probes breadth of world state under a controlled design rather than rewarding memorization of frequent scenes. The right panel reports condition-type diagnostic frontiers, the best observed value for each viewpoint condition type on each evaluation dimension. These frontiers preview the main result pattern: current systems preserve visible continuation, instantiate camera motion, expose the hidden target again, and preserve the returned event endpoint at different strengths.

Table 2 is the model-level WRBench diagnostic profile. Its rows are grouped by viewpoint condition type: source-video, geometry-cache, model-inferred, and prompt-only conditions. WRBenchLib records the intended viewpoint intervention, delivered condition, generated video, and measurement evidence before aggregation, so each row is read through the evidence path actually delivered to the benchmark. The central question is endpoint binding: when the camera leaves and returns, does the target-relative displacement, spatial relation, contact outcome, or state change remain bound to the same evolving world state?

- Table 2 Model-level WRBench diagnostic profile by viewpoint condition type. For 23 models, CamPrec is strict local requested-control precision, CamAlign common-yaw intent, Reobs. support the fraction of outputs with judgeable hidden-and-returned evidence, and Reobs. spatial/state conditional means over that subset. The rightmost column Avg. is the mean of the six metric scores (CamAlign, Integ., the two visible and two re-observed scores), excluding re-observation support. Daggers mark sparse support; best/second-best marks are computed within each viewpoint condition type (per column), not across the whole table.

Cam Prec.

Cam Align. Integ.

Vis. spatial

Reobs. spatial

Reobs. support

Vis. state

Reobs.

Model

state Avg. Source-video condition

ReCamMaster 0.717 0.729 0.740 58.5% 0.715 0.535 0.665 0.616 0.667 HyDRA 0.822 0.855 0.691 33.2% 0.648 0.500 0.509 0.445 0.608 InSpatio World 14B 0.693 0.661 0.824 62.3% 0.821 0.668 0.734 0.664 0.729 Geometry-cache condition

Gen3C 0.699 0.764 0.749 73.0% 0.723 0.558 0.681 0.640 0.686 Spatia 0.704 0.482 0.763 25.8% 0.731 0.541 0.600 0.586 0.617 VerseCrafter 0.781 0.667 0.846 28.0% 0.707 0.508 0.607 0.584 0.653 Model-inferred condition

- Wan-Fun 2.1-14B 0.757 0.526 0.846 18.2% 0.733 0.530 0.659 0.621 0.652

- Wan-Fun 2.1-1.3B 0.771 0.729 0.842 13.8% 0.725 0.513 0.709 0.657 0.696

- Wan-Fun 2.2-5B 0.724 0.335 0.812 12.0% 0.805 0.607 0.709 0.664 0.655

- Wan-Fun 2.2-A14B 0.758 0.553 0.848 17.6% 0.810 0.625 0.698 0.649 0.697 Lingbot World Cam 0.513 0.175 0.870 6.0% 0.876 0.735 0.717† 0.663† 0.673 Lingbot World Act 0.468 0.168 0.856 6.4% 0.874 0.719 0.771† 0.725† 0.685 LiveWorld 0.812 0.856 0.775 39.6% 0.703 0.541 0.661 0.600 0.689 Hunyuan GameCraft 0.534 0.361 0.705 6.0% 0.672 0.440 0.554† 0.490† 0.537 Hunyuan WorldPlay 0.708 0.261 0.870 20.8% 0.737 0.523 0.640 0.603 0.606 MagicWorld 0.764 0.720 0.543 15.0% 0.623 0.458 0.584 0.574 0.584 Prompt-only condition Hailuo 2.3 – 0.075 0.829 6.3% 0.891 0.759 0.719† 0.642† 0.652 HappyHorse 1.0 I2V – 0.025 0.860 10.3% 0.875 0.715 0.779 0.695 0.658 Kling v2.6 – 0.094 0.864 3.3% 0.854 0.674 0.711† 0.617† 0.636 Wan2.2 I2V Plus – 0.013 0.800 2.0% 0.829 0.644 0.714† 0.610† 0.602

- Wan2.6 I2V – 0.016 0.856 12.7% 0.855 0.682 0.659 0.556 0.604

- Wan2.7 I2V – 0.020 0.750 6.0% 0.848 0.676 0.715† 0.638† 0.608 WanX2.1 I2V Turbo – 0.030 0.713 1.0% 0.839 0.651 0.855† 0.777† 0.644

#### 4.1 Diagnostic Analysis: From Common Failures to Model Design Implications

Overview. A persistent world model must bind three things into one shared state: how the camera moves, how the target is displaced, and how an event changes the world. WRBench’s most informative failures are coupling failures, because a model can render readable frames, plausible camera motion, and a coherent visible action, yet still fail to bring the hidden target back into a judgeable view, or bring it back with the wrong spatial relation or state. We read every model along three axes of increasing depth. (i) Across evaluation dimensions in Section 4.1.1, we isolate the failure shared by all 23 models, which is that visible plausibility, re-observation access, and re-observed-state correctness do not move together. (ii) Across world-model paradigms in Section 4.1.2, the viewpoint interface reorders which models can even pose the return test, but not whether they pass it. (iii) Along model series in Section 4.1.3, we trace which increments of scale, architecture, and training actually repair that shared failure. The first two axes establish a model-independent problem, while the third asks what fixes it.

###### 4.1.1 Common Failures across Evaluation Dimensions

Finding 1. World consistency must be measured on return, not read off frame quality. Visible-fidelity and re-observation-access scores form their own correlated blocks but predict re-observed-state correctness only weakly, so it is a distinct capability that benchmarks and models must target directly rather than infer from image quality or camera precision.

Visual integrity and camera access predict whether the return question is asked, not whether it is answered correctly. A coherent world model would make the diagnostics move as one body, with better rendering buying more returns and a returned object carrying the right position and state. Instead, Figure 4 shows three loosely-linked groups. Visible spatial and state lock together

- at r=0.97, and the re-observed pair forms its own block at r=0.94, yet the two blocks track each other only moderately, since visible→re-observed reaches just r=0.60–0.79. Re-observation support, which records whether the object ever leaves and returns in a checkable form, sits apart from all of them and even inverts, falling to r= − 0.42 with visible spatial and to −0.15 with both visual integrity and re-observed state.

Each capability a model adds removes one excuse and exposes the next, so the universal failure lives at a single link, the re-observed state, which cleaner images and more access reach rather than remove. The cleanest-image systems test the least. The prompt-only APIs and LingBot post the top visible scores, with Hailuo at 0.891/0.759 and LingBot at 0.876/0.735, yet they keep the object on screen, so it returns in only 3–6% of clips and the hidden-state question is rarely posed. Source-video and geometry-cache interfaces remove that barrier: Gen3C returns the object in 73% of clips, while InSpatio and ReCamMaster do so in 58–62%. This only surfaces the next failure, because once the object is back and checkable, its re-observed state still scores only ≈ 0.62–0.66.

- Figure 5 isolates the mechanism. Turning the camera swings re-observation support by roughly two orders of magnitude, from near zero under a static camera to ∼40% under yaw, while the re-observed scores barely move; across the two well-supported yaw directions, support triples from 13%→40% yet re-observed state shifts under 0.01. Camera motion therefore decides whether the test can be run, not its result. Appendix E.1 shows one clip where all three stage-local successes hold yet the returned world state is wrong. The decoupling is sharpest at the extremes, where the model with the best camera execution returns the worst world state while the one with the highest access degrades the scene as the camera moves, as Appendix E.5 details.

|1.00|0.84|-0.22|-0.58|-0.53|-0.36|-0.33|0.35|
|---|---|---|---|---|---|---|---|
|0.84|1.00|-0.34|-0.75|-0.65|-0.52|-0.38|0.69|
|-0.22|-0.34|1.00|0.62|0.53|0.41|0.34|-0.15|
|[Figure 148]<br><br>-0.58<br>-0.53<br>|-0.75<br>-0.65<br>|0.62 0.53|1.00 0.97|0.97 1.00|0.79 0.74|0.64 0.60|-0.42<br>-0.32<br>|
|-0.36|-0.52|0.41|0.79|0.74|1.00|0.94|-0.24|
|-0.33|-0.38|0.34|0.64|0.60|0.94|1.00|-0.15|
|0.35|0.69|-0.15|-0.42|-0.32|-0.24|-0.15|1.00|

Cam Prec.

1.0

|[Figure 149]|
|---|

Cam Align.

0.5

Integ.

Vis. Spatial

0.0

Vis. State

Reobs. Spatial

0.5

Reobs. State

1.0

Support

CamPrec.CamAlign.Integ.Vis.SpatialVis.StateReobs.SpatialReobs.StateSupport

- Figure 4 Metric correlation structure. Modellevel Pearson correlations among WRBench diagnostic metrics over the 23 evaluated rows. Visible and re-observed consistency form local blocks, while support remains separate from visual integrity.

Per-condition Mean Score

0.00 0.25 0.50 0.75 1.00

Reobs. Support Static

0.00 0.13 0.40

| |
|---|

Yaw L R Yaw R L

Reobs. State

0.51 0.61 0.62

| |
|---|

Reobs. Spatial

0.52 0.66 0.67

0.65

Vis. State

0.58

| |
|---|

0.58

0.83

Vis. Spatial

0.76

| |
|---|

0.76

0.77 0.77 0.90

Integ.

| |
|---|

Cam. Align.

0.45 0.47 0.95

Cam. Intent

0.58 0.60 0.95

0.0 0.5 1.0 1.5 2.0 2.5

Stacked Mean Score

Figure 5 Camera motion governs access, not correctness. Grouped bars compare static, left-to-right yaw, and rightto-left yaw conditions. Camera motion mainly changes reobservation support, while re-observed consistency moves much less.

Finding 2. World models struggle with object state changes, while spatial changes help them leverage pretrained motion priors to return the object more reliably. Object transformation, not trajectory, is the open problem. Factoring every Natural-25 event into a spatial change where the object is relocated and an in-place state change where it transforms where it sits, e.g. folding or tipping, a move hands the model a new coordinate to track and the static scene to anchor against, so the object returns roughly right, whereas an in-place change offers no such anchor and leaves the altered object to drift and smear on return.

Moving an object and changing it in place pull in opposite directions. Having localized the failure to the re-observed link, we ask which events break it. Figure 6 factors every Natural-25 event into two independent axes, whether the object moves and whether its state changes, toggling each while holding the other fixed. Adding a move barely touches the visible position score, at +0.008 (n.s.), while lifting the later scores: visible state rises +0.070 and re-observed state +0.038

- at paired Wilcoxon p < 0.01. Adding an in-place change does the reverse, and the tell is where it lands. It depresses the visible position score the most, at −0.114 (p < 0.001), while leaving the visible state score essentially flat at −0.031 (n.s.), and it drags down both re-observed scores, re-observed spatial −0.075 (p < 0.01) and re-observed state −0.068 (p < 0.001). The position hit is the mechanism: a relocation gives the model a new coordinate to track and the static scene to anchor against, whereas an in-place change offers no new location, so the altered object drifts and smears where it sits. The damage stays local, since overall visual integrity falls only −0.029, so the frame holds while the changed object degrades.

Two checks rule out the easy reading. The effect is not just rarer returns. At matched return rates, in right-to-left yaw where moved and in-place events return at 40.5% vs. 40.4%, in-place events still score lowest on visible position and state, at 0.661/0.502 vs. 0.819/0.629. Nor do the axes interact: the per-model interaction term on re-observed state is indistinguishable from zero, with mean −0.009, Wilcoxon p = 0.45, n = 13, so a move adds its benefit on top of an in-place change rather than rescuing it. The single hardest case for today’s video world models is thus a sharp one, preserving an object’s changed state across a viewpoint change when the change moved it nowhere. Appendix E.2 traces it frame by frame, showing that in a fixed scene the changed object returns in the wrong state, is silently erased, or is lost during occlusion, while the room

##### (a) (b)

[Figure 150]

[Figure 151]

- Figure 6 Event-factor pressure splits by factor. Boxplots show per-model metric distributions under the Natural-25 2 × 2 event design; bars summarize event-factor pressure on out-of-view behavior. Contrasts separate relocation from in-place state change; stars denote paired Wilcoxon tests (* p < 0.05, ** p < 0.01,

*** p < 0.001; n.s. not significant).

behind it stays sharp.

These two failures, a re-observed-state link that neither quality nor access ever closes and an in-place hard case that no event design escapes, are common to all 23 models and independent of architecture or interface. The next two subsections ask what does move them, namely the world-model paradigm through which a model is told where to look in Section 4.1.2, and the scale, architecture, and training increments within a model series in Section 4.1.3.

###### 4.1.2 World-Model Paradigms by Viewpoint Condition Type

Finding 3. World-model controllability has mastered static re-rendering, not dynamic evolution. Access rises with how much already-seen footage a paradigm carries, but every paradigm only replays the recorded scene, so the next controllability target is off-screen dynamics, not just viewpoint.

We read the four viewpoint condition types as controllability paradigms rather than individual designs, where each paradigm fixes what a model can control about an already-seen scene: a prompt-only condition carries a sentence, a model-inferred condition carries only internal camera, action, or state controls, a source-video condition carries a reference stream of appearance and dynamics, and a geometry-cache condition carries a point-cloud, 3D, or 4D record of the scene. The right panel of Figure 3 previews how these paradigms compare, plotting the best value each condition type reaches on every dimension. The access and visible-quality frontiers separate sharply by paradigm, while the re-observed-consistency frontier rises far less and, where it does look high, rests on sparse re-observation. A richer paradigm therefore mostly buys access and readable frames rather than a correct return. The paradigms differ in one thing, how much external record of the already-seen scene they externalize, and the two questions from Section 4.1.1, whether the hidden object returns and whether it returns in the right state, separate cleanly along that single axis.

Table 3 ranks representative paradigms by how often they re-expose the hidden target, and access falls monotonically down the ladder as the external record thins. Prompt-only conditions sit at the floor, with Hailuo at 6.3% and Kling at 3.3%: given only a sentence, the camera moves conservatively and keeps the target framed, so a hidden-then-returned test is rarely even created. A model-inferred condition raises access, since Wan-Fun A14B reaches 17.6% while synthesizing the new view internally, though at lower visible consistency than the prompt-only systems, 0.718 versus 0.825. Source-video and geometry-cache conditions top the ladder, with Gen3C at 73%, InSpatio at 62%, ReCamMaster at 58%, and VerseCrafter at 28%, because an external record can be replayed or re-projected to put the target back where it was last seen. The amount of

Table 4 Re-observed-state event gap. Representative interfaces with re-observed state on relocation vs. inplace events; Subtype follows the viewpoint-condition taxonomy, and Gap is relocation minus in-place, the in-place penalty.

- Table 3 Condition-interface access profile. Representative interfaces ranked by Reobs. support; Subtype follows the viewpoint-condition taxonomy; Cam Align. is common-yaw alignment and Vis. avg. mean visible consistency.

Spatial event

Cam Align.

Reobs. support

State event Gap

Vis. avg.

Model Subtype

Model Subtype

Gen3C Geometry-cache 73.0% 0.764 0.641 InSpatio 14B Source-video 62.3% 0.661 0.745 ReCamMaster Source-video 58.5% 0.729 0.625 VerseCrafter Geometry-cache 28.0% 0.667 0.608 Wan-Fun A14B Model-inferred 17.6% 0.553 0.718 Hailuo 2.3 Prompt-only 6.3% 0.075 0.825 Kling v2.6 Prompt-only 3.3% 0.094 0.764

Gen3C Geometry-cache 0.711 0.559 +0.152 InSpatio 14B Source-video 0.720 0.591 +0.129 Spatia Geometry-cache 0.633 0.512 +0.121 VerseCrafter Geometry-cache 0.638 0.527 +0.111 ReCamMaster Source-video 0.686 0.589 +0.097 HyDRA Source-video 0.475 0.414 +0.061 Wan-Fun A14B Model-inferred 0.682 0.611 +0.071

already-seen footage carried, rather than the cache label, sets access. Gen3C derives its 3D cache from a source video and so behaves as V2V and tops the ladder, whereas the text- and image-conditioned (TI2V) geometry caches such as VerseCrafter and Spatia carry less and sit well below the source-video rows. Access is therefore a property of the paradigm rather than of any one design, since handing the model a record of the scene, in whatever form, is what lets the question be posed at all. Appendix E.3 shows the two ends of this spectrum frame by frame, and Appendix F.3 details what each paradigm externalizes.

Re-observed-state correctness, however, does not track that ladder. Table 4 reports re-observed state on the two event families from Section 4.1.1, where in-place change trails relocation on every paradigm. On the geometry-cache and source-video rows that lead on access, the shortfall is a consistent ≈ 0.10–0.15, with Gen3C at 0.711 versus 0.559, InSpatio at 0.720 versus 0.591, and Spatia at 0.633 versus 0.512, exactly the universal hard case the prior section isolated. The ordering inside that gap is the tell: the paradigm best at re-projecting a static scene, the geometry cache in Gen3C, opens the largest gap of all at 0.152, since a stored record of where surfaces were cannot reconstruct a change it never observed. The decoupling is sharpest read across paradigms. Moving from the model-inferred to the geometry-cache paradigm more than quadruples access, from Wan-Fun A14B at 17.6% to Gen3C at 73%, yet leaves relocation re-observed state essentially flat, from 0.682 to 0.711, and pushes in-place re-observed state down, from 0.611 to 0.559. A richer paradigm therefore only moves the bottleneck from whether the object returns to whether the return reflects an unobserved change. This second wall is identical across all four paradigms, because it is the in-place collapse of Section 4.1.1 re-expressed along the model axis: the paradigm reorders which models can pose the test but not the ceiling on answering it. Appendix E.4 runs one in-place fold through all four paradigms and finds four distinct failure signatures, none recovering the unobserved change.

###### 4.1.3 Series Increments: Scaling, Architecture, and Training

Series diagnostics as a world-model stress map. Across all twelve Wan-derived rows and all three axes, the same pattern holds: access improves more readily than endpoint persistence, so the in-place change that prior sections isolated as the universal hard case stays the unbound endpoint that no increment yet reaches. We treat the twelve rows as one controlled stress map over a shared video prior rather than a capacity leaderboard, because each row shares a documented Wan base but varies one increment while holding the family roughly fixed. Figure 7 reads that map along three axes, scale and version in panel a, architecture design in panel b, and training signal in panel c, with per-row mechanism coding in Table 11 and Appendix F.2. Each axis asks the endpoint question from Sections 4.1.1–4.1.2: when an event resolves out of view and the

-0.014

Cam Prec.

+0.034

+0.004

Integ.

+0.036

+0.008

Vis. Spatial

+0.006

+0.018

Vis. State

+0.018

- Wan2.1 1.3B 14B

| |
|---|

- Wan2.2 5B A14B

-0.051

Ret. Spatial

- -0.011
- -0.015

-0.036

Ret. State

+0.044

Reobs. Support

+0.056

0.05 0.00 0.05

Metric Change

InSpatio World 14B

Lingbot World

0.65

Wan-Fun 2.1-14B

Returned-StateConsistency

LiveWorld

0.60

ReCamMaster

Spatia VerseCrafter

0.55

Camera fine-tune Geometry carrier Memory / action

0.50

| |
|---|

0.45

| |
|---|

Hydra

0 10 20 30 40 50 60 70

Re-observation Support (%)

Camera / Content

4

Layout / Motion

3

Reappear / OOS

2

Pose / Action

1

Endpoint Persistence

none

0 1 2 3 4

Rows With Public Evidence

- Figure 7 Series diagnostics. Panels summarize scale/version, architecture, and training-signal diagnostics for Wan-derived rows. The common pattern is that access improves more readily than endpoint persistence.

camera later returns, does the increment only make that moment easier to see and score, which is re-observation support, or does it also preserve how the event resolved, which is re-observed spatial and re-observed state consistency, conditional over the judgeable re-observation subset?

Finding 4. Regular video-supervised scaling will not deliver world-state evolution, so it has to be designed in. Larger Wan backbones add re-observation access while the 2.1→2.2 upgrade adds visible fidelity, which are the observable axes such training already optimizes, yet conditional re-observed state, scored only on the clips where the object is re-observed, stays in a fixed band, so endpoint persistence is a target for architecture and training rather than for more parameters or standard video data.

Scaling and version upgrades buy observable quality but not endpoint persistence, because modern video scaling is tuned to observable axes. Wan’s own reports frame scaling as improving motion quality, visual fidelity, camera control, and instruction adherence: 1.3B→14B lifts the WanBench weighted score 0.689→0.724 and the VBench total 83.96%→86.22%, while Wan 2.2 explicitly targets cinematic aesthetics and complex motion [38]. All of these are scored from generated pixels and text alignment, never from hidden-state persistence after occlusion [9]. WRBench measures the orthogonal axis. The only clean within-family scale pairs are the four Wan-Fun rows in Figure 7a. Wan 2.1 1.3B→14B raises re-observation support 13.8%→18.2% but lowers re-observed state consistency 0.657→0.621, and Wan 2.2 5B→A14B raises support 12.0%→17.6% while re-observed state slips 0.664→0.649, even though A14B does improve requested-camera precision, visual integrity, and visible consistency over 5B. A version axis sits inside the same rows: the Wan 2.2 variants reach markedly higher visible spatial and state consistency, at 0.805/0.607 and 0.810/0.625, than either Wan 2.1 row at 0.725/0.513 and 0.733/0.530, yet re-observed state never leaves the 0.621–0.664 band. Read against the other eight rows, where source-video and geometry carriers span 25.8–62.3% support while LingBot enters the judgeable subset only 6.0–6.4% of the time, the shortfall is compositional rather than a scaling law: a stronger prior surfaces more returns and cleaner frames without preserving what happened out of view, which is exactly the in-place hard case of Sections 4.1.1–4.1.2. Scaling and version claims should therefore report re-observation support and re-observed state consistency separately, and A14B denotes active-14B MoE context rather than dense parameter scaling, with the full per-row map in Appendix F.2.

Finding 5. World models need a what-memory, not just a where-memory. Every architecture caches geometry, appearance, or motion to re-render where the scene was, so how the camera is encoded is second-order, while the missing, highest-leverage component is a state writer that records what changed while hidden.

The state carrier, not the camera-conditioning format, is the architectural bottleneck: reading the full roster as a ladder shows no carrier closes the endpoint (Figure 7b). Two architectural decisions sit behind each camera-controllable row. The first is the camera-conditioning representation, whether Plücker-coordinate ray embeddings as in CameraCtrl [4], discretized or one-hot pose tokens, or direct injection of the camera-to-world (C2W) extrinsic matrix as in MotionCtrl [3]. This choice is second-order for WRBench. Camera-control rows cluster on access and requested-camera precision regardless of encoding, and within-method ablations report only a small representation effect next to the access-and-endpoint gap measured here, so how the camera is encoded is not the bottleneck. The decision that moves WRBench is the state carrier, which never closes

- as the roster climbs. Camera-lens control alone, the four Wan-Fun rows at 12–18% support, stores nothing out of view; external geometry adapters such as VerseCrafter’s GeoAdapter and Spatia’s point-cloud memory, at 26–28%, add a static layout record yet sit near the series floor on re-observed state at D6 ≈ 0.58; source-video carriers such as ReCamMaster and InSpatio,
- at 58–62%, stream appearance and dynamics to the highest access but borrow them from the

condition at D6 ≈ 0.62–0.66; explicit reappearance memory, HyDRA’s Memory Tokenizer and Dynamic Retrieval Attention, wins the highest requested-camera precision at 0.822 yet the weakest re-observed state of any row at 0.445; a dedicated state adapter, LiveWorld at 39.6%, lifts access without closing the endpoint at 0.600; and pose/action conditioning in LingBot preserves the strongest visible state of any Wan-derived row but lets support collapse to 6%. Each rung repairs the previous weak link, yet every carrier stores where to look back rather than what changed while hidden. LiveWorld’s own w/o Event Evo ablation isolates the gap: removing its generative evolution engine collapses foreground fidelity while background geometry stays intact, which confirms that a spatial memory alone never writes hidden dynamics. Read by how each carrier fails when forced to render the unobserved, the modes are distinct but equivalent in outcome. A geometry or source cache replays where the scene was, a generative-forward simulator such as LiveWorld hallucinates the subject forward and returns it duplicated, and a memory trained on synthetic exit and entry data such as HyDRA reverts to its training domain, as Appendix E.5 shows. Architecture progress should therefore target writing the event endpoint into state rather than the camera-encoding format, with the full per-row reading in Appendix F.2.

Finding 6. Endpoint persistence is an unwritten training objective, so write it with a long-to-short recipe. No public loss or documented post-training stage supervises the unobserved outcome, so we propose a long-to-short recipe: learn persistence on long horizons, then add explicit camera-execution supervision.

No public training signal writes the unobserved endpoint into state, so reading the public training ladder shows exactly where the gradient stops, as Figure 7c records, treating unreported recipes as evidence gaps rather than negative evidence. Camera-control fine-tuning on the Wan-Fun rows, whose public scope and control data are unreported, supplies lens conditioning only, so support stays low at 12.0–18.2% and scaling expands judgeable returns without improving re-observed state. ReCamMaster [32] adds the first explicit disentanglement signal, where MultiCamVideo pairs one event across ten synchronized cameras to separate viewpoint motion from scene content, so support jumps to 58.5%, yet re-observed state stays 0.616 because no label writes the hidden endpoint. Geometry and layout training, including VerseCrafter’s frozen-backbone GeoAdapter on rendered

- 4D maps, Spatia’s point-cloud memory, and InSpatio’s depth-guided source path, raises access to 25.8–62.3% while re-observed state holds at 0.58–0.66, because external structure carries where to look rather than what changed. Reappearance and state-channel training, such as HyDRA’s HM-World exit and entry events and LiveWorld’s out-of-sight adapter, moves retrieval and latent dynamics without binding the endpoint, since HyDRA reaches the top requested-camera precision but the weakest re-observed state at 0.445. LingBot [39] inverts the ladder. Its long-trajectory

world-model training holds the visible world state best of any Wan-derived row at 0.876/0.735, yet the same objective binds the camera to object dynamics, so requested displacement barely executes and support collapses to 6.0–6.4% with returned rows n=30/32. This leaves the persisted state unjudgeable exactly when WRBench would check it. We read this as a reasoned proposal rather than a demonstrated result: a long-to-short strategy that learns persistence from long horizons, then adds a shorter, controllable camera-execution objective with explicit requested-displacement supervision, so that the preserved state becomes re-observable on return. Endpoint-directed reward or policy training remains only proposed in Appendix H, and the target is the in-place endpoint of Sections 4.1.1–4.1.2, an outcome that must survive non-observation and that no current public signal writes back into state. Crucially, no documented stage is reinforcement learning on the endpoint. Every post-training recipe here is distribution-matching distillation or reward-weighted DMD, and the synthetic paired data that buys the strongest access can revert to its training domain when asked for the unobserved, as Appendices F.2 and E.5 show.

#### 4.2 Validating Human Alignment of WRBench

Human calibration grounds the diagnostic profile. The human study validates WRBench dimensions as dimension-specific comparisons rather than collapsing them into a single preference label. Agreement is strong enough to support the main claims about requested-view execution, visual integrity, re-observation support, and conditional re-observed consistency.

- Table 5 shows the calibration pattern. AC1

Table 5 Human calibration snapshot. Agree. is AC1 on the main reliability set, Bridge N counts pairs where an evaluator–human bridge exists, and Rev. counts opposite-direction threshold decisions; visual integrity uses a separate 190-pair holdout.

is high for requested-view execution, visual integrity, re-observed spatial consistency, and re-observed event-state consistency. Oppositedirection threshold decisions are rare: none for visible spatial and visible event-state, one of 136 for re-observed spatial consistency, and eight of 136 for re-observed event-state consistency. Human disagreement mainly appears as ties around small differences rather than as systematic reversal, which is exactly the pattern expected from a benchmark that separates nearby diagnostic states.

Agree. (AC1)

Bridge N

Rev. n/N

Rank ρ

Metric

View exec. 0.898 – – – Visual integrity 0.877 0.709 190 – Vis. spatial 0.788 0.386 228 0/228 Vis. event-state 0.790 0.484 229 0/229 Reobs. spatial 0.875 0.667 136 1/136 Reobs. event-state 0.937 0.660 136 8/136

This calibration is especially important for the access–re-observed-consistency separation. A video that never hides the target, or never returns identifiable evidence, cannot support re-observedstate evaluation. Once a clip does provide judgeable hidden-and-returned evidence, re-observed spatial and re-observed state consistency evaluate the endpoint itself. Human-aligned judgeability therefore makes the denominator meaningful and keeps access distinct from correctness.

The same design supports model comparison and future reward use. Sparse re-observed rows are flagged because they describe a small judgeable subset, while re-observed metrics are interpreted only on clips where the hidden-and-returned evidence exists. Preference or reward mining can use this structure directly: reward visible plausibility, camera access, and re-observed endpoint consistency as separate targets, and avoid treating a clip that skips the re-observed-state test as evidence of successful evolution consistency.

### 5 Conclusion

We introduced WRBench, a diagnostic benchmark that evaluates whether video generation models maintain evolving world state under camera-induced partial observation. Across 23 models and 9,600 generated outputs, WRBench reveals a preservation–access–re-observed-consistency gap: models can preserve visible evidence, execute plausible camera motion, or expose the target region

again, but they do not reliably maintain the event endpoint after re-observation. The shared bottleneck is endpoint binding, because a persistent world model should not merely render a plausible returned view but should write the event-induced object relation, contact endpoint, posture, containment, or collision endpoint back into the returned scene. By separating requestedcamera precision, prompt-camera alignment, visual integrity, visible spatial/state consistency, re-observation support, and re-observed spatial/state consistency, WRBench provides a compact diagnostic profile that evaluates future systems as persistent world models rather than viewconditioned renderers that master static re-rendering but not dynamic world evolution.

### References

- [1] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.
- [2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [3] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
- [4] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.
- [5] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed worldconsistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6121–6132, 2025.
- [6] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4245–4257, 2026.
- [7] Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, and Yanwei Fu. Versecrafter: Dynamic realistic video world model with 4d geometric control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 40277–40290, 2026.
- [8] Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models. arXiv preprint arXiv:2603.07145, 2026.
- [9] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [10] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [11] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.
- [12] Weixi Feng, Jiachen Li, Michael Saxon, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. Tc-bench: Benchmarking temporal compositionality in text-to-video and image-to-video generation. arXiv preprint arXiv:2406.08656, 2024.

- [13] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In International Conference on Learning Representations, volume 2025, pages 102075–102121, 2025.
- [14] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.
- [15] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.
- [16] Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, and Jiale Zhao. T2vphysbench: A first-principles benchmark for physical consistency in text-to-video generation. arXiv preprint arXiv:2505.00337, 2025.
- [17] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024.
- [18] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2vcompbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.
- [19] Yiping Wang, Xuehai He, Kuan Wang, Luyao Ma, Jianwei Yang, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Is your world simulator a good story presenter? a consecutive events-based benchmark for future long video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13629–13638,

- 2025.

[20] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. Advances in Neural Information Processing Systems, 38,

- 2026.

- [21] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27713–27724, 2025.
- [22] Xiaojie Xu, Zhengyuan Lin, Kang He, Yukang Feng, Xiaofeng Mao, Yuanyang Yin, Kaipeng Zhang, and Yongtao Ge. Worldmark: A unified benchmark suite for interactive video world models. arXiv preprint arXiv:2604.21686, 2026.
- [23] Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation. arXiv preprint arXiv:2605.25874, 2026.
- [24] Jianjie Fang, Yingshan Lei, Qin Wan, Ziyou Wang, Yuchao Huang, Yongyan Xu, Baining Zhao, Weichen Zhang, Chen Gao, Xinlei Chen, et al. iworld-bench: A benchmark for interactive world models with a unified action generation framework. arXiv e-prints, pages arXiv–2605, 2026.

- [25] Ziqi Ma, Mengzhan Liufu, and Georgia Gkioxari. Out of sight, out of mind? evaluating state evolution in video world models. arXiv preprint arXiv:2603.13215, 2026.
- [26] Shengjun Zhang, Zhang Zhang, Simin Huang, Zhenyu Tang, Hanyang Wang, Chensheng Dai, Min Chen, Yifan Li, Yuxin Li, Yingjie Chen, et al. Mbench: A comprehensive benchmark on memory capability for video world models. arXiv preprint arXiv:2606.00793, 2026.
- [27] Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. Mind: Benchmarking memory consistency and action control in world models. arXiv preprint arXiv:2602.08025, 2026.
- [28] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Advances in Neural Information Processing Systems, 36:62352–62387, 2023.
- [29] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024.
- [30] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.
- [31] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26057–26068, 2025.
- [32] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14834–14844, 2025.
- [33] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω. arXiv preprint arXiv:2605.15195, 2026.
- [34] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.
- [35] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [36] Kilem Li Gwet. Computing inter-rater reliability and its variance in the presence of high agreement. British Journal of Mathematical and Statistical Psychology, 61(1):29–48, 2008.
- [37] Klaus Krippendorff. Computing krippendorff’s alpha-reliability. 2011.
- [38] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [39] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.
- [40] Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026.
- [41] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Hujun Bao, Hongjia Zhai, Jialin Liu, Jing Guo, Nan Wang, et al. Inspatio-world: A real-time 4d world simulator via spatiotemporal autoregressive modeling. arXiv preprint arXiv:2604.07209, 2026.
- [42] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2(3):6, 2025.
- [43] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.
- [44] Guangyuan Li, Siming Zheng, Shuolin Xu, Jinwei Chen, Bo Li, Xiaobin Hu, Lei Zhao, and Peng-Tao Jiang. Magicworld: Interactive geometry-driven video world exploration. arXiv preprint arXiv:2511.18886, 2025.

### Appendix

Table of contents:

- • §A: Use of Large Language Models (LLMs)
- • §B: Reproducibility and Metric–Dataset Records
- • §C: Limitations and Future Work
- • §D: Metric–Dataset Common-Problem Checks
- • §D.1: Visual-Integrity Implementation
- • §E: Frame-Level Failure Forensics
- • §F: Model Subtype and Series Reading Guide
- • §G: Human Validation Notes
- • §H: Preference-Pair Export and Reward/Policy Outlook
- • §H.2: Dense-Control and Future Benchmark Extensions

### A Use of Large Language Models (LLMs)

Large language models were used as editorial and coding assistants for manuscript organization, wording, LaTeX editing, and consistency checks. All scientific claims, metric definitions, numerical results, tables, figures, and citations remain the responsibility of the authors. LLM assistance was not used as a substitute for benchmark generation, measurement records, human annotation, or empirical validation; any LLM-suggested prose was reviewed against the WRBench evidence dimensions and the paper’s denominator conventions before inclusion.

### B Reproducibility and Metric–Dataset Records

The main tables are generated after fixing the prompt set, model roster, metric definitions, and aggregation scripts.

- • Benchmark prompts and first frames include initial-state, event, object/action, and cameratrajectory metadata.
- • The model roster records model identifiers, conditioning interface, generation settings, and reference-video conditioning for V2V methods.
- • CamPrec, visual integrity, visible-frame metrics, re-observed-state consistency, and reobservation records are exported before aggregation.
- • Aggregation scripts generate model, family, camera-protocol, reasoning-tier, and object/action slices.
- • The human-validation protocol records sampling, annotation instructions, agreement computation, and calibration examples for ambiguous re-observation cases.

- Table 6 Metric common-problem checks. Rows state each metric’s denominator, diagnostic use, and diagnostic reading.

Metric Denominator Diagnoses Not for Strict CamPrec 7,500 local rows Requested trajectory

Prompt-only API rows

execution

Common-yaw CamAlign

7,800 yaw rows Shared yaw intent Full trajectory precision

Visual integrity 9,600 outputs Readable visual evidence Event or physics correctness Visible spatial 9,600 outputs Observable spatial errors Hidden-state consistency Visible state 9,600 outputs Observable event-state errors Hidden-state consistency Re-observation support

9,600 outputs Judgeable re-observation access

State correctness

Re-observed spatial 2,073 judgeable rows

Re-observed spatial consistency

Full-denominator persistence

Re-observed state 2,073 judgeable rows

Re-observed event-state consistency

Full-denominator persistence

### C Limitations and Future Work

WRBench targets viewpoint-conditioned dynamic world-state attribution: whether camera motion, target-relative displacement, spatial relations, and event endpoints stay bound when observability changes. Natural-25 and the evaluation dimensions are designed to make that endpoint-binding problem measurable. The next step is to use these diagnostics as a model-design loop: train systems with explicit state carriers, mine reward or policy signals from axis-level records, and extend the benchmark with finer VLM-labeled masks, boxes, and dense-control settings. Dense controls should enter with leakage-aware input policies, because target-view depth, segmentation, edge maps, or endpoint controls can reveal the post-event state and turn a world-state test into control following.

### D Metric–Dataset Common-Problem Checks

The main text analyzes WRBench through diagnostic metrics before comparing viewpoint condition types. Table 6 records the intended common-problem interpretation for each metric. The same generated-video dataset is therefore not collapsed into one score: each denominator answers a different question about where a world-state claim becomes unsupported.

#### D.1 Visual-Integrity Implementation

The visual-integrity score uses the fixed DINOv2 proxy described in Section 3.4. For each generated video v, we sample frames with a time-based sampler at 3 fps, retain the first and last frames, and cap the sequence at 24 frames. Each sampled RGB frame is resized while preserving the full field of view and padded to the DINOv2 patch grid before standard rescaling and ImageNet normalization. We avoid center cropping because WRBench scenes often place prompt-critical subjects, targets, or failure regions away from the image center. With patch size 14, the processed frame yields one CLS token and a rectangular grid of patch tokens; patches that contain only padding are masked out for local matching.

Let gt denote the ℓ2-normalized DINOv2 CLS token for sampled frame t, let pt,i denote its normalized patch token at grid index i, and let Ωt be the set of non-padding patch indices. The global component measures long-range frame-level appearance continuity over the sampled clip:

sglobal(v) = g1⊤gT 1 0

, (11)

- Table 7 Re-observed consistency for high-re-observation-support models. Rows list re-observation support and conditional re-observed consistency.

Reobs.

Model / group Condition type

support Reobs. spatial Reobs. state Gen3C Geometry-cache 73.0% 0.681 0.640 InSpatio World 14B Source-video 62.3% 0.734 0.664 ReCamMaster Source-video 58.5% 0.665 0.616 LiveWorld Model-inferred 39.6% 0.661 0.600 HyDRA Source-video 33.2% 0.509 0.445

where [·]10 clips cosine similarity to [0,1]. The local component measures whether local semantic structures can still be matched across adjacent sampled frames. For adjacent frames t and t + 1,

we compute patch-token cosine similarities only over valid patches, m(ijt) = [p⊤t,ipt+1,j]10 for i ∈ Ωt and j ∈ Ωt+1, and form a bidirectional best-match set

m(ijt)

m(ijt)

. (12)

Bt = max

∪ max

j∈Ωt+1

i∈Ωt

j∈Ωt+1

i∈Ωt

The adjacent-frame local score is the 20th percentile of Bt, and the video-level local score is again the 20th percentile over adjacent-frame local scores:

slocal(v) = P20 {P20(Bt)}Tt=1−1 . (13)

This low-tail aggregation is a heuristic that makes the score sensitive to localized collapse, ghosting, disappearance, hard cuts, or identity drift even when the median frame pair remains visually similar. The best-match formulation does not require a patch to match the same image coordinate, so it is more tolerant to object and camera motion than fixed-location patch cosine. It is still not an object detector, object tracker, or prompt-grounded subject mask.

The global term uses the DINOv2 CLS token rather than a valid-patch mean. Padding patches are excluded from local patch matching, but the CLS token can in principle attend to padded regions. We therefore include a valid-patch-mean global alternative as a sensitivity check. On the adjudicated visual-integrity bridge slice, the CLS-based headline candidate has higher exact agreement, tie accuracy, and weighted κ than the valid-patch-mean alternative, so the patch-mean global descriptor is reported only as a diagnostic and is not used for the headline visual-integrity score. The score serves as visual-integrity evidence rather than object-level state tracking or event correctness.

#### D.2 Additional Common-Problem Tables and Case Montages

To keep §4.1.1 focused on the two central claims, this appendix expands the same slices at finer granularity. Table 7 separates re-observation support from re-observed consistency, Table 13 summarizes directional access, and Tables 8 and 9 expand the event/action diagnostics.

Family-level slices follow the same reading and are therefore summarized rather than tabled: higher re-observation support changes what can be judged, but it does not make re-observed consistency automatic; state-only pressure appears as an event-design effect rather than a family leaderboard.

Selected family-by-action slices were used only for case mining. The compact appendix keeps the action-level pressure table and the per-sample montages in Figures 8 through 11, rather than adding another slice table.

- Figure 8 Directional re-observation-support contrast (Wan-Fun 2.1). A kitchen bowl-placement task (spatial event) under two yaw directions. Top (L→R): the subject is dragged with the camera and never exits, so no return is created and re-observed consistency is undefined. Bottom (R→L): the subject leaves

and re-enters, creating judgeable re-observation (D6 = 0.71). Visible spatial/state scores stay similar, so only the camera channel decides whether returned evidence exists.

- Table 8 Event-change pressure on diagnostic metrics. Read as a 2 × 2 design: none toggles neither factor, spatial only the spatial-displacement factor, state-only only the state-change factor, and full both.

Reobs.

Event change

support Integ. Vis. spatial Vis. state Reobs. spatial Reobs. state none 24.2% 0.826 0.820 0.580 0.678 0.618 spatial 22.9% 0.780 0.825 0.644 0.705 0.668 state-only 20.5% 0.779 0.697 0.541 0.601 0.547 full 18.9% 0.762 0.711 0.598 0.641 0.603

### E Frame-Level Failure Forensics

The findings in Sections 4.1.1 and 4.1.2 are summary statistics over the 9,600 evaluated clips. This appendix grounds every finding in the raw video. For each claim we fix a scene and vary exactly one factor, then read evenly spaced frames from the generated clip to show what the picture does where the metric drops. All frames are decoded directly from the released per-sample outputs (the row manifest in Appendix B); the absolute clip paths are recorded as comments next to each figure for reproducibility, and the scores in the row labels are the same per-clip D2–D6 values used in the aggregate analysis. We cover, in order: visual quality versus a correct return (Section 4.1.1, Finding 1); the in-place-change penalty and its distinct failure modes (Finding 2); camera-channel access (Section 4.1.2); and the shared in-place failure across interfaces (Section 4.1.2). Throughout, D2 is visual integrity, D3/D4 are visible spatial/state, and D5/D6 are re-observed spatial/state.

#### E.1 Visual quality does not certify a correct return

Finding 1 (Section 4.1.1) claims that a clean image, a correct camera move, and a subject that returns are three separate successes that need not compose. Figure 12 shows a single clip in which all three hold yet the return is wrong. VerseCrafter renders the meeting-room scene at high visual integrity (D2 = 0.85) and executes the requested yaw with an accurate visible layout of the conference table and chairs (D3 = 0.81), and the person re-enters the frame at the end of the sweep. But the return is wrong: the prompted event was to sit on a chair, yet when the subject comes back into view the person is still standing by the screen rather than seated, so the returned world state no longer matches the event that should have occurred while the subject was out of view, and re-observed-state collapses to D6 = 0.54. High image quality and a correct camera path are therefore certificates of an in-frame continuation, not of a correct reappearance; only the re-observed-consistency probes catch the difference. This is the per-clip face of the loosely-coupled metric structure in Figure 4.

- Table 9 Overall metric pressure by action. Rows summarize re-observation-support, visual, and re-observedconsistency metrics by action.

Vis. spatial

Reobs. spatial

Reobs. support Integ.

Vis. state

Reobs. state

Action

fold 19.1% 0.789 0.844 0.652 0.677 0.599 jump 18.0% 0.657 0.789 0.637 0.726 0.649 knock 24.2% 0.643 0.812 0.624 0.656 0.568 place 21.9% 0.810 0.791 0.605 0.673 0.627 sit 21.0% 0.819 0.747 0.574 0.642 0.604 tip 24.1% 0.706 0.787 0.575 0.717 0.640

[Figure 153]

- Figure 9 Re-observed consistency under matched re-observation (loading dock). A loading-dock boxplacement task (full event, R→L yaw) on two source-video interfaces, where both rows create judgeable re-observation. Top (Spatia): the box returns mislocated and malformed, so re-observed consistency stays low (D5 = 0.68, D6 = 0.44). Bottom (InSpatio): the box returns correctly placed on the pallet, with high re-observed consistency (D5 = 0.89, D6 = 0.82). Judgeable re-observation alone does not guarantee a correct return.

#### E.2 Why an in-place change degrades across re-observation

The in-place penalty is not a single artifact but a family of failures that share one cause, because there is no new coordinate to anchor the unobserved change. Figure 13 samples three in-place actions across strong models and finds three distinct signatures. In fold (Spatia), the person returns but the blanket comes back in an unresolved, wrong fold state. In knock (InSpatio), the most telling mode appears: the cup returns upright, as if the event never happened, because the carrier replays the last observed intact configuration and silently erases the change. In sit (HyDRA), the failure is loss of the target: the seated person dissolves during occlusion and the scene briefly collapses, then a person reappears displaced, driving re-observed-spatial to D5 = 0.27. Wrong-state, erasure, and target-loss are different surface forms of the same missing capability: binding an endpoint that was never in view.

#### E.3 Why access depends on the camera channel

Figure 14 grounds the access half of the analysis: whether a clip ever creates a hidden-thenreturned event at all. The top two rows fix a Gen3C lobby scene and flip only the yaw direction. Right-to-left (middle row) sweeps the standing person cleanly out of frame, so that the empty lobby is visible through the occlusion, and brings the person back, producing a valid re-observation; this direction yields re-observation support of 88.5%. Left-to-right (top row) does not: in this direction the prescribed yaw does not sweep the subject out of frame, so the target stays visible throughout and no hidden-then-returned event is created. This is a layout effect rather than a model failure—the subject’s starting position clears the view only under the opposite yaw—which is why Gen3C’s 0.65 retention ratio (Table 13) serves as the geometry-calibrated baseline for reading other models’ directional gaps.

The bottom row shows the opposite extreme. A prompt-only system (Kling) renders a clean,

- Figure 10 State-only re-observed contrast (Wan-Fun 2.2-A14B vs. Wan2.7). A fixed sofa-sitting task (state-only event, R→L yaw) under matched, judgeable re-observation. Top (Wan-Fun 2.2-A14B): the subject returns crouched on the open floor away from the sofa, so re-observed-state consistency stays low (D6 = 0.57). Bottom (Wan2.7): the subject returns seated at the sofa with higher re-observed-state consistency (D6 = 0.79). State-only pressure therefore persists inside re-observed-state consistency even after re-observation is available.

[Figure 155]

- Figure 11 Re-observed event-state contrast (kitchen cup knock). The same cup-knock task (spatial event, R→L yaw) on two interfaces. Top (Hunyuan GameCraft): the knocked cup returns in an inconsistent state, so re-observed event-state stays low (D6 = 0.51). Bottom (InSpatio): the cup’s re-observed state is reconstructed more consistently (D6 = 0.81). Re-observed event-state consistency varies sharply across models under a fixed event and yaw.

high-integrity clip (D2 ≈ 0.93) but executes such a timid camera motion that the reading student never leaves the frame across the entire sweep; no occlusion, no return, and therefore no judgeable re-observation. This is why prompt-only interfaces top the visible scores yet sit at the bottom of re-observation support (Section 4.1.2, Table 3): their bottleneck is not rendering quality but the failure to create the test. Together the three rows make the central access claim concrete: camera motion governs whether the hidden-state question can be asked, and that is decided by the camera channel, not by visible image quality.

#### E.4 All interfaces fail on the same in-place case

The access advantage of source-video and geometry-cache interfaces (Section 4.1.2) buys the right to be tested; it does not buy a correct return. Figure 15 fixes one in-place event, folding a blanket on a bed, and one camera path, then runs it through all four interface types. None recovers the folded state, and each fails in the signature of its own mechanism. The geometry-cache interface (Gen3C) melts the bed and blanket into warped geometry as it re-projects a cache that never observed the fold (D6 = 0.38). The source-video interface (ReCamMaster) hallucinates as the camera sweeps out, fusing two separate beds into one merged structure instead of preserving the single bed (D6 = 0.45). The model-inferred interface (Wan-Fun A14B) keeps the cleanest frame of the four (D2 = 0.89) but couples the subject to ego-motion, dragging the person along the camera direction instead of letting the fold be re-observed (D6 = 0.58). The prompt-only interface (Kling) never lets the subject leave the frame, so it produces no re-observation to score at all. Across four different mechanisms, namely geometric melt, hallucinated structure, camera–subject coupling,

0% 20% 40% 70%

90%

100%

=0.85,=0.81,=0.54DDD236

VerseCrafter,meetingroom

|[Figure 156]|
|---|

|[Figure 157]|
|---|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 12 Visual integrity is not world-state correctness (VerseCrafter). A meeting-room clip keeps high visual integrity (D2 = 0.85) and accurate visible layout (D3 = 0.81) under a correct yaw, yet the prompted sit never happens: on return the subject is still standing, so re-observed-state collapses (D6 = 0.54).

fold(Spatia)

=0.85,=0.39DD26

[Figure 162]

0% 20% 40% 55% 80%

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

|[Figure 167]|
|---|

100%

knock(InSpatio)

- =0.80,=0.44DD26

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

|[Figure 173]|
|---|

sit(HyDRA)

- =0.81,=0.27DD25

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

|[Figure 178]|[Figure 179]|
|---|---|

- Figure 13 Distinct in-place failure modes across actions. Each row is one in-place clip on a strong model under a right-to-left yaw, with the red frame marking the failure: fold returns a wrong fold state, knock returns the cup upright so the change is silently erased, and sit loses the seated person during occlusion (D5 = 0.27). Different surface artifacts share one cause: no anchor for an unobserved change.

and no test created, no interface returns the correctly folded state, so the unobserved in-place endpoint is preserved by none of them.

#### E.5 High Scores, Frame-Level Failures: The Metric Paradox

Aggregate dimension scores can rank a model at the top of one axis while a different axis quietly collapses; the headline number and the failure then live on separate axes, and only frame-level reading separates them. Three of the strongest single-axis scorers make the point (Figure 16).

HyDRA: best camera execution, worst returned world. HyDRA controls the camera better than any other row yet returns the least faithful world. It posts the highest requested-camera precision of all 23 rows (D1 = 0.822) and strong common-yaw alignment (0.855), yet the weakest re-observed state of any model (D6 = 0.445) and a low visual integrity (D2 = 0.691). The frames explain the split (Figure 16a). HyDRA’s hybrid memory is trained only on the fully synthetic, engine-rendered HM-World corpus, so it executes the prescribed camera trajectory precisely, but once the yaw exposes a region the source never carried, it fills that region from its synthetic training prior rather than the real scene, so that a backyard becomes a generic paved urban plaza with skyscrapers or the view collapses into a dark void. Precise camera control therefore coexists with a hallucinated world: the model is overfit to its training domain exactly where WRBench probes the unobserved.

Gen3C: top access, first-frame ghosting and progressive degradation. Gen3C leads re-observation support (73%) because its 3D cache re-projects the seen scene. The same reprojection pipeline, however, leaves two frame-level artifacts that the aggregate D2 (0.749) only partly reflects (Figure 16b). The first ∼0.4s is a translucent, doubled overlay while the cache initializes, and once the camera leaves the cached frustum the scene degrades into smears, darkening, and object

t=0.0s

t=1.0s

t=2.0s

t=2.8s

t=4.0s

t=5.0s

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

(dragged,57.5%)

Gen3CyawL->R

(cleanOoV,88.5%) Klingprompt-only

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

|[Figure 191]|
|---|

Gen3CyawR->L

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

(neverexits)

- Figure 14 Re-observation access is set by the camera channel. Evenly spaced frames over one camera sweep. Top (Gen3C, L→R): the camera does not swing far enough to move the subject out of frame, so no hidden-then-returned event is created (57.5% support). Middle (Gen3C, R→L): the same scene cleanly hides and returns the subject (88.5% support), the 0.65 retention of Table 13. Bottom (Kling, prompt-only):

a clean clip (D2 ≈ 0.93) whose timid camera keeps the subject framed, so no hidden-then-returned event is posed.

ghosting. Access is bought with a cache whose generative infill is not yet photometrically closed.

LiveWorld: a plausible reappearance of the wrong world. LiveWorld’s easy clips return cleanly, so its aggregate visible scores are high (visible spatial >0.9); the failure concentrates on the in-place-change events, where re-observed state falls to D6≈0.31–0.40. There the cost of its monitor-agent design becomes visible, because it rolls the backbone forward to hallucinate the out-of-sight subject rather than retrieving a stored state (Figure 16c): a single seated person is re-synthesized on return as two crouching figures, or an extra displaced body is spawned near the original, while the static room stays sharp. The subject is generated forward, not preserved, so the model returns a confident but duplicated occupant and the event endpoint is lost.

### F Model Subtype and Series Reading Guide

The analysis is organized around metric–dataset common problems, model subtype and viewpoint condition type, targeted model series, human-aligned interpretation, and future reward or policytraining use. This appendix records the assignment rules used for the condition-type and series comparisons. The generated artifacts use V2V, TI2V, and I2V labels; methods that take text plus a source/reference video are assigned to the source-video condition rather than a separate top-level family.

#### F.1 Subtype Taxonomy

The subtype taxonomy is a claim-scoping device, not a new aggregation script, and Table 10 summarizes the resulting grouping. Its primary organizing variable is the dominant viewpoint condition type: the form of input that provides information about the requested viewpoint change. Source-video rows use a reference stream for appearance, layout, and partial event evidence. Geometry-cache rows use point clouds, 3D caches, or 3D–4D controls to make parts of cameratarget access computable. Model-inferred rows have no external view-state reference and must synthesize the new view under local camera, action, or state controls. Prompt-only rows expose natural-language camera intent rather than a certified requested-control interface and therefore use CamAlign/common-yaw diagnostics rather than strict CamPrec. Camera-control interfaces and dynamic-state evidence sources are recorded as modifiers, not as peer subtype names.

The unit of assignment is the evaluated WRBench row, not the model family. If a future system

0%

20%

40% 55%

80%

100%

|[Figure 198]|[Figure 199]|
|---|---|

geom-cache:Gen3C

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

D6=0.38

src-video:ReCamMaster

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

D6=0.45 model-inferred:Wan-FunA14B

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

D6=0.58 prompt-only:Kling

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

noOoVcreated

- Figure 15 One in-place event, four interfaces, four failure signatures. The same blanket-fold event and camera path run through each interface type. Geom-cache (Gen3C): the bed and blanket melt into warped geometry (red frames, D6 = 0.38). Source-video (ReCamMaster): as the camera sweeps out, two separate beds are hallucinated into one merged structure (D6 = 0.45). Model-inferred (Wan-Fun A14B): cleanest frame (D2 = 0.89) but the subject is dragged along the camera direction (D6 = 0.58). Prompt-only (Kling): the subject never exits, so no re-observation is created. No interface recovers the unobserved in-place change.

- Table 10 Appendix viewpoint condition taxonomy. Rows are grouped by viewpoint condition type; control and state modules are modifiers.

Viewpoint condition type Representative rows Assignment rule Source-video condition ReCamMaster, HyDRA,

Externalized appearance/layout evidence.

InSpatio World 14B

Geometry-cache condition

Gen3C, Spatia, VerseCrafter Computable access; static-memory channel.

Model-inferred condition Wan-Fun, LingBot, LiveWorld, Hunyuan GameCraft, Hunyuan WorldPlay, MagicWorld

Local generation with modifiers.

Prompt-only condition Hailuo, Kling, Wan API rows, HappyHorse

CamAlign only; no strict CamPrec.

exposes multiple interfaces, it should be assigned by the interface and evidence path used in the reported WRBench run. For hybrids, an external source-video or geometry-cache condition takes precedence when it supplies the view-state information used for re-observation; otherwise local camera/action/state modules are recorded as modifiers within the model-inferred condition. Proprietary API rows remain prompt-only unless strict requested-control artifacts are available.

Representative source-video rows cite ReCamMaster [32], HyDRA [40], and InSpatio World 14B [41]. Geometry-cache rows cite Gen3C [5], Spatia [6], and VerseCrafter [7]. Model-inferred rows cite Wan-Fun [38], LingBot [39], LiveWorld [8], Hunyuan GameCraft [42], Hunyuan WorldPlay [43], and MagicWorld [44]; prompt-only product rows without paper entries are left uncited.

The series comparisons are diagnostic contrasts rather than scalar rankings. LingBot versus Wan-Fun separates visible preservation from camera-target access; Wan scaling separates the returned-evidence frontier from endpoint binding; source, memory, geometry, and local-control routes separate access mechanisms; and the Hunyuan split illustrates that interface compatibility is not the same as re-observed consistency.

[Figure 222]

- Figure 16 High scores, frame-level failures. Each row is a high scorer on one axis whose failure lives on

another. (a) HyDRA (D1 = 0.822, best; D6 = 0.445, worst): at the yaw extreme the real backyard is replaced by a generic urban plaza as the synthetic-domain training prior takes over the unobserved region. (b) Gen3C (access 73%, best): translucent first-frame ghosting (0.0–0.4s) and progressive degradation once the camera leaves the cached frustum. (c) LiveWorld (visible spatial >0.9 on easy clips; D6≈0.31–0.40 on in-place changes): the out-of-sight subject is hallucinated forward and returns duplicated, one seated person re-synthesized as two, while the room stays sharp.

- Table 11 Wan-derived model increments behind the series diagnostics. Rows are grouped by documented Wan base or reported Wan-backed release family; “Unreported” marks public-release gaps rather than negative evidence.

Evaluated model

Architecture Training update Data evidence Diagnostic mechanism

Wan2.1-T2V-1.3B Wan-Fun

Control-Camera camera-lens conditioning

Public fine-tuning scope unreported

Camera-control data unreported Camera condition only; no public camera/content disentanglement or state supervision

- 2.1-1.3B

ReCamMaster Source-video rerendering plus target-camera encoder

Camera encoder and 3D-attention layers trained; base mostly frozen

MultiCamVideo: 13,600 scenes, 10 synchronized cameras, paired-view sampling; toy→full ablation FVD 179 → 123

Same event under multiple cameras separates camera motion from scene content; paired-view diversity is the critical training factor

HyDRA Memory Tokenizer plus Dynamic Retrieval Attention

Memory and retrieval modules; 10,000 iterations; affinity-vs-FOV retrieval ablation

HM-World: 59,000 clips, fully engine-rendered (17 scenes, 49 subjects), decoupled camera/subject trajectories and exit-entry events

Reappearance memory under camera/subject decoupling; synthetic-domain prior fills the unobserved region (App. E.5); endpoint writing is separate

+0.018 subject consistency

- Wan2.1-T2V-14B

Wan-Fun

Control-Camera camera-lens conditioning

Public fine-tuning scope unreported

Camera-control data unreported Same camera condition at larger Wan2.1 scale; no new endpoint supervision

- 2.1-14B

VerseControl4D: 35,000 train / 1,000 validation; rendered RGB/depth, trajectories, masks

GeoAdapter only; Wan encoder, diffusion transformer, and decoder frozen; 3D-Gaussian-traj ablation beats bbox/point (ObjMC 2.51 vs 4.52/6.90)

VerseCrafter GeoAdapter with rendered 4D control maps

External 4D geometry supplies camera/object motion without backbone retraining; trajectories are prescribed, not learned out-of-sight state

LiveWorld Monitor-agent generative out-of-sight evolution (VACE state adapter + LoRA on frozen Wan-14B); not test-time optimization

State adapter 10,000 steps; rank-64 low-rank adapter 5,000 steps; w/o-Evo ablation VQA 59 → 18

Out-of-sight dynamics hallucinated forward by monitor rollouts; returns a duplicated subject on in-place change (App. E.5)

MIRA, RealEstate10K, and SpatiaVID_HQ examples; full mix partially public

- Wan2.1-I2V-14B-480P InSpatio World 14B

STAR video-to-video control with DA3 depth and point-cloud rendering

JDMD dual-teacher distillation (synthetic V2V + real T2V; no RL stage); 14B recipe unreported

Reference video, DA3 depth, point-cloud rendering, trajectory text; official config names the I2V-14B generator

Geometry/sourcevideo signal separates view access from internal state writing

- Wan2.2-TI2V-5B Wan-Fun

Control-Camera conditioning on dense TI2V-5B

Public fine-tuning scope unreported

Camera-control data unreported; Wan2.2 reports

Camera condition plus stronger visible prior; control and base-data effects coupled

- 2.2-5B

+65.6% images, +83.2% videos, aesthetic filtering

RealEstate 40,000 plus SpatialVID-HD 10,000; pose estimation, dynamic-object removal

Spatial memory stores layout and camera path, not dynamic event endpoints; SAM2 strips dynamics before the SLAM update, so the cloud is static-only

Spatia Persistent 3D

Control blocks about 8,000 steps; rank-64 low-rank adapter about 5,000 steps

point-cloud memory with VACE controls

- Wan2.2-I2V-A14B/MoE family Wan-Fun

Control-Camera conditioning on Wan2.2-Fun-A14B

Public fine-tuning scope unreported

Camera-control data unreported; HF metadata/config identify I2V-A14B high/low-noise MoE lineage

Camera condition under I2V-A14B/MoE prior; control data remain unreported

- 2.2-A14B

LingBot World Base-Cam pose-conditioned DiT modulation

Family-level pre-training prior; middle training for memory/action; causal and few-step post-training

Family-level data evidence: real/open videos, game logs, Unreal synthetic trajectories, synchronized camera/action data, captions; Base-Cam split unreported

Explicit per-frame pose competes with subject-continuation prior; memory is emergent from long context, with no explicit out-of-sight module

Action-conditioned state channel, distinct from camera-pose conditioning

Family-level data evidence: hybrid game/simulation action data central; act-preview split unreported

Family-level action adapters and adaptive layer normalization; main diffusiontransformer blocks frozen

LingBot Act Base-Act action adapters with continuous camera

Note. Base groups use the authors’ release-family labels when documented: T2V, I2V, TI2V, A14B, and MoE are retained only as compact base identifiers. ReCamMaster’s evaluated open port is Wan2.1-T2V-1.3B; its paper model used an internal text-to-video backbone. InSpatio’s official inference config names a Wan2.1-I2V-14B-480P generator for the 14B V2V path, while the released training recipe remains unreported. Wan-Fun 2.2-A14B is grouped with I2V-A14B/MoE because HF metadata/config identify Wan2.2-I2V-A14B high/low-noise MoE lineage, while public Fun documentation reports the release as A14B Control-Camera and leaves fine-tuning scope unreported. LingBot rows use I2V-A14B/MoE inference, but their training/data entries are family-level evidence rather than checkpoint-specific recipes. Runtime profiles such as frame count and frame rate are omitted because they describe the generation interface, not the model-side increment.

#### F.2 Series Comparison Notes

LingBot vs Wan-Fun. LingBot [39] and Wan-Fun [38] isolate the visible-preservation versus camera-target-displacement tradeoff inside local TI2V control. LingBot World and LingBot Act have the strongest local visible spatial/state scores, but their re-observation support is only 6.0%/6.4%, so their re-observed-consistency values are sparse-re-observation readings. Wan-Fun executes stronger camera displacement and reaches higher re-observation support, but its visible event scores are lower than LingBot’s. This comparison isolates how camera motion couples to target displacement across access routes.

Wan scaling diagnostic. Wan-Fun scaling provides a within-family diagnostic rather than a universal capacity rule. In both Wan 2.1 and Wan 2.2, the larger/backbone-upgraded variant improves or preserves visible rendering and raises re-observation support more clearly than it improves conditional re-observed consistency. The diagnostic reading is specific: larger backbones can expand the set of judgeable re-observed samples without automatically binding the hidden event endpoint.

Architectural state-carrier ladder (full roster). Coding each evaluated increment by its state carrier, the module that decides whether an out-of-view region can be revisited, rather than by its camera-encoding format, orders the full roster as one ladder (Table 11). Camera-lens control (Wan-Fun 2.1-1.3B/14B, 2.2-5B/A14B) aims the view but stores nothing out of sight (re-observation support 12.0–18.2%; re-observed state 0.62–0.66). External geometry adapters add a static spatial record, namely VerseCrafter’s GeoAdapter over rendered 4D maps (28.0%, re-observed state 0.584) and Spatia’s persistent point cloud (25.8%, 0.586); these sit among the lowest re-observed state in the series, above only HyDRA. Source-video carriers stream appearance and dynamics for the highest access (ReCamMaster 58.5%/0.616; InSpatio 62.3%/0.664) but borrow temporal evidence from the condition. Explicit reappearance memory (HyDRA’s Memory Tokenizer and Dynamic Retrieval Attention) earns the highest requested-camera precision (0.822) and the weakest re-observed state (0.445). A dedicated out-of-sight state adapter (LiveWorld) reaches mid access (39.6%) without closing the endpoint (0.600). Pose- and action-conditioned rows (LingBot World/Act) preserve the strongest visible spatial/state consistency among the Wanderived rows (0.876/0.735 and 0.874/0.719) while camera execution collapses (requested-camera precision 0.513/0.468; support 6.0%/6.4%; returned rows n=30/32). The ladder is monotonic in access repair but flat in endpoint binding: every carrier stores or replays where a scene can be revisited, none yet stores what resolved there while hidden, and the camera-encoding format does not predict re-observed state.

Public training-signal ladder (full roster). The same rows order by the public training signal they expose (Figure 7c); unreported recipes are gaps, not negative evidence. Wan-Fun control-camera fine-tuning (scope and data unreported) supervises camera execution only. ReCamMaster’s MultiCamVideo (13,600 scenes, ten synchronized cameras, paired-view sampling) supervises camera/content disentanglement but is tied to that paired distribution. Rendered-geometry training supervises external layout: VerseCrafter trains GeoAdapter alone with the backbone frozen (VerseControl4D, 35,000 clips), and Spatia trains control blocks and a low-rank adapter over RealEstate and SpatialVID-HD, while InSpatio’s depth/point-cloud source path keeps its 14B recipe unreported. HyDRA trains memory and retrieval on HM-World (59,000 clips with exit–entry events) for reappearance; LiveWorld trains a VACE-initialized state adapter for an out-of-sight channel. LingBot’s long-trajectory world-model curriculum most directly supervises objects evolving and reappearing within a sequence, which is why it holds visible world state best, though its re-observed metrics are sparse-re-observation readings (n=30/32). No evaluated row exposes public supervision on whether a contact, containment, posture, or collision outcome survives non-observation; that zero-count tier, together with the ReCamMaster-versus-LingBot contrast, motivates the long-to-short hypothesis and the endpoint-directed reward/policy outlook of Appendix H.

Test-time and post-training paradigm (axes beyond camera encoding). The roster also separates on two axes the camera-encoding view misses. First, test-time compute: all rows are amortized feedforward except LiveWorld, which adds runtime multi-agent orchestration, where VLM-registered monitors roll the backbone forward to hallucinate out-of-sight dynamics. This is a distinct paradigm from amortized latent/retrieval memory, and not per-instance test-time optimization. Second, post-training: every increment that post-trains does so by distribution-matching distillation (ReCamMaster/InSpatio dual-teacher JDMD, Hunyuan WorldPlay Context-Forcing, LingBot causal-plus-DMD) or reward-weighted DMD (MagicWorld), not reinforcement learning; “reward” in this literature shapes a distillation batch, not an environment-return objective, and none of it supervises the hidden event endpoint. Third, training-data domain: synthetic paired corpora (ReCamMaster’s MultiCamVideo, HyDRA’s HM-World) buy strong access and camera-alignment priors but risk a synthetic-domain fallback in unobserved regions (Appendix E.5), whereas realmonocular training (Spatia, VerseCrafter, MagicWorld) trades that for weaker geometric replay. These three axes reframe the series result: the endpoint gap is an unwritten training objective, not a property of the camera interface.

Hunyuan GameCraft vs Hunyuan WorldPlay. Hunyuan GameCraft [42] and Hunyuan WorldPlay [43] function as local-control modifiers. GameCraft’s player-action interface is partly mismatched to WRBench’s prescribed yaw and target-region re-observation protocol, while WorldPlay’s chunk memory is more compatible with re-observation pressure. WorldPlay therefore improves access relative to GameCraft, but it still does not establish re-observed consistency because scene-context memory is not the same as writing the exact contact, containment, posture, or collision endpoint into state.

#### F.3 Condition-Type Detail Notes

This appendix provides the expanded condition-interface analysis behind Section 4.1.2. The main text uses the compact evidence chain; the notes below specify what each viewpoint condition type externalizes, what advantage it buys, and why that advantage does not yet amount to hidden endpoint binding.

Source-video condition. Source-video rows externalize some world evidence into an input stream. ReCamMaster [32], HyDRA [40], and InSpatio World 14B [41] instantiate source-world transforma-

tion under a target camera path, rather than standalone prompt-only simulation. ReCamMaster and HyDRA supply a source or reference stream that already contains appearance, layout, and temporal evidence, giving a route to re-observation support that does not require an explicit point-cloud cache. The tradeoff is that dynamics are borrowed from the conditioning stream rather than inferred as a persistent hidden state. High re-observation support in this group means that the source/reference condition can make the target region visible again; returned visibility still must be checked separately for spatial and state consistency.

Geometry-cache condition. Point clouds, 3D caches, spatial memory, and 4D controls make camera-target access more computable. Gen3C [5] is the access reference because its pointcloud/3D-cache condition creates the strongest re-observation support, but its re-observedconsistency values are still not saturated. A previously observed region can be re-rendered, warped, or constrained by a stored spatial representation, which explains why high-re-observationsupport rows are dominated by geometry or source-scene conditions. The same channel clarifies the measured variable: an event endpoint that occurs while the relevant evidence is not observed still needs re-observed-state evidence. Spatia [6] and VerseCrafter [7] show the same pattern from another angle: geometry can remember where physical surfaces and viewpoints are, while the abstract endpoint state may remain unbound.

Model-inferred condition. Wan-Fun [38], LingBot [39], LiveWorld [8], Hunyuan GameCraft [42], Hunyuan WorldPlay [43], and MagicWorld [44] are grouped together because their row-level modifiers do not create a separate viewpoint condition type. Camera-lens control, pose conditioning, action tokens, chunk memory, state adapters, and history retrieval change the metric signature inside local control. Local camera-control TI2V rows synthesize the scene from a first frame and camera/action controls rather than re-rendering a source stream or a cache. Their visible evidence can be strong, and when a returned target is judgeable their conditional re-observed consistency is not necessarily the weakest signal. The recurring bottleneck is retrievable re-observation support: visible generation quality and returned-evidence availability are supplied by different mechanisms.

Prompt-only condition. Prompt-only rows are outside strict requested-control CamPrec. They preserve visible content well and pass static-hold sanity checks, but the common-yaw CamAlign diagnostic is weak and re-observed-consistency rows are often sparse. These rows show that visually plausible prompt-camera continuation can fail the camera-intent bridge. Hailuo and Kling, for example, can keep high visible spatial/state scores while rarely creating the hiddenthen-returned evidence needed to evaluate re-observed consistency; the active bottleneck is sparse returned-evidence creation rather than conditional re-observed consistency alone.

#### F.4 Directional Access Diagnostic

Yaw-direction slices require a geometry-anchored layout baseline. Gen3C shows that the benchmark layout itself can make one yaw direction easier than the other, so a raw left–right gap is not automatically a model failure. The stronger signal is when a method falls far below the geometrycalibrated retention pattern despite executing camera-like motion. The row-level diagnostic therefore supports condition-type analysis and hard-case mining, while the main text keeps the central claim on access, re-observation support, and conditional re-observed consistency.

### G Human Validation Notes

This appendix gives the calibration evidence behind Section 3.5. Human comparison families follow the same reported dimensions used by WRBench: requested-camera precision, promptcamera alignment, visual integrity, visible spatial/state correctness, re-observation support, and

- Table 13 Geometry-calibrated yaw access. Entries report re-observation support count over yaw-direction rows, followed by the percentage; retention is L→R divided by R→L.

Model R→L reobs. L→R reobs. Retention Gen3C 177/200 (88.5%) 115/200 (57.5%) 0.65 Hailuo 2.3 15/100 (15.0%) 4/100 (4.0%) 0.27 HappyHorse 1.0 I2V 6/100 (6.0%) 25/100 (25.0%) 4.17 Hunyuan GameCraft 15/200 (7.5%) 9/200 (4.5%) 0.60 Hunyuan WorldPlay 102/200 (51.0%) 0/200 (0.0%) 0.00 HyDRA 95/200 (47.5%) 38/200 (19.0%) 0.40 InSpatio World 14B 166/200 (83.0%) 83/200 (41.5%) 0.50 Kling v2.6 10/100 (10.0%) 0/100 (0.0%) 0.00 Lingbot World Cam 30/200 (15.0%) 0/200 (0.0%) 0.00 Lingbot World Act 32/200 (16.0%) 0/200 (0.0%) 0.00 LiveWorld 85/200 (42.5%) 113/200 (56.5%) 1.33 MagicWorld 62/200 (31.0%) 13/200 (6.5%) 0.21 ReCamMaster 168/200 (84.0%) 66/200 (33.0%) 0.39 Spatia 122/200 (61.0%) 7/200 (3.5%) 0.06 VerseCrafter 139/200 (69.5%) 1/200 (0.5%) 0.01

- Wan-Fun 2.1-14B 91/200 (45.5%) 0/200 (0.0%) 0.00

- Wan-Fun 2.1-1.3B 66/200 (33.0%) 3/200 (1.5%) 0.05
- Wan-Fun 2.2-5B 57/200 (28.5%) 3/200 (1.5%) 0.05

- Wan-Fun 2.2-A14B 86/200 (43.0%) 2/200 (1.0%) 0.02 Wan2.2 I2V Plus 5/100 (5.0%) 1/100 (1.0%) 0.20

- Wan2.6 I2V 26/100 (26.0%) 12/100 (12.0%) 0.46
- Wan2.7 I2V 13/100 (13.0%) 5/100 (5.0%) 0.38 WanX2.1 I2V Turbo 1/100 (1.0%) 2/100 (2.0%) 2.00

re-observed consistency. The complete annotation inventory covers 2,641 annotation rows across five semantic families; the released subset contains 2,547 deduplicated annotator verdicts over 1,156 comparison pairs. Table 14 records the release registry by annotation family, and Table 15 reports inter-annotator agreement by diagnostic dimension.

Table 16 reports the world-state evaluator–human bridge that underlies these checks. The fixed DINOv2 visual-integrity proxy is calibrated separately from the world-state VLM probes, with exact/strict/tie pairwise agreement of 0.730/0.773/0.702, weighted κ = 0.678, and rank ρ = 0.709 on a 190-pair family holdout. The world-state bridge uses 230 pairwise comparisons over 261 unique videos; the recovered 2026-06-05 measurement-available bridge has effective visible-spatial / visibleevent-state / re-observed-spatial / re-observed-event-state denominators of 228/229/136/136. These bridge denominators are distinct from the 645-pair main human-comparison reliability set and the 2,547-verdict released human-data subset.

The thresholded decision check separates ranking from calibration. Under the reported evaluator configuration, visible spatial and visible event-state metrics introduce no human-preference-toopposite reversals, while re-observed spatial and re-observed event-state contain one and eight reversals, respectively. Re-observed-metric scores remain conditional on judgeability: agreement on whether hidden-and-returned evidence exists is reported separately from conditional re-observedconsistency accuracy.

- Table 14 Human annotation release registry. Rows list annotation families, release roles, row counts, label counts, and uses. Family Role Rows Labels Use Main Reliability anchor 360 (355 active) 4-way Human-human agreement. Supp. Robustness 290 3-way Agreement checks. Src/geo Model slices 276 138 triples Source/geometry diagnostics. Broad Case mining 1418 401 triples Hard-pair evidence. Stratified Re-observed coverage 297 297 triples Re-observed-metric probes.

- Table 15 Human agreement by diagnostic dimension. Rows report κ, α, and AC1 for the main and supplementary annotation sets.

Metric Main set Supp. set

κ α AC1 κ α AC1 Requested view 0.535 0.485 0.898 0.524 0.568 0.677 Visual integrity 0.536 0.468 0.877 0.546 0.644 0.700 Visible spatial 0.316 0.214 0.788 0.188 0.307 0.481 Visible event-state 0.504 0.424 0.790 0.497 0.614 0.608 Re-observed spatial 0.632 0.560 0.875 0.403 0.716 0.880 Re-observed event-state 0.620 0.433 0.937 0.409 0.754 0.880

### H Preference-Pair Export and Reward/Policy Outlook

The same decoupled metric records can be exported as preference pairs. This appendix specifies how WRBench records can be converted into preference pairs for future reward-model or policytraining experiments. It documents the export interface only; the main paper does not use preference optimization, downstream reward-model training, DPO, or policy-improvement results as evidence. Table 17 summarizes the export counts after missing assets are removed.

#### H.1 Reward and Policy-Training Outlook

The export interface turns WRBench diagnostics into reward and policy-training inputs. A reward model can mine pairs from the same diagnostic dimensions, for example preferring stronger target-relative camera displacement when visual integrity and returned-evidence judgeability are preserved. A policy-training study can then test whether optimizing those pairs changes held-out WRBench evidence dimensions.

The present paper provides the measurement handoff: metric records, judgeability flags, pair construction rules, and dynamic-versus-dynamic filtering. Reward optimization and policy training become downstream tests over the same evidence graph. The diagnostic structure matters because rewarding visible plausibility alone could reinforce exactly the failure WRBench exposes: keeping the salient subject in frame while failing to create a judgeable hidden-and-returned world-state test.

#### H.2 Dense-Control and Future Benchmark Extensions

Dense-control world-model interfaces should be added to WRBench only with leakage-aware settings. If target-view depth, segmentation, edge maps, LiDAR, HD maps, or other dense controls already encode the post-event endpoint, a high score can reflect control following rather than hidden-state inference. This is why Cosmos-style or other dense-control systems should enter as future model clusters with explicit input-policy labels.

Three settings separate the claims:

- Table 16 World-state evaluator–human bridge. Decision κ is thresholded agreement, Rank ρ and Holdout ρ are Spearman alignment, and J, NJ, and Bal. denote judgeable, non-judgeable, and balanced judgeability agreement for re-observed metrics. Metric Evaluator agreement Judgeability agree.

Decision κ Rank ρ Holdout ρ J NJ Bal. Visible spatial 0.199 0.386 0.360 – – – Visible event-state 0.271 0.484 0.444 – – – Re-observed spatial 0.388 0.667 0.844 0.941 0.966 0.954 Re-observed event-state 0.427 0.660 0.775 0.941 0.946 0.943

- Table 17 Preference-pair export summary. Export counts after removing missing assets.

Set Input Missing Kept Stage 1 880 39 841

- Ext. 1 900 135 765
- Ext. 2 900 0 900 Total 2680 174 2506

- 1. Source-only control. Controls are extracted from the source view or source video only, so the target-view endpoint is not supplied.
- 2. Endpoint-masked target control. Target-view controls are allowed, but the object/contact endpoint region is masked or ambiguous.
- 3. Full target control. Target-view dense controls provide an upper-bound control-following condition, not evidence of internal world-state persistence.

Reward work or policy training should follow the same separation. WRBench records can mine preference pairs over target-relative camera displacement, visual integrity, judgeable re-observation, and re-observed consistency, but model-gain claims require held-out evidence after training.

