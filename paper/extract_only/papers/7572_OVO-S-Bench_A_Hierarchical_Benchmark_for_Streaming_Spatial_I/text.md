## arXiv:2606.03890v1[cs.CV]2Jun2026

[Figure 1]

2026-6-3

# OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs

###### Yifei Li1,2,†, Pengyiang Liu3,†, Yuhang Zang2,*, Zhongyue Shi3, Qi Fu3, Hongye Hao3 and Jiwen Lu1

1Tsinghua University, 2Shanghai AI Laboratory, 3Beihang University

Multimodal agents in robotics, AR, and autonomous driving must reason about places and layouts from continuous egocentric streams, often using evidence outside the current view. Existing benchmarks either evaluate offline over full videos or target events rather than spatial structure. We introduce OVO-S-Bench, a fully human-annotated benchmark for streaming spatial intelligence, comprising 1,680 questions over 348 source videos. Annotation involves 12 trained annotators (each also serving as a blind cross-reviewer) across roughly 804 person-hours of multi-round quality assurance. Each question carries a query timestamp and an evidence interval, and at evaluation, the model sees only the prefix preceding the query. Questions span four levels of increasing abstraction: instantaneous egocentric perception, spatiotemporal context tracking, spatial simulation and reasoning, and allocentric mapping. Across 38 proprietary and open-source MLLMs, Gemini-3.1-Pro trails human experts by 27 points (59.2 vs. 86.6), with allocentric mapping as the dominant bottleneck. Notably, streaming and spatially fine-tuned MLLMs underperform their own backbones. We further find that chain-of-thought reasoning amplifies spatial errors when ungrounded in the stream. By exposing these limitations, OVO-S-Bench establishes a demanding testbed for next-generation streaming spatial MLLMs.

L1 InstantaneousEgocentric Perception Understand the current view

L2 Spatial-TemporalContext Tracking

L3 GenerativeSpatial Reasoning

L4 GlobalMappingTopological

||[Figure 2]<br><br>20<br><br>40<br><br>60<br><br>80<br><br>L4.3 100 Trajectory|L1.2 Spatial|
|---|---|
|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>0<br><br>L3.2|L2.3|
<br><br>L1.3 Dynamic<br><br>L2.1<br><br>Revisit<br><br>L2.2<br><br><br>Memory<br><br>Change Temporal<br><br>L3.3<br><br>Planning<br><br>L4.1<br><br>Direction<br><br>L4.2<br><br><br>Topology<br><br>T<br><br>Human-offline 92.7 Gemini-3.1-Pro 59.3<br><br>Qwen3.5-27B 52.5 Qwen3.5-27B-Thinking<br><br>[Figure 7]<br><br>L1.1Geometry<br><br>L3.1Physics|
|---|

Build an allocentric topology

Track places after they leave view

Simulate spatial changes

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Room F

t = 234.3s t = 309.7s

[Figure 13]

Room G

figre1_1.png

Revisit

What supports the tall green plant?

What is the direction of Room G

Can the TV fit through the window after a 90° rotation?

Is the current place the same as at 234.3s?

relative to Room F?

Evidence: 0-1467s

Evidence: 311.3-313.3s

Evidence: 234.3-309.7s

Evidence: 591-593s

[Figure 14]

[Figure 15]

Revisit

[Figure 16]

[Figure 17]

[Figure 18]

53.3

Answer: same place

Answer: tabletop

Answer: Yes

Answer: southeast

GPT-5.4 51.5

Figure 1: Overview of OVO-S-Bench. The benchmark evaluates streaming spatial understanding across four levels, from instantaneous egocentric perception and spatiotemporal context tracking to generative spatial reasoning and global topological mapping. The right panel summarizes representative model behavior across task families.

###### 1. Introduction

Multimodal agents acting in the physical world must reason about space from a continuous egocentric stream. A household robot fetching an item must recall where it was last seen as the camera moves between rooms,

† Equal contribution. * Project leader. Project Page: https://internlm.github.io/OVO-S-Bench/

Table 1: Comparison with spatial and streaming video benchmarks. Stream indicates a prefix-only protocol at query time; Avg. Query Time equals full video duration for offline benchmarks and the mean query timestamp (prefix length) for streaming ones (“N/A” if not open-sourced; OST-Bench is reported in frames as it releases only sampled frames). L1–L4 follow the taxonomy in Section 3; ✓ marks tasks at that level. OVO-S-Bench is the only entry with all four levels checked under a streaming protocol; per-benchmark discussion in Appendix F.

Benchmark Venue Input Stream Avg. Query Time Source #Src. #Tasks Anno. L1 L2 L3 L4 Image / Multi-image / Top-view (no video)

EmbSpatial-Bench [9] ACL’24 Image ✗ — Indoor 3 6 Auto. ✓ ✗ ✗ ✗ TopViewRS [22] EMNLP’24 Top-view ✗ — Indoor 1 9 Manual ✓ ✗ ✓ ✓ MMSI-Bench [60] NeurIPS’25 Multi-img. ✗ — Mixed 8 11 Manual ✓ ✓ ✗ ✗

Video (offline)

VSI-Bench [57] CVPR’25 Video ✗ 1.9 min Indoor 3 8 Auto. ✓ ✓ ✓ ✗ DISJOINT-3DQA [43] EMNLP’25 Video ✗ N/A Indoor 1 5 Manual ✓ ✓ ✗ ✗ STI-Bench [25] ICCV’25 Video ✗ 0.38 min Mixed 3 8 Auto. ✓ ✓ ✗ ✗ MMSI-Video-Bench [27] arXiv’25 Video ✗ 1.2 min Mixed 26 13 Manual ✓ ✓ ✓ ✗ VSI-SUPER [59] ICLR’26 Video ✗ 56.6 min Indoor 3 2 Hybrid ✓ ✓ ✓ ✗

Video (online / streaming)

StreamingBench [29] ICLR’25 Video ✓ 4.0 min Mixed 1 18 Hybrid ✓ ✓ ✗ ✗ OVBench [18] CVPR’25 Video ✓ 5.5 min Mixed 7 16 Hybrid ✓ ✓ ✗ ✗ OVO-Bench [37] CVPR’25 Video ✓ 4.0 min Mixed 12 12 Manual ✓ ✓ ✗ ✗ OST-Bench [28] NeurIPS’25 Video ✓ 20.12 frames Indoor 3 15 Auto. ✓ ✓ ✗ ✗ ODV-Bench [64] NeurIPS’25 Video ✓ 12.6 s Outdoor 6 12 Hybrid ✓ ✓ ✓ ✗ StreamEQA [48] CVPR’26 Findings Video ✓ 12.0 min Indoor 1 42 Hybrid ✓ ✓ ✓ ✗

OVO-S-Bench (ours) — Video ✓ 8.8 min Mixed 9 30 Manual ✓ ✓ ✓ ✓

and an AR navigation assistant must guide the wearer back to a landmark they walked past minutes ago [15, 34, 35]. In each setting, the evidence required to answer a query often lies outside the current frame, requiring the agent to retain, update, and reason about spatial structure over time rather than from any single image.

Existing benchmarks leave the streaming-spatial regime untested. Spatial benchmarks study 3D relations, multi-view reasoning, and embodied question answering but assume static or offline visual context [19, 57, 60], while long-video and streaming benchmarks target event understanding, narrative memory, or response timing rather than spatial structure in a continuous visual stream [28, 29, 37, 48]. Prior spatial benchmarks grant offline access, prior streaming benchmarks treat space as a side task, and none requires building an allocentric map from a causal egocentric stream (Table 1).

We introduce OVO-S-Bench, a benchmark for streaming spatial intelligence over continuous egocentric video. Each question includes a query timestamp and an evidence interval, and at evaluation, the model sees only the prefix preceding the query, thereby simulating an online agent without access to future frames. Questions span four levels of increasing spatial abstraction: L1 instantaneous egocentric perception, L2 spatiotemporal context tracking, L3 spatial simulation and reasoning, and L4 allocentric spatial mapping (§3). Higher levels generally demand evidence that persists longer, integrates across viewpoints, and supports allocentric abstraction beyond the current frame.

Every question is written and verified by human experts. Annotators with 3D-vision backgrounds source long videos from indoor walkthroughs, egocentric activities, outdoor scenes, driving videos, and 3D environments, marking each item with options, a query timestamp, and an evidence interval. An independent group crossreviews answerability and evidence sufficiency, and a text-only LLM probe flags items solvable from option text or world knowledge alone for revision; recurring leakage patterns are folded back into the guideline (§3.2).

We evaluate 38 systems across proprietary MLLMs, general video backbones, streaming-specialized architectures, and spatially fine-tuned variants, paired with Random, Text-Only, and a stratified Human baseline (§4.2). Three findings stand out. Allocentric mapping is the dominant bottleneck: L4 is the lowest-scoring level for 28 of 34 systems, and even Gemini-3.1-Pro (59.2 overall) lags human experts by 27 points. Specialization hurts the backbone: 13 of 15 streaming and spatially fine-tuned methods score below their vanilla base, regressing most on the cross-frame levels they were designed to help. Chain-of-thought is double-edged: explicit reasoning aids the cross-frame integration L2 demands (mean ∆= + 3.9) but degrades clean current-view perception on L1 (mean ∆= − 1.0) and amplifies hallucinated spatial relations when ungrounded in the stream.

###### 2. Related Work

Benchmarking Visual Spatial Understanding. Spatial benchmarks span image, multi-image, and video regimes. Image and top-view benchmarks evaluate metric estimation, object relations, and 3D geometry on static scenes [6, 19, 22]; MMSI-Bench [60] extends to multi-image spatial intelligence; and video benchmarks add temporal observations for spatial recall [57], spatio-temporal precision [25], and dynamic camera/object motion [69]. Hierarchical efforts organize spatial abilities into progressively harder levels [53, 55]. All assume offline visual context: models can re-attend to any frame at query time, so the streaming constraint that makes spatial evidence ephemeral never surfaces.

Embodied and egocentric settings sharpen the need for persistent spatial memory. OpenEQA [35] frames embodied QA around episodic memory and active exploration, EmbSpatial-Bench [9] and Open3D-VQA [68] evaluate spatial relations in embodied or aerial environments, and DisJoint-3DQA [43] studies disjoint-frame reasoning where queried objects are not co-visible. These settings still operate over a curated context window rather than a continuous causal stream, leaving open whether models sustain a spatial model across arbitrary query timestamps.

Multimodal Models for Spatial Reasoning. Methods for spatial reasoning in MLLMs cluster around three strategies. Data scaling fine-tunes general-purpose backbones on large spatial corpora: VST pairs a 4M perception set with a 135K reasoning set under SFT+RL [58], SenseNova-SI extends the recipe to 8M samples [5], and Holi-Spatial mines spatial supervision from raw video streams [11]. Architectural augmentation injects 3D priors via spatially grounded supervision [6], dedicated spatial tokens [49], or camera-aware fusion [70]. Inference-time adaptation reshapes how the stream is consumed: Cambrian-S frames spatial supersensing as predictive world modeling with an active-memory manager [59], and Spatial-TTT adapts fast weights at test time [30].

Streaming Video Understanding. Streaming video understanding has two arms. Long-form benchmarks, including EgoSchema, LongVideoBench, and LVBench [36, 46, 51], test temporal understanding under offline access. Streaming benchmarks, including StreamingBench, OVBench, OVO-Bench, OST-Bench, and VCBench [18, 28, 29, 31, 37], enforce causal input and bounded memory but target event understanding, multi-turn interaction, or counting rather than spatial structure. Streaming MLLMs respond with memory banks, token pruning, KV-cache management, and event-segment memories [39, 40, 54, 61, 64–66], and recent work asks whether such designs beat sliding-window baselines [44]. All target event memory; persistent spatial evidence over a continuous egocentric stream is the dimension OVO-S-Bench makes empirical for the first time.

###### 3. OVO-S-Bench

OVO-S-Bench organizes questions into four levels by the spatial state a model must access at query time (Figure 3). The levels progress from evidence directly available in the current view to allocentric map queries that require cross-viewpoint integration, reflecting a gradient of persistence and abstraction. Each level contains several task families and canonical task types; full definitions and examples are provided in Appendix D.

###### 3.1. Four-Level Streaming Spatial Taxonomy

- L1: Instantaneous Egocentric Perception. L1 tests whether a model understands the spatial structure currently in view. Questions can be answered from the frames near the query timestamp alone, without recalling any past observation. Task families cover egocentric metric perception (distance, scale, clearance, viewpoint height), local spatial relations (containment, occlusion, support, visible layout), and dynamic spatial perception (camera motion, object motion, relative speed).
- L2: Spatiotemporal Context Tracking. L2 evidence has appeared in the video prefix but is no longer visible at query time. The model must keep spatial facts bound to specific places, objects, and timestamps after visual support disappears. Task families include scene revisit recognition (whether a place has been seen and how

L1: Instantaneous Egocentric Perception

###### L4.2: Topology

L4.1: Direction Inst./Scene Spatial Attribute

Inst./Scene Spatial Attribute

Geometry Inst./Scene Spatial Attribute

Spatial Inst./Scene Spatial Attribute

###### Dynamic

L1.1: Geometry Inst./SceneSpatialAttribute

L1.2 Spatial Inst./Scene Spatial Attribute

###### L1.3 Dynamic

Inst./Scene Spatial Attribute

Inst./Scene Spatial Attribute

[Figure 19]

[Figure 20]

[Figure 21]

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

[Figure 32]

[Figure 33]

[Figure 34]

In the current view, what object is directly supporting the tall green plant?

###### If facing north when entering the main door, what direction is the TV from the fridge?

###### What is the main connection between Area D and Room E?

###### Approximately how high is the camera above the ground?

In the current view, what object is directly supporting the tall green plant? Anwser: The tabletop

approximately how high is the camera above the ground?

###### What is the camera's dominant translation direction?

how many times has camera visited this living room?

Anwser: A small door or entrance leading directly into Room E

Anwser: About 0.5m

Anwser: Downward

Anwser: The tabletop

Anwser: About 0.5m

Anwser: Downward

Anwser: west

L2: Spatial-Temporal Context Tracking

L4.3: Trajectory

L4.3: Trajectory Inst./Scene Spatial Attribute

Inst./Scene Spatial Attribute

Inst./SceneMemorySpatial Attribute

Inst./SceneTemporalSpatial Attribute

Inst./SceneL2.2: SpatialMemoryAttribute

Inst./SceneL2.3: TemporalSpatial Attribute

Inst./SceneRevisitSpatial Attribute

Inst./SceneL2.1:SpatialRevisitAttribute

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

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

###### Based on the video, which bird's-eye view corresponds to the scene shown?

Based on the video, which bird's-eye view corresponds to the scene shown?

What order does the protagonist visit the rooms after entering?

###### What color is the statue that is directly behind the camera?

###### What order does the protagonist visit the rooms after entering?

what color is the statue that is directly behind the camera?

###### How many times has camera visited this living room?

how many times has camera visited this living room?

Anwser: Hallway->bathroom>bedroom

Anwser: Hallway->Bathroom->Bedroom

Anwser: White

Anwser: Four times

Anwser: White

Anwser: Four times

Anwser: A photo as shown below

Anwser: A photo as shown below

L3: Generative Spatial Reasoning

[Figure 55]

[Figure 56]

Physics Inst./Scene Spatial Attribute

Change

L3.1: Physics Inst./Scene Spatial Attribute

L3.2: Change

L3.3: Planning

Inst./Scene Spatial Attribute

Inst./Scene Spatial Attribute

Inst./Scene Spatial Attribute

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

What new object appears on the car now, vs. the prior moment?

###### Can the TV pass the left window after a 90° vertical rotation?

What new object appears on the car now, vs. the prior moment?

Can the TV pass the left window after a 90° vertical rotation?

###### From A to Billiard Table, which route is the shortest or most reasonable?

Anwser: True

Anwser: A Golden Retriever

Anwser: Bathroom A->Room B->Corridor C->Living room->Recreation Room->Billiard Table

Anwser: Yes

Anwser: A Golden Retriever

L4: Global Topological Mapping

- Figure 2: Representative OVO-S-Bench examples. Each card pairs a spatial question with visual evidence, illustrating the progression from current-view perception to allocentric mapping.

often), spatial memory beyond the view (location and state of out-of-view entities), and chronological spatial memory (visit order and temporal-depth comparisons).

- L3: Spatial Simulation and Reasoning. L3 requires the model to operate on spatial structure rather than merely retrieve an observation. Questions involve mental rotation, state-change inference, and path feasibility; evidence spans vary by task, from local for simulation items to extended for route planning. Task families cover spatial simulation (reorientation, removal consequences, physical feasibility), spatiotemporal consistency verification (whether a state has changed between observations), and spatial route planning (route selection, detours, reachability).
- L4: Allocentric Spatial Mapping. L4 requires the model to integrate the egocentric stream into an allocentric representation and query its global structure. Evidence typically spans multiple viewpoints and may cover the entire explored region. Three task families target allocentric direction reasoning (global bearings among rooms or landmarks), topological structure reasoning (adjacency, connectivity, boundaries, and transitions), and trajectory-map alignment (matching the first-person path to a bird’s-eye map).

- 3.2. Benchmark Construction

Video sources. OVO-S-Bench draws from 9 publicly available or accessible sources covering five regimes: indoor walkthroughs (RoomTour3D [16]), egocentric activities (Ego4D [15]), outdoor/world scenes (Sekai [26], OmniWorld [71], and YouTube walking tours), driving videos (CODa [24] and Honda HDD [42]), and spatially annotated 3D environments (ARKitScenes [4] and VSI-Bench [57]).

Human annotators write every item. Auto-generated captions and detections rarely capture the spatial evidence our targets require. Annotators choose clips from these sources with stable motion, clear viewpoints, and enough spatial variation for the target level. For each item, they record the video, task label, question, options, answer, query timestamp, and evidence interval. Some task types employ specialized construction techniques such as image editing to generate spatial-change contrasts.

Each item follows the streaming setting. The answer must be derivable from the video prefix before the query timestamp. Annotators mark the shortest interval that contains the needed evidence and write distractors that are plausible under the visual context but wrong under the annotated evidence.

Quality control removes shortcuts. We first run a text-only model to flag items that leak the answer through

OVO-S Spatial Taxonomy (b) Question counts per level

###### (a)

Trajectory

Alig n

olute

e

pth

e

nc

L1

T

elativ

ale

e

- o
- p olo

g

Dista

(614) 1.1 Egocentric Metric 158 1.2 Local Spatial Relations 193

1.3 Ego-/Object Motion 263

Structure

m e

D

erin

bs

wpoint Height

Sc

bject

A

R

nt

Ord

###### L2

gical

2.3 Chronological Memory 147

(513) 2.1 Visual Scene Re-ID 198 2.2 Out-of-View State 168

Vie

O

ment

Trajectory

Alocentric Direction

###### L3 (279)

Map Align-

Contain

###### 3.1 Mental Simulation 150

PeErgMcoeecptertiincotnric

3.2 63 3.3 Path 66

Topological Structure

ment

Occlusion

###### L4 (274)

###### 4.1 Allocentric Direction 131

4.2 Spatial 90 4.3 53

Alocentric

Reachability

Direction

| | | | |
|---|---|---|---|
| | | | |

ROerbosuttaincgle

0 200 400 600 Number of questions

Support

L ocal

S patial Relatio ns

OPpattihmal

L4: Map Queries

(c) Source distribution per level

L1: Cu

atial R

g

ute Pla

Local Layout

nin

L1 (614)

###### L2 (513)

###### L3 (279)

###### L4 (274)

- o

n

si

-ste

n

c

y

S

- p

o

n

Perception

DCetheacntigoen

RoomTour3D 163

RoomTour3D 174 Ego4D 166

RoomTour3D 308 Ego4D 113

RoomTour3D 159

atiote

rent-Vie w

oral C

s

atial

Ego4D 74 YouTube 17

Honda 59 ARKitScenes 33

p

n

p

m

Ego Translation

eratio

S

CODa 112 OmniWorld 111

Sekai 53 OmniWorld 22

-

p

Physical Feasibility

CODa 17 OmniWorld 8

OmniWorld 12

S

Si

3:

Sekai 32 VSI-Bench 19

CODa 17

Ego4D 9 CODa 2

p

S p

- m

ulatio

- n

mic S p

n

O

L

ptio

atial P

atial

0 300

0 300

Spatial What-If

a

n

e

Ego Rotation

0 300

0 300 Number of questions

y

erc

L2: Out-ofView

D

Mental Rotation

###### (d) Evidence-interval length per level (s)

Chronological

Spatial Me

MOobtiojenct

Memory

& Fiting

Temporal

1,000

Relative Speed

Depth

mory

Scene Revisit Recognition

SBpeaytoianld M Veimewory

###### 229 s

Motion State

Chronological

100

Recal

###### 41 s

Time (s, log)

Revisit Detection

mory

Transient Count

10

Me

d-

era

e

w

n

C o u

R evisit

nc

hin

atio

ut-of-Vie

m

Infere

###### 2 s

###### 2 s

e

ntin g

a

aliz

B

C

1

L1 (n=614)

L2 (n=513)

L3 (n=279)

L4 (n=274)

oc

O

L

- Figure 3: Taxonomy and benchmark statistics for OVO-S-Bench. The left panel gives the four-level spatial taxonomy, while the right panels report task-family counts, source distribution, and evidence-interval lengths by level.

wording, common sense, option asymmetry, or a single lexical cue; this matters most for L2 through L4, where room priors and route stereotypes create shortcuts. We then probe the item under the intended streaming setting to expose ambiguity, easy visual shortcuts, and timestamp errors. A second annotator reviews the item without seeing the original answer and checks whether the answer and evidence interval are sufficient. Disagreements are revised, and recurring problems are folded back into the guideline. The full protocol, source details, and construction techniques are in Appendix E.

- 3.3. Benchmark Statistics

The released benchmark comprises 1,680 questions over 348 source videos from 9 datasets, organized into 30 canonical task types across four levels. The mean prefix at query time is 8.8 minutes. Evidence-span duration varies by level (L1 median 2.0s, L2 36.8s, L3 2.0s, L4 278.7s), reflecting the spatial persistence each level demands. Figure 3b–d reports per-level task-family counts, source distribution, and evidence-interval lengths; L4 is reported as three map-query families.

- 4. Experiments

- 4.1. Evaluation Setup

All systems are evaluated under the same streaming protocol: each source video is truncated at the annotated query timestamp 𝑡𝑞, and the model receives 128 frames uniformly sampled from the resulting prefix together with the question and multiple-choice options. For streaming-architecture models that implement a native sequential ingestion path, we instead feed the video at each model’s published streaming rate and query the resulting compressed state (Appendix G.4). This prefix-only setup follows OVO-Bench and OST-Bench [28, 37], so no model sees frames after 𝑡𝑞. Proprietary models are queried through their official APIs, all other models run locally with their published defaults, and answers are extracted by regular expression without further post-processing.

We evaluate 38 systems (Tab. 2) spanning seven families: proprietary MLLMs (GPT-5.4 [38], Gemini-3.1Pro/Flash-Lite [12, 13], Grok-4.1-Fast [52]); open-source general backbones (InternVL-3.5 [47], Qwen2.5-

- Table 2: Main results on OVO-S-Bench (streaming protocol, multiple-choice accuracy). R is rank; top-three level ranks are shaded gray and overall ranks green (darker is better); bold marks the best per column; baselines and controls are unranked. *L4.3 uses image options unsupported by InfiniPot-V.

L1: Instant. Ego. Perc. L2: Context Tracking L3: Spatial Reasoning L4: Allocentric Mapping

Model Params

Avg. Rank 1.1 1.2 1.3 Avg. R 2.1 2.2 2.3 Avg. R 3.1 3.2 3.3 Avg. R 4.1 4.2 4.3 Avg. R

Baseline & Controls

Text-Only (GPT-5.4) – 38.64 41.37 35.21 38.41 – 36.01 30.08 40.78 35.62 – 54.00 9.52 53.03 38.85 – 31.68 54.17 20.75 35.53 – 37.10 – Random Baseline – 33.93 25.49 30.00 29.81 – 37.44 26.77 41.08 35.10 – 50.00 25.00 25.00 33.33 – 25.00 31.25 25.00 27.08 – 31.33 – Human (offline) – 97.47 97.93 95.44 96.95 – 91.41 81.55 85.71 86.22 – 93.33 95.24 93.94 94.17 – 86.14 98.33 83.02 89.16 – 92.20 – Human (streaming) – 91.77 95.85 92.02 93.21 – 86.87 77.98 78.23 81.03 – 88.67 87.30 83.33 86.43 – 79.21 86.67 71.70 79.19 – 86.61 –

Closed-source proprietary MLLMs

GPT-5.4 – 52.08 63.70 48.13 54.64 4 46.25 57.92 68.61 57.59 2 61.33 36.51 54.55 50.80 9 32.67 62.50 26.42 40.53 17 50.89 5 Gemini-3.1-Pro – 58.69 71.18 55.89 61.92 1 57.92 59.97 74.23 64.04 1 67.33 41.27 59.09 55.90 3 60.40 81.67 22.64 54.90 1 59.19 1 Gemini-3.1-Flash-Lite – 51.19 63.38 47.70 54.09 5 47.98 47.61 60.98 52.19 8 58.67 49.21 54.55 54.14 5 35.64 68.33 24.53 42.84 7 50.81 7 Grok-4.1-Fast – 48.34 49.95 36.13 44.80 17 42.26 42.03 55.48 46.59 19 56.00 34.92 54.55 48.49 15 24.75 55.83 24.53 35.04 25 43.73 19

Open-source general video MLLMs InternVL-3.5 8B 47.86 54.60 35.31 45.92 13 40.95 40.31 56.22 45.83 21 54.00 26.98 60.61 47.20 18 34.65 55.00 28.30 39.32 18 44.57 16 InternVL-3.5 38B 58.64 55.56 49.74 54.65 3 50.30 48.00 64.91 54.40 7 61.33 22.22 53.03 45.53 22 38.61 68.33 18.87 41.94 10 49.13 8 InternVL-3.5 241B-A28B 61.30 59.36 45.98 55.55 2 48.75 49.86 68.54 55.72 3 53.33 33.33 68.18 51.62 8 42.57 65.83 13.21 40.54 16 50.85 6

- Qwen2.5-VL 7B 42.89 40.26 38.87 40.67 28 40.24 36.31 59.98 45.51 22 54.67 23.81 59.09 45.86 21 47.52 65.83 20.75 44.70 5 44.19 17 Qwen3-VL 4B 32.91 53.19 43.65 43.25 20 47.26 34.25 63.04 48.18 14 57.33 38.10 68.18 54.54 4 25.74 70.00 28.30 41.35 11 46.83 10 Qwen3-VL 32B 48.83 57.07 44.51 50.14 8 50.71 34.58 70.43 51.91 9 60.67 26.98 68.18 51.94 7 32.67 68.33 22.64 41.22 12 48.80 9 Qwen3-VL 235B-A22B 50.36 56.09 50.91 52.45 6 49.76 45.61 70.17 55.18 5 63.33 46.03 74.24 61.20 1 45.54 70.83 20.75 45.71 3 53.64 2

- Qwen3.5 4B 40.28 51.13 44.82 45.41 15 47.44 34.19 62.98 48.20 13 52.00 39.68 56.06 49.25 14 27.72 65.83 22.64 38.73 19 45.40 13 Qwen3.5 9B 43.25 55.03 44.13 47.47 11 44.52 38.53 65.05 49.37 12 56.67 33.33 60.60 50.20 11 25.74 63.33 20.75 36.61 22 45.91 12 Qwen3.5 27B 46.50 58.13 49.88 51.50 7 55.24 41.53 68.67 55.15 6 54.00 36.51 66.67 52.39 6 35.64 71.67 35.85 47.72 2 51.69 4 Qwen3.5 397B-A17B 46.71 53.32 48.68 49.57 9 55.65 41.22 69.30 55.39 4 57.33 38.10 78.79 58.07 2 34.65 75.00 26.42 45.36 4 52.10 3 Gemma-4 E2B 42.12 42.36 31.93 38.80 33 35.00 31.97 42.60 36.52 35 55.33 12.70 50.00 39.34 33 28.71 37.50 22.64 29.62 35 36.07 36 Gemma-4 E4B 41.33 47.33 34.10 40.92 26 41.43 38.69 48.16 42.76 30 52.00 14.29 62.12 42.80 27 24.75 47.50 24.53 32.26 32 39.69 30 Gemma-4 26B-A4B 47.90 56.23 43.86 49.33 10 47.26 43.31 49.10 46.55 20 62.67 25.40 46.97 45.01 25 19.80 55.00 13.21 29.34 36 42.56 24 GLM-4.6V-Flash 9B 41.93 56.01 35.87 44.60 19 43.81 43.36 56.79 47.99 16 48.67 34.92 56.06 46.55 19 28.71 57.50 15.09 33.77 28 43.23 22

Streaming video MLLMs

Flash-VStream 7B 15.26 22.70 17.99 18.65 38 31.85 21.61 36.34 29.93 38 35.33 9.52 22.73 22.53 38 14.85 54.17 16.98 28.67 38 24.94 38 StreamForest 7B 47.69 56.74 35.48 46.64 12 42.74 36.64 56.22 45.20 24 60.00 30.16 59.09 49.75 12 26.73 59.17 18.87 34.92 26 44.13 18 StreamingVLM 7B 40.29 41.43 34.25 38.65 35 47.74 39.72 64.11 50.52 10 46.67 28.57 50.00 41.75 30 38.61 64.17 20.75 41.18 13 43.03 23

Token-compression and memory-based methods

InfiniPot-V 7B 38.66 41.66 36.82 39.05 31 34.82 26.81 45.53 35.72 36 56.67 17.46 51.52 41.88 29 33.66 47.50 –* 40.58 15 39.31 32 HERMES 7B 42.00 46.35 34.29 40.88 27 40.48 36.92 58.85 45.41 23 56.00 30.16 62.12 49.43 13 41.58 70.00 16.98 42.86 6 44.64 15 FluxMem 7B 41.46 44.58 42.89 42.98 22 43.33 39.36 59.98 47.56 17 53.33 28.57 54.55 45.48 24 48.51 64.17 15.09 42.59 9 44.65 14 StreamingTOM 7B 35.60 43.34 32.79 37.24 36 50.48 37.14 56.91 48.18 14 51.33 23.81 40.91 38.68 34 23.76 54.17 22.64 33.52 29 39.41 31

Spatially fine-tuned MLLMs

Spatial-MLLM 7B 34.39 34.35 38.41 35.72 37 40.00 29.03 48.53 39.19 33 54.00 17.46 31.82 34.43 36 23.76 55.00 30.19 36.32 23 36.41 35 Cambrian-S 7B 42.79 43.75 34.14 40.23 29 38.57 32.00 49.53 40.03 32 54.00 15.87 40.91 36.93 35 18.81 50.00 20.75 29.86 34 36.76 34 Cambrian-S-LFP 7B 38.72 42.14 35.62 38.82 32 37.80 25.61 50.72 38.04 34 52.00 14.29 36.36 34.22 37 22.77 42.50 20.75 28.68 37 34.94 37 VST-7B-SFT 7B 48.70 42.92 38.11 43.25 20 41.85 28.69 61.41 43.98 27 50.67 26.98 53.03 43.56 26 24.75 62.50 26.42 37.89 21 42.17 26 VST-7B-RL 7B 51.07 44.82 41.10 45.66 14 44.23 29.03 59.35 44.20 25 52.00 20.63 50.00 40.88 32 26.73 60.83 26.42 37.99 20 42.18 25 SenseNova-SI-1.5 8B 43.99 44.67 37.74 42.13 25 45.83 27.69 53.66 42.40 31 54.00 31.75 42.42 42.72 28 21.78 42.50 33.96 32.75 30 40.00 29 Spatial-TTT 2B 47.25 39.84 29.08 38.73 34 40.24 23.92 42.16 35.44 37 48.67 38.10 36.36 41.04 31 23.76 44.17 30.19 32.71 31 36.98 33

Embodied foundation models

Cosmos-Reason1 7B 48.20 43.85 42.31 44.79 18 36.85 35.94 58.29 43.69 28 55.33 20.63 60.61 45.52 23 30.69 50.00 15.09 31.93 33 41.48 28 VeBrain 7B 43.88 45.36 38.98 42.74 24 40.24 32.25 60.04 44.18 26 54.00 25.40 59.09 46.16 20 42.57 66.67 13.21 40.82 14 43.47 21 RynnBrain 8B 45.20 49.42 41.33 45.32 16 49.29 38.94 62.54 50.26 11 54.00 33.33 54.55 47.29 17 35.64 64.17 28.30 42.70 8 46.39 11 RoboBrain2.5 4B 36.70 45.60 38.04 40.12 30 42.92 31.64 55.72 43.43 29 52.00 30.16 62.12 48.09 16 22.77 54.17 30.19 35.71 24 41.84 27 RoboBrain2.5-NV 8B 34.68 51.38 42.53 42.86 23 45.95 35.56 58.41 46.64 18 56.67 28.57 66.67 50.63 10 26.73 49.17 26.42 34.10 27 43.56 20

VL [3], Qwen3-VL [2], Qwen3.5 [41], Gemma-4 [14], GLM-4.6V-Flash [63]); streaming video MLLMs (FlashVStream [65], StreamForest [64], StreamingVLM [56]); token-compression and memory methods (InfiniPotV [20], HERMES [67], FluxMem [54], StreamingTOM [7]); spatially fine-tuned MLLMs (Spatial-MLLM [49], Cambrian-S±LFP [59], VST-SFT/RL [58], SenseNova-SI [5], Spatial-TTT [30]); and embodied foundation models (Cosmos-Reason1 [1], VeBrain [32], RynnBrain [8], RoboBrain2.5 [45]). We anchor accuracy with three baselines: Random, Text-Only (GPT-5.4 with question and options only), and a Human baseline evaluated by five independent participants on the full question set under both the streaming protocol (participants view the prefix before the question is revealed at 𝑡𝑞) and an offline protocol (participants receive the question alongside the full video and may re-scan freely).

- 4.2. Benchmark Results

- Tab. 2 supports four observations about the current state of streaming spatial intelligence.

Significant Gap with Human Performance. The strongest system Gemini-3.1-Pro, reaches 59.2, far below human experts under the same streaming protocol (86.6, or 92.2 offline). The best open-source model Qwen3VL-235B-A22B attains only 53.6, trailing human-streaming by 33 points. Random (31.3) and Text-Only (37.1) baselines fall below all general-backbone and proprietary systems, confirming that the human-model gap reflects genuine visual-streaming difficulty rather than language priors.

Model ΔOverall ΔL1 ΔL2 ΔL3 ΔL4 GLM-4.6V-Flash -0.9 +1.7 -1.1 -2.2 -1.9 InternVL-3.5-8B +0.3 -1.3 +6.0 -1.8 -1.7 InternVL-3.5-38B 0.0 -7.7 +1.7 +3.2 +2.7 Qwen3-VL-4B -2.5 -1.8 +0.1 -4.1 -4.3 Qwen3-VL-32B +1.0 -0.8 +3.5 +2.3 -1.2 Qwen3.5-4B -1.3 -4.9 +4.3 -6.1 +1.5 Qwen3.5-9B +6.7 +3.1 +9.4 +4.3 +10.0 Qwen3.5-27B +4.3 -0.5 +6.3 +1.5 +10.0 Grok-4.1-Fast +3.2 +3.2 +5.2 -0.3 +4.5

n=929

GLM-4.6V

n=904

InternVL-8B

n=849

InternVL-38B

n=896

Qwen3-VL-4B

n=836

Qwen3-VL-32B

n=947

Qwen3.5-4B

n=840

Qwen3.5-9B

n=734

Qwen3.5-27B

0 200 400 600 800 1000

Wrong-trace count

No-conclusion

Non-visual

Visual-content

Direction

Temporal-binding

- Table 3: Thinking-mode versus non-thinkingmode variants on OVO-S-Bench. ∆ is thinking minus non-thinking accuracy.

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

Figure 4: Wrong-trace counts (bar length) and errorclass composition (segments) across eight thinkingmode models.

Allocentric Mapping Bottleneck. L4 is the lowest-scoring level for 28 of 34 systems, with an average gap of 9.3% between L1–L3 and L4 and gaps above 10% even on the largest open-source backbones (Qwen3-VL235B-A22B: 10.6, InternVL-3.5-241B-A28B: 13.8). The six exceptions all have L1 below 41 (Qwen2.5-VL-7B, Flash-VStream, and four others), so their flipped ordering reflects degraded current-view perception rather than competent allocentric mapping. Among L1–L3 the ordering is unstable (29 of 34 systems peak at L2 or L3): although evidence demand grows monotonically from L1 to L4, accuracy does not, and drops sharply once an allocentric map must be abstracted from the entire explored region.

Narrow but Uneven Closed-Source Lead. The closed-source advantage is only 5.6 points overall (Gemini-

- 3.1-Pro 59.2 vs. Qwen3-VL-235B-A22B 53.6), narrower than the 10+ point gap reported on recent video and multimodal benchmarks [10]. The gap is uneven across levels: closed-source widens its lead on the memory-heavy L2 (+5.9, above the +5.6 overall) but narrows on L4 (+4.1), and on L3 the best open-source backbone exceeds Gemini-3.1-Pro by 5.3 points (61.2 vs. 55.9). The closed-source group is itself heterogeneous: GPT-5.4 (50.9) and Grok-4.1-Fast (43.7) trail Gemini-3.1-Pro by 8 and 15 points, suggesting streaming spatial competence is not yet a generic capability of frontier multimodal systems.

Specialized Methods Lag Behind Backbones. No streaming-architecture or spatially fine-tuned variant outperforms its comparable general backbone, and 13 of 15 lag behind their own base. Streaming MLLMs are tuned to compress memory for narrative QA, and spatial fine-tuning [5, 49] is tuned on discrete-frame QA; neither regime exercises persistent spatial state over a continuous prefix, consistent with the L2–L4 losses we measure. We dissect this further in Section 4.3.3.

- 4.3. Analysis Experiments

###### 4.3.1. Effect of Thinking Mode

- Tab. 3 compares thinking-mode and non-thinking-mode variants for nine models, where the paired delta isolates the effect of explicit chain-of-thought.

Model-Dependent Thinking Gains. Gains do not scale with backbone size: mid-size Qwen3.5 posts the strongest gain (from −1.3 at 4B to +6.7 at 9B), while Qwen3-VL and InternVL-3.5 hover near zero, Grok-4.1-Fast rises by +3.2, and GLM-4.6V-Flash falls by −0.9. Across levels, thinking consistently helps L2 (mean ∆ = +3.9, 8/9 pairs positive) and shows a small mean drop on L1 (mean ∆ = −1.0, 6/9 pairs negative), suggesting chain-of-thought aids the cross-frame integration but offers little for clean current-view perception.

Probing Thinking-Mode Failure Patterns. We probe failure modes of thinking-mode models in Tab. 3 by submitting every wrong-answer trace, together with the question, options, ground-truth, and predicted letters, and 4–6 frames sampled at midpoints of the annotated evidence intervals, to GPT-5.4 as a judge. The judge assigns one of five labels: no-conclusion error (no extractable letter), non-visual error (commits via world priors without citing visible content), visual-content error (asserts a spatial relation the frames do not show), direction error (correct objects, inverted L/R or F/B), and temporal-binding error (correct object, wrong moment or visit). Three votes per trace are aggregated by majority. Please refer to Appendix C for full setup and cross-judge calibration. Across the eight thinking-mode models (Fig. 4), failures concentrate on mis-grounded visual evidence: non-visual and visual-content errors jointly cover 60–80% of wrong traces in GLM-4.6V-Flash,

- Table 4: Frame-sampling sensitivity on a five-model representative subset (Overall accuracy; bold per row). Policy definitions and the full policy / model matrix are in Appendix A.

Model 1@q 16f@4fps U-128 U-256 log-128 oracle Gemini-3.1-Flash-Lite 47.95 49.72 50.88 48.33 48.55 49.70 Qwen3-VL-32B 43.95 45.80 48.71 48.93 51.40 48.80 Qwen3.5-27B 47.49 48.42 51.56 50.64 51.74 51.87 VST-7B-RL 41.24 39.10 42.13 41.99 44.35 42.13 RoboBrain2.5-NV-8B 39.91 41.22 43.56 43.87 46.84 43.56

- Table 5: Specialized methods versus their base backbones on OVO-S-Bench. ∆ is specialized minus base accuracy.

Method Base ΔOverall ΔL1 ΔL2 ΔL3 ΔL4 Spatially fine-tuned

Spatial-MLLM-7B Qwen2.5-VL-7B -7.70 -4.82 -6.16 -11.43 -8.39 VST-7B-SFT Qwen2.5-VL-7B -1.98 +2.57 -1.36 -2.30 -6.82 VST-7B-RL Qwen2.5-VL-7B -1.98 +4.92 -1.15 -4.98 -6.71 Spatial-TTT-2B Qwen3-VL-2B -3.54 -1.81 -10.12 -4.28 +2.04 SenseNova-SI-1.5 InternVL-3.5-8B -4.55 -3.72 -3.43 -4.47 -6.57

Streaming architectures

StreamingVLM-7B Qwen2.5-VL-7B -1.14 -2.09 +5.17 -4.11 -3.53 Flash-VStream-7B Qwen2-VL-7B -18.35 -21.69 -14.68 -20.28 -16.73

Token-compression and memory

FluxMem Qwen2.5-VL-7B +0.45 +2.23 +2.05 -0.37 -2.11 HERMES Qwen2.5-VL-7B +0.44 +0.14 -0.09 +3.57 -1.85 StreamingTOM Qwen2.5-VL-7B -4.78 -3.43 +2.67 -7.18 -11.18

Embodied foundation models

Cosmos-Reason1-7B Qwen2.5-VL-7B -2.63 +4.25 -1.66 -0.33 -12.78 VeBrain-7B Qwen2.5-VL-7B -0.64 +2.21 -1.17 +0.31 -3.89 RoboBrain2.5-4B Qwen3-VL-4B -4.90 -2.93 -4.60 -6.44 -5.64 RynnBrain-8B Qwen3-VL-8B -1.24 -0.05 -0.28 -4.95 +0.33 RoboBrain2.5-8B-NV Qwen3-VL-8B -4.07 -2.51 -3.89 -1.61 -8.27

Qwen3-VL, and InternVL-3.5. Qwen3.5-thinking is the sole exception, exhausting its 32K thinking-token budget without emitting an answer letter.

###### 4.3.2. Frame Sampling Sensitivity

We ask whether frame-selection strategy explains the gap to human performance. Eight policies span cover four regimes: a one-frame query baseline (1@q); a 16-frame sliding window at 4 fps (16f@4fps); uniform budgets U-32/64/128/256 over the entire prefix; a recency-weighted variant log-128 (60% within 30s of the query, 30% in 30s–5min, 10% earlier), and an offline oracle oracle-evidence that places 128 frames inside the annotated evidence intervals. Formal definitions are in Appendix A.

Sampling Strategies Help Little. Tab. 4 reports overall accuracy under six policies on a representative subset. Neither the oracle nor U-256 improves over U-128: the oracle gain is at most +0.3 across all five models, and U-256 stays in [−2.6,+0.3]. The recency-weighted log-128 matches or exceeds the oracle on three of five models (Qwen3-VL-32B +2.6, VST-7B-RL +2.2, RoboBrain2.5-NV +3.3 vs. U-128) and underperforms on the other two, so smarter sampling does not consistently close the gap. Short-context priors are level-selective: 1@q wins Qwen3.5-27B L1 by +7.8 but loses L4 by −4.1 versus uniform, replicating the perception–memory trade-off. Together, these results indicate that the 27-point gap to human performance does not reduce to a retrieval problem solvable by better frame selection.

###### 4.3.3. Specialized Methods versus Backbones

Streaming, token-compression, and spatially fine-tuned MLLMs target the regime OVO-S-Bench evaluates. Tab. 5 pairs each specialized method with its base backbone under the same protocol.

Specialization Loses Across Levels. Specialization is net-negative overall: 13 of 15 methods lose on overall accuracy (median −2.0, range −18.4 to +0.5), and only HERMES (+0.4) and FluxMem (+0.5) exceed their base. L4 is the most uniformly damaged level: 13 of 15 methods regress on allocentric mapping

###### (a) Retention over t/tq

###### (b) Evidence recall by level

###### (c) ER vs. correctness

1.0

1.0

2.0

|HERMES (202k bins)<br><br>StreamingTOM (214k bins)<br><br>FluxMem (48k bins)<br><br>uniform reference|
|---|

HERMES StreamingTOM

HERMES (N=1667, r=-0.062)

StreamingTOM (N=1667, r=-0.093)

| |
|---|

0.8

0.8

FluxMem

FluxMem (N=1667, r=-0.036)

Retained-bindensity

1.5

P(correct)

MeanER

0.6

0.6

1.0

0.4

0.4

random (1/4)

0.5

0.2

0.2

0.0

0.0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

L1 L2 L3 L4

0.0 0.2 0.4 0.6 0.8 1.0

t/tq (0=start, 1=query)

Level

ER quintile mid

Figure 5: Token-compression diagnostics. (a) Retained 1-s bin density vs. 𝑡/𝑡𝑞 (dotted: uniform). (b) Evidence Recall (ER) by level. (c) ER vs. 𝑃(correct); legend shows per-method Pearson 𝑟.

(mean ∆ = −6.1, only Spatial-TTT +2.0 and RynnBrain +0.3 are positive), and Flash-VStream-7B (−16.7) and Cosmos-Reason1-7B (−12.8) show the largest drops. L1 is occasionally improved: perception-style specialization lifts current-view scores (VST-7B-RL +4.9, Cosmos-Reason1 +4.3, VST-7B-SFT +2.6). The L1/L4 split is consistent with training regimes optimized for fragmented spatial QA, narrative event memory, or robotic action grounding rather than long-range geometric evidence. The pattern recurs across all three specialization families despite non-overlapping training distributions, which argues against a pure domain-shift explanation, though a controlled fine-tuning study would be needed to fully isolate the cause.

Retention Is Not the Bottleneck. For HERMES, StreamingTOM, and FluxMem, we instrument inference to dump, at every query timestamp, the original-video seconds surviving in memory, then compute the per-query Compression Ratio (CR; fraction of the prefix kept) and Evidence Recall (ER; fraction of the annotated evidence interval that survives). Compression sheds evidence on memory-heavy levels: mean ER on L4 ranges from 0.14 (FluxMem) to 0.42 (StreamingTOM), against 0.60–0.76 on L1 (Figure 5b), so the L2/L4 cost of compression originates before reasoning begins. Yet retention alone does not predict correctness: the per-query Pearson correlation between ER and correctness is essentially zero across all three methods (𝑟 ∈ [−0.07,0.00], panel c), and the retained-frame time distribution stays close to the uniform reference (panel a) rather than concentrating on the evidence interval. Consistent with Sec. 4.3.2, higher retention does not translate into correctness under these compressors.

###### 5. Conclusion

We introduce OVO-S-Bench, a fully human-annotated benchmark that exposes the streaming spatial intelligence gap of current video MLLMs. Its 1,680 questions span four levels of spatial abstraction under a strict streaming protocol with annotated query timestamps and evidence intervals. Across 38 evaluated systems, Gemini-3.1-Pro (59.2) trails human experts by 27 points, with allocentric mapping as the dominant bottleneck; streaming and spatially fine-tuned variants score below their own backbones, and chain-of-thought amplifies spatial errors when ungrounded in the stream. We hope OVO-S-Bench helps shift spatial-intelligence evaluation from offline, discrete-frame benchmarks to continuous, streaming ones, where models must perceive, remember, and reason about space as it unfolds in real time, bringing evaluation closer to the embodied setting in which spatial intelligence must operate.

Limitations. While OVO-S-Bench covers the broadest mix of streaming-spatial scenes to date, from indoor walkthroughs to driving videos, the evaluated model remains a passive observer of pre-recorded streams and cannot close the perception-action loop that real embodied agents rely on. Each item is also scored via multiple-choice under a single streaming protocol, which simplifies evaluation but may obscure partial spatial knowledge that would surface under open-ended or interactive scoring. The specialization analysis (Section 4.3.3) does not control for domain shift between each method’s training data and OVO-S-Bench; controlled fine-tuning would be needed to fully isolate architectural limitations. This reflects a current trade-off in the field, as prior streaming-spatial benchmarks recover the loop only by trading scene diversity for controlled 3D environments, via camera-pose graphs over indoor scans [28] or a full physics simulator [17]. Moving forward, we plan to extend the protocol to a closed-loop regime that preserves the same diversity, toward benchmarks that evaluate not only what models perceive but also what they choose to perceive.

###### References

- [1] Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025. 4.1
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 4.1
- [3] Shuai Bai, Ke qin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025. 4.1
- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitScenes: A diverse real-world dataset for 3d indoor scene understanding using mobile RGB-D data. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2021. 3.2, 10
- [5] Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, et al. Scaling spatial intelligence with multimodal foundation models. arXiv preprint arXiv:2511.13719, 2025. 2, 4.1, 4.2
- [6] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024. 2
- [7] Xueyi Chen, Keda Tao, Kele Shao, and Huan Wang. Streamingtom: Streaming token compression for efficient video understanding. arXiv preprint arXiv:2510.18269, 2025. 4.1
- [8] Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, et al. Rynnbrain: Open embodied foundation models. arXiv preprint arXiv:2602.14979, 2026. 4.1
- [9] Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 346–355,

2024. 1, 2, F.1, 12

- [10] Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, et al. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026. 4.2
- [11] Yuanyuan Gao, Hao Li, Yifei Liu, Xinhao Ji, Yuning Gong, Yuanjun Liao, Fangfu Liu, Manyuan Zhang, Yuchen Yang, Dan Xu, et al. Holi-spatial: Evolving video streams into holistic 3d spatial intelligence. arXiv preprint arXiv:2603.07660, 2026. 2
- [12] Google DeepMind. Gemini 3.1 flash-lite. https://deepmind.google/models/gemini/ flash-lite/, March 2026. 4.1
- [13] Google DeepMind. Gemini 3.1 pro. https://blog.google/products/gemini/gemini-3-pro/, February 2026. 4.1
- [14] Google DeepMind. Gemma 4. https://ai.google.dev/gemma/docs/core/model_card_4, April

2026. Model card; technical report forthcoming. 4.1

- [15] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 1, 3.2, 10

- [16] Mingfei Han, Liang Ma, Kamila Zhumakhanova, Ekaterina Radionova, Jingyi Zhang, Xiaojun Chang, Xiaodan Liang, and Ivan Laptev. RoomTour3D: Geometry-aware video-instruction tuning for embodied navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3.2, 10
- [17] Yining Hong, Jiageng Liu, Han Yin, Manling Li, Leonidas Guibas, Li Fei-Fei, Jiajun Wu, and Yejin Choi. ESI-Bench: Towards embodied spatial intelligence that closes the perception-action loop. arXiv preprint arXiv:2605.18746, 2026. 5
- [18] Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online video understanding: Ovbench and videochat-online. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3328–3338, 2025. 1, 2, F.3
- [19] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135, 2025. 1, 2
- [20] Minsoo Kim, Kyuhong Shim, Jungwook Choi, and Simyung Chang. Infinipot-v: Memory-constrained kv cache compression for streaming video understanding. Advances in Neural Information Processing Systems, 38:138983–139013, 2026. 4.1
- [21] J. Richard Landis and Gary G. Koch. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174, 1977. E.5.2
- [22] Chengzu Li, Caiqi Zhang, Han Zhou, Nigel Collier, Anna Korhonen, and Ivan Vulić. TopViewRS: Visionlanguage models as top-view spatial reasoners. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1786–1807, 2024. 1, 2, F.1
- [23] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025. 12
- [24] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, Xiaodan Liang, Zhenguo Li, and Hang Xu. CODA: A real-world road corner case dataset for object detection in autonomous driving. In Proceedings of the European Conference on Computer Vision, 2022. 3.2, 10
- [25] Yun Li, Yiming Zhang, Tao Lin, XiangRui Liu, Wenxiao Cai, Zheng Liu, and Bo Zhao. Sti-bench: Are mllms ready for precise spatial-temporal world understanding? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5622–5632, 2025. 1, 2, F.2, 12
- [26] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration, 2025. 3.2, 10
- [27] Jingli Lin, Runsen Xu, Shaohao Zhu, Sihan Yang, Peizhou Cao, Yunlong Ran, Miao Hu, Chenming Zhu, Yiman Xie, Yilin Long, et al. Mmsi-video-bench: A holistic benchmark for video-based spatial intelligence. arXiv preprint arXiv:2512.10863, 2025. 1, F.2, 12
- [28] Jingli Lin, Chenming Zhu, Runsen Xu, Xiaohan Mao, Xihui Liu, Tai Wang, and Jiangmiao Pang. Ost-bench: Evaluating the capabilities of mllms in online spatio-temporal scene understanding. Advances in Neural Information Processing Systems, 38, 2026. 1, 2, 4.1, 5, F.3, 12
- [29] Junming Lin, Zheng Fang, Chi Chen, Haoxuan Cheng, Zihao Wan, Fuwen Luo, Ziyue Wang, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12147–12151. IEEE, 2026. 1, 2, F.3
- [30] Fangfu Liu, Diankun Wu, Jiawei Chi, Yimo Cai, Yi-Hsin Hung, Xumin Yu, Hao Li, Han Hu, Yongming Rao, and Yueqi Duan. Spatial-ttt: Streaming visual-based spatial intelligence with test-time training. arXiv preprint arXiv:2603.12255, 2026. 2, 4.1

- [31] Pengyiang Liu, Zhongyue Shi, Hongye Hao, Qi Fu, Xueting Bi, Siwei Zhang, Xiaoyang Hu, Zitian Wang, Linjiang Huang, and Si Liu. Vcbench: A streaming counting benchmark for spatial-temporal state maintenance in long videos, 2026. 2
- [32] Gen Luo, Ganlin Yang, Ziyang Gong, Guanzhou Chen, Haonan Duan, Erfei Cui, Ronglei Tong, Zhi Hou, Tianyi Zhang, Zhe Chen, et al. Visual embodied brain: Let multimodal large language models see, think, and control in spaces. arXiv preprint arXiv:2506.00123, 2025. 4.1
- [33] Yulin Luo, Chun-Kai Fan, Menghang Dong, Jiayu Shi, Mengdi Zhao, Bo-Wen Zhang, Cheng Chi, Jiaming Liu, Gaole Dai, Rongyu Zhang, et al. Robobench: A comprehensive evaluation benchmark for multimodal large language models as embodied brain. arXiv preprint arXiv:2510.17801, 2025. 12
- [34] Zhaoyang Lv, Nicholas Charron, Pierre Moulon, Alexander Gamino, Cheng Peng, Chris Sweeney, Edward Miller, Huixuan Tang, Jeff Meissner, Jing Dong, et al. Aria everyday activities dataset. arXiv preprint arXiv:2402.13349, 2024. 1
- [35] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16488–16498, 2024. 1, 2
- [36] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 2
- [37] Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18902–18913, 2025. 1, 2, 4.1, F.3
- [38] OpenAI. Introducing GPT-5.4. https://openai.com/index/introducing-gpt-5-4/, March 2026. 4.1
- [39] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24045–24055, 2025. 2
- [40] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. Advances in Neural Information Processing Systems, 37:119336–119360, 2024. 2
- [41] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, February 2026. Alibaba Cloud. 4.1
- [42] Vasili Ramanishka, Yi-Ting Chen, Teruhisa Misu, and Kate Saenko. Toward driving scene understanding: A dataset for learning driver behavior and causal reasoning, 2018. 3.2, 10
- [43] Sahithya Ravi, Gabriel Herbert Sarch, Vibhav Vineet, Andrew D Wilson, and Balasaravanan Thoravi Kumaravel. Out of sight, not out of context? egocentric spatial reasoning in vlms across disjoint frames. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 16146– 16161, 2025. 1, 2, F.2
- [44] Yujiao Shen, Shulin Tian, Jingkang Yang, and Ziwei Liu. A simple baseline for streaming video understanding. arXiv preprint arXiv:2604.02317, 2026. 2, A.4
- [45] Huajie Tan, Enshen Zhou, Zhiyu Li, Yijie Xu, Yuheng Ji, Xiansheng Chen, Cheng Chi, Pengwei Wang, Huizhu Jia, Yulong Ao, et al. Robobrain 2.5: Depth in sight, time in mind. arXiv preprint arXiv:2601.14352,

2026. 4.1

- [46] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958–22967, 2025. 2
- [47] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 4.1
- [48] Yifei Wang, Zhenkai Li, Tianwen Qian, Huanran Zheng, Zheng Wang, Yuqian Fu, and Xiaoling Wang. Streameqa: Towards streaming video understanding for embodied scenarios. arXiv preprint arXiv:2512.04451, 2025. 1, F.3
- [49] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. Advances in neural information processing systems, 38:13569–13597,

2026. 2, 4.1, 4.2

- [50] Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, and Weidi Xie. Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv e-prints, pages arXiv–2505, 2025. 12
- [51] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828– 28857, 2024. 2
- [52] xAI. Grok 4.1 fast and the Agent Tools API. https://x.ai/news/grok-4-1-fast, November 2025. 4.1
- [53] Yuxi Xiao, Longfei Li, Shen Yan, Xinhang Liu, Sida Peng, Yunchao Wei, Xiaowei Zhou, and Bingyi Kang. Spatialtree: How spatial abilities branch out in mllms. In The First Workshop on Efficient Spatial Reasoning,

2026. 2, 12

- [54] Yiweng Xie, Bo He, Junke Wang, Xiangyu Zheng, Ziyi Ye, and Zuxuan Wu. Fluxmem: Adaptive hierarchical memory for streaming video understanding. arXiv preprint arXiv:2603.02096, 2026. 2, 4.1
- [55] Peiran Xu, Sudong Wang, Yao Zhu, Jianing Li, Gege Qi, and Yunjian Zhang. Spatialbench: Benchmarking multimodal large language models for spatial cognition. arXiv preprint arXiv:2511.21471, 2025. 2
- [56] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025. 4.1
- [57] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 1, 2, 3.2, 10, F.2, 12
- [58] Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, et al. Visual spatial tuning. arXiv preprint arXiv:2511.05491, 2025. 2, 4.1
- [59] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis L Brown II, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. In The Fourteenth International Conference on Learning Representations, 2025. 1, 2, 4.1, F.2
- [60] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025. 1, 2, F.1, 12
- [61] Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, et al. Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10807–10816, 2025. 2
- [62] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025. 12

- [63] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025. 4.1
- [64] Xiangyu Zeng, Kefan Qiu, Qingyu Zhang, Xinhao Li, Jing Wang, Jiaxin Li, Ziang Yan, Kun Tian, Meng Tian, Xinhai Zhao, et al. Streamforest: Efficient online video understanding with persistent event memory. Advances in Neural Information Processing Systems, 38:75804–75835, 2026. 1, 2, 4.1, F.3
- [65] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memory-based real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 4.1
- [66] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-vstream: Efficient real-time understanding for long video streams. In Proceedings of the IEEE/CVF international conference on computer vision, pages 21059–21069, 2025. 2
- [67] Haowei Zhang, Shudong Yang, Jinlan Fu, See-Kiong Ng, and Xipeng Qiu. Hermes: Kv cache as hierarchical memory for efficient streaming video understanding. arXiv preprint arXiv:2601.14724, 2026. 4.1
- [68] Weichen Zhang, Zile Zhou, Xin Zeng, Liu Xuchen, Jianjie Fang, Chen Gao, Jinqiang Cui, Yong Li, Xinlei Chen, and Xiao-Ping Zhang. Open3d-vqa: A benchmark for embodied spatial concept reasoning with multimodal large language model in open space. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12784–12791, 2025. 2
- [69] Ziang Zhang, Zehan Wang, Guanghao Zhang, Weilong Dai, Yan Xia, Ziang Yan, Minjie Hong, and Zhou Zhao. Dsi-bench: A benchmark for dynamic spatial intelligence. arXiv preprint arXiv:2510.18873, 2025. 2
- [70] Ruosen Zhao, Zhikang Zhang, Jialei Xu, Jiahao Chang, Dong Chen, Lingyun Li, Weijian Sun, and Zizhuang Wei. Spacemind: Camera-guided modality fusion for spatial reasoning in vision-language models. arXiv preprint arXiv:2511.23075, 2025. 2
- [71] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, et al. OmniWorld: A multi-domain and multi-modal dataset for 4d world modeling, 2025. 3.2, 10

### Appendix

###### Content of Appendices

- Section A. Frame-Sampling Sensitivity.

- ∙ § A.1. Sampling Policy Definitions.
- ∙ § A.2. Overall-Accuracy Matrix.
- ∙ § A.3. Per-Level Breakdown.
- ∙ § A.4. Additional Analysis.
- ∙ § A.5. Length versus Difficulty.

- Section B. Per-Family Scaling Curves.

∙ Per-level accuracy versus backbone scale for Qwen3-VL, Qwen3.5, and InternVL-3.5.

- Section C. CoT Failure Taxonomy.

- ∙ § C.1. Judge Setup and Pre-filtering.
- ∙ § C.2. Cross-Judge Calibration.
- ∙ § C.3. Per-Level Failure Distribution.
- ∙ § C.4. Worked Examples.

- Section D. Detailed Category Definitions.

∙ L1–L4 subcategory definitions with accepted/rejected examples.

- Section E. Annotation Protocol & Construction Pipeline.

- ∙ § E.2. Source Datasets and Licensing.
- ∙ § E.3. General Annotation Protocol.
- ∙ § E.4. Level-Specific Construction Techniques.
- ∙ § E.5. Quality Control Pipeline.

- Section F. Extended Comparison with Prior Benchmarks.

∙ Per-benchmark discussion of what each benchmark does, shared points with OVO-S-Bench, and key differences.

- ∙ § F.5. Text-Only Shortcut Analysis.
- ∙ § F.6. Source-Robustness Analysis.

- Section G. Evaluation Methods Details.

- ∙ § G.1. Models Evaluated.
- ∙ § G.2. Decoding Parameters.
- ∙ § G.3. Default Frame Sampling.
- ∙ § G.4. Streaming-Model Ingestion Protocol.
- ∙ § G.5. Prompt Templates.
- ∙ § G.6. Policy-Specific Inputs.
- ∙ § G.7. Answer Extraction.
- ∙ § G.8. Hardware and Software.

- Section H. Future Work.
- Section I. Additional Examples.

###### A. Frame-Sampling Sensitivity

This appendix accompanies Section 4.3.2. We formalize the eight sampling policies, then report the full eight-model × eight-policy accuracy matrix at the Overall and per-level granularity. Three observations not included in the main text follow.

###### A.1. Sampling Policy Definitions

Let the video frame rate be 𝑓, the query time 𝑡𝑞 seconds, and the query frame index 𝑛𝑞 = ⌊𝑡𝑞 · 𝑓⌋. Let 𝑁 be the total frame budget and ℰ = {[𝑠𝑖,𝑒𝑖]}𝐾𝑖=1 the set of annotated evidence intervals (in seconds). Every policy outputs an ordered frame index set 𝒯 ⊆ {0,1,...,𝑛𝑞}, i.e. never consumes content posterior to the query, satisfying the streaming protocol.

###### Table 6: Frame-sampling policies, 𝑁=128 unless noted. Streaming policies obey the prefix-only constraint; the oracle requires evidence annotations and is not available at deployment.

Policy Prefix coverage Type

single@query (𝑁=1) last frame only streaming nearest-16f@4fps (𝑁=16) last 3.75s at 4fps streaming uniform-128 uniform over [0, 𝑛𝑞] streaming log-decay-128 3-band recency-weighted (60/30/10%) streaming oracle-evidence only inside ℰ offline

single@query. The naive streaming baseline that uses only the query frame: 𝒯sgl = {𝑛𝑞 } (𝑁=1).

nearest-16f@4fps. A short causal sliding window with step ∆ = max(1,round(𝑓/𝑓𝑤)):

𝒯rcw = {𝑛𝑞 − 𝑘∆ : 0 ≤ 𝑘 < 𝑁, 𝑛𝑞 − 𝑘∆ ≥ 0}.

The default 𝑁=16, 𝑓𝑤=4fps covers the last (𝑁 − 1)/𝑓𝑤 = 3.75s before the query.

uniform-𝑁. Uniform sampling of 𝑁 frames over the entire prefix [0,𝑛𝑞]:

{︀

(︀𝑁𝑘−1 𝑛𝑞

)︀

}︀

round

𝒯uni =

: 0 ≤ 𝑘 < 𝑁

.

We sweep 𝑁 ∈ {32,64,128,256}; uniform-128 is the default policy used in Table 2.

oracle-evidence. An offline oracle that allocates the 𝑁=128 budget inside the annotated evidence intervals only. Each interval [𝑠𝑖,𝑒𝑖] maps to a frame-domain interval [˜𝑠𝑖,𝑒˜𝑖] clipped to ≤ 𝑛𝑞, with length 𝐿𝑖 = 𝑒˜𝑖 − 𝑠˜𝑖 and per-interval budget 𝑁𝑖 = max(1,round(𝑁𝐿𝑖/

∑︀

𝑗 𝐿𝑗)) (rounding corrected on the longest interval). Each interval is then uniformly sampled:

⋃︁𝐾

linspace(˜𝑠𝑖,𝑒˜𝑖,𝑁𝑖).

𝒯evid =

𝑖=1

When ℰ = ∅ the policy degrades to uniform-𝑁 over [0,𝑛𝑞]. This serves as an upper bound: how well a model could do if it had access only to the truly relevant frames.

log-decay-128. A recency-weighted three-band schedule with 𝑁=128, band boundaries 𝛽1 = 30s, 𝛽2 = 300s, and weights (𝛼1,𝛼2,𝛼3) = (0.60,0.30,0.10):

- ℬ1 = [𝑛𝑞 − 𝛽1𝑓, 𝑛𝑞], 𝑁1 = round(𝛼1𝑁),
- ℬ2 = [𝑛𝑞 − 𝛽2𝑓, 𝑛𝑞 − 𝛽1𝑓], 𝑁2 = round(𝛼2𝑁),
- ℬ3 = [0, 𝑛𝑞 − 𝛽2𝑓], 𝑁3 = 𝑁 − 𝑁1 − 𝑁2.

Each band is uniformly sampled. When the prefix is shorter than 𝛽2𝑓 the missing band is skipped and its budget rolls into the surviving bands. This policy realizes a coarse exponential prior of near-importance without using ground-truth evidence.

Table 6 compares the five families at 𝑁=128.

- A.2. Overall-Accuracy Matrix

###### Table 7 gives the Overall accuracy of eight models under all eight policies. Empty cells (“–”) indicate that the model’s context window cannot accommodate the required frame count.

- Table 7: Full Overall-accuracy matrix on OVO-S-Bench (%). Eight representative models × eight frame-sampling policies. Per-row best is bolded. “–” = context-window overflow.

Model single@q nearest-16f U-32 U-64 U-128 U-256 oracle-E log-128 Gemini-3.1-Flash-Lite 47.95 49.72 52.45 51.26 50.88 48.33 49.70 48.55 InternVL-3.5-38B 45.04 44.45 48.94 49.49 49.08 – 50.18 49.33 Qwen3-VL-32B 43.95 45.80 45.89 47.18 48.71 48.93 48.80 51.40 Qwen3.5-27B 47.49 48.42 48.12 49.72 51.56 50.64 51.87 51.74 VST-7B-RL 41.24 39.10 41.50 43.62 42.13 41.99 42.13 44.35 Cambrian-S-7B 35.93 36.01 34.93 36.66 36.76 36.20 37.17 37.67 VeBrain-7B 43.51 45.17 41.65 43.82 43.47 43.30 43.53 45.40 RoboBrain2.5-NV-8B 39.91 41.22 40.82 42.63 43.56 43.87 43.56 46.84

- Table 8: Per-level accuracy (%) under eight frame-sampling policies. Per-row best is bolded. “–” = contextwindow overflow.

(a) L1: Instantaneous Egocentric Perception

Model 1@q 16f U32 U64 U128 U256 Ora Log Gemini-3.1-Flash-Lite 59.53 59.46 53.12 53.14 53.88 53.40 53.28 53.60 InternVL-3.5-38B 59.87 61.15 53.35 54.04 54.44 – 54.97 58.48 Qwen3-VL-32B 56.46 60.74 49.36 49.66 49.93 51.37 50.19 55.34 Qwen3.5-27B 59.12 61.88 49.52 50.87 51.29 50.63 51.06 53.87 VST-7B-RL 57.81 52.45 45.47 45.96 45.45 46.16 45.45 50.73 Cambrian-S-7B 39.14 41.12 38.25 40.16 40.23 39.81 40.23 42.21 VeBrain-7B 52.64 54.43 41.67 42.08 42.74 43.02 42.74 44.52 RoboBrain2.5-NV-8B 53.79 53.25 41.14 43.42 42.86 44.06 42.86 50.14

###### (c) L3: Spatial Reasoning

Model 1@q 16f U32 U64 U128 U256 Ora Log Gemini-3.1-Flash-Lite 50.85 53.12 54.56 55.18 54.14 51.32 54.29 53.65 InternVL-3.5-38B 46.06 44.70 51.07 48.94 45.53 – 46.01 47.35 Qwen3-VL-32B 52.25 50.25 50.13 48.42 51.94 55.03 52.45 54.02 Qwen3.5-27B 51.40 50.54 50.98 50.86 52.39 51.80 53.85 56.04 VST-7B-RL 38.60 40.56 39.00 42.11 40.88 41.07 40.88 40.87 Cambrian-S-7B 41.36 38.64 36.32 38.47 36.93 36.77 37.01 38.47 VeBrain-7B 46.01 45.87 42.23 47.46 46.16 44.88 46.67 43.78 RoboBrain2.5-NV-8B 43.65 44.28 42.81 46.66 50.63 51.08 50.63 49.22

###### (b) L2: Context Tracking

Model 1@q 16f U32 U64 U128 U256 Ora Log Gemini-3.1-Flash-Lite 36.90 39.25 56.88 54.11 52.03 47.96 49.03 42.41 InternVL-3.5-38B 33.32 34.89 52.67 55.03 54.40 – 54.41 52.98 Qwen3-VL-32B 33.93 34.55 50.13 52.27 51.75 53.53 52.31 50.45 Qwen3.5-27B 35.82 35.71 48.89 52.35 54.83 54.09 55.16 52.46 VST-7B-RL 31.63 28.71 44.17 44.86 44.20 43.65 44.20 45.22 Cambrian-S-7B 34.56 34.29 36.79 38.56 40.03 39.75 39.99 42.18 VeBrain-7B 31.82 33.21 42.45 43.89 44.18 46.00 44.18 47.73 RoboBrain2.5-NV-8B 28.45 31.10 42.30 44.12 46.64 46.31 46.64 46.64

###### (d) L4: Allocentric Mapping

Model 1@q 16f U32 U64 U128 U256 Ora Log Gemini-3.1-Flash-Lite 44.51 47.03 45.27 42.61 43.46 40.66 42.17 44.55 InternVL-3.5-38B 40.90 37.07 38.68 39.94 41.94 – 45.32 38.50 Qwen3-VL-32B 33.15 37.65 33.95 38.35 41.22 35.77 40.26 45.80 Qwen3.5-27B 43.61 45.54 43.10 44.79 47.72 46.03 47.42 44.57 VST-7B-RL 36.94 34.68 37.34 41.54 37.99 37.09 37.99 40.56 Cambrian-S-7B 28.68 29.98 28.37 29.45 29.86 28.49 31.44 27.84 VeBrain-7B 43.58 47.16 40.25 41.84 40.82 39.29 40.54 45.59 RoboBrain2.5-NV-8B 33.74 36.24 37.01 36.32 34.10 34.02 34.10 41.36

###### A.3. Per-Level Breakdown

Table 8 reports the per-level accuracies (L1–L4) of the same eight models under the same eight policies. Per-row best is bolded. The level breakdown drives the three observations in §A.4.

###### A.4. Additional Analysis

The oracle is not an upper bound. On five of eight models (Cambrian-S-7B, VeBrain-7B, VST-7B-RL, Qwen3VL-32B, RoboBrain2.5-NV-8B), log-128 outperforms oracle-evidence on Overall by 0.5–3.3 points. The oracle places all 128 frames inside the annotated evidence interval, whereas the rest of the prefix is discarded; for questions whose answer also depends on global context (e.g. L3 counterfactuals and L4 trajectory queries), this aggressive concentration removes useful framing. The log-128 schedule, in contrast, retains 10% of the budget on the far past, which appears to compensate.

Backbone-dependent value of long context. Doubling the uniform budget from 128 to 256 frames yields negligible or negative deltas for all seven models with data (range −2.6 to +0.3): Gemini-3.1-Flash-Lite −2.6, Qwen3.5 −0.9, Cambrian-S −0.6, VeBrain −0.2, VST-7B-RL −0.1, Qwen3-VL +0.2, RoboBrain2.5-NV +0.3. No model benefits by more than 0.3 points, consistent with the scaling-curve finding (§B) that L4 saturates early across families and with the observation in [44] that the marginal value of longer visual context is model-dependent.

Short-context priors and the perception–memory trade-off. Across all four large general backbones the one-frame query (1@q) wins L1 by 5–8 points over uniform-128 (Gemini +5.7, InternVL-3.5-38B +5.4, Qwen3-

Scope Evidence 𝜌 𝑛 Prefix 𝜌 𝑛 L1 -0.04 629 0.00 629 L2 0.08 540 -0.08 540 L3 0.02 279 0.24 279 L4 0.17 274 0.26 274 Level-controlled 0.06 1722 0.06 1722 Level+source-controlled 0.04 1722 0.01 1722

Length vs. query difficulty, stratified by level

Meanaccuracyacrossreportedmodels(%)

100

80

60

40

20

L1 L2

L3 L4

0

2 4 6 8 10 12 Evidence span (log2 seconds, L4.3=prefix)

0 2 4 6 8 10 12 14 Visible prefix length (log2 seconds)

- Table 9: Spearman 𝜌 between query-level mean accuracy and input length. Positive 𝜌 means longer inputs correlate with higher accuracy (opposite to the “longer

Figure 6: Query-level mean accuracy versus evidence-span duration (log scale), colored by level. Trend lines are per-level LOWESS fits.

= harder” hypothesis). All values are small and none survive source controls.

L1 L2 L3 L4 Overall

| | | | |wen3|-V|L| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | || |
|---|
| | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |

###### Qwen3.5

###### InternVL-3.5

60

Accuracy(%)

| |
|---|

50

40

30

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Total parameters (B, log scale)

- Figure 7: Per-level accuracy versus total parameter count (log scale) within three open-source families. Solid lines: L1–L4. Dashed line with square markers: overall accuracy. Open rings overlay MoE checkpoints.

VL-32B +6.5, Qwen3.5-27B +7.8) but loses L2 by 15–21 points (Gemini −15.1, Qwen3-VL −17.8, Qwen3.5 −19.0, InternVL-3.5 −21.1). The trade-off is sharpest for VST-7B-RL (+12.4 on L1 vs. −12.6 on L2) and is the empirical signature of the perception–memory trade-off [44] for OVO-S-Bench: a clean current view helps egocentric perception but cannot answer questions that require integrating across the prefix.

###### A.5. Length versus Difficulty

Evidence spans grow with taxonomy level (L1 median 2.0 s, L2 36.8 s, L3 2.0 s, L4 278.7 s), raising the question of whether length alone explains difficulty. We compute the query-level mean accuracy over the 38 reported models and correlate it with three length variables: visible prefix duration, evidence-span duration, and full source-video duration. Tab. 9 reports Spearman 𝜌 per level and under two sets of controls.

Although evidence spans grow with level, length alone does not explain difficulty: the level-controlled correlations between accuracy and prefix length (𝜌 = 0.06) or evidence-span length (𝜌 = 0.06) are small, and full-video-duration correlation is similarly weak (𝜌 = 0.03, 𝑝 = 0.29, computed analogously). With source controls these correlations shrink further. The small evidence-span effect is positive (longer spans correlate with slightly higher accuracy), opposite to the direction predicted by a “longer = harder” explanation. The allocentric-mapping bottleneck documented in Section 4.2 is therefore not reducible to a generic long-video effect; it reflects the abstraction demanded by L4, not the temporal extent of the evidence.

###### B. Per-Family Scaling Curves

This appendix accompanies the corollary in Section 4.2 that L4 saturates early. We trace per-level accuracy as a function of backbone scale within three open-source families: Qwen3-VL, Qwen3.5, and InternVL-3.5. The curves include dense and MoE checkpoints from the small to frontier-scale variants in each family, all under default 128-frame uniform sampling with reasoning off, so the curves isolate scale. MoE checkpoints are marked with open rings, since the total parameter count overstates per-token compute.

Three Scaling Regimes. Figure 7 reveals three regimes. L1 and L2 scale log-linearly: end-to-end gains are +11.8/ + 9.5 on Qwen3-VL, +13.9/ + 14.5 on Qwen3.5, and +20.3/ + 16.2 on InternVL-3.5, with the two curves tracking each other within 3 points at every size. L3 climbs but unevenly: Qwen3.5 gains +25.2 overall yet Qwen3-VL drops from 54.5 at 4B to 49.9 at 30B-A3B before recovering to 61.2 at 235B-A22B, and the InternVL-3.5 38B dense checkpoint (45.5) underperforms its 30B-A3B MoE sibling (49.0), suggesting L3 depends on training-data composition as much as raw parameter count. L4 saturates early: all three families plateau in the 40–46 range from the mid-tier onward; InternVL-3.5 L4 moves only +0.8 over a 60× parameter increase (39.7 at 4B vs. 40.5 at 241B-A28B), and remains roughly 39 points below human-streaming (79.2), indicating the allocentric bottleneck is not a capacity problem and will not close from scaling alone.

###### C. CoT Failure Taxonomy

This appendix accompanies the failure-pattern analysis in Section 4.3.1. We give the formal judge setup (§C.1), the cross-judge calibration (§C.2), the per-level distribution (§C.3), and one worked example per error class (§C.4).

###### C.1. Judge Setup and Pre-filtering

Inputs and judge function. For each thinking-mode model ℳ in Table 3, let 𝒬−ℳ = {𝑞 : 𝑎ℳ𝑞 ̸= 𝑎*𝑞 } be the set of queries on which ℳ’s extracted letter 𝑎ℳ𝑞 differs from the ground truth 𝑎*𝑞. For each 𝑞 ∈ 𝒬−ℳ the judge 𝐽 applies

𝐽(𝑞,𝒪𝑞, 𝑎*𝑞, 𝑎ℳ𝑞 , 𝜏𝑞ℳ, ℱ𝑞)

→ ℓ ∈ {T1a,T1b,T2a,T2b,T2c},

where 𝒪𝑞 is the multiple-choice option set, 𝜏𝑞ℳ the thinking trace (truncated to head-3,000 + tail-1,500 characters when |𝜏| > 5,000), and ℱ𝑞 = {𝑓1,...,𝑓𝑘} with 𝑘 ∈ [4,6] a set of 512px JPEG frames extracted at midpoints of the annotated evidence intervals ℰ𝑞. The five label codes map to the paper-text classes no-conclusion error (T1a), non-visual error (T1b), visual-content error (T2a), direction error (T2b), and temporal-binding error (T2c).

Judge model. All wrong-answer traces are judged by gpt-5.4. To verify that the five-class scheme is robust to judge choice, we additionally re-judge all traces with gemini-3.1-flash-lite; see §C.2.

Voting. Every query is judged 𝑟=3 times under the same prompt, using stream=true, JSON-object output, and max_tokens=200. The primary label is the majority vote; records without a ≥2-vote majority are marked disagreement and excluded from the figures (<0.5%).

Pre-filtering and signal-tagging. We drop 39 queries whose text contains an absolute timestamp (“at 134 seconds”), a prompt-engineering artifact rather than a reasoning failure. Decoding-truncated traces (|𝜏| > 50,000 characters without a </think> close tag) are directly tagged T1a without calling the judge; this affects 591 Qwen3.5 traces that overran the 32K thinking-token cap.

Prompt skeleton. The system prompt fixes the five-class taxonomy and decision rules (Tier-1 takes precedence when both apply; prior phrases without visual citations route to non-visual error; direction errors require explicit textual evidence of the wrong direction). The user turn injects the per-query tuple (Question, Options, Ground-truth, Predicted, Trace, Evidence frames), the last being the 𝑘 images appended as base64 image_url fields. The verbatim prompts are released with the code.

###### C.2. Cross-Judge Calibration

We re-judged all wrong-answer traces with gemini-3.1-flash-lite under the identical prompt. Exact agreement is 75.9% with Cohen’s 𝜅=0.65, which falls in the substantial band of the Landis–Koch convention and is sufficient for the qualitative cross-model claims made in Section 4.3.1.

###### (a) L1: Egocentric perception

###### (b) L2: Context tracking

n=351

n=259

GLM-4.6V

GLM-4.6V

n=369

n=234

InternVL-8B

InternVL-8B

n=351

n=222

InternVL-38B

InternVL-38B

n=379

n=240

Qwen3-VL-4B

Qwen3-VL-4B

n=342

n=209

Qwen3-VL-32B

Qwen3-VL-32B

n=385

n=229

Qwen3.5-4B

Qwen3.5-4B

n=349

n=197

Qwen3.5-9B

Qwen3.5-9B

n=324

n=182

Qwen3.5-27B

Qwen3.5-27B

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250

###### (c) L3: Spatial reasoning

###### (d) L4: Allocentric mapping

n=144

n=175

GLM-4.6V

GLM-4.6V

n=140

n=161

InternVL-8B

InternVL-8B

n=134

n=142

InternVL-38B

InternVL-38B

n=126

n=151

Qwen3-VL-4B

Qwen3-VL-4B

n=132

n=153

Qwen3-VL-32B

Qwen3-VL-32B

n=168

n=165

Qwen3.5-4B

Qwen3.5-4B

n=150

n=144

Qwen3.5-9B

Qwen3.5-9B

n=124

n=104

Qwen3.5-27B

Qwen3.5-27B

0 25 50 75 100 125 150 175

0 25 50 75 100 125 150 175

Wrong-trace count

Wrong-trace count

No-conclusion

Non-visual

Visual-content

Direction

Temporal-binding

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

- Figure 8: Per-level failure-pattern distribution. Each panel restricts the data in Figure 4 to one question level. Rows and colors match the main figure. L2 (top-right) makes the universal temporal-binding error pattern visible; L4 (bottom-right) makes the family bifurcation between non-visual (GLM, Qwen3-VL) and direction (InternVL-3.5) errors visible.

###### C.3. Per-Level Failure Distribution

- Figure 8 reproduces Figure 4 broken down by question level. Two patterns are invisible in the pooled view.

L2 elicits temporal-binding errors universally. All eight models score 17–36% temporal-binding error on L2 (spatiotemporal context tracking; mean 28.5%), against ≤ 5.6% on the other three levels. This mirrors the +3.9 average L2 gain from thinking mode in Table 3: explicit reasoning helps most precisely where the bottleneck is pinning a fact to a specific prior moment.

L4 bifurcates by family. The L4 panel reveals two distinct failure-mode clusters. GLM and Qwen3-VL fall back on world priors (69–85% non-visual error, ≤ 2% direction error), while InternVL-3.5 attempts cardinal-frame reasoning but inverts directions (27–30% direction error, ≤ 18% non-visual error). The L4 bottleneck in Table 2 is therefore not a single capacity gap and unlikely to close from a backbone-agnostic remedy.

###### C.4. Worked Examples

We present two annotated cases per error class, organized by tier. Tier 1 (Fig. 9) covers process-level failures where the model either fails to reach a conclusion (T1a, no-conclusion error) or bypasses the visual evidence entirely (T1b, non-visual error). Tier 2 (Fig. 10) covers reasoning-level failures where the model engages with the frames but mis-grounds the spatial content (visual-content error), inverts a directional relation (direction error), or binds a correct spatial fact to the wrong moment (temporal-binding error). Each card shows the evidence frames, the question and ground truth, the model’s predicted answer, an excerpt of the chain-of-thought trace, and an error analysis.

###### T1: Process-Level Failures

###### No-conclusion Error

[Figure 67]

[Figure 68]

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

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

t=9.0 t=9.5 t=10.0 t=10.5 t=11.0

t=78.1 t=125.0 t=778.1 t=816.1 t=1215.0

Red sofa Road Greenery Fence

Red sofa Greenery Greenery

Object Appearance

Revisit Counting

MLLM: A: A Hello Kitty GT: C: A bouquet of red roses

###### MLLM: D: 8 GT: E: 5

Q Compared with the immediately preceding moment in the video, which object

Q Including this moment, how many times has this same scene appeared so far

has newly appeared on the Red sofa in the current view?

in the video?

CoT...but butno. no.Wait,Wait,no, no,I thinkI'llthego withanswertheiscorrectA? No. option.Wait, no,Wait,I'm no,stuck.theWait,answerno,isthenone, correct answer is none, but no. Wait, no, the user's question is about the Red sofa. Let's check the options. The correct answer is none, but no. Wait, no, I think the answer is A. No. Wait, no, I'll just pick the letter. Wait, no, the correct answer is none, but no. Wait, no, I think the answer is A. No. Wait, no, I'm wrong. Wait, no, the answer is none, but no. Wait, no, I'll go with the correct option. Wait, no, the answer is A. No. Wait, no, I think the answer is none, but no. Wait, no, I'm done. <|begin_of_box|>A<|end_of_box|>

CoT...70. meTheshowsseventieththe sameframesceneshowsagain.the 69.sameTheroadsixty-ninthand surroundings.frame is71.anotherThe seventy-firstcontinuation. frame is another instance of the same scene. 72. The seventy-second frame continues the same view. 73. The seventy-third frame shows the same scene again. 74. The seventyfourth frame is another continuation. 7 Answer: D To determine how many times this scene has appeared, I analyzed the visual elements in the frames, focusing on the paved road, greenery, fence, and surrounding environment. By mentally scanning through the provided frames, I identified multiple instances of the same scene. The scene has appeared 8 times, includi ...

###### Error Analysis

###### Error Analysis

The model falls into a circular "Wait, no..." loop and finally guesses without using the visual frames.

The model keeps enumerating imagined frames and fails to reach a valid conclusion before giving an unreliable count.

###### Non-visual Error

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

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

t=518.7 t=519.2 t=519.7 t=520.2 t=520.7

t=209.2 t=209.7 t=210.2 t=210.7 t=211.2

Player bottom edge columned platform stairs

Road trees fences fences stream

Distance Estimation

Height Estimation

MLLM: A: About 3 meters GT: D: About 1 meter

MLLM: B: About 3.5 m GT: A: About 2.5 m

Q In the query moment, about how far is the player from the bottom edge of the

Q In the current frame, approximately how high is the camera above the ground

lowest stair step in front of the columned platform?

plane?

CoTGotplayerit,tolet'sthe bottomtackle this.edge ofThethequestionlowest stairis aboutstepestimatingin front theof adistancecolumnedfromplatformthe . Looking at the relevant frame (the one with the player near the columned platform stairs), we need to assess typical game distances. In many games like Assassin's Creed, such distances are often around 3-5 meters. But let's think about the options. The options are A (3m), B (5m), C (8m), D (1m). In the context of the game's scale, when a character is approaching stairs, being about 3-5 meters away is plausible. But which is more accurate? Let's consider the visual. If the player is running towards the stairs, the distance to the ...

CoT...stoneuralwallroadon thewithrighta mixandof grassnaturalfieldsand builton theenvironments.left, transitioningThe roadintostartsa morewithwoodeda area with leafless trees and a stone hedge. The road continues to a residential area with houses and fences, indicating a village setting. The final frames show a wooden bridge over a small stream, suggesting the path leads to a crossing point. The consistent elevation and absence of significant changes in height throughout the frames suggest the camera is maintained at a moderate height, likely around 3.5 meters, which is typical for a cyclist or a person walking with a camera mounted on a bike or a stabilizing device. Answer: B

###### Error Analysis

###### Error Analysis

The model relies on generic game-world distance priors instead of estimating from the visible stair and player layout.

The model ignores the actual stairway view and invents a rural-road scene to justify a prior-based height estimate.

- Figure 9: Tier 1: Process-level failures. Two examples each for T1a (no-conclusion error, top) and T1b (non-visual error, bottom). Left and right columns show cases from different task families. Evidence frames are sampled at midpoints of the annotated evidence interval.

###### D. Detailed Category Definitions

This appendix provides formal definitions for the 30 canonical task types used in OVO-S-Bench. The taxonomy is paper-facing and collapses finer raw annotation labels into canonical categories where appropriate; in particular, L4 is reported as three map-query families. The four levels form a cognitive progression: L1 See (instantaneous egocentric perception), L2 Remember (spatial memory after evidence leaves the view), L3 Imagine (simulation, verification, and planning over spatial state), and L4 Map (integrating first-person experience into a queryable allocentric representation). For each task type we give a Definition, an Accepted example, and a Rejected near-miss that clarifies the boundary with a neighboring type or level.

- D.1. L1: Instantaneous Egocentric Perception

L1 questions are answerable from the frames near the query timestamp alone, without recalling any past observation. The assignment rule is query-time visibility: if the spatial cue needed to answer is visible near the query, the item belongs to L1; if the same cue must be recalled after the camera has moved away, it belongs to L2.

###### Tier 2: Reasoning-Level Failures

###### Visual-content Error

Cleaning Rag

Tabletop bucket Pliers

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Containment Relation MLLM: inside GT: outside

Support Relation MLLM: sink GT: tabletop

Q In the current view, are the pliers located inside, outside, or on the rim of the plastic bucket?

Q In the current frame, which object is directly supporting the cleaning rag?

COT The video frames show the pliers located inside the plastic bucket. The bucket contains various items, and the pliers are clearly visible among them, indicating they are not on the rim or outside the bucket. ... Answer: They are placed inside the plastic bucket.

COT ...The question asks which object is directly supporting the cleaning rag. Looking at the frames where the cleaning rag is being used, it seems to be near the sink, so the sink appears to be the support.

The model wrongly states pliers are inside the bucket, while they are held outside all the time.

The model falsely imagines a sink and judges based on wrong spatial info instead of real visual facts.

Direction Error

Yawing right

Two cups

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Ego-motion Rotation MLLM: left GT: right

Relative Physical Scale MLLM: right GT: left

Q Based on the visual flow, is the camera currently yawing left, yawing right, or not rotating?

Q Which of the two measuring cups is larger?

COT ...The question is about two measuring cups. Looking at the relevant image (probably one of the kitchen shots), we need to compare the sizes. In the image where measuring cups are visible, the one on the right is larger. So the answer is The one on the right.

COT ...the perspective shifts to show more of the road on the left and less of the building, indicating a leftward rotation. ... 3. The final frames show the camera has rotated further left, revealing more of the street and buildings on the left side. ... Answer: A. Yawing left.

Camera yaws right with clockwise visual shift. Rotation detected correctly, yet direction judged wrong as left, resulting in axis inversion.

The model spots two measuring cups yet wrongly thinks right one is larger, reversing their actual sizes.

Temporal-binding Error

Pool table Behind Room1 once Room1 twice Room2 once

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Revisit Counting MLLM: Two GT: One

Out-of-ViewLocalization MLLM: In front GT: Behind

Q Which visit is it to the bedroom with the patterned accent wall behind the bed?

Q At the current moment, where is the pool table located relative to the camera? COT ...No pool table yet. ... Frame 61.0s: The camera pans into a new room. A blue pool table is clearly visible in the center of the room. ... C. In front of the camera: Correct. The table is directly ahead in the shot.

COT ...Frame 20-22: First bedroom (gray walls, two paintings) - Frame 62-63: Second bedroom (patterned accent wall, black and white city map style) ... - 38-39: Back to bedroom 1 ... So the bedroom with the patterned accent wall is the second bedroom. So the visit number is 2.

The reasoning refers to earlier footage of the pool table, yet fails to focus on the current moment. It overlooks the camera pan at 64.0s, where the table no longer in front.

Scene matches bedroom, but answer refers to room sequence instead of entry times; error stems from incorrect visit judgment.

- Figure 10: Tier 2: Reasoning-level failures. Two examples each for visual-content error (top), direction error (middle), and temporal-binding error (bottom). Each card shows the CoT excerpt and a bolded error analysis.

Can you provide the correct answer? How can I improve the description to get the correct answer? What other information do I need to get the correct answer?

- D.1.1. Egocentric Metric Perception

Object-to-Object Absolute Distance. Definition. Estimate the absolute distance between two entities (observer-to-object or object-to-object) visible in the current view. The answer is a numeric magnitude choice (meters, steps, car-lengths, etc.).

Accepted. “About how far is the player from the bottom edge of the lowest stair step?” A: ∼3m; B: ∼5m; C: ∼8m; D: ∼1m. Both referents are visible; the question asks for an absolute metric.

Rejected. “Which object is closer to you — the lamp or the sofa?” — rejected: no absolute distance is requested; this is a depth-ordering judgment → Object Depth Ordering.

Relative Physical Scale. Definition. Compare the physical size (height, width, volume) of two objects visible in the current frame. The answer is a relative judgment, not an absolute metric.

Accepted. “Which of the two measuring cups is larger — the one on the left or the one on the right?” Both objects are simultaneously visible; the question asks which is physically bigger.

Rejected. “Which object is nearer to the camera — the cup or the plate?” — rejected: this asks about distance from the viewpoint, not physical size → Object Depth Ordering.

Object Depth Ordering. Definition. Determine the relative distance of two or more objects from the camera or observer (which is closer, which is farther). The answer is an ordering or binary comparison along the depth axis.

Accepted. “Which piece of furniture is closer to you — the coffee table or the bookshelf?” A: coffee table; B: bookshelf. Both are visible; the question asks about relative depth.

Rejected. “About how many meters separate the coffee table and the bookshelf?” — rejected: this asks for an absolute metric → Object-to-Object Absolute Distance.

Viewpoint Height Perception. Definition. Estimate the camera or observer’s current elevation relative to the ground plane or a visible reference (floor level, building stories, terrain).

Accepted. “Based on the current view, approximately which floor is the camera at?” A: ground floor; B: 2nd floor; C: 4th floor; D: rooftop. Height cues (horizon line, visible floors below) are in the current frame.

Rejected. “How many floors have you ascended since entering the building?” — rejected: requires recalling the entry point → L2 Chronological Recall.

###### D.1.2. Local Spatial Relations

Containment Relation. Definition. Determine whether one object is inside, on top of, or within the bounds of another container or surface visible in the current view. The answer identifies the containing entity.

Accepted. “Where is the red mug right now?” A: inside the cabinet; B: on the table; C: in the sink; D: on the shelf. The mug and its container are both visible.

Rejected. “Where did you last see the red mug?” — rejected: the mug has left the field of view; answering requires memory → L2 Out-of-View Localization.

Occlusion Recognition. Definition. Identify which object partially or fully occludes another in the current view, or determine what is hidden behind a visible occluder.

Accepted. “Which object is partially blocking the TV screen in the current frame?” A: floor lamp; B: potted plant; C: curtain; D: nothing.

Rejected. “Is the painting that was previously hidden behind the door still occluded?” — rejected: requires comparing two temporal states → L3 Object-Level Change Detection.

Support Relation. Definition. Identify the physical support relation in the current view: what is holding up or bearing the weight of a target object.

Accepted. “In the current frame, which surface is directly supporting the cleaning rag?” A: bathroom sink; B: shoulder; C: tabletop; D: wooden floor.

Rejected. “If you remove the middle shelf, what will fall?” — rejected: requires counterfactual simulation → L3 Spatial What-If Consequence.

Local Topology Description. Definition. Describe the spatial arrangement (left–right, front–back, above– below) of multiple objects visible simultaneously in the current frame. The answer is an ordering or layout description.

Accepted. “From left to right, what is the order of the three items on the desk?” A: lamp–book–cup; B: book–lamp–cup; C: cup–book–lamp; D: lamp–cup–book.

Rejected. “Combining what you saw in the kitchen and the living room, describe the overall furniture layout.”

— rejected: requires integrating multiple viewpoints into a global representation → L4 Topological Structure Reasoning.

###### D.1.3. Dynamic Spatial Perception

Ego-motion Translation. Definition. Classify the observer’s or camera’s translational motion (moving forward, backward, left, right, up, down, or stationary) from the frames near the query. The question targets position change, not heading change.

Accepted. “Is the camera currently moving forward, backward, or staying in place?” A: forward; B: backward; C: stationary.

Rejected. “Is the camera turning left or right?” — rejected: heading change without position displacement is rotational → Ego-motion Rotation.

Ego-motion Rotation. Definition. Classify the observer’s or camera’s rotational motion (panning left/right, tilting up/down, rolling) from the frames near the query. The question targets heading or orientation change.

Accepted. “Which direction is the camera currently panning?” A: left; B: right; C: not panning. Rejected. “Is the camera advancing down the hallway?” — rejected: forward displacement is translational

→ Ego-motion Translation.

Object Motion Separation. Definition. Identify an independently moving object in the scene (i.e., its motion is not explained by camera ego-motion alone) and determine its motion direction or trajectory.

Accepted. “Which direction is the white sedan in the adjacent lane traveling?” A: same direction as you; B: opposite; C: turning left; D: stationary.

Rejected. “Where is the cyclist who passed you earlier now?” — rejected: the cyclist has left the view; answering requires memory → L2 Out-of-View Localization.

Relative Speed Judgment. Definition. Compare the speeds of two moving entities (or the observer and an object) visible in the current clip. The answer is a relative judgment (faster, slower, same).

Accepted. “Who is moving faster — your car or the truck in the right lane?” A: your car; B: the truck; C: about the same.

Rejected. “Are you moving faster now than you were at the intersection 2 minutes ago?” — rejected: requires recalling a past speed → L2 Temporal Depth Estimation.

Judgment of Motion State. Definition. Classify whether a visible entity is currently stationary or in motion (and optionally whether it is accelerating or decelerating) based on the frames near the query.

Accepted. “Is the pedestrian ahead currently stationary or walking?” A: stationary; B: walking toward you; C: walking away; D: running.

Rejected. “Did the elevator stop between the time you first saw it and now?” — rejected: spans a temporal interval beyond the current view → L2 Chronological Recall.

###### D.2. L2: Spatiotemporal Context Tracking

L2 evidence has appeared in the video prefix but is no longer visible at query time. The model must retain spatial facts—locations, directions, counts, temporal order—bound to specific places, objects, and timestamps after visual support disappears. The L1/L2 boundary is strict: if the cue is visible near the query frame, the item is L1; if it must be recalled, it is L2.

###### D.2.1. Scene Revisit Recognition

Revisit Detection. Definition. Determine whether the scene currently in view is the same place that was observed at an earlier point in the video prefix. The model must match appearance or spatial layout across temporally separated observations.

Accepted. “Is the room you are looking at now the same living room you walked through earlier?” A: Yes; B: No. The earlier visit is no longer in view; recognition requires memory.

Rejected. “How many times have you been in this hallway?” — rejected: this asks for a count, not a binary detection → Revisit Counting.

Revisit Counting. Definition. Count how many times a specific location has been visited (entered the field of view) across the video prefix up to the query time.

Accepted. “How many times have you passed through this corridor so far?” A: 1; B: 2; C: 3; D: 4. Requires tracking repeated visits over time.

Rejected. “Is this the same corridor you saw before?” — rejected: binary same/different judgment without counting → Revisit Detection.

###### D.2.2. Spatial Memory Beyond the View

Out-of-View Localization. Definition. Recall the location (room, surface, or spatial region) where an object or landmark was last observed, given that it is no longer visible at query time.

Accepted. “In which room did you last see the red backpack?” A: kitchen; B: bedroom; C: living room; D: hallway. The backpack is not in the current view.

Rejected. “The backpack is to your left or right?” — rejected: this asks for egocentric direction relative to the observer’s current heading, not a location label → Behind-Camera Inference.

Behind-Camera Inference. Definition. Infer the egocentric direction (left, right, behind, ahead) of a landmark or object relative to the observer’s current position and heading, given that the target is not currently visible. The key distinction from Out-of-View Localization is that the answer is a direction from the observer, not a place name.

Accepted. “The elevator you used earlier is now in which direction relative to you?” A: behind; B: to your left; C: to your right; D: ahead. Requires binding the elevator’s remembered position to the observer’s current heading.

Rejected. “From the global floor plan, which cardinal direction is the elevator from the living room?” rejected: queries the allocentric map in absolute coordinates → L4 Allocentric Direction Reasoning.

Transient Count Memory. Definition. Recall the number of instances of a specific object class or spatial event observed during the video prefix, where those instances are no longer simultaneously visible.

Accepted. “How many red chairs have you seen in total across all rooms so far?” A: 2; B: 3; C: 4; D: 5. The chairs appeared in different rooms at different times; counting requires memory.

Rejected. “How many chairs are visible in the current frame?” — rejected: all instances are simultaneously visible → L1 Local Topology Description.

###### D.2.3. Chronological Spatial Memory

Chronological Recall. Definition. Recall the temporal sequence in which multiple locations, objects, or spatial events were encountered during the video prefix.

Accepted. “Which room did you visit first — the kitchen or the bedroom?” A: kitchen; B: bedroom. Both rooms have been visited; the order must be recalled from memory.

Rejected. “From left to right, what is the order of doors along this corridor?” — rejected: asks about spatial arrangement visible now → L1 Local Topology Description.

Temporal Depth Estimation. Definition. Estimate how long ago a spatial event occurred relative to the query time, or compare the durations spent in different locations.

Accepted. “Did you spend more time in the living room or the study?” A: living room; B: study; C: about the same. Requires temporal duration tracking across the prefix.

Rejected. “Which room did you visit first?” — rejected: asks for order, not duration or recency → Chronological Recall.

###### D.3. L3: Spatial Simulation and Reasoning

L3 requires the model to operate on spatial structure—mentally rotating objects, predicting consequences of hypothetical changes, verifying consistency across observations, or planning paths—rather than merely retrieving a stored fact. The L2/L3 boundary: L2 asks “what was there?”; L3 asks “what would happen if...?” or “which route...?”. The L3/L4 boundary: L3 targets an operation or decision over a local spatial state; L4 targets a property of the global map structure itself.

###### D.3.1. Spatial Simulation

Mental Rotation & Fitting. Definition. Mentally rotate, flip, or reorient an object and judge whether it fits into a target space, or determine its appearance after transformation. The core ability is spatial transformation without physical execution.

Accepted. “If you rotate the L-shaped sofa 90° clockwise, will it fit into the corner between the wall and the bookshelf?” A: Yes; B: No. Requires imagining the transformed configuration.

Rejected. “Is the sofa currently touching the wall?” — rejected: no transformation is needed; the relation is directly visible → L1 Support Relation or Local Topology Description.

Spatial What-If Consequence. Definition. Predict the spatial consequence of a hypothetical action—moving, removing, or adding an object—on the surrounding layout. This includes reasoning about what becomes visible, reachable, or blocked after the change.

Accepted. “If you remove the bookshelf in the middle of the room, can you see the window from the doorway?”

- A: Yes; B: No. Requires simulating the removal and its effect on line-of-sight.

Rejected. “Has the bookshelf been moved since you last saw it?” — rejected: this compares two observed states rather than simulating a hypothetical → L3 Object-Level Change Detection.

Physical Stability & Feasibility. Definition. Judge whether a proposed spatial action is physically feasible given visible constraints (gravity, collision, clearance, weight). This includes passability judgments that require reasoning beyond simple gap measurement.

Accepted. “Can the refrigerator be moved through this doorway without tilting it?” A: Yes; B: No. Requires comparing the object’s dimensions against the opening and reasoning about physical constraints.

Rejected. “Is the doorway wide enough for a person to walk through?” — rejected: simple gap estimation with no complex physical reasoning → L1 Object Depth Ordering or Absolute Distance.

D.3.2. Spatiotemporal Consistency Verification

Object-Level Change Detection. Definition. Compare the spatial state of a location or object across two different observation times and determine whether a change has occurred (object added, removed, moved, or altered). The model must hold a prior state in memory and compare it against a current or later observation.

Accepted. “Compared with the first time you saw this desk, what object has newly appeared?” A: a laptop;

- B: a vase; C: a stack of books; D: nothing changed. Requires remembering the earlier state and detecting the difference.

Rejected. “What objects are currently on the desk?” — rejected: no comparison across time; only current-view enumeration → L1 Local Topology Description.

###### D.3.3. Spatial Route Planning

Optimal Path Reasoning. Definition. Select the best (shortest, safest, or most efficient) route between two points based on the spatial layout observed so far. The model must reason over remembered spatial structure to compare route options.

Accepted. “To reach the kitchen from here, is it shorter to go through the living room or through the hallway?” A: living room; B: hallway; C: about the same.

Rejected. “Are the kitchen and living room directly connected?” — rejected: asks about topological adjacency, not route optimality → L4 Topological Structure Reasoning.

Obstacle-Aware Rerouting. Definition. Given that a previously viable path is now blocked or unavailable, determine an alternative route using the observed spatial layout.

Accepted. “The main corridor is blocked by construction. Based on what you have seen, which alternative path leads to the exit?” A: through the conference room; B: through the storage area; C: no alternative exists.

Rejected. “Is the main corridor currently passable?” — rejected: binary passability judgment on a visible gap → L1 (if visible) or L2 (if recalled).

Reachability Judgment. Definition. Determine whether a target location is reachable from the current position given the spatial structure observed so far, without necessarily specifying the route.

Accepted. “Based on what you have explored, can you reach the rooftop terrace from your current position?” A: Yes; B: No; C: Cannot determine.

Rejected. “Is the rooftop terrace adjacent to the stairwell?” — rejected: asks about topological adjacency → L4 Topological Structure Reasoning.

###### D.4. L4: Allocentric Spatial Mapping

L4 requires the model to integrate the egocentric video stream into an allocentric (viewer-independent) representation and query its global structure. Evidence typically spans multiple viewpoints and may cover the entire explored region. The L3/L4 boundary: L3 asks “how do I get there?” or “what happens if I change X?” (an operation); L4 asks “what is the map structure?” (a property of the global layout). L4 is reported as three canonical map-query families.

###### D.4.1. Allocentric Direction Reasoning

Allocentric Direction Reasoning. Definition. Determine the global (viewer-independent) directional relationship between two rooms, landmarks, or route segments. Answers are expressed in cardinal directions, clock-face bearings, or absolute spatial terms (north/south/east/west), not egocentric terms (left/right/behind). This family also covers directional ordering: given a reference axis, rank landmarks by their position along that axis.

Accepted. “From the overall layout, which direction is the kitchen relative to the living room?” A: north; B: south; C: east; D: west. Requires building a global frame from the first-person trajectory.

Rejected. “Is the kitchen to your left or right?” — rejected: egocentric direction relative to the observer’s current heading → L2 Behind-Camera Inference (if out of view) or L1 Local Topology Description (if visible).

###### D.4.2. Topological Structure Reasoning

Topological Structure Reasoning. Definition. Reason about the connectivity, adjacency, boundaries, and transitions among spatial units (rooms, zones, floors) as a graph structure. This includes: (a) adjacency and

connectivity—whether two spaces are directly connected without passing through a third; (b) boundary and transition—identifying what separates two adjacent spaces (door, wall, open archway); and (c) local connection graph—enumerating the direct neighbors of a given spatial node.

Accepted. “Which rooms are directly connected to the hallway?” A: bedroom and kitchen only; B: bedroom, kitchen, and bathroom; C: kitchen and bathroom only; D: all four rooms. Requires integrating observations from multiple viewpoints into a topological graph.

Rejected. “To get from the bedroom to the kitchen, should you go left or right at the hallway junction?” rejected: asks for a route decision (operation over the map) → L3 Optimal Path Reasoning.

###### D.4.3. Trajectory Map Alignment

Trajectory Map Alignment. Definition. Match the first-person trajectory observed in the video to a bird’seye-view map, floor plan, or route diagram. This includes path-to-map matching (which map corresponds to the walked path) and route option selection (which drawn route on a given map matches the observed trajectory).

Accepted. “Which of the following floor plans best matches the path you have walked?” [four candidate top-down maps]. Requires converting the egocentric experience into an allocentric trace and comparing against candidate maps.

Rejected. “Based on the map, which route should you take to reach the exit?” — rejected: asks for a planning decision using the map as input → L3 Optimal Path Reasoning.

###### E. Annotation Protocol & Construction Pipeline

This appendix details the full annotation workflow, level-specific construction techniques, and quality-control pipeline referenced in Section 3.2.

###### E.1. Annotation Setup

The benchmark is annotated by a team of 12 trained volunteers, all with prior coursework or research experience in 3D computer vision or embodied perception. Before production annotation, every team member completes a unified training session covering the four-level taxonomy (§3), the item-writing standards in §E.3, and the level-boundary rules. Annotation is performed inside an in-house web-based tool that supports frame-byframe navigation, key-frame jumping, question and option editing, named-entity labeling for spatial regions and landmarks, and source-metadata viewing. Each annotator contributes approximately 67 hours, for a total of about 804 person-hours. Annotators are graduate students and research staff recruited from the authors’ institutions; participation is part of their research duties and no additional monetary compensation was provided. A pilot annotation round precedes production: pilot items are reviewed jointly with two senior authors of this paper—both with prior research experience in 3D computer vision and embodied perception—who serve as adjudicators throughout the project, and feedback from the pilot is folded into the official guideline. During production, the two adjudicators perform periodic spot checks and convene case-by-case discussions for ambiguous items.

###### E.2. Source Datasets and Licensing

OVO-S-Bench reuses publicly released or publicly accessible videos and contributes human-written streaming spatial QA annotations. We do not redefine the license of raw videos; users should obtain source videos under the original dataset or platform terms listed in Table 10.

Training-data overlap. Source videos from Ego4D, ARKitScenes, and other public datasets appear in many MLLMs’ training corpora; although all questions, options, and query timestamps are written from scratch, scene-level familiarity could inflate L4 scores on previously seen layouts. The stable cross-source ranking (Appendix F.6, 𝜌 = 0.92) suggests this effect does not dominate the overall ranking.

- Table 10: Source datasets used in OVO-S-Bench. Video counts use unique normalized source-video keys; question counts are accepted benchmark items.

Source Regime Q. Vid. Levels License / distribution note RoomTour3D [16] Indoor walkthroughs 804 89 L1–L4 Original RoomTour3D terms;

OVO-S releases annotations/evaluation metadata.

Ego4D [15] Egocentric activities 362 90 L1–L4 Original Ego4D terms; raw

videos are not relicensed. Sekai [26] Outdoor/world scenes 85 18 L1–L2 Original Sekai terms. OmniWorld [71] Outdoor/world scenes 153 31 L1–L4 Original OmniWorld terms. YouTube walking tours Public walking tours 17 14 L3 Public URLs/creator terms; re-

distribution follows platform/source terms.

CODa [24] Driving 148 16 L1–L4 Original CODa terms. Honda HDD [42] Driving 59 40 L4 Original HDD terms. ARKitScenes [4] 3D environments 33 33 L4 Original ARKitScenes terms. VSI-Bench [57] Spatial benchmark-derived 19 17 L1 Original VSI-Bench/source

terms.

###### E.3. General Annotation Protocol

Every item must be uniquely answerable from the visual evidence inside the prefix preceding its query timestamp, without relying on world knowledge, language regularities, or option-elimination heuristics. Annotators record, for each item, the source video, the canonical task label, the question, the options, the answer, the query timestamp, and the evidence interval.

Video Selection Criteria. Annotators choose clips with stable camera motion, clear viewpoints, and sufficient spatial variation for the target level. Indoor walkthroughs, egocentric activities, and 3D-environment renderings are favored for L2–L4, where memory and global mapping require sustained spatial structure; driving and outdoor footage is admitted mainly for L1 metric and dynamic items, where the relevant cue is local.

Query Time and Evidence Interval. The query timestamp 𝑡𝑞 controls which prefix the model can see. Annotators place 𝑡𝑞 only after every cue needed to answer the item has appeared in the stream. For higher-level items, 𝑡𝑞 may be deliberately advanced beyond the last evidence frame—provided the intervening content does not change the answer—to test whether the model retains the spatial fact across a longer prefix. The evidence interval is the shortest temporally contiguous span in the prefix that suffices to determine the answer; it is used only for verification, oracle sampling, and analysis, never as model input. All evidence intervals lie strictly before 𝑡𝑞 and cover only the segments that bear directly on the answer.

Options and Distractors. The correct option must be uniquely supported by the evidence interval. All options share consistent surface form—comparable length, register, and level of specificity—so the answer is not telegraphed by wording. Distractors are visually plausible and stay within the same answer type, scale, and axis as the correct answer (direction vs. direction, metric distance vs. metric distance, and so on). Items in which a distractor is obviously wrong by world knowledge or trivially excludable are revised.

Level Assignment. Level assignment follows the highest spatial capability the item demands. L1 covers instantaneous perception of structure currently in view; L2 covers retrieval of evidence that has appeared in the prefix but is no longer visible; L3 covers operation over current or remembered spatial state, including reasoning about post-edit object or state changes; L4 covers global structure, route relationships, and allocentric mapping. When an item invokes more than one capability, it is labeled at the highest one that is required to answer it.

###### E.4. Level-Specific Construction Techniques

We organize this section by tooling rather than by level: the default workflow handles most items across L1–L4, and three specialized techniques apply only to specific task families.

Default Workflow. The default applies to L1, L2, and the L3 Spatial Simulation and Spatial Route Planning families and the L4 Allocentric Direction Reasoning and Topological Structure Reasoning families. For L1 the query timestamp is placed where the relevant cue is currently visible; for L2 it is placed after the cue has left the current view, with an earlier interval marked as evidence, and the annotator verifies that the cue is not re-disclosed by any later frame in the prefix. For Distance Estimation (under L1 Egocentric Metric Perception), inter-object distances are computed from the source video’s coordinate annotations and metadata, and options are designed so that only the metadata-consistent option is correct. Across the L3 and L4 families that follow this workflow, distractors are visually plausible but inconsistent with the layout, clearance, or connectivity that the camera has already traversed in the prefix.

Image Editing. This is the one place where the source video itself is modified, used only for L3 Spatiotemporal Consistency Verification. We use a generative image-editing model to insert a target object into a chosen frame and into the next several seconds of frames, so the edit reads as continuous video content rather than a single-frame artifact. The item is then written on top of the edited stream, with the evidence interval covering the edited segment.

Named-Entity Labels. L3 Spatial Route Planning and all L4 items typically reference multiple rooms or regions, several of which are not visible at the query timestamp. Phrases such as “the kitchen” are often ambiguous in these settings, so before writing the question the annotator names the rooms, regions, and landmarks involved using neutral identifiers (e.g., “Room X”, “Area Y”) and reuses these names verbatim in the question and the options. The named entities are also recorded in the item metadata so that reviewers and downstream analysis can resolve them back to the underlying scene elements.

Bird’s-Eye Rendering. L4 Trajectory Map Alignment—specifically Path-to-Map Matching—additionally requires bird’s-eye material as the candidate options. The correct option is rendered from the source video’s absolute coordinates and metadata; distractor maps are produced by perturbing path shape, orientation, or visit order while keeping the rendering style identical. Every L4 item is double-checked before inclusion to ensure that the question, the named entities, the bird’s-eye material when present, and the answer are mutually consistent.

- E.5. Quality Control Pipeline

- E.5.1. Text-only Probe and Shortcut Filtering

We first run a text-only probe to remove items that can be guessed without the video. Reviewers check whether the wording or the option set leaks the answer through any of: a correct option that is more specific or more wordy than its distractors; obvious world-knowledge giveaways; distractors that are excludable a priori as implausible; or lexical overlap between the question and a single option. For spatial-relation, distance, direction, and trajectory items, reviewers additionally check that distractors stay within the same answer type and a comparable scale, so that no option can be eliminated at a glance. Items that can be answered primarily by language pattern, common-sense inference, or option asymmetry are revised; items whose ambiguity cannot be removed by revision are dropped.

- E.5.2. Blind Cross-Review and Adjudication

Items that survive the shortcut filter are forwarded to a second annotator for blind review. The second reviewer answers the item and inspects the question, options, query timestamp, and evidence interval without seeing the original annotator’s notes. Items on which the two annotators agree are included in the final candidate pool. Items with disagreements are escalated to the two senior authors, who serve as adjudicators (§E.1); they resolve each case by re-examining the video evidence against the annotation guidelines. We log the number of rejected, revised, and finally accepted items. Only items that pass the shortcut filter, survive blind review, and are uniquely supported by their annotated evidence are retained in the released benchmark.

Inter-Annotator Agreement. To quantify annotation reliability, we randomly sampled 150 items (stratified across L1–L4) and had a second trained annotator independently judge each item under the blind-review

- Table 11: Per-item field schema of the released JSONL file.

Field Type Description id int (0–1679) Global unique item index category_index string Within-category item tag, formatted as {subcategory}_{index} source_dataset string Source video dataset name video_id string Video identifier within the source dataset video_path string Relative path to the video file level int (1–4) Taxonomy level task_main_category string Main category code (e.g., 1.1, 4.1) task_subcategory string Subcategory code (e.g., 1.1.1, 2.2.3) task_type_name string Canonical task type name (maps to definitions in Appendix D) question string The spatial question text options dict Answer options keyed by letter (A, B, C, D, ...) query_times list[float] Query timestamp(s) in seconds evidence_times list[list[2]] Evidence interval(s), each [𝑡start, 𝑡end] in seconds answers list[string] Correct answer letter(s), one per query time

|{"id": 0, "category_index": "1.1.1_0", "source_dataset": "CODa_full", "video_id": "10", "video_path": "CODa_full/10.mp4", "level": 1, "task_main_category": "1.1", "task_subcategory": "1.1.1", "task_type_name": "Object-to-Object Absolute Distance", "question": "How far is the yellow wet-floor<br><br>caution sign from the camera?", "options": {"A": "About 0.5 m", "B": "About 6 m",<br><br>"C": "About 3 m", "D": "About 1.5 m"}, "query_times": [22.8], "evidence_times": [[21.8, 23.8]], "answers": ["D"]}<br><br>|
|---|

- Figure 11: Example item from the released JSONL file (L1, Absolute Distance).

protocol described above (accept, revise, or reject). Cohen’s 𝜅 between the original annotator and the blind reviewer was 0.87, indicating almost perfect agreement [21]. The primary source of disagreement was the L2/L3 boundary for items involving spatial change detection, which motivated a guideline update in Round 3.

###### E.5.3. Iterative Guideline Refinement

Recurring disagreement patterns are summarized into guideline updates and pushed back to the team between annotation rounds, so that subsequent items inherit the correction without re-litigating the same edge cases. We maintain a versioned guideline document that records each update and the issue it resolved.

###### E.6. Data Format & Export

The benchmark is released as a single JSONL file (ovo_s_bench_l1_l4.jsonl), with one JSON object per item (1680 items total, covering L1–L4). Table 11 describes each field. An abbreviated example is shown in Figure 11.

Notes. The three list fields query_times, evidence_times, and answers are equal-length and positionally aligned. Most items have a single query time; the exception is Revisit Counting (2.1.2), where the same question is posed at multiple timestamps during streaming playback and the correct answer increments as more revisits accumulate. All evidence intervals lie strictly before their corresponding query time, enforcing the online protocol. The number of options is typically four but may vary (up to seven for Revisit Counting).

###### F. Extended Comparison with Prior Benchmarks

Table 1 positions OVO-S-Bench against 14 representative spatial and streaming video benchmarks. This appendix expands each row into a structured comparison: what the benchmark does, the point on which it converges with our work and supports a specific claim of ours, and the remaining difference. We close with a synthesis of how OVO-S-Bench resolves the recurring gaps.

###### F.1. Image, Multi-image, and Top-view Benchmarks

EmbSpatial-Bench [9]. An automatic single-image MCQ benchmark built from MP3D, AI2-THOR, and ScanNet, with 3.6K items on six egocentric relations (above/below/left/right/close/far). Shared point. Even the strongest LVLMs at the time (GPT-4V 36%, Qwen-VL-Max 49%) trail human accuracy (90%) on basic egocentric relations, supporting our claim that current MLLMs are far from saturating even L1 perception. Difference. A static image strips away every temporal cue, so L2 memory, L3 simulation, and L4 mapping are all absent, and the streaming protocol does not apply.

TopViewRS [22]. A top-view map benchmark over 7 Matterport3D scenes (11.4K MCQs) covering recognition, localization, static spatial reasoning, and dynamic spatial reasoning along a navigation path. Shared point. It confirms a >50% human–model gap that widens monotonically from recognition to spatial reasoning, paralleling the L1→L4 difficulty gradient we observe. Difference. The bird’s-eye map is provided directly as model input, so the agent never has to reconstruct an allocentric layout from an egocentric stream; OVO-S-Bench requires exactly that reconstruction at L4.

MMSI-Bench [60]. A fully human-curated multi-image benchmark (1K Q, ∼2.55 images per question) covering 10 atomic relation types plus a multi-step reasoning split. Shared point. It documents a ∼55-point human–model gap on multi-image spatial reasoning and identifies four recurring failure modes (grounding, overlap-matching, situation-transformation, spatial-logic) that closely match the cross-frame errors we observe at L2. Difference. The setting is discrete multi-image rather than continuous video; there is no streaming protocol and no L3 simulation or L4 allocentric mapping.

###### F.2. Offline Video Benchmarks

VSI-Bench [57]. Template-generated MCQs over 288 indoor walkthroughs (ScanNet/ScanNet++/ ARKitScenes), 5K items across configuration, measurement, and spatiotemporal tasks. Shared point. It was the first to show that linguistic CoT prompting hurts spatial-video QA and that the dominant bottleneck is spatial reasoning rather than perception or language understanding – both findings we replicate (Tab. 3, §4.3.1). Difference. The model has full offline access to the video at query time; no L4 allocentric map queries are defined; and all evidence remains re-attendable across the entire clip.

DISJOINT-3DQA [43]. A generative QA benchmark over synthetic Aria homes (5.4K Q) constructed so the two queried objects are never co-visible in the same frame. Shared point. It provides the cleanest existing test for cross-frame spatial memory – precisely the L2 capability we formalize – and quantitatively shows performance degrades as the spatial separation of the two anchors grows. Difference. The benchmark is synthetic and single-source, has no streaming protocol, and is limited to L1+L2; L3 spatial simulation and L4 allocentric mapping are absent.

STI-Bench [25]. A 2K-question benchmark over 300 videos from Waymo, ScanNet, and Omni6DPose, covering eight precise spatial-temporal estimation tasks from dimension and pose to displacement, speed, and trajectory. Shared point. It confirms that even SOTA proprietary MLLMs (Gemini-2.5-Pro 41.4%) fail at precise quantitative spatial-temporal estimation, supporting our argument that L1 metric perception is far from saturated. Difference. Tasks are pure numerical estimation under offline access; there is no streaming protocol, no L3 simulation or route planning, and no L4 allocentric mapping.

MMSI-Video-Bench [27]. A fully human-annotated video extension of MMSI-Bench with 1.1K Q over 1.3K clips drawn from 26 datasets, organized into Spatial Construction, Motion, Planning, Prediction, and CrossVideo Reasoning. Shared point. It reports the largest known human–AI gap on video spatial intelligence (∼58 points), finds that spatially fine-tuned models fail to generalize, and that neither 3D cues nor CoT yield meaningful gains – three observations we corroborate in our scaling and CoT analyses. Difference. The protocol is offline-video with no prefix-only constraint, items are not paired with query timestamps and evidence intervals, and the taxonomy contains no L4 allocentric-mapping tasks; “multi-video” here means cross-clip rather than cross-room within a single egocentric stream.

VSI-SUPER [59]. Two long-horizon stress tests (VSR for out-of-place object recall and VSC for continual counting) over 10–240min room-tour videos. Shared point. It demonstrates empirically that brute-force context expansion alone cannot close the spatial-streaming gap – Gemini-2.5-Flash exceeds its context window before 120min and scores near zero on VSC – supporting our motivation that streaming requires native selective spatial memory. Difference. Only two needle-in-haystack-style tasks are defined; the full L1–L4 taxonomy is absent, the query is not constrained to a strict causal prefix at evaluation time, and there is no allocentric L4 layer.

###### F.3. Streaming Video Benchmarks

StreamingBench [29]. The first streaming-video MCQ benchmark, with 4.5K Q over 900 YouTube clips spanning 18 real-time visual, omni-source, and contextual tasks. Shared point. It pioneered the prefix-only protocol we adopt and showed that even the strongest model trails humans by ∼25 points, validating streaming as a fundamentally harder regime. Difference. Spatial understanding is one task in 18; the rest target event, audio, and conversation context; L3 simulation and L4 allocentric mapping are not defined.

OVBench [18]. A streaming benchmark with ∼7K Q over 7 source datasets, organized as 16 fine-grained subtasks along the Past/Current/Future temporal axis. Shared point. It formalizes the Past/Current/Future temporal scope we re-use, and shows that offline MLLMs adapted via sliding windows outperform natively online models – a pattern we also replicate. Difference. Subtasks are event-level (action, object, trajectory) rather than spatial-structure-level; neither L3 spatial simulation nor L4 allocentric mapping is defined.

OVO-Bench [37]. A human-curated streaming benchmark (2.8K Q) over 12 source datasets spanning Backward Tracing, Real-Time Visual Perception, and Forward Active Responding. Shared point. It confirms that streaming exposes pervasive hallucination and a ∼30-point human–model gap even on its best-performing modes, motivating a strict prefix-only protocol. Difference. Spatial structure is captured only by the single STU task; there is no L3 simulation, no L4 allocentric mapping, and no annotated evidence interval that our oracle and analysis tooling rely on.

OST-Bench [28]. A rule-based streaming benchmark for online spatiotemporal understanding from agent exploration, with 10K Q over 1.4K indoor scenes (Agent State / Agent Visible Info / Agent–Object Spatial Relationships, 15 subtypes). Shared point. It is the closest prior work in spirit to OVO-S-Bench: it enforces streaming over egocentric indoor input, exposes a “spatio-temporal reasoning shortcut” where models avoid retrieving past evidence, and reports a sharp accuracy decline as exploration grows – phenomena we also observe and quantify. Difference. Task families are limited to agent-state and agent–object dyadic relations (L1+L2); there is no L3 spatial simulation, no L4 allocentric mapping, and the source distribution is indoor-only, missing outdoor, driving, and 3D-rendered environments.

ODV-Bench [64]. A streaming benchmark over 6 autonomous-driving datasets with 12 tasks across statictarget perception, dynamic-target prediction, and event-oriented risk analysis. Shared point. It confirms that streaming spatial reasoning is critical in safety-relevant outdoor settings and that online models underperform offline ones even within their target domain. Difference. The source is driving-only with no indoor or 3Drendered environments; the L3-style content is restricted to short-horizon future prediction with no allocentric mapping; query timestamps are very short (∼13s on average), so the prefix-memory pressure is minimal.

- Table 12: GPT-5.4 text-only accuracy on spatial benchmarks (question and options only, no visual input). ∆ is text-only minus random baseline. OVO-S-Bench row from Tab. 2; others from a 20% stratified sample.

Benchmark N Text-only (%) Random (%) Δ (pp) EmbSpatial-Bench [9] 728 67.5 25.0 +42.5 STI-Bench [25] 413 37.1 20.0 +17.1 SpatialTree-Bench [53] 850 32.1 21.9 +10.3 ViewSpatial-Bench [23] 1143 34.1 25.9 +8.2 Mind-Cube [62] 4231 39.7 32.5 +7.2 OVO-S-Bench (ours) 1680 37.1 31.3 +5.8 OST-Bench [28] 1112 44.2 40.5 +3.7 ERQA [62] 119 27.7 26.1 +1.6 SpatialScore-Hard [50] 68 30.9 30.6 +0.3 MMSI-Video-Bench [27] 222 22.1 23.2 -1.1 MMSI-Bench [60] 190 25.8 27.0 -1.2 RoboBench [33] 106 23.6 25.0 -1.4 VSI-Bench [57] 498 27.1 29.3 -2.2

StreamEQA [48]. A hybrid-pipeline streaming benchmark over the HD-EPIC kitchen dataset, with ∼21K Q spanning 42 sub-tasks across three embodied levels (Perception/Interaction/Planning) and three temporal modes (Backward/Realtime/Forward). Shared point. It documents a 1.6× larger human–AI gap on embodied streaming than on non-embodied streaming benchmarks, reinforcing our claim that the embodied spatial dimension compounds streaming difficulty. Difference. The source domain is a single one (kitchen activities); the focus is interaction and procedural planning over object handling rather than spatial structure; our L4 (cross-room allocentric mapping) falls entirely outside its scope.

###### F.4. Synthesis: How OVO-S-Bench Addresses the Recurring Gaps

Three structural gaps recur across these 14 benchmarks. First, spatial benchmarks (image, multi-image, offline video) expose large spatial-reasoning gaps but never enforce a streaming prefix-only protocol, so it remains unclear whether the same gaps survive when evidence is ephemeral. Second, streaming benchmarks enforce causal input but instantiate it on event, narrative, or counting tasks, leaving spatial structure as at most a single subtask. Third, no prior video benchmark, whether streaming or offline, systematically covers all four levels of spatial abstraction; in particular, the L4 allocentric-mapping level (global direction, topology, trajectory–map alignment) is absent from every prior video benchmark.

OVO-S-Bench unifies these three threads. (i) Each item carries a query timestamp and an evidence interval, and the model sees only the prefix preceding the query, so the streaming protocol is enforced at the item granularity rather than the dataset granularity. (ii) Source videos span indoor walkthroughs, outdoor and driving footage, egocentric activities, and 3D-rendered environments – together covering the union of domains explored separately by prior benchmarks. (iii) Tasks are stratified into four levels of spatial abstraction (L1 instantaneous egocentric perception, L2 spatiotemporal context tracking, L3 spatial simulation and reasoning, L4 allocentric spatial mapping), and the previously untested L4 level is annotated with named-entity, bird’s-eye, and topological supervision derived from each source’s metadata. This design yields the first empirical test of whether the spatial gaps documented under offline access also exist (and become larger) when the model must reason from a causal egocentric stream.

###### F.5. Text-Only Shortcut Analysis

To contextualize the residual text-only advantage on OVO-S-Bench, we run the same GPT-5.4 text-only probe (question and options only, no visual input) on a 20% stratified sample of 12 peer spatial benchmarks (Tab. 12). Text-only deltas above random range from +42 to −2 percentage points across the surveyed benchmarks, with OVO-S-Bench’s +5.8 pp (Tab. 2) near the median. OVO-S-Bench’s construction pipeline applies a text-only probe and blind cross-review to suppress language shortcuts (§E.5); the residual delta is within the range observed on benchmarks that employ similar filtering.

Model Paper (%) Rpaper w/o RT3D (%) Rw/o ΔR Gemini-3.1-Pro 59.2 1 54.4 1 0 Qwen3-VL-235B-A22B 53.6 2 49.7 2 0 InternVL-3.5-241B 50.9 6 48.8 3 -3 InternVL-3.5-38B 49.1 8 47.9 4 -4 GPT-5.4 50.9 5 47.8 5 0 Qwen3.5-27B 51.7 4 47.6 6 +2 Gemini-3.1-Flash-Lite 50.8 7 46.8 7 0 Qwen3.5-397B-A17B 52.1 3 46.0 8 +5 Qwen3-VL-32B 48.8 9 45.7 9 0 Qwen3.5-9B 45.8 12 43.6 10 +2

Per-source accuracy for top reported models (scored pool)

80

[Figure 163]

gemini-3.1-pro-preview qwen3-vl-235b-a22b qwen3.5-397b-a17b qwen3.5-27b

71 56 49 62 71 29 27 50 100

[Figure 164]

59 52 40 67 55 27 18 41 100

70

61 48 38 59 52 27 30 32 0

58 52 39 59 50 21 42 27 0

60

gpt-5.4 internvl3.5-241b-a28b

58 52 41 60 52 25 30 9 0

Accuracy(%)

58 54 44 53 54 25 9 55 0

50

gemini-3.1-flash-lite-preview internvl3.5-38b qwen3-vl-32b

58 49 43 59 51 25 27 14 100

- 57 48 37 61 61 32 18 55 0
- 57 49 35 59 57 16 30 36 0

40

qwen3-vl-4b rynnbrain-8b

55 46 30 51 47 24 30 5 0

30

52 44 34 51 50 20 30 36 0

qwen3.5-9b

52 46 35 58 55 24 15 18 0

20

RoomTour3D Ego4DOmniWorld CODa SekaiHondaHDDARKitScenes VSI-Bench YouTube

Figure 12: Per-source accuracy (%) for the top-8 reported models. Sources ordered leftto-right by question count (descending).

- Table 13: Ranking stability after removing RoomTour3D

(48% of items). Rpaper and Rw/o are rankings among 38 reported systems. Spearman 𝜌 = 0.92 (𝑝 < 10−4) across all 38 systems.

###### F.6. Source-Robustness Analysis

RoomTour3D contributes 48% of benchmark questions (Tab. 10). To test whether this concentration drives the main-table ranking, we recompute overall accuracy after removing all RoomTour3D items. Tab. 13 reports the top-10 models under both rankings: the top two (Gemini-3.1-Pro, Qwen3-VL-235B-A22B) hold their positions, and the Spearman rank correlation across all 38 reported systems is 𝜌 = 0.92 (𝑝 < 10−4). Per-source accuracy varies widely (Fig. 12), reflecting genuine cross-domain difficulty rather than a single-source artifact.

###### G. Evaluation Methods Details

This appendix specifies the inference configuration used to produce every score in Section 4. We give the model inventory and provenance (§G.1), the decoding-parameter recipes (§G.2), the default frame-sampling policy (§G.3), the streaming-model ingestion protocol (§G.4), the prompt templates used at the model interface (§G.5), the verbatim policy-specific prompts used in the §A matrix (§G.6), the deterministic answer-extraction rule (§G.7), and the hardware and software stack (§G.8). The governing principle throughout is that every per-model choice (resolution, frame count, decoding temperature, reasoning-mode toggle) follows the recipe published by the model’s authors; nothing is tuned on OVO-S.

###### G.1. Models Evaluated

The set of evaluated models is listed in Table 2 (the main results table, including closed-source proprietary MLLMs, open-source general video MLLMs, streaming video MLLMs, token-compression / memory-based methods, spatially fine-tuned MLLMs, and embodied foundation models). Open-source models are loaded from each author’s official HuggingFace release; no weights, tokenizer, or vocabulary are modified, and inference is driven through the authors’ released code path (vLLM for models whose card lists vLLM as a supported backend; the authors’ provided inference scripts otherwise; see §G.8). Closed-source models are accessed through their providers’ OpenAI-compatible endpoints. To allow re-evaluation against any future API revision, we record the access window: all closed-source runs reported in this paper were collected between 202602 and 2026-04 against the providers’ stable (non-preview) endpoint tier where one exists, and against the latest preview tier otherwise (gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview). The exact endpoint identifier is the model-name string shown in Table 2.

###### G.2. Decoding Parameters

Non-reasoning baseline. Every model whose row in Table 2 does not carry a thinking-mode toggle is run under the same decoding protocol: greedy decoding (𝑇=0), no nucleus filtering, and a 1,024-token output cap. The cap is comfortably larger than the answer-letter response targeted by our prompt (§G.5) and accommodates models that emit a short justification before the letter.

- Table 14: Reasoning-mode decoding recipes for the thinking-variant rows of Tab. 3. Each row reproduces the recipe documented on the corresponding model card. Max tok caps the thinking trace plus answer combined; truncation is tagged T1a (see §C.1).

Family 𝑇 top-𝑝 Thinking toggle Max tok Qwen3-VL-thinking 0.0 – enable_thinking=true 8,192 Qwen3.5-thinking 0.0 – enable_thinking=true 32,768 InternVL-3.5-thinking 0.6 0.95 R1 system prompt + CoT user template 16,384 GLM-4.6V-Flash-thinking 0.0 – chat_template_kwargs.enable_thinking=true 16,384 Grok-4.1-Fast-Reasoning 0.0 – built-in (no user toggle) 4,096

Reasoning variants. For the reasoning-mode rows of Table 3, decoding follows the recipe published by each model’s authors verbatim. Table 14 reproduces these recipes. Two design choices in the table merit note. First, the non-zero 𝑇=0.6 and top-𝑝=0.95 used for InternVL-3.5-thinking are the authors’ R1-style anti-repetition recipe documented on the model card; the same model card prescribes the chain-of-thought user template paired with the R1 system prompt, which is why InternVL-3.5-thinking is the only row that overrides the default prompt template (§G.5). Second, the per-family thinking-token cap (8K–32K) is the published thinking budget for each family; we observed truncation only on Qwen3.5-thinking when the trace exceeds 32K (591 traces across L1–L4, all tagged T1a per §C.1), which is recorded as a model-side failure rather than a configuration artifact.

###### G.3. Default Frame Sampling

For each query at timestamp 𝑡𝑞 the default visual input is 𝑁=128 frames sampled uniformly across the prefix [0,𝑡𝑞]. The choice of 𝑁 is the largest uniform-frame budget that fits inside every evaluated model’s published context window without truncation; Section A reports the sensitivity of overall and per-level accuracy to this choice. Frames are resized so the shorter side equals the spatial resolution prescribed by each model’s training recipe (this falls in the 336–512px range across the evaluation set), and the longer side is preserved up to the model’s published per-frame visual-token cap where one exists (e.g. Qwen3-VL’s max_pixels setting, Flash-VStream’s 50,176-pixel cap). Frame extraction is cached: the decoded 𝑁-frame tensor is keyed by the SHA-1 of the video file and the sampling specification, so a re-run of the same model on the same annotation reads from cache and is bit-for-bit identical to the first run.

###### G.4. Streaming-Model Ingestion Protocol

Native protocol. Streaming-architecture models (StreamingVLM, StreamForest, Flash-VStream in the main results; HERMES, InfiniPot-V, FluxMem, StreamingTOM in the token-compression block) consume the video as a sequential stream and apply a model-side memory or KV-compression mechanism that is calibrated for a specific ingest rate. We therefore do not impose the uniform 128-frame default on these models for their main-result row in Table 2; instead the video is read from the start until 𝑡𝑞 at the model’s published streaming rate, and answers are generated using the resulting compressed state. Table 15 lists, for each streaming model, the ingest rate and the chunk or compression schedule we use. Every entry is the value provided by the model’s authors (the model card, the released inference script, or the corresponding paper); no streaming hyperparameter was tuned on OVO-S, and the compression-side hyperparameters (e.g. StreamingTOM’s contrastive-token retention thresholds) take the authors’ released defaults.

###### G.5. Prompt Templates

Default template. With the exception of the InternVL-3.5-thinking row (see below), every model in every evaluation track receives the same prompt template. The question text and option strings are substituted into the placeholders verbatim:

You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question. Question: {question} Options:

- Table 15: Streaming ingest rate and compression schedule for each streaming-architecture model. All values from the corresponding paper, model card, or released inference script; no streaming hyperparameter is tuned on OVO-S.

Model Ingest rate Memory / compression unit

StreamingVLM 2fps continuous (linear-attention context) StreamForest 1fps 64-frame budget Flash-VStream 2fps continuous, 50,176-pixel per-frame cap HERMES 2fps 16-frame chunk, KV pruned per chunk InfiniPot-V 1fps KV-budget per chunk FluxMem 1fps flux-memory replacement schedule StreamingTOM 1fps 32-frame encoder batch, contrastive-token retention

{options_text} Instructions:

- - Respond with ONLY the letter of your answer (e.g., “A” or “B”).
- - Do not include any explanation or additional text. Your answer:

Chain-of-thought template (reasoning models with a CoT recipe). InternVL-3.5-thinking is the only main-text row whose model card prescribes an explicit chain-of-thought user template alongside the R1 system prompt; for that row we replace the Instructions block above with the following, while leaving the question and option blocks unchanged:

Instructions:

- - Think step by step about the spatial relationships and scene content shown in the frames.
- - After your reasoning, output your final answer on a new line in the format: Answer: X

All other thinking-mode rows (Qwen3-VL-thinking, Qwen3.5-thinking, GLM-4.6V-Flash-thinking, Grok-4.1-FastReasoning) use the default template above; the reasoning trace is gated by the model-side thinking-mode flag listed in Table 14, not by the user prompt.

###### G.6. Policy-Specific Inputs

A practical clarification on the matrix in §A: the eight frame-sampling policies (single@query, nearest-16f@4fps, uniform-32, uniform-64, uniform-128, uniform-256, log-decay-128, oracle-evidence) differ only in the visual input fed to the model, not in the textual prompt. Every policy submits exactly the same multiple-choice prompt produced by the template above, with the same question and option strings. The text-only control baseline reported in the Baseline & Controls block of Table 2 is the same prompt submitted with an empty image list, and is evaluated under GPT-5.4 to bound how much of the benchmark is answerable from textual cues alone.

###### G.7. Answer Extraction

We adopt multiple-choice scoring throughout because it enables deterministic, reproducible evaluation without relying on LLM-as-judge for open-ended answers, which would itself introduce the spatial grounding errors this benchmark is designed to measure.

Model outputs are scored by a deterministic post-processor. For thinking-mode outputs we first remove any <think>...</think> span so that subsequent matching applies only to the post-thinking tail; outputs without a closing </think> after the published thinking budget are treated as truncated and tagged T1a per §C.1. The extractor then tries the following patterns in order on the (de-thinked) response, restricted to the letter range A–G, and returns the first match:

- 1. Within the last 300 characters: a tail Answer: X (including final answer, final).

- 2. Within the last 300 characters: a Cosmos-style <answer> X tag.
- 3. Within the last 300 characters: a bare single letter at the very end (e.g. “...therefore B”).
- 4. Within the last 300 characters: a GLM-style <|begin_of_box|> X marker.
- 5. A single letter at the start of the stripped response.
- 6. Anywhere in the response: answer / choice / option(s): X with a true separator.
- 7. Anywhere in the response: a parenthesized or bracketed letter, e.g. “(B)” or “[C]”.

Responses for which none of these patterns matches are scored as wrong; they are not skipped, and they contribute to the denominator of every per-task and overall accuracy in Section 4. The order above is the tail-first order that minimizes accidental matches against letters embedded in reasoning prose; the only step that scans the full response is step (vi), which requires the trigger word answer, choice, or option(s) followed by a real separator.

###### G.8. Hardware and Software

All open-source inference is run on NVIDIA H-series GPUs (H800 80GB and H200 141GB). Models whose authors release a vLLM-compatible inference path are served with vLLM ≥ 0.11 in offline batched mode; models without such a path (predominantly the streaming-architecture and token-compression entries) are served through the authors’ released inference scripts at the dependency versions pinned in those repositories. All closed-source inference goes through the providers’ OpenAI-compatible HTTPS endpoints. Video frame decoding uses decord as the primary backend with an OpenCV fallback for codecs unsupported by decord. The decoded 𝑁-frame tensor for each (model, query, sampling specification) tuple is cached on disk under a content-addressed key (the SHA-1 of the video file together with the sampling specification), so repeated evaluation against the same query reads from cache and produces a bit-for-bit identical visual input; the cache footprint for the full benchmark is on the order of 50GB.

###### H. Future Work

The first direction is to automate more of the data construction pipeline, following recent efforts that evolve raw video streams into structured spatial supervision. The second direction is model design: mechanisms such as 3D-aware memory, test-time adaptation, world-model imagination, and structured spatial maps may be necessary for robust spatial streaming. The third direction is evaluation: future versions should include more open-ended answers, richer human baselines, and interactive embodied tasks. We also plan to maintain a contamination-aware release protocol, including a lightweight public subset and a protected evaluation split if the benchmark is used for leaderboard evaluation.

###### I. Additional Examples

The following pages present representative examples across all levels and task families of OVO-S-Bench. Each page pairs streaming spatial questions with visual evidence, query timestamps, and answers.

#### Level1.1 Egocentric Metric Estimation

Query Time: 770.0 Question: Based on the current frame, what is the approximate distance from the camera to the black nightstand next to the bed? Options: A: 0.5 meters B: 6 meters C: 4 meters D: 2 meters

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

##### CODa_fullRoomTour3DOmniWorld

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Query Time: 770.0

Query Time: 53.5 Question: In the current view, which is taller, the blue trash can or the metal trash can? Options: A: The blue trash can is taller. B: The metal trash can is taller.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Query Time: 53.5

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Query Time: 165.667 Question: In the current frame, which is closer to the camera: the fire pit or the person? Options: A: The fire pit is closer to the camera.

Query Time: 165.667

B: The person is closer to the camera.

#### Level1.2 Local Spatial Relationships

Query Time: 212.5 Question: In the current view, what best describes the spatial relationship between the white car and the building on the left? Options: A: The car is in direct physical contact with the exterior of the building.

B: The car is separated from the building by a row of parked bicycles.

[Figure 201]

##### CODa_fullRoomTour3DOmniWorld

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

###### Query Time: 212.5

Query Time: 40.8 Question: In the current frame, is the potted plant occluded? If so, what is it occluded by? Options: A: The plant is partially blocked by the wooden column.

###### C: The plant is not occluded and is fully visible.

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Query Time: 40.8

Query Time: 429.667

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Query Time: 429.667 Question: In the current view, what object is directly supporting the empty vase? Options: A: the cabinet B: cannot be determined from the video

C: the floor D: the desk

#### Level1.3 Dynamical Spatial Perception

Query Time: 21.333

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

##### Ego4DCODa_fullOmniWorld

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Query Time: 21.333 Question: Based on the visual flow, what is the camera's dominant translation direction in this segment? Options: A: Translating downward B: Translating upward

C: Translating leftward D: Translating rightward E: Translating forward F: Translating backward

Query Time: 52.4 Question: Based on the visual flow, is the camera currently yawing left, yawing right, or not rotating? Options: A: Yawing left B: Yawing right C: Not rotating

Query Time: 52.4

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Query Time: 6.5

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Query Time: 6.5 Question: In this segment, what is the motion state of the person in dark clothing on the left side of the walkway relative to the surrounding scene? Options: A: The person in dark clothing on the left side of the walkway is moving independently relative to the surrounding scene.

#### Level2.1 Visual Scene Re-identification

Query Time: 113.3 Question: At the current moment, is the camera showing the same place as at 73 seconds? Options: A: Different from the referenced moment B: Same as the referenced moment

##### OmniWorldRoomTour3D

Query Time: 113.3

###### Reference Time: 73.0

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Query Time: 27.3 Question: By the time the video reaches the queried moment, how many times has this same hallway scene been shown in total?

- Options: A: 2 B: 4 C: 1 D: 3

Query Time: 27.3

Query Time: 55.3 Question: How many separate times does the camera wearer walk on an outdoor street with the herringbone brick pattern during the video?

- Options: A: 3 B: 2 C: 1 D: 4

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

##### CODa_full

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Query Time: 55.3

#### Level2.2 Out-of-View State Maintenance

Query Time: 178.5 Question: At the current moment, where is the giant humanoid statue located relative to the camera, outside the current view? Options: A: Out of view to the left B: Out of view to the right

C: Out of view behind the camera D: Out of view in front of the camera

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

##### OmniWorldEgo4D

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Query Time: 178.5

Query Time: 248.0 Question: Earlier in the video, a national flag was visible. Which country's flag was it? Options: A: The flag of the United Kingdom B: The flag of France

C: The flag of the United States D: The flag of Canada

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

Query Time: 248.0

Query Time: 198.3 Question: As the camera moved past the staircase, how many beige balloons were attached near the stairs? Options: A: 4 B: 3 C: 6 D: 5

Query Time: 198.3

RoomTour3D

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

#### Level2.3 Chronological Event Sequencing

Query Time: 12.2 Question: In the video, was the indoor hall visited before or after the street/outdoor walkway with trees? Options: A: The hall was visited after the street. B: The hall was visited before the street.

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

##### CODa_fullEgo4D

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Query Time: 12.2

Query Time: 1133.0 Question: Thinking about the route shown in the video, what is the last indoor room that we visited before going outdoors? Options: A: a toilet B: a dining room C: a kitchen D: a living room

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

Query Time: 1133.0

Query Time: 184.0 Question: Which option was shown for a longer duration in the video: the bathroom or the pool table room? Options: A: the bathroom B: the pool table room

Query Time: 184.0

RoomTour3D

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

#### Level3.1 Spatial What-if Simulation

Query Time: 79.3 Question: After rotating the wall painting in the current view to a vertical orientation, can it pass through the doorway in front? Options: A: Yes B: No

Query Time: 79.3

##### RoomTour3DRoomTour3DOmniWorld

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Query Time: 202.0 Question: In the current configuration, if the green pillow were translated only backward, would it make contact with the TV? Options: A: Yes, it would make contact.

B: No, it would not make contact.

Query Time: 202.0

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Query Time: 99.6

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Query Time: 99.6 Question: In the current view, is there enough space between the pillars in front for a bicycle to pass through? Options: A: No B: Yes

#### Level3.2 Spatiol-Temporal Consistency Verification

Query Time: 10.0 Question: Compared with the immediately preceding moment in the video, which object has newly appeared on the White mortar mixer in the current view? Options: A: A red umbrella B: A Homer Simpson

C: A Hello Kitty D: A Golden Retriever

###### Query Time: 10.0

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

##### Ego4DEgo4DEgo4D

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

Query Time: 44.0 Question: Compared with the immediately preceding moment in the video, which object has newly appeared on the floor in the current view? Options: A: A Golden Retriever B: A teddy bear

C: A yellow parrot D: A Hello Kitty

Query Time: 44.0

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

Query Time: 9.0

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

Query Time: 9.0 Question: Compared with the immediately preceding moment in the video, which object has newly appeared on the square table in the current view? Options: A: A yellow parrot B: A red umbrella

C: A Hello Kitty D: A Homer Simpson

#### Level3.3 Navigation-Oriented Spatial Planning

Query Time: 392.0

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

##### CODa_fullRoomTour3DRoomTour3D

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

Query Time: 392.0 Question: From Bathroom A to Billiard Table, which route is the shortest or most reasonable? Options: A: Bathroom A -> Room A -> Corridor A -> Kitchen -> Billiard Table

- B: Bathroom A -> Kitchen -> Room A -> Living room -> Billiard Table
- C: Bathroom A -> Corridor A -> Room A -> Recreation Room -> Billiard Table
- D: Bathroom A -> Room A -> Corridor A -> Living room -> Recreation Room -> Billiard Table

Query Time: 147.2 Question: If Path is blocked by an obstacle near Tree A, how should one detour from Avenue to Door? Options: A: Detour via Avenue -> Door

- B: Detour via Avenue -> Tree A -> Door
- C: Detour via Avenue -> Tree A -> Path -> Door
- D: No valid detour can be confirmed from the observed layout.

Query Time: 147.2

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

Query Time: 271.0

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

Query Time: 271.0 Question: Without passing through Path B, which option best describes whether Backyard is reachable from Garage based on the observed layout? Options: ...... C: Reachable via Garage -> Backyard D: Not reachable without passing through Path B.

#### Level4 Allocentric Spatial Graph

Query Time: 171.333 Question: Assuming the entrance faces due north, what is the order of the three rooms from east to west? Options: A: Room B, Room C, Room A, B: Room A, Room B, Room C,

C: Room A, Room C, Room B, D: Room C, Room B, Room A

##### RoomTour3DRoomTour3D

Query Time: 171.333

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

Query Time: 720.0 Question: Based on the full video, which of the following most accurately describes the topological relationship between Area A and Corridor A? Options: A: There is no evidence of a connection between the two.,

###### Query Time: 720.0

- B: Area A is an exterior area that leads into Corridor A.,
- C: Area A must be accessed through Area D to reach Corridor A.,
- D: Area A is an open functional space within Corridor A.

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

##### Honda

Query Time: 4144.2

Query Time: 4144.2 Question: Based on the vehicle's driving status in the video, select the correct bird's-eye view trajectory map. Options: A: B: C: D:

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

