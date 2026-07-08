# arXiv:2605.25077v1[cs.CV]24May2026

## WorldCraft: From Camera Navigation to Object Manipulation in Interactive Video World Models

#### Bohai Gu∗1,2, Taiyi Wu2, Yueyang Yuan3, Jian Liu1, Xiaocheng Lu1, Dazhao Du1, Jie Zhang1, Jinxiang Lai1, Shuai Yang4, Xiaotong Zhao2, Alan Zhao2, Song Guo†1

1The Hong Kong University of Science and Technology 2AI Technology Center, Tencent Video, Tencent

3Wuhan University 4Peking University

### Abstract

Recent video-based world models have made pixel-space environments interactive at the camera level: users can navigate viewpoints while the model generates coherent visual continuations. Yet their action spaces remain incomplete: users can move the camera, but cannot act on individual objects. Since real-world interaction is inherently object-centric, such models remain closer to passive scene observers than truly manipulable environments. We present WorldCraft, a framework that expands interactive video world models from camera navigation to object-level trajectory actions. Given a user click and a sketched path, WorldCraft generates future frames in which the selected object follows the prescribed trajectory while the camera continues to navigate the scene. WorldCraft achieves this through a trajectory-centric control pipeline: First, Normalized World Trajectory (NWT) represents user-drawn motion in a camera-invariant world coordinate system and dynamically re-projects it under the current camera pose, separating object motion from camera-induced screen-space displacement; Spatial-Pathway LoRA (SP-LoRA) then injects this world-space signal through the model’s spatialcontrol pathway, adding object manipulation capability while preserving the pretrained camera controller; ﬁnally, Trajectory-Anchored State Persistence (TASP) treats the world trajectory as a persistent spatial state and refreshes autoregressive memory after trajectory-conditioned generation, allowing moved objects to reappear at their updated positions after leaving the camera view. Experiments show that WorldCraft enables accurate object control, preserves the video-based world model’s camera ﬁdelity under camera-only evaluation, and maintains object state across long autoregressive rollouts with off-camera excursions.

### 1 Introduction

A world model [4, 3] is a learned simulator that predicts future states given the current state and an action. Recent video-based world models realize this idea directly in pixel space, such as Genie 3 [2] and WorldPlay [23] have made impressive progress on camera-level interaction: users can navigate viewpoints and the model generates coherent visual continuations. Yet their action space stops at the camera. Real interaction is inherently object-centric. A robot must predict what happens after pushing a cup along a tabletop [24, 27]; a driving simulator must model a pedestrian stepping into the road [20, 6]; an interactive game must allow users to move entities, not just the viewpoint. In all these cases, the action is a continuous trajectory attached to a speciﬁc object. Without such object-level actions, video world models are more like passive scene observers than manipulable environments.

∗Work done as an intern at Tencent AI Technology Center, Tencent Video. †Corresponding author.

Preprint.

- Table 1: Capability comparison. WorldCraft uniquely supports composable camera-object control with autoregressive long-video generation.

Action Space Camera-Traj. Off-Cam. AutoMethod Camera Object Traj. Composable State regressive Trajectory-guided generation

DragAnything [25] ✓ Wan-Move [19] ✓

Interactive video world models

GameCraft [11] ✓ ✓ Genie 3 [2] ✓ ✓ WorldPlay [23] ✓ ✓

WorldCraft (Ours) ✓ ✓ ✓ ✓ ✓

Object-level actions in an interactive world model are not the same as trajectory-guided video generation under a static camera [19, 25]. They introduce three coupled challenges. First, cameratrajectory coupling: when the camera moves, every object’s screen-space position changes even if the object is stationary, so pixel trajectories entangle object motion with ego-motion. Second, controllability preservation: adding trajectory control to an existing camera-capable backbone should not overwrite the camera controller, yet our analysis shows that camera and trajectory control share the same spatial pathway inside the transformer. Third, off-camera state prediction: moving an object changes the world state even when the camera looks away. Autoregressive memory [23] stores only the last observed appearance and position. Thus, if an object moves while out of view, memory alone will anchor it to a stale location when the camera returns. Solving object-level interaction therefore requires three mechanisms: a camera-invariant trajectory representation, a non-destructive adaptation strategy, and a persistent spatial state signal.

We introduce WorldCraft, a framework that equips an interactive video world model with objectlevel trajectory actions while preserving camera control. Given a user click to select an object and a sketched path to specify its motion, WorldCraft generates future frames in which the selected object follows the prescribed trajectory as the camera simultaneously navigates the scene. It augments the WorldPlay [23] backbone through three novel components: Normalized World Trajectory (NWT), Spatial-Pathway LoRA (SP-LoRA), and Trajectory-Anchored State Persistence (TASP). NWT disentangles object motion from camera ego-motion through dynamic re-projection; SP-LoRA adapts the shared spatial-control pathway without disrupting the base camera controller; and TASP uses the trajectory as a global “where” signal that complements autoregressive memory’s “what” signal when the object leaves and re-enters view. Table 1 highlights this distinction: prior world models support autoregressive camera interaction without object actions, while prior trajectory methods manipulate objects only in non-interactive or non-autoregressive settings. To the best of our knowledge, WorldCraft is the ﬁrst to combine both action modalities in a single autoregressive world model. Our contributions are:

- 1. Object-level actions for interactive video world models. We formulate object trajectory control as a new action modality for autoregressive video world models, enabling users to manipulate selected entities while continuing camera navigation.
- 2. Normalized World Trajectory. We lift user trajectories into a normalized world-space coordinate system and dynamically re-project them under the current camera pose, yielding a camerainvariant representation that disentangles ego-motion from object motion.
- 3. Off-camera state prediction. We use the world-space trajectory as a persistent spatial state signal for off-camera objects and refresh autoregressive memory so moved objects reappear at their updated positions.

### 2 Related work

Interactive video world models. Recent video world models have made controllable simulation in pixel space increasingly practical. Early models such as GameNGen [18] and GameGen-

X [1] demonstrated that video generators can be driven by action inputs, but their control spaces remain limited to camera motion or game-style discrete commands. Subsequent camera-centric world models pushed this line further. Yume [13], GameCraft [11], and Matrix-Game 2.0 [5] improve long-horizon visual simulation and camera controllability in open or game-like environments, while WorldPlay [23] provides the strongest open-sourced baseline by combining camera-action conditioning with autoregressive memory. However, across this literature, the action space is still fundamentally viewpoint-centric: users can move the camera, but cannot directly manipulate individual objects. WorldCraft extends this family of autoregressive world models by introducing object-trajectory actions that are composable with existing camera control.

Trajectory-guided video generation. Several works enable object motion control in video diffusion models through trajectory conditioning. DragNUWA [28] introduces trajectory-guided video synthesis by conditioning generation on motion trajectories together with image and text inputs. DragAnything [25] strengthens object-speciﬁc control through entity representations that bind trajectories to selected targets. MotionCtrl [22] further uniﬁes camera and object motion control within a video diffusion framework through dedicated control branches. Most recently, Wan-Move [19] shows that strong trajectory following can be achieved by reusing displaced ﬁrst-frame latent features as in-context conditioning, without redesigning the backbone. In our experiments, DragAnything and Wan-Move serve as the main trajectory-control baselines in the static-camera regime (Table 2). Unlike all these methods, WorldCraft operates in an autoregressive world-model setting with an existing camera-action space, where object motion must remain compatible with simultaneous camera motion and long-horizon memory.

### 3 Method

WorldCraft adds object-level trajectory actions to an autoregressive video world model while preserving its camera-control capabilities. Figure 1 summarises the full pipeline. We ﬁrst describe the base architecture and trajectory injection mechanism (§3.1-3.2), then present three technical contributions corresponding to the three named components: Normalized World Trajectory (§3.3), Spatial-Pathway LoRA (§3.4), and Trajectory-Anchored State Persistence (§3.5).

#### 3.1 Preliminaries and notation

WorldCraft builds on WorldPlay [23], an autoregressive video world model based on HunyuanVideo1.5 [16]. The DiT [14] takes a 65-channel input: 32 channels of noisy latent zt, 32 channels of image conditioning cimg (ﬁrst-frame latent, zero-padded for subsequent frames), and 1 channel of task mask m. Given an initial frame and a sequence of camera actions, WorldPlay generates videos chunk by chunk using a DiT backbone with two camera-control interfaces: an action encoder action_in for discrete camera actions, and Projective Positional Encoding (ProPE) with per-block projections prope_proj(l) for injecting camera pose. During autoregressive generation, each new chunk attends to cached key-value memories from previous chunks [29]. We use Kt ∈ R3×3 and Et ∈ SE(3) to denote the camera intrinsics and world-to-camera extrinsics at frame t. Let π(K,E,P) denote perspective projection of a 3D point into screen space, and let π−1(K,E,p,d) denote back-projection of a screen-space point p at depth d.

#### 3.2 In-context trajectory conditioning

We inject trajectory information by replacing the image-conditioning channels (32-64) with ﬁrstframe latent features displaced to the target positions. Given the ﬁrst-frame latent z0 ∈ RC×1×H×W and N point trajectories {p(tn)}n,t, the trajectory condition cˆtraj is:

cˆtraj[:, t, h(tn), wt(n) ] ← z0[:, 0, h(0n), w0(n) ], ∀ n, t (1)

where (h(tn),wt(n)) is the latent-space coordinate of track n at frame t; unassigned positions remain zero. The model input becomes [zt ; cˆtraj ; m], preserving full compatibility with the pretrained PatchEmbed [16]. This design yields an informative prior even before trajectory-speciﬁc training: displaced ﬁrst-frame features serve as positional cues that the base model’s spatial understanding partially decodes, producing coarse trajectory following at zero shot. The three components below

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### Trajectory Injection Model Architecture

Copy & Replace

###### Causal Self-Attention

×N

Scale

[Figure 5]

FeedForward

||t = 0| | | |
|---|---|---|---|
| |[Figure 6]<br><br>| | |
| | | | |
<br><br>…<br><br>|t|= 1| | |
|---|---|---|---|
| | | |[Figure 7]<br><br>|
| | | | |
<br><br>|t|=|2| |
|---|---|---|---|
| | | |[Figure 8]<br><br>|
| | | | |
<br><br>|t|=|3| |
|---|---|---|---|
|[Figure 9]<br><br>| | | |
| | | | |
<br><br>|
|---|

Linear Zero-init Linear

AdaLN

Re-project

Dproj

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |[Figure 10]| | |
| | | | | | | |
| | | | | | | |

TrajectoryCondition

Action-In

Cross-Attention LayerNorm

Causal Attention Causal Attention

Target Feature Extraction

RT (Dproj)TR-1 (Dproj)-1 I (Dproj)-1

Lift

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Scale

Normalized World Space

Q K V

Causal Self-Attention

[Figure 11]

|Depth Anything V2<br><br>[Figure 12]|
|---|

Linear Linear Linear

AdaLN

Input Frame

Tokens Continues action

Discrete action

[Figure 13]

[Figure 14]

[Figure 15]

###### Autoregressive Generation Pipeline

###### Select Frames as Memory

[Figure 16]

World Model

FirstChunkCurrentChunk

###### Memory Refresh for OOC

[Figure 17]

Trajectory Condition

[Figure 18]

[Figure 19]

[Figure 20]

Generate

Outdated Memory

| |
|---|

[Figure 21]

↑ ← ↓ →

[Figure 22]

| |
|---|

↑ ← ↓ →

GeneratedChunk

↑ ← ↓ →

Trajectory Condition

…

[Figure 23]

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>…|
|---|

…

Misleading

←

←

Memory Bank

[Figure 31]

[Figure 32]

Pending Deletion

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Anchor Frame Update

| |↑| |
|---|---|---|
|←|↓|→|

| |
|---|

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Anchor Frame

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Last Frame

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

…

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

…

[Figure 63]

[Figure 64]

Normalized World Space

→

→

Replace

| |↑| |
|---|---|---|
|←|↓|→|

[Figure 65]

Target Generation (With Trajectory)

Anchor Frame

Last Frame

Camera action Target Feature Zero

Trajectory Condition Latent Noise Latent Finetune with Lora

Trajectory

- Figure 1: WorldCraft overview. (Top-left) WorldCraft lifts a user-speciﬁed 2D trajectory into a camera-decoupled normalized world space and re-projects it into per-frame trajectory conditions under the given camera actions. (Top-right) The trajectory and camera controls are injected through a lightweight pathway-selective LoRA on the spatial-control pathway, while the backbone attention and MLP layers remain frozen. (bottom) During autoregressive generation, WorldCraft updates the anchor frame and memory bank across chunks, and refreshes outdated memories to support longhorizon out-of-camera object reasoning.

build on this injection mechanism: NWT determines what coordinates to inject, SP-LoRA determines which parameters to adapt, and TASP determines how memory interacts with the injected signal across chunks.

#### 3.3 Normalized World Trajectory

We represent object trajectories in a normalized world-space coordinate system instead of raw screen-space pixels. The motivation is that a screen-space trajectory observed in video entangles two factors: the object’s own motion and the apparent displacement caused by camera ego-motion, i.e., changes in the camera viewpoint. At inference time, however, users typically specify only the desired object motion, without manually compensating for how the camera will move. To bridge this gap, we anchor the trajectory to the ﬁrst-frame camera coordinate system and re-project it under the current camera pose at each generation step. This representation decouples object motion from camera ego-motion, allowing camera and object controls to compose naturally, while also providing a spatial signal that remains well-deﬁned even when the object projection leaves the visible frame (off-camera persistence, exploited by TASP in §3.5).

Deﬁnition. Given a user-speciﬁed screen trajectory {pusert } and a reference depth d, we ﬁrst lift each point to a normalized coordinate on the ﬁrst-frame reference plane:

qt = (

), (2)

(xt − cx)/fx (yt − cy)/fy

where fx,fy,cx,cy are the ﬁrst-frame intrinsic parameters. Intuitively, qt describes the object’s position on a ﬁrst-frame-anchored reference plane, rather than its instantaneous screen-space location. At each generation step, we re-project this anchored coordinate into the current camera view:

panchoredt = π(Kt, EtE−0 1 · lift(qt,d)), (3)

where Et denotes the world-to-camera extrinsic matrix and lift(qt,d) maps the normalized coordinate to a 3D point on the reference depth plane. This re-projection automatically folds camerainduced pixel displacement into the trajectory signal consumed by the model.

Algorithm 1 Inference with Normalized World Trajectory Require: First frame I0; user screen trajectory {pusert }Tt=0; camera pose sequence {Et}Tt=0; ﬁrst-

frame intrinsics K0; chunk size C. Ensure: Generated video {It}Tt=1.

# Stage 1: lift user trajectory into world space (once).

- 1: d ← D(I0)[puser0 ] ▷ depth at initial click
- 2: for t = 0,...,T do
- 3: qt ← (

(xusert − cx)/fx, (ytuser − cy)/fy

)

- 4: end for # Stage 2: autoregressive generation with iterative depth reﬁnement.
- 5: for chunk k = 1,..., ⌈T/C⌉ do
- 6: for t in chunk k do
- 7: panchoredt ← π

(

Kt, Et · E−0 1 · lift(qt,d)

)

- 8: end for
- 9: Ichunk k ← WorldModel

(

I<chunk k, {panchoredt }, {Et})

- 10: d ← ReﬁneDepth(Ichunk k, panchoredt

last

, d)

- 11: end for
- 12: return {It}Tt=1

Composable camera-object control. Trajectories extracted from videos by point tracking are screen-space observations that naturally entangle object motion with camera motion:

(

)

, (4)

pt = π

Kt, Et, Pworld(t)

At inference time, however, users typically specify the desired object motion without compensating for the camera trajectory, leading to a mismatch if the raw user path is used directly as screen-space conditioning. NWT closes this gap by anchoring the user trajectory in the ﬁrst-frame coordinate system and re-projecting it under the current camera pose. As a result, user-speciﬁed object motion and model-driven camera motion can be composed without requiring the user to manually anticipate camera-induced screen displacement.

Depth estimation and iterative anchor reﬁnement. Eq. 3 requires the object depth d, which we initialize by querying the monocular depth map [26] at the user’s initial click position: d0 = D(I0)[puser0 ]. In long autoregressive rollouts, however, keeping both the depth and the visual anchor ﬁxed to the ﬁrst frame can accumulate projection errors across chunks. We therefore apply iterative anchor reﬁnement: after each generated chunk, we update the autoregressive anchor to the latest reliable frame, re-estimate the object depth from that frame, and use the updated anchor-depth pair for subsequent re-projection. This forms a closed-loop correction that keeps the trajectory condition aligned with the generated video, reducing geometric drift while preserving the normalized worldspace trajectory as the global control signal.

#### 3.4 Spatial-Pathway LoRA

Camera viewpoint control and object trajectory control may appear to require different mechanisms, but in a video DiT they share the same underlying goal: controlling where visual content appears in

- 3D space. A camera action induces a global spatial transformation over the entire scene, whereas an object trajectory induces a local spatial transformation on a target instance. We hypothesize that both signals should therefore be handled primarily by the model’s spatial-control pathway, rather than by modules responsible for semantic routing or channel mixing. Empirical conﬁrmation. To verify this hypothesis, we measure the relative weight change

∆(rell) = ∥Wft(l) − Wbase(l) ∥F ∥Wbase(l) ∥F

after full-parameter trajectory ﬁne-tuning [12]. The optimizer concentrates updates on the spatialcontrol pathway: the action encoder and ProPE projections change by 10-25× more than attention and feed-forward layers, with detailed statistics reported in Appendix Table 7. This suggests that

trajectory control mainly requires adapting how spatial intent is mapped into feature-space positions, rather than modifying the model’s global attention or semantic processing.

Pathway-selective adaptation. Motivated by this observation, we adapt only the spatial-control pathway with a lightweight LoRA [7], enabling object-level trajectory control while preserving the camera ﬁdelity of the pretrained backbone. Speciﬁcally, we apply low-rank updates only to the action encoder and ProPE projection layers:

α r

, l ∈ {action_in, prope_proj(1..L)}. (5)

W′(l) = W(l) + B(l)A(l) ·

All other parameters remain frozen. Since the pretrained camera-control pathway is largely preserved and the trajectory adapter introduces only a low-rank perturbation, WorldCraft can add objectlevel control without overwriting the backbone’s camera behavior. In contrast, adapting attention Q/K/V or feed-forward layers modiﬁes global routing and feature mixing, which can interfere with camera control, as conﬁrmed by our ablations in §4.4.

#### 3.5 Trajectory-Anchored State Persistence

A world model is, at its core, a state predictor: when the camera looks elsewhere, the world continues to evolve, and a capable model should predict that off-camera state. TASP resolves this via two coordinated mechanisms.

- (i) Trajectory as persistent spatial signal. The world-space trajectory {qt} remains well-deﬁned when the object is off-screen because Eq. 2 does not depend on visibility. During the camera-away interval (t0,t1), the trajectory signal is still injected via in-context conditioning (§3.2): Eq. 3 projects qt into screen coordinates at every step, producing valid spatial tokens even when the projected position falls outside the visible frustum. Upon camera return at t1, the re-projected position lands inside the frame at the correct updated location.
- (ii) Pre-exit memory ﬁltering.

=When the camera turns away from an object at time t0 and returns at t1, the autoregressive memory Mt0

holds a frozen state snapshot: its keys and values encode the object at its pre-departure screen

location. At t1, attention retrieval conﬁdently reproduces that stale location even if the trajectory has moved the object elsewhere.

We resolve this with pending deletion with dynamic mask: at each re-entry chunk, we identify memory frames in the pre-exit zone (the last k temporal latent before off-screen happens) and mask them from the retrieval set if their FOV similarity with the current chunk exceeds a threshold τ:

simFOV(Vf, Vcur) > τ =⇒ f ∈/ M (6)

where Vf and Vcur are the view matrices of memory frame f and the current chunk’s ﬁrst frame. Frames outside the pre-exit zone or with dissimilar FOV are retained, preserving the appearance prior. The two mechanisms are complementary: trajectory supplies the correct where at re-entry, while pre-exit ﬁltering suppresses stale memory context that would affect this updated spatial cue.

### 4 Experiments

#### 4.1 Implementation details

Base model and training. We build on WorldPlay [23], an 8B-parameter video world model based on HunyuanVideo-1.5 [16]. We train WorldCraft using a three-stage progressive schedule. Stage 0 uses real-world videos for domain adaptation; Stage 1 introduces trajectory control on static-camera data using BI attention and SP-LoRA; Stage 2 extends training to dynamic-camera sequences with AR attention. We set the LoRA rank to 32 and otherwise follow the training conﬁguration of WorldPlay. All experiments are conducted on 8 NVIDIA H200 GPUs with AWS cloud serves. Details are provided in Appendix.

Evaluation data. We construct three quantitative test sets plus qualitative demonstrations, all evaluation is on held-out splits disjoint from training: (i) Trajectory Accuracy (TA) set: 50 clips with static camera, paired with object masks from SAM 2 [15] and ground-truth object trajectories from

- Table 2: Trajectory control under static camera (61 frames, 50 clips). All methods share the same ﬁrst frame and trajectory condition. Best in bold.

Visual Quality VBench++ Method PSNR↑ SSIM↑ LPIPS↓ DINO↑ SubjC↑ BgC↑ Temp↑ TE↓

DragAnything [25] 15.97 0.600 0.468 0.777 0.896 0.913 0.938 39.86 Wan-Move [19] 16.42 0.592 0.375 0.782 0.927 0.943 0.985 44.08 WorldCraft (ours) 17.23 0.616 0.363 0.807 0.942 0.945 0.989 38.90

- Table 3: Camera ﬁdelity on camera-only input. WorldCraft preserves camera accuracy at 61 frames and outperforms all methods at the 253-frame extended horizon.

Short-term (61 frames) Long-term (253 frames)

Method RPErot↓ RPEtrans↓ RPEcam↓ PSNR↑ SSIM↑ LPIPS↓ RPErot↓ RPEtrans↓ RPEcam↓

Yume [13] 0.261 0.0143 0.0169 12.39 0.2931 0.5718 0.374 0.0247 0.0285 Matrix-Game 2.0 [5] 0.342 0.0137 0.0196 12.96 0.3235 0.5326 0.162 0.0243 0.0261 GameCraft [11] 0.252 0.0130 0.0157 12.42 0.2861 0.5529 0.198 0.0243 0.0265 WorldPlay (base) 0.120 0.0155 0.0165 13.77 0.3434 0.4700 0.130 0.0262 0.0276 WorldCraft (ours) 0.131 0.0161 0.0170 13.95 0.3474 0.4621 0.123 0.0225 0.0233

CoTracker [10]. (ii) Camera Fidelity (CF) set: 50 clips with dynamic camera and no trajectory. We use the per-latent camera poses (extrinsic Et + intrinsic Kt) extracted by ViPE [8] to test both basic camera controllability. (iii) Composable (Camera + Trajectory) set: a separate 45-clip test set stratiﬁes three camera-rotation buckets (small < 15◦, mid 15-45◦, large ≥ 45◦; 15 clips each). This set is disjoint from the TA set (which has no camera motion) and from the CF set (which has no trajectory), and is constructed speciﬁcally to evaluate the simultaneous camera+trajectory regime that only WorldCraft supports in ablation study.

Evaluation horizon and metrics. We evaluate trajectory control with Trajectory Error (TE) (mean CoTracker [10] L2 pixel error between the tracked object and the speciﬁed trajectory) and we evaluate visual quality using VBench++ [9] consistency scores: Subject Consistency(SubjC), Background Consistency(BgC), Temporal Flickering(Temp); We evaluate camera control with average Relative Pose Errors in translation (RPErot), rotation ( RPEtrans), and camera extrinsics (RPEcam), between camera trajectories estimated from generated videos by ViPE [8] and ground-truth camera motion. We apply Sim(3) Umeyama alignment [17] to compensate for differences in scale and coordinate frames. Visual quality is also reported with PSNR/SSIM [21]/LPIPS [30] following WorldPlay [23].

#### 4.2 Quantitative results

We evaluate quantitatively on the two regimes in which external baselines are applicable: trajectory control under static camera against trajectory-guided single-clip methods: DragAnything [25] and Wan-Move [19] (Table 2), and camera ﬁdelity under camera-only input against camera-controlled world models: Yume [13], Matrix-Game 2.0 [5], GameCraft [11] and WorldPlay [23] (Table 3).

Trajectory control quality. Under static camera with an identical ﬁrst frame and object-trajectory condition, WorldCraft achieves the lowest trajectory error (TA) while simultaneously producing the best pixel ﬁdelity (PSNR/SSIM [21]/LPIPS [30]), semantic consistency (DINO), and VBench++ consistency scores across all 50 TA clips (Table 2).

Camera ﬁdelity. On camera-only inputs, WorldCraft retains the camera-control capability of the base model. At 61 frames, its RPErot is 0.131, compared with 0.120 for WorldPlay, and far below the next-best external baseline (0.252). At 253 frames, a horizon 4× longer than the main protocol, WorldCraft further reduces the error to 0.123, outperforming WorldPlay (0.130). These results show that adding object-level trajectory control does not trade off against camera ﬁdelity; instead, WorldCraft maintains, and in long rollouts slightly improves, the stability of camera control (Table 3).

WorldPlay

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

|D|
|---|

|D|
|---|

|D|
|---|

|D|
|---|

|D|
|---|

|D|
|---|

Ours

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

| |↑| |
|---|---|---|
|←|↓|→|

W A S D

|D|
|---|

|D|
|---|

|D|
|---|

|D|
|---|

|D|
|---|

|D|
|---|

Wan-Move

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Ours

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

W A S D

← → ↑ ↓

- Figure 2: Qualitative comparison of trajectory control. WorldCraft achieves precise and composable controllability, jointly controlling camera motion and target-object trajectories.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

|←|
|---|

|←|
|---|

|←|
|---|

|→|
|---|

|→|
|---|

|→|
|---|

Ours

|[Figure 98]<br><br>| |↑| |
|---|---|---|
|←|↓|→|
<br><br>| |W| |
|---|---|---|
|A|S|D|
|
|---|

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

WorldPlay

|←|
|---|

|←|
|---|

|←|
|---|

|→|
|---|

|→|
|---|

|→|
|---|

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Matrix-game 2.0

|←|
|---|

|←|
|---|

|←|
|---|

|→|
|---|

|→|
|---|

|→|
|---|

- Figure 3: Long-horizon comparisons with off-camera motion. Given the same initial frame and camera actions, the goose moves right while the camera pans left and then returns. WorldCraft maintains scene consistency and, via TASP, recovers the goose at the correct off-camera-updated position when it re-enters view, whereas baselines either lose scene consistency or cannot track the off-camera object state.

#### 4.3 Qualitative results

We present qualitative comparisons along three axes: Trajectory control (Figure 2), long-horizon camera rollout with off-camera demonstration (Figure 3), and part-level control, multi-object control, long-term control (253-frame) of WorldCraft in Figure 4.

Trajectory control. Figure 2 compares WorldCraft with two state-of-the-art baselines in their respective domains. The upper examples show that although WorldPlay produces plausible scenes, it does not support precise object-level control along complex trajectories; The lower examples further show that, under sparse trajectory signals, WorldCraft faithfully follows the prescribed trajectory throughout the rollout, highlighting its superior object-level controllability.

Long-horizon rollout (off-camera demonstration). Figure 3 compares WorldCraft with cameracontrolled world models, including Matrix-Game 2.0 and WorldPlay. In this example, the goose moves to the right, while the camera ﬁrst pans left and then returns right with the same magnitude. Matrix-Game 2.0 exhibits clear scene inconsistency after the camera returns, whereas WorldPlay maintains coherent background structure. Beyond preserving scene consistency, WorldCraft further leverages the TASP mechanism to track the goose even when it moves off camera, correctly predicting its position once the camera returns. This demonstrates WorldCraft’s ability to maintain object-level state over long-horizon rollouts under off-camera motion.

- Table 4: NWT representation ablation. Trajectory error on the composable set across camerarotation magnitudes. World-space trajectories with iterative depth reﬁnement perform best, especially under large rotations.

Representation TE (small rot.) ↓ TE (mid rot.) ↓ TE (large rot.) ↓

Pixel space (raw user traj.) 35.82 40.69 45.28 World space + single-shot depth 33.82 37.69 41.28 World space + iterative depth 30.82 32.10 34.65

Part

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

| |↑| |
|---|---|---|
|←|↓|→|

W A S D

Multi

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

| |W| |
|---|---|---|
|A|S|D|

← → ↑ ↓

|W|
|---|

|W|
|---|

|W|
|---|

|W|
|---|

|W|
|---|

Long

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

W A S D

← → ↑ ↓

|D|
|---|

|D|
|---|

|W|
|---|

|W|
|---|

|W|
|---|

- Figure 4: Extended capabilities. Part: part-level control-the shield follows the trajectory while the body stays still. Multi: multi-object control-three objects steered simultaneously along independent trajectories. Long: 253-frame autoregressive rollout with long trajectory (∼10.5s at 24fps).

WorldCraft also supports (i) Part-level control (Figure 4, top) and (ii) Multi-object control simultaneously (Figure 4, middle), even in (iii) Long-horizon generation (Figure 4, bottom), extending to 253 frames (∼10.5s) with composable camera-object control.

#### 4.4 Ablation studies

Normalized world-space trajectory. We isolate two design choices in the world-space representation: (i) pixel-space versus world-space coordinates, and (ii) single-shot versus iterative monocular depth estimation. Table 4 reports trajectory error after grouping examples by camera-rotation magnitude. Compared with raw pixel-space conditioning, world-space trajectories consistently improve trajectory accuracy, showing that anchoring trajectories in 3D space better composes object motion with camera motion. Iterative depth reﬁnement provides an additional gain, especially under large camera rotations, where repeated re-estimation helps correct projection drift accumulated during autoregressive rollout.

Spatial-pathway LoRA and curriculum. Table 5 ablates both the adaptation target and the training strategy. In Table 5(a), full ﬁne-tuning obtains the lowest trajectory error, but substantially degrades camera ﬁdelity, as indicated by its much higher rotational RPE. Conventional LoRA on Q/K/V and MLP layers, as well as variants that further add V and MLP layers to the spatial pathway, require many more trainable parameters yet do not improve the TE-RPE trade-off. In contrast, adapting only the spatial-control pathway, namely prope_proj and action_in, achieves the best overall balance with only ∼50M trainable parameters, supporting our choice to inject trajectory control through the camera-control pathway rather than generic attention or feed-forward layers. Table 5(b) further shows that, under the same SP-LoRA adaptation, the Static-BI → Dynamic-AR training strategy best preserves both trajectory control and camera ﬁdelity.

- Table 5: Adaptation and training ablation. (a) Spatial-pathway LoRA, provides the best TE-RPE trade-off with the fewest trainable parameters. (b) The Static-BI → Dynamic-AR curriculum gives the best joint preservation of trajectory control and camera ﬁdelity.

Conﬁguration #Params TE↓ RPErot ↓

- (a) Layer selection, trained with Static-BI → Dynamic-AR Spatial-pathway LoRA (ProPE + action) ∼50M 38.90 0.131 Spatial pathway + V + MLP (blocks 28-42) ∼120M 46.60 0.136 Spatial pathway + V + MLP (all blocks) ∼180M 47.65 0.169 Q/K/V + MLP (conventional LoRA) ∼200M 49.43 0.139 Full ﬁne-tune 8B 37.20 0.237

- (b) Training strategy, using Spatial-pathway LoRA

Dynamic data, AR attention ∼50M 46.63 0.170 Dynamic BI → Dynamic AR ∼50M 38.25 0.177 Static BI → Dynamic AR ∼50M 38.90 0.131

### 5 Conclusion

We introduced WorldCraft, a framework that extends camera-controlled video world models with precise object-level action control. WorldCraft identiﬁes the shared spatial-control pathway underlying camera motion and object trajectories, and adapts it with a lightweight pathway-selective LoRA to add trajectory controllability while preserving the base model’s camera ﬁdelity. At the input level, normalized world-space trajectories decouple object motion from ego-motion, enabling composable camera-object control and providing a persistent spatial signal for off-camera motion. Together with TASP-based memory refresh and progressive training, WorldCraft supports long-horizon autoregressive generation with both scene-level consistency and object-level state preservation. These results point toward interactive world models that can not only navigate scenes, but also manipulate and reason about objects within them.

### References

- [1] GameGen-X Authors. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024.
- [2] Google DeepMind. Genie 3: A large-scale foundation world model. Technical report, DeepMind, 2024.
- [3] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
- [4] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations (ICLR), 2020.
- [5] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.
- [6] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.
- [7] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.
- [8] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025.
- [9] Ziqi Huang et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint, 2024.
- [10] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.
- [11] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. volume 2, page 6, 2025.
- [12] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. In Forty-ﬁrst International Conference on Machine Learning, 2024.
- [13] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.
- [14] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- [15] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations (ICLR), 2025.

- [16] Tencent Hunyuan. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [17] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, 1991.
- [18] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In International Conference on Learning Representations (ICLR), 2025.
- [19] Wan-Move Authors. Wan-move: Wan move anything. In Advances in Neural Information Processing Systems (NeurIPS), 2025.
- [20] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European conference on computer vision (ECCV), pages 55–72. Springer, 2024.
- [21] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.
- [22] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A uniﬁed and ﬂexible motion controller for video generation. In ACM SIGGRAPH, 2024.
- [23] WorldPlay Team. Worldplay: Interactive video generation with autoregressive world models. 2025. Tencent Hunyuan.
- [24] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on robot learning (CoRL), pages 2226–2240. PMLR, 2023.
- [25] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.
- [26] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [27] Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Pack Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In International Conference on Learning Representations (ICLR), 2024. Outstanding Paper Award.
- [28] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.
- [29] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 22963–22974, 2025.
- [30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

### A Progressive training

The shared spatial pathway identiﬁed in §3.4 implies that trajectory training and camera control occupy the same parameter subspace. Naïvely training trajectory control therefore risks catastrophic interference with the base model’s camera capabilities.

Three-stage pipeline. We design a progressive training strategy that systematically avoids both failure modes by gradually increasing data complexity and constraining the attention mode:

Stage 0 adapts the pretrained model (trained on synthetic data) to the target real-data domain via fullparameter ﬁne-tuning at very low learning rate (5 × 10−7, 2000 steps). No trajectory conditioning is used (trajectory_rate=0), so no camera–trajectory conﬂict arises. Stage 1 trains trajectory control on static-camera data using BI attention and layer-selective LoRA (Eq. 5). Static cameras eliminate the screen-space entanglement of Eq. 4 (Et = E0 implies pt ≈ Pworld(t)), providing

[Figure 129]

[Figure 130]

- Figure 5: Curated training set statistics (N=27,027 clips after ﬁltering). (a) Representative samples with the ﬁrst frame, the SAM2 mask contour of the selected subject (blue), and the multi-point trajectory overlay (start in green, end in red, path in yellow). Subjects range from vehicles and pedestrians to pushed or carried objects under diverse weather and lighting. (b) Distribution of object displacement magnitude, measured as the net 2D displacement of the subject centroid across the 97-frame window, normalized by the frame diagonal. The distribution is right-skewed with

median 15.2% and p95=43.6%, covering small to large object motions. (c) Joint distribution of camera translation (world units, symlog axis) and object displacement (% diagonal). Using thresholds of 0.5 world-units for camera and 10% diagonal for object, four regimes partition the dataset: 47% static-cam / moving-obj (purely object-centric), 23% static-cam / static-obj, 7% moving-cam / static-obj (pure ego-motion), and 23% moving-cam / moving-obj, the WorldCraft-speciﬁc regime that demands composable control and is absent from most existing trajectory datasets.

a clean trajectory→object-motion mapping. LoRA protects camera parameters in the frozen base weights. Stage 2 extends to dynamic-camera data with AR attention, teaching the model to handle simultaneous camera and object motion.

### B Scalable data curation pipeline

No existing world-model dataset provides the structured supervision our method requires: each training sample must contain a video clip, camera intrinsics and extrinsics, a per-frame binary mask identifying the moving subject, and a multi-point trajectory describing its motion. We build an automatic pipeline that extracts these (video, camera, mask, trajectory) tuples from unlabeled video at scale, using only off-the-shelf vision models and physical-plausibility ﬁltering. Figure 6 illustrates the end-to-end ﬂow.

Camera estimation. For videos lacking camera annotations (e.g., WISA-80K), we run ViPE to recover per-frame intrinsics and SE(3) poses at metric scale. For SpatialVID-HQ, camera parameters are provided as normalized intrinsics and world-to-camera quaternion–translation pairs; we convert these to pixel-space intrinsics and world-to-camera 4×4 matrices, then Slerp-interpolate the sparse annotation frames (sampled at ⌊fps/5⌋ intervals) to the full frame rate.

Subject discovery. The central challenge is identifying which object to track—videos may contain dozens of moving entities, and only a subset exhibit the kind of coherent, spatially signiﬁcant motion

[Figure 131]

- Figure 6: Automatic data curation pipeline. Given unlabeled video, we extract camera parameters, discover the salient moving subject, track it with SAM2 to obtain per-frame masks, and run CoTracker to produce multi-point trajectories. Physical-plausibility ﬁlters remove degenerate samples. The pipeline adapts to two data sources with complementary strengths: WISA-80K contributes diverse real-world scenes via a fully automatic VLM-guided discovery; SpatialVID-HQ provides metric camera annotations that we combine with a novel tracklet-based subject selection.

suitable for trajectory training. We employ two complementary strategies depending on the data source:

- • VLM-guided discovery (WISA-80K). We sample 5 frames uniformly from each clip and query a vision-language model (Qwen3-VL-8B) with a structured prompt asking it to identify the most salient moving subject and describe its appearance. The VLM response is parsed into a text query, which is passed to GroundingDINO to produce a bounding box localized in the video. This approach requires no category-speciﬁc priors and naturally adapts to open-vocabulary scenes.
- • Multi-frame tracklet matching (SpatialVID-HQ). SpatialVID-HQ provides binary dynamicregion masks (dyn_masks) that mark all moving pixels per frame, but do not distinguish individual objects. In crowded scenes a single frame may contain 30–300 connected components of varying size; naïvely selecting the largest component yields multi-person merged blobs (area ratio > 0.3) rather than individual entities. We instead perform cross-frame association: on each annotated frame, we extract connected components ﬁltered to 0.1%–30% of image area, then greedily match components across frames by centroid proximity (threshold: 15% of image diagonal). The resulting tracklets capture per-object temporal persistence; we score each tracklet by score = nframes×sarea×scoherence, where sarea peaks for objects occupying 0.5%–15% of the frame and scoherence penalizes erratic centroid jumps. The top-scoring tracklet yields the target frame, bounding box, centroid, and a per-pixel component mask for SAM2 initialization.

Tracking and trajectory extraction. Given the discovered subject (bounding box and, when available, centroid and component mask), we initialize SAM2 video segmentation with a compound prompt: per-pixel mask logits provide the strongest initialization signal, the centroid serves as a positive point to anchor identity, and the bounding box acts as a spatial fallback. This triple-prompt strategy substantially reduces identity switches in crowded scenes compared to box-only prompting. SAM2 propagates bidirectionally from the prompt frame, producing per-frame binary masks.

- Table 6: Training data statistics. “Camera source” indicates whether camera parameters are estimated by our pipeline or provided by the dataset. “Subject method” indicates how the target object is identiﬁed.

Dataset Raw After curation Camera source Subject method Traj. points

WISA-80K 80K 8.5K ViPE (estimated) VLM + GroundingDINO 20 SpatialVID-HQ 100K 24K Provided (metric) Tracklet matching 20 OmniWorld — 2K Provided Provided —

Total — ∼34.5K — — —

We then identify the optimal 97-frame window by sliding over the SAM2 output and selecting the interval with maximum mask coverage (≥ 30% of frames must contain a valid subject mask). Within this window, we verify that the subject exhibits meaningful displacement: centroid net displacement must exceed a minimum threshold, ﬁltering out near-stationary objects whose “motion” is merely camera-induced parallax.

Finally, CoTracker3 tracks 20 query points seeded from the subject mask (one centroid plus 19 uniformly sampled interior points), producing dense multi-point trajectories over 97 frames. The center-point trajectory serves as the primary training signal; the remaining 19 tracks provide auxiliary supervision for spatial extent.

Training data summary. Table 6 summarizes the curated datasets. All videos are standardized to 30fps and 97 frames (≈3.2s). Camera parameters are converted to a uniﬁed format of perframe 3×3 intrinsic and 4×4 world-to-camera matrices. Video latents are pre-cached through the HunyuanVideo VAE encoder to accelerate training. Figure 5 reports sample-level statistics of the 27,027 ﬁltered clips: object-displacement magnitude and the joint distribution of camera and object motion.

### C Additional analysis details

We additionally present a series of activation-level experiments that progressively characterize how trajectory control interacts with the camera pathway inside the transformer. All activation experiments probe the prope_proj output of each of the 54 DiT double-stream blocks (the shared spatialcontrol pathway identiﬁed in §3.4). We perform a single denoising step per forward pass rather than full generation: since all our measurements compare relative signals (e.g. camera-only vs. camera+trajectory under identical noise input), a ﬁxed step t serves as a control, and trends reported below are stable across step choice.

How large is the trajectory signal? We perform four forward passes with different input conditions (baseline, camera-only, trajectory-only, combined) and decompose the per-layer activation delta. Figure 7(a) shows the trajectory signal as a thin additive layer on top of the camera signal across all 54 blocks. The stacked-bar structure conﬁrms that LoRA does not overwrite the camera representation; it adds a small trajectory-speciﬁc perturbation (mean traj/cam energy ratio = 0.42).

Is the camera effect direction preserved? To directly test whether camera control is preserved under trajectory conditioning, we employ a 2 × 2 counterfactual design: two camera poses (A, B) crossed with two trajectories (α, β). For each layer, the camera effect vector is cα = h(B,α) − h(A,α); camera invariance is measured by cos(cα,cβ). A value near 1 indicates that the direction of the camera effect is unchanged by trajectory variation. Critically, this probe is evaluated on the prope_proj output, which encodes continuous camera pose (viewmat) and is the layer that shares parameters with the trajectory LoRA. If trajectory updates had corrupted the continuous camera pathway, the camera effect direction would rotate; if only the added perturbation were small but misaligned, the magnitude would match but the direction would drift. Figure 7(b) shows that from block 5 onward, cos(cα,cβ) is consistently above 0.85, with a cross-block mean of 0.89. This demonstrates that the camera effect direction is highly stable regardless of trajectory input, conﬁrming that pathway-selective LoRA achieves an asymmetric decoupling: camera control is preserved while trajectory control is added.

(a) Trajectory as additive perturbation on camera

(b) Camera effect invariance across trajectory variation

(c) Camera preservation across LoRA strategies

1.00

Camera effect

1.0

Trajectory effect

0.6

Activationdelta(prope_proj)

Cosinesim(basevsvariant)

0.8

0.5

0.95

0.4

cos(,)cc

0.6

0.90

0.3

0.4

0.85

0.2

0.2

Config A (prope+action)

IQR

0.1

- Config E (+V+MLP 28-42)

- Config F (Q/K/V+MLP)

- Config G (A+V+MLP all)

Median cos(c , c )

0.80

Mean = 0.894

0.0

0.0

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

Block

Block

Block

- Figure 7: Activation-level analysis of camera–trajectory interaction. (a) Trajectory effect is a small additive perturbation on top of the camera signal across all blocks (stacked bars on prope_proj).

(b) Counterfactual camera invariance: cos(cα,cβ) per block, where c is the camera effect vector measured under two different trajectories. Mean = 0.89, conﬁrming that trajectory variation does not alter the camera effect direction. (c) Layer selection ablation on camera preservation: perblock cosine similarity averaged across all hooked layer types (img_mod, prope_proj, Q/K/V, MLP, img_attn_proj) between base model and each LoRA variant under camera-only input. Pathwayselective adaptation (prope+action) preserves camera activations far better than Q/K/V LoRA, which causes signiﬁcant degradation in mid-to-late blocks.

0.2 0.1 0.0 0.1 0.2 0.3 0.4

PC1

0.20

0.15

0.10

0.05

0.00

0.05

0.10

0.15

PC2

Block 5 overlap=0.40

Camera

Traj

0.4 0.3 0.2 0.1 0.0 0.1 0.2

PC1

0.20

0.15

0.10

0.05

0.00

0.05

0.10

0.15

0.20

Block 15 overlap=0.39

Camera

Traj

2.5 2.0 1.5 1.0 0.5 0.0 0.5 1.0

PC1

1.5

1.0

0.5

0.0

0.5

1.0

Block 27 overlap=0.72

Camera

Traj

0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00

PC1

0.6

0.4

0.2

0.0

0.2

0.4

0.6

Block 40 overlap=0.59

Camera

Traj

0.50 0.25 0.00 0.25 0.50 0.75 1.00 1.25 1.50

PC1

0.6

0.4

0.2

0.0

0.2

0.4

0.6

0.8

Block 50 overlap=0.59

Camera

Traj

PCA: camera vs trajectory token deltas (prope_proj)

- Figure 8: Shared control subspace. 2D PCA projection of token-level camera-control updates u (blue) and trajectory-control updates v (red) at the peak block. The two distributions are aligned along the same principal directions rather than forming orthogonal subspaces, indicating that trajectory control is injected within the camera-compatible spatial-control subspace.

Do camera and trajectory share a feature subspace? A deeper question is why trajectory control can be added without destructively interfering with the original camera-control ability. We analyze token-level activation updates induced by camera control, u = hcam − hbase, and trajectory control, v = hcam+traj −hcam. Using PCA and cosine-based subspace overlap from principal angles, Figure 8 shows that the two updates are most aligned in the middle layers, where spatial layout and geometric control are primarily represented. Early and late layers show lower overlap, indicating that the two signals remain sufﬁciently distinguishable for low-level visual encoding and ﬁnal rendering. Together with the preserved camera accuracy in Table 5, this supports our claim that WorldCraft adds object-level control by reusing the existing spatial pathway while avoiding destructive interference with camera control.

### D Limitations.

Our mechanism only persists the state of entities with user-speciﬁed trajectories; predicting the uninstructed dynamics of the broader off-camera world remains an open problem. Camera-trajectory compensation relies on monocular depth estimation, which introduces projection error at large camera rotations. Trajectory control operates at the granularity of latent tokens (16×16 pixels), limiting precision for very small objects.

### E Broader impact.

WorldCraft takes a step toward world models that support not only passive observation but active manipulation, a capability relevant to embodied AI, content creation, and simulation. Beyond inter-

- Table 7: Top-30 parameters ranked by relative weight change ∆rel = ∥Wft−Wbase∥F/∥Wbase∥F after full-parameter trajectory ﬁne-tuning of WorldPlay (8B). The ranking is dominated by action_in (ranks 1–2), prope_proj (22 of the top 30 rows), and the final_layer adaLN modulation, conﬁrming that the optimizer concentrates updates on the spatial-control pathway. No attention Q/K/V, attention-output projection, or MLP parameter appears in the top 30.

Rank ∆rel ∥∆W∥F ∥Wbase∥F Params Category Parameter

- 1 0.9619 0.2835 0.2947 4,194,304 action action_in.mlp.W
- 2 0.1725 0.0037 0.0213 2,048 action action_in.mlp.b
- 3 0.1371 0.1214 0.8860 4,194,304 prope blk53.prope.W
- 4 0.0996 1.7929 18.0008 8,388,608 other final.adaLN.W
- 5 0.0991 0.0017 0.0171 2,048 prope blk53.prope.b
- 6 0.0945 0.1342 1.4196 4,194,304 prope blk52.prope.W
- 7 0.0903 0.0018 0.0203 2,048 prope blk52.prope.b
- 8 0.0883 0.0018 0.0206 2,048 prope blk51.prope.b
- 9 0.0827 0.0018 0.0214 2,048 prope blk50.prope.b
- 10 0.0824 0.1103 1.3388 4,194,304 prope blk3.prope.W
- 11 0.0820 0.0981 1.1956 4,194,304 prope blk2.prope.W
- 12 0.0768 0.1212 1.5794 4,194,304 prope blk51.prope.W
- 13 0.0765 0.0018 0.0240 2,048 prope blk32.prope.b
- 14 0.0759 0.1171 1.5435 4,194,304 prope blk5.prope.W
- 15 0.0758 0.0018 0.0240 2,048 prope blk30.prope.b
- 16 0.0755 0.1239 1.6413 4,194,304 prope blk6.prope.W
- 17 0.0753 0.0019 0.0247 2,048 prope blk29.prope.b
- 18 0.0753 0.0018 0.0240 2,048 prope blk31.prope.b
- 19 0.0747 0.0020 0.0268 2,048 prope blk26.prope.b
- 20 0.0745 0.1182 1.5865 4,194,304 prope blk4.prope.W
- 21 0.0739 0.0019 0.0251 2,048 prope blk28.prope.b
- 22 0.0735 0.1269 1.7267 4,194,304 prope blk8.prope.W
- 23 0.0730 0.0020 0.0273 2,048 prope blk25.prope.b
- 24 0.0730 0.1247 1.7075 4,194,304 prope blk7.prope.W
- 25 0.0725 0.0021 0.0286 2,048 prope blk24.prope.b
- 26 0.0723 0.0019 0.0266 2,048 prope blk27.prope.b
- 27 0.0723 0.1304 1.8030 4,194,304 prope blk9.prope.W
- 28 0.0713 0.0021 0.0298 2,048 prope blk23.prope.b
- 29 0.0697 0.0017 0.0247 2,048 prope blk34.prope.b
- 30 0.0697 0.1138 1.6325 4,194,304 prope blk50.prope.W

active control, this elevates trajectory from a user-facing interface to a world-state communication channel: in autonomous settings such as a self-driving simulator where occluded pedestrians continue walking, the trajectory signal lets the world model maintain globally consistent dynamics without continuous visual observation of every entity.

