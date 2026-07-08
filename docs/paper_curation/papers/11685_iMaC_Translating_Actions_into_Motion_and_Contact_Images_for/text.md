# arXiv:2606.09813v1[cs.RO]8Jun2026

## iMaC: Translating Actions into Motion and Contact Images for Embodied World Models

Zhenyu Wu1∗, Xiuwei Xu2∗, Yukun Zhou3, Yifan Li3, Qiuping Deng3, Xiaofeng Wang3, Zheng Zhu3, Bingyao Yu2, Ziwei Wang4, Jiwen Lu2, Haibin Yan1

1Beijing University of Posts and Telecommunications 2Tsinghua University 3GigaAI 4Nanyang Technological University

##### Project page: https://imac-wm.github.io/

Abstract: Embodied world models promise to serve as real-world simulators for robot policy evaluation and closed-loop rollout, but their reliability depends on how precisely they condition future video prediction on robot actions. Existing action-conditioned video models often encode future actions as compact vectors and inject them through learned conditioning modules, leaving the model to infer fine-grained spatial consequences indirectly. This abstraction is limiting for real manipulation, where centimeter-level action differences can determine contact, object motion, and task outcome. Toward more spatially explicit action conditioning, we present images of Motion and Contact (iMaC), an embodied world model that converts future actions into image-like controls to guide video generation with precise robot appearance and robot-scene spatial relations. iMaC first uses the robot URDF and forward kinematics to render future robot-observation control videos (i.e., motion images) from future joint actions. It also predicts depth as an auxiliary signal to strengthen spatial understanding, and uses the resulting 3D pointclouds to build two-stream geometry controls (i.e., contact images) between the current scene and future robot. These controls describe both the future robot observations and the spatial interactions that drive scene dynamics. To enhance long-horizon manipulation, iMaC further leverages a training-time rollout strategy to support minute-level generation and reduce exposure bias across generated chunks. Experiments on eight challenging long-horizon real-robot manipulation tasks show that iMaC can evaluate the relative performance of different policy checkpoints, with world-model success estimates strongly positively correlated with real-world policy performance.

Keywords: Action-Conditioned Video Generation, Robot Policy Evaluation

### 1 Introduction

World models [1, 2, 3, 4, 5, 6] have long been viewed as a foundation for planning and control: an agent can choose actions by predicting their consequences before executing them in the real world. Recent progress in video generation has renewed this idea for robotics, where future states can be represented directly as visual observations rather than manually engineered simulator states. Such embodied world models are especially attractive for robot policy evaluation [7, 8, 9]. Real-world evaluation is slow, expensive, difficult to reproduce across policy checkpoints, and often unsafe for rare failure cases. In contrast, a learned real-world simulator can roll out a policy in generated observations, providing a scalable way to compare policies, analyze failures, and support closedloop improvement without building task-specific physical simulators or collecting large numbers of hardware trials. For this use case, the model must also sustain long closed-loop rollouts: after each generated chunk, its own predicted observation becomes the next reference, so small visual or geometric errors can accumulate over time.

The usefulness of a learned real-world simulator depends critically on whether the world model is truly responsive to the robot’s actions. A policy evaluator must not only produce realistic videos; it must predict how different future actions change contact, object motion, and task outcome. This requirement is particularly stringent in manipulation, where a few centimeters can decide whether a gripper touches an object, misses it, pushes it into a different pose, or causes an entirely different downstream trajectory. Existing robotic world models have made important progress toward controllable video prediction, but many of them still encode actions or proxy actions as compact vectors [7, 10, 9] and inject them through mechanisms such as cross-attention [11], AdaLN [12], FiLM [13], or related learned conditioning modules. Such designs are convenient for adapting large video generators, yet the model must infer precise spatial consequences from an abstract signal. Other works attempt to express actions more explicitly: EVAC [14] and ABot-PhysWorld [15] use projected spheres or projection-based action maps, while Action Images [16] uses Gaussian mixture maps to represent actions. These representations make actions more visible to the video model, but they remain primarily action visualizations. They do not directly control the future robot body state in the generated frames, nor do they explicitly describe the interaction geometry between the future robot and the current scene. Spatially precise action representation that controls both future robot motion and contact-relevant robot-scene geometry therefore remains under exploration.

In this paper, we present images of Motion and Contact (iMaC), an embodied world model that translates future actions into dense image-like controls for future-video prediction. Given an initial multi-view RGB observation and a future action sequence, iMaC predicts the future video while using the action sequence to specify not only where the robot will move, but also how that motion relates geometrically to the observed scene. It first converts future joint actions into motion images: rendered robot-observation control videos produced by applying the robot URDF and forward kinematics to obtain future robot configurations and rendering the future robot body from the camera views. These controls specify the robot’s future visual state directly, reducing the burden on the model to hallucinate robot motion from a compact action vector. To capture how this motion may affect the scene, iMaC further constructs contact images, two-stream geometry controls built from robot and scene pointclouds. One stream measures distances from the current scene to the future gripper, while the other measures distances from the future robot to the current scene, encoding contact-relevant spatial relations between action-induced robot motion and the observed environment. These motion and contact images are injected as video controls, preserving the scalability of image-to-video modeling while making action conditioning spatially explicit. To support longhorizon evaluation, we further propose training-time rollouts in which generated chunks provide the next reference observation, reducing the train-test mismatch that arises during closed-loop generation. We conduct experiments on eight challenging long-horizon real-robot manipulation tasks and show that iMaC can rank different policies with different checkpoints by performance, with world-model evaluation scores strongly positively correlated with real-world success rates.

### 2 Related Work

Video Generation Models for Robotics: Video generation models are increasingly used in robotics for offline data generation, including cross-embodiment demonstration transfer, diverse visual composition, missing-view synthesis, human-robot demonstration alignment, and larger synthetic data engines built with 3D reconstruction or scene editing [17, 18, 19, 20, 21, 22]. Another line uses video models as embodied world models for forecasting observations in planning, policy decoding, manipulation, training, and evaluation [4, 23, 5, 24, 7, 10, 9, 25, 14, 26, 15]. These works show the value of scalable video priors, but manipulation rollout remains sensitive to action representation: compact or latent actions learn spatial consequences indirectly, while sparse projected maps expose limited geometry. iMaC addresses this bottleneck by translating future actions into URDF/FK-based motion images and two-stream pointcloud-based contact images.

Evaluation for Robotic Policies: Reliable policy evaluation is a central bottleneck for robot learning: real-world trials are authoritative but expensive and difficult to reproduce across checkpoints or rare failure cases [27, 28, 29, 30]. Physics simulators, manipulation benchmarks, and real-to-sim

or digital-twin systems improve repeatability and alignment [31, 32, 33, 34, 35, 36, 37, 38, 39, 40], but they still require assets, tuned dynamics, and careful scene construction. Video world models offer a complementary in-silico evaluator, with recent work using action-conditioned rollouts to compare policies, test OOD or safety settings, and obtain scores correlated with real-world performance [7, 41, 8, 9, 42]. iMaC follows this direction while targeting the action sensitivity of long-horizon manipulation through explicit robot-motion and contact-geometry controls.

### 3 Approach

#### 3.1 Problem Formulation and Overview

World model for policy evaluation: Let ot denote the robot RGB observation at time t. A robot policy π maps ot and an optional language instruction l to a future action sequence,

at:t+H−1 = π(ot,l). (1)

Given the current observation and the policy-proposed actions, an action-conditioned world model predicts the future observation chunk

oˆt+1:t+H = fθ(ot,at:t+H−1,l). (2)

During policy evaluation, π and fθ form a closed loop: the policy acts on generated observations, and the world model predicts the visual consequences of those actions. This paper focuses on learning fθ, not improving π itself, but the core objective is to make world-model rollouts reliable enough to compare policy checkpoints and estimate their real-world performance.

IT2V world-model backbone: iMaC builds on a WAN2.2 image-to-video (IT2V) DiT [43]. In our implementation, ot contains one fixed head-camera view and two wrist-camera views, arranged as a single image mosaic so that multi-view prediction can still be handled by a single-image IT2V model. Rollout is generated chunk-wise. The first chunk uses the given initial image as the reference image; for later chunks, the last generated frame of the previous chunk becomes the next reference image. For each chunk, the reference image and target future video are encoded together by the WAN VAE encoder E. Let zr be the clean reference latent and x1 be the clean future-video latent. We sample x0 ∼ N(0,I), τ ∼ U(0,1), and noise only the future latent:

xτ = (1 − τ)x0 + τx1. (3)

iMaC constructs three action-derived control videos: motion images Cm, scene-to-gripper contact images Cs→g, and robot-to-scene contact images Cr→s. After VAE encoding and control-specific patchification, these controls are added to the noised future tokens, while the reference tokens remain clean:

hτ = [Pv(zr) ; Pv(xτ) + Pm(E(Cm)) + Ps→g(E(Cs→g)) + Pr→s(E(Cr→s))], (4)

where Pv is the WAN video patchify layer and Pm, Ps→g, and Pr→s are control-specific patchify layers. The DiT predicts the flow only for future tokens, with objective

0,x1,τ ∥vθ(hτ,τ,l) − (x1 − x0)∥22 , (5)

Lfm = Ex

Besides RGB prediction, iMaC also constructs an auxiliary depth prediction branch, which provides geometric state for constructing pointcloud controls in subsequent chunks. Sec. 3.2 converts actions into URDF/FK-rendered motion images that specify future robot appearance, Sec. 3.3 builds twostream contact images from robot and scene pointclouds to encode contact-relevant geometry, and

- Sec. 3.4 describes training-time rollouts for long-horizon generation.

#### 3.2 Motion Images from Robot Kinematics

Predicting a future manipulation video can be decomposed conceptually into predicting the future robot appearance and predicting the future scene response. The scene response is governed by

Update Ref Images

Last Frame

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Ref RGB Images Ref Depth Images

Scene Reconstruction

Contact Image

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Projection

Robot-centricImage

[Figure 18]

[Figure 19]

VAE Decoder

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Scene-centricImage

Camera Pose Gripper Point

Diffusion Transformer

URDF

[Figure 24]

[Figure 25]

[Figure 26]

Forward Kinematics

[Figure 27]

VAE Encoder

N Step Action Chunk

Motion Image

Noised Future

- Figure 1: Overall iMaC pipeline. Given a reference observation and future actions, iMaC translates actions into motion images from robot kinematics and contact images from robot-scene geometry, injects these image-like controls into an IT2V world model, and rolls out future video chunks for policy evaluation.

contact-rich physical dynamics and is difficult to infer from actions alone. In contrast, the robot part of the future video is largely determined by the commanded future action sequence and the robot kinematic model. This distinction is important for policy evaluation: if the generated gripper or arm deviates from the action actually proposed by the policy, the subsequent contact pattern can be wrong even when the rendered scene remains visually plausible. iMaC therefore avoids asking the world model to infer future robot motion only through a compact action embedding. Instead, it translates the future actions into dense robot-observation control videos, i.e., motion images.

Given the future joint action sequence at:t+H−1, iMaC applies the robot URDF and forward kinematics to obtain the corresponding future robot configurations. Let ϕ denote the robot controller’s action-to-joint-state update, and let KURDF be forward kinematics defined by the robot URDF. For the k-th future step and camera view v, we construct

qt+k = ϕ(qt,at:t+k−1), Mt+k = KURDF(qt+k), Cmt+k,v = R(Mt+k;Kv,Ttv+k), (6)

where qt+k is the future joint state, Mt+k denotes the posed robot model, R is the renderer, and (Kv,Ttv+k) are the camera intrinsics and extrinsics. We render the robot model from the same camera views used by the world model, including the fixed head camera and the two wrist cameras whose poses are obtained from forward kinematics. The rendered three-view robot observations are arranged in the same mosaic format as the predicted video, yielding a control video Cmt+1:t+H. These motion images specify the future robot body and gripper appearance directly in image space. Since the control branch is injected by latent-wise addition to the noised future-video tokens in Eq. 4, the rendered robot observations provide strong pixel-level guidance for the part of the future video whose geometry is known from the robot model.

iMaC is used as a world model for evaluating policies on a specified robot platform. In this setting, access to the robot URDF is a natural requirement, analogous to knowing the hardware platform in real-world policy evaluation. The construction therefore does not require additional human annotation or task-specific labeling beyond the future action sequence already needed for actionconditioned rollout.

#### 3.3 Contact Images from Robot-Scene Geometry

Taming World Model for RGB-D Prediction: Beyond RGB prediction, iMaC predicts depth to improve the world model’s spatial understanding and to provide geometry for subsequent contactcontrol construction. The initial reference depth is estimated from the three RGB views and their camera poses, where the wrist-camera poses are obtained from URDF and forward kinematics, using

Depth Anything 3 (DA3) [44]. For later chunks, depth is predicted by the world model together with RGB, so the generated final frame provides both the next visual reference and geometric state.

To keep depth compatible with the image-to-video backbone, we encode each depth map as a colorized image following VisionBanana [45]. The model input and output are organized as a six-panel mosaic: the first row contains the three RGB views, and the second row contains the corresponding colorized depth views. RGB and depth therefore pass through the same VAE encoder E. Since the future-video latent now has this six-panel layout, each three-view control video is vertically duplicated before control injection, so that the control tokens are spatially aligned with both the RGB and depth rows. We use this dimension-matched version of the controls in Eq. 4, and the injection remains the same latent-wise addition.

Two-stream Geometry Controls: Given the current predicted depth and the future actions, iMaC builds two contact-image streams following a bidirectional robot-scene distance construction. We first remove the robot from the current depth using the rendered robot mask at the reference step, then lift the remaining pixels from all views into a scene pointcloud Pst. Future full-robot pointclouds Prt+k and gripper pointclouds Pgt+k ⊂ Prt+k are obtained from the URDF/FK-predicted robot configurations.

The first stream is robot-to-scene: each future robot point stores its nearest distance to the current scene,

drt+→ks(r) = min p∈Pst

∥r − p∥2 , r ∈ Prt+k. (7) These distances are projected with the future robot pose into the robot render mask and densified inside the mask, producing the robot-centric contact images Crt+1:→st+H. The second stream is sceneto-gripper: each current scene point stores its nearest distance to the future gripper,

dst+→kg(p) = min

∥p − g∥2 , p ∈ Pst. (8)

g∈Pgt+k

These distances are projected back to the current scene pixels and densified inside the scene mask, producing the scene-centric contact images Cst+1:→gt+H. Both distance videos are colorized with a fixed heatmap after sequence-level distance normalization. Together, the two streams tell the world model where the future robot body approaches the scene and which current scene regions are close to the future gripper, providing contact-relevant spatial guidance beyond the rendered motion images.

#### 3.4 Training-time Rollout for Long Video Generation

Long-horizon policy evaluation requires the world model to operate on its own generated observations. At inference time, iMaC predicts a future chunk, takes the final generated RGB-D-style frame as the next reference, constructs new motion and contact images from the next action chunk and the predicted depth, and repeats this process. If training always conditions on ground-truth reference frames, the model only learns under clean contexts, while closed-loop evaluation conditions on imperfect generated contexts. This train-test mismatch is a form of exposure bias and can cause visual, depth, and contact-control errors to accumulate across chunks.

To reduce this mismatch, iMaC performs training-time rollout over multiple consecutive chunks. Given a training sequence, we split it into R chunks of length H. For the first chunk, the reference is the ground-truth initial RGB-D-style observation. For chunk r, we train the model with the usual flow-matching objective in Eq. 5 using the corresponding ground-truth future RGB and depth latents

as targets and the action-derived controls for that chunk. Let x(1r) denote the clean RGB future latent and let d(1r) denote the clean depth future latent. We sample noise and τ as in Sec. 3.1, form noisy latents x(τr) and d(τr), and obtain one-step clean estimates from the predicted flows,

x(τr) = (1 − τ)x(0r) + τx(1r), d(τr) = (1 − τ)d(0r) + τd(1r), xˆ(1r) = x(τr) + (1 − τ)vθx(·), dˆ(1r) = d(τr) + (1 − τ)vθd(·). (9) where vθx and vθd are the RGB and depth flow predictions. The final frames decoded from xˆ(1r) and dˆ(1r) are detached and used as the reference RGB and depth for chunk r + 1. Thus, later chunks are

trained under generated references while still receiving paired supervision from the recorded videoaction sequence. The depth reference is updated in the same way as RGB, so the model also learns to construct subsequent geometric state from its own depth predictions rather than from ground-truth.

Training under self-generated context is also studied in Self-Forcing [46], but the objective and setting are different. Self-Forcing targets open-ended autoregressive video diffusion, where a text prompt can correspond to many plausible videos and a self-generated sample has no unique paired target; it therefore relies on distribution-matching objectives such as DMD [47, 48], SiD [49, 50], or GAN losses [51]. iMaC instead studies action-conditioned robot world modeling. Given the reference observation and future robot actions, each chunk in the recorded manipulation sequence provides aligned RGB-D supervision, even when the reference for that chunk is generated by the model. We therefore keep the standard flow-matching/MSE losses for RGB and depth. Operationally, iMaC also uses a simpler rollout update: rather than unrolling a full autoregressive diffusion sampler with rolling KV cache, it obtains the next reference by a one-step flow clean estimate, VAE-decodes the predicted RGB-D latents, detaches the final frame, and recomputes the next chunk’s action-derived controls from the updated RGB-D reference.

### 4 Experiment

#### 4.1 Experimental Setup

Tasks and data. We evaluate iMaC on eight real-world manipulation tasks that require contactsensitive prediction over closed-loop rollouts. Each task contains paired multi-view RGB videos and robot action trajectories collected from a mixture of teleoperation and policy rollouts, including both successful and failed executions. Appendix A provides task descriptions and visualizations. The observation contains one fixed head-camera view and two wrist-camera views; during training, iMaC additionally uses the corresponding depth-color targets described in Sec. 3.3. At test time, policies act from RGB observations, while depth is only used internally to construct pointcloudbased contact images during chunk-wise rollout.

World-model evaluation protocol. For policy evaluation, a policy is deployed in the learned world model in closed loop: the policy predicts the next action chunk from the current generated observation, iMaC predicts the future video chunk, and the final generated frame becomes the next reference. We evaluate two VLA policy families, π0.5 [52] and GigaBrain-0.5 [53], using three checkpoints from each model. The three checkpoints for each policy family sample early, intermediate, and late stages of training. This protocol evaluates whether world-model scores preserve performance differences both across policy families and across training stages within each family. Each checkpoint is evaluated on the same initial task configurations in both the real world and the iMaC world model, with one or two evaluation groups according to task availability. Each group contains 30 episodes, and repeated groups provide an estimate of evaluation repeatability for the same policy checkpoint. We treat real-world performance as the reference measurement and test whether normalized world-model scores preserve the relative ranking of policies and checkpoints, which is the key property needed for model-based checkpoint selection.

Baselines and metrics. We compare iMaC with action-conditioned world-model baselines that inject future actions through learned action embeddings or sparse action-image controls. The main video-prediction metrics are computed between generated and ground-truth future videos under the same initial observations and action sequences; Table 1 reports task-averaged video-quality metrics. For policy evaluation, we report the correlation between world-model and real-world scores for each task. We ablate URDF/FK-rendered motion images, two-stream contact images, and the source of depth used for contact-image construction.

Implementation details. We train iMaC in two stages. The first stage trains a shared model on data from all eight tasks, and the second stage finetunes a task-specific model on each individual task; the final evaluation therefore uses one world model per task. Because contact images depend on the quality of predicted depth used for pointcloud construction, the first stage uses only motion-

Table 1: Quantitative comparison of future-video prediction quality. We report mean and standard deviation across the eight real-world tasks. Lower is better for MSE, FID, and FVD; higher is better for PSNR and SSIM.

Method MSE ↓ FID ↓ PSNR ↑ SSIM ↑ FVD ↓

Ctrl-World [7] 0.030 ± 0.012 48.64 ± 10.68 16.22 ± 1.74 0.730 ± 0.037 591.47 ± 160.30 ABot-PhysWorld [15] 0.041 ± 0.017 74.23 ± 22.50 14.41 ± 1.62 0.630 ± 0.071 642.98 ± 105.27 iMaC w/o contact images 0.028 ± 0.009 38.81 ± 9.89 16.34 ± 1.39 0.735 ± 0.039 523.94 ± 156.84 iMaC 0.028 ± 0.010 36.96 ± 9.16 16.39 ± 1.41 0.735 ± 0.037 489.51 ± 92.65

NormalizedSuccessRateiniMaCWorldModel

0.5 GigaBrain-0.5 Checkpoint 1

Checkpoint 2 Checkpoint 3

| |
|---|

Task 1

Task 2

###### Task 3

###### Task 4

- 0.8

- 1.0 r=0.956

| | | | | | | |
|---|---|---|---|---|---|---|
| |r=0.93|1| | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

r=0.678

r=0.915

0.6

0.4

0.2

0.0

###### Task 5

###### Task 6

###### Task 7

Task 8

- 0.8

- 1.0 r=0.428

r=0.870

r=0.856

r=0.833

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Success Rate in Real World

- Figure 2: Correlation between normalized policy success rates measured in the iMaC world model

and in the real world. Each subplot evaluates three checkpoints from π0.5 and three checkpoints from GigaBrain-0.5 on matched initial configurations; repeated groups for the same checkpoint share the same color and marker. Most tasks show strong positive correlation, while Tasks 3 and 5 expose the missing-observation failure mode analyzed in Appendix B and Fig. 7.

image controls, and contact-image controls are introduced during the second-stage finetuning after the model produces clearer RGB-D predictions. Training-time rollout is also warmed up: for the first 40 epochs, chunks use clean reference observations before enabling one-step generated references.

#### 4.2 Main Results

Table 1 evaluates future-video prediction before closed-loop policy evaluation. iMaC obtains the best task-averaged FID, PSNR, SSIM, and FVD, while matching the best MSE within rounding, showing that motion images and contact images improve action-conditioned prediction quality by specifying future robot state and dense robot-scene distance cues. Appendix B provides additional rollout and control-video visualizations.

- Figure 2 evaluates iMaC as a closed-loop policy evaluator. Across six of the eight tasks, worldmodel scores are strongly aligned with real-world performance, with per-task correlations between 0.833 and 0.956. This indicates that iMaC usually preserves the relative ranking of policy families and training checkpoints, which is the key requirement for checkpoint selection before additional hardware evaluation. The two lower-correlation tasks, Task 3 (r = 0.678) and Task 5 (r = 0.428), are not arbitrary outliers: both depend on height relations that are weakly observed by the available camera views. In Task 3, the model must know whether the box ear has been lifted high enough to clear the side wall before entering the slot; in Task 5, it must know whether the dustpan entrance is flush with the tabletop or raised above the paper trash. Appendix B analyzes these missingobservation cases.

- w/o Contact w/o Motion DA3 Depth iMaC GT
- Figure 3: Ablation of contact/motion images and the depth source for contact-image construction.

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

[Figure 66]

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

#### 4.3 Ablation Study

Motion and contact images. The first two columns isolate the two action-derived controls. Without contact images, the world model lacks contact-aware guidance for action following: the gripper does not grasp the cloth, even though later frames still generate an interaction-like cloth motion. Without motion images, the model lacks direct guidance about the future robot configuration; the gripper repeatedly attempts the motion but cannot produce the precise grasp.

Depth source for contact images. The third and fourth columns keep motion and contact controls but compare how the depth used for contact-image construction is obtained. Using DA3 depth partially improves action following because both controls are present, but its contact geometry is less consistent than iMaC’s RGB-D world-model state: compared with the ground truth, the gripper still misses the cloth corner. Additional qualitative rollout and control visualizations are provided in Appendix B.

### 5 Limitation

iMaC relies on accurate 3D information to train depth prediction and to construct pointcloud-based contact images. In the current system, depth supervision is estimated by Depth Anything 3 (DA3) from multi-view RGB observations and camera poses, which can introduce centimeter-level errors in manipulation scenes. The two-stream contact images remain useful because they are heatmaps over distance fields, so the model can exploit coarse approaching and separating trends rather than exact metric contact at every pixel. Higher-quality depth sensors or manipulation-adapted depth models should further improve contact timing, collision localization, and long-horizon rollout reliability.

### 6 Concluding Remark

We presented iMaC, an embodied world model that translates future robot actions into motion images and contact images for spatially explicit action conditioning. By combining URDF/FKrendered robot controls, auxiliary depth prediction, two-stream pointcloud-based contact images, and training-time rollout, iMaC gives an image-to-video model direct guidance about future robot state and robot-scene geometry. Across eight real-world manipulation tasks, iMaC is evaluated as a learned real-world simulator for ranking checkpoints of π0.5 and GigaBrain-0.5, with world-model scores positively correlated with real-world performance. iMaC is intended to complement, not eliminate, real-world evaluation: generated rollouts can help rank policies and reduce hardware trials, while final deployment decisions should still account for residual model bias and rare physical failures.

### References

- [1] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.
- [2] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.
- [3] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.
- [4] M. Yang, Y. Du, K. Ghasemipour, J. Tompson, D. Schuurmans, and P. Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023.
- [5] S. Zhou, Y. Du, J. Chen, Y. Li, D.-Y. Yeung, and C. Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024.
- [6] Q. Garrido, T. Nagarajan, B. Terver, N. Ballas, Y. LeCun, and M. Rabbat. Learning latent action world models in the wild. arXiv preprint arXiv:2601.05230, 2026.
- [7] Y. Guo, L. X. Shi, J. Chen, and C. Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025.
- [8] Gemini Robotics Team. Evaluating gemini robotics policies in a veo world simulator. arXiv preprint arXiv:2512.10675, 2025.
- [9] Y. Wang, R. Syed, F. Wu, M. Zhang, A. Onol, J. Barreiros, H. Nayyeri, T. Dear, H. Zhang, and Y. Li. Interactive world simulator for robot policy training and evaluation. arXiv preprint arXiv:2603.08546, 2026.
- [10] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin, et al. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.
- [11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In NeurIPS, volume 30, pages 5998–6008, 2017.
- [12] W. Peebles and S. Xie. Scalable diffusion models with transformers. In ICCV, pages 4195– 4205, October 2023.
- [13] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. C. Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, 2018.
- [14] Y. Jiang, S. Chen, S. Huang, L. Chen, P. Zhou, Y. Liao, X. He, C. Liu, H. Li, M. Yao, et al. Enerverse-ac: Envisioning embodied environments with action condition. arXiv preprint arXiv:2505.09723, 2025.
- [15] Y. Chen, R. Chen, D. Huo, Y. Yang, D. Qi, H. Liu, T. Lin, S. Zeng, J. Xiao, X. Chang, et al. Abot-physworld: Interactive world foundation model for robotic manipulation with physics alignment. arXiv preprint arXiv:2603.23376, 2026.
- [16] H. Zhen, Z. Gao, Q. Sun, Y. Zhao, Y. Yang, Y. Du, T.-H. Wang, Y.-L. Qiao, and C. Gan. Action images: End-to-end policy learning via multiview video generation. arXiv preprint arXiv:2604.06168, 2026.
- [17] L. Liu, X. Wang, G. Zhao, K. Li, W. Qin, J. Qiu, Z. Zhu, G. Huang, and Z. Su. Robotransfer: Geometry-consistent video diffusion for robotic visual policy transfer. arXiv preprint arXiv:2505.23171, 2025.

- [18] Z. Tong, D. Chen, S. Hu, H. Fan, L. Chen, G. Ren, H. Tang, H. Dong, and L. Shao. Fidelityaware data composition for robust robot generalization. arXiv preprint arXiv:2509.24797, 2025.
- [19] Z. Qian, X. Chi, Y. Li, S. Wang, Z. Qin, X. Ju, S. Han, and S. Zhang. Wristworld: Generating wrist-views via 4d world models for robotic manipulation. arXiv preprint arXiv:2510.07313, 2025.
- [20] H. Li, I. Zhang, R. Ouyang, X. Wang, Z. Zhu, Z. Yang, Z. Zhang, B. Wang, C. Ni, W. Qin, et al. Mimicdreamer: Aligning human and robot demonstrations for scalable vla training. arXiv preprint arXiv:2509.22199, 2025.
- [21] Y. Zhao, H. Fan, D. Chen, S. Chen, L. Chen, X. Li, G. Ren, and H. Dong. Real2edit2real: Generating robotic demonstrations via a 3d control interface. In CVPR, 2026.
- [22] G. Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J. Zhu, K. Li, M. Xu, et al. Gigaworld-0: World models as data engine to empower embodied ai. arXiv preprint arXiv:2511.19861, 2025.
- [23] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel. Learning universal policies via text-guided video generation. NeurIPS, 2023.
- [24] J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia, Y.-C. Lin, et al. Dreamgen: Unlocking generalization in robot learning through video world models. arXiv preprint arXiv:2505.12705, 2025.
- [25] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.
- [26] J. Xiao, Y. Yang, X. Chang, R. Chen, F. Xiong, M. Xu, W.-S. Zheng, and Q. Zhang. Worldenv: Leveraging world model as a virtual environment for vla post-training. arXiv preprint arXiv:2509.24948, 2025.
- [27] H. Kress-Gazit, K. Hashimoto, N. Kuppuswamy, P. Shah, P. Horgan, G. Richardson, S. Feng, and B. Burchfiel. Robot learning as an empirical science: Best practices for policy evaluation. arXiv preprint arXiv:2409.09491, 2024.
- [28] P. Atreya, K. Pertsch, T. Lee, M. J. Kim, A. Jain, A. Kuramshin, C. Eppner, C. Neary, E. Hu, F. Ramos, et al. Roboarena: Distributed real-world evaluation of generalist robot policies. arXiv preprint arXiv:2506.18123, 2025.
- [29] Z. Zhou, P. Atreya, Y. L. Tan, K. Pertsch, and S. Levine. Autoeval: Autonomous evaluation of generalist robot manipulation policies in the real world. arXiv preprint arXiv:2503.24278, 2025.
- [30] Y. R. Wang, C. Ung, G. Tannert, J. Duan, J. Li, A. Le, R. Oswal, M. Grotz, W. Pumacay, Y. Deng, et al. Roboeval: Where robotic manipulation meets structured and scalable evaluation. arXiv preprint arXiv:2507.00435, 2025.
- [31] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In IROS, pages 5026–5033. IEEE, 2012.
- [32] Y. Zhu, J. Wong, A. Mandlekar, R. Martin-Martin, A. Joshi, S. Nasiriany, and Y. Zhu. robosuite: A modular simulation framework and benchmark for robot learning. arXiv preprint arXiv:2009.12293, 2020.
- [33] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, et al. Sapien: A simulated part-based interactive environment. In CVPR, pages 11097–11107, 2020.

- [34] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. In NeurIPS, 2023.
- [35] W. Pumacay, I. Singh, J. Duan, R. Krishna, J. Thomason, and D. Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191, 2024.
- [36] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.
- [37] M. Torne, A. Simeonov, Z. Li, A. Chan, T. Chen, A. Gupta, and P. Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. arXiv preprint arXiv:2403.03949, 2024.
- [38] A. Badithela, D. Snyder, L. Zha, J. Mikhail, M. O’Kelly, A. Dixit, and A. Majumdar. Reliable and scalable robot policy evaluation with imperfect simulators. arXiv preprint arXiv:2510.04354, 2025.
- [39] K. Zhang, S. Sha, H. Jiang, M. Loper, H. Song, G. Cai, Z. Xu, X. Hu, C. Zheng, and Y. Li. Realto-sim robot policy evaluation with gaussian splatting simulation of soft-body interactions. arXiv preprint arXiv:2511.04665, 2025.
- [40] L. Meyer, F. Erich, Y. Yoshiyasu, M. Stamminger, N. Ando, and Y. Domae. Pegasus: Physically enhanced gaussian splatting simulation system for 6dof object pose dataset generation. In IROS, pages 10710–10715. IEEE, 2024.
- [41] J. Quevedo, A. K. Sharma, Y. Sun, V. Suryavanshi, P. Liang, and S. Yang. Worldgym: World model as an environment for policy evaluation. arXiv preprint arXiv:2506.00613, 2025.
- [42] A. Majumdar, M. Sharma, D. Kalashnikov, S. Singh, P. Sermanet, and V. Sindhwani. Predictive red teaming: Breaking policies without breaking robots. arXiv preprint arXiv:2502.06575, 2025.
- [43] Team Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [44] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [45] V. Gabeur, S. Long, S. Peng, P. Voigtlaender, S. Sun, Y. Bao, K. Truong, Z. Wang, W. Zhou, J. T. Barron, et al. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026.
- [46] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. NeurIPS, 38:167283–167308, 2026.
- [47] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.
- [48] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and B. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.
- [49] M. Zhou, H. Zheng, Z. Wang, M. Yin, and H. Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In ICML, 2024.
- [50] M. Zhou, H. Zheng, Y. Gu, Z. Wang, and H. Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. In ICLR, 2025.

- [51] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NeurIPS, 2014.
- [52] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn,

N. Fusai, M. Y. Galliker, et al. π0.5: A vision-language-action model with open-world generalization. In CoRL, 2025.

- [53] GigaBrain Team, B. Wang, B. Li, C. Ni, G. Huang, G. Zhao, H. Li, J. Li, J. Lv, J. Liu, et al. Gigabrain-0.5 m*: A vla that learns from world model-based reinforcement learning. arXiv preprint arXiv:2602.12099, 2026.

Appendix

- A Task Suite

This appendix provides additional materials for the eight real-world manipulation tasks used in

- Sec. 4.1. Fig. 4 visualizes the task suite with representative initial and final observations. Across tasks, initial configurations vary in object placement, object pose, and robot approach state, while rollouts cover the full manipulation sequence from the initial observation to the task outcome. These details complement the main paper’s policy-evaluation results by clarifying what interaction outcomes the world model must predict during closed-loop rollout.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Task4: open the box lid and pour the chips from the box onto the plate

Task1: put banana into basket Task2: put green bowl into pink plate Task3: stack the box ears

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Task5: sweep the trash into the dustpan using a broom

Task7: tear off a piece of clear tape and stick it onto the metal box

Task6: fold the shirt

Task8: put panda into pink plate

- Figure 4: Visualization of the eight real-world manipulation tasks used for world-model-based policy evaluation. Each task is shown with an initial frame and a representative final frame from a successful rollout. Each task requires contact-sensitive prediction over closed-loop rollouts, so the world model must preserve both policy-dependent robot motion and the resulting scene changes.

- Task 1. The language instruction is “put banana into basket.” The scene contains a banana and a basket on the tabletop, with distractor objects possibly present in the workspace. The policy must reach the banana, grasp or push it into a controllable pose, and place it inside the basket. Success is achieved when the banana rests inside the basket at the end of the rollout. This task tests whether the world model can predict object transport into a container while preserving the basket geometry and the banana’s pose across multiple views.
- Task 2. The language instruction is “put green bowl into pink plate.” The initial scene contains a green bowl, a pink plate, and additional bowls or plates that create visual ambiguity. The policy must identify the green bowl, move it without disturbing the target plate excessively, and place it onto the pink plate. Success is achieved when the green bowl is stably placed on the pink plate. This task requires the generated rollout to preserve object identity, relative placement, and contact between two shallow objects.
- Task 3. The language instruction is “stack the box ears.” The task starts from an open cardboard box whose side ears are outside the closing slot. The policy must manipulate the ears so that they are

- folded and inserted into the slot rather than merely pushed against the box side. Success is achieved when the box ears are stacked into the closed configuration. This task is sensitive to small height and alignment differences: the world model must predict whether the ear clears the side wall and enters the slot, not only whether the visible box top appears closed.
- Task 4. The language instruction is “open the box lid and pour the chips from the box onto the plate.” The initial scene contains a small chip box and a plate. The policy must open the box, lift and tilt it, and pour the chips onto the plate. Success is achieved when the chips are transferred from the box to the plate. This task combines articulated object manipulation with granular object motion, requiring the world model to predict both the box pose and the downstream motion of the chips.
- Task 5. The language instruction is “sweep the trash into the dustpan using a broom.” The scene contains scattered trash, a dustpan, and a broom. The policy must control the broom to collect the trash and move it into the dustpan opening. Success is achieved when the trash ends inside the dustpan. This task stresses contact-rich pushing: small errors in broom pose, dustpan placement, or trash trajectory can change the final outcome.
- Task 6. The language instruction is “fold the shirt.” The task starts from a spread-out shirt on the tabletop. The policy must grasp and fold the cloth into a compact folded state. Success is achieved when the shirt is folded into the desired final configuration. This task requires the world model to forecast deformable object motion, including large nonrigid changes in cloth shape that are only partially constrained by the robot trajectory.
- Task 7. The language instruction is “tear off a piece of clear tape and stick it onto the metal box.” The scene contains a tape dispenser, clear tape, and a metal box. The policy must pull a piece of tape from the dispenser, tear it off, move it to the box, and press it onto the box surface. Success is achieved when a piece of clear tape is attached to the metal box. This task is visually challenging because the tape is small and transparent, so the world model must infer subtle contact and attachment states from limited visual evidence.
- Task 8. The language instruction is “put panda into pink plate.” The scene contains a panda toy, a pink plate, and distractor objects such as a banana or another plate. The policy must localize the panda toy, move it to the target plate, and release it inside the plate boundary. Success is achieved when the panda rests on the pink plate at the end of the rollout. This task evaluates whether the world model can preserve small-object identity and predict precise placement into a shallow target region.

### B Generated Rollout and Control Visualizations

We include qualitative rollout visualizations in Figs. 5 and 6 to show how iMaC sustains long futurevideo generation under closed-loop chunk-wise rollout and how its generated videos align with the corresponding video controls. Each example presents the reference observation, future RGB prediction, auxiliary depth prediction, URDF/FK-rendered motion images, and the two contactimage streams. The first visualizations focus on normal generation behavior across tasks. The generated RGB videos preserve the multi-view scene layout over long horizons, keep the robot motion consistent with the commanded trajectory, and produce object motion at the time and location suggested by the controls. The auxiliary depth predictions remain spatially coherent enough to support subsequent pointcloud construction, even when RGB details become less sharp over later chunks. Each figure uses the same row order so the reader can compare the generated video against the controls available to the model: motion images reveal the action-implied future robot body, while the scene-to-gripper and robot-to-scene contact images highlight where future gripper motion and robot geometry approach observed scene regions.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

#1 #2 #3 #4

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

#5 #6 #7 #8

- Figure 5: Long-horizon iMaC rollouts with paired video controls. For each task, generated RGB and depth are shown together with motion images and the two contact-image streams used during generation.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

#1 #2 #3 #4

#5 #6 #7 #8

- Figure 6: Additional iMaC rollouts with the same generated-video and video-control layout. The fixed layout shows how differences in generated scene motion are supported by the corresponding motion and contact images.

Task3

Task5

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Real World

World Model

Lacking the perspective to perceive the shape of the box's ears

Lacking the observation of relative object heights

- Figure 7: Focused failure-case visualization for missing task-relevant observations. Even with plausible generated videos and reasonable action-derived controls, the world model can mispredict scene evolution when all available views omit a physical relation that determines task success.

Failure analysis. Long closed-loop video generation can fail in several ways. Many failure cases reflect common limitations of current video models, including low visual fidelity, accumulated temporal error, and insufficiently accurate action following. In our setting, however, we also observe a failure mode that is more directly tied to world-model-based policy evaluation: the model cannot reliably infer task-relevant physical relations that are missing from all available observations. Fig. 7 visualizes this boundary through two representative cases. Task 3, “stack the box ears,” exposes this ambiguity in a box-ear insertion scenario. The initial view can make the task appear as if the two side ears only need to be folded inward into the slot, but in the real scene the ears are long and can be blocked by the box side; the robot must first lift the ear so that its lower edge is above and aligned with the slot before inserting it. The available views do not reliably reveal the height relation between the lower edge of the ear and the slot, so the model cannot determine how high the ear must be lifted to pass over the side wall. As a result, it may generate an apparently plausible motion in which the ear is lifted and then lowered, while incorrectly predicting that the ear has entered the slot. Task 5, “sweep the trash into the dustpan using a broom,” exhibits an analogous height ambiguity. Whether the paper trash can be swept into the dustpan depends on the height of the dustpan entrance relative to the tabletop: if the entrance is flush with the table, the trash can be pushed inside, whereas a raised dustpan blocks the trash. The three available views, including wrist-camera views, often do not provide a clean side observation of this height relation because the key local region can be occluded by the gripper, broom, or dustpan itself. The world model may therefore generate plausible broom motion and trash displacement while mispredicting whether the trash can physically pass into the dustpan. These examples suggest that, beyond improving video quality and control following, reliable learned real-world simulation also depends on camera coverage that captures the physical state variables that decide task success.

