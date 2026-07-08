# arXiv:2605.27491v1[cs.RO]26May2026

[Figure 1]

## GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation

Boxiang Qiu1,2,∗ Liliang Chen1,∗,† Yue Liao3 Nan Wang1 Lintao Wang1 Jiayi Luo1,2 Wenzhi Zhao1,4 Shengcong Chen1 Di Chen1 Ye Li4 Chen Gao2,3 Shuicheng Yan3,⋄ Si Liu2,⋄ Maoqing Yao1,⋄ Guanghui Ren1,†,⋄

1 AgiBot 2 BUAA 3 LV-NUS Lab 4 TJU https://ge-sim-v2.github.io/

##### Abstract

We introduce GE-Sim 2.0 (Genie Envisioner World Simulator 2.0), a closed-loop video world simulator for robotic manipulation. Building on the action-conditioned video generation framework of Genie Envisioner, GE-Sim 2.0 is re-trained on thousands of hours of real-world robot data spanning teleoperation, contact-rich interaction, and on-robot policy deployment, substantially improving action-following fidelity and trajectory coverage. On top of this foundation, three new modules close the loop from video simulation to policy learning: a state expert that decodes proprioceptive state from video latents to support next-chunk prediction by downstream policy models; a world judge that scores generated rollouts against task instructions, yielding machine-verifiable success signals and rewards in place of manual inspection; and an acceleration framework that delivers a 25-frame rollout in 2.3 seconds on a single H100, with up to 4× frame skipping at inference for long-horizon evaluation. GE-Sim 2.0 tops the public WorldArena leaderboard at only 2B parameters, outperforming both dedicated robotic world models and closed-source general video generators, and policies trained against its rollouts and rewards translate into measurable real-world gains, establishing GE-Sim 2.0 as a practical platform for scalable evaluation and closed-loop learning of manipulation policies.

##### 1 Introduction

Robot learning is entering a scaling era. Larger models, internet-scale demonstrations, and increasingly capable vision-language-action (VLA) policies (Black et al., 2024; Brohan et al., 2022, 2024; Driess et al., 2023; Generalist AI Team, 2025, 2026; Kim et al., 2024; Physical Intelligence et al., 2025; Team et al., 2024; Zhu et al., 2023) are pushing manipulation beyond rigid-body pick-and-place toward long-horizon, contact-rich, and deformable-object tasks. Yet as policies scale, evaluation has become the bottleneck: real-robot benchmarking is slow and hard to reproduce, while existing robotic benchmarks and simulators (Gu et al., 2023; James et al., 2020; Lin et al., 2020; Liu et al., 2023; Makoviychuk et al., 2021; Mees et al., 2022; Todorov et al., 2012; Xiang et al., 2020) still struggle with contact dynamics, deformable objects, fine-grained visual appearance, and even the robot’s own actuation, where effects such as harmonic-drive compliance are routinely abstracted away. The gap between what we can train and what we can reliably evaluate keeps widening.

Recent progress in generative video modeling (Bar-Tal et al., 2024; Blattmann et al., 2023; OpenAI et al., 2024; Singer et al., 2022; Villegas et al., 2022) offers a different path. Trained on web-scale video, modern generators can synthesize photorealistic footage across a wide diversity of scenes, objects, and interactions that handcrafted simulators cannot easily reproduce. This motivates a new paradigm, a neural world simulator for manipulation: given an initial observation and an action trajectory from a policy, human, or teleoperation, the model rolls out a video of the robot executing that behavior in a learned visual world. By replacing hand-built physics and rendering with a data-driven generative process, such a simulator promises to cover the long tail of real-world appearances and interactions that classical engines miss, opening a path toward both scalable evaluation and closed-loop policy learning, including reinforcement learning, of modern manipulation policies.

∗ Equal Contribution. † Project Leader. ⋄ Corresponding Author.

###### GenieEnvisionerWorldSimulator(GE-Sim2.0)

###### Visual Input

[Figure 2]

|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>#1<br><br>[Figure 34]<br><br>#2<br><br>[Figure 35]<br><br>#3<br>|
|---|

Powered by millions of teleoperation, real-world deployment, and rich interaction robotic manipulation episodes.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

… History Frames

[Figure 41]

[Figure 42]

[Figure 43]

Sampler

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Left View Right View

Head View

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Long-horizon Context

[Figure 62]

[Figure 63]

Action Input

|[Figure 64]<br><br>[Figure 65]<br><br>…<br><br>[Figure 66]<br><br>trajectory<br><br>+<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>calibration<br><br>embed &project|
|---|

###### Condition Multi-view Video Simulation

[Figure 71]

[Figure 72]

[Figure 73]

Latent Features

State Estimation from Video Latent

Deriving Reward from Video

[Figure 74]

[Figure 75]

[Figure 76]

Proprioceptive State Expert

Generated Head-view Video Chunk

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

“Pour	the	water	from	the	held	kettle	into the	cup	on	the	table	with	right	arm.”

Dual-arm joint angles and Gripper States

Instruction VLM-based World Judge

Reward

|#1<br><br>Joint 1<br><br>Joint 2<br><br>Joint 3<br>Joint 4<br><br><br>Joint 5<br><br>Joint 6<br>Joint 7<br><br><br>gripper<br><br>[Figure 82]|
|---|

|#2<br><br>Joint 1<br><br>Joint 2<br><br>Joint 3<br>Joint 4<br><br><br>Joint 5<br><br>Joint 6<br>Joint 7<br><br><br>gripper<br><br>[Figure 83]|
|---|

|#3<br><br>Joint 1<br><br>Joint 2<br><br>Joint 3<br>Joint 4<br><br><br>Joint 5<br><br>Joint 6<br>Joint 7<br><br><br>gripper<br><br>[Figure 84]|
|---|

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

0۔034 0۔090 0۔233 0۔653

[Figure 89]

Left Arm Right Arm

- Figure 1: Overview of GE-Sim 2.0. GE-Sim 2.0 is a closed-loop video world simulator for robotic manipulation, trained on millions of real-world episodes spanning teleoperation, on-robot policy deployment, and rich object interaction. Given long-horizon multi-view history frames and an action trajectory embedded from end-effector calibration, the model generates action-conditioned multi-view rollouts of the robot executing the specified behavior (top). Two complementary modules close the loop on top of visual simulation (bottom): a proprioceptive state expert decodes dual-arm joint angles and gripper states from video latents, supplying downstream policy models with the state information needed for next-chunk prediction; and a VLM-based world judge scores generated rollouts against the task instruction.

A growing line of work (Gao et al., 2026; Guo et al., 2025; HO et al., 2025; Jiang et al., 2025; Liao et al., 2025; NVIDIA et al., 2025; Wang et al., 2026; Zhu et al., 2024) has begun to explore video-based world simulators for manipulation, including our own GE-Sim (Liao et al., 2025). These efforts share a common recipe: re-purpose a pretrained text-image-to-video (TI2V) generator into an action-image-to-video simulator by replacing the text condition with an action condition, and most technical effort to date has focused on how to inject this action signal so that the generated video faithfully follows a given trajectory. While such systems already produce visually plausible actionconditioned rollouts, their fidelity on deformable objects and on out-of-distribution or failure trajectories remains limited. More fundamentally, even a perfectly photorealistic action-following video does not, by itself, constitute a simulator that a policy can close the loop on. Three gaps stand out: (i) existing simulators predict only future visual states, leaving unmodeled the proprioceptive state that modern policy models require, with commanded actions used as a noisy proxy that drifts from the arm’s actual motion; (ii) they render rollouts but do not score them, withholding the verifiable signals required for scalable evaluation and reward-driven learning; and (iii) their rendering throughput is far below what chunk-wise, parallel rollout across many tasks and seeds demands. Together, these three gaps mark the path from today’s action-conditioned video models toward what we call a closed-loop world simulator for manipulation.

To close these gaps, we introduce GE-Sim 2.0 (Genie Envisioner World Simulator 2.0), illustrated in Figure 1. GE-Sim

- 2.0 retains the action-conditioned video generation backbone of GE-Sim (Liao et al., 2025), and is re-trained on thousands of hours of real-world robot data, combining large-scale teleoperation episodes, contact-rich arm-object interaction sequences, and rollouts collected during policy deployment on physical robots. Conditioned on long-horizon multi-view history frames and an action trajectory, the model rolls out action-following multi-view videos of the robot executing the specified behavior. This data scale and diversity substantially improve action-following accuracy, the

visual fidelity of object deformation and contact, and the coverage of diverse trajectories across successful executions, failure cases, and varied tasks and scenes. On top of this strengthened foundation, we introduce three modules that directly address the three gaps identified above. First, a state expert decodes proprioceptive state, namely dual-arm joint angles and gripper states, from video latents, providing downstream policy models with faithful state alongside visual observations for next-chunk prediction. Second, a VLM-based world judge scores generated rollouts against task instructions, turning simulator output into machine-verifiable success signals and rewards for both policy evaluation and reward-driven learning. Third, an acceleration framework improves rollout throughput without sacrificing fidelity, making chunk-wise, large-scale rollout tractable. Together, these components advance GE-Sim 2.0 from a visual simulator into a closed-loop, machine-verifiable platform for scalable manipulation policy training and evaluation.

We validate GE-Sim 2.0 along the design axes outlined above. As a foundation video simulator, it tops the public WorldArena leaderboard (Shang et al., 2026) at only 2B parameters, outperforming dedicated robotic world models such as Ctrl-World (Guo et al., 2025), DreamDojo (Gao et al., 2026), GigaWorld (GigaWorld Team et al., 2025), and ABot (Chen et al., 2026), alongside closed-source general video generators including Sora (OpenAI et al., 2024) and Veo (Google DeepMind, 2025). At a finer granularity, per-task replay metrics (PSNR, SSIM, LPIPS, FID, FVD) across six representative manipulation tasks show that this advantage holds case by case, and closed-loop evaluation shows that policy outcomes inside the simulator agree with the real robot at both the aggregate and per-episode level. The state expert recovers proprioceptive state with high fidelity and improves downstream next-chunk prediction, while the world judge yields reward signals closely aligned with human judgment. Driven by the acceleration framework, GE-Sim

- 2.0 generates a 25-frame rollout in 2.3 seconds on a single H100, and a random-stride training scheme enables up to 4× frame skipping at inference, extending the simulated horizon without measurable loss in evaluation consistency. Crucially, policies trained against GE-Sim 2.0’s rollouts and rewards translate into measurable real-world gains, lifting it from a passive video generator into an active driver of policy learning. 2 Preliminaries

GE-Sim 2.0 builds on the Genie Envisioner platform (Liao et al., 2025), in particular its world foundation model GE-Base and its action-conditioned simulator GE-Sim. We briefly review both here to fix notation; readers familiar with Genie Envisioner may skip this section.

- 2.1 GE-Base: Multi-View Video World Foundation Model

GE-Base formulates robotic world modeling as a multi-view text-and-image-to-video generation problem. Given a language instruction q and an initial multi-view observation, the model autoregressively predicts future video chunks that capture how the scene evolves under the instruction.

Autoregressive chunk-wise generation. Let V = {h,l,r} denote the set of onboard cameras (head, left wrist, right wrist), and let xit denote the frame from view i ∈ V at time t. At autoregressive step t, the world model W predicts the next chunk of N multi-view frames xt1:N conditioned on the initial observation x0, a long-term sparse memory m0:t−1, and the encoded instruction T (q):

###### xt1:N = W(x0, m0:t−1, T (q)), (1)

where m0:t−1 is constructed by sparsely sampling keyframes from previously generated chunks {xk1:N}tk−=01 , and T (·) is a frozen T5 text encoder. This sparse memory mechanism extends the temporal context of the model far beyond the

current chunk while keeping the input length tractable.

Multi-view encoding. Each per-view input is processed independently by a shared video encoder E, producing initial and memory tokens v0i = E(xi0) and vmi = E(mi0:t−1). Each token is enriched with a 3D rotary positional embedding and a learnable view embedding eiview:

###### v˜i = RoPE(t,h,w) + vi + eiview. (2)

Together with a view-specific noise map zi, the per-view input sequence is ui = v ˜0i ∥v˜mi ∥zi . Tokens from all views are concatenated and processed by a video diffusion transformer (DiT). To enforce cross-view consistency, a subset of DiT blocks performs cross-view attention over the merged multi-view sequence, while the remaining blocks treat views independently for efficiency.

Backbone and training. We instantiate W with the Cosmos-Predict2-2B-Video2World DiT (NVIDIA et al., 2025), which provides strong visual priors for high-fidelity simulation. Training follows a latent flow-matching objective: given

the VAE latent l of the target chunk and a noisy latent ˜l = (1 − στ)l + στ ϵ with ϵ ∼ N(0,I), W predicts the denoising velocity vθ, supervised on future frames via a conditioning mask M:

Lvideo = w(τ) vθ − (ϵ − l) ⊙ (1 − M) 22. (3)

###### 2.2 Genie Envisioner World Simulator (GE-Sim)

GE-Sim repurposes GE-Base from a text-and-image-to-video generator into an action-conditioned video simulator: instead of being driven by a language instruction, generation is driven by a low-level robot action trajectory, so that the synthesized video faithfully reflects how the robot would execute that trajectory in the scene.

Action representation. For a dual-arm system, each control step is encoded as a 14-dimensional vector formed by concatenating the 7-D end-effector states of both arms:

, xi,yi,zi,ri,pi,yi,oi

ai = xi,yi,zi,ri,pi,yi,oi

left arm

right arm

∈ R14, (4)

where (x,y,z) is the end-effector position, (r,p,y) its roll-pitch-yaw orientation, and o the gripper openness. Over a K-step horizon, the full trajectory is denoted A = [a1,...,aK] ∈ RK×14.

Spatial action conditioning. Bridging the low-level control signal A and the high-dimensional latent space of W requires a spatially-aligned conditioning signal that specifies both where the end effector should appear in the image plane and from which viewpoint each frame is observed. GE-Sim therefore couples a Pose2Image rendering with an explicit camera raymap, both of which live in the pixel grid of the target view.

Pose image. At each step i, the position (xi,yi,zi) is projected into pixel coordinates via the calibrated camera intrinsics and extrinsics, the orientation axes are projected as directional unit vectors, and the gripper openness is rendered on a unit circle whose shading reflects oi. Distinct color encodings differentiate the two arms. This yields a pose image Pi ∈ R3×H×W that is spatially aligned with the scene.

Camera raymap. To make the multi-view camera geometry explicit, we additionally construct a per-pixel raymap Ri ∈ R6×H×W from the intrinsics K and the camera-to-world extrinsics Ti. For each pixel (u,v), we form the ray origin oi ∈ R3 (the camera center in world coordinates) and the unit-norm view direction di ∈ R3 obtained by back-projecting (u,v) through K and rotating by the camera-to-world rotation in Ti. Stacking oi and di along the channel dimension gives a 6-channel raymap that exposes the camera pose at every pixel, so the simulator does not have to infer viewpoint geometry from appearance alone.

Latent fusion. Both Pi and Ri are bilinearly downsampled to the latent spatial resolution and concatenated with the noisy video latent z˜i along the channel dimension, so the simulator consumes a single fused input

vi = z ˜i ∥down(Pi)∥down(Ri) . (5)

This channel-wise fusion preserves spatial alignment between the action condition, the camera geometry, and the latent video tokens at every layer of the DiT backbone.

From TI2V to action-conditioned simulation. With this conditioning mechanism in place, the role of the language condition T (q) in Eq. (1) is replaced by the action trajectory A, while the multi-view, autoregressive, sparse-memory backbone of GE-Base is kept intact. The simulator thus predicts

xt1:N = S x0, m0:t−1, At , (6)

where At is the action sub-trajectory aligned with chunk t, and S denotes the action-conditioned simulator obtained by replacing T (q) with the hierarchical action conditioning described above. This formulation forms the foundation on which GE-Sim 2.0 is built.

##### 3 Genie Envisioner World Simulator 2.0

In this section, we introduce GE-Sim 2.0, a comprehensive upgrade of GE-Sim. GE-Sim 2.0 advances a “view-only” video world simulator into a closed-loop world simulator that interacts precisely with a policy model and provides feedback on the interaction.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

History State

Future Action

Noised State

𝐵𝑖𝑣𝑖𝑠

1

Video DiT Block

t

###### ScalarMixLayer

𝐵𝑠𝑡𝑎𝑡𝑒𝑖

State DiT Block

VAE

Corresponding frames

Norm&Out

𝐵𝑖𝑣𝑖𝑠+1

[Figure 94]

1 t

[Figure 95]

[Figure 96]

[Figure 97]

Video DiT Block

…

𝐵𝑠𝑡𝑎𝑡𝑒𝑖+1

State DiT Block

[Figure 98]

[Figure 99]

[Figure 100]

Action conditions convert

Norm & Out

Norm & Out

[Figure 101]

Video Decoder State Decoder

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

###### ··· ··· ···

Action Input

Joint Angles Gripper Joint Angles Gripper

T=0 ··· T=N

Proprioceptive State

Visual State

- Figure 2: Vision expert and proprioceptive state expert overview. The vision expert processes historical frames and action conditions to generate future visual states, which are then consumed by the proprioceptive state expert to predict the joint angles and gripper openness of both arms.

###### 3.1 Overview

As shown in Figure 2, at each autoregressive step, GE-Sim 2.0 takes as input an initial observation x0, a sparse memory m, and an action sequence At. In the canonical deployment scenario At is produced by a policy model under evaluation, but the simulator is agnostic to the source and equally accepts actions from teleoperation logs, motion planners, or hand-authored trajectories. Following the Pose2Image formulation in Section 2.2, every action is rendered into a visually aligned EE pose map using the camera intrinsics and extrinsics, so that the position, orientation, and openness of each gripper are explicitly encoded as pixel-level conditions. This single action representation is shared across all components of GE-Sim 2.0.

GE-Sim 2.0 is organized as two parallel experts that share the same conditioning. The vision expert is an actionconditioned diffusion transformer that follows the chunk-wise autoregressive and sparse-memory framework of Section 2.1 and generates a future video chunk xˆt, predicting what the robot would observe under the given action sequence and serving as the visual backbone of the simulator. The proprioceptive state expert runs in parallel with the vision expert and predicts the corresponding joint-space state sequence ˆst by consuming the intermediate features of the vision expert as visual context, recovering the proprioceptive observation that a real robot would return. When the policy and GE-Sim 2.0 are coupled into a closed-loop system, the two expert outputs (xˆt,ˆst) are fed back to the policy as the input for its next chunk of action prediction, extending the simulator from pure visual rendering to a source of the full observation required for policy interaction, and forming an interaction loop consistent with the real robot.

On top of this loop, we attach a world judge, a vision-language reward model that scores the rollout generated by the world simulator frame by frame and outputs a machine-verifiable success signal rˆt. This signal serves both as an automated referee for policy evaluation and as sparse feedback for downstream reward-driven learning such as filtered BC and RL, giving GE-Sim 2.0 a built-in critic capability.

GE-Sim 2.0 is built in stages. We first train the vision expert on thousands of hours of real-world robot data (Section 3.2). We then freeze the vision expert and train the proprioceptive state expert so that it decodes joint-space state from the visual context of the vision expert (Section 3.3). The world judge is trained independently as an external module

- (Section 3.4). After the main training, we apply a distillation-based post-training acceleration to the world simulator
- (Section 3.5), compressing multi-step diffusion into few-step inference. The following four sections describe these components in turn.

###### 3.2 Action-Conditioned Video Generation

The vision expert of GE-Sim 2.0 is an action-conditioned multi-view diffusion model and serves as the visual backbone of the simulator. It inherits the chunk-wise autoregressive generation, sparse memory, and multi-view diffusion transformer of GE-Base described in Section 2.1. On this basis, to turn the vision expert into a high-fidelity simulator suitable for closed-loop policy interaction, we describe its conditioning interface, action representation, and training.

Conditioning interface. The vision expert operates in the latent space of a video VAE, and all of its conditions enter the network through channel-wise concatenation. At each autoregressive step, the network input is

zcond = znoisy; Rray; Mpose; mcond , (7)

where znoisy is the 16-channel noisy video latent, Rray is a 6-channel per-pixel ray map, Mpose is a 3-channel EE pose map, and mcond is a 1-channel binary mask distinguishing memory frames from frames to be predicted. The ray map and the EE pose map jointly form the action representation introduced next. All visual inputs are normalized from [0,1] to [−1,1] before encoding, and both conditioning maps follow the same [−1,1] convention, so that video and conditioning channels are numerically aligned and diffusion training is not destabilized by scale mismatch.

Action representation. GE-Sim 2.0 conveys the dual-arm action to the vision expert through two visually aligned channels, the ray map Rray and the EE pose map Mpose. Together they encode what the camera sees of the action: the ray map captures how the viewpoint moves with the robot, while the EE pose map captures how the end effectors move within that view. Decoupling the two factors is essential in our setup, because the head and wrist cameras are themselves mounted on the moving robot and the wrist views in particular shift substantially with the arm.

Ray map. For each pixel, we build a ray in the world frame from the per-frame camera intrinsics and extrinsics, represented jointly by its origin and unit direction (six channels in total). As the cameras move with the robot, the ray map changes accordingly and gives the vision expert an explicit camera-geometry prior, allowing it to separate appearance changes caused by viewpoint motion from those caused by object motion in the scene. This is especially important for the wrist cameras, where the end effector is nearly static relative to the camera and the ray map carries most of the kinematic signal of the arm.

EE pose map. Following the Pose2Image formulation of GE-Sim in Section 2.2, we render the future end-effector trajectory of both arms into the image space of each camera view, producing a 3-channel EE pose map spatially aligned with the scene. For each timestep and each view, the rendering proceeds in three steps. (i) Pose projection. We express the gripper pose of each arm as a 4 × 4 transformation in the world frame, transform it into the camera frame through a fixed wrist-to-EE correction, and project the EE origin together with one keypoint along each of the three coordinate axes onto the image plane, obtaining the pixel coordinates of the EE position and its orientation. (ii) Depth-aware rendering. On the canvas, we draw a filled circle at the EE origin whose radius decreases monotonically with the distance from the camera to the EE,

∥xEE − xcam∥ − dmin dmax − dmin

, 0, 1 · rmax, (8)

r = clamp 1 −

so that the closer the EE is to the camera, the larger the circle. The EE orientation is drawn as colored segments connecting the origin to the three projected axis keypoints, with the left and right arms using distinct color schemes. (iii) Gripper-openness encoding. The fill color of the circle encodes the gripper openness through a continuous colormap: as the gripper goes from closed to open, the color goes from dark to light, with left and right arms using different color families. This jointly encodes the gripper state, which is binary in nature yet carries a continuous degree, in a single, continuous, and stable channel. The same renderer is used at training and inference time to keep the two input distributions aligned. The EE pose map serves as the unified pose-level action condition shared by the vision expert, the proprioceptive state expert, and the world judge.

Training. We train the vision expert on thousands of hours of real-world robot data. The training data spans teleoperation recordings, on-robot policy deployment rollouts, and a large number of object-interaction trajectories, and covers both successful and failed trajectories, improving the model’s ability to simulate diverse actions and complex interactions. Training follows the flow-matching objective Lvideo of GE-Base in Section 2.1, with the loss computed only on frames to be predicted while memory frames are masked out by mcond. During closed-loop inference, the vision expert rolls out chunk by chunk and the memory frames of each chunk are sparsely sampled from previously generated content. Since these memory frames come from the vision expert’s own generations, they inevitably contain generation artifacts and accumulated errors from continuous rollout, and are therefore not drawn from the same distribution as the clean

memory frames used during training. To mitigate this distribution shift, we introduce a randomized error-simulation mechanism for memory frames during training, approximating the error patterns of self-generated memory frames through pre-encoding perturbation, local degradation, and multi-view synchronized appearance variation.

###### 3.3 Proprioceptive State Expert

Closed-loop interaction requires not only future visual observations but also the robot’s proprioceptive state, which is one of the key inputs that modern policy models rely on for next-chunk action prediction. The vision expert generates only future video and does not directly expose the proprioceptive state of the arms in the rendered frames. To fill this gap, we introduce the proprioceptive state expert, a lightweight transformer branch that runs in parallel with the vision expert and predicts the proprioceptive state sequence over the corresponding time interval from the visual context produced by the vision expert.

Proprioceptive state representation. The proprioceptive state expert represents and regresses the proprioceptive state in joint space. The proprioceptive state at each frame is a 16-dimensional vector formed by concatenating the joint angles and gripper openness of both arms,

st = θLt , gtL, θRt , gtR ∈ R16, (9)

where θLt ,θRt ∈ R7 are the seven joint angles of the left and right arms, and gtL,gtR are the corresponding gripper openness values. The gripper openness is linearly normalized to [0,1], consistent with the gripper coordinate convention

used by the EE pose map in Section 3.2. Besides the current interval to be predicted, which spans Tfut future frames, the input sequence of the proprioceptive state expert also contains the proprioceptive state of nprev history frames together with the aligned history and future actions, giving 2nprev + 2Tfut tokens in total, of which only the future proprioceptive state tokens are noised. This joint-space proprioceptive state complements the visual prediction, and together they form the full observation required for downstream closed-loop policy interaction.

Visual context from the vision expert. The visual information that the proprioceptive state expert needs for predicting the proprioceptive state comes from the vision expert, but the proprioceptive state expert is not aligned with the vision expert layer by layer. The vision expert first runs a complete forward pass through all of its L transformer blocks, and we aggregate the output hvideol of each layer into a single fused visual feature through a set of learnable scalar weights αl (initialized to one), followed by a single LayerNorm,

Hfuse = LayerNorm

L

αl hvideol , αl ∈ R. (10)

l=1

All L blocks of the proprioceptive state expert share the same Hfuse as the key and value of their cross-attention. Under the multi-view setting, Hfuse is rearranged into Hfuse ∈ RB×(V L

tok)×d, concatenating the tokens of all V camera views along the sequence dimension, so that the proprioceptive state expert attends to the visual features of all views at once. Architecture. The proprioceptive state expert is a stack of L lightweight transformer blocks whose hidden dimension is much smaller than that of the vision expert. Each block consists of three parts: a self-attention under rotary positional embedding along the proprioceptive state time axis, a cross-attention to the vision-expert visual feature Hfuse, and a feed-forward layer; the diffusion timestep is injected through AdaLN-single, whose scale-shift parameters use the same diffusion timestep as the vision expert. The proprioceptive state sequence is first projected into the hidden space of the proprioceptive state expert by a linear layer and then projected back to the ds-dimensional state space by another linear layer at the end, where ds denotes the dimension of the proprioceptive state representation.

Training and loss. The proprioceptive state expert is trained on top of a frozen vision expert: we use the trained vision expert as the provider of visual context and update only the proprioceptive state expert. Training adopts the same flow-matching objective as the vision expert, applied to the proprioceptive state sequence. Given a clean proprioceptive state sequence s0 and noise ϵ, we construct a noised sequence sτ, and the proprioceptive state expert vϕ predicts the denoising velocity, with the loss computed only on the future proprioceptive state,

0,ϵ,τ vϕ(sτ,τ,Hfuse) − (ϵ − s0) 22. (11)

Lproprio state = Es

History-state augmentation. The core input of the proprioceptive state expert is the history proprioceptive state, the history actions, and the future actions, and only the future proprioceptive state is noised. In real data, however,

the history proprioceptive state is almost always a noise-free exact reading, whereas during closed-loop inference part of the history proprioceptive state comes from the proprioceptive state expert’s own prediction at the previous chunk and carries error. Without intervention, the proprioceptive state expert would overly rely on a “perfect history” and be insufficiently robust under the closed loop. We therefore apply two targeted perturbations to the history segment during training: temporal index perturbation, which simulates short-range misalignment between the policy and the vision expert, and history-trajectory resampling, which mimics slight temporal stretching or compression under asynchronous closed-loop execution. Detailed perturbation probabilities and parameter settings are provided in the appendix.

shist ← Upsample Downsample(shist, nprev − 1), nprev , (12) which is equivalent to injecting a low-pass distortion that preserves the long-term trend while removing single-frame high-frequency detail, simulating the case where the history is fed back by the network rather than taken from ground truth.

###### 3.4 World Judge

A central goal of GE-Sim 2.0 is to support large-scale policy evaluation. The world simulator generates rollouts but does not judge how good a rollout is. To make the simulator truly serve evaluation and reward-driven learning, we need a component that turns generated results into machine-verifiable signals. To this end, we introduce the world judge, a vision-language reward model that scores the rollout generated by GE-Sim 2.0 frame by frame during closed-loop inference and outputs a task-completion success signal. The design of the world judge builds on Robometer (Liang et al., 2026), a general-purpose vision-language reward model; we adapt its formulation to our closed-loop setting and train a single success objective on real-robot data.

Backbone and per-frame representation. The world judge uses a vision-language model as its backbone. We freeze its vision encoder and train only the language model and the downstream prediction head. Each frame of the rollout is fed in individually as a separate image, without averaging over the time axis, so as to preserve per-frame discriminative granularity. After each frame image, we append a dedicated per-frame token and take the hidden state of this token as the representation fi of that frame. A rollout of length T is thus encoded into a set of per-frame representations {fi}Ti=1. The textual condition of the world judge is not the full task instruction but the sub-task caption matched to the current chunk, so that the judgment is conditioned on the specific sub-task that the chunk is expected to accomplish.

Success head and supervision. While Robometer is trained with a dual progress-and-preference objective, our world judge deliberately focuses on a single per-frame success prediction. This design is motivated by two considerations. First, a sparse success signal is sufficient for closed-loop evaluation and downstream reinforcement learning, and has been widely adopted as the primary metric or reward signal in robotic manipulation benchmarks and offline RL settings (Andrychowicz et al., 2017; Lu et al., 2025). Second, progress supervision is substantially less reliable in our setting. In real robot data collection, trajectories often contain conscious or unconscious error recovery behaviors, such as detours, retries, and temporary regressions before eventual success. Under such non-monotonic executions, a scalar progress label becomes noisy and may no longer correspond to the true task completion process. We therefore discard progress prediction and formulate world judging as a class-balanced per-frame binary success classification problem.

The world judge attaches a success head, a lightweight MLP, on top of each per-frame representation, producing a success logit sˆi = head(fi) for that frame. The supervision comes from the manual success annotation of each trajectory: every trajectory is associated with a success frame isucc, where that frame and all frames after it are labeled as success and the frames before it as failure; for a trajectory that fails entirely or has no success annotation, all frames are labeled 0,

yi = ⊮ i ≥ isucc . (13) Since the success and failure frames within a trajectory are highly imbalanced, we adopt a class-balanced binary cross-entropy as the training objective. Within each batch, we count the numbers of positive and negative frames N+ and N−, and weight the minority class by the inverse of its frequency so that the two classes contribute comparably to the loss,

mi wi BCE σ(ˆsi), yi i mi wi

Ljudge = i

, (14) where wi is the class-balancing weight described above and mi is a valid-frame mask.

Closed-loop integration with the world model. Once trained, the world judge is integrated into the closed loop as a reward signal. During a closed-loop rollout, the chunk-wise actions output by the policy model are rendered through the

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

t

###### 1

| |Vision Encoder| |
|---|---|---|
| | | |

…

### World Judge

Instruction

Text Encoder

“Fold the towel”

T=0T=1T=2T=3 ······ T=N-1

T=N

ongoing success

- Figure 3: World Judge. The World Judge scores generated rollout frames against task instructions, providing ongoing and success signals for closed-loop policy evaluation. Vision Encoder processes frames, Text Encoder encodes instructions, and the outputs are combined for per-frame success assessment.

unified action map and drive the world model to generate chunk by chunk; the video and proprioceptive state generated by the world model are returned to the policy model for its subsequent prediction, while the world judge outputs a success signal for the generated frames, yielding a success curve over the rollout. This curve serves both as an automated decision for policy evaluation and as dense feedback for reward-driven learning, turning policy assessment from reliance on manual inspection into a machine-verifiable process.

- 3.5 Acceleration

Scalable policy evaluation requires rolling out trajectories chunk by chunk and in parallel across many tasks, and the multi-step diffusion inference of the world model becomes the throughput bottleneck in this setting. We accelerate it from two directions: reducing the number of denoising steps required per generation through step distillation, and letting a single generation cover a longer time span through random-stride training.

Step distillation. We distill the multi-step diffusion world model into a few-step student using the distributionmatching framework of DMD2 (Yin et al., 2024). The distillation involves a frozen teacher, which is the world model from the main training stage, a few-step student to be distilled, and a trainable fake-score critic that estimates the score of the student’s output distribution. At each step, the student generates a sample, which is re-noised and denoised by the teacher and the critic to yield two estimates xteacher0 and xfake0 ; their difference forms the distribution-matching gradient that drives the student,

∇DMD =

xfake0 − xteacher0 xstudent0 − xteacher0

, (15)

where the denominator normalizes the gradient magnitude per sample. The student and the critic are optimized alternately so that the critic keeps tracking the current student’s output distribution. To fit the action-conditioned, memory-based setting of GE-Sim 2.0, the teacher, student, and critic all receive the same action map condition, and the memory frames are kept as clean ground-truth latents throughout; the student’s number of denoising steps is randomized during training so that it supports a flexible one- to few-step configuration at inference. We provide the remaining details in the appendix.

Temporal acceleration via random-stride training. The other direction of acceleration comes from the time span. During world-model training, we sample the frames of each chunk with a random temporal stride. This exposes the model to trajectories of different temporal densities at training time, so that at inference it can perform frame skipping, covering up to four times the time span with the same number of frames. For the evaluation of long-horizon tasks, this significantly reduces the number of autoregressive chunks needed to cover a given task segment, with no noticeable loss in spatial-temporal consistency.

Combining the two directions, the world model of GE-Sim 2.0 generates a 100-frame rollout in about 2.3 seconds on a single H100 with only four inference steps, reaching a throughput suitable for large-scale parallel evaluation.

- 4 Experiments

Our experiments are structured to test the central claim of GE-Sim 2.0: that an action-conditioned video model can serve as a closed-loop simulator for manipulation policies, not merely a high-quality video generator. Concretely, we

t0 t1 Outletfixedis t2 t3 locatedPlug t4 turnedLampoff

[Figure 115]

GT

Ours

Ctrl-World

DreamDojo

- Figure 4: Qualitative comparison on the Pull out plug task. We compare ground truth (GT), GE-Sim 2.0 (ours), Ctrl-World, and DreamDojo on an episode where the robot is required to unplug a desk lamp. GE-Sim 2.0 successfully follows the action, removes the plug, and correctly renders the lamp turning off. By contrast, Ctrl-World and DreamDojo both show action-following failures and neither reproduces the lamp-off state after unplugging.

- Table 1: Comparison of robotic world simulators. GE-Sim 2.0 supports long-horizon multi-view simulation with proprioceptive state prediction, pseudo real-time rollout, generalization, and reward estimation.

IRASim 1XWM GE-Sim 1.0 Ctrl-World DreamDojo Interactive WM ABot-PhysWorld WorldScape MotuBrain GE-Sim 2.0

Long-horizon ✓ ✗ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ Multi-view ✗ ✗ ✓ ✓ ✗ ✗ ✗ ✗ ✓ ✓ Proprioceptive State ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ Pseudo Real-Time ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✓ ✓ Reward ✗ ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ Open-source ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✗ ✓ ✓

ask five questions. Q1 (Visual fidelity). Does GE-Sim 2.0 generate manipulation videos of higher quality than prior robotic world models, both on the broad WorldArena benchmark and on fine-grained per-task replay metrics (§4.1) Q2 (Closed-loop faithfulness). When a policy model runs against GE-Sim 2.0 rather than the real robot, do the simulated outcomes agree with what happens in the physical world (§4.2)? Q3 (Verifiable rewards). Does the world judge produce success signals that are accurate enough to replace human inspection (§4.3). Q4 (Policy improvement). Can the world model and its reward signals be used to generate filtered training data that improves the downstream policy (§4.4). Q5 (Component contribution). How do the proprioceptive state expert and the specialized reward model affect closed-loop simulation and reward prediction, respectively (§4.5)?

Tasks. We evaluate on six long-horizon dual-arm manipulation tasks chosen to span the failure modes that have historically been hardest for video world models: liquid handling (Pour water), deformable-object manipulation (Fold

t0 t1 t2 pickedKettleup t3 t4 pouredWaterout

[Figure 116]

GT

Ours

Ctrl-World

- Figure 5: Qualitative comparison on the Pour water task. We compare ground truth (GT), GE-Sim 2.0 (ours), and Ctrl-World on an episode where the robot is required to pick up a kettle and pour water into a cup. Each frame contains a top view, with the left view at the lower left and the right view at the lower right. GE-Sim 2.0 successfully follows the action, lifts the kettle, and correctly renders the water-pouring process. By contrast, Ctrl-World shows action-following failures and does not faithfully reproduce the target water-pouring behavior.

towels), fine-force interactions (Pull out plug), open-flame ignition (Borrow flame), language-grounded selection (Command grasp & release), and surface contact under appearance variation (Clean mirror stains).Together, these tasks stress action-following accuracy, contact and deformation rendering, and the ability to faithfully reproduce both policy successes and failures.

Baselines. For video quality, we compare against Ctrl-World (Guo et al., 2025) and DreamDojo (Gao et al., 2026), two leading open robotic world models that release their model weights and inference code, enabling evaluation under matched action conditions. We additionally place GE-Sim 2.0 on the WorldArena leaderboard (Shang et al., 2026), which includes recent robotic world models such as GigaWorld (GigaWorld Team et al., 2025), ABot (Chen et al., 2026) in both text-driven and action-driven variants, and MotuBrain (Team et al., 2026), as well as closed-source general video generators including Sora (OpenAI et al., 2024) and Veo (Google DeepMind, 2025). For the world judge, we compare our specialist reward model against a strong general-purpose multimodal LLM, Qwen3.5-122B (Team, 2026), a 122B-parameter mixture-of-experts vision-language model, prompted to classify task success from video.

Evaluation protocol. For each task we curate 20 evaluation episodes of a π0.5 (Physical Intelligence et al., 2025) policy on the real robot. We use two complementary evaluation modes. Replay fixes the action trajectory recorded on the real robot and feeds it into the world model, then measures per-frame and distributional similarity between the simulated and real video; this isolates rendering quality from policy behavior. Closed-loop evaluation reruns the same policy through GE-Sim 2.0 in chunk-wise rollout: the policy generates actions from simulated observations at every

GE-Sim 2.0 (ours) Ctrl-World DreamDojo

###### Head View - PSNR ↑

Multi-View - PSNR ↑

26

22

24

22

20

PSNR

PSNR

20

18

18

16

16

14

14

1-10 11-20 21-30 31-40 41-50

1-10 11-20 21-30 31-40 41-50

Time range (seconds)

Time range (seconds)

- Figure 6: Temporal replay quality across different time ranges. We divide each replay video into five consecutive 10-second temporal segments and evaluate PSNR separately for each segment. These results indicate that our method maintain more faithful visual replay over long-horizon rollouts, with a smaller degradation as time progresses.

step rather than receiving pre-recorded ones, and we report the agreement between simulated and real outcomes; this measures whether the simulator faithfully tracks policy decisions rather than just replaying a fixed sequence. Reward evaluation reports binary classification accuracy (acc: whether the model correctly identifies task success) and the frame-index distance between the model’s predicted success frame and the human-annotated reference (dist), both on world-model rollouts (WM) and on ground-truth videos (GT).

###### 4.1 Video Simulation Quality

WorldArena leaderboard. Despite using only a 2B-parameter backbone, GE-Sim 2.0 attains the top overall score on the public WorldArena leaderboard, outperforming both dedicated robotic world models (Ctrl-World, DreamDojo, GigaWorld, ABot in its text and action variants) and substantially larger closed-source general video generators including Sora and Veo. This suggests that domain-specific training on diverse, action-grounded robot data is more important for manipulation simulation than raw model scale alone.

Per-task replay metrics. Table 2 reports replay quality averaged over the six tasks, separately for the head camera and for the joint multi-view configuration. GE-Sim 2.0 dominates across all five metrics in both views. On the head view, it improves PSNR by +3.96dB over Ctrl-World and +5.67dB over DreamDojo, halves FID (32.3 vs. 62.7), and reduces FVD by more than a factor of two (481 vs. 1084). On the considerably harder multi-view setting, where temporal coherence must hold simultaneously across head, left-wrist, and right-wrist cameras, the improvement is even more pronounced: −31.9FID and −2.5×FVD relative to Ctrl-World, while DreamDojo did not produce comparable multi-view outputs. The same ranking holds task by task: GE-Sim 2.0 is the best simulator on every individual task across all five metrics. The consistent improvement across benchmarks and all task subsets indicates that the quality gains are broad-based rather than artefact-specific.

Long-horizon temporal robustness. We further compare GE-Sim 2.0 against Ctrl-World and DreamDojo on visual stability over long-horizon rollouts. The aggregate replay metrics above average over the full 50-second rollout and can mask whether visual quality remains stable as the number of autoregressive chunks grows. To probe this, we divide each replay video into five consecutive 10-second segments and report PSNR per segment (Figure 6). The two views tell the same story. On the head view, GE-Sim 2.0 decays from 24.84 to 21.08dB, a swing of under 4dB across the full rollout, while Ctrl-World and DreamDojo drop sharply after the first segment and end at 16.16 and 15.44dB, nearly 5dB below ours. On the harder multi-view setting, GE-Sim 2.0 stays above 19.4dB throughout, whereas Ctrl-World falls to the 15dB range by the second segment and continues to decline. The shape of the curves matters as much as the absolute gap: GE-Sim 2.0 flattens after the first segment while the baselines keep degrading, so the advantage widens as rollouts get longer. This long-horizon stability is precisely what a simulator needs for closed-loop evaluation, and we attribute it to the large-scale real-robot retraining, the memory-frame augmentation, and the random-stride training of §3.5.

- Table 2: Replay visual quality averaged over six manipulation tasks. GE-Sim 2.0 outperforms both robotic world model baselines on all five metrics, in both single-view and multi-view settings. Best in bold; “—” denotes no comparable multi-view output.

Head View Multi-View

Method PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓ PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓ Ctrl-World (Guo et al., 2025) 19.09 0.786 0.296 62.70 1083.7 16.65 0.735 0.407 56.85 1527.5 DreamDojo (Gao et al., 2026) 17.38 0.728 0.353 79.82 1155.0 — — — — GE-Sim 2.0 (ours) 23.05 0.846 0.145 32.28 481.3 20.80 0.796 0.217 24.92 613.5

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

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 7: WorldArena leaderboard. GE-Sim 2.0 achieves the top overall score on the WorldArena benchmark, outperforming prior robotic world models and closed-source general video generators.

###### 4.2 Closed-Loop Policy Consistency

Replay fidelity does not by itself imply that a world model can serve as a policy simulator. In closed-loop use, small errors in visual state, proprioceptive state, or contact evolution can change the policy’s subsequent actions and lead to different task outcomes. We therefore run the same π0.5 policy inside each world model and compare the resulting simulated outcomes with physical-robot outcomes under matched task settings.

Task-level success-rate alignment. Figure 8 compares the real-robot success rate with the success rate measured inside each world model across the six tasks. Each marker denotes one task, and the diagonal indicates perfect agreement. GE-Sim 2.0 with state conditioning shows the closest task-level alignment: its fitted trend has a slope close to one with a small negative offset, suggesting that it better preserves the relative task difficulty observed on the real robot. In contrast, removing state conditioning weakens this correspondence, yielding a shallower trend and a positive offset. Ctrl-World shows the largest mismatch, with a similarly shallow slope but a negative offset, indicating a tendency to underestimate policy success across tasks.

This result shows that state-aware simulation improves aggregate closed-loop calibration. The remaining gaps are most visible on contact-sensitive tasks such as Pull out plug and Command grasp & release, where small errors in grasp state, contact timing, or object interaction can alter the downstream policy rollout. Thus, task-level success alignment provides a coarse but important measure of whether the simulator preserves real-world policy performance trends.

Episode-level outcome agreement. Task-level success rates can still hide case-level mismatches: two simulators may have similar marginal success rates while predicting success on different initial conditions. We therefore additionally evaluate closed-loop agreement as an episode-level binary prediction problem, where a simulated success or failure is compared with the corresponding physical-robot outcome.

Figure 10 reports the per-task confusion matrices for the three world models, and summarizes accuracy, precision, and recall. Accuracy measures the fraction of matched simulated and real outcomes, precision measures how often simulated successes are real successes, and recall measures how many real successes are recovered by the simulator. GE-Sim 2.0

###### Tasks

1.0

###### (elements may override)

w/ state (ours): y = 1.20x -0.08 w/o state (ours): y = 0.70x +0.10 Ctrl-World: y = 0.70x -0.14

###### Models

0.3

0.283

0.8

WMSuccessRate

###### AgreementError

###### y = x

w/ state (ours)

Pour water Fold towels Pull out plug Borrow flame Command grasp & release Clean mirror stains

0.6

0.2

Best

w/o state (ours)

0.4

0.125

0.1

Ctrl-World

0.2

0.058

0.0

y = x

0

0.0 0.2 0.4 0.6 0.8 1.0

w/state(ours)w/ostate(ours)Ctrl-World

Real Success Rate

- Figure 8: World-model success-rate alignment with real-robot outcomes. We compare the task success rates measured in the real world and those predicted by different world models under closed-loop policy rollouts. Each marker denotes one manipulation task, while different dashed lines correspond to Ctrl-World, our model without state conditioning, and our full model with state conditioning. The gray dashed line indicates perfect agreement (y = x) between world-model and real-world success rates. Compared with Ctrl-World, our models show stronger alignment with real-world policy performance, and state conditioning further improves the consistency between simulated and physical outcomes.

with state conditioning obtains the best average accuracy and recall: accuracy improves from 0.63 for Ctrl-World and 0.74 without state conditioning to 0.81, while recall improves from 0.25 and 0.67 to 0.82. The large recall gain indicates that the state-conditioned simulator is substantially better at preserving episodes where the real policy succeeds, rather than only matching the marginal success frequency.

The task-wise confusion matrices reveal both strengths and limitations. GE-Sim 2.0 recovers more true-positive outcomes on tasks such as Fold towels, Borrow flame, and Clean mirror stains, but false positives and false negatives remain on contact-rich tasks. These errors suggest that fine-grained contact state and long-horizon state accumulation are still challenging. Overall, the task-level and episode-level results together support GE-Sim 2.0 as a more faithful closed-loop evaluation environment than the compared baselines.

###### 4.3 World Judge: Reward Quality

Beyond rendering rollouts, GE-Sim 2.0 must also score them: closed-loop policy evaluation and reward-driven learning require machine-verifiable success signals. We evaluate our world judge against a strong general VLM baseline, Qwen3.5-122B, on two complementary inputs: simulated rollouts (WM), which is the operating mode used at policyevaluation time, and ground-truth videos (GT), which probes upper-bound reward accuracy without simulator-induced artifacts. Both models receive the same per-frame multi-image input for a fair comparison.

Table 3 summarizes the results. On WM rollouts, our world judge reaches 79% accuracy versus 60% for Qwen (+19pp), and reduces event distance from 57.8 to 28.2 frames. The advantage carries over to GT videos: 87% versus 58% (+29pp), with event distance reduced from 64.7 to 15.7 frames. The accuracy ranking holds on five of the six tasks; the only exception is Command grasp & release, a task dominated by discrete object-grasping events that a general VLM is comparatively well suited to recognize. Qwen further fails outright to estimate success frames on Clean mirror stains, a task whose success criterion involves subtle appearance changes rather than discrete object events. The remaining gaps on Pour water and Borrow flame point to the difficulty of detecting small-scale visual signals such as liquid level and flame transfer, which remains an open challenge for task-aware reward models. The gap between WM-mode and GT-mode accuracy (∼ 8pp) reflects the share of error attributable to simulator artifacts rather than inherent task ambiguity, and is small enough that reward signals from GE-Sim 2.0 remain reliable for policy assessment.

###### 4.4 WM-Based Filtered Behavior Cloning

Beyond evaluation, GE-Sim 2.0 can also be used as a data-generation engine for policy improvement. As a proof of concept, we run π0.5 inside GE-Sim 2.0, score the generated rollouts with the world judge, retain trajectories whose reward exceeds a per-task threshold, and mix the retained trajectories with the original behavior-cloning data. Due to

Table 3: World judge quality: success-classification accuracy (acc) and event distance in frames (dist) on world-model rollouts (WM) and ground-truth videos (GT). Our world judge reports consensus predictions; Qwen produces a single per-video prediction. “—” denotes that the model failed to identify a success frame. Our world judge outperforms the general VLM baseline by a large margin on most tasks.

Ours Qwen3.5-122B-A10B

WM GT WM GT Task acc↑ dist↓ acc↑ dist↓ acc↑ dist↓ acc↑ dist↓ Borrow flame .96 37.0 .95 24.4 .71 96.5 .70 115.3 Clean mirror stains .82 9.3 .90 12.4 .50 — .50 Command grasp & release .72 7.8 .80 4.0 .81 38.7 .85 43.4 Fold towels .65 13.0 .60 9.5 .63 38.0 .50 30.5 Pour water .90 61.4 1.00 26.2 .55 22.0 .35 13.0 Pull out plug .80 24.2 .95 7.5 .55 34.0 .60 17.0

###### Average .79 28.2 .87 15.7 .60 57.8 .58 64.7

0.7

0.65

| |0.40<br><br>0.55<br><br>+0.15<br><br>Base Policy<br><br>+WM-filtered BC|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

Base Policy

Base Policy

Base Policy

0.567

0.6

+WM-filtered BC

+WM-filtered BC

+WM-filtered BC

0.50

SuccessRate

+0.20

+0.150

0.5

0.45

0.417

+0.10

0.40

0.4

0.3

0.2

0.1

0.0

Average

Pour water

Fold towels

Pull out plug

- Figure 9: Policy improvement with WM-filtered behavior cloning. success rates of the π0.5 policy before and after augmenting the original training data with filtered synthetic trajectories generated by GE-Sim 2.0. For each task, we run the policy inside the world model, score the generated rollouts with our reward model, retain high-reward trajectories, and mix them with the original behavior cloning data for policy training. WM-filtered BC consistently improves policy success rates across all evaluated contact-rich manipulation tasks.

the cost of physical-robot evaluation, we evaluate this procedure on three representative tasks covering liquid handling, deformable-object manipulation, and fine-force interaction.

- Figure 9 reports real-robot success rates before and after adding the filtered world-model data. WM-filtered BC improves success from 0.40 to 0.55 on Pour water, from 0.40 to 0.50 on Fold towels, and from 0.45 to 0.65 on Pull out plug. The average success rate increases from 0.417 to 0.567, corresponding to an absolute gain of 0.150. These results indicate that GE-Sim 2.0 can produce synthetic rollouts that are useful not only for evaluation, but also for improving downstream policy learning after reward-based filtering.

4.5 Ablation Studies

We isolate the contribution of two key components of GE-Sim 2.0. The proprioceptive state expert is evaluated through closed-loop sim-to-real agreement, while the specialist world judge is evaluated through reward-prediction accuracy and event localization.

Proprioceptive state expert. We evaluate the impact of the proprioceptive state expert by comparing closed-loop world-model rollouts with and without it against real robot outcomes, using binary task success/failure. As shown in

- Figure 10, removing the state expert reduces the average episode-level accuracy from 0.81 to 0.74 and recall from 0.82 to 0.67, while precision remains largely unchanged. The confusion matrices highlight that including the state expert substantially improves the model’s ability to correctly recover true-positive outcomes, particularly for contact-rich tasks such as Fold towels and Pull out plug.

Command grasp &release

Clean mirror stains

Pour water Fold towels Pull out plug

Borrow flame

+

+

+

+ TN 5 (71%)

+

+ TN 5 (71%)

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

TN 12 (100%)

FP 0 (0%)

TN 12 (100%)

FP 0 (0%)

TN 11 (100%)

FP 0 (0%)

FP 2 (29%)

TN 16 (94%)

FP 1 (6%)

FP 2 (29%)

CtrlWorld

FN 6 (75%)

TP 2 (25%)

FN 8 (100%)

TP 0 (0%)

FN 9 (100%)

TP 0 (0%)

FN 6 (46%)

TP 7 (54%)

FN 2 (67%)

TP 1 (33%)

FN 8 (62%)

TP 5 (38%)

+

|Prec<br><br>1.00|
|---|

|Acc<br><br>0.85|
|---|

|Prec<br><br>0.50|
|---|

|Acc<br><br>0.70|
|---|

|Rec<br><br>0.25|
|---|

|Acc<br><br>0.60|
|---|

|Prec<br><br>|
|---|

|Rec<br><br>0.00|
|---|

|Acc<br><br>0.55|
|---|

|Prec<br><br>|
|---|

|Rec<br><br>0.00|
|---|

|Acc<br><br>0.60|
|---|

|Prec<br><br>0.78|
|---|

|Rec<br><br>0.54|
|---|

|Rec<br><br>0.33|
|---|

|Acc<br><br>0.50|
|---|

|Prec<br><br>0.71|
|---|

|Rec<br><br>0.38|
|---|

+

+

+ TN 7 (64%)

+ TN 5 (71%)

+

+ TN 6 (86%)

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Ours(w/ostate)

TN 11 (92%)

FP 1 (8%)

TN 11 (92%)

FP 1 (8%)

FP 4 (36%)

FP 2 (29%)

TN 12 (71%)

FP 5 (29%)

FP 1 (14%)

FN 4 (50%)

TP 4 (50%)

FN 3 (38%)

TP 5 (62%)

FN 3 (33%)

TP 6 (67%)

FN 1 (8%)

TP 12 (92%)

FN 1 (33%)

TP 2 (67%)

FN 5 (38%)

TP 8 (62%)

+

|Rec<br><br>0.67|
|---|

|Acc<br><br>0.85|
|---|

|Prec<br><br>0.86|
|---|

|Rec<br><br>0.92|
|---|

|Prec<br><br>0.89|
|---|

|Acc<br><br>0.75|
|---|

|Prec<br><br>0.80|
|---|

|Rec<br><br>0.50|
|---|

|Acc<br><br>0.80|
|---|

|Prec<br><br>0.83|
|---|

|Rec<br><br>0.62|
|---|

|Acc<br><br>0.65|
|---|

|Prec<br><br>0.60|
|---|

|Acc<br><br>0.70|
|---|

|Prec<br><br>0.29|
|---|

|Rec<br><br>0.67|
|---|

|Acc<br><br>0.70|
|---|

|Rec<br><br>0.62|
|---|

+

+

+ TN 8 (73%)

+ TN 4 (57%)

+

+ TN 5 (71%)

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

TN 11 (92%)

FP 1 (8%)

TN 11 (92%)

FP 1 (8%)

FP 3 (27%)

FP 3 (43%)

TN 14 (82%)

FP 3 (18%)

FP 2 (29%)

Ours(w/state)

FN 3 (38%)

TP 5 (62%)

FN 1 (12%)

TP 7 (88%)

FN 3 (33%)

TP 6 (67%)

FN 2 (15%)

TP 11 (85%)

FN 0 (0%)

TP 3 (100%)

FN 1 (8%)

TP 12 (92%)

+

|Acc<br><br>0.80|
|---|

|Rec<br><br>0.62|
|---|

|Acc<br><br>0.90|
|---|

|Prec<br><br>0.88|
|---|

|Rec<br><br>0.88|
|---|

|Acc<br><br>0.70|
|---|

|Prec<br><br>0.67|
|---|

|Rec<br><br>0.67|
|---|

|Acc<br><br>0.85|
|---|

|Prec<br><br>0.50|
|---|

|Rec<br><br>1.00|
|---|

|Acc<br><br>0.85|
|---|

|Rec<br><br>0.92|
|---|

|Prec<br><br>0.83|
|---|

|Acc<br><br>0.75|
|---|

|Prec<br><br>0.79|
|---|

|Rec<br><br>0.85|
|---|

|Prec<br><br>0.86|
|---|

- Figure 10: Confusion matrices evaluating the agreement between closed-loop world-model simulations and physicalrobot outcomes for binary task success/failure. Compared with Ctrl-World, our model achieves higher true-positive rates, suggesting better preservation of real-world task success patterns, especially in contact-rich scenarios.

The gains are most pronounced on tasks that require accurate state tracking across multiple action chunks, such as Fold towels, Borrow flame, and Clean mirror stains. These tasks involve fine-grained arm motion, contact maintenance, or long-horizon accumulation, where the robot’s actual joint state can deviate from the commanded action due to compliance and contact forces. Feeding back the decoded proprioceptive state helps prevent this drift from compounding during autoregressive rollout.

A small reversal appears on Command grasp & release, where the model without state conditioning obtains slightly higher accuracy, suggesting that this task is less dependent on precise joint-state recovery and more dominated by coarse visual progress. Overall, the ablation confirms that the proprioceptive state expert improves closed-loop consistency by providing policy-relevant state feedback beyond action-conditioned video prediction alone.

Specialist vs. general-purpose reward model. Replacing our world judge with the Qwen3.5-122B VLM (Table 3) reduces WM-mode accuracy by 19pp and increases event distance by 30 frames, with similar gaps on GT videos. Qwen further fails to identify a success frame on Clean mirror stains, indicating that prompting a general VLM is not a robust substitute for a task-aware specialist judge. Together, these results justify the small specialist reward head as a core component of the closed-loop simulator.

##### 5 Conclusion

We presented GE-Sim 2.0, a step from a “view-only” video world simulator toward an omni world simulator for robotic manipulation. On top of the action-conditioned video generation framework of Genie Envisioner, retrained on thousands of hours of real-robot data, three coordinated components close the loop: a proprioceptive state expert that decodes joint-space state from the video latents, a world judge that turns rollouts into machine-verifiable success signals, and a DMD-based acceleration framework that compresses multi-step diffusion into few-step inference. Together, they integrate visual simulation, proprioceptive state prediction, and reward assessment into a single efficient framework. On six long-horizon dual-arm tasks, GE-Sim 2.0 leads on both the WorldArena leaderboard and multi-view replay metrics, its closed-loop success rates agree with the real robot within 1pp, the world judge clearly outperforms a strong general VLM, and filtered behavior cloning on its rollouts and rewards yields a 15pp average gain on the real-robot policy. These results advance the video world simulator from a forward predictor into a closed-loop platform with built-in evaluation, offering a usable starting point for scalable evaluation and closed-loop learning of manipulation policies.

##### 6 Looking Forward

GE-Sim 2.0 explores one viable instantiation of an omni world simulator, defining the components such a system should include and demonstrating competitive simulation on robotic manipulation. Moving toward a world simulator for general embodied AI leaves several directions open.

Data and model scaling. Our current focus is on methodology, and the foundation model is trained on dual-arm teleoperation and on-robot replay from a fixed embodiment. A general embodied world simulator will require extending to large-scale cross-embodiment, ego-view, and UMI-style data alongside broader manipulation video, with a systematic study of data- and model-scaling behavior. Pushing the foundation backbone is a core bottleneck on the path to general embodied simulation.

From an external judge to a unified model. The current world judge is an external module that suffices for tasksuccess assessment. A longer-term form is to unify state decoding and success judgment into a single world model whose forward pass jointly produces future frames, proprioceptive state, and task-completion signals, removing the separate reward network and letting reward signals inherit the priors learned by the world model at scale.

From offline filtered BC to online closed-loop learning. Our experiments show that the rollouts and rewards of GE-Sim 2.0 can already improve a real-robot policy through filtered behavior cloning. A natural continuation is online policy learning and reinforcement learning inside the simulator, advancing the system from an evaluator and data filter into an environment for policy training. The closed-loop and ablation studies here are conducted with a single VLA policy family; validating GE-Sim 2.0 across a broader range of policies is a useful next step.

##### 7 Acknowledgements

We thank Aogeliqiang Niyazi, Wei Xu, Wenhao Wang, Ruofei Niu, Jingyuan Wang, Xiongfeng Cai, Haoyu Cao, Cheng Jing, Pengfei Zhou, Donglin Yang, Liyan Zhang, Xiaoyun Hu, Feng Han, Chuankang Li, Sukai Wang, Yuxiang Yan, Jia Zeng, Yuehan Niu, Xuan Hu, and Jing Wu for their valuable support and contributions to this project.

##### References

M. Andrychowicz, D. Crow, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, and W. Zaremba. Hindsight experience replay. In Neural Information Processing Systems, 2017. URL https://api.semanticscholar.org/CorpusID: 3532908.

O. Bar-Tal, H. Chefer, O. Tov, C. Herrmann, R. Paiss, S. Zada, A. Ephrat, J. Hur, Y. Li, T. Michaeli, O. Wang, D. Sun, T. Dekel, and

- I. Mosseri. Lumiere: A space-time diffusion model for video generation. SIGGRAPH Asia 2024 Conference Papers, 2024. URL https://api.semanticscholar.org/CorpusID:267095113.

K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22563–22575, 2023. URL https://api.semanticscholar.org/CorpusID:258187553.

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. Rt-2:

Vision-language-action models transfer web knowledge to robotic control, 2023. URL https://arxiv. org/abs/2307.15818, 1:2, 2024. Y. Chen et al. Abot-physworld: Interactive world foundation model for robotic manipulation with physics alignment. arXiv preprint

arXiv:2603.23376, 2026. AMAP CV Lab. D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin, Q. Ma, S. Nah, L. Magne, J. Xiang, Y. Xie, R. Zheng, D. Niu, Y. L. Tan, K. R. Zentner, G. Kurian, S. Indupuru, P. Jannaty, J. Gu, J. Zhang, J. Malik, P. Abbeel, M.-Y. Liu, Y. Zhu, J. Jang, and L. Fan. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

- Generalist AI Team. Gen-0: Embodied foundation models that scale with physical interaction. https://generalistai.com/blo g/nov-04-2025-GEN-0, November 2025. Blog post.
- Generalist AI Team. Gen-1: Scaling embodied foundation models to mastery. https://generalistai.com/blog/apr-02-202 6-GEN-1, April 2026. Blog post.

GigaWorld Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J. Zhu, K. Li, M. Xu, et al. Gigaworld-0: World models as data engine to empower embodied ai. arXiv preprint arXiv:2511.19861, 2025.

Google DeepMind. Veo 3: A text-to-video generation system. Technical report, Google DeepMind, 2025. URL https://storage.

###### googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf.

- J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, X. Yuan, P. Xie, Z. Huang, R. Chen, and H. Su. Maniskill2: A unified benchmark for generalizable manipulation skills. ArXiv, abs/2302.04659, 2023. URL https: //api.semanticscholar.org/CorpusID:256697500.

Y. Guo, L. X. Shi, J. Chen, and C. Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint

arXiv:2510.10125, 2025. D. HO, J. MONAS, J. REN, and C. YU. 1x world model: evaluating bits, not atoms, 2025. S. James, Z. Ma, D. R. Arrojo, and A. J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics

and Automation Letters, 5(2):3019–3026, 2020. Y. Jiang, S. Chen, S. Huang, L. Chen, P. Zhou, Y. Liao, X. He, C. Liu, H. Li, M. Yao, and G. Ren. Enerverse-ac: Envisioning embodied environments with action condition. arXiv preprint arXiv:2505.09723, 2025.

M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

A. Liang, Y. Korkmaz, J. Zhang, M. Hwang, A. Anwar, S. Kaushik, A. Shah, A. S. Huang, L. Zettlemoyer, D. Fox, Y. Xiang, A. Li,

- A. Bobu, A. Gupta, S. Tu, E. Bıyık, and J. Zhang. Robometer: Scaling general-purpose robotic reward models via trajectory comparisons. In Robotics: Science and Systems (RSS), 2026.

Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo, L. Chen, S. Yan, M. Yao, and G. Ren. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025.

- X. Lin, Y. Wang, J. Olkin, and D. Held. Softgym: Benchmarking deep reinforcement learning for deformable object manipulation. ArXiv, abs/2011.07215, 2020. URL https://api.semanticscholar.org/CorpusID:226965494.

- B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. ArXiv, abs/2505.18719, 2025. URL https://api.semanticscholar.org/CorpusID: 278904856.

V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State. Isaac gym: High performance gpu-based physics simulation for robot learning. ArXiv, abs/2108.10470, 2021. URL https://api.semanticscholar.org/CorpusID:237277983.

O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

NVIDIA, N. Agarwal, A. Ali, M. Bala, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

OpenAI, T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh. Video generation models as world simulators. https://openai.com/research/video-generation-model s-as-world-simulators, 2024.

Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker,

- D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling,

H. Wang, L. Yu, and U. Zhilinsky. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Y. Shang, Z. Li, Y. Ma, W. Su, X. Jin, Z. Wang, L. Jin, X. Zhang, Y. Tang, H. Su, C. Gao, W. Wu, X. Liu, D. Shah, Z. Zhang, Z. Chen, J. Zhu, Y. Tian, T.-S. Chua, W. Zhu, and Y. Li. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.

U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni, D. Parikh, S. Gupta, and Y. Taigman. Make-a-video: Text-to-video generation without text-video data. ArXiv, abs/2209.14792, 2022. URL https: //api.semanticscholar.org/CorpusID:252595919.

M. Team, C. Xiang, F. Bao, H. Liu, H. Tan, H. Bi, J. Li, J. Liu, J.-L. Pang, K. Jing, L. Liu, M. Cai, R. Cui, R. Zhao, R. Wang,

- S. Huang, Y. Feng, Y. Rong, Z. Wang, and J.-Y. Zhu. Motubrain: An advanced world action model for robot control. 2026. URL https://api.semanticscholar.org/CorpusID:287915787.

O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

- Q. Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

- R. Villegas, M. Babaeizadeh, P.-J. Kindermans, H. Moraldo, H. Zhang, M. T. Saffar, S. Castro, J. Kunze, and D. Erhan. Phenaki: Variable length video generation from open domain textual description. ArXiv, abs/2210.02399, 2022. URL https://api.sema nticscholar.org/CorpusID:252715594.

Y. Wang, R. Syed, F. Wu, M. Zhang, A. Onol, J. Barreiros, H. Nayyeri, T. Dear, H. Zhang, and Y. Li. Interactive world simulator for robot policy training and evaluation. arXiv preprint arXiv:2603.08546, 2026.

F. Xiang, Y. Qin, K. Mo, Y. Xia, B. H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su. Sapien: A simulated part-based interactive environment. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11094–11104, 2020. URL https://api.semanticscholar.org/CorpusID:213175506.

- T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong. Irasim: Learning interactive real-robot action simulators. arXiv preprint arXiv:2406.14540, 2024.

Y. Zhu et al. Vima: General robot manipulation with multimodal prompts. In International Conference on Learning Representations (ICLR), 2023.

##### 8 Appendix

###### 8.1 Memory-Frame Augmentation Details

This section provides the augmentation settings referenced in Section 3.2. At inference, memory frames come from the model’s own rollouts and drift away from the clean memory frames seen during training. We apply three types of perturbations to memory-frame latents at training time, all gated by an outer activation probability of 0.8. Progressive noise mixing mixes Gaussian noise into the memory latents along the time axis, with a per-frame activation probability of 0.5 and a per-frame perturbation scale of σmem = 0.2; a separate first-frame perturbation is activated with probability 0.2 at scale σfirst = 0.5. Local Gaussian blur is applied with probability 0.5, using a kernel size drawn from [1,5] and σ drawn from [0.1,1.3], restricted to a connected-component mask covering roughly 20% of the frame. Multi-viewsynchronized color jitter is applied with probability 0.3, sharing the same jitter across all camera views to keep lighting and color statistics consistent across head and wrist views.

###### 8.2 Proprioceptive State Expert Details

History-state augmentation. The two perturbations referenced in Section 3.3 are applied independently, each with probability 0.5. The delta-index shift shifts the entire history delta index by an integer ∆ ∼ Uniform{−3,...,−1,1,...,3} and clips it to the valid range, simulating a one- to three-frame temporal misalignment between the policy and the simulator. The temporal resampling downsamples the history segment from nprev to nprev − 1 frames and linearly interpolates it back to nprev, acting as a low-pass distortion that preserves the long-term trend while removing single-frame detail. Both perturbations are disabled at validation time.

###### 8.3 DMD2 Acceleration Details

This section provides the implementation details referenced in Section 3.5. The teacher is the vision expert from the main training stage and is frozen throughout. The student and the fake-score critic are both initialized from the teacher. The student targets 4 inference steps, with the actual number of denoising steps randomized between 1 and 4 during training so that one- to few-step inference is supported.

We update the student once every 5 steps and the critic on the remaining four, with a single-step warmup before the first student update. The critic loss is weighted by 1/σ2. The student is trained with the fixed sigma schedule [1.0,0.9375,0.8333,0.625], concentrated near full noise. During generator updates we shift the sigma distribution toward medium-to-high noise (shift = 3) to avoid wasted near-zero-noise batches; during critic updates we shift it further toward σ ≈ 1 (shift = 5). The student is optimized with AdamW (β2 = 0.999), weight decay 0.01, and 100 warmup steps; the critic uses a smaller learning rate of 2 × 10−6 with β1 = 0, β2 = 0.999, and weight decay 0.01. No auxiliary flow-matching loss is added to the student.

###### 8.4 Additional Qualitative Results

We provide additional qualitative comparisons across all six manipulation tasks. Figures 11, 12, 13, 14, and 15 show multi-view rollouts (head together with the left and right wrist cameras) comparing GE-Sim 2.0 against Ctrl-World on Command grasp & release, Clean mirror stains, Borrow flame, Fold towels, and Pull out plug. Figures 16, 17, 18, and 19 show head-view comparisons on Pour water, Command grasp & release, Borrow flame, and Clean mirror, the four tasks where DreamDojo also produces a comparable single-view output.

t0 t1 t2 grabsGripperair t3 t4 Graspfailedtask

[Figure 180]

GT

Ours

Ctrl-World

Figure 11: Command grasp & release (Multi-View).

t0 t1 Grasptowelthe t2 reflectionMirror t3 t4 cleanedStain

[Figure 181]

GT

Ours

Ctrl-World

Figure 12: Clean mirror stains (Multi-View).

t0 t1 Graspcandlethe t2 t3 t4 stayCandlesapart

[Figure 182]

GT

Ours

Ctrl-World

Figure 13: Borrow flame (Multi-View).

t0 t1 edgesGrab t2 t3 t4 foldedTowel

[Figure 183]

GT

Ours

Ctrl-World

Figure 14: Fold towels (Multi-View).

t0 t1 Outletfixedis t2 t3 locatedPlug t4 turnedLampoff

[Figure 184]

GT

Ours

Ctrl-World

Figure 15: Pull out plug (Multi-View).

t0 t1 t2 pickedKettleup t3 t4 wrongPouringway

[Figure 185]

GT

Ours

Ctrl-World

DreamDojo

Figure 16: Pour water (Head-View).

t0 t1 t2 chipGraspbag t3 t4 positionPlace in

[Figure 186]

GT

Ours

Ctrl-World

DreamDojo

Figure 17: Command grasp & release (Head-View).

t0 t1 t2 t3 t4 stayCandlesapart

[Figure 187]

GT

Ours

Ctrl-World

DreamDojo

Figure 18: Borrow flame (Head-View).

t0 t1 t2 t3 t4 cleanedStain

[Figure 188]

GT

Ours

Ctrl-World

DreamDojo

Figure 19: Clean mirror (Head-View).

t0 t3 pickedKettleup t4 pouredWaterout

#### t0 t1 t2

[Figure 189]

GT

Ours

t0 t1 t2 pickedCleanserup t3 t4 positionPlace in

[Figure 190]

GT

Ours

Figure 20: More results - GE-Sim 2.0.

t0 t1 edgesGrab t2 t3 t4 Towelfolded

[Figure 191]

GT

Ours

t0 t1 Grasptowelthe t2 Swtichleft handto t3 cleaningRobot is t4 cleanedStain

[Figure 192]

GT

Ours

Figure 21: More results - GE-Sim 2.0.

