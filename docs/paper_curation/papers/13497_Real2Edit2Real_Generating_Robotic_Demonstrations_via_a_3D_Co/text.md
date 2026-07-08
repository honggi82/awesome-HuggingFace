## arXiv:2512.19402v2[cs.RO]21Mar2026

### Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface

Yujie Zhao1,2∗ Hongwei Fan1,2∗ Di Chen3 Shengcong Chen3 Liliang Chen3 Xiaoqi Li1,2 Guanghui Ren3 Hao Dong1,2† 1CFCS, School of Computer Science, Peking University 2PKU-AgiBot Lab 3AgiBot https://real2edit2real.github.io/

Demo Generation Few-Shot Spatial Generalization

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### Real Demo vs. Gen Demo

| | |[Figure 7]<br><br>Robust to Object Spatial Configurations 10–50×Data Efficiency<br><br>[Figure 8]<br><br>[Figure 9]|
|---|---|---|

80

AverageSuccessRate(%)

# Real2Edit2Real

70

60

No Simulation RGB Only

50

VLA Compatible

40

###### Novel Placements Novel Trajectories

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

0 10 20 30 40 50

1 2 5

The Number of Real Demonstrations

[Figure 16]

[Figure 17]

[Figure 18]

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

Mug to Basket Pour Water Lift Box Scan Barcode

Figure 1. The overview of Real2Edit2Real. Real2Edit2Real generates diverse robotic demonstrations, featuring 10-50× improvement on data efficiency compared with real-world collection across four tasks.

#### Abstract

Recent progress in robot learning has been driven by large-scale datasets and powerful visuomotor policy architectures, yet policy robustness remains limited by the substantial cost of collecting diverse demonstrations, particularly for spatial generalization in manipulation tasks. To reduce repetitive data collection, we present Real2Edit2Real, a framework that generates new demonstrations by bridging 3D editability with 2D visual data through a 3D control interface. Our approach first reconstructs scene geometry from multi-view RGB observations with a metricscale 3D reconstruction model. Based on the reconstructed geometry, we perform depth-reliable 3D editing on point clouds to generate new manipulation trajectories while geometrically correcting the robot poses to recover physically

consistent depth, which serves as a reliable condition for synthesizing new demonstrations. Finally, we propose a multi-conditional video generation model guided by depth as the primary control signal, together with action, edge, and ray maps, to synthesize spatially augmented multi-view manipulation videos. Experiments on four real-world manipulation tasks demonstrate that policies trained on data generated from only 1–5 source demonstrations can match or outperform those trained on 50 real-world demonstrations, improving data efficiency by up to 10-50×. Moreover, experimental results on height and texture editing demonstrate the framework’s flexibility and extensibility, indicating its potential to serve as a unified data generation framework. Project website is https://real2edit2real. github.io/.

*: Equal contributions. †: Corresponding author.

#### 1. Introduction

Recent advances in robotics have demonstrated remarkable progress in visuomotor policy learning, driven by largescale datasets and powerful model architectures such as the Diffusion Policy [5, 6] and Vision-Language-Action (VLA) models [1, 2, 4, 16, 21]. However, the performance of these methods heavily relies on the availability of diverse and high-quality demonstrations. Spatial generalization, in particular, remains a bottleneck for policy robustness [39, 55]. In manipulation tasks where objects are randomly arranged in space, achieving a high success rate typically requires a large amount of data to cover diverse spatial configurations, and increases substantial data collection costs.

To alleviate the burden of repetitive data collection, an efficient strategy is to synthesize new demonstrations from limited existing data. MimicGen-style works [9, 11, 18, 33] segmented demonstration trajectories according to object interactions, and then transformed and interpolated these object-centric segments to generate new trajectories that adapt to novel object arrangements. Real2Render2Real [57] synthesized demonstrations from a human manipulation video through pose tracking and trajectory interpolation. Since these approaches render robotic videos within a graphics engine, they inevitably face the visual and physics gaps, which remain a significant challenge in robotics. Moreover, they require assets for the manipulated objects, which prevents them from directly augmenting an existing demonstration. DemoGen [55] augmented real-world point-cloud demonstrations through 3D editing and enhanced spatial generalization of 3D Diffusion Policy [58], but it cannot be applied to RGB images and 2D policies, which remain the dominant paradigm in current robot learning and deployment. Consequently, to our best knowledge, there is no existing method to rapidly scale up real-world

- 2D multi-view manipulation videos with novel trajectories, while preserving both visual realism and interaction fidelity.

To mitigate this research gap, we introduce Real2Edit2Real, a demonstration generation framework that bridges the gap between 3D editability and 2D visual data via a 3D control interface, achieving spatially augmented multi-view robotic demonstrations. As shown in Figure 1, Real2Edit2Real does not rely on simulation engines or digital assets, and directly generates multi-view manipulation data from raw RGB demonstrations, featuring novel object placements and corresponding new trajectories, which can be used for VLA training. Our key insight is that depth inherently encodes robot motion and object interactions, making it a natural interface between 3D modalities and 2D observations. Specifically, our proposed framework works with three modules: (1) Metric-scale geometry reconstruction of robot manipulation scenarios, where we propose a hybrid training paradigm that leverages real and simulated data to co-train a feed-forward

3D reconstruction model. (2) Depth-reliable spatial editing which combines point-cloud editing with trajectory planning to generate feasible manipulation trajectories while geometrically correcting the robot’s poses, thereby producing kinematically consistent depth maps that serve as reliable control signals for subsequent video generation. (3) 3D-Controlled video generation for multi-view consistent demonstrations, where we construct a video generation model conditioned on depth, together with edges, actions, and ray maps. With Real2Edit2Real, we can edit 2D videos through a unified 3D control interface, which facilitates data augmentation for robotic manipulation, thereby enhancing the robustness of downstream policies.

To evaluate the quality and efficiency of generated demonstrations, we conduct experiments on four real-world robotic manipulation tasks, covering single-arm to dualarm manipulation. Experimental results indicate that policies trained on data generated from as few as 1–5 source demonstrations can achieve comparable or higher success rates than those trained on 50 real-world demonstrations, improving data efficiency by up to 10–50×. Additionally, Real2Edit2Real supports extended editing such as object height and background texture, demonstrating the framework’s flexibility and extensibility, and suggesting its potential as a unified robotic data generation framework.

#### 2. Related Work

##### 2.1. Demonstration Generation

Due to the difficulty and cost of collecting robotic demonstration data, generating numerous demonstrations from zero or one demonstration has been proposed to rapidly extend robotic data. One line of works [4, 7, 10, 37, 45] use simulation engines to automatically collect demonstrations with pre-defined tasks and motion planning. However, the lack of real-world interaction leads to the Sim2Real gap. Another line of works [9, 11, 18, 25, 33, 55–57] focus on generating from one or a few collected demonstrations. MimicGen family [9, 11, 18, 25, 33] generates from one demonstration with carefully designed task segments. DemoGen [55] combines MimicGen-style generation with point cloud editing. However, it is incompatible with the widely used setting of multi-view RGB cameras for visuomotor policy training. Real2Render2Real [57] and RoboSplat [56] use 3D Gaussian Splatting (3DGS) [20] with trajectory generation to reduce the gaps in visual fidelity and interaction reality. These works expose two disadvantages. First, the rendering-based techniques that they used still bring the visual domain gap, limiting the Gen2Real performance. Second, the dense image captures that 3DGS requires restrict the scalability of generation. By bridging point-cloud-based demonstration editing and 3D-controlled video generation, Real2Edit2Real jointly achieves scalabil-

ity, visual quality, and real-world interaction of generated demonstrations in one framework.

##### 2.2. Geometry Reconstruction

Reconstructing the detailed environmental geometry is the key to generating controllable and multi-view consistent demonstrations. Early methods such as NeuS [48] and

- 2DGS [12] use radiance fields and gaussian splatting as
- 3D representations, which require dense image captures and minute-level post-optimization, both restricting their application scope and efficiency. Recently, feed-forward geometry reconstruction [19, 24, 32, 47, 50, 51, 60] unlocks sparse-view reconstruction in seconds, bringing the potential of recovering geometry from robot camera views [27, 36, 52]. However, directly using the feed-forward model suffers from the domain gap, including wrong camera pose and misaligned metrics between pretrained data and realworld manipulation scenarios. We propose an effective training recipe of VGGT [47] on robotic data to both utilize multi-view consistent depth maps and camera poses from simulated data, and precise geometry metrics from real-world data. The aligned geometry prediction ensures a high-quality point cloud, which in turn enables the generation of reliable depth maps for demonstration synthesis.

##### 2.3. Video Generation in Robotics

Recent progress in video generation [3, 8, 23, 34] has improved conditional and temporal consistency, empowering several downstream applications in robotics. First, predicting video generation can serve as an enhancement [14, 31, 43, 61] to regular policy learning by jointly predicting future action and observation in one model. Second, actionconditioned video generation [17, 26, 40, 46] performs as the policy evaluator or realistic simulation environment, which receives future action and feeds generated observation back to the pretrained policy model. Third, learning multi-modal conditioned video generation [28, 30, 36, 44, 54] and editing these conditions during inference generalizes the original demonstration across diverse camera poses and textures. However, these works neglect spatial generalization, which is also fundamental for scaling robotic data. In contrast, Real2Edit2Real generates spatially augmented multi-view demonstrations and can be easily combined with first-frame editing to support beyond.

#### 3. Method

##### 3.1. Overview

We consider a humanoid robot scenario with multi-view videos from the head, left wrist, and right wrist cameras. Then we can formulate our problem as follows:

O = {Oi}Ni=1 = Real2Edit2Real(O,K) (1)

Here, O = (I,q,a) consists of multiview videos

I = {(Iht,Ilt,Irt)}Nt=1, joint angles q, and actions a from one source demonstration. K represents the robot kine-

matic model (URDF) and camera parameters, where K = (Krobot,Kcam). With our proposed Real2EditReal framework, the source demonstration O can be augmented to a large set of demonstrations O with novel object spatial configurations and trajectories.

As shown in Figure 2, our framework consists of three main components: metric-scale geometry reconstruction, depth-reliable spatial editing, and 3D-controlled video generation. In Sec. 3.2, we first introduce a hybrid training paradigm that leverages both real and simulated data to enhance the capability of the reconstruction model R in robotic environments, which predicts depth maps D and camera poses T from the source demo as Eq. 2, where D = {(Dht ,Dlt,Drt)}Nt=1 and T = {(Tht,Tlt,Trt)}Mt=1.

###### D,T = R(I) (2)

- In Sec. 3.3, we detail the pipeline E of depth-reliable spatial editing, shown in Eq. 3, where we synthesize novel trajectories based on motion planning and point-cloud editing, while correcting the robot’s poses to obtain physically consistent depth maps that serve as reliable control signals.

{(Di,Ti,qi,ai)}Ni=1 = E(D,T,K,q,a) (3)

- In Sec. 3.4, we propose a multi-view video generation model G, which produces the complete robotic manipulation video from the first frame controlled by depth, together with Canny edges C(·), actions, and ray maps, as Eq. 4.

{Ii}Ni=1 = {G(Di,Ti,ai,C(Di))}Ni=1 (4)

##### 3.2. Metric-scale Geometry Reconstruction

Motivated by the fact that 3D data affords greater flexibility for editing than 2D imagery, we initially conduct geometric reconstruction of the source demonstrations. To achieve scale-aware geometry reconstruction and improve its accuracy in humanoid robot scenes, which only include three cameras from the head and both wrists, we propose a hybrid training paradigm that combines real and simulated data to fine-tune VGGT [47], enhancing metric-scale depth map and camera pose prediction in humanoid scenarios.

Camera Pose. Camera poses obtained via hand–eye calibration in real-robot settings are susceptible to mechanical tolerances, calibration drift, and kinematic inaccuracies, leading to misalignment. Conversely, simulated data offers perfectly accurate and metrically consistent camera poses, since the virtual sensors are derived directly from the robot’s standardized URDF model. Therefore, we only use simulated data to supervise the camera loss: Lcamera =

v∈{h,l,r} L1(Tˆvsim,Tvsim), where we use L1 loss because of the precise ground truth.

###### Depth-reliable Editing 3D-Controlled Video Generation

- 1 2 3 Source Point Cloud

###### Metric-scale Reconstruction

Left Head

First Frame Temporal Depth

Source Demo

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Right

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

Edit

Head

[Figure 51]

Edit

[Figure 52]

[Figure 53]

[Figure 54]

Infer

[Figure 55]

[Figure 56]

Generated Point Cloud Multi-view Video Diffusion

[Figure 57]

Metric VGGT

[Figure 58]

[Figure 59]

Train

Left Hand

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

Gen Demo

Pose Correction

Right Hand

Real Data

Sim Data

- Figure 2. The framework of Real2Edit2Real. (1) Reconstruct metric-scale geometry from multi-view observations. (2)Synthesize novel trajectories with reliable depth rendering. (3) Generate demonstrations controlled by temporal depth signals.

Depth Map. Real-world datasets capture metric-scale geometry through depth sensors, but the acquired depth maps are often highly noisy due to sensor limitations, reflective or textureless surfaces. In contrast, simulated data provides noise-free and geometrically precise depth maps, but the object and scene scale may deviate from real-world distributions. To leverage the complementary strengths of both fields of data, we compute the depth loss using a mixed method: Ldepth = v∈{h,l,r}((Lconf(M(Dˆvreal),M(Dvreal)) + Lconf(Dˆvsim,Dvsim)). Lconf is the depth loss with uncertainty used in [47], and we use a threshold mask M to filter invalid noise in real depth maps.

In addition, the simulation data also supervises the point

map loss: Lpointmap = v∈{h,l,r} Lconf(Pˆvsim,Pvsim). The total training loss is shown as Eq.5, where we apply the

weight λ of 10 to Lcamera to ensure that its magnitude is comparable to other losses, which stabilizes optimization.

L = λLcamera + Ldepth + Lpointmap (5)

##### 3.3. Depth-reliable Spatial Editing

Based on point-cloud editing and motion planning, we can synthesize novel object placements and corresponding manipulation trajectories in 3D space. To obtain reliable depth from the edited point clouds, we improve the spatial editing pipeline with techniques such as background inpainting and depth filtering. Crucially, we introduce robot pose correction during spatial editing, ensuring the resulting depth maps are consistent with the robot’s kinematics.

Trajectory Synthesis. Inspired by the previous work [55], we decompose a point-cloud demonstration into two types of segments: the motion segment, where the robot moves freely in space, and the skill segment, where the robot interacts with objects. Given a transformation that relocates an object, we apply the same transformation to the robot’s point cloud in the corresponding skill segment. This en-

Algorithm 1 Depth-reliable Spatial Editing Pipeline Require: Source point clouds Probot,Pobject,Pbg, joint

states Q, action trajectory A, camera poses T .

Ensure: Novel depth sequence D⋆, joint states Q⋆, action

trajectory A⋆, camera poses T ⋆.

function RENDERDEPTH(P,T ,Q) D1 ← ProjectPointCloud(P,T ) D2 ← RenderLinkDepth(Q,T ) return Merge(D1,D2)

end function Random Sample a Object Transform T ∈ R4×4 D⋆ ← list(), Q⋆ ← list(), A⋆ ← list(), T ⋆ ← list() // Motion Segment A⋆start ← A0, A⋆end ← TAend, for t in Motion do

Tt,A⋆t,Q⋆t ← MotionPlan(A⋆start,A⋆end,t) Ptarm ← FK(Ptrobot,Qt) Ptee ← Ptrobot \ Ptarm Pt⋆ ← TtPtee ∪ TPtobject ∪ Pbg D⋆ ← D⋆ ∪ RenderDepth(Pt⋆,TtTt,Q⋆t) Q⋆ ← Q⋆ ∪ Q⋆t, A⋆ ← A⋆ ∪ A⋆t, T ⋆ ← T ⋆ ∪ TtTt

###### end for

// Skill Segment for t in Skill do

Q⋆t ← IK(TAt) Ptarm ← FK(Ptrobot,Qt) Ptee ← Ptrobot \ Ptarm Pt⋆ ← T(Ptee ∪ Ptobject) ∪ Pbg D⋆ ← D⋆ ∪ RenderDepth(Pt⋆,TTt,Q⋆t) Q⋆ ← Q⋆ ∪ Q⋆t, A⋆ ← A⋆ ∪ TAt, T ⋆ ← T ⋆ ∪ TTt

end for return D⋆, Q⋆, A⋆, T ⋆

sures that the robot-object relation remains consistent with the original demonstration, thereby preserving realistic in-

[Figure 75]

Transformer Backbone Generated Video

[Figure 76]

Unified Text Prompt

|Text Encoder|
|---|

[Figure 77]

[Figure 78]

[Figure 79]

|Cross-Attention|
|---|

[Figure 80]

[Figure 81]

[Figure 82]

Best quality, consistent and smooth …

[Figure 83]

[Figure 84]

[Figure 85]

3D Control Interface

|Intra-view|
|---|

|Intra-view|
|---|

|Intra-view|
|---|

[Figure 86]

[Figure 87]

[Figure 88]

|Cross-Attention|
|---|

|Memory Cache<br><br>[Figure 89]|
|---|

|Intra-view|
|---|

|Intra-view|
|---|

|Intra-view|
|---|

Sample

Canny Edge

[Figure 90]

[Figure 91]

[Figure 92]

|Cross-Attention|
|---|

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

|Cross-view Self-Attention|
|---|

[Figure 100]

[Figure 101]

[Figure 102]

###### Control Condition

[Figure 103]

|Latent Embedding|VAE Encoder|
|---|---|
| | |

###### Action Camera

- Figure 3. The framework of 3D-Controlled Video Generation. We utilize depth as the 3D control interface, in conjunction with edges, actions, and ray maps, to guide the generation of multi-view demonstrations.

teractions. The new motion segment is generated through motion planning [42].

build a Transformer-based video generation model followed by [26, 34, 35]. The model works with three key designs: dual-attention mechanism, depth control interface, and smooth object relocation.

Depth Projection. Since the camera is rigidly attached to the robot’s end-effector, we apply the same transformation to the camera pose to obtain its updated position after editing. This allows us to project the edited point cloud to generate the corresponding depth map. However, the projected depth maps often suffer from holes and noise due to changes in object positions and varying camera distances, which can cause sparse or missing regions. We mitigate these artifacts through background [49] inpainting and depth filtering.

Dual-attention Mechanism. The dual-attention mechanism consists of intra-view attention and cross-view attention. Intra-view attention performs self-attention over the tokens of each individual view, capturing detailed spatial context within that view. Cross-view attention, on the other hand, computes self-attention across all views simultaneously, enabling the model to utilize the multi-view correspondence. This dual-attention design not only facilitates multi-view consistency in the generated videos by allowing interactions of visual features across different scale contexts, but also significantly reduces computational cost compared to applying global attention at every layer.

Robot Pose Correction. A major challenge arises from incorrect robot poses, as the previous editing process treats the entire robot as a rigid body. Instead, only the endeffector should be transformed, and the remaining arm must be realigned to preserve kinematic validity. To ensure the reliability of depth maps, we segment out the original robot links with URDF and source joint states, and re-render the arm depth with synthesized actions. This process yields physically plausible robot configurations and produces accurate depth observations without rigid-body artifacts. To summarize the overall procedure, we illustrate in Algorithm 1 an example pipeline consisting of one motion segment followed by one skill segment.

Depth Control Interface. We use depth as the 3D control interface of video generation. Specifically, we concatenate the depth map with the image latent representations and feed them jointly into the transformer backbone, enabling the model to condition video generation on 3D structural cues. This design ensures the synthesized robotic demonstrations remain consistent with the geometry information in the depth sequence, which encodes robot motion and object interactions. In addition, we incorporate auxiliary conditioning signals, including Canny edges, action maps, and ray maps, which further sharpen object boundaries, improve motion grounding, and enhance multi-view consistency, respectively. Overall, these structured controlling signals provide strong 3D inductive biases, enabling more realistic manipulation behaviors in the generated videos.

##### 3.4. 3D-Controlled Video Generation

After obtaining the edited depth, action, and camera pose sequences from the 3D editing pipeline described in Sec-

- tion 3.3, we then convert them into 2D visual observations required for policy training. We propose a 3D-conditioned video generation model that, starting from the first frame, synthesizes novel robot manipulation videos with realistic visual appearance, multi-view consistency, and physically plausible interactions. As shown in Figure 3, we

Smooth Object Relocation. With the condition’s control, the model is able to generate the manipulation video from the first frame, but how to relocate the objects in the first

Mug to Basket Pour Water Lift Box Scan Barcode Total Go-1 π0.5 Go-1 π0.5 Go-1 π0.5 Go-1 π0.5 Go-1 π0.5

# Demo

Real 10 8 / 20 8 / 20 5 / 20 1 / 20 11 / 20 13 / 20 5 / 20 4 / 20 36.3% 32.5% Real 20 12 / 20 14 / 20 7 / 20 2 / 20 12 / 20 15 / 20 8 / 20 5 / 20 48.8% 45.0% Real 50 14 / 20 13 / 20 8 / 20 8 / 20 15 / 20 17 / 20 12 / 20 11 / 20 61.3% 61.3%

- Real 1 Gen 200 14 / 20 15 / 20 12 / 20 10 / 20 12 / 20 10 / 20 14 / 20 11 / 20 65.0% 57.5%

- Real 2 Gen 200 15 / 20 15 / 20 10 / 20 10 / 20 14 / 20 17 / 20 17 / 20 14 / 20 70.0% 70.0% Real 5 Gen 200 17 / 20 18 / 20 12 / 20 12 / 20 16 / 20 18 / 20 18 / 20 17 / 20 78.8% 81.3%

- Table 1. Success rates of Real2Edit2Real on four real-world manipulation tasks and two VLA policies Go-1 and π0.5. Both policies trained on data generated from only 1–5 source demonstrations can match or outperform those trained on 50 real-world demonstrations, improving data efficiency by up to 10-50×.

frame remains difficult. To make full use of the depthcontrolled video generation model, we convert the object relocation to a smooth transformation, where we interpolate both the translation and rotation of objects during spatial editing to synthesize object moving trajectories before manipulation starts. Through this smooth relocation, we convert image editing to video generation and process the object relocation and demonstration generation together by

its internal motion API. Three RGB cameras are mounted on its head, left wrist, and right wrist. The workspace is a 50cm × 40cm rectangular area on the white desktop. Please refer to the supplementary materials for more details.

VLA Policy. We conduct experiments on two VLA policies: Go-1 [4] and π0.5 [16]. For Go-1, we only finetune the action expert while keeping the backbone frozen because we use the same embodiment as its pretrained data. The action is the 6D end-effector pose. For π0.5, due to embodiment mismatch, we perform full finetuning. The action is the 7-DoF joint angles. Each training typically consists of 10K iterations; for the smaller training data cases, we proportionally reduce the iteration count to 100 epochs. Both policies are trained on 8 H100 GPUs for 2-4 hours.

- 3D-controlled video generation, achieving a unified and efficient generation framework.

#### 4. Experiments

To demonstrate the effectiveness of our data generation framework, we present the following experiments. In Sec-

- tion 4.2, we evaluate its impact on real-world policy learning across four manipulation tasks. In Section 4.3, we showcase the flexibility of our proposed framework through height and texture editing. In Section 4.4, we conduct ablations to validate the necessity of each module and our key designs. Finally, in Section 4.5, we provide visualizations of our novel demonstration generation.

##### 4.2. Gen2Real Policy Learning

Tasks. As shown in Figure 1, we conduct real-robot experiments on four tasks, covering from single-arm to dual-arm manipulation:

- • Mug to Basket: The robot uses its right arm to grasp the mug and place it stably inside the basket.
- • Pour Water: The robot uses its left arm to pick up the kettle and pour water into the paper cup by aligning the spout with the cup.
- • Lift Box: The robot grasps both sides of the box using its two arms and lifts it.
- • Scan Barcode: The robot grasps a snack with its left hand and a barcode scanner with its right hand, and scans the barcode by aligning the scanner with it.

##### 4.1. Implementation Details

Real2Edit2Real. For Metric-VGGT, we sample 40K frames from the Agibot-DigitalWorld dataset [59] as simulation training data and collect 100K real robot data with depth sensors as real training data. We full-finetune VGGT [47] on 8 H100 GPUs for 150K iterations, 20 hours with the learning rate 2e-4 and backbone learning rate 2e-

Settings. For real-world data, we collect demonstrations via teleoperation by placing objects in diverse configurations that uniformly cover the workspace, including variations in both position and orientation. For generation experiments, we randomly sample a specified number of source demonstrations from the collected data and apply randomized object relocations around their original poses. This procedure yields 200 synthesized demonstrations, and training is performed solely on these generated samples. During evaluation, objects are uniformly randomly placed across the

###### 5. For the video generation model, we sample 7K episodes from 64 tasks in the AgiBot-World dataset [4] as training data. We train the video generation model by fine-tuning the backbone of GE-Sim [26](based on Cosmos-Predict-

- 2B [59]) on 8 H100 GPUs for 20K iterations, 60 hours with the learning rate 1e-4, batch size 16. With parallelization across 8 H100 GPUs, the average generation time for a 20second, 30-FPS episode is 48.6 seconds. Please refer to the supplementary materials for more details. Hardware Setup. We use the Agibot Genie G1 robot with

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

# Demo White Green Black Blue Red Total Real 50 7/10 4/10 5/10 5/10 4/10 50% Real 1 Gen 200 7/10 4/10 6/10 4/10 5/10 52% Real 1 Gen 200* 6/10 7/10 6/10 7/10 8/10 68%

[Figure 112]

[Figure 113]

[Figure 114]

Gen

[Figure 115]

Raise

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Source

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Table 3. Performance comparison of Go-1 under different desktop textures. Real 1 Gen 200* means generating data includes different textures. Our method is robust to the texture variation.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

(a) (b)

Real Data VGGT Metric-VGGT (ours)

Figure 4. Experiment setups of (a) height and (b) texture editing.

[Figure 132]

# Demo Tabletop Platform Total Tabletop Real 20 5/5 0/5 50% Tabletop Real 1 Gen 40 4/5 4/5 80%

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

- Table 2. Performance comparison of Go-1 on height generalization. Our method successfully generalizes to the unseen height.

| | |
|---|---|
| | |

workspace to assess spatial generalization.

Results. Table 1 shows the manipulation success rate of four tasks with different training data. Real data results show that when the number of demonstrations is fewer than 20, the average task success rate drops below 50%, indicating that VLAs exhibit limited spatial generalization when trained with scarce data. Conversely, policies trained on 200 demonstrations generated from only a single source demonstration by Real2Edit2Real achieve comparable spatial generalization to those trained on 50 real-world demonstrations. As the number of source demonstrations increases, the average success rate of policies trained on the same 200 generated demonstrations improves significantly. When trained with data generated from 5 real demonstrations, the policies of Go-1 and π0.5 achieve average success rates of 78.8% and 81.3%, surpassing those trained on 50 real demonstrations by 17.5% and 20%, respectively. This improvement arises because increasing source demonstrations introduces more diverse robot–object interaction patterns and expands the spatial coverage of the generated data. Overall, the experimental results demonstrate that Real2Edit2Real improves data efficiency by 10-50× through data generation, confirming its effectiveness as a demonstration generation framework for robotics.

Figure 5. Ablation study of geometry reconstruction. The left endeffector is in red, and the right end-effector is in yellow.

w/o RPC w/ RPC

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

- Figure 6. Ablation study of robot pose correction (RPC). Target Depth w/o SOR w/ SOR

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Smooth Object Relocation

- Figure 7. Ablation study of smooth object relocation (SOR).

##### 4.3. More Applications

generates video from the first frame, we can easily edit the video through first-frame editing, like changing the background texture. Table 3 illustrates the performance under different desktop textures shown in Figure 4 and indicates that our method can generate demonstrations with different textures to improve policy robustness.

Height Editing. With smooth object relocation described in Sec. 3.4, we can also edit the object height as shown in Figure 4. Table 2 shows the performance comparison on height generalization. The policy trained by 20 real demonstrations on the tabletop completely fails on the platform height because of OOD. If we generate 20 demonstrations on the tabletop and 20 demonstrations on the platform through our framework, the policy can achieve 80% success rate.

##### 4.4. Ablation Study

Geometry Reconstruction. Figure 5 provides point-cloud visualizations of real data, VGGT, and ours. We can see that

Texture Editing. Since the model described in Sec. 3.4

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Mug to Basket

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

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

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Pour Water

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Lift Box

[Figure 211]

[Figure 212]

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

Scan Barcode

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Figure 8. Visualization of videos generated by Real2Edit2Real on all four real-world tasks.

our model predicts the cleanest point cloud and the most accurate camera pose in robot scenarios, while the real-world data suffer from inaccurate camera poses and the VGGT reconstructions contain substantial clutter and noise.

rect manipulation trajectories, and maintain both multi-view consistency and realistic visual appearance. Please refer to the supplementary materials for more visualizations.

Robot Pose Correction. Figure 6 shows the ablation results of robot pose correction. Without robot pose correction, the erroneous depth maps lead to blurry and inconsistent generation results. With robot pose correction, the depth maps become kinematically consistent, allowing the model to generate realistic robot motions in the synthesized videos. Smooth Object Relocation. Figure 7 shows the ablation results of smooth object relocation. Without smooth object relocation, the generated object placements often exhibit noticeable errors, leading to unusable demonstrations. In contrast, smooth object relocation enables the precise placement of objects at the target locations.

#### 5. Conclusion

In this work, we introduce Real2Edit2Real, a framework that enables scalable demonstration generation by linking 3D editability with 2D visual data. Through metric-scale geometry reconstruction, depth-reliable spatial editing, and 3D-controlled video generation, our approach synthesizes realistic and kinematically consistent multi-view manipulation demonstrations. Experiments across four real-world manipulation tasks show that policies trained on data generated from as few as 1–5 demonstrations can match or surpass those trained on 50 real-world demonstrations, improving data efficiency by up to 10–50×. Additional results on height and texture editing further highlight the extensibility of our framework, suggesting its potential to serve as a unified engine for scalable data generation.

##### 4.5. Visualization

Figure 8 is the visualization of videos generated by Real2Edit2Real on all four real-world tasks. The generated videos successfully relocate the objects, synthesize the cor-

#### 6. Acknowledgement

This work was supported by the National Natural Science Foundation of China (62376006). We would like to thank Zizhao Tong from University of Chinese Academy of Sciences for his fruitful discussion and Haolin Chen from Zhongguancun Academy for his technical support.

#### References

- [1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang,

and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2

- [2] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In arXiv preprint arXiv:2307.15818, 2023. 2
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 3

- [4] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, Shu Jiang, Yuxin Jiang, Cheng Jing, Hongyang Li, Jialu Li, Chiming Liu, Yi Liu, Yuxiang Lu, Jianlan Luo, Ping Luo, Yao Mu, Yuehan Niu, Yixuan Pan, Jiangmiao Pang, Yu Qiao, Guanghui Ren, Cheng Ruan, Jiaqi Shan, Yongjian Shen, Chengshi Shi, Mingkang Shi, Modi Shi, Chonghao Sima, Jianheng Song, Huijie Wang, Wenhao Wang, Dafeng Wei, Chengen Xie, Guo Xu, Junchi Yan, Cunbiao Yang, Lei Yang, Shukai Yang, Maoqing Yao, Jia Zeng, Chi Zhang, Qinglin Zhang, Bin Zhao, Chengyue Zhao, Jiaqi Zhao, and Jianchao Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems, 2025. 2, 6, 15
- [5] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In

- Proceedings of Robotics: Science and Systems (RSS), 2023. 2, 17
- [6] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research,

2024. 2

- [7] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Wenhao Zhang, et al. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. arXiv preprint arXiv:2505.03233, 2025. 2
- [8] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 3

- [9] Caelan Garrett, Ajay Mandlekar, Bowen Wen, and Dieter Fox. Skillmimicgen: Automated demonstration generation for efficient skill learning and deployment. In 8th Annual Conference on Robot Learning, 2024. 2, 13
- [10] Xiaoshen Han, Minghuan Liu, Yilun Chen, Junqiu Yu, Xiaoyang Lyu, Yang Tian, Bolun Wang, Weinan Zhang, and Jiangmiao Pang. Re3sim: Generating high-fidelity simulation data via 3d-photorealistic real-to-sim for robotic manipulation. arXiv preprint arXiv:2502.08645, 2025. 2
- [11] Ryan Hoque, Ajay Mandlekar, Caelan Garrett, Ken Goldberg, and Dieter Fox. Intervengen: Interventional data generation for robust and data-efficient robot imitation learning. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2840–2846, 2024. 2
- [12] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 3
- [13] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 16
- [14] Siyuan Huang, Liliang Chen, Pengfei Zhou, Shengcong Chen, Zhengkai Jiang, Yue Hu, Yue Liao, Peng Gao, Hongsheng Li, Maoqing Yao, et al. Enerverse: Envisioning embodied future space for robotics manipulation. arXiv preprint arXiv:2501.01895, 2025. 3
- [15] Yan Huang, Shoujie Li, Xingting Li, and Wenbo Ding. Umigen: A unified framework for egocentric point cloud generation and cross-embodiment robotic imitation learning. arXiv preprint arXiv:2511.09302, 2025. 13
- [16] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling,

- Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization, 2025. 2, 6
- [17] Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, et al. Enerverse-ac: Envisioning embodied environments with action condition. arXiv preprint arXiv:2505.09723, 2025. 3
- [18] Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Fan, and Yuke Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025. 2
- [19] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025. 3
- [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42

(4), 2023. 2

- [21] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2
- [22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 14
- [23] Kuaishou Technology. Kling AI: Next-Generation AI Creative Studio. https://klingai.com/. 3
- [24] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r, 2024. 3
- [25] Chengshu Li, Mengdi Xu, Arpit Bahety, Hang Yin, Yunfan Jiang, Huang Huang, Josiah Wong, Sujay Garlanka, Cem Gokmen, Ruohan Zhang, et al. Momagen: Generating demonstrations under soft and hard constraints for multi-step bimanual mobile manipulation. arXiv preprint arXiv:2510.18316, 2025. 2
- [26] Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, et al. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025. 3, 5, 6, 15, 17
- [27] Tao Lin, Gen Li, Yilei Zhong, Yanwen Zou, Yuxin Du, Jiting Liu, Encheng Gu, and Bo Zhao. Evo-0: Vision-languageaction model with implicit spatial understanding. arXiv preprint arXiv:2507.00416, 2025. 3
- [28] Liu Liu, Xiaofeng Wang, Guosheng Zhao, Keyu Li, Wenkang Qin, Jiaxiong Qiu, Zheng Zhu, Guan Huang, and Zhizhong Su. Robotransfer: Geometry-consistent video diffusion for robotic visual policy transfer. arXiv preprint

arXiv:2505.23171, 2025. 3, 13

- [29] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 14
- [30] Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, and Shuran Song. Geometry-aware 4d video generation for robot manipulation. arXiv preprint arXiv:2507.01099, 2025. 3
- [31] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation. In European Conference on Computer Vision, pages 349–366. Springer, 2024. 3
- [32] Yuanxun Lu, Jingyang Zhang, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Long Quan, Xun Cao, Yao Yao, and Shiwei Li. Matrix3d: Large photogrammetry model all-in-one. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11250–11263, 2025. 3
- [33] Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. In 7th Annual Conference on Robot Learning, 2023. 2, 13
- [34] NVIDIA, Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, Prithvijit Chattopadhyay, Mike Chen, Yongxin Chen, Yu Chen, Shuai Cheng, Yin Cui, Jenna Diamond, Yifan Ding, Jiaojiao Fan, Linxi Fan, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Ruiyuan Gao, Yunhao Ge, Jinwei Gu, Aryaman Gupta, Siddharth Gururani, Imad El Hanafi, Ali Hassani, Zekun Hao, Jacob Huffman, Joel Jang, Pooya Jannaty, Jan Kautz, Grace Lam, Xuan Li, Zhaoshuo Li, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Yen-Chen Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Yifan Lu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Seungjun Nah, Yashraj Narang, Abhijeet Panaskar, Lindsey Pavao, Trung Pham, Morteza Ramezanali, Fitsum Reda, Scott Reed, Xuanchi Ren, Haonan Shao, Yue Shen, Stella Shi, Shuran Song, Bartosz Stefaniak, Shangkun Sun, Shitao Tang, Sameena Tasmeen, Lyne Tchapmi, WeiCheng Tseng, Jibin Varghese, Andrew Z. Wang, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Jiashu Xu, Dinghao Yang, Xiaodong Yang, Haotian Ye, Seonghyeon Ye, Xiaohui Zeng, Jing Zhang, Qinsheng Zhang, Kaiwen Zheng, Andrew Zhu, and Yuke Zhu. World simulation with video foundation models for physical ai,

2025. 3, 5

- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. 5
- [36] Zezhong Qian, Xiaowei Chi, Yuming Li, Shizun Wang, Zhiyuan Qin, Xiaozhu Ju, Sirui Han, and Shanghang Zhang. Wristworld: Generating wrist-views via 4d world models for robotic manipulation. arXiv preprint arXiv:2510.07313,

2025. 3

- [37] M Nomaan Qureshi, Sparsh Garg, Francisco Yandun, David Held, George Kantor, and Abhisesh Silwal. Splatsim: Zeroshot sim2real transfer of rgb manipulation policies using gaussian splatting. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 6502–6509. IEEE, 2025. 2
- [38] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 14
- [39] Vaibhav Saxena, Matthew Bronars, Nadun Ranawaka Arachchige, Kuancheng Wang, Woo Chul Shin, Soroush Nasiriany, Ajay Mandlekar, and Danfei Xu. What matters in learning from large-scale datasets for robot manipulation. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [40] Yu Shang, Xin Zhang, Yinzhou Tang, Lei Jin, Chen Gao, Wei Wu, and Yong Li. Roboscape: Physics-informed embodied world model. arXiv preprint arXiv:2506.23135, 2025. 3
- [41] Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick Labatut, and Piotr Bojanowski. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 17
- [42] Balakumar Sundaralingam, Siva Kumar Sastry Hari, Adam Fishman, Caelan Garrett, Karl Van Wyk, Valts Blukis, Alexander Millane, Helen Oleynikova, Ankur Handa, Fabio Ramos, et al. Curobo: Parallelized collision-free robot motion generation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8112–8119. IEEE,

2023. 5

- [43] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. In The Thirteenth International Conference on Learning Representations. 3
- [44] Zizhao Tong, Di Chen, Sicheng Hu, Hongwei Fan, Liliang Chen, Guanghui Ren, Hao Tang, Hao Dong, and Ling Shao. Fidelity-aware data composition for robust robot generalization. arXiv preprint arXiv:2509.24797, 2025. 3, 13
- [45] Marcel Torne Villasevil, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, and Pulkit Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024. 2
- [46] Boyuan Wang, Xinpan Meng, Xiaofeng Wang, Zheng Zhu, Angen Ye, Yang Wang, Zhiqin Yang, Chaojun Ni, Guan Huang, and Xingang Wang. Embodiedreamer: Advancing real2sim2real transfer for policy training via embodied world modeling. arXiv preprint arXiv:2507.05198, 2025. 3
- [47] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt:

- Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3, 4, 6, 13
- [48] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. Advances in Neural Information Processing Systems, 34:27171–27183, 2021. 3
- [49] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025. 5, 13
- [50] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 3
- [51] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning, 2025. 3
- [52] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. arXiv preprint arXiv:2507.07982, 2025. 3
- [53] Xiuwei Xu, Angyuan Ma, Hankun Li, Bingyao Yu, Zheng Zhu, Jie Zhou, and Jiwen Lu. R2rgen: Real-to-real 3d data generation for spatially generalized manipulation. arXiv preprint arXiv:2510.08547, 2025. 13
- [54] Yuan Xu, Jiabing Yang, Xiaofeng Wang, Yixiang Chen, Zheng Zhu, Bowen Fang, Guan Huang, Xinze Chen, Yun Ye, Qiang Zhang, et al. Egodemogen: Novel egocentric demonstration generation enables viewpoint-robust manipulation. arXiv preprint arXiv:2509.22578, 2025. 3
- [55] Zhengrong Xue, Shuying Deng, Zhenyang Chen, Yixuan Wang, Zhecheng Yuan, and Huazhe Xu. DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning. In Proceedings of Robotics: Science and Systems, LosAngeles, CA, USA, 2025. 2, 4, 13
- [56] Sizhe Yang, Wenye Yu, Jia Zeng, Jun Lv, Kerui Ren, Cewu Lu, Dahua Lin, and Jiangmiao Pang. Novel demonstration generation with gaussian splatting enables robust one-shot manipulation. arXiv preprint arXiv:2504.13175, 2025. 2, 13
- [57] Justin Yu, Letian Fu, Huang Huang, Karim El-Refai, Rares Andrei Ambrus, Richard Cheng, Muhammad Zubair Irshad, and Ken Goldberg. Real2render2real: Scaling robot data without dynamics simulation or robot hardware. In 9th Annual Conference on Robot Learning, 2025. 2, 13
- [58] Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In Proceedings of Robotics: Science and Systems (RSS), 2024. 2
- [59] Jiyao Zhang, Mingjie Pan, Baifeng Xie, Yinghao Zhao, Wenlong Gao, Guangte Xiang, Jiawei Zhang, Dong Li, Zhijun Li, Sheng Zhang, Hongwei Fan, Chengyue Zhao, Shukai Yang, Maoqing Yao, Chuanzhe Suo, and Hao Dong. Agibot digitalworld. https://agibot-digitalworld.com/,

2025. 6, 15

- [60] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21936–21947, 2025. 3
- [61] Wentao Zhao, Jiaming Chen, Ziyu Meng, Donghui Mao, Ran Song, and Wei Zhang. Vlmpc: Vision-language model predictive control for robotic manipulation. arXiv preprint arXiv:2407.09829, 2024. 3

### Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface Supplementary Material

Source Generation

Method

No Simulation RGB Only VLA Compatible Novel Texture Novel Trajectory

MimicGen [33] ✗ ✗ ✓ ✓ ✓ SkillMimicGen [9] ✗ ✗ ✓ ✓ ✓ RoboSplat [56] ✓ ✗ ✓ ✓ ✓ Real2Render2Real [57] ✓ ✗ ✓ ✓ ✓ DemoGen [55] ✓ ✗ ✗ ✗ ✓ R2RGen [53] ✓ ✗ ✗ ✗ ✓ UMIGen [15] ✓ ✗ ✗ ✗ ✓ RoboTransfer [28] ✓ ✓ ✓ ✓ ✗ MVAug [44] ✓ ✓ ✓ ✓ ✗

Real2Edit2Real (ours) ✓ ✓ ✓ ✓ ✓

Table 4. Comparison with Other One-to-many Demonstration Generation Methods.

#### 7. Contribution Clarification

ing its unified and flexible design.

To better clarify our contribution, we provide a detailed comparison between our method and other one-to-many demonstration generation approaches, as shown in Table 4. Simulation-based methods like MimicGen [33] and SkillMimicGen [9] rely on simulators and require scene and object assets, which not only introduce a significant simto-real gap but also make it difficult to perform data augmentation directly on real-world data. Methods such as RoboSplat [56] and Real2Render2Real [57] are built on 3D Gaussian Splatting. Although they do not require a simulation engine, they still rely on dense scanning to reconstruct the objects or scenes. This means that they cannot perform data generation using only the RGB observations from the original demonstrations, which significantly limits their scalability. Another line of research, including DemoGen [55], R2RGen [53], and UMIGen [15] , generates new 3D point-cloud demonstrations through pointcloud editing. However, their reliance on depth sensors limits their compatibility with the current mainstream VLA paradigm that uses multi-view RGB inputs, and also prevents them from performing texture-level augmentation. Methods based on video generation, such as RoboTransfer [28] and MVAug [44], can directly augment multi-view 2D demonstrations, but they only enhance visual aspects such as texture, without increasing the diversity of object spatial distributions or robot trajectories.

In contrast, our method requires no simulator and directly augments the original RGB observations, significantly improving scalability. It simultaneously generates new textures and trajectories for VLA training, highlight-

#### 8. Real2Edit2Real Implementation Details

In this section, we provide more details of the proposed framework, Real2Edit2Real:

- • In Section 8.1, we provide additional information for the hybrid training paradigm.
- • In Section 8.2, we explain the full pipeline of depthreliable spatial editing in detail.
- • In Section 8.3, we discuss more about 3D-controlled video generation model.

##### 8.1. Metric-scale Geometry Reconstruction

Data Visualization. Fig. 9 shows the visualization of the training data. We can see that real-world depth maps are often noisy and contain large invalid regions, whereas synthetic depth is clean and accurate. By training with our proposed hybrid training paradigm, our model learns to reconstruct geometry in metric scale in the real world, effectively compensating for the limitations of depth sensors.

Training Details. In Table 5, we provide the details of finetuning VGGT [47] to Metric-VGGT.

##### 8.2. Depth-reliable Spatial Editing

Background Depth Completion. As we mentioned in the manuscript, projecting edited point clouds to depth maps may cause missing regions in the background due to the object moving and novel robot motion. To mitigate this artifact, we first inpaint the background, which deletes the foreground objects and robot in the multi-view first frames with an image-edit model [49]. Figure 10 provides the prompt

Head Left Hand Right Hand

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

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Real

Sim

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

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

Figure 9. Training Data Visualization of Metric-VGGT. Depth visualization: red is the nearest, blue is the farthest.

[Figure 287]

[Figure 288]

[Figure 289]

"While keeping everything else in the image unchanged, remove the black gripper and the black wire."

"While keeping everything else in the image

unchanged, remove the white robotic arm." "While keeping everything else in the image

[Figure 290]

[Figure 291]

[Figure 292]

unchanged, remove xxx on the table."

Figure 10. Prompt Used for Background Inpainting. In the prompt, xxx means the manipulated objects.

Figure 11. Example of the Inpainted Background.

we used for image editing, and Figure 11 shows an example of the inpainted background. Then, we reconstruct the metric geometry of the background with Metric-VGGT. To correct the metric-scale inconsistencies introduced by image editing, we incorporate an additional point cloud alignment procedure, as shown in Algorithm 2.

Grounded-SAM [22, 29, 38] and robotic dual arms through forward kinematics. Following, we provide an example to detail the full spatial editing pipeline. Algorithm 3, 4 shows the spatial editing pipeline of the Mug to Basket task. The Object Relocation Segment produces the depth sequence for Smooth Object Relocation described in the manuscript.

Spatial Editing Pipeline. After getting a completed background point cloud, we separate foreground objects through

###### Config Value

Base Model VGGT-1B Training Real Data 100,000 Training Sim Data 40,000 Fine-Tuning Scheme Full Parameter Total Training Steps 150,000 Learning Rate 2e-4 Backbone Learning Rate 2e-5 LR Scheduler Cosine Annealing Scheduler ETA Minimum 1e-6 Weight Decay 1e-2 View Num 3 Global Batch Size 16 Gradient Accumulation Steps 4 Mixed Precision bf16 Optimizer AdamW Training Image Size 518

Table 5. Training Details of Metric-VGGT.

Algorithm 2 Background Point-Cloud Alignment Require: Origin first frame point cloud Po, unaligned

background point cloud Pedit, table mask Mtable.

Ensure: Metric-aligned background point cloud Pbg.

function ESTIMATEPLANE(P) plane ← RansacPlaneSegment(P) // plane : [a,b,c,d],ax + by + cz + d = 0 return plane

end function planeo ← EstimatePlane(Po[Mtable]) planeedit ← EstimatePlane(Pedit[Mtable]) scale ← planeo[3]/planeedit[3] Pbg ← scale × Pedit return Pbg

##### 8.3. 3D-Controlled Video Generation

Training Data. For training the 3D-controlled multi-view video generation model, we sample 7K episodes of 64 tasks from the Agibot-World datasets [4]. To get the control conditions of the training data, we used the Metric-VGGT to predict the depth maps and compute the Canny Edges from depth. To ensure the 3D control condition remains consistent across multi-view and temporal, we perform global normalization on the depth sequences of all three views within a training chunk, rather than normalizing each depth map individually.

Condition Dropout. In the training stage, we finetune the backbone of GE-Sim [26] (based on CosmosPredict-2B [59]) with sampling data from the Agibot-World Dataset [4]. In multi-condition compositional generation, intensity-based conditions such as depth maps and Canny

Algorithm 3 Pipeline of Mug to Basket

Require: Source point clouds Pl, Pr, Pmug, Pbasket, background point cloud Pbg, joint states Q, action trajectory A, camera poses T h, T l, T r, skill-1 start timestep t1, skill-1 end timestep t2, skill-2 start timestep t3, skill-2 end timestep t4.

Ensure: Novel depth sequence Dh⋆, Dl⋆, Dr⋆, joint states

Q⋆, action trajectory A⋆, camera poses T ⋆.

function RENDERDEPTH(P,T ,Q) D1 ← ProjectPointCloud(P,T ) D2 ← RenderLinkDepth(Q,T ) return Merge(D1,D2)

end function Sample a Object Transform Pair Tmug,Tbasket ∈ R4×4 Dh⋆ ← list(), Dl⋆ ← list(), Dr⋆ ← list() Q⋆ ← list(), A⋆ ← list(), T ⋆ ← list() // Object Relocation Segment for t in range(0, 30) do

Tmugt ,Tbaskett ← Interpolate(Tmug,Tbasket,30,t) Pt⋆ ← Tmugt P0mug∪Tbaskett P0basket∪P0l ∪P0r∪Pbg Dh∗ ← Dh∗ ∪ RenderDepth(Pt⋆,T0h,Q0) Dl∗ ← Dl∗ ∪ RenderDepth(Pt⋆,T0l,Q0) Dr∗ ← Dr∗ ∪ RenderDepth(Pt⋆,T0r,Q0) Q⋆ ← Q⋆ ∪ Q0, A⋆ ← A⋆ ∪ A0 T ⋆ ← T ⋆ ∪ (T0h,T0l,T0r)

end for // Motion-1 Segment A⋆start ← A0, A⋆end ← TmugAt

,

1

for t in range(0, t1) do Tt,A⋆t,Q⋆t ← MotionPlan(A⋆start,A⋆end,t) Ptree ← Ptr \ FK(Ptr,Qt) Pt⋆ ← TtPtree∪Ptl∪TmugPtmug∪TbasketPtbasket∪

Pbg

Dh⋆ ← Dh⋆ ∪ RenderDepth(Pt⋆,Tth,Q⋆t) Dl⋆ ← Dl⋆ ∪ RenderDepth(Pt⋆,Ttl,Q⋆t) Dr⋆ ← Dr⋆ ∪ RenderDepth(Pt⋆,TtTtr,Q⋆t) Q⋆ ← Q⋆ ∪ Q⋆t, A⋆ ← A⋆ ∪ A⋆t T ⋆ ← T ⋆ ∪ (Tth,Ttl,TtTtr)

end for // Skill-1 Segment for t in range(t1, t2) do

Q⋆t ← IK(TmugAt) Ptree ← Ptr \ FK(Ptr,Qt) Pt⋆ ← Tmug(Ptree∪Ptmug)∪Ptl∪TmugPtmug∪Pbg Dh⋆ ← Dh⋆ ∪ RenderDepth(Pt⋆,Tth,Q⋆t) Dl⋆ ← Dl⋆ ∪ RenderDepth(Pt⋆,Ttl,Q⋆t) Dr⋆ ← Dr⋆ ∪ RenderDepth(Pt⋆,TmugTtr,Q⋆t) Q⋆ ← Q⋆ ∪ Q⋆t, A⋆ ← A⋆ ∪ TmugAt T ⋆ ← T ⋆ ∪ (Tth,Ttl,TmugTtr)

###### end for

Algorithm 4 Continued to Pipeline of Mug to Basket // Motion-2 Segment A⋆start ← TmugAt2

###### Config Value

Base Model GE-Sim-2B Training Data 7000 Episodes 64 Tasks Fine-Tuning Scheme Full Parameter Total Training Steps 20,000 Learning Rate 1e-4 LR Scheduler Constant with Warmup LR Warmup Steps 1000 Weight Decay 5e-5 Global Batch Size 16 Gradient Accumulation Steps 1 Max Gradient Norm 1.0 Mixed Precision bf16 Optimizer AdamW Training Resolution 384×512 Video Chunk Length 25 Memory Frames 4

, A⋆end ← TbasketAt

,

3

- for t in range(t2, t3) do Tt,A⋆t,Q⋆t ← MotionPlan(A⋆start,A⋆end,t) Ptree ← Ptr \ FK(Ptr,Qt) Pt⋆ ← Tt(Ptree∪Ptmug)∪Ptl∪TbasketPtbasket∪Pbg Dh⋆ ← Dh⋆ ∪ RenderDepth(Pt⋆,Tth,Q⋆t) Dl⋆ ← Dl⋆ ∪ RenderDepth(Pt⋆,Ttl,Q⋆t) Dr⋆ ← Dr⋆ ∪ RenderDepth(Pt⋆,TtTtr,Q⋆t) Q⋆ ← Q⋆ ∪ Q⋆t, A⋆ ← A⋆ ∪ A⋆t

T ⋆ ← T ⋆ ∪ (Tth,Ttl,TtTtr) end for // Skill-2 Segment

- for t in range(t3, t4) do Q⋆t ← IK(TbasketAt) Ptree ← Ptr \ FK(Ptr,Qt) Pt⋆ ← Tbasket(Ptree ∪ Ptmug ∪ Ptbasket) ∪ Ptl ∪ Pbg Dh⋆ ← Dh⋆ ∪ RenderDepth(Pt⋆,Tth,Q⋆t) Dl⋆ ← Dl⋆ ∪ RenderDepth(Pt⋆,Ttl,Q⋆t) Dr⋆ ← Dr⋆ ∪ RenderDepth(Pt⋆,TbasketTtr,Q⋆t) Q⋆ ← Q⋆ ∪ Q⋆t, A⋆ ← A⋆ ∪ TbasketAt

Table 6. Training Details of 3D-controlled Video Generation Model.

[Figure 293]

T ⋆ ← T ⋆ ∪ (Tth,Ttl,TbaseketTtr) end for return D⋆, Q⋆, A⋆, T ⋆

edges tend to dominate the visual information, potentially diminishing the influence of other control signals during training [13]. However, these two conditions always introduce noise after spatial editing. To improve robustness against imperfect control signals, we apply random dropout to the depth and Canny edge conditions during training, where they are independently dropped with a probability of 0.5, and jointly dropped with a probability of 0.1. By randomly masking portions of these inputs, the model is encouraged to rely on complementary visual evidence, rather than depending solely on the intensity conditions, ultimately improving the realism of the generated videos under noisy conditions.

Figure 12. Visualization of Manipulation Workspace.

dataset. This setup enables a more reliable assessment of

- our framework’s generalization to unseen objects. In addition, the real-world testing laboratory is also absent from the training data used to develop our framework.

Data Generation. To generate demonstrations, we first reconstruct the source demonstrations and apply a confidence threshold between 30% and 50% to remove spuri-

- ous points. During spatial editing, we define an augmentation region around the object’s original location, typically a 40cm×40cm square, and augment object rotations within a 30°–60° range. For video generation, we set the diffusion step to 6 and the memory length to 4, for which we uniformly sample from the already generated frames, including both the first and last frames.

Training Details. In Table 6, we provide the details of training the 3D-controlled video generation model.

#### 9. Experiment Details

Workspace. Figure 12 shows the workspace of four realrobot manipulation tasks in the manuscript. The workspace is determined by the maximal range in which the robot’s kinematic configuration can perform the intended tasks.

Object Set. Figure 13 shows all the objects we used in the manipulation tasks. Because it is impractical to verify every object in the training set, all objects in the figure are newly purchased to minimize any potential overlap with the

[Figure 294]

###### R10 R20 R50 R1G200 R2G200 R5G200 0/20 9/20 11/20 13/20 14/20 17/20

Table 8. Performance of Diffusion Policy on the Mug to Basket task. R means the number of real demonstrations, and G means the number of generated demonstrations.

100

Methodology

Teleop

R2E2R (8 GPUs)

R2E2R (16 GPUs) R2E2R (32 GPUs) R2E2R (64 GPUs)

80

Figure 13. Visualization of Manipulation Objects.

SuccessRate(%)

60

#### 10. Additional Experiments

| |
|---|

##### 10.1. Quantitative Results of Video Generation

40

To evaluate the performance of our video generation model, we conduct experiments using data collected from the four real-world robot tasks to prevent data contamination. Table 7 shows the quantitative results compared to GESim [26] with the conditional I2V setting. It demonstrates that our video generation module produces robot demonstrations with significantly enhanced visual realism.

20

0

102 103 104

Time Consumption (Seconds)

Method FVD ↓ LPIPS ↓ SSIM ↑ PSNR ↑

Figure 14. Time Analysis of Real2Edit2Real. We report the success rate (%) relative to the total time consumption (seconds, logscale) for data generation.

GE-Sim [26] 663.4 0.2038 0.7491 20.41 Ours 352.9 0.1252 0.8647 22.95

Table 7. Quantitative Results of Video Generation. We compare our method with GE-Sim across several standard metrics on conditional I2V. Bold numbers indicate the best performance.

##### 10.2. Diffusion Policy on Mug to Basket

To further validate the quality of the data generated by Real2Edit2Real, we conduct additional Diffusion Policy [5] experiments on the Mug to Basket task. We use a ViT-S encoder initialized with DINO-v3 [41] weights and train the Diffusion Policy in a full-parameter manner on different training data. Table 8 shows the success rate of diffusion polices trained with real demonstrations and generated demonstrations, which indicates that generating data from only a few source demonstrations, like 1-5, can make DP surpass that trained with 50 real demonstrations on this task.

##### 10.3. Generation Time Analysis

- Figure 14 presents the generation time analysis of the Real2Edit2Real framework. While the integration of video generation modules introduces a computational bottleneck at lower GPU counts, our approach exhibits par-

allel scalability. By leveraging multi-GPU acceleration, the data generation time is significantly reduced, allowing Real2Edit2Real to surpass the success rates of manual teleoperation in a short time.

##### 10.4. Generation Data Scaling Analysis

In the manuscript, we investigate how increasing the number of source demonstrations affects policy performances. Here, we additionally examine, within our proposed Real2Edit2Real framework, the impact of generating more demonstrations. To this end, we produce varying numbers of demonstrations from a single source demonstration and evaluate the resulting policies, using the same training and evaluation protocols as in the manuscript. Experimental results are shown in Figure 15 and Table 9. The results indicate: (1) Both policies exhibit consistently improved success rates when scaling up generated demonstrations. (2) When we generate more than 300 demonstrations from only one demo, the average success rates surpass that of 50 real demonstrations.

Mug to Basket Pour Water Lift Box Scan Barcode Total Go-1 π0.5 Go-1 π0.5 Go-1 π0.5 Go-1 π0.5 Go-1 π0.5

# Demo

Real 10 8 / 20 8 / 20 5 / 20 1 / 20 11 / 20 13 / 20 5 / 20 4 / 20 36.3% 32.5% Real 20 12 / 20 14 / 20 7 / 20 2 / 20 12 / 20 15 / 20 8 / 20 5 / 20 48.8% 45.0% Real 50 14 / 20 13 / 20 8 / 20 8 / 20 15 / 20 17 / 20 12 / 20 11 / 20 61.3% 61.3%

Real 1 Gen 50 8 / 20 9 / 20 11 / 20 4 / 20 10 / 20 6 / 20 11 / 20 4 / 20 50.0% 28.8% Real 1 Gen 100 12 / 20 11 / 20 12 / 20 6 / 20 10 / 20 7 / 20 12 / 20 9 / 20 57.5% 41.3% Real 1 Gen 200 14 / 20 15 / 20 12 / 20 10 / 20 12 / 20 10 / 20 14 / 20 11 / 20 65.0% 57.5% Real 1 Gen 300 15 / 20 16 / 20 12 / 20 12 / 20 14 / 20 11 / 20 18 / 20 11 / 20 73.8% 62.5% Real 1 Gen 400 15 / 20 19 / 20 14 / 20 15 / 20 15 / 20 13 / 20 18 / 20 13 / 20 77.5% 75.0%

Table 9. Scaling Analysis of Generated Demonstrations. This compares the performance of polices trained with different numbers of demonstrations generated from only one source demonstration. We can see that increasing the number of generated demonstrations leads to improved success rates for both policies. When we generate more than 300 demonstrations from only one demo, the average success rates even surpass that of 50 real demonstrations.

Success Rate vs. Generated Demonstrations

1.0

0.8

SuccessRate

0.6

0.4

0.2

Go-1 0.5

0.0

50 100 150 200 250 300 350 400

Generated Demonstrations

- Figure 15. Scaling Analysis of Generated Demonstrations. Bold curves denote the task-averaged performance, while the faint translucent curves visualize the trajectories of individual tasks. Both policies exhibit consistently improved success rates when scaling up generated demonstrations.

##### 10.5. Ablation Study of Control Conditions

In the manuscript, we introduced our 3D-controlled video generation model, which uses depth as the 3D control interface and incorporates Canny edges computed from the depth map as an auxiliary condition. To investigate the roles of depth and Canny edges in video generation, we conduct qualitative ablation studies by removing each condition individually. Fig. 17, 18, 19, 20 show the results on four tasks, respectively. The results demonstrate that removing either the depth control or the Canny edge constraint leads to issues such as object blurring and incorrect interactions, which substantially degrade the quality of the generated demonstrations.

#### 11. Additional Visualizations

Figure 21, 22, 23, 24 show more visualizations of the four real-world manipulation tasks.

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Articulated Object

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Deformable Object

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Figure 16. Visualization of Failure Cases.

#### 12. Limitation and Discussion

Despite the advantages of our proposed Real2Edit2Real framework, which enables scalable multi-view demonstration augmentation, it still has certain limitations.

Video Generation Time. As analyzed in Section 10.3, the video generation module currently poses a computational bottleneck within our framework, particularly in resourceconstrained scenarios. Future research could explore the integration of acceleration techniques from the generative modeling community, such as KV caching and model distillation, to further enhance the throughput of our data generation pipeline.

Object Generalization. As illustrated in Figure 16, our generative model exhibits limitations in object generalization, particularly when handling articulated or deformable objects. This stems primarily from the lack of these object categories in our training distribution, which can lead to visual artifacts such as motion blurring or structural inconsistency during video synthesis. To mitigate this, future work could focus on scaling up the diversity and volume of training data to enhance the model’s robustness across a broader spectrum of object geometries and physical properties.

[Figure 331]

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

Full Model

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

w/o Depth

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

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

w/o

Canny

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

Figure 17. Ablation Study of Control Conditions on Mug to Basket.

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Full Model

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

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

w/o Depth

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

w/o

Canny

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

Figure 18. Ablation Study of Control Conditions on Pour Water.

[Figure 439]

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

[Figure 452]

Full Model

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

w/o Depth

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

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

[Figure 488]

w/o

Canny

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

Figure 19. Ablation Study of Control Conditions on Lift Box.

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

Full Model

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

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

[Figure 524]

[Figure 525]

w/o Depth

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

w/o

Canny

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

- Figure 20. Ablation Study of Control Conditions on Scan Barcode.

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

###### Figure 21. Visualization of Generated Videos on Mug to Basket.

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

###### Figure 22. Visualization of Generated Videos on Pour Water.

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

###### Figure 23. Visualization of Generated Videos on Lift Box.

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

###### Figure 24. Visualization of Generated Videos on Scan Barcode.

