arXiv:2505.21437v1[cs.GR]27May2025

# CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects

Huaijin Pi1 Zhi Cen2 Zhiyang Dou1 Taku Komura1 1The University of Hong Kong 2Zhejiang University

https://phj128.github.io/page/CoDA

[Figure 1]

[Figure 2]

[Figure 3]

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

(a) Text (``a person uses the mixer’’) to whole-body motion

(b) Keyframe object pose to whole-body motion

(c) Hand-only trajectories to whole-body motion

[Figure 7]

|[Figure 8]<br><br>①|
|---|

|[Figure 9]<br><br>③|
|---|

|[Figure 10]<br><br>②|
|---|

|[Figure 11]<br><br>④|
|---|

(d) Walking and opening the box simultaneously

- Figure 1: Our approach enables: (a) generating whole-body manipulation of articulated objects from text input (e.g., “a person uses the mixer”); (b) manipulating the object to a target pose and articulation (the blue object is the target pose); (c) synthesizing whole-body motion guided by trajectories from hand-only data; (d) generating motions involving simultaneous walking and object manipulation (e.g., opening a box while walking).

## Abstract

Synthesizing whole-body manipulation of articulated objects, including body motion, hand motion, and object motion, is a critical yet challenging task with broad applications in virtual humans and robotics. The core challenges are twofold. First, achieving realistic whole-body motion requires tight coordination between the hands and the rest of the body, as their movements are interdependent during manipulation. Second, articulated object manipulation typically involves high degrees of freedom and demands higher precision, often requiring the fingers to be placed at specific regions to actuate movable parts. To address these challenges, we propose a novel coordinated diffusion noise optimization framework. Specifically, we perform noise-space optimization over three specialized diffusion models for the body, left hand, and right hand, each trained on its own motion dataset to improve generalization. Coordination naturally emerges through gradient flow

Preprint. Under review.

along the human kinematic chain, allowing the global body posture to adapt in response to hand motion objectives with high fidelity. To further enhance precision in hand-object interaction, we adopt a unified representation based on basis point sets (BPS), where end-effector positions are encoded as distances to the same BPS used for object geometry. This unified representation captures fine-grained spatial relationships between the hand and articulated object parts, and the resulting trajectories serve as targets to guide the optimization of diffusion noise, producing highly accurate interaction motion. We conduct extensive experiments demonstrating that our method outperforms existing approaches in motion quality and physical plausibility, and enables various capabilities such as object pose control, simultaneous walking and manipulation, and whole-body generation from hand-only data. The code will be released for reproducibility.

## 1 Introduction

Human-object interaction (HOI) motion generation has broad applications in virtual reality, character animation, and robotics. These interactions range from simple activities like sitting on a chair [89, 79] to more complex tasks involving articulated object manipulation [6, 34], such as opening a box or a microwave. This paper focuses on the challenging setting of whole-body manipulation of articulated objects. Given an initial pose of the human and the object, along with a textual instruction, our goal is to synthesize realistic, physically plausible interaction sequences that involve coordinated body, hand, and articulated object motion.

Most prior works on HOI generation [26, 79, 121, 4, 46, 47, 13] suffer from two key limitations. First, they typically focus on either body-only motion [79, 46, 47] or hand-only manipulation [118, 132, 6, 34, 13]. Although hand-only methods can produce plausible contact behaviors in short-range scenarios, they fail to capture important whole-body dynamics such as bending down, reaching forward, or walking while manipulating objects. Such whole-body behaviors are essential for generating realistic human-object interactions, especially when manipulation is not restricted to a fixed space. Second, most existing works target rigid objects [115, 47, 13], while articulated objects introduce more complex motion patterns and require continuous in-hand adjustments.

Whole-body manipulation of articulated objects is highly challenging. First, it demands coordinated motion between the body and hands to reflect natural physical behaviors. Body movement affects how the hands approach and manipulate objects, and conversely, hand-object interactions can influence global posture. Second, precise control of finger positions is essential to maintain accurate, physically plausible contact throughout the sequence. This is especially important for articulated objects, where the manipulation often requires placing the fingers at specific regions to actuate the articulation while avoid colliding with other parts.

To address these challenges, we propose a novel framework called CoDA (Coordinated Diffusion noise optimization for whole-body manipulation of Articulated objects), which jointly synthesizes the motions of the human body, hands, and articulated objects. Our core idea is to optimize the input noise vectors of three specialized diffusion models, which independently model the body, left hand, and right hand, to generate coordinated whole-body motion. This decoupled design allows each component to be trained on its own data source, such as using large-scale datasets like AMASS

- [63] for body motion, manipulation datasets like ARCTIC [19] and GRAB [93] for hand motion, thereby improving generalization across diverse motions. Coordination naturally emerges during optimization, as gradients from hand motion objectives flow through the human kinematic chain, allowing the global posture to adapt in response to fine-grained hand motion. This optimization further enables precise control over hand-object contact, while the diffusion noise space [40] provides strong motion priors to preserve naturalness in the generated sequences.

To enable precise manipulation while accounting for object geometry and articulation, we adopt a basis point set (BPS) representation [81, 132] to encode both the object surface and end-effector trajectories in a unified form. Specifically, we represent the positions of the end-effectors, namely the wrists and fingertips, by their distances to the same BPS used for encoding the object geometry. The unified representation captures the relative spatial relationship between the hand and the object geometry as well as its articulation during complex manipulation tasks. The generated trajectories, based on this representation, provide a continuous target signal for optimizing whole-body motion.

We evaluate our approach on both the ARCTIC [19] dataset of articulated object manipulation and the GRAB [93] dataset of rigid object interactions. Our method achieves state-of-the-art performance on both benchmarks, outperforming existing approaches in motion quality and physical plausibility. Beyond benchmark evaluation, our framework enables several compelling capabilities, as illustrated in Figure 1. It supports object pose control at specific times, and coordinated whole-body behaviors involving simultaneous locomotion and manipulation, which are absent from the ARCTIC dataset. In addition, our framework allows us to leverage hand-only datasets [2] to generate whole-body motion, enabling broader data usage and generalization. To the best of our knowledge, this is the first work to jointly generate body, hand, and articulated object motions for whole-body manipulation tasks.

## 2 Related work

Human-object interaction. Human-object interaction (HOI) generation [89, 46, 14] has received increasing attention due to its potential to enable virtual humans to perform various actions in

#### 3D environments. Early works focus on generating static interactions such as sitting or lying on furniture [89, 26, 133, 137, 131], using either auto-regressive pipelines or whole-sequence generation [103, 105, 67, 1, 139]. Recent methods explore diffusion-based models [35, 79, 45, 4, 121, 38] and apply guidance techniques [16, 30] to improve human-scene contact quality. Beyond static objects, several works consider dynamic objects [115, 116] or generate human motion conditioned on given object trajectories [46, 15]. For example, OMOMO [46] proposes a two-stage framework that first generates wrist trajectories and then completes body motion accordingly. Other approaches [74, 115, 47, 17, 88, 117] jointly generate body and object motion, and incorporate contact-aware guidance into the diffusion process to improve the quality. Another line of research [27, 71, 113, 96, 72, 106] enables physically simulated characters to perform scene-level interactions by learning control policies through environment interaction. These methods mainly focus on navigation and interactions with large-scale objects such as furniture or obstacles. While generating plausible body motion, they ignore finger motion, which is crucial for fine-grained manipulation.

Hand-object interaction. ManipNet [126] synthesizes object manipulation given wrist and object trajectories, using multiple representations to model the hand-object relationship. GRIP [95] design a temporal hand-object spatial feature for stable grasping. Some works [140, 55] address the task of denoising noisy hand motion to recover clean interaction sequences. While these methods explore various representations for modeling hand-object spatial relationships, they rely on access to predefined wrist and object trajectories. Some works [138, 132, 128] explore settings where only the object trajectory is provided. CAMS [138] introduces a canonicalized representation to enable precise contact generation. [132, 128] generate manipulation by predicting contact maps as intermediate representations. Other works generate hand and object motion jointly, without relying on predefined trajectories. DiffH2O [13] applies grasp guidance to diffusion models for more coherent hand-object interactions.. Text2HOI [6] employs cascaded diffusion to iteratively refine the results. HOIGPT [34] leverages separate codebooks for hands and objects, and jointly predicts motion and text. Physics-based approaches [12, 127] generate grasping motions through reinforcement learning in simulated environments. Despite their differences, all these methods ignore the body context.

Whole-body interaction. Although there are several whole-body manipulation datasets [93, 19, 39, 38, 123, 62, 57], only a few works consider body and hand interaction simultaneously. [94, 110] assume the object is static and only synthesize approaching and grasp motion. IMoS [20] demonstrates full-body manipulation with given finger motion; it generates body motion auto-regressively and optimizes object trajectories by assuming a static hand-object contact frame. TOHO [49] synthesizes whole-body interactions using implicit representations [28], relying on the same contact assumption to recover object motion. DiffGrasp [135] generates whole-body motion conditioned on given object trajectories using diffusion models, and introduces hand-object guidance to improve interaction quality. HOIFHLI [111] employs LLM [70] to analyze the scene and plan motions for grasping and relocating rigid objects. Other works [107, 108, 122] employ physics-based tracking to mimic manipulation behaviors. More recent work extends this to synthesize humanoid grasping [3, 60, 51], but the generated motions remain unnatural and do not involve complex manipulation. All of the above methods focus exclusively on rigid object interaction and do not address articulated objects. Compared to rigid object interaction, articulated object manipulation is more complex, as it often requires placing the fingers at specific regions to actuate the articulation.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

“A person uses the box.”

Idle Object BPS

Initial Pose Text

[Figure 16]

[Figure 17]

Body Diffusion

Left Hand Diffusion

Right Hand Diffusion

[Figure 18]

Object Trajectory Diffusion

[Figure 19]

[Figure 20]

[Figure 21]

Moving Object BPS

Forward Kinematics

Object Trajectory

[Figure 22]

[Figure 23]

[Figure 24]

End-effector Trajectory Diffusion

[Figure 25]

[Figure 26]

[Figure 27]

End-effector BPS

End-effector Trajectory

Forward Backward

(b) Whole-body Motion Optimization

(a) Object and End-effector Generation

- Figure 2: Pipeline overview. (a) Given the initial human pose, object pose, and text, we first generate the articulated object trajectory and the corresponding end-effector trajectories via two conditional diffusion models. (b) We then optimize the latent noise inputs of three decoupled diffusion models by propagating gradients through the kinematic chain, guided by end-effector tracking, penetration, and regularization losses. Finally, we forward the optimized noise through the diffusion models to synthesize coherent whole-body motion aligned with the generated object motion.
- 3 Preliminary

In this section, we define the input and output in this paper. Given the initial pose of a human and an articulated object, along with a textual instruction, our goal is to generate a full sequence including the whole-body human motion (body and fingers) and the articulated object motion over time.

Object representations. The objects from the ARCTIC [19] dataset are two-part articulated objects with 7 degrees of freedom. We use So = {To,Ro,a} to indicate the object pose, where the object state So ∈ R7 consists of object translation To ∈ R3, object rotation Ro ∈ R3, and the angle of the rotational joint a ∈ R1 between the two parts of the object.

Motion representations. We use SMPL-X [73], which is a parametric human body model to represent the whole body, including the face and fingers. SMPL-X is a differentiable function that takes input shape, pose, and expression parameters and outputs a 3D mesh with 10,475 vertices and 20,908 triangles. The vertices are posed with linear blend skinning with a rigged skeleton which is learned from the data. As we focus on the body motion with two hands, we remove the face related parameters. Θ = {θ,t} is the pose parameters to drive the SMPL-X model, where θ ∈ R52×3 represents joint angles and t ∈ R3 is the root translation.

Text descriptions. In the ARCTIC [19] and the GRAB [93] dataset, each sequence is annotated with an action label. Following previous work [20, 6], we construct the text description using the template “A person <action> the <object>.”. For example, “A person uses the box.”.

- 4 Method

The overview of our pipeline is shown in Figure 2. We first generate the motion of the articulated object (Section 4.1), then predict the end-effector trajectories (Section 4.2), and finally synthesize the whole-body motion by optimizing the noise of decoupled diffusion models (Section 4.3).

### 4.1 Object motion generation

Given the initial object pose and the textual instruction, we train a diffusion model [97] to generate the object future trajectory. The input includes the CLIP [82] feature of the text, the initial object pose, and the object geometry embedding. We represent the object geometry using the normalized part-based BPS descriptor [132], which will be formally defined in Section 4.2 and Figure 3. The output is a sequence of object states over time.

### 4.2 End-effector trajectory generation

Given the generated object trajectory, we extract its geometry representation and combine it with the trajectory itself and the textual instruction as input to a diffusion model that predicts end-effector trajectories. Instead of directly predicting 3D joint coordinates [46], we design a distance-based representation that encodes end-effector positions in the same space as the object geometry.

Unified BPS-based representation for object and end-effectors. We first present the object geometry representation. Following previous work [132], we adopt the normalized part BPS [81] to represent the object geometry. Specifically, the object mesh is first normalized to the unit scale by dividing all vertex coordinates by the maximum distance from the object origin to any vertex. Then a pre-defined fixed set of basis points P ∈ RK×3, shared across all objects, are uniformly sampled within the unit sphere centered at the object origin. The BPS representation is computed as the distances from each basis point to the nearest vertex on each of the two rigid object parts, resulting in an object geometry vector O ∈ RK×2.

We then introduce end-effector BPS, a distance-based representation tailored for encoding the positions of end-effectors in the object coordinate system. The end-effectors include both wrists and fingertips, comprising a total of 12 joints (2 wrists and 10 fingertips). As shown in figure 3, at each frame, for each of the 12 end-effectors, we compute a K-dimensional vector of Euclidean distances to the basis points. We use the same pre-defined set of basis points P ∈ RK×3 in object geometry representation [132]. This results in a (12 × K)-dimensional end-effector BPS vector per frame. The diffusion model outputs a sequence of end-effector BPS over time, along with binary contact labels for each fingertip, indicating whether it is close to the object surface.

Given the generated end-effector BPS sequence, we recover the end-effector trajectories by solving a simple optimization problem. For each end-effector at each frame, we minimize the following loss to infer its 3D position:

[Figure 28]

[Figure 29]

p∗e = arg min

L(pe), (1)

pe

(a) Object BPS (b) End-effector BPS

∥∥pe − Pj∥2 − dj∥2, (2)

L(pe) =

Figure 3: The illustration of the endeffector BPS. (a) is the object BPS [132]. (b) is the proposed end-effector BPS representation. Gray points denote the basis points; pink/yellow are two object parts; blue indicates a fingertip. Only one end-effector and 64 basis points are visualized for simplicity.

j

where pe is the optimized 3D position and dj is the predicted distance to the j-th basis point Pj. By sharing the basis point set with the object BPS representation, our method provides a consistent spatial reference frame that facilitates geometric alignment between end-effectors and object parts.

RoPE-based object motion encoding. To better encode the object trajectory, we adapt the idea of CaPE [44], which encodes relative camera pose information via RoPE [92]. In our case, each object pose is also represented as a 4 × 4 transformation matrix. Inspired by CaPE [44], we use the object pose to transform the query and key features in each attention layer. This enables the model to encode the relative object motion within a local temporal window, providing temporally-aware conditioning for the generation. We refer readers to Section C for more details.

### 4.3 Whole-body motion generation

The goal of this stage is to generate coherent whole-body motion that aligns with the predicted end-effector trajectories and articulated object motion. Rather than directly predicting whole-body

poses conditioned on end-effectors [46], we adopt an optimization-based approach inspired by DNO [40]. Specifically, we optimize the noise input to the diffusion models (Figure 2 (b)), and then forward the optimized noise through the diffusion models to generate the final motion. To further improve motion quality, we decouple the body into three components: body, left hand, and right hand, and train separate diffusion models for each. This decoupled design enables us to train each module using individual data, such as training the hand models using the ARCTIC [19] and GRAB [93], and the body-only model without hands on the AMASS [63]. Such specialization improves generalization by allowing novel combinations of finger and body motion to be synthesized. Moreover, this formulation facilitates gradient flow through the kinematic chain during optimization, which improves coordination between the body and hands.

Decoupled motion diffusion model. We adopt a decoupled human representation for whole-body motion, dividing the human pose into three components: body, left hand, and right hand. Formally, for each frame i, the whole-body pose Θi is represented as:

x = {xb,xlh,xrh}, (3) xb = {r˙x,r˙z,ry,r˙a,θb}, (4) xlh = {θlh}, (5) xrh = {θrh}, (6)

where xb denotes the body component, including root velocities r˙x,r˙z ∈ R (projected on the XZplane), root height ry ∈ R, angular velocity r˙a ∈ R, and body joint rotations θb ∈ R6×J

b, while θlh ∈ R6×J

rh represent the left and right hand joint rotations, respectively. All joint rotations are encoded using the 6D representation [142], with Jb = 22, Jlh = Jrh = 15 joints for the body and each hand. We train three separate diffusion models, Mb, Mlh, and Mrh, to model the motion manifolds of the body and hands individually.

lh and θrh ∈ R6×J

Optimization over diffusion noise. Given the trained diffusion models for body, left hand, and right hand, we optimize the noise vectors z = {zb,zlh,zrh} to generate whole-body motion as shown in Figure 2 (b). Let f(z) denote the process that maps the input noise to global joint positions through diffusion models and forward kinematics:

f(z) = FK(Mb(zb),Mlh(zlh),Mrh(zrh)), (7)

where FK(·) converts root translation and local joint rotations into global joint positions. We formulate motion generation as minimizing a loss L over the diffusion noise:

z∗ = arg min

L(f(z)). (8)

z

The overall loss function consists of three components with different weights λee, λpen, and λreg:

L = λeeLee + λpenLpen + λregLreg, (9) where Lee, Lpen, and Lreg are the end-effector tracking, penetration, and regularization losses. We encourage the generated global fingertip positions pˆf to follow the predicted trajectories pf from the previous stage. We also constrain the relative fingertip positions to the wrist joints:

Lee = ∥pˆf − pf∥1 + ∥pˆrf − prf∥1, (10) where prf and pˆrf denote the relative fingertip positions with respect to the wrist. To reduce hand-object interpenetration, we penalize fingertip joints that fall inside the object mesh:

Lpen =

j

min SDF(Jj) − 0.01,0.00 1 (11)

where SDF(Jj) is the signed distance at the j-th hand joint, assuming 1cm finger thickness. We add a regularization term to discourage foot floating and foot sliding:

Lreg = ∥min(Jy) − 0.02∥1 + 1left. Jli − Jli−1 1 + 1right. Jri − Jri−1 1 , (12)

where Jy denotes the height of all joints in the body, and Jli and Jri denote the 3D positions of the left and right foot joints at frame i, respectively. The binary indicators 1 denote whether the left or

right foot is in contact with the ground, based on a height threshold of 0.02 meters.

We adopt DDIM [87] sampling to efficiently generate motion sequences during optimization following DNO [40]. The loss is computed on the final output, and gradients are propagated back through the DDIM solver to update the noise. After optimization, we pass the optimized noise into the decoupled diffusion models to generate the final whole-body motion. Combined with the previously generated object trajectory, this yields a complete human-object manipulation sequence. This noisespace optimization avoids high-dimensional pose regression, reduces artifacts, and produces natural whole-body motions aligned with the object manipulation process.

## 5 Experiments

### 5.1 Implementation details

We adopt a transformer-based diffusion architecture similar to MDM [97] for all models in our framework. During inference, we perform noise optimization using DDIM [87] with T=10 for 800 steps and a cosine-decayed learning rate, following the DNO [40] strategy. All experiments are conducted on a single NVIDIA A100 GPU. More training details are in Section D.

### 5.2 Dataset and evaluation metrics

Dataset. We evaluate on ARCTIC [19] for articulated object manipulation and on GRAB [93] for rigid object interaction. ARCTIC contains around 2 hours of motion data featuring 10 subjects interacting with 11 articulated objects, including complex motions such as bimanual grasps and in-hand manipulation. Following the protocol in [132], we randomly sample 4 sequences per object category to construct the test set. The GRAB dataset covers about 4 hours of interaction from 10 subjects with 51 rigid objects, focusing primarily on grasping and simple lifting actions. Similar to [20], we use data from the last subject as the test set. For training object motion and end-effector trajectories generation, ARCTIC is used for articulated objects, and GRAB is used for rigid objects. The body motion model is trained on ARCTIC, GRAB, and AMASS [63], while the two hand motion models are trained on ARCTIC and GRAB.

Evaluation metrics. Similar to [13, 6], we evaluate the motion quality using the following metrics: (1) Frechet Inception Distance (FID) measures the feature-level distance between generated and real motions, using a motion feature extractor trained on the dataset following [23]. (2) R-Precision quantifies the alignment between generated motion and the corresponding textual prompt, measured using Top-3 accuracy. (3) Diversity reflects the variation among generated motion samples. (4) Foot skating indicates motion realism by detecting undesired foot sliding, following the computation in [53, 79]. We additionally report physical realism metrics following [13]: (5) Interpenetration volume (IV) computes the number of hand vertices that penetrate the object mesh. (6) Interpenetration depth (ID) measures the maximum penetration depth of hand vertices into the object. (7) Contact ratio (CR) is defined as the average proportion of hand vertices within 5 mm of the object surface. We also conduct a user study involving 16 participants to evaluate the generated motion sequences.

### 5.3 Comparison with baselines

Baselines. As there is no existing method that jointly generates body, hand, and articulated object motion, we adapt several representative methods to our task: IMoS [20], MDM [97], OMOMO [46], Text2HOI [6], and CHOIS [47]. IMoS is a CVAE-based [86] auto-regressive model, while MDM is a full-sequence diffusion-based [31] model. Text2HOI is originally designed for hand-object interaction with multiple diffusion models for iterative refinement. CHOIS is a diffusion-based model that incorporates contact guidance during inference. We extend them to jointly generate whole-body motion and object motion. OMOMO first generates wrist motion and then synthesizes body motion. We extend it to a three-stage model: first generating object motion, then predicting fingertip and wrist trajectories, and finally producing whole-body motion.

Quantitative results. We report quantitative results on ARCTIC and GRAB in Table 1 and Table 3, and user study results in Table 2. Our method achieves the best performance on nearly all metrics across both datasets. While it ranks slightly lower in diversity, it significantly outperforms all baselines in the user study, indicating superior perceptual quality and physical plausibility.

Table 1: Comparison on the ARCTIC [19] dataset. The right arrow → means the closer to real motion the better. IV, ID, and CR denote interpenetration volume, interpenetration depth, and contact ratio. The best and second-best results are highlighted green and yellow, respectively.

Methods FID↓ R-Precision↑ Diversity→ Foot skating↓ IV↓ ID↓ CR↑ Real − 0.531 8.664 0.002 4.68 11.47 0.085 IMoS [20] 6.686 0.305 6.144 1.469 14.28 13.24 0.010 MDM [97] 3.972 0.209 8.167 0.027 16.90 15.85 0.033 Text2HOI [6] 6.654 0.234 5.923 0.028 12.72 17.14 0.010 OMOMO [46] 3.710 0.406 6.110 0.028 13.77 15.16 0.061 CHOIS [47] 3.758 0.367 7.423 0.023 17.19 15.84 0.030 Ours 2.283 0.477 7.208 0.002 5.25 12.87 0.086

### Table 2: User study on the ARCTIC [19] dataset.

Metrics Ours CHOIS [47] OMOMO [46] Text2HOI [6] Best Motion Realism Rate ↑ 88.7% 1.1% 9.9% 0.3%

Best Physical Plausibility Rate ↑ 87.3% 1.4% 10.2% 1.1%

Qualitative results. As demonstrated in Figure 4, our method achieves significantly better hand-object contact compared to baselines. We provide more results in the supplementary material.

### 5.4 Ablation study

We ablate key components of our framework to understand their impact on overall performance: (a) A single model to jointly predict object motion and end-effector trajectories. (b) Predicting relative coordinate of end-effectors to the object center without end-effector BPS. (c) Using object velocity and rotational velocity as the trajectory input without RoPE-based representation. (d) Removing the optimization process and using a conditional diffusion model with fingertip trajectories as input. (e) Using a single diffusion model for the entire body without the decoupled body-hand representation. (f) Excluding the AMASS [63] dataset during training the body motion model. As shown in Table 4, each component contributes to the performance improvement.

### 5.5 More discussions

Generalization to different object geometry. To further validate generalization to unseen object geometries of the same category, we train the object motion and end-effector trajectory models on the hand-only dataset [138], using 7 training and 3 testing objects. Despite the dataset containing only hand motion, our method successfully generates whole-body motion, as shown in Figure 5.

Various capabilities. Our approach enables various capabilities. First, it allows control over keyframe object poses by setting them as optimization targets for object trajectory generation (Figure 6). Second, it can synthesize whole-body motions that involve simultaneous locomotion and manipulation, even though such combinations are not present in the training dataset [19] (Figure 7). Third, it enables generating whole-body motion guided by hand-only datasets [2], using wrist and fingertip trajectories as optimization targets (Figure 9). Furthermore, our generated whole-body

### Table 3: Comparison on the GRAB [93] dataset.

Methods FID↓ R-Precision↑ Diversity→ Foot skating↓ IV↓ ID↓ CR↑ Real − 0.727 15.045 0.010 5.84 13.41 0.049 IMoS [20] 52.290 0.180 8.374 0.152 11.57 20.35 0.000 MDM [97] 26.734 0.289 8.627 0.109 12.96 16.03 0.001 Text2HOI [6] 30.101 0.320 10.302 0.086 12.52 14.55 0.000 OMOMO [46] 25.017 0.391 9.294 0.094 11.03 14.03 0.004 CHOIS [47] 25.835 0.320 9.887 0.055 9.31 14.37 0.002 Ours 21.544 0.438 9.387 0.046 4.93 10.23 0.040

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

|[Figure 38]|
|---|

[Figure 39]

|[Figure 40]|
|---|

[Figure 41]

OMOMO

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

CHOIS

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

|[Figure 65]|
|---|

Ours

Figure 4: Qualitative comparison. Given the text “A person uses the ketchup.”, our method generates the whole-body motion with better hand-object contact compared to baselines.

### Table 4: Ablation study on the ARCTIC [19] dataset.

Methods FID↓ R-Precision↑ Diversity→ Foot skating↓ IV↓ ID↓ CR↑ Real − 0.531 8.664 0.002 4.68 11.47 0.085

- (a) w/o separate models 3.790 0.438 6.939 0.002 8.21 13.16 0.103

- (b) w/o end-effector BPS 4.069 0.453 6.888 0.002 8.09 13.54 0.093
- (c) w/o RoPE motion 2.714 0.469 7.021 0.002 6.12 12.66 0.093

- (d) w/o optimization 4.883 0.414 6.406 0.030 16.39 16.13 0.095

- (e) w/o decoupled 2.699 0.438 7.142 0.008 12.45 16.29 0.082

- (f) w/o AMASS 3.305 0.453 6.859 0.003 5.46 13.04 0.089 Ours 2.283 0.477 7.208 0.002 5.25 12.87 0.086

motion can serve as a reference for controlling humanoids [75, 59, 107] in physics-based simulators

- [64], where the object is physically manipulated by the humanoid, rather than being directly assigned a target trajectory (Figure 8). We provide more qualitative results in Section E.

Limitations. First, the optimization process is slower than other generative methods [97], limiting real-time applications. Second, due to the limited object diversity in existing datasets [19], the model struggles to generalize to novel object categories. Third, our framework only focuses on single-object manipulation; extending it to handle multiple interacting objects or multi-step sequential interactions remains an open direction. Finally, enabling both the body and fingers to reason about and avoid obstacles in complex scenes, such as surrounding geometry or other objects, is still a difficult problem. More limitations are discussed in Section F.

## 6 Conclusion

In this paper, we present a coordinated diffusion noise optimization framework for synthesizing wholebody manipulation of articulated objects. By optimizing over the noise space of separately trained diffusion models for the body, left hand, and right hand, our method enables natural coordination between the body and hands. We introduce a unified distance-based representation built on basis point sets to generate end-effector trajectories, facilitating precise hand-object interactions. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in motion quality and physical plausibility. It also supports various capabilities such as object pose control, simultaneous manipulation and locomotion, and whole-body motion generation from hand-only data.

## References

- [1] Joao Pedro Araújo, Jiaman Li, Karthik Vetrivel, Rishi Agarwal, Jiajun Wu, Deepak Gopinath, Alexander William Clegg, and Karen Liu. Circle: Capture in rich contextual environments. In CVPR, 2023. 3
- [2] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. HOT3D: Hand and object tracking in 3D from egocentric multi-view videos. CVPR, 2025. 3, 8
- [3] Jona Braun, Sammy Christen, Muhammed Kocabas, Emre Aksan, and Otmar Hilliges. Physically plausible full-body hand-object interaction synthesis. In International Conference on 3D Vision (3DV), 2024. 3
- [4] Zhi Cen, Huaijin Pi, Sida Peng, Zehong Shen, Minghui Yang, Zhu Shuai, Hujun Bao, and Xiaowei Zhou. Generating human motion in 3d scenes from text descriptions. In CVPR, 2024. 2, 3
- [5] Zhi Cen, Huaijin Pi, Sida Peng, Qing Shuai, Yujun Shen, Hujun Bao, Xiaowei Zhou, and Ruizhen Hu. Ready-to-react: Online reaction policy for two-character interaction generation. In ICLR, 2025. 19
- [6] Junuk Cha, Jihyeon Kim, Jae Shin Yoon, and Seungryul Baek. Text2hoi: Text-guided 3d motion generation for hand-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1577–1585, 2024. 2, 3, 4, 7, 8, 24
- [7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022. 19
- [8] Changan Chen, Juze Zhang, Shrinidhi Kowshika Lakshmikanth, Yusu Fang, Ruizhi Shao, Gordon Wetzstein, Li Fei-Fei, and Ehsan Adeli. The language of motion: Unifying verbal and non-verbal language of 3d human motion. In arXiv, 2024. 19
- [9] Ling-Hao Chen, Shunlin Lu, Wenxun Dai, Zhiyang Dou, Xuan Ju, Jingbo Wang, Taku Komura, and Lei Zhang. Pay attention and move better: Harnessing attention for interactive motion generation and training-free editing. arXiv preprint arXiv:2410.18977, 2024. 19
- [10] Rui Chen, Mingyi Shi, Shaoli Huang, Ping Tan, Taku Komura, and Xuelin Chen. Taming diffusion probabilistic models for character control. In ACM SIGGRAPH 2024 Conference Papers, pages 1–10, 2024. 19
- [11] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, and Gang Yu. Executing your Commands via Motion Diffusion in Latent Space. arXiv e-prints, art. arXiv:2212.04048, December 2022. doi: 10.48550/arXiv.2212.04048. 19
- [12] Sammy Christen, Muhammed Kocabas, Emre Aksan, Jemin Hwangbo, Jie Song, and Otmar Hilliges. D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20577–20586, 2022. 3
- [13] Sammy Christen, Shreyas Hampali, Fadime Sener, Edoardo Remelli, Tomas Hodan, Eric Sauser, Shugao Ma, and Bugra Tekin. Diffh2o: Diffusion-based synthesis of hand-object interactions from textual descriptions. In SIGGRAPH Asia 2024 Conference Papers, 2024. 2, 3, 7
- [14] Peishan Cong, Ziyi Wang, Zhiyang Dou, Yiming Ren, Wei Yin, Kai Cheng, Yujing Sun, Xiaoxiao Long, Xinge Zhu, and Yuexin Ma. Laserhuman: Language-guided scene-aware human motion generation in free environment. arXiv preprint arXiv:2403.13307, 2024. 3
- [15] Peishan Cong, Ziyi Wang, Yuexin Ma, and Xiangyu Yue. Semgeomo: Dynamic contextual human motion generation with semantic and geometric guidance. arXiv preprint arXiv:2503.01291, 2025. 3

- [16] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 3, 19
- [17] Christian Diller and Angela Dai. Cg-hoi: Contact-guided 3d human-object interaction generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19888–19901, 2024. 3
- [18] Zhiyang Dou, Xuelin Chen, Qingnan Fan, Taku Komura, and Wenping Wang. C· ase: Learning conditional adversarial skill embeddings for physics-based characters. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023. 19
- [19] Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, and Otmar Hilliges. ARCTIC: A dataset for dexterous bimanual hand-object manipulation. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 3, 4, 6, 7, 8, 9, 21, 23
- [20] Anindita Ghosh, Rishabh Dabral, Vladislav Golyanik, Christian Theobalt, and Philipp Slusallek. Imos: Intent-driven full-body motion synthesis for human-object interactions. In Computer Graphics Forum, 2023. 3, 4, 7, 8
- [21] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Commun. ACM,

2020. 19

- [22] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 19
- [23] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In CVPR, 2022. 7, 19
- [24] Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. Momask: Generative masked modeling of 3d human motions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1900–1910, 2024. 19
- [25] Félix G Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. Robust motion in-betweening. ACM Trans. Graph., 2020. 19
- [26] Mohamed Hassan, Duygu Ceylan, Ruben Villegas, Jun Saito, Jimei Yang, Yi Zhou, and Michael J. Black. Stochastic scene-aware motion prediction. In ICCV, 2021. 2, 3, 19
- [27] Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, and Xue Bin Peng. Synthesizing physical character-scene interactions. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–9, 2023. 3
- [28] Chengan He, Jun Saito, James Zachary, Holly Rushmeier, and Yi Zhou. Nemf: Neural motion fields for kinematic animation. In NeurIPS, 2022. 3
- [29] Gustav Eje Henter, Simon Alexanderson, and Jonas Beskow. Moglow: Probabilistic and controllable motion synthesis using normalising flows. ACM Trans. Graph., 2020. 19
- [30] Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. arXiv e-prints, art. arXiv:2207.12598, July 2022. 3, 19
- [31] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 7, 19, 20
- [32] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video Diffusion Models. arXiv e-prints, art. arXiv:2204.03458, April 2022. 19
- [33] Daniel Holden, Taku Komura, and Jun Saito. Phase-functioned neural networks for character control. ACM Trans. Graph., 2017. 19

- [34] Mingzhen Huang, Fu-Jen Chu, Bugra Tekin, Kevin J Liang, Haoyu Ma, Weiyao Wang, Xingyu Chen, Pierre Gleize, Hongfei Xue, Siwei Lyu, Kris Kitani, Matt Feiszli, and Hao Tang. Hoigpt: Learning long sequence hand-object interaction with language models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, USA, 2025. 2, 3
- [35] Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, and Song-Chun Zhu. Diffusion-based generation, optimization, and planning in 3d scenes. In CVPR, 2023. 3
- [36] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991. doi: 10.1162/neco.1991.3. 1.79. 19
- [37] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems, 36:20067–20079,

2023. 19

- [38] Nan Jiang, Zimo He, Zi Wang, Hongjie Li, Yixin Chen, Siyuan Huang, and Yixin Zhu. Autonomous character-scene interaction synthesis from text instruction, 2024. URL https: //arxiv.org/abs/2410.03187. 3
- [39] Nan Jiang, Zhiyuan Zhang, Hongjie Li, Xiaoxuan Ma, Zan Wang, Yixin Chen, Tengyu Liu, Yixin Zhu, and Siyuan Huang. Scaling up dynamic human-scene interaction modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1737–1747, 2024. 3
- [40] Korrawe Karunratanakul, Konpat Preechakul, Emre Aksan, Thabo Beeler, Supasorn Suwajanakorn, and Siyu Tang. Optimizing diffusion noise can serve as universal motion priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1334–1345, 2024. 2, 6, 7, 19, 20
- [41] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv e-prints, 2014. 20
- [42] Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. arXiv e-prints, 2013. 19
- [43] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. NeurIPS, 2018. 19
- [44] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9503–9513, 2024. 5, 19
- [45] Nilesh Kulkarni, Davis Rempe, Kyle Genova, Abhijit Kundu, Justin Johnson, David Fouhey, and Leonidas Guibas. Nifty: Neural object interaction fields for guided human motion synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 947–957, 2024. 3
- [46] Jiaman Li, Jiajun Wu, and C Karen Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023. 2, 3, 5, 6, 7, 8
- [47] Jiaman Li, Alexander Clegg, Roozbeh Mottaghi, Jiajun Wu, Xavier Puig, and C Karen Liu. Controllable human-object interaction synthesis. In European Conference on Computer Vision, pages 54–72. Springer, 2024. 2, 3, 7, 8, 23
- [48] Peizhuo Li, Kfir Aberman, Zihan Zhang, Rana Hanocka, and Olga Sorkine-Hornung. Ganimator: Neural motion synthesis from a single sequence. ACM Trans. Graph., 2022. 19
- [49] Quanzhou Li, Jingbo Wang, Chen Change Loy, and Bo Dai. Task-oriented human-object interactions generation with implicit neural representations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3035–3044, 2024. 3

- [50] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13401–13412, 2021. 19
- [51] Yitang Li, Mingxian Lin, Zhuo Lin, Yipeng Deng, Yue Cao, and Li Yi. Learning physicsbased full-body human reaching and grasping from brief walking references. arXiv preprint arXiv:2503.07481, 2025. 3
- [52] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36, 2024. 19
- [53] Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne. Character controllers using motion vaes. ACM Trans. Graph., 2020. 7
- [54] Hanchao Liu, Xiaohang Zhan, Shaoli Huang, Tai-Jiang Mu, and Ying Shan. Programmable motion generation for open-set motion control tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1399–1408, 2024. 19
- [55] Xueyi Liu and Li Yi. Geneoh diffusion: Towards generalizable hand-object interaction denoising via denoising diffusion. In The Twelfth International Conference on Learning Representations, 2024. 3
- [56] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum? id=Bkg6RiCqY7. 20
- [57] Jintao Lu, He Zhang, Yuting Ye, Takaaki Shiratori, Sebastian Starke, and Taku Komura. Choice: Coordinated human-object interaction in cluttered environments for pick-and-place actions. arXiv preprint arXiv:2412.06702, 2024. 3
- [58] Shunlin Lu, Jingbo Wang, Zeyu Lu, Ling-Hao Chen, Wenxun Dai, Junting Dong, Zhiyang Dou, Bo Dai, and Ruimao Zhang. Scamo: Exploring the scaling law in autoregressive motion generation model. arXiv preprint arXiv:2412.14559, 2024. 19
- [59] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023. 9, 19, 22
- [60] Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris M. Kitani, and Weipeng Xu. Omnigrasp: Simulated humanoid grasping on diverse objects. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=Glt37xoU7e. 3, 24
- [61] Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris M. Kitani, and Weipeng Xu. Universal humanoid motion representations for physics-based control. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=OrOd8PxOO2. 19
- [62] Xintao Lv, Liang Xu, Yichao Yan, Xin Jin, Congsheng Xu, Shuwen Wu, Yifan Liu, Lincheng Li, Mengxiao Bi, Wenjun Zeng, and Xiaokang Yang. Himo: A new benchmark for fullbody human interacting with multiple objects, 2024. URL https://arxiv.org/abs/2407.

12371. 3

- [63] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In ICCV, 2019. 2, 6, 7, 8, 19, 21
- [64] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021. 9, 19, 23
- [65] Julieta Martinez, Michael J. Black, and Javier Romero. On human motion prediction using recurrent neural networks. In CVPR, 2017. 19

- [66] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=aBsCjcPu_tE. 19
- [67] Aymen Mir, Xavier Puig, Angjoo Kanazawa, and Gerard Pons-Moll. Generating continual human motion in diverse 3d scenes. In 2024 International Conference on 3D Vision (3DV), pages 903–913. IEEE, 2024. 3
- [68] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 20
- [69] Zachary Novack, Julian McAuley, Taylor Berg-Kirkpatrick, and Nicholas J Bryan. Ditto: Diffusion inference-time t-optimization for music generation. arXiv preprint arXiv:2401.12179,

2024. 19

- [70] OpenAI. Openai: Introducing chatgpt. https://openai.com/blog/chatgpt, 2022. 3, 19
- [71] Liang Pan, Jingbo Wang, Buzhen Huang, Junyu Zhang, Haofan Wang, Xu Tang, and Yangang Wang. Synthesizing physically plausible human motions in 3d scenes. In 2024 International Conference on 3D Vision (3DV), pages 1498–1507. IEEE, 2024. 3
- [72] Liang Pan, Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura, and Jingbo Wang. Tokenhsi: Unified synthesis of physical human-scene interactions through task tokenization. In CVPR, 2025. 3
- [73] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, 2019. 4
- [74] Xiaogang Peng, Yiming Xie, Zizhao Wu, Varun Jampani, Deqing Sun, and Huaizu Jiang. Hoi-diff: Text-driven synthesis of 3d human-object interactions using diffusion models. arXiv preprint arXiv:2312.06553, 2023. 3
- [75] Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne. Deepmimic: Exampleguided deep reinforcement learning of physics-based character skills. ACM Transactions On Graphics (TOG), 37(4):1–14, 2018. 9, 19, 22
- [76] Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. ACM Transactions on Graphics (ToG), 40(4):1–20, 2021. 19
- [77] Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions On Graphics (TOG), 41(4):1–17, 2022. 19
- [78] Mathis Petrovich, Michael J. Black, and Gül Varol. Action-conditioned 3D human motion synthesis with transformer VAE. In ICCV, 2021. 19
- [79] Huaijin Pi, Sida Peng, Minghui Yang, Xiaowei Zhou, and Hujun Bao. Hierarchical generation of human-object interactions with diffusion probabilistic models. In ICCV, 2023. 2, 3, 7
- [80] Matthias Plappert, Christian Mandery, and Tamim Asfour. The kit motion-language dataset. Big data, 2016. 19
- [81] Sergey Prokudin, Christoph Lassner, and Javier Romero. Efficient learning on point clouds with basis point sets. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4332–4341, 2019. 2, 5
- [82] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 5
- [83] Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. In NeurIPS, 2019. 19

- [84] Davis Rempe, Zhengyi Luo, Xue Bin Peng, Ye Yuan, Kris Kitani, Karsten Kreis, Sanja Fidler, and Or Litany. Trace and pace: Controllable pedestrian animation via guided trajectory diffusion. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 19
- [85] Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng, Zechen Hu, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou. World-grounded human motion recovery via gravity-view coordinates. In SIGGRAPH Asia Conference Proceedings, 2024. 20
- [86] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. In NeurIPS, 2015. 7
- [87] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=St1giarCHLP. 7, 20
- [88] Wenfeng Song, Xinyu Zhang, Shuai Li, Yang Gao, Aimin Hao, Xia Hou, Chenglizhao Chen, Ning Li, and Hong Qin. Hoianimator: Generating text-prompt human-object animations using novel perceptive diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 811–820, June 2024. 3
- [89] Sebastian Starke, He Zhang, Taku Komura, and Jun Saito. Neural state machine for characterscene interactions. ACM Trans. Graph., 2019. 2, 3, 19
- [90] Sebastian Starke, Yiwei Zhao, Taku Komura, and Kazi Zaman. Local motion phases for learning multi-contact character movements. ACM Trans. Graph., 2020.
- [91] Sebastian Starke, Ian Mason, and Taku Komura. Deepphase: Periodic autoencoders for learning motion phase manifolds. ACM Trans. Graph., 2022. 19
- [92] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 5
- [93] Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas. Grab: A dataset of whole-body human grasping of objects. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 581–600. Springer, 2020. 2, 3, 4, 6, 7, 8
- [94] Omid Taheri, Vasileios Choutas, Michael J. Black, and Dimitrios Tzionas. Goal: Generating 4d whole-body motion for hand-object grasping. In CVPR, 2022. 3
- [95] Omid Taheri, Yi Zhou, Dimitrios Tzionas, Yang Zhou, Duygu Ceylan, Soren Pirk, and Michael J. Black. GRIP: Generating interaction poses using latent consistency and spatial cues. In International Conference on 3D Vision (3DV), 2024. URL https://grip.is.tue. mpg.de. 3
- [96] Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic: Unified physics-based character control through masked motion inpainting. ACM Transactions on Graphics (TOG), 43(6):1–21, 2024. 3
- [97] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In ICLR, 2023. 5, 7, 8, 9, 19, 20, 23
- [98] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In CVPR, 2023. 19
- [99] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 19, 20
- [100] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7280–7290, 2023. 19
- [101] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22532–22541, 2023. 19

- [102] Weilin Wan, Zhiyang Dou, Taku Komura, Wenping Wang, Dinesh Jayaraman, and Lingjie Liu. Tlcontrol: Trajectory and language control for human motion synthesis. In ECCV 2024, pages 37–54. Springer Nature Switzerland, 2024. 19
- [103] Jiashun Wang, Huazhe Xu, Jingwei Xu, Sifei Liu, and Xiaolong Wang. Synthesizing long-term 3d human motion and interaction in 3d scenes. In CVPR, 2021. 3
- [104] Jiashun Wang, Jessica Hodgins, and Jungdam Won. Strategy and skill learning for physicsbased table tennis animation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 19
- [105] Jingbo Wang, Yu Rong, Jingyuan Liu, Sijie Yan, Dahua Lin, and Bo Dai. Towards diverse and natural scene-aware 3d human motion synthesis. In CVPR, 2022. 3
- [106] Wenjia Wang, Liang Pan, Zhiyang Dou, Zhouyingcheng Liao, Yuke Lou, Lei Yang, Jingbo Wang, and Taku Komura. Sims: Simulating human-scene interactions with real world script planning. arXiv preprint arXiv:2411.19921, 2024. 3
- [107] Yinhuai Wang, Jing Lin, Ailing Zeng, Zhengyi Luo, Jian Zhang, and Lei Zhang. Physhoi: Physics-based imitation of dynamic human-object interaction. arXiv preprint arXiv:2312.04393, 2023. 3, 9, 22
- [108] Yinhuai Wang, Qihan Zhao, Runyi Yu, Ailing Zeng, Jing Lin, Zhengyi Luo, Hok Wai Tsui, Jiwen Yu, Xiu Li, Qifeng Chen, et al. Skillmimic: Learning reusable basketball skills from demonstrations. arXiv preprint arXiv:2408.15270, 2024. 3
- [109] Jungdam Won, Deepak Gopinath, and Jessica Hodgins. Physics-based character controllers using conditional vaes. ACM Trans. Graph., 41(4), July 2022. ISSN 0730-0301. doi: 10.1145/3528223.3530067. URL https://doi.org/10.1145/3528223.3530067. 19
- [110] Yan Wu, Jiahao Wang, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, and Siyu Tang. Saga: Stochastic whole-body grasping with contact. In ECCV, 2022. 3
- [111] Zhen Wu, Jiaman Li, Pei Xu, and C Karen Liu. Human-object interaction from human-level instructions. arXiv preprint arXiv:2406.17840, 2024. 3
- [112] Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, and Jingbo Wang. Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space. arXiv preprint arXiv:2503.15451,

2025. 19

- [113] Zeqi Xiao, Tai Wang, Jingbo Wang, Jinkun Cao, Wenwei Zhang, Bo Dai, Dahua Lin, and Jiangmiao Pang. Unified human-scene interaction via prompted chain-of-contacts. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=1vCnDyQkjg. 3
- [114] Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. In The Twelfth International Conference on Learning Representations, 2024. 19
- [115] Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui. Interdiff: Generating 3d human-object interactions with physics-informed diffusion. In ICCV, 2023. 2, 3
- [116] Sirui Xu, Yu-Xiong Wang, Liangyan Gui, et al. Interdreamer: Zero-shot text to 3d dynamic human-object interaction. Advances in Neural Information Processing Systems, 37:52858– 52890, 2024. 3
- [117] Mengqing Xue, Yifei Liu, Ling Guo, Shaoli Huang, and Changxing Ding. Guiding humanobject interactions with rich geometry and relations. arXiv preprint arXiv:2503.20172, 2025. 3
- [118] Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, and Cewu Lu. OakInk: A large-scale knowledge repository for understanding hand-object interaction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2

- [119] Heyuan Yao, Zhenhua Song, Baoquan Chen, and Libin Liu. Controlvae: Model-based learning of generative controllers for physics-based characters. ACM Transactions on Graphics (TOG), 41(6):1–16, 2022. 19
- [120] Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, and Libin Liu. Moconvq: Unified physics-based motion control via scalable discrete representations. ACM Transactions on Graphics (TOG), 43(4):1–21, 2024. 19
- [121] Hongwei Yi, Justus Thies, Michael J Black, Xue Bin Peng, and Davis Rempe. Generating human interaction motions in scenes with text control. In European Conference on Computer Vision, pages 246–263. Springer, 2025. 2, 3
- [122] Runyi Yu, Yinhuai Wang, Qihan Zhao, Hok Wai Tsui, Jingbo Wang, Ping Tan, and Qifeng Chen. Skillmimic-v2: Learning robust and generalizable interaction skills from sparse and noisy demonstrations. arXiv preprint arXiv:2505.02094, 2025. 3
- [123] Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu. Oakink2: A dataset of bimanual hands-object manipulation in complex task completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 445–456, June 2024. 3
- [124] Haotian Zhang, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, and Kayvon Fatahalian. Learning physically simulated tennis skills from broadcast videos. ACM Trans. Graph., 2023. 19
- [125] He Zhang, Sebastian Starke, Taku Komura, and Jun Saito. Mode-adaptive neural networks for quadruped motion control. ACM Trans. Graph., 2018. 19
- [126] He Zhang, Yuting Ye, Takaaki Shiratori, and Taku Komura. Manipnet: Neural manipulation synthesis with a hand-object spatial representation. ACM Trans. Graph., 2021. 3
- [127] Hui Zhang, Sammy Christen, Zicong Fan, Luocheng Zheng, Jemin Hwangbo, Jie Song, and Otmar Hilliges. Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation. In 2024 International Conference on 3D Vision (3DV), pages 235–246. IEEE,

2024. 3

- [128] Jiajun Zhang, Yuxiang Zhang, Liang An, Mengcheng Li, Hongwen Zhang, Zonghai Hu, and Yebin Liu. Manidext: Hand-object manipulation synthesis via continuous correspondence embeddings and residual-guided diffusion. arXiv preprint arXiv:2409.09300, 2024. 3, 21
- [129] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In CVPR, 2023. 19
- [130] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. arXiv preprint, 2022. 19
- [131] Wanyue Zhang, Rishabh Dabral, Thomas Leimkühler, Vladislav Golyanik, Marc Habermann, and Christian Theobalt. Roam: Robust and object-aware motion generation using neural pose descriptors. In 2024 International Conference on 3D Vision (3DV), pages 1392–1402. IEEE,

2024. 3

- [132] Wanyue Zhang, Rishabh Dabral, Vladislav Golyanik, Vasileios Choutas, Eduardo Alvarado, Thabo Beeler, Marc Habermann, and Christian Theobalt. Bimart: A unified approach for the synthesis of 3d bimanual interaction with articulated objects. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3, 5, 7, 21
- [133] Xiaohan Zhang, Bharat Lal Bhatnagar, Sebastian Starke, Vladimir Guzov, and Gerard PonsMoll. Couch: Towards controllable human-chair interactions. In ECCV, 2022. 3
- [134] Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, and Wanli Ouyang. Motiongpt: Finetuned llms are general-purpose motion generators. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7368–7376, 2024. 19

- [135] Yonghao Zhang, Qiang He, Yanguang Wan, Yinda Zhang, Xiaoming Deng, Cuixia Ma, and Hongan Wang. Diffgrasp: Whole-body grasping synthesis guided by object motion using a diffusion model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 10320–10328, 2025. 3
- [136] Zeyu Zhang, Akide Liu, Ian Reid, Richard Hartley, Bohan Zhuang, and Hao Tang. Motion mamba: Efficient and long sequence motion generation. In European Conference on Computer Vision, pages 265–282. Springer, 2025. 19
- [137] Kaifeng Zhao, Yan Zhang, Shaofei Wang, Thabo Beeler, and Siyu Tang. Synthesizing diverse human motions in 3d indoor scenes. In ICCV, 2023. 3
- [138] Juntian Zheng, Qingyuan Zheng, Lixing Fang, Yun Liu, and Li Yi. Cams: Canonicalized manipulation spaces for category-level functional hand-object manipulation synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 585–594, 2023. 3, 8, 21
- [139] Yang Zheng, Yanchao Yang, Kaichun Mo, Jiaman Li, Tao Yu, Yebin Liu, C Karen Liu, and Leonidas J Guibas. Gimo: Gaze-informed human motion prediction in context. In ECCV,

2022. 3

- [140] Keyang Zhou, Bharat Lal Bhatnagar, Jan Eric Lenssen, and Gerard Pons-Moll. Toch: Spatiotemporal object-to-hand correspondence for motion refinement. In European Conference on Computer Vision, pages 1–19. Springer, 2022. 3
- [141] Wenyang Zhou, Zhiyang Dou#, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. Emdm: Efficient motion diffusion model for fast and high-quality motion generation. In ECCV 2024, pages 18–38. Springer Nature Switzerland, 2024. 19
- [142] Yi Zhou, Connelly Barnes, Lu Jingwan, Yang Jimei, and Li Hao. On the continuity of rotation representations in neural networks. In CVPR, 2019. 6
- [143] Qingxu Zhu, He Zhang, Mengting Lan, and Lei Han. Neural categorical priors for physicsbased character control. ACM Trans. Graph., 42(6), dec 2023. ISSN 0730-0301. doi: 10.1145/3618397. URL https://doi.org/10.1145/3618397. 19

## A Introduction

This supplementary document provides additional details and analysis of our proposed method. In Section B, we provide a more comprehensive review of related work. Section C includes more details about our method. In Section D, we describe implementation details and experimental settings. Section E provides deeper analysis of the generated motions and showcases various capabilities enabled by our method. Finally, Section F discusses more limitations and broader impacts of this work.

We provide a supplementary webpage that presents high-resolution visualizations of all generated motion sequences.

## B Additional Related Work

Motion generation. Human motion generation is a long-standing research problem [33, 125, 89– 91, 23, 97, 10, 102, 141]. Recent approaches adopt a wide range of neural architectures, including Mixture of Experts (MoE) [36, 125], recurrent neural network [65], transformer [99, 25, 78], and Mamba [22, 136]. To enhance motion diversity and realism, various generative paradigms have been explored, such as generative adversarial networks [21, 48], normalizing flow [43, 29], variational autoencoder [42, 78, 26], VQ-VAE [83, 129, 37, 134, 58], diffusion models [31, 97, 130, 11, 114, 112, 5] and mask modeling [7, 24]. With the availability of the large-scale datasets [80, 63, 50, 23, 52], motion generation has been conditioned on diverse modalities such as text [129, 24, 9], music [98], and audio [8]. In parallel, physics-based methods [75, 76, 59, 84] have enabled simulated humanoids [64] to perform various motor skills [124, 143, 104] through reinforcement learning. Several recent works [77, 109, 119, 18, 120, 61] learn latent representations of human motion that support skill reuse.

Diffusion noise optimization. Diffusion models [31] have shown great success in various generative tasks [32]. To better control the generation process, classifier guidance [16] and classifier-free guidance [30] have been proposed. SDEdit [66] enables image editing by injecting noise into the reverse stochastic differential equation (SDE) process. DOODL [100] introduces an optimizationbased approach that directly updates the input noise of diffusion models by leveraging the invertible ordinary differential equation (ODE) [101]. This framework has been extended to other domains such as music generation [69]. In the motion domain, DNO [40] applies this idea for body-only motion editing, while ProgMoGen [54] uses LLMs [70] to select constraints and performs noise-space optimization for open-set motion control. However, these works are limited to body motion. In contrast, our work tackles the more complex setting of whole-body manipulation of articulated objects. We perform coordinated optimization over three diffusion models specialized for the body, left hand, and right hand, enabling coherent and physically plausible motion across the whole body.

## C Method Details

We adopt a RoPE-based encoding scheme for object motion, inspired by 6-DoF CaPE [44], to effectively capture the temporal dynamics of objects.

Given the object pose at i-th frame, we could use Pi to denote its 4 × 4 transformation matrix:

Ri ti 0 1

, (13)

Pi =

where Ri is the rotation matrix and ti is the translation vector. The relative position embedding function π(v,P) should statisfy the following properties:

⟨π (v1,P1),π (v2,P2)⟩ = π v1,P−2 1P1 ,π (v2,I) . (14) where v is the position embedding. Under this constraint, the attention between two transformed features can be equivalently rewritten as:

ϕ P−2 1P1 v1 ⊤ (ϕ(I)v2) = v1⊤ϕ P⊤1 P−⊤2 v2 (15) = v1⊤ϕ P⊤1 ϕ P−⊤2 v2 = (ϕ(P1)v1)⊤ ϕ P−⊤2 v2 (16) = π (v1,ϕ(P1)),π v2,ϕ P−⊤2 . (17)

Therefore, the relative embedding function can be implemented as π(v,P) = ϕ(P)v, where the transformation ϕ(P) is defined as:





Ψ 0 ··· 0 0 Ψ 0 . . 0

P if key P−⊤ if query

(18)

,Ψ =

ϕ(P) =

 

 

... 0 0 ··· 0 Ψ

Here, the embedding dimension is assumed to be divisible by 4, and Ψ is either P or P−⊤ depending on whether the input is a key or query.

This formulation allows each frame’s object pose to attend to others within a temporal window using relative transformations, enabling more expressive modeling of object trajectories compared to simple velocity inputs. Following [85], we restrict the attention window to 120 neighboring frames during training and inference.

## D Implementation Details

### D.1 Training details

Model architectures. We adopt a transformer-based diffusion architecture [99, 31], similar to MDM [97], for all models in our framework. All diffusion models are 8 transformer encoder layers, with 4 attention heads and 512 hidden units, and the feed-forward layer has 1024 hidden units.

Network training. For object motion and end-effector trajectories diffusion models, we train them for 500 epochs with batchsize 32 and AdamW [56] optimizer, with a learning rate 1 × 10−4. For body motion and hand motion diffusion models, we train them for 2000 epochs with batchsize 128 and AdamW optimizer, with a learning rate 1 × 10−4. All diffusion models use 1000 sampling steps during training [31], with a variance schedule increasing from β1 = 0.0001 to βT = 0.02 using the cosine schedule [68]. At inference time, we accelerate sampling for object motion and fingertip distance generation using DDIM [87] with T = 100. All our experiments are conducted on a single NVIDIA A100 GPU.

### D.2 Optimization details

End-effector trajectories generation. Given the generated end-effector BPS, we use the Adam [41] optimizer with a cosine-decayed learning rate 0.05 for 800 steps to calculate the end-effector trajectories.

Whole-body motion generation. During inference, we perform noise optimization using DDIM [87] with T=10 for 800 steps and a cosine-decayed learning rate 0.05, following the DNO [40] strategy. The optimization loss with different weights λee, λpen, and λreg is as follows:

L = λeeLee + λpenLpen + λregLreg, (19)

where Lee, Lpen, and Lreg are the end-effector tracking, penetration, and regularization losses. For first 300 steps, we set fingertip part as zero and only use the wrist targets, as the body motion might

be very different from the generated end-effector trajectories, with λee = 1, λpen = 0, λreg = 0. For 300 to 500 steps, we optimize with all end-effectors, and set λee = 1, λpen = 0, λreg = 0. For 500 to 800 steps, we enable the penetration loss and regularization loss, and set λee = 1, λpen = 5.0, λreg = 1.0.

### D.3 Ablation study implementation details

We present more details about the implementation of each variant in the ablation study. (a) A single model to jointly predict object motion and end-effector trajectories. In our proposed pipeline, we utilize different models for object motion and end-effector trajectories. Instead, we train a variant where a single model predicts both object motion and end-effector trajectories. The difference is that in our pipeline, the input of the end-effector trajectory model includes the frame-wise object

geometry embedding, as the object geometry of articulated objects is dynamic and changes over time. (b) Predicting the relative coordinate of end-effectors to the object center without end-effector BPS. In this variant, we directly predict the coordinate of end-effectors in the object coordinate system. This variant validates our design of using end-effector BPS to represent the end-effector trajectories. (c) Using object velocity and rotational velocity as the trajectory input without RoPEbased representation. This variant is to validate the effectiveness of the RoPE-based representation of object trajectories, which could capture the relative object trajectories in a local temporal window of each frame. (d) Removing the optimization process and using a conditional diffusion model with fingertip trajectories as input. This variant is to validate the effectiveness of the optimization process, which could generate more realistic whole-body motion. (e) Using a single diffusion model for the entire body without the decoupled body-hand representation. This variant is to validate the effectiveness of the decoupled representation. (f) Excluding the AMASS [63] dataset during training the body motion model. This variant validates the effectiveness of including more body data (the AMASS [63] dataset) for training the body motion model.

## E More Analysis

### E.1 Generalization to different object geometry

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

[Figure 69]

[Figure 70]

[Figure 71]

- Figure 5: Generalization to different object geometry. We train the object motion and end-effector trajectory models on hand-only data [138] with different object geometries. These models are integrated into our framework to provide optimization targets, enabling realistic whole-body motion synthesis for unseen object geometry.

To validate the generalization to different object geometries, we train the object motion and endeffector trajectory models on the hand-only dataset [138]. Following previous work [138, 128, 132], we use 7 training and 3 testing objects. Thanks to our multi-stage design, the optimization only relies on the object motion and end-effector trajectories. Therefore, our method could generate whole-body motion, as shown in Figure 5.

### E.2 Object motion control

To enable controllable object motion, we apply diffusion noise optimization to the object motion generation model by specifying keyframe object poses as targets. As shown in Figure 6, our method successfully generates manipulation sequences where the object is guided to reach the desired poses, producing plausible whole-body interactions.

### E.3 Simultaneous locomotion and manipulation

We demonstrate simultaneous locomotion and manipulation in Figure 7. Starting from the generated object motion, we manually add horizontal translation to simulate object movement in different directions. This translation is then set as the root position target in the optimization process. Our method successfully produces whole-body motions that combine manipulation with walking forward, backward, left, and right. Notably, such combinations are not present in the ARCTIC [19] dataset, which only features manipulation while standing still.

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 6: Object motion control. Our method could generate coherent whole-body motion with the object motion keyframe. The blue object indicates the object motion keyframe.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

a) Walk forward b) Walk backward

c) Walk to the right d) Walk to the left

- Figure 7: Simultaneous locomotion and manipulation. Our method enables the human to manipulate objects while simultaneously a) walking forward, b) walking backward, c) walking to the right, and d) walking to the left. The transparency of the meshes indicates time progression, where more transparent frames correspond to earlier frames.

### E.4 Deployment on simulated humanoids

As shown in Figure 8, our generated whole-body motion can serve as a reference for controlling humanoids in physics-based simulators. We apply physical motion tracking methods [75, 59, 107] to track the synthesized motions. The humanoid is able to physically interact with objects and perform coordinated manipulation behaviors in the simulated environment.

### E.5 Generating whole-body motion from hand-only dataset

We demonstrate that our framework can generate whole-body motion from hand-only datasets. Specifically, we use the object trajectories provided by the dataset, and extract the end-effector trajectories (fingertips and wrists) as optimization targets. Our method only requires 3D positions of the fingertips and wrists, without the need for full joint rotations or detailed hand mappings, making it easier to apply. As shown in Figure 9, our method successfully produces realistic whole-body motions aligned with the provided hand-object interactions.

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

- Figure 8: Deployment on simulated humanoids. We apply existing motion tracking techniques to deploy the generated motion to a simulated humanoid. The articulated object is physically manipulated by the humanoid within the physics simulator [64].

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

- Figure 9: Generating whole-body motion from hand-only dataset. We use the fingertip and object trajectories from the dataset and assign them as the optimization targets. After the optimization, we could get the whole-body motion.

### E.6 Inference speed

Table 5 presents the inference time of each module in our pipeline. All results are measured on a single NVIDIA A100 GPU, generating a motion sequence of 300 frames. The majority of the time cost comes from the whole-body motion optimization process, which involves iterative diffusion sampling and gradient-based updates. Although slower than feed-forward models [47], this process enables high-quality, physically plausible whole-body interactions.

## F More Discussions

### F.1 Limitations

The optimization process is relatively slow compared to feed-forward generative models such as MDM [97], which may hinder real-time deployment or interactive applications. Reducing inference time while maintaining quality remains an important direction for future work.

Due to the limited diversity of object categories in current datasets like ARCTIC [19], our model struggles to generalize to novel objects with significantly different geometries, topologies, or manipulation affordances. Our method only considers the rotational articulated objects, but other manipulation tasks, such as pushing or pulling, are not considered. Scaling to broader object types would require more diverse and high-quality motion data.

Our current framework focuses on single-object manipulation. Extending it to multi-object scenarios, such as opening a drawer and retrieving an item, or performing sequential multi-step tasks, poses both modeling and optimization challenges and remains an open problem.

Our method does not explicitly model obstacle avoidance. While the resulting body and hand motions are physically plausible, the character may occasionally intersect with surrounding geometry in

### Table 5: Inference time.

Module Object motion End-effector Whole-body motion Time 0.52 secs 3.66 secs 16.93 mins

cluttered or constrained scenes. Enabling both the fingers and the full body to reason jointly about nearby obstacles and environment geometry is an important direction for improving interaction realism.

Our text conditioning is currently limited by the simplicity of available annotations. While Text2HOI [6] provides rule-based captions for hand-object interactions, they are typically segmented into short atomic motions, whereas we aim to model longer and more coherent manipulation sequences. Developing richer textual annotations and grounding them to temporally extended actions is a promising avenue for future work.

Our method primarily focuses on manipulation tasks, with the body posture adapting naturally to support the behaviors. However, some interactions require direct contact between the object and other body parts, such as pressing a box against the torso or holding an item between the arm and the body. Modeling such whole-body contact behaviors remains largely unexplored and could further expand the expressiveness of interaction generation.

### F.2 Broader impact

Our method can be used to create a realistic manipulation sequence, which could be rendered as a video. It also has the potential to be transferred to humanoid robots [60]. Therefore, our method has a potential positive social impact to help build the development of character animation and humanoid robotics.

