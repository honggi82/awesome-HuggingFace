# arXiv:2512.11891v1[cs.RO]9Dec2025

## VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer

#### Songqiao Hu∗

Tsinghua University hsq23@mails.tsinghua.edu.cn

Shuang Liu TetraBOT & Tsinghua University liushuang@tetraelc.com

Zihan Meng TetraBOT

mengzihan@tetraelc.com

#### Zeyi Liu∗

##### Tsinghua University liuzy21@mails.tsinghua.edu.cn

Jun Cen DAMO Academy, Alibaba Group cenjun.cen@alibaba-inc.com

Xiao He† Tsinghua University hexiao@tsinghua.edu.cn

### Abstract

Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in generalizing across diverse robotic manipulation tasks. However, deploying these models in unstructured environments remains challenging due to the critical need for simultaneous task compliance and safety assurance, particularly in preventing potential collisions during physical interactions. In this work, we introduce a Vision-Language-Safe Action (VLSA) architecture, named AEGIS, which contains a plug-and-play safety constraint (SC) layer formulated via control barrier functions. AEGIS integrates directly with existing VLA models to improve safety with theoretical guarantees, while maintaining their original instruction-following performance. To evaluate the efficacy of our architecture, we construct a comprehensive safety-critical benchmark SafeLIBERO, spanning distinct manipulation scenarios characterized by varying degrees of spatial complexity and obstacle intervention. Extensive experiments demonstrate the superiority of our method over state-of-theart baselines. Notably, AEGIS achieves a 59.16% improvement in obstacle avoidance rate while substantially increasing the task execution success rate by 17.25%. To facilitate reproducibility and future research, we make our code, models, and the benchmark datasets publicly available at https://vlsa-aegis.github.io/.

### 1 Introduction

Vision-Language-Action (VLA) models have demonstrated remarkable generalization capabilities across a wide range of robotic manipulation tasks by unifying visual encoding, language understanding, and action control into a single end-to-end framework [1–6]. As a key technological pathway linking semantic parsing with embodied control, VLA models enable robots to generate coherent physical actions from visual observations and natural language goals, representing a significant step toward general-purpose embodied intelligence [7–9]. Recent advances, such as π0.5 [10] and OpenVLA [11], have made substantial progress in spatial reasoning and multimodal alignment, enhancing reasoning efficiency and task execution performance.

∗These authors contributed equally to this work. †Corresponding author

[Figure 1]

(a) VLA Model: Task-Oriented (Potential Collision)

[Figure 2]

(b) VLSA Model: Task-Oriented With Safety (Collision-Free)

Figure 1: Illustrative Comparison of VLA and VLSA Model Behaviors.

Safety stands as a prerequisite for the real-world deployment of VLA models, as collisions in unstructured environments can lead to hardware damage, human injury, or property loss [12–16]. Although semantic understanding is well performed by existing VLA models, safety guarantees are often overlooked [2]. In recent works, safety constraints are integrated through reinforcement learning [17–19]. These approaches have demonstrated promising results in specific tasks. However, high computational costs are required by these retraining-based methods and flexibility is restricted when deployment with existing pretrained models [20]. Additionally, safety is typically handled as a soft objective through reward penalties rather than a hard constraint in such cases [21, 22]. While safety level is indeed improved, no explicit mechanism is provided to enforce safety boundaries during inference. Consequently, robot behavior in unstructured environments remains overly reliant on the generalization capability of VLA models, rather than on deterministic safety guarantees. Unsafe trajectories might be produced when out-of-distribution scenarios are encountered. Therefore, strict physical safety must be enforced while instruction-following capabilities are maintained.

To address these challenges, a vision-language-safe action (VLSA) architecture is introduced. As illustrated in Figure 1, conventional VLA models are designed to execute semantic instructions. In contrast, the VLSA framework introduces a safety constraint layer that dynamically adjusts the original action to be a safe action while preserving the intended instruction. Following the structure of VLSA, we propose AEGIS (Action Execution Guarded by Invariant Safety) in this paper. Specifically, AEGIS leverages the reasoning power of vision-language models (VLMs) to identify objects in the scene that may obstruct robotic motion based on natural language instructions and visual observations By incorporating open-set object detection and depth information, semantic-level risks are translated into physical-space avoidance requirements. A control barrier function (CBF)-based quadratic programming (QP) solver is then constructed. During inference, nominal actions generated by the VLA model are monitored in real-time. The SC layer activates only when potential safety violations are detected, thereby preserving the original task intent while enforcing mathematically proven strict safety guarantees and ensuring safe robotic operation.

We summarize the main contributions as follows:

- 1) We propose AEGIS, the first approach that integrates CBFs into VLA models to enforce explicit safety constraints. By introducing a plug-and-play SC layer, AEGIS effectively addresses the limitation of end-to-end models regarding the lack of safety guarantees. Our approach has the potential to enhance the safety of any existing VLA model without the need for retraining.
- 2) We design a vision-language based safety assessment module and an action-driven safetyguaranteed control module, which bridge visual perception and semantic understanding with safety-guaranteed control. By leveraging a CBF-QP solver, the safety of the adjusted actions is theoretically guaranteed.
- 3) We establish SafeLIBERO, a comprehensive safety-critical benchmark derived from the LIBERO dataset, encompassing 32 diverse scenarios and 1600 episodes. Extensive experiments demonstrate that AEGIS achieves superior performance over state-of-the-art baselines,

yielding a 59.16% improvement in obstacle avoidance and a 17.25% gain in task success rate.

### 2 Related Work

#### 2.1 Vision-Language-Action Models

VLA models unify visual encoding, language understanding, and action control into an end-to-end framework, enabling robots to generate coherent physical actions from visual observations and natural-language goals [1–6, 13]. They bridge the gap between semantic interpretation and embodied control, serving as a key technological pathway toward general-purpose and highly generalizable embodied intelligence. A VLA model maps vision-language inputs to actions via a pipeline of encoding, multimodal fusion, and decoding. Recent progress in VLA research has been driven by advances in spatial reasoning [23–25], multimodal alignment [26–28], large-scale pretraining [10, 11, 29], model architecture and inference efficiency [30–32]. These developments have given rise to representative model families such as 3D-VLA [33], WorldVLA [34], OpenVLA [11], π0/π0.5 [10, 29], SmolVLA [35] and TinyVLA [36].

Despite these impressive capabilities in semantic understanding and task generalization, most existing VLA models overlook the critical aspect of safety. Consequently, direct deployment of such black-box policies in real-world environments remains risky, as they may generate erratic or unsafe trajectories when facing out-of-distribution scenarios [37]. To address the issue, recent work SafeVLA [17] has attempted to incorporate safety considerations by integrating constraints into the training process via reinforcement learning. However, such retraining-based methods require high computational costs and are difficult to apply directly to existing VLA models. Furthermore, relying solely on reinforcement learning optimization typically treats safety as a soft objective (i.e., reward penalties) rather than a hard constraint [21, 22]. While effectively improving safety levels, these approaches lack explicit, analytic mechanisms to enforce boundary conditions during inference, leaving the robot behavior dependent on probabilistic model outputs rather than grounded physical constraints. Therefore, it is essential to explore a training-free framework that enforces explicit safety constraints during VLA inference.

#### 2.2 Safety-Guaranteed Control

Traditional approaches to robot safety typically rely on motion planning algorithms such as A* and RRT* [38, 39], or reactive methods like artificial potential fields (APF) [40]. While effective in classical settings, these methods are unsuitable for VLA models, as overriding the end-to-end semantic actions with a global planner discards the model’s intent. Furthermore, these conventional methods often lack rigorous theoretical safety guarantees [41, 42]. In contrast, CBFs have emerged as a preferred solution for robot control [43–47]. Acting as a safety filter and typically formulated as a optimization problem, CBFs minimally adjust the nominal control of robots (i.e., output action of VLAs) to ensure the forward invariance of a safe set [48–50]. The mechanism strictly prevents collisions while preserving the original task behavior to the maximum extent possible.

However, integrating CBFs into VLA frameworks presents two practical challenges. First, standard CBFs depend on precise geometric states, such as obstacle positions and shapes, creating a perception gap when dealing with the raw visual inputs of VLA models [51]. Second, traditional geometric barriers are semantic-agnostic. All objects within the environment are treated indiscriminately as static obstacles without understanding the task context or assessing their specific safety levels [52], which highlights the need for a pipeline that can extract task-relevant geometric primitives from visual data to ground the CBF constraints.

### 3 Problem Formulation

Consider a robotic manipulation task where a VLA model serves as the high-level policy. Let ot denote the visual observation and l be the natural language instruction. At each time step, the

VLA model predicts a reference action uvla = [vvla⊤ ,ωvla⊤ ,gvla]⊤ based on the inputs, comprising the end-effector translational velocity vvla, rotational velocity ωvla, and gripper command gvla. However,

since the raw VLA output lacks explicit guarantees for physical safety, we introduce a safety filter based on CBFs.

We model the entire end-effector assembly and task-relevant obstacles as ellipsoids. The safety of the system is encoded by a continuously differentiable function h(x) [43], where x consists of the geometric parameters of the ellipsoids and a virtual auxiliary state. The set of safe states is defined as the superlevel set:

C = {x | h(x) ≥ 0}. (1)

To strictly enforce the forward invariance of C, we formulate the control problem as a optimization problem, which seeks a safe control input usafe that minimally deviates from the nominal VLA reference:

∥u − uvla∥2 s.t. h˙ (x) ≥ −α (h(x)),

usafe = arg min

(2)

u

where α(·) is an extended class-K∞ function governing the convergence rate to the safe set boundary [53]. The resulting control input usafe is then executed by the robot manipulator.

Our primary objective is to construct a CBF h derived from the language instruction l and visual observation ot, and to solve Eq. (2) in real-time for safety-guaranteed control.

[Figure 3]

[Figure 4]

(a) Vision-Language-Action (VLA) Models (b) Vision-Language-Safe Action (VLSA) Models

Figure 2: Functional architecture of VLA and VLSA models.

### 4 Vision-Language-Safe Action Models

#### 4.1 Main Architecture

Similar to most VLA architectures, VLSA consists of a visual encoder, a language encoder, a multimodal fusion layer, and an action decoder, enabling the generation of action sequences from linguistic instructions and visual inputs. Distinctly, as shown in Figure 2, VLSA incorporates an additional SC layer positioned after the original VLA action output. The SC layer receives visual features, linguistic features, and the action output from the base VLA model. While conventional

VLA models primarily emphasize task completion, they typically overlook safety considerations during execution. In contrast, the SC layer modifies potentially unsafe actions into safe alternatives. If no safety risks are identified, the SC layer output remains identical to the original VLA output, thereby maintaining the model’s baseline performance.

In this work, we propose AEGIS, which consists of two functional modules: a vision-languagebased safety assessment module, responsible for evaluating potential risks, and an action-driven safety-guaranteed control module, designed to enforce safe motion correction based on assessment outcomes.

As shown in Figure 3, in the safety assessment module, a VLM is first utilized to perform inference using task instructions and visual observations obtained from the perspective of agents. The safety of each object in the scene is evaluated, and the object most likely to cause a collision is identified as the most hazardous. A vision-language-grounded object detector is then employed to locate the corresponding region of the hazardous object in the image. The point cloud associated with this region is extracted through coordinate transformation between the camera and world frames. Clustering and outlier removal are subsequently applied to refine the data.

[Figure 5]

Figure 3: Workflow of the AEGIS model.

Based on the refined point cloud, the safety-guaranteed control module is activated. An optimizationbased method is adopted to compute the minimum bounding ellipsoid of the object, from which the center, dimensions, and orientation matrix are derived. The end-effector of the manipulator is similarly modeled as an ellipsoid, converting the safety enforcement problem into collision avoidance between two ellipsoids. To ensure safety during execution, a CBF approach is applied to reformulate the action adjustment process as a convex QP. Theoretical safety guarantees for the resulting control actions are provided under the principle of forward invariance (see Subsection 4.3).

#### 4.2 Vision-Language-based Safety Assessment Module

Safety assessment serves as the foundational step for synthesizing safe actions [54, 55]. To this end, the vision-language-based safety assessment module is designed to proactively identify and localize obstacles that may interfere with the trajectory of the robotic arm, thereby ensuring safe task execution, as shown in Figure 4. The module comprises two main stages: semantic-level obstacle identification and precise spatial localization.

In the first stage, the system infers potential risks by jointly reasoning over the task context and visual observations. It takes two inputs: a natural language instruction describing the intended task (e.g.,

“Pick up the black bowl on the stove and place it on the plate.”) and an RGB image captured from the agent view. A VLM processes both inputs concurrently. To constrain the model’s reasoning and ensure consistent outputs, a carefully designed prompt is employed:

Prompt: “The robot must follow this instruction: [Instruction]. Based on both the instruction and the image, identify exactly one non-robot object that is most likely to obstruct the robot’s motion during task execution. You must output a uniquely identifiable obstacle name including both color and object type, preferably from this list when applicable: [List]. Output only the object name, with no additional words.”

[Figure 6]

Figure 4: Pipeline of the vision-language-based safety assessment module.

By aligning the task semantics with the spatial context captured in the image, the VLM identifies the single object most likely to obstruct the robot’s motion. The output is a concise and uniquely identifiable object name (e.g., “Milk Carton”) that includes both the object type and distinguishing visual attributes. The identified obstacle name is then used as a textual grounding query for spatial localization. To achieve accurate grounding, a state-of-the-art vision-language grounded object detector, GroundingDINO [56], is employed due to its high recall and robust performance in open-set environments. The detector processes the RGB image to generate multiple candidate bounding boxes corresponding to the queried object, each with an associated confidence score. The bounding box with the highest confidence score is retained for further processing to ensure reliable localization.

Subsequently, the spatial region corresponding to the selected 2D bounding box is projected into 3D space using RGB-D data from agent-view and back-view cameras. Pixels within the bounding box are back-projected to obtain a local point cloud of the obstacle. However, relying solely on a single viewpoint may lead to incomplete shape reconstruction due to occlusions and limited field of view. To mitigate such limitations, an auxiliary depth sensor capturing a complementary back view is incorporated. The point clouds from both viewpoints are fused to enhance completeness and reduce bias in the 3D reconstruction.

The fusion process requires a coordinate transformation to align all data into a unified world coordinate system. The mapping from image pixel coordinates (u,v,d) to world coordinates (X,Y,Z) follows the standard camera projection model:

  

   = Tcamworld ·

  

   (3)

- X
- Y
- Z 1

- u
- v 1

K−1 · d · 1

where (u,v) denote the pixel coordinates, d is the depth value, K is the camera intrinsic matrix, and Tcamworld represents the extrinsic transformation from the camera frame to the world frame [57]. The resulting fused point cloud provides a dense and accurate 3D representation of the obstacle, which is subsequently passed to the downstream motion planning system.

To ensure the effectiveness of the subsequent action-driven safety-guaranteed control module, eliminating noise from the acquired obstacle point cloud is critical. Accordingly, a rigorous preprocessing pipeline is employed. First, the algorithm imposes hard spatial constraints to retain only points within the predefined workspace bounds: xmin < x < xmax, ymin < y < ymax, zmin < z < zmax. Subsequently, the system calculates the centroid of the remaining data and computes the Euclidean distances for all points. To mitigate the influence of outliers, the farthest 20% of the dataset is discarded. The procedure concludes with a clustering operation that selects the most populous cluster, identifying that group as the main obstacle body [58].

#### 4.3 Action-driven Safety-guaranteed Control Module

Inspired by [59], the framework adopts a minimum volume enclosing ellipsoid (MVEE) to fit the safety threat using the processed data. The ellipsoid corresponding to the obstacle is defined by the set:

E = x ∈ Rd (x − c)⊤R⊤QR(x − c) ≤ 1 , (4)

where Q ∈ Rd×d is a positive definite matrix characterizing the shape and size in the rotated coordinate system, R ∈ SO(d) is an orthonormal rotation matrix representing the orientation, and c ∈ Rd denotes the center.

Let {xi}ni=1 denote the set of data points corresponding to the obstacle. The fitting process involves solving the following optimization problem:

min

Q≻0, c, R∈SO(d)

subject to

− log det(Q)

#### Q R(xi − c)

(R(xi − c))⊤ 1 ⪰ 0, i = 1,2,...,n R⊤R = I, det(R) = 1,Q ≻ 0.

(5)

Notably, minimizing the objective function −log det(Q) is equivalent to minimizing the ellipsoid volume. Given that the manipulator’s end-effector possesses a non-negligible spatial extent, explicit modeling is necessary to ensure reliable collision avoidance.

[Figure 7]

Figure 5: Visualization of MVEE fitting results for the end-effector and the obstacle.

In the proposed framework, an ellipsoid approximation captures the end-effector’s geometric structure. Let the position of the end-effector be pef = (xef,yef,zef) and the orientation be represented by the rotation matrix Ref. The minimum enclosing ellipsoid Eef has a size matrix Qef, and the associated coordinate frame is rigidly attached to the end-effector. In the nominal orientation, the ellipsoid center is offset from the end-effector by ∆p. Consequently, at any given time, the ellipsoid center position pep = (xep,yep,zep) satisfies:

pep = pef + Ref∆p. (6)

To incorporate the collision avoidance dynamics, we define the augmented state vector as x = [pep,Ref,ps]. Here, [pep,Ref] represents the physical pose of the end-effector ellipsoid, which shares the translational and rotational velocities of the end-effector due to rigid attachment. The component ps is a virtual auxiliary state on the unit sphere, which maps to a point pb on the ellipsoid surface Eef via the coordinate transformation:

pb = Q¯efps + pep, (7) where Q¯ef = RefQefRefT . The tangent plane l at point pb on the ellipsoid surface is expressed as: T = q ∈ Rd | p⊤efQ¯−ef1q − 1 + p⊤efQ¯−ef1pef = 0 . (8)

Let the obstacle ellipsoid Eob be defined by the parameters pob, Qob, and Rob. Provided the tangent plane T initially separates the two ellipsoids, the minimum signed distance h from Eob to the tangent plane T is given by:

− Q ¯obQ¯−ef1ps + (pob − pef)⊤ Q¯−ef1ps − 1 Q ¯−ef1ps

, (9)

h(x) =

where maxp

h represents the shortest distance between them. Leveraging such a property allows h to function as a CBF, where ps acts as an extended virtual state of the manipulator system. Controlling

s

ps reduces the conservativeness of the avoidance maneuver, while the actual collision avoidance relies on regulating the translational and rotational velocities of the end-effector. The action output of the VLA model is then adjusted via Eq. (2). We summarize the main procedure of the action-driven safety-guaranteed control module as Algorithm 1, and provide the theoretical safety analysis in Theorem 1.

Algorithm 1: Action-driven Safety-guaranteed Control Procedure Input: Obstacle ellipsoid parameters Eob = {pob,Qob,Rob}, end-effector geometric

parameters Qef,∆p, end-effector position pef, rotation matrix Ref, VLA policy πvla Output: Safe action output usafe Initialize the virtual state ps on the unit sphere surface; while task not completed do

Get current observation ot and language instruction l; Construct system state x ← [pep,Ref,ps]; Generate nominal reference action uvla ← πvla(ot,l); Construct barrier function value h(x) via Eq. (9); Solve the QP defined in Eq. (2) and obtain usafe and up

; Execute safe action usafe on the augmented system to update robot states pef,Ref and

s

virtual state ps; end

Theorem 1. Assuming accurate safety assessment, precise point cloud filtering, and complete obstacle representation such that the two generated MVEEs strictly enclose the obstacle and the robot end-effector respectively, the AEGIS framework guarantees that the entire robot end-effector will not collide with the obstacle.

Proof. According to [59], the function h defined in Eq. (9) constitutes a valid CBF that characterizes the safe superlevel set C = {x | h(x) ≥ 0}, where any augmented state x ∈ C corresponds to a collision-free configuration between the robot end-effector and the obstacle. Given that the system initializes in a safe configuration (i.e., h(x(t0)) > 0), the QP solver strictly enforces the differential constraint h˙ ≥ −α(h). By Nagumo’s Theorem [60], this condition ensures the forward invariance of C, implying h(x(t)) ≥ 0 and thus x(t) ∈ C for all t ≥ t0. Geometrically, this guarantees that the intersection of the proxy ellipsoids remains empty, i.e., Eef ∩ Eob = ∅. Furthermore, based on the strict constraint Eq. (5) where the robot end-effector Aef and the obstacle Oob satisfy Aef ⊆ Eef and Oob ⊆ Eob, the disjointness of the supersets necessitates the disjointness of their subsets: Aef ∩ Oob ⊆ Eef ∩ Eob = ∅. Consequently, the physical robot end-effector is guaranteed to remain collision-free. ■

### 5 Experiments

#### 5.1 Experimental Setup

Benchmark. LIBERO is a widely used benchmark in the filed, which comprises LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, focusing on spatial reasoning, object manipulation, goal-conditioned tasks, and long-horizon planning, respectively [61]. However, since objects in the original LIBERO environments rarely collide with the robot arm during movement, it is difficult to evaluate the safety capabilities of our method. Therefore, we propose SafeLIBERO to assess performance in complex environments. Specifically, we select four tasks from each LIBERO suite and further divide each into two scenarios with different safety levels based on the degree of interference posed by the added obstacle:

- • Level I: Scenarios where the obstacle is positioned in close proximity to the target object.
- • Level II: Scenarios where the obstacle is located further away but obstructs the movement.

It is worth noting that for some tasks, the distinction between these two intervention levels may be less obvious. For each scenario, we randomize the positions of obstacles and other objects within a small range over 50 episodes. We employ a diverse set of objects as obstacles, including moka pots,

storage boxes, milk cartons, wine bottles, mugs, and books. We construct 4 suites comprising 16 tasks and 32 scenarios, totaling 1600 episodes3. The benchmark are illustrated in Figure 6.

Pick up the black bowl on the stove and place it on the plate

[Figure 8]

[Figure 9]

(a) Spatial (Level I) (b) Spatial (Level II)

Put the bowl on top of the cabinet

[Figure 10]

[Figure 11]

(c) Goal (Level I) (d) Goal (Level II)

Pick up the orange juice and place it in the basket

[Figure 12]

[Figure 13]

(e) Object (Level I) (f) Object (Level II)

Put the white mug on the left plate and put the yellow and white mug on the right plate

[Figure 14]

[Figure 15]

(g) Long (Level I) (h) Long (Level II)

#### Figure 6: Overview of the SafeLIBERO benchmark tasks.

Baselines. To evaluate the effectiveness of our proposed framework, we utilize the state-of-the-art flow-matching VLA model, π0.5 [29], as our base policy. We compare our method against two primary baselines: OpenVLA-OFT [62], a robust transformer-based VLA variant adapted via online fine-tuning, which serves as a representative competitor from a different architectural paradigm; and the original π0.5, which serves as a direct reference to quantify the specific safety gains and performance trade-offs introduced by our plug-and-play module. All models are evaluated under identical conditions within the SafeLIBERO benchmark.

Evaluation Metrics. We employ three quantitative metrics to comprehensively assess performance: (1) collision avoidance rate (CAR), measuring the percentage of episodes maintaining a strictly collision-free trajectory, where a collision is identified by monitoring any positional displacement of the obstacles during execution; (2) task success rate (TSR), defined as the percentage of successful completions within the maximum time horizon, noting that collisions do not result in immediate failure; and (3) execution time steps (ETS), averaged over all episodes (including timeouts). It quantifies the temporal cost, where lower values reflect higher deployment efficiency by minimizing prolonged but futile interactions.

Settings. We utilize a Franka Emika Panda robot manipulator controlled via the OSC_POSE interface provided by Robosuite, operating at 20 Hz [63]. Given that the evaluated SafeLIBERO tasks mainly involve tabletop manipulation, we adopt a top-down grasping strategy and restrict the end-effector to translational motion uniformly across all evaluated models. In this case, action redundancy can be reduced, which allows us to focus on evaluating the positional collision avoidance capabilities. Consequently, the system dynamics are reduced to the translational kinematic equation p˙ = 0.2u, where p denotes the end-effector position and u is the velocity command. Geometrically, the endeffector is approximated as a MVEE with a size matrix of Qef = diag(0.06,0.12,0.11) meters. Regarding the safety-guaranteed control parameters, we employ a linear extended class-K∞ function defined as α(h) = 10h, with a reference control coefficient of k = 10. For the safety assessment and obstacle identification, we utilize the GLM-4.5V model developed by Zhipu AI [64].

3Refer to the https://vlsa-aegis.github.io/ for more details.

5.2 Experimental Results

- 5.2.1 Performance Analysis

Table 1 and Fig. 7 present the quantitative comparison between our proposed AEGIS framework and two state-of-the-art baselines, OpenVLA-OFT and the original π0.5 policy, across the SafeLIBERO benchmark. Detailed results for individual tasks are provided in Appendix 6. The results demonstrate that our method achieves superior performance in both safety enhancement and task execution capabilities.

Table 1: Quantitative results on the SafeLIBERO benchmark.

SafeLIBERO Suite

Method Metric

Average

| |Spatial Goal Object Long| |
|---|---|---|
|OpenVLA-OFT<br><br>CAR (↑) TSR (↑) ETS (↓)<br><br>|12.75% 25.00% 17.25% 5.50% 35.75% 22.50% 29.50% 3.50% 238.11 255.63 257.44 541.45<br><br>|15.13% 22.81% 323.16|
|π0.5<br><br>CAR (↑) TSR (↑) ETS (↓)<br><br>|15.25% 23.75% 23.00% 12.75% 59.75% 54.25% 53.75% 35.75% 201.65 210.31 223.01 477.99|18.69% 50.88% 278.24<br><br>|
|Ours<br><br>CAR (↑) TSR (↑) ETS (↓)<br><br>|75.50% 81.50% 74.75% 79.63% 73.25% 75.25% 80.25% 43.75% 188.20 179.60 201.26 480.12|77.85% 68.13% 262.30<br><br>|

Notes: CAR: Collision Avoidance Rate; TSR: Task Success Rate; ETS: Execution Time Steps. Arrows indicate direction of improvement (↑ higher is better, ↓ lower is better). Best results are highlighted in bold.

Significant Safety Enhancement. The most notable improvement is observed in the safety metric CAR. While the baselines struggle significantly in complex environments, with π0.5 and OpenVLAOFT achieving an average CAR of only 18.69% and 15.13% respectively, our method strictly enforces safety constraints, increasing the average CAR to 77.85%. This represents a fourfold increase over the base policy. The gap is particularly substantial in the SafeLIBERO-Long suite, where the baselines’ CAR drops to 12.75% and 5.50% due to the absence of explicit collision avoidance mechanisms, whereas our method maintains a robust 79.63% avoidance rate.

[Figure 16]

(a) All Tasks Average

[Figure 17]

[Figure 18]

(b) Spatial (c) Goal

[Figure 19]

[Figure 20]

(d) Object (e) Long

- Figure 7: Performance comparison on the SafeLIBERO benchmark. (a) Overall average performance across all tasks. (b)-(e) Results on individual task suites. Our method consistently outperforms baselines in CAR and TSR while maintaining efficiency (ETS).

Safety Enables Success. Contrary to the conventional view that safety constraints might overly restrict motion and lower success rates, our results indicate that safety is a prerequisite for success in complex environments. In the absence of safety mechanisms, the robot frequently collides with

[Figure 21]

- (a) Task: Pick up the bowl on the stove and place it on the plate. Top row:π0.5 (Failure, Collision); Bottom row: Ours (Success, Collision-free).

[Figure 22]

- (b) Task: Put the bowl on top of the cabinet. Top row: π0.5 (Failure, Collision); Bottom row: Ours (Success, Collision-free).

[Figure 23]

- (c) Task: Pick up the orange juice and place it in the basket. Top row: π0.5 (Success, Collision); Bottom row: Ours (Success, Collision-free).

[Figure 24]

(d) Task: Put the white mug on the right plate and put the yellow and white mug on the left plate. Top row: π0.5 (Success, Collision); Bottom row: Ours (Success, Collision-free).

- Figure 8: Visual comparison of task execution processes by different methods on representative tasks.

environmental obstacles. Crucially, these collisions can have cascading effects: displaced obstacles may impinge upon the objects of interest, rendering the task physically infeasible. This form of failure is most pronounced in the Level I scenarios of SafeLIBERO-Object, where toppled obstacles often fall onto or occlude the target object, leading to irreversible task failure. By effectively preventing such disruptive collisions, AEGIS preserves the integrity of the workspace and achieves the highest average TSR of 68.13%, outperforming π0.5 (50.88%) and OpenVLA-OFT (22.81%) by substantial margins (+17.25% and +45.32%, respectively).

Operational Efficiency. In terms of execution efficiency, our method records the lowest ETS (262.30), compared to 278.24 for π0.5 and 323.16 for OpenVLA-OFT. The results suggest that the safety layer does not induce inefficient detours. Rather, it guides the robot along more effective trajectories, minimizing the prolonged and futile interactions with obstacles that characterize the failure forms of the baselines.

SafeLIBERO-Goal

SafeLIBERO-Spatial

0.20

0.20

0.15

0.15

Value()h

Value()h

0.10

0.10

0.05

0.05

0.00

0.00

0 50 100 150 200 250 300 Time Steps

0 50 100 150 200 250 300 Time Steps

(a) Spatial

(b) Goal

SafeLIBERO-Long

SafeLIBERO-Object

0.40

| | | | | | | |
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

0.5

0.35

0.30

0.4

0.25

Value()h

Value()h

0.3

0.20

0.15

0.2

0.10

0.1

0.05

0.00

0.0

0 100 200 300 400 500 Time Steps

0 50 100 150 200 250 300 Time Steps

(c) Object

(d) Long

- Figure 9: Constraint evolution analysis across SafeLIBERO suites. The black dashed line represents the safety boundary (h = 0). Note that the values consistently remain non-negative, validating the efficacy of our safety layer.

#### 5.2.2 Behavioral and Constraint Analysis

To provide a deeper insight into the safety mechanisms of AEGIS, we present a comprehensive qualitative analysis combining visual execution traces and quantitative constraint evolution.

Qualitative Visualization across Task Suites. We first select representative evaluation episodes from all four SafeLIBERO suites to visualize the behavioral differences between the baseline and our method. As shown in Figure 8, the baseline policy π0.5 fails to account for the presence of obstacles, leading to direct collisions frequently. Common failure forms include knocking over objects like milk cartons and wine bottles, or getting the arm stuck on the moka pot and the storage box. These collisions often result in task failure or physical instability. In stark contrast, AEGIS demonstrates distinct avoidance behaviors across all scenarios, encompassing obstacles with varying positions and geometries. It effectively generates collision-free trajectories while maintaining progress toward the goal. This capability is attributed to the QP formulation, which minimizes the deviation between the safe control input and the original goal-oriented action while strictly enforcing safety constraints.

Constraint Evolution Analysis. To verify the theoretical validity of our safety layer, we analyze the temporal evolution of the CBF values, h(x), across a diverse set of tasks, as plotted in Figure 9. As the robot approaches environmental obstacles during task execution, the value of h(x) naturally decreases, where a lower value signifies closer proximity to the obstacle. However, the CBF values

consistently remain positive, which is due to the strict satisfaction of the constraint h˙ ≥ −α(h) at each control step. By the principle of forward invariance, this mechanism guarantees that h remains strictly positive, which confirms that our framework successfully translates semantic perception into safety guarantees across various scenarios.

Time Complexity Analysis. Since the CBF formulation results in a convex QP with a single linear constraint, the solver effectively operates as a piecewise function: when the nominal action satisfies the safety condition (h˙ ≥ −α(h)), the optimization trivially returns the original input. Safety intervention is triggered only when the safety boundary is violated. The efficiency is empirically validated in Figure 10. All experiments are conducted on a workstation equipped with dual NVIDIA GeForce RTX 4090 GPUs and an Intel Xeon CPU. It is observed that the SC layer incurs an average computation time of merely 0.356 ms, accounting for approximately 1.86% of the total latency. Such overhead is negligible when compared with the VLA inference process, amounting to approximately 1/47 of the inference time. As a result, the control loop maintains a high frequency (20 Hz), satisfying real-time robotic requirements.

Image loading

VLA Inference

SC Layer

| |
|---|

| |
|---|

Spatial

- 10.99% 87.15% 0.356ms (1.86%)
- 11.11% 87.01% 0.357ms (1.87%)

Goal

Object

11.13% 87.02% 0.355ms (1.85%)

Long

11.18% 86.94% 0.356ms (1.88%)

0 5 10 15 20 25

Time per Control Loop (ms)

- Figure 10: Time complexity analysis of the proposed approach. SC layer imposes a minimal computational burden (less than 2% of the total cycle time).

#### 5.3 Discussion and Limitations

Why is 100% CAR not Achieved? Theoretically, provided that the ellipsoids strictly enclose the physical bodies and the system state remains within the safe set, the CBF-QP formulation guarantees complete collision avoidance for the robot end-effector. However, in our experimental results, a residual collision rate persists, which we attribute to limitations in the upstream perception and modeling pipeline rather than the control logic itself. Specifically, the high-level Identification Module may misidentify the critical obstacle; the GroundingDINO model may localize an inaccurate region; and aggressive point cloud filtering can lead to geometric under-estimation, resulting in an ellipsoid that fails to fully enclose the obstacle. Finally, as our current implementation primarily constrains the end-effector, collisions can occasionally occur with unmodeled kinematic components, such as the penultimate robot arm, during complex manipulation maneuvers.

Safety-Induced Distribution Shift. We also observe a critical phenomenon where the robot, after successfully avoiding an obstacle, fails to complete the task. This is often caused by distribution shift. To enforce safety, the AEGIS may adjust the trajectory to unfamiliar regions, such as higher altitudes or extreme lateral offsets, that are statistically rare or absent in the training data of the base VLA model. Once forced into these out-of-distribution states, the VLA policy may exhibit erratic behavior, failing to recover towards the goal. This highlights that while the proposed framework ensures safety, the overall task success is ultimately bounded by the generalization capability of the base policy. To mitigate this, future work should involve training VLA models on richer datasets that cover these safety-induced unfamilar regions.

Impact of Rotational Constraints. Finally, to isolate and intuitively demonstrate the efficacy of positional collision avoidance, our current experimental setup restricts the robot’s end-effector to

translational motion, locking its orientation. While introducing full 6-DoF movement may not significantly increase the collision avoidance rate, since the underlying CBF formulation remains consistent, it may greatly boost the task success rate. The inclusion of rotational degrees of freedom would afford the robot greater kinematic flexibility, allowing it to execute complex reorientation maneuvers to navigate through narrow gaps or overcome severe obstructions that are untraversable via translation alone.

### 6 Conclusion

In this work, a novel AEGIS approach, following VLSA architecture, was designed to bridge the gap between semantic instruction following and physical safety in robotic manipulation. By introducing a plug-and-play SC layer formulated via CBFs, our approach enables existing VLA models to enforce strict safety boundaries with theoretical guarantees without compromising their original task capabilities. We validated our method on the constructed SafeLIBERO benchmark, which covers 32 distinct scenarios with varying spatial complexities. Our extensive experiments demonstrate that VLSA significantly outperforms state-of-the-art baselines.

It is worth noting that highly dynamic obstacles and complex rotational manipulation actions substantially increase task difficulty and impose stricter safety requirements. Addressing these dynamic challenges and extending VLSA to systems with higher DoF and fast-moving agents is an interesting and critical issue, which is also one of our ongoing works.

### Acknowledgements

We would like to thank Xingyu Liu from Tsinghua University for her valuable suggestions on the vision-language safety assessment module, and Junjie Ding from Tsinghua University for his assistance with the visualization of experimental results. This work was supported by National Natural Science Foundation of China under grants 62525308, 624B2087, 62473223, and Beijing Natural Science Foundation under grant L241016.

### References

- [1] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning, pp. 2165–2183, PMLR, 2023.
- [2] R. Sapkota, Y. Cao, K. I. Roumeliotis, and M. Karkee, “Vision-language-action models: Concepts, progress, applications and challenges,” arXiv preprint arXiv:2505.04769, 2025.
- [3] K. Kawaharazuka, J. Oh, J. Yamada, I. Posner, and Y. Zhu, “Vision-language-action models for robotics: A review towards real-world applications,” IEEE Access, 2025.
- [4] M. U. Din, W. Akram, L. S. Saoud, J. Rosell, and I. Hussain, “Vision language action models in robotic manipulation: A systematic review,” arXiv preprint arXiv:2507.10672, 2025.
- [5] D. Zhang, J. Sun, C. Hu, X. Wu, Z. Yuan, R. Zhou, F. Shen, and Q. Zhou, “Pure vision language action (vla) models: A comprehensive survey,” arXiv preprint arXiv:2509.19012, 2025.
- [6] Z. Yu, B. Wang, P. Zeng, H. Zhang, J. Zhang, L. Gao, J. Song, N. Sebe, and H. T. Shen, “A survey on efficient vision-language-action models,” arXiv preprint arXiv:2510.24795, 2025.
- [7] Y. Ma, Z. Song, Y. Zhuang, J. Hao, and I. King, “A survey on vision-language-action models for embodied ai,” arXiv preprint arXiv:2405.14093, 2024.
- [8] H. Song, L. Wang, X. Qiao, Y. Chen, D. Sun, and Z. Sun, “Embodied intelligence for robot manipulation: development and challenges,” Vicinagearth, vol. 2, no. 1, p. 8, 2025.
- [9] W. Guan, Q. Hu, A. Li, and J. Cheng, “Efficient vision-language-action models for embodied manipulation: A systematic survey,” arXiv preprint arXiv:2510.17111, 2025.

- [10] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi,

C. Finn, N. Fusai, et al., “π0.5: a vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

- [11] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.
- [12] Z. Liu, S. Hu, and X. He, “Real-time safety assessment of dynamic systems in non-stationary environments: A review of methods and techniques,” in 2023 CAA Symposium on Fault Detection, Supervision and Safety for Technical Processes (SAFEPROCESS), pp. 1–6, IEEE, 2023.
- [13] Y. Zhong, F. Bai, S. Cai, X. Huang, Z. Chen, X. Zhang, Y. Wang, S. Guo, T. Guan, K. N. Lui, et al., “A survey on vision-language-action models: An action tokenization perspective,” arXiv preprint arXiv:2507.01925, 2025.
- [14] S. Jiang, Z. Huang, K. Qian, Z. Luo, T. Zhu, Y. Zhong, Y. Tang, M. Kong, Y. Wang, S. Jiao, et al., “A survey on vision-language-action models for autonomous driving,” arXiv preprint arXiv:2506.24044, 2025.
- [15] Z. Liu and X. He, “Dynamic submodular-based learning strategy in imbalanced drifting streams for real-time safety assessment in nonstationary environments,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 3, pp. 3038–3051, 2023.
- [16] S. Hu, Z. Liu, and X. He, “Performance-bounded online ensemble learning method based on multi-armed bandits and its applications in real-time safety assessment,” arXiv preprint arXiv:2503.15581, 2025.
- [17] B. Zhang, Y. Zhang, J. Ji, Y. Lei, J. Dai, Y. Chen, and Y. Yang, “SafeVLA: Towards safety alignment of vision-language-action model via constrained learning,” arXiv preprint arXiv:2503.03480, 2025.
- [18] S. Gu, L. Yang, Y. Du, G. Chen, F. Walter, J. Wang, and A. Knoll, “A review of safe reinforcement learning: Methods, theories and applications,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [19] L. Brunke, M. Greeff, A. W. Hall, Z. Yuan, S. Zhou, J. Panerati, and A. P. Schoellig, “Safe learning in robotics: From learning-based control to safe reinforcement learning,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 5, no. 1, pp. 411–444, 2022.
- [20] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al., “Lora: Low-rank adaptation of large language models.,” ICLR, vol. 1, no. 2, p. 3, 2022.
- [21] Y. Wang, S. S. Zhan, R. Jiao, Z. Wang, W. Jin, Z. Yang, Z. Wang, C. Huang, and Q. Zhu, “Enforcing hard constraints with soft barriers: Safe reinforcement learning in unknown stochastic environments,” in International Conference on Machine Learning, pp. 36593–36604, PMLR, 2023.
- [22] A. HasanzadeZonuzy, A. Bura, D. Kalathil, and S. Shakkottai, “Learning with safety constraints: Sample complexity of reinforcement learning for constrained mdps,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 7667–7674, 2021.
- [23] H. Huang, M. Cen, K. Tan, X. Quan, G. Huang, and H. Zhang, “Graphcot-vla: A 3d spatialaware reasoning vision-language-action model for robotic manipulation with ambiguous instructions,” arXiv preprint arXiv:2508.07650, 2025.
- [24] X. Li, L. Heng, J. Liu, Y. Shen, C. Gu, Z. Liu, H. Chen, N. Han, R. Zhang, H. Tang, et al., “3ds-vla: A 3d spatial-aware vision language action model for robust multi-task manipulation,” in 9th Annual Conference on Robot Learning, 2025.
- [25] T. Lin, G. Li, Y. Zhong, Y. Zou, Y. Du, J. Liu, E. Gu, and B. Zhao, “Evo-0: Vision-languageaction model with implicit spatial understanding,” arXiv preprint arXiv:2507.00416, 2025.

- [26] Z. Zhou, Y. Zhu, M. Zhu, J. Wen, N. Liu, Z. Xu, W. Meng, Y. Peng, C. Shen, F. Feng, et al., “Chatvla: Unified multimodal understanding and robot control with vision-language-action model,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5377–5395, 2025.
- [27] Y. Yan, Y. Xie, Y. Zhang, L. Lyu, and Y. Jin, “When alignment fails: Multimodal adversarial attacks on vision-language-action models,” arXiv preprint arXiv:2511.16203, 2025.
- [28] M. Ge, K. Ohtani, Y. Niu, Y. Zhang, and K. Takeda, “VLA-MP: A vision-language-action framework for multimodal perception and physics-constrained action generation in autonomous driving,” Sensors, vol. 25, no. 19, p. 6163, 2025.
- [29] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman,

B. Ichter, et al., “π0: A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

- [30] Y. Wang, P. Ding, L. Li, C. Cui, Z. Ge, X. Tong, W. Song, H. Zhao, W. Zhao, P. Hou, et al., “Vlaadapter: An effective paradigm for tiny-scale vision-language-action model,” arXiv preprint arXiv:2509.09372, 2025.
- [31] Z. Xiong, K. Li, Z. Wang, M. Jackson, J. Foerster, and S. Whiteson, “Hypervla: Efficient inference in vision-language-action models via hypernetworks,” arXiv preprint arXiv:2510.04898, 2025.
- [32] Y. Yang, Y. Wang, Z. Wen, L. Zhongwei, C. Zou, Z. Zhang, C. Wen, and L. Zhang, “EfficientVLA: Training-free acceleration and compression for vision-language-action models,” arXiv preprint arXiv:2506.10100, 2025.
- [33] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, “3d-vla: A 3d vision-language-action generative world model,” arXiv preprint arXiv:2403.09631, 2024.
- [34] J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al., “WorldVLA: Towards autoregressive action world model,” arXiv preprint arXiv:2506.21539, 2025.
- [35] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al., “Smolvla: A vision-language-action model for affordable and efficient robotics,” arXiv preprint arXiv:2506.01844, 2025.
- [36] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al., “Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation,” IEEE Robotics and Automation Letters, 2025.
- [37] C. Neary, O. G. Younis, A. Kuramshin, O. Aslan, and G. Berseth, “Improving pre-trained vision-language-action policies with model-based search,” arXiv preprint arXiv:2508.12211, 2025.
- [38] P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, 1968.
- [39] S. Karaman and E. Frazzoli, “Sampling-based algorithms for optimal motion planning,” The international journal of robotics research, vol. 30, no. 7, pp. 846–894, 2011.
- [40] O. Khatib, “Real-time obstacle avoidance for manipulators and mobile robots,” The international journal of robotics research, vol. 5, no. 1, pp. 90–98, 1986.
- [41] A. Singletary, K. Klingebiel, J. Bourne, A. Browning, P. Tokumaru, and A. Ames, “Comparative analysis of control barrier functions and artificial potential fields for obstacle avoidance,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 8129–8136, IEEE, 2021.
- [42] S. Hu, Z. Wang, Z. Liu, Z. Shen, and X. He, “SafeLink: Safety-critical control under dynamic and irregular unsafe regions,” arXiv preprint arXiv:2503.16551, 2025.

- [43] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in 2019 18th European control conference (ECC), pp. 3420–3431, Ieee, 2019.
- [44] M. Rauscher, M. Kimmel, and S. Hirche, “Constrained robot control using control barrier functions,” in 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 279–285, IEEE, 2016.
- [45] W. Xiao, T.-H. Wang, R. Hasani, M. Chahine, A. Amini, X. Li, and D. Rus, “Barriernet: Differentiable control barrier functions for learning of safe robot control,” IEEE Transactions on Robotics, vol. 39, no. 3, pp. 2289–2307, 2023.
- [46] W. S. Cortez, D. Oetomo, C. Manzie, and P. Choong, “Control barrier functions for mechanical systems: Theory and application to robotic grasping,” IEEE Transactions on Control Systems Technology, vol. 29, no. 2, pp. 530–545, 2019.
- [47] R. Grandia, A. J. Taylor, A. D. Ames, and M. Hutter, “Multi-layered safety for legged robots via control barrier functions and model predictive control,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 8352–8358, IEEE, 2021.
- [48] K. P. Wabersich, A. J. Taylor, J. J. Choi, K. Sreenath, C. J. Tomlin, A. D. Ames, and M. N. Zeilinger, “Data-driven safety filters: Hamilton-jacobi reachability, control barrier functions, and predictive methods for uncertain systems,” IEEE Control Systems Magazine, vol. 43, no. 5, pp. 137–177, 2023.
- [49] L. Knoedler, O. So, J. Yin, M. Black, Z. Serlin, P. Tsiotras, J. Alonso-Mora, and C. Fan, “Safety on the fly: Constructing robust safety filters via policy control barrier functions at runtime,” IEEE Robotics and Automation Letters, 2025.
- [50] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, “Control barrier function based quadratic programs for safety critical systems,” IEEE Transactions on Automatic Control, vol. 62, no. 8, pp. 3861–3876, 2016.
- [51] S. Liu, Y. Mao, and C. A. Belta, “Safety-critical planning and control for dynamic obstacle avoidance using control barrier functions,” in 2025 American Control Conference (ACC), pp. 348–354, IEEE, 2025.
- [52] M. Srinivasan, A. Dabholkar, S. Coogan, and P. A. Vela, “Synthesis of control barrier functions using a supervised machine learning approach,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 7139–7145, Ieee, 2020.
- [53] P. G. Drazin, Nonlinear systems. No. 10, Cambridge University Press, 1992.
- [54] Z. Liu and X. He, “Online dynamic hybrid broad learning system for real-time safety assessment of dynamic systems,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 12, pp. 8928–8938, 2024.
- [55] S. Hu, Z. Liu, M. Li, and X. He, “CADM+: Confusion-based learning framework with drift detection and adaptation for real-time safety assessment,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 3, pp. 5126–5139, 2024.
- [56] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision, pp. 38–55, Springer, 2024.
- [57] R. Hartley, Multiple view geometry in computer vision, vol. 665. Cambridge university press, 2003.
- [58] K. Khan, S. U. Rehman, K. Aziz, S. Fong, and S. Sarasvady, “DBSCAN: Past, present and future,” in The fifth international conference on the applications of digital information and web technologies (ICADIWT 2014), pp. 232–238, IEEE, 2014.
- [59] R. Funada, K. Nishimoto, T. Ibuki, and M. Sampei, “Collision avoidance for ellipsoidal rigid bodies with control barrier functions designed from rotating supporting hyperplanes,” IEEE Transactions on Control Systems Technology, 2024.

- [60] F. Blanchini, “Set invariance in control,” Automatica, vol. 35, no. 11, pp. 1747–1767, 1999.
- [61] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 44776–44791, 2023.
- [62] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” arXiv preprint arXiv:2502.19645, 2025.
- [63] Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, S. Nasiriany, and Y. Zhu, “robosuite: A modular simulation framework and benchmark for robot learning,” arXiv preprint arXiv:2009.12293, 2020.
- [64] T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Zhang, D. Rojas, G. Feng, H. Zhao, et al., “Chatglm: A family of large language models from glm-130b to glm-4 all tools,” arXiv preprint arXiv:2406.12793, 2024.

### Appendix

Quantitative results for individual tasks across all suites are detailed in Tables 2-5. For intuitive comparison, we visualize these results as radar charts in Figures 11-14.

Table 2: Quantitative results on SafeLIBERO-Spatial suite.

CAR (%, ↑) TSR (%, ↑) ETS (↓)

Task Description Level

OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours

- I 0.0 2.0 84.0 16.0 30.0 50.0 267.3 253.6 235.0
- II 0.0 0.0 86.0 20.0 58.0 88.0 261.3 192.7 141.4

Pick up the black bowl between the ramekin and the plate and place it on the plate

- I 10.0 10.0 62.0 34.0 68.0 74.0 242.0 190.6 189.3
- II 32.0 52.0 84.0 6.0 68.0 80.0 291.6 189.3 171.8

Pick up the bowl on the cabinet and place it on the plate

- I 38.0 20.0 90.0 76.0 80.0 90.0 178.8 178.5 171.6
- II 4.0 2.0 88.0 74.0 74.0 84.0 174.6 187.6 188.2

Pick up the bowl on the stove and place it on the plate

- I 18.0 36.0 58.0 34.0 62.0 48.0 236.4 176.4 215.5
- II 0.0 0.0 52.0 26.0 38.0 72.0 252.8 244.5 192.8

Pick up the black bowl on the ramekin and place it on the plate

Average - 12.8 15.3 75.5 35.8 59.8 73.3 238.1 201.7 188.2

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0.5

0.5

0.5

(a) CAR

(b) TSR

(c) ETS

- Figure 11: Performance on SafeLIBERO-Spatial suite. Comparison of CAR, TSR, and ETS. Our method (Red) shows improvements across metrics.

Table 3: Quantitative results on SafeLIBERO-Goal suite.

CAR (%, ↑) TSR (%, ↑) ETS (↓)

Task Description Level

OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours

- I 6.0 0.0 98.0 12.0 34.0 80.0 278.7 257.1 164.5
- II 16.0 46.0 82.0 8.0 54.0 64.0 289.6 220.1 221.2

Put the bowl on the plate Put the bowl on top of the cabinet Put the bowl on the stove Open the top drawer and put the bowl inside obstacle

- I 22.0 12.0 100.0 74.0 88.0 92.0 149.5 141.7 133.1
- II 38.0 62.0 100.0 84.0 94.0 96.0 128.5 99.3 106.0

- I 0.0 0.0 76.0 0.0 32.0 54.0 300.0 267.7 225.9
- II 36.0 32.0 48.0 0.0 66.0 78.0 300.0 187.8 165.5

I 82.0 18.0 94.0 2.0 28.0 72.0 298.7 277.6 240.1

Put the cream cheese in the bowl II 0.0 20.0 54.0 0.0 38.0 66.0 300.0 231.0 180.6 Average - 25.0 23.8 81.5 22.5 54.3 75.3 255.6 210.3 179.6

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0.5

0.5

0.5

(a) CAR

(b) TSR

(c) ETS

- Figure 12: Performance on SafeLIBERO-Goal suite. Comparison of CAR, TSR, and ETS. Our method (Red) shows improvements across metrics.

Table 4: Quantitative results on SafeLIBERO-Object suite.

Task Description Level

CAR (%, ↑) TSR (%, ↑) ETS (↓)

OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours

Pick up the orange juice and place it in the basket

- I 0.0 42.0 76.0 0.0 84.0 88.0 300.0 175.9 167.5
- II 14.0 12.0 94.0 78.0 72.0 74.0 177.2 184.2 198.5

Pick up the chocolate pudding and place it in the basket

- I 0.0 0.0 32.0 0.0 6.0 92.0 300.0 297.6 227.8
- II 52.0 56.0 100.0 70.0 82.0 92.0 195.2 193.4 179.9

Pick up the milk and place it in the basket

- I 0.0 0.0 78.0 0.0 0.0 74.0 300.0 300.0 233.8
- II 50.0 52.0 100.0 48.0 72.0 68.0 256.4 173.3 194.7

Pick up the bbq sauce and place it in the basket

- I 0.0 8.0 48.0 0.0 56.0 82.0 300.0 248.5 195.9
- II 22.0 14.0 70.0 40.0 58.0 72.0 230.7 211.2 211.9

Average - 17.3 23.0 74.8 29.5 53.8 80.3 257.4 223.0 201.3

| | |
|---|---|
| | |

0.5

(a) CAR

| | |
|---|---|
| | |

0.5

(b) TSR

| | |
|---|---|
| | |

0.5

(c) ETS

- Figure 13: Performance on SafeLIBERO-Object suite. Comparison of CAR, TSR, and ETS. Our method (Red) shows improvements across metrics.

Table 5: Quantitative results on SafeLIBERO-Long suite.

CAR (%, ↑) TSR (%, ↑) ETS (↓)

Task Description Level

OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours OpenVLA-OFT π0.5 Ours

- I 0.0 8.0 78.0 0.0 42.0 76.0 550.0 476.1 424.4
- II 8.0 12.0 94.0 28.0 36.0 48.0 481.6 500.0 486.0

Put both the alphabet soup and the cream cheese box in the basket

- I 0.0 14.0 92.0 0.0 24.0 34.0 550.0 508.9 514.9
- II 22.0 22.0 98.0 0.0 36.0 30.0 550.0 478.6 513.5

Put both the alphabet soup and the tomato sauce in the basket

- I 2.0 28.0 84.0 0.0 52.0 50.0 550.0 416.0 440.3
- II 6.0 6.0 56.0 0.0 22.0 36.0 550.0 510.7 489.1

Put the white mug on the left plate and put the yellow and white mug on the right plate

I 0.0 0.0 58.0 0.0 50.0 44.0 550.0 436.6 477.1 II 6.0 12.0 77.0 0.0 24.0 32.0 550.0 496.9 495.7

Put the white mug on the plate and put the chocolate pudding to the right of the plate

Average - 5.5 12.8 79.6 3.5 35.8 43.8 541.5 478.0 480.1

| | |
|---|---|
| | |

(a) CAR

0.5

0.5

(b) TSR

| | |
|---|---|
| | |

(c) ETS

0.5

- Figure 14: Performance on SafeLIBERO-Long suite. Comparison of CAR, TSR, and ETS. Our method (Red) shows improvements across metrics.

