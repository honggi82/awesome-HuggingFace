# arXiv:2509.18084v2[cs.RO]24Sep2025

[Figure 1]

## ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces

### Jiawen Tian†, Liqun Huang, Zhongren Cui, Jingchao Qiao, Jiafeng Xu, Xiao Ma, Zeyu Ren†

ByteDance Seed †Corresponding authors

### Abstract

This paper introduces ByteWrist, a novel highly-flexible and anthropomorphic parallel wrist for robotic manipulation. ByteWrist addresses the critical limitations of existing serial and parallel wrists in narrow-space operations through a compact three-stage parallel drive mechanism integrated with arc-shaped end linkages. The design achieves precise RPY (Roll-Pitch-Yaw) motion while maintaining exceptional compactness, making it particularly suitable for complex unstructured environments such as home services, medical assistance, and precision assembly. The key innovations include: (1) a nested three-stage motor-driven linkages that minimize volume while enabling independent multi-DOF control, (2) arc-shaped end linkages that optimize force transmission and expand motion range, and (3) a central supporting ball functioning as a spherical joint that enhances structural stiffness without compromising flexibility. Meanwhile, we present comprehensive kinematic modeling including forward / inverse kinematics and a numerical Jacobian solution for precise control. Empirically, we observe ByteWrist demonstrates strong performance in narrow-space maneuverability and dual-arm cooperative manipulation tasks, outperforming Kinova-based systems. Results indicate significant improvements in compactness, efficiency, and stiffness compared to traditional designs, establishing ByteWrist as a promising solution for next-generation robotic manipulation in constrained environments.

Date: September 23, 2025 Correspondence: tianjiawen.robot@bytedance.com,renzeyu.93@bytedance.com Project Page: https://bytewrist.github.io/

### 1 INTRODUCTION

With the rapid advancement of robotic technology, robot application scenarios have shifted dramatically from open and highly structured environments to complex and unstructured settings such as home services, medical assistance, and precision assembly [1, 2]. Among these scenarios, operations in narrow and confined spaces, such as handling objects in cluttered home cabinets, performing minimally invasive surgical interventions in human body cavities, and assembling precision components in automotive engine compartments have put forward increasingly stringent requirements for the flexibility, compactness, and dynamic responsiveness of robotic end-effectors, especially robotic wrists.

Traditional serial robotic wrists, composed of multiple sequentially connected rotational joints, have the advantages of straightforward kinematic modeling [3, 4] and a large rotational range per joint [5, 6]. However,

[Figure 2]

[Figure 3]

(a) Components Description in Wireframe. (b) Parameters Description in Prototype. Figure 1 Design and Development of ByteWrist.

their open-chain structure leads to accumulated errors, low structural stiffness, and a relatively bulky overall volume. When operating in narrow spaces, the multi-link serial structure is prone to collision with the surrounding environment, limiting their maneuverability. In contrast, parallel robotic wrists, which actuate the end platform through multiple parallel branches, demonstrate higher structural rigidity, superior load capacity, and better motion accuracy.

However, existing parallel wrist configurations with different structural designs and actuation methods struggle to simultaneously meet the requirements of compactness and flexibility [7], as well as those of high rigidity, high load capacity and simple transmission mode [8–14]. This renders them less suitable for anthropomorphic manipulation tasks that necessitate both high dexterity and strong spatial adaptability.

To address these limitations, this paper proposes a novel parallel robotic wrist named ByteWrist. The core design concept of ByteWrist is to integrate a compact three-stage parallel drive mechanism with arc-shaped end linkages, achieving precise RPY motion [15–18] of the end platform while maintaining a small structural footprint. Compared with existing parallel and serial wrists, ByteWrist features three key innovations:

- 1. High Compactness: The nested three-stage motor-driven linkages that minimize the overall volume while enabling independent control of multiple degrees of freedom.
- 2. High Efficiency: The arc-shaped end linkages optimize the force transmission path and expand the effective motion range of the end platform.
- 3. High Stiffness: The central supporting ball functioned as a spherical joint enhances structural stiffness without sacrificing flexibility.

This paper is organized as follows: Section II elaborates on the mechanical structure of ByteWrist, encompassing its drive mechanism, linkage design, and key structural parameters. Section III establishes the forward and inverse kinematic models for the wrist, and proposes a numerical solution for the Jacobian matrix to ensure precise control. Section IV validates the performance of ByteWrist through three sets of experiments: motion range testing, a narrow-space maneuverability comparison with Kinova, and a dual-arm chest-front cooperative manipulation for clothes hanging. Finally, Section V summarizes the research findings and discusses prospective improvements.

### 2 STRUCTURE OF BYTEWRIST

ByteWrist is illustrated in Fig. 1a, which is driven by three stage motors. The output of the first-stage motor is connected to the first-stage driving linkage, which is further linked to the parallel platform via an arc-shaped end linkage. Meanwhile, the second-stage motor is mounted inside the first-stage driving linkage, its output connects to the second-stage driving linkage, which is also linked to the parallel platform through an arc-shaped end linkage. Similarly, the third-stage motor is fixed within the second-stage driving linkage, with its output attached to the third-stage driving linkage that connects to the parallel platform via an arc-shaped end linkage.

All three stage driving linkages and arc-shaped end linkages, as well as arc-shaped end linkages and parallel platform, are connected via revolute pairs. All these six revolute pairs are oriented toward the center of the parallel platform.

To enhance the stiffness of the parallel platform, a supporting ball is mounted at its center and connected to the platform via a spherical joint. By controlling the movement of three stage motors, the end parallel platform can achieve precise RPY motion.

As illustrated in Fig. 1b, the prototype of ByteWrist adopts Quasi-Direct Drive [19–21] based actuators manufactured by RobStride Dynamics. R1 denotes the rotation radius of the connection point between all three stage driving linkages and arc-shaped end linkages, R2 denotes the rotation radius of the connection point between arc-shaped end linkages and the parallel platform. The center distance between these two rotation planes is h.

The arc-shaped end linkage consists of a 90-degree arc and two straight segments connected to the arc, where the radius of the arc is Rl and the length of two straight segments are l1 and l2 respectively, as described in Fig. 1b.

Above mentioned parameters are shown in Table 1.

Table 1 Structural Parameters of ByteWrist (Unit: mm)

R1 R2 h Rl l1 l2 27.35 30 27.35 25 5 13.68

### 3 FORWARD AND INVERSE KINEMATICS OF BYTEWRIST

To achieve precise control of ByteWrist, deriving its forward and inverse kinematics is essential. For clarity of description, its structure is simplified as illustrated in Fig. 2.

[Figure 4]

Figure 2 Coordinate System Definition of ByteWrist.

Points P1, P2 and P3 lie on the axis of the revolute joint connecting the arc-shaped end linkage and the driving linkage, and are situated on the inner surface of arc-shaped end linkages. Points P4, P5, and P6 lie on the axis of the revolute joint between the arc-shaped end linkage and the parallel platform, and are similarly located on the inner surface of arc-shaped end linkages. Point O0 is the center of the circle passing through points P1,

P2 and P3 with a radius of R1. Correspondingly point O1 is the center of the circle passing through points P4, P5 and P6 with a radius of R2. For the coordinate system {O0X0Y0Z0}, when the parallel wrist is in its initial position, O0⃗X0 points to P1, while O0⃗Z0 aligns with the axis of the driving motor, with its positive direction defined as Fig. 2 presents. For the coordinate system {O1X1Y1Z1}, its coordinate axis are parallel to coordinate system {O0X0Y0Z0}, and the distance between O1 and O0 is h. θ0 is the angle of ∠P1O1O0, and θ1, θ2, θ3 are respectively the angles between the three driving linkages and the O0⃗X0 direction, note that at initial position θ1 = 0◦, θ2 = 240◦, θ3 = 120◦.

Generally, in the design process, R1, R2, and h are structural parameters to be determined with high priority by taking the wrist actuators diameter into consideration. Once these three parameters are confirmed, the relationships between the parameters Rl, l1, and l2 of the arc-shaped end linkage can be calculated using (1). After Rl is further determined, the overall structural design can be completed.

 

θ0 = arctan(R1/h) Rl + l2 = h/cosθ0 Rl + l1 = R2

(1)



#### 3.1 Forward Kinematics

The kinematics of the parallel wrist involves solving the relationship between θ1, θ2, θ3 and the RPY angles of the parallel platform. For the forward kinematics part, the inputs are θ1, θ2, θ3, and the outputs are RPY angles of the parallel platform.

When three driving linkages rotate to θ1, θ2, and θ3, the arc-shaped end linkages P1P4, P2P5 and P3P6 rotate around O1⃗P1, O1⃗P2 and O1⃗P3 by φ1, φ2 and φ3 respectively. As can be seen from the geometric structure of the parallel wrist, the coordinates of points P1, P2, and P3 are Pi = (R1 cosθi,R1 sinθi,0)(i = 1,2,3).

The coordinates of points Pi(i = 1 ∼ 6) in their initial positions are denoted as Pi0. In the initial position, θ1 = 0, θ2 = 4π/3, and θ3 = 2π/3. It can be calculated that P40 = (−R2 sinθ1,R2 cosθ1,h), P50 = (−R2 sinθ2,R2 cosθ2,h), P60 = (−R2 sinθ3,R2 cosθ3,h).

When the parallel wrist moves to any position, the coordinates of points P4, P5, and P6 can be derived as (2).

 

- Pi+3.x = −R2 cosφi sinθi + R2 sinφi cosθ0 cosθi
- Pi+3.y = R2 cosφi cosθi + R2 sinφi cosθ0 sinθi Pi+3.z = R2 sinφi sinθ0 + h

(2)



(i = 1,2,3)

Thus vectors O1⃗P4, O1⃗P5, O1⃗P6 can be expressed in (3). O1P⃗i+3 = (Pi+3.x,Pi+3.y,R2 sinφi sinθ0)(i = 1,2,3) (3) The angle between each pair of these three vectors is 2π/3, and (4) can be obtained.

 

- O1⃗P4·O1⃗P5

- |O1⃗P4|·|O1⃗P5|

= cos 23π O1⃗P4·O1⃗P6 |O1⃗P6|·|O1⃗P6|

= cos 23π O1⃗P5·O1⃗P6

- |O1⃗P5|·|O1⃗P6|

(4)



= cos 23π

By solving the system of equations (4) simultaneously, the values of φ1, φ2 and φ3 can be calculated for any given inputs θ1, θ2, θ3. Given that this system of equations is nonlinear, the Newton-Raphson method is adopted herein for iterative solution.

This paper adopts the RPY method to describe the attitude of the parallel platform. Let {O2X2Y2Z2} denote the coordinate system of the parallel platform after rotation. Specifically, the platform undergoes rotations

around the X-axis, Y-axis and Z-axis of the original coordinate system {O1X1Y1Z1} by γ(Roll), β(Pitch) and α(Yaw) respectively. The rotation matrix describing the attitude transformation of the parallel platform is expressed in (5).

- RX(γ) =

 

1 0 0 0 cosγ −sinγ 0 sinγ cosγ

 ,

- RY (β) =

 

cosβ 0 sinβ 0 1 0

−sinβ 0 cosβ

 ,

- RZ(α) =

 

 ,

(5)

cosα −sinα 0 sinα cosα 0

0 0 1

1R2 = RZ(α)RY (β)RX(γ)

 

 

R11 R12 R13 R21 R22 R23 R31 R32 R33

=

The normal vector of the parallel platform is n = (A,B,C), where A, B, and C can be calculated by (6):

 

- A = (P5.y − P4.y)(P6.z − P4.z) − (P5.z − P4.z)(P6.y − P4.y)
- B = (P5.z − P4.z)(P6.x − P4.x) − (P5.x − P4.x)(P6.z − P4.z)
- C = (P5.x − P4.x)(P6.y − P4.y) − (P5.y − P4.y)(P6.x − P4.x)

(6)



- R12, R22, and R32 can be calculated by (7).  

- P4.x
- P4.y
- P4.z − h

  = 1R2

 

- P40.x
- P40.y
- P40.z − h

  = 1R2

 

0 R2 0

  (7)

Furthermore, R11, R21, and R31 can be calculated by (8).

 

- P5.x
- P5.y
- P5.z − h

  = 1R2

 

- P50.x
- P50.y
- P50.z − h

  = 1R2

 

−R2 sin 43π R2 cos 43π 0

  (8)

- R13, R23, and R33 can be calculated by (9).  

 

  = 1R2

  (9)

0

- A
- B
- C

√A2 + 0B2 + C2

Then, the RPY angles of the parallel platform can be solved by (10):

 

α = atan2(R21,R11) β = atan2(−R31, R322 + R332 ) γ = atan2(R32,R33)

(10)



#### 3.2 Inverse Kinematics

For inverse kinematics, the input is RPY angles of the parallel platform, and the output is θ1, θ2 and θ3. After given the RPY angles, the rotation matrix 1R2 can be solved through (5). Furthermore, the coordinates of points P4, P5, and P6 can be calculated using (7), (8), and (11).

 

  (11)

  = 1R2

 

  = 1R2

 

−R2 sin 23π R2 cos 23π 0

- P6.x
- P6.y
- P6.z − h

- P60.x
- P60.y
- P60.z − h

The values of φ1, φ2, φ3, θ1, θ2, and θ3 can be derived using (2).

##### 3.3 Jacobian Matrix This paper adopts the numerical method to solve the Jacobian matrix, which can be expressed as (12):

  

   (12)

 

  =

∂α ∂θ1

∂α ∂θ2

∂α ∂θ3

J11 J12 J13 J21 J22 J23 J31 J32 J33

∂β ∂θ1

∂β ∂θ2

∂β

J =

∂θ3 ∂γ

∂γ ∂θ2

∂γ ∂θ3

∂θ1

For a small step ∆θ, J11 can be calculated by (13).

∂α ∂θ1 ≈

α(θ1 + ∆θ,θ2,θ3) ∆θ

J11 =

(13)

The other elements of the Jacobian matrix J can be calculated in the same way.

As discussed previously, the iterative Newton-Raphson method inherently gives rise to rounding errors. In the numerical solution of the Jacobian matrix of the wrist, special consideration must be given to the effect of step size ∆θ on the precision of the results.

The motion process of the wrist under the conditions θ1 = 0,θ2 = 4π/3 and θ3 ranging from 2π/3 to π is simulated using Creo software. The RPY angles of the parallel platform were recorded at a step size of π/3000. A series of step sizes (1e − 1, 1e − 2, ..., 1e − 7) are selected to calculate both the maximum error and root mean square error (RMSE) of the elements in Jacobian matrix. Comparative analysis revealed that the smallest values of both maximum error and RMSE for matrix J occur when the step size ∆θ is 1e − 3. Consequently, a step size of ∆θ = 1e − 3 is chosen for solving the Jacobian matrix via the numerical method.

### 4 EXPERIMENTS

In this chapter, three experiments are conducted to verify the performance of ByteWrist in terms of flexibility, high integration and anthropomorphic. Prior to these experiments, an introduction to the ByteMini robotic system is provided.

- 4.1 Prototype of ByteMini To verify ByteWrist functionality, we integrate them into our 22-DoF dual-arm mobile robot, ByteMini (Fig.

- 3). Key design features include:

- 1. Arms:7-DoF in SRS (Spherical-Revolute-Spherical) configuration with ByteWrist as wrist modules.
- 2. Grippers:1-DoF grippers paired with RealSense D405 cameras for close-range vision.
- 3. Waist:1-DoF lifting mechanism in high stiffness for height adjustment.
- 4. Chassis:3-DoF omnidirectional mobile platform for flexible movement.
- 5. Head:2-DoF in Pitch and Yaw motion, integrated with a RealSense D457 camera for primary vision.

###### 6. Computing and Power:Dell T3280 computer as the main controller and 4.08 kWh battery for power supply.

[Figure 5]

Figure 3 Description of the ByteMini Robot.

#### 4.2 Wrist Motion Range

Owing to structural constraints, the motion range of ByteWrist must comply with the requirement specified in (14).

β2 + γ2 < 0.72 (14)

To verify the wrist range motion and high flexibility, this study designs a circular trajectory, enabling the wrist to move in accordance with (15).

β2 + γ2 = 0.68 α = 0

(15)

As illustrated in Fig. 4(a-c), the input of wrist movement and the actual position feedback are plotted for cycles T = 4s, 2s and 1s respectively. Postures of ByteWrist at 8 distinct moments are presented in Fig. 4(d-k), each corresponding to different pitch and yaw angles. Throughout the wrist movement, the yaw direction remains constant. Experimental results demonstrate that the robotic wrist exhibits flexible motion within this operational region.

In the actual control of the wrist, the three aforementioned motion exhibit a delay of approximately 0.06s, attributed to communication and computation latencies. Owing to variations in motion speed, the errors between the state and command curves are 0.064 rad, 0.127 rad, and 0.247 rad, respectively, when the pitch or roll angle is 0 rad. Consequently, in high-dynamic applications, the influence induced by such lag must be considered, whereas it can be neglected in low-speed control scenarios.

#### 4.3 Confined-Space Maneuverability

To verify the flexibility of ByteWrist in confined spaces, this study designs a comparative experiment of objects grasping inside a glove box between ByteMini and Kinova[22].

[Figure 6]

[Figure 7]

[Figure 8]

(a) T = 4 s (b) T = 2 s (c) T = 1 s

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

(d) Pitch=0.48, Roll=0.48 (e) Pitch=0.68, Roll=0 (f) Pitch=0.48, Roll=-0.48 (g) Pitch=0, Roll=-0.68

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

(h) Pitch=-0.48, Roll=-0.48 (i) Pitch=-0.68, Roll=0 (j) Pitch=-0.48, Roll=0.48 (k) Pitch = 0, Roll = 0.68 Figure 4 Motion Range of ByteWrist, here we define moving down and left (along the direction of the gripper) as pitch and roll positive direction (Unit: Radian).

###### A glove box shown in Fig. 5 is designed for this experiment. This box is constructed from transparent acrylic material, with dimensions of 1000mm (length) × 500mm (width) × 600mm (height). There are two access holes (each in diameter 200mm) on the front panel of the cabinet, with center-to-center distance in 500mm.

[Figure 17]

Figure 5 Glove Box Experimental Scenario Setup.

###### Nine grasping target objects were arranged inside the glove box, including a plastic doll, a plush bear, a bath

flower, and six donut toys. The specific distribution of these objects in the glove box is illustrated in Fig. 5. The design objective is to evaluate the difficulty of grasping tasks in different spatial regions through two access holes, thereby achieving the assessment of the maneuverability of robotic arms.

The grasping experiment procedure is as follows. The ByteMini robot is tele-operated to extend both arms into the glove box. Subsequently, its waist and mobile chassis are kept stationary, and the current pose of the arms are recorded as the initial pose. Next, the arms are tele-operated via Quest VR to attempt grasping the objects inside the glove box and placing them into the blue storage basket. Once the object is placed into the blue storage box, the robot arm is subsequently returned to its initial pose in preparation for the next grasping attempt. Throughout the procedure, the duration to grasp each object (between two initial poses) is recorded. The grasping sequence is as follows: plastic doll, plush bear, bath flower, and Donut Toys 1–6. As shown in Fig. 6, the ByteMini robot with ByteWrist is capable of achieving accurate grasping of all nine objects, with the time required for each grasp summarized in Table 2a.

[Figure 18]

[Figure 19]

[Figure 20]

(a) Plastic Doll (b) Plush Bear (c) Bath Flower

[Figure 21]

[Figure 22]

[Figure 23]

(d) Donut Toy 1 (e) Donut Toy 2 (f) Donut Toy 3

[Figure 24]

[Figure 25]

[Figure 26]

(g) Donut Toy 4 (h) Donut Toy 5 (i) Donut Toy 6 Figure 6 ByteMini Performance in the Confined-Space Grasping Experiment.

To enable comparison with the traditional serial wrist, a dual-arm robot featuring a fixed base is constructed in this study, as illustrated in Fig. 7. The robot two arms consist of Kinova Gen3, which utilizes a 7-DoF SRS configuration similar to ByteMini. The key distinction lies in Kinova adopting a serial configuration for the wrist three joints. A Robotiq 2F-85 gripper is affixed as the end effector of each arm. The identical grasping experiment is conducted using the Kinova dual-arm robot, with grasping times recorded, and the experimental results are summarized in Table 2b.

Between Table 2a and 2b, we can notice that for the same experiment task, the Kinova dual-arm robot takes approximately twice as long as ByteMini, which reflects the difference in the flexibility of the robots wrist. Three main reasons lead to Kinova’s longer grasping duration are as follows.

- 1. Larger Arm Angle Adjustment: For the grasping of the plastic doll, bath flower, donut toy 4, and donut

(a) Grasping Time by ByteMini (Unit: s)

Plastic Doll Plush Bear Bath Flower 20 49 21 Donut Toy 1 Donut Toy 2 Donut Toy 3 22 38 23 Donut Toy 4 Donut Toy 5 Donut Toy 6 17 27 17 Total Time 234

(b) Grasping Time by Kinova (Unit: s)

Plastic Doll Plush Bear Bath Flower 47 56 67 Donut Toy 1 Donut Toy 2 Donut Toy 3 64 51 75 Donut Toy 4 Donut Toy 5 Donut Toy 6 23 59 34 Total Time 476

Table 2 Grasping Time Comparison Between ByteMini and Kinova

[Figure 27]

Figure 7 The Kinova Dual-arm Robot Setup.

toy 6, the Kinova’s forearm—featuring a serial configuration and greater length—requires retraction from the glove box to adjust its posture prior to grasp completion. As illustrated in Fig. 8a, to compensate for the limited range of wrist rotation, the upper arm executes a large-angle rotational motion, resulting in increased grasping duration.

- 2. Gripper Camera Out of the Box: When attempting to grasp donut toy 1 and donut toy 3, the rotation of the Kinova’s wrist will cause the collision between the gripper camera and the inner front wall of the glove box, if the gripper camera remains in the glove box. Thus, the gripper camera must first be moved out of the glove box before grasping, which leads to longer grasping time.
- 3. Slight Collisions Occur: For the grasping of donut toy 2 and donut toy 5, slight collisions occur between the forearm links and the glove box due to the forearm’s series configuration and greater length, which brings more time for adjustment and recovery.
- 4.4 Dual-Arm Manipulation of Deformable Objects

GR-3[23, 24] is a large-scale vision-language-action (VLA) model[25–28], and ByteMini robots are deployed for GR-3 data collection and model rollout. The dexterous clothes-hanging task in GR-3 imposes high requirements on the capabilities of the robot, where the robot is required to achieve dual-arm collaboration in the chest area and perform manipulation of deformable objects with high precision and high dexterity.

As shown in Fig. 9, ByteMini is required to perform following actions in the clothes-hanging manipulation:

[Figure 28]

[Figure 29]

[Figure 30]

(a) Larger Arm Angle Adjustment. (b) Gripper Camera Out of the Box. (c) Slight Collisions Occur. Figure 8 Causes of Longer Grasping Duration for Kinova Dual-Arm Robot in the Confined Space Grasping Experiment.

pick up the clothes hanger with its left gripper, cooperate with both arms to hang the clothes left shoulder on the hanger, switch to holding the hanger with its right gripper, cooperate with both arms to hang the clothes right shoulder on the hanger, and hang the entire assembly onto the crossbar.

ByteMini has not only completed 116 hours of data collection for the clothes-hanging task, but also successfully achieved fully automated dexterous cloth manipulation, demonstrating the flexibility and robustness of ByteWrist based robotic arms. Last but not least, thanks to the design of ByteWrist, the clothes-hanging motion of the ByteMini robot exhibits remarkable antropomorphic, with the overall movement being smooth and natural, as illustrated in Fig. 9.

[Figure 31]

[Figure 32]

[Figure 33]

(a) Pick Up the Clothes Hanger. (b) Hang the Clothes Left Shoulder. (c) Switch the Holding Gripper.

[Figure 34]

[Figure 35]

[Figure 36]

(d) Hang the Clothes Right Shoulder. (e) Pick Up the Assembly. (f) Hang the Assembly on the Rail. Figure 9 GR-3 Task: ByteWrist Based Dual-Arm Manipulation of Deformable Objects.

### 5 CONCLUSIONS

This paper presents the design, modeling, and experimental validation of ByteWrist, a compact parallel robotic wrist developed for flexible operation in narrow and confined spaces. The research achievements and key conclusions are summarized as follows:

- 1. Innovative Structural Design: ByteWrist adopts a three-stage motor-driven parallel mechanism, combined with arc-shaped end linkages and a central supporting ball. This design achieves a balance between compactness and stiffness. The prototype test confirms that the wrist can stably realize continuous

- RPY motion, meeting the requirements of anthropomorphic manipulation in confined spaces.
- 2. Complete Kinematic Modeling: The forward and inverse kinematic models of ByteWrist are established. For forward kinematics, the Newton-Raphson method is used to solve the nonlinear equations, realizing

the mapping from driving linkage angles θ1,θ2,θ3 to parallel platform RPY angles. For inverse kinematics, the rotation matrix is derived based on given RPY angles to calculate the required driving linkage angles. Additionally, a numerical method with an optimized step size (∆θ = 1e − 3 ) is proposed to solve the Jacobian matrix, providing a theoretical basis for high-precision motion control.

- 3. Excellent Performance Validation: Experimental results demonstrate ByteWrist can achieve highdynamic and large-angle motion. Compared with the Kinova dual-arm robot with serial wrists, ByteMini (integrated with ByteWrist) features higher integration and greater flexibility. ByteMini completes 116 hours of data collection for dexterous clothes-hanging tasks and realizes fully automated operation, verifying ByteWrist’s ability to cooperate with dual arms for high-precision deformable object manipulation.

For future improvements, we will consider optimize the structure parameters for extending the motion range. Meanwhile more light-weight design and reliable electrical wire routing will also be considered.

### 6 ACKNOWLEDGMENT

The authors sincerely thank Jiajun Zhang, Mingyu Lei, Yang Liu and Hao Niu for setting up the Kinova dual-arm robot. Meanwhile, special thanks go to the GR-3 Team, who provides an excellent VLA model for dexterous clothing-hanging manipulation task.

### References

- [1] Liyana Wijayathunga, Rassau Alexander, and Chai Douglas. Challenges and solutions for autonomous ground robot scene understanding and navigation in unstructured outdoor environments: A review. Applied Sciences, 13.17:9877, 2023.

- [2] Klamt Tobias and et al. Flexible disaster response of tomorrow: Final presentation and evaluation of the centauro system. IEEE Robotics and Automation Magazine, 26.4:59–72, 2019.

- [3] Dragomir Nenchev, Tsumaki Yuichi, and Takahashi Mitsugu. Singularity-consistent kinematic redundancy resolution for the srs manipulator. In International Conference on Intelligent Robots and Systems (IROS), volume 2. IEEE/RSJ, 2009.

- [4] Paul Zsombor-Murray and Gfrerrer Anton. 3r wrist positioning-a classical problem and its geometric background. In Computational Kinematics: Proceedings of the 5th International Workshop on Computational Kinematics, volume 4. Springer Berlin, 2004.

- [5] Fan Hangbing, Wei Guowu, and Ren Lei. Prosthetic and robotic wrists comparing with the intelligently evolved human wrist: A review. Robotica, 40.11:4169–4191, 2022.

- [6] Xu Feng, Zi Bin, Yu Zhaoyi, Zhao Jiahao, and Ding Huafeng. Design and implementation of a 7-dof cable-driven serial spray-painting robot with motion-decoupling mechanisms. Mechanism and Machine Theory, 192:105549, 2024.

- [7] Peter Vischer and Clavel Reymond. Argos: A novel 3-dof parallel wrist mechanism. The International Journal of Robotics Research, 19.1:5–11, 2000.

- [8] Pang Zaixiang and et al. Design and analysis of a flexible, elastic, and rope-driven parallel mechanism for wrist rehabilitation. Applied Bionics and Biomechanics, 2020.1:8841400, 2020.

- [9] Wu Guanglei and Niu Bin. Dynamic stability of a tripod parallel robotic wrist featuring continuous end-effector rotation used for drill point grinder. Mechanism and Machine Theory, 129:36–50, 2018.

- [10] Wu Yuanqing and Carricato Marco. Design of a novel 3-dof serial-parallel robotic wrist: A symmetric space approach. Robotics Research, 1:389–404, 2017.

- [11] Sharafatdin Yessirkepov, Umurzakov Timur, and Folgheraiter Michele. Design and analysis of a parallel elastic shoulder joint for humanoid robotics application. IEEE Access, 2025.

- [12] Baggetta Mario and et al. Virtual and physical prototyping of a cable-driven compliant robotic wrist. IEEE/ASME Transactions on Mechatronics, 2025.

- [13] Pollen Robotics. Orbita : A 3-dof joint on reachy. Forum, 2020.
- [14] Pollen Robotics. Reachy 2 is the first open-source humanoid robot specifically designed for the development of embodied ai and real-world applications. Tech Report, 2024.
- [15] Cheng Min and et al. Development of a redundant anthropomorphic hydraulically actuated manipulator with a roll-pitch-yaw spherical wrist. Frontiers of Mechanical Engineering, 16.4:698–710, 2021.

- [16] Bai Quan and et al. Coordinated motion planning of the mobile redundant manipulator for processing large complex components. The International Journal of Advanced Manufacturing Technology, 121.9:6703–6721, 2022.

- [17] Da Song and et al. Modeling and control system experiment of a novel series three-axis stable platform. Mechanical Sciences, 15.1:209–221, 2024.

- [18] Marcelo Ang and Tourassis Vassilios. Singularities of euler and roll-pitch-yaw representations. IEEE Transactions on Aerospace and Electronic Systems, 3:317–324, 2007.

- [19] Paul Zsombor-Murray and Gfrerrer Anton. Mini cheetah: A platform for pushing the limits of dynamic quadruped control. In Mini cheetah: A platform for pushing the limits of dynamic quadruped control. IEEE, 2019.

- [20] Katz B.G. A low cost modular actuator for dynamic robots. Doctoral dissertation, 2018.
- [21] P.M Wensing, A Wang, S Seok, D Otten, J Lang, and S Kim. Proprioceptive actuator design in the mit cheetah: Impact mitigation and high-bandwidth physical interaction for dynamic legged robots. IEEE Transactions on Robotics, 33:509–522, 2017.

- [22] Campeau-Lecours Alexandre and et al. Kinova modular robot arms for service robotics applications. In Rapid Automation: Concepts, Methodologies, Tools, and Applications, pages 693–719. IGI global, 2019.

- [23] Cheang Chilam and et al. Gr-3 technical report. arXiv preprint, 2025.
- [24] Cheang Chilam and et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint, 2024.
- [25] Kawaharazuka Kento and et al. Vision-language-action models for robotics: A review towards real-world applications. TechRxiv, 2025.
- [26] Sapkota Ranjan and et al. Vision-language-action models: Concepts, progress, applications and challenges. arXiv preprint, 2025.
- [27] TianYu Xiang and et al. Parallels between vla model post-training and human motor learning: Progress, challenges, and trends. arXiv preprint, 2025.
- [28] Ma Yueen and et al. A survey on vision-language-action models for embodied ai. arXiv preprint, 2024.

