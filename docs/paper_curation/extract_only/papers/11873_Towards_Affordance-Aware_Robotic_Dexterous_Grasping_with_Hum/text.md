# arXiv:2508.08896v4[cs.RO]11Nov2025

## Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors

### Haoyu Zhao *1,2, Linghao Zhuang *1, Xingyue Zhao *2, Cheng Zeng5, Haoran Xu4, Yuming Jiang2, Jun Cen2,3,4, Kexiang Wang2, Jiayan Guo2, Siteng Huang†2,3,4, Xin Li2,3, Deli Zhao2,3, Hua Zou†1

1 Wuhan University 2 DAMO Academy, Alibaba Group 3 Hupan Lab 4 Zhejiang University 5 Tsinghua University

Seen Obj. HLS

###### Abstract

A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on lowlevel grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories. Here is our project page: https://afforddex.github.io/.

Unseen Cat. HLS

Seen Obj. AS

Unseen Cat. AS

Unseen Obj. HLS

Unseen Obj. AS

[Figure 1]

[Figure 2]

[Figure 3]

UniDexGrasp UniDexGrasp++ AffordDex (Ours)

Figure 1: Performance comparision among UniDexGrasp (Xu et al. 2023), UniDexGrasp++ (Wan et al. 2023), and our AffordDex, on the vision-based setting. we report human-likeness score (HLS) and affordance score (AS) across seen objects, unseen objects, and unseen categories. We also present a qualitative comparison, where AffordDex performs natural and safe grasping by avoiding the blade.

### 1 Introduction

Dexterous grasping, as a foundational capability for robotic manipulation, has garnered significant attention from both academia and industry (Zhao et al. 2024, 2025b). Compared to simpler end-effectors (e.g., parallel jaws, vacuum grippers), five-fingered dexterous hands closely resemble human hand structure, providing substantially enhanced flexibility, precision, and task adaptability (Zhong et al. 2025). Furthermore, anthropomorphic robots expedite the collection of rich human demonstration data via teleoperation (Li et al. 2025a). Consequently, this synergy has fueled rapid progress, with recent algorithms achieving high success rates in generalizing grasps to novel objects (Fang et al. 2022, 2020; Gou et al. 2021; Wang et al. 2021; Xu et al. 2023; Wan et al. 2023).

Due to the high degrees of freedom (DOFs) of dexterous hands, traditional motion planning-based methods (Andrews and Kry 2013; Bai and Liu 2014) struggle to handle such complex hand joint movements. Recent advancements in reinforcement learning (RL) (Wan et al. 2023; Mandikal and Grauman 2022; Christen et al. 2022; Nagabandi et al. 2020; Mandikal and Grauman 2021) have shown promising results in complex dexterous manipulation. However, the goal of grasping is not merely to lift an object. It involves alignment with human intent and preparation for subsequent manipulation tasks, such as avoiding the blade of a knife or preparing to open a bottle cap. Existing methods, while focused on low-level grasp stability metrics, largely overlook this crucial synthesis of affordance-aware positioning and human-like kinematics, limiting their utility in real-world, multi-step manipulation scenarios.

* Equal contributions. †Corresponding Author

In this work, we focus on the critical aspect of safety and functional correctness by modeling negative affordances—regions to be avoided, which provide clear, unambiguous negative constraints and thus simplify the learning problem. We propose AffordDex, a novel framework that learns a universal grasping policy that is both human-like in its motion and functionally aware of object affordances. We achieve this through a structured, two-stage training paradigm. In the first stage, we pre-train a base policy on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapts the general human-like motions from the pre-trained policy to specific objects. This refinement is critically guided by our proposed Negative Affordance-aware Segmentation (NAA) module, which provides explicit visual-geometric constraints on functionally inappropriate contact regions. Moreover, the training is enhanced with a teacher-student distillation framework, which leverages ground-truth state information to ensure the final vision-based policy is highly effective and robust.

As illustrated in Fig. 1, AffordDex produces grasps that are not only successful but also remarkably human-like and functionally correct, such as safely grasping a knife by its handle. Extensive experiments validate that our method significantly outperforms existing approaches across benchmarks of seen objects, unseen objects, and even objects across datasets. In summary, AffordDex makes the following contributions:

- • We propose AffordDex, a two-stage framework that synergistically and effectively integrates human motion priors with functional affordance constraints to achieve generalizable and anthropomorphic dexterous grasping.
- • We introduce a Negative Affordance-aware Segmentation (NAA) module that, by reformulating segmentation as a VLM-guided classification problem, provides explicit geometric constraints to prevent functionally improper grasps.
- • Extensive experiments demonstrate AffordDex achieves SOTA success rates across multiple levels of generalization while producing grasps that are qualitatively superior in human-likeness and functional appropriateness.

### 2 Related Work

#### 2.1 Dexterous Grasping

Robotic grasping (Fang et al. 2022, 2020; Gou et al. 2021; Wang et al. 2021) has been a longstanding research, aiming to enable robots to interact with objects reliably and adaptively. While significant advances have been made with simple parallel-jaw grippers (Fang et al. 2020; Mahler et al. 2019), their limited dexterity restricts adaptability to objects with intricate geometries. Dexterous, multi-fingered hands (Xu et al. 2023; Wan et al. 2023) offer a solution but pose a severe control challenge for traditional analytical methods (Bai and Liu 2014; Liu et al. 2021), motivating the shift towards learning-based approaches.

One paradigm decouples the grasping process into static grasp pose generation followed by a dynamic grasping

through trajectory planning or goal-conditioned reinforcement learning (RL) (Wan et al. 2023; Christen et al. 2022; Wang et al. 2025). For example, UniDexGrasp++ proposes geometry-aware curriculum learning and leverages the geometry feature for RL. However, these RL-based methods may produce physically unrealistic joint configurations. An alternative paradigm directly learns the entire grasping trajectory through expert demonstrations from humans or reinforcement learning agents (Xu et al. 2023; Liu et al. 2024; Huang et al. 2023; Lu et al. 2024; Zhang et al. 2024b,a). These approaches tend to achieve more natural motions but suffer from poor generalization to novel objects due to the limited diversity of demonstrations and inherent policy constraints. To address these failures, our AffordDex combines strong, human-derived motion priors to ensure natural movement with affordance-based guidance to achieve robust generalization, resulting in a policy that is both natural and functionally effective across a wide range of objects.

#### 2.2 Affordance Prediction

Affordance defines the action possibilities an object offers to an agent (Gibson 2014). In robotics, this translates to identifying object regions suitable for specific interactions, such as grasping, pushing, or lifting. Predicting such affordances is therefore critical for advanced visual understanding and robotic manipulation, as evidenced by extensive research (Li et al. 2019; Cao et al. 2020; Corona et al. 2020; Jiang et al. 2021; Lu et al. 2025; Shao et al. 2025). While pioneering works like GanHand (Corona et al. 2020) introduced generative models for multi-object on-table grasps, and GEAL (Lu et al. 2025) pioneered a dual-branch architecture for crossmodal (3D point cloud to 2D) representation learning, their learned affordances are often task- or category-specific. This inherent specialization limits their ability to generalize to novel objects or adapt to different downstream manipulation requirements. By contrast, humans exhibit exceptional proficiency in inferring universal affordances from visual cues (Zhao et al. 2025a). Inspired by this capability, AffordDex learns to infer functional affordances directly from multi-view rendered images of 3D objects, enabling a generalizable grasping policy that is not constrained to specific object categories or predefined tasks.

### 3 Methodology

To generate grasps with affordance-aware positioning and human-like kinematics, crucial for facilitating downstream manipulation, we propose a novel two-stage framework. The first stage establishes a strong human motion prior by pretraining a base policy πH, on a large-scale human motion dataset (Zhan et al. 2024) via imitation learning. This constrains the policy to a manifold of natural, human-like movements. In the second stage, we freeze the weights of πH and train a lightweight residual module via reinforcement learning (RL) to adapt these general motions to specific object interactions. This RL refinement stage is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which provides explicit constraints on where not to touch an object, and a teacher-student dis-

[Figure 4]

| | | |
|---|---|---|
|ol|icy<br><br>[Figure 5]|[Figure 6]|
| | | |

[Figure 7]

Robot State Object State Point Cloud Negative Affordance

[Figure 8]

P

Imitation Learning

[Figure 9]

[Figure 10]

Policy

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

| | | |
|---|---|---|
|o|licy<br><br>[Figure 15]| |
| | | |

Human hand trajectories

P

[Figure 16]

Human Hand Trajectory Imitating

[Figure 17]

State-based Input

Imitation Learning

Affordance Description

[Figure 18]

[Figure 19]

Add

[Figure 20]

Robot State Point Cloud Negative Affordance

Which areas are

Texture

[Figure 21]

unsuitable for grasping?

[Figure 22]

[Figure 23]

[Figure 24]

Policy

[Figure 25]

Blade edge.

[Figure 26]

Object

Vision-based Input

[Figure 27]

[Figure 28]

|[Figure 29]<br><br>[Figure 30]| |[Figure 31]|[Figure 32]|
|---|---|---|---|

[Figure 33]

###### SAM

CLIP Negative

Affordance

Negative Affordance-aware Segmentation Affordance-aware Residual Learning

Figure 2: Pipeline of AffordDex. To generate grasps with affordance-aware positioning and human-like kinematics, crucial for facilitating downstream manipulation, we propose a novel two-stage framework. The first stage establishes a strong human motion prior by training a base policy πH, on a human motion dataset via imitation learning. This constrains the policy to a space of natural, human-like movements. Subsequently, the second stage employs reinforcement learning (RL) to refine this coarse policy πH for precise, functional interaction. We fine-tune πH with a residual module that is guided by our Negative Affordance-aware Segmentation (NAA) module, which provides explicit constraints on where not to touch the object. The entire learning pipeline is further enhanced by a teacher-student distillation framework, leveraging privileged inputs to significantly boost the final grasping performance.

tillation framework that leverages privileged state information to significantly boost the final policy’s performance. An overview of our method is illustrated in Fig. 2.

#### 3.1 Human Hand Trajectory Imitating

In this stage, our objective is to learn a base policy πH, that captures the kinematic priors of natural human hand motions. We formulate this task as a reinforcement learning (RL) problem where the policy πH(at|StH) learns to generate dexterous hand action at based on the current state StH at time t. To facilitate the following fine-tuning stage, the state consists of robot state Rt, object state Ot, and point cloud representation of object Pt, i.e., StH = {Rt,Ot,Pt}.

Reward function. We design a reward function rH to promote both precise imitation of human hand trajectories and the motion stability. It is composed of two terms: a finger imitation reward rfingerH and a smoothness reward rsmoothH .

The finger imitation reward rfingerH encourages the dexterous hand to closely track the reference finger poses from human hand dataset. Following (Li et al. 2025b), we define this reward based on the distance between the corresponding keypoints F on the robot dexterous hand and the MANO

hand. The reward at time t is formulated as:

F

wf · exp −λf ∥jd,f − jh,f∥22 , (1)

rfingerH =

f=1

where jd,f is the position of the f-th keypoint on the dexterous hand, jh,f is its corresponding target position from the reference trajectory, wf is weight and λf is the decay rate.

The smoothness reward rsmoothH encourages energyefficient movements by penalizing excessive power consumption. This is computed as the element-wise product of joint velocities and applied torques. A detailed formulation of our reward function is available in Supp. Mat.

#### 3.2 Negative Affordance-aware Segmentation

A significant limitation of prior work in grasp synthesis (Xu et al. 2023; Wan et al. 2023; Zhong et al. 2025), is its neglect of the semantic and functional context of the interaction. A classic example is a knife: while its blade is geometrically stable for grasping, any such grasp is functionally incorrect and unsafe. To address limitation, we introduce the Negative Affordance-aware Segmentation (NAA) module to incorporate negative affordances—reasoning about which parts of an object should not be touched. The proposed NAA has the ability to operate in an open-vocabulary manner by harnessing the rich world knowledge embedded in Vision-Language Models (VLMs) (Radford et al. 2021; Achiam et al. 2023),

automatically benefiting from future progress in foundation models. This ensures that the generated grasps are not only geometrically stable but also semantically coherent and taskaware.

VLMs struggle to interpret non-textured 3D meshes, as these models primarily rely on rich visual cues learned from images. To bridge this gap, we first apply procedural texturing to the raw meshes by (Zhang et al. 2024c), which generates semantically plausible textures based on geometric analysis, ensuring robustness across different object shapes. Next, we render the textured object from six cardinal directions to create a multi-view image set I as a holistic visual representation. While this may not capture all concavities in highly complex objects, we found it provides a sufficient basis for affordance prediction for objects in the benchmark datasets, representing a practical trade-off between coverage and computational cost. We then query GPT-4V (Achiam et al. 2023) to elicit a detailed description of the object’s negative affordances.

VLMs (Radford et al. 2021) and Multimodal Large Language Models (MLLMs) (Achiam et al. 2023) excel at image-level understanding but struggle with the fine-grained spatial localization required for segmentation. To solve this, instead of asking CLIP (Radford et al. 2021) to find “blade part” from the image, we turn the segmentation task into a much simpler classification task. We generate a set of precise object-part masks Mi and use them as a visual prompt to let CLIP identify which mask in Mi has the highest semantic similarity to the textual description “blade part”.

Specifically, for each image Ii ∈ I, we prompt Segment Anything Model (SAM) (Kirillov et al. 2023) with a dense grid of points G overlaid on Ii, which prompts SAM to perform an exhaustive segmentation, identifying all potential objects and parts. The resulting collection of masks is then refined using Non-Maximum Suppression (NMS) to eliminate duplicates, yielding a clean candidate masks set Mi:

Mi = NMS(SAM(Ii,Gi)). (2) For each mask Mij ∈ Mi, we generate a visually

prompted image Iij by blurring regions outside the mask with Gaussian filter following (Yang et al. 2023). The

prompted image set {Iij} is then passed to CLIP along with the text query to compute a similarity score for each image-

text pair. The mask with the highest similarity score is selected as the final segmentation mask. The mask is then projected into 3D space to segment the corresponding regions of the object’s point cloud to get the negative affordance Nt, as shown in Fig. 3. Our NAA is an offline, one-time process. About 160 seconds per object on an RTX 4090 is a reasonable one-time trade-off.

#### 3.3 Affordance-aware Residual Learning

The negative affordance predicted from proposed NAA, we use a residual module R to refine the pre-trained policy πH. Since visual pose estimation is inherently less precise than using privileged state information, directly training an effective vision-based policy can be challenging. Therefore, we first train a state-based teacher policy πT which can access

##### Same Cat. Different Cat.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

A knife. A hammer. A candle. A camera.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

A knife.

A hammer. A lightbulb. A headphone.

Figure 3: Visualization of Negative Affordances Predicted by our NAA. The point cloud, highlighted in red, represents the negative affordances identified on various objects. These points denote regions that are functionally unsafe or inappropriate for grasping, such as a knife’s blade.

the ground-truth states of the environment, such as object states, to learn residual actions to refine the initial actions predicted by πH. Once the teacher policy πT finishes training, we use an imitation learning algorithm, DAgger (Ross, Gordon, and Bagnell 2011), to distill πT to a vision-based student policy πS that can access oracle information and let policy help and ease the vision-based policy learning.

State-based teacher policy. In this stage, the inputs are robot state Rt, object state Ot, the scene point clouds Pt, and predicted negative affordance Nt. Here the scene point cloud is fused by multi-view depth cameras. Our goal is to learn residual actions ∆at = πT(StT) with predicted negative affordance by PPO (Schulman et al. 2017). The final action is computed with an element-wise addition:

###### at = πH(StT) + πT(StT). (3)

Reward function. The reward function rT is defined as:

rT = −rdT − rgT + rsT − rnT (4)

where the grasp reward rdT penalizes the distance between the dexterous hand and the object, encouraging the hand to

maintain contact with the object surface for a secure grasp. The goal reward rgT penalizes the distance between the object and the target goal, and the success reward rsT provides a bonus when the object successfully reaches the goal. Also the negative affordance reward rnT penalizes the dexterous hand to approach the predicted negative affordance. The formal definitions of all rewards are available in our Supp. Mat.

Vision-based student policy. For vision-based policy, we only allow it to access information available in the real world, including robot state Rt, the scene points clouds Pt, and predicted negative affordance Nt. Then, we distill the

UniDexGrasp UniDexGrasp++

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

DexGrasp Anything Ours

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

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

- Figure 4: Qualitative Comparison on UniDexGrasp (Xu et al. 2023) and OakInk2 (Zhan et al. 2024). A comparison of grasps generated by our AffordDex with several baselines, including UniDexGrasp (Xu et al. 2023), UniDexGrasp++ (Wan et al. 2023), and DexGrasp Anything (Zhong et al. 2025).

teacher policy πT into vision-based student policy πS using DAgger (Ross, Gordon, and Bagnell 2011), i.e.,

πS = arg min

∥πT(StT) − πS(StS)∥, (5)

πS

where the state for the teacher policy StT = {Rt,Ot,Pt,Nt}, and the state for the student policy StS = {Rt,Pt,Nt}.

### 4 Experiments

#### 4.1 Datasets

UniDexGrasp (Xu et al. 2023). This dataset contains 3165 different object instances spanning 133 categories. Evaluation is conducted on these 3,200 seen objects, as well as on 140 unseen objects from seen categories and 100 unseen objects from unseen categories. Each environment is randomly initialized with one object and its initial pose, and the environment consists of a panoramic 3D point cloud Pt captured from the fixed cameras for vision-based policy learning.

OakInk2 (Zhan et al. 2024). This dataset record the manipularion processes with pose and shape of the human upperbody and objects. We pre-train our πH using about 2,200 right hand manipulation sequences in this dataset. Also we employ objects in OakInk2 to evaluate the generalization capabilities for grasping.

#### 4.2 Metrics

Following previous works (Xu et al. 2023; Wan et al. 2023; Wang et al. 2025), each object is randomly rotated and dropped onto the table to enhance the diversity of its initial poses. We report the success rate of grasp Succ, Humanlikeness Score HLS, and Affordance Score AS across all objects and grasp attempts. A grasp is considered successful if the object reaches the target goal within 200 steps in

simulator. The Human-likeness Score HLS assesses the anthropomorphic quality of the grasp, which is obtained by prompting the Gemini 2.5 Pro (Comanici et al. 2025) to analyze a visual sequence of the grasp execution. This metric is specifically to rate the resemblance of the dexterous hand’s motion to that of a typical human, yielding a quantitative measure of naturalness. The Affordance Score AS, in contrast, evaluates the functional correctness of the grasp by penalizing contact with inappropriate object parts. This metric is calculated using a point cloud of 100 “negative affordance” points sampled from our NAA. Specifically, the score is incremented by one for each fingertip that maintains a distance greater than 2cm from any point in this negative set, thus rewarding functionally sound grasps.

#### 4.3 Implementation Details

We conduct our experiments in IssacGym (Makoviychuk et al. 2021) simulator. During training, 4096 environments are simulated in parallel on an NVIDIA RTX 4090 GPU. For the network architecture, we use MLP with 4 hidden layers (1024,1024,512,512) for the policy network and value network in the state-based setting, and an additional PointNet+Transformer (Mu et al. 2021) to encode the 3D scene point cloud input in the vision-based setting. Other detailed hyperparameters are shown in our Supp. Mat.

Dexterous hand configuration. We use the Shadow Hand, which features 24 active degrees of freedom (DOFs). The wrist has 6 DOFs controlled by force and torque, while the fingers have 18 active DOFs controlled by joint angles. Specifically, the thumb has 5 DOFs, the little finger has 4, and the remaining three fingers each have 3. Additionally, each finger, excluding the thumb, includes a passive, noncontrolled DOF.

- Table 1: Quantitative comparisons on UniDexGrasp (Xu et al. 2023) and OakInk2 (Zhan et al. 2024). HLS denotes Humanlikeness Score, while AS is Affordance Score.

Method Seen Obj.

Unseen Obj. Seen Cat.

Unseen Obj. Unseen Cat.

OakInk2

Succ ↑ HLS ↑ AS ↓ Succ ↑ HLS ↑ AS ↓ Succ ↑ HLS ↑ AS ↓ Succ ↑ HLS ↑ AS ↓

State-Based Setting PPO (Schulman et al. 2017) 24.3 - - 20.9 - - 17.2 - - - - DAPG (Rajeswaran et al. 2017) 20.8 - - 15.3 - - 11.1 - - - - GSL (Jia et al. 2022) 57.3 - - 54.1 - - 50.9 - - - - ILAD (Wu, Wang, and Wang 2023) 31.9 - - 26.4 - - 23.1 - - - - UniDexGrasp (Xu et al. 2023) 79.4 6.9 12 74.3 6.4 15 70.8 6.3 18 68.4 5.9 18 UniDexGrasp++ (Wan et al. 2023) 87.9 5.4 28 84.3 5.2 26 83.1 5.0 27 79.6 4.9 28 DexGrasp Anything (Zhong et al. 2025) 71.2 - 20 69.1 - 18 67.3 - 22 65.9 - 24 AffordDex 89.2 8.6 4 87.7 8.5 7 85.2 8.1 9 82.2 8.2 10

Vision-Based Setting PPO (Schulman et al. 2017) 20.6 - - 17.2 - - 15.0 - - - - DAPG (Rajeswaran et al. 2017) 20.8 - - 15.3 - - 11.1 - - - - GSL (Jia et al. 2022) 54.1 - - 50.2 - - 44.8 - - - - ILAD (Wu, Wang, and Wang 2023) 27.6 - - 23.2 - - 20.0 - - - - UniDexGrasp (Xu et al. 2023) 73.7 6.2 16 68.6 6.1 18 65.1 6.0 17 62.8 5.6 20 UniDexGrasp++ (Wan et al. 2023) 85.4 5.4 29 79.6 5.1 25 76.7 4.8 28 74.4 4.7 29 AffordDex 87.0 8.3 10 82.8 7.8 14 79.2 8.0 15 77.3 7.8 13

- Table 2: Ablation Study on UniDexGrasp (Xu et al. 2023) in Seen Object.

HTI NAA Distillation Succ ↑ HLS ↑ AS ↓ State-Based Setting

85.4 5.2 27 ✓ 87.9 8.2 22

✓ ✓ 89.2 8.6 4

Vision-Based Setting

70.1 5.0 27 ✓ 84.9 5.6 28

✓ ✓ 85.8 7.2 13 ✓ ✓ 86.9 8.1 20 ✓ ✓ ✓ 87.0 8.3 10

- Table 3: Results on UniDexGrasp (Xu et al. 2023) in Seen Object in state-based setting.

thing (Zhong et al. 2025), diffusion-based dexterous grasp generation models.

Since DexGrasp Anything (Zhong et al. 2025) generates only the final static grasp pose rather than a grasping motion, the HLS is not applicable in this setting. To evaluate grasp robustness, we apply a random external force ranging from 0 to 200N to the object to simulate object’s gravity.

Tab. 1 compares our AffordDex with these SOTA methods using a universal model for dexterous robotic grasping across both state-based, vision-based and even across dataset settings. Our method achieves highest scores in grasping success rate, outperforming other state-of-the-art methods. This improvement stems from our proposed Human Hand Trajectory Imitating, where the policy learns to generate correct and stable grasp poses from human hand motion. This not only greatly enhances the grasp success rate but also leads to a significant improvement in our method’s human-likeness score (HLS). The significantly lower Affordance Score AS validates the effectiveness of our Negative Affordance-aware Segmentation module. This result indicates that the module successfully guides the policy away from functionally inappropriate regions, leading to grasps at the most suitable locations.

Method Succ ↑ HLS ↑ AS ↓ UniDexGrasp++ (Wan et al. 2023) 87.9 5.4 28

UniDexGrasp++ + HTI 88.2 7.8 23 UniDexGrasp++ + NAA 88.0 5.9 19

UniDexGrasp++ + HTI + NAA 88.8 8.0 12

#### 4.4 Comparison with SOTA Methods

As illustrated in Fig. 4, our method generates a diverse set of grasps. Crucially, it consistently identifies functionally appropriate grasp locations and forms natural hand postures. This combination of functional awareness and naturalness makes the generated poses highly effective for direct application in downstream manipulation tasks.

We evaluate AffordDex by first training a state-based policy and then distilling it into a vision-based one. For comparison, we compare AffordDex with several state-of-theart methods. These include RL and imitation learning algorithms like PPO (Schulman et al. 2017), DAPG (Rajeswaran et al. 2017), and ILAD (Wu, Wang, and Wang 2023). We also compare against methods with advanced learning paradigms such as GSL (Jia et al. 2022) (generalistspecialist), UniDexGrasp++ (Wan et al. 2023) (geometryaware curriculum learning), UniDexGrasp++ (Wan et al. 2023) which proposes a geometry-aware curriculumn learning and generalist-specialist learning, and DexGrasp Any-

#### 4.5 Ablation study

Unless otherwise specified, the ablation studies are conducted on the seen objects under the state-based setting.

Human Hand Trajectory Imitating. The results in Tab. 2 and Fig. 8 show the critical role of pre-training on human

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

###### w/o HTI w/ HTI

- Figure 5: Ablation Study on Human Hand Trajectory Imitating (HTI). Without the human motion prior, the policy converges to a solution that, while potentially successful, is kinematically awkward and non-humanlike.

w/o NAA w/ NAA

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

- Figure 6: Ablation Study on proposed NAA, which guides the policy to a correct and safte position. The higher Affordance Score (AS) for the NAA-guided grasp confirms its superior functional quality.

trajectories (HTI). When this imitation stage is omitted, the policy, while still capable of finding geometrically stable grasps, produces motions that are kinematically unnatural. This is quantitatively reflected in a sharp increase in the Human-Likeness Score (HLS). Such configurations are not merely an aesthetic issue, they can be inefficient, unpredictable, and detrimental to downstream tasks that require fluid, human-centric interaction.

NAA. As shown in Fig. 7, a naive approach combining an MLLM (Achiam et al. 2023) with SAM (Kirillov et al. 2023) (denoted as GPT+SAM) proves ineffective for this task. This baseline first uses the MLLM’s coarse localization ability to provide prompts to SAM. However, because MLLMs like GPT-4V (Achiam et al. 2023) excel at image-level understanding but struggle with the fine-grained spatial localization required for segmentation, this process often results in the segmentation of the entire object. In contrast, our NAA module solves this by converting the segmentation task into a simpler classification problem. By first using SAM to generate accurate mask proposals and then using CLIP to select the one with the highest semantic similarity to the negative affordance description, NAA achieves precise segmentation.

As shown in Fig. 6 and Tab. 2, the guidance from NAA results in a significant decrease in the Affordance Score AS, which indicates that the policy successfully learns to make contact at more rational and safer locations on the object. By generating functionally sound grasps, AffordDex greatly improves the feasibility of performing downstream tasks.

Teacher-student distillion. Without the teacher-student dis-

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Input Object w/ GPT+SAM w/ NAA

Figure 7: Ablation Study on proposed NAA, which has capability to segment fine-grained negative affordance.

tillion (Distillation) grasping accuracy decreases significantly. This is primarily due to the lack of privileged information guidance, which makes it challenging for the single-stage RL policy to learn the position to grasp. As shown in Tab. 2, the policy without teacher-student distillation demonstrates lower grasping success rate.

#### 4.6 Extension to Other Grasping Methods

Notably, our proposed modules demonstrate strong generalizability by significantly enhancing other RL-based methods, such as UniDexGrasp++ (Wan et al. 2023). Specifically, the Human Hand Trajectory Imitating (HTI) module markedly improves the naturalness and human-likeness of its generated poses. Simultaneously, the Affordance-aware Residual Learning, guided by negative affordances from our Negative Affordance-aware Segmentation (NAA), substantially boosts the semantic appropriateness of its grasp locations on the object, as shown in Tab. 3.

### 5 Conclusion

In this paper, we present AffordDex, a novel framework for generating dexterous grasps that are not only successful but also human-like and functionally correct. Our key insight is that the challenges of naturalness and functional correctness can be effectively decoupled and then synergized: a strong motion prior learned from human data constrains the policy to a manifold of natural poses, while a visual understanding of negative affordances guides the policy to safe and appropriate contact regions. Extensive experiments validate that this approach significantly outperforms stateof-the-art baselines in success rate, pose naturalness, and contact appropriateness. We believe this work lays a crucial foundation for more general-purpose embodied agents and opens new avenues for research in dexterous manipulation.

Limitation. A limitation of our approach stems from its reliance on a fixed set of six rendered views for negative affordance prediction, which can fail to capture all functionally relevant parts on geometrically complex or concave objects. This can lead to imprecise negative affordance segmentation due to occlusion. Future work could overcome this by adopting volumetric-based affordance learning on implicit 3D representations, which are inherently robust to viewpoint-specific occlusions.

### 6 Acknowledgement

This work was supported by Central Guidance for Local Science and Technology Development Fund (ZYYD2025QY19), an Bingtuan Science and Technology Program (2022DB005). This work was supported by DAMO Academy through Academy Research Intern Program.

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Andrews, S.; and Kry, P. G. 2013. Goal directed multi-finger manipulation: Control policies and analysis. Computers & Graphics, 37(7): 830–839.

Bai, Y.; and Liu, C. K. 2014. Dexterous manipulation using both palm and fingers. In International Conference on Robotics and Automation, 1560–1565. IEEE.

Cao, Z.; Gao, H.; Mangalam, K.; Cai, Q.-Z.; Vo, M.; and Malik, J. 2020. Long-term human motion prediction with scene context. In Proc. of European Conf. on Computer Vision, 387–404. Springer.

Christen, S.; Kocabas, M.; Aksan, E.; Hwangbo, J.; Song, J.; and Hilliges, O. 2022. D-Grasp: Physically plausible dynamic grasp synthesis for hand-object interactions. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 20577–20586.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Corona, E.; Pumarola, A.; Alenya, G.; Moreno-Noguer, F.; and Rogez, G. 2020. Ganhand: Predicting human grasp affordances in multi-object scenes. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 5031–5041.

Fang, H.; Fang, H.-S.; Xu, S.; and Lu, C. 2022. Transcg: A large-scale real-world dataset for transparent object depth completion and a grasping baseline. Robotics and Automation Letters, 7(3): 7383–7390.

Fang, H.-S.; Wang, C.; Gou, M.; and Lu, C. 2020. Graspnet1billion: A large-scale benchmark for general object grasping. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 11444–11453.

Gibson, J. J. 2014. The theory of affordances:(1979). In The people, place, and space reader, 56–60. Routledge.

Gou, M.; Fang, H.-S.; Zhu, Z.; Xu, S.; Wang, C.; and Lu, C. 2021. RGB matters: Learning 7-dof grasp poses on monocular rgbd images. In International Conference on Robotics and Automation, 13459–13466. IEEE.

Huang, S.; Wang, Z.; Li, P.; Jia, B.; Liu, T.; Zhu, Y.; Liang, W.; and Zhu, S.-C. 2023. Diffusion-based generation, optimization, and planning in 3d scenes. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 16750–16761.

Jia, Z.; Li, X.; Ling, Z.; Liu, S.; Wu, Y.; and Su, H. 2022. Improving policy optimization with generalist-specialist learning. In Proc. of Intl. Conf. on Machine Learning, 10104– 10119. PMLR.

Jiang, H.; Liu, S.; Wang, J.; and Wang, X. 2021. Handobject contact consistency reasoning for human grasps generation. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 11107–11116.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.Y.; et al. 2023. Segment anything. In Proc. of IEEE Intl. Conf. on Computer Vision, 4015–4026.

Li, H.; Zhao, Q.; Xu, H.; Jiang, X.; Ben, Q.; Jia, F.; Zhao, H.; Xu, L.; Zeng, J.; Wang, H.; et al. 2025a. TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation. arXiv preprint arXiv:2505.12748.

Li, K.; Li, P.; Liu, T.; Li, Y.; and Huang, S. 2025b. Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 6991–7003.

Li, X.; Liu, S.; Kim, K.; Wang, X.; Yang, M.-H.; and Kautz,

- J. 2019. Putting humans in a scene: Learning affordance in 3d indoor environments. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 12368–12376.

Liu, T.; Liu, Z.; Jiao, Z.; Zhu, Y.; and Zhu, S.-C. 2021. Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator. Robotics and Automation Letters, 7(1): 470–477.

Liu, Y.; Yang, Y.; Wang, Y.; Wu, X.; Wang, J.; Yao, Y.; Schwertfeger, S.; Yang, S.; Wang, W.; Yu, J.; et al. 2024. RealDex: Towards human-like grasping for robotic dexterous hand. arXiv preprint arXiv:2402.13853.

Lu, D.; Kong, L.; Huang, T.; and Lee, G. H. 2025. Geal: Generalizable 3d affordance learning with cross-modal consistency. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 1680–1690.

Lu, J.; Kang, H.; Li, H.; Liu, B.; Yang, Y.; Huang, Q.; and Hua, G. 2024. UGG: Unified generative grasping. In Proc. of European Conf. on Computer Vision, 414–433. Springer.

Mahler, J.; Matl, M.; Satish, V.; Danielczuk, M.; DeRose, B.; McKinley, S.; and Goldberg, K. 2019. Learning ambidextrous robot grasping policies. Science Robotics, 4(26): eaau4984.

Makoviychuk, V.; Wawrzyniak, L.; Guo, Y.; Lu, M.; Storey,

- K.; Macklin, M.; Hoeller, D.; Rudin, N.; Allshire, A.; Handa, A.; et al. 2021. Isaac gym: High performance gpubased physics simulation for robot learning. arXiv preprint arXiv:2108.10470.

- Mandikal, P.; and Grauman, K. 2021. Learning dexterous grasping with object-centric visual affordances. In international conference on robotics and automation, 6169–6176. IEEE.
- Mandikal, P.; and Grauman, K. 2022. Dexvip: Learning dexterous grasping with human hand pose priors from video. In Conference on Robot Learning, 651–661. PMLR.

Mu, T.; Ling, Z.; Xiang, F.; Yang, D.; Li, X.; Tao, S.; Huang, Z.; Jia, Z.; and Su, H. 2021. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. arXiv preprint arXiv:2107.14483.

Nagabandi, A.; Konolige, K.; Levine, S.; and Kumar, V. 2020. Deep dynamics models for learning dexterous manipulation. In Conference on Robot Learning, 1101–1112. PMLR.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In Proc. of Intl. Conf. on Machine Learning, 8748–8763.

Rajeswaran, A.; Kumar, V.; Gupta, A.; Vezzani, G.; Schulman, J.; Todorov, E.; and Levine, S. 2017. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087.

Ross, S.; Gordon, G.; and Bagnell, D. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of International Conference on Artificial Intelligence and Statistics, 627–635.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Shao, Y.; Zhai, W.; Yang, Y.; Luo, H.; Cao, Y.; and Zha, Z.-J. 2025. GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 17326–17336.

Wan, W.; Geng, H.; Liu, Y.; Shan, Z.; Yang, Y.; Yi, L.; and Wang, H. 2023. UniDexGrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning. In Proc. of IEEE Intl. Conf. on Computer Vision, 3891–3902.

Wang, C.; Fang, H.-S.; Gou, M.; Fang, H.; Gao, J.; and Lu, C. 2021. Graspness discovery in clutters for fast and accurate grasp detection. In Proc. of IEEE Intl. Conf. on Computer Vision, 15964–15973.

Wang, W.; Wei, F.; Zhou, L.; Chen, X.; Luo, L.; Yi, X.; Zhang, Y.; Liang, Y.; Xu, C.; Lu, Y.; et al. 2025. UniGraspTransformer: Simplified policy distillation for scalable dexterous robotic grasping. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 12199–12208.

Wu, Y.-H.; Wang, J.; and Wang, X. 2023. Learning generalizable dexterous manipulation from human grasp affordance. In Conference on Robot Learning, 618–629. PMLR.

Xu, Y.; Wan, W.; Zhang, J.; Liu, H.; Shan, Z.; Shen, H.; Wang, R.; Geng, H.; Weng, Y.; Chen, J.; et al. 2023. UniDexGrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 4737–4746.

Yang, L.; Wang, Y.; Li, X.; Wang, X.; and Yang, J. 2023. Fine-grained visual prompting. Proc. of Advances in Neural Information Processing Systems, 36: 24993–25006.

- Zhan, X.; Yang, L.; Zhao, Y.; Mao, K.; Xu, H.; Lin, Z.; Li, K.; and Lu, C. 2024. OakInk2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 445–456.

Zhang, H.; Christen, S.; Fan, Z.; Hilliges, O.; and Song, J. 2024a. Graspxl: Generating grasping motions for diverse objects at scale. In Proc. of European Conf. on Computer Vision, 386–403. Springer.

Zhang, H.; Christen, S.; Fan, Z.; Zheng, L.; Hwangbo, J.; Song, J.; and Hilliges, O. 2024b. ArtiGrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation. In International Conference on 3D Vision, 235–246. IEEE.

Zhang, H.; Pan, Z.; Zhang, C.; Zhu, L.; and Gao, X. 2024c. TexPainter: Generative Mesh Texturing with Multi-view Consistency. In ACM SIGGRAPH Conference, 1–11.

- Zhao, H.; Wang, H.; Zhao, X.; Fei, H.; Wang, H.; Long, C.; and Zou, H. 2025a. PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting. In Proc. of IEEE Intl. Conf. on Computer Vision, 5242–5252.

Zhao, H.; Yang, C.; Wang, H.; Zhao, X.; and Shen, W. 2024. SG-GS: Photo-realistic animatable human avatars with semantically-guided gaussian splatting. arXiv preprint arXiv:2408.09665.

Zhao, H.; Zeng, C.; Zhuang, L.; Zhao, Y.; Xue, S.; Wang, H.; Zhao, X.; Li, Z.; Li, K.; Huang, S.; et al. 2025b. HighFidelity Simulated Data Generation for Real-World ZeroShot Robotic Manipulation Learning with Gaussian Splatting. arXiv preprint arXiv:2510.10637.

Zhong, Y.; Jiang, Q.; Yu, J.; and Ma, Y. 2025. Dexgrasp anything: Towards universal robotic dexterous grasping with physics awareness. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 22584–22594.

### 7 Details about Our Method

Algorithm 1: Overall AffordDex Framework Require: Human hand motion dataset DH, object dataset

###### DO

Ensure: Final vision-based grasping policy πS

- 1: Stage 1: Human Hand Trajectory Imitating
- 2: Pre-train a base policy πH on DH via imitation learning to establish a human motion prior.
- 3: Stage 2: Affordance-aware Residual Learning and Distillation
- 4: Generate negative affordance point clouds {nO}O∈DO for all objects using the NAA module.
- 5: Learn a state-based teacher policy πT by fine-tuning πH with a residual module, guided by the negative affordances {nO}.
- 6: Distill the teacher policy πT into a vision-based student policy πS via a teacher-student framework (e.g., DAgger).
- 7: return Final vision-based policy πS.

### 8 Details about Baselines

PPO. We use Proximal Policy Optimization (PPO) (Schulman et al. 2017), a standard on-policy reinforcement learning (RL) algorithm, as the foundation for our RL-based training stages.

DAPG. To leverage expert data, we employ DemoAugmented Policy Gradient (DAPG) (Rajeswaran et al. 2017). This imitation learning (IL) method accelerates policy training by combining a policy gradient loss with a behavior cloning loss on expert demonstrations. These demonstrations are generated via motion planning.

GSL. We also include Generalist-Specialist Learning (GSL) (Jia et al. 2022), a three-stage training paradigm. It first trains a generalist policy, then fine-tunes specialized policies on difficult task subsets, and finally uses demonstrations from these specialists to train a final, more capable generalist via IL. For fair comparison, our GSL implementation uses PPO for the RL components and DAPG for the final IL stage.

ILAD. To further improve generalization, we compare against ILAD (Wu, Wang, and Wang 2023), an IL method that builds upon DAPG. It introduces an auxiliary objective that forces the policy to learn a geometric object representation from the same motion-planned demonstrations, enhancing its adaptability.

UniDexGrasp and UniDexGrasp++. UniDexGrasp (Xu et al. 2023) is a two-stage learning method for dexterous grasping. In the first stage, it trains a state-based teacher policy using Reinforcement Learning. To manage training across numerous objects, it introduces Object Curriculum Learning (OCL), which starts with a single object and gradually incorporates more objects from similar semantic categories. In the second stage, this proficient teacher policy

is distilled into a vision-based student policy using DAgger (Ross, Gordon, and Bagnell 2011), enabling it to operate from point cloud inputs.

Building upon this foundation, UniDexGrasp++ aims to significantly enhance the policy’s generalizability across thousands of object instances with diverse geometries. It argues that curriculum based on semantic categories can be suboptimal and instead proposes two novel, geometrycentric techniques: Geometry-aware Curriculum Learning (GeoCurriculum) and Geometry-aware iterative GeneralistSpecialist Learning (GiGSL). GeoCurriculum organizes the training progression based on object geometric features rather than categories, creating a more effective learning path for grasping. GiGSL further refines the policy by iteratively training a generalist model on all objects and specialist models on geometrically challenging subsets. These innovations lead to a more robust universal grasping policy that substantially outperforms its predecessor.

DexGrasp Anything. DexGrasp Anything (Zhong et al. 2025) is a recent method for generating high-quality, static dexterous grasp poses. It utilizes a diffusion-based generative model to address object diversity and hand complexity. Its core innovation is the direct integration of physical constraints (e.g., collision-freeness, stability) into both the training and sampling phases of the diffusion process. Since it generates only the final static grasp pose rather than a grasping motion, we apply a random external force ranging from 0 to 200N to the object to simulate object’s gravity.

For PPO, DAPG, GSL, ILAD, UniDexGrasp, and UniDexGrasp++, we maintain the same experimental settings as reported in the UniDexGrasp++ paper (Wan et al. 2023) to ensure a fair comparison. For the DexGrasp Anything baseline, we directly utilized the officially released, pretrained model weights provided by the authors.

### 9 Experiment Details

We use PPO to train Reinforcement learning. We use DAgger-based policy distillation to distill the state-based policy into a vision-based one. We set wf = 1 in Eq. 1. We follow FGVP and set NMS IoU threshold in NAA to 0.7 (robust in practice).

For UniDexGrasp in Seen Object, we conduct 5 independent runs using fixed random seed of 42, to assess the performance consistency, reported in the table below.

#### 9.1 Experiment Setup

State Definition. The full state of the teacher policy StT = {Rt,Ot,Pt,Nt}. The complete object point clouds are assumed to be perfectly accurate. Object states Ot, including positions, rotations, and velocities, are directly accessible. To accelerate the training process, we sample 1024 points from the object and the hand in the scene point cloud Pt.

The full state of the student policy StS = {Rt,Pt,Nt}. Partial object point clouds are reconstructed and segmented from depth data captured by five cameras around the table. The hand-object distance is computed using the partial object point cloud in the vision-based setting.

[Figure 95]

Figure 8: Illustration of the simulation environment.

- Table 4: Ablation Study on UniDexGrasp dataset (Xu et al.

2023) in Seen Object in state-based setting.

|Configuration<br><br>|Succ ↑ HLS ↑ AS ↓|
|---|---|
|λsmooth = 0.02 λsmooth = 0.05 λsmooth = 0.1<br><br>|89.0 7.9 4 89.2 8.6 4 88.2 8.5 6<br><br>|
|λfinger = 0.5<br><br>λfinger = 0.8<br>λfinger = 1.0<br><br><br>|87.8 8.5 4 89.2 8.6 4<br><br>88.5 8.2 6<br>|

Action Space. The action space consists of the motor commands for the 24 actuators of the dexterous hand. The first 6 actuators control the global position and orientation, while the remaining 18 control the finger joints. We normalize the action range to (−1,1).

Camera Setup. Following a similar approach to UniDexGrasp++ (Wan et al. 2023), five RGBD cameras are mounted around the table, as shown in Fig. 8. The cameras are positioned relative to the table center at coordinates (0.0,0.0,0.55), (0.5,0.0,0.15), (−0.5,0.0,0.15), (0.0,0.5,0.15), (0.0,−0.5,0.15), with their focal points aligned at (0,0,0.15). In the vision-based setting, the depth images captured by these cameras are fused to generate a scene point cloud, from which the partial point cloud of the object is segmented.

Reward Function for Human Hand Trajectory Imitating. The goal of our human hand trajectory imitation reward, rH is to encourage the agent to mirror the articulation and motion smoothness of a reference human trajectory. It is formulated as a weighted sum of two key components: a finger imitation reward and a smoothness reward.

The finger imitation reward rfingerH encourages the agent’s hand to accurately track the reference finger poses. We define this reward based on the squared Euclidean distance between corresponding keypoints on the agent’s hand and the human reference hand:

rfingerH =

F

wf · exp −λf ∥jd,f − jh,f∥22 , (6)

f=1

where jd,f is the position of the f-th keypoint on the dexterous hand, and jh,f is its corresponding target position from the reference trajectory. The parameters wf and λf are set according to the anatomical group to which keypoint f belongs. Specifically, we group keypoints into two levels:

- • For keypoints f in Level-1 (base joints), we use a stricter decay rate of λf = 50.
- • For keypoints f in Level-2 (middle joints), we use λf = 40.

The weights wf are configured to combine these rewards appropriately. This hierarchical parameterization allows the model to prioritize accurate positioning of the more critical base joints.

The smoothness reward rsmoothH is designed to alleviate jerky motions, penalizing the power exerted on each joint, defined as the element-wise product of joint velocities and torques

n

rtsmooth = −wsmooth

|τt,i · q˙t,i|, (7)

i=1

where wsmooth is a positive weighting coefficient, n is the total number of actuated joints, and τt,i and q˙t,i are the torque and angular velocity of the i-th joint at timestep t, respectively. The product τt,i · q˙t,i represents the instantaneous power exerted by the actuator of joint i. By taking the absolute value, we penalize any high-power action, including both strong acceleration and aggressive braking, thus encouraging the policy to learn energy-efficient and hardwarefriendly motions.

The final reward in this stage is:

rH = λsmoothrtsmooth + λfingerrfingerH , (8)

where we set λsmooth = 0.05 and λfinger = 0.8. We also conduct ablation studies on these weights in Tab. 4.

Reward Function for Affordance-aware Residual Learning. The reward function described in Eq.(4) of the main paper comprises four components: a grasp reward rdT, a goalreaching reward rgT, a success bonus rsT, and a negative affordance penalty rnT. Each component is detailed below.

The grasp reward rdT encourages the hand to approach and stay close to the object. It is defined as a penalty proportional to the distance between the hand and the object’s center:

rdT = λTd ∥pdex − pobj∥, (9) where pdex and pobj are the Cartesian position of the dexterous hand and the object, respectively. We set λTd to -1.

The goal reward rgT guides the object towards the target goal. It is structured as a penalty on the distance between the object and the goal:

rgT = λTg ∥pobj − pgoal∥, (10) where pgoal is the position of the target goal. We set λTg to -1.

A bonus is provided upon task completion. This is triggered when the object enters a small threshold radius around the goal:

###### rsT = λTs I(∥pobj − pgoal∥ < αs), (11)

- Table 5: Ablation Study on UniDexGrasp dataset (Xu et al.

2023) in Seen Object in state-based setting.

|Configuration|Succ ↑ HLS ↑ AS ↓|
|---|---|
|λTd = −0.5<br>λTd = −1<br><br>λTd = −2<br>|84.2 7.2 5 89.2 8.6 4 88.4 8.4 4<br><br>|
|λTg = −0.5<br><br>λTg = −1<br>λTg = −2<br><br><br>|87.9 8.1 4 89.2 8.6 4<br><br>88.9 8.3 6<br>|
|λTs = 0.5<br>λTs = 1<br><br>λTs = 2<br>|83.1 7.0 6 89.2 8.6 4 89.0 8.5 5<br><br>|
|λTn = −5 λTn = −10 λTn = −20<br><br>|89.1 8.5 16<br><br>89.2 8.6 4 87.1 8.0 0<br><br><br>|

where I(·) is the indicator function, which evaluates to 1 if the condition is true and 0 otherwise. αs is set to 0.05. We set λTs to 1.

The negative affordance reward rnT penalizes the dexterous hand fingertips to approach the predicted negative affordance:

∥pftip − pn∥ < αn , (12)

rnT = λTn

I min

pn∈Pn

f∈Kc

where αn is set to 0.03, Kc is the set of the hand’s fingertips, pftip is the position of fingertip f, and Pn is the set of points representing the negative affordance. We set λTn = −10. Ablation studies are conducted on these weights to evaluate their contribution. The results are presented in Hyperparameters part.

Human-likeness Score (HLS). The Human-likeness Score (HLS) is designed to quantify the anthropomorphic quality of the generated grasping motion. We leverage the advanced multi-modal capabilities of Gemini 2.5 Pro (Comanici et al. 2025) to serve as an expert evaluator.

For each grasp, a video sequence of the complete grasp execution is provided as input to the model. The model is then prompted with a specific set of instructions to assess the motion based on key kinematic criteria. The exact prompt provided to the model is as follows:

Expert Role: You are an expert in hand kinematics evaluation.

Task: Evaluate the similarity between the robotic hand motion in the simulation video and that of a real human hand, based on the following three criteria:

- • Motion trajectory
- • Velocity smoothness
- • Joint coordination

Output Format: Return only a valid JSON, following this format: { "score": <1-10> }.

VLM-based Negative Affordance Identification VLMs struggle to interpret non-textured 3D meshes, as these models primarily rely on rich visual cues learned from images. To bridge this gap, we first apply procedural texturing to the raw meshes using Tex-Painter (Zhang et al. 2024c), which generates semantically plausible textures based on geometric analysis. Next, we render the textured object from six cardinal directions to create a multi-view image set, I, as a holistic visual representation. While this may not capture all concavities in highly complex objects, we found it provides a sufficient basis for affordance prediction for the objects in our benchmark datasets.

We then query GPT-4V (Achiam et al. 2023) with the multi-view images to elicit a description of the object’s negative affordances. The exact prompt is as follows:

Expert Role: You are an expert in identifying which part of a physical object should not be touched, especially in robotic grasping tasks.

Task: You will be given 6-view images of an object. Your task is:

- • Identify one non-touchable part.
- • Infer the object’s identity from the image name.

Output Format: Return one sentence only, following this format: "This is a . of should not be touched."

Points selected in Negative Affordance-aware Segmentation The core idea is to systematically prompt SAM with a dense, regular grid of points covering the entire image. This process ensures that objects of various sizes and locations are likely to be ”hit” by at least one point prompt, triggering SAM to segment them. The procedure consists of the following steps:

- 1. Grid Definition: We define a regular grid of points to be overlaid on the input image. The density of this grid is controlled by a single hyperparameter, g = 16, which represents the number of points along each of the image’s dimensions. This results in a total of g2 keypoints.
- 2. Keypoint Generation: For an input image I with width W and height H, the set of grid points G is generated.

The coordinates (xi,yj) for each point are calculated by uniformly spacing them across the image dimensions. The formula for a point at grid position (i,j) is:

G = (xi,yj) | xi =

i g + 1 · W, yj =

j g + 1 · H

(13) for all i,j ∈ {1,2,...,g}. Using g + 1 as the denominator ensures that the points are placed within the image boundaries and not on the absolute edges.

- 3. Prompting SAM: Each of the g2 points in the set G is used as an individual positive point prompt for SAM. This yields a large collection of raw segmentation masks,

Mraw = SAM(I,G), where many masks may be redundant or highly overlapping.

4. Filtering with NMS: To produce a refined set of unique proposals, we apply Non-Maximum Suppression (NMS) to the raw masks. The NMS algorithm filters out duplicate masks based on the Intersection over Union (IoU) of their corresponding bounding boxes, retaining only the most confident and distinct object proposals.

Hyperparameters. As shown in Tab. 5, we conduct a series of ablation studies to validate our design choices and analyze the model’s sensitivity to key hyperparameters. Our investigation focuses on the weights of our proposed reward function. The goal is to find a robust set of parameters that balances efficient learning with task performance.

The results indicate that our chosen default configuration is robust and well-justified. For the grasp reward weight λTd and the goal reward weight λTg , we observe an optimal value of -1.0. Deviating from this value, by either decreasing the penalty to -0.5 or increasing it to -2.0, leads to a degradation in both Succ and HLS. This suggests a critical balance: a penalty that is too weak fails to prevent undesirable actions, while an overly strong penalty can cause the policy to focus myopically on avoiding penalties rather than achieving a stable grasp, ultimately degrading task success. Similarly, for the bonus reward weight λTs , the optimal value is 1.0. A lower weight of 0.5 fails to sufficiently incentivize the target behavior, resulting in a significant drop in performance. Conversely, increasing the weight to 2.0 offers no additional benefit and slightly hinders performance, indicating that an excessive bonus can also be detrimental. Similarly, the negative affordance weight λTn requires careful tuning. An overly strong penalty can make the agent too conservative, causing it to avoid the target object altogether in its effort to steer clear of negative affordances. Conversely, an insufficient penalty fails to effectively deter the agent from frequently approaching these undesirable regions.

### 10 Future Experiments

We chose simulation to enable large-scale generalization experiments. The policy is vision-based and avoids privileged information, designing it for sim-to-real transfer, which is a key future goal.

