# arXiv:2506.07961v2[cs.RO]14Oct2025

[Figure 1]

[Figure 2]

## BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models

### Peiyan Li1,2,3,†, Yixiang Chen1,3, Hongtao Wu2,†,∗, Xiao Ma2,†, Xiangnan Wu1 Yan Huang1,3,4, Liang Wang1,3, Tao Kong2, Tieniu Tan1,3,5,∗

1CASIA 2ByteDance Seed 3UCAS 4FiveAges 5NJU †Project Lead, ∗Corresponding Author

### Abstract

Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low data efficiency. In this paper, we introduce a new paradigm for constructing 3D VLAs. Specifically, we first pre-train the VLM backbone to take 2D images as input and produce 2D heatmaps as output. Using this pre-trained VLM as the backbone, we then fine-tune the entire VLA model while maintaining alignment between inputs and outputs by: (1) projecting raw point cloud inputs into multi-view images, and (2) predicting heatmaps before generating the final action. Extensive experiments show that the resulting model, BridgeVLA, can learn 3D manipulation both efficiently and effectively. BridgeVLA outperforms state-of-the-art baselines across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 95.4% on 10+ tasks with only 3 trajectories per task, while other VLA methods such as π0 fail completely.

Project Page: https://bridgevla.github.io/ Corresponding Email: wuhongtao.123@bytedance.com; tnt@nlpr.ia.ac.cn

### 1 Introduction

Leveraging pre-trained vision-language models (VLMs) [1–4] for developing large vision-language-action (VLA) models has become a promising method for learning generalizable and robust manipulation policies [5–9]. However, most VLA models only incorporate 2D image inputs and require extensive efforts on data collection. On the other hand, 3D robot policies leverage 3D structural priors in model design and demonstrate exceptional sample efficiency in learning complex 3D robot manipulation tasks [10–14]. Can we develop a unified 3D VLA model which combines the effectiveness of VLA models with the efficiency from 3D policies?

Although there have been some works exploring integrating 3D information into VLMs for developing 3D VLA

Previous framework

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

| |
|---|

Language

3D Actions

| |
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

... ... ...

... ... ...

Previous

>>>> >>>>

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

VLM

3D-VLA

| |
|---|

| |
|---|

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

###### ... ... ... ...

3D info

[Figure 27]

[Figure 28]

“Put the ring on the maroon

"Find all instances

[Figure 29]

3D Projection

2D Heatmap

spoke"

of frock"

2D Finetune 2D Pretrain

Our framework

3D actions

[Figure 30]

[Figure 31]

BridgeVLA

3D Actions

Real World Simulation

Reshape & Upsample

Reshape & Upsample

Image Instructions

... ...

... ...

>>>> >>>>

[

VLM BridgeVLA

###### Real-World Generalization with ≤ 10 Trajectories

SoTA on RLBench

SoTA on

COLOSSEUM

... ...

###### ... ...

[Figure 32]

[Figure 33]

[Figure 34]

| |
|---|

| |
|---|

88.2

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

"Find all instances

81.4

| |
|---|

SuccessRate(%)

SuccessRate(%)

“Put the ring

64.0

62.9

on the maroon

56.7

Background Lighting Distractor

of frock"

49.4

spoke"

35.4

27.9

[Figure 39]

[Figure 40]

[Figure 41]

| |
|---|

| |
|---|

| |
|---|

Text

2D Image

3D Scene

1D Feature

Combination Height Category

Token

Token

Token

Token

- Figure 1 Overview. BridgeVLA is a novel 3D VLA model that aligns the input and output within a unified 2D image space. It is pre-trained on object grounding using 2D heatmaps and fine-tuned on action prediction for 3D manipulation. Experiment results in both simulation and the real world show that it is able to learn 3D manipulation both efficiently and effectively.

models [15, 16], these works typically convert actions into token sequences that do not have spatial structure and use next-token prediction to predict actions. This strategy fails to take advantage of the 3D structural priors as previous efficient 3D policies [10–14] that align the observation input and action output into a unified space, therefore leading to poor sample efficiency. Another significant challenge in developing 3D VLA models lies in the misalignment between the 3D inputs used in action fine-tuning and the 2D image inputs used in original VLM pre-training, causing a large distributional shift from the original VLM pre-training.

To tackle the challenges mentioned above, as inllustrated in Fig. 1, we present BridgeVLA, a novel 3D VLA model that achieves remarkable sample efficiency and strong generalization capabilities. To ensure input alignment with the pre-trained VLM backbone, BridgeVLA transforms a 3D point cloud observation into multiple 2D images captured from different orthographic projection views [13, 14]. To leverage the structural priors of the 3D input, BridgeVLA is trained to predict 2D heatmaps for translational action prediction. The

- 2D heatmaps, generated from the tokens corresponding to the projection images, share the same resolution as these images, aligning the input observations and output actions within a unified spatial structure. Given that the original VLM is pre-trained to predict token sequences, which is incompatible with our VLA’s 2D heatmap output, we also introduce a scalable pre-training method, which trains the model to ground objects with heatmaps conditioned on text inputs. This pre-training method equips the VLM with the capabilities to predict heatmaps before downstream fine-tuning for policy learning. Overall, our design aligns the input and output within a shared 2D space in both pre-training and fine-tuning.

We perform extensive experiments in both simulation and the real world to evaluate the proposed method. Results show that BridgeVLA is able to learn 3D manipulation both efficiently and effectively. It outperforms state-of-the-art baseline methods in RLBench [17], improving the average success rate from 81.4% to 88.2%. In COLOSSEUM [18], it showcases strong performance in challenging generalization settings, boosting the success rate from 56.7% to 64.0%. In GemBench [19], it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, we evaluate on seven different settings, spanning from

visual perturbations to manipulating objects from unseen categories. BridgeVLA surpasses a state-of-the-art method by 32% on average and demonstrates strong performance in generalizing to multiple out-of-distribution settings. Notably, BridgeVLA is able to achieve a success rate of 96.8% on 10+ tasks using only 3 trajectories per task for training, highlighting its superb sample efficiency. In summary, the contributions of this paper are threefold:

- • We introduce BridgeVLA, a novel 3D VLA model that efficiently and effectively learns 3D robot manipulation with a vision-language model via input-output alignment with 2D heatmaps.
- • We propose a scalable pre-training method to equip the model with the capability to predict heatmaps conditioned on text inputs via object grounding.
- • We conduct extensive experiments in both simulation and real-world environments to thoroughly evaluate the proposed method. Results show that BridgeVLA outperforms state-of-the-art methods in both settings and achieves exceptional sample efficiency in real-robot experiments.

### 2 Related Work

Language-Conditioned Visuomotor Policies. Most language-conditioned visuomotor policies employ transformers to process 2D visual inputs and directly generate 3D actions for manipulation [5–9, 20–25]. In these works, leveraging pre-trained vision-language models (VLMs) for developing large vision-language-action (VLA) models has become popular for its effectiveness on learning complex manipulation [5–9]. However, such 2D image-based policies typically require significant efforts on data collection, often needing hundreds of trajectories per task to learn effectively. On the other hand, 3D manipulation policies hold great potential for efficient learning by taking advantage of the spatial structure inherent in the 3D inputs. A popular line of works take as inputs point cloud data [11, 12, 26–28]. For example, Act3D [12] proposes to create a 3D feature cloud by lifting image features to the observation point cloud and predicts translational actions via classification for 3D points in the observation space. Another line of works utilize voxels to represent the observation space and predict translational actions within the voxel space, unifying the input observation and output actions within the same space [10, 29]. Recently, RVT [13] and RVT-2 [14] propose to leverage orthographic projection of 3D point clouds to convert 3D signals to 2D images to avoid high computational cost on processing 3D inputs. Different from the above methods, our method aims to unify the effectiveness of VLA models and the efficiency of 3D policies within a single cohesive framework, combining the best of both worlds.

###### 3D Vision-Language-Action (VLA) Models. While 2D VLA models have been extensively studied, 3D VLA models [15, 28, 30, 31] remain relatively under-explored. Zhen et al. [15] build 3D-VLA on top of a large language model (LLM) and train the model to perform 3D reasoning, multi-modal goal generation, and robot planning. Lift3D [30] proposes to enhance 2D foundation models (e.g., DINOv2 [32]) with implicit and explicit

- 3D robotic representation for learning 3D manipulation policies. FP3 [28] leverages a transformer to fuse the information from point clouds, proprioceptive states, and language instructions. PointVLA [31] utilizes a VLM and a point cloud encoder to process 2D images and 3D point clouds, respectively. The embeddings from both encoders are injected into an action expert for action prediction. SpatialVLA [16] introduces Ego3D position encoding to inject 3D information into 2D image observation and adaptive action grids to represent robot movement in a more transferable way. Our method is different from the above methods in that it is designed in a way to take advantage of the spatial structure of 3D inputs in action prediction. In addition, it bridges the gap between the 2D image inputs of pre-trained VLMs and the 3D inputs by projecting the 3D inputs into multiple 2D images instead of injecting 3D information into the VLMs. Such design enables it to simultaneously leverages the broad knowledge in the VLM backbone and the spatial structure priors embedded in 3D inputs. 3 BridgeVLA

###### 2D Heatmap Pre-training

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Freeze Weights Unfreeze

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

Rearrange

Convex Upsample

[Figure 59]

Preprocess

[Figure 60]

SigLIP

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Weights Image

Gemma

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Token Text Token

"Find all

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

instances

[Figure 79]

[Figure 80]

Tokenizer

of frock"

- 2D Detection Data
- 3D Point

Copy

[Figure 81]

Weights

Copy

###### 3D Action Fine-tuning

Weights

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

Orthographic

Projection

Rearrange

[Figure 96]

[Figure 97]

Convex

[Figure 98]

[Figure 99]

[Figure 100]

SigLIP

[Figure 101]

Upsample

[Figure 102]

Gemma

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

"Put the ring on

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

the maroon spoke"

Tokenizer

[Figure 113]

[Figure 114]

MLP Rotation Gripper Collision

[Figure 115]

Cloud Data

- Figure 2 Model Architecture. (a) 2D Heatmap Pre-training: we train BridgeVLA on 2D object detection datasets. The model takes as inputs an image and a language describing the target object and outputs a 2D heatmap which highlights regions of interest that correspond to the target object. Note that the bounding box shown here is for illustrative purposes only; it is not present in the image when input to the model. (b) 3D Action Fine-tuning: the model takes as inputs three orthographic projection images of a 3D point cloud and a language instruction. It outputs

- three 2D heatmaps, which highlight the position of the end-effector in the next keyframe across all three views. For the remaining action components, it uses an MLP to process the image feature tokens to predict the rotation action, gripper action, and collision flag of the next keyframe.

###### 3.1 Preliminaries

BridgeVLA aims to learn a multi-task 3D robot manipulation policy π, which maps the observation o and a language instruction l to an action a:

π : (o,l)  → a (1)

We assume access to a set of expert demonstrations D = {τi}Ni=1 containing N trajectories. And each trajectory contains a language instruction and a sequence of observation-action pairs, i.e., τi = {li,(oi1,ai1),...,(oiH,aiH)}. The observation o is one or multiple RGB-D images captured from one or multiple viewpoints. Following prior works [10, 12, 13], the action a consists of a 6-DoF end-effector pose T ∈ SE(3), a target gripper state g ∈ {0,1}, and a collision flag c ∈ {0,1} of the next key frame. The collision flag c indicates whether the motion planner should avoid collisions while moving towards the target pose. A key frame typically captures important or bottleneck steps in a trajectory (detailed in appendix B.1) [33]. BridgeVLA operates through an iterative process: 1) predicting the action at conditioned on the current observation ot and instruction l, 2) moving to the predicted next keyframe pose Tt using a sampling-based motion planner [34–36], 3) updating observation and repeating until task completion or reaching a maximum step Hmax.

As illustrated in Fig. 2, BridgeVLA employs a dual-phase training recipe. During pre-training, it is trained to predict 2D heatmaps on object detection datasets. During fine-tuning, point clouds are projected into multiple 2D images as inputs to the VLM backbone. The model is trained to predict 2D heatmaps for estimating the translational action and other action components. This design aligns the input and output within a shared 2D space in both pre-training and fine-tuning.

###### 3.2 2D-Heatmap Pre-training

The VLM backbone was originally pre-trained to predict token sequences without spatial structure. To equip it with the same ability to predict heatmaps as downstream policy learning, we introduce a pre-training stage which trains the model to ground target objects via heatmaps. Concretely, we leverage the 120K object detection split of RoboPoint [37] as our pre-training dataset. For each image, we construct the ground-truth heatmap Hgt from the bounding boxes of all objects of interest. Specifically, for each object, we construct a

probability map with spatial truncation:

Higt(x) =

pi(x) if pi(x) ≥ pmin 0 otherwise

(2)

where x = (u,v) denotes the pixel position, pi(x) = exp −∥x − xi∥2/2σ2 , xi is the center of the object bounding box, and pmin is a probability threshold. For all the objects of interest, we fuse the probability map of all objects via averaging and normalization to obtain Hgt:

N

Havg(x)

1 N

Higt(x) (3)

Hgt(x) =

, where Havg(x) =

x∈Ω Havg(x)

i=1

where Ω denotes the pixel space. Please refer to Fig. 9 for samples of the ground-truth heatmaps.

As illustrated in Fig. 2, we input an image along with the text prompt describing the objects of interest into the VLM backbone of BridgeVLA. In this paper, we employ PaliGemma [1] as the VLM backbone, which consists of a SigLIP vision encoder [38] and a Gemma transformer backbone [39]. During its pre-training, PaliGemma takes as input one or multiple 2D images together with a prefix text (e.g., a question about the image) and outputs a suffix text (e.g., an answer to the question). While the model uses causal attention for predicting suffix text tokens, it adopts bidirectional attention for the image tokens and the prefix text tokens. This allows the image tokens to fuse information from the prefix text.

To predict the heatmap, we first rearrange the output image tokens according to their patch positions to reconstruct the spatial feature grid. A convex upsampling block [40] then converts the grid into a heatmap with the same resolution as the input image. Unlike fixed methods (e.g., bilinear or nearest-neighbor), this upsampling module learns pixel-wise interpolation weights, allowing for finer spatial detail recovery. The whole pipeline is trained with a cross-entropy loss to predict heatmaps that localize the position of all objects of interest in the image. We emphasize that the proposed pre-training strategy outputs a spatially aware 2D heatmap, in contrast to the conventional next-token-prediction used in prior works [15, 16]. Moreover, this approach is highly scalable, as it can, in principle, leverage any vision-language datasets that can be formulated as a heatmap prediction tasks, such as keypoint detection and semantic segmentation.

###### 3.3 3D Action Fine-tuning

During fine-tuning, we first reconstruct a point cloud of the scene from the RGB-D images captured from calibrated cameras. To align with the 2D image input of the VLM backbone, we render three orthographic projection images of the point cloud from three viewpoints (top, front, and right) and use these images as the input images for the VLM backbone as in RVT [13] and RVT-2 [14]. These images, along with the task instruction, are then fed into the pre-trained VLM backbone to generate a heatmap for each of the three views. Importantly, we do not incorporate any additional information (e.g., robot states) during the VLM forward pass to minimize the distribution shift between pre-training and fine-tuning.

For translational actions, we back-project the heatmaps of all three views to estimate the scores of all 3D point grids distributed uniformly across the robot workspace. The position of the 3D point with the highest score determines the end-effector’s translation in the next keyframe. Similar to previous works [13, 14], we use Euler angles to represent rotational actions where each axis is discretized into 72 bins. To predict the rotation, binary gripper action, and collision avoidance flag, we integrate features from global and local contexts. For the global feature, max-pooling is applied to the output tokens of each inputted orthographic projection image, resulting in three tokens in total – one for each view. For the local feature, we extract a token from the heatmap peak of each view, also resulting in three tokens in total. All these tokens are concatenated and passed through MLP to predict the rotation action, gripper action, and collision avoidance flag.

Following the approach in prior works [14, 29], BridgeVLA adopts a coarse-to-fine refinement strategy for accurate action prediction. After the initial prediction on the original point cloud, we zoom in and crop the point cloud with a cuboid centered at the predicted translation. A second forward pass is performed on the cropped, zoomed-in point cloud. The predicted action from the second pass is used for execution.

Overall Task Success Rate (%)

Push Buttons Image-BC (CNN) [10, 41] 1.3 11.72 0.0 0.0 0.0 0.0 0.0 4.0 0.0 0.0

Close Jar

Drag Stick

Insert Peg

Meat off Grill

Open Drawer

Place Cups

Place Wine

Avg. SR (%) ↑

Avg. Rank ↓

Models

Image-BC (ViT) [10, 41] 1.3 12.19 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

C2F-ARM-BC [10, 29] 20.1 10.72 24.0 24.0 4.0 20.0 20.0 0.0 8.0 72.0 HiveFormer [42] 45.3 8.47 52.0 76.0 0.0 100.0 52.0 0.0 80.0 84.0

PolarNet [26] 46.4 7.61 36.0 92.0 4.0 100.0 84.0 0.0 40.0 96.0 PerAct [43] 49.4 7.0 55.2±4.7 89.6±4.1 5.6±4.1 70.4±2.0 88.0±5.7 2.4±3.2 44.8±7.8 92.8±3.0 Act3D [12] 65.0 4.89 92.0 92.0 27.0 94.0 93.0 3.0 80.0 99.0

RVT [13] 62.9 4.92 52.0±2.5 99.2±1.6 11.2±3.0 88.0±2.5 71.2±6.9 4.0±2.5 91.0±5.2 100.0±0.0 3D Diffuser Actor [11] 81.3 2.67 96.0±2.5 100.0±0.0 65.6±4.1 96.8±1.6 89.6±4.1 24.0±7.6 93.6±4.8 98.4±2.0

RVT-2 [14] 81.4 2.75 100.0±0.0 99.0±1.7 40.0±0.0 99.0±1.7 74.0±11.8 38.0±4.5 95.0±3.3 100.0±0.0 BridgeVLA w/o heat 31.4 10.06 49.3±2.3 65.3±2.3 0.0±0.0 81.3±4.6 74.7±10.1 1.3±2.3 32.0±14.4 54.7±6.1

BridgeVLA w pos 56.2 5.97 96.0±0.0 58.7±6.1 26.7±2.3 96.0±0.0 97.3±2.3 14.7±4.6 81.3±8.3 86.7±2.3 BridgeVLA 88.2 2.03 100.0±0.0 100.0±0.0 88.0±2.8 100.0±0.0 100.0±0.0 58.4±10.0 88.0±2.8 98.4±2.2

Turn Tap Image-BC (CNN) [10, 41] 0.0 8.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0 8.0

Models Put in

Put in Drawer

Put in Safe

Screw Bulb

Slide Block

Sort Shape

Stack Blocks

Stack Cups

Sweep to Dustpan

Cupboard

Image-BC (ViT) [10, 41] 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 16.0 C2F-ARM-BC [10, 29] 0.0 4.0 12.0 8.0 16.0 8.0 0.0 0.0 0.0 68.0 HiveFormer [42] 32.0 68.0 76.0 8.0 64.0 8.0 8.0 0.0 28.0 80.0

PolarNet [26] 12.0 32.0 84.0 44.0 56.0 12.0 4.0 8.0 52.0 80.0 PerAct [43] 28.0±4.4 51.2±4.7 84.0±3.6 17.6±2.0 74.0±13.0 16.8±4.7 26.4±3.2 2.4±2.0 52.0±0.0 88.0±4.4 Act3D [12] 51.0 90.0 95.0 47.0 93.0 8.0 12.0 9.0 92.0 94.0

RVT [13] 49.6±3.2 88.0±5.7 91.2±3.0 48.0±5.7 81.6±5.4 36.0±2.5 28.8±3.9 26.4±8.2 72.0±0.0 93.6±4.1

- 3D Diffuser Actor [11] 85.6±4.1 96.0±3.6 97.6±2.0 82.4±2.0 97.6±3.2 44.0±4.4 68.3±3.3 47.2±8.5 84.0±4.4 99.2±1.6 RVT-2 [14] 66.0±4.5 96.0±0.0 96.0±2.8 88.0±4.9 92.0±2.8 35.0±7.1 80.0±2.8 69.0±5.9 100.0±0.0 99.0±1.7

BridgeVLA w/o heat 5.3±2.3 0.0±0.0 58.7±22.7 2.7±2.3 64.0±0.0 4.0±4.0 0.0±0.0 0.0±0.0 32.0±4.0 40.0±10.6

BridgeVLA w pos 10.7±2.3 78.7±2.3 97.3±4.6 16.0±4.0 72.0±0.0 21.3±8.3 17.3±2.3 4.0±4.0 53.3±2.3 84.0±0.0 BridgeVLA 73.6±4.6 99.2±1.8 99.2±1.8 87.2±6.6 96.0±2.8 60.8±7.7 76.8±8.7 81.6±3.6 87.2±1.8 92.8±3.3

Table 1 Results on RLBench. The "Avg. Rank" column reports the average rank of each method across all 18 tasks, where lower values indicate better overall performance. "BridgeVLA w/o heat" refers to the ablated version that directly predicts actions without using intermediate heatmaps. "BridgeVLA w pos" refers to the ablated version that incorporates position features into the image features. BridgeVLA achieves the best performance in 10 out of the 18 tasks.

The training loss during fine-tuning consists of four components: L = Ltrans + Lrot + Lgripper + Lcollision (4)

Similar to pre-training, Ltrans is a cross-entropy loss that supervises the heatmap prediction for translational actions. The ground-truth heatmap for each orthographic view is the normalized single-object probability map defined in Eq. 2, where xi represents the projected pixel position of the ground-truth end-effector position in the next keyframe. As we discretize the Euler angles for rotation into bins, we also apply cross-entropy loss in Lrot to supervise rotation prediction. For gripper action and collision avoidance, we use the binary cross-entropy loss in Lgripper and Lcollision as supervision. To enhance geometric robustness, random rigid-body transformations are applied jointly to the point cloud and the ground-truth action during training. Additional training details can be found in Appendix A.

- 4 Experiments

In this section, we perform extensive experiments in both simulation and the real world to evaluate the proposed method. Through the experiments, we aim to answer five questions:

- Q1: How effectively does BridgeVLA learn 3D robot manipulation compared to state-of-the-art methods when sufficient data is available?
- Q2: Does BridgeVLA learn more efficiently than existing state-of-the-art methods when data is limited (e.g., 3 trajectories per task)?
- Q3: How robust is BridgeVLA in handling visual disturbances (e.g., distractors, background, and lighting)?

- Q4: How well does BridgeVLA generalize to novel object-skill combinations and objects from previously unseen categories?
- Q5: Are our architectural designs (e.g., predicting heatmaps before outputting actions) truly useful when constructing 3D VLA?

###### 4.1 Simulation Experiments

###### 4.1.1 Experiments on RLBench

Setup. RLBench [17] implements tasks in CoppeliaSim [44] using a Franka Panda robot mounted with a parallel-jaw gripper. The observation contains four RGB-D images captured from four calibrated cameras positioned at the front, left shoulder, right should, and wrist. Following previous works [10–14], we perform experiments on 18 tasks from RLBench. These tasks span 1) non-prehensile manipulation (e.g., slide block to target), 2) pick-and-place (e.g., stack cups), and 3) high-precision insertion (e.g., insert peg). Each task is provided with 100 expert demonstrations. And each demonstration is paired with language instruction and multiple keyframes. Models are evaluated via binary success rates over 25 trials per task, with a maximum of 25 action steps per trial.

Baselines. We compare BridgeVLA with multiple baselines. (1) Image-BC (CNN) and Image-BC (ViT) [41] are two 2D baseline methods which predict the actions directly from 2D images using CNN and ViT as the backbone, respectively. (2) C2F-ARM-BC [29] predicts the next keyframe action in the voxel space with a coarse-to-fine strategy. (3) PerAct [10] also operates in the voxel space and predicts the action with a perciever transformer [43]. (4) HiveFormer incorporates historical information using a unified multi-modal transformer architecture. (5) PolarNet employs PointNext [45] to encode the 3D scene and predicts both heatmaps and offsets for all points to estimate translational actions. (6) Act3D [12] predicts the next keyframe action by selecting the point with the highest score from a set of randomly sampled points in the workspace. (7) 3D Diffuser Actor [11] generates 3D trajectories via a diffusion process conditioned on 3D observation and language instructions. (8) RVT [13] uses multi-view transformer to aggregate information from multiple orthographic views of the point cloud observation. (9) And RVT-2 [14], the current state-of-the-art method, further improves the precision of its prior via a coarse-to-fine strategy.

Results. In total, we evaluate BridgeVLA five times to minimize statistical bias. The results are shown in

- Table 1. BridgeVLA outperforms all the comparing baseline methods, achieving an average success rate of 88.2% and an average rank of 1.9 across all the 18 tasks, establishing a new state of the art in this benchmark. These results address Q1, demonstrating the effectiveness of BridgeVLA in learning complex 3D manipulation tasks. We highlight that BridgeVLA outperforms the best baseline method by a large margin in Insert Peg (88.0% vs 40.0%) and Sort Shape (60.8% vs 35.0%). These two tasks demand highly precise alignment between the peg and hole and the block and sorter, respectively. The high success rates of our method showcase its strong capabilities of learning precise manipulation which is highly desirable in many industrial applications. Among the 18 tasks, BridgeVLA performs the worst in Place Cups, despite surpassing all the comparing baseline methods. We hypothesize this is because the target keypoints are often occluded in all orthographic projection views, which makes learning and prediction more challenging. In the future, we plan to explore dynamically selecting the projection views for rendering to avoid this problem.

###### 4.1.2 Experiments on COLOSSEUM

Setup. To systematically evaluate the generalization capabilities of BridgeVLA, we further evaluate on the COLOSSEUM benchmark [18]. The COLOSSEUM benchmark is an extension to the RLBench benchmark. The model is trained on the data from the original RLBench benchmark but evaluated in environments spanning 12 axes of perturbations. These perturbations, which are unseen during training, encompass changes in object texture, color, and size, backgrounds, lighting, distractors and camera poses. In total, the COLOSSEUM creates 20,371 unique task perturbations instances to comprehensively evaluate the generalization capabilities of the model. Specifically, our evaluation includes three steps: 1) train the model with the original RLBench data without perturbations (100 trajectories per task) on 20 tasks, 2) evaluate each task over 25 trials per

Overall Success Rate (%)

Models Avg. SR (%) ↑ Avg. Rank ↓ All Perturbations MO-COLOR RO-COLOR MO-TEXTURE RO-TEXTURE MO-SIZE

R3M-MLP[46] 0.8 5.71 0.6 0.4 0.0 0.0 0.0 1.8 MVP-MLP[47] 1.6 5.0 0.8 1.2 0.0 0.4 0.0 4.44

PerAct[43] 27.9 3.71 7.2 24.0 29.2 28.8 17.71 35.6 RVT[13] 35.4 3.28 6.4 26.0 31.3 44.8 41.1 35.3

RVT-2[14] 56.7 1.92 15.6 ± 0.8 53.0 ± 0.9 54.6 ± 0.6 59.7 ± 0.7 56.7 ± 1.4 60.9 ± 0.9

BridgeVLA (Ours) 64.0 1.07 18.7 ± 2.2 60.5 ± 1.1 63.8 ± 0.1 63.5 ± 1.5 68.4 ± 3.3 69.3 ± 1.0

Models RO-SIZE Light Color Table Color Table Texture Distractor Background Texture RLBench Camera Pose

R3M-MLP[46] 0.0 1.0 1.4 0.2 1.6 1.2 2.0 0.8 MVP-MLP[47] 0.0 1.6 1.6 1.0 3.8 2.2 2.0 2.6

PerAct[43] 29.3 29.1 30.4 23.2 27.1 33.5 39.4 36.3 RVT[13] 40.5 34.0 30.0 45.2 18.8 46.4 53.4 42.2

RVT-2[14] 53.4 ± 1.5 58.0 ± 1.1 62.6 ± 0.9 56.6 ± 0.9 60.8 ± 0.5 68.7 ± 1.1 68.8 ± 1.3 64.4 ± 0.5 BridgeVLA (Ours) 61.7 ± 0.8 69.7 ± 1.2 75.7 ± 0.9 71.3 ± 0.7 51.8 ± 1.5 74.8 ± 1.0 73.1 ± 0.2 73.8 ± 0.3

- Table 2 Results on the COLOSSEUM Benchmark. The table shows the success rates across 14 generalization settings. The “Avg. Rank” column reports the average rank of each method across all perturbations, where lower values indicate better overall performance. Compared to the state-of-the-art baseline, BridgeVLA improves the average success rate by 7.3%.

perturbation, 3) compute the average success rate of all evaluated tasks for every perturbation. Besides the 12 types of perturbations, we also evaluate on basic variations from the original RLBench (denoted as RLBench in Tab. 2), and a more challenging setting which combines all the 12 types of perturbations (denoted as All Perturbations in Tab. 2).

Baselines. We compare BridgeVLA with five baseline methods. R3M-MLP and MVP-MLP are two 2D methods that utilize pre-trained visual encoders to process observation images and an MLP for action prediction. Specifically, R3M-MLP uses R3M [46] that is pre-trained on large-scale egocentric human videos; MVP-MLP uses MVP [47] that is pre-trained on millions of in-the-wild data. Both visual encoders show strong adaptability on various robotics tasks in both simulation and the real world. We also compare with

- three 3D methods introduced in Sec. 4.1.1, i.e., PerAct [10], RVT [13], and RVT-2 [14].

- Results. Results are shown in Tab. 2. BridgeVLA outperforms all the comparing baseline methods in terms of average success rate, significantly outperforming the best baseline method by 7.3%. Among all the 14 evaluated perturbations, our method ranks the best among all methods in 13 of them. These results address Q3, showcasing that BridgeVLA possesses strong robustness against visual perturbation. More detailed results can be found in Tab. 6 and 7.

###### 4.1.3 Experiments on GemBench

Setup. To further evaluate the generalization capabilities of BridgeVLA, we perform experiments on the GemBench benchmark. GemBench [19] is a hierarchical generalization benchmark built on the RLBench simulator [17]. Its training set contains 16 tasks (31 variations) covering seven core action primitives—press, pick, push, screw, close, open, and stack/put. The test set consists of 44 tasks (92 variations), categorized into four increasingly challenging settings:

- L1 (Novel Placements): L1 consists of the original 16 tasks (31 variations). The object placements are randomized within the workspace. In addition, chromatic distractors are introduced to test the ability to handle additional visual complexity.
- L2 (Novel Rigid Objects): L2 involves 15 unseen tasks (28 variations) that require interaction with 8 novel rigid objects using learned primitives. The generalization capabilities are evaluated across two categories: novel object-color compositions and novel object shapes.
- L3 (Novel Articulated Objects): L3 consists of 18 unseen tasks (21 variations) that involve interacting with articulated objects. It evaluates the generalization capabilities across three categories: novel action-part compositions, novel object instances, and novel object categories.

###### Method Average L1 L2 L3 L4

Hiveformer [42] 30.4 60.3 ± 1.5 26.1 ± 1.4 35.1 ± 1.7 0.0 ± 0.0 PolarNet [26] 38.4 77.7 ± 0.9 37.1 ± 1.4 38.5 ± 1.7 0.1 ± 0.2 3D Diffuser Actor [11] 44.0 91.9 ± 0.8 43.4 ± 2.8 37.0 ± 2.2 0.0 ± 0.0 RVT-2 [14] 44.0 89.1 ± 0.8 51.0 ± 2.3 36.0 ± 2.2 0.0 ± 0.0 3D-LOTUS [19] 45.7 94.3 ± 1.4 49.9 ± 2.2 38.1 ± 1.1 0.3 ± 0.3 3D-LOTUS++ [19] 48.0 68.7 ± 0.6 64.5 ± 0.9 41.5 ± 1.8 17.4 ± 0.4 BridgeVLA (Ours) 50.0 91.1 ± 1.1 65.0 ± 1.3 43.8 ± 1.2 0.0 ± 0.0

- Table 3 Results on GemBench. We show the average success rates on the four evaluation settings of GemBench. BridgeVLA establishes a new state of the art on this benchmark, achieving an average success rate of 50.0%.

L4 (Novel Long-Horizon Tasks): L4 includes 6 complex long-horizon tasks (12 variations) that require combining multiple actions to finish a whole task.

Baselines. In total, we compare with six baseline methods. 3D-LOTUS [19] processes point cloud inputs through a language-conditioned point cloud transformer architecture [48]. It showcases notable multitasking capabilities and high training efficiency. Its enhanced variant, 3D-LOTUS++ [19], integrates the generalization capabilities of large-scale models into 3D-LOTUS with a modular architecture consisting of three components: (1) LLM-based task planning [49], (2) VLM-based object grounding [50, 51], and (3) motion control inherited from 3D-LOTUS. We also compare with four methods introduced in Sec. 4.1.1, i.e., Hiveformer [42], PolarNet [26], 3D Diffuser Actor [11], RVT-2 [14]

- Results. Results are shown in Tab. 3 and more detailed results can be found in Appendix B.4. BridgeVLA consistently outperforms all the comparing baseline methods in terms of average success rate across the four evaluation settings. Notably, BridgeVLA achieves state-of-the-art results in both the L2 and L3 settings, demonstrating strong generalization capabilities, addressing Q4. However, similar to most baseline approaches, BridgeVLA exhibits limited performance in the L4 setting, where each task comprises multiple sub-tasks. In the future, we plan to explore leveraging large language models (LLMs) for long-horizon task decomposition and further improve the performance in such setting.

###### 4.2 Real-Robot Experiments

Setup. In this section, we perform real-robot experiments to validate the effectiveness of BridgeVLA in the real world. Our real-robot setup includes a Franka Research 3 robot arm mounted with a parallel-jaw gripper (Fig. 3). A static ZED 2i depth camera is used to provide the colored point cloud observation. In total, we evaluate on 13 tasks (see Tab. 12 for a full list of tasks). These tasks ranges from simple pick-and-place to complex long-horizon tasks, requiring the robot to open a drawer and put items into the drawer. Each task contains 3-9 keyframes (see Fig. 7 and 8 for visualization). For each task, we collect 10 expert trajectories for training.

In total, we design 7 different settings to comprehensively evaluate our model’s performance. (1) Basic: The model is evaluated in environments that are similar to the training data. (2) Distractor: Distractor objects that are visually similar to at least one target object are added to the scene. (3) Lighting: The model is tested in a visually distinct lighting condition in which the lights are turned off. (4) Background: Three different tablecloths are used to change the background. (5) Height: All objects for manipulation are placed on a drawer that is 9.5cm high. (6) Combination: We combine objects and skills that are not paired together in the training datasets. That is, while the objects (e.g., red block and green plate) and skill (e.g., place A in B) are seen during training, the instruction that pairs them together is novel (e.g., place the red block in the green plate). In total, we evaluate 13 novel object-skill combinations (Fig. 11 and 12). (7) Category: To test whether BridgeVLA is able to transfer the broad knowledge from pre-training to downstream policy learning, we evaluate on manipulating objects from categories that are unseen in the robot training data. In total, we test 7 novel objects (Fig. 13). Distractor, Lighting, Background, and Height aim to evaluate the robustness

Put the coke can in the top shelf SpatialVLA(50) [16] 1/10 1/10 5/10 6/10 3/10 1/10 2/10

Put the soda can in the bottom shelf

Put the giraffe in the lower drawer

Place the red block in the blue plate

Put the RedBull can in the top shelf

Put the RedBull can in the bottom shelf

Method

Press Sanitizer

- SpatialVLA(10) [16] 0/10 0/10 0/10 2/10 0/10 0/10 0/10 π0 [52] 0/10 0/10 2/10 1/10 0/10 1/10 0/10

- ACT [24] 2/10 2/10 3/10 2/10 3/10 1/10 2/10

- RVT-2 [14] 10/10 8/10 8/10 10/10 9/10 10/10 10/10

- BridgeVLA 9/10 9/10 10/10 10/10 10/10 10/10 10/10 Method

Place the orange block in the green plate

Place the red block in the purple plate

Place the yellow block in the green plate

Put the zebra in the upper drawer

Put the zebra in the lower drawer

Put the wolf in the upper drawer

Average

SpatialVLA(50) [16] 6/10 3/10 5/10 2/10 0/10 2/10 28.5% SpatialVLA(10) [16] 1/10 1/10 0/10 0/10 0/10 0/10 3.1% π0 [52] 0/10 0/10 1/10 0/10 0/10 0/10 3.8% ACT [24] 2/10 3/10 4/10 1/10 2/10 1/10 22.3% RVT-2 [14] 10/10 9/10 9/10 7/10 8/10 9/10 90%

- BridgeVLA 10/10 10/10 10/10 9/10 10/10 9/10 96.9%

- Table 4 Per-task Success Rate in the Basic Setting. Except for SpatialVLA(50), which was trained with 50 trajectories, all other methods were trained with 10 trajectories. BridgeVLA outperforms all baseline methods, achieving an almost perfect success rate of 96.9%.

against visual disturbances, while Combination and Category evaluate the generalization capabilities for unseen instructions.

To demonstrate BridgeVLA’s advantages over existing manipulation policy, we compare it with four types of representative methods:

- 1) SpatialVLA [16]: A state-of-the-art 3D VLA model that incorporates 3D information through Ego3D positional encoding and leverages Adaptive Action Grids to accelerate inference.
- 2) π0 [52]: A state-of-the-art 2D VLA model pretrained on a large-scale cross-embodiment dataset. It adopts a vision-language model (VLM) backbone and employs a flow matching action expert to generate final actions.
- 3) ACT [24]: A state-of-the-art 2D non-VLA model using a Conditional Variational Autoencoder (CVAE) to model action distributions. Though effective for fine-grained manipulation, ACT does not support language conditioning, so we train a separate single-task model for each task, which should theoretically perform better than a multi-task version.
- 4) RVT-2 [14]: A state-of-the-art 3D non-VLA model performing the best in our simulation experiments. (See Sec. 4.1)

Results. We first compare BridgeVLA with these baselines on the basic setting. For every task, we evaluated every baseline over 10 trials to ensure statistical robustness. For fair comparison, we photographed each test scene and manually aligned the scenes across all methods. Results are provided in Tab. 4. As we can see, most methods completely fails when given only 10 trajectories per task except two 3D related methods: RVT-2 and BridgeVLA. Notably, although SpatialVLA also utilises 3D information, its data efficiency is still very low. Even when the data are increased to 50 trajectories per task, its success rate is still much lower than BridgeVLA, which indicates only adding 3D information is not enough for constructing 3D VLA model and a carefully designed network architecture is still very important. To assess the data efficiency of BridgeVLA, we also train the model with only 3 trajectories per task. Remarkably, despite the limited data, BridgeVLA achieves a success rate of 95.4% in Basic, matching the performance achieved with 10 trajectories per task. This result underscores the data efficiency of the proposed method, directly addressing Q2. Detailed per-task results are provided in Appendix C.5 and observations about the baselines are detailed in Appendix C.2.

Considering that only RVT-2 and BridgeVLA have a good performance in basic setting, we only further evaluate these two methods on other generalization settings. The average success rates are shown in Fig. 3. BridgeVLA outperforms RVT-2 in all the seven settings. As we can see, RVT-2 struggles in both visual generalization settings and semantic generalization settings, while BridgeVLA performs much better, especially in Lighting and Combination. These results addresses Q3 and Q4, indicating that BridgeVLA is able to handle visual disturbance and novel instructions robustly.

Although our method outperforms baseline methods in the Category setting, its absolute success rate is not high. A common failure mode is that the robot often ignores the target object and moves directly to the

###### Real Robot Setup Basic Setting Generalization Settings

Background Lighting Distractor

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Franka Research 3

ZED 2i

Combination Height Category

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- Figure 3 Real-Robot Experiments and Results. We use a Franka Research 3 robot arm and a ZED 2i camera to capture point clouds of the scene. To evaluate the model’s performance, we design 7 different settings including one basic setting and six generalization settings. Experimental results show that BridgeVLA outperforms the state-of-the-art baseline method RVT-2 [14] by an average of 32%.

destination during pick-and-place manipulation. We believe this relatively low performance is not due to BridgeVLA forgetting the knowledge gained from pre-training, as it still predicts heatmaps accurately when provided with samples from the pre-training dataset after fine-tuning (see Fig. 4 and Appendix C.4). Instead, we hypothesize that the reduced performance stems from two factors: 1) The images in the pre-training dataset are mostly captured from third-person views, which differ significantly from the projection images in our robot data; 2) The pre-training task focuses solely on object localization, whereas manipulation involves predicting keypoints that do not correspond to an object. To address these issues, we plan to expand both the scale and diversity of the pre-training dataset and explore more expressive action-decoding methods to better leverage the preserved pre-training knowledge.

###### 4.3 Ablation Studies

To prove the effectiveness of our model design and provide insights for the community, we conduct three ablation studies:

Whether we need to predict heatmaps before predicting actions. Our approach avoids direct action prediction by first generating 2D heatmaps using a convex upsampling module. Target positions are then computed by projecting 3D workspace points onto the heatmaps and selecting the point with the highest mean probability. For ablation, we replaced the convex upsampling module (309M parameters) with a similarly sized Transformer decoder (303M) to directly predict target positions, supervised by MSE loss. All other modules were kept the same as before. We performed a hyperparameter grid search and evaluated the model on RLBench. Results are shown in the Tab. 1. Replacing heatmap prediction with direct position regression reduced the average success rate from 88.2% to 31.4%, confirming the effectiveness of our heatmap-based design. The ablated model was also harder to train and more sensitive to hyperparameters—requiring a batch size of 192 and careful learning rate tuning—while our original model trains reliably even with a batch size of 64. We see three main reasons for this outcome: (1) Heatmaps offer denser supervision than 3D position vectors, enabling more effective learning. (2) Projecting 3D points onto heatmaps introduces helpful spatial priors, easing the learning process. (3) The 2D heatmaps share the same spatial structure as the input images,

Find all instances of sneaker Find all instances of apple Find all instances of panda bear

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Input Image

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Prediction

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Ground Truth

- Figure 4 Prediction on Pre-training Data after Fine-tuning. To simulate the multi-view inputs during fine-tuning, we repeat each pre-training image three times and feed them into the fine-tuned model to generate heatmaps. Note that these samples are not cherry-picked. Additional samples can be found in Appendix C.4. enhancing alignment and improving performance.

Whether we need to remove the 3D position input to the VLM backbone. Unlike typical 3D VLA models like SpatialVLA, we deliberately avoid using per-pixel 3D position inputs and rely solely on RGB images. This design preserves alignment between the input feature spaces of fine-tuning and VLM pretraining, which we find crucial for effective vision-language-action (VLA) modeling. To test this, we added a 3D convolutional module to encode per-pixel 3D positions, fused them with 2D features, and fed the result into the backbone. Although this adds richer spatial cues, it alters the image feature distribution seen during pretraining, leading to a performance drop from 88.2% to 56.2% on RLBench. Detailed results are shown in the Tab. 1.

Whether we need to do 2D heatmap pre-training to the VLM backbone. The original VLM backbone can not predict heatmaps, while our downstream policy learning requires such ability. To bridge the gap, we do 2D heatmap pre-training to the VLM backbone. To verify its effectiveness, we ablate this pre-training and evaluate model’s performance in the real world, the results are shown in Fig. 3. As we can see, BridgeVLA w/o Pre-train is not able to generalize well in both language-related generalization settings and can not even beat RVT-2, while BridgeVLA achieves the best performance across the two generalization settings especially in Combination, highlighting its ability to understand language semantics. We hypothesize that the 2D-heatmap pre-training equips BridgeVLA with the ability to connect the semantics in language instructions with image observations in the heatmap space.

The above experiment results address Q5 and highlight the effectiveness of our architectural designs.

- 5 Conclusions & Future Work

This paper has introduced BridgeVLA, a novel and efficient 3D vision-language-action (VLA) model built on top of a pre-trained vision-language model (VLM) [1]. Keys to our method are that (1) it converts 3D inputs to 2D images to align with the 2D image inputs of the pre-trained VLM; (2) it aligns the input observation and the output action to a unified 2D image space via 2D heatmap prediction; (3) it adopts a scalable pre-training method to equip the VLM with the capability to predict heatmaps before fine-tuning on action prediction. Extensive experiments show that the proposed method is able to learn 3D manipulation efficiently and effectively in both simulation and the real world. In the future, we plan to explore pre-training on more diverse tasks, including semantic segmentation and keypoint detection. We also want to incorporate more expressive action-decoding methods (e.g., diffusion [21]) into the framework to continue improving the policy performance.

### References

- [1] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

- [2] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [4] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning, 2024.

- [5] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, et al. Openvla: An open-source vision-language-action model. In 8th Annual Conference on Robot Learning.

- [6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control,

2024. URL https://arxiv. org/abs/2410.24164.

- [7] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

- [8] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023.

- [9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

- [10] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. In Conference on Robot Learning, pages 785–799. PMLR, 2023.

- [11] Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. In 8th Annual Conference on Robot Learning, 2024.

- [12] Theophile Gervet, Zhou Xian, Nikolaos Gkanatsios, and Katerina Fragkiadaki. Act3d: 3d feature field transformers for multi-task robotic manipulation. In Conference on Robot Learning, pages 3949–3965. PMLR, 2023.

- [13] Ankit Goyal, Jie Xu, Yijie Guo, Valts Blukis, Yu-Wei Chao, and Dieter Fox. Rvt: Robotic view transformer for 3d object manipulation. In Conference on Robot Learning, pages 694–710. PMLR, 2023.

- [14] Ankit Goyal, Valts Blukis, Jie Xu, Yijie Guo, Yu-Wei Chao, and Dieter Fox. Rvt-2: Learning precise manipulation from few demonstrations. In RSS 2024 Workshop: Data Generation for Robotics, 2024.

- [15] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.

- [16] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

- [17] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020.

- [18] Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191, 2024.

- [19] Ricardo Garcia, Shizhe Chen, and Cordelia Schmid. Towards generalizable vision-language robotic manipulation: A benchmark and llm-guided 3d policy. arXiv preprint arXiv:2410.01345, 2024.

- [20] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

- [21] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2024.

- [22] Peiyan Li, Hongtao Wu, Yan Huang, Chilam Cheang, Liang Wang, and Tao Kong. Gr-mg: Leveraging partiallyannotated data via multi-modal goal-conditioned policy. IEEE Robotics and Automation Letters, 2025.

- [23] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

- [24] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

- [25] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations, 2024.

- [26] Shizhe Chen, Ricardo Garcia Pinel, Cordelia Schmid, and Ivan Laptev. Polarnet: 3d point clouds for languageguided robotic manipulation. In Conference on Robot Learning, pages 1761–1781. PMLR, 2023.

- [27] Wentao Yuan, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. M2t2: Multi-task masked transformer for object-centric pick and place. arXiv preprint arXiv:2311.00926, 2023.

- [28] Rujia Yang, Geng Chen, Chuan Wen, and Yang Gao. Fp3: A 3d foundation policy for robotic manipulation. arXiv preprint arXiv:2503.08950, 2025.

- [29] Stephen James, Kentaro Wada, Tristan Laidlow, and Andrew J Davison. Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13739–13748, 2022.

- [30] Yueru Jia, Jiaming Liu, Sixiang Chen, Chenyang Gu, Zhilue Wang, Longzan Luo, Lily Lee, Pengwei Wang, Zhongyuan Wang, Renrui Zhang, et al. Lift3d foundation policy: Lifting 2d large-scale pretrained models for robust 3d robotic manipulation. arXiv preprint arXiv:2411.18623, 2024.

- [31] Chengmeng Li, Junjie Wen, Yan Peng, Yaxin Peng, Feifei Feng, and Yichen Zhu. Pointvla: Injecting the 3d world into vision-language-action models. arXiv preprint arXiv:2503.07511, 2025.

- [32] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

- [33] Edward Johns. Coarse-to-fine imitation learning: Robot manipulation from a single demonstration. In 2021 IEEE international conference on robotics and automation (ICRA), pages 4613–4619. IEEE, 2021.

- [34] Ioan A. Şucan, Mark Moll, and Lydia E. Kavraki. The Open Motion Planning Library. IEEE Robotics & Automation Magazine, 19(4):72–82, December 2012. https://ompl.kavrakilab.org.

- [35] James J Kuffner and Steven M LaValle. Rrt-connect: An efficient approach to single-query path planning. In Proceedings 2000 ICRA. Millennium conference. IEEE international conference on robotics and automation. Symposia proceedings (Cat. No. 00CH37065), volume 2, pages 995–1001. IEEE, 2000.

- [36] David Coleman, Ioan Sucan, Sachin Chitta, and Nikolaus Correll. Reducing the barrier to entry of complex robotic software: a moveit! case study. arXiv preprint arXiv:1404.3785, 2014.

- [37] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024.

- [38] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [39] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

- [40] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.

- [41] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 991–1002. PMLR, 2022.

- [42] Pierre-Louis Guhur, Shizhe Chen, Ricardo Garcia Pinel, Makarand Tapaswi, Ivan Laptev, and Cordelia Schmid. Instruction-driven history-aware policies for robotic manipulations. In Conference on Robot Learning, pages 175–187. PMLR, 2023.

- [43] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver io: A general architecture for structured inputs & outputs. In International Conference on Learning Representations.

- [44] Eric Rohmer, Surya PN Singh, and Marc Freese. V-rep: A versatile and scalable robot simulation framework. In 2013 IEEE/RSJ international conference on intelligent robots and systems, pages 1321–1326. IEEE, 2013.

- [45] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. Advances in neural information processing systems, 35:23192–23204, 2022.

- [46] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022.

- [47] Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik. Masked visual pre-training for motor control. arXiv preprint arXiv:2203.06173, 2022.

- [48] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4840–4851, 2024.

- [49] AI@Meta. Llama 3 model card. 2024.
- [50] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36:72983–73007, 2023.

- [51] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.

- [52] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. \π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Pretrain RLBench Finetune COLOSSEUM Finetune GemBench Finetune Real-robot Finetune

learning rate 5e-5 8e-5 8e-5 8e-5 2e-5 optimizer AdamW AdamW AdamW AdamW AdamW batch size 384 192 192 160 192

warmup steps 400 – – – –

Table 5 Training hyperparameters for BridgeVLA

## Appendix

### A Training & Inference Details

Detailed training configurations are summarized in Tab. 5. Throughout both pre-training and fine-tuning, we keep the SigLIP vision encoder and language token embeddings frozen.

Computational Resources:

- 1. Pre-training: 8 NVIDIA A100 GPUs for 3,800 steps (≈2 hours)
- 2. RLBench fine-tuning: 48 NVIDIA H100 GPUs for 83,000 steps (≈20 hours)
- 3. COLOSSEUM fine-tuning: 48 NVIDIA H100 GPUs for 83,000 steps (≈20 hours)
- 4. GemBench fine-tuning: 40 NVIDIA A100 GPUs for 50 epochs (≈2.1 hours)
- 5. Real-world fine-tuning: 8 NVIDIA A100 GPUs for 300 epochs (≈1.5 hours)

For inference, we run BridgeVLA on a machine equipped with an NVIDIA RTX 4090 GPU. To evaluate its inference speed, we conducted 100 trials. From point cloud input to action output, the average end-to-end inference time is 0.21 seconds.

### B Simulation Experiments

###### B.1 Key frame Selection

For all the simulation and real-robot experiments, we adopt the same key frame selection strategy as PerAct [10]. A time step is labeled as a key frame if (i) the robot is stationary, (ii) the gripper state changes, or (iii) the step is the final state of the episode. The robot is considered stationary when the absolute velocities of all joints fall below 0.1rad/s.

###### B.2 Data

Following [10, 13, 14], we select 18 tasks from RLBench [17] to evaluate the performance of our method on complex manipulation tasks. These tasks are visualized in Fig. 5.

To assess the generalization capability of BridgeVLA, we also evaluate on the COLOSSEUM benchmark [18] and GemBench [19]. The COLOSSEUM benchmark includes 20 basic tasks and 12 types of perturbations. These perturbations, which are unseen during training, encompass changes in object texture, color, and size, backgrounds, lighting, distractors and camera poses. The benchmark evaluates on all the 12 types of perturbations, a setting with basic variations from the original RLBench, and a more challenging setting which combines all the 12 types of perturbations. We visualize all perturbations except the one from the original RLBench in Fig. 6.

For GemBench, the training set includes 16 tasks (31 variations) spanning seven fundamental action primitives (press, pick, push, screw, close, open, stack/put). The test set includes 44 tasks (92 variations) organized into four increasingly challenging settings. Unlike RLBench and COLOSSEUM, where demo augmentation is used,

we train BridgeVLA using only keyframes from each trajectory without performing any demo augmentation in GemBench.

###### B.3 Detailed Results on COLOSSEUM

For the COLOSSEUM results in Tab.2, we use the results of R3M-MLP [46], MVP-MLP [47], RVT [13], and PerAct [10] from the original COLOSSEUM paper [18]. For RVT-2 [14] and BridgeVLA, we perform our own training and evaluation process. We performed three test repetitions and report the average success rate and variance of BridgeVLA and RVT-2 for each task under different perturbations in Tab.6 and Tab.7, respectively.

###### B.4 Detailed Results on GemBench

We show per-task success rates on the four settings of GemBench in Tab. 8, 9, 10, 11. The results of baseline methods are sourced from [19]. In total, we evaluate on 5 random seeds to reduce statistical variance. And for every seed, we run 20 trials per task variation.

### C Real-Robot Experiments

###### C.1 Experiment Setup

Fig. 3 illustrates our real-robot setting. The platform comprises a 7-DoF Franka Research 3 manipulator with a parallel-jaw gripper and a ZED 2i stereo camera mounted on a tripod for capturing point clouds of the workspace. We collect expert trajectories with a kinestheic teaching approach. We first move the manipulator to keypoints of an expert trajectory and then play back the keypoints to record the observation and action at each keypoint.

###### C.2 Basic Setting

This setting provides a scene similar to the training dataset, where only the object layouts are modified. To highlight BridgeVLA’s advantages over existing manipulation policies, we compare it with four representative methods in this setting. The behaviors of these baselines are as follows:

SpatialVLA [16]: In the experimental setup, we initially trained SpatialVLA using only 10 trajectories per task. However, it failed on nearly all tasks, often struggling to move toward the correct target object. To improve performance, we augmented the dataset with an additional 40 trajectories per task. While this improved performance, it still lagged significantly behind BridgeVLA—particularly on more challenging tasks, such as "Put the giraffe in the lower drawer." These findings suggest that BridgeVLA provides a more effective and data-efficient solution for 3D VLA.

π0 [52]: Similarly, π0 fails with only 10 trajectories per task, likely due to overfitting—it performs well on the training set but often fails during online testing. Common failure modes include missing or failing to grasp the target and prematurely opening the gripper before reaching the goal. Notably, both BridgeVLA and π0 share the same PaliGemma backbone and are trained end-to-end. This highlights a key contribution of our work: while VLAs like π0 perform well with large-scale data, they struggle in low-data regimes—even on simpler tasks, such as "Press sanitizer." In contrast, BridgeVLA achieves near-perfect success and generalizes robustly across diverse settings.

ACT [24]: ACT also underperforms compared to BridgeVLA. It demonstrates limited spatial generalization, performing well only in areas densely covered during training, but often failing when the target is near the workspace boundaries. This behavior is consistent with its design: ACT models actions using a Gaussian prior, which assigns low probability to peripheral regions, limiting its spatial generalization capabilities.

RVT-2 [14]: RVT-2 performs the best among all the baselines. It can successfully solve most tasks, but it is not as robust as BridgeVLA. For instance, it sometimes fails to pick up the block precisely or place the object accurately, leading to task failure. Meanwhile, by utilizing the capabilities of VLM, BridgeVLA’s advantages are further amplified in generalization settings, as detailed in Sec. 4.2.

###### C.3 Generalization Settings

We evaluate on a total of six generalization settings: Distractor, Lighting, Background, Height, Combination, and Category. For Distractor, Lighting, Background, and Height, we visualize these settings in Fig. 10. We visualize the settings of Combination and Category in Fig. 11 and Fig. 12, respectively.

In Distractor, we add distractor objects that are visually similar to at least one target object to the scene. In Lighting, we evaluate the model in a novel lighting condition in which the lights are off. In Background, we use three different tablecloths to change the background. For Height, we elevate all objects for manipulation with a drawer that is about 10cm high. Distractor, Lighting, Background, and Height aim to evaluate the robustness against visual disturbances.

In Combination, we combine objects and skills that are not paired together in the training datasets. That is, while the object for manipulation and the manipulation skill are seen during training, the instruction that pairs them together is novel. The setting of Combination helps us evaluate whether the model is able to generalize across novel object-skill combinations. In Category, we want to evaluate whether BridgeVLA is able to manipulate objects from categories that are unseen in the robot training data. In total, we test 7 novel objects.

###### C.4 Preservation of Object Grounding Capability after Fine-tuning

We observe that even after fine-tuning on robot action data, BridgeVLA retains the object grounding capability learned during pre-training. We visualize its predictions on the pre-training dataset after fine-tuning in Fig. 14. It is important to note that the samples in Fig. 14 are not cherry-picked. BridgeVLA does not forget its pre-training knowledge after 3D action fine-tuning.

###### C.5 Per-task Success Rate

###### We showcase per-task success rates of BridgeVLA in the basic setting in Tab. 12. Notably, BridgeVLA achieves exceptionally high success rates even with only 3 trajectories per task, highlighting its superb sample efficiency.

BackgroundTexture

AllPerturbations

MO-TEXTURE

RO-TEXTURE

TableTexture

CameraPose

MO-COLOR

RO-COLOR

TableColor

LightColor

Distractor

MO-SIZE

RLBench

RO-SIZE

Original

Task Name

basketball_in_hoop 100.0±0.0 4.0±3.3 94.7±1.9 96.0±0.0 84.0±5.7 - 100.0±0.0 68.0±0.0 100.0±0.0 100.0±0.0 100.0±0.0 37.3±1.9 100.0±0.0 100.0±0.0 100.0±0.0 close_box 100.0±0.0 72.0±0.0 94.7±1.9 - - - 93.3±1.9 - 100.0±0.0 100.0±0.0 98.7±1.9 98.7±1.9 100.0±0.0 97.3±1.9 100.0±0.0 close_laptop_lid 100.0±0.0 11.1±15.7 82.7±3.8 - - - 67.9±14.6 - 89.3±8.2 92.0±0.0 97.3±3.8 82.7±6.8 96.0±3.3 100.0±0.0 96.0±0.0 empty_dishwasher 0.0±0.0 0.0±0.0 1.3±1.9 1.3±1.9 - 1.3±1.9 4.0±3.3 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 1.3±1.9 1.3±1.9 0.0±0.0

- get_ice_from_fridge 94.7±1.9 5.3±1.9 86.7±1.9 90.7±7.5 90.7±5.0 - 84.0±3.3 73.3±1.9 96.0±3.3 98.7±1.9 89.3±7.5 56.0±8.6 94.7±1.9 96.0±3.3 98.7±1.9 hockey 57.3±5.0 9.3±3.8 44.0±6.5 50.7±8.2 - 50.7±13.2 46.7±8.2 65.3±5.0 45.3±1.9 64.0±8.6 53.3±1.9 20.0±3.3 56.0±5.7 49.3±5.0 50.7±5.0 insert_onto_square_peg 93.3±3.8 23.3±2.4 52.0±3.3 94.7±1.9 - 76.0±8.6 85.3±3.8 70.7±3.8 84.0±0.0 88.0±3.3 88.0±3.3 44.0±11.8 86.7±1.9 77.3±5.0 96.0±0.0 meat_on_grill 96.0±0.0 9.3±1.9 32.0±0.0 88.0±5.7 - - 100.0±0.0 - 100.0±0.0 92.0±6.5 90.7±1.9 98.7±1.9 97.3±1.9 100.0±0.0 100.0±0.0 move_hanger 37.3±3.8 2.7±3.8 26.7±3.8 46.7±3.8 - - - - 52.0±0.0 84.0±0.0 52.0±5.7 52.0±5.7 33.3±5.0 42.7±1.9 24.0±0.0 open_drawer 96.0±0.0 60.0±3.3 97.3±1.9 - - - 90.7±1.9 - 88.0±3.3 93.3±1.9 100.0±0.0 90.7±1.9 100.0±0.0 94.7±1.9 96.0±0.0 place_wine_at_rack_location 88.0±5.7 17.3±13.6 82.7±5.0 89.3±7.5 - 92.0±6.5 93.3±3.8 90.7±3.8 90.7±5.0 97.3±1.9 88.0±3.3 74.7±3.8 90.7±6.8 92.0±3.3 92.0±8.6 put_money_in_safe 94.7±1.9 6.7±5.0 78.7±1.9 74.7±1.9 81.3±6.8 89.3±5.0 92.0±3.3 - 37.3±12.4 84.0±3.3 84.0±3.3 84.0±3.3 89.3±1.9 86.7±8.2 86.7±1.9 reach_and_drag 100.0±0.0 0.0±0.0 89.3±3.8 96.0±0.0 94.7±5.0 84.0±5.7 94.7±1.9 38.7±5.0 92.0±3.3 88.0±5.7 78.7±3.8 28.0±8.6 100.0±0.0 100.0±0.0 94.7±3.8 scoop_with_spatula 96.0±3.3 6.7±1.9 94.7±1.9 93.3±1.9 85.3±3.8 85.3±3.8 78.7±3.8 86.7±5.0 90.7±1.9 88.0±6.5 77.3±1.9 20.0±5.7 90.7±6.8 89.3±1.9 93.3±1.9 setup_chess 10.7±1.9 0.0±0.0 1.3±1.9 8.0±0.0 8.0±3.3 - 13.3±1.9 - 12.0±5.7 21.3±8.2 13.3±3.8 5.3±1.9 20.0±5.7 16.0±5.7 4.0±3.3 slide_block_to_target 100.0±0.0 24.0±3.3 74.7±1.9 - 92.0±3.3 - - - 100.0±0.0 100.0±0.0 98.7±1.9 84.0±9.8 100.0±0.0 100.0±0.0 100.0±0.0 stack_cups 58.7±3.8 29.3±1.9 66.7±1.9 - 50.7±1.9 - 44.0±3.3 - 62.7±1.9 64.0±3.3 65.3±8.2 26.7±7.5 73.3±8.2 64.0±14.2 72.0±8.6 straighten_rope 61.3±6.8 8.0±5.7 16.0±5.7 - 48.0±3.3 - - - 61.3±9.4 65.3±1.9 54.7±8.2 37.3±5.0 70.7±8.2 66.7±7.5 72.0±6.5 turn_oven_on 93.3±1.9 85.3±3.8 94.7±3.8 - - - 90.7±1.9 - 93.3±3.8 94.7±7.5 96.0±3.3 96.0±3.3 96.0±0.0 88.0±3.3 100.0±0.0 wipe_desk 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 - 0.0±0.0 - 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 Task Mean 73.9±0.7 18.7±2.2 60.5±1.1 63.8±0.1 63.5±1.5 68.4±3.3 69.3±1.0 61.7±0.8 69.7±1.2 75.7±0.9 71.3±0.7 51.8±1.5 74.8±1.0 73.1±0.2 73.8±0.3

###### Table 6 Success Rates of BridgeVLA under Different Perturbations of COLOSSEUM.

BackgroundTexture

AllPerturbations

MO-TEXTURE

RO-TEXTURE

TableTexture

CameraPose

MO-COLOR

RO-COLOR

TableColor

LightColor

Distractor

MO-SIZE

RLBench

RO-SIZE

Original

Task Name

basketball_in_hoop 100.0±0.0 10.0±2.0 99.0±1.7 94.0±2.0 97.0±1.7 - 100.0±0.0 86.0±3.5 95.0±1.7 94.0±2.0 84.0±6.3 89.0±3.3 100.0±0.0 99.0±1.7 100.0±0.0 close_box 93.0±4.4 36.0±8.5 70.0±6.6 - - - 86.0±3.5 - 99.0±1.7 97.0±1.7 91.0±4.4 93.0±3.3 97.0±1.7 94.0±2.0 99.0±1.7 close_laptop_lid 86.0±4.5 40.0±0.0 89.0±3.3 - - - 62.0±2.0 - 84.0±4.0 92.0±0.0 96.0±2.8 89.0±5.2 99.0±1.7 87.0±3.3 92.0±0.0 empty_dishwasher 0.0±0.0 0.0±0.0 1.0±1.7 0.0±0.0 - 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0

- get_ice_from_fridge 95.0±1.7 11.0±4.4 88.0±5.7 77.0±5.2 89.0±1.7 - 78.0±3.5 79.0±3.3 83.0±5.9 89.0±1.7 70.0±4.5 86.0±4.5 81.0±5.2 96.0±2.8 96.0±2.8 hockey 19.0±4.4 0.0±0.0 26.0±4.5 30.0±4.5 - 40.0±4.9 24.0±2.8 13.0±3.3 12.0±8.5 15.0±3.3 9.0±3.3 10.0±6.0 14.0±2.0 17.0±3.3 19.0±3.3 insert_onto_square_peg 31.0±3.3 0.0±0.0 13.0±1.7 35.0±9.5 - 32.0±2.8 33.3±8.6 21.0±1.7 30.0±2.0 9.0±1.7 4.0±4.9 9.0±3.3 35.0±3.3 35.0±1.7 23.0±4.4 meat_on_grill 100.0±0.0 89.0±1.7 100.0±0.0 100.0±0.0 - - 100.0±0.0 - 99.0±1.7 98.0±2.0 100.0±0.0 99.0±1.7 100.0±0.0 100.0±0.0 100.0±0.0 move_hanger 91.0±5.2 0.0±0.0 61.0±4.4 83.0±18.4 - - - - 55.0±5.9 69.0±5.9 29.0±5.2 92.0±2.8 94.0±2.0 87.0±4.4 22.0±2.0 open_drawer 99.0±1.7 25.0±4.4 63.0±4.4 - - - 92.0±0.0 - 88.0±0.0 92.0±0.0 99.0±1.7 86.0±8.2 100.0±0.0 95.0±1.7 95.0±1.7 place_wine_at_rack_location 96.0±4.9 28.0±6.3 74.0±4.5 98.0±2.0 - 93.0±5.2 87.0±3.3 90.0±6.6 81.0±7.1 87.0±4.4 95.0±6.6 83.0±3.3 89.0±5.9 96.0±2.8 91.0±5.2 put_money_in_safe 77.0±4.4 9.0±1.7 45.0±3.3 22.0±3.5 55.0±6.6 73.0±3.3 69.0±1.7 - 56.0±2.8 70.0±4.5 72.0±6.3 82.0±6.6 79.0±3.3 77.0±8.7 62.0±6.0 reach_and_drag 86.0±6.6 0.0±0.0 72.0±5.7 80.0±5.7 60.0±6.9 67.0±5.9 87.0±6.6 55.0±4.4 68.0±2.8 76.0±2.8 71.0±5.2 61.0±6.6 88.0±2.8 86.0±3.5 81.0±5.9 scoop_with_spatula 89.0±5.2 2.0±3.5 75.0±4.4 87.0±3.3 84.0±4.9 92.0±7.5 94.0±4.5 83.0±5.9 54.0±2.0 79.0±5.2 74.0±6.0 83.0±5.9 92.0±2.8 91.0±1.7 89.0±4.4 setup_chess 3.0±1.7 0.0±0.0 0.0±0.0 4.0±2.8 4.0±4.0 - 17.0±7.1 - 7.0±5.2 7.0±3.3 9.0±7.1 14.0±4.5 14.0±3.5 16.0±8.9 9.0±3.3 slide_block_to_target 100.0±0.0 11.0±4.4 45.0±1.7 - 97.0±1.7 - - - 84.0±4.9 96.0±0.0 83.0±5.2 82.0±8.7 100.0±0.0 100.0±0.0 100.0±0.0 stack_cups 35.0±5.2 0.0±0.0 47.0±5.9 - 45.0±5.9 - 23.0±4.4 - 18.0±2.0 16.0±4.0 13.0±9.5 19.0±7.7 24.0±2.8 43.0±9.1 40.0±2.8 straighten_rope 66.0±11.5 0.0±0.0 25.0±3.3 - 66.0±10.0 - - - 53.0±1.7 68.0±2.8 39.0±11.4 42.0±7.2 72.0±8.5 69.0±6.6 75.0±4.4 turn_oven_on 91.0±4.4 50.0±10.8 68.0±4.9 - - - 83.0±1.7 - 95.0±3.3 97.0±1.7 95.0±3.3 96.0±0.0 96.0±4.9 89.0±7.1 96.0±2.8 wipe_desk 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 - 0.0±0.0 - 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 0.0±0.0 Task Mean 67.8±1.5 15.6±0.8 53.0±0.9 54.6±0.6 59.7±0.7 56.7±1.4 60.9±0.9 53.4±1.5 58.0±1.1 62.6±0.9 56.6±0.9 60.8±0.5 68.7±1.1 68.8±1.3 64.4±0.5

###### Table 7 Success Rates of RVT-2 under Different Perturbations of COLOSSEUM.

Close Fridge+0

Close Jar+15

Close Jar+16

CloseLaptop Lid+0

Close Microwave+0

LightBulb In+17

LightBulb In+19

Open Box+0

Open Door+0

Open Drawer+0

Method Avg.

Hiveformer [42] 60.3±1.5 96±4.2 64±13.9 92±2.7 90±3.5 88±7.6 12±4.5 13±6.7 4±4.2 53±15.2 15±12.2 PolarNet [26] 77.6±0.9 99±2.2 99±2.2 99±2.2 95±3.5 98±2.7 72±12.5 71±6.5 32±11.5 69±8.9 61±12.4 3D diffuser actor [11] 91.9±0.8 100±0.0 100±0.0 100±0.0 99±2.2 100±0.0 85±5.0 88±2.7 11±2.2 96±4.2 82±9.1 RVT-2 [14] 89.0±0.8 77±11.0 97±4.5 98±2.7 77±13.0 100±0.0 93±5.7 91±8.2 7±4.5 98±4.5 93±5.7 3D-LOTUS [19] 94.3±3.5 96±3.7 100±0.0 100±0.0 98±2.5 98±4.0 84±7.4 85±9.5 99±2.0 77±2.5 83±8.7 3D-LOTUS++ [19] 68.7±0.6 95±0.0 100±0.0 99±2.0 28±2.5 87±5.1 55±10.5 45±8.9 55±8.9 79±9.7 68±12.5 BridgeVLA (Ours) 91.1±1.1 99±2.0 98±4.0 100±0.0 97±2.5 85±5.5 90±5.5 87±7.5 76±10.2 70±12.3 86±5.8

Open Drawer+2

Pick& Lift+0

Pick& Lift+2

Pick& Lift+7

PickUp Cup+8

PickUp Cup+9

PickUp Cup+11

Push Button+0

Push Button+3

Push Button+4

PutIn Cupboard+0

Method

Hiveformer [42] 59±7.4 86±4.2 92±6.7 93±2.7 83±7.6 69±12.9 61±19.8 84±11.9 68±6.7 87±7.6 34±8.2 PolarNet [26] 90±7.1 92±9.1 84±7.4 88±5.7 82±7.6 79±4.2 72±10.4 100±0.0 100±0.0 99±2.2 52±7.6 3D diffuser actor [11] 97±4.5 99±2.2 99±2.2 99±2.2 96±2.2 97±4.5 98±2.7 98±2.7 96±4.2 98±2.7 85±5.0 RVT-2 [14] 94±4.2 99±2.2 98±2.7 100±0.0 99±2.2 99±2.2 99±2.2 100±0.0 100±0.0 100±0.0 88±8.4 3D-LOTUS [19] 93±6.0 99±2.0 100±0.0 99±2.0 97±4.0 96±3.7 94±4.9 99±2.0 99±2.0 100±0.0 89±5.8 3D-LOTUS++ [19] 75±4.5 97±6.0 94±3.7 93±5.1 86±8.0 88±6.8 91±4.9 100±0.0 100±0.0 100±0.0 1±2.0 BridgeVLA(Ours) 99±2.0 99±2.0 100±0.0 98±2.5 96±2.0 94±3.7 99±2.0 100±0.0 98±4.0 98±4.0 74±6.6

PutIn Cupboard+3

PutMoney InSafe+0

PutMoney InSafe+1

Reach& Drag+14

Reach& Drag+18

Slide Block+0

Slide Block+1

Stack Blocks+30

Stack Blocks+36

Stack Blocks+39

Method

Hiveformer [42] 74±6.5 85±3.5 88±2.7 37±5.7 32±7.6 99±2.2 91±12.4 6±5.5 7±4.5 6±4.2 PolarNet [26] 88±4.5 93±4.5 95±5.0 99±2.2 99±2.2 100±0.0 0±0.0 34±10.8 30±9.4 36±12.9 3D diffuser actor [11] 82±11.5 95±5.0 98±2.7 100±0.0 99±2.2 100±0.0 89±4.2 88±7.6 85±6.1 89±5.5 RVT-2 [14] 80±6.1 93±8.4 96±8.5 85±10.0 94±2.2 100±0.0 37±6.7 88±5.7 93±2.7 88±11.5 3D-LOTUS [19] 72±11.2 94±3.7 99±2.0 99±2.0 100±0.0 100±0.0 100±0.0 94±5.8 91±6.6 90±4.5 3D-LOTUS++ [19] 2±2.5 22±6.8 16±4.9 94±3.7 62±8.7 100±0.0 65±5.5 86±5.8 20±4.5 28±13.6 BridgeVLA (Ours) 84±6.6 79±9.7 86±3.7 96±5.8 97±4.0 100±0.0 90±5.5 77±8.1 87±4.0 85±7.8

###### Table 8 Per-task Success Rate on GemBench Level 1.

PickUp Cup+13 Hiveformer 26.1±1.4 97±2.7 85±10.0 88±2.7 21±6.5 9±4.2 8±6.7 30±7.1 22±13.5 26±10.6

Push Button+13

Push Button+15

Push Button+17

Pick& Lift+14

Pick& Lift+16

Pick& Lift+18

PickUp Cup+10

PickUp Cup+12

Method Avg.

- PolarNet 37.1±1.4 100±0.0 100±0.0 85±7.9 3±4.5 1±2.2 0±0.0 48±11.0 46±8.9 16±6.5 3D diffuser actor 43.4±2.8 87±13.0 81±6.5 60±9.4 9±4.2 18±9.1 0±0.0 84±5.5 60±11.7 62±13.0 RVT-2 51.0±2.3 100±0.0 100±0.0 100±0.0 47±7.6 29±9.6 8±4.5 81±8.2 59±9.6 72±9.7 3D-LOTUS 49.9±2.2 99±2.0 100±0.0 100±0.0 3±2.5 18±8.7 33±9.3 89±3.7 78±8.7 57±7.5 3D-LOTUS++ 64.5±0.9 99±2.0 100±0.0 99±2.0 94±3.7 96±3.7 95±3.2 79±4.9 89±9.7 84±10.2 BridgeVLA (Ours) 65.0±1.3 100±0.0 100±0.0 100±0.0 74±9.7 89±4.9 0±0.0 91±3.7 90±3.2 90±6.3

Stack Blocks+24

Stack Blocks+27

Stack Blocks+33

Slide Block+2

Slide Block+3

Close Jar+3

Close Jar+4

LightBulb In+1

LightBulb In+2

Lamp On+0

Method

- Hiveformer 0±0.0 4±4.2 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 4±4.2 0±0.0 7±4.5 PolarNet 1±2.2 2±2.7 6±8.2 0±0.0 0±0.0 20±10.6 82±5.7 22±11.5 17±8.4 14±10.8 3D diffuser actor 66±13.9 82±2.7 50±14.6 0±0.0 0±0.0 23±16.8 82±5.7 51±17.8 60±10.0 7±7.6 RVT-2 18±4.5 56±16.7 45±13.7 0±0.0 1±2.2 7±7.6 77±5.7 68±14.4 6±6.5 0±0.0 3D-LOTUS 13±8.1 40±9.5 69±5.8 0±0.0 0±0.0 71±5.8 90±4.5 24±4.9 41±8.6 0±0.0 3D-LOTUS++ 22±9.3 83±7.5 59±3.7 27±9.8 5±3.2 98±2.5 96±3.7 56±9.7 43±7.5 2±2.0 BridgeVLA (Ours) 61±10.7 51±13.2 79±8.6 12±9.3 3±4.0 66±6.6 88±4.0 66±8.6 74±5.8 7±4.0 Method

Reach& Drag+5

Reach& Drag+7

PutCube InSafe+0

Pick&Lift Cylinder+0

Pick&Lift Star+0

Pick&Lift Moon+0

Pick&Lift Toy+0

PutIn Cupboard+7

PutIn Cupboard+8

- Hiveformer 1±2.2 0±0.0 4±2.2 78±5.7 73±7.6 88±2.7 87±4.5 0±0.0 0±0.0 PolarNet 61±8.2 10±6.1 40±14.1 93±6.7 88±8.4 93±6.7 90±3.5 0±0.0 0±0.0 3D diffuser actor 0±0.0 64±6.5 3±2.7 99±2.2 43±17.9 91±9.6 30±9.4 0±0.0 3±4.5 RVT-2 91±2.2 89±6.5 6±5.5 98±2.7 98±4.5 94±4.2 78±8.4 0±0.0 0±0.0 3D-LOTUS 95±4.5 18±10.8 25±5.5 88±8.7 69±6.6 80±8.4 96±3.7 0±0.0 0±0.0 3D-LOTUS++ 94±2.0 64±12.4 37±5.1 91±2.0 94±3.7 29±6.6 71±2.0 1±2.0 0±0.0

###### BridgeVLA (Ours) 94±3.7 96±3.7 3±2.5 98±2.5 99±2.0 95±3.2 93±5.1 0±0.0 0±0.0 Table 9 Per-task Success Rate on GemBench Level 2.

Open Box2+0 Hiveformer 35.1±1.7 0±0.0 1±2.2 34±9.6 52±9.1 15±7.1 32±11.5 5±3.5

Close Door+0

Close Box+0

Close Fridge2+0

CloseLaptop Lid2+0

Close Microwave2+0

Open Door2+0

Method Avg.

- PolarNet 38.5±1.7 0±0.0 0±0.0 78±5.7 26±8.2 74±6.5 33±6.7 23±8.4 3D diffuser actor 37.0±2.2 0±0.0 0±0.0 97±2.7 23±6.7 88±7.6 86±7.4 67±9.8 RVT-2 36.0±2.2 1±2.2 2±2.7 72±6.7 42±14.0 71±8.9 79±6.5 5±6.1 3D-LOTUS 38.1±1.1 0±0.0 58±8.1 36±9.7 54±10.7 85±7.1 42±6.8 11±6.6 3D-LOTUS++ 41.5±1.8 1±2.0 29±8.6 93±2.5 50±9.5 99±2.0 52±10.3 16±8.0 BridgeVLA (Ours) 43.8±1.2 0±0.0 1±2.0 95±5.5 77±4.0 54±10.2 68±10.8 74±4.9

Open Drawer2+0

Open Drawer3+0

OpenDrawer Long+0

OpenDrawer Long+1

OpenDrawer Long+2

OpenDrawer Long+3

Toilet SeatUp+0

Open Fridge+0

Method

Hiveformer 59±11.9 39±11.9 78±8.4 82±4.5 49±4.2 57±11.5 6±4.2 0±0.0 PolarNet 91±4.2 29±8.2 84±11.9 88±5.7 63±8.4 37±7.6 2±2.7 4±2.2 3D diffuser actor 19±8.2 1±2.2 15±5.0 35±13.7 26±9.6 79±12.9 0±0.0 7±5.7 RVT-2 81±11.9 0±0.0 84±8.2 39±10.8 11±8.9 75±6.1 7±5.7 0±0.0 3D-LOTUS 90±3.2 22±8.1 56±13.9 33±11.2 17±8.1 75±6.3 0±0.0 4±5.8 3D-LOTUS++ 70±5.5 41±4.9 72±4.0 52±10.8 23±8.1 78±5.1 8±5.1 0±0.0 BridgeVLA (Ours) 65±6.3 87±6.0 59±8.6 34±8.0 18±10.3 85±8.4 6±5.8 7±2.5

Close Grill+0 Hiveformer 100±0.0 0±0.0 0±0.0 0±0.0 83±5.7 44±10.8 PolarNet 100±0.0 0±0.0 1±2.2 4±4.2 29±11.9 42±11.5 3D diffuser actor 100±0.0 0±0.0 2±4.5 0±0.0 66±7.4 65±13.7 RVT-2 93±5.7 0±0.0 0±0.0 6±2.2 78±8.4 9±4.2 3D-LOTUS 100±0.0 0±0.0 0±0.0 0±0.0 87±8.1 29±6.6 3D-LOTUS++ 86±6.6 0±0.0 13±8.1 0±0.0 69±5.8 19±13.9

OpenLaptop Lid+0

Open Microwave+0

PutMoney InSafe+2

Open Drawer+1

Close Drawer+0

Method

###### BridgeVLA (Ours) 95±0.0 0±0.0 2±2.5 0±0.0 58±12.9 35±12.3 Table 10 Per-task Success Rate on GemBench Level 3.

Push Buttons4+1

Push Buttons4+2

Push Buttons4+3

TakeShoes OutOfBox+0

PutItems InDrawer+0

PutItems InDrawer+2

Method Avg.

Hiveformer 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 PolarNet 0.1±0.2 1±2.2 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 3D diffuser actor 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 RVT-2 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 3D-LOTUS 0.3±0.3 3±4.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 3D-LOTUS++ 17.4±0.4 76±7.4 49±8.6 37±8.1 0±0.0 0±0.0 0±0.0 BridgeVLA (Ours) 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0

PutItems InDrawer+4

Stack Cups+0

Stack Cups+3

PutAllGroceries InCupboard+0

Method

Tower4+1 Tower4+3

Hiveformer 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 PolarNet 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 3D diffuser actor 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 RVT-2 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 3D-LOTUS 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 3D-LOTUS++ 0±0.0 17±10.8 30±13.4 0±0.0 0±0.0 0±0.0 BridgeVLA (Ours) 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0 0±0.0

###### Table 11 Per-task Success Rate on GemBench Level 4.

Close jar Insert onto square peg Light blub in

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Meat off grill Place shape in shape sorter Place wine at rack location

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Open drawer Push buttons Place cups

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Put groceries in cupboard Put item in drawer Put money in safe

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Reach and drag Slide block to color target Stack cups

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Sweep to dustpan of size Turn tap Stack blocks

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Figure 5 Visualization of 18 RLBench [17] Tasks.

Basketball in hoop Close box Close laptop lid

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

All Variations

Empty dishwasher Get ice from fridge Insert onto square peg

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

ManipulationObject-Color

Hockey Meat on grill Place wine at rack location

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

ReceivingObject-Color

Put money in safe Reach and drag Scoop with spatula

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

ManipulationObject-Texture

Hockey Insert onto square peg Place wine at rack location

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

ReceivingObject-Texture

Setup chess Turn oven on Wipe desk

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

ManipulationObject-Size

Basketball in hoop Hockey Insert onto square peg

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

ReceivingObject-Size

Stack cups Strengthen rope Turn oven on

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Light-Color

Put money in safe Reach and drag Scoop with spatula

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Table-Color

Close laptop lid Get ice from fridge Open drawer

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Table-Texture

Move hanger Setup chess Slide block to target

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

Distractor

Basketball in hoop Open drawer Wipe desk

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Backgroun d-Texture

Close box Empty dishwasher Setup chess

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Camera Pose

###### Figure 6 Visualization of Perturbations in COLOSSEUM [18].

Task 3 trajectories 10 trajectories

Put the RedBull can in the top shelf 9/10 10/10 Put the soda can in the bottom shelf 9/10 9/10

Put the RedBull can in the bottom shelf 10/10 10/10 Put the coke can in the top shelf 10/10 10/10

Place the red block in the blue plate 10/10 10/10 Place the orange block in the green plate 10/10 10/10

Put the wolf in the upper drawer 7/10 9/10

Place the red block in the purple plate 10/10 10/10 Place the yellow block in the green plate 10/10 10/10

Press sanitizer 10/10 10/10

Put the zebra in the upper drawer 9/10 9/10 Put the giraffe in the lower drawer 10/10 9/10 Put the zebra in the lower drawer 10/10 10/10

Table 12 Per-task Success Rates of BridgeVLA in the Basic Setting.

Initial Scene Place orange block in green plate

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Place red block in blue plate

[Figure 294]

Initial Scene

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

##### Place red block in purple plate

[Figure 301]

Initial Scene

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Place yellow block in green plate

[Figure 308]

Initial Scene

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

Press sanitizer

[Figure 315]

Initial Scene

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Put soda can in bottom shelf

[Figure 322]

Initial Scene

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

##### Put coke can in top shelf

[Figure 329]

Initial Scene

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Figure 7 Real-Robot Rollouts (I).

Put Redbull can in bottom shelf

[Figure 336]

Initial Scene

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Put Redbull can in top shelf

[Figure 343]

Initial Scene

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Put giraffe in lower drawer

[Figure 350]

Initial Scene

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

##### Put wolf in upper drawer

[Figure 357]

Initial Scene

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

Put zebra in upper drawer

[Figure 364]

Initial Scene

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Put zebra in lower drawer

[Figure 371]

Initial Scene

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Figure 8 Real-Robot Rollouts (II).

half eaten frosted donut behind cup horse in the back kid in glasses playing wii

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

orange slice under grapes person taking photo right animal

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

small lamb sitting on ground on the right hand side next to two others. the woven place matt

tractor with orange

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

white bowl with vegetables. white doughnut right white keyboard

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

zebra facing out number three

15 an elephant with other two elephants

from tree

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

Find all instances of a circular frame

baby far middle right elephant

with spokes

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

Find all instances of a decorative arrangement of flowers.

Find all garments from waist to knee or ankle, covering each leg separately Find all instances of bike

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Find all instances of boot Find all instances of clock tower Find all instances of cup

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

Find all instances of a piece of furniture holding one or more electric light bulbs Find all instances of surfboard

greenest apple by banana

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

white bowl with vegetables Find all instances of street sign the woven place matt

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

###### Figure 9 Visualization of Pre-training Data. We list some samples of pre-training data. For every sample, the left shows the original image; the middle shows the bounding boxes of the objects of interest; the right shows the ground-truth heatmap used for training.

Lighting

Background

Distractor

Height

#### Task

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

Put the RedBull can in

| |
|---|

| |
|---|

| |
|---|

the top shelf

| |
|---|

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

Put the soda can

in the bottom shelf

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

Place the red

block in the blue

| |
|---|

| |
|---|

| |
|---|

plate

| |
|---|

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

Place the orange

block in the green

| |
|---|

| |
|---|

| |
|---|

plate

| |
|---|

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

Press sanitizer

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

Put the zebra in

the upper drawer

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

Put the giraffe

in the lower drawer

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 10 Visualization of the Distractor, Lighting, Background, and Height settings.

[Figure 496]

[Figure 497]

[Figure 498]

put the wolf in

| |
|---|

the lower drawer

| |
|---|

| |
|---|

[Figure 499]

[Figure 500]

[Figure 501]

put the

| |
|---|

| |
|---|

giraffe in the

| |
|---|

upper drawer

[Figure 502]

[Figure 503]

[Figure 504]

place the

orange block

in the purple

| |
|---|

| |
|---|

plate

| |
|---|

[Figure 505]

[Figure 506]

[Figure 507]

place the red block in the

green plate

| |
|---|

| |
|---|

| |
|---|

[Figure 508]

[Figure 509]

[Figure 510]

place the

orange block in

| |
|---|

| |
|---|

the blue plate

| |
|---|

[Figure 511]

[Figure 512]

[Figure 513]

place the

yellow block in the blue plate

| |
|---|

| |
|---|

| |
|---|

[Figure 514]

[Figure 515]

[Figure 516]

place the yellow

block in the

| |
|---|

| |
|---|

| |
|---|

purple plate

[Figure 517]

[Figure 518]

[Figure 519]

put the soda can

| |
|---|

in the top shelf

| |
|---|

| |
|---|

- Figure 11 Visualization of the Combination Setting (I). During training, the manipulated objects and skills are seen, but their combinations are unseen.

[Figure 520]

[Figure 521]

[Figure 522]

Put the red block

in the bottom shelf

| |
|---|

| |
|---|

| |
|---|

[Figure 523]

[Figure 524]

[Figure 525]

Put the orange

block in the lower

drawer

| |
|---|

| |
|---|

| |
|---|

[Figure 526]

[Figure 527]

[Figure 528]

Put the soda can in the upper drawer

| |
|---|

| |
|---|

| |
|---|

[Figure 529]

[Figure 530]

[Figure 531]

Put the Redbull can in the green

| |
|---|

| |
|---|

| |
|---|

plate

[Figure 532]

[Figure 533]

[Figure 534]

Place the zebra in the blue plate

| |
|---|

| |
|---|

| |
|---|

- Figure 12 Visualization of the Combination Setting (II). During training, the manipulated objects and skills are seen, but their combinations are unseen.

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

Press the

mouse

| |
|---|

| |
|---|

| |
|---|

[Figure 539]

[Figure 540]

[Figure 541]

Put the apple in

the top shelf

| |
|---|

| |
|---|

| |
|---|

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

Put the peach in

the bottom shelf

| |
|---|

| |
|---|

| |
|---|

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

Put the sneaker

in the upper

| |
|---|

| |
|---|

| |
|---|

drawer

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

Put the panda in

the lower

| |
|---|

| |
|---|

| |
|---|

drawer

[Figure 554]

[Figure 555]

[Figure 556]

Place the bread

in the green

| | | |
|---|---|---|
| | | |

plate

| |
|---|

| |
|---|

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

Place the bottle

in the blue plate

| |
|---|

| |
|---|

| |
|---|

###### Figure 13 Visualization of the Category Setting. In total, we evaluate on 7 objects from novel categories that are unseen during training.

Find all instances of a round faster Find all instances of baggage Find all instances of cutlery

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

Find all instances of mug Find all instances of neckwear Dog laying down

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

Find all instances of a container Find all instances of a long tube made of

Find all instances of alarm clock

metal or plastic

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

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

Find all instances of bedding white doughnut right Find all instances of veil

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

###### Figure 14 Visualization of BridgeVLA’s Prediction on Pre-training Dataset after Fine-tuning. To simulate the multi-view inputs during fine-tuning, we repeat the input image three times and feed them into the fine-tuned model to generate heatmaps. For each sample, the first row shows the input image; the second row shows the heatmap prediction; the third row shows the ground truth.

