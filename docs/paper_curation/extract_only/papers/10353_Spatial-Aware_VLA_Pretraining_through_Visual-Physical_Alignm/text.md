# arXiv:2512.13080v1[cs.RO]15Dec2025

∝ ∝

∝

∝BeingBeyond

BeingBeyond

智在⽆界

智在⽆界

## Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos

#### Yicheng Feng1,3 Wanpeng Zhang1,3 Ye Wang2,3 Hao Luo1,3 Haoqi Yuan1,3 Sipeng Zheng3 Zongqing Lu1,3,†

1Peking University 2Renmin University of China 3BeingBeyond https://beingbeyond.github.io/VIPA-VLA

#### Abstract

Vision-Language-Action (VLA) models provide a promising paradigm for robot learning by integrating visual perception with language-guided policy learning. However, most existing approaches rely on 2D visual inputs to perform actions in 3D physical environments, creating a significant gap between perception and action grounding. To bridge this gap, we propose a Spatial-Aware VLA Pretraining paradigm that performs explicit alignment between visual space and physical space during pretraining, enabling models to acquire 3D spatial understanding before robot policy learning. Starting from pretrained vision-language models, we leverage large-scale human demonstration videos to extract 3D visual and 3D action annotations, forming a new source of supervision that aligns 2D visual observations with 3D spatial reasoning. We instantiate this paradigm with VIPA-VLA, a dual-encoder architecture that incorporates a 3D visual encoder to augment semantic visual representations with 3D-aware features. When adapted to downstream robot tasks, VIPA-VLA achieves significantly improved grounding between 2D vision and 3D action, resulting in more robust and generalizable robotic policies.

Date: December 15, 2025

#### 1 Introduction

The rapid progress of large-scale vision-language models (VLMs) has showcased remarkable capabilities in learning joint representations across modalities[1–5]. This progress has opened up new opportunities for robot policy learning, where VLMs offer a solid foundation for understanding both visual observations and language instructions, thereby guiding robot interaction with the physical world. Building on this foundation, the vision-language-action (VLA) paradigm has recently demonstrated the potential to develop generalist robot policies across a wide range of tasks[6–10].

Nevertheless, current VLA models typically rely on 2D visual inputs to perceive the world while performing actions in a 3D physical environment, leaving a substantial gap between visual perception and embodied action. This weak correspondence limits their ability to ground actions in physical space. For effective policy learning, an agent must not only interpret pixels but also understand how these visual cues map to 3D geometry and how physical actions interact with the surrounding environment. While humans can infer 3D

†Correspondence to Zongqing Lu <lu@beingbeyond.com>.

|Spatial Relationship Task Completion<br><br>[Figure 1]<br><br>[Figure 2]<br><br>| | |
|---|---|
| | |
<br><br>| | |
|---|---|
| | |
<br><br>Hand Movement<br><br>[Figure 3]<br><br>[Figure 4]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Camera Movement<br><br>[Figure 5]<br><br>[Figure 6]<br><br>3D vision annotations<br><br>|[Figure 7]|
|---|
<br><br>3D action annotations<br><br>|
|---|

|Robot Data<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>|
|---|

|| |
|---|
<br><br>Visual Tokens<br><br>Text Tokens<br><br>| |
|---|
<br><br>Motion<br><br>Tokens<br><br>| |
|---|
<br><br>Action Queries<br><br>| |
|---|
<br><br>Action|
|---|

|Spatial-Aware VLA Pretraining<br><br>VIPA-VLA<br><br>Projector<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>VIPA-VLA<br><br>Projector Stage 1 Stage 2<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|Post Training<br><br>| |
|---|
<br><br>| |
|---|
<br><br>VIPA-VLA<br><br>Projector<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Stage 3|
|---|

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Human videos

- Figure 1: Overview of the proposed Visual-Physical Alignment framework. We start from diverse human demonstration videos to extract 3D visual annotations and 3D action annotations, capturing how humans interact with the physical world with corresponding visual observations. These annotations enable the two-stage Spatial-Aware VLA Pretraining, which teaches VLA models to ground 2D visual inputs in 3D spatial understanding: (1) 3D-Visual Pretraining: starting from a VLM backbone, human demonstration videos with 3D visual annotations are used to align 2D visual features with 3D spatial representations via a dual-encoder fusion module. (2) 3D-Action Pretraining: human hand trajectories provide 3D motion supervision, enabling the model to learn physically grounded action priors. Then in the third stage, the pretrained model, VIPA-VLA, is adapted to robot manipulation tasks, resulting in robust and generalizable policies in simulation and real-world settings.

space from 2D visual signals, existing VLA models largely overlook this aspect, resulting in poor spatial grounding and limited generalization.

To bridge this gap, we introduce a new Spatial-Aware VLA Pretraining paradigm that enables models to acquire 3D spatial awareness before learning robotic policies. Starting from pretrained VLMs, we leverage large-scale human demonstration videos as a rich source of supervision, where implicit correspondences between 2D visual observations and 3D physical actions naturally exist. Compared to robot data, human demonstrations are easier to obtain across diverse environments and naturally provide rich evidence of how actions are carried out in the physical world under diverse visual contexts. By extracting 3D cues such as hand-object relationships and motion trajectories from these videos, we construct pretraining tasks that teach the model to align 2D vision with 3D spatial understanding—forming what we term visual-physical alignment. This spatially grounded pretraining provides a strong foundation for subsequent VLA post-training, allowing the model to learn and generalize more effectively in robot manipulation tasks.

To instantiate this paradigm, we present VIPA-VLA (Visual-Physical-Alignment-VLA), a dual-encoder architecture that augments semantic visual representations with explicit 3D spatial features. Alignment between vision, language, and 3D action is achieved through Spatial-Aware VLA Pretraining on our dataset Hand3D, which is constructed from diverse human manipulation recordings with annotated hand poses. From these videos, we derive two forms of spatial supervision: 3D visual annotations, providing coarse-grained spatial grounding between visual inputs and physical configurations, and 3D action annotations, offering fine-grained motion supervision through 3D trajectories that encode human manipulation dynamics. The pretrained model is then adapted to robotic tasks via post-training, transferring the learned spatial reasoning capability to robotic control.In summary, this work makes the following contributions:

- • We propose a new Spatial-Aware VLA Pretraining paradigm that bridges 2D visual perception and 3D physical action through large-scale human video data.
- • We construct Hand3D, a dataset of human manipulation videos with 3D visual and action annotations enabling visual–physical alignment supervision.
- • We present VIPA-VLA, a dual-encoder VLA architecture pretrained with our paradigm, which significantly enhances spatial grounding and generalization in downstream robot tasks.

#### 2 Related Work

- 2.1 VLA models

Vision-Language-Action (VLA) models aim to leverage joint representations of visual observations and language instructions to enable action execution in physical environments[8, 11–21]. Some prior works tokenize robot actions and finetune pretrained VLMs on robot datasets[9, 10]. To support this pretraining paradigm, large-scale robot datasets have been introduced[22–24]. Other works incorporate additional action experts to improve the quality of action generation[6, 7, 25, 26]. Some approaches leverage demonstration videos for representation learning[27–29] or future state prediction to enhance action learning[23, 30, 31]. More recently, models such as GR00T-N1[7] and π0.5[32] combine massive multimodal internet data with robot datasets for large-scale VLA pretraining, significantly improving generalization. In contrast, our work focuses on understanding the 3D physical space to better ground actions in environments with Spatial-Aware VLA Pretraining from human videos.

- 2.2 3D Aware Models

Recent advances in 3D vision-language models (3D VLMs) aim to extend the perceptual capacity of multimodal models beyond 2D images by incorporating 3D spatial information[33–39]. Some approaches leverage datasets with explicit 3D annotations[40–42], while others rely on geometric priors or depth estimation models to infer

- 3D structure from monocular visual inputs[43–47]. These methods have improved the ability of VLMs to reason about spatial relationships, object geometry, and scene structure. However, the focus of existing 3D VLMs is primarily on perception rather than action. They enhance spatial understanding of static observations but do not explicitly establish the correspondence between 3D perception and the physical action space required for robot policy learning. Consequently, their utility in training VLA models remains limited.

Recent works have also explored enhancing VLAs with 3D spatial reasoning[37]. For instance, 3D-VLA[48] introduces a generative world model with diffusion-based rendering of future images and point clouds, while SpatialVLA[49] improves policy learning through ego-centric position encoding and adaptive action grids. Our work differs in that we aim to endow VLAs with 3D spatial understanding through a pretraining stage, by leveraging 3D-aware annotations extracted from human videos and incorporating a 3D encoder to explicitly align 2D visual observations with 3D physical space. This Spatial-Aware Pretraining equips VLAs with stronger 3D grounding prior to downstream policy learning.

- 2.3 Learning from human videos

A line of research has explored leveraging human demonstration videos to facilitate robot policy learning. Some efforts adopt representation learning approaches to extract latent features from human activities[23, 27, 38], yet such representations are typically implicit and provide limited guidance for explicit action grounding. Other methods attempt to directly align the human action space with the robot action space[50–54]. However, these approaches suffer from embodiment mismatch, as the physical capabilities and kinematics of humans and robots differ significantly. In addition, several works explore extracting interaction-related knowledge from human videos, such as affordances or grasping strategies[55–57], which provide helpful cues but still stop short of explicitly bridging perception and action in 3D physical space. Beyond these, Being-H0[15] also leverages human videos for VLA pretraining, but its primary focus lies in learning manipulation motion sequences. In contrast, we highlight that, despite the embodiment gap, human videos contain rich information about how actions are executed in the 3D physical world across diverse visual contexts. Our work leverages this information as supervision for Spatial-Aware Pretraining, enabling VLA models to acquire grounded correspondences between 2D visual observations and 3D action space before downstream policy learning.

#### 3 Method

In this section, we first provide preliminaries on vision-language models (VLMs) and vision-language-action models (VLAs) (Sec. 3.1). Then we illustrate the Visual Phisical Alignment in Sec. 3.2, including the spatialaware dataset Hand3D (Sec. 3.2.1), the model architecture VIPA-VLA (Sec. 3.2.2) and the Spatial-Aware VLA

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Instructional Data

|| |
|---|
<br><br>|
|---|

### Hand3D

Hand

[Figure 24]

[Figure 25]

&

Camera

Spatial Relationship

Spatial

[Figure 26]

Calibration

[Figure 27]

[Figure 28]

[Figure 29]

Task Completion

[Figure 30]

Depth Map

Cut3R

Object

Hand Movement

[Figure 31]

[Figure 32]

Localization

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Object Proposal

Camera Movement

Grounding

DINO

3D Spatial Relationships

- Figure 2: Overview of the Hand3D-visual. By integrating point cloud estimation, object localization, and hand pose annotations from human manipulation videos, we bridge 2D visual observations with 3D physical action space to provide visual-physical aligment supervision for VLA models.

Pretraining (Sec. 3.2.3). In Sec. 3.3 we describe the post-training procedure for adapting the model to robot tasks.

##### 3.1 Preliminaries

We begin by briefly formulating the setting of vision-language models (VLMs) and vision-language-action models (VLAs). A VLM is designed to map a set of visual inputs v = {v1,...,vT} (e.g., images or video frames) and a language instruction l (e.g., a question or command) into an aligned embedding space, producing a textual output y such as an answer or a caption:

y = fVLM(v,l).

In contrast, a VLA extends this formulation to the action domain, where the model must not only capture the semantic alignment between vision and language but also generate executable actions in a 3D physical environment. Given visual observations v and an instruction l, the VLA predicts an action chunk at = {at1,...,atH}:

at = fVLA(v,l).

This distinction highlights the central challenge addressed in this work: while VLMs excel at visual–semantic reasoning, VLAs additionally require grounding visual perception in 3D physical action space, which motivates our proposed Spatial-Aware VLA Pretraining paradigm.

- 3.2 Visual-Physical Alignment

- 3.2.1 Hand3D

Existing 3D VLMs typically obtain 3D supervision either from datasets with explicit 3D annotations or by estimating depth from 2D inputs. We follow their principles, but we focus on human demonstration videos, which inherently reveal how actions unfold in 3D space. These videos naturally provide rich source of visual-physical correspondences, allowing us to construct annotations that align 2D visual observations with action-relevant 3D information, forming the foundation for visual–physical alignment in VLA models.

Human video collection. We start from the collection of human manipulation videos covering diverse tasks and interaction scenarios. Following UniHand[15], we aggregate data from a wide range of different sources: (1) Motion capture datasets: Arctic[58], HOI4D[59], FPHA[60], H2O[61], OAKINK2[62], TACO[63], DexYCB[64], (2) VR-recorded datasets: EgoDex[65] and (3) Pseudo-annotated datasets: Taste-Rob[66], each providing recordings of humans performing object-centric manipulations and the corresponding hand motion sequences. To ensure consistency across heterogeneous datasets, we align all hand annotations to the unified MANO[67] representation, which serves as a standard parametric model of human hand pose and shape. For datasets with 3D hand joint annotations, we fit the MANO parameters via gradient optimization.

For Taste-Rob, which is a video-only dataset, we estimate the hand MANO parameters with HaWoR[68]. This alignment enables us to obtain high-quality, normalized annotations of hand trajectories that can be directly associated with the corresponding visual frames.

3D aware annotations. While prior works have explored learning from human videos, they largely remain confined to the 2D visual space, which restricts their ability to capture the physical grounding required for 3D action execution in robotic manipulation. We address this by extracting 3D visual annotations and 3D action annotations from human demonstrations.

As shown in Figure 2, we construct 3D visual annotations from human manipulation videos by combining point cloud estimation, object proposals, and hand pose information. Specifically, we first employ the Cut3R[47] model to obtain a dense per-frame point cloud estimation P = {(xi,yi,zi)}Ni=1, which provides 3D coordinates for each image pixel. Cut3R is chosen for its robustness in dynamic scenes and pretraining on human–object interaction data. For object localization, we apply Gemini-2.5-flash[69] to generate object proposals and GroundingDINO[70] to obtain 2D bounding boxes Bo, which approximate the spatial extent of each object in the image. By combining these bounding boxes with the depth estimation from P, we localize each object in the 3D space and obtain its approximate spatial position.

In parallel, we utilize MANO-based hand pose annotations m = {θ,r,τ,β}, provided in the dataset, together with camera extrinsics [R |t] and intrinsics K. The MANO parameters are transformed into the camera coordinate system to yield 3D joint locations Jh = {(xj,yj,zj)}21j=1. These joint coordinates are projected onto the image plane,

(u,v) = Π K[R |t](x,y,z)⊤ , Π(x′,y′,z′) = (x′/z′, y′/z′).

By checking the visibility of the projected joints, we filter out frames where the hand is outside the view. Previous works have often relied on depth estimation models to introduce 3D information; however, the relative scale provided by such models typically does not match the actual physical space and may introduce misalignment. This mismatch can be problematic for action learning, since actions are executed at real-world physical scales. To unify scales between the relative point cloud estimation P and the physical space, we perform scale calibration by matching hand joint positions Jh, which are absolute spatial positions, with corresponding point cloud coordinates.

To formalize this process, let Jhz = {jkz} denote the set of hand joint depths in absolute physical space, and J˜hz = {˜jkz} the corresponding depths obtained from the point cloud estimation P. We estimate a scale factor s as

s = mediank∈Ω jkz/˜jkz , where Ω is the set of valid joints determined by visibility and valid depth estimations. Applying s to the point cloud P yields the calibrated representation sP which ensures that both hands and objects are represented in a consistent 3D physical coordinate system. Consequently, we obtain spatially aligned annotations that encode the 3D geometry of hands and objects in consecutive frames.

Instructional data curation. Building on the spatial annotations derived from human manipulation videos, we construct a set of vision–question–answering style labels that explicitly encode the relationship between

- 2D visual observations and 3D action information. This step is crucial for grounding visual inputs to physical actions, enabling VLA models to understand how 3D movements are manifested in visual space. To achieve this, we leverage Gemini-2.5-flash[69] to generate four complementary categories of question–answer pairs: (1) Spatial Relationship: Given information about a hand and an object within several consecutive frames, we generate natural language questions and answers that describe their 3D spatial relations in the last frame (e.g., relative position, distance, or contact state). (2) Task Completion: Conditioned on a task description and corresponding video frames, we formulate instructions about how the hand should move in 3D to interact with the target object. (3) Hand Movement: Given two frames, we annotate the hand’s 3D trajectory between them, including movement direction and distance. (4) Camera Movement: The relative 3D transformation of the camera for pairs of frames, expressed in terms of rotation direction and degree.

For the 3D spatial relationships or hand and camera movements used in Category 1–3 we represent each by the pair (direction, distance). Let v = (x,y,z) ∈ R3 denote the displacement vector that characterizes the

relative 3D offset between two entities (e.g., object and hand, or hand positions across frames). The distance is defined as the Euclidean norm dist(v) = ∥v∥2. The direction is derived from the unit vector vˆ = v/∥v∥2 and discretized into axis-aligned language tokens with a component-wise threshold γ to filter out negligible components:

D = {right/left if |xˆ| > γ, up/down if |yˆ| > γ,

forward/backward if |zˆ| > γ},

For camera movement, we consider the relative transformation between two camera poses (R1,t1) and (R2,t2), and decompose it into a rotation, represented by an axis-angle description (u,ϕ), where u is the rotation axis and ϕ the rotation degree; and the translation, expressed as a direction vector and magnitude.

Through this process, dense 3D geometry are transformed into compact, linguistically grounded labels directly used to produce VQA-style supervisory pairs, linking human visual demonstrations to physically grounded actions. Importantly, this formulation allows the model not only to perceive spatial configurations but also to reason about dynamic changes—hand motions, task-oriented manipulations, and even camera movements—thereby strengthening the bridge between

Table 1: Data distribution of Hand3D-visual.

Category Count Proportion Sources Arctic 100,772 33.5% OakInk2 45,926 15.3% TACO 39,087 13.0% H2O 32,390 10.8% HOI4D 28,977 9.6% EgoDex 28,200 9.4% FPHA 12,918 4.3% Taste-Rob 7,181 2.4% Dex-YCB 4,917 1.6% Task Types Task Completion 206,409 68.7% Spatial Relations 74,887 24.9% Hand Movements 18,867 6.2% Camera Movements 205 0.1% Total 300,368 100%

- 2D visual observations and 3D action policies. With large-scale human data from nine heterogeneous human manipulation video sources, we uniformly sampled approximately 4K clips for annotation, ensuring coverage across diverse scenes, task types, and object interactions, and our pipeline produced around 300K instruction–answer pairs, which we denote as Hand3D-visual.

We provide some examples of the Hand3D-visual in Figure 3. The dataset covers four categories of manipulation-related tasks, encompassing diverse forms of spatial understanding essential for interaction. Such diversity ensures that the pretraining stage exposes the model to a wide range of manipulation scenarios, providing rich supervision for learning robust visual–physical alignment.

|[Figure 38]<br><br>Spatial Relationship<br><br>Question: Where is the black mug in relation to the right hand, and how far away is it? Answer: left and down and forward, 0.44m<br><br>Task Completion<br><br>[Figure 39]<br><br>Question: How should the right hand move to cut the white paper?<br><br>Answer: right and backward, 0.21m<br><br>Hand Movement Camera Movement<br><br>[Figure 40]<br><br>[Figure 41]<br><br>Question: How did the camera move between these two frames? Answer: left, 12.384 degrees<br><br>[Figure 42]<br><br>[Figure 43]<br><br>Question:<br><br>How far and in what direction did the left hand travel? Answer: right and down and backward, 0.18m<br><br>|
|---|

Figure 3: Examples of Hand3D-visual.

To complement the 3D visual annotations, we further construct 3D action annotations that capture the dynamic aspect of human manipulation in physical space. From hand motion sequences in human videos, we extract wrist trajectories as (xt,yt,zt) spatial coordinate series, which are discretized via uniform binning into motion tokens (mx1,my1,mz1,...,mxt ,myt ,mzt). Following UniHand[15], we obtain text instructions for videos, yielding 4M video–instruction–motion pairs. Specifically, each video is first segmented into 10s chunks and Gemini-2.5-Flash[69] is applied to generate chunk-level and second-level annotations including

task instructions and hand-object interactions. Then we utilize Gemini-2.5-Pro to construct three kinds of VQA-style tasks based on the annotations: instructional motion generation, motion translation and contextual motion prediction. We further filter out samples with insignificant 3D hand movement, resulting in a 1M curated dataset Hand3D-action, providing fine-grained 3D motion patterns and spatial trajectories, as well as higher-level physical reasoning knowledge such as object affordance and task decomposition. The detailed composition and distribution of the Hand3D-visual and Hand3D-action are provided in Table 1 and Table 2.

Table 2: Data distribution of Hand3D-action.

Category Count Proportion Sources EgoDex 758,050 73.5% Arctic 104,032 10.1% TACO 77,100 7.5% OakInk2 54,470 5.3% HOI4D 19,812 1.9% H2O 17,386 1.7% Task Types

Instructional Motion Generation 610,192 59.2% Contextual Motion Prediction 280,545 27.2% Motion Translation 140,113 13.6% Total 1,030,850 100%

###### 3.2.2 VIPA-VLA

Previous VLA models typically employ a semantic vision encoder to extract high-level visual semantics. However, such encoders lack the ability to provide features of 3D spatial structures. To address this limitation, we propose VIPA-VLA, a dual-encoder architecture that integrates both semantic and spatial visual representations (Figure 4 left). Specifically, in addition to a semantic vision encoder, we incorporate a 3D vision encoder Cut3R [47], which provides explicit geometric understanding of the scene. The semantic encoder produces visual embeddings Vsem, while the 3D encoder outputs spatial embeddings Vspa.

To effectively combine these complementary features, we introduce a fusion layer implemented as a crossattention module. The fusion layer design is inspired by VLM-3R[71], which follows a cross-attention mechanism, where semantic visual features attend to 3D spatial features extracted by the 3D encoder. Given vision features Vsem ∈ RN

s×ds, we first project them into a shared attention space, and then perform cross-attention where visual tokens query 3D spatial tokens. The output is projected back to the vision feature dimension using an output projection, resulting in Fspa. To integrate the spatial information while maintaining the pretrained visual semantics, the fusion layer applies a residual connection with a learnable scaling parameter α:

v×dv and spatial features Vspa ∈ RN

Vf = Vsem + αFspa Finally, dropout and layer normalization are applied to stabilize optimization.

To further enable the model to to understand fine-grained 3D motion trajectories extracted from human demonstrations, we extend the LLM’s token embedding space by introducing a set of motion tokens which discretize the 3D physical space in the second pretraining stage. Specifically, we tokenize each wrist trajectory point (xt,yt,zt) into three discrete motion tokens. To convert continuous 3D coordinates into discrete indices, we apply a uniform discretization over pre-defined bounded ranges. For each axis a ∈ {x,y,z}, we define a clipping range: a ∈ [amin,amax], and in our implementation, the x and y axes share the range [−0.5,0.5], while the z axis uses [0,1]. This effectively discretizes a 1m3 physical space directly in front of the camera into a structured token space. Each coordinate is discretized into one of K bins, ma ∈ {0,...,K − 1}, where we use K = 1024 in our experiments. Thus, each 3D waypoint (xt,yt,zt) is converted into a triplet of motion tokens mxt ,myt ,mzt, yielding the final tokenized motion sequence (mx1,my1,mz1,...,mxt ,myt ,mzt).

3D visual ann. 3D action ann.

text vision

action query

Pre-training Post-training

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

| |
|---|

| |
|---|

VIPA-VLA

###### VIPA-VLA

Robotic Dataset

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

| | | |
|---|---|---|
| |𝒉𝒄𝒐𝒏𝒅| |

text tokens motion tokens visual tokens

[Figure 44]

Fusion

𝒔𝒕

Hand3D

Action Head

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

| |
|---|

| |
|---|

| |
|---|

[Figure 45]

|[Figure 46]|
|---|

Cut3R ViT

|𝐚𝒕|
|---|

Figure 4: Model architecture of VIPA-VLA. A dual-encoder including a semantic vision encoder and a 3D encoder produces fused spatial–semantic features through a cross-attention fusion layer. During pre-training, the vision tokens are aligned with text and motion tokens using 3D visual and 3D action annotations. During post-training, action queries interact with fused visual–language features to produce conditions, which is combined with the robot state and processed by a flow-matching action head to predict actions for robotic manipulation.

###### 3.2.3 Spatial-Aware VLA Pretraining

To effectively align semantic perception with physical understanding, we design a Spatial-Aware VLA Pretraining strategy composed of two stages, enabling progressive learning of visual–physical alignment (Figure 1). We first initialize our model using a pretrained VLM to inherit semantic understanding, while incorporating the pretrained 3D vision encoder and a randomly initialized fusion layer. In Stage 1, we freeze all pretrained parameters and train only the fusion layer using the 3D visual annotation VQA data. The objective is to align the semantic embeddings Vsem and spatial embeddings Vspa through the fusion layer, encouraging reasoning about 3D spatial relations.

In Stage 2, we extend the LLM vocabulary to include a set of motion tokens and then train the model on

- 3D action annotations. In this stage, the semantic and spatial encoders are frozen, and the LLM is trained to predict motion tokens conditioned on fused visual and textual inputs. This phase enables the model to acquire fine-grained spatial reasoning and action-level understanding, learning how visual cues correspond to physically grounded motion patterns. Through the Spatial-Aware VLA Pretraining, VIPA-VLA progressively aligns 2D semantic perception, 3D spatial understanding, and action reasoning — achieving comprehensive visual–physical alignment for downstream robotic policy learning.

##### 3.3 Post-Training

After pretraining, VIPA-VLA has acquired alignment between 2D observations and 3D physical/action representations. In the post-training stage, we fine-tune this initialization on downstream robot tasks by attaching an action head and training it to produce executable action chunks. For implementation, we use a diffusion transformer (DiT) model, parameterized with θ.

Let v and l denote visual frames and language instruction, and let Qa be a fixed set of action queries. We extract conditional context from the pretrained VLM backbone as hcond = VLMϕ(v,l,Qa), where the hidden states corresponding to Qa are used as the DiT conditioning. The ground-truth action chunk for timestep t is denoted by at, and the robot state embedding by st. Following flow matching[83], we generate a noisy action trajectory by linearly interpolating between a random noise vector ϵ and the ground-truth action:

a˜t(τ) = (1 − τ)ϵ + τ at, τ ∼ U(0,1).

The DiT receives the concatenation of a˜t(τ) and the state embedding st as its input representation: hDiT = concat a ˜t(τ), st . Conditioned on hcond, the DiT predicts the instantaneous flow vector vθ. The training

- Table 3: Success rates (%) on the LIBERO benchmark. Results are reported for four task suites, with each suite evaluated across 500 trials. Experiments are conducted under both single-view and two-view input settings. All results are from public reports, except for GR00T N1.5∗, which we reproduce using the released model. The results of MaIL† is from the ED-Ma version.

Model Robo-PT LIBERO-S (%) LIBERO-O (%) LIBERO-G (%) LIBERO-L (%) Avg. (%) Models with single-view input

TraceVLA[72] 84.6 85.2 75.1 54.1 74.8 OpenVLA[10] 84.7 88.4 79.2 53.7 76.5 SpatialVLA[49] 88.2 89.9 78.6 55.5 78.1 DiT Policy[73] 84.2 96.3 85.4 63.8 82.4 CoT-VLA[74] 87.5 91.6 87.6 69.0 83.9 ThinkAct[75] 88.3 91.4 87.1 70.9 84.4 TriVLA[76] 91.2 93.8 89.8 73.2 87.0 4D-VLA[77] 88.9 95.2 90.9 79.1 88.6 GR00T N1.5∗[7] 91.4 97.6 94.0 85.6 92.1 VIPA-VLA 92.6 97.2 94.2 85.6 92.4

Models with two-view input

MaIL†[78] 74.3 90.1 81.8 78.6 83.5 π0-FAST [79] 96.4 96.8 88.6 60.2 85.5 MolmoAct[80] 87.0 95.4 87.6 77.2 86.6 GR00T N1[7] 94.4 97.6 93.0 90.6 93.9 π0 [6] 98.0 96.8 94.4 88.4 94.4 UniVLA[81] 95.4 98.8 93.6 94.0 95.5 π0.5[82] 98.8 98.2 98.0 92.4 96.9 VIPA-VLA 96.6 98.6 97.0 95.0 96.8

objective minimizes the deviation between this prediction and the oracle transport direction given by at − ϵ: LFM = Ea

t,τ,ϵ,v,l ∥vθ − (at − ϵ)∥22 .

During training, only the LLM backbone and the action head fθ are updated to adapt the model to action execution. The illustration is shown in Figure 4 (right).

#### 4 Experiment

##### 4.1 Implementation Details

VIPA-VLA is initialized from InternVL3.5-2B[84]. During the first pre-training stage, for the spatial relationship and task completion tasks, the model is given 1–4 randomly sampled consecutive frames as input, while for the hand movement and camera movement tasks, two frames are provided. All video frames are sampled at 1 fps. In the second pre-training stage and the post-training stage, we use single frame as visual input.

The training consists of three stages, and the training detials are as follows:

- Stage 1 We pretrain VIPA-VLA with Hand3D-visual for a single epoch in the first stage, cost around 6 hours on 8 NVIDIA A800 GPUs. Only the fusion layer is optimized, with AdamW optimizer using the learning rate of 1e − 5. We set warm up ratio to 0.03 and the weight decay to 0.01, with a cosine scheduler. The global batch size is 32. For the fusion layer, the spatial scaling parameter α is initialized as 0.5. The image frames are resized to 448 × 448.
- Stage 2 In the second stage we use Hand3D-action to optimize the fusion layer and the LLM backbone of VIPA-VLA, requiring around 20 hours on 8 NVIDIA A800 GPUs. The training hyper-parameters are kept the

- Table 4: Success rates (%) on the RoboCasa benchmark. Models are evaluated on 24 tasks (8 for Pick & Place, 6 for Doors / Drawers, 10 for Others), with each task evaluated across 50 trials.

Model Pick & Place Doors / Drawers Others Avg. GR00T N1 18.6 50.2 39.1 36.0 π0.5 21.5 57.8 44.9 41.4 VIPA-VLA 20.8 67.7 52.8 45.8

same as in the first stage.

- Stage 3 In the post-training stage, we freeze the visual encoder and the 3D encoder, and the global batch size is 128 (for LIBERO and real robot tasks) or 256 (for RoboCasa), with a learning rate 5e − 5. The warm up ratio is 0.05 and the weight decay is 1e − 5. For both the LIBERO benchmark and the real robot tasks, we train VIPA-VLA for 30K steps, requiring around 5 hours on 8 NVIDIA A800 GPUs. For the RoboCasa benchmark, we train for 60K steps, taking approximately 40 hours on 8 NVIDIA A800 GPUs.

##### 4.2 Simulation Robot Tasks

Benchmarks. We first evaluate VIPA-VLA’s performance for robotic manipulation in simulation environments, which assesses how well the model can generalize spatially grounded reasoning from human demonstrations to robotic scenarios. Specifically, we adopt LIBERO[85], which is a standard benchmark for evaluating robot manipulation capabilities, emphasizing robustness and generalization, and RoboCasa[86], which poses significantly greater challenges than LIBERO—featuring more diverse layouts, cluttered scenes, and more complex visual observations, demanding more accurate and robust 3D spatial understanding from VLA models. LIBERO consists of four task suites: Spatial, Object, Goal and Long, covering different task settings and scene layouts.

Results on LIBERO We report the average success rates across 500 trials for each task suite in Table 3. VIPA-VLA achieves strong performance across all LIBERO task suites under both single-view and two-view input settings. This demonstrates that the Spatial-Aware VLA Pretraining learned from human videos generalizes effectively to simulated robotic environments. Notably, our model does not rely on any robot data during pretraining, yet it surpasses a number of previous methods and achieves results comparable to π-series and GR00T models, which are strong baselines pretrained on large-scale robot datasets. This indicates that the spatial reasoning ability acquired from human demonstrations—bridging 2D visual cues with 3D spatial understanding—can significantly help robotic manipulation even in simulation. Among baselines that explicitly model spatial reasoning, such as SpatialVLA[49], TraceVLA[72], and MolmoAct[80], our approach consistently performs better, highlighting the effectiveness of learning visual–physical alignment from diverse human demonstrations.

Results on RoboCasa For experiments on RoboCasa, we adopt a three-view input configuration and train VIPA-VLA using only 50 human demonstrations per task. As shown in Table 4, across the diverse RoboCasa tasks, VIPA-VLA achieves the best overall performance, demonstrating strong generalization under multi-view observations and limited demonstration data. Notably, our model attains significant improvements in the Doors/Drawers categories (+9.9%), which require precise spatial localization. These results highlight that the Spatial-Aware Pretraining effectively equips the model with more reliable 2D–3D grounding, enabling accurate spatial reasoning even in visually diverse and geometrically challenging environments.

##### 4.3 Real Robot Tasks

We further evaluate our model in real-world robotic manipulation tasks, which provide a more realistic

- 3D physical setting and require more accurate spatial perception and action grounding, making them ideal for assessing the benefits of our Spatial-Aware VLA Pretraining with human videos. The experiments are conducted with a 7-DoF Franka Research 3 arm, a 6-DoF Inspire hand, and two RealSense L515 cameras for visual observation.

We design three manipulation tasks to test different aspects of spatial perception and reasoning: (1) PutThree-Obj – sequentially place three fruits into a drawer, assessing the ability to localize and manipulate multiple objects; (2) Wipe-Board – use a cloth to remove all pen marks from a whiteboard, requiring flexible spatial reasoning over irregularly shaped target regions; (3) Water-Plant – pick up a watering can and water a plant, demanding precise spatial localization and control of fine-grained 3D motion. Each task is further decomposed into several sub-tasks, which are used to measure fine-grained execution performance. And we also design unseen environment setups for each task for evaluation of the generalization abilities. Details of the configurations are as follows:

- • Put-Three-Obj: The task requires the robot to open the drawer, sequentially pick and place three fruits (apple, banana, plum) from the table into the drawer, and close the drawer. We define five sub-tasks: open drawer, pick & place apple, pick & place banana, pick & place plum, and close drawer. For the unseen environment setup, the table surface is modified by covering it with tablecloths of unseen colors.

- • Wipe-Board: The task requires the robot to picks up a cloth and wipes the pen marks off the whiteboard. There are three sub-tasks defined: pick up cloth, wipe pen marks, and clean entire whiteboard. For the unseen environment setup, we change the color of the marker used to draw the pen marks.

- • Water-Plant: The task requires the robot to pick up a watering can, locate the plant, and water it by pressing the spray handle. There are three sub-tasks defined: pick up watering can, locate plant, and press spray handle.

The illustration of the tasks are shown in Figure 5. For each task, we collect 50 teleoperated trajectories for post-training and conduct 10 evaluation trials to measure task success rates. During evaluation, we randomize the placement of objects on the table and the distribution of pen marks on the whiteboard for each trial. This setup ensures that the learned policies are tested on their generalization ability rather than memorization of fixed configurations. Besides the success rates of the whole tasks, we also report the success rates computed by the proportion of completed sub-tasks, averaged across all trials.

Seen Environments Unseen Environments

|Put-Three-Obj<br><br>[Figure 47]|Wipe-Board<br><br>[Figure 48]|Water-Plant<br><br>[Figure 49]|
|---|---|---|

|Put-Three-Obj<br><br>[Figure 50]|Wipe-Board<br><br>[Figure 51]|
|---|---|

Figure 5: Real Robot Task Settings.

As shown in Table 5, VIPA-VLA achieves the best overall performance across all real-world manipulation tasks, demonstrating that the spatial understanding acquired during pretraining effectively helps real robotic manipulations, leading to more robust and generalizable behavior in real-world manipulation scenarios. Compared with the ablation baseline InternVL3.5, our model achieves substantial improvements on all tasks, confirming the effectiveness of our proposed Spatial-Aware Pretraining and dual-encoder architecture in enhancing spatial grounding and generalization. Although VIPA-VLA achieves lower complete-task success rate on Put-Three-Obj, it attains higher sub-task success rate. This is because baseline models often fail early in the sequence—sometimes unable to complete even the first step—indicating less stable spatial reasoning and control. In contrast, VIPA-VLA exhibits more consistent progress across task stages, reflecting stronger robustness in real-world execution.

- Table 5: Success rates (%) of VIPA-VLA v.s. baseline models on real robot tasks. Results are shown as sub-task / whole-task success rates.

Task Put-Three-Obj Wipe-Board Water-Plant

GR00T N1.5 [7] 48% / 40% 57% / 30% 53% / 30% Being-H0[15] 38% / 20% 40% / 10% 37% / 20% InternVL3.5 [84] 34% / 10% 43% / 10% 37% / 20% VIPA-VLA 52% / 10% 83% / 60% 57% / 50%

We further evaluate the models in unseen environments. As shown in Table 6, VIPA-VLA maintains strong performance when evaluated in unseen visual environments, significantly outperforming all baselines. While

- Table 6: Success rates (%) of VIPA-VLA v.s. baseline models on real robot tasks with unseen environments. Results are shown as sub-task / whole-task success rates.

Task Put-Three-Obj-Unseen Wipe-Board-Unseen

GR00T N1.5 [7] 28% / 0% 43% / 10% Being-H0[15] 16% / 0% 33% / 10% InternVL3.5 [84] 42% / 10% 40% / 10% VIPA-VLA 44% / 20% 83% / 50%

other models often exhibit a sharp degradation in performance—often failing to complete even the initial steps—VIPA-VLA retains high sub-task and whole-task success rates, demonstrating enhanced robustness and generalization. These results highlight that the Spatial-Aware Pretraining effectively equips the model with transferable 3D spatial understanding learned from human videos. By grounding 2D visual observations into 3D physical representations, our model generalizes more reliably to new scenes.

We show qualitative examples of VIPA-VLA executing real robot tasks in Figure 6. Across diverse scenes and object configurations, the model demonstrates reliable spatial grounding and robust generalization: it accurately localizes objects, adjusts to varying layouts, and completes multi-step manipulation sequences with consistent trajectories. These examples highlight the model’s ability to transfer its spatial-aware pretraining to real-world settings, enabling precise and stable task execution even under unseen spatial variations.

Put-Three-Obj

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Put-Three-Obj-Unseen

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Wipe-Board

Wipe-Board-Unseen

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Water-Plant

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Figure 6: Qualitative examples of VIPA-VLA performing real robot tasks.

We also provide analysis of typical failure cases of VIPA-VLA and the baseline model InternVL-3.5 in Figure 7. For VIPA-VLA, failures generally occur in fine-grained manipulation steps—such as slight misalignment when grasping a small object—while the overall spatial localization and action planning remain correct. These errors often arise from subtle control inaccuracies rather than misunderstanding the physical space. In contrast, the baseline model frequently fail due to inaccurate spatial grounding: they may reach toward incorrect regions or misjudge object positions that reflect poor 2D–3D correspondence. Such failures indicate weaker spatial understanding, causing the policy to be stuck before meaningful manipulation begins.

##### 4.4 Ablation Studies

Architecture and the Spatial-Aware VLA Pretraining. We conduct ablation studies on the LIBERO benchmark to evaluate the effectiveness of our model architecture and the proposed Spatial-Aware VLA Pretraining. This analysis isolates the contributions of the dual-encoder design and the pretraining strategy with visual-physical alignment.

|VIPA-VLA|VIPA-VLA|
|---|---|

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

|InternVL-3.5|InternVL-3.5|InternVL-3.5|
|---|---|---|

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Figure 7: Failure examples of VIPA-VLA and InternVL-3.5 performing real robot tasks. Table 7: Ablations (%) on the LIBERO benchmark.

Model Spa. Obj. Goal Long Avg. VIPA-VLA 92.6 97.2 94.2 85.6 92.4

- – Pretraining 90.8 97.0 93.0 84.0 91.2 (-1.2%)

- – Dual Encoder 90.0 97.2 92.4 81.8 90.4 (-2.0%)

- – Both 89.2 95.2 90.0 80.4 88.7 (-3.7%)

As shown in Table 7, both the Spatial-Aware Pretraining on human videos and the proposed dual-encoder architecture contribute to overall performance improvements. Removing either component leads to a consistent performance drop across all task suites. The pretraining provides the model with rich 3D motion and perception priors learned from human demonstrations, while the dual encoder enables more effective fusion of 3D visual representations. When combined, these two designs further enhance the model’s spatial reasoning and generalization, resulting in the strongest performance.

Effect of Spatial-Aware VLA Pretraining. Here we first evaluate the effectiveness of the first stage pretraining on 3D visual annotations in improving spatial understanding capabilities. Specifically, we assess how well the model can reason about spatial relations and 3D geometry in visual scenes. We report results on the Hand3D-test, which consists of 2K VQA pairs from unseen videos not included in the training corpus. The evaluation focuses on two aspects: distance accuracy and direction accuracy. Direction accuracy is measured by assessing the correctness of predictions along the three principal axes; each axis contributes a binary score of 0 or 1. Distance accuracy is quantified by the average prediction error in meters.

We compare VIPA-VLA-PT against two ablations: (1)InternVL3.5: the backbone VLM without additional pretraining. (2)InternVL3.5+Hand3D: The backbone model pretrained with Hand3D, without the proposed dual encoder architecture. This setup allows us to disentangle the contribution of Hand3D pretraining and the architecture of VIPA-VLA. For InternVL3.5, we provide an example answer in the prompt for valid answering format.

Table 8: Evaluation of 3D spatial understanding.

Model Dist. Err. (m) ↓ Dir. Scr. ↑

InternVL3.5 0.18 1.22/3 InternVL3.5 +Hand3D 0.14 1.75/3

###### VIPA-VLA-PT 0.12 1.82/3

From the results shown in Table 8, the backbone InternVL3.5 performs poorly on the spatial reasoning tasks, indicating its limited understanding of spatial relationships. Pretraining on Hand3D notably improves performance, confirming that 3D visual annotations effectively enhance the model’s spatial grounding through visual-physical alignment. Our proposed VIPA-VLA further achieves a consistent performance gain, demonstrating that introducing the 3D visual encoder and the fusion mechanism brings additional benefits beyond data-level supervision. Notably, only the fusion layer is trained at this stage, showing that aligning

[Figure 104]

- Figure 8: Comparison between VIPA-VLA-PT and InternVL3.5 on Hand3D-test. Left: histogram of direction scores; Right: histogram of distance errors

semantic and spatial visual features alone already yields substantial gains.

We further show detailed comparison between VIPA-VLA-PT and InternVL3.5 in Figure 8. After Spatial-Aware Pretraining on 3D visual annotations, our model makes substantially fewer severe errors and attains highly accurate predictions in about 30% of the cases, reflecting precise and reliable 3D spatial grounding ability obtained in this stage.

We then show qualitative results of the second stage pretraining on 3D action annotations in Figure 9. The predicted trajectories closely align with the instructions, showing that the model effectively understands the mapping between visual observations and physical space. Compared to the ground-truth trajectories, which often exhibit noisy and redundant motion due to natural human variability, our model produces smoother and more goal-directed trajectories. Moreover, the predictions reflect learned affordance knowledge — for example, when manipulating a wooden spoon, the model grasps it near the handle end.

|Place the yellow plate on top of the blue plate|Grasp the wooden spoon and move it towards the white pot|Grasp the wooden spoon and<br><br>move it towards the cutting<br><br>board|Grasp the hammer and lift it<br><br>upward|
|---|---|---|---|
|[Figure 105]|[Figure 106]|[Figure 107]|[Figure 108]|

- Figure 9: Visualization of the predicted motion trajectories from VIPA-VLA after second stage pretraining (blue lines) and the ground-truth trajectories (red lines).

#### 5 Conclusion

In this work, we address the critical 2D–3D gap in vision-language-action (VLA) models through SpatialAware VLA Pretraining from large-scale human videos. We introduce Hand3D, a dataset providing 3D visual and action annotations derived from human manipulation, and propose VIPA-VLA, a dual-encoder architecture designed for visual-physical alignment. The proposed paradigm effectively aligns 2D perception with 3D physical understanding, yielding strong improvements in spatial reasoning and downstream robotic manipulation. Our results validate the effectiveness of Spatial-Aware Pretraining from human demonstrations. In future, this pretraining paradigm can be combined with robot data pretraining to achieve a more comprehensive and effective overall pretraining strategy.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [4] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

- [5] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

- [6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164, 2024.

- [7] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

- [8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

- [9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

- [10] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

- [11] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023.

- [12] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

- [13] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

- [14] Jianglong Ye, Keyi Wang, Chengjing Yuan, Ruihan Yang, Yiquan Li, Jiyue Zhu, Yuzhe Qin, Xueyan Zou, and Xiaolong Wang. Dex1b: Learning with 1b demonstrations for dexterous manipulation. arXiv preprint

- arXiv:2506.17198, 2025.

[15] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: Vision-language-action pretraining from large-scale human videos. arXiv preprint

- arXiv:2507.15597, 2025.

- [16] Ankit Goyal, Valts Blukis, Jie Xu, Yijie Guo, Yu-Wei Chao, and Dieter Fox. Rvt2: Learning precise manipulation from few demonstrations. In RSS, 2024.

- [17] Jiawei He, Danshi Li, Xinqiang Yu, Zekun Qi, Wenyao Zhang, Jiayi Chen, Zhaoxiang Zhang, Zhizheng Zhang, Li Yi, and He Wang. Dexvlg: Dexterous vision-language-grasp model at scale. arXiv preprint arXiv:2507.02747, 2025.

- [18] Ziye Huang, Haoqi Yuan, Yuhui Fu, and Zongqing Lu. Efficient residual learning with mixture-of-experts for universal dexterous grasping. In ICLR, 2025.

- [19] Weikang Wan, Haoran Geng, Yun Liu, Zikang Shan, Yaodong Yang, Li Yi, and He Wang. Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning. In ICCV, 2023.

- [20] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Heming Cui, et al. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. arXiv preprint arXiv:2505.03233, 2025.

- [21] Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Yitao Liang, Yaodong Yang, and Yuanpei Chen. Dexgraspvla: A vision-language-action framework towards general dexterous grasping. arXiv preprint arXiv:2502.20900, 2025.

- [22] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In ICRA, 2024.

- [23] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

- [24] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

- [25] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In RSS, 2024.

- [26] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

- [27] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022.

- [28] Lirui Wang, Xinlei Chen, Jialiang Zhao, and Kaiming He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. In Neurips, 2024.

- [29] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In CoRL, 2023.

- [30] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In ICLR, 2024.

- [31] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025.

- [32] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Rich Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

- [33] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding, reasoning, and planning. In CVPR, 2024.

- [34] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024.

- [35] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024.

- [36] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual understanding and reasoning. arXiv preprint arXiv:2403.11401, 2024.

- [37] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. In ICML, 2024.

- [38] Boshen Xu, Yuting Mei, Xinbi Liu, Sipeng Zheng, and Qin Jin. Egodtm: Towards 3d-aware egocentric videolanguage pretraining. arXiv preprint arXiv:2503.15470, 2025.

- [39] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. In NeurIPS, 2023.

- [40] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.

- [41] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In ECCV, 2020.

- [42] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In ECCV, 2020.

- [43] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Muller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.

- [44] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024.

- [45] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024.

- [46] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025.

- [47] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025.

- [48] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.

- [49] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

- [50] Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video. arXiv preprint arXiv:2410.24221, 2024.

- [51] Haoqi Yuan, Bohan Zhou, Yuhui Fu, and Zongqing Lu. Cross-embodiment dexterous grasping with reinforcement learning. In ICLR, 2025.

- [52] Himanshu Gaurav Singh, Antonio Loquercio, Carmelo Sferrazza, Jane Wu, Haozhi Qi, Pieter Abbeel, and Jitendra Malik. Hand-object interaction pretraining from videos. arXiv preprint arXiv:2409.08273, 2024.

- [53] Yaru Niu, Yunzhe Zhang, Mingyang Yu, Changyi Lin, Chenhao Li, Yikai Wang, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Bingqing Chen, et al. Human2locoman: Learning versatile quadrupedal manipulation with human pretraining. arXiv preprint arXiv:2506.16475, 2025.

- [54] Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, David J Yoon, Ryan Hoque, Lars Paulsen, et al. Humanoid policy˜ human policy. arXiv preprint arXiv:2503.13441, 2025.

- [55] Hanzhi Chen, Boyang Sun, Anran Zhang, Marc Pollefeys, and Stefan Leutenegger. Vidbot: Learning generalizable 3d actions from in-the-wild 2d human videos for zero-shot robotic manipulation. In CVPR, 2025.

- [56] Teli Ma, Jia Zheng, Zifan Wang, Ziyao Gao, Jiaming Zhou, and Junwei Liang. Glover++: Unleashing the potential of affordance learning from human behaviors for robotic manipulation. arXiv preprint arXiv:2505.11865, 2025.

- [57] Alexey Gavryushin, Xi Wang, Robert JS Malate, Chenyu Yang, Xiangyi Jia, Shubh Goel, Davide Liconti, René Zurbrügg, Robert K Katzschmann, and Marc Pollefeys. Maple: Encoding dexterous robotic manipulation priors learned from egocentric videos. arXiv preprint arXiv:2504.06084, 2025.

- [58] Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges. Arctic: A dataset for dexterous bimanual hand-object manipulation. In CVPR, 2023.

- [59] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In CVPR, 2022.

- [60] Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim. First-person hand action benchmark with rgb-d videos and 3d hand pose annotations. In CVPR, 2018.

- [61] Taein Kwon, Bugra Tekin, Jan Stühmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In ICCV, 2021.

- [62] Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu. Oakink2: A dataset of bimanual hands-object manipulation in complex task completion. In CVPR, 2024.

- [63] Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi. Taco: Benchmarking generalizable bimanual tool-action-object understanding. In CVPR, 2024.

- [64] Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, et al. Dexycb: A benchmark for capturing hand grasping of objects. In CVPR, 2021.

- [65] Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

- [66] Hongxiang Zhao, Xingchen Liu, Mutian Xu, Yiming Hao, Weikai Chen, and Xiaoguang Han. Taste-rob: Advancing video generation of task-oriented hand-object interaction for generalizable robotic manipulation. In CVPR, 2025.

- [67] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 2017.

- [68] Jinglei Zhang, Jiankang Deng, Chao Ma, and Rolandos Alexandros Potamias. Hawor: World-space hand motion reconstruction from egocentric videos. In CVPR, 2025.

- [69] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [70] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.

- [71] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025.

- [72] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. TraceVLA: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. In ICLR, 2025.

- [73] Zhi Hou, Tianyi Zhang, Yuwen Xiong, Hengjun Pu, Chengyang Zhao, Ronglei Tong, Yu Qiao, Jifeng Dai, and Yuntao Chen. Diffusion transformer policy. arXiv preprint arXiv:2410.15959, 2024.

- [74] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In CVPR, 2025.

- [75] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Visionlanguage-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815, 2025.

- [76] Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Xiangyang Xue, and Yanwei Fu. Trivla: A unified triple-system-based unified vision-language-action model for general robot control. arXiv preprint arXiv:2507.01424, 2025.

- [77] Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, et al. 4d-vla: Spatiotemporal vision-language-action pretraining with cross-scene calibration. arXiv preprint arXiv:2506.22242, 2025.

- [78] Xiaogang Jia, Qian Wang, Atalay Donat, Bowen Xing, Ge Li, Hongyi Zhou, Onur Celik, Denis Blessing, Rudolf Lioutikov, and Gerhard Neumann. MaIL: Improving imitation learning with selective state space models. In CoRL, 2024.

- [79] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

- [80] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.

- [81] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025.

- [82] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. π0.5: a vision-language-action model with open-world generalization. In CoRL, 2025.

- [83] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023.

- [84] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

- [85] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

- [86] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

