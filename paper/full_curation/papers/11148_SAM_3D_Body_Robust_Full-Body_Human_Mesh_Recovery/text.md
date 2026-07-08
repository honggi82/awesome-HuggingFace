# arXiv:2602.15989v1[cs.CV]17Feb2026

## SAM 3D Body: Robust Full-Body Human Mesh Recovery

Xitong Yang⋆, Devansh Kukreja⋆, Don Pinkus⋆, Anushka Sagar, Taosha Fan, Jinhyung Park◦, Soyong Shin◦, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Matt Feiszli†, Jitendra Malik†, Piotr Dollar†, Kris Kitani†

Meta Superintelligence Labs ⋆Core Contributor, ◦Intern, †Project Lead

We introduce SAM 3D Body (3DB), a promptable model for single-image full-body 3D human mesh recovery (HMR) that demonstrates state-of-the-art performance, with strong generalization and consistent accuracy in diverse in-the-wild conditions. 3DB estimates the human pose of the body, feet, and hands. It is the first model to use a new parametric mesh representation, Momentum Human Rig (MHR), which decouples skeletal structure and surface shape. 3DB employs an encoder–decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling userguided inference similar to the SAM family of models. We derive high-quality annotations from a multi-stage annotation pipeline that uses various combinations of manual keypoint annotation, differentiable optimization, multi-view geometry, and dense keypoint detection. Our data engine efficiently selects and processes data to ensure data diversity, collecting unusual poses and rare imaging conditions. We present a new evaluation dataset organized by pose and appearance categories, enabling nuanced analysis of model behavior. Our experiments demonstrate superior generalization and substantial improvements over prior methods in both qualitative user preference studies and traditional quantitative analysis. Both 3DB and MHR are open-source.

Demo: https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d Code: https://github.com/facebookresearch/sam-3d-body Website: https://ai.meta.com/sam3d

[Figure 1]

Figure 1 Full-body human mesh recovery results using SAM 3D Body (3DB). Our model demonstrates robust performance in estimating challenging poses across diverse viewpoints and produces accurate body and hand pose estimations within a unified framework.

1 Introduction

Estimating 3D human pose (skeleton pose and structure) and shape (soft body tissue) from images is an essential capability for vision and embodied AI systems to understand and interact with people. Despite notable progress in human mesh recovery (HMR) Goel et al. (2023); Dwivedi et al. (2024); Patel and Black (2025); Wang et al. (2025b,c), existing approaches still exhibit unsatisfactory robustness when applied to in-the-wild images, which limits their applicability to real-world scenarios such as robotics Peng et al. (2018); Patel et al. (2022); Vasilopoulos et al. (2020) and biomechanics Pearl et al. (2023). In particular, current models often fail on individuals presenting challenging poses, severe occlusion, or captured from uncommon viewpoints. They also struggle to reliably estimate both the overall body pose and the fine details of the

hands and feet in a unified full-body framework.

We argue that the primary challenges in developing a robust full-body human mesh recovery model stem from both the data and model aspects. First, collecting large-scale and diverse human pose datasets with high-quality mesh annotations is inherently difficult and computationally costly. Most existing datasets either suffer from low diversity due to laboratory capture settings Chao et al. (2021); Ionescu et al. (2013); Joo et al. (2017) or from low mesh quality resulting from pseudo-labeling Von Marcard et al. (2018); Andriluka et al. (2014). Second, current HMR architectures do not adequately address the distinct optimization mechanisms required for body and hand pose estimation, nor do they incorporate effective training strategies to handle uncertainty and ambiguity from monocular images.

In this work, we present SAM 3D Body (3DB), a robust full-body HMR model fueled by large-scale, high-quality human pose data curated by our data engine.

Robust Full-body HMR Model. We make three main contributions to improve model performance on both body and hand pose estimation. (i) We propose a novel promptable encoder–decoder architecture Kirillov et al.

- (2023); Ravi et al. (2024) that enables the model to condition on optional 2D keypoints, masks or camera information for controllable pose estimation. This promptable design naturally facilitates interactive guidance in ambiguous or challenging scenarios during training, and provides a coherent approach to integrate hand and body predictions. (ii) Our model utilizes a shared image encoder and two separate decoders for the body and hands. This two-way-decoder design effectively alleviates conflicts in optimizing body and hand pose estimation, which arise from differences in input resolution, camera estimation, and supervision objectives. (iii) Unlike most prior work that relies on the SMPL Loper et al. (2015) human mesh model, we build 3DB on a new parametric mesh representation, MHR Ferguson et al. (2025), which decouples skeletal pose and body shape, providing richer control and interpretability for full-body reconstruction.

Data Engine for Diverse Human Pose and High-quality Annotation. HMR methods have increasingly turned to large-scale training data for higher performance Goel et al. (2023); Cai et al. (2023); Yin et al. (2025). However, high-quality 3D supervision remains scarce, and existing in-the-wild datasets are still limited in scale and diversity. To this end, we design a new data creation pipeline that features: (i) Data Quality: Our annotation pipeline combines various combinations of components such as geometric constraints, parametric priors, and dense keypoint regression, which automatically yields high-quality 3D human mesh annotations. (ii) Data Quantity: We curate data from large licensed stock photo repositories, multiple multi-view capture datasets, and synthetic data. We create a large scale of 7 million images with high-quality annotation. (iii) Data Diversity: Our data is diversified using a VLM-based data engine that mines for in-the-wild challenging images and routes them for annotation. This ensures coverage of rare poses, difficult viewpoints, and varied appearances, providing a more diverse dataset for supervision.

Together, the data engine and full-body HMR model enable 3DB to recover high-fidelity full-body human meshes from a single image. 3DB achieves state-of-the-art performance across both body and hand pose estimation. Extensive experiments demonstrate that 3DB consistently outperforms prior HMR methods on standard metrics, generalizes better to unseen datasets, and is preferred by users in a study of 7,800 participants, achieving a significant 5 : 1 win rate in visual quality. To our knowledge, it is the first single model that delivers the best performance to body-specialized models and comparable performance to hand-specialized models, while providing interactive control and strong robustness under challenging poses and in-the-wild scenarios.

- 2 Related Work

Human Mesh Models: The most widely used human mesh model is SMPL Loper et al. (2015), which parameterizes human body into pose and shape. SMPL-X Pavlakos et al. (2019) goes further to include hands (MANO Romero et al. (2022)) and faces (FLAME Li et al. (2017)). SMPL models intertwine the skeletal structure and soft-tissue mass within the shape space, which can limit interpretability (e.g., the parameters do not always map directly to bone lengths) and controllability. Alternatively, Momentum Human Rig Ferguson et al. (2025), an enhancement of ATLAS Park et al. (2025), explicitly decouples the skeletal structure and body shape, and we adopt it as our representation of the human body.

[Figure 2]

[Figure 3]

||[Figure 4]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|
<br><br>Body and hand boxes|
|---|

[Figure 5]

|[Figure 6]<br><br>CROSS-ATTENTION<br><br>BODY DECODER|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

| |
|---|

| |
|---|

| |
|---|

+

+

Pose

[Figure 10]

[Figure 11]

[Figure 12]

| |
|---|

| |
|---|

| |
|---|

+

+

Shape

[Figure 13]

|[Figure 14]<br><br>CROSS-ATTENTION<br><br>HAND DECODER|
|---|

#### ENCODER

[Figure 15]

[Figure 16]

[Figure 17]

| |
|---|

| |
|---|

| |
|---|

+

+

Skeleton

[Figure 18]

[Figure 19]

[Figure 20]

| |
|---|

| |
|---|

| |
|---|

+

+

Camera

Feature

Mask

Position

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Momentum Human Rig (MHR)

*3DKeypoints

- *2DKeypoints
- *Learnable token

CameraToken

*HandPosition

Keypointprompts

MHRToken

- Figure 2 SAM 3D Body Model Architecture. We employ a promptable encoder–decoder architecture with a shared image encoder and separate decoders for body and hand pose estimation.

Human Mesh Recovery (HMR): Early HMR methods like HMR 2.0 Goel et al. (2023) were body-only methods that predicted the body without articulated hands or feet Kolotouros et al. (2019); Li et al. (2022); Dwivedi et al. (2024). Instead, 3DB follows the more recent paradigm of full-body methods Baradel et al. (2024); Choutas et al. (2020); Rong et al. (2021); Cai et al. (2023); Wang et al. (2025c) that estimate body+hands+feet. There are also part-specific hand mesh recovery methods Pavlakos et al. (2024); Potamias et al. (2025) that only estimate the pose and shape of the hands, which usually have more accurate performance compared to full-body methods. In contrast, 3DB shows strong performance on both hand and full-body estimation.

Promptable Inference: Promptable inference, popularized by the SAM family Kirillov et al. (2023); Ravi et al.

- (2024), enables user or system-provided prompts (such as 2D keypoints or masks) to guide model predictions. Similarly to Wang et al. (2025c), our approach supports various prompt types, including 2D keypoints and masks, and by integrating prompt tokens directly into the transformer architecture, enables user-guided mesh recovery.

Data Quality and Annotation Pipelines: A major bottleneck in HMR is the quality of training data. Many datasets rely on pseudo-ground-truth (pGT) meshes obtained from monocular fitting Kolotouros et al. (2019); Kanazawa et al. (2018), which often contain systematic errors in pose, shape, and camera parameters Patel and Black (2025). Recent work Dwivedi et al. (2024); Wang et al. (2025b) highlights the impact of annotation noise on reported metrics and generalization. To address this, multi-view datasets Martinez et al. (2024); Khirodkar et al. (2024); Moon et al. (2020) and synthetic data have been used in our work to provide higher-fidelity supervision. Our method builds on these insights by employing a scalable data engine that mines challenging cases using vision-language models, and by leveraging a multi-stage annotation pipeline that combines dense keypoint detection, strong parametric priors, and robust optimization.

### 3 SAM 3D Body Model Architecture

Our goal is to recover 3D human meshes (i.e., MHR parameters) accurately, robustly and interactively from a single image. To this end, we design 3DB as a promptable encoder–decoder architecture (see Figure 2) with a rich set of prompt tokens. 3DB is designed to be interactive as it can accept 2D keypoints or masks, allowing users or downstream systems to guide inference.

- 3.1 Image Encoder

The human-cropped image I is normalized and passed through a vision backbone to produce a dense feature map F. An optional set of hand crops Ihand can also be provided to obtain hand crop feature maps Fhand:

F = ImgEncoder(I), (1) Fhand = ImgEncoder(Ihand). (2)

3DB considers two optional prompts: 2D keypoints and segmentation masks. Keypoint prompts are encoded by positional encodings summed with learned embeddings and are provided as additional tokens for the pose decoder. Mask prompts are embedded using convolutions and summed element-wise with the image embedding Kirillov et al. (2023).

- 3.2 Decoder Tokens
- 3DB has two decoders: The body decoder outputs the full-body human rig and an optional hand decoder can provide enhanced hand pose results. The pose decoders take a set of query tokens as input to predict the parameters of MHR and camera parameters. There are four types of query tokens: MHR+camera, 2D keypoint prompt, auxiliary 2D/3D keypoint tokens and optional hand position tokens.

MHR+Camera Token: The initial estimate of MHR and (optionally) camera parameters is embedded as a learnable token for MHR parameter estimation:

Tpose = RigEncoder(Einit) ∈ R1×D, (3) Einit ∈ Rd

. (4)

init

2D Keypoint Prompt Tokens: If 2D keypoint prompts K are provided (e.g., from a user or detector), they are encoded as:

Tprompt = PromptEncoder(K) ∈ RN×D, (5)

K ∈ RN×3, (6) where each keypoint is represented by (x,y,label).

Hand Position Tokens: The hand token, Thand ∈ R2×D, is used in the body decoder to locate the hand positions inside the human images. This set of tokens is optional, without which 3DB can still produce a full-body human rig because the output from body decoder already includes hands.

Auxiliary Keypoint Tokens: To further enhance interactivity and model capacity, we include learnable tokens for all 2D and 3D keypoints.

- Tkeypoint2D ∈ RJ

- 2D×D, (7)

Tkeypoint3D ∈ RJ

- 3D×D. (8)

These tokens allow the model to reason about specific joints and support downstream tasks such as keypoint prediction or uncertainty estimation.

3.3 MHR Decoder

All tokens are concatenated to form the full set of queries: T = [Tpose, Tprompt, Tkeypoint2D, Tkeypoint3D, Thand] (9)

This flexible assembly enables the model to operate in both fully automatic and user-guided modes, adapting to the available prompts. The body decoder attends to both the query tokens T, the full-body image features F,

2D+J3D)×D. (10)

O = Decoder(T,F) ∈ R(3+N+J

Through cross-attention, the body decoder fuses prompt information with visual context, enabling robust and editable mesh recovery. Optionally, the hand decoder can take the same prompt information while attends to the hand crop features Fhand to provide another output token Ohand.

The first output token of O is passed through an MLP to regress the final mesh parameters: θ = MLP(O0) ∈ Rd

out, where θ = {P,S,C,Sk} are the predicted MHR parameters: pose, shape, camera pose and skeleton, respectively. Another set of outputs can be computed from Ohand for a pair of MHR hands, which can be merged to the body output to improve the estimation of the hand.

- 4 Model Training and Inference

Model Training. 3DB is trained with a comprehensive multi-task loss terms, Ltrain = i λiLi, where each Li is a task-specific loss targeting a specific prediction head or anatomical structure. λi are hyper-parameters set empirically. To stabilize training, certain loss terms (e.g., 3D keypoints) are introduced with a warm-up schedule, gradually increasing their weights over the course of training. We also simulate an interactive setup Kirillov et al. (2023); Sofiiuk et al. (2022) for training by randomly sampling prompts in multiple rounds per sample. This multi-task, prompt-aware loss design provides strong supervision across all outputs. We describe the losses in details below.

- 2D/3D Keypoint Loss: We supervise 2D/3D joint locations using an L1 loss, incorporating learnable per-joint uncertainty to modulate the loss based on prediction confidence. For 3D body and hand keypoints, we normalize them with their respective pelvis and wrist locations before computing the loss. Hand keypoints are weighted according to annotation availability. 2D keypoints are supervised in the cropped image spaces, and we upweight the loss for the user-provided keypoint to encourage prompt consistency when keypoint prompts are available.

Parameter Losses: MHR parameters (pose, shape) are supervised with L2 regression losses, and joint limit penalties are imposed to discourage anatomically implausible poses.

Hand Detection Loss: 3DB can localize the hand position by a built-in hand detector. We apply GIoU loss and L1 loss to supervise the hand box regression. We also predict the uncertainty of hand boxes and turn off the hand decoder on hand-occluded samples during inference.

Full-body Inference. During inference, we use the body decoder output by default, with the option to merge hand decoder output when hands are detected. The benefit of the hand decoder comes from the hand-specific data used during training and the flexibility of a free-moving wrist due to the dedicated prediction head. By unifying the hand decoder’s output to the body decoder, our model can provide full-body prediction with improved hand pose estimation. However, we noticed that simply integrating the hand decoder’s output into the middle of the Momentum Human Rig kinematic tree can lead to errors in adjacent joints, particularly at the elbows. Therefore, we leverage the promptability of 3DB to mitigate these errors introduced by the hand decoder’s output. Specifically, we use the wrist location from the hand decoder as well as the elbow location predicted by the body decoder to prompt the body decoder to generate a refined full-body pose estimation result. Finally, the predicted local MHR parameters are merged to a full-body configuration following the kinematic tree of the mesh model. We show qualitative comparisons of our strategy in Figure 9.

- 5 Data Engine for Diversity

Obtaining highly accurate human mesh annotations paired with the images can be computationally costly. Instead, one common strategy is to annotate a large video collection and leveraging temporal constraints to get more reliable pseudo annotations. While it is possible to get a large number of training images from videos, the poses, appearance, imaging conditions, and background might be very similar. In order to increase the diversity of our training dataset, we implemented an automated data engine that selectively routes difficult images for annotation, enabling scalable and efficient dataset curation.

At the core of our data engine is a Vision-Language Model (VLM) driven mining strategy. Rather than relying on simple heuristics or random sampling, we leverage VLMs to automatically generate and update mining rules that identify high-value images for annotation. The VLM identifies images exhibiting challenging scenarios for pose estimation, including occlusion (where the human subject is partially hidden by objects or other people), unusual poses (rare or complex body configurations such as acrobatics or dance), interaction (human-object or human-human activities like holding tools or group actions), extreme scale (subjects appearing at atypical distances from the camera), low visibility (poor lighting, motion blur, or partial visibility), and hand-body coordination (tight coupling of hand and body poses, as in sign language or sports).

Mining rules are automatically updated iteratively based on failure analysis of the current model, allowing the engine to adaptively focus on the most challenging or informative samples. Failure analysis is performed semi-manually, by evaluating 3DB on the current set of annotated images, visualizing the most challenging

[Figure 27]

- Figure 3 Left: GUI of our annotation tool for annotating 2D keypoints. Right: Comparison of the dense (thin) and sparse (thick) keypoints for pseudo annotation.

[Figure 28]

- Figure 4 Example of single-image MHR mesh fitting for ITW datasets. Source: SA-1B Kirillov et al. (2023).

images using keypoint location error, and then manually annotating the image with a few words. These words and images are used to create text prompt for the VLM. New images selected by the VLM are then routed for manual annotation. By focusing annotation efforts on the most informative samples, our data engine enables efficient search through tens of millions of images, while maximizing the value and diversity of each annotated image. By collecting a highly diverse dataset, it provides the basis on which to build a very robust HMR model that works on a wide range of in-the-wild images.

### 6 Data Annotation and Mesh Fitting

In addition to the robustness enabled by the data diversity derived from our data engine, the accuracy of our model depends heavily on the quality of our annotations. To this end, we designed a multi-stage annotation pipeline that produces accurate 3D mesh pseudo-ground truth from both in-the-wild single image datasets and a variety of multi-view datasets, using various combinations of manual 2D keypoint annotation, sparse and dense keypoint detection, geometric constraints, temporal constraints, strong parametric priors, and robust optimization methods.

- 6.1 Manual Annotation

Given a set of images selected by the data engine, we use a current version of 3DB to estimate initial 2D joint positions. Then, a team of trained annotators review and manually correct the estimated joint locations if needed, as shown in Figure 3(a). The annotators also assign a per-joint visibility label according to a strict rubric. Joints with substantial occlusion or other factors that would prevent accurate placement (e.g., 50% occlusion, motion blur) are marked as not visible.

[Figure 29]

[Figure 30]

(a) (b)

- Figure 5 Examples of MHR mesh fitting results. (a) Multi-view mesh fitting. Source: EgoExo4D Grauman et al.

- (2024). (b) Scan-based mesh fitting. Source: Re:Interhand Moon et al. (2023).

6.2 Single-Image Mesh Fitting

For each image, we first obtain the initial estimation of MHR parameters from a current version of 3DB’s predictions, as well as the 595 dense 2D keypoints predicted from a high-capacity keypoint detector. MHR fitting is then performed via gradient-based refinement of the model parameters, minimizing a composite fitting loss Lfit = j λjLj, where each Lj is a task-specific loss including 2D keypoint loss, initialization-anchored regularization and priors. Hyper-parameters λj are set via cross-validation. We apply several loss terms and priors to make the fitting goal: 2D Keypoint Loss is the L2 distance between projected and detected dense 2D keypoints, to ensure minimal 2D reprojection error. Initialization-Anchored Regularization penalizes deviation from the initial prediction by applying L2 losses on both the Momentum Human Rig parameters and their corresponding 3D keypoints, thereby preventing model drift. Pose and Shape Prior enforces anatomical plausibility via a learned Gaussian Mixture prior and L2 regularization. Following the pipeline above, we derive the image to MHR fittings as training supervision as in Figure 4.

Dense keypoint detector. The configuration of 595 dense keypoints is chosen as it represents the minimal manifold of a human body mesh for capturing diverse body shapes and hand poses. The dense keypoint detector adopts a standard Transformer encoder-decoder architecture. However, unlike prior models that only exploit visual cues from pixels Patel and Black (2025); Hewitt et al. (2024); Cuevas-Velasquez et al.

- (2025), our dense keypoint detector leverages additional sparse keypoint guidance obtained from the manual annotation step to predict accurate 2D dense keypoints from in-the-wild images, as illustrated in Figure 3(b). We first train the model on 3D datasets (e.g., Goliath and Synthetic), and use it for multi-stage mesh fitting on the in-the-wild datasets (e.g., COCO, AI Challenger, MPII). We then project the MHR mesh to dense keypoints for a second round of dense keypoint detector training and apply this iterative training scheme twice.

- 6.3 Multi-View Mesh Fitting

Though single-view mesh fitting is effective for a large and diverse set of images, the annotation quality tends to be lower fidelity due to the depth ambiguities and natural occlusion. Therefore, we also exploit multi-view mesh fitting on suitable datasets. For multi-view video datasets, we further extend the pipeline to jointly fit mesh across all frames and camera views, leveraging both spatial and temporal cues. Synchronized 2D keypoints are extracted for each camera and frame, then triangulated to obtain sparse 3D keypoints.

The mesh model is initialized from these triangulated points and camera parameters and refined via second-order optimization-based update of the model parameters, minimizing a composite fitting loss, Lmulti = k λkLk, where each Lk is a task-specific loss including the 2D keypoint loss and the regularization and priors as single-view mesh fitting, together with additional 3D keypoint loss and temporal smoothness: 3D Keypoint Loss is the L2 distance between mesh joints and triangulated 3D keypoints obtained from multi-view geometry, providing strong spatial supervision. Temporal Smoothness Loss encourages estimated pose parameters to temporally

###### Table 1 List of 3DB training datasets. ⋆ denotes the datasets providing samples to train the hand decoder.

Dataset # Images/Frames # Subjects # Views

MPII human pose Andriluka et al. (2014) 5K 5K+ 1 MS COCO Lin et al. (2014) 24K 24K+ 1 3DPW Von Marcard et al. (2018) 17K 7 1 AIChallenger Wu et al. (2019) 172K 172K+ 1 SA-1B Kirillov et al. (2023) 1.65M 1.65M+ 1 Ego-Exo4D Grauman et al. (2024) 1.08M 740 4+ DexYCB Chao et al. (2021) 291K 10 8 EgoHumans Khirodkar et al. (2023) 272K 50+ 15 Harmony4D Khirodkar et al. (2024) 250K 24 20 InterHand Moon et al. (2020)⋆ 1.09M 27 66 Re:Interhand Moon et al. (2023)⋆ 1.50M 10 170 Goliath Martinez et al. (2024)⋆ 966K 120+ 500+ Synthetic⋆ 1.63M – –

smooth, penalizing abrupt changes in motion and promoting realistic temporal dynamics. λk are set via cross-validation. Optimization alternates between updating camera parameters, shape, skeleton, and pose, with robust keypoint filtering (e.g., robust losses, RANSAC, smoothing). Body specific parameters (e.g., shape, skeleton parameters) are optimized jointly across frames. The mesh fitting happens on body full-body data and hand data as shown in Figure 5.

- 7 Training Datasets

We train our model on a mix of single-view, multi-view, and synthetic datasets listed in Table 1, covering general body pose, hands, interactions, and “in-the-wild’ conditions to ensure the quality, quantity and diversity of training data.

Single-view in-the-wild: We utilize datasets that captures people in unconstrained environments with diverse appearance, pose, and scene conditions. For this, we use AIChallenger Wu et al. (2019), MS COCO Lin et al. (2014), MPII Andriluka et al. (2014), 3DPW Von Marcard et al. (2018), and a subset of SA-1B Kirillov et al.

- (2023).

Multi-view consistent: To incorporate geometric consistency for more reliable annotations, we use multi-view data from Ego-Exo4D Grauman et al. (2024), Harmony4D Khirodkar et al. (2024), EgoHumans Khirodkar et al. (2023), InterHand2.6M Moon et al. (2020), DexYCB Chao et al. (2021) and Goliath Martinez et al.

- (2024).

High-fidelity synthetic: We use a photorealistic synthetic extension of the Goliath dataset Martinez et al. (2024). It provides millions of frames with ground-truth MHR parameters across diverse identities, clothing, and contexts. Synthetic data ensures accurate supervision for human mesh recovery, complementing real-world datasets that prioritize diversity over quality.

Hand datasets: These datasets (marked with ⋆ in Table 1), such as Re:Interhand Moon et al. (2023), are used to train both the body and hand decoder. We provide wrist-truncated hand samples to train the hand decoder.

- 8 Evaluation

We follow prior HMR work and report standard pose and shape evaluation metrics: MPJPE Martinez et al. (2017), PA-MPJPE Zhang et al. (2020), PVE Li et al. (2021), and PCK Zhang et al. (2020). To evaluate on SMPL-based datasets, a MHR mesh is mapped to the SMPL mesh format. We present results with two variants of the model; 3DB-H leverages the commonly used ViT-H (632M) backbone, and 3DB-DINOv3 uses the recent DINOv3 (840M) Siméoni et al. (2025) encoder. We resize the input to 512 × 512 for the image encoder and use an off-the-shelf field-of-view (FOV) estimator (e.g., MoGe-2 Wang et al. (2025a)) to provide camera intrinsics for model inference.

- Table 2 Comparison on five common benchmarks. The best results are highlighted in bold, while the second-best results are underlined. Results evaluated using publicly released checkpoint denoted by †. Models trained using RICH denoted by ∗.

3DPW (14) EMDB (24) RICH (24) COCO LSPET Models PA-MPJPE ↓ MPJPE ↓ PVE ↓ PA-MPJPE ↓ MPJPE ↓ PVE ↓ PA-MPJPE ↓ MPJPE ↓ PVE ↓ PCK@0.05 ↑ PCK@0.05 ↑

HMR2.0b Goel et al. (2023) 54.3 81.3 93.1 79.2 118.5 140.6 48.1† 96.0† 110.9† 86.1 53.3 CameraHMR Patel and Black (2025) 35.1 56.0 65.9 43.3 70.3 81.7 34.0 55.7 64.4 80.5† 49.1† PromptHMR Wang et al. (2025c) 36.1 58.7 69.4 41.0 71.7 84.5 37.3 56.6 65.5 79.2† 55.6† SMPLerX-H Cai et al. (2023) 46.6† 76.7† 91.8† 64.5† 92.7† 112.0† 37.4† 62.5† 69.5† – – NLF-L+fit∗ Sárándi and Pons-Moll (2024) 33.6 54.9 63.7 40.9 68.4 80.6 28.7† 51.0† 58.2† 74.9† 54.9†

image

WHAM Shin et al. (2024) 35.9 57.8 68.7 50.4 79.7 94.4 – – – – – TRAM Wang et al. (2024) 35.6 59.3 69.6 45.7 74.4 86.6 – – – – – GENMO Li et al. (2025) 34.6 53.9 65.8 42.5 73.0 84.8 39.1 66.8 75.4 – –

video

3DB-H (Ours) 33.2 54.8 64.1 38.5 62.9 74.3 31.9 55.0 61.7 86.8 68.9 3DB-DINOv3 (Ours) 33.8 54.8 63.6 38.2 61.7 72.5 30.9 53.7 60.3 86.5 67.8

- 8.1 Evaluating Performance on Common Datasets

We first evaluate 3DB on five standard benchmark datasets in Table 2, comparing with a wide variety of state-of-the-art (SoTA) mesh recovery methods. 3DB outperforms all other single-image methods and is even competitive with video-based approaches that additionally leverage temporal information.

In particular, our model achieves superior results in the EMDB and RICH datasets, which are out-of-domain (i.e., not included in the training set), indicating better generalization than previous SoTA methods. 3DB exceeds the second best model, NLF, on all datasets in terms of 3D metrics except for RICH which dataset NLF uses in training while our model does not. 3DB is also state-of-the-art on PCK for 2D evaluation on the COCO and LSPET datasets, demonstrating strong 2D alignment.

- 8.2 Evaluating Performance on New Datasets

Throughout our experiments, we found that mesh recovery models are particularly fragile in out-of-domain settings due to camera, appearance, and pose differences. To understand how methods perform on new, unseen data distributions, we additionally evaluate on five new datasets (38.6K images) in Table 3. The five new datasets include (1) Ego-Exo4D Grauman et al. (2024), (2) Harmony4D Khirodkar et al. (2024), (3) Goliath Martinez et al. (2024), (4) in-house synthetic data and (5) SA1B-Hard. Ego-Exo4D captures humans in diverse, skilled activities, divided into physical (EE4D-Phys) and procedural (EE4D-Proc) domains. Harmony4D focuses on close multi-human interaction in dynamic sports settings. Goliath offers diverse motions in a precise, studio environment. The synthetic dataset consists of single-human images with diverse camera angles and parameters. SA1B-Hard is a subset of 2.6K images extracted from SA1B using our data engine. Together, these five new datasets present a challenging new testbed for mesh recovery methods.

- As it is difficult to compare methods using the exact same training data and methodology due to prohibitive data usage licenses, unclear descriptions of training data, and lack of training code (CameraHMR, PromptHMR, and NLF are trained on 6, 9, and 48 datasets, respectively), we fairly test the generalization ability of 3DB by using a leave-one-out training procedure. This ensures a fair comparison with prior work which have also not seen these datasets. To serve as an in-domain, upper bound comparison, we also show the performance of

- 3DB when trained on the full dataset (i.e., training data is also sampled from these new datasets). For both the baselines and our model, we use ground truth camera intrinsics for model inference for all 3D datasets, except for SA1B-Hard which we used FOV estimated by MoGe-2 Wang et al. (2025a).

We present the results in Table 3. Despite being trained on a large number of datasets, we find that prior work still struggle with these five domains, incurring a significant drop in performance. In contrast, our leave-one-out model shows strong generalization, owing to our more diverse data distribution and stronger training framework. Interestingly, we notice that existing methods constantly trade places for second across different datasets, reflecting strong dataset-specific biases. This indicates that each baseline overfit to a narrow slice of the underlying data distribution.

###### Table 3 Comparison on five new benchmark datasets. The best results are highlighted in bold, while the second-best results are underlined. MPJPE is computed on 24 SMPL keypoints.

EE4D-Phy EE4D-Proc Harmony4D Goliath Synthetic SA1B-Hard Models PVE ↓ MPJPE ↓ PVE ↓ MPJPE ↓ PVE ↓ MPJPE ↓ PVE ↓ MPJPE ↓ PVE ↓ MPJPE ↓ Avg-PCK ↑

CameraHMR Patel and Black (2025) 71.1 58.8 70.3 60.2 84.6 70.8 66.7 54.5 102.8 87.2 63.0 PromptHMR Wang et al. (2025c) 74.6 63.4 72.0 62.6 91.9 78.0 67.2 56.5 92.7 80.7 59.0 NLF Sárándi and Pons-Moll (2024) 75.9 68.5 85.4 77.7 97.3 84.9 66.5 58.0 97.6 86.5 66.5

3DB-H Leave-one-out (Ours) 49.7 44.3 52.9 47.4 63.5 54.0 54.2 46.5 85.6 75.5 73.1 3DB-H Full dataset (Ours) 37.0 31.6 41.9 36.3 41.0 33.9 34.5 28.8 55.2 47.2 76.6

###### Table 4 Comparison on Freihand for hand pose estimation. Methods using Freihand for training are denoted by †.

Method PA-MPVPE ↓ PA-MPJPE ↓ F@5 ↑ F@15 ↑ LookMa Hewitt et al. (2024) 8.1 8.6 0.653 METRO Lin et al. (2021)† 6.3 6.5 0.731 0.984 HaMeR Pavlakos et al. (2024)† 5.7 6.0 0.785 0.990 MaskHand Saleem et al. (2025)† 5.4 5.5 0.801 0.991 WiLoR Potamias et al. (2025)† 5.1 5.5 0.825 0.993

3DB-H (Ours) 6.3 5.5 0.735 0.988 3DB-DINOv3 (Ours) 6.2 5.5 0.737 0.988

- 8.3 Evaluating Hand Pose Estimation Performance

One significant characteristic of 3DB is its strong performance in estimating hand shape and pose. Previous full-body human pose estimation methods Cai et al. (2023); Baradel et al. (2024); Lin et al. (2023) revealed a notable gap in hand pose accuracy compared to hand-only pose estimation methods Pavlakos et al. (2024); Potamias et al. (2025). This performance gap arises from two main factors. First, hand-only methods can leverage large-scale datasets of hand poses, whereas full-body methods cannot utilize these datasets because of the absence of full-body images and annotations. Second, a free-moving wrist allows hand pose models to more easily fit finger poses with 2D and 3D alignment, while for full-body methods, wrist rotation and position are highly constrained by the body’s pose and position. Despite these challenges, 3DB demonstrates strong hand pose accuracy. 3DB benefits from the flexible model training design that incorporates both hand and body data and the hand decoder. Additionally, being promptable, 3DB provides a natural mechanism to align the wrists of the body prediction with those of the hands. We evaluate 3DB’s hand estimation on the representative FreiHand Zimmermann et al. (2019) benchmark in Table 4. For fair comparison against hand-only models, we use the output from our hand decoder for evaluation. Despite not training on the Freihand dataset, which gives a strong in-domain boost, 3DB’s hand pose estimation accuracy is already comparable to SoTA hand pose estimation methods that include Freihand alongside many other hand-centric datasets.

- 8.4 Evaluating 2D Categorical Performance

To better understand the strengths and weaknesses of models on a variety of image types, we compare the performance across our 24 categories defined over SA1B-Hard Kirillov et al. (2023). Our proposed evaluation set is designed to capture a broad spectrum of human appearance and activity in images, ensuring robust evaluation across real-world scenarios. It consists of 24 total categories, which are organized under several high-level groups: Body Shape, Camera View, Hand, Multi-person, Pose and Visibility.

We use the PCK (Percentage of Correct Keypoints) metric for 17 body keypoints and 6 feet keypoints. Results are reported using Avg-PCK, which is PCK averaged over a range of thresholds (i.e. 0.01, 0.025, 0.05, 0.075, 0.1 of the human bounding box size). Results in Table 5 show that 3DB outperforms all baselines on all categories. Qualitative examples are given in Figure 6.

One notable significance is for categories of Visibility - Truncation where the model shows significant advantages than CameraHMR or PromptHMR. Essentially, 3DB has learned a much stronger pose prior when dealing with body truncation in images. Other rows with the large improvements are Pose - Inverted body and Pose Leg or arm splits. We largely attribute these improvements to the increased distribution of hard poses selected by the data engine.

- Table 5 2D categorical performance analysis on the SA-1B Hard dataset. CameraHMR Patel and Black (2025) PromptHMR Wang et al. (2025c) 3DB aPCK(body) aPCK(feet) aPCK(body) aPCK(feet) aPCK(body) aPCK(feet)

Body_shape - In-the-wild 87.64 78.56 85.73 77.87 90.76 92.12 Camera_view - Back or side view 59.69 46.64 61.92 47.74 76.27 66.81 Camera_view - Bottom-up view 55.18 34.84 46.56 29.25 69.62 55.35 Camera_view - Others 51.48 33.80 54.39 38.55 76.62 71.52 Camera_view - Overhead view 55.08 39.46 43.65 24.63 73.33 66.94 Hand - Crossed or overlapped fingers 73.20 62.85 72.48 62.43 81.36 84.04 Hand - Holding objects 76.73 72.11 73.57 68.92 83.40 85.92 Hand - Self-occluded hands 73.22 58.06 72.43 56.19 80.07 80.82 Multi_people - Contact or interaction 63.23 51.65 61.77 47.60 74.81 69.92 Multi_people - Overlapped 53.11 41.88 57.17 41.43 70.82 64.71 Pose - Contortion or bending 47.08 32.78 42.61 20.98 65.20 53.04 Pose - Crossed legs 63.95 32.24 56.15 27.35 76.40 58.80 Pose - Inverted body 46.12 30.01 39.83 24.64 78.18 72.19 Pose - Leg or arm splits 57.51 31.43 54.76 33.11 83.69 72.49 Pose - Lotus pose 63.19 14.38 54.85 12.87 74.53 57.97 Pose - Lying down 51.29 35.88 44.59 26.88 71.35 66.53 Pose - Sitting on or riding 79.66 71.65 70.15 61.16 84.85 81.51 Pose - Sports or athletic activities 78.93 69.34 73.62 60.37 85.10 82.80 Pose - Squatting or crouching or kneeling 62.74 41.47 54.41 33.84 72.85 61.85 Visibility - Occlusion (foot cues) 62.93 26.83 58.00 30.81 75.43 54.74 Visibility - Occlusion (hand cues) 61.01 53.89 58.55 51.13 76.04 72.01 Visibility - Truncation (lower-body truncated) 39.27 - 46.50 - 61.95 Visibility - Truncation (others) 79.18 74.82 77.06 74.99 84.23 86.72 Visibility - Truncation (upper-body truncated) 62.37 54.90 56.01 49.28 64.49 70.99

- 8.5 Evaluating 3D Categorical Performance

Categorical 3D analysis using existing single view datasets is challenging as the underlying pseudo ground truth are low-fidelity approximations of the real geometry. In order to perform a more detailed categorical analysis of HMR methods, we constructed an evaluation dataset using a mix of synthetic and real data from multi-view datasets with high camera counts (more than 100 cameras).

To comprehensively evaluate 3D human mesh reconstruction performance for HMR, we define a set of 28 distinct categories based on interpretable scene and subject attributes, such as occlusion, truncation, viewpoint, pose difficulty, shape, and interaction. Unlike the manual classification used for 2D categories, these 3D categories are automatically generated using rule-based criteria applied to metadata and geometric cues. This systematic approach enables consistent, scalable, and objective analysis of model performance across diverse real-world conditions.

Based on results from Table 6, 3DB demonstrates superior performance in challenging scenarios. Particularly within the very hard pose categories, 3DB consistently outperforms both CameraHMR and PromptHMR in the pose_3d:very_hard category and in pose_2d:very_hard. These results indicate that 3DB possesses inherent strengths in accurately estimating poses under the most challenging conditions.

Additionally, 3DB exhibits a significant advantage in handling the truncation:severe scenario in comparison to CameraHMR and achieves better performance in the viewpoint:topdown_view category in comparison to PromptHMR.

- 8.6 Qualitative Results

In addition to quantitative gains, our model shows clear qualitative improvements over baselines. Figure 6 compares SAM 3D Body to six state-of-the-art methods on the SA1B-Hard dataset, highlighting challenging cases with complex poses, shapes, and occlusions. As shown, SAM 3D Body consistently achieves more accurate body pose and shape recovery, especially for fine details like limbs and hands. The 2D overlays in Figure 6 further illustrate better alignment with input images, demonstrating the robustness of our approach even under difficult conditions. When we focus on hand-crop images where the human body is invisible or truncated out of images, we demonstrate the effectiveness of model as in Figure 7. Here, we only visualize the mesh output by the hand decoder for simplicity and clearness.

- 8.7 Human Preference Study

We conducted a large-scale user preference study to evaluate the perceptual quality of human reconstructions produced by 3DB compared with existing approaches on the SA1B-Hard dataset. While quantitative metrics

###### Table 6 3D categorical performance analysis.

CameraHMR Patel and Black (2025) PromptHMR Wang et al. (2025c) 3DB PVE MPJPE PA-MPJPE PVE MPJPE PA-MPJPE PVE MPJPE PA-MPJPE

aux:depth_ambiguous 126.25 102.25 81.33 109.58 91.77 69.24 64.38 52.72 39.85 aux:orient_ambiguous 84.26 71.77 45.07 83.79 72.93 46.17 42.35 36.64 25.16 aux:scale_ambiguous 118.18 104.77 50.93 112.95 102.28 47.26 58.64 51.16 27.67 fov:medium 82.88 68.81 46.86 76.31 64.84 42.85 43.58 36.97 25.57 fov:narrow 82.15 69.82 49.73 90.41 77.95 53.49 52.14 43.89 36.18 fov:wide 71.55 60.05 38.66 74.98 64.55 42.87 37.97 33.06 22.44 interaction:close_interaction 107.59 90.95 57.62 115.19 98.12 64.87 54.23 44.98 29.76 interaction:mild_interaction 89.98 75.28 52.93 106.55 90.38 62.74 42.63 34.65 27.16

- pose_2d:hard 117.91 107.74 77.16 117.73 110.64 79.16 62.93 57.50 45.58

- pose_2d:very_hard 150.20 140.61 92.66 150.15 145.07 95.40 62.22 56.84 42.39

- pose_3d:hard 133.89 121.11 84.21 129.30 118.59 81.82 71.42 63.68 49.10

- pose_3d:very_hard 213.66 206.34 143.23 186.35 179.46 129.51 114.20 110.62 86.43 pose_prior:average_pose 68.52 56.70 37.22 70.32 59.73 39.42 36.06 30.95 21.35 pose_prior:easy_pose 57.83 47.31 29.92 62.85 53.58 32.80 29.53 24.66 17.20 pose_prior:hard_pose 94.64 80.04 54.53 88.12 76.19 51.15 51.65 44.24 31.09 shape:average_bmi 70.35 58.07 38.08 71.01 60.25 39.90 36.58 31.41 21.31 shape:high_bmi 84.52 69.96 47.55 79.49 67.83 43.04 43.33 36.49 22.45 shape:low_bmi 80.93 65.70 42.71 69.92 58.76 37.30 38.74 32.73 21.82 shape:very_high_bmi 87.18 72.91 47.54 81.17 69.05 44.03 48.51 41.11 24.80 shape:very_low_bmi 108.16 91.25 47.26 94.16 81.12 38.64 51.76 45.69 22.97 truncation:left_body 135.30 113.17 87.98 127.53 110.67 91.33 91.28 76.46 62.23 truncation:lower_body 127.81 97.84 75.82 151.52 118.65 83.79 92.87 67.10 60.77 truncation:right_body 110.28 91.58 71.17 115.71 98.43 72.15 75.04 62.84 50.62 truncation:severe 230.51 213.64 124.01 186.57 168.22 122.70 126.53 113.66 88.42 truncation:upper_body 85.59 79.68 56.36 86.06 80.88 56.94 50.83 48.79 38.39 viewpoint:average_view 75.61 62.69 41.90 74.17 62.80 41.81 41.25 35.22 24.41 viewpoint:bottomup_view 89.83 72.25 53.00 95.46 78.87 55.57 56.50 47.07 34.03 viewpoint:topdown_view 101.69 91.13 59.15 104.29 97.92 63.39 42.84 38.78 27.90

[Figure 31]

- Figure 6 Qualitative comparison of 3DB against state-of-the-art HMR methods. Source: SA-1B Kirillov et al. (2023).

capture geometric and numeric accuracy, they do not always align with the human perception accuracy.

We designed six independent pairwise comparison studies, each comparing 3DB against one baseline method: HMR2.0b Goel et al. (2023), CameraHMR Patel and Black (2025), NLF Sárándi and Pons-Moll (2024), PromptHMR Wang et al. (2025c), SMPLer-X Cai et al. (2023), and SMPLest-X Yin et al. (2025). The study encompassed 7,800 unique participants (1,300 unique per comparison) resulting in over 20,000 total responses. Each participant was presented with a video stimuli. The left and right sides of the video displayed

[Figure 32]

- Figure 7 Qualitative results of hand estimation using the hand decoder of 3DB. Source: Freihand Zimmermann et al.

(2019).

HMR2.0b CameraHMR NLF PromptHMR SMPLer-X SMPLest-X

0

20

40

60

80

100

96.2 (77/80)

95.0 (76/80) 83.8 (67/80)

97.5 (78/80)

98.8 (79/80)

100.0 (80/80)

WinRate(%)

3DB Baseline

- Figure 8 Comparison of 3DB win rate against baselines for human preference study. Win rate (%) and number of wins out of 80.

reconstructions from the two methods, and a video transition effect as used to fade-in the reconstruction result over the image. Participants were instructed to choose which 3D reconstruction better matched the original image by answering: “Which 3D model of the person better matches the original image, left or right?”. We quantify results using win rate and vote share. Win rate is the percentage of stimuli for which 3DB received more votes than the baseline. As summarized in Figure 8, 3DB consistently outperforms all baselines. Focusing on the strongest baseline, NLF, 3DB achieves a win rate of 83.8%.

- 9 Conclusion

We have presented 3DB, a robust HMR model for body and hands. Our approach leverages the Momentum Human Rig parametric body model, employs a flexible encoder–decoder architecture, and supports optional prompts such as 2D keypoints or masks to guide inference. A central advance of our work is in the supervision pipeline. Instead of relying on noisy monocular pseudo-ground-truth, we leverage multi-view capture systems, synthetic sources, and a scalable data engine that actively mines and annotates challenging samples. This strategy yields cleaner and more diverse training signals, supporting generalization beyond curated benchmarks.

- At the same time, 3DB employs a separate hand decoder to enhance the hand pose estimation with hand crops as input which makes it comparable to SoTA hand pose estimation methods.

Acknowledgements

We gratefully acknowledge the following individuals for their contributions and support: Vivian Lee, George Orlin, Nikhila Ravi, Andrew Westbury, Jyun-Ting Song, Zejia Weng, Xizi Zhang, Yuting Ye, Federica Bogo, Ronald Mallet, Ahmed Osman, Rawal Khirodkar, Javier Romero, Carsten Stoll, Shunsuke Saito, Jean-Charles Bazin, Sofien Bouaziz, Yuan Dong, Su Zhaoen, Alexander Richard, Michael Zollhoefer, Roman Radle, Sasha Mitts, Michelle Chan, Yael Yungster, Azita Shokrpour, Helen Klein, Mallika Malhotra, Ida Cheng, Eva Galper.

References

Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, pages 3686–3693, 2014.

Fabien Baradel, Matthieu Armando, Salma Galaaoui, Romain Brégier, Philippe Weinzaepfel, Grégory Rogez, and Thomas Lucas. Multi-hmr: Multi-person whole-body human mesh recovery in a single shot. In European Conference on Computer Vision, pages 202–218. Springer, 2024.

Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Wang Yanjun, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, et al. Smpler-x: Scaling up expressive human pose and shape estimation. Advances in Neural Information Processing Systems, 36:11454–11468, 2023.

Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, et al. Dexycb: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9044–9053, 2021.

Vasileios Choutas, Georgios Pavlakos, Timo Bolkart, Dimitrios Tzionas, and Michael J Black. Monocular expressive body regression through body-driven attention. In European Conference on Computer Vision, pages 20–40. Springer, 2020.

Hanz Cuevas-Velasquez, Anastasios Yiannakidis, Soyong Shin, Giorgio Becherini, Markus Höschle, Joachim Tesch, Taylor Obersat, Tsvetelina Alexiadis, Eni Halilaj, and Michael J Black. Mamma: Markerless & automatic multi-person motion action capture. arXiv preprint arXiv:2506.13040, 2025.

Sai Kumar Dwivedi, Yu Sun, Priyanka Patel, Yao Feng, and Michael J Black. Tokenhmr: Advancing human mesh recovery with a tokenized pose representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1323–1333, 2024.

Aaron Ferguson, Ahmed AA Osman, Berta Bescos, Carsten Stoll, Chris Twigg, Christoph Lassner, David Otte, Eric Vignola, Fabian Prada, Federica Bogo, et al. Mhr: Momentum human rig. arXiv preprint arXiv:2511.15586, 2025.

Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4d: Reconstructing and tracking humans with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14783–14794, 2023.

Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400, 2024.

Charlie Hewitt, Fatemeh Saleh, Sadegh Aliakbarian, Lohit Petikam, Shideh Rezaeifar, Louis Florentin, Zafiirah Hosenie, Thomas J Cashman, Julien Valentin, Darren Cosker, and Tadas Baltrušaitis. Look ma, no markers: holistic performance capture without the hassle. ACM Transactions on Graphics (TOG), 43(6), 2024.

Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence, 36(7):1325–1339, 2013.

Hanbyul Joo, Tomas Simon, Xulong Li, Hao Liu, Lei Tan, Lin Gui, Sean Banerjee, Timothy Scott Godisart, Bart Nabbe, Iain Matthews, et al. Panoptic studio: A massively multiview system for social interaction. IEEE Transactions on Pattern Analysis and Machine Intelligence, 16, 2017.

Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7122–7131, 2018.

Rawal Khirodkar, Aayush Bansal, Lingni Ma, Richard Newcombe, Minh Vo, and Kris Kitani. Ego-humans: An ego-centric 3d multi-human benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19807–19819, 2023.

Rawal Khirodkar, Jyun-Ting Song, Jinkun Cao, Zhengyi Luo, and Kris Kitani. Harmony4d: A video dataset for in-the-wild close human interactions. Advances in Neural Information Processing Systems, 37:107270–107285, 2024.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. Learning to reconstruct 3d human pose and shape via model-fitting in the loop. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2252–2261, 2019.

Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, and Cewu Lu. Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3383–3393, 2021.

Jiefeng Li, Jinkun Cao, Haotian Zhang, Davis Rempe, Jan Kautz, Umar Iqbal, and Ye Yuan. Genmo: A generalist model for human motion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11766–11776, 2025.

Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6):194–1, 2017.

Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan. Cliff: Carrying location information in full frames into human pose and shape estimation. In European Conference on Computer Vision, pages 590–606. Springer, 2022.

Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, and Yu Li. One-stage 3d whole-body mesh recovery with component aware transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21159–21168, 2023.

Kevin Lin, Lijuan Wang, and Zicheng Liu. End-to-end human pose and mesh reconstruction with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1954–1963, 2021.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Jingyuan Liu, Li-Yi Wei, Ariel Shamir, and Takeo Igarashi. ipose: Interactive human pose reconstruction from video. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pages 1–14, 2024.

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, October 2015.

Julieta Martinez, Rayat Hossain, Javier Romero, and James J Little. A simple yet effective baseline for 3d human pose estimation. In Proceedings of the IEEE international conference on computer vision, pages 2640–2649, 2017.

Julieta Martinez, Emily Kim, Javier Romero, Timur Bagautdinov, Shunsuke Saito, Shoou-I Yu, Stuart Anderson, Michael Zollhöfer, Te-Li Wang, Shaojie Bai, et al. Codec avatar studio: Paired human captures for complete, driveable, and generalizable avatars. Advances in Neural Information Processing Systems, 37:83008–83023, 2024.

Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori, and Kyoung Mu Lee. Interhand2. 6m: A dataset and baseline for 3d interacting hand pose estimation from a single rgb image. In European Conference on Computer Vision, pages 548–564. Springer, 2020.

Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe De Bree, et al. A dataset of relighted 3d interacting hands. Advances in Neural Information Processing Systems, 36:17689–17701, 2023.

Jinhyung Park, Javier Romero, Shunsuke Saito, Fabian Prada, Takaaki Shiratori, Yichen Xu, Federica Bogo, Shoou-I Yu, Kris Kitani, and Rawal Khirodkar. Atlas: Decoupling skeletal and shape parameters for expressive parametric human modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6508–6518, 2025.

Austin Patel, Andrew Wang, Ilija Radosavovic, and Jitendra Malik. Learning to imitate object interactions from internet videos. arXiv preprint arXiv:2211.13225, 2022.

Priyanka Patel and Michael J Black. Camerahmr: Aligning people with perspective. In 2025 International Conference on 3D Vision (3DV), pages 1562–1571. IEEE, 2025.

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019.

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3d with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9826–9836, 2024.

Owen Pearl, Soyong Shin, Ashwin Godura, Sarah Bergbreiter, and Eni Halilaj. Fusion of video and inertial sensing data via dynamic optimization of a biomechanical model. Journal of biomechanics, 155:111617, 2023.

Xue Bin Peng, Angjoo Kanazawa, Jitendra Malik, Pieter Abbeel, and Sergey Levine. Sfv: Reinforcement learning of physical skills from videos. ACM Transactions On Graphics (TOG), 37(6):1–14, 2018.

Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. Wilor: End-to-end 3d hand localization and reconstruction in-the-wild. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12242–12254, 2025.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. arXiv preprint arXiv:2201.02610, 2022.

Yu Rong, Takaaki Shiratori, and Hanbyul Joo. Frankmocap: A monocular 3d whole-body pose estimation system via regression and integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1749–1759, 2021.

Muhammad Usama Saleem, Ekkasit Pinyoanuntapong, Mayur Jagdishbhai Patel, Hongfei Xue, Ahmed Helmy, Srijan Das, and Pu Wang. Maskhand: Generative masked modeling for robust hand mesh reconstruction in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8372–8383, 2025.

István Sárándi and Gerard Pons-Moll. Neural localizer fields for continuous 3d human pose and shape estimation. Advances in Neural Information Processing Systems, 37:140032–140065, 2024.

Soyong Shin, Juyong Kim, Eni Halilaj, and Michael J Black. Wham: Reconstructing world-grounded humans with accurate 3d motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2070–2080, 2024.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Konstantin Sofiiuk, Ilya A Petrov, and Anton Konushin. Reviving iterative training with mask guidance for interactive segmentation. In 2022 IEEE international conference on image processing (ICIP), pages 3141–3145. IEEE, 2022.

Vasileios Vasilopoulos, Georgios Pavlakos, Sean L Bowman, J Diego Caporale, Kostas Daniilidis, George J Pappas, and Daniel E Koditschek. Reactive semantic planning in unexplored semantic environments using deep perceptual feedback. IEEE Robotics and Automation Letters, 5(3):4455–4462, 2020.

Timo Von Marcard, Roberto Henschel, Michael J Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3d human pose in the wild using imus and a moving camera. In Proceedings of the European conference on computer vision (ECCV), pages 601–617, 2018.

Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546, 2025a.

Shengze Wang, Jiefeng Li, Tianye Li, Ye Yuan, Henry Fuchs, Koki Nagano, Shalini De Mello, and Michael Stengel. Blade: Single-view body mesh estimation through accurate depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21991–22000, 2025b.

Yufu Wang, Ziyun Wang, Lingjie Liu, and Kostas Daniilidis. Tram: Global trajectory and motion of 3d humans from in-the-wild videos. In European Conference on Computer Vision, pages 467–487. Springer, 2024.

Yufu Wang, Yu Sun, Priyanka Patel, Kostas Daniilidis, Michael J Black, and Muhammed Kocabas. Prompthmr: Promptable human mesh recovery. In Proceedings of the computer vision and pattern recognition conference, pages 1148–1159, 2025c.

Jiahong Wu, He Zheng, Bo Zhao, Yixin Li, Baoming Yan, Rui Liang, Wenjia Wang, Shipei Zhou, Guosen Lin, Yanwei Fu, et al. Large-scale datasets for going deeper in image understanding. In International Conference on Multimedia and Expo (ICME), pages 1480–1485. IEEE, 2019.

Jie Yang, Ailing Zeng, Feng Li, Shilong Liu, Ruimao Zhang, and Lei Zhang. Neural interactive keypoint detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15122–15132, 2023.

Wanqi Yin, Zhongang Cai, Ruisi Wang, Ailing Zeng, Chen Wei, Qingping Sun, Haiyi Mei, Yanjun Wang, Hui En Pang, Mingyuan Zhang, et al. Smplest-x: Ultimate scaling for expressive human pose and shape estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Yifei Yin, Chen Guo, Manuel Kaufmann, Juan Jose Zarate, Jie Song, and Otmar Hilliges. Hi4d: 4d instance segmentation of close human interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17016–17027, 2023.

Jianfeng Zhang, Xuecheng Nie, and Jiashi Feng. Inference stage optimization for cross-scenario 3d human pose estimation. Advances in neural information processing systems (NeurIPS), 33:2408–2419, 2020.

Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, and Thomas Brox. Freihand: A dataset for markerless capture of hand pose and shape from single rgb images. In Proceedings of the IEEE/CVF international conference on computer vision, pages 813–822, 2019.

## Appendix

- A Author Contributions

SAM 3D Body represents a joint effort by the entire team – all members contributed to paper writing and project release, with each author contributing to the following areas:

Model: Xitong Yang (model lead); Jinkun Cao (hand pose, model improvement); Jinhyung Park (MHR integration, model improvement); Nicolas Ugrinovic (multi-person interaction); Jiawei Liu (SAM 3D unification).

Data: Devansh Kukreja (data engine and infrastructure); Don Pinkus (manual annotation tooling); Taosha Fan (multi-view mesh fitting); Soyong Shin (single-view mesh fitting, dense keypoint detector); Jinhyun Park (MHR mesh fitting); Jinkun Cao (hand and whole-body data).

Evaluation: Xitong Yang (internal/external benchmarks); Jinkun Cao (hand pose evaluation); Jiawei Liu (human preference study, visualization); Nicolas Ugrinovic (mutli-person evaluation).

Leadership and XFN: Kris Kitani, Anushka Sagar, Piotr Dollar, Matt Feiszli, Jitendra Malik.

- B Evaluating 3DB Prompt Following

3DB is a promptable model that supports conditioning on 2D keypoints or segmentation masks for controllable human pose estimation. In this section, we evaluate the model’s ability to follow the provided prompts and analyze their impact on pose estimation performance.

2DKeypointPrompt. The 2D keypoint prompt provides a user-friendly mechanism for adjusting pose estimation by specifying joint locations in the image Liu et al. (2024); Yang et al. (2023). In Table 7, we present an ablation study on varying the number of keypoint prompts provided during inference, where the keypoint with the largest error is selected for prompting. We observe that the model effectively follows the prompts and both 2D and 3D performance improve as more prompts are provided (noise scale = 0). Notably, although the keypoint prompt is provided in the 2D image space, 3DB is able to leverage this information to infer a more accurate 3D body pose.

We further evaluate the sensitivity of our model to the quality of keypoint prompts, as shown in Table 7 for #Prompt = 1. The noise scale is defined relative to the bounding box size, as in PCK. We observe that the model is robust to small keypoint inaccuracy (noise scale < 0.05), as such noise naturally exists in annotations of in-the-wild datasets. When the noise level becomes larger, performance degrades because the model tends to follow the incorrect keypoint prompts.

Finally, our full-body inference pipeline leverages the 2D keypoint prompting capability to improve hand pose estimation quality, as described in Section 4. To illustrate the impact of this strategy, we provide a qualitative comparison in Figure 9. From this comparison, it is evident that without keypoint prompting, 2D keypoint alignment at the wrist and hand joints is significantly worse than that achieved by the default inference. On the other hand, without integrating the hand decoder during inference, the predicted wrist rotation is often suboptimal, leading to inferior 2D finger joint alignment.

Maks Prompt. The capability of mask conditioning is essential when handling multiple people with close interaction, where the standard bounding box information is insufficient to clearly specify the person of interest for the model Yin et al. (2023); Wang et al. (2025c). To assess the impact of incorporating masks as additional input to our model, we compare the model inference result with and without mask-conditioning on three multi-person (MP) datasets. We follow the prior work to provide ground-truth segmentation masks when available Yin et al. (2023); Wang et al. (2025c), and extract SAM2 Ravi et al. (2024) masks for the ITW dataset SA1B using the bounding boxes. As shown in Table 8, conditioning our model with person-specific segmentation masks yields significant improvements – the same model (3DB-DINOv3) with mask conditioning improves PVE by 33.1 and MPJPE by 29.4 on Hi4D. Notably, both Hi4D Yin et al. (2023) and Harmony4D Khirodkar et al. (2024) are multi-person dataset that captures close interactions between two individuals, featuring frames with significant occlusion, which poses a challenge for most HMR

###### Table 7 Ablation on 2D keypoint prompting with 3DB-H. We report results under varying numbers of prompts, as well as different noise scales for a single prompt.

# Prompts 0 1 2 Noise scale 0 0 0.01 0.03 0.05 0.1 0

COCO (PCK@0.05↑) 86.7 90.2 90.2 89.5 87.6 80.9 93.0 EMDB (MPJPE ↓) 63.3 60.1 60.3 61.5 63.3 67.8 58.9

###### Table 8 Comparison on mask-conditioned inference with 3DB-DINOv3 on multi-person datasets.

Hi4D Yin et al. (2023) Harmony4D SA1B-Hard SA1B-MP Models PVE ↓ MPJPE ↓ PVE ↓ MPJPE ↓ Avg-PCK ↑ Avg-PCK ↑

3DB (w/o mask) 91.4 76.4 42.7 35.6 75.4 67.9 3DB (w/ mask) 58.3 47.0 36.5 30.1 76.3 72.3

methods, especially in disambiguating between individuals. Using segmentation masks for each person as additional input, 3DB effectively addresses this challenge and accurately predicts the corresponding person. For the experiments on our SA1B-Hard dataset, we observe that the performance gain on the "Multi-person" subset (+4.4%) is more significant than that on the overall dataset (+0.9%), indicating the importance of mask-conditioning for multi-person scenarios.

- C Limitations

We discuss some limitations of 3DB as presented in this paper and suggest possible next steps to address these limitations. First, 3DB processes each individual separately, without taking multi-person or humanobject interactions into account. This limits its ability to accurately interpret relative positions and physical interactions. A natural next step would be to incorporate interactions among humans, objects, and the environment into the model’s training process. Second, while our model has achieved significant improvements in hand pose estimation as part of the full-body estimation task, its accuracy does not surpass that of specialized hand-only pose estimation methods. Additionally, due to the limited availability of high-quality full-body data during training, the hand estimation performance from the body decoder alone is also suboptimal. This limitation can be addressed by incorporating more diverse full-body data into the training of 3DB. Third, both 3DB and the underlying mesh model MHR fall short in modeling human body shapes across all age groups. As a result, they may produce suboptimal pose estimations and shape modeling for children.

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

Image Crop W/o keypoint prompt W/o decoder unifying Default Inference

###### Figure 9 Qualitative comparison to show the impact from using keypoint prompting and unifying the predictions from hand decoder and body decoder.

