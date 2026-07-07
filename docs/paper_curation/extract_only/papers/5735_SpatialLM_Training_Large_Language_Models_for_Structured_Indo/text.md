# arXiv:2506.07491v2[cs.CV]5Nov2025

## SPATIALLM: Training Large Language Models for Structured Indoor Modeling

Yongsen Mao1∗ Junhao Zhong1∗ Chuan Fang2 Jia Zheng1 Rui Tang1 Hao Zhu1 Ping Tan2 Zihan Zhou1 1Manycore Tech Inc. 2Hong Kong University of Science and Technology https://manycore-research.github.io/SpatialLM

### Abstract

SPATIALLM is a large language model designed to process 3D point cloud data and generate structured 3D scene understanding outputs. These outputs include architectural elements like walls, doors, windows, and oriented object boxes with their semantic categories. Unlike previous methods which exploit task-specific network designs, our model adheres to the standard multimodal LLM architecture and is fine-tuned directly from open-source LLMs.

To train SPATIALLM, we collect a large-scale, high-quality synthetic dataset consisting of the point clouds of 12,328 indoor scenes (54,778 rooms) with groundtruth 3D annotations, and conduct a careful study on various modeling and training decisions. On public benchmarks, our model gives state-of-the-art performance in layout estimation and competitive results in 3D object detection. With that, we show a feasible path for enhancing the spatial understanding capabilities of modern LLMs for applications in augmented reality, embodied robotics, and more.

### 1 Introduction

3D indoor environments are ubiquitous in our daily lives. We spend a lot of time and perform various activities in such environments every day. Therefore, a long-standing goal of artificial intelligence is to teach machines to perceive, reason about, and interact with 3D indoor scenes as humans do. In this work, we focus on the task of structured indoor modeling, which aims to extract structured indoor scene descriptions from raw sensorial inputs (i.e., RGBD scans). Specifically, the scene description includes both the architectural layout (i.e., walls, doors, and windows) and 3D object bounding boxes in the indoor environment. Such 3D structure information has been shown to benefit a wide range of real-world applications such as scene editing [40], augmented reality [4, 2], and robot navigation [17].

Compared to 3D formats such as meshes, voxels, or implicit functions, the structured scene description provides a highly compact yet flexible scene representation. In this paper, our first contribution is to regard the structured descriptions as scripts of a general-purpose language (i.e., Python) and propose to predict the language in text form, as illustrated in Figure 1. This design choice has several advantages: (i) it is human interpretable and editable; (ii) it can be easily extended to incorporate any new classes without affecting existing content in the scripts; and (iii) it allows us to leverage the strong built-in coding capability of pre-trained large language models.

Therefore, our goal is to implement our method by directly fine-tuning open-source LLMs. While modern (multimodal) LLMs have revolutionized fields like natural language processing, 2D image understanding and generation, few work has attempted to use LLMs for 3D structured scene modeling. This is partly due to the lack of large-scale, high-quality datasets for the task. In this paper, as our second contribution, we collect a new synthetic dataset, which consists of the point clouds of 12,328

∗Equal contribution.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

[Figure 1]

[Figure 2]

Projector(MLP)

PointCloud

[Figure 3]

Encoder

LLM

Detect walls, doors, windows, bboxes . . .

[Figure 4]

Figure 1: The overall pipeline of SPATIALLM. Given the point cloud input, it employs a standard “Encoder-MLP-LLM” architecture for multimodal feature alignment (left), and generates structured scene descriptions in pure text form as output (middle). The reconstructed 3D structure is further overlaid on the point cloud for visualization (right).

distinct scenes (54,778 rooms) paired with 3D structure information, and conduct the first empirical study on the best strategy of aligning the point cloud input with LLMs for structured scene modeling.

Building upon our findings, we introduce SPATIALLM, a large language model which is capable of processing point cloud input and generating structured scene descriptions. As our third contribution, we show that, by first training on our large-scale dataset and then on smaller downstream task data, our model gives competitive performance on public benchmarks for layout estimation and 3D object detection, respectively. Additionally, we provide proof-of-concept zero-shot experiments which show that our model can handle point clouds from diverse sources such as monocular video sequences. These results suggest that SPATIALLM may serve as a base for developing future solutions for scenarios where enhanced spatial understanding and reasoning is required (e.g., embodied AI).

### 2 Related Work

- 2.1 Structured Indoor Modeling

Structured indoor modeling aims to recover structural elements (i.e., walls, doors, and windows) as well as 3D objects from unstructured point clouds. In the literature, some studies focus on room layout estimation from point clouds. Early methods [37, 38] rely on traditional geometric analysis algorithms (i.e., RANSAC) to detect wall planes. SceneCAD [1] jointly optimizes the layout estimation and CAD model alignment of the scanned scene. RoomFormer [60] proposes to train a Transformer-based network to predict corners and rooms. Meanwhile, a number of recent methods [35, 46, 51, 47, 25] have advanced the state-of-the-art on 3D object detection from point clouds.

The closest work to ours is SceneScript [2], which also casts structured indoor reconstruction as a sequence modeling problem. It demonstrates great flexibility as the language can be tailored to different scenarios without changing the method. But unlike our method, SceneScript uses domainspecific tokens for language commands, thus requiring a specialized decoder. Instead, SPATIALLM strictly adheres to the “Encoder-MLP-LLM” architecture and is directly fine-tuned based on modern foundation models.

- 2.2 Large Language Models for 3D Scene Understanding Recently, there has been a growing interest in empowering LLMs to understand and reason about
- 3D spaces, as defined in public benchmarks such as ScanRefer [9] and ScanQA [3]. One line of work [53, 20, 19, 29, 22] first uses 3D detectors to locate objects in the point cloud, then extracts features of individual objects using a pre-trained object-level encoder. In this paper, we study how such 3D detectors can be learned with modern LLMs.

Other methods directly extract scene-level features from the point cloud and align the features with LLMs [18, 66, 10, 16]. 3D-LLM [18] uses pre-trained image encoder to obtain 2D features from multi-view images and back-projects onto the point cloud [21]. It then leverages a Q-former [28] to map them to a fixed number of tokens as input to the LLM. LL3DA [10] adopts a 3D visual encoder

@dataclass class Wall:

- a_x: int
- a_y: int
- a_z: int

- b_x: int
- b_y: int
- b_z: int height: int

@dataclass class Door:

wall_id: str

- position_x: int
- position_y: int
- position_z: int width: int height: int

@dataclass class Window:

wall_id: str

- position_x: int
- position_y: int
- position_z: int width: int height: int

@dataclass class Bbox:

class: str

- position_x: int
- position_y: int
- position_z: int angle_z: int

- scale_x: int
- scale_y: int
- scale_z: int

Figure 2: Definition of our structured representation for layouts and objects.

pre-trained on ScanNet detection as the scene encoder. ScanReason [66] further employs a 3D point cloud encoder to obtain fine-grained geometric information for accurate object grounding. SceneLLM [16] tokenizes the 3D point cloud features by simply dividing the space into a fixed-resolution voxel grid and aggregating features in the same voxel.

Besides point cloud-based representations, recent studies have explored other ways to integrate

- 3D information in LLMs. SpatialVLM [8] primarily focuses on synthesizing large-scale 3D VQA data to enhance the spatial reasoning capabilities of the VLM. LLaVA-3D [67] injects 3D position embeddings into 2D CLIP features to establish a unified architecture for 2D image understanding and 3D scene understanding. SpatialRGPT [11] extracts a 3D scene graph from a single image to enhance grounded spatial reasoning in LLMs. Video-3D LLM [63] learns representations from video frames and 3D coordinates to leverage the strong priors of pre-trained 2D Video LLMs. LSceneLLM [65] uses an adaptive self-attention module to capture fine-grained details in local areas. Our work differs from these studies in that we focus on analyzing the structural information of the scene.

### 3 SPATIALLM

Given the point cloud P (typically derived from RGBD scans), our goal is to recover the 3D layout L, which consists of all walls, doors and windows, and a set of objects O in the scene. As discussed before, we represent L and O as scripts of a general-purpose language, and fine-tune LLMs to predict the language in text form. Figure 2 shows all the parameters we use for the layout and objects.

To the best of our knowledge, we are the first to employ LLMs for structured scene reconstruction from point clouds. Thus, we first propose a new dataset suitable for training the LLMs on (Section 3.1). Then, with the new dataset, we set out to investigate the following two key questions: (i) What point cloud features to use for the structured indoor modeling task (Section 3.2)? (ii) How to integrate the features into LLMs (Section 3.3)?

- 3.1 Dataset and Metrics Table 1: Quantitative statistics of indoor RGBD scan datasets.

Table 1 provides a list of popular indoor RGBD scan datasets. We can make a few observations:

annotations layouts objects

Dataset (year) source #scenes

ScanNet (2017) [13] real 1,513 • Matterport3D (2017) [7] real 90 • 3RScan (2019) [50] real 1,482 • SceneCAD (2020) [1] real 1,151 • • ARKitScenes (2021) [4] real 5,047 • MultiScan (2022) [34] real 273 • ScanNet++ (2023) [59] real 1,006 • HM3DSem (2023) [58] real 181 •

First, the number of scenes in real datasets is relatively small, ranging from about one hundred to a few thousands. To collect such a dataset, one needs to take RGBD scans in real scenes and manually annotate 3D information, which is a highly laborious process. Further, most real datasets provide annotations on the objects (i.e., semantic labels and 3D boxes) only. One exception is SceneCAD [1], which manually adds wall annotations to ScanNet dataset. But it only contains 1,151 rooms.

Structured3D (2020) [64] syn. 3,500 • Hypersim (2021) [45] syn. 461 • ProcTHOR (2022) [14] syn. 12,000 • • HSSD (2024) [14] syn. 211 • • ASE (2024) [2] syn. 100,000 •

SPATIALLM dataset (ours) syn. 12,328 • •

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

ProcTHOR [14] HSSD [24] ASE [2] SPATIALLM dataset

- Figure 3: Dataset visual quality comparison. The layout and object placements in ProcTHOR [14] and ASE [2] are program-generated, which exhibit noticeable differences from real-world statistics. The scenes in HSSD [24] and our dataset are fully human-authored. But HSSD only has 211 scenes.

Second, for synthetic datasets, it is much cheaper to obtain ground-truth 3D annotations. But to ensure the realism of the RGBD scans, the 3D scenes must be professionally constructed. As a result, the diversity of 3D scenes in some datasets is still limited. For example, Hypersim [45] and HSSD [24] only have 461 and 211 scenes, respectively. On the other hand, one may create a large number of scenes by employing a rule-based procedural modeling pipeline, as did in ProcTHOR [14] and ASE [2]. But the quality of the scenes is not as high.

SPATIALLM dataset. In this paper, we make an effort to build a dataset suitable for training LLMs for structured indoor scene reconstruction. To this end, we take advantage of access to a large repository of interior designs from an online platform2 in the interior design industry. Most designs are created by professional designers and used for real-world production. We parse each 3D house into individual rooms and use a few rules to filter the rooms. This results in 12,328 distinct scenes with 54,778 rooms.

For the objects, we keep annotations on a selected set of 59 common categories (excluding wall, door, and window), and filter out small objects with side lengths all < 15cm. In the end, our dataset contains 412,932 annotated object instances of 35,426 unique CAD models.

For each scene, we generate photo-realistic RGBD images utilizing an industry-leading rendering engine. At the time of rendering, a camera trajectory traversing each room is simulated, based on which the images are taken at an interval of 0.5m. Figure 3 compares the visual quality of our dataset to several recent synthetic datasets. Finally, the dataset is divided into 11,328/500/500 scenes for training/validation/testing.

Evaluation metrics. We evaluate our method on both layout estimation and 3D object detection. For layout estimation, let Lˆ = {ˆli}miˆ=1 and L = {li}mi=1 denote the set of predicted structural elements and ground truth, respectively. Since each element is a plane segment in 3D space, we compute the 2D intersection-over-union (IoU2D) by projecting each predicted element to the plane of the ground-truth element to find the optimal assignment with the Hungarian algorithm, and report the F1-scores at 0.25 and 0.5 thresholds. For 3D object detection, we match the predictions Oˆ = {oˆi}niˆ=1 with the ground truth O = {oi}ni=1 via Hungarian matching. Then, we compute the 3D intersection-over-union (IoU3D) between each matched pair and report the F1-scores at 0.25 and 0.5 thresholds.

#### 3.2 Point Cloud Encoders

Extracting meaningful representations for a point cloud is challenging because of its irregular format. Learning-based approaches to process 3D point clouds can be roughly classified into three groups: Mapping-based methods are inspired by the success of 2D image backbones such as CLIP [44] and the DINO series [6, 39], and map pixel-aligned image features to 3D. For example, Lexicon3D [33]

2https://www.kujiale.com/

Table 2: Experiment on the encoder design.

Params F1 - layout F1 - object

Encoder

(M) IoU2D@0.25 IoU2D@0.5 IoU3D@0.25 IoU3D@0.5

Voxelize (P+DINOv2) 0.0 10.8 5.5 0.1 0.0 Rand. sampling (P+DINOv2) 0.0 10.6 4.9 0.2 0.0

3DCNN enc. (P) 10.5 79.4 76.3 59.8 46.1 3DCNN enc. (P+DINOv2) 11.6 81.5 79.4 57.1 45.0 3DCNN enc. (P) + Voxelize (DINOv2) 11.9 83.8 81.6 62.9 46.7

Sonata/PTv3 enc. (P) 108.8 84.6 82.6 65.1 49.4

projects 3D points onto the image plane to obtain their corresponding pixel features; ConceptFusion [21] builds multimodal 3D maps via traditional SLAM and multi-view fusion approaches. Voxel-based methods first transform irregular point clouds to regular representations via 3D voxelization. Then, sparse convolution algorithms [12] can be readily used for efficient computation. Point-based methods, by contrast, process point clouds directly as sets embedded in continuous space. Recent approaches in this category [62, 54–56] have successfully explored the self-attention mechanism in Transformers and demonstrated superior performance on downstream applications.

Note that all methods discussed above produce one feature per point by default. Given that a scenelevel point cloud consists of hundreds of thousands of points or more, these features are too much in terms of quantity for LLMs to consume directly. Since most modern open-source LLMs take an “Encoder-MLP-LLM” architecture for multimodal feature alignment (e.g., the Llama series [32] and Qwen series [42, 43]), we can define a point cloud encoder for LLMs as a mapping which takes a set of N points as input and generates K feature embeddings, where K ≪ N:

F = E(P), P ∈ RN×6,F ∈ RK×D. (1) Here, each input point is a 6-dimensional vector (XYZ and RGB), and D is the feature dimension.

Recently, a few attempts have been made to convert point representations into feature embeddings [2, 16]. However, this step has not been carefully investigated. In this section, we conduct experiments to study (i) the effectiveness of different encoders, and (ii) the impact of K, i.e., the number of output features. Unless otherwise stated, Qwen2.5-0.5B [42] is employed as our base model. We use Sonata [56] as the point cloud encoder, and a two-layer MLP as the projector.

Experiment on the encoder design. We consider six possible point cloud encoders. The first group of encoders adopts a mapping-based method to fuse 2D image features to 3D points. Following Lexicon3D [33], we pool pixel-level DINOv2 features to form 3D point representations. Then, one of the two routes may be taken to reduce the points into a manageable length of tokens:

- • Voxelize (P+DINOv2): divides the space into a fixed-resolution voxel grid and aggregates all the points in the same voxel – similar to the approach used in Scene-LLM [16];
- • Rand. sampling (P+DINOv2): uses Farthest Point Sampling [41] to randomly sample a fixed number of points with their features.

From Table 2 we can see, these two options result in extremely low F1-scores. We hypothesize that this is because too much spatial information is lost during the naive down-sampling step. While this may be acceptable for certain applications, such as visual Q&A, it is problematic for our task, which relies heavily on the spatial information to reconstruct the geometric coordinates of objects.

The second group of encoders adopts a standard 3DCNN encoder to generate pooled features. We follow SceneScript [2] to build our encoder with a sparse 3D convolution library [48]. Here we consider three variants:

- • 3DCNN enc. (P): takes raw points (6-dim XYZRGB values) as input;
- • 3DCNN enc. (P+DINOv2): takes the points with the associated DINOv2 features as input;
- • 3DCNN enc. (P) + Voxelize (DINOv2): concatenates the output of 3DCNN with aggregated DINOv2 features.

As shown in Table 2, a learned encoder is effective in keeping the necessary semantic and geometric information for reconstruction. We also find that incorporating DINOv2 features yields slightly better results, possibly due to the enhanced contextual information.

The last encoder we consider is Sonata [56], a variant of Point Transformer V3 (PTv3) [55] which removes the decoder and focuses on self-supervised learning on the encoder. Compared to other backbones, such an “encoder-only” design makes it quite convenient for adapting to LLMs. Indeed, as shown in Table 2, with raw points as input, it already achieves the best results among all encoders.

Experiment on the spatial resolution. Recall in Eq. (1) that the parameter K denotes the number of feature embeddings or visual tokens, which is critical to the performance of LLMs. In this experiment, we study the effect of K.

Table 3: Experiment on the spatial resolution. Avg. (K) indicates the average value of K under the corresponding resolution.

F1 - layout F1 - object IoU2D@0.25 IoU2D@0.5 IoU3D@0.25 IoU3D@0.5

Res. Avg. (K)

- 1× 123 84.6 82.6 65.1 49.4 1.25× 197 84.8 82.6 65.3 52.2

- 1.5× 286 86.9 85.1 68.7 58.6

2× 510 87.4 85.3 71.6 61.1

- 2.5× 788 86.1 83.9 72.2 61.7 4× 2,000 85.8 83.7 69.9 58.9

However, fixing K is difficult because different point clouds generally produce varying K values. Instead, since the encoders employ a coarse-to-fine structure for point cloud processing, we set the resolution at the finest level at 5cm and use a 5-level hierarchical structure by default. A detailed illustration of the point cloud encoder hierarchy is presented in the appendix (see Figure 11). Then, we gradually increase the resolution up to 4×, and report the results in Table 3. As can be seen, model performance steadily improves from 1× to 2× resolutions, but further increasing the resolution hurts the performance. This is likely due to the excessively long token sequences.

#### 3.3 Training Schedules

As most recent MLLMs start from pre-trained unimodal backbones, how to align the multimodal content with the language model plays a critical role in the performance of the resulting MLLMs. For example, a common practice adopted by many MLLMs [30, 49] is the inclusion of a two-stage training schedule: (i) an alignment stage that trains the randomly initialized MLP projector only, followed by (ii) a fine-tuning stage that trains both the projector and language model while keeping the vision encoder frozen. Other studies (e.g., [23]) advocate a single-stage training protocol, suggesting that including the projector training stage is unnecessary. Furthermore, when adapting the LLMs to rich visual content such as long videos, some research [68] has shown that a three-stage pipeline, which progressively unfreezes different components in different stages, improves the performance.

Table 4: Experiment on training schedule. For each training stage, we use three circles to represent the three components in the “Encoder-MLP-LLM” architecture. “◦” and “•” indicate whether a component is frozen or trainable, respectively.

Training stages F1 - layout F1 - object 1st 2nd 3rd IoU2D@0.25 IoU2D@0.5 IoU3D@0.25 IoU3D@0.5

- ◦ • • - - 73.8 68.4 40.2 22.7

- • • • - - 84.6 82.6 65.1 49.4

- • • ◦ ◦ • • - 70.9 62.7 27.4 13.7

- • • ◦ • • • - 85.3 83.4 60.7 45.6

- ◦ • ◦ • • ◦ ◦ • • 76.9 70.1 31.1 16.0

- ◦ • ◦ • • ◦ • • • 83.2 80.3 57.1 41.2

In this experiment, we test six possible training configurations, ranging from one-stage to three-stage, to evaluate the influence of different training strategies. As shown in Table 4, we found that a single-stage fine-tuning yields the best results. Adding more stages generally degrades the model performance. We also found that making all parameters trainable in a single stage is important, especially for 3D object detection. This may indicate that, compared to modern image encoders such

as SigLIP [61] and DINOv2 [39], current pre-trained point cloud encoders are not as versatile in supporting downstream tasks yet.

### 4 Experiments

In this section, we compare SPATIALLM to the current state-of-the-art on scene layout estimation (Section 4.1) and 3D object detection (Section 4.2), respectively. Following our empirical study, we use Qwen2.5-0.5B [42] as the base model and Sonata [56] as the point cloud encoder. We set of resolution at the finest level at 2.5cm and employ a single-stage training on our dataset.

Besides testing on established RGBD benchmarks, we also investigate whether our model can handle inputs from other sources, such as point clouds reconstructed from RGB videos (Section 4.3). Other extensions and more technical details can be found in the appendix.

#### 4.1 Layout Estimation

Layout estimation focuses on predicting architectural elements, i.e., walls, doors, and windows, within an indoor scene. In this experiment, we include results from two representative baselines, namely RoomFormer [60] and SceneScript [2]. As the state-of-the-art specialist model for layout estimation, RoomFormer employs a highly specialized network architecture, using two-level queries to predict room polygons and their corners, respectively. SceneScript, on the other hand, casts 3D structure estimation as an auto-regressive “next-token-prediction” problem. However, unlike our method, it uses specialized structured language commands to describe the scene and trains a Transformer decoder from scratch.

Following the setting of RoomFormer, we perform evaluation on the Structured3D benchmark [64], which contains 3,500 residential houses with diverse floorplans. We use the original data split of 3000/250/250 for training/validation/testing, respectively. For RoomFormer, we directly download the model checkpoint from its website3 for inference. Note that RoomFormer takes in 2D density maps projected from point clouds and outputs 2D layouts, which are converted to 3D by extruding based on the lowest/highest points in the point cloud. For SceneScript, we download the pre-trained model and code4, and fine-tune it on Structured3D with the default hyperparameter settings.

For our model, we include three variants: (i) trained on Structured3D only, (ii) trained on

Table 5: Experiment on layout estimation.

Params F1 - Structured3D

Method

- our dataset only, and (iii) first trained on our dataset, then fine-tuned on Structured3D. As can be seen in Table 5, directly training LLMs on the smaller Structured3D dataset leads to poor performance. In contrast, by first training on our new dataset and then on Structured3D, our model outperforms both baselines for this task. This demonstrates the benefits of our new dataset.

(M) IoU2D@0.25 IoU2D@0.5

RoomFormer [60] 42.2 83.4 81.4 SceneScript [2] 29.9 90.4 89.2

SPATIALLM (ft. Structured3D) 603.5 32.6 18.1 SPATIALLM (ft. Ours) 603.5 59.9 44.7 SPATIALLM (ft. Ours → Structured3D) 603.5 94.3 93.5

Figure 4 provides some qualitative comparisons. Since RoomFormer detects room elements independently, we can see that some doors and windows are not attached to any wall. Auto-regressive methods like SceneScript and our method can preserve the relationships between structures by design. But we observe that SceneScript tends to miss some elements in its output.

#### 4.2 3D Object Detection

Next, we compare SPATIALLM to existing methods on 3D object detection. We include results from two representative baselines, namely V-DETR [47] and SceneScript. As the state-of-the-art

- 3https://github.com/ywyue/RoomFormer
- 4https://github.com/facebookresearch/scenescript

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Ground truth RoomFormer [60] SceneScript [2] SPATIALLM Figure 4: Qualitative results on Structured3D dataset.

specialist model for the task, V-DETR builds upon the DETR framework [5] and makes several taskspecific improvements, including a 3D vertex relative position encoding (3DV-RPE), object-based normalization, among others. Following prior practice, we perform evaluation on ScanNet [13], which contains 1,513 3D indoor scans with annotations of 18 object categories. The training set is composed of 1,201 scenes, while 312 scenes are used for validation/testing. Since auto-regressive models (i.e., SPATIALLM and SceneScript) do not produce confidence scores, we report F1 scores instead of mean Average Precision (mAP), as suggested in [2].

For V-DETR, we directly use the model checkpoint downloaded from its website5. A confidence threshold of 0.5 is applied for both objectness and semantic predictions, followed by non-maximum suppression (NMS). For SceneScript, we fine-tune the model on ScanNet with the default hyperparameter settings. During fine-tuning, we apply the same data augmentations as used in PTv3 [55] and V-DETR [47].

Table 6: Experiment on 3D object detection. “*”: numbers reported on the overlapping subset of object classes between our dataset and ScanNet.

Params F1 - ScanNet

Method

(M) IoU3D@0.25 IoU3D@0.5

V-DETR [47] 79.4 65.1 56.8 SceneScript [2] 29.9 49.1 36.8

SPATIALLM (ft. ScanNet) 603.5 2.9 0.7 SPATIALLM (ft. Ours) 603.5 33.8∗ 22.6∗ SPATIALLM (ft. Ours → ScanNet) 603.5 65.6 52.6

For our model, we again include three variants: (i) trained on ScanNet only, (ii) trained on our dataset only, and (iii) first trained on our dataset, then fine-tuned on ScanNet. As can be seen in Table 6, the first variant has very low F1 scores, suggesting that ScanNet itself is too small to train LLMs on. By first training on our new dataset and then on ScanNet, our model outperforms SceneScript and achieves competitive results against the state-of-the-art specialist V-DETR.

- Figure 5 shows some qualitative results. Compared to SceneScript, SPATIALLM demonstrates improved detection accuracies across categories. In comparison to V-DETR, we observe that our method has some difficulty in locating objects in “picture”, “sink” and “showercurtain” classes. Notably, “picture” and “sink” are the smallest objects in ScanNet by volume (< 0.1m3), whereas “showercurtain” category is not included in our dataset and has the fewest occurrences in the ScanNet.

5https://github.com/V-DETR/V-DETR

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Ground truth V-DETR [47] SceneScript [2] SPATIALLM Figure 5: Qualitative results on ScanNet dataset.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 6: Qualitative results of zero-shot detection on videos.

#### 4.3 Zero-shot Detection on Videos

Although significant progress has been made in SFM and SLAM techniques over the past decades, reliably inferring 3D structure from videos remains a highly challenging problem. Recently, DUSt3R [52] and its extensions [27, 15, 36, 31] provide a novel solution that employs deep networks for simultane-

- ous point matching and camera motion estimation. These techniques are shown to be remarkably robust in dealing with real-world videos or unconstrained image collections.

In this experiment, we collect a set of 107 videos of virtual indoor scene tours and use MASt3RSLAM [36] to reconstruct a point cloud from each video. Then, we test SPATIALLM in a zero-shot setting on the reconstructed point clouds. Unlike RGBD scans, these reconstructions have significant noises, occlusions, and geometric artifacts, posing a greater challenge for structured 3D understanding.

- Figure 6 shows some qualitative results. Without fine-tuning, SPATIALLM demonstrates strong resilience to imperfect data and preserves consistent layout and object-level predictions. Trained on synthetic datasets with accurate bounding boxes, the model infers full object sizes and orientations from sparse or occluded inputs. In the first example, it predicts beds and nightstands extending to the floor, even when those regions are not visible. Leveraging the auto-regressive nature of LLMs, SPATIALLM is also capable of generating plausible room layouts from partial observations. In the second example, the model reasonably reconstructs areas such as the balcony and dining space, filling in missing elements based on contextual understanding.

### 5 Conclusion

In this study, we demonstrate the feasibility of training LLMs for structured indoor modeling tasks. We regard it as a meaningful step towards building future foundation models that can not only understand, but also reason about, interact with, and even create structured 3D scenes.

Limitations. SPATIALLM has several limitations. First, it falls short of delivering a universal, state-of-the-art model for extracting 3D structure from arbitrary point clouds. There are significant differences between point clouds from different sources, such as monocular videos, RGBD scans, and LiDAR sensors. As shown in Section 4, although our model exhibits reasonable generalizability across datasets, fine-tuning on a specific dataset is still needed to achieve the best performance. Further scaling up the data and model size could be a promising future direction. Second, in this paper, we focus on training LLM for structured indoor modeling only, without considering how it affects the model’s natural skills in natural language processing and reasoning. To address this, a more thorough benchmarking may be performed. Third, our current approach models indoor layouts using a set of predefined object categories, limiting the LLMs in leveraging their open-ended language capabilities. Future directions include extending our work to support open-vocabulary object detection and 3D visual question answering (VQA), enabling more flexible and generalizable scene understanding.

Impacts. We envision that SPATIALLM will be used in real-world applications such as layout estimation and 3D object detection. For example, people may take our code and fine-tune the model on their own datasets. On the methodology side, we hope that our work will inspire future studies on MLLM models for 3D scene understanding, reasoning, and creation.

### Acknowledgments and Disclosure of Funding

This work was partially supported by the Key R&D Program of Zhejiang Province (2025C01001) and the HKUST project 24251090T019.

### References

- [1] Armen Avetisyan, Tatiana Khanova, Christopher B. Choy, Denver Dash, Angela Dai, and Matthias Nießner. SceneCAD: Predicting object alignments and layouts in RGB-D scans. In ECCV, pages 596–612, 2020.
- [2] Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan P. Frost, Luke Holland, Campbell Orme, Jakob Engel, Edward Miller, Richard A. Newcombe, and Vasileios Balntas. SceneScript: Reconstructing scenes with an autoregressive structured language model. In ECCV, pages 247–263, 2024.
- [3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. ScanQA: 3D question answering for spatial scene understanding. In CVPR, pages 19107–19117, 2022.
- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitScenes: A diverse real-world dataset for 3D indoor scene understanding using mobile RGB-D data. In NeurIPS Datasets and Benchmarks, 2021.
- [5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, pages 213–229, 2020.
- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9630–9640, 2021.
- [7] Angel X. Chang, Angela Dai, Thomas A. Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. In 3DV, pages 667–676, 2017.

- [8] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas J. Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In CVPR, pages 14455–14465, 2024.
- [9] Dave Zhenyu Chen, Angel X. Chang, and Matthias Nießner. ScanRefer: 3D object localization in RGB-D scans using natural language. In ECCV, pages 202–221, 2020.
- [10] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. LL3DA: visual interactive instruction tuning for omni-3d understanding, reasoning, and planning. In CVPR, pages 26418–26428, 2024.
- [11] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. SpatialRGPT: Grounded spatial reasoning in vision-language models. In NeurIPS, pages 135062–135093, 2024.
- [12] Christopher B. Choy, JunYoung Gwak, and Silvio Savarese. 4D spatio-temporal convnets: Minkowski convolutional neural networks. In CVPR, pages 3075–3084, 2019.
- [13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In CVPR, pages 5828–5839, 2017.
- [14] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. ProcTHOR: Large-scale embodied AI using procedural generation. In NeurIPS, pages 5982–5994, 2022.
- [15] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jérôme Revaud. MASt3R-SfM: a fully-integrated solution for unconstrained structure-frommotion. In 3DV, 2025.
- [16] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-LLM: Extending language model for 3d visual reasoning. In WACV, pages 2195–2206, 2025.
- [17] Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Wang. Vision-and-language navigation: A survey of tasks, methods, and future directions. In ACL, pages 7606–7623, 2022.
- [18] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3d world into large language models. In NeurIPS, pages 20482–20494, 2023.
- [19] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, and Zhou Zhao. Chat-Scene: Bridging 3d scene and large language models with object identifiers. In NeurIPS, pages 113991–114017, 2024.
- [20] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. In ICML, pages 20413–20451, 2024.
- [21] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd. Omama, Ganesh Iyer, Soroush Saryazdi, Tao Chen, Alaa Maalouf, Shuang Li, Nikhil Varma Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, K. Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. ConceptFusion: Open-set multimodal 3d mapping. In RSS, 2023.
- [22] Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. SceneVerse: Scaling 3d vision-language learning for grounded scene understanding. In ECCV, pages 289–310, 2024.
- [23] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic VLMs: Investigating the design space of visually-conditioned language models. In ICML, 2024.
- [24] Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva. Habitat Synthetic Scenes Dataset (HSSD-200): An Analysis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal Navigation. In CVPR, pages 16384–16393, 2024.

- [25] Maksim Kolodiazhnyi, Anna Vorontsova, Matvey Skripkin, Danila Rukhovich, and Anton Konushin. UniDet3D: Multi-dataset indoor 3d object detection. In AAAI, pages 4365–4373, 2025.
- [26] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [27] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with MASt3R. In ECCV, pages 71–91, 2024.
- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: bootstrapping languageimage pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742, 2023.
- [29] Dingning Liu, Xiaoshui Huang, Yuenan Hou, Zhihui Wang, Zhenfei Yin, Yongshun Gong, Peng Gao, and Wanli Ouyang. Uni3D-LLM: Unifying point cloud perception, generation and editing with large language models. CoRR, abs/2402.03327, 2024.
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, pages 34892–34916, 2023.
- [31] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yanchao Yang, Qingnan Fan, and Baoquan Chen. SLAM3R: real-time dense scene reconstruction from monocular RGB videos. In CVPR, pages 16651–16662, 2025.
- [32] LLama Team. The Llama 3 Herd of Models. CoRR, abs/2407.21783, 2024.
- [33] Yunze Man, Shuhong Zheng, Zhipeng Bao, Martial Hebert, Liangyan Gui, and Yu-Xiong Wang. Lexicon3D: Probing visual foundation models for complex 3d scene understanding. In NeurIPS, pages 76819–76847, 2024.
- [34] Yongsen Mao, Yiming Zhang, Hanxiao Jiang, Angel X. Chang, and Manolis Savva. MultiScan: Scalable RGBD scanning for 3d environments with articulated objects. In NeurIPS, pages 9058–9071, 2022.
- [35] Ishan Misra, Rohit Girdhar, and Armand Joulin. An end-to-end transformer model for 3d object detection. In ICCV, pages 2886–2897, 2021.
- [36] Riku Murai, Eric Dexheimer, and Andrew J. Davison. MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors. In CVPR, pages 16695–16705, 2025.
- [37] Srivathsan Murali, Pablo Speciale, Martin R. Oswald, and Marc Pollefeys. Indoor Scan2BIM: Building information models of house interiors. In IROS, pages 6126–6133, 2017.
- [38] Sebastian Ochmann, Richard Vock, and Reinhard Klein. Automatic reconstruction of fully volumetric 3d building models from point clouds. CoRR, abs/1907.00631, 2019.
- [39] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024.
- [40] Akshay Gadi Patil, Supriya Gadi Patil, Manyi Li, Matthew Fisher, Manolis Savva, and Hao Zhang. Advances in data-driven analysis and synthesis of 3d indoor scenes. Comput. Graph. Forum, 43(1), 2024.
- [41] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J. Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, pages 5099–5108, 2017.
- [42] Qwen Team. Qwen2.5 Technical Report. CoRR, abs/2412.15115, 2024.
- [43] Qwen Team. Qwen2.5-VL Technical Report. CoRR, abs/2502.13923, 2025.

- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [45] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, pages 10912–10922, 2021.
- [46] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. FCAF3D: fully convolutional anchor-free 3d object detection. In ECCV, pages 477–493, 2022.
- [47] Yichao Shen, Zigang Geng, Yuhui Yuan, Yutong Lin, Ze Liu, Chunyu Wang, Han Hu, Nanning Zheng, and Baining Guo. V-DETR: DETR with vertex relative position encoding for 3d object detection. In ICLR, 2024.
- [48] Haotian Tang, Zhijian Liu, Xiuyu Li, Yujun Lin, and Song Han. TorchSparse: Efficient point cloud inference engine. In MLSys, pages 302–315, 2022.
- [49] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, Xichen Pan, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, pages 87310–87356, 2024.
- [50] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. RIO: 3d object instance re-localization in changing indoor environments. In ICCV, pages 7657–7666, 2019.
- [51] Haiyang Wang, Lihe Ding, Shaocong Dong, Shaoshuai Shi, Aoxue Li, Jianan Li, Zhenguo Li, and Liwei Wang. CAGroup3D: Class-aware grouping for 3d object detection on point clouds. In NeurIPS, pages 29975–29988, 2022.
- [52] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jérôme Revaud. DUSt3R: Geometric 3d vision made easy. In CVPR, pages 20697–20709, 2024.
- [53] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, Tao Jin, and Zhou Zhao. Dataefficiently learn large language model for universal 3D scene perception. In NAACL, pages 313–333, 2025.
- [54] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point Transformer V2: Grouped vector attention and partition-based pooling. In NeurIPS, pages 33330–33342, 2022.
- [55] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler, faster, stronger. In CVPR, pages 4840–4851, 2024.
- [56] Xiaoyang Wu, Daniel DeTone, Duncan P. Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob J. Engel, Richard A. Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Self-supervised learning of reliable point representations. In CVPR, pages 22193–22204, 2025.
- [57] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In CVPR, pages 21469–21480, 2024.
- [58] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Théophile Gervet, John M. Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, Alexander William Clegg, and Devendra Singh Chaplot. Habitat-matterport 3d semantics dataset. In CVPR, pages 4927–4936, 2023.
- [59] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV, pages 12–22, 2023.
- [60] Yuanwen Yue, Theodora Kontogianni, Konrad Schindler, and Francis Engelmann. Connecting the dots: Floorplan reconstruction using two-level queries. In CVPR, pages 845–854, 2023.

- [61] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pages 11941–11952, 2023.
- [62] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip H. S. Torr, and Vladlen Koltun. Point Transformer. In ICCV, pages 16239–16248, 2021.
- [63] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3D LLM: Learning position-aware video representation for 3d scene understanding. In CVPR, pages 8995–9006, 2025.
- [64] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A large photo-realistic dataset for structured 3d modeling. In ECCV, pages 519–535, 2020.
- [65] Hongyan Zhi, Peihao Chen, Junyan Li, Shuailei Ma, Xinyu Sun, Tianhang Xiang, Yinjie Lei, Mingkui Tan, and Chuang Gan. LSceneLLM: Enhancing large 3d scene understanding using adaptive visual preferences. In CVPR, pages 3761–3771, 2025.
- [66] Chenming Zhu, Tai Wang, Wenwei Zhang, Kai Chen, and Xihui Liu. ScanReason: Empowering 3d visual grounding with reasoning capabilities. In ECCV, pages 151–168, 2024.
- [67] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. LLaVA-3D: A simple yet effective pathway to empowering lmms with 3d-awareness. CoRR, abs/2409.18125, 2024.
- [68] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, and Xide Xia. Apollo: An exploration of video understanding in large multimodal models. In CVPR, pages 18891–18901, 2024.

### A SPATIALLM Dataset

SPATIALLM dataset comprises 12,328 distinct scenes with 54,778 rooms, featuring a total of 403,291 walls, 123,301 doors, and 48,887 windows, with an overall floor area of approximately 863,986 m2.

Object categories. We have curated a set of 59 commonly occurring object categories, organized by functional and semantic similarity, as shown in Table 7. Statistics of these categories can be found in Figure 7.

Table 7: Categories in the SPATIALLM dataset.

Super Categories Categories

Seatings sofa, chair, dining_chair, bar_chair, stool Beddings bed, pillow Cabinetry wardrobe, nightstand, tv_cabinet, wine_cabinet, bathroom_cabinet,

shoe_cabinet, entrance_cabinet, decorative_cabinet, washing_cabinet, wall_cabinet, sideboard, cupboard

Tables coffee_table, dining_table, side_table, dressing_table, desk Kitchen appliances integrated_stove, gas_stove, range_hood, microwave_oven, sink, stove,

refrigerator Bathroom fixtures hand_sink, shower, shower_room, toilet, tub Lighting illumination, chandelier, floor_standing_lamp Decoration wall_decoration, painting, curtain, carpet, plants, potted_bonsai Electronics tv, computer, air_conditioner, washing_machine Others clothes_rack, mirror, bookcase, cushion, bar, screen, combination_sofa,

dining_table_combination, leisure_table_and_chair_combination, multifunctional_combination_bed

Figure 8 plots the object-to-room correlation heatmap, which reveals strong correlations between object distribution and specific room types. Additionally, each room type contains a variety of object categories, highlighting the diversity of our dataset.

Dataset visualizations. We provide visualizations of the ground-truth point cloud and layout annotations in Figure 9 to showcase the quality of the photo-realistic scans and the realism of the layout. Additionally, we show rendered images of objects in Figure 10 to illustrate the diversity and quality of the objects in our dataset.

### B Implementation Details of SPATIALLM

In Figure 12, we show the full prompt and an example response of SPATIALLM. Note that we shift the point cloud and the corresponding layout and object box coordinates to ensure that they are all non-negative. Next, we quantize the coordinates into 1,280 bins, each with a resolution of 2.5cm. LLM predicts integer values, which are mapped back to continuous values using the inverse transformation and normalization, restoring them to the original coordinate system.

Data augmentations. We apply a number of augmentations to the point cloud, which are summarized in Table 8 and explained below.

We begin with a cuboid crop following V-DETR [47]. This process involves randomly selecting a point as the center of the cuboid and then determining random dimensions for the cuboid relative to the input point cloud size. We maintain a minimum aspect ratio of 0.8 and ensure that at least 50,000 points remain after the cropping process.

Then, we apply four types of jitter noises to the point cloud: (i) jitter that simulates the minor uncertainties inherent in sensor accuracy; (2) jitter that replicates noise in areas with greater uncertainty; (3) jitter that reflects noise along the edges, typically from floating points between foreground objects and the background; and (4) jitter that represents distant floating points outside the room, often attributed to glossy windows, reflective mirrors, and points along the path to the outdoors.

[Figure 37]

[Figure 38]

- Figure 7: Statistics of the SPATIALLM dataset. Top: Number of objects in each category. Bottom: Distribution of physical sizes in each category. We compute the object size as the diagonal length of the object’s bounding box (in meters). All objects in our dataset have realistic physical sizes.

[Figure 39]

Figure 8: Room object correlation of SPATIALLM dataset.

Compute resources. We train SPATIALLM for 4 epochs with a total batch size of 64. The learning rate is set at 10−4, using a cosine scheduler with a warm-up ratio of 0.03. The parameters for AdamW optimizer are as follows: adam_beta1 is 0.9, adam_beta2 is 0.99, and adam_epsilon is 1 × 10−8. We utilized 32 NVIDIA H20 GPUs, and training on SPATIALLM dataset takes approximately one day.

For experiments on Structured3D [64] and ScanNet [13], we use a batch size of 8. Fine-tuning on Structured3D involves 50 epochs, while keeping the other settings the same. For fine-tuning on ScanNet, we introduce an additional pre-training stage to mitigate the difference in label spaces across datasets. Specifically, we map the classes in our dataset to those of ScanNet, and further train the model on our dataset for 3 epochs, followed by fine-tuning on ScanNet for 30 epochs.

### C Supplementary Results for SPATIALLM Experiments

In this section, we present per-category quantitative results for (i) layout estimation, (ii) 3D object detection, and (iii) zero-shot detection on videos.

Table 8: Point cloud data augmentations.

Method Parameters cuboid crop min_points=50,000, aspect=0.8, min_crop=0.75, max_crop=1.0, p=0.1 random jitter σ=0.025, clip=0.05, ratio=0.8, p=0.9 random jitter σ=0.2, clip=0.2, ratio=0.05, p=0.8

- random jitter σ=0.4, clip=1.0, ratio=0.001, p=0.75
- random jitter σ=0.5, clip=4.0, ratio=0.0005, p=0.65 elastic distort params=[[0.2, 0.4], [0.8, 1.6]], p=[0.8, 0.5] random rotate axis=z, angle=[0, 90, 180, 270] random scale scale=[0.75, 1.25] auto contrast p=0.2 chromatic translation p=0.75, ratio=0.1 chromatic jitter std=0.05; p=0.8 color drop p=0.1

#### C.1 Layout Estimation

Per-category results. In Table 9, we report the F1 scores in percentages at 0.25 and 0.5 thresholds of IoU2D for architectural elements i.e., walls, doors and windows, and the averaged score across the categories on Structured3D test set.

Table 9: Experiment results on layout estimation.

IoU2D@0.25 IoU2D@0.5 avg. wall door window avg. wall door window

Method

RoomFormer [60] 83.4 86.0 85.1 78.9 81.4 84.6 83.1 76.6 SceneScript [2] 90.4 92.9 92.8 85.5 89.2 91.9 92.5 83.3

SPATIALLM (ft. Structured3D) 32.6 54.1 19.4 24.3 18.1 33.5 9.4 11.5 SPATIALLM (ft. Ours) 59.9 71.5 42.7 65.6 44.7 64.8 39.7 29.6 SPATIALLM (ft. Ours → Structured3D) 94.3 95.3 95.9 91.7 93.5 94.5 95.7 90.2

#### C.2 3D Object Detection

Per-category results. We report the F1 scores in percentages at 0.25 and 0.5 thresholds of IoU3D for 18 ScanNet categories and the averaged score across the categories in Table 10 and Table 11, respectively.

For IoU3D@0.25, we have +16.5% overall gain compared to SceneScript, and +0.5% difference compared to V-DETR. For IoU3D@0.5, we have +15.8% overall gain compared to SceneScript, and -4.2% difference compared to V-DETR. The largest gaps between our model and V-DETR correspond to “picture”, “sink”, and “shower_curtain”. Note that these are either the smallest or the least occurring objects in ScanNet.

- Table 10: 3D object detection results with IoU3D@0.25. †: trained on ScanNet only. ∗: trained on our dataset only. “n/a” indicates the category is not included in our dataset.

bookshelf

s.curtain

garbbin.

window

average

bathtub

counter

refrige.

cabinet

curtain

picture

toilet

chair

table

desk

door

sink

sofa

bed

Method

V-DETR [47] 65.1 77.4 80.1 48.8 44.4 82.5 55.6 66.1 68.0 62.3 54.0 47.4 50.8 74.0 79.5 72.7 55.3 98.0 54.9 SceneScript [2] 49.1 64.5 71.1 39.7 30.3 81.1 43.4 30.9 53.4 41.7 42.6 11.8 28.2 58.6 48.9 68.3 55.8 77.5 36.5

SPATIALLM† 2.9 6.5 7.0 2.5 1.2 4.0 1.4 2.3 4.8 2.5 0.3 0.0 1.4 2.3 0.9 1.7 4.7 5.0 3.7 SPATIALLM∗ 33.8 57.0 49.2 24.4 12.2 50.9 n/a 15.3 48.1 n/a n/a 9.7 23.4 n/a 3.6 38.3 31.4 75.6 n/a SPATIALLM 65.6 80.6 79.9 52.0 40.0 86.6 51.0 66.8 62.8 67.1 55.6 29.7 53.6 81.6 70.7 78.9 63.9 95.4 64.3

- Table 11: 3D object detection results with IoU3D@0.5. †: trained on ScanNet only. ∗: trained on our dataset only. “n/a” indicates the category is not included in our dataset.

bookshelf

s.curtain

garbbin.

window

average

bathtub

counter

refrige.

cabinet

curtain

picture

toilet

chair

table

desk

door

sink

sofa

bed

Method

V-DETR [47] 56.8 68.8 75.5 50.4 39.1 76.2 43.6 54.1 52.1 50.9 47.8 42.7 49.2 61.5 61.4 65.4 50.6 88.6 43.9 SceneScript [2] 36.8 54.8 65.9 37.3 21.2 73.7 25.3 16.9 44.5 23.5 30.5 3.0 23.3 19.5 24.9 62.0 50.1 64.2 21.7

SPATIALLM† 0.7 2.2 2.0 0.0 0.1 0.6 1.4 0.8 1.7 0.4 0.0 0.0 1.4 0.0 0.0 0.0 0.2 2.0 0.0 SPATIALLM∗ 22.6 37.6 41.0 13.5 6.8 30.7 n/a 1.0 34.3 n/a n/a 2.6 17.7 n/a 1.1 28.4 19.5 59.0 n/a SPATIALLM 52.6 74.2 71.7 47.7 29.5 79.3 40.1 53.3 52.6 46.6 41.7 10.2 42.9 50.6 40.2 71.1 60.3 90.5 45.0

#### C.3 Zero-shot Detection on Videos

Per-category results. We collect 107 virtual room-tour videos and use MASt3R-SLAM [36] to reconstruct a point cloud from each video. Note that these videos are rendered from virtual scenes created by professional designers, thus ground-truth 3D annotations are available. To calibrate the reconstructed point cloud with ground truth 3D annotations, we align the estimated camera trajectory from MASt3R-SLAM to the ground-truth camera trajectory. Then, we perform zero-shot detection with SPATIALLM on the calibrated point clouds.

In Table 12, we report the quantitative results on the top-20 occurring categories in the video test set. Note that this is a challenging task, because the reconstructed point clouds are often noisy and highly incomplete. Furthermore, because of the differences in scale and position due to imperfect alignment of camera trajectories, the objects in the point cloud may not match well with the ground truth 3D annotations, leading to lower F1 scores.

Table 12: Zero-shot results on layout estimation and 3D object detection with IoU@0.25.

- (a) Layout estimation avg. wall door window 55.7 68.2 47.4 51.4

- (b) 3D object detection

nightstand

chandelier

diningtab.

coffeetab.

wardrobe

painting

aircond.

average

sidetab.

refrige.

cabinet

dresser

curtain

carpet

plants

tvcab.

chair

stool

sofa

bed

tv

36.9 37.0 67.0 36.8 39.6 95.2 69.1 32.3 11.2 24.2 26.3 27.3 64.9 9.7 24.0 46.7 30.8 16.7 38.2 24.1 18.0

### D Extensions of SPATIALLM

In this section, we present more proof-of-concept experiment results on future extensions of SPATIALLM, including (i) handling point clouds from diverse sources, and (ii) adapting to various downstream tasks via language-based prompts.

#### D.1 Supporting Point Clouds from Diverse Sources

Point clouds offer a lightweight and flexible 3D representation that can be readily generated from a variety of sources. Meanwhile, our dataset is large and diverse, enabling strong zero-shot performance over point cloud inputs in both synthetic and real-world data. In this experiment, we examine the following four types of input sources:

- • Text-to-3D: We first generate an image via Flux-dev [26] with prompt “a low poly 3D living room in cartoon style”, then convert the image into 3D using TRELLIS [57].

- • Hand-held video camera: We take a real-world video using a hand-held camera and perform 3D reconstruction using MASt3R-SLAM [36].
- • LiDAR-based reconstruction: We take point clouds reconstructed from captures using iPhone ARKit from MultiScan [34].
- • Synthetic mesh: We create point clouds by randomly sampling mesh surfaces from a synthetic dataset HSSD [24].

As can be seen in Figure 13, our model is robust to variations in point cloud origin, structure, and appearance.

#### D.2 Task Adaptation via Language-based Prompts

A key idea of our work is to represent the 3D scene structures in pure text form. This enables SPATIALLM to be conveniently adapted to different downstream tasks via natural language instruction, without changing the underlying model architecture. Below, we demonstrate two such extensions.

Detection with user-specified categories. We enable the model to perform object detection conditioned on user-specified categories by leveraging the flexibility of LLMs. During training, the input prompt is customized to indicate which object categories to detect. The model then learns to selectively predict oriented bounding boxes (OBBs) for instances matching the specified categories, all within a unified training procedure. This experiment highlights the model’s adaptability through prompt-based control. Some qualitative results are provided in Figure 14.

Semantic label completion. Besides natural language prompts, SPATIALLM also supports structured language scripts as input, enabling novel tasks such as semantic label completion. In this task, the model receives object bounding boxes and their poses written in a general-purpose language (i.e., Python), while the object categories are left unspecified. The model then predicts the semantic label of each object based on the spatial layout and the corresponding point cloud data. Note that this problem often arises in real-world design workflows that involve 3D assets with known positions but without semantic labels.

Compared to 3D object detection, this task is relatively easy. Nevertheless, it demonstrates the versatility and potential use of large language models for structured 3D understanding. We show some qualitative results in Figure 15. Our model achieves an overall classification accuracy of 96.8% on the test set of our dataset.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

livingroomlivingroomlivingroomlivingroombedroombathroombedroombedroom

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

- Figure 9: Point clouds with overlaid 3D layout and object annotations in SPATIALLM dataset.

sofa

coffee table

tv cabinet

bed

nightstand

wardrobe

chair

table

desk

shoe cabinet

chandelier

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

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

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

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

dressing table

- Figure 10: Example objects in SPATIALLM dataset. Note that each category has a diverse set of objects with high quality in both geometry and appearance.

[Figure 168]

- Figure 11: Five-level hierarchical structure of the point cloud encoder [56]. D indicates the feature dimension at each level, and grid pool downscales the point cloud resolution by 2 times along each dimension.

#### Response:

#### Prompt:

[Figure 169]

- wall_0=Wall(144,15,30,139,15,30,101)
- wall_1=Wall(144,15,30,144,26,30,101)
- wall_2=Wall(139,15,30,128,16,30,101)
- wall_3=Wall(128,16,30,118,18,30,101)
- wall_4=Wall(47,18,30,118,18,30,101)
- wall_5=Wall(47,18,30,47,42,30,101)
- wall_6=Wall(144,26,30,152,26,30,101)
- wall_7=Wall(152,26,30,152,42,30,101)
- wall_8=Wall(47,42,30,67,42,30,101)
- wall_9=Wall(67,42,30,67,53,30,101)
- wall_10=Wall(138,42,30,152,42,30,101)
- wall_11=Wall(138,42,30,138,53,30,101)
- wall_12=Wall(49,53,30,67,53,30,101)
- wall_13=Wall(49,53,30,49,151,30,101)
- wall_14=Wall(138,53,30,152,53,30,101)
- wall_15=Wall(152,53,30,152,170,30,101)
- wall_16=Wall(19,151,30,49,151,30,101)
- wall_17=Wall(19,151,30,19,269,30,101)
- wall_18=Wall(118,170,30,152,170,30,101)
- wall_19=Wall(118,170,30,118,269,30,101)
- wall_20=Wall(19,269,30,118,269,30,101)

<point_cloud> Detect walls, doors, windows, boxes. The reference code is as follows:

@dataclass class Wall:

a_x: int a_y: int a_z: int

- b_x: int
- b_y: int
- b_z: int height: int

- door_0=Door(wall_16 ,32,151,60,30,75)
- door_1=Door(wall_17 ,19,194,60,27,75)
- door_2=Door(wall_19 ,118,213,60,37,75)
- door_3=Door(wall_20 ,98,269,60,30,75)

@dataclass class Door:

- window_0=Window(wall_4 ,94,18,74,73,75)
- window_1=Window(wall_17 ,19,162,60,17,75)

wall_id: str

- bbox_0=Bbox(curtain ,93,20,74,320,165,10,96)
- bbox_1=Bbox(plants ,144,46,53,320,27,27,74)
- bbox_2=Bbox(air_conditioner ,147,58,55,412,15,15,81)
- bbox_3=Bbox(plants ,58,61,52,480,24,31,70)
- bbox_4=Bbox(tv_cabinet ,55,103,41,480,111,19,34)
- bbox_5=Bbox(tv ,50,103,69,480,77,6,44)
- bbox_6=Bbox(combination_sofa ,119,110,45,480,166,105,48)
- bbox_7=Bbox(painting ,152,110,80,480,102,6,41)
- bbox_8=Bbox(floor -standing_lamp ,145,163,55,320,21,21,80)
- bbox_9=Bbox(dining_chair ,50,200,43,320,26,28,41)
- bbox_10=Bbox(dining_chair ,67,214,43,480,26,28,41)
- bbox_11=Bbox(dining_table ,49,228,50,480,77,40,43)
- bbox_12=Bbox(dining_chair ,67,233,43,480,26,28,41)
- bbox_13=Bbox(painting ,21,236,77,480,58,6,39)
- bbox_14=Bbox(shoe_cabinet ,113,246,68,480,52,16,120)
- bbox_15=Bbox(dining_chair ,41,252,43,480,26,28,41)
- bbox_16=Bbox(dining_chair ,67,252,43,480,26,28,41)
- bbox_17=Bbox(sideboard ,54,264,68,320,98,16,120)

- position_x: int
- position_y: int
- position_z: int width: int height: int

@dataclass class Window:

wall_id: str

- position_x: int
- position_y: int
- position_z: int width: int height: int

@dataclass class Bbox:

class: str

- position_x: int
- position_y: int
- position_z: int angle_z: int

- scale_x: int
- scale_y: int
- scale_z: int

Figure 12: An example conversation of SPATIALLM.

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

##### Figure 13: Qualitative results on various input sources. First row: Text-to-3D. Second row: Handheld video camera. Third row: LiDAR-based reconstruction. Fourth row: Synthetic mesh.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

all objects bed chair desk

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

layout dining_chair tv, tv_cabinet sofa, coffee_table

- Figure 14: Qualitative results of detection with user-specified categories. Labels below each figure indicate the category specified for detection.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Input GT SPATIALLM

- Figure 15: Qualitative results of semantic label completion. Black-colored input boxes denote objects with unknown categories, which are classified by SPATIALLM.

