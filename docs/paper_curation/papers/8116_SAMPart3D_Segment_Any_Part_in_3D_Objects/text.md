# arXiv:2411.07184v2[cs.CV]16Nov2024

## SAMPart3D: Segment Any Part in 3D Objects

Yunhan Yang1 Yukun Huang1 Yuan-Chen Guo2 Liangjun Lu1 Xiaoyang Wu1 Edmund Y. Lam1 Yan-Pei Cao2† Xihui Liu1

1 The University of Hong Kong 2 VAST Project Page: https://yhyang-myron.github.io/SAMPart3D-website

[Figure 1]

Figure 1. SAMPart3D is able to segment any 3D object into semantic parts across multiple levels of granularity, without the need for predefined part label sets or text prompts. It supports a range of applications, including part-level editing and interactive segmentation.

### Abstract

3D part segmentation is a crucial and challenging task in 3D perception, playing a vital role in applications such as robotics, 3D generation, and 3D editing. Recent methods harness the powerful Vision Language Models (VLMs) for 2D-to-3D knowledge distillation, achieving zero-shot 3D part segmentation. However, these methods are limited by their reliance on text prompts, which restricts the scalability to large-scale unlabeled datasets and the flexibility in handling part ambiguities. In this work, we introduce SAMPart3D, a scalable zero-shot 3D part segmentation framework that segments any 3D object into semantic parts at multiple granularities, without requiring predefined part label sets as text prompts. For scalability, we use text-agnostic vision foundation models to distill a 3D feature extraction backbone, allowing scaling to large unlabeled 3D datasets to learn rich 3D priors. For flexibility,

: Corresponding author, †: Project leader.

we distill scale-conditioned part-aware 3D features for 3D part segmentation at multiple granularities. Once the segmented parts are obtained from the scale-conditioned partaware 3D features, we use VLMs to assign semantic labels to each part based on the multi-view renderings. Compared to previous methods, our SAMPart3D can scale to the recent large-scale 3D object dataset Objaverse and handle complex, non-ordinary objects. Additionally, we contribute a new 3D part segmentation benchmark to address the lack of diversity and complexity of objects and parts in existing benchmarks. Experiments show that our SAMPart3D significantly outperforms existing zero-shot 3D part segmentation methods, and can facilitate various applications such as part-level editing and interactive segmentation.

### 1. Introduction

3D part segmentation is a fundamental 3D perception task that is essential for various application areas, such as robotic manipulation, 3D analysis and generation, part-level edit-

ing [38] and stylization [7].

In the past few years, data-driven fully supervised methods [23, 30, 31, 34, 53] have achieved excellent results on closed-set 3D part segmentation benchmarks [3, 27]. However, these methods are limited to segmenting simple objects due to the restricted quantity and diversity of 3D data with part annotations. Despite the recent release of largescale 3D object datasets [9, 10, 45], acquiring part annotations for such vast amounts of 3D assets is time-consuming and labor-intensive, which prevents 3D part segmentation from replicating the success of data scaling and model scaling in 2D segmentation [21].

To achieve zero-shot 3D part segmentation in the absence of annotated 3D data, several challenges need to be addressed. The first and most significant challenge is how to generalize to open-world 3D objects without 3D part annotations. To tackle this, recent works [1, 20, 25, 47, 56] have utilized pre-trained 2D foundation vision models, such as SAM [21] and GLIP [22], to extract visual information from multi-view renderings and project it onto 3D primitives, achieving zero-shot 3D part segmentation. However, these methods rely solely on 2D appearance features without 3D geometric cues, leading to the second challenge: how to leverage 3D priors from unlabeled 3D shapes. PartDistill [41] has made a preliminary exploration by introducing a 2D-to-3D distillation framework to learn 3D point cloud feature extraction, but it cannot scale to large 3D datasets like Objaverse [9] due to the need for predefined part labels and the constrained capabilities of GLIP. Building on existing works, we further explore the third challenge: the ambiguity of 3D parts, which manifests primarily in semantics and granularity. Semantic ambiguity arises from the vague textual descriptions of parts. Existing methods rely on vision-language models (VLMs) like GLIP, which require a part label set as text prompt. Unfortunately, not all 3D parts can be clearly and precisely described in text. Granularity ambiguity considers that a 3D object can be segmented at multiple levels of granularity. For example, the human body can be divided into broader sections, such as upper and lower halves, or into finer parts like limbs, torso, and head. Previous methods rely on fixed part label sets and lack flexible control over segmentation granularity.

To tackle the three aforementioned challenges, in this work, we propose SAMPart3D, a scalable zero-shot 3D part segmentation framework that segments object parts at multiple granularities without requiring preset part labels as text prompts. We argue that previous works overly rely on predefined part label sets and GLIP, limiting their scalability to complex, unlabeled 3D datasets and their flexibility in handling semantic ambiguity of 3D parts. To address this, we abandon GLIP and instead utilize the more low-level, text-independent DINOv2 [29] model for 2D-to3D feature distillation, eliminating the reliance on part label

sets and enhancing both scalability and flexibility. Besides, to handle the ambiguity in segmentation granularity, we employ a scale-conditioned MLP [19] distilled from SAM for granularity-controllable 3D part segmentation. The distillation from DINOv2 and SAM is divided into two training stages to balance efficiency and performance. After obtaining the segmented 3D parts, we adaptively render multi-view images for each part based on its visual area, then use the powerful Multi-modal Large Language Models (MLLMs) [6, 15, 24, 42] to assign semantic descriptions for each part based on the renderings, yielding the final part segmentation results.

In summary, our contributions are as follows:

- • We introduce SAMPart3D, a scalable zero-shot 3D part segmentation framework that segments object parts at multiple granularities without requiring preset part labels as text prompts.
- • We propose a text-independent 2D-to-3D distillation, which enables learning 3D priors from large-scale unlabeled 3D objects and can handle part ambiguity in both semantic and granularity aspects. The distillation is twostage, striking a balance between segmentation performance and training efficiency.
- • We introduce PartObjaverse-Tiny, a 3D part segmentation dataset which provides detailed semantic and instance level part annotations for 200 complex 3D objects.
- • Extensive experiments demonstrate that SAMPart3D achieves outstanding part segmentation results on complex and diverse 3D objects compared to existing zeroshot 3D part segmentation methods. Furthermore, our method can facilitate various applications, such as interactive segmentation and part-level editing.

### 2. Related Work

2D Foundation Models. Recently, 2D vision foundation models have advanced significantly due to large-scale data and model size growth. Based on learning strategies, these models can be grouped into: traditional models, textually-prompted models, and visually-prompted models. Traditional models rely solely on images and use selfsupervised objectives, like masked patch reconstruction in MAE [14] and self-distillation in DINO [2, 29]. Textuallyprompted models use large-scale text-image pairs, as seen in CLIP [36], which aligns images and text for strong zero-shot performance. Since text prompts are less effective for fine-grained tasks like segmentation [44], visuallyprompted models use visual cues like bounding boxes, points, or masks. SAM [21], for example, employs visual prompts for zero-shot segmentation on new domains. Recently, efforts [4, 17, 32, 48, 50, 51, 57] have explored using 2D foundation models for 3D content understanding. In this work, we integrate multiple 2D vision foundation models for zero-shot 3D semantic part segmentation at varying

(a) (b)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Segmentation-Aware 3D Features

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

MLP

[Figure 14]

[Figure 15]

PTv3 object

PTv3 object

[Figure 16]

MLP

2D-to-3D Distillation

Contrastive Learning

Large-Scale 3D Dataset

3D Features

XYZ & Normal & RGB XYZ & Normal & RGB

Multi-view Rendering Multi- Scale 2D Segmentation Masks

Multi-view Rendering DINOv2 Features

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

… … … …

DINOv2 SAM

(c)

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Frozen

“The Right Foot” “The Right Shoe”

MLLM

Clustering

[Figure 37]

Learnable

Segmentation-Aware 3D Features

Multi-View Rendering with Mask Annotation

3D Part Segmentation

Figure 2. An overview pipeline of SAMPart3D. (a) We first pre-train 3D backbone PTv3-object on 3D large-scale data Objaverse, distilling visual features from FeatUp-DINOv2. (b) Next, we train light-weight MLPs to distill 2D masks to scale-conditioned grouping. (c) Finally, we cluster the feature of point clouds and highlight the consistent 2D part area with 2D-3D mapping on multi-view renderings, and then query semantics from MLLMs.

dational models. Unlike previous methods directly transferring 2D pixel-wise or bounding-box-wise predictions to 3D segmentation, PartDistill [41] adopts a cross-modal teacherstudent distillation framework to learn a 3D student backbone for extracting point-specific geometric features from unlabeled 3D shapes. In this work, we extend cross-modal distillation to a more challenging large-scale 3D dataset, Objaverse [9], without requiring text prompts for 3D parts, and achieve granularity-controllable segmentation.

granularities.

3D Part Segmentation. 3D part segmentation aims to divide a 3D object into semantic parts, which is a longstanding problem in 3D computer vision. Early works [23, 30, 31, 34, 53] primarily focused on exploring network architecture designs to better learn 3D representations. Qi et al. [31] propose a hierarchical neural network named PointNet++, which leverages neighborhoods at multiple scales to achieve both robustness and detail capture. Zhao et al. [53] design an expressive Point Transformer layer, which can be used to construct high-performing backbones for semantic segmentation of 3D point clouds. These methods typically employ fully supervised training [53], requiring timeconsuming and labor-intensive 3D part annotations. Limited by the scale and diversity of 3D part datasets [3, 27], they struggle to achieve generalization on complex 3D objects in open-world scenarios.

### 3. Method

As shown in Figure 2, the proposed framework SAMPart3D consists of three stages: large-scale pre-training to learn a 3D feature extraction backbone from a vast number of unlabeled 3D objects, as described in Section 3.1; sample-specific fine-tuning to train a lightweight MLP for scale-conditioned grouping, as described in Section 3.2; and training-free semantic querying for assigning semantic labels to each part using a multimodal large language model, as described in Section 3.3.

Zero-shot 3D Part Segmentation. To overcome the the limitations of 3D annotated data and pursue zero-shot capabilities, recent 3D part segmentation methods [1, 25, 37, 39, 41, 47, 55–57] leverage 2D priors from foundation vision models [21, 22, 29, 36]. PartSLIP [25] leverages the image-language model GLIP [22] to solve both semantic and instance segmentation for 3D object parts, where the GLIP model is expected to predict multiple bounding boxes for all part instances. The subsequent PartSLIP++ [56] integrates the pretrained 2D segmentation model SAM [21] into the PartSLIP pipeline, yielding more accurate pixelwise part annotations than the bounding boxes used in PartSLIP. Similarly, ZeroPS [47] introduces a two-stage 3D part segmentation and classification pipeline, bridging the multiview correspondences and the prompt mechanism of foun-

#### 3.1. Large-scale Pre-training: Distilling 2D Visual Features to 3D Backbone

In this stage, we aim to learn a 3D feature extraction backbone that leverages the geometric cues of 3D objects and learns 3D priors from a large-scale collection of unlabeled 3D objects.

Training Data. Unlike the previous work PartDistill [41], which was trained on limited categories of 3D objects, we utilize the large-scale 3D object dataset Objaverse [9] as our training data. Objaverse encompasses over 800K 3D assets spanning diverse object categories, providing rich 3D priors

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Semantic Segmentation

Body, Flowerpot, Leaf, Head, Head Leaves, Eyebrow, Mud

Balustrade, Body, Bench, Head, Inside Seat,

Base, Body, Enclosure, Fan Blades, Motor, Plug, Wire

Capsule, Dvd, Hair,

Head, Head Band, Foot, Hand, Body, Stethoscope

Billboard, Tire, Light, Seat, Main Billboard, Nets, Roof, Staircase, Steering Wheel

[Figure 42]

[Figure 43]

[Figure 44]

Instance Segmentation

[Figure 45]

Back Cylinder, Head, Left Arm, Left Bumper, Left Hand, Left Hub, Left Track, Lower Body, Right Arm, Right Bumper,

Body, Hair, Head, Horns, Left Hind Leg, Left Right Leg,

Armrest, Balcony, Big Iron Fence, Blocking The Door, Body, Cask, Deck, Gold,

Head, Left Arm, Left Foot, Left Hand, Lower Body,

Hanging Car, Knife, Ladder, Left Arrow Stand, Left Cannon, Right Arrow Stand, Right Cannon, Main Mast, Medium Mast, Rope, Small Iron Fence, Small Mast, Sub-Mast, Top

Right Front Leg, Right Hind Leg, Sunglasses, Tail

Right Arm, Right Foot,

Right Hand, Upper Body

Right Hand, Right Hub, Right Track, Upper Body

Figure 3. Visualization of PartObjaverse-Tiny with part-level semantic and instance segmentation labels.

point-wise supervision in 3D feature extraction.

for zero-shot 3D part segmentation. Additionally, considering that current 3D feature extraction network architectures are primarily developed for 3D point clouds, we randomly sample point clouds from the mesh surfaces of 3D objects as the input to our backbone.

Specifically, for each training iteration, we sample a batch of 3D objects, with each object represented by a point cloud X ∈ RN×3, where N denotes the number of 3D points. We input the point cloud X into the PTv3-object backbone, resulting in 3D features F3D ∈ RN×C, where C is the feature dimension of 384 consistent with DINOv2 features. Then, to obtain corresponding 2D visual features for supervision, we render images from K different views for each object and extract the corresponding DINOv2 features. Utilizing the mapping relationship between point clouds and pixels, we can directly obtain the 2D features F2D ∈ RN×C of the 3D point cloud. However, considering occlusion, not all 3D points can be assigned 2D features given a single view. To address this, we use depth information to determine the occlusion status of the point cloud following [16]. For occluded 3D points, we directly assign their original 3D features from F3D. Finally, by averaging

Backbone for 3D Feature Extraction. Building upon the state-of-the-art point cloud perception backbone, Point Transformer V3 (PTv3) [8, 46], we further tailor the architecture to accommodate the characteristics of 3D objects, resulting in PTv3-object. Specifically, PTv3 is designed for scene-level point clouds, incorporating numerous downsampling layers for a large receptive field and low computational load. However, the number of points and spatial extent required to represent an object is much smaller than that required for a scene. Therefore, we removed most of the down-sampling layers from PTv3 and instead stacked more transformer blocks to enhance detail preservation and feature abstraction. Note that our learning framework is model-agnostic and can be used with other more advanced network architectures for 3D feature extraction.

- the 2D features from all K rendered views, we obtain the final 2D features of the point cloud:

F2D =

1 K

K

k=1

F2D(k), (1)

where F2D(k) represents the obtained 2D features of the point cloud at the k-th view, and we simply choose a mean

squared error (MSE) loss:

Lpre = (F3D − F2D)2 (2)

as the learning objective for distilling 2D visual features to

- the 3D backbone.

Distilling 2D Visual Features to 3D Backbone. To train the backbone for 3D feature extraction on large-scale unlabeled 3D objects from Objaverse [9], pre-trained 2D vision foundation models are needed as supervision. Previous methods, such as PartDistill [41], use VLMs as supervision and require part label sets as text prompts, making it difficult to scale to Objaverse. Therefore, we abandon VLMs and instead utilize the more low-level, text-independent DINOv2 [29] model as supervision for visual feature distillation. In particular, the visual features extracted by DINOv2 are low-resolution and lack detail, making them unsuitable for subsequent part segmentation. To address this, we employ the recently proposed feature upsampling technique, FeatUp [13], to enhance the DINOv2 features for use as

#### 3.2. Sample-specific Fine-tuning: Distilling 2D Masks for Multi-granularity Segmentation

After pre-training the backbone via distilling the 2D visual features, we can effectively extract 3D features of any 3D object. These 3D features are used together with 2D segmentation masks from SAM [21] for zero-shot 3D part segmentation. Furthermore, considering the ambiguity in segmentation granularity, we aim to introduce a scale value to control the granularity of the segmentation. To this end, we introduce a scale-conditioned lightweight MLP that enables 3D part segmentation at various scales, inspired by GARField [19] and GraCo [54].

Long Skip Connection. Although the pre-trained backbone is able to extract rich 3D features, the low-level cues of point cloud (critical for point-wise prediction tasks) are lost due to overly deep networks. Therefore, we introduce a MLP-based long skip connection module to capture the low-level features of point cloud. Specifically, we first assign the normal values of the faces to each corresponding point in the point cloud to provide the shape information of the mesh. Then, these normal values, along with color and coordinates, serve as inputs for the long skip connection module, the outputs of which are added to the outputs of the 3D backbone to complement low-level features.

Scale-conditioned Grouping. We first render multi-view images of the 3D object and utilize SAM to generate 2D masks of these multi-view renderings. For each mask, we can find the relevant points and calculate the 3D scale σ with:

σ = (εσx)2 + (εσy)2 + (εσz)2, (3)

where σx,σy,σz are the standard deviations of coordinates in the x,y,z directions, respectively; ε is a scaling factor for better distinguishing the scales of different masks, which we set to 10.

Then, we sample paired pixels on the valid region of 2D renderings for contrastive learning. Specifically, for two 3D points pi and pj mapping from a 2D pixel pair, we can obtain their features:

##### Fi = FBi (σi) + FPi (σi), Fj = FBj (σj) + FPj (σj), (4)

where FB(σ) is the feature derived from backbone PTv3object, and FP(σ) represents the positional embedding derived from positional encoding module. The final contrastive loss is:

Lcon = ∥Fi − Fj∥, if C(i,j) = 1 ReLU(m − ∥Fi − Fj∥), if C(i,j) = 0

(5)

where C(i,j) is a binary function that indicates whether the pair (i,j) is from the same mask (1) or different masks (0), and m is a lower margin.

After training the scale-conditioned MLP, we can obtain the segmentation-aware features of 3D point cloud conditioned on a scale. By applying clustering algorithms such as HDBSCAN [26] to these grouping features, we can segment the 3D point cloud into different parts. The segmentation of 3D mesh can be easily derived from the segmentation of 3D point cloud using a voting algorithm.

- 3.3. Semantic Querying with MLLMs

After obtaining the part segmentation results of a 3D object, we query the semantics label of each part using the powerful Multimodal Large Langauge Models (MLLMs), as shown in Figure 2 (c). Utilizing the 3D-to-2D mapping, we can identify the corresponding 2D area of each 3D part in the multi-view renderings, which enables view-consistent highlighting of 3D parts in 2D renderings. By inputting these highlighted results into MLLMs, we can perform per-part semantic querying.

Specifically, we first select several canonical views of an object for rendering and part highlighting. This enriches the object details in the rendered images and facilitates the perception of MLLMs. Next, we choose a view with the largest rendered area for the part of interest and highlight the corresponding part area, ensuring comprehensive incorporation of this part’s details. Finally, we combine these images and feed them into MLLMs to obtain the semantic labels of the part.

- 4. Experiments

- 4.1. PartObjaverse-Tiny

Current part segmentation datasets typically include limited object categories and incomplete part annotations. This makes existing datasets unsuitable for evaluating the 3D segmentation performance of arbitrary objects and parts. Therefore, we annotate a subset of Objaverse, named PartObjaverse-Tiny, which consists of 200 shapes with fine-grained annotations. Following GObjaverse [35], we divide these 200 objects into 8 categories: Human-Shape (29), Animals (23), Daily-Used (25), Buildings&&Outdoor (25), Transportations (38), Plants (18), Food (8) and Electronics (34). Each major category includes multiple smaller object categories, for example, Transportation includes cars, motorcycles, airplanes, cannons, ships, among others. For each object, we meticulously segment and annotate it into fine-grained, semantically coherent parts. We present examples of PartObjaverse-Tiny dataset for semantic segmentation and instance segmentation in Figure 3.

- 4.2. Experiments Results

Multi-granularity 3D part segmentation. To demonstrate the generalization ability of our model, we use the model pretrained on Objaverse [9] to segment objects in GSO [11],

Scale Factor Mesh 0.0 0.5 1.0 1.5

Scale Factor Mesh 0.0 0.5 1.0 1.5

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

OmniObjectGSO

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

Vroid3D Generated

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

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Meshes

- Figure 4. Visualization of multi-granularity 3D part segmentation on GSO [11], OmniObject3D [45], Vroid [5] and 3D generated meshes.

OmniObject3D [45] and Vroid [5] datasets, as well as on 3D meshes generated from TripoAI [40] and Rodin [18]. Multigranularity segmentation results are shown in Figure 4.

Metrics. We utilize class-agnostic mean Intersection over Union (mIoU) to evaluate part segmentation results without semantics. Follow [43, 47], for each ground-truth part, we calculate the IoU with every predicted part, and assign the maximum IoU as the part IoU. Finally, we calculate the average of part IoU as class-agnostic mIoU. For semantic evaluation, we follow [25, 27], utilizing category mIoU and mean Average Precision (mAP) with a 50% IoU threshold

- as metrics for semantic and instance segmentation, respectively. We consider each object as a separate category, calculate the IoU/AP50 of each part separately, and compute the mIoU/mAP50 of this object.

Comparison with Existing Methods. For the PartObjaverse-Tiny dataset, we evaluate our method against PointCLIP [51], PointCLIPv2 [57], SATR [1] and PartSLIP [25] for zero-shot semantic segmentation, as shown in Table 1; against SAM3D [48] and PartSLIP for zero-shot class-agnostic part segmentation in Table 2; and against PartSLIP for instance segmentation in Table 3. For methods such as PartSlip and SATR, which utilize the GLIP detection model, the resulting segmentation often exhibits numerous blank areas. To address this, we employ the k-Nearest Neighbors (kNN) method, assigning the label of each face to that of the nearest face with predicted results. We present qualitative comparisons in Figure 5. Note that our pre-training dataset excludes the 200 3D objects in PartObjaverse-Tiny for fair comparison.

We further compare our method with PointCLIPv2, Part-

SLIP, ZeroPS [47] and PartDistill [41] on the PartNetE [25] dataset, as shown in Table 4. We assign the label ”others” for unlabeled areas in the PartNetE dataset.

#### 4.3. Ablation Analysis

We conduct ablation studies on SAMPart3D, and the quantitative comparison are shown in Table 5. To save pretraining time, we utilize a high-quality subset of Objaverse with 36k objects for ablation studies. We pre-train the original PTv3 backbone on this dataset. And we also pre-train our PTv3-object on this dataset with the same number of parameters as PTv3 for fair comparison.

Necessity of Pre-training. We ablate the proposed largescale pretraining on Objaverse, distilling the knowledge from powerful 2D foundational model DINOv2. Without pre-training, we randomly initialize the PTv3-object backbone and retain other contents of the pipeline. Without pretraining, the model lacks rich semantic part information, which hinders its ability to effectively encode 3D objects. This limitation not only impacts the segmentation results but also leads to unstable training.

PTv3-object v.s. PTv3. We modify the original PTv3 backbone to PTv3-object, enhancing the model’s encoding capabilities and ensuring the effective transmission of information at each point cloud. The second and fourth rows of Table 5 show comparison of PTv3-object and PTv3.

Significance of Long Skip Connection. When training the grouping field, we freeze the backbone and only train MLPs for efficient training. At this stage, without the long skip connection, the model can only accept 3D embedding inputs rich in part semantic information from our backbone,

GT Ours PartSLIP SATR

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Body, Head,

Arm, Foot, Hand, Leg, Tail

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Flag, Staircase Ground Floor, Windows, Roof, Second Floor,

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Body, Collar,

Head, Foot, Horn, Hand

- Figure 5. Qualitative comparison with PartSLIP [25] and SATR [1] in the semantic segmentation task on the PartObjaverse-Tiny dataset.

|Method<br><br>|Overall|Human-Shape Animals Daily-Used Buildings Transportations Plants Food Electronics|
|---|---|---|
|PointCLIP PointCLIPv2 SATR PartSLIP Ours|5.4 9.5 12.3 24.3 34.7<br><br>|3.5 4.5 6.5 5.5 3.6 8.8 12.3 5.6 6.8 10.0 11.3 8.4 6.5 15.8 15.3 9.9<br><br>15.6 16.5 12.7 7.9 9.4 17.2 14.5 9.7<br><br>39.3 41.1 19.0 13.0 17.1 31.7 17.3 18.5 44.4 51.6 33.6 20.7 26.6 42.6 35.1 31.1<br><br>|

Table 1. Zero-shot semantic segmentation on PartObjaverse-Tiny, reported in mIoU (%).

|Method<br><br>|Overall|Human-Shape Animals Daily-Used Buildings Transportations Plants Food Electronics<br><br>|
|---|---|---|
|PartSLIP SAM3D Ours|35.2 43.6 53.7<br><br>|45.0 50.1 34.4 22.5 26.3 44.6 33.4 32.0 47.2 45.0 43.1 38.6 39.4 51.1 46.8 43.8 54.4 59.0 52.1 46.2 50.3 60.7 59.8 54.5|

Table 2. Zero-shot class-agnostic part segmentation on PartObjaverse-Tiny, reported in mIoU (%).

|Method<br><br>|Overall<br><br>|Human-Shape Animals Daily-Used Buildings Transportations Plants Food Electronics|
|---|---|---|
|PartSLIP Ours<br><br>|16.3 30.2|23.0 34.1 13.1 6.7 10.4 28.9 7.2 10.2 36.9 43.7 29.0 19.0 21.4 38.5 39.4 27.7<br><br>|

Table 3. Zero-shot instance segmentation on PartObjaverse-Tiny, reported in mAP50 (%).

|Method|PointCLIPv2 PartSLIP ZeroPS PartDistill Ours<br><br>|
|---|---|
|Overall<br><br>|16.1 34.4 39.3 39.9 41.2|

Table 4. Zero-shot semantic segmentation on PartNetE [25], reported in mIoU (%).

but lacks direct input of 3D information. This leads to difficulty in training convergence, affecting final results.

DINOv2’s Feature v.s. SAM’s Feature for Pre-training. We utilize DINOv2 as our teacher model for pre-training, since DINOv2 contains extensive semantic information of the part and the whole object. We also attempt to directly utilize the features of SAM for pre-training, but find that the 3D backbone can not learn any knowledge that is helpful for part segmentation or perceiving objects. By pre-training on 3D large-scale data with more 3D spatial information in-

puts, our backbone’s encoding capability exceeds the direct fusion of DINOv2’s feature. We show qualitative comparison of our backbone’s encoding feature, DINOv2’s fusion feature and SAM’s fusion feature in Figure 7.

#### 4.4. Applications

Recent several works [7, 12, 33, 38, 49, 52] attempt to generate or edit the style or material of parts for 3D objects. However, due to the lack of accurate 3D segmentation methods, most of these methods are designed to use 2D segmentation methods or traditional simple 3D segmentation methods instead. This limitation may cause these methods to struggle with complex objects or generate inconsistencies in 3D editing. Our model is capable of segmenting any 3D object at various scales, thereby serving as a robust tool for

(b) Part Material Editing

(a) Part Segmentation Controlled by 2D Maks

[Figure 110]

[Figure 111]

[Figure 112]

3D Shape 2D Control Mask Controled 3D Part Segmentation

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

(c) Part Shape Editing & Animation (d) Click-Based Hierarchical Segmentation

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

scale = 0.5 1.0 1.5

- Figure 6. The resulting 3D part segmentation can directly support various applications, including part segmentation controlled by 2D masks, part material editing, part geometry editing, and click-based hierarchical segmentation.

|Method|Pre-train Data|Overall<br><br>|Human-Shape Animals Daily-Used Buildings Transportations Plants Food Electronics|
|---|---|---|---|
|w.o. pre. PTv3 w.o. skip Ours Ours|36k 36k 36k 200k<br><br>|43.4 46.7 48.7 50.5 53.7<br><br>|48.5 45.7 44.9 31.7 37.2 54.5 48.1 44.8<br><br>50.9 48.7 47.8 38.5 43.0 51.5 52.0 47.0<br>51.1 51.0 49.0 40.5 44.3 59.0 53.1 49.5<br><br><br>53.3 53.4 51.1 41.6 45.5 58.7 57.2 51.8<br>54.4 59.0 52.1 46.2 50.3 60.7 59.8 54.5<br>|

Table 5. Ablation study on PartObjaverse-Tiny, reported in mIoU (%).

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Mesh Ours DINOv2 SAM

[Figure 154]

[Figure 155]

- Figure 7. Visualization and qualitative comparison of the features encoded by our backbone, DINOv2, and SAM. Due to the utilization of 3D information from point clouds, our backbone can produce more accurate and fine-grained visual semantic features.

Part Material Editing. Leveraging precise segmentation results for 3D objects, our approach enables the customization and editing of materials and styles for individual components. This greatly improves the adaptability of 3D models to different design needs, enabling detailed personalization and optimization of textures. As shown in Figure 6 (b), we can edit the material of parts for realistic outcomes.

Part Shape Editing and Part Animation. The 3D object part segmentation results produced by our model can be directly applied to part shape editing tasks. By using these segmentation results, we can modify selected components within Blender, streamlining the workflow for 3D model refinement and customization. As shown in Figure 6 (c), we can use the segmentation results of the windmill and chimney to alter the shape of object parts. Similarly, we can adjust the angle of the windmill to showcase part animation.

3D generation and editing. At the same time, our method can serve as a pipeline for creating 3D part data, to create more 3D part assets for training 3D perception and generating models.

Click-based Hierarchical Segmentation. SAMPart3D is capable of accepting scale control to segment objects from coarse to fine. Therefore, similar to GraCo [54] in 2D, our model can accept a click in 3D space with a scale value to perform hierarchical segmentation of 3D objects. As shown in the Figure 6 (d), given a click at a location, the segmented area can be adjusted by controlling the scale.

Part Segmentation Controlled by 2D Masks. As illustrated in Figure 6 (a), our model seamlessly adapt to part segmentation guided by 2D segmentation masks. By utilizing 2D masks from a given viewpoint, we substitute SAM’s masks with these input masks and compute a scale factor for the visible points within that view for inference.

### 5. Conclusion and Discussions

We propose SAMPart3D, a zero-shot 3D part segmentation framework that can segment 3D objects into semantic parts

- at multiple granularities. Additionally, we introduce a new 3D part segmentation benchmark, PartObjaverse-Tiny, to address the shortcomings in diversity and complexity of existing annotated datasets. Experimental results demonstrate the effectiveness of SAMPart3D. References

- [1] Ahmed Abdelreheem, Ivan Skorokhodov, Maks Ovsjanikov, and Peter Wonka. Satr: Zero-shot semantic segmentation of 3d shapes. In ICCV, 2023. 2, 3, 6, 7
- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2
- [3] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv:1512.03012,

2015. 2, 3

- [4] Runnan Chen, Youquan Liu, Lingdong Kong, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, Yu Qiao, and Wenping Wang. Clip2scene: Towards label-efficient 3d scene understanding by clip. In CVPR, 2023. 2
- [5] Shuhong Chen, Kevin Zhang, Yichun Shi, Heng Wang, Yiheng Zhu, Guoxian Song, Sizhe An, Janus Kristjansson, Xiao Yang, and Matthias Zwicker. Panic-3d: Stylized singleview 3d reconstruction from portraits of anime characters. In CVPR, 2023. 6
- [6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv:2312.14238, 2023. 2
- [7] SeungJeh Chung, JooHyun Park, Hyewon Kan, and HyeongYeop Kang. 3dstyleglip: Part-tailored text-guided 3d neural stylization. arXiv:2404.02634, 2024. 2, 7
- [8] Pointcept Contributors. Pointcept: A codebase for point cloud perception research. https://github.com/ Pointcept/Pointcept, 2023. 4
- [9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2, 3, 4, 5
- [10] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. In NeurIPS, 2024. 2
- [11] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In ICRA. IEEE, 2022. 5, 6

- [12] Ye Fang, Zeyi Sun, Tong Wu, Jiaqi Wang, Ziwei Liu, Gordon Wetzstein, and Dahua Lin. Make-it-real: Unleashing large multimodal model’s ability for painting 3d objects with realistic materials. arXiv:2404.16829, 2024. 7
- [13] Stephanie Fu, Mark Hamilton, Laura Brandt, Axel Feldman, Zhoutong Zhang, and William T Freeman. Featup: A model-agnostic framework for features at any resolution. arXiv:2403.10516, 2024. 4, 11
- [14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2
- [15] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv:2402.11530,

2024. 2

- [16] Wenbo Hu, Hengshuang Zhao, Li Jiang, Jiaya Jia, and TienTsin Wong. Bidirectional projection network for cross dimension scene understanding. In CVPR, 2021. 4
- [17] Tianyu Huang, Bowen Dong, Yunhan Yang, Xiaoshui Huang, Rynson WH Lau, Wanli Ouyang, and Wangmeng Zuo. Clip2point: Transfer clip to point cloud classification with image-depth pre-training. In ICCV, 2023. 2
- [18] HYPERHUMAN. Rodin website. https : / / hyperhuman.deemos.com/rodin, 2024. 6
- [19] Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg, Matthew Tancik, and Angjoo Kanazawa. Garfield: Group anything with radiance fields. In CVPR, 2024. 2, 5
- [20] Hyunjin Kim and Minhyuk Sung. Partstad: 2d-to-3d part segmentation task adaptation. arXiv:2401.05906, 2024. 2
- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 2, 3, 5
- [22] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In CVPR, 2022. 2, 3
- [23] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. In NeurIPS, 2018. 2, 3
- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 2
- [25] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. In CVPR, 2023. 2, 3, 6, 7
- [26] Leland McInnes, John Healy, Steve Astels, et al. hdbscan: Hierarchical density based clustering. J. Open Source Softw.,

2017. 5, 11

- [27] Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A largescale benchmark for fine-grained and hierarchical part-level 3d object understanding. In CVPR, 2019. 2, 3, 6, 11
- [28] OpenAI. Gpt-4o website. https://openai.com/ index/hello-gpt-4o/, 2024. 11
- [29] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv:2304.07193, 2023. 2, 3, 4
- [30] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017. 2, 3
- [31] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017. 2, 3
- [32] Zhangyang Qi, Ye Fang, Zeyi Sun, Xiaoyang Wu, Tong Wu, Jiaqi Wang, Dahua Lin, and Hengshuang Zhao. Gpt4point: A unified framework for point-language understanding and generation. In CVPR, 2024. 2
- [33] Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. arXiv:2407.06191, 2024. 7
- [34] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. In NeurIPS, 2022. 2, 3
- [35] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to3d. arXiv:2311.16918, 2023. 5
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3
- [37] George Tang, William Zhao, Logan Ford, David Benhaim, and Paul Zhang. Segment any mesh: Zero-shot mesh part segmentation via lifting segment anything 2 to 3d. arXiv:2408.13679, 2024. 3
- [38] Konstantinos Tertikas, Despoina Paschalidou, Boxiao Pan, Jeong Joon Park, Mikaela Angelina Uy, Ioannis Emiris, Yannis Avrithis, and Leonidas Guibas. Generating part-aware editable 3d shapes without 3d supervision. In CVPR, 2023.

- 2, 7

[39] Anh Thai, Weiyao Wang, Hao Tang, Stefan Stojanov, Matt Feiszli, and James M Rehg. 3x2: 3d object part segmentation by 2d semantic correspondences. arXiv:2407.09648, 2024.

- 3

- [40] TripoAI. Tripoai website. https://www.tripo3d. ai/, 2024. 6
- [41] Ardian Umam, Cheng-Kun Yang, Min-Hung Chen, JenHui Chuang, and Yen-Yu Lin. Partdistill: 3d shape part segmentation by vision-language model distillation. arXiv:2312.04016, 2023. 2, 3, 4, 6
- [42] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv:2311.03079, 2023. 2
- [43] Xiaogang Wang, Xun Sun, Xinyu Cao, Kai Xu, and Bin Zhou. Learning fine-grained segmentation of 3d shapes without part labels. In CVPR, 2021. 6

- [44] Meng Wei, Xiaoyu Yue, Wenwei Zhang, Shu Kong, Xihui Liu, and Jiangmiao Pang. Ov-parts: Towards openvocabulary part segmentation. In NeurIPS, 2024. 2
- [45] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In CVPR,

2023. 2, 6

- [46] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler, faster, stronger. In CVPR, 2024. 4
- [47] Yuheng Xue, Nenglun Chen, Jun Liu, and Wenyun Sun. Zerops: High-quality cross-modal knowledge transfer for zeroshot 3d part segmentation. arXiv:2311.14262, 2023. 2, 3, 6
- [48] Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. Sam3d: Segment anything in 3d scenes. arXiv:2306.03908, 2023. 2, 6
- [49] Yunhan Yang, Yukun Huang, Xiaoyang Wu, Yuan-Chen Guo, Song-Hai Zhang, Hengshuang Zhao, Tong He, and Xihui Liu. Dreamcomposer: Controllable 3d object generation via multi-view conditions. In CVPR, 2024. 7
- [50] Yingda Yin, Yuzheng Liu, Yang Xiao, Daniel Cohen-Or, Jingwei Huang, and Baoquan Chen. Sai3d: Segment any instance in 3d scenes. In CVPR, 2024. 2
- [51] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In CVPR,

2022. 2, 6

- [52] Shangzhan Zhang, Sida Peng, Tao Xu, Yuanbo Yang, Tianrun Chen, Nan Xue, Yujun Shen, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou. Mapa: Text-driven photorealistic material painting for 3d shapes. arXiv:2404.17569, 2024. 7
- [53] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In ICCV, 2021. 2, 3
- [54] Yian Zhao, Kehan Li, Zesen Cheng, Pengchong Qiao, Xiawu Zheng, Rongrong Ji, Chang Liu, Li Yuan, and Jie Chen. Graco: Granularity-controllable interactive segmentation. In CVPR, 2024. 5, 8
- [55] Ziming Zhong, Yanyu Xu, Jing Li, Jiale Xu, Zhengxin Li, Chaohui Yu, and Shenghua Gao. Meshsegmenter: Zero-shot mesh semantic segmentation via texture synthesis. In ECCV. Springer, 2024. 3
- [56] Yuchen Zhou, Jiayuan Gu, Xuanlin Li, Minghua Liu, Yunhao Fang, and Hao Su. Partslip++: Enhancing low-shot 3d part segmentation via multi-view instance segmentation and maximum likelihood estimation. arXiv:2312.03015, 2023. 2, 3
- [57] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. Pointclip v2: Prompting clip and gpt for powerful 3d open-world learning. In ICCV, 2023. 2, 3, 6

### 6. Supplemental Material

#### 6.1. Implementation Details

In the large-scale pre-training stage, we train the PTv3object backbone on 200K high-quality objects from Objaverse with a batch size of 32, taking 7 days on eight A800 GPUs. Each object is rendered from 36 views, including 6 fixed views (+x, +y, +z, -x, -y, -z) and 30 random views. For each iteration, we pick 6 fixed views and 2 random views to cover most of the area for each object. We use the pre-trained DINOv2-ViTS14 model to encode the rendered images into visual features. These features are then upsampled by the FeatUp [13] model to obtain pixel-wise features as supervision.

In the sample-specific fine-tuning stage, we initially sample 15K point clouds on the mesh surface for the object of interest. We render 36 views for the object, consistent with the settings used during the pre-training stage. We input the 2D rendered images into SAM to obtain the 2D segmentation masks. For each iteration, we randomly pick 90 rendered images (with replacement) and sample 256 valid pixels from each image, obtaining 23,040 3D points mapped from pixels as inputs. The modules used for scaleconditioned grouping and long skip connection each employ 6-layer and 4-layer MLPs, with hidden dimensions set to 384. This stage requires 1 minute to generate masks using SAM, followed by 5 minutes of training MLPs.

After the fine-tuning stage, we can obtain the segmentation-aware features of 3D point cloud conditioned on a scale. We use the clustering algorithm HDBSCAN [26] for feature grouping, and utilize GPT-4o [28] for per-part semantic querying.

#### 6.2. Ablation Analysis of Segmentation Scale

Figure 4 in the paper presents a visualization of segmentation results across different scale factors. For the quantitative analysis of scale, we use five scale values [0.0, 0.5,

- 1.0, 1.5, 2.0] for each object, automatically selecting the result closest to the ground truth for evaluation. We conduct measurements using five values: [0.0, 0.5, 1.0, 1.5,
- 2.0] and compare them with our mixed scale results. Since the dataset is manually annotated, the number of suitable scales for different objects varies, leading to differences between individual and mixed scale results. The results of class-agnostic part segmentation results at different scales is shown in Table 6.

#### 6.3. Limitations

The training for the grouping field stage uses masks from SAM segmented at different scales. If some of these masks are inaccurate, it can affect the final results. Also, training the grouping field for each object is still slow. A better solution might be to utilize our method as a pipeline to annotate

a large number of part data, and then train a large 3D part segmentation model.

#### 6.4. Visualization on PartNetE Dataset

We visualize the segmentation results of our SAMPart3D on the PartNetE [27] dataset in Fig. 8. Compared to previous methods, our SAMPart3D can segment all fine-grained parts of 3D objects from PartNetE, even for parts that are not annotated in the dataset.

#### 6.5. More Qualitative Results of Our SAMPart3D

We show more visualization results of multi-granularity point clouds and meshes produced by our SAMPart3D in Figure 9.

#### 6.6. More PartObjaverse-Tiny Visualization

We demonstrate more visualization of dataset PartObjaverse-Tiny. We present more examples of semantic segmentation annotations in Figure 10, and instance segmentation annotations in Figure 11.

|Method<br><br>|Overall|Human-Shape Animals Daily-Used Buildings Transportations Plants Food Electronics|
|---|---|---|
|0.0 0.5 1.0 1.5 2.0 mixed-scale<br><br>|49.1 48.9 39.6 31.5 24.4 53.7|52.6 54.8 45.3 42.4 47.8 55.2 42.5 49.5 51.1 53.1 47.1 41.3 46.9 58.0 50.9 48.0 35.2 43.0 42.0 37.4 34.1 41.9 50.4 43.4 26.7 31.0 34.3 20.9 24.9 34.1 52.3 35.6 21.5 24.6 30.6 22.5 15.4 30.3 48.4 24.9 54.4 59.0 52.1 46.2 50.3 60.7 59.8 54.5<br><br>|

Table 6. Zero-shot class-agnostic part segmentation on PartObjaverse-Tiny across different scale values, reported in mIoU (%).

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

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Figure 8. Visualization of segmentation results on PartNetE dataset.

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

[Figure 211]

[Figure 212]

[Figure 213]

###### Figure 9. Visualization of multi-granularity segmentation of point clouds and meshes.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Base, Cab, Chassis, Tire, Window, Secondary Tires, Vitta

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Back Cover, Button, Enclosure, Inside, Knob, Screen, Watchband

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Body, Electrical Box, Front, Exhaust Pipe, Pipeline, Top, Wire

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Big Stab, Body, Flower, Flowerpot, Mud, Stab

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

House Beams, Support, Plank, Roof, Rope, Swivel Rod, Water Well

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Leaf, Earth, Flower, Flowerpot, Grass, Petals, Stalks, Stones

- Figure 10. Visualization of PartObjaverse-Tiny with part-level annotations with semantic labels for segmentation segmentation.

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Frame, Left Glass, Left Rims, Right Glass, Right Rims

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Brackets, Door, Foundation, Front Column, Left Lantern, Left Railing, Rear Column, Right Lantern, Right Railing, Roof, Support Frame, Wall, Windows, Wing Angles

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Body, Button, Glass, Knob, Lens, Ornament, Screen

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Head, Left Arm, Left Ear, Left Foot, Left Hand, Left Leg, Lower Body, Others, Right Arm, Right Ear, Right Foot, Right Hand, Right Leg, Upper Body

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Beard, Body, Book, Cloak, Hair, Hat, Head, Left Arm, Left Foot, Left Hand, Right Arm, Right Foot, Right Hand, Sceptre

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Body, Hide Buckle, Left Buckle, Left Front Leg, Left Hand, Left Hide Leg, Left Machine Gun, Left Tube, Legs Low Phong, Right Buckle, Right Front Leg, Right Hand, Right Hide Leg, Right Machine Gun, Right Tube, Top Body, Top Trim

- Figure 11. Visualization of PartObjaverse-Tiny with part-level annotations with semantic labels for instance segmentation.

