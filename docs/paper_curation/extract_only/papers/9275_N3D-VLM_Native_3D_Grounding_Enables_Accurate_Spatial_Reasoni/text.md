# arXiv:2512.16561v1[cs.CV]18Dec2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

## N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models

Yuxin Wang1,2* Lei Ke2 Boqiang Zhang2 Tianyuan Qu2,3 Hanxun Yu2,4 Zhenpeng Huang2,5 Meng Yu2 Dan Xu1† Dong Yu2

1HKUST 2Tencent AI Lab 3CUHK 4ZJU 5NJU Project Page: https://n3d-vlm.github.io

|[Figure 7]<br><br>lemon: (0.6, -0.31, 0.85)<br><br>dishwasher: (0.98, -0.54, 5.28)<br><br>blender: (-1.28, -0.11,|
|---|

[Figure 8]

|[Figure 9]<br><br>blender: (-1.28, -0.11,<br><br>lemon: (0.6, -0.31, 0.85)<br><br>dishwasher: (0.98, -0.54, 5.28)|
|---|

2.12)

r: 2.12)

[Figure 10]

###### Spatial Reasoning & Answer:

###### Question: Which one of dishwasher and blender is closer to the reference object lemon?

Native 3D Grounding in [x, y, z, w, h, l]: lemon: [0.6, -0.31, 0.85, 0.11, 0.20, 0.10] dishwasher: [0.98, -0.54, 5.28, 0.66, 0.88, 0.27] blender: [-1.28, -0.11, 2.12, 0.34, 0.44, 0.21]

👤 🤖 🤖

- 1. Distance (lemon, dishwasher) = 4.4408.
- 2. Distance (lemon, blender) = 2.3565.
- 3. Since 2.3565 < 4.4408, blender is closer

##### Unified 3D Grounding & to lemon than dishwasher Spatial Reasoning

|[Figure 11]<br><br>0 1<br><br>2<br><br>3<br><br>6<br><br>5<br><br>4<br><br>7<br><br>8 9<br><br>10<br><br>[Figure 12]<br><br>|
|---|

|[Figure 13]<br><br>0 1<br><br>2<br><br>3<br><br>6<br><br>5<br><br>4<br><br><br>7<br><br>8 9<br><br>10<br><br>[Figure 14]|
|---|

[Figure 15]

[Figure 16]

Question: Which goose is the furthest from the camera?

Spatial Reasoning & Answer:

###### Native 3D Grounding in [x, y, z, w, h, l]:

👤 🤖 🤖

- 1. The z coordinates is [20.55, 20.92, 28.42, 21.58, ... 22.81].
- 2. Since goose_2 has the biggest z coordinate, goose_2 is the furthest.

- goose_0: [3.69, -0.7, 20.55, 0.93, 0.38, 0.77]
- goose_1: [2.28, -0.55, 20.92, 0.96, 0.41, 0.81]

... goose_10: [-3.88, -0.02, 22.81, 1.87, 0.89, 1.41]

Figure 1. Our unified vision-language model N3D-VLM performs native 3D grounding and subsequent spatial reasoning and answering. Given an RGB image and the corresponding text question, the model is capable of predicting 3D bounding boxes for specified objects and explicitly reasoning about spatial relations in 3D space.

1

###### Abstract

While current multimodal models can answer questions based on 2D images, they lack intrinsic 3D object perception, limiting their ability to comprehend spatial relationships and depth cues in 3D scenes. In this work, we propose N3D-VLM, a novel unified framework that seamlessly integrates native 3D object perception with 3Daware visual reasoning, enabling both precise 3D grounding and interpretable spatial understanding. Unlike conventional end-to-end models that directly predict answers from RGB/RGB-D inputs, our approach equips the model with native 3D object perception capabilities, enabling it to directly localize objects in 3D space based on textual descriptions. Building upon accurate 3D object localization, the model further performs explicit reasoning in 3D, achieving more interpretable and structured spatial understanding. To support robust training for these capabilities, we develop a scalable data construction pipeline that leverages depth estimation to lift large-scale 2D annotations into 3D space, significantly increasing the diversity and coverage for 3D object grounding data, yielding over six times larger than the largest existing single-image 3D detection dataset. Moreover, the pipeline generates spatial questionanswering datasets that target chain-of-thought (CoT) reasoning in 3D, facilitating joint training for both 3D object localization and 3D spatial reasoning. Experimental results demonstrate that our unified framework not only achieves state-of-the-art performance on 3D grounding tasks, but also consistently surpasses existing methods in 3D spatial reasoning in vision-language model.

###### 1. Introduction

Recent vision-language models (VLMs) [2, 11, 19, 28] have expanded beyond text-only understanding to handle diverse multimodal tasks such as image and video analysis, OCR, and visual reasoning. However, real-world applications often demand a deeper grasp of 3D structure and spatial relationships, which current VLMs largely lack. Effective 3D spatial reasoning requires accurate object-level perception in 3D space; without it, models struggle to infer spatial configurations or reason about physical environments. Advancing toward truly multimodal intelligence therefore requires moving beyond 2D language-centric perception toward robust 3D spatial ability to perceive, ground, and reason about the 3D world from visual inputs.

Specialized VLMs enhance 3D spatial understanding capability through diverse input modalities and architectural designs. Some models integrate external perception models [24, 26] to obtain auxiliary object information, such as

*Work done during an internship at Tencent AI Lab. †Corresponding author.

- 2D/3D bounding boxes or segmentation masks [8, 10, 34]. Others assume that 3D object bounding boxes or spatial layouts are provided in advance [15, 23, 39]. Alternatively, recent approaches have explored using VLMs to directly localize objects in point clouds [1, 22]. However, these methods typically focus on object detection in constrained scenes with limited object categories, and do not support explicit spatial reasoning. Although these approaches have advanced specific aspects of 3D spatial understanding, they either depend heavily on external modules or predefined spatial information, or remain confined to narrow perception tasks, which makes it challenging to generalize and integrate them into unified vision-language systems.

Based on these observations, we argue that 3D spatial understanding could be decomposed into two core abilities:

- 3D object localization and subsequent 3D spatial reasoning. This perspective motivates our design, where explicit 3D object perception serves as a critical foundation for spatial reasoning. By first detecting objects in 3D space, models can reason more effectively over structured representations such as 3D bounding boxes, enhancing both the accuracy and interpretability of the reasoning process.

To this end, we propose N3D-VLM, a unified visionlanguage model that integrates 3D detection, grounding, and CoT reasoning. The model is equipped with inherent and generalizable 3D object perception, allowing it to accurately localize objects and capture depth cues in physical space. Building on these 3D perception results, N3D-VLM performs spatial reasoning tasks—such as computing interobject distances based on 3D coordinates or inferring relative sizes from 3D bounding box dimensions, as shown in Fig. 1. Recent works like [14, 21] utilized 3D coordinates or

- 2D grounding data to aid spatial understanding. However, our approach fundamentally differs: our model explicitly predicts comprehensive 3D bounding boxes, enabling more generalizable and interpretable spatial reasoning.

A key challenge in developing general 3D object perception lies in the scarcity of large-scale 3D training data. Existing datasets [4, 37], typically captured from indoor [3, 25] or autonomous driving scenes [5, 9], suffer from limited diversity, small scale, and narrow category coverage. In contrast, large-scale 2D detection datasets [13, 16, 27] provide richer scene variety and class diversity.

To overcome 3D data scarcity, we leverage a depth and camera estimation model [32] to lift 2D annotations into

- 3D space, generating abundant 3D detection and grounding data. We further construct spatial QA datasets to supervise CoT-based spatial reasoning. Another challenge is ensuring consistent real-world scale and camera geometry in 3D outputs; we address this by using the same depth model during data construction and introducing a depth-aware positional encoding to inject explicit depth cues. Jointly training on these heterogeneous sources enables our model to achieve

robust 3D localization and effective CoT spatial reasoning grounded in explicit 3D perception.

In summary, we present a comprehensive solution for 3D grounded reasoning by introducing a model with native 3D obejct localization capabilities, marking a significant advancement in 3D visual understanding. Our main contributions are threefold:

- • We propose a unified model that takes RGB-D input and features native 3D object detection/grounding abilities, enabling spatial reasoning based on detection outcomes.
- • We design a data construction pipeline that leverages depth estimation to convert large-scale 2D annotations into 3D space, addressing the shortage of diverse and high-volume 3D object perception training data.
- • We establish a spatial reasoning benchmark with explicit reasoning process, covering both single-object and multiobject scenarios across a wide variety of question types.

Our model consistently surpasses existing methods in both object localization and spatial reasoning metrics, demonstrating state-of-the-art effectiveness and generalization. We will release code, checkpoints, and datasets upon acceptance.

###### 2. Related Work

###### 2.1. VLM for 3D Spatial Understanding

Recently, an increasing number of approaches aim to extend general-purpose VLMs [2, 11, 19, 28] with 3D spatial understanding, by leveraging 3D point clouds, video [7, 23, 31, 35, 39] or RGB/RGB-D inputs [6, 8, 10, 40]. For example, GPT4Scene [23] generates object-marked images from point clouds, and uses them along BEV (bird’s-eye view) images, to perform 3D captioning and question answering in reconstructed scenes. Think-in-Space [35] supports a broader range of spatial questions, including multiplechoice route planning and relative distance reasoning based on video inputs. However, these methods often rely on additional 3D information or object-level annotations, and are typically constrained to limited indoor environments. Other approaches focus on inferring spatial relations from 2D inputs. SpatialVLM [6] enables spatial question answering (e.g., left, right, front, behind) from a single RGB image. SpatialRGPT further extends this capability to region-level reasoning, allowing spatial understanding between specified image regions. Nevertheless, these methods generally lack explicit 3D spatial understanding, which either rely on black-box end-to-end reasoning to produce answers, or requires external modules for region localization.

###### 2.2. VLM for 3D Object Localization

Other 3D visual grounding approaches can localize object positions in 3D space based on point cloud or video input [15, 18, 20, 34]. For example, VLM-Grounder [34]

performs 2D segmentation on selected video frames, followed by multi-view matching and ensemble projection to localize objects in 3D. SeeGround [15] supports 3D visual grounding but assumes known object positions after selecting a specific view. These methods typically rely on external segmentation tools or require additional objectlevel information. In contrast, our approach explicitly localizes 3D objects and outputs 3D bounding boxes. SpatialLM [22] also performs 3D grounding from point clouds and produces 3D bounding boxes. However, it is limited to indoor scenes with a small set of object categories and lacks spatial reasoning capabilities beyond grounding. SpatialReasoner [21] enhances spatial understanding by estimating object position and orientation. However, it operates in constrained scenarios, does not capture object size, and demonstrates limited generalization. By contrast, our method generalizes well across diverse scenes and outputs full 3D bounding boxes with complete object dimensions and positions, supporting both 3D detection, 3D grounding and downstream spatial reasoning.

###### 3. The Proposed Framework: N3D-VLM

Given an RGB image and its corresponding monocular depth map, represented as (I,D), we aim to train a visionlanguage model (VLM) that takes RGB-D input and outputs 3D object detection and grounding results in the form of explicit 3D bounding box in the camera coordinate system. Furthermore, the model can leverage these grounded 3D objects to perform spatial reasoning and answer spatial understanding questions. The depth map D can be easily obtained using monocular depth estimation models [32, 36].

Our framework consists of two main components: 3D data construction and model design with training and evaluation, as illustrated in Fig. 2 and Fig. 3, respectively. As shown in Fig. 2, we begin by lifting 2D annotations from existing large-scale, category-rich 2D detection and grounding datasets into 3D domain, forming a diverse repository of 3D detection annotations. Based on this 3D repository, we further construct datasets for 3D detection, 3D grounding, and 3D spatial reasoning QA, which are used for both training and evaluation. Fig. 3a depicts the architecture of our model, which accepts RGB-D inputs. For grounding tasks, the model predicts structured language descriptions that correspond explicitly to the 3D bounding boxes of objects. For spatial reasoning tasks, the model performs spatial reasoning explicitly over grounded 3D objects to answer spatial understanding questions.

###### 3.1. 3D Data Generation

3D Detection Annotation Repository. As shown in Fig. 2, we construct a large-scale 3D detection repository by lifting existing 2D detection datasets into 3D. Starting from images annotated with 2D bounding boxes, we first obtain

[Figure 17]

[Figure 18]

###### 2D Detection Annotation

###### 3D Detection QA:

3D Spatial Reasoning QA: (e.g., Direction)

2

Lifting Structured

1

Template + LLM Rephrasing

2

###### Question: Detect all 3D bounding boxes in the image. Answer:

Language

Question: If you are at 5_stroller, where will you find 1_boy in clock direction? Answer:

[Figure 19]

- bbox(id=0, class=stroller, ...),

- bbox(id=1, class=boy, ...),

- bbox(id=2, class=ice cream, ...),

5

<think> The 3D center coordinates of 5_stroller and 1_boy is [-2.18, 1.45, 6.11] and [0.29, -0.12, 1.72].

0

- Step 1: Compute the vector from 5_stroller to 1_boy:

[0.29- -2.18, 1.72 - 6.11] = [2.47, -4.39].

- Step 2: Compute the angle to the positive x-axis:

atan2(2.47, -4.39) = 2.63 rad, which is 150.64 degrees.

- Step 3: Convert the angle to clock position, clock

..., bbox(id=5, class=stroller, ...)

2

1

3 4

2

Structured Language

###### 3D Grounding QA:

position = 12 - (150.64 // 30) = 7.

###### Question: Locate the 3D bounding boxes of all the boys. Answer:

</think> Finally, 1_boy is roughly at 7 o'clock of 5_stroller.

3D Detection Annotation Repository

- bbox(id=0, class=boy, ...),
- bbox(id=1, class=boy, ...)

#### Data Generation for 3D Localization & Spatial Reasoning

- Figure 2. Illustration of our data construction pipeline. We first lift annotations from existing 2D detection datasets with diverse object categories into 3D space, resulting in a large-scale and category-rich 3D detection annotation repository. Based on this repository, we generate data for 3D detection, 3D grounding, and 3D spatial reasoning QA tasks.

(b) 3D Detection Dataset Comparison

[Figure 20]

3D bbox annotation

Structured Language

Structured Language.

scale_x

Generation of 3D Localization Data. After constructing the 3D detection annotation repo generate corresponding 3D detection a pairs to facilitate training VLMs for 3D object localization. Specifically, we represent each 3D bounding box using a structured language format, shown in Fig. 3c. Similar to [1, 22], each box is encoded as:

object segmentation masks [24] and estimate monocular depth for each image [32]. The resulting depth maps are back-projected to generate point clouds in the camera space. By combining these point clouds with category and segmentation labels, we derive 3D bounding boxes for each object. During this lifting process, we apply rule-based filters to automatically remove ou invalid depth values and discard implausible are excessively large or small. Leveraging and category richness of 2D datasets, our corpus inherits these advantages, and pr foundation for boosting VLMs’ 3D localization capabilities. As illustrated in Fig. 2b, our dataset contains 2.78 million samples, which substantially more than existing single-image 3D detection Omni3D (∼234K) [4] and DetAny3D (∼ dataset is constructed from three major sources: COCO [17], OpenImages [13], and Objects365 [27]. COCO and Objects365 offer high-quality

[Figure 21]

Structured Language

|Turn to Structured Language.<br><br>repository, we further and grounding QA|
|---|

- scale_y
- scale_z

center

Grounding QA:

SpatialReasoningQA: 3D Detection Dataset Cruation

Template + LLM Rephrasing

3D bbox annotation

[Figure 22]

[Figure 23]

|Model<br><br>outlier points from boxes that the scale constructed 3D provides a strong|
|---|

| |
|---|

|Turn to Structured,s ,sLanguage.)|
|---|

(u, v)

Template + LLM Rephrasing

image

2D projection

bbox(id,class,u,v,z,sx y z

(c) 3D Object Representation

where id and class denotes the object identifier and category. (u,v) is the 2D projection of the 3D center on the image plane, and z is its depth. s x, s y, and s z represent the box dimensions along the three spatial axes. Note that, given known camera intrinsics, (u,v,z) and (x,y,z) are interconvertible via a deterministic projection. For 3D detection QA, answers are directly derived from the structured annotations. For 3D grounding QA, we adopt two strategies. First, we select objects uniquely identifiable by category (i.e., categories that appear only once) and convert them into grounding questions. This extends to cases where multiple distinct categories are each uniquely identifiable. When multiple instances of the same category appear (see Fig. 2), we formulate questions like “Locate all the boys in the image.” For objects not easily described by category, we

(a) 3D Detection Annotation from 2D

point cloud

|Reasoning Template<br><br>datasets+ LLM rephrasingsuch as 450K) [37]. Our|
|---|

- 2D annotations across dozens to hundreds of object categories. For OpenImages, where the average number of boxes per image is low, we apply RAM [38] to re-detect objects, producing open-vocabulary 2D detections that are subsequently lifted into 3D.

[Figure 24]

[Figure 25]

[Figure 26]

- Question 1 (3D Localization): Detect the left boy and the stroller on the right farther away.
- Question 2 (3D Spatial Reasoning): If you're at the rightmost stroller, where is the left boy in clock direction? Given 3D bbox of both objects: bbox(...), bbox(...).

Projection

[Figure 27]

Image Point Cloud

🔥

###### Encoder PE

###### Large Language Model

🔥

###### Answer 1 (3D Localization):

[Figure 28]

Projection

- bbox(id=0, class=boy, u, v, z, w, h, l)
- bbox(id=1, class=stroller, u, v, z, w, h, l)

(b) Quantitative Comparison

scale_x

id: int class: str

[Figure 29]

[Figure 30]

Answer 2 (3D Spatial Reasoning): <think> ... </think> The boy is at ~7 o'clock of the stroller.

[Figure 31]

- scale_y
- scale_z 2D projection

- position_u: int
- position_v: int position_z: int

center (u, v) 3D bbox

| |
|---|

- scale_x: int
- scale_y: int
- scale_z: int }

(c) Structured Language for 3D Bounding Box

(a) Model Architecture Design

- Figure 3. Illustration of our model design and quantitative comparison. (a) Overview of our model architecture and the cascaded spatial reasoning process. (b) Quantitative comparison showing that our model outperforms existing methods. (c) Definition of structured language representation for 3D bounding boxes.

either use referring expressions from existing datasets [12], or refer to them by rendering their 2D bounding boxes on the image. This process yields a diverse set of 3D detection and grounding QA pairs, supporting effective training of VLMs for 3D object localization.

from Fig. 2, the question asks for the clock direction of the boy relative to the stroller. The reasoning process is first generated based on a predefined template: given the 3D bounding boxes of both objects, we compute the vector from the stroller to the boy on the xz-plane, calculate its angle with respect to the positive x-axis, and convert the angle into a clock position. This explicit geometric reasoning is embedded in the answer chain, enabling more interpretable and intuitive explanations. More question templates and task definitions are provided in the Appendix.

Generation of 3D Spatial Reasoning Data. We further construct 3D spatial reasoning questions and explicit reasoning answers based on the 3D detection annotation repository. As shown in Fig. 2, we randomly sample objects from an image and apply predefined question templates, e.g., asking for the clock direction between two objects. For each question, we generate a deterministic reasoning process and answer based on the 3D bounding boxes, which is then rephrased using an LLM to improve naturalness. Following [8], we design both open-ended and numerically grounded questions. Specifically, we adopt question types similar to SpatialRGPT, including comparisons of relative scale (e.g., wider/narrower, taller/shorter), spatial relations (e.g., above/below, left/right, front/behind), absolute distances between objects, clock directions, and object dimensions (e.g., height, width). In addition, we also extend to multi-object reasoning involving three or more objects. For example, we ask about relative distances among three objects or spatial configurations among a dozen objects, as illustrated in Fig. 1. All answers include deterministic numerical computations and interpretable reasoning steps, grounded in 3D object bounding boxes. In the example

###### 3.2. Model Architecture

3D-aware Visual Encoding. To ensure that the 3D bounding boxes predicted by our VLM are in real-world scale and aligned with an existing coordinate system, we adopt the depth estimation model [32], which predicts both depth maps and camera intrinsics. We then use the predicted depth as an additional input to our model. This guarantees that all predicted 3D bounding boxes are expressed in metric scale and aligned with the coordinate system defined by [32].

As illustrated in Fig. 3, we design a 3D-aware visual encoding pipeline to incorporate geometric information into the vision-language model. Given an RGB image I ∈ RH×W×3, its corresponding depth map D ∈ RH×W, and camera intrinsics intr ∈ R3×3. We first back-project each pixel to a 3D point in the camera coordinate system: P ∈

RH×W×3:

 

 , (1)

uj vi 1

Pij = Dij · intr−1 ·

where (uj,vi) are the pixel coordinates. This yields a dense point cloud P ∈ RH×W×3, which is then downsampled to Pˆ ∈ Rh×w×3 to match the spatial resolution of the image features Fimg ∈ Rh×w×c extracted by the vision encoder. To inject spatial information, each 3D coordinate (x,y,z) ∈ Pˆ is encoded using sinusoidal positional encoding. For each axis α ∈ {x,y,z}, we compute:

α 100002i/c

, PE(α,2i + 1) = cos

PE(α,2i) = sin

α 100002i/c

, (2)

for i = 0,1,..., 2c − 1. The final coordinate embedding is obtained by summing the encodings across all three axes:

ecoord =

PE(k). (3)

k∈{x,y,z}

We then add the coordinate embedding to the image features to obtain the fused representation as follows:

F˜img = Fimg + ecoord. (4)

The fused feature map F˜img, which encodes both visual and spatial cues, is then passed to the language model along with the prompt tokens for autoregressive prediction.

Training Strategy and Inference Pipeline. Our model is based on Qwen2.5-VL [2] and trained in two stages. In the first stage, we train the model for 3D object localization using the dataset described in Sec. 3.1. In the second stage, we train the model for grounding-based 3D spatial reasoning using a mixture of 3D spatial reasoning data and a subset of the localization data. All parameters of the encoder and the language model are learnable throughout both stages.

At inference time, our model supports two usage modes. The first, illustrated in Fig. 1, allows users to ask spatialrelated questions directly. The model automatically decomposes the query into two steps: 3D object grounding, followed by spatial reasoning based on the grounding results. In the second mode, shown in Fig. 3, users can explicitly request 3D grounding first, then issue follow-up questions according to the grounding output. In both cases, reasoning is performed conditionally on the grounding results.

###### 3.3. N3D-Bench

Given the narrow range of scenes and object categories, current benchmarks fall short in representing the complexity of real-world 3D spatial understanding. Based on our

- 3D spatial data generation pipeline, we manually curated 1,200 open-ended and 800 numerically questions with CoT

Table 1. Comparison between our proposed N3D-Bench and SpatialRGPT-Bench [8].

Comparision SpatialRGPT-Bench [8] N3D-Bench

# Questions 1406 2000 # Object Categories 88 264 Objects / Question {1,2} {1,2,3,>3} View Change ✗ ✓ CoT Reasoning ✗ ✓

reasoning to construct N3D-Bench. As shown in Tab. 1, our benchmark significantly extends SpatialRGPT-Bench 1 in both object category coverage and question complexity. N3D-Bench includes references to 264 object categories, which is three times more than SpatialRGPT-Bench. It also introduces questions involving spatial relations among three or more objects, as well as viewpoint-shifted reasoning (e.g., “from the opposite view”), which are not considered in SpatialRGPT-Bench. Additionally, we introduce explicit CoT reasoning grounded in 3D object bounding box, offering interpretable intermediate steps beyond direct answers. These enhancements make N3D-Bench a more comprehensive and challenging benchmark for evaluating 3D spatial reasoning over both single and multiple objects.

###### 4. Experiments

###### 4.1. Experimental Setup

Dataset. As described in Sec. 3.2, we train our model on 3D object localization and spatial reasoning data derived from OpenImages, Objects365, and COCO. For 3D spatial reasoning, we evaluate on three benchmarks: our proposed N3D-Bench, SpatialRGPT-Bench [8] (1,404 openended and numerical questions), and CV-Bench-3D [30] (1,200 multiple-choice questions). For 3D grounding, we evaluate on the RefCOCO series [12] , along with an additional test set constructed from Objects365.

Metrics. For spatial reasoning, we report the accuracy. For open-ended questions, we use GPT-4o [11] in an LLM-as-ajudge setup to access correctness. For numerical questions, we extract predicted values via string matching and apply a ±25% tolerance, following [8]. For all methods evaluated on SpatialRGPT-Bench, except SpatialRGPT itself, object references are provided via bounding boxes drawn on the image. For 3D grounding, to ensure fair comparison across varying depth scales and box types, we compute projected IoU and projected center offset by projecting predicted 3D bounding box onto the image plane and comparing it with ground-truth 2D boxes. We also try to align the predicted boxes to the depth of the ground-truth boxes and compute 3D IoU and 3D center offset. To mitigate the alignment noise, 3D metrics are reported on a sampled subset.

- Table 2. Quantitative comparison on spatial reasoning benchmarks. Our N3D-VLM consistently outperforms baseline methods across all three spatial reasoning benchmarks, achieving state-of-the-art performance on open-ended, numerical, and multiple-choice questions.

Category Method

N3D-Bench SpatialRGPT-Bench [8] CV-Bench-3D [30] Open-ended Numerical Open-ended Numerical Multi-choice

Closed-source

GPT-4o [11] 63.5 27.8 76.3 55.8 72.4 Gemini-2.5-Flash [28] 64.2 36.7 82.4 42.2 86.0

Open-source

Qwen2.5-VL-7B [2] 55.0 22.5 74.4 38.2 75.8 Qwen3-VL-8B [29] 66.3 36.3 89.2 40.7 91.3 SpatialLadder-3B [14] 48.9 18.1 55.9 26.5 74.9 SpatialReasoner-7B [21] 54.8 27.4 63.2 33.7 80.3 SpatialRGPT-VILA-1.5-8B [8] 63.1 50.4 92.7 62.7 63.3

Ours

N3D-VLM-3B 77.0 90.1 80.5 73.3 96.3 N3D-VLM-7B 89.7 92.1 95.7 78.0 93.3

- Table 3. Quantitative comparison of 3D grounding using IoU and center offset of projected 3D bounding boxes. Our N3D-VLM achieves the best performance on both projected IoU and offset, demonstrating our superior accuracy in localizing objects in 3D space.

Method

Refcoco [12] Refcoco+ [12] Refcocog [12] Objects365 [27] Proj. IoU ↑ Proj. Offset ↓ Proj. IoU ↑ Proj. Offset ↓ Proj. IoU ↑ Proj. Offset ↓ Proj. IoU ↑ Proj. Offset ↓

Qwen3-VL-8B [29] 0.37 0.16 0.34 0.26 0.36 0.14 0.28 0.12 Qwen3-VL-30B-A3B [29] 0.38 0.14 0.36 0.16 0.38 0.13 0.28 0.13

Our N3D-VLM-7B 0.59 0.06 0.53 0.10 0.54 0.08 0.61 0.05

- Table 4. Quantitative comparison of 3D bounding boxes. Our N3D-VLM achieves the best results across 3D IoU and 3D offset, demonstrating superior performance in 3D object localization.

Table 5. Ablation studies on model design. Variants (3) and (4), which adopt our model design shown in Fig. 3(a), achieve the best 3D detection performance, when compared with other variants.

Refcoco/+/g [12] 3D IoU ↑ 3D Offset ↓

Method F1@0.25 ↑ P@0.25 ↑ R@0.25 ↑

Method

- (0) SpatialLM [22]-340K 2.2 2.3 2.4

- (1) 3B-340K-nodepth 9.4 10.9 9.4
- (2) 3B-340K-cameraxy 10.8 11.6 10.7

- (3) 3B-340K-imageuv 12.8 13.6 12.9
- (4) 3B-1.7M-imageuv 22.9 24.3 22.9

Qwen3-VL-8B [29] 0.20 1.88 Qwen3-VL-30B-A3B [29] 0.27 1.86

Our N3D-VLM-7B 0.48 0.36

###### 4.2. Main Results

ing performance between our model and Qwen3-VL [29]. As shown in Tab. 3 and 4, our method consistently outperforms Qwen3-VL in both projected 2D metrics and aligned 3D bounding box evaluation, demonstrating stronger 3D grounding capabilities. Fig. 4 further shows that our model accurately localize objects in 3D space, producing precise 3D bounding boxes across diverse scenarios, including indoor and outdoor environments. We provide more grounding results comparison and analysis in the Appendix.

3D Spatial QA. Tab. 2 compares the accuracy of our method with baseline approaches across three benchmarks. Our model achieves the highest accuracy on all benchmarks, demonstrating that our native 3D grounding significantly improves performance on 3D spatial reasoning tasks. Compared to the base model Qwen2.5-VL-7B, our 7B model shows substantial gains, especially on numerical questions, which indicates that grounding-based reasoning enhances the model’s quantitative understanding beyond standard QA capabilities. While Qwen3-VL improves over Qwen2.5-VL in spatial reasoning, its numerical reasoning remains limited, with only 36.3% and 40.7% accuracy. In contrast, our model achieves 92.1% and 78.0% on the same tasks, demonstrating significantly stronger numerical reasoning. Although SpatialRGPT performs well on numerical questions in N3D-Bench, achieving 50.4% which outperforms Qwen3-VL, but still far below our model’s 92.1%.

###### 4.3. Ablation Study on Model Design

We conduct ablation studies of the 3D detection task on Objects365 dataset to evaluate the effectiveness of our model design. Tab. 5 reports the 3D IoU results on a validation set of 5,565 images covering 341 object classes. Row (0) shows that the SpatialLM [22] architecture, which combines a point cloud encoder [33] and an LLM [2], performs poorly on our more diverse point cloud data, indicating limited generalization. Variants (1) and (2) are based on our

3D Object Grounding. Tab. 3, Tab. 4 and Fig. 4 present the quantitative and qualitative comparisons of 3D ground-

[Figure 32]

[Figure 33]

Locate the 3D bbox of all the towel.

Locate the 3D bbox of all the guitar.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

###### (a) Qwen3-VL-8B (b) Ours (a) Qwen3-VL-8B (b) Ours

Locate the 3D bbox of "the cow that is the smallest" and "the cow on the far right of the herd".

[Figure 38]

###### Locate the 3D bbox of "purple umbrella" and "child in white shirt".

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

(a) Qwen3-VL-8B (b) Ours (a) Qwen3-VL-8B (b) Ours

- Figure 4. Qualitative comparison of 3D grounding capability with Qwen3-VL-8B [29]. Compared to Qwen3-VL-8B, our N3DVLM generates 3D bounding boxes that more accurately close to the ground truth, reflecting stronger 3D understanding and localization precision. In the visualization, green boxes represent ground truth 3D bounding boxes, and red boxes indicate model’s predictions.

model architecture, with (1) removing depth input and (2) using depth but directly predicting 3D coordinates in the camera coordinate system. Variants (3) and (4) follow our full design (Fig. 3a), which takes depth input and predicts

- 2D pixel-space coordinates shown in Fig. 3c. Comparing

(1) and (3), we can observe that incorporating depth input improves 3D detection accuracy, increasing the F1 score from 9.4 to 12.8. Comparing (2) and (3), we can find that predicting the center coordinates in pixel space outperforms camera-space prediction, likely because the base model is pretrained with 2D perception data, which is naturally aligned with the image-space representations. These results suggest that using depth input and predicting pixelspace coordinates, is more effective than excluding depth or directly predicting in the camera coordinate system. Finally, scaling the training set from 340K to 1.7M samples in (4) leads to substantial improvements, demonstrating the effectiveness of our large-scale 3D data generation pipeline.

###### 4.4. 3D Grounding Helps Spatial Reasoning

We design two experiments to demonstrate that native 3D grounding explicitly improves spatial reasoning. First, we feed the grounding results from our model into Qwen3-VL, prompting it to reason based on our 3D grounding output.

Table 6. Ablation studies on the effectiveness of 3D grounding. Our intermediate 3D grounding results can improve Qwen3-VL’s performance, while training our model directly on QA data leads to degraded results. These findings demonstrate that our unified N3D-VLM, which first performing grounding, then spatial reasoning, effectively enhances the spatial understanding.

Method N3D-open ↑ N3D-num ↑

Qwen3-VL-8B [29] (direct answer) 66.3 36.3 Qwen3-VL-8B [29] (ground. given) 71.3 54.6 Improvement (%) +7.5% +50.4%

Our QAonly-7B 80.6 62.4 Our N3D-VLM-7B 89.7 92.1

As shown in Tab. 6, Qwen3-VL achieves higher accuracy with the grounding results compared to answering directly, indicating that 3D grounding enhances spatial reasoning. However, its performance still lags behind our 7B model, suggesting that our model performs strong spatial reasoning based on 3D grounding. In the second experiment, we use the same architecture but train it end-to-end for question answering without separating grounding and reasoning. This setup underperforms our full model, confirming that explicitly decomposing the task into grounding and reasoning leads to better performance under same architecture.

###### 5. Conclusion

We present a unified framework N3D-VLM that bridges

- 3D object perception and spatial reasoning within a single model. By enabling native 3D grounding and explicit 3Daware reasoning, our approach offers both accurate localization and interpretable spatial understanding. Supported by a scalable data construction pipeline that projects 2D annotations into 3D space and facilitates the creation of explicit reasoning datasets, our model demonstrates reasonable generalization in 3D grounding and provides a structured foundation for spatial reasoning. Extensive experiments validate the effectiveness of our framework, which achieves strong performance in both 3D grounding and 3D spatial reasoning among existing vision-language models.

###### References

- [1] Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan Frost, Luke Holland, Campbell Orme, et al. Scenescript: Reconstructing scenes with an autoregressive structured language model. In ECCV, 2024. 2, 4
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 6, 7
- [3] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021. 2
- [4] Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3d: A large benchmark and model for 3d object detection in the wild. In CVPR, 2023. 2, 4
- [5] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 2
- [6] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024. 3
- [7] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In CVPR, 2024. 3
- [8] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In NeurIPS, 2024. 2, 3, 5, 6, 7
- [9] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. In IJRR,

2013. 2

- [10] Peiqi He, Zhenhao Zhang, Yixiang Zhang, Xiongjun Zhao, and Shaoliang Peng. Spatial-ormllm: Improve spatial relation understanding in the operating room with multimodal large language model. arXiv preprint arXiv:2508.08199,

2025. 2, 3

- [11] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 3, 6, 7, 5
- [12] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In EMNLP, 2014. 5, 6, 7
- [13] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. In IJCV,

2020. 2, 4

- [14] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025. 2, 7
- [15] Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, and Junwei Liang. Seeground: See and ground for zero-shot openvocabulary 3d visual grounding. In CVPR, 2025. 2, 3
- [16] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 2
- [17] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 4
- [18] Dingning Liu, Xiaomeng Dong, Renrui Zhang, Xu Luo, Peng Gao, Xiaoshui Huang, Yongshun Gong, and Zhihui Wang. 3daxiesprompts: Unleashing the 3d spatial task capabilities of gpt-4v. arXiv preprint arXiv:2312.09738, 2023. 3
- [19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 3
- [20] Zhenyang Liu, Yikai Wang, Sixiao Zheng, Tongying Pan, Longfei Liang, Yanwei Fu, and Xiangyang Xue. Reasongrounder: Lvlm-guided hierarchical feature splatting for open-vocabulary 3d visual grounding and reasoning. In CVPR, 2025. 3
- [21] Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso de Melo, Jianwen Xie, and Alan Yuille. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. arXiv preprint arXiv:2504.20024, 2025. 2, 3, 7, 6
- [22] Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. Spatiallm: Training large language models for structured indoor modeling. arXiv preprint arXiv:2506.07491, 2025. 2, 3, 4, 7
- [23] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428, 2025. 2, 3
- [24] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2, 4
- [25] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021. 2
- [26] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask transformer for 3d semantic instance segmentation. arXiv preprint arXiv:2210.03105, 2022. 2
- [27] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, 2019. 2, 4, 7

- [28] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2, 3, 7
- [29] Qwen Team. Qwen3-vl github repository. https:// github.com/QwenLM/Qwen3-VL, 2025. 7, 8, 2, 3, 4, 5
- [30] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 6, 7
- [31] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. arXiv preprint arXiv:2504.01901, 2025. 3
- [32] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546,

2025. 2, 3, 4, 5

- [33] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Selfsupervised learning of reliable point representations. In CVPR, 2025. 7
- [34] Runsen Xu, Zhiwei Huang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Vlm-grounder: A vlm agent for zero-shot 3d visual grounding. arXiv preprint arXiv:2410.13860, 2024. 2, 3
- [35] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025. 3
- [36] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In NeurIPS, 2024. 3
- [37] Hanxue Zhang, Haoran Jiang, Qingsong Yao, Yanan Sun, Renrui Zhang, Hao Zhao, Hongyang Li, Hongzi Zhu, and Zetong Yang. Detect anything 3d in the wild. arXiv preprint arXiv:2504.07958, 2025. 2, 4
- [38] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In CVPR, 2024. 4
- [39] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In CVPR, 2025. 2, 3
- [40] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024. 3

## N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models

### Supplementary Material

In this supplementary material, we provide more additional experiments in Sec. A. We also present a video demo for more qualitative results in Sec. B. We illustrate the data distribution of our proposed N3D-Bench in Sec. C.

###### A. Additional Experiments

###### A.1. 3D Grounding Comparison

We present a qualitative comparison of the native 3D grounding capability of our N3D-VLM with two baseline methods, SpatialLM [22] and Qwen3-VL-8B [29], as shown in Fig. 5 and Fig. 6. SpatialLM is a vision-language model designed for 3D grounding in the indoor scenes using point cloud input. We generate point clouds for SpatialLM by combining the input image with depth maps obtained via [32]. For Qwen3-VL-8B [29], we apply the depth alignment strategy described in the main paper to align its predictions with our ground truth. Fig. 5 shows two indoor scene examples. In the first, our method accurately localizes the pillows, while both baselines either detect only a subset of objects or exhibit inaccurate spatial prediction. In the second scene, our method also provides a precise localization of the washing machines. Fig. 6 presents two diverse outdoor scenes. SpatialLM fails in these cases, as it is limited to a predefined set of indoor categories. In both examples, our N3D-VLM outperforms Qwen3VL-8B for detection, demonstrating superior native 3D grounding capabilities, which are essential for reliable spatial reasoning.

We also consider SpatialReasoner [21], a related method capable of predicting 3D object centers. However, it does not support explicit 3D bounding box output, limiting its capacity for comprehensive spatial perception. In our evaluation, we observed that even when prompted to output object centers, the results were inconsistent and often failed to follow a coherent 3D coordinate format.

###### A.2. 3D Spatial Reasoning Comparison

We present a qualitative comparison of the 3D spatial reasoning capabilities of our N3D-VLM with GPT-4o [11], Qwen3-VL-8B [29], SpatialRGPT [8], and SpatialReasoner [21], as shown in Fig. 7 and Fig. 8. Fig. 7 compares our N3D-VLM with GPT-4o and Qwen3-VL-8B. In Example 1, both GPT-4o and Qwen3-VL fail to answer the question correctly due to incorrect reasoning about the change in viewpoint. In contrast, our N3D-VLM provides the correct answer by leveraging accurate 3D grounding and reasoning over the grounded results. In Example 2, GPT-4o and

Qwen3-VL-8B overly rely on commonsense priors while neglecting visual cues. Our method, empowered by native 3D grounding, directly localizes the bounding box to answer the question accurately. Fig. 8 compares our method with SpatialRGPT and SpatialReasoner on questions involving relative distance comparison and depth comparison among three objects. SpatialRGPT either provides incorrect answers or misunderstands the question entirely. Although SpatialReasoner has the potential to reason using 3D coordinates, it fails to complete the reasoning process. For instance, in Example 1, it incorrectly attempts to calculate absolute distances for a depth-ordering question, resulting in rigid and ultimately incorrect reasoning.

In contrast, our N3D-VLM not only accurately localizes 3D bounding boxes of objects but also reliably performs reasoning over grounded results, demonstrating its effectiveness in 3D spatial understanding tasks.

As a supplement to Table 2 in the main paper, Tab. 7 and Tab. 8 report accuracy scores across different question types. Our N3D-VLM consistently outperforms our base model Qwen2.5VL [2], confirming that native 3D grounding significantly enhances the model’s ability to understand various forms of spatial questions.

###### A.3. Failure Cases

We present two failure cases of 3D grounding in Fig. 9. In the first example, a duck’s reflection on the water surface (highlighted by a green circle) is mistakenly identified as a real object, suggesting that our model could benefit from a better understanding of specular reflections. In the second example, although N3D-VLM successfully detects 30 jellyfish with accurate 3D bounding boxes, several jellyfish are still missed, as indicated in the green circle. These cases highlight that our method, while effective, still has room for improvement. Enhancing the 3D grounding capabilities may further boost overall performance in 3D spatial understanding tasks.

###### B. Video Demo

As a supplement to Sec. A, we have included a video demo to showcase the qualitative results in a video format.

###### C. N3D-Bench Details C.1. Distribution Summary

In Table 1 of the main paper, we compare our N3D-Bench with SpatialRGPT-Bench, showing that N3D-Bench covers

[Figure 44]

[Figure 45]

Locate the 3D bbox of all the pillow.

Image Point Cloud

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

###### (a) SpatialLM (b) Qwen3-VL-8B (c) Our N3D-VLM

[Figure 52]

[Figure 53]

Locate the 3D bbox of all the washing/dry machine.

Image

Point Cloud

[Figure 54]

[Figure 55]

[Figure 56]

(a) SpatialLM (b) Qwen3-VL-8B (c) Our N3D-VLM

- Figure 5. Qualitative comparison of 3D grounding capability with SpatialLM [22] and Qwen3-VL-8B [29] in indoor scenes. our N3D-VLM accurately localizes objects such as pillows and washing machines, while baselines either miss objects or exhibit inaccurate prediction. In the visualization, green boxes represent ground truth 3D bounding boxes, and red boxes indicate model’s predictions.

a broader range of object categories and question types. As a supplement to Table 1, we further present the dis-

tribution of question types and object categories in N3DBench. Specifically, Fig. 11 shows the distribution of ques-

[Figure 57]

[Figure 58]

Locate the 3D bbox of all the pig.

Image Point Cloud

[Figure 59]

[Figure 60]

| |
|---|

(a) SpatialLM (b) Qwen3-VL-8B (c) Our N3D-VLM

[Figure 61]

[Figure 62]

[Figure 63]

Locate the 3D bbox of all the person.

Image Point Cloud

[Figure 64]

(a) SpatialLM (b) Qwen3-VL-8B (c) Our N3D-VLM

- Figure 6. Qualitative comparison of 3D grounding capability with SpatialLM [22] and Qwen3-VL-8B [29] in outdoor scenes. Our N3D-VLM outperforms Qwen3-VL-8B and SpatialLM, highlighting its superior native 3D grounding capability for reliable spatial reasoning. In the visualization, green boxes represent ground truth 3D bounding boxes, and red boxes indicate model’s predictions.

tion types. The dataset contains 2,000 questions grouped into 11 major categories, reflecting both high diversity and balanced coverage. For example, the questions include reasoning about relative distance comparison and depth comparison among three or more objects. For relative spatial relations such as left/right, we also incorporate variations in viewpoint, including phrasing like “from the opposite di-

rection of the camera.” Fig. 10 illustrates the distribution of object categories involved in the questions. N3D-Bench includes 264 commonly encountered indoor and outdoor object classes, derived from the Objects365 [27] dataset.

[Figure 65]

🤖 GPT-4o: Yes, the person is positioned behind the orange/tangerine as seen from the side opposite the photographer.

❌ Wrong answer

🤖 Qwen3-VL-8B: (…) The girl is standing behind the bowl of fruit. (…) The phrase "as seen from the side opposite the photographer" is a bit ambiguous, but it doesn't change the fundamental spatial relationship. (…)

👤 Question: Is person positioned Final answer: Yes. behind orange/tangerine as seen from the side opposite the photographer?

❌ Wrong answer

[Figure 66]

Groundtruth answer: No.

🤖 Our-N3D-VLM: (3D Grounding)

- bbox_0=Bbox(person,0.49,0.59,2.69,0.59,0.33,0.38)\n
- bbox_1=Bbox(orange/tangerine,0.72,0.86,1.37,0.22,0.07,0.12)\n (...)

(3D Spatial Reasoning) </think> Step 1: Compare the Z coordinates of the centers. Z_A

= 2.69, Z_B = 1.37 Step 2: Determine if person is in front of or behind of orange/tangerine. Since Z_A = 2.69 > Z_B = 1.37, and we are considering the opposite view, person is in front of orange/tangerine. </think>

✅ Correct answer

Final answer: No, it is not.

- (a) Example 1
- (b) Example 2

[Figure 67]

🤖 GPT-4o: The height of the person cannot be determined from the image alone.

❌ Refuse to answer.

🤖 Qwen3-VL-8B: (…) In wheelchair tennis, players are often tall. For example, many top players are between 1.7 and 1.9 meters tall. This is because a taller player has a better reach and can hit the ball more effectively. Make a reasonable estimate (…) would be around 1.75 meters. Final Answer: 1.75 m.

👤 Question: What is the height of the person?

❌ Wrong answer

Groundtruth answer: 0.9 m.

[Figure 68]

🤖 Our-N3D-VLM: (3D Grounding) bbox_0=Bbox(person,0.28,0.49,2.98,0.62,0.90,0.77)\n (...)

(3D Spatial Reasoning) <think>

The 3D bbox of person is [0.27, 0.02, 2.98, 0.62, 0.9, 0.77]. So, the height of person is 0.9 meters.

</think> Final answer: person is 0.9 meters in height. ✅ Correct answer

- Figure 7. Qualitative comparison of 3D spatial reasoning capability with GPT-4o [11] and Qwen3-VL-8B [29]. Our N3D-VLM outperforms GPT-4o and Qwen3-VL-8B by leveraging accurate 3D grounding and 3D spatial reasoning.

[Figure 69]

🤖 SpatialRGPT: bicycle, bowl/basin, trophy

❌ Wrong answer

🤖 SpatialReasoner: <think> The 3D location of the bicycle is (0.9, 0.7, 3.1). The 3D location of the bowl/basin is (-0.4, 1.2, 2.1). The 3D location of the trophy is (0.8, 1.5, 4.6). The L2 distance between the bicycle and the bowl/basin is 1.57. The L2 distance between the bicycle and the trophy is 1.77.(...) </think> Final answer: bicycle, bowl/basin

👤 Question: Could you list bicycle, bowl/basin, and trophy from the closest to the most distant relative to the camera?

❌ Fail to reason and answer

Groundtruth: bowl/basin, bicycle, trophy.

[Figure 70]

🤖 Our-N3D-VLM: (3D Grounding)

trophy

- bbox_0=Bbox(bicycle,0.18,0.73,5.34,1.44,1.31,1.21)\n
- bbox_1=Bbox(bowl/basin,0.63,0.56,3.44,0.19,0.09,0.26)\n
- bbox_2=Bbox(trophy,0.30,0.26,6.00,1.43,0.76,0.92)\n (...)

(3D Spatial Reasoning) </think> The z coordinate of bicycle, bowl/basin and trophy is 5.34, 3.44 and 6.0. Sort the objects by z coordinate (from near to far): bowl/basin, bicycle and trophy. </think>

bowl/basin

bicycle

✅ Correct answer

Final answer: bowl/basin < bicycle < trophy

- (a) Example 1
- (b) Example 2

[Figure 71]

🤖 GPT-4o: The ambulance is closer to the viewer than the truck. ❌ Fail to answer.

🤖 Qwen3-VL-8B: <think> The 3D location of the ambulance is (4.5, 2.6, 7.9). The 3D location of the truck is (4.1, 2.5, 12.0). The 3D location of the SUV is (-0.2, 1.6, 3.7). The L2 distance between the ambulance and the truck is 4.18. The L2 distance between the ambulance and the SUV is 7.70. (...) </think> Final answer: truck

👤 Question: Considering suv as the reference object, is ambulance or truck closer? Groundtruth: ambulance.

❌ Fail to reason and answer

🤖 Our-N3D-VLM: (3D Grounding)

- bbox_0=Bbox(suv,0.50,0.68,4.92,4.20,1.70,0.63)\n
- bbox_1=Bbox(ambulance,0.08,0.54,7.99,1.02,2.30,2.45)\n
- bbox_2=Bbox(truck,0.91,0.59,14.32,1.12,0.54,0.67)\n (...)

ambulance

[Figure 72]

(3D Spatial Reasoning) <think>We can calculate the Euclidean distance between suv and ambulance, and suv and truck. Distance of ambulance and suv = √((4.32 - -0.03)² + (-0.34 - -0.85)² + (7.99 - 4.92)²) = (...)

suv

truck

= 5.3486. Distance of truck and suv = √((-7.45 - -0.03)² + (-1.23 - -0.85)² + (14.32 - 4.92)²) = (...) = 11.9817. 3. Since 5.3486 < 11.9817, ambulance is closer to suv than truck in 3D.</think>

Final answer: ambulance is closer to suv. ✅ Correct answer

- Figure 8. Qualitative comparison of 3D spatial reasoning capability with SpatialRGPT [8] and SpatialReasoner [21]. N3D-VLM outperforms SpatialRGPT and SpatialReasoner by accurately interpreting the question and reasoning over explicit 3D bounding boxes.

- Table 7. Detailed quantitative comparison on spatial reasoning benchmark N3D-Bench. Dark green indicates the highest accuracy, while light green denotes the second-highest accuracy.

Category Method

N3D-Bench Open-ended Numerical

left/right front/behind wide/thin tall/short big/small relative dis. depth comp. width/height distance ver./hor. dis. direction Closed-source

GPT-4o [11] 61.50 51.00 65.33 62.00 68.50 70.00 74.22 24.50 14.00 14.00 25.50 Gemini-2.5-Flash [28] 63.00 52.00 62.31 62.00 74.00 70.00 73.44 36.00 20.00 24.50 25.00

Open-source

Qwen2.5-VL-7B [2] 51.50 45.00 58.00 54.00 61.00 61.97 58.14 12.50 15.00 18.50 17.00 Qwen3-VL-8B [29] 71.50 48.00 62.50 63.50 71.50 61.97 66.67 27.50 33.00 21.50 27.50 SpatialLadder-3B [14] 45.50 50.50 52.50 47.00 52.00 52.11 41.09 8.00 16.00 17.00 12.00 SpatialReasoner-7B [21] 38.50 51.50 62.50 58.50 59.50 64.79 54.26 16.00 36.00 16.50 13.50 SpatialRGPT-VILA-1.5-8B [8] 50.50 48.00 78.39 69.00 75.50 61.43 54.69 34.00 39.50 36.00 62.00

Ours

N3D-VLM-3B 97.50 95.50 45.50 66.50 62.50 92.96 93.80 75.00 96.00 93.00 83.50 N3D-VLM-7B 95.00 93.00 90.00 85.50 80.00 92.96 95.35 82.50 96.50 93.00 87.00

- Table 8. Detailed quantitative comparison on spatial reasoning benchmark SpatialRGPT-Bench. Dark green indicates the highest accuracy, while light green denotes the second-highest accuracy.

SpatialRGPT-Bench Open-ended Numerical

Category Method

below/above left/right big/small tall/short wide/thin behind/front distance hor. dis. ver. dis. width height direction Closed-source

GPT-4o [11] 88.33 78.10 82.08 74.11 68.27 65.45 30.30 42.31 53.12 52.24 71.70 65.42 Gemini-2.5-Flash [28] 85.83 92.23 88.68 81.25 79.81 67.27 12.06 12.93 18.18 64.55 77.24 72.64

Qwen2.5-VL-7B [2] 76.67 89.52 71.70 71.43 73.08 64.55 21.15 28.74 21.62 35.29 49.50 57.94 Qwen3-VL-8B [29] 93.33 96.19 88.68 84.82 88.46 83.64 29.73 26.23 29.25 45.80 55.64 59.05 SpatialLadder-3B [14] 58.33 56.19 55.66 54.46 55.77 54.55 24.32 21.31 35.85 21.80 33.08 23.71 SpatialReasoner-7B [21] 68.33 48.57 68.87 66.07 76.92 50.0 37.16 34.43 17.92 34.62 35.34 42.53 SpatialRGPT-VILA-1.5-8B [8] 99.17 100.0 84.90 89.28 91.34 90.90 45.9 68.0 56.6 48.9 61.7 95.3

Open-source

N3D-VLM-3B 97.50 100.0 75.47 73.21 39.42 94.55 72.30 78.69 83.96 48.12 69.70 93.46 N3D-VLM-7B 100.0 99.05 94.34 92.86 92.31 95.45 81.08 85.25 88.68 54.17 64.29 96.26

Ours

[Figure 73]

[Figure 74]

|[Figure 75]|
|---|

|[Figure 76]|
|---|

Locate the 3D bbox of all the duck.

Locate the 3D bbox of all the jellyfish.

(b) Native 3D Grounding

(a) Image

(a) Image (b) Native 3D Grounding

- Figure 9. Failure cases of our native 3D grounding. N3D-VLM misidentifies a specular reflection and misses several objects, suggesting room for improvement in handling reflections and dense object scenes.

[Figure 77]

###### Figure 10. Distribution of object classes in N3D-Bench.

[Figure 78]

###### Figure 11. Distribution of question types in N3D-Bench.

