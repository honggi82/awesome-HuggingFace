## Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models

Runsen Xu1,2 Weiyao Wang1 Hao Tang1 Xingyu Chen1 Xiaodong Wang1 Fu-Jen Chu1 Matt Feiszli1 Kevin J. Liang1 1FAIR, Meta 2The Chinese University of Hong Kong

# arXiv:2505.17015v2[cs.CV]22May2026

###### Camera Movement

Visual Correspondence

###### Depth Perception

###### Object Perception

###### Object Movement

- B

- C
- D A

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Which vector is the annotated point in Image1's translation?

Which labeled point in Image-2 matches the circle in Image-1?

How many degrees did the camera rotate in yaw?

[Figure 11]

How far away is the highlighted point from the viewer in the image (mm)?

What is the length of the desk that is commonly visible in these images?

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Multi-SpatialMLLM: It rotated `47` degrees.

Multi-SpatialMLLM: The point's displacement vector is `[ -52 , 43 , -41 ]` mm.

Multi-SpatialMLLM: Point `A`.

Multi-SpatialMLLM: It’s `1591` mm.

Multi-SpatialMLLM: It’s `3256`.

Determine the new position of point [ 353 , 239 ] from Image-1 in Image-2.

In Image-2, which point is nearer to the camera's position [ 258 , 139 ] or [ 472 , 148 ]?

What is the displacement vector of the camera's movement?

[Figure 16]

[Figure 17]

[Figure 18]

What is the height of the desk?

[Figure 19]

How many millimeters has the point [ 300 , 300 ] Image-1 been displaced?

[Figure 20]

Multi-SpatialMLLM: The displacement vector is `[ 900 , 71 , 213 ]` mm.

Multi-SpatialMLLM: The height is approximately `950` millimeters.

Multi-SpatialMLLM: The new position is `[ 531 , 288 ]` .

Multi-SpatialMLLM: The point at `[ 472, 148 ]` is closer.

Multi-SpatialMLLM: The point did not move.

Input Referencing: Coordinate, Visual Prompt, Semantic Label Output Format: Qualitative, Scalar, Multiple-Choice, Vector, Coordinate

Figure 1. We present Multi-SpatialMLLM, a model capable of multi-frame spatial understanding, a capability overlooked by previous spatial understanding research. Multi-SpatialMLLM can support different types of input referencing and outputs for various tasks.

### Abstract

### 1. Introduction

Recent years have witnessed tremendous advances in multi-modal large language models (MLLMs), which have evolved into versatile AI assistants capable of a wide array of tasks [14, 27, 35, 50, 63]. Despite these strides, deploying such assistants as “brains in a jar” within digital platforms limits their potential. Instead, there is a growing push to integrate MLLMs directly into real-world applications, such as robotics [6, 36] and autonomous vehicles [59], to facilitate interactions with the environment. This shift imposes a requirement for human-like spatial understanding. However, current MLLMs struggle with surprisingly basic spatial understanding, even confusing left and right [58].

Multi-modal large language models (MLLMs) have rapidly advanced in visual tasks, yet their spatial understanding remains limited to single images, leaving them ill-suited for physical-world applications that require multi-frame reasoning. In this paper, we propose a framework to equip MLLMs with multi-frame spatial understanding by integrating fundamental spatial skills, including depth perception, visual correspondence, and dynamic perception. We design a novel data pipeline and collect the MultiSPA dataset of more than 27 million samples spanning diverse 3D and 4D scenes to enable training. Alongside MultiSPA, we introduce a comprehensive benchmark that tests a wide spectrum of spatial tasks under uniform metrics. Our resulting model, Multi-SpatialMLLM, achieves significant gains over baselines and proprietary systems, demonstrating scalable and generalizable multi-frame perception. We further observe multi-task benefits and emergent spatial capabilities in challenging scenarios, and showcase how our model can serve as a multi-frame reward annotator for robotics.

Previous works [8, 12] attribute these deficiencies primarily to a shortage of specialized training data, and addresses it by incorporating spatial data into model training, leading to notable improvements. However, these works focus on single-image scenarios, thus restricting the model’s perception to a static field-of-view without any dynamic information. We instead aim for more comprehensive spatial understanding, enabling MLLMs to reason across multiple images. Inspired by the long-standing Structure-from-

- Table 1. Comparison of spatial understanding datasets. Our MultiSPA is the first large-scale dataset for multi-frame spatial understanding, with diverse referencing and output formats. We generate 27M samples here and can scale further if needed.

Dataset Split Multi-Frames GT Annotation Referencing Output # Images # QAs BLINK [20] eval dot MCQ 877 572 UniQA-3D [80] eval dot, semantic MCQ 2450 2450 Q-Spatial [38] eval semantic scalar 271 271 VSR [41] train, eval semantic true/false 11K 11K SpatialVLM [8] train, eval Only eval semantic qual., MCQ, scalar 10M 2B SpatialRGPT [12] train, eval Only eval mask qual., scalar 1M 8.7M MultiSPA train, eval dot, coord., semantic qual., MCQ, scalar, coord., vec. 1.1M 27M+

Motion problem [24] from 3D computer vision, we focus on integrating three fundamental capabilities into MLLMs: (1) depth perception, to infer relative distances and threedimensional structures (2) visual correspondence, to match overlapping regions across images for consistent scene association, and (3) dynamic perception, to perceive selfmovement (camera motion) and object motion.

The challenge in achieving this goal is the scarcity of suitable training data. Because manual annotation at the required scale can be expensive, prior works [8, 12] have resorted to single-view data from in-the-wild images [34], relying on off-the-shelf modules such as monocular depth estimators [26] and open-vocabulary object detectors [55]. However, this approach often produces noisy annotations. Moreover, our objective is to collect multi-frame spatial data, which requires both spatial and temporal alignment—an open challenge [68, 73] in unstructured in-thewild images. Consequently, we leverage existing annotated 3D [13] and 4D [30, 33, 51] datasets for data collection.

We develop a data engine that samples image pairs with a uniform overlap distribution, then backprojects spatially and temporally aligned point clouds to establish pixel correspondences. Leveraging these correspondences in addition to camera movement and projection information, we create high quality question–answer pairs via diverse, LLMgenerated templates. In contrast to previous methods that rely on semantic labels [8] or object masks [12] for referencing, our framework supports multiple modalities, including visual point annotations, pixel coordinates, and semantic labels, thus broadening potential downstream applications. The collected data encompasses both qualitative and quantitative spatial information, ranging from text to scalar, 2D pixel locations, and 3D displacement vectors.

In total, we curate a dataset named MultiSPA consisting of more than 27 million samples, which we use to train our Multi-SpatialMLLM, as illustrated in Fig. 1. To the best of our knowledge, MultiSPA is the first large-scale dataset dedicated to multi-frame spatial understanding. Alongside the dataset, we introduce a novel MultiSPA benchmark to evaluate multi-frame spatial reasoning in MLLMs, covering diverse tasks and output formats under a unified metric.

Extensive experiments show that Multi-SpatialMLLM

substantially outperforms base models and proprietary systems, and even matches specialized 3D perception models. We show that its multi-frame spatial understanding generalizes to held-out benchmarks and scales with data volume, task diversity, and trainable parameters, without degrading the base models’ general abilities. To further provide insights into spatial training, we find that multi-task training on MultiSPA provides notable benefits and, for the first time, reveals preliminary signs of emergent behavior on spatial tasks. Finally, we demonstrate our model’s potential as a multi-frame reward annotator for robot learning.

### 2. Related Work

Multi-modal large language models. We refer to multimodal large language models (MLLMs) or vision-language models (VLMs) as large language models extended to handle image inputs [3, 10, 14, 21, 23, 39, 42, 46, 79], typically by incorporating an image encoder [11, 28, 52] converting images into tokens, which are then projected into the LLM’s [2, 63, 65] text latent space and processed alongside text tokens. Training MLLMs uses the same language modeling objective as LLMs. Thanks to Internetscale image–text training data, e.g. captioning or OCR corpora [56, 57], MLLMs have demonstrated remarkable performance on tasks like multi-modal dialogue. However, these training data lack sufficient spatial annotations, resulting in deficient spatial understanding. We address this limitation by further fine-tuning existing MLLMs on newly collected spatial data. Notably, we preserve the original model architecture to maintain the wide range of capabilities and application scenarios derived from large-scale pre-training. Spatial understanding benchmarks for MLLMs. Researchers have explored deploying MLLMs on real-world platforms such as robotics [6, 16, 31, 36, 71] and autonomous vehicles [59], applications requiring human-like spatial understanding to perceive and interact with the environment. However, spatial understanding is a complex, fundamental ability that is difficult to define formally, so researchers have introduced various benchmarks targeting different aspects of spatial reasoning for evaluation purposes. Most prior works focus on single-image spatial understanding, primarily assessing inter-object spatial rela-

tions [17, 29, 38, 41, 58, 64, 78] or spatial recognition inspired by cognitive science [53]. Some benchmarks extend beyond single images: BLINK [20] and UniQA-3D [80] evaluate spatial relationships across image pairs, while video-based benchmarks [40, 75] like VSI-Bench [72] introduce scene-level spatial reasoning. Though these share some similar tasks with ours, such as qualitative camera movement estimation and keypoint matching, our proposed benchmark includes additional tasks like object movement perception, and supports more diverse input and output formats, instead of just the limited multiple-choice format.

Improving MLLMs for spatial understanding. Existing benchmarks highlight the limitations of MLLMs in spatial understanding, prompting several recent works [5, 7, 8, 12, 18, 48, 60, 70]. Most of these focus on single-image understanding. SpatialVLM [8] was the first to identify the lack of spatial training data as a key limitation, demonstrating significant improvements by fine-tuning MLLMs on a curated spatial dataset. SpatialRGPT [12] extended this approach by introducing mask-based reference and incorporating depth images. SpatialPIN [48] explored an alternative strategy, avoiding model fine-tuning and instead leveraging specialized perception models to extract spatial information for MLLMs. Unlike prior efforts, we focus on enabling MLLMs to reason across multiple images for spatial understanding by fine-tuning them on our newly collected dataset. Concurrently, SAT [54] also explores multi-frame spatial reasoning, but it relies on simulated data, potentially introducing a sim-to-real gap. In contrast, our dataset is significantly larger, derived from real-world images across diverse scenarios, and covers a broader range of spatial reasoning tasks. See Tab. 1 for a comparison of our dataset with other popular spatial datasets and benchmarks.

### 3. MultiSPA Dataset and Benchmark

In this section, we introduce our MultiSPA dataset in Sec. 3.1, describe the data generation pipeline in Sec. 3.2, and present the MultiSPA benchmark in Sec. 3.3.

#### 3.1. MultiSPA Dataset

Tasks definitions. We aim to equip MLLMs with the ability to integrate multiple images for spatial understanding. Building on the three fundamental capabilies discussed in Sec. 1, we introduce the following five tasks to generate training data: 1) Depth perception, 2) Visual correspondence, 3) Camera movement perception, 4) Object movement perception, and 5) Object size perception. Fig. 1 shows examples of these five tasks.

Referencing and output types. As summarized in Tab. 1, we reference specific pixels or objects in spatial QA data using visual dots (points) or semantic labels, as opposed to masks[12] to avoid additional dependency on a segmentation module [32]. Additionally, we introduce pixel coordi-

nates as a straightforward referencing method that preserves the original images without requiring extra annotations. Beyond referencing, most existing datasets constrain spatial tasks to multiple-choice formats or limit outputs to qualitative or scalar quantitative answers. We broaden these restrictions by incorporating diverse quantitative outputs such as pixel coordinates and displacement vectors. Detailed descriptions of each task are provided as follows, with examples of each in the supplementary material.

Depth perception. We divide this task into two subtasks: direct depth estimation and depth comparison. In the first task, a single pixel is specified in an image, and the model must estimate its depth. In the second, two pixels are specified, and the model must identify the one closer to the camera. Both subtasks support referencing pixels either via visually annotated dots or pixel coordinates.

Visual correspondence. Given two images and a pixel location in the first image, the model must identify the corresponding pixel with the same 3D position in the second image, either qualitatively or quantitatively. In the qualitative version, the pixel is annotated visually in both images, and an additional three pixels in the second image are labeled to form a multiple-choice question. The model’s goal is to pick the correct label. In the quantitative version, only the pixel coordinates are specified, and the model must output the corresponding coordinates in the second image.

Camera movement perception. Given two images, the model must estimate the relative movement of the camera from the first view to the second, including both translation and orientation. We define multiple output levels, from coarse to fine-grained. In the simplest variant, the model must only identify the camera’s movement direction along three translational axes: “right” or “left”, “forward” or “backward”, and “up” or “down”, as well as its rotation direction in two axes: rotating “left” or “right” and tilting “up” or “down”. A more challenging variant requires the model to estimate scalar values of the overall translation distance or rotation angle. Finally, the most detailed form requires predicting the camera’s displacement vector. In total, we have nine question types for this category.

Object movement perception. Given two images and a pixel location on a specific object (or object part) in the first image, the model estimates the pixel’s overall translation distance or, at a finer level, the pixel’s displacement vector with respect to the first view. The camera may remain still or move during capture, and pixel referencing can be done either via visual annotations or pixel coordinates.

Object size perception. Given several images of a target object, the model estimates the object’s height, width, and length. We treat this as a higher level of spatial understanding compared with the previous four tasks. The model must integrate information across all images to infer the object’s size. We use semantic labels to refer to the object.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

#### 3.2. MultiSPA Data Generation

[Figure 27]

Visible Points Calculation

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Data Format. Following common MLLM fine-tuning strategies [10, 14, 43], we format our data as QA pairs as:

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

User: <image>...<image>{description}{question} Assistant: {answer}

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Point Cloud

[Figure 43]

[Figure 44]

We use GPT-4o [50] to generate diverse templates for task descriptions, questions, and answers. Please refer to the supplementary material for detailed templates for each task. To facilitate answer extraction, we enclose the answer in backticks (‘‘). For numerical answers of metric length, we use millimeters as the unit and round to the nearest integer. For pixel coordinates, we normalize the values to maintain compatibility with varying image resolutions as follows:

Overlap Calculation

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

16.7% 7.9% 0%

[Figure 52]

Figure 2. Overlap ratio calculation of image pairs.

y H × 1000 (1)

x W × 1000 ,ynorm =

Image pairs sampling. Although depth estimation is performed on single images, we also require image pairs with overlapping regions to construct multi-frame spatial understanding data. For each scene, we define the overlap ratio between two images as the IoU of their visible points:

xnorm =

where x,y are the original pixel coordinates, and W,H are the width and height of the image, respectively.

Source datasets. We leverage existing annotated scene datasets for high-quality data collection. Specifically, we use the 4D datasets Aria Digital Twin (ADT) [51] and Panoptic Studio (PStudio) [30], with 3D tracking annotations from the TAPVid3D [33] dataset for the object movement perception task, and the 3D dataset ScanNet [13] for other tasks. Our data generation pipeline can be used for other datasets as long as they have the same spatial annotations. Further details are in the supplementary material.

Overlap(i,j) = |Pi ∩ Pj| |Pi ∪ Pj|

. (4)

We only consider image pairs with overlap ratios between 6% and 35%, as ratios outside this range indicate either too little or too much shared content. Fig. 2 visualizes the calculation of the overlap ratio. Notably, the overlap ratio exhibits a long-tailed distribution, where most pairs have a low overlap. We do not use all image pairs, and to achieve balanced sampling, we divide the overlapping pairs into bins based on their overlap ratios. We then evenly allocate a target number of samples among these bins while prioritizing bins with fewer samples. For each task, we sample image pairs with different random seeds to ensure diversities. More details are in the supplementary material.

- 3.2.1. Static Scene Data Generation Visible points calculation. For each scene, ScanNet [13]

provides a reconstructed point cloud Pscene = {pW}, where each point pW = (X,Y,Z)T is in world coordinates. Each RGB image Ii has a depth map Di, an extrinsic matrix Ei (camera to world), and an intrinsic matrix Ki. We transform and project each point pW onto Ii via:

Visual correspondence data generation. For an sampled image pair (Ii,Ij), we randomly select one point from their co-visible points Pi ∩Pj and use its projected pixel coordinates in both images to construct the QA pair.

 

  =

 

 . (2)

- pCi [0]
- pCi [1]
- pCi [2]

- u
- v 1

pW 1

Ki pCi [2]

pCi = (Ei)−1

,

Camera movement perception data generation. In the ScanNet[13] dataset, the camera coordinate system is defined with its origin at the top-left of the image, where the x-axis points to the right, the y-axis points downward, and the z-axis points forward. For an image pair (Ii,Ij), we compute the relative camera pose with respect to the first image as Eij = E−i 1Ej ∈ R4×4. The translation component is given by the displacement vector (dij)T = [xij,yji,zji] = Eij[0:3,3], and its norm, ∥dij∥, represents the overall translation distance. When xij > 0, yji > 0, and zji > 0, we label the camera motion as “right”, “down”, and “forward”, respectively; otherwise, the movement is considered “left”,

We maintain all visible points of image i, denoted as Pi, by selecting those whose projected coordinates (u,v) lie within the image bounds and are not occluded:

0 < pCi [2] < Di(u,v). (3) Depth perception data generation. To create depth perception data, we randomly sample images for each scene. For each image Ii, we sample one or two visible points from Pi, record their 2D coordinates (u,v) and corresponding depth pCi [2], and fill in the templates to construct QA pairs.

“up”, or “backward.” To determine orientation, we measure rotation angles around the gravity direction and the tilt relative to the ground plane (details in the supplementary material). Finally, we format all these spatial parameters into QA templates to construct the camera movement data.

[Figure 53]

[Figure 54]

All Group-1

Object size perception data generation. For this task, we require a set of images that not only share overlapping regions but also jointly cover the entire target object. To ensure that the model learns to reason across all images, only the complete image set should cover the object’s full dimensions, while no proper subset does. To achieve this, we propose a BFS-based minimum-coverage-set search algorithm that iteratively explores image combinations with early pruning. For each object in ScanNet [13], we use the size of the target object’s 3D bounding box as its “height,” “width,” and “length” and combine these with the searched image sets to construct QA pairs. More details are in the supplementary material.

[Figure 55]

[Figure 56]

Group-2 Group-3

Figure 3. Visualization of rigid body segmentation results.

QA samples from 1.1M unique images. For each subtask in MultiSPA, we hold out 300 samples as evaluation sets, resulting in a total of 7,800 benchmark samples. We ensure that the images in the benchmark come from scenes or scenarios distinct from those in the training split. The dataset distribution is provided in the supplementary material.

- 3.2.2. Dynamic Data Generation TAPVid3D [33] provides temporally aligned point cloud

tracking sequences {Pt}Tt=1, along with the corresponding video frames {It}Tt=1, camera extrinsics {Et}Tt=1, and intrinsics {Kt}Tt=1 for the ADT[51] and PStudio datasets[30]. We use these datasets to construct object movement perception QA pairs. We randomly select one point from the tracked sequences, then choose two images (Ii,Ij) to form the image pair. Similar to the camera movement data generation procedure, we compute each point’s displacement vector and translation distance between these two frames using the camera extrinsics. To ensure diversity, we adopt two additional modules described as follows to sample the points and image pairs (more in supplementary material).

Rigid body segmentation. Point clouds from TAPVid3D typically belong to the same object or a local region, but different parts may be unevenly represented (see Fig. 3). For instance, a moving human often has more points on the torso than on the arms. Random sampling yields a distribution skewed toward the dominant body part, which follows a single movement pattern. Thus, we devise a clusteringbased rigid body segmentation method to group the point clouds according to inter-point distance changes over time, and sample each group separately to enhance diversity.

Image pairs bin sampling. Given a selected point that appears in T frames, one could form up to T(T2−1) image pairs. However, similar to ScanNet, these pairs exhibit a long-tailed distribution of motion magnitudes. We therefore bin the image pairs by the object translation distances and perform balanced sampling for each bin, ensuring diversity across small and large displacements.

- 3.3. MultiSPA Benchmark

Evaluation metric. Our MultiSPA benchmark supports diverse answer formats. The required answer format is specified in the question, and a regular expression is used to extract the answer from model responses. Accuracy is calculated using task-specific criteria. For qualitative and multiple-choice answers, exact matching is used. For scalar and vector, a prediction vpred is correct if: ∥vpred − vgt∥2 ≤ 0.2 · ∥vgt∥2. For pixel coordinates, a prediction is correct if within 5% image width pixels of the ground truth.

### 4. Experimental Results

#### 4.1. Multi-Frame Spatial Understanding

Implementation details. Our preliminary studies show that InternVL2 [63] exhibits stronger instruction-following capabilities than other popular MLLMs (e.g., LLaVAOneVision [35], VILA [39]). Hence, we adopt the 8B InternVL2 model as our base, fine-tuning it on the MultiSPA training split. Specifically, we employ LoRA [25] with rank R = 16 to update the LLM backbone, while freezing the image encoder and projection layer. More results of other settings are in the supplementary material. We use a cosine learning rate scheduler with lr = 4 × 10−5 and the AdamW [45] optimizer. For research efficiency, we train on a subset of MultiSPA (3M QA samples) for one epoch, mixed with 60K general image-based instruction-following samples to preserve the base model’s original abilities. The training is conducted on 24 nodes of 8×32G V100 GPUs with a batch size of 192, taking 50 hours to complete.

Baselines. We include different sizes of InternVL2 [63] and SpatialRGPT [12] as baselines to investigate how our proposed training data improves performance. We also evaluate three popular proprietary models, including “Claude3.5-Sonnet-20241022” [1], “Gemini2.0-Flash” [15], and

Using both ScanNet [13] and TAPVid3D [33], we employ our proposed data generation pipeline to create over 27M

- Table 2. Evaluation on the MultiSPA benchmark. Our Multi-SpatialMLLM significantly outperforms baselines across both qualitative and quantitative subtasks, demonstrating an average 36% gain and surpassing even larger proprietary models.

Multi-SpatialMLLM InternVL-8B InternVL-26B SpatialRGPT [12] Claude-3.5 Gemini-2.0 GPT-4o Average 56.11 (+35.68) 20.43 21.36 16.24 27.50 30.31 28.87

Depth Perception Comparison 74.00 (+24.50) 49.50 50.50 52.17 38.17 57.00 54.84

Value 75.33 (+71.99) 3.34 2.50 3.00 34.84 28.67 22.50 Visual Correspondence

Coordinate 49.00 (+47.33) 1.67 1.67 0.00 1.33 5.67 2.00

MCQ 90.00 (+56.67) 33.33 44.00 30.67 54.67 73.00 67.67 Camera Orientation

Direction 90.83 (+42.66) 48.17 49.34 54.67 62.17 62.17 58.84

Degree 45.50 (+42.16) 3.34 5.17 4.84 10.50 16.34 17.50 Camera Translation

Direction 85.89 (+33.56) 52.33 50.22 53.22 55.11 51.89 54.78 Distance 42.33 (+28.00) 14.33 13.00 7.33 16.33 14.00 13.67

Vector 18.00 (+17.67) 0.33 0.67 0.00 0.33 0.33 0.00 Object Movement

Distance 40.42 (+31.58) 8.84 8.75 2.57 8.50 9.42 8.92

Vector 12.92 (+10.42) 2.50 3.58 0.00 1.92 2.33 5.25 Object Perception

Size 49.11 (+21.66) 27.45 26.89 22.33 46.11 42.89 40.44

[Figure 57]

Figure 4. Scalability of Multi-SpatialMLLM.

“GPT-4o-20241120” [50], as representative models to highlight the limitations of multi-frame spatial understanding even in SOTA MLLMs. Since these baselines often either refuse to answer questions or produce responses failing to adhere to the required format, we employ prompts to encourage them to provide answers or guess values when uncertain. We further use GPT-4 for post-processing to ensure their outputs conform to the prescribed answer format and can be extracted for evaluation accordingly.

MultiSPA benchmark. Tab. 2 summarizes model accuracy on our MultiSPA benchmark. We observe that most existing MLLMs have limited multi-frame spatial understanding ability, performing slightly above random (about 50–60% accuracy) on qualitative tasks such as depth comparison and camera translation direction. Even worse, these baselines fail on tasks requiring quantitative outputs, such as coordinate-based visual correspondence and camera or object movement vectors. SpatialRGPT [12] is trained on vast

single-image spatial data, but it performs worse than zeroshot InternVL [10] on MultiSPA, confirming that singleimage supervision doesn’t transfer to multi-image tasks.

By contrast, our Multi-SpatialMLLM significantly improves performance across all tasks, achieving an average 36% gain over the base model. On relatively easier qualitative tasks, it gets 80–90% accuracy and outperforms all proprietary models. Even on challenging tasks like predicting camera movement vectors, our model attains much higher accuracy, whereas all baselines remain near zero. It is notable that our model has only 8B parameters, which is likely far fewer than those of closed-source models. Yet, with the MultiSPA dataset, it matches or even exceeds their performance, validating the effectiveness of our proposed data.

Scalability of Multi-SpatialMLLM. Certain tasks, like estimating the camera’s displacement vector, are more difficult to learn, so using the same amount of training data brings less improvement. However, the multi-frame spatial understanding ability of Multi-SpatialMLLM is still scalable even for these challenging tasks. To verify this, we select the camera movement vector prediction task as a case study and gradually increase the training data and trainable parameters as shown in Fig. 4. We observe consistent improvements by adding more data and increasing model capacity. With 2.5M samples, the 26B variant achieves around 72% accuracy. These findings encouragingly suggest that further scaling up training data and model capacity holds promise for even more powerful spatial understanding.

Comparisons with expert models. It’s worth noting that Multi-SpatialMLLM is a general VLM, but its 3D perception capabilities can match those of SOTA specialized mod-

Table 3. Evaluation on the BLINK[20] benchmark. Model Avg. V.C. R.D. M.V. S.R. Gemini-2.0 75.7 88.4 83.9 42.9 83.9 Claude-3.5 67.7 74.4 63.7 54.1 75.5 GPT-4o 73.8 84.9 71.0 53.4 81.8 InternVL-8B 57.9 39.0 71.8 49.6 76.2 InternVL-26B 61.4 47.1 78.2 44.4 79.7 SpatialRGPT [12] 52.5 34.3 61.3 44.4 74.1 Multi-SpatialMLLM 84.3 89.5 79.8 94.7 74.8

els. For example, our best-performing 26B model achieves 72% accuracy on camera vector prediction, on par with VGGT [67]. Additional comparisons with expert models are provided in the supplementary material.

#### 4.2. Generalization of Multi-SpatialMLLM

We study the generalization ability of Multi-SpatialMLLM by evaluating it on the held-out external benchmarks and on standard VQA benchmarks. We also demonstrate the multitask benefits introduced by our MultiSPA data.

Held-out benchmarks. To verify whether our model’s multi-frame spatial understanding generalizes to other datasets, we perform zero-shot evaluation on BLINK [20], a diverse benchmark for assessing MLLM perception (Tab. 3). We focus on four splits relevant to spatial reasoning: Visual Correspondence (V.C.), Relative Depth (R.D.), Multi-View Reasoning (M.V.), and Spatial Reasoning (S.R.). Our model never sees BLINK images during fine-tuning, and BLINK’s image resolutions and distributions differ from our training data. We find that all baselines fail on the M.V. task and struggle with V.C. In contrast, our Multi-SpatialMLLM achieves almost 90% accuracy on these tasks and delivers an average 26.4% improvement over the base model, even outperforming proprietary models. This result demonstrates that the multi-frame spatial understanding learned by our model is transferable across datasets. We do not observe gains on the S.R. task, possibly because this task focuses on topological position relations between two objects within a single image, which differs significantly from our multi-frame training data geared toward integrating spatial cues from multiple viewpoints.

To further test generalization, we also render synthetic multi-view images from unseen 3RScan [66] scene meshes and evaluate our 8B and 26B models on the most challenging camera vector prediction task. Our models still achieve 18.7% and 27.7% accuracy, respectively, while zero-shot baselines remain at 0%, confirming the generalization ability of Multi-SpatialMLLM.

Standard VQA benchmarks. We evaluate our MultiSpatialMLLM on several popular standard VQA benchmarks, as shown in Tab. 4. These benchmarks target

Table 4. Evaluation on standard VQA benchmarks.

Model POPE VizWiz OCRVQA MathVista MMStar CCBench

InternVL-8B 84.5 33.2 42.7 58.5 61.1 77.3 Ours 85.3 30.7 42.7 57.6 59.7 75.7

various MLLM capabilities, such as general perception (POPE [37] and VizWiz [22]), optical character recognition (OCRVQA [49]), reasoning (MathVista [47] and MMStar [9]), and Chinese VQA (CCBench [44]). The results show rough parity across the benchmarks, indicating that our model retains most of its original standard VQA proficiency and can be used as a general-purpose MLLM, without being overfit to just multi-frame spatial reasoning.

Table 5. Model performance w./wo. multi-task training.

Camera Movement Vector Object Movement

Single Task 9.30 17.50 Multiple Tasks 18.00 (+8.70) 22.04 (+4.56)

Multi-task generalization and synergy. While each of the tasks proposed in Sec. 3.1 is focused on a narrower subgoal, ultimately the aim is to collectively improve multiframe spatial understanding; we thus prefer that our training data has synergistic generalization effects, as opposed to balancing potentially antagonistic tasks individually. We observe that this is indeed the case by comparing training on just the 500K samples from the camera-movement subset (without any other task data) versus the full training set of 3M samples: we observe that the additional data from the additional tasks indeed increases the accuracy on camera movement questions from 9.3% to 18.0%. We further compare two training configurations for object movement: (1) a dataset of 400K object movement samples alone, and (2) the same 400K object movement samples plus 400K additional samples from camera movement, visual correspondence, and depth estimation. The average accuracy on object movement subtasks increases from 17.5% to 22.04% with the additional data, as shown in Tab. 5. Importantly, these extra 400K samples only involve ScanNet [13] images, whereas the object movement data originate from PStudio [30] and ADT [51], and the two sets do not share question types or data sources. This improvement demonstrates that spatial understanding learned from different datasets and task types can transfer, highlighting an additional scalability dimension beyond merely data volume and model capacity: task diversity. We provide additional results, including further verification of multi-task synergy and subset contributions, in the supplementary material.

#### 4.3. Emergence of spatial understanding

We have shown that our model’s multi-frame spatial understanding is scalable (Fig. 4 and Tab. 5). However, we also investigate whether certain spatial reasoning abilities only appear in sufficiently large models, mirroring the emergent

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Frame-0 Frame-1 Frame-3 -

Frame-5

[Figure 62]

[Figure 63]

How far did the annotated point move from Image A to B?

[Figure 64]

GPT4o: The annotated point moved about 0.4 meters.

InternVL-8B: A few centimeters to the left on the table.

Multi-SpatialMLLM: The point did not move.

Image-A Image-B

Figure 5. Demonstrations of Multi-SpatialMLLM in zero-shot robotics tasks. Our model accurately identifies static objects and predicts movement distances, aligning with the ground truth. It exhibits potential for novel applications like multi-frame reward annotation.

###### Table 6. Multiple-choice visual correspondence accuracy.

because our training set does not include any robotic scenario. As shown in Fig. 5, when asked about the movement of a static blue cube, GPT-4o and the base model respond incorrectly, while ours accurately identifies no movement.

Model Size Encoder Size LLM Size Acc. (Baseline v.s. Hard)

8B 300M 7B 33.3 v.s. 25.67 13B 6B 7B 44.0 v.s. 42.67 26B 6B 20B 44.0 v.s. 82.33

Multi-Frame reward annotator. Prior works [8, 12] have shown that MLLMs with spatial understanding can act as reward annotators in robot learning, but they only handle single-frame inputs. In contrast, our model supports multiframe tasks such as perceiving object movement across consecutive frames. In Fig. 5, we provide Frame-0 and subsequent frames (Frame-1 to Frame-5), then query our model about the object’s displacement. Our model successfully estimates an increasing trend in movement distances, aligning with the ground truth. Though the predicted values are not exact (due to differing resolutions and domains), these results underscore our model’s generalization ability and highlight potential novel applications as a reward annotator or evaluator for robot tasks involving multi-frame spatial understanding, such as “move the object by n meters.”

phenomena observed in text-based LLMs [69].

We explore this through our multiple-choice visual correspondence task for preliminary study. By default, when generating distractor pixels in the second image, we pick them randomly; we denote this as Easy, as distractors may be quite far from the answer. For a more challenging scenario, we deliberately select distractors near the correct pixel, thus requiring higher discriminative power from the model (Hard version). We train various sizes of the base models on these Hard samples and then test on the Easy samples, to gauge whether they can effectively learn from the Hard data. Tab. 6 shows that only the 26B variant improves over the base model, whereas both the 8B and 13B models (the latter equipped with a larger 6B vision encoder) fail to learn effectively from the Hard samples and even show reduced performance. As a reference, training the 8B model on the same number of Easy samples yields 93.33% accuracy on the test set. These findings suggest that learning difficult spatial tasks may require sufficiently large model capacity—potentially pointing to an “emergent” aspect of multi-frame spatial understanding. We leave deeper investigation of this interesting phenomenon to future work.

### 5. Conclusion

In this work, we extend MLLMs’ spatial understanding to multiple frames, a capability overlooked in previous research. We develop a data generation pipeline that produces the first large-scale dataset and a benchmark, MultiSPA, dedicated to this goal. Our extensive experiments demonstrate the effectiveness, scalability, and generalization of the proposed Multi-SpatialMLLM, revealing key observations such as multi-task benefits and emergent behaviors in challenging spatial tasks. The model also opens up new applications, including acting as a multi-frame reward annotator. We discuss limitations and provide more experimental results in the supplementary material.

#### 4.4. Demonstrations of Multi-SpatialMLLM

In Fig. 1, we demonstrate our Multi-SpatialMLLM’s multiview spatial understanding. We further test its real-world performance on newly collected images of a robot arm stacking cubes. These robot scenes are out-of-distribution

### Appendix

- A. MultiSPA Data Samples and Distributions 9
- B. MultiSPA Data Templates 9
- C. Details of Source Datasets 9
- D. Image Pairs Sampling 9
- E. Rotation Angles 9
- F. BFS-Based Minimum-Coverage-Set Search 10
- G. Clustering-Based Rigid Body Segmentation 10
- H. Different Fine-Tuning Strategies 10
- I. Comparison with Expert Models 11
- J. More on VQA Benchmarks 11
- K. Multi-Task Synergy 11
- L. Generalization to More Images. 12
- M. Limitations 12

### A. MultiSPA Data Samples and Distributions

Our MultiSPA dataset has 26 subtasks in total. Each task with an example is shown from Fig. 7 to Fig. 15. We show the distribution of samples across different task types in Tab. 7. Note that our data engine is scalable, and we can generate more training samples for these tasks.

### B. MultiSPA Data Templates

Due to paper length limits, we only show part of the templates in Listing 4. Other templates are similar to those shown in this supplementary material.

### C. Details of Source Datasets

ScanNet. ScanNet [13] is an RGB-D dataset containing more than 1,500 indoor scans. Each scan provides reconstructed point clouds, 3D camera poses, camera intrinsics, depth maps, and 3D instance and semantic segmentation masks. Our data generation pipeline utilizes all these annotations, though segmentation masks are optional if object perception data is not required.

PStudio. The CMU Panoptic Studio dataset [30] comprises 65 sequences (5.5 hours total) of multiple people interacting with one another or with objects, captured within a light stage. It offers multi-view images, 3D body skeletons, and facial landmark annotations.

ADT. Aria Digital Twin [51] is an egocentric video dataset recorded with Aria glasses. It contains 200 sequences of real-world indoor activities, each with precise 6DoF camera poses, 3D human poses, 2D image segmentations, and depth maps, as well as a digital twin environment.

TAPVid3D. TAPVid3D [33] is a dataset for tracking 3D points in space. It provides temporal 3D point tracking, constructed from PStudio, ADT, and DriveTrack [4]. It leverages official annotations to produce temporally aligned 3D point sequences, along with camera pose sequences and intrinsics. We use these annotations for our data generation. Note that we exclude the DriveTrack split because its camera poses are insufficiently accurate.

### D. Image Pairs Sampling

To ensure a balanced selection of image pairs based on their overlap ratio, we adopt the procedure as follows. First, we separate pairs with zero overlap and randomly sample a predefined number of them. Next, we partition all nonzerooverlap pairs into bins according to their overlap ratio. We then distribute the sampling quota across bins in proportion to the number of bins, sorting them by bin size in ascending order to prevent smaller bins from being overshadowed by larger ones. Finally, we either sample or exhaust each bin, carrying over any unused quota to subsequent bins. This step balances pairs of different overlap levels, mitigating issues caused by long-tail distributions. The main content of the full algorithm is shown in Listing 1.

### E. Rotation Angles

Beyond translation, we estimate two orientation angles: yaw and pitch. We do not model roll, as it typically remains small in real-world use cases (e.g., autonomous vehicles, robotics, wearable devices). Formally, let E ∈ R4×4 be the camera pose in world coordinates, which has the z−axis aligned with the gravity direction, and R its upper-left 3×3 submatrix:

##### R = E[0:3,0:3]. (5)

We then extract yaw and pitch by focusing on the camera’s forward (i.e., z-) axis:

 

 . (6)

0

- 0
- 1

zfwd = R

Yaw is defined as the angle of this rotated z-axis in the horizontal plane, measured around the gravity axis:

180 π

. (7) Pitch is the angle of zfwd relative to the ground plane:

yaw = arctan2 zfwd[1], zfwd[0] ×

pitch = arcsin

zfwd[2] ∥zfwd∥

180 π

. (8)

×

###### Table 7. Distribution of samples across different task types.

Depth Estimation Visual Correspondence Camera Orientation Camera Translation Object Movement Object Perception 1.6M 1.5M 4M 9M 13M 1.5M

- Table 8. Results of different fine-tuning strategies for InternVL2-8B. Increasing the number of trainable parameters yields larger gains from MultiSPA. Tuning the vision encoder significantly improves performance on fine-grained spatial understanding tasks, such as predicting camera motion vectors.

LLM-LoRA LLM-LoRA+MLP LLM-LoRA+MLP+VE LLM-Full+MLP+VE Average 56.11 57.46 65.41 66.72

Depth Estimation Comparison 74.00 73.34 81.00 80.84

Value 75.33 75.84 78.50 79.67 Visual Correspondence

Coordinate 49.00 52.67 63.67 71.33

MCQ 90.00 90.33 94.67 95.67 Camera Orientation

Direction 90.83 91.50 94.00 94.17

Degree 45.50 46.84 59.67 60.50 Camera Translation

Direction 85.89 87.66 91.78 92.56 Distance 42.33 46.33 72.00 77.67

Vector 18.00 24.67 43.00 53.00 Object Movement

Distance 40.42 38.17 39.08 47.83

Vector 12.92 15.84 19.50 26.75 Object Perception

Size 49.11 46.33 48.00 47.22

With these two angles, we can determine whether the camera rotates left or right, and tilt up or tilt down.

### F. BFS-Based Minimum-Coverage-Set Search

To ensure that an object’s full dimensions are captured across multiple images, we develop a breadth-first search (BFS) algorithm that identifies minimal sets of images whose combined coverage meets each dimension’s size requirement. In particular, for each axis (height, width, length), we track which subsets of object points are visible per image. If the difference between the minimum and maximum coordinates of all selected points along that axis meets a target threshold (based on the object’s 3D bounding box size), we consider it covered. Our BFS proceeds in two phases at each iteration:

- 1. Phase A: Coverage check. We examine the current sets and mark any that fully cover the object on the chosen axis. These sets are recorded as “minimal,” and any set that is a superset of a previously found minimal set is pruned.
- 2. Phase B: Expansion. We expand the remaining (uncovered) sets to the next level by appending additional images, while pruning those that cannot possibly achieve coverage in deeper levels.

This process continues until either no further expansion is

possible or the maximum number of images is reached. The final result is a collection of minimal sets that together span the object’s relevant dimension. Although we include pruning steps, the search still becomes expensive when considering sets of three or more images. Hence, we only use two images for object size perception in our data. Listing 2 shows a simplified implementation.

### G. Clustering-Based Rigid Body Segmentation

In TAPVid3D [33], all points in a sequence often belong to the same object or scene region, but they can be unevenly distributed (e.g., a human torso versus arms). To sample diverse motion patterns, we segment the point cloud into multiple rigid bodies, each undergoing a distinct motion. Our method accumulates inter-point distance changes over time and applies hierarchical clustering to identify coherent groups. We also filter groups with too less points to avoid noise. Listing 3 is a simplified code snippet.

### H. Different Fine-Tuning Strategies

For research efficiency, we adopt a resource-light setting: we fine-tune the LLM backbone with LoRA while freezing the vision encoder and the MLP projection layer. This limits the trainable components and thus the benefits from our

proposed datasets. We also evaluate alternative fine-tuning strategies; results are shown in Tab. 8.

We observe that increasing the number of trainable parameters yields larger gains from MultiSPA. Notably, tuning the vision encoder significantly improves performance, especially on challenging fine-grained spatial perception tasks such as camera and object motion prediction. For example, enabling vision-encoder tuning increases accuracy on camera translation vector prediction from 24.67% to 43.00%, highlighting the importance of the vision encoder for learning spatially relevant visual representations. These results are consistent with the scalability trends reported in the main paper.

### I. Comparison with Expert Models

Multi-SpatialMLLM is a general-purpose VLM capable of broad vision–language tasks. Enabled by MultiSPA data and training, it acquires multi-frame spatial understanding, such as camera motion estimation, depth perception, and visual correspondence (image matching), traditionally achieved by specialized 3D vision models. We compare these capabilities to expert systems; results are summarized in Tab. 9.

For camera vector prediction, we compare to the SOTA VGGT [67] using our fully fine-tuned 26B variant trained on 2.5M camera-vector samples. Because VGGT does not output metric-scale vectors, we report results after scale normalization. For the depth comparison task, we evaluate against Depth-Anything [74]; for image matching, we evaluate against LoFTR [61], using our fully fine-tuned 8B variant trained on 3M mixed MultiSPA samples. We could not include a 26B fully fine-tuned model on the mixed MultiSPA set due to computational limits at submission.

Nevertheless, our models match VGGT and LoFTR, and our 8B model remains competitive with Depth-Anything. Based on the scaling trends reported in the main paper, we expect further gains with larger models and data. Our goal is not to surpass specialized systems, but to show that a general-purpose VLM can attain comparable spatial perception performance while retaining broad vision–language capabilities.

Table 9. Comparison with expert models. For the camera vector prediction task, we compare VGGT to our 26B model; for depth comparison and image matching, we compare to our 8B model.

VGGT v.s. Ours Depth Anything v.s. Ours LoFTR v.s. Ours 86 v.s. 82 86 v.s. 76 71 v.s. 71

### J. More on VQA Benchmarks

More results on spatial benchmarks. We extend our evaluation to held-out spatial benchmarks, including the popular

60

50

40

Avg.perf.

30

20

Original

10

Fine-Tuned Model

0

0k 1k 2k 3k 4k 5k 6k 7k 8k 9k 10k Training iteration

Figure 6. Forgetting curves of Multi-SpatialMLLM training.

CVBench-3D [64] and ERQA [62]. As shown in Tab. 10, our model outperforms the baseline on both benchmarks, with particularly strong gains on the ERQA multi-frame subset, confirming its effective generalization to unseen benchmarks.

Table 10. Evaluation on spatial benchmarks.

Model CVBench-3D [64] ERQA [62] ERQA: M.V. [62]

InternVL-8B 78.2 35.8 13.5 Multi-SpatialMLLM 81.7 36.2 21.6

More results on general benchmarks. In addition to evaluating our models on the general VQA benchmarks reported in the main paper, we further test them on three more widely-used VQA benchmarks to verify that our fine-tuning preserves most of the general VLM capabilities, as shown in Tab. 11.

Table 11. Evaluation on standard VQA benchmarks.

Model MMMU [77] MME [19] MMVet [76]

InternVL-8B 47.7 85.8 62.0 Multi-SpatialMLLM 48.4 84.9 58.1

Forgetting curve. We also track the average performance on all 10 general VQA benchmarks throughout training, as shown in Fig. 6. The results demonstrate that our model maintains stable performance, alleviating concerns about forgetting. We hypothesize that the learned spatial skills are largely orthogonal to general capabilities, thereby minimizing interference with the model’s pre-existing knowledge during fine-tuning.

### K. Multi-Task Synergy

Further verification. To further validate the benefits of multi-task training, we use the multiple-choice visual correspondence (V.C.) task, which provides a strong baseline (random guess: 25%). We report results under different cross-task training configurations in Tab. 12.

We find that including a small set of target-task samples, 1K V.C. examples, which alone are insufficient to im-

prove accuracy, enables the model to leverage knowledge from other tasks (50K samples each). Experiments show that adding data from camera movement, object perception, and depth perception consistently boosts V.C. performance, further confirming the multi-task synergy highlighted in the main paper.

- Table 12. Multi-task synergy for visual correspondence (V.C.). Incorporating training samples from other tasks improves V.C. performance. Abbreviations: Cam., camera movement; Obj., object perception; Depth., depth perception.

Zero-Shot V.C. V.C. + Cam. V.C. + Obj. V.C. + Depth. 33.3 33.1 40.6 41.0 57.3

Benefits of different tasks. We also observe that depth perception contributes the largest gains to V.C. The main paper shows that adding samples from other datasets helps, but it also remains unclear which task contributes most.

To study this, we adopt the same multi-task setting but include 400K training samples from a single task at a time, V.C., depth perception, or object perception. As shown in Tab. 13, V.C. and depth samples yield the largest improvements, reinforcing the benefits of multi-task training and suggesting that V.C. and depth perception may be more fundamental spatial tasks, consistent with the classic Structurefrom-Motion pipeline [24].

- Table 13. Benefits of different tasks. “Single” denotes training without samples from other tasks. Abbreviations: V.C., visual correspondence; Depth., depth perception; Obj., object perception.

Single w. V.C. w. Depth. w. Obj. Camera Movement 9.30 15.00 15.00 12.00

Object Movement 17.50 21.88 22.59 18.54

L. Generalization to More Images.

In the main paper, we primarily train on two-frame samples. To test generalization to more frames, we evaluate camera direction prediction by asking fully fine-tuned models to predict motion between the first and last frame while inserting 1–4 intermediate frames. As shown in Table 14, accuracy remains at 85%, demonstrating robustness and generalization to varying frame counts.

- Table 14. Performance of camera prediction when adding different numbers of frames.

# Frames 1 2 3 4 Acc. 85.0 85.3 85.3 85.3

### M. Limitations

Our generated data in the main paper uses two-frame scenarios. We show that models trained on two images gener-

alize to more images, and our data-generation pipeline naturally scales to additional frames. However, future work is still needed to extend beyond pairs to exploit multi-view inputs for stronger spatial reasoning. Another limitation is that although we observe signs of the emergent phenomenon, further investigation is required to clarify what exact spatial abilities drive such emergence.

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47
- 48
- 49
- 50
- 51
- 52
- 53
- 54
- 55
- 56
- 57
- 58
- 59

|def sample_dataframe(df, all_overlap_samples, non_overlap_samples, overlap_min=0, overlap_max=100, interval=1):<br><br># 1) Sample pairs with overlap == 0 non_overlap_df = df[df["overlap"] == 0].copy() sampled_non_overlap_df = (non_overlap_df if len(non_overlap_df) <= non_overlap_samples<br><br>else non_overlap_df.sample(n=non_overlap_samples))<br><br># 2) Partition the remaining pairs (overlap != 0) into bins remaining_df = df[df["overlap"] != 0].copy() bins = np.arange(overlap_min, overlap_max + interval, interval) remaining_df["overlap_group"] = pd.cut(remaining_df["overlap"], bins=bins, include_lowest=True) remaining_df.dropna(subset=["overlap_group"], inplace=True) bin_groups = [] for ovlp_bin, group_df in remaining_df.groupby("overlap_group"):<br><br>bin_groups.append((ovlp_bin, group_df))<br><br>if not bin_groups: final_df = sampled_non_overlap_df.copy() final_df.drop(columns=["overlap_group"], errors="ignore", inplace=True) return final_df<br><br># 3) Distribute all_overlap_samples evenly across bins N = len(bin_groups) base_quota = all_overlap_samples // N remainder = all_overlap_samples % N bin_quotas = [base_quota] * N for i in range(remainder):<br><br>bin_quotas[i] += 1<br><br># 4) Sort bins by size (ascending) and sample bin_data = [] for i, (ovlp_bin, group_df) in enumerate(bin_groups):<br><br>bin_data.append({ "group_df": group_df, "quota": bin_quotas[i], "size": len(group_df)<br><br>}) bin_data.sort(key=lambda x: x["size"]) sampled_df = pd.DataFrame() leftover = 0 for info in bin_data:<br><br>group, quota, size = info["group_df"], info["quota"], info["size"] current = quota + leftover if size <= current:<br><br>sampled_df = pd.concat([sampled_df, group], ignore_index=True) leftover = current - size<br><br>else: sampled_df = pd.concat([sampled_df, group.sample(n=current)], ignore_index=True) leftover = 0<br><br>if leftover > 0: print(f"Warning: leftover {leftover} samples not used.")<br><br># 5) Combine sampled bins with zero-overlap samples final_df = pd.concat([sampled_df, sampled_non_overlap_df], ignore_index=True) final_df.drop(columns=["overlap_group"], errors="ignore", inplace=True) return final_df<br><br><br>|
|---|

###### Listing 1. The image pairs sampling algorithm for static scene data

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40

|def compute_coverage(points, mask, axis): """Returns the min-to-max spread along ’axis’ for points indicated by ’mask’.""" if not mask.any(): return 0.0 coords = points[mask][:, axis] return coords.max() - coords.min()<br><br>def covers_dimension(coverage, target_dim, tol): """Checks if ’coverage’ is within tolerance of the target dimension.""" return abs(coverage - target_dim) <= tol * target_dim<br><br>def bfs_min_coverage(images, visibility, points, obj_mask, axis, target_dim, tol, max_k=2): """ Finds minimal image sets up to size ’max_k’ that meet coverage criteria along ’axis’. ’images’ is a list of candidate frames, ’visibility’ maps frame->boolean mask, ’obj_mask’ indicates the object points in ’points’. """ # Prepare BFS queue: each item is (set_of_images, combined_mask, last_idx) queue = [] for i, img in enumerate(images):<br><br>mask_i = visibility[img] & obj_mask queue.append(([img], mask_i, i))<br><br>solutions = [] k = 1 while k <= max_k and queue:<br><br>next_level = [] for combo, comb_mask, last_idx in queue:<br><br>cov = compute_coverage(points, comb_mask, axis) if covers_dimension(cov, target_dim, tol):<br><br>solutions.append(combo) elif k < max_k:<br><br># Expand only if we have not reached max_k for j in range(last_idx + 1, len(images)):<br><br>mask_j = visibility[images[j]] & obj_mask next_mask = comb_mask | mask_j next_level.append((combo + [images[j]], next_mask, j))<br><br>queue = next_level k += 1<br><br>return solutions<br><br>|
|---|

###### Listing 2. Simplifed version of BFS-based minimum-coverage-set search with pruning.

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31

|def smooth_distance_changes(dist_t, dist_prev, smooth_factor=0.01): """Zeroes out small distance changes to reduce noise.""" diff = np.abs(dist_t - dist_prev) return np.where(diff > smooth_factor, diff, 0)<br><br>def rigid_body_segmentation(points, thr=0.1, smooth_factor=0.01): """ points: Shape (T, N, 3), with T time steps & N points. thr: Threshold for clustering distance. smooth_factor: Ignored small changes. Returns: A list of groups, each group is a list of point indices. """ T, N, _ = points.shape cum_loss = np.zeros((N, N))<br><br># Accumulate distance changes over time for t in range(1, T):<br><br>dist_t = squareform(pdist(points[t])) dist_prev = squareform(pdist(points[t - 1])) cum_loss += smooth_distance_changes(dist_t, dist_prev, smooth_factor)<br><br># Hierarchical clustering Z = linkage(squareform(cum_loss), method="average") labels = fcluster(Z, thr, criterion="distance")<br><br># Group points by label groups = [] for label_id in range(1, labels.max() + 1):<br><br>group = np.where(labels == label_id)[0].tolist() groups.append(group)<br><br>return groups<br><br>|
|---|

###### Listing 3. Rigid body segmentation with smoothing and hierarchical clustering.

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47
- 48
- 49
- 50
- 51
- 52
- 53

|# Depth Estimation-Dot TASK_DESCRIPTION = [<br><br>"<image>\nGiven an image with an annotated point, complete the question-answer task.",<br><br>] TEMPLATES = {<br><br>"questions": [ "What is the depth of the annotated point in the image (in mm)?",<br><br>], "answers": [<br><br>"The depth of the annotated point is ‘{depth}‘ mm.", ]<br><br>} # Visual Correspondence Multiple-Choice TASK_DESCRIPTION = [<br><br>"Image-1: <image>\nImage-2: <image>\nGiven these two images, find the corresponding points between them.",<br><br>] TEMPLATES = {<br><br>"questions": [ "Which point labeled A, B, C, or D in Image-2 corresponds to the circle point in Image-1? Please answer with the correct label from Image-2.",<br><br>], "answers": [<br><br>"The correct point is labeled ‘{correct_label}‘.", ]<br><br>} # Object Perception TASK_DESCRIPTION = [<br><br>"Assume the scene remains unchanged. Your task is to determine the spatial properties based on the images. You need to integrate and analyze information from all provided images to get the answer.",<br><br>] QUESTION_TEMPLATES = [<br><br>"What is the {dimension} (in millimeters) of the {object_category} itself commonly visible in these images?",<br><br>] ANSWER_TEMPLATES = [<br><br>"The {dimension} is approximately ‘{value_mm}‘ millimeters.",<br><br>] # Object Movement-Coordinate-Distance TASK_DESCRIPTION = [<br><br>"Image-1: <image>\nImage-2: <image>\nGiven two images, analyze the movements of objects in the images and the cameras that captured them. The movement should be relative to the first image. Note that the objects in the images and the camera may or may not have moved.",<br><br>] QUESTION_TEMPLATES = [<br><br>"How far did the point at [ {x1} , {y1} ] in Image-1 travel between the two shots? The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height."<br><br>] ANSWER_TEMPATES = [<br><br>"The point traveled a total of ‘{total_distance}‘ mm.", ]<br><br>|
|---|

###### Listing 4. Part of the templates used by MultiSPA dataset

Depth Comparison-Coordinate

[Figure 65]

|human: <image> Given an image with two points specified by their coordinates, determine which point is closer to or farther from the camera. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the yaxis represents the height. Which of these coordinates is more remote: [ 9 , 40 ] or [ 961 , 34 ]?<br><br>gpt: The point at `[ 961 , 34 ]` has the longer distance to the camera.|
|---|

Depth Comparison-Dot

[Figure 66]

|human: <image> Examine the image with two annotated points and decide which point is positioned closer to the camera and which one is positioned farther from it. Can you identify which annotated point has the shorter distance to the camera?<br><br>gpt: The annotation `B` marks the closer point.|
|---|

Depth Estimation-Coordinate

|human: <image> Given an image and a specific point's 2D coordinates, perform. The coordinates [ x , y ] are<br><br>normalized to 0-1 and then scaled by 1000, with [ 0 , 0 ] starting at the top-left. The x-axis represents the width, and the y-axis represents the height. What is the depth at point [ 204 , 486 ] in the image (in mm)?<br><br>gpt: Measured depth of this point is `1941` mm|
|---|

[Figure 67]

Depth Estimation-Dot

[Figure 68]

|human: <image> For the image with a noted point, ascertain the depth-related answers. Identify the depth value for the annotated point in millimeters.<br><br>gpt: The depth value of the marked point is `2249` mm.|
|---|

###### Figure 7. Data samples of depth perception.

Visual Correspondence-Coordinate

[Figure 69]

|human: Image-1: <image> Image-2: <image> Identify equivalent point positions between images. The [ x , y ] coordinates are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at top-left. Width follows x-axis, height follows y-axis. Find the matching position in Image-2 for point [ 314 , 111 ] from Image-1.<br><br>gpt: The point has moved to `[ 363 , 70 ]` in Image-2.|
|---|

Visual Correspondence-Dot

[Figure 70]

|human: Image-1: <image> Image-2: <image> Determine the corresponding points between these images. Which point labeled A, B, C, or D in Image-2 corresponds to the circle point in Image-1? Please answer with the correct label from Image-2.<br><br>gpt: The correct point in Image-2 is `D`.|
|---|

###### Figure 8. Data samples of visual correspondence.

- Camera Translation Direction-X axis

|human: Image-1: <image> Image-2: <image> Compare these images to infer the camera's motion. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. Is the camera vantage now elevated or depressed relative to its old vantage?<br><br>gpt: It exhibits a noticeable move `up` along the vertical axis.|
|---|

[Figure 71]

- Camera Translation Direction-Y axis

[Figure 72]

|human: Image-1: <image> Image-2: <image> Observe and describe the camera's motion from one image to the other. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. If we consider the vantage, is it pushing in or pulling away from the center of the view?<br><br>gpt: The camera moved `forward` in depth.|
|---|

- Camera Translation Direction-Z axis

|human: Image-1: <image> Image-2: <image> Observe and describe the camera's motion from one image to the other. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. Between these two frames, is the camera more to the left or to the right now?<br><br>gpt: It shows a noticeable move `left` along the horizontal axis.|
|---|

[Figure 73]

###### Figure 9. Data samples of camera movement-translation direction.

Camera Orientation Direction-Yaw

|human: Image-1: <image> Image-2: <image> Observe and describe the camera's motion from one image to the other. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. Yaw is an angle describing the camera's rotation left or right around vertical. Please tell me if it pivoted left or pivoted right?<br><br>gpt: We confirm the camera has undergone a `left` yaw rotation.|
|---|

[Figure 74]

Camera Orientation Direction-Pitch

[Figure 75]

|human: Image-1: <image> Image-2: <image> Examine how the camera shifted its position/orientation between these shots. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. By pitch angle, we refer to the camera's up-down rotation relative to the ground. Are we seeing an upward tilt or a downward tilt?<br><br>gpt: It shows a notable pitch rotation of `down`.|
|---|

###### Figure 10. Data samples of camera movement-orientation direction.

Camera Orientation Degree-Yaw

|human: Image-1: <image> Image-2: <image> Determine the change in camera viewpoint across these two images. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. How many degrees does the camera revolve side-to-side from its original orientation?<br><br>gpt: The camera's left-right rotation measures about `61` degrees.|
|---|

[Figure 76]

Camera Orientation Degree-Pitch

[Figure 77]

|human: Image-1: <image> Image-2: <image> Examine how the camera shifted its position/orientation between these shots. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. Pitch is the angle capturing up-down spin. Could you share the camera's pitch rotation in degrees?<br><br>gpt: Observations indicate a pitch rotation of `1` degrees.|
|---|

###### Figure 11. Data samples of camera movement-orientation degree.

Camera Translation Distance

|human: Image-1: <image> Image-2: <image> Identify the differences in camera pose when comparing these images. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. Does the camera have a large or small travel distance, and how much is it?<br><br>gpt: Comparing both images, the camera moved roughly `742` mm in distance.|
|---|

[Figure 78]

Camera Translation Vector

|human: Image-1: <image> Image-2: <image> Examine how the camera shifted its position/orientation between these shots. The movement should be relative to the first image. Note that the objects in the images are assumed to be static. The first image orients X->right, Y->down, Z>forward. Please specify that vector showing exactly how the camera moved in all three dimensions.<br><br>gpt: Summarily, the camera's movement vector is `[ 18 , 148 , 287 ]` mm.|
|---|

[Figure 79]

###### Figure 12. Data samples of camera movement-translation distance and vector.

Object Perception-Height

|human: Image-1: <image> Image-2: <image> Determine the spatial properties of the scene based on the images. Collate and interpret data from all provided images to conclude the answer. Find the height (in millimeters) of the monitor which is commonly visible in these images.<br><br>gpt: The height is understood to be `575` millimeters.|
|---|

[Figure 80]

Object Perception-Length

|human: Image-1: <image> Image-2: <image> Find the spatial properties of the scene using the images. You need to gather and interpret information from all provided images to obtain the answer. Calculate the length (in millimeters) of the desk that is commonly visible in these images.<br><br>gpt: The length is figured to be `1591` millimeters.|
|---|

[Figure 81]

Object Perception-Width

|human: Image-1: <image> Image-2: <image> Using the images, identify the spatial properties of the scene. Synthesize and scrutinize information from all provided images to deduce the answer. What is the width (in millimeters) of the bed that is commonly visible across these images?<br><br>gpt: From my estimation, the width is `1595` millimeters.|
|---|

[Figure 82]

###### Figure 13. Data samples of object perception.

Object Movement-Coordinate-Distance (PStudio)

|human: Image-1: <image> Image-2: <image> Evaluate these two images to determine how objects and camera positioning have changed, using the first image as baseline. Be aware that motion may affect either, both, or neither. Could you give me the magnitude of the point's displacement from [ 416 , 846 ] in Image-1? The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.<br><br>gpt: The total movement of the point measures `1147` mm.|
|---|

[Figure 83]

Object Movement-Coordinate-Distance (ADT)

|human: Image-1: <image> Image-2: <image> Investigate the movement patterns in these two images, considering both object and camera motion relative to the first frame. Note that changes could affect either, both, or neither. Please specify the total movement distance of the point at [ 343 , 881 ] in Image-1. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the yaxis represents the height.<br><br>gpt: The total distance covered by the point is `114` mm.|
|---|

[Figure 84]

Object Movement-Dot-Distance (PStudio)

[Figure 85]

|human: Image-1: <image> Image-2: <image> Map the positional changes between these two images, considering both object and camera movement from their initial state. Either, both, or neither may have shifted. Could you measure the total displacement for the annotated point in Image-1's movement?<br><br>gpt: The point's total displacement measures `1219` mm.|
|---|

Object Movement-Dot-Distance (ADT)

|human: Image-1: <image> Image-2: <image> Identify any spatial changes between these two images, examining both object and camera motion from their initial positions. Remember that movement isn't required. I'm interested in the point-to-point distance of the annotated point in Image-1's shift.<br><br>gpt: The measured distance of the point's movement is `17` mm.|
|---|

[Figure 86]

###### Figure 14. Data samples of object movement-distance.

Object Movement-Coordinate-Vector (PStudio)

|human: Image-1: <image> Image-2: <image> Study these two images and identify any changes in position of objects or camera, using the first image as reference. Both elements may or may not show movement. Given X=right, Y=down, Z=forward, I'd like you to list the point at [ 408 , 435 ] in Image-1's total displacement vector `[ x , y , z ]` in mm. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The xaxis represents the width, and the y-axis represents the height.<br><br>gpt: We have `[ 161 , 143 , -109 ]` mm as the shift.|
|---|

[Figure 87]

Object Movement-Coordinate-Vector (ADT)

[Figure 88]

|human: Image-1: <image> Image-2: <image> Compare these two images and describe how objects and the camera have moved relative to their positions in the first image. Keep in mind that either or both may have changed position. Remember X->right, Y->down, Z->forward. Please give me the exact displacement as `[ x , y , z ]` in mm for the point at [ 677 , 792 ] in Image-1. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height. gpt: The point's spatial transformation vector is `[ -372 , 66 , -149 ]` mm.|
|---|

Object Movement-Dot-Vector (PStudio)

|human: Image-1: <image> Image-2: <image> Analyze the positional shifts between these two frames, tracking both object and camera movement relative to the first image. Note that either or neither might have moved. Between these images, using X=right, Y=down, Z=forward, what's the vector that the annotated point in Image-1 traveled in mm?<br><br>gpt: The point's positional change vector is `[ -1 , -21 , 206 ]` mm.|
|---|

[Figure 89]

Object Movement-Dot-Vector (ADT)

[Figure 90]

|human: Image-1: <image> Image-2: <image> Study these two images and identify any changes in position of objects or camera, using the first image as reference. Both elements may or may not show movement. The first image is oriented such that positive X is right, Y is down, and Z is forward. Could you provide the annotated point in Image1's movement vector in 3D coordinates?<br><br>gpt: The overall shift is `[ 167 , 320 , 564 ]` mm.|
|---|

###### Figure 15. Data samples of object movement-vector.

### References

- [1] anthropic. Claude 3.5 sonnet. https : / / www . anthropic . com / news / claude - 3 - 5 - sonnet,

2024. 5

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv:2309.16609, 2023. 2
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv:2308.12966, 2023. 2
- [4] Arjun Balasingam, Joseph Chandler, Chenning Li, Zhoutong Zhang, and Hari Balakrishnan. Drivetrack: A benchmark for long-range point tracking in real-world videos. In CVPR,

2024. 9

- [5] Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. arXiv:2412.03548, 2024. 3
- [6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. Pi0: A vision-languageaction flow model for general robot control. https: //physicalintelligence.company/blog/pi0,

2024. 1, 2

- [7] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In ICRA, 2025. 3
- [8] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024. 1, 2, 3, 8
- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024. 7
- [10] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 2024. 2, 4, 6

- [11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR,

2024. 2

- [12] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In NeurIPS, 2024. 1, 2, 3, 5, 6, 7, 8
- [13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 2, 4, 5, 7, 9
- [14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. NeurIPS, 20243. 1, 2, 4
- [15] Google Deepmind. Gemini 2.0: our new ai model for the agentic era. https://blog.google/technology/ google - deepmind / google - gemini - ai update - december - 2024 / ceo - message, 2024. 5
- [16] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv:2303.03378, 2023. 2
- [17] Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. arXiv:2406.05756, 2024. 3
- [18] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv:2505.20279, 2025. 3
- [19] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 11
- [20] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In ECCV, 2024. 2, 3, 7
- [21] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv:2305.04790, 2023. 2
- [22] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, 2018. 7
- [23] Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu

- Guo, et al. Imagebind-llm: Multi-modality instruction tuning. arXiv:2309.03905, 2023. 2
- [24] R. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, 2003. 2, 12
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR,

2022. 5

- [26] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2
- [27] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv:2302.14045, 2023. 1
- [28] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 2
- [29] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, 2017. 3
- [30] Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh. Panoptic studio: A massively multiview system for social motion capture. In Proceedings of the IEEE international conference on computer vision, 2015. 2, 4, 5, 7, 9
- [31] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An opensource vision-language-action model. In CoRL, 2024. 2
- [32] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 3
- [33] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Jo˜ao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. arXiv preprint arXiv:2407.05921, 2024. 2, 4, 5, 9, 10
- [34] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV,

2020. 2

- [35] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv:2408.03326, 2024. 1, 5
- [36] Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and

- Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2024. 1, 2
- [37] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023. 7
- [38] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models. arXiv:2409.09788, 2024. 2, 3
- [39] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, 2024. 2, 5
- [40] JingLi Lin, Chenming Zhu, Runsen Xu, Xiaohan Mao, Xihui Liu, Tai Wang, and Jiangmiao Pang. Ost-bench: Evaluating the capabilities of mllms in online spatio-temporal scene understanding. arXiv:2507.07984, 2025. 3
- [41] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. TACL, 2023. 2, 3
- [42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2
- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR,

2024. 4

- [44] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281,

2023. 7

- [45] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. ICLR, 2019. 5
- [46] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv:2403.05525, 2024. 2
- [47] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024. 7
- [48] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. In NeurIPS, 2024. 3
- [49] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019. 7
- [50] OpenAI. Gpt-4o. https://openai.com/index/ hello-gpt-4o/, 2024. 1, 4, 6
- [51] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In ICCV, 2023. 2, 4, 5, 7, 9
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [53] Santhosh Kumar Ramakrishnan, Erik Wijmans, Philipp Kraehenbuehl, and Vladlen Koltun. Does spatial cognition emerge in frontier models? arXiv:2410.06468, 2024. 3
- [54] Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 2024. 3
- [55] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the” edge” of open-set object detection, 2024. 2
- [56] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 2
- [57] Christoph Schuhmann, Andreas K¨opf, Richard Vencu, Theo Coombes, and Romain Beaumont. Laion coco: 600m synthetic captions from laion2b-en, 2022. 2
- [58] Fatemeh Shiri, Xiao-Yu Guo, Mona Far, Xin Yu, Reza Haf, and Yuan-Fang Li. An empirical analysis on spatial reasoning capabilities of large multimodal models. In EMNLP,

2024. 1, 3

- [59] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In ECCV, 2024. 1, 2
- [60] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. arXiv:2411.16537, 2024. 3
- [61] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, 2021. 11
- [62] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 11
- [63] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https:// github.com/InternLM/InternLM, 2023. 1, 2, 5
- [64] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 3, 11
- [65] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023. 2

- [66] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Niessner. Rio: 3d object instance re-localization in changing indoor environments. In ICCV,

2019. 7

- [67] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 7, 11
- [68] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025. 2
- [69] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv:2206.07682, 2022. 8
- [70] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv:2505.23747, 2025. 3
- [71] Runsen Xu, Zhiwei Huang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Vlm-grounder: A vlm agent for zero-shot 3d visual grounding. In CoRL, 2024. 2
- [72] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv:2412.14171, 2024. 3
- [73] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 2
- [74] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 11
- [75] Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv:2504.15280, 2025. 3
- [76] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 11
- [77] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 11
- [78] Wenyu Zhang, Wei En Ng, Lixin Ma, Yuwen Wang, Jungqi Zhao, Boyang Li, and Lu Wang. Sphere: A hierarchical evaluation on spatial perception and reasoning for visionlanguage models. arXiv preprint arXiv:2412.12693, 2024. 3
- [79] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv:2304.10592, 2023. 2

[80] Yiming Zuo, Karhan Kayan, Maggie Wang, Kevin Jeon, Jia Deng, and Thomas L Griffiths. Towards foundation models for 3d vision: How close are we? arXiv preprint arXiv:2410.10799, 2024. 2, 3

