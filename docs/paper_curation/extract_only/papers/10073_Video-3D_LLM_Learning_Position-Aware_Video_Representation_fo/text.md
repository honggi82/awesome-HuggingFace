## Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding

Duo Zheng* Shijia Huang* Liwei Wang† The Chinese University of Hong Kong

{dzheng23, sjhuang, lwwang}@cse.cuhk.edu.hk

# arXiv:2412.00493v2[cs.CV]27Mar2025

### Abstract

Initialize Image Video

[Figure 1]

VideoLLM

[Figure 2]

The rapid advancement of Multimodal Large Language Models (MLLMs) has significantly impacted various multimodal tasks. However, these models face challenges in tasks that require spatial understanding within 3D environments. Efforts to enhance MLLMs, such as incorporating point cloud features, have been made, yet a considerable gap remains between the models’ learned representations and the inherent complexity of 3D scenes. This discrepancy largely stems from the training of MLLMs on predominantly 2D data, which restricts their effectiveness in comprehending 3D spaces. To address this issue, in this paper, we propose a novel generalist model, i.e., Video3D LLM, for 3D scene understanding. By treating 3D scenes as dynamic videos and incorporating 3D position encoding into these representations, our Video-3D LLM aligns video representations with real-world spatial contexts more accurately. In addition, we have implemented a maximum coverage sampling technique to optimize the trade-off between computational cost and performance. Extensive experiments demonstrate that our model achieves state-of-the-art performance on several 3D scene understanding benchmarks, including ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, and SQA3D. Our code is available at https://github.com/LaVi-Lab/Video-3DLLM.

[Figure 3]

[Figure 4]

[Figure 5]

###### MLLM3DLLM

Pretrain

Pretrain

Initialize

Point cloud/Voxel-level Representation

Position-Aware Video Representation

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Video-3D LLM

Finetune

Finetune

|(𝑥 ,𝑦 ,𝑧 )|(𝑥 ,𝑦 ,𝑧 )|
|---|---|
|(𝑥 ,𝑦 ,𝑧 )|(𝑥 ,𝑦 ,𝑧 )|

|(𝑥 , 𝑦 , 𝑧 )|(𝑥 , 𝑦 , 𝑧 )|
|---|---|
|(𝑥 , 𝑦 , 𝑧 )|(𝑥 , 𝑦 , 𝑧 )|

Global Coordinates

(a) Previous Work (b) Our Method

Figure 1. Comparison of previous work and our method: (a) Previous 3D LLMs are initialized on MLLMs trained solely on imagetext pairs, and learn point cloud or voxel representations via finetuning on 3D scenes. The 3D point clouds are reconstructed from RGB-D videos. (b) Our method directly utilizes video frames and 3D coordinates as input, where the 3D coordinates are converted from depths through coordinate transformation. We then transfer the ability of video understanding to 3D scene understanding by injecting position information into video representations.

cessitate spatial understanding and reasoning in 3D environments.

Recent studies [9, 10, 15–17, 20, 40, 42, 51] have focused on adapting MLLMs for enhanced 3D scene understanding. As depicted in Figure 1 (a), these approaches develop comprehensive 3D scene-level representations using a variety of techniques. These include harnessing features from point clouds [9, 10, 42], lifting multi-view image features to 3D space [15, 16, 51], and exploiting characteristics from recognized objects [17, 18, 20, 40].

### 1. Introduction

The rapid development of Multimodal Large Language Models (MLLMs) [1, 11, 14, 23, 24, 29, 30, 33, 39, 47] has demonstrated substantial capabilities in various multi-modal tasks, attracting significant attention from both academia and industry sectors. However, despite these advancements, recent studies [4, 27, 28, 32] indicate that current MLLMs face challenges when addressing tasks that ne-

Although significant advancements have been achieved, a noticeable gap exists between the representations learned by MLLMs and the complexity of 3D scenes. This gap stems from the fact that MLLMs are primarily trained on fundamentally different data types, namely 2D images. While it is possible to further finetune MLLMs with 3D data, such as point clouds or voxels, the limited availability

*Equal contribution. †Corresponding author.

of labeled 3D scene data poses a challenge. Consequently, the 2D visual knowledge embedded in MLLMs fails to fully unleash its potential in understanding 3D environments.

In parallel, the abundance of video data has spurred interest in adapting Video LLMs to different domains, e.g., 3D question answering [23, 28, 30] and robotic manipulation [22, 41, 54]. These methods benefit from extensive internet video datasets and pre-trained video models, revealing the immense potential for extending video modality to 3D modeling. However, as early attempts, they are still far from creating a model capable of handling diverse 3D tasks. Moreover, the absence of integrated spatial information—such as 3D locations and spatial relationships—in video representations constrains their capability to fully comprehend the 3D physical world. For instance, tasks that require an intricate understanding of 3D spatial relationships cannot rely solely on RGB data. This limitation underscores the necessity for incorporating more comprehensive spatial modeling into Video LLMs to enhance their effectiveness in 3D applications.

Therefore, in this paper, we propose a generalist model for 3D scene understanding, namely Video-3D LLM. Our primary motivation is to effectively leverage the spatiotemporal priors inherent in Video LLMs and advance this modeling approach to a variety of 3D scene understanding tasks. As shown in Figure 1, our model is based on a Video LLM framework that processes 3D videos, i.e., video frames accompanied by the corresponding 3D global coordinates. The frames are sampled from raw RGB videos and the 3D coordinates are obtained by backprojecting each pixel in the depth images1. To establish the correspondence in visual appearance and position information, we learn positionaware video representations by injecting 3D global coordinates into video features. Specifically, we encode the coordinate to 3D position encoding and add it to the video representations, serving as the input for the Video LLM.

Our model offers several significant advantages. Firstly, it aligns video representations with their real-world spatial contexts, thereby equipping our model to handle various 3D tasks such as 3D visual grounding, 3D dense captioning, and 3D question answering. Secondly, it maintains both temporal and spatial contextual information in the video data, which helps to reduce the discrepancy between the pre-training data and actual 3D scenes. Additionally, we have developed a maximum coverage sampling strategy for frame selection. This approach views frame selection as a maximum coverage problem and adopts a greedy algorithm for its resolution. This strategy ensures the selection of the most informative frames, thus improving the model’s capacity to discern diverse and essential spatio-temporal features within the video, while also ensuring efficient inference per-

1Indoor 3D datasets [12, 38, 44] are captured as RGB-D streams and then reconstructed into 3D point clouds.

formance.

Our approach is to train a single model in a multi-task manner on varying 3D scene understanding tasks, including 3D question answering, 3D dense captioning, and 3D visual grounding. Extensive experiments demonstrate that our Video-3D LLM achieves state-of-the-art performance on five 3D scene understanding benchmarks, i.e., ScanRefer [5], Multi3DRefer [46], Scan2Cap [6], ScanQA [2] and SQA3D [31]. Notably, our method surpasses the previous state-of-the-art LLaVA-3D [51] by using only 26% of its 3D data (223k vs. 859k), achieving improvements of 4.1% Acc@0.25 on ScanRefer, 4.6 CIDEr@0.5IoU on Scan2Cap, 2.9% EM on ScanQA, and 3.0% EM on SQA3D. The impressive performance reveals the immense potential for adapting video models to 3D modality, establishing a new paradigm in 3D scene understanding.

### 2. Related Work

LLMs for 3D Scene Understanding. Recently, there has been a growing interest in integrating 3D information in LLMs [9, 10, 16–18, 20, 40, 42, 51], which advances the progress of 3D scene understanding. 3D-LLM [16] introduces the LLM-based model for 3D physical world, which takes 3D features from rendered 2D images as input. PointLLM [42] utilizes a point encoder with a strong LLM for point cloud understanding. LL3DA [9] leverages a Q-former to extract useful information from point cloud features, endowing humans with the capability to interact in 3D environments. Grounded 3D-LLM [10] further introduces a projection module based on 3D detectors, which allows for generating object proposals from pointlevel features. Chat3D[40], LEO [18], and ChatScene [17] take use of off-the-shelf 3D detectors for proposal generation, and then incorporate the object-centric representations into LLMs. LLaVA-3D [51] introduces 3D-patch representations, which aggregate 2D-patch features in voxel space. Robin3D [20] tries to enhance 3D scene understanding via data generation. It is important to note that existing 3D Large Language Models (3D LLMs) typically transform 3D scenes into voxel-level or point cloud-level 3D representations as input for modeling purposes. However, these approaches create a disconnect with pre-trained multimodal large language models (MLLMs), which are primarily trained on extensive 2D datasets, such as images, and only fine-tuned on a limited amount of 3D scene data. To address this challenge, our method incorporates 3D information (e.g., coordinates in Figure 1 (b)) into a new video representation. This enhancement maximizes the use of pre-trained 2D Video LLMs, leveraging their full potential.

Video-Language Models for 3D Understanding. We have witnessed the rapid development of Video LLMs [25, 26, 30, 45, 47]. There is also a trend of leverag-

ing Video LLMs for 3D tasks, including 3D question answering [23, 28, 30] and robotic manipulation [22, 41, 54]. LLaVA-OneVision [23] and Oryx MLLM [30] incorporate 3D question-answering datasets into instruction data, which deliver competent results on 3D question-answering tasks. However, these models do not capture detailed 3D spatial information, which limits their performance in addressing other 3D tasks that require precise spatial alignments. Furthermore, recent work [28] has emphasized the importance of identifying key object correspondences across frames through visual prompting. In contrast, our approach directly incorporates 3D positional information into video representation learning, which enhances the capability to tackle more complex tasks that demand a thorough spatial understanding of the 3D environment.

### 3. Method

We propose a generalist model for 3D scene understanding, namely Video-3D LLM, which comprises a visual encoder, a 3D position encoding module, and a Video LLM backbone. As shown in Figure 2, the model takes the input as: 1) video streams captured from the 3D scene and 2) the associated 3D coordinate maps obtained through back-projection across all frames. In contrast to prior work [9, 16, 51] that converts video frames into explicit 3D representations (e.g., point clouds or voxels), our model directly processes temporal sequences of video frames, preserving both temporal and spatial contextual information in the video streams.

Given that entire frame sequences can be redundant, implementing an effective frame selection strategy is crucial,

- as it significantly influences both performance and computational efficiency. Subsequently, our goal is to enhance the Video LLM with position awareness and adapt it to various 3D scene understanding tasks. This is achieved by encoding spatial coordinates into the 3D position encoding (3D-PE) and integrating them into the video representation learning. In this section, we detail three key components of our approach: the frame sampling strategy (3.1), position-aware video representation (3.2), and the multi-task training (3.3).

- 3.1. Frame Sampling Strategy

Representing 3D scenes as video sequences presents two primary challenges: (1) Due to GPU memory constraints, the Video LLM can only process a limited number of frames

- at a time. This necessitates the sampling of a subset of frames from the extensive raw video sequence to manage resources effectively. (2) It is crucial for the video sequence to encompass as much of the entire 3D scene as possible, since any omission of scene content could result in a significant and irreversible decline in model performance. To address these challenges, we introduce a maximum coverage strategy for frame sampling. This approach involves preprocessing the selected frames offline and applying the

strategy consistently during both the training and inference phases to ensure comprehensive scene coverage and efficient memory usage.

Frame Sampling as Maximum Coverage Problem. Given a raw RGB-D video, each frame captures a portion of the 3D scene. We aim to select fewest possible frames that maximize coverage of the 3D scene, which could be formulated as a maximum coverage problem. Formally, let F = {f1,f2,...,fn} represent the set of all frames, and V = {v1,v2,...,vm} represent the set of all voxels in the 3D space. Each frame fk covers a subset of voxels Vk ⊆ V . To identify the contained voxels of each frame, we first back-project each pixel from the depth image to global coordinates, then discretize these coordinates into voxel grids.

The objective is to find a subset of frames S ⊆ F such that the union of covered voxels f

k∈S Vk is maximized.

Greedy Solution. Since the maximum coverage problem is NP-hard, we employ a greedy algorithm to solve it, which can obtain an approximation ratio of 1 − 1/e [21]. As illustrated in the Algorithm 1, the approach iteratively selects the frame with the largest increase in uncovered voxel coverage. Frames are added until the desired number is reached or the coverage ratio exceeds a predefined threshold. This stop condition ensures a balance between computational efficiency and the coverage for varying scenes.

Algorithm 1 Maximum Coverage Sampling

Require: Set of frames F = {f1, f2, . . . , fn}, voxel sets Vk for

each frame fk, budget K

Ensure: Subset S ⊆ F maximizing voxel coverage

- 1: Initialize S ← ∅
- 2: Initialize U ← ∅ {Set of covered voxels}
- 3: while size of S is less than K do
- 4: Select f∗ = arg maxfk∈F\S |Vk \ U|
- 5: Add f∗ to S
- 6: Update U ← U ∪ Vf∗
- 7: if Stop condition is met then
- 8: break
- 9: end if
- 10: end while
- 11: return S

#### 3.2. Position-Aware Video Representation

After completing frame sampling, we obtain a sequence of RGB frames, depth images, and the camera’s intrinsic and extrinsic parameters. To create a position-aware video representation, we first transform the depth information into 3D coordinates within a global coordinate system. We then encode the visual embeddings along with the 3D position encodings (3D-PE) to enhance spatial awareness.

Camera Coordinate Transformation. Given a depth image dk ∈ RH×W, an extrinsic matrix Tk ∈ R4×4, and a camera intrinsic matrix K, we can backproject each pixel at

VG: “<ground>” QA: “left side”

𝐞

VideoLLM

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Video Frames Caption: “a white table”

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Position-Aware Video Representations

Word Embedding

{𝐞 }

3D-PE

Vision Encoder

VideoLLM

(0.8, -3.2, 0.4)

“Given an object located at <coord>, describe the object in detail.”

Global Coordinates

###### (b) Task Example: 3D Dense Captioning

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

3D-PE Module

𝐞

[Figure 17]

| | | | |
|---|---|---|---|
| | | |[Figure 18]|
| | | | |
| | | | |

{𝐞 }

| | |
|---|---|
| | |

[Figure 19]

[Figure 20]

[Figure 21]

𝐞

| | |
|---|---|
| | |

Box Center (1.7, 1.2, 0.4)

Coordinate Transformation

𝐞

VG: “Identify the object according to the following description. A beige wooden working table. placed on the side of the room.” QA: “On what side of the bed is the guitar located?” Caption: “Given an object located at <coord>(0.8, -3.2, 0.4), describe the object in detail.”

[Figure 22]

|𝑂|O|O|O|O|
|---|---|---|---|---|

|𝐻|
|---|

|𝑂 𝐻|𝑂 𝐻|𝑂 𝐻|𝑂 𝐻|𝑂 𝐻|
|---|---|---|---|---|

Video-3D

<ground>

(a) Model Architecture

###### (c) Task Example: 3D Visual Grounding

Figure 2. The overview of the model architecture. (a) shows the integration of video sequence and global coordinates for creating positionaware video representations. (b) and (c) detail the examples of 3D dense captioning and 3D visual grounding, respectively. Our approach can generalize well to other 3D tasks.

the position (i,j) in the depth image to the global coordinates ck(i,j):

the coordinate onto a discrete grid. The encoding for the x coordinate is then defined as:

 

 

  

   . (1)

j i 1

dk(i, j) · K−1 ·

ck(i, j) 1

x 100002i/⌊

= Tk ·

, (3)

PE(x,2i) = sin

3⌋

d

1

x 100002i/⌊

We perform the above process for each sampling frame, resulting in a set of global coordinates {ck}lk=1 and their corresponding RGB images {fk}lk=1. Here, both ck and fk are in RH×W×3.

. (4)

PE(x,2i + 1) = cos

3⌋

d

Similar calculations are applied to the y and z coordinates. The PE of (x,y,z) are concatenated to obtain the final coordinate embeddings ecoordk ∈ RH

′×W′×d. Lastly, the coordinate embeddings are added to the visual embeddings to form the position-aware video representations, denoted by:

Visual Embedding. We first encode each frame into visual embeddings via a Vision Transformer (ViT) [13]. In concrete, given a frame fi ∈ RH×W×3, the image will first be split into a series of patches at the patch size P, which are then fed into the ViT to produce visual embeddings eimgk ∈ RH

evisk = eimgk + ecoordk . (5)

′×W′×d, where H′ = HP , W′ = WP and d is the feature dimension.

#### 3.3. Multi-Task Training

Our approach is to build a generalist model capable of handling multiple tasks with the single learned model. To achieve this, we train our model on a diverse, multi-task dataset that encompasses various 3D scene understanding tasks. During training, we randomly sample a single task type for each batch and train exclusively on data specific to that task. For general 3D scene understanding tasks, such as 3D question answering and 3D dense captioning, we use cross-entropy loss to supervise text generation. For the 3D visual grounding task, to locate more accurately, we use the designed 3D visual grounding loss to supervise the 3D proposal selection.

3D Position Encoding. Since the video frame is divided into image patches, we need to pool the coordinate information of each image patch. For each global coordinate map ck ∈ RH×W×3, we divide the coordinates into patches identical to those of the image patches. Subsequently, we compute the average coordinates for each patch with:

1 P2

c′k(i, j) =

ck(u, v), (2)

(u,v)∈P(i,j)

where P(i,j) denotes the patch region corresponding to position (i,j) and c′k ∈ RH

′×W′×d are the average coordinates for P(i,j). Due to the small size of the patches, the averaged coordinates retain sufficiently precise positional information. We also explored alternative coordinate pooling methods in our ablation study.

Cross-Entropy Loss. Given a position-aware video representation and a textual instruction, the language modeling objective aims to optimize the cross-entropy loss LCE:

We adopt sinusoidal position encoding [36] to encode the coordinates. For 3D coordinates (x,y,z), we first map

LCE = − log(y|{evisk }lk=1, {etextk }qk=1), (6)

where y is the ground truth response, {evisk }lk=1 are position-aware video representations for the video and

{etextk }qk=1 are text embeddings.

For the dense captioning task, we ask the model to describe objects based on their center coordinates. As shown in Fig 2 (b), we obtain the 3D position encoding of the bounding box center in the same manner as described in Sec 3.2. This position encoding is added to the embedding of the special token ⟨coord⟩ to provide location information.

3D Visual Grounding Objective. Previous studies [16, 50] have demonstrated that directly outputting 3D bounding boxes is quite challenging for LLM. To enable our model to perform 3D visual grounding, we follow previous work [17, 19, 52] to model the task as a proposal classification task–selecting the target objects from a list of detected proposals. As illustrated in Fig 2 (c), given a list of object proposals, we extract object features for each object from the visual embeddings. Specifically, for each object bk, we check each patch to see if more than 50% of its points are contained within bk, and then apply average pooling to the features of all selected patches, to obtain the 2D object features eobj-rgbk . Lastly, we add the 3D position encoding of the center coordinate eobj-coordi with eobj-rgbi to obtain the object representation eobji . During training, we utilize InfoNCE loss [35] to optimize the similarity between the ground truth object feature and the hidden states h of the ⟨ground⟩ token:

exp(f(eobjk ) · g(h)/τ) k∈O exp(f(eobjk ) · g(h)/τ)

LGrd = k∈O+

, (7)

where O+ and O respectively represent the sets of positive objects and all objects, f and g are two-layer learnable MLPs, and τ is the temperature.

### 4. Experiments

In this section, we first compare the overall performance of Video-3D LLM with top-tier models and also investigate the effectiveness of all components.

#### 4.1. Experimental Setup

Datasets. We conduct experiments across five 3D scene understanding benchmarks. For visual grounding, we test our model on ScanRefer [5] and Multi3DRefer [46], which require localizing objects in single-target and multiple-target scenarios, respectively. For dense captioning, we utilize the Scan2Cap [6] benchmark, which involves densely generating descriptions for all objects in 3D scenes. For question answering, we use the ScanQA [2] for spatial reasoning and SQA3D [31] for situated reasoning. All these datasets are sourced from the ScanNet [12], a richly annotated RGB-D video dataset containing 1,513 scans in 3D scenes. We preprocess video frames for each scan at 3 FPS and extract

the corresponding extrinsic and camera intrinsic parameters. For evaluation, we follow previous work [10, 17, 51] to adopt the validation sets for ScanRefer, Multi3DRefer, Scan2Cap, and ScanQA, and adopt the test set for SQA3D.

Metrics. We adopt widely used evaluation metrics for each of these benchmarks. For ScanRefer [5], we report thresholded accuracy metrics, specifically Acc@0.25 and Acc@0.5, where a prediction is considered correct if its Intersection over Union (IoU) with the ground truth exceeds 0.25 and 0.5, respectively. For Multi3DRefer [46], which involves grounding a variable number of target objects, we use the F1 score at IoU thresholds of 0.25 and 0.5. For Scan2Cap [6], we apply CIDEr@0.5IoU and BLEU4@0.5IoU (denoted as C@0.5 and B-4@0.5), combining traditional image captioning metrics with IoU between predicted and reference bounding boxes. For ScanQA [2], we use CIDEr [37] and exact match accuracy, referred to as C and EM, respectively. Finally, for SQA3D [31], we evaluate performance using exact match accuracy (EM).

Implementation Details. We build Video-3D LLM based on the LLaVA-Video 7B [47], an open-sourced video LLM based on the QWen2-7B [43]. We use the Adam optimizer to train our model for one epoch with a batch size of 16 and a warmup ratio of 0.03. During the warmup phase, the learning rates peak at 1e-5 for the LLM and 2e-6 for the vision encoder. All experiments are conducted on 8 A10080G GPUs. For 3D visual grounding and dense captioning, in training, we use the ground truth objects as the candidates. While in inference, we follow [17, 18] to employ Mask3D [34] to generate object proposals. The temperature τ for InfoNCE loss is 0.07.

#### 4.2. Comparison with State-of-the-art Methods

##### 4.2.1. Comparison Baselines

For a comprehensive comparison, we include both expert models designed for specific tasks and LLM-based models.

Expert Models. For ScanRefer [5], we compare our method with ScanRefer [5], MVT [19], 3DVG-Trans [48], ViL3DRel [8]. M3DRef-CLIP [46] further extends 3D grounding capabilities to multi-target scenarios. Scan2Cap [6] and ScanQA [2] provide initial benchmarks for the Scan2Cap and ScanQA datasets, respectively. 3D-VisTA [52] is pre-trained on large-scale scene-text pairs and then finetuned on specific tasks.

- 2D LLMs. Oryx [30] has included the ScanQA dataset in its training stage. We also test the zero-shot performance of LLaVA-Video [47].
- 3D LLMs. 3D-LLM [16] is the first LLM-based model for 3D scene undersanding. SceneLLM and LL3DA [9] enrich the 3D representations with point cloud features. Chat3D [40], LEO [18], and ChatScene [17] incorporate

ScanRefer Multi3DRef Scan2Cap ScanQA SQA3D Acc@0.25 Acc@0.5 F1@0.25 F1@0.5 B-4@0.5 C@0.5 C EM EM

3D Generalist

Method

Expert Models ScanRefer [5] 37.3 24.3 MVT [19] 40.8 33.3 3DVG-Trans [48] 45.9 34.5 ViL3DRel [8] 47.9 37.7 M3DRef-CLIP [46] 51.9 44.7 42.8 38.4 Scan2Cap [6] 22.4 35.2 ScanQA [2] 64.9 21.1 47.2 3D-VisTA [52] 50.6 45.8 34.0 66.9 69.6 22.4 48.5

- 2D LLMs Oryx-34B [30] – – – – – – 72.3 – – LLaVA-Video-7B [47] – – – – – – 88.7 – 48.5

- 3D LLMs

3D-LLM(Flamingo)[16] 21.2 – – – – – 59.2 20.4 – 3D-LLM(BLIP2-flant5)[16] 30.3 – – – – – 69.4 20.5 – Chat-3D [40] – – – – – – 53.2 – Chat-3D v2 [17] ✓ 42.5 38.4 45.1 41.6 31.8 63.9 87.6 – 54.7 LL3DA [9] ✓ – – – – 36.0 62.9 76.8 – – SceneLLM [15] ✓ – – – – – – 80.0 27.2 53.6 LEO [18] ✓ – – – – 38.2 72.4 101.4 21.5 50.0 Grounded 3D-LLM [10] ✓ 47.9 44.1 45.2 40.6 35.5 70.6 72.7 – – PQ3D [53] ✓ 57.0 51.2 – 50.1 36.0 80.3 – – 47.1 ChatScene [17] ✓ 55.5 50.2 57.1 52.4 36.3 77.1 87.7 21.6 54.6 LLaVA-3D [51] ✓ 54.1 42.4 – – 41.1 79.2 91.7 27.0 55.6 Video-3D LLM (MC) ✓ 57.9 51.2 57.9 52.4 40.2 80.0 100.5 29.5 57.7 Video-3D LLM (Uniform) ✓ 58.1 51.7 58.0 52.7 41.3 83.8 102.1 30.1 58.6

- Table 1. Overall performance comparison. “Expert models” are customized for specific tasks through task-oriented heads. “3D Generalist” means the model can perform multiple 3D tasks in a single model. LLaVA-Video is assessed in a zero-shot setting.

object representations into 3D LLMs. Grounded 3D-LLM [10], PQ3D [53] and LLaVA-3D [51] deliver impressive results on 3D visual grounding by co-training a 3D detector. All these methods apply 3D point cloud features or projecting multi-view image features into 3D space, while ours directly works on the video representations.

##### 4.2.2. Results

We present the overall comparison with leading methods in Table 1. “Video-3D LLM (Uniform)” is trained using uniform sampling with 32 frames, while “Video-3D LLM (MC)” is trained using maximum coverage sampling with a coverage ratio of 95% and a maximum frame number of 32. “Video-3D LLM (Uniform)” achieves state-of-the-art performance on a variety of tasks including 3D visual grounding, 3D dense captioning, and 3D question answering, while “Video-3D LLM (MC)” delivers similar results with only half the inference time (527ms vs. 1050 ms).

3D Visual Grounding. For 3D visual grounding, we follow previous work [17, 19, 52] to detect all objects and then make predictions over the object proposals2. Our model achieves the highest accuracy, with Acc@0.25 at 58.1% and

2For a fair comparison, we use the Mask3D-generated object proposals provided by LEO [18] for both 3D visual grounding and dense captioning.

Acc@0.5 at 51.7% on ScanRefer, and F1@0.25 at 58.0% and F1@0.5 at 52.7% on Multi3DRefer. As previous 3D LLMs either use detected object proposals (e.g., ChatScene, Chat3D) or train an additional grounding module based on a 3D detector (e.g., Grounded 3d-LLM, PQ3D, LLaVA3D), we can compare with these 3D LLMs fairly. Specifically, our model improves Acc@0.25 by 2.6% on ScanRefer and F1@0.25 by 0.9% on Multi3DRefer compared to ChatScene, which uses the same object proposal as ours.

3D Dense Captioning. Following the previous setting [17, 18], we generate captions for each detected object proposal2. Our method demonstrates superior performance in Scan2Cap, achieving 41.3 at B-4@0.5 and 83.8 at C@0.5. The results reveal that our method connects video content with its position information by injecting 3D-PE into video representations.

3D Question Answering. Our model outperforms the best competitors with 30.1% EM on ScanQA and 58.6% EM on SQA3D, which could be attributed to the strong representations inherited from Video LLMs. Existing Video LLMs achieve competent results in a zero-shot manner, suggesting that current 3D QA tasks may not sufficiently address the challenges of spatial reasoning in 3D scenes.

Frame Number

Sampling Strategy

Inference Time

ScanRefer Multi3dRefer Scan2Cap ScanQA SQA3D

Acc@0.25 Acc@0.5 F1@0.25 F1@0.5 B-4@0.5 C@0.5 C EM EM Fixed Frame Number

|8<br><br>Uniform MC|309ms<br><br>|48.93 43.50 49.80 45.40 37.34 68.82 94.98 27.57 56.77 53.47 47.41 53.55 48.54 38.77 73.08 96.37 28.00 56.97<br><br>|
|---|---|---|
|16<br><br>Uniform MC<br><br>|537ms<br><br>|55.42 49.17 54.95 49.82 39.39 76.96 99.86 28.96 57.70<br>56.46 50.11 56.65 51.39 39.59 76.84 100.63 29.49 57.82<br>|
|32<br><br>Uniform MC<br><br>|1050ms<br><br>|58.11 51.72 58.02 52.68 41.30 83.76 102.06 30.09 58.56 58.27 51.68 57.93 52.50 40.32 81.58 102.33 30.35 59.25|

###### Adaptive Frame Number

≈18 MC∗ 527ms 57.86 51.18 57.87 52.40 40.18 80.00 100.54 29.50 57.72 Previous SOTA

LLaVA-3D [51] 433ms 54.1 42.4 – – 41.1 79.2 91.7 27.0 55.6

- Table 2. Ablation study for the effect of frame sampling strategy. “MC” represents maximum coverage sampling. “MC∗” denotes sampling frames until over 95% of the scene’s voxels are covered or a maximum of 32 frames is reached.

3D-PE Coord.

Scan2Cap ScanRefer ScanQA

C@0.5 Acc@0.25 Acc@0.5 EM None

Avg

31.03 57.50 50.84 30.03 MLP 76.23 59.63 52.98 29.62

Sin 83.76 58.11 51.72 30.09

Sin

Center 80.88 57.53 51.06 29.39 Min-Max 82.75 58.05 51.77 30.18 Avg 83.76 58.11 51.72 30.09

- Table 3. Ablation study for the effect of coordinate encoding. “Coord.” means the method for aggregating the coordinates.
- 4.3. Ablation Study

Patch Size

ScanRefer Multi3DRefer

Loss

Acc@0.25 Acc@0.5 Acc@0.5 Acc@0.5

14 InfoNCE 56.44 50.08 56.31 51.05 27 InfoNCE 55.23 48.93 56.13 50.90 14 BCE 51.63 45.82 46.07 41.47

Table 4. Ablation study for the effect of visual grounding. We train the model separately on the ScanRefer and Multi3DRefer datasets.

Scan2Cap ScanRefer ScanQA

C@0.5 B@0.5 Acc@0.25 Acc@0.5 EM

Voxel 54.9 34.3 56.1 49.8 28.1 Video 83.8 (+28.9) 41.3 (+7.0) 58.1 (+2.0) 51.7 (+1.9) 30.1 (+2.0)

Table 5. Ablation study for the 3D scene representation.

Effectiveness of Frame Sampling. In Table 2, we assess the effectiveness of the frame sampling strategy across varying numbers of frames. To evaluate the model’s efficiency, we calculated the average inference speed on ScanQA. In the fixed frame number setting, we sample frames until reaching the desired number. As shown in the table, the model’s performance improves with an increasing number of frames, though inference time also rises. Video-3D LLM surpasses many previous methods with only 8 frames, demonstrating that our position-aware video paradigm can effectively model 3D scene understanding tasks. With 8 and 16 frames, maximum coverage sampling (“MC”) significantly enhances performance for all tasks by capturing a more complete 3D scene. Notably, with 8 frames, it improves Acc@0.25 by 4.54 on ScanRefer and C@0.5 by 4.26 on Scan2Cap compared to the uniform sampling strategy (“Uniform”). With 32 frames, the “Uniform” strategy shows comparable results with “MC”, as this number of frames is more than enough to cover the 3D scene. In the adaptive frame number setting, frames are sampled until over 95% of the scene’s voxels are covered or a maximum of 32 frames is reached. This allows for flexible adjustment based on scene size. As shown in the results, “MC∗” uses an average of 18 frames across all scenes, achieving similar performance to the 32-frame uniform strategy while offering better inference speed. Meanwhile, compared to

LLaVA-3D [51], “MC∗” shows superior performance while achieving similar inference speed. Maximum coverage sampling is performed once per scene, and the average time spent on ScanQA is only 17.8ms per question, which is negligible compared to the inference time.

Effectiveness of 3D-PE. Table 3 presents the performance using different 3D position encoding (3D-PE) and coordinate aggregation strategies. The model was trained on the entire multi-task dataset, with 32 frames under the uniform frame sampling. We first assess the impact of different 3DPE using average coordinate aggregation. The introduction of 3D-PE leads to a consistent performance increase in 3D dense captioning and 3D visual grounding in all aggregation variants. Specifically, for dense captioning tasks, which require locating objects within video sequences based on query bounding boxes, the absence of 3D-PE results in significantly degraded performance. Additionally, MLP position encoding proves more effective for grounding tasks, while Sin PE works better for others. We then evaluate different coordinate aggregation strategies using Sin position encoding. Since the 3D coordinates are more complex and variable than those in 2D images, simply using the coordinates at the center of a 2D patch doesn’t accurately reflect the spatial location. As shown in the result, using average

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

Uniform (16 frames)

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

MC (16 frames)

(d) It is a white window. the white window is on the right side, towards the back of the room.

(b) The trash can is between the copier and the table. the trash can is a gray rectangular prism.

(c) A white dresser. in front of it is a red armchair.

(a) There is a circular wooden end table. it is next to a teal couch and a blue armchair.

(e) There is a brown chair near the center of the room at a brown table. its left side faces the window to the room.

(f) There's a small cabinet sitting underneath the window behind the larger of the trow desks.

Query

###### Figure 3. The visualization results on ScanRefer. The green/red/blue colors indicate the correct/incorrect/ground truth boxes.

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

Entire Video

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

MC (16 frames)

GT: This is a white copy machine. It is located in the corner to the left of the wooden counter.

GT: The stool placed on hardwood floor is dark blue. It has four legs.

GT: Chair located directly behind the green chair. Chair is next to the stairs behind the green chair.

Caption

Ours: The copier is in the back left corner of the room. It is to the right of the bulletin board.

Ours: This is a stool. It sets between the couch and the wall.

Ours: The chair is the northeastern-most one in the room. The chair has a curved backside and four legs.

Figure 4. The visualization results on Scan2Cap. The input boxes are marked in blue.

3D coordinates better represents the 3D location within a patch, while using the minimum and maximum 3D coordinates within the patch is also a viable alternative.

Ablation for 3D Visual Grounding. As illustrated in Table 4, we investigate the effect of varying patch sizes of object embeddings and loss functions. Since LLaVA-Video [47] downsamples the image patches again before feeding them into LLM, we can generate object embeddings with patch sizes of 14 or 27. For ScanRefer, switching the patch size from 27 to 14 leads to improved accuracy, with Acc@0.5 from 48.93% to 50.08%. This improvement may be attributed to the smaller patch size to capture more precise object features. Additionally, since we model the grounding task by leveraging the similarity between object embeddings and the ⟨ground⟩ token, the use of BCE loss may impose overly strict constraints. Replacing BCE loss with InfoNCE loss consistently improves performance.

Effectiveness of Representing 3D Scenes as Videos. To eliminate the influence of the backbone, we conducted ablation experiments by comparing our 3D-as-video paradigm with voxel modeling based on the same backbone, LLaVA-

Video. Following LLaVA-3D, we average the patch features corresponding to the same voxel to obtain the voxel features, and sample 3,096 of these features as input for the LLM. As shown in Table 5, there is a notable improvement of 28.9 C@0.5 on Scan2Cap, from 83.8 to 54.9. Note that our dataset is only 26% of LLaVA-3D, resulting the lower results compared to the LLaVA-3D. For ScanQA, which relies more heavily on visual perception, our approach still achieves a 2% gain in accuracy compared to voxel modeling. This indicates that our approach leverages spatiotemporal priors specifically, rather than merely benefiting from stronger visual features.

Visualization. Figure 3 presents visualization results on the validation split of ScanRefer. The first and second rows indicate the rendered point clouds using uniform sampling and maximum coverage sampling with 16 frames. We observe that maximum coverage sampling usually provides more complete scenes than uniform sampling. For complex cases like (b-e), uniform sampling tends to miss smaller or peripheral objects, leading to prediction failures. Figure 4 presents visualization results on the Scan2Cap validation set. The first two rows show the rendered point clouds of the full video and the frames sampled using the MC strategy. With only 16 frames, nearly the entire scene’s information is captured. Meanwhile, using the proposed 3D-PE, the model accurately describes the specified target object.

### 5. Conclusion

In this paper, we propose a novel paradigm to effectively exploit Video LLMs for 3D scene understanding. It incorporates 3D position encoding into video representations and employs a carefully-designed multi-task training recipe, thereby facilitating a suite of 3D scene understanding tasks. We further introduce a maximum coverage sampling strategy to optimize the trade-off between computational costs and model performance. Our extensive experimental results demonstrate the superiority of our method.

### Acknowledgements

This work is supported by National Key R&D Program of China (Project No. 2022ZD0161200, 2022ZD0161201). This work is also supported by Hong Kong Research Grant Council - Early Career Scheme (Grant No. 24200223).

### References

- [1] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Ana¨ıs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. ArXiv preprint, abs/2312.11805,

2023. 1

- [2] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 19107–19117. IEEE, 2022. 2, 5, 6, 13, 14
- [3] Daigang Cai, Lichen Zhao, Jing Zhang, Lu Sheng, and Dong Xu. 3djcg: A unified framework for joint dense captioning and visual grounding on 3d point clouds. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 16443–16452. IEEE, 2022. 13, 14
- [4] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas J. Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 14455–14465. IEEE, 2024. 1
- [5] Dave Zhenyu Chen, Angel X. Chang, and Matthias Nießner. Scanrefer: 3d object localization in RGB-D scans using natural language. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XX, pages 202–221. Springer, 2020. 2, 5, 6, 13, 14
- [6] Dave Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X. Chang. Scan2cap: Context-aware dense captioning in RGB-D scans. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 1925, 2021, pages 3193–3203. Computer Vision Foundation / IEEE, 2021. 2, 5, 6, 13
- [7] Dave Zhenyu Chen, Qirui Wu, Matthias Nießner, and Angel X. Chang. D3net: A unified speaker-listener architecture

- for 3d dense captioning and visual grounding. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXII, pages 487–505. Springer, 2022. 13, 14
- [8] Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Language conditioned spatial relation reasoning for 3d object grounding. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28

- December 9, 2022, 2022. 5, 6, 14

- [9] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. LL3DA: visual interactive instruction tuning for omni-3d understanding, reasoning, and planning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26418–

26428. IEEE, 2024. 1, 2, 3, 5, 6, 13, 14

- [10] Yilun Chen, Shuai Yang, Haifeng Huang, Tai Wang, Ruiyuan Lyu, Runsen Xu, Dahua Lin, and Jiangmiao Pang. Grounded 3d-llm with referent tokens. ArXiv preprint, abs/2405.10370,

2024. 1, 2, 5, 6, 14

- [11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Intern VL: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 24185–24198. IEEE, 2024. 1
- [12] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas A. Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 2432–2443. IEEE Computer Society, 2017. 2, 5
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7,

2021. OpenReview.net, 2021. 4

- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aur´elien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozi`ere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy,

- Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gr´egoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. ArXiv preprint, abs/2407.21783, 2024. 1
- [15] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual understanding and reasoning. ArXiv preprint, abs/2403.11401, 2024. 1, 6, 13, 14
- [16] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10

- 16, 2023, 2023. 1, 2, 3, 5, 6, 14

- [17] Haifeng Huang, Zehan Wang, Rongjie Huang, Luping Liu, Xize Cheng, Yang Zhao, Tao Jin, and Zhou Zhao. Chat-3d v2: Bridging 3d scene and large language models with object identifiers. ArXiv preprint, abs/2312.08168, 2023. 1, 2, 5, 6, 13, 14
- [18] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27,

2024. OpenReview.net, 2024. 1, 2, 5, 6, 13, 14

- [19] Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. Multiview transformer for 3d visual grounding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 15503–15512. IEEE, 2022. 5, 6, 14
- [20] Weitai Kang, Haifeng Huang, Yuzhang Shang, Mubarak Shah, and Yan Yan. Robin3d: Improving 3d large language model via robust instruction tuning. ArXiv preprint, abs/2410.00255, 2024. 1, 2
- [21] Samir Khuller, Anna Moss, and Joseph (Seffi) Naor. The budgeted maximum coverage problem. Information Processing Letters, 70(1):39–45, 1999. 3
- [22] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Paul Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. ArXiv preprint, abs/2406.09246, 2024. 2, 3
- [23] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and

- Chunyuan Li. Llava-onevision: Easy visual task transfer. ArXiv preprint, abs/2408.03326, 2024. 1, 2, 3
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, pages 19730–

19742. PMLR, 2023. 1

- [25] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 22195–

22206. IEEE, 2024. 2

- [26] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: on pre-training for visual language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26679–26689. IEEE,

2024. 2

- [27] Xiongkun Linghu, Jiangyong Huang, Xuesong Niu, Xiaojian (Shawn) Ma, Baoxiong Jia, and Siyuan Huang. Multimodal situated reasoning in 3d scenes. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024,

2024. 1

- [28] Benlin Liu, Yuhao Dong, Yiqin Wang, Yongming Rao, Yansong Tang, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondence elicit 3d spacetime understanding in multimodal language model. ArXiv preprint, abs/2408.00754, 2024. 1, 2, 3
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 1, 13
- [30] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx MLLM: on-demand spatial-temporal understanding at arbitrary resolution. ArXiv preprint, abs/2409.12961, 2024. 1, 2, 3, 5, 6, 14
- [31] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. SQA3D: situated question answering in 3d scenes. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net,

2023. 2, 5, 13

- [32] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul McVay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent-Pierre Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Alexander Sax, and Aravind Rajeswaran. Openeqa: Embodied question answering in the era of foundation models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle,

WA, USA, June 16-22, 2024, pages 16488–16498. IEEE,

2024. 1

- [33] OpenAI. GPT-4 technical report. ArXiv preprint, abs/2303.08774, 2023. 1
- [34] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask transformer for 3d semantic instance segmentation. In IEEE International Conference on Robotics and Automation, ICRA 2023, London, UK, May 29 - June 2, 2023, pages 8216–

8223. IEEE, 2023. 5, 13

- [35] A¨aron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. ArXiv preprint, abs/1807.03748, 2018. 5
- [36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. 4
- [37] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 4566–4575. IEEE Computer Society, 2015. 5
- [38] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. RIO: 3d object instance re-localization in changing indoor environments. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 7657–7666. IEEE, 2019. 2
- [39] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. ArXiv preprint, abs/2409.12191, 2024. 1
- [40] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. ArXiv preprint, abs/2308.08769, 2023. 1, 2, 5, 6, 14
- [41] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 2, 3
- [42] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXV, pages 131–147. Springer, 2024. 1, 2
- [43] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin,

- Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. ArXiv preprint, abs/2407.10671, 2024. 5
- [44] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 12–22. IEEE, 2023. 2
- [45] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 543–553, Singapore, 2023. Association for Computational Linguistics. 2
- [46] Yiming Zhang, ZeMing Gong, and Angel X. Chang. Multi3drefer: Grounding text description to multiple 3d objects. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, page

15179. IEEE, 2023. 2, 5, 6, 13, 14

- [47] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. 1, 2, 5, 6, 8, 13, 14
- [48] Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3dvgtransformer: Relation modeling for visual grounding on point clouds. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 2908–2917. IEEE, 2021. 5, 6, 14
- [49] Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and Liwei Wang. Towards learning a generalist model for embodied navigation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 13624–13634. IEEE, 2024. 14
- [50] Chenming Zhu, Tai Wang, Wenwei Zhang, Kai Chen, and Xihui Liu. Scanreason: Empowering 3d visual grounding with reasoning capabilities. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part VIII, pages 151–168. Springer, 2024. 5
- [51] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. ArXiv preprint, abs/2409.18125, 2024. 1, 2, 3, 5, 6, 7, 13, 14
- [52] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 2899–2909. IEEE, 2023. 5, 6, 13, 14

- [53] Ziyu Zhu, Zhuofan Zhang, Xiaojian Ma, Xuesong Niu, Yixin Chen, Baoxiong Jia, Zhidong Deng, Siyuan Huang, and Qing Li. Unifying 3d vision-language understanding via promptable queries. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLIV, pages 188–206. Springer,

2024. 6, 14

- [54] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong T. Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, CoRL 2023, 6-9 November 2023, Atlanta, GA, USA, pages 2165–2183. PMLR, 2023. 2, 3

Appendix

- A. Implementation Details

Dataset Statistics. We present the detailed statistics for training and testing data in Table 6 and 7, respectively. Following previous work [17, 18], we adopt the validation set for ScanRefer [5], Multi3DRefer [46], Scan2Cap [6], ScanQA [6], and the test set for SQA3D [31]. All data have been converted to LLaVA [29] format, and we conduct statistics in this format.

Evaluation Details. For ScanRefer [5], we select the object with the highest similarity as the prediction. For Multi3DRefer [46], we select objects with the highest probabilities such that their cumulative probability exceeds a given threshold p, which is empirically set to 0.25. For Scan2Cap [6], we follow [18] to evaluate the captioning performance by inserting “sos” and “eos” at the start and end of the prediction, respectively. Responses are generated using greedy sampling for 3D dense captioning and 3D question answering tasks.

- B. Detailed Comparison

SQA3D. We conduct a detailed evaluation on the test split of the SQA3D [31] dataset. As shown in Table 8, our model achieves the best performance on all categories of questions with an average EM at 58.86%, outperforming the previous state-of-the-art method LLaVA-3D [51] by 2.94% on the average EM.

Scan2Cap. As shown in Table 9, we provide a detailed comparison on the validation set of Scan2Cap [6]. During inference, we utilize the object proposals from [18], which include 50 predicted objects extracted with Mask3D [34] for each scan. From the table, we can see our method achieves state-of-the-art results on CIDEr and BLEU-4 at 83.77 and 42.43, respectively.

ScanRefer. We present a detailed comparison for ScanRefer [5] in Table 10. The table shows that our method reaches a peak of 58.12% Acc@0.25 and 51.72% Acc@0.5, surpassing ChatScene [17] by 2.6% and 1.5%, respectively.

Multi3DRefer. We follow previous work [46] to report the metrics across all question types, where “ZT” denotes zerotarget, “ST” denotes single-target, “MT” denotes multitarget, “w/ D” and “w/o D” denote ‘with and without distractors, respectively. As shown in Table 11, our method outperforms previous methods on “ZT w/o D”, “ZT w/ D”, and “ST w/D” types. However, the performance for “MT” is lower than ChatScene [17], suggesting that our method still struggles to distinguish similar objects.

ScanQA. We test our model on the validation set of

Ques length

Answer Length

Data Count

Scan Count

ScanRefer [5] 36,665 562 24.9 – Multi3DRefer [46] 43,838 562 34.8 – Scan2Cap [6] 36,665 562 13.0 17.9 ScanQA [2] 26,515 562 13.7 2.4 SQA3D [31] 79,445 518 37.8 1.1

- Table 6. Detailed statistics for training data. We report the average lengths for questions and answers, respectively.

Data Count

Scan Count

Ques length

Answer Length

ScanRefer [5] (Val) 9,508 141 25.0 – Multi3DRefer [46] (Val) 11,120 141 34.7 – Scan2Cap [6] (Val) 2,068 141 13.0 18.7 ScanQA [2] (Val) 4,675 71 13.8 2.4 SQA3D [31] (Test) 3,519 67 36.3 1.1

- Table 7. Detailed statistics for testing data. We report the average lengths for questions and answers, respectively.

Method

Test set

Avg. What Is How Can Which Others

SQA3D [31] 31.6 63.8 46.0 69.5 43.9 45.3 46.6 3D-VisTA [52] 34.8 63.3 45.4 69.8 47.2 48.1 48.5 LLaVA-Video[47] 42.7 56.3 47.5 55.3 50.1 47.2 48.5 Scene-LLM [15] 40.9 69.1 45.0 70.8 47.2 52.3 54.2 LEO [18] – – – – – – 50.0 ChatScene [17] 45.4 67.0 52.0 69.5 49.9 55.0 54.6 LLaVA-3D [51] – – – – – – 55.6 Video-3D LLM (Uniform) 51.1 72.4 55.5 69.8 51.3 56.0 58.6 Video-3D LLM (MC) 50.0 70.7 57.9 69.8 50.1 55.8 57.7

- Table 8. Performance comparison on the test set of SQA3D [31].

Method

@0.5 C B-4 M R

Scan2Cap [6] 39.08 23.32 21.97 44.48 3DJCG [3] 49.48 31.03 24.22 50.80 D3Net [7] 62.64 35.68 25.72 53.90 3D-VisTA [52] 66.9 34.0 27.1 54.3 LL3DA [9] 65.19 36.79 25.97 55.06 LEO [18] 68.4 36.9 27.7 57.8 ChatScene [17] 77.19 36.34 28.01 58.12 LLaVA-3D [51] 79.21 41.12 30.21 63.41 Video-3D LLM (Uniform) 83.77 42.43 28.87 62.34 Video-3D LLM (MC) 80.00 40.18 28.49 61.68

- Table 9. Performance comparison on the validation set of Scan2Cap [6]. C, B-4, M, R represent CIDEr, BLEU-4, Meteor, Rouge-L, respectively.

ScanQA [2]. Compared to previous top-tier models, our Video-3D LLM achieves a relative improvement of 10.7% and 11.9% on EM@1 and CIDEr, respectively.

Unique Multiple Overall Acc@0.25 Acc@0.5 Acc@0.25 Acc@0.5 Acc@0.25 Acc@0.5

Method Venue

ScanRefer [5] ECCV20 76.33 53.51 32.73 21.11 41.19 27.40 MVT [19] CVPR22 77.67 66.45 31.92 25.26 40.80 33.26 3DVG-Transformer [48] ICCV21 81.93 60.64 39.30 28.42 47.57 34.67 ViL3DRel [8] NeurIPS22 81.58 68.62 40.30 30.71 47.94 37.73 3DJCG [3] CVPR22 83.47 64.34 41.39 30.82 49.56 37.33 D3Net [7] ECCV22 – 72.04 – 30.05 – 37.87 M3DRef-CLIP [46] ICCV23 85.3 77.2 43.8 36.8 51.9 44.7 3D-VisTA [52] ICCV23 81.6 75.1 43.7 39.1 50.6 45.8 3D-LLM (Flamingo) [16] NeurIPS23 – – – – 21.2 – 3D-LLM (BLIP2-flant5) [16] NeurIPS23 – – – – 30.3 – Grounded 3D-LLM [10] ArXiv24 – – – – 47.9 44.1 PQ3D [53] ECCV24 86.7 78.3 51.5 46.2 57.0 51.2 ChatScene [17] NeurIPS24 89.59 82.49 47.78 42.90 55.52 50.23 LLaVA-3D [51] ArXiv24 – – – – 54.1 42.2 Video-3D LLM (Uniform) – 87.97 78.32 50.93 45.32 58.12 51.72 Video-3D LLM (MC) – 86.61 77.02 50.95 44.96 57.87 51.18

- Table 10. Performance comparison on the validation set of ScanRefer [5]. “Unique” and “Multiple” depends on whether there are other objects of the same class as the target object.

Method

ZT w/o D ZT w/ D ST w/o D ST w/ D MT ALL

F1 F1 F1@0.25 F1@0.5 F1@0.25 F1@0.5 F1@0.25 F1@0.5 F1@0.25 F1@0.5

M3DRef-CLIP [46] 81.8 39.4 53.5 47.8 34.6 30.6 43.6 37.9 42.8 38.4 D3Net [7] 81.6 32.5 – 38.6 – 23.3 – 35.0 – 32.2 3DJCG [3] 94.1 66.9 – 26.0 – 16.7 – 26.2 – 26.6 Grounded 3D-LLM [10] – – – – – – – – 45.2 40.6 PQ3D [53] 85.4 57.7 – 68.5 – 43.6 – 40.9 – 50.1 ChatScene [17] 90.3 62.6 82.9 75.9 49.1 44.5 45.7 41.1 57.1 52.4 Video-3D LLM (Uniform) 94.7 78.5 82.6 73.4 52.1 47.2 40.8 35.7 58.0 52.7 Video-3D LLM (MC) 94.1 76.7 81.2 72.6 52.7 47.4 40.6 35.3 57.9 52.4

- Table 11. Performance comparison on the validation set of Multi3DRefer [46]. ZT: zero-target, ST: single-target, MT: multi-target, D: distractor.

Method Venue EM B-1 B-2 B-3 B-4 ROUGE-L METEOR CIDEr ScanQA [2] CVPR22 21.05 30.24 20.40 15.11 10.08 33.33 13.14 64.86 3D-VisTA [52] ICCV23 22.4 – – – 10.4 35.7 13.9 69.6 Oryx-34B [30] ArXiv24 – 38.0 24.6 – – 37.3 15.0 72.3 LLaVA-Video-7B [47] ArXiv24 – 39.71 26.57 9.33 3.09 44.62 17.72 88.70 3D-LLM (Flamingo) [16] NeurIPS23 20.4 30.3 17.8 12.0 7.2 32.3 12.2 59.2 3D-LLM (BLIP2-flant5) [16] NeurIPS23 20.5 39.3 25.2 18.4 12.0 35.7 14.5 69.4 Chat-3D [40] ArXiv23 – 29.1 – – 6.4 28.5 11.9 53.2 NaviLLM [49] CVPR24 23.0 – – – 12.5 38.4 15.4 75.9 LL3DA [9] CVPR24 – – – – 13.53 37.31 15.88 76.79 Scene-LLM [15] ArXiv24 27.2 43.6 26.8 19.1 12.0 40.0 16.6 80.0 LEO [18] ICML24 – – – – 11.5 39.3 16.2 80.0 Grounded 3D-LLM [10] ArXiv24 – – – – 13.4 – – 72.7 ChatScene [17] NeurIPS24 21.62 43.20 29.06 20.57 14.31 41.56 18.00 87.70 LLaVA-3D [51] arXiv24 27.0 – – – 14.5 50.1 20.7 91.7 Video-3D LLM (Uniform) – 30.10 47.05 31.70 22.83 16.17 49.02 19.84 102.06 Video-3D LLM (MC) – 29.50 46.23 31.22 22.71 16.28 48.19 19.36 100.54

- Table 12. Performance comparison on the validation set of ScanQA [2]. EM indicates exact match accuracy, and B-1, B-2, B-3, B-4 denote BLEU-1, -2, -3, -4, respectively.

