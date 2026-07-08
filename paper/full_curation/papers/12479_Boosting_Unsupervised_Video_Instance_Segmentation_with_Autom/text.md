# arXiv:2512.06864v1[cs.CV]7Dec2025

#### Boosting Unsupervised Video Instance Segmentation with Automatic Quality-Guided Self-Training

Kaixuan Lu1 Mehmet Onurcan Kaya1,2 Dim P. Papadopoulos1,2 1Technical University of Denmark 2Pioneer Centre for AI

s232248@student.dtu.dk, monka@dtu.dk, dimp@dtu.dk

###### Abstract

Video Instance Segmentation (VIS) faces significant annotation challenges due to its dual requirements of pixellevel masks and temporal consistency labels. While recent unsupervised methods like VideoCutLER eliminate optical flow dependencies through synthetic data, they remain constrained by the synthetic-to-real domain gap. We present AutoQ-VIS, a novel unsupervised framework that bridges this gap through quality-guided self-training. Our approach establishes a closed-loop system between pseudo-label generation and automatic quality assessment, enabling progressive adaptation from synthetic to real videos. Experiments demonstrate state-of-the-art performance with 52.6 AP50 on YouTubeVIS-2019 val set, surpassing the previous state-of-the-art VideoCutLER by 4.4%, while requiring no human annotations. This demonstrates the viability of quality-aware self-training for unsupervised VIS. The source code of our method is available at here.

###### 1. Introduction

Video Instance Segmentation (VIS) is the challenging task of simultaneously detecting, segmenting, and tracking object instances across video sequences [10, 18, 32, 33, 39, 45]. This capability is fundamental for scene understanding in applications ranging from autonomous driving [43] to video content editing [47]. However, training highperformance VIS models typically requires pixel-level annotations across all frames [39]. This process is expensive due to the labor-intensive nature of annotating temporal consistency and instance identities. As a result, there is an urgent need to develop unsupervised video instance segmentation frameworks that can accurately interpret video content and function effectively across diverse, unlabeled environments.

Prior work [1, 4, 8, 17, 20, 37, 44] in unsupervised video segmentation predominantly addresses Video Object Segmentation (VOS), focusing on separating a sin-

###### Multi-Round Self-Training

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

VIS Model Unlabeled Videos

|Pseudo Masks<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

Train with both synthetic dataset and ﬁltered pseudo masks

|Filter with a Fixed Threshold|
|---|

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>Synthetic Videos with Annotations|
|---|

| | |
|---|---|
|[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>Quality Scores<br><br>0.9 0.8<br><br>0.9 0.5<br><br>| |

Mask Quality Predictor

Figure 1. AutoQ-VIS overview. In the initial training stage, both the VIS model and the mask quality predictor are trained on synthetic videos with pseudo annotations [31]. During the multiround self-training stage, the VIS model generates pseudo masks on unlabeled videos, which are then scored by the frozen quality predictor. Pseudo masks with high predicted quality are selected and added to the training set. The VIS model is subsequently retrained on both the synthetic data and the selected pseudo labels, enabling iterative refinement and progressive performance gains.

gle foreground object via motion or consistency cues. While OCLR [34] introduces unsupervised VIS that supports multiple instances, its predefined object count during training prevents dynamic adaptation to varying numbers of instances during inference. Furthermore, prior approaches [4, 17, 20, 34, 37] rely on optical flow estimators (e.g., RAFT [27]) that are trained on human-annotated datasets. VideoCutLER [31] marks a breakthrough in unsupervised VIS and achieves unprecedented performance by demonstrating multi-instance segmentation without optical flow dependencies. Its core innovation lies in synthetic video generation via spatial augmentations of CutLER [30] pseudo-labels from ImageNet [7]. However, VideoCutLER remains constrained by synthetic-to-real domain gaps and static instance modeling, i.e., the synthetic videos lack nat-

ural and realistic motion patterns.

Building upon VideoCutLER’s synthetic data paradigm, which generates training videos through spatial augmentations of static image pseudo-labels, we introduce AutoQVIS to address its critical domain gap limitation. While VideoCutLER’s synthetic videos provide initial instance awareness, they lack natural motion patterns and real-world appearance variations, hindering adaptation to authentic video dynamics. Our framework bridges this syntheticto-real domain gap through a self-training loop that progressively adds quality-filtered pseudo-labels from unlabeled real videos. Inspired by Mask Scoring R-CNN [11], which directly predicts mask quality scores via an auxiliary branch, we implement a quality assessment module for pseudo-label filtering of instance masks.

AutoQ-VIS advances unsupervised video instance segmentation through an iterative self-training paradigm with quality-aware pseudo-label selection (Fig. 1). The system initializes using synthetic video data from VideoCutLER, which provides pseudo-labels to bootstrap a VideoMask2Former [2, 3] VIS model and a specialized mask quality predictor (Sec. 3.1). The mask quality predictor estimates mask IoU quality scores by analyzing frame-level features and mask predictions. During multi-round optimization, the VIS model generates pseudo-labels on unlabeled videos, which are then scored by the quality predictor. High-quality pseudo-labels surpassing a fixed threshold are progressively incorporated into the training set, enabling dataset augmentation without any human supervision

- (Sec. 3.2). To enhance the mask head training, we employ a DropLoss that zeros out mask losses whose maximum ground-truth overlap falls below 0.01 (Sec. 3.3). By alternating rounds of VIS training (with occasional weight resets) and quality-based dataset expansion, AutoQ-VIS dynamically enriches its training dataset and steadily sharpens segmentation performance.

AutoQ-VIS establishes a new state-of-the-art in unsupervised video instance segmentation, achieving 52.6% AP50 on the YouTubeVIS-2019 validation set [39]. This represents a significant +4.4% AP50 improvement over the previous best-performing method VideoCutLER [31], demonstrating our framework’s effectiveness in motion pattern understanding and instance-level discrimination. To further validate the approach’s generalizability, we conduct additional evaluation on the UVO-Dense benchmark [29], where our method achieves consistent performance gains (1.1% AP50 improvement over the previous state-of-the-art approaches) under dense object scenarios.

Our key contributions are threefold: (1) AnnotationFree VIS Framework: We propose AutoQ-VIS, an unsupervised framework that overcomes annotation dependency through cyclic pseudo-label refinement with automated quality control, enabling video instance segmenta-

tion training directly from unlabeled videos. (2) Automatic Quality Assessment: We propose a simple quality predictor that reliably filters pseudo labels across self-training rounds. (3) New State-of-the-art Performance: Our AutoQ-VIS achieves 52.6 AP50 on YouTubeVIS-2019 [39] val split, surpassing the previous state-of-the-art VideoCutLER [31] by 4.4 AP50.

###### 2. Related Work

Unsupervised video object segmentation (VOS) [14, 16, 21, 24, 36] targets identifying and segmenting moving objects as foreground elements in video sequences, generating pixel-level binary masks that distinguish these objects from background content without any human-annotated labels. The task applies to both scenarios, including those with a single object instance and those with multiple concurrent instances. It is worth noting that some materials [15, 26, 40, 46] refer to video salient object detection (SOD) as Unsupervised VOS, which is a different task that requires human-annotated labels. Current unsupervised video segmentation methods predominantly rely on detection based on motion or consistency cues. Motion Group [37] uses optical flow to train an unsupervised video object segmentation network, and it uses only optical flow as input. Similar to Motion Group [37], many methods [4, 17, 20, 28, 34, 41] use optical flow as input or use it to supervise the model. However, they all rely on optical flow estimators (e.g., RAFT [27]) that are trained on human-annotated datasets, and most of them [4, 17, 20, 41] can only track one object at a time. While some unsupervised video object segmentation methods [28, 34] support multi-object tracking, their multi-object tracking performance is very limited, which makes them unsuitable for the video instance segmentation task.

Unsupervised video instance segmentation (VIS) [10, 18, 32, 33, 39, 45] targets identifying and segmenting moving objects from backgrounds and discriminating between distinct instances, all without any human-annotated labels. VideoCutLER [31] is the first unsupervised video segmentation method that is built for the VIS task, without using any human-annotated labels in the whole pipeline. VideoCutLER [31] establishes a milestone in unsupervised video instance segmentation (VIS), surpassing previous benchmarks by demonstrating multi-instance segmentation without using optical flow. While its key innovation — a synthetic video generation pipeline leveraging spatial augmentations of CutLER [30] pseudo-labels (originally derived from ImageNet [7]) — enables remarkable performance, the approach remains fundamentally limited by syntheticto-real domain discrepancies and rigid instance modeling. Specifically, the static object configurations in synthesized videos fail to capture authentic motion dynamics observed in natural video sequences.

Classical VideoMask2Former Model for Video Instance Segmentation

|Class|
|---|

| | |
|---|---|
| | |

|Mask|
|---|

Transformer Decoder

Input Frames

Predicted Early Layer Features Masks

Last Layer Features

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

| | |
|---|---|
| | |

Pixel Decoder

Backbone

Feature for a Frame

Mask for a Frame

Linear Layer

[Figure 27]

Convolution Layer (Kernel Size: , Stride: )

| | |
|---|---|
| | |

Linear Layer

Max-Pool Max-Pool

|Linear Layer| |
|---|---|
| | |

Convolution Layer (Kernel Size: , Stride: )

[Figure 28]

| |
|---|

Predicted IoU Score

Concatenate

Frame-Based Quality Predictor Head based on Mask Scoring R-CNN

- Figure 2. Network architecture of VideoMask2Former [2, 3] and Mask Quality Predictor. Our quality predictor integrates mask predictions and pixel decoder features following [11], employing a sequential architecture with four convolution layers (3×3 kernels, final layer stride of 2 for spatial reduction) followed by three fully-connected layers that ultimately produce mask IoU predictions.

Looped self-training method is the method that uses the pseudo labels generated by the model to retrain the model [5, 30, 35, 38, 48, 49]. This method is widely used in different areas related to unsupervised or semi-supervised training, for example, unsupervised domain adaptation [38, 48, 49], semi-supervised image classification [35], semisupervised 3D instance segmentation [5], unsupervised object detection and segmentation [30], and human-in-theloop systems for object detection [13, 22], instance segmentation [6, 23], and 3D shape segmentation [42].

###### 3. Method

AutoQ-VIS operates through four parts: (1) Initial Training (Sec. 3.1): Jointly pretrain VideoMask2Former and the mask quality predictor on VideoCutLER’s synthetic videos; (2) Multi-Round Self-Training (Sec. 3.2): Iteratively generate pseudo-labels on real unlabeled videos, filter via quality scores, and augment training data; (3) DropLoss

- (Sec. 3.3): Suppress low-IoU mask predictions to enhance mask head training. This quality-guided pipeline progressively improves segmentation accuracy without any human annotations; (4) Adaptive fusion: Augment the dataset dynamically through adaptive fusion of newly curated annotations.

###### 3.1. Initial training stage

Video instance segmentation (VIS) model. Following VideoCutLER [31], we use the VideoMask2Former [2, 3] with the ResNet-50 [9] backbone as our video instance segmentation (VIS) model.

Quality predictor. For the quality predictor, as shown in Fig. 2, we use an architecture inspired by Mask Scoring R-CNN [11]. Our architecture processes individual frame features and single-object mask predictions per inference step. Supervision is established through thresholdbinarized (0.5) mask IoU between predictions and matched ground truths, optimized via ℓ2 regression loss. Although the architecture of our quality predictor and Mask Scoring R-CNN is similar, there is a key difference between our model and theirs. In Mask Scoring R-CNN, the input predicted mask is threshold-binarized (0.5). In our case, we find it crucial to use the raw predicted mask instead of the threshold-binarized predicted mask. If we use the threshold-binarized predicted mask, the quality predictor cannot produce any meaningful output.

Synthetic videos. VideoCutLER [31] provides a highquality pseudo-labeled synthetic video dataset that was built on the unlabeled images from ImageNet [7], which is very suitable to train and initialize our VIS model and quality predictor. We also use the trained VideoMask2Former model [2] from VideoCutLER to initialize our VIS model.

###### 3.2. Multi-round self-training

As shown in Fig. 1, we optimize the VIS model through iterative self-training and dynamic dataset augmentation. The training dataset is initialized using the synthetic videos from VideoCutLER [31]. Empirically, we find that executing model parameter restoration from the initial model weight (trained in Sec. 3.1) achieves a better performance.

After training the VIS model, we use the predicted masks on unlabeled videos with a confidence score over 0.25 as pseudo-labels. Then we use the quality predictor to predict the IoU of each pseudo-label predicted mask. Let IoUˆ l denote the predicted IoU of label l, and sl denotes the confidence score of label l from the class head of the VIS model (see Fig. 2). We define the quality score of label l as Ql = IoUˆ l · sl.

We implement quality-based pseudo-label selection using a fixed quality score threshold τth. For each pseudolabel l, we select it if Ql ≥ τth. At the end of each round, we add all the pseudo-labels to the training dataset.

###### 3.3. DropLoss for mask head

We enhance the mask head training by suppressing loss contributions from low-overlap predictions, following CutLER [30]. For each predicted mask mi, we discard its loss contribution if its maximum ground truth IoU falls below

the threshold τIoU:

Ldrop(mi) = 1(IoUmaxi > τIoU)Lvanilla(mi) (1)

Here, IoUmaxi is the maximum overlap between mi and any ground truth mask, while Lvanilla denotes the original mask head loss from VideoMask2Former [2, 3]. We employ a low threshold (τIoU = 0.01) to filter only near-zero overlap predictions.

###### 3.4.Adaptivefusionofnewcuratedannotationsand old curated annotations

To augment the training dataset, at the end of each round of self-training, we need to merge the old curated annotations in the training dataset with the new curated annotations.

The augmentation process operates in two distinct modes depending on whether the video containing the new detections already exists in the training set. For novel videos, we perform bulk insertion of all qualified detections. For existing videos, we implement an instance-level fusion protocol that intelligently merges new predictions with old annotations while preserving temporal consistency.

Let dv denote an object detection for video v, formally:

dv = {(St,mt)}T

t (2)

v

Here, St ∈ {0,1} indicates whether the object mask (pseudo label) of dv at frame t is selected for self-training, mt denotes the binary object mask at frame t.

Let T (k) = {(v,Dv)} denote the training dataset at iteration k, where Dv contains all preserved detections (old annotations) for video v, and Dretained(k) denote the set of detections that we retained after the filtering using the quality score threshold τth at the end of iteration k (new detections). We need to merge the Dretained(k) into the T (k) = {(v,Dv)}. For each video, if it’s not in the training dataset, we simply add all its detections to the training dataset. Formally, let Dretained(k) ,v denote the detections belong to video v:

Dretained(k) ,v = d ∈ Dretained(k) d belongs to video v (3)

Let Vnew(k) = π1(Dretained(k) ) \ π1(T (k)) denote completely new videos, where π1 projects to video identifiers. Let Tnew(k) denote labels of new videos:

v,Dretained(k) ,v (4)

Tnew(k+1) =

v∈Vnew(k)

For videos already present in the training set, we implement a temporal-aware fusion protocol that resolves conflicts between new detections and old annotations.

To merge the existing detections and new detections, we need to identify whether two detections overlap. We define

the spatiotemporal overlap predicate for two detections:

ϕ(dnew,v,dexist,v) ≜ ∃t ∈ [1,Tv] : ∥mtnew ∩ mtexist∥ ∥mtnew ∪ mtexist∥

≥ 0.5

(5)

If in any frames, two detections’ masks overlap (IoU ≥ 0.5), we consider them overlapping detections. If two detections overlap, we need to fuse them. We fuse the new detection and the existing detection frame by frame. For one frame, if only the existing detection’s label is selected, we use its label; otherwise, we use the new detection’s label. Let F denote the fusion operation:

F(dnew,v,dexist,v) = {Smerget ,mtmerge}T

t=1 (6) where:

v

Smerget = max(Snewt ,Sexistt ) (7)

mtexist if Sexistt = 1 ∧ Snewt = 0 mtnew otherwise

mtmerge =

(8)

If the new detection does not overlap with any existing detections, we simply add it. Let Vexist(k) = π1(T (k)) denote existing videos. For each existing video v ∈ Vexist(k) , we process all new detections Dretained(k) ,v through:

Dv(k+1) = Dv(k) Dretained(k) ,v (9) where the operation is defined as:

###### D D′ =

(D ⊕ d′) (10)

d′∈D′

with per-detection fusion:

 

D ∪ {dnew} if ∀dexist ∈ D,

¬ϕ(dnew,dexist) D \ dexist ∪ F(dnew,dexist) if ∃dexist ∈ D,

D⊕dnew =



ϕ(dnew,dexist)

(11)

In the end, we merge the labels of new videos and existing videos:

T (k+1) = Tnew(k+1) ∪ Texist(k+1) (12) where:

Texist(k+1) =

v,Dv(k+1) (13)

v∈Vexist(k)

During training, for each video, we need to sample three frames as the input of the model (following VideoCutLER [31]). To provide good-quality labels, we only sample those frames where all their detections are selected. For

each video v ∈ Vtrain(k+1), we define the eligible frame set:

Feligible(k+1)(v) = t ∈ [1,Tv] ∀d ∈ Dv(k+1) : Sdt = 1 (14)

YouTubeVIS-2019 Method

AP50 AP75 AP APS APM APL AR10 MotionGroup [37] 0.5 0.0 0.1 0.0 0.4 0.1 1.2 OCLR [34] 5.5 0.3 1.6 0.1 1.6 6.1 11.5 CutLER [30] 36.4 13.5 16.0 3.5 13.9 26.0 29.8 VideoCutLER [31] 48.2 22.9 24.5 6.7 17.7 36.3 42.3 AutoQ-VIS 52.6 28.2 28.1 6.7 21.2 40.7 42.5 vs. previous SOTA +4.4 +5.3 +3.6 +0.0 +3.5 +4.4 +0.2

Table 1. YouTubeVIS-2019 val. We reproduced MotionGroup [37], OCLR [34], CutLER [30], and VideoCutLER [31] results with the official code and checkpoints. AutoQ-VIS outperforms the state-of-the-art VideoCutLER by 4.4 AP50. We evaluate results on YouTubeVIS-2019’s val split in a class-agnostic manner.

The training batch is constructed through uniform sampling:

Btrain(k+1)(v) = UniformSample

3

Feligible(k+1)(v) (15)

Although we only use a subset of pseudo-labels, all pseudo-labels persist in the training set regardless of selection status, enabling progressive refinement.

###### 4. Experiments

Datasets. Our model is trained on synthetic videos from VideoCutLER [31] and the unlabeled train split of YouTubeVIS-2019 [39]. We evaluate our model’s performance on the val split of YouTubeVIS-2019 in a classagnostic manner. YouTubeVIS-2019 contains 2,883 highresolution YouTube videos, annotated at 6 FPS (2,238 training videos, 302 validation videos, and 343 test videos).

Evaluation metrics. Following VideoCutLER [31], we use Average Precision (AP) and Average Recall (AR) as evaluation metrics. We evaluate the models in a class-agnostic manner, treating all classes as a single one during evaluation.

Implementation details. For the initial training stage, we use the pretrained VideoMask2Former model [2, 3] from VideoCutLER [31] to initialize our VIS model. Then we train the VIS model and quality predictor on synthetic videos from VideoCutLER for 8,000 iterations using a single V100 GPU, with a batch size of 2 and a learning rate of 2 × 10−5. For each round of multi-round self-training, we train the VIS model for 10,000 iterations on two V100 GPUs, with a batch size of 4 and a learning rate of 2×10−5. In practice, we find that two rounds of self-training and a quality score threshold τth of 0.75 provide the best performance.

To balance data distribution between VideoCutLER’s [31] extensive synthetic videos and our pseudolabeled videos, we implement a balanced stochastic

YouTubeVIS-2019 Method

AP50 AP75 AP APS APM APL AR10 Theoretical limit 76.8 48.7 46.8 13.5 43.6 62.9 58.0 Practical limit 62.7 33.2 33.9 4.3 27.3 53.2 47.5 AutoQ-VIS 52.6 28.2 28.1 6.7 21.2 40.7 42.5

Table 2. Comparison with the theoretical and practical limit. Theoretical Limit: Upper-bound performance achieved by training on ground-truth labels from YouTubeVIS-2019 train split in class-agnostic mode, representing ideal supervision conditions. Practical Limit: Best attainable performance when using all pseudo-labels with IoU ≥ 0.5 against ground truth, simulating perfect pseudo-label selection.

sampling strategy. Each training batch has an equal probability (50%) of being drawn from either the synthetic videos or the pseudo-labeled set. This prevents overfitting to either domain while maintaining the synthetic data’s regularization benefits during self-training.

###### 4.1. Experimental results

Comparison with the-state-of-the-art method. We compare AutoQ-VIS with previous top-performing methods on YouTubeVIS-2019 [39] in Tab. 1. AutoQ-VIS achieves a remarkable improvement (about 4.4% AP50). Especially for AP75, AutoQ-VIS can outperform the state-of-the-art VideoCutLER [31] by 5.3%. The observed improvements of +2.4% APL, +1.2% APM, and +0.0% APL suggest that our pipeline primarily enhances segmentation performance for medium and large objects, with negligible gains on other categories.

Comparison with the theoretical and practical limit. Tab. 2 reveals a significant performance gap (10.1 AP50) between AutoQ-VIS and the practical upper bound, indicating substantial potential for improvement through enhanced pseudo-label utilization. The theoretical limit represents a fully supervised training using all ground-truth annotations from YouTubeVIS-2019 train set. The practical limit represents an oracle experiment that simulates perfect pseudo-label selection by using all predictions with IoU ≥ 0.5 against ground truth.

Qualitative results. Qualitative results in Fig. 3 illustrate AutoQ-VIS’s capability in integrated multi-object segmentation, temporally consistent tracking, and mask-level quality prediction throughout video sequences. Fig. 4 demonstrates AutoQ-VIS’s advancements over VideoCutLER [31]. As we observe, AutoQ-VIS is capable of discovering more objects and producing higher-quality segmentation masks.

###### 4.2. Ablation studies

Component ablation analysis. Tab. 3 quantifies individual component contributions through progressive additions.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Time

- Figure 3. The qualitative results of AutoQ-VIS on YouTubeVIS-2019 val split. The quality scores are shown in the center of each object. The visual results demonstrate AutoQ-VIS’ proficiency in simultaneous multi-instance segmentation, persistent object tracking, and per-mask quality assessment across video sequences.

Video

CutLER Our

Method Ground

Truth

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

[Figure 45]

- Figure 4. The qualitative comparison on YouTubeVIS-2019 val split. AutoQ-VIS demonstrates superior instance discovery capabilities compared to VideoCutLER [31]: (1) Enhanced multiobject detection capacity, particularly for semantically distinct instances (e.g., person and bull in Column 2); (2) Improved segmentation fidelity through precise boundary delineation (e.g., the leopard in Column 3). (3) Better comprehensive instance coverage, eliminating false negatives (e.g., detecting humans in Columns 1 & 4 that VideoCutLER completely misses, even without occlusion or scale challenges).

YouTubeVIS-2019 Method

AP50 AP75 AP APS APM APL AR10 w/o quality predictor 50.5 25.9 27.2 5.7 19.0 40.5 43.3 w/o DropLoss 48.0 23.7 24.6 3.8 16.9 37.8 39.2 w/o resetting each round 51.6 28.0 28.2 6.4 22.8 40.6 42.9 w/o adaptive fusion 49.7 26.5 26.8 6.1 18.9 40.1 41.4 w/o freezing predictor 49.2 25.8 26.5 6.9 18.8 39.3 40.9 AutoQ-VIS 52.6 28.2 28.1 6.7 21.2 40.7 42.5 VideoCutLER [31] 48.2 22.9 24.5 6.7 17.7 36.3 42.3

Table 3. Ablation study on the contribution of each component. Without quality predictor: We remove the quality predictor, and use the confidence score sl from the class head of VIS model (see Fig. 2) as quality score Ql with threshold τth = 0.85. Without DropLoss: We use the vanilla loss for the mask head instead of the DropLoss. Without resetting each round: The model weights are not reset at the beginning of each round. Without Adaptive fusion: Instead of fusing the newly curated annotations and old curated annotations, we simply replace the old curated annotations in the training dataset with new curated annotations. Without freezing predictor: We do not freeze the weight of the quality predictor and train it together with the VIS model.

The DropLoss contributes the most significant performance improvement (+4.6% AP50) in our framework. While originally developed by CutLER [30] for image-level segmentation training, this technique was notably absent in Video-

CutLER’s [31] video instance segmentation pipeline. Our quantitative results in Tab. 3 demonstrate the importance of reintroducing DropLoss in our video-domain self-training

YouTubeVIS-2019 Self-training

AP50 AP75 AP APS APM APL AR10

- 1 round 51.3 26.5 26.9 7.0 22.3 38.4 43.2
- 2 rounds 52.6 28.2 28.1 6.7 21.2 40.7 42.5
- 3 rounds 52.0 27.0 27.7 6.2 21.8 40.1 42.1

- Table 4. Ablation study on the number of self-training rounds. Our framework achieves peak performance (52.6 AP50) at the second round before gradual degradation (-0.6 AP50) from pseudolabel noise accumulation. This establishes round 2 as the optimal stopping point to balance accuracy and error propagation risks.

YouTubeVIS-2019 τth

AP50 AP75 AP APS APM APL AR10 0.95 48.7 25.3 26.1 5.9 18.8 38.5 40.7 0.85 48.8 24.6 26.0 5.8 18.6 38.6 41.2 0.75 52.6 28.2 28.1 6.7 21.2 40.7 42.5 0.50 52.4 25.7 27.1 6.5 20.1 39.9 42.2

- Table 5. Ablation study on the quality score threshold τth. Optimal performance (52.6 AP50) emerges at τth = 0.75, balancing valid sample retention and noise suppression. Lower threshold (τth = 0.50) degrades results by admitting too many lowquality predictions, while the higher thresholds (τth = 0.95 and τth = 0.80) oversuppress valid samples.

pipeline.

Subsequent components demonstrate incremental gains: freezing the quality predictor contributes +3.4% AP50, adaptive fusion adds +2.9% AP50, and the quality predictor itself provides +2.1% AP50 improvement. Freezing quality predictor prevents overestimation bias by decoupling the optimization process between the VIS model and the quality predictor. Without parameter freezing, both components would reinforce pseudo-label errors through mutual confirmation during self-training cycles, thereby inducing progressively amplified confidence miscalibration in successive iterations. Adaptive fusion of new curated annotations and old curated annotations helps the model achieve progressive improvements. Notably, even the confidence score baseline surpasses VideoCutLER by +2.3 AP50, demonstrating fundamental advantages of our self-training method. While model resetting yields marginal gains (+1.0 AP50, +0.2 AP75), it maintains baseline AP performance.

Self-training round analysis. Tab. 4 tests how many selftraining rounds work best. The model hits its peak (52.6 AP50) at round 2, then slowly gets worse. This drop occurs because, as we perform more rounds, mistakes in the pseudo-labels pile up.

Quality score threshold τth analysis. Tab. 5 examines the impact of quality score thresholds on pseudo-label selection. We observe a non-monotonic relationship: While

Quality Score vs. IoU

1.0

| |[Figure 46]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.6

IoU

0.4

0.2

0.0 0.2 0.4 0.6 0.8 1.0 Quality Score, ( s=0.57)

Confidence Score vs. IoU

1.0

| |[Figure 47]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.6

IoU

0.4

0.2

0.0 0.2 0.4 0.6 0.8 1.0 Confidence Score, ( s=0.42)

Figure 5. Visualized comparison of quality score Ql and confidence score sl on YouTubeVIS-2019 val split. Here, ρs denotes the Spearman’s rank correlation coefficient. Subplot (a) visualizes quality scores Ql and their ground truth IoU. Subplot (b) visualizes confidence scores sl and their ground truth IoU.

lower thresholds (τth ≤ 0.75) generally yield superior performance by retaining more valid samples, excessively lenient selection (τth = 0.50) introduces noisy supervision, degrading results. The optimal balance occurs at τth = 0.75, achieving peak performance.

Quality score vs. confidence score. As shown in Fig. 5, our quality score Qs has a higher correlation to the IoU of the pseudo label and the ground truth label than the confidence score of VideoMask2Former [2], which proves our quality predictor’s effectiveness in pseudo-label quality assessment. This results in a significant improvement in the final VIS model performance (+2.1 AP50) in Tab. 3.

###### 4.3. Discussion

Alignment Between Quality Score, IoU, and Area Distributions. As shown in Figs. 6 and 7, our quality score exhibits a strong alignment with ground-truth IoU across different object sizes, reflecting mask reliability with only a limited overestimation for large instances. This consistency ensures that the confidence assigned to pseudo labels is well aligned with their true segmentation quality. Furthermore, the area distribution analysis across self-training rounds (Fig. 7) indicates that the pseudo-label set remains stable, without substantial drift over iterations. Compared to ground-truth annotations, however, pseudo labels display a bias toward larger objects, which is consistent with the slight overestimation observed in quality scores. Impor-

Average Quality Score by Area Category

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

AverageQualityScore

0.3

0.2

0.1

0.0

Small Medium Large Area Category

Average IoU by Area Category

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.3

AverageIoU

0.2

0.1

0.0

Small Medium Large Area Category

- Figure 6. Comparison of average ground-truth IoU and quality score across different mask sizes. The definitions of Small, Medium, and Large follow those in COCO [19]. Here, IoU refers to the intersection-over-union between the predicted mask and the ground-truth annotation. We compare the average IoU and quality score across these area categories. The results demonstrate a strong alignment between our quality score and the ground-truth IoU, with only a limited overestimation for large objects. This minor bias does not substantially affect the distribution of pseudo labels across training rounds (as shown in Fig. 7).

6 4 2 0 Log Area

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

- 0.5

Density

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

Distribution of Mask Areas

Small/Medium Threshold Medium/Large Threshold Ground Truth

- Round 0

- Round 1

- Round 2

- Figure 7. Comparison of object area distributions between pseudo labels across different rounds and ground-truth labels. Here, Log Area refers to the natural logarithm of the normalized area (i.e., object area divided by image area). The definitions of Small, Medium, and Large follow those in COCO [19]. Round 0 denotes pseudo labels produced by the model in the beginning of the self-training. As shown, the object area distributions remain largely stable across different rounds. Nonetheless, relative to the ground truth, our pseudo-label set exhibits a higher proportion of large objects.

tantly, this bias remains limited and does not hinder the effectiveness of iterative self-training.

UVO-Dense AP50 AP75 AP APS APM APL AR10 CutLER [30] 10.3 3.4 4.6 1.3 11.5 17.3 8.8 VideoCutLER [31] 13.5 5.1 6.2 1.7 14.8 26.5 12.3 AutoQ-VIS 14.6 5.8 7.0 1.9 16.0 28.9 12.3 vs. prev. SOTA +1.1 +0.7 +0.8 +0.2 +1.2 +2.4 +0.0

Method

Table 6. UVO-Dense val. We reproduced the state-of-the-art VideoCutLER [31] with the official code and checkpoint. AutoQVIS outperforms the previous state-of-the-art VideoCutLER by 1.1 AP50. We evaluate results on UVO-Dense’s val split in a class-agnostic manner.

Generalizability. To further assess the generalizability of our pipeline, we conduct experiments on UVO-Dense [29]. Following the same setup as in YouTubeVIS-2019, we train our model on synthetic videos from VideoCutLER [31] together with the unlabeled train split of UVO-Dense, and evaluate it on the val split. UVO-Dense comprises 759 videos sourced from Kinetics-400 [12], annotated at 30 FPS (503 for training and 256 for validation). Compared to YouTubeVIS-2019, UVO-Dense is considerably more challenging: it contains seven times more instance annotations per video and features crowded scenes with complex background motions. The training procedure remains identical to that used for YouTubeVIS-2019 [39], except that we lower the quality score threshold to 0.3 to accommodate the higher object density in UVO-Dense.

Table 6 compares AutoQ-VIS with prior state-of-the-art methods reported in [29]. AutoQ-VIS achieves a consistent improvement of about 1.1% in AP50, highlighting the robustness and generalizability of our pipeline.

Evaluation Scope and Protocol Differences. We do not report results on YouTubeVIS-2021 [39], OVIS [25], or similar benchmarks because they do not release groundtruth annotations for the val split, and class-agnostic VIS evaluation requires access to these labels. We also clarify why our reproduced VideoCutLER [31] baseline (48.2 AP50) differs from the number in its original paper (50.7 AP50): VideoCutLER reports results on the train split, whereas we evaluate on the val split, following same standard VIS evaluation practice.

###### 5. Conclusion

We present AutoQ-VIS, a quality-aware self-training framework that advances unsupervised video instance segmentation through iterative pseudo-label refinement with automatic quality control. By establishing a closed-loop system of pseudo-label generation and automatic quality assessment, our method achieves state-of-the-art performance (52.6 AP50) on YouTubeVIS-2019 val split without requiring any human annotations. The simple quality predictor proves effective in pseudo-label quality assessment.

###### Acknowledgements

D. P. Papadopoulos and M. O. Kaya were supported by the DFF Sapere Aude Starting Grant “ACHILLES”.

###### References

- [1] Nikita Araslanov, Simone Schaub-Meyer, and Stefan Roth. Dense unsupervised learning for video segmentation. Advances in Neural Information Processing Systems, 34: 25308–25319, 2021. 1
- [2] Bowen Cheng, Anwesa Choudhuri, Ishan Misra, Alexander Kirillov, Rohit Girdhar, and Alexander G Schwing. Mask2former for video instance segmentation. arXiv preprint arXiv:2112.10764, 2021. 2, 3, 4, 5, 7
- [3] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 2, 3, 4, 5
- [4] Subhabrata Choudhury, Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian Rupprecht. Guess What Moves: Unsupervised Video and Image Segmentation by Anticipating Motion. In British Machine Vision Conference (BMVC),

2022. 1, 2

- [5] Ruihang Chu, Xiaoqing Ye, Zhengzhe Liu, Xiao Tan, Xiaojuan Qi, Chi-Wing Fu, and Jiaya Jia. Twist: Two-way inter-label self-training for semi-supervised 3d instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1100–1109,

2022. 3

- [6] Thanos Delatolas, Vicky Kalogeiton, and Dim P Papadopoulos. Learning the what and how of annotation in video object segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024. 3
- [7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 1, 2, 3
- [8] Emanuela Haller and Marius Leordeanu. Unsupervised object segmentation in video by efficient selection of highly probable positive features. In Proceedings of the IEEE international conference on computer vision, pages 5085–5093,

2017. 1

- [9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 3
- [10] Miran Heo, Sukjun Hwang, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. Vita: Video instance segmentation via object token association. Advances in neural information processing systems, 35:23109–23120, 2022. 1, 2
- [11] Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, and Xinggang Wang. Mask scoring r-cnn. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6409–6418, 2019. 2, 3

- [12] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950,

2017. 8

- [13] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 2020. 3
- [14] Hala Lamdouar, Charig Yang, Weidi Xie, and Andrew Zisserman. Betrayed by motion: Camouflaged object discovery via motion segmentation. In Proceedings of the Asian conference on computer vision, 2020. 2
- [15] Minhyeok Lee, Suhwan Cho, Dogyoon Lee, Chaewon Park, Jungho Lee, and Sangyoun Lee. Guided slot attention for unsupervised video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3807–3816, 2024. 2
- [16] Fuxin Li, Taeyoung Kim, Ahmad Humayun, David Tsai, and James M Rehg. Video segmentation by tracking many figureground segments. In Proceedings of the IEEE international conference on computer vision, pages 2192–2199, 2013. 2
- [17] Long Lian, Zhirong Wu, and Stella X Yu. Bootstrapping objectness from videos by relaxed common fate and visual grouping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14582– 14591, 2023. 1, 2
- [18] Chung-Ching Lin, Ying Hung, Rogerio Feris, and Linglin He. Video instance segmentation tracking with a modified vae architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13147–13157, 2020. 1, 2
- [19] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 8
- [20] Etienne Meunier, Ana¨ıs Badoual, and Patrick Bouthemy. Em-driven unsupervised learning for efficient motion segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4462–4473, 2022. 1, 2
- [21] Peter Ochs, Jitendra Malik, and Thomas Brox. Segmentation of moving objects by long term video analysis. IEEE transactions on pattern analysis and machine intelligence, 36(6): 1187–1200, 2013. 2
- [22] Dim P Papadopoulos, Jasper RR Uijlings, Frank Keller, and Vittorio Ferrari. We don’t need no bounding-boxes: Training object class detectors using only human verification. In CVPR, 2016. 3
- [23] Dim P Papadopoulos, Ethan Weber, and Antonio Torralba. Scaling up instance annotation via label propagation. In ICCV, 2021. 3
- [24] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video

- object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732, 2016. 2
- [25] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip Torr, and Song Bai. Occluded video instance segmentation: A benchmark. International Journal of Computer Vision, 2022. 8
- [26] Sucheng Ren, Wenxi Liu, Yongtuo Liu, Haoxin Chen, Guoqiang Han, and Shengfeng He. Reciprocal transformations for unsupervised video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15455–15464, 2021. 2
- [27] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 1, 2

- [28] Carles Ventura, Miriam Bellver, Andreu Girbau, Amaia Salvador, Ferran Marques, and Xavier Giro-i Nieto. Rvos: Endto-end recurrent network for video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5277–5286, 2019. 2
- [29] Weiyao Wang, Matt Feiszli, Heng Wang, and Du Tran. Unidentified video objects: A benchmark for dense, openworld segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10776– 10785, 2021. 2, 8
- [30] Xudong Wang, Rohit Girdhar, Stella X Yu, and Ishan Misra. Cut and learn for unsupervised object detection and instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3124– 3134, 2023. 1, 2, 3, 5, 6, 8
- [31] Xudong Wang, Ishan Misra, Ziyun Zeng, Rohit Girdhar, and Trevor Darrell. Videocutler: Surprisingly simple unsupervised video instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22755–22764, 2024. 1, 2, 3, 4, 5, 6, 8
- [32] Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia. End-to-end video instance segmentation with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8741–8750, 2021. 1, 2
- [33] Junfeng Wu, Yi Jiang, Song Bai, Wenqing Zhang, and Xiang Bai. Seqformer: Sequential transformer for video instance segmentation. In European Conference on Computer Vision, pages 553–569. Springer, 2022. 1, 2
- [34] Junyu Xie, Weidi Xie, and Andrew Zisserman. Segmenting moving objects via an object-centric layered representation. Advances in neural information processing systems, 35: 28023–28036, 2022. 1, 2, 5
- [35] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687– 10698, 2020. 3
- [36] Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen,

- and Thomas Huang. Youtube-vos: Sequence-to-sequence video object segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 585– 601, 2018. 2
- [37] Charig Yang, Hala Lamdouar, Erika Lu, Andrew Zisserman, and Weidi Xie. Self-supervised video object segmentation by motion grouping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7177–7188,

2021. 1, 2, 5

- [38] Jihan Yang, Shaoshuai Shi, Zhe Wang, Hongsheng Li, and Xiaojuan Qi. St3d: Self-training for unsupervised domain adaptation on 3d object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10368–10378, 2021. 3
- [39] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5188–5197, 2019. 1, 2, 5, 8
- [40] Shu Yang, Lu Zhang, Jinqing Qi, Huchuan Lu, Shuo Wang, and Xiaoxing Zhang. Learning motion-appearance coattention for zero-shot video object segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1564–1573, 2021. 2
- [41] Yanchao Yang, Antonio Loquercio, Davide Scaramuzza, and Stefano Soatto. Unsupervised moving object detection via contextual information separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 879–888, 2019. 2
- [42] Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. A scalable active framework for region annotation in 3d shape collections. ACM Transactions on Graphics (ToG), 2016. 3
- [43] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2636–2645, 2020. 1
- [44] Kaihua Zhang, Zicheng Zhao, Dong Liu, Qingshan Liu, and Bo Liu. Deep transport network for unsupervised video object segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8781–8790,

2021. 1

- [45] Tao Zhang, Xingye Tian, Yu Wu, Shunping Ji, Xuebo Wang, Yuan Zhang, and Pengfei Wan. Dvis: Decoupled video instance segmentation framework. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1282–1291, 2023. 1, 2
- [46] Tianfei Zhou, Jianwu Li, Shunzhou Wang, Ran Tao, and Jianbing Shen. Matnet: Motion-attentive transition network for zero-shot video object segmentation. IEEE transactions on image processing, 29:8326–8338, 2020. 2
- [47] Tianfei Zhou, Fatih Porikli, David J Crandall, Luc Van Gool, and Wenguan Wang. A survey on deep learning technique for video segmentation. IEEE transactions on pattern analysis and machine intelligence, 45(6):7099–7122, 2022. 1

- [48] Yang Zou, Zhiding Yu, BVK Kumar, and Jinsong Wang. Unsupervised domain adaptation for semantic segmentation via class-balanced self-training. In Proceedings of the European conference on computer vision (ECCV), pages 289– 305, 2018. 3
- [49] Yang Zou, Zhiding Yu, Xiaofeng Liu, BVK Kumar, and Jinsong Wang. Confidence regularized self-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5982–5991, 2019. 3

#### Boosting Unsupervised Video Instance Segmentation with Automatic Quality-Guided Self-Training

##### Supplementary Material

###### 6. Detailed methodology

This section supplements what is not clearly stated in Sec. 3.2.

###### 6.1. Automated pseudo-annotation with spatiotemporal NMS

After the training of the VIS model, we use it to label the unlabeled videos. Let D = {di}Ni=1 denote the initial detection set per video, where each detection d = (si,{mti}Tt=1) contains:

- • si ∈ [0,1]: Confidence score
- • {mti}Tt=1: Binary mask sequence across T frames We filter those detections that have confidence scores

larger than or equal to 0.25:

Dfiltered = {di ∈ D | si ≥ 0.25} (16)

However, the detection sets may contain duplicate detections. To solve this problem, we need to perform spatiotemporal non-maximum suppression. First, we sort the detections based on their confidence scores:

Dsorted = {d(k)}|Dk=1filtered| s.t. ∀i < j : s(i) ≥ s(j) (17)

Our method eliminates redundant detections through spatiotemporal overlap analysis: Any lower-confidence prediction is suppressed if exhibiting mask overlap (IoU ≥ 0.5) with higher-confidence detections in at least one video frame. Formally, let Dsuppressed ⊆ Dsorted represent the preserved detection set after suppression:

d(k) ∈ Dsuppressed ⇐⇒ ∄d(p) ∈ Dsuppressed where p < k s.t. ∃t ∈ [1,T] :

∥mt(k) ∩ mt(p)∥ ∥mt(k) ∪ mt(p)∥

≥ 0.5

Frame-specific overlap condition

(18)

where ∥ · ∥ denotes pixel cardinality. This temporalexistential criterion suppresses duplicates appearing in any frame of the video sequence.

###### 6.2. Confidence-aware filtration via quality predictor

Let Dglobal(k) denote the union of Dsuppressed of all videos:

Dglobal(k) =

Dsuppressed(k) ,v (19)

v∈V

where Dsuppressed(k) ,v denotes preserved detections for video v after spatiotemporal NMS at iteration k.

For each detection d ∈ Dglobal(k) , we define:

Qtd = sd · IoUˆ td ∀t ∈ [1,Tv] (20)

where Qtd is the quality score of detection d in frame t, sd is the confidence score of detection d, and Tv denotes the number of frames in video v.

We implement quality-based pseudo-label selection using a fixed quality threshold τth. For each detection d ∈ Dglobal(k) across all videos:

Sdt =

1 Qtd ≥ τ(k) 0 otherwise

(21)

where Sdt denote whether pseudo-label of detection d in frame t is selected. For each detection, if its results are not

selected in any frames, we discard it:

Dretained(k) = dv ∈ Dglobal(k)

Tv

Sdt > 0 (22)

t=1

where Dretained(k) is the set of detections we retain in iteration k.

###### 7. Additional qualitative visualizations

We provide additional qualitative results of our VIS model and quality predictor in Fig. 8 and Fig. 9.

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

## Time

Figure 8. Qualitative results of our VIS model on YouTubeVIS-2019 val split.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

### Time

Figure 9. Qualitative results of our quality predictor on YouTubeVIS-2019 train split. The quality scores are shown in the center of each pseudo label.

