# arXiv:2411.11922v2[cs.CV]30Nov2024

## SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

Cheng-Yen Yang Hsiang-Wei Huang Wenhao Chai Zhongyu Jiang Jenq-Neng Hwang University of Washington

{cycyang, hwhuang, wchai, zyjiang, hwang} @ uw.edu

### Abstract

The Segment Anything Model 2 (SAM 2) has demonstrated strong performance in object segmentation tasks but faces challenges in visual object tracking, particularly when managing crowded scenes with fast-moving or self-occluding objects. Furthermore, the fixed-window memory approach in the original model does not consider the quality of memories selected to condition the image features for the next frame, leading to error propagation in videos. This paper introduces SAMURAI, an enhanced adaptation of SAM 2 specifically designed for visual object tracking. By incorporating temporal motion cues with the proposed motionaware memory selection mechanism, SAMURAI effectively predicts object motion and refines mask selection, achieving robust, accurate tracking without the need for retraining or fine-tuning. SAMURAI operates in real-time and demonstrates strong zero-shot performance across diverse benchmark datasets, showcasing its ability to generalize without fine-tuning. In evaluations, SAMURAI achieves significant improvements in success rate and precision over existing trackers, with a 7.1% AUC gain on LaSOText and a 3.5% AO gain on GOT-10k. Moreover, it achieves competitive results compared to fully supervised methods on LaSOT, underscoring its robustness in complex tracking scenarios and its potential for real-world applications in dynamic environments. Code and results are available at https://github.com/yangchris11/samurai.

### 1. Introduction

Segment Anything Model (SAM) [26] has demonstrated impressive performance in segmentation tasks. Recently, SAM 2 [35] incorporates a streaming memory architecture, which enables it to process video frames sequentially while maintaining context over long sequences. While SAM 2 has shown remarkable capabilities in Video Object Segmentation (VOS [46]) tasks, generating precise pixel-level masks for objects throughout a video sequence, it still faces chal-

lenges in Visual Object Tracking (VOT [36]) scenarios.

The primary concern in VOT is maintaining consistent object identity and location despite occlusions, appearance changes, and the presence of similar objects. However, SAM 2 often neglects motion cues when predicting masks for subsequent frames, leading to inaccuracies in scenarios with rapid object movement or complex interactions. This limitation is particularly evident in crowded scenes, where SAM 2 tends to prioritize appearance similarity over spatial and temporal consistency, resulting in tracking errors. As illustrated in Figure 1, there are two common failure patterns: confusion in crowded scenes and ineffective memory utilization during occlusions.

To address these limitations, we propose incorporating motion information into SAM 2’s prediction process. By leveraging the history of object trajectories, we can enhance the model’s ability to differentiate between visually similar objects and maintain tracking accuracy in the presence of occlusions. Additionally, optimizing SAM 2’s memory management is crucial. The current approach [14, 35] of indiscriminately storing recent frames in the memory bank introduces irrelevant features during occlusions, compromising tracking performance. Addressing these challenges is essential to adapt SAM 2’s rich mask information for robust video object tracking.

To this end, we propose SAMURAI, a SAM-based Unified and Robust zero-shot visual tracker with motionAware Instance-level memory. Our proposed method incorporates two key advancements: (1) a motion modeling system that refines the mask selection, enabling more accurate object position prediction in complex scenarios, and (2) an optimized memory selection mechanism that leverages a hybrid scoring system, combining the original mask affinity, object, and motion scores to retain more relevant historical information, so as to enhance the model’s overall tracking reliability.

In conclusion, this paper makes the following contributions:

• We enhance the visual tracking accuruacy of SAM 2

[Figure 1]

###### Case 1: Ambiguous prediction in crowded scene with similar appearance Consider motion during mask selection!

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

| | |
|---|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 8]

###### Case 2: Ambiguous prediction in occlusion resulting bad memory feature Motion-aware memory selection!

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

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

GT Ambiguous Prediction Incorrect Prediction

Figure 1. Illustration of two common failure cases in visual object tracking using SAM 2: (1) In a crowded scene with similar appearances between target and background objects, SAM 2 tends to ignore the motion cue and predict where the mask has the higher IoU score. (2) The original memory bank simply chooses and stores the previous n frames into the memory bank, resulting in introducing some bad features during occlusion.

by incorporating motion information through motion modeling, to effectively handle the fast-moving and occluded objects.

- • We proposed a motion-aware memory selection mechanism that reduces error in crowded scenes in contrast to the original fixed-window memory by selectively storing relevant frames decided by a mixture of motion and affinity scores.
- • Our zero-shot SAMURAI achieves state-of-the-art

performance on LaSOT, LaSOText, GOT-10k, and other VOT benchmarks without additional training or fine-tuning, demonstrating strong generalization of our proposed modules across diverse datasets.

### 2. Related Works

#### 2.1. Visual Object Tracking (VOT)

Visual Object Tracking (VOT) [36] aims to track objects in challenging video sequences that include variations in object scale, occlusions, and complex backgrounds so as to elevate the robustness and accuracy of tracking algorithms. Siamese-based [10, 52] and transformer-based [12, 47] trackers are common approaches by learning embedding similarity. However, due to lacking self-correction of these trackers in the single forward pass evaluation scheme, they can easily drift toward distractors. To this end, recent works [18, 49] further introduce memory bank and attention to find a better mapping between current frame and history information.

#### 2.2. Segment Anything Model (SAM)

The Segment Anything Model (SAM) [26] has sparked considerable follow-up research since its introduction. SAM introduces a prompt-based segmentation approach, where

users could input points, bounding boxes, or text to guide the model in segmenting any object within an image. The use of SAM has wide-ranging applications like in video understanding [7, 38, 39] and editing [6]. Since then, various works have built upon SAM. For example, SAM 2 [35] expands the model’s capabilities to video segmentation [11], incorporating memory mechanisms for tracking objects across multiple frames in dynamic video sequences. Additionally, efforts have been made to create more efficient variants of SAM for resource-constrained environments, aiming to reduce its computational demands [45, 54]. Research in medical imaging has also adopted SAM for specialized tasks [30]. Recently, SAM2Long [14] uses treebased memory to enhance object segmentation for long video. However, their higher FPS video sequences and deeper memory tree architectures require exponentially more computing power and memory storage due to the overhead of storing exact paths and time-constrained memory paths. On the other hand, our proposed SAMURAI model, which is built upon SAM 2, has been trained on large-scale segmentation datasets to overcome these challenges and ensure good performance and generalization ability.

#### 2.3. Motion Modeling

Motion modeling is an important component in tracking tasks, which can be categorized into heuristic and learnable approaches. Heuristic methods, such as the widelyused Kalman Filter (KF) [24], rely on fixed motion priors and predefined hyper-parameters to predict object trajectories. While KF has proven effective in many tracking benchmarks, it often fails in scenarios with intense or abrupt motion. Other methods [1] attempt to counteract intense or abrupt object motion by compensating for camera movement before applying traditional KF-based predic-

tion. However, both the standard and noise scale adaptive (NSA) Kalman Filters [15] come with a multitude of hyperparameters, potentially restricting their effectiveness to specific types of motion scenarios. In contrast, learnable motion models have attracted increasing interest due to their data-driven nature. Tracktor [2] is the first to use trajectory boxes as Regions of Interest (RoI) in a Faster-RCNN to extract features and regress the object’s position across frames. MotionTrack [43] enhances tracking by learning past trajectory representations to predict future movements. MambaTrack [22] further explores different learning-based motion models architecture like transformer [40] and statespace model (SSM) [21]. Our approach is also a learningbased motion modeling with an enhanced heuristic scheme.

### 3. Revisiting Segment Anything Model 2

Segment Anything Model 2 [34] contains (1) an image encoder, (2) a mask decoder with a prompt encoder, (3) a memory attention layer, and (4) a memory encoder. We will introduce some preliminaries of SAM 2 and specifically point out the part where SAMURAI is being added.

Prompt Encoder. The prompt encoder design follows SAM [35], in which it takes two types of prompts, including sparse (e.g., points, bounding boxes) and dense (e.g., masks). The prompt tokens output by the prompt encoder can be represented as xprompt ∈ Ntokens × d. In the visual object tracking, where the ground-truth bounding box of the target object of the first frame t0 is provided, SAM 2 takes the positional encoding for the top-left and bottomright points as inputs while the rest of the sequence uses the predicted mask M¯ t−1 from the previous frame as the input to the prompt encoder.

Mask Decoder. The memory decoder is designed to take the memory-conditioned image embeddings produced by the memory attention layer along with the prompt tokens from the prompt encoder as its inputs. Its multi-head branches can then generate a set of predicted masks, along with the corresponding mask affinity score smask (it is referred to as IoU score in [34, 35]), and one object score sobj for the frame as outputs.

M = {(M0,smask,0),(M1,smask,1),...}. (1)

The affinity mask score prediction of SAM 2 is supervised with MAE loss as it can represent the overall confidence of the mask, while the object prediction is supervised with cross-entropy loss to determine whether a mask should exist in the frame or not. In the original implementation, the final output mask, M¯ = Mi, is selected based on the highest affinity score among the Nmask output masks.

smask,i where sobj,i > 0 (2)

i = arg max

i∈[0,Nmask−1]

However, the affinity score is not a very robust indicator in the case of visual tracking, especially in crowded scenarios where similar objects self-occlude with each other. We introduce an extra motion modeling to keep track of the motion of the target and provide an additional motion score to aid the selection of the prediction.

Memory Attention Layer. The Memory attention block first performs self-attention with the frame embeddings and then performs cross-attention between the image embeddings and the contents of the memory bank. The unconditional image embeddings, therefore, get contextualized with the previous output masks, previous input prompts, and object pointers.

Memory Encoder and Memory Bank. After the mask decoder generates output masks, the output mask is passed through a memory encoder to obtain a memory embedding. A new memory is created after each frame is processed. These memory embeddings are appended to a Memory Bank, which is a first-in-first-out (FIFO) queue of the latest memories generated during video decoding. At any given time t in the sequence, we can form the memory bank Bt as:

] (3)

Bt = [mt−1,mt−2,...,mt−N

mem

which takes the past Nmem frames’ output m as the components of the memory bank.

This straightforward fixed-window memory implementation may suffer from encoding the incorrect or lowconfidence object, which will cause the error to propagate considerably when in the context of a long sequence visual tracking task. Our proposed motion-aware memory selection will replace the original memory bank composition to ensure that better memory features can be kept and conditioned onto the image feature.

### 4. Method

SAM 2 has demonstrated strong performance in basic Visual Object Tracking (VOT) and Video Object Segmentation (VOS) tasks. However, the original model can mistakenly encode incorrect or low-confidence objects, leading to substantial error propagation in long-sequence VOT.

To address the above issues, we propose a Kalman Filter (KF)-based motion modeling on top of the multi-masks selection (in 4.1) and an enhanced memory selection based on a hybrid scoring system that combines affinity and motion scores (in 4.2). These enhancements are designed to

[Figure 15]

Figure 2. The overview of our SAMURAI visual object tracker.

strengthen the model’s ability to track objects accurately in complex video scenarios. Importantly, this approach does not require fine-tuning, nor does it require additional training, and it can be integrated directly into the existing SAM 2 model. By improving the selection of predicted masks without additional computational overhead, this method provides a reliable, real-time solution for online VOT.

#### 4.1. Motion Modeling

Motion modeling has long been an effective approach to Visual Object Tracking (VOT) and Multiple Object Tracking (MOT) [1, 5, 51] in resolving association ambiguities. We employ the linear-based Kalman filter [24] as our baseline to demonstrate the incorporation of motion modeling in improving tracking accuracy.

In our visual object tracking framework, we integrate the Kalman filter to enhance bounding box position and dimension predictions, which in turn helps select the most confident mask out of N candidates from M. We define the state vector x as:

x = [x,y,w,h,x,˙ y,˙ w,˙ h˙ ]T (4) where x, y represents the center coordinate of the bounding box, w and h denote its width and height, respectively, and their corresponding velocities are represented by the dot notation. For each mask Mi, the corresponding bounding box di is derived by computing the minimum and maximum x and y coordinates of the mask’s non-zero pixels. The Kalman filter operates in a predict-correct cycle, where the state prediction xˆt+1|t is given by:

xˆt+1|t = Fxˆt|t, (5)

skf = IoU(xˆt+1|t,M) (6)

the KF-IoU score skf is then computed by calculating the Intersection over Union (IoU) between the predicted masks

M and the bounding box derived from the Kalman filter’s predicted state. We then select the mask that maximizes a weighted sum of the KF-IoU score and the original affinity score:

M∗ = arg max

(αkf ·skf(Mi)+(1−αkf)·smask(Mi)).

Mi

(7) Finally, the update is performed using:

xˆt|t = xˆt|t−1 + Kt(zt − Hxˆt|t−1) (8)

where zt is the measurement, the bounding box derived from the mask we selected, used to update. F is the linear state transition matrix, Kn is the Kalman gain, and H is the observation matrix. Furthermore, to ensure the robustness of the motion modeling after the targeted object reappears or the poor mask qualities for a certain period of time, we also maintain a stable motion state where we take consideration of the motion module if and only if the tracked object is being successfully update in the past τkf frames.

#### 4.2. Motion-Aware Memory Selection

The original SAM 2 prepares the conditioned visual feature of the current frame based on selecting Nmem from the previous frames. In [34], the implementation simply selects the Nmem most recent frames based on the qualities of the target. However, this approach has the weakness of not being able to handle longer occlusion or deformation, which is common in visual object tracking tasks.

To construct an effective memory bank of object cues considering motion, we employ a selective approach for choosing frames from previous time steps based on three scoring: the mask affinity score, object occurrence score, and motion score. We select the frame as an ideal candidate for memory if and only if all three scores meet their corresponding thresholds (e.g., τmask, τobj, τkf). We iterate

Table 1. Visual object tracking results on LaSOT [16], LaSOText [17], and GOT-10k [23]. † LaSOText are evaluated on trackers to be trained with LaSOT. ‡ GOT-10k protocol only allows trackers to be trained using its corresponding train split. The T, S, B, L represents the size of the ViT-based backbone while the subscript is the search region. Bold represents the best while underline represents the second.

LaSOT LaSOText† GOT-10k ‡ AUC(%) Pnorm(%) P(%) AUC(%) Pnorm(%) P(%) AO(%) OP0.5(%) OP0.75(%)

Trackers Source

Supervised method SiamRPN++ [27] CVPR’19 49.6 56.9 49.1 34.0 41.6 39.6 51.7 61.6 32.5 DiMP288 [13] CVPR’20 56.3 64.1 56.0 - - - 61.1 71.7 49.2 TransT256 [8] CVPR’21 64.9 73.8 69.0 - - - 67.1 76.8 60.9 AutoMatch255 [53] ICCV’21 58.2 67.5 59.9 - - - 65.2 76.6 54.3 STARK320 [48] ICCV’21 67.1 76.9 72.2 - - - 68.8 78.1 64.1 SwinTrack-B384 [28] NeurIPS’22 71.4 79.4 76.5 - - - 72.4 80.5 67.8 MixFormer288 [12] CVPR’22 69.2 78.7 74.7 - - - 70.7 80.0 67.8 OSTrack384 [50] ECCV’22 71.1 81.1 77.6 50.5 61.3 57.6 73.7 83.2 70.8 ARTrack-B256 [41] CVPR’23 70.8 79.5 76.2 48.4 57.7 53.7 73.5 82.2 70.9 SeqTrack-B384 [9] CVPR’23 71.5 81.1 77.8 50.5 61.6 57.5 74.5 84.3 71.4 GRM-B256 [20] CVPR’23 69.9 79.3 75.8 - - - 73.4 82.9 70.4 ROMTrack-B256 [4] ICCV’23 69.3 78.8 75.6 47.2 53.5 52.9 72.9 82.9 70.2 TaMOs-B384 [32] WACV’24 70.2 79.3 77.8 - - - - - EVPTrack-B384 [37] AAAI’24 72.7 82.9 80.3 53.7 65.5 61.9 76.6 86.7 73.9 ODTrack-B384 [55] AAAI’24 73.2 83.2 80.6 52.4 63.9 60.1 77.0 87.9 75.1 ODTrack-L384 [55] AAAI’24 74.0 84.2 82.3 53.9 65.4 61.7 78.2 87.2 77.3 HIPTrack-B384 [3] CVPR’24 72.7 82.9 79.5 53.0 64.3 60.6 77.4 88.0 74.5 AQATrack-B256 [44] CVPR’24 71.4 81.9 78.6 51.2 62.2 58.9 73.8 83.2 72.1 AQATrack-L384 [44] CVPR’24 72.7 82.9 80.2 52.7 64.2 60.8 76.0 85.2 74.9 LoRAT-B224 [29] ECCV’24 71.7 80.9 77.3 50.3 61.6 57.1 72.1 81.8 70.7 LoRAT-L224 [29] ECCV’24 74.2 83.6 80.9 52.8 64.7 60.0 75.7 84.9 75.0 Zero-shot method

SAMURAI-T Ours 69.3 76.4 73.8 55.1 65.6 63.7 79.0 89.6 72.3 SAMURAI-S Ours 70.0 77.6 75.2 58.0 69.6 67.7 78.8 88.7 72.9 SAMURAI-B Ours 70.7 78.7 76.2 57.5 69.3 67.1 79.6 90.8 72.9 SAMURAI-L Ours 74.2 82.7 80.2 61.0 73.9 72.2 81.7 92.2 76.9

back in time from the current frame and repeat the verification. We select Nmem memories based on the above scoring function and obtain a motion-aware memory bank Bt:

Bt = {mi|f(smask,sobj,skf) = 1,t − Nmax ≤ i < t}

(9) where Nmax is the maximum number of frames to look back. The motion-aware memory bank Bt is subsequently passed through the memory attention layer and then directed to mask decoder Dmask to perform mask decoding at current timestamp. Note that we follow the Nmem = 7 as the SAM 2 is trained under these specific memory bank settings.

The proposed motion modeling and memory selection module can significantly enhance visual object tracking without the need for retraining and does not add any computational overhead to the existing pipeline. It is also modelagnostic and potentially applicable to other tracking frameworks beyond SAM 2. By combining motion modeling with intelligent memory selection, we can enhance tracking performance in challenging real-world applications without sacrificing efficiency.

### 5. Experiments

#### 5.1. Benchmarks

We evaluate the zero-shot performance of our SAMURAI on the following VOT benchmarks:

LaSOT [16] is a visual object tracking dataset comprising 1,400 videos across 70 diverse object categories with an average sequence length of 2,500 frames. It is divided into training and testing sets, consisting of 1,120 and 280 sequences, respectively, with 16 training and 4 testing sequences for each category.

LaSOText [17] is an extension to the original LaSOT dataset, introducing an additional 150 video sequences across 15 new object categories. These new sequences are specifically designed to focus on occlusions and variations in small objects, which is more challenging, and the standard protocol is to evaluate the models trained on LaSOT directly on the LaSOText.

GOT-10k [23] comprises over 10,000 video segments of real-world moving objects, spanning more than 560 object

Success (LaSOT)

80

OverlapPrecision[%]

SAMURAI-L [74.2]

LoRAT-L [74.2]

60

ODTrack-B [74.0]

AQATrack-L [72.7]

HIPTrack-B [72.7]

OSTrack [71.1]

40

SAMURAI-B [70.6]

TaMoS-B [70.2]

ROMTrack-B [69.2]

SAM2.1-L [68.5] SAM2.1-B [66.0] TransT [64.9]

20

AutoMatch [58.2]

DiMP [56.3]

0

0.0 0.2 0.4 0.6 0.8 1.0 Overlap threshold

Normalized Precision (LaSOT)

80

DistancePrecision[%]

ODTrack-B [84.2]

LoRAT-L [83.6]

60

AQATrack-L [82.9]

HIPTrack-B [82.9]

SAMURAI-L [82.7]

OSTrack [81.1]

40

TaMoS-B [79.3]

ROMTrack-B [78.8] SAMURAI-B [78.7] SAM2.1-L [76.2]

20

TransT [73.8]

SAM2.1-B [73.5]

AutoMatch [67.4]

DiMP [64.1]

0

0.0 0.1 0.2 0.3 0.4 0.5 Location error threshold

Success (LaSOText)

80

OverlapPrecision[%]

60

SAMURAI-L [61.0]

SAM2.1-L [58.6]

SAMURAI-B [57.5]

40

SAM2.1-B [55.5]

HIPTrack-B [53.0]

LoRAT L [52.8] AQATrack-L [52.7]

ODTrack L [52.4] OSTrack [50.5]

20

LoRAT B [50.3] ROMTrack B [48.9] ARTrack [48.4]

0

0.0 0.2 0.4 0.6 0.8 1.0 Overlap threshold

Normalized Precision (LaSOText)

80

DistancePrecision[%]

60

SAMURAI-L [73.9]

SAM2.1-L [71.1]

SAMURAI-B [69.3]

40

SAM2.1-B [67.2]

LoRAT L [64.7] HIPTrack-B [64.3]

AQATrack-L [64.2]

ODTrack L [63.9] LoRAT B [61.5] OSTrack [61.3]

20

ROMTrack B [59.3] ARTrack [57.7]

0

0.0 0.1 0.2 0.3 0.4 0.5 Location error threshold

Figure 3. SUC and Pnorm plots of LaSOT and LaSOText.

classes and 80+ motion patterns. A key aspect of GOT-10k is its one-shot evaluation protocol, which requires trackers to be trained exclusively on the designated training split, with 170 videos reserved for testing.

TrackingNet [33] is a large-scale tracking dataset that covers a wide selection of object classes in broad and diverse contexts in the wild. It has a total of 30,643 videos split into 30,132 training videos and 511 testing videos.

NFS [25] consists of 100 videos with a total of 380k frames captured with higher frame rate (240 FPS) cameras from real-world scenarios. We use the 30 FPS version of the data with artificial motion blur following other VOT works.

OTB100 [42] is one of the earliest visual tracking benchmarks that annotated sequences with attribute tags. It contains 100 sequences with an average length of 590 frames.

#### 5.2. Quantitative Results

Results on LaSOT and LaSOText. Table 1 presents the visual object tracking results on the LaSOT and LaSOText datasets. Our method, SAMURAI, demonstrates significant improvements over both the zero-shot and supervised methods on all three metrics (shown in Figure 3). Although the supervised VOT method such as [29, 55] show quite impressive results, the zero-shot SAMURAI in contrast show its great generalization ability with comparalbe zero-shot performance. Furthermore, all SAMURAI models surpass the state-of-the-art on all metrics on LaSOText.

- Table 2. Visual object tracking results on AUC (%) of our proposed method with state-of-the-art methods on TrackingNet [33], NFS [25], and OTB100 [42] datasets. Bold represents the best while underline represents the second.

Trackers TrackingNet NFS OTB100 Supervised method DiMP288 [13] 74.0 61.8 TransT256 [8] 81.4 65.7 STARK320 [48] 82.0 - 68.5 KeepTrack [31] - 66.4 70.9 AiATrack320 [19] 82.7 67.9 69.6 OSTrack384 [50] 83.9 66.5 55.9 SeqTrack-B384 [9] 83.9 66.7 HIPTrack-B384 [3] 84.5 68.1 71.0 AQATrack-L384 [44] 84.8 - LoRAT-L224 [29] 85.0 66.0 72.3 Zero-shot method SAMURAI-L (Ours) 85.3 69.2 71.5

- Table 3. Ablation on the effectiveness of the proposed modules.

Motion Memory AUC(%) Pnorm(%) P(%)

× × 68.32 76.16 73.59 ✓ × 70.81 78.87 76.47 × ✓ 72.67 80.67 78.23 ✓ ✓ 74.23 82.69 80.21

- Table 4. Ablation on the sensitivity of the motion weight αkf.

αkf AUC(%) Pnorm(%) P(%) 0.00 72.67 80.67 78.23 0.15 74.23 82.69 80.21 0.25 73.76 81.86 79.53 0.50 72.92 80.49 78.34

Results on GOT-10k. Table 1 also presents the visual object tracking results on the GOT-10k dataset. Note that the GOT-10k protocol only allows trackers to be trained using its corresponding train split, as some papers may refer to them as a one-shot method. SAMURAI-B shows a 2.1% improvement on AO and 2.9% on OP0.5 over SAM2.1-B while SAMURAI-L shows a 0.6% improvement on AO and 0.7% on OP0.5. All SAMURAI models surpass the stateof-the-art on all metrics on GOT-10k.

Results on TrackingNet, NFS, and OTB100. Table 2 presents the visual object tracking results on four widely compared benchmarks. Our zero-shot SAMURAI-L model is comparable to or can surpass the state-of-the-art supervised method on AUC, showcasing the capability of our model on various datasets and generalization ability.

Table 5. Visual object tracking results of the proposed SAMURAI compare to the baseline SAM-based tracking method.

LaSOT LaSOText

Trackers

AUC(%) Pnorm(%) P(%) AUC(%) Pnorm(%) P(%) SAM2.1-T [34] 66.70 73.70 71.22 52.25 62.03 60.30 SAMURAI-T 69.28 (+2.58) 76.39 (+2.69) 73.78 (+2.56) 55.13 (+2.88) 65.60 (+2.57) 63.72 (+3.42) SAM2.1-S [34] 66.47 73.67 71.25 56.11 67.57 65.81 SAMURAI-S 70.04 (+3.57) 77.55 (+3.88) 75.23 (+3.98) 57.99 (+1.88) 69.60 (+2.03) 67.73 (+1.92) SAM2.1-B [34] 65.97 73.54 70.96 55.51 67.17 64.55 SAMURAI-B 70.65 (+4.68) 78.69 (+4.15) 76.21 (+5.25) 57.48 (+1.97) 69.28 (+2.11) 67.09 (+2.54) SAM2.1-L [34] 68.54 76.16 73.59 58.55 71.10 68.83 SAMURAI-L 74.23 (+5.69) 82.69 (+6.53) 80.21 (+6.62) 61.03 (+2.48) 73.86 (+2.76) 72.24 (+3.41)

Table 6. Attribute-wise AUC(%) Results for LaSOT [16] and LaSOText [17].

LaSOT

Trackers

ARC BC CM DEF FM FOC IV LR MB OV POC ROT SV VC

SAM2.1-B [34] 64.7 62.8 67.7 67.1 56.1 57.6 63.0 55.4 67.1 56.2 64.6 62.8 65.5 59.8 SAMURAI-B 69.6 68.0 73.1 72.0 62.5 63.0 69.6 63.2 70.2 64.5 69.1 68.0 70.3 64.1

- % Gain +7.6% +8.3% +8.0% +7.3% +11.4% +9.4% +10.5% +14.1% +4.6% +14.8% +7.0% +8.3% +7.3% +7.2% SAM2.1-L [34] 67.3 64.3 69.4 70.8 58.4 59.3 63.9 59.7 67.8 61.9 68.0 67.2 68.1 61.1 SAMURAI-L 73.1 69.5 77.0 75.7 63.9 66.8 72.8 67.6 73.8 70.4 72.8 72.7 73.7 71.4

- % Gain +8.9% +8.1% +11.0% +6.9% +9.4% +12.7% +14.3% +13.2% +8.0% +7.8% +7.1% +8.2% +9.2% +16.8%

LaSOText

Trackers

ARC BC CM DEF FM FOC IV LR MB OV POC ROT SV VC

SAM2.1-B 53.4 49.3 58.6 75.4 42.1 42.5 69.5 45.3 42.6 46.1 56.3 61.6 54.4 57.1 SAMURAI-B 54.8 52.5 67.8 73.3 45.9 45.9 67.3 47.4 43.7 48.1 56.7 62.9 56.2 61.8

- % Gain +2.6% +6.5% +15.7% +10.6% +9.0% +8.0% -2.3% +4.6% +2.6% +6.1% +11.5% +2.1% +3.3% +8.2% SAM2.1-L 56.6 53.2 62.8 75.6 46.1 47.6 71.4 48.8 47.1 50.9 60.0 63.2 57.7 61.9 SAMURAI-L 58.4 55.4 73.1 77.5 50.7 50.9 69.0 52.1 49.4 53.3 60.9 64.7 59.9 66.6

- % Gain +3.2% +4.0% +16.5% +5.2% +9.9% +6.9% -3.5% +6.3% +5.4% +4.3% +3.2% +2.6% +4.0% +7.4%

#### 5.3. Ablation Studies

Effect of the Individual Modules. We demonstrate the effect of the with or without memory selection on various settings in Table 3. Both of the proposed modules had a positive impact on the SAM 2 model, while combining both can achieve the best AUC on the LaSOT dataset with an AUC of 74.23% and Pnorm of 82.60%.

Effect of the Motion Weights. We showcase the effect of the weighting of the score of deciding which mask to trust in Table 4. The trade-off between motion score and mask affinity score demonstrates a significant impact on tracking performance. Our experiments reveal that setting the motion weight αmotion = 0.2 yields the best AUC and Pnorm score on the LaSOT dataset, indicating an optimal balance enhances both accuracy and robustness in mask selection.

Baseline Comparison. To demonstrate the effectiveness of the proposed motion modeling and motion-aware memory selection mechanism in SAMURAI, we conduct a detailed apple-to-apple comparison of the SAM 2 [34] at all of the backbone variations on LaSOT and LaSOText. The baseline SAM 2 employs the original memory selection and directly predicts the mask with the highest IoU score. Table 5 shows that the proposed method consistently improves

upon the baseline with a significant margin on all three metrics, which underscores the robustness and generalization of our approach across different model configurations.

Attribute-Wise Performance Analysis. We analysis the LaSOT and LaSOText based on the 14 attributes defined in [16, 17]. In Table 6, SAMURAI shows consistent success in improving upon the original baseline across all attributes in both datasets but the label IV (Illumination Variation) label on LaSOText. By considering motion scoring, the performance gains on attributes like CM (Camera Motion) and FM (Fast Motion) are the largest among the rest, the SAMURAI has a 16.5% and 9.9% gain on CM and FM respectively from LaSOText dataset which is considered one of the most challenging datasets in VOT. Furthermore, the occlusion-related attributes like FOC (Full Occlusion) and POC (Partial Occlusion) also greatly benefited from the proposed motion-aware instance-level memory selection, which showed steady improvement across all model variants and datasets. These findings suggest that the SAMURAI incorporates simple motion estimation to better account for global camera or rapid object movements for better tracking.

Runtime Analysis. The incorporation of the motion modeling and an enhanced memory selection method into

[Figure 16]

HIPTrack LoRAT SAMURAI (Ours) GT

[Figure 17]

SAM2.1 (Baseline) SAMURAI (Ours) GT

Figure 4. Visualization of tracking results comparing SAMURAIwith existing methods. (Top) Conventional VOT methods often struggle in crowded scenarios where the target object is surrounded by objects with similar appearances. (Bottom) The baseline SAM-based method suffers from fixed-window memory composition, leading to error propagation and reduced overall tracking accuracy due to ID switches.

our tracking framework introduces minimal computational overhead, and the runtime measurements conducted on one NVIDIA RTX 4090 GPU remain consistent with the baseline model.

#### 5.4. Qualitative Results

Qualitative comparison between SAMURAI and other methods [3, 29, 34] are shown in Figure 4. SAMURAI demonstrates superior visual object tracking results in scenes where multiple objects with similar appearances are present in the video. The short-term occlusions in these examples make it challenging for existing VOT methods to predict or localize the same object consistently over time. Furthermore, the comparison between SAMURAI and the original baseline with visualized masks showcases the improvement gained by adding the motion modeling and memory selection modules, the predicted masks are not always a reliable source to serve as memory therefore hav-

ing a systematic way of deciding which to trust is valuable. These enhancements benefit the existing framework by providing better guidance for visual tracking without the need to retrain the model or fine-tune it.

### 6. Conclusion

We present SAMURAI, a visual object tracking framework built on top of the segment-anything model by introducing the motion-based score for better mask prediction and memory selection to deal with self-occlusion and abrupt motion in crowded scenes. The proposed modules show consistent improvement on all variations of the SAM models across multiple VOT benchmarks on all metrics. This method does not require re-training nor fine-tuning while demonstrating robust performance on multiple VOT benchmarks with the capability of real-time online inferences.

### References

- [1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Botsort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022. 2, 4
- [2] Philipp Bergmann, Tim Meinhardt, and Laura Leal-Taixe. Tracking without bells and whistles. In Proceedings of the IEEE/CVF international conference on computer vision, pages 941–951, 2019. 3
- [3] Wenrui Cai, Qingjie Liu, and Yunhong Wang. Hiptrack: Visual tracking with historical prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 5, 6, 8
- [4] Yidong Cai, Jie Liu, Jie Tang, and Gangshan Wu. Robust object modeling for visual tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9589–9600, 2023. 5
- [5] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric sort: Rethinking sort for robust multi-object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9686–9696, 2023. 4
- [6] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050, 2023. 2
- [7] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jeng-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051, 2024. 2
- [8] Xin Chen, Bin Yan, Jiawen Zhu, Dong Wang, Xiaoyun Yang, and Huchuan Lu. Transformer tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8126–8135, 2021. 5, 6
- [9] Xin Chen, Houwen Peng, Dong Wang, Huchuan Lu, and Han Hu. Seqtrack: Sequence to sequence learning for visual object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14572– 14581, 2023. 5, 6
- [10] Zedu Chen, Bineng Zhong, Guorong Li, Shengping Zhang, and Rongrong Ji. Siamese box adaptive network for visual tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6668–6677,

2020. 2

- [11] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1316–1326, 2023. 2
- [12] Yutao Cui, Cheng Jiang, Limin Wang, and Gangshan Wu. Mixformer: End-to-end tracking with iterative mixed attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13608–13618,

2022. 2, 5

- [13] Martin Danelljan, Luc Van Gool, and Radu Timofte. Probabilistic regression for visual tracking. In Proceedings of

- the IEEE/CVF conference on computer vision and pattern recognition, pages 7183–7192, 2020. 5, 6
- [14] Shuangrui Ding, Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Yuwei Guo, Dahua Lin, and Jiaqi Wang. Sam2long: Enhancing sam 2 for long video segmentation with a training-free memory tree. arXiv preprint arXiv:2410.16268, 2024. 1, 2
- [15] Yunhao Du, Junfeng Wan, Yanyun Zhao, Binyu Zhang, Zhihang Tong, and Junhao Dong. Giaotracker: A comprehensive framework for mcmot with global information and optimizing strategies in visdrone 2021. In Proceedings of the IEEE/CVF International conference on computer vision, pages 2809–2819, 2021. 3
- [16] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5374–5383,

2019. 5, 7

- [17] Heng Fan, Hexin Bai, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Harshit, Mingzhen Huang, Juehuan Liu, et al. Lasot: A high-quality large-scale single object tracking benchmark. International Journal of Computer Vision, 129: 439–461, 2021. 5, 7
- [18] Zhihong Fu, Qingjie Liu, Zehua Fu, and Yunhong Wang. Stmtrack: Template-free visual tracking with space-time memory networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13774–13783, 2021. 2
- [19] Shenyuan Gao, Chunluan Zhou, Chao Ma, Xinggang Wang, and Junsong Yuan. Aiatrack: Attention in attention for transformer visual tracking. In European Conference on Computer Vision, pages 146–164. Springer, 2022. 6
- [20] Shenyuan Gao, Chunluan Zhou, and Jun Zhang. Generalized relation modeling for transformer tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18686–18695, 2023. 5
- [21] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 3
- [22] Hsiang-Wei Huang, Cheng-Yen Yang, Wenhao Chai, Zhongyu Jiang, and Jenq-Neng Hwang. Exploring learningbased motion models in multi-object tracking. arXiv preprint arXiv:2403.10826, 2024. 3
- [23] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic object tracking in the wild. IEEE transactions on pattern analysis and machine intelligence, 43(5):1562–1577, 2019. 5
- [24] Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. 1960. 2, 4
- [25] Hamed Kiani Galoogahi, Ashton Fagg, Chen Huang, Deva Ramanan, and Simon Lucey. Need for speed: A benchmark for higher frame rate object tracking. In Proceedings of the IEEE international conference on computer vision, pages 1125–1134, 2017. 6
- [26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-

- head, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 1, 2
- [27] Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, and Junjie Yan. Siamrpn++: Evolution of siamese visual tracking with very deep networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4282–4291, 2019. 5
- [28] Liting Lin, Heng Fan, Zhipeng Zhang, Yong Xu, and Haibin Ling. Swintrack: A simple and strong baseline for transformer tracking. Advances in Neural Information Processing Systems, 35:16743–16754, 2022. 5
- [29] Liting Lin, Heng Fan, Zhipeng Zhang, Yaowei Wang, Yong Xu, and Haibin Ling. Tracking meets lora: Faster training, larger model, stronger performance. In European Conference on Computer Vision, pages 300–318. Springer, 2025. 5, 6, 8
- [30] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15(1):654, 2024. 2
- [31] Christoph Mayer, Martin Danelljan, Danda Pani Paudel, and Luc Van Gool. Learning target candidate association to keep track of what not to track. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13444– 13454, 2021. 6
- [32] Christoph Mayer, Martin Danelljan, Ming-Hsuan Yang, Vittorio Ferrari, Luc Van Gool, and Alina Kuznetsova. Beyond sot: Tracking multiple generic objects at once. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6826–6836, 2024. 5
- [33] Matthias Muller, Adel Bibi, Silvio Giancola, Salman Alsubaihi, and Bernard Ghanem. Trackingnet: A large-scale dataset and benchmark for object tracking in the wild. In Proceedings of the European conference on computer vision (ECCV), pages 300–317, 2018. 6
- [34] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3, 4, 7, 8
- [35] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 1, 2, 3
- [36] Giorgio Roffo, Simone Melzi, et al. The visual object tracking vot2016 challenge results. In Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 810 and 15-16, 2016, Proceedings, Part II, pages 777–823. Springer International Publishing, 2016. 1, 2
- [37] Liangtao Shi, Bineng Zhong, Qihua Liang, Ning Li, Shengping Zhang, and Xianxian Li. Explicit visual prompts for visual object tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4838–4846, 2024. 5
- [38] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo,

- Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 2
- [39] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Question-aware sparse memory for long video question answering. arXiv preprint arXiv:2404.17176, 2024. 2
- [40] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 3
- [41] Xing Wei, Yifan Bai, Yongchao Zheng, Dahu Shi, and Yihong Gong. Autoregressive visual tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9697–9706, 2023. 5
- [42] Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang. Object tracking benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9):1834–1848, 2015. 6
- [43] Changcheng Xiao, Qiong Cao, Yujie Zhong, Long Lan, Xiang Zhang, Zhigang Luo, and Dacheng Tao. Motiontrack: Learning motion predictor for multiple object tracking. Neural Networks, 179:106539, 2024. 3
- [44] Jinxia Xie, Bineng Zhong, Zhiyi Mo, Shengping Zhang, Liangtao Shi, Shuxiang Song, and Rongrong Ji. Autoregressive queries for adaptive tracking with spatio-temporal transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19300– 19309, 2024. 5, 6
- [45] Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, et al. Efficientsam: Leveraged masked image pretraining for efficient segment anything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16111–16121, 2024. 2
- [46] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 1
- [47] Bin Yan, Houwen Peng, Jianlong Fu, Dong Wang, and Huchuan Lu. Learning spatio-temporal transformer for visual tracking. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10448–10457, 2021. 2
- [48] Bin Yan, Houwen Peng, Jianlong Fu, Dong Wang, and Huchuan Lu. Learning spatio-temporal transformer for visual tracking. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10448–10457, 2021. 5, 6
- [49] Tianyu Yang and Antoni B Chan. Learning dynamic memory networks for object tracking. In Proceedings of the European conference on computer vision (ECCV), pages 152– 167, 2018. 2
- [50] Botao Ye, Hong Chang, Bingpeng Ma, Shiguang Shan, and Xilin Chen. Joint feature learning and relation modeling for tracking: A one-stream framework. In European Conference on Computer Vision, pages 341–357. Springer, 2022. 5, 6
- [51] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every

- detection box. In European conference on computer vision, pages 1–21. Springer, 2022. 4
- [52] Zhipeng Zhang and Houwen Peng. Deeper and wider siamese networks for real-time visual tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4591–4600, 2019. 2
- [53] Zhipeng Zhang, Yihao Liu, Xiao Wang, Bing Li, and Weiming Hu. Learn to match: Automatic matching network design for visual tracking. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13339–13348,

2021. 5

- [54] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. arXiv preprint arXiv:2306.12156, 2023. 2
- [55] Yaozong Zheng, Bineng Zhong, Qihua Liang, Zhiyi Mo, Shengping Zhang, and Xianxian Li. Odtrack: Online dense temporal token learning for visual tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7588–7596, 2024. 5, 6

