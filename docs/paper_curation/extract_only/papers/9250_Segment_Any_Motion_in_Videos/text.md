# arXiv:2503.22268v2[cs.CV]14Apr2025

## Segment Any Motion in Videos

Nan Huang1,2 Wenzhao Zheng1 Chenfeng Xu1 Kurt Keutzer1 Shanghang Zhang2 Angjoo Kanazawa1 Qianqian Wang1 1UC Berkeley 2Peking University

##### Abstract

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Moving object segmentation is a crucial task for achieving a high-level understanding of visual scenes and has numerous downstream applications. Humans can effortlessly segment moving objects in videos. Previous work has largely relied on optical flow to provide motion cues; however, this approach often results in imperfect predictions due to challenges such as partial motion, complex deformations, motion blur and background distractions. We propose a novel approach for moving object segmentation that combines long-range trajectory motion cues with DINO-based semantic features and leverages SAM2 for pixel-level mask densification through an iterative prompting strategy. Our model employs Spatio-Temporal Trajectory Attention and Motion-Semantic Decoupled Embedding to prioritize motion while integrating semantic support. Extensive testing on diverse datasets demonstrates state-of-the-art performance, excelling in challenging scenarios and fine-grained segmentation of multiple objects. Our code is available at https://motion-seg.github.io/.

Images Inputs

Long-range Tracks Inputs

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Filtered Dynamic Tracks Dynamic Masks

Figure 1. Our method is capable of handling challenging scenarios, including articulated structures, shadow reflections, dynamic background motion, and drastic camera movements, while producing per object level fine-grained moving object masks.

nificant progress. Taking points, masks, or bounding boxes as prompts, SAM2 [51] segments and tracks the associated objects in videos effectively. However, SAM2 cannot natively handle MOS, as it has no mechanism to detect which objects are moving.

##### 1. Introduction

We propose an innovative combination of long-range tracks with SAM2 for moving object segmentation to exploit the capabilities of SAM2. First, point tracking captures valuable long-range pixel motion information which is robust to deformation and occlusion, as shown in Fig. 2. At the same time, we incorporate DINO feature [12, 45], to add semantic context as a complementary source of information to support motion-based segmentation. We depart from traditional MOS approaches by training a model on extensive datasets that effectively combines motion and semantic information at a high level. Given a set of long-range 2D tracks, our model is designed to identify those tracks that correspond to moving objects. Once these dynamic tracks are identified, we apply a sparse-to-dense mask densification strategy, which uses an Iterative Prompting method in conjunction with SAM2 [51] to transform the sparse, point-level mask into a pixel-level segmentation. Since the primary objective is moving object segmentation, we em-

Segmenting moving objects in videos is crucial for a range of applications, including action recognition, autonomous driving [10, 22, 66], and 4D reconstruction [58]. Many prior works address this problem under terms such as Video Object Segmentation (VOS) or motion segmentation. In this paper, we define our task as moving object segmentation (MOS) – segmenting objects that exhibit observable motion within the video. This definition differs from Video Object Segmentation, which includes objects that have the potential to move even if they remain static in the video, and from motion segmentation, which may also capture background motion, such as flowing water. This task is challenging as it implicitly requires distinguishing between camera motion and object motion, robustly tracking objects despite deformations, occlusions, rapid or transient movement, and segmenting them out with precise, clean masks.

Recently, promptable visual segmentation has made sig-

phasize motion cues while using semantic information as secondary support. To effectively balance these two types of information, we propose two specialized modules. (1) Spatio-Temporal Trajectory Attention. Given the long-term nature of input tracks, our model incorporates spatial attention to capture relationships between different trajectories and temporal attention to monitor changes within individual trajectories over time. (2) Motion-Semantic Decoupled Embedding. We implement special attention mechanisms to prioritize motion patterns and process semantic features in supplementary pathways.

We trained our model on extensive datasets, including both synthetic [19, 28] and real-world data [36]. Due to the self-supervised nature of DINO features [45], our model demonstrates strong generalization capabilities, even when primarily trained on synthetic data. We evaluated our approach on benchmarks [34, 43, 47, 48] that were not part of the training data, and the results show that our method significantly outperforms baseline models in diverse tasks.

While previous MOS methods leverage optical flow [6, 8, 9] to capture motion information, either by identifying different motion groups [6, 46, 53, 59] or by using learningbased models [8, 9, 18, 40, 49] to derive pixel masks from optical flow. However, optical flow is limited to short-range motion and can lose track over extended durations. Other methods [3, 7, 14, 42] rely on point trajectories as motion cues, but traditionally utilize spectral clustering on affinity matrices which struggle with complex motions. Though some methods also attempt to take advantage of appearance cues [24, 61] to help understand motion better, they typically handle different modalities in diverse separate stages, limiting the effective integration of their complementary information. Addressing these limitations, our unified framework achieves threefold integration: long-range trajectory, DINO feature, and SAM2. This design explains the model’s exceptional capability in handling challenging cases like articulated motion and reflective surfaces as shown in the Fig. 1, and the superior performance in fine-grained segmentation of multiple objects.

In summary, we make the following contributions:

- • We introduce an innovative combination of long-range tracks with SAM2, which enables efficient mask densification and tracking across frames.
- • To obtain motion labels for trajectories, we propose a method that differs from traditional affinity matrix-based approaches and introduce the Motion-Semantic Decoupled Embedding, which enables a more effective integration of motion and semantic information, enhancing track-level segmentation by balancing these cues.
- • Extensive results on multiple benchmarks demonstrate the effectiveness of our method, particularly in finegrained moving object segmentation.

[Figure 13]

[Figure 14]

[Figure 15]

RCF

-All

[Figure 16]

[Figure 17]

[Figure 18]

ABR

[Figure 19]

[Figure 20]

[Figure 21]

Ours

[Figure 22]

[Figure 23]

[Figure 24]

| |
|---|

| |
|---|

| |
|---|

RGB

t=60 t=85 t=125 Time

Figure 2. The effectiveness of long-range tracks. Over longer periods of time, if a moving object experiences factors such as occlusion or changes in lighting, it can negatively affect the tracking performance of optical-flow-based methods for that object.

##### 2. Related Work

Flow-based Moving Object Segmentation. Traditionally, optical flow based methods [6, 46, 53, 59] segment moving objects by grouping motion cues to create a moving object mask. These methods typically employ iterative optimization or statistical inference techniques to estimate motion models and identify motion regions simultaneously. Recently, numerous deep learning-based approaches [8, 9, 18, 37, 40, 49, 62] have used CNN encoders or transformer to extract motion cues from optical flow, followed by decoders to produce the final segmentation. The main distinctions among these methods lie in model architecture; for instance, methods that encode semantic information often utilize multiple CNN encoders to process different data modalities separately. In general, optical-flowbased methods struggle to distinguish independent object motion from apparent motion caused by depth differences. Furthermore, strong brightness changes also adversely affect these methods. Additionally, optical-flow-based methods are limited to short temporal sequences; they perform poorly if objects move slowly or are occluded.

Trajectory-based Moving Object Segmentation. Trajectory-based methods can be typically classified into two categories: two-frame and multi-frame methods. Two-frame methods [3, 14, 25] generally estimate motion parameters by solving an iterative energy minimization problem, which are recently powered with various convolutional neural network (CNN) models [54, 75]. Multi-frame methods, in contrast, often utilize spectral clustering based on affinity matrices. These matrices are derived through techniques such as geometric model fitting [2, 23, 32, 64], subspace fitting [17, 50, 55, 57], or pairwise motion affinities that integrate motion and appearance information [7, 24, 30, 42]. Recent work has

###### (a) Tracks Label Prediction

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Motion-Semantic

Spatial-Temporal Trajectory Attention

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Decoupled Embedding

[Figure 34]

DINO

Position Embedding

Motion Encoder Dynamic Tracks Selection

Tracks Decoder

Featured Tracks

2D Tracks and Depths

###### (b) Iterative Prompting

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

SAM2

###### SAM2

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

…

time

Iterative Grouping

Grouped Dynamic Tracks

Predicted Fine-grained Masks

- Figure 3. Overview of Our Pipeline. We take 2D tracks and depth maps generated by off-the-shelf models [15, 67] as input, which are then processed by a motion encoder to capture motion patterns, producing featured tracks. Next, we use tracks decoder that integrates DINO feature [45] to decode the featured tracks by decoupling motion and semantic information and ultimately obtain the dynamic trajectories(a). Finally, using SAM2 [51], we group dynamic tracks belonging to the same object and generate fine-grained moving object masks(b).

focused on the search for more effective motion models. For instance, [1] uses the trifocal tensor to analyze point trajectories, arguing that it provides more reliable matches over three images than fundamental matrices can over two. However, the trifocal tensor also poses challenges: it is difficult to optimize and prone to failure when the three camera positions are nearly collinear[44]. Other studies [27, 65] have proposed geometric model fusion techniques to combine different models. Some recent work has explored integrating multiple motion cues [21, 41]. For example, [24] investigates combining point trajectories and optical flow, using well-crafted geometric motion models to fuse the two affinity matrices through co-regularized multi-view spectral clustering. However, these approaches still face inherent issues due to their reliance on affinity matrices. They tend to capture only local similarities, leading to poor global consistency, resulting in inconsistent segmentation. Furthermore, affinity matrices is difficult to capture dynamic changes in motion features like speed and direction over time. In contrast, we address the challenge of capturing motion similarities across diverse motion types.

Unsupervised Video Object Segmentation. Unsupervised Video Object Segmentation (VOS) aims to automatically identify and track salient objects in raw video footage, while semi-supervised VOS relies on first-frame ground truth annotations to segment objects in subsequent frames [47, 48]. In this work, we focus on Unsupervised VOS, referred to here simply as ”VOS”. Recently, many approaches [69, 72] have combined motion and appearance information. For instance, MATNet [74] introduces a motionattentive transition model for unsupervised VOS, leverag-

ing motion cues to guide segmentation with a primary focus on appearance. RTNet [52] presents a method based on reciprocal transformations, using the consistency of object appearance and motion between consecutive frames to achieve segmentation. FSNet [26] employs a full-duplex strategy with a dual-path network to jointly model both appearance and motion. Overall, VOS generally targets salient objects in videos, regardless of whether the object is moving. Although many VOS methods incorporate motion information, it is often not their primary focus.

##### 3. Method

Our objective is, given a video, to identify moving objects and generate pixel-level dynamic masks. Fig. 3 provides an overview of our pipeline. The central insight is that longrange tracks not only capture motion patterns that facilitate video understanding but also offer long-range prompts essential for promptable visual segmentation. Thus, we use long-range point tracks as motion cues, serving as the primary input in Sec. 3.1, where we apply spatial-temporal attention to capture context-aware feature. In Sec. 3.2, we further incorporate and decouple the use of semantic information with motion cues to decode features, helping the model predict the final motion labels. After identifying dynamic tracks, we leverage these long-range tracks to prompt SAM2 [51] iteratively, as described in Sec. 3.3.

###### 3.1. Motion Pattern Encoding

Point trajectories carry valuable information for understanding motion, and related MOS methods can be typi-

cally classified into two categories: two-frame and multiframe methods. However, as discussed in Sec. 2, two-frame methods [3, 14, 25] often suffer from significant temporal inconsistencies and exhibit degraded performance when input flows are noisy. Multi-frame methods, in contrast, often utilize spectral clustering based on affinity matrices. Nevertheless, they remain highly sensitive to noise and struggle to handle global, dynamic, and complex motion patterns effectively.

To address these limitations, and inspired by ParticleSFM [73], we propose a method that leverages longrange point tracks [15], processed through a specialized trajectory processing model, to predict per-trajectory motion labels. As illustrated in Fig. 3, our proposed network adopts an encoder-decoder architecture. The encoder directly processes long-range trajectory data and applies a Spatio-Temporal Trajectory Attention mechanism across trajectories. This mechanism integrates both spatial and temporal cues, capturing both local and global information across time and space, in order to embed the motion pattern of each trajectory.

Given that the accuracy and quality of long-range trajectories significantly impact model performance, we utilize BootsTAP [15] to generate the tracks, which provides a confidence score for each track at each time step, enabling us to mask out low-confidence points. Furthermore, due to the movement of dynamic objects and camera motion, the visibility of long-range tracks can vary over time, as they may be occluded or move out of the frame. This variability in visibility and confidence makes each trajectory data highly irregular, motivating our use of a transformer model, inspired by sequence modeling approaches in natural language processing [56, 73], to handle the data effectively.

Our input data comprises long-range trajectories, with each trajectory consisting of normalized pixel coordinates (ui,vi), visibility ρi and confidence scores ci, where i ∈ (0,time). Masks Mi is applied to indicate points where the pixel coordinates are either invisible or low-confidence. Additionally, we integrate monocular depth maps di estimated by Depth-Anything [67], which, despite some noise, provide valuable insights into the underlying 3D scene structure, enhancing understanding of spatial layout and occlusions. To further enrich the input data and strengthen temporal motion cues, we compute frame-to-frame differences in both trajectory coordinates (∆ui,∆vi) and depth ∆di for adjacent frames.

Since adjacent sampling points in coordinates can lead to oversmoothing of spatially close features, we draw inspiration from NeRF [39] to address this issue. Specifically, we apply frequency transformations for positional encoding to better capture fine-grained spatial details.

The final augmented trajectories pass through two MLPs to generate intermediate features, which are then fed into

the transformer encoder. Given the long-range nature of the input data, we propose a Spatio-Temporal Trajectory Attention for our encoder E, interleaves attention layers that operate alternately across track and temporal dimensions [4, 29]. This design allows the model to capture both the temporal dynamics within each trajectory and the spatial relationships across different trajectories. Finally, to obtain a feature representation for each entire trajectory rather than individual points, we perform max-pooling along the temporal dimension, following [73]. This process yields a single feature vector for each trajectory, naturally forming a high-dimensional featured track that implicitly captures the unique motion pattern of each trajectory.

###### 3.2. Per-trajectory Motion Prediction

Though we encoded motion pattern in Sec. 3.1, it is still challenging to distinguish moving objects based solely on motion cues, because learning to differentiate between object motion and camera motion from highly abstracted trajectories is difficult for the model. Providing the model with texture, appearance, and semantic information can simplify this task by helping it understand which objects are likely to move or be moved. Some approaches directly apply semantic segmentation models [5, 20, 68, 71] where potentially moving pixels are identified based on semantic labels. While these methods can be effective in specific scenarios, they are intrinsically limited for general moving object segmentation, as they depend entirely on predefined semantic classes. Recently, many MOS [61, 70] and VOS [11, 33, 35] methods combine appearance information and motion cues, but they do so in two separate stages, often using RGB images to refine masks. However, relying on raw RGB data may fail to capture high-level information, and applying the two modalities in separate stages limits the effective integration of their complementary information.

To address these limitations, we incorporate DINO features predicted by DINO v2 [45], a self-supervised model, which helps generalize the inclusion of appearance information. However, we observed that simply introducing DINO features as input makes the model overly reliant on semantics as shown in Fig. 8 and discussed in Sec. 4.5, reducing its ability to differentiate between moving and static objects within the same semantic category. To overcome this issue, we propose a Motion-Semantic Decoupled Embedding, enabling the transformer decoder D to prioritize motion information while still considering semantic cues.

We obtain the final embedded featured tracks P through the process described in Sec. 3.1:

P = E((γ(u),γ(v),γ(∆u),γ(∆v),d,∆d,ρ,c),M). (1)

We then design a transformer-based decoder, where the encoder layer performs attention only on the embedded featured tracks, which contain motion information exclusively.

After computing the attention-weighted feature, we concatenate the DINO feature and pass this concatenated feature through a feed-forward layer. In the decoder layer, selfattention is still applied only to the motion features; however, multi-head attention is used to attend to a memory that includes semantic information. Finally, we apply a sigmoid activation function to produce the final output, yielding the predicted label for each trajectory.

We then compute the loss between these predicted labels and per-track ground truth labels using a weighted binary cross-entropy loss [73]. We assign ground truth labels to each trajectory by checking if the sampled point coordinates lie within the ground truth dynamic masks. If a point falls inside the mask, it is labeled as dynamic.

###### 3.3. SAM2 Iterative Prompting

As depicted in Fig. 3, after obtaining the predicted label of each trajectory and filter dynamic trajectories, we use these trajectories as point prompts for SAM2[51] with an iterative, two-stage prompting strategy. The first stage focuses on grouping trajectories belonging to the same object and storing the trajectories of each distinct object in memory. In the second stage, this memory is used as a prompt for SAM2 [51] to generate dynamic masks.

The motivation behind this approach is twofold. First, it is necessary because SAM2 requires object IDs as input. However, if we assign the same object ID to all dynamic objects (e.g., assigning 1 to represent all dynamic objects), SAM2 would struggle to simultaneously segment multiple objects that share the same ID. Second, this method offers the benefit of achieving finer-grained segmentation.

In the first stage, we select the time frame with the maximum number of visible points and locate the densest point among all visible points in that frame. This point serves as the initial prompt for SAM2 [51], which then generates an initial mask for that frame. After generating this mask, we apply dilation to expand its boundaries, excluding all points within the expanded mask area to remove edge points and assume that these points belong to the same object. We then proceed to the next frame with the highest number of visible points and repeat this process until the remaining visible points across all frames are too few to process. The trajectories identified as belonging to the same object are stored in memory, with unique object IDs assigned to each. We only save the points within the undilated mask for each object.

In the second stage, we use this memory to refine prompt selection by locating the densest point within the stored trajectories and the two points furthest from this point. Leveraging the long-range nature of trajectories, we prompt SAM2 at regular intervals to prevent it from losing track of the object over extended distances. Since SAM2 may generate partial object masks (e.g., parts of a person’s clothing), we perform post-processing on all masks to merge

those that overlap internally or appear within the same mask boundaries. This results in a complete mask for each distinct object.

##### 4. Experiments

###### 4.1. Implementation Details

Training Dataset. We train our model using three datasets: Kubric[19], Dynamic Replica[28], and HOI4D [36], sampling them at a ratio of 35%, 35%, and 30% respectively. Kubric [19] is a synthetic dataset composed of sequences of 24 frames showing 3D rigid objects falling under gravity and bouncing. We generate dynamic masks for each sequence based on the motion labels of individual objects. Dynamic Replica [28] is another synthetic dataset, created for 3D reconstruction, that includes long-term tracking annotations and object masks, featuring articulated models of humans and animals. We calculate dynamic masks by analyzing the 3D tracks to determine whether each object is in motion, providing accurate motion segmentation for this dataset. HOI4D [36] is a real-world, egocentric dataset that contains common objects involved in human-object interactions. This dataset provides official motion segmentation masks, making it ideal for real-world training of our model. Data Sampling. During training, we randomly sample a variable number of tracking points, enhancing the model’s robustness to different track counts. For the Dynamic Replica dataset [28], which contains 300 frames, we speed up training by sampling 1/4 of the frames at regular intervals randomly. This approach preserves the large camera motion characteristics of the dataset. We find that including the Dynamic Replica dataset is essential for helping the model understand camera motion effectively.

###### 4.2. Benchmark and metrics

We evaluate our model using several established datasets for moving object video segmentation. DAVIS17-Moving[13] is a subset of the DAVIS2017 dataset[48], designed specifically for moving object detection and segmentation. In DAVIS17-Moving, all moving instances within each video sequence are labeled, while static objects are excluded. Following the same criteria, we created DAVIS16-Moving as a subset of the DAVIS2016 dataset [47]. Additionally, we report performance on other popular video object segmentation benchmarks, including DAVIS2016[47], SegTrackv2[34], and FBMS-59 [43].

For evaluation, we benchmark our moving object video segmentation performance using region similarity (J) and contour similarity (F) metrics, as outlined in [35, 38, 61].

###### 4.3. Moving Object Segmentation

We selected methods that specifically target moving object segmentation as baselines [35, 38, 60, 61, 70]. For

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

CISOCLR-TTAABRGTOursRCF-AllEM

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

- Figure 4. Qualitative comparison on DAVIS17-moving benchmarks. For each sequence we show moving object mask results. Our method successfully handles water reflections (left), camouflage appearances (middle), and drastic camera motion (right).

RGB GT Ours CIS OCLR-TTA ABR RCF-All EM

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

- Figure 5. Qualitative comparison on FBMS-59 benchmarks. The masks produced by us are geometrically more complete and detailed.

OCLR [60], we report results for two versions: OCLR-flow, which uses only flow input, and a second version OCLRTTA that incorporates test-time adaptation on top of OCLRflow. For RCF [35], the first stage, RCF-stage 1, focuses on motion information, while the second stage, RCF-All, further optimizes the results from the first stage. We report results for both stages. For all baselines, we apply a fully connected conditional random field (CRF) [31] to refine the masks and achieve the best possible results.

Notably, for multi-object scenarios, we follow the common practice [16, 60, 61, 70] of grouping all foreground objects together for evaluation purposes, which we refer to as MOS. Although our approach is capable of generating highly accurate, fine-grained per-object masks, as detailed in Sec. 4.4, we term this second evaluation method

as fine-grained MOS. Table 1 compares the performance of our model with several baseline methods on the MOS task. Our method achieves state-of-the-art F-scores across all datasets, and our region similarity (J) scores are either the best or second-best across multiple datasets, further validating the effectiveness of our approach. Fig. 4 shows our visual results on the DAVIS16-Moving dataset, where our method accurately identifies object boundaries without incorrectly labeling moving backgrounds. Moreover, our masks exhibit strong geometric structure, particularly in challenging scenarios with significant camera motion. Fig. 5 and Fig. 6 present qualitative results on the FBMS59 and SegTrack v2 benchmarks, respectively. Our method performs exceptionally well in maintaining mask geometry, and even in cases where the RGB images are blurred or of

RGB GT Ours CIS OCLR-TTA ABR RCF-All EM

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

Figure 6. Qualitative comparison on SegTrack v2 benchmarks. Our method succeeds even under motion blur conditions. Table 1. Quantitative comparison on MOS task which grouping all foreground objects together for evaluation.

Model Settings DAVIS2016-Moving SegTrackv2 FBMS-59 DAVIS2016 Motion Appearance J &F ↑ J ↑ F ↑ J ↑ J ↑ F ↑ J &F ↑ J ↑ F ↑

Methods

CIS [70] Optical Flow RGB 66.2 67.6 64.8 62.0 63.6 - 68.6 70.3 66.8 EM [38] Optical Flow ✗ 75.2 76.2 74.3 55.5 57.9 56.0 70.0 69.3 70.7 RCF-Stage1 [35] Optical Flow ✗ 77.3 78.6 76.0 76.7 69.9 - 78.5 80.2 76.9 RCF-All [35] Optical Flow DINO 79.6 81.0 78.3 79.6 72.4 - 80.7 82.1 79.2 OCLR-flow [60] Optical Flow ✗ 70.0 70.0 70.0 67.6 65.5 64.9 71.2 72.0 70.4 OCLR-TTA [60] Optical Flow RGB 78.5 80.2 76.9 72.3 69.9 68.3 78.8 80.8 76.8 ABR [61] Optical Flow DINO 72.0 70.2 73.7 76.6 81.9 79.6 72.5 71.8 73.2

###### Ours Trajectory DINO 89.5 89.2 89.7 76.3 78.3 82.8 90.9 90.6 91.0

Table 2. Quantitative comparison on DAVIS17-Moving dataset for MOS and Fine-grained MOS tasks.

[Figure 122]

[Figure 123]

[Figure 124]

OCLR-TTARGBABRGTOurs

MOS Fine-grained MOS

Methods

[Figure 125]

[Figure 126]

[Figure 127]

###### J ↑ F ↑ J &F ↑ J ↑ F ↑

OCLR-flow [60] 69.9 70.0 44.4 42.1 46.8 OCLR-TTA [60] 76.0 75.3 49.1 48.4 49.9 ABR [61] 74.6 75.2 51.1 50.9 51.2

[Figure 128]

[Figure 129]

[Figure 130]

###### Ours 90.0 89.0 80.5 77.4 83.6

[Figure 131]

[Figure 132]

[Figure 133]

ability to perform this task. Table 2 shows that our method significantly outperforms the baselines, demonstrating its superior capability in producing accurate per-object masks. Additionally, Fig. 7 illustrates that, first, our method accurately identifies each object, effectively distinguishing different objects with similar motion patterns. Second, it ensures the completeness of each object mask, handling challenging cases such as articulated human structures and occluded objects while maintaining mask integrity.

[Figure 134]

[Figure 135]

[Figure 136]

- Figure 7. Qualitative comparison on Fine-grained MOS task which will produce per-object level masks.

low quality, our reliance on long-range trajectories enables accurate identification of moving objects.

###### 4.5. Ablation Study

###### 4.4. Fine-grained Moving Object Segmentation

We investigate the effectiveness of our method and its various components on the DAVIS17-Moving and DAVIS16Moving datasets. The former is used for fine-grained MOS, while the latter focuses on MOS. All models are trained for a full number of epochs.

Building on the initial MOS task, this task not only identifies moving objects but also classifies them within their motion context to generate fine-grained, per-object masks. We evaluate our approach for multi-moving object segmentation specifically on the DAVIS2017-Moving dataset. For a fair comparison, we only include baselines that claim the

We conducted several experiments to assess the importance of each component. The w/o DINO configuration ex-

[Figure 137]

[Figure 138]

[Figure 139]

Complete Model w/o DINO w/o Space-time Attention

[Figure 140]

[Figure 141]

[Figure 142]

Complete Model w/o Decouple Embedding w/o Motion-only Encoding

- Figure 8. Visual comparison for the ablation study on two critical and challenging cases. The top sequence shows scenarios involves drastic camera motion and complex motion patterns, while the bottom sequence with both static and dynamic objects of the same category. The experimental setup is detailed in Sec. 4.5.

Table 3. Quantitative comparison for the ablation study on the DAVIS17-Moving and DAVIS16-Moving benchmarks, which evaluate fine-grained MOS and MOS tasks, respectively. The experimental setup is detailed in Sec. 4.5.

DAVIS17-Moving DAVIS16-Moving J &F ↑ J ↑ F ↑ J &F ↑ J ↑ F ↑

Methods

w/o Depth 69.2 65.6 72.8 82.5 78.6 86.4 w/o Tracks 19.6 14.5 24.7 20.9 9.8 31.9 w/o DINO 65.0 62.1 67.9 75.5 71.4 79.5 w/o MOE 72.0 68.7 75.4 81.8 81.0 82.7 w/o MSDE 63.0 59.3 66.7 78.2 77.3 79.1 w/o PE 66.4 64.7 68.2 82.0 81.5 82.5 w/o ST-ATT 65.5 61.9 69.1 78.3 74.3 82.4

Ours-full 80.5 77.4 83.6 89.1 89.0 89.2

cludes DINO features entirely during training, while w/o MOE (Motion-only Encoding) concatenates DINO features with motion cues before the motion encoder, allowing both the encoder and decoder layers to incorporate DINO information throughout. w/o MSDE (Motion-Semantic Decoupled Embedding) excludes DINO features from the motion encoder but concatenates them with the embedded featured tracks from the encoder output, introducing semantic information through self-attention in the tracks decoder. We also test configurations w/o depth and w/o tracks, removing specific inputs to observe their impact on performance. Additionally, w/o PE (Positional Embedding) omits NeRF-like positional embedding in the motion encoder, and w/o STATT (Spatial-temporal Attention) replaces spatial-temporal attention with conventional attention.

Table 3 presents the quantitative results. We find that excluding depth as input or positional encoding impacts performance less than other components, but it still falls significantly short of the best results. When tracks are removed and only DINO features and depth maps are used, performance drops drastically, indicating that the model struggles to learn effectively without trajectory-based information. We further analyze the key components in two chal-

lenging scenarios presented below.

Drastic Camera Motion. We observed that in highly challenging scenes, such as those with drastic camera movement or rapid object motion, relying solely on motion information is insufficient. As shown in the upper part of Fig. 8, the colored points represent dynamic points predicted by the model, while the hollow points indicate invisible points at that moment. In this example, without DINO feature information, the model incorrectly classifies the stationary road surface as dynamic, despite the fact that the road lacks the ability to move. This information can be effectively supplemented by incorporating DINO features. Additionally, we found that adding spatial-temporal attention within the motion encoder is particularly beneficial in these difficult scenarios, as it provides the model with richer motion information to capture the long-range motion patterns of tracks, as illustrated in Fig. 8.

Distinguishing Moving and Static Objects of the Same Category. Results show that excluding DINO features entirely results in a performance drop, and the manner in which these features are integrated significantly affects the model’s output. Simply incorporating DINO as an input during the motion encoding stage causes the model to rely heavily on semantic information, often leading it to assume that objects of the same type share the same motion state. In contrast, our Motion-Semantic Decoupled Embedding architecture effectively reduces this over-reliance on semantics, allowing the model to differentiate between moving and static objects within the same category, as illustrated in the lower part of Fig. 8.

##### 5. Conclusion

In this work, we present a novel approach that leverages long-range tracks which departs from traditional affinity matrix-based methods. Trained on extensive datasets, our model accurately identifies dynamic tracks, which, when combined with SAM2, produce precise moving object masks. Our carefully designed model architecture is tailored to handle long-range motion information while effectively balancing motion and appearance cues. Experiments show that our method achieves state-of-the-art results across multiple benchmarks, with particularly strong performance in per-object-level segmentation.

##### 6. Limitation

During testing, we identified several limitations of our approach, which we believe can offer valuable insights. We discuss these limitations below and leave addressing these fundamental directions for future work.

Dependency on Tracking Estimators. We utilizes off-theshelf long-range tracking estimators, whose accuracy can

greatly influence overall performance, as shown in Tab. 3.

Fast-Moving Objects with Brief Appearances. In longrange videos, objects moving rapidly and appearing briefly pose significant challenges. Specifically, if an object moves quickly and is captured in only a few frames, resulting in very short object tracks, our method is likely to fail.

Dominant Motion vs. Minor Motion. In scenes with multiple moving objects, the method may struggle to capture objects with subtle movements, particularly when another object exhibits more pronounced motion, causing the less dynamic object to be overlooked.

Partial Segmentation. The method occasionally produces incomplete segmentation masks. For instance, when a person moves, if the dynamic track prompts provided to SAM2 are located on the person’s clothing, segmentation might capture only the clothing rather than the entire figure, leading to partial or fragmented results.

Homogeneous Motion State. Our segmentation framework also faces difficulties when motion differentiation within the scene is limited. Specifically, when most objects share similar motion states—either predominantly moving or static—our approach cannot effectively distinguish individual objects, leading to segmentation failures.

##### References

- [1] Federica Arrigoni, Luca Magri, and Tomas Pajdla. On the Usage of the Trifocal Tensor in Motion Segmentation, page 514–530. 2020. 3
- [2] Federica Arrigoni, Luca Magri, and Tomas Pajdla. On the Usage of the Trifocal Tensor in Motion Segmentation, page 514–530. 2020. 2
- [3] Daniel Barath and Jiri Matas. Progressive-x: Efficient, anytime, multi-model fitting algorithm. arXiv: Computer Vision and Pattern Recognition,arXiv: Computer Vision and Pattern Recognition, 2019. 2, 4
- [4] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? Cornell University - arXiv,Cornell University - arXiv, 2021. 4
- [5] Berta Bescos, Jose M. Facil, Javier Civera, and Jose Neira. Dynaslam: Tracking, mapping and inpainting in dynamic scenes. IEEE Robotics and Automation Letters, page 4076–4083, 2018. 4
- [6] Pia Bideau and Erik Learned-Miller. It’s moving! a probabilistic model for causal motion segmentation in moving camera videos. Cornell University - arXiv,Cornell University - arXiv, 2016. 2
- [7] Thomas Brox and Jitendra Malik. Object segmentation by long term analysis of point trajectories. In European conference on computer vision, pages 282–295. Springer, 2010. 2
- [8] M. B¨osch. Deep learning for robust motion segmentation with non-static cameras. arXiv: Computer Vision and Pattern Recognition,arXiv: Computer Vision and Pattern Recognition, 2021. 2

- [9] Zhe Cao, Abhishek Kar, Christian Hane, and Jitendra Malik. Learning independent object motion from unlabelled stereoscopic videos. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2
- [10] Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, and Li Zhang. Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering. arXiv:2311.18561,

2023. 1

- [11] Suhwan Cho, Minhyeok Lee, Seunghoon Lee, Dogyoon Lee, Heeseung Choi, Ig-Jae Kim, and Sangyoun Lee. Dual prototype attention for unsupervised video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19238–19247, 2024. 4
- [12] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers, 2023. 1
- [13] Achal Dave, Pavel Tokmakov, and Deva Ramanan. Towards segmenting anything that moves. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0–0, 2019. 5
- [14] Andrew Delong, Anton Osokin, Hossam N. Isack, and Yuri Boykov. Fast approximate energy minimization with label costs. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2010. 2, 4
- [15] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Jo˜ao Carreira, and Andrew Zisserman. BootsTAP: Bootstrapped training for tracking any point. arXiv,

2024. 3, 4, 1

- [16] Suyog Dutt Jain, Bo Xiong, and Kristen Grauman. Fusionseg: Learning to combine motion and appearance for fully automatic segmentation of generic objects in videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3664–3673, 2017. 6
- [17] E. Elhamifar and R. Vidal. Sparse subspace clustering: Algorithm, theory, and applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 2765–2781,

2013. 2

- [18] Muhammad Faisal, Ijaz Akhter, Mohsen Ali, and Richard Hartley. Epo-net: Exploiting geometric constraints on dense trajectories for motion saliency. arXiv: Computer Vision and Pattern Recognition,arXiv: Computer Vision and Pattern Recognition, 2019. 2
- [19] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam Laradji, HsuehTi (Derek) Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, and Andrea Tagliasacchi. Kubric: a scalable dataset generator. 2022. 2, 5
- [20] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask r-cnn. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 386–397, 2020. 4

- [21] Christian Homeyer and RobertBosch Gmbh. On moving object segmentation from monocular video with transformers. 3
- [22] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. S3gaussian: Self-supervised street gaussians for autonomous driving. arXiv preprint arXiv:2405.20323, 2024. 1
- [23] Yuxiang Huang and John Zelek. Motion segmentation from a moving monocular camera. 2023. 2
- [24] Yuxiang Huang, Yuhao Chen, and John Zelek. Zero-shot monocular motion segmentation in the wild by combining deep learning with geometric motion model fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 2733– 2743, 2024. 2, 3
- [25] Hossam Isack and Yuri Boykov. Energy-based geometric multi-model fitting. International Journal of Computer Vision, page 123–147, 2012. 2, 4
- [26] Ge-Peng Ji, Keren Fu, Zhe Wu, Deng-Ping Fan, Jianbing Shen, and Ling Shao. Full-duplex strategy for video object segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4922–4933, 2021. 3
- [27] Yangbangyan Jiang, Qianqian Xu, Ke Ma, Zhiyong Yang, Xiaochun Cao, and Qingming Huang. What to select: Pursuing consistent motion segmentation from multiple geometric models. Proceedings of the AAAI Conference on Artificial Intelligence, page 1708–1716, 2022. 3
- [28] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. CVPR, 2023. 2, 5
- [29] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In Proc. ECCV, 2024. 4
- [30] Laurynas Karazija, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Learning segmentation from point trajectories,

2024. 2

- [31] Philipp Kr¨ahenb¨uhl and Vladlen Koltun. Efficient inference in fully connected crfs with gaussian edge potentials. Neural Information Processing Systems,Neural Information Processing Systems, 2011. 6
- [32] Taotao Lai, Hanzi Wang, Yan Yan, Tat-Jun Chin, and WanLei Zhao. Motion segmentation via a sparsity constraint. IEEE Transactions on Intelligent Transportation Systems, page 973–983, 2017. 2
- [33] Minhyeok Lee, Suhwan Cho, Dogyoon Lee, Chaewon Park, Jungho Lee, and Sangyoun Lee. Guided slot attention for unsupervised video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3807–3816, 2024. 4
- [34] Fuxin Li, Taeyoung Kim, Ahmad Humayun, David Tsai, and James M. Rehg. Video segmentation by tracking many figure-ground segments. In 2013 IEEE International Conference on Computer Vision, pages 2192–2199, 2013. 2, 5, 1

- [35] Long Lian, Zhirong Wu, and Stella X. Yu. Bootstrapping objectness from videos by relaxed common fate and visual grouping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14582–14591, 2023. 4, 5, 6, 7, 2
- [36] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. Hoi4d: A 4d egocentric dataset for category-level humanobject interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21013–21022, 2022. 2, 5
- [37] Etienne Meunier and Patrick Bouthemy. Unsupervised motion segmentation in one go: Smooth long-term model over a video, 2024. 2
- [38] Etienne Meunier, Ana¨ıs Badoual, and Patrick Bouthemy. Em-driven unsupervised learning for efficient motion segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4462–4473, 2023. 5, 7, 2
- [39] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 4
- [40] Eslam Mohamed, Mahmoud Ewaisha, Mennatullah Siam, Hazem Rashed, Senthil Yogamani, Waleed Hamdy, Mohamed El-Dakdouky, and Ahmad El-Sallab. Monocular instance motion segmentation for autonomous driving: Kitti instancemotseg dataset and multi-task baseline. In 2021 IEEE Intelligent Vehicles Symposium (IV), 2021. 2
- [41] Michal Neoral and Jan Sochman.ˇ Monocular arbitrary moving object discovery and segmentation. 3
- [42] Peter Ochs, Jitendra Malik, and Thomas Brox. Segmentation of moving objects by long term video analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 1187–1200, 2014. 2
- [43] Peter Ochs, Jitendra Malik, and Thomas Brox. Segmentation of moving objects by long term video analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(6): 1187–1200, 2014. 2, 5, 1
- [44] H. Opower. Multiple view geometry in computer vision. Optics and Lasers in Engineering, page 85–86, 2002. 3
- [45] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 1, 2, 3, 4
- [46] Anestis Papazoglou and Vittorio Ferrari. Fast object segmentation in unconstrained video. In 2013 IEEE International Conference on Computer Vision, 2013. 2
- [47] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Computer Vision and Pattern Recognition, 2016. 2, 3, 5, 1
- [48] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alexander Sorkine-Hornung, and Luc Van Gool.

- The 2017 davis challenge on video object segmentation. arXiv:1704.00675, 2017. 2, 3, 5, 1
- [49] Mohamed Ramzy, Hazem Rashed, AhmadEl Sallab, and Senthil Yogamani. Rst-modnet: Real-time spatio-temporal moving object detection for autonomous driving. Cornell University - arXiv,Cornell University - arXiv, 2019. 2
- [50] S Rao, R Tron, R Vidal, and Yi Ma. Motion segmentation in the presence of outlying, incomplete, or corrupted trajectories. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(10):1832–1845, 2010. 2
- [51] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 1, 3, 5
- [52] Sucheng Ren, Wenxi Liu, Yongtuo Liu, Haoxin Chen, Guoqiang Han, and Shengfeng He. Reciprocal transformations for unsupervised video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15455–15464, 2021. 3
- [53] Hicham Sekkati and Amar Mitiche. A variational method for the recovery of dense 3d structure from motion. Robotics and Autonomous Systems, 55(7):597–607, 2007. 2
- [54] Pavel Tokmakov, Cordelia Schmid, and Karteek Alahari. Learning to segment moving objects. International Journal of Computer Vision,International Journal of Computer Vision, 2017. 2
- [55] Roberto Tron and Rene Vidal. A benchmark for the comparison of 3-d motion segmentation algorithms. In 2007 IEEE Conference on Computer Vision and Pattern Recognition, page 1–8, 2007. 2
- [56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, AidanN. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Neural Information Processing Systems,Neural Information Processing Systems,

2017. 4

- [57] Ren´e Vidal. Subspace clustering. IEEE Signal Processing Magazine,IEEE Signal Processing Magazine, 2011. 2
- [58] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video, 2024. 1
- [59] Andreas Wedel, Annemarie Meißner, Clemens Rabe, Uwe Franke, and Daniel Cremers. Detection and Segmentation of Independently Moving Objects from Dense Scene Flow, page 14–27. 2009. 2
- [60] Junyu Xie, Weidi Xie, and Andrew Zisserman. Segmenting moving objects via an object-centric layered representation. In NeurIPS, 2022. 5, 6, 7, 2
- [61] Junyu Xie, Weidi Xie, and Andrew Zisserman. Appearancebased refinement for object-centric motion segmentation. In ECCV, 2024. 2, 4, 5, 6, 7
- [62] Junyu Xie, Charig Yang, Weidi Xie, and Andrew Zisserman. Moving object segmentation: All you need is sam (and flow),

2024. 2

- [63] Junyu Xie, Charig Yang, Weidi Xie, and Andrew Zisserman. Moving object segmentation: All you need is sam (and flow). In ACCV, 2024. 2
- [64] Xun Xu, Loong-Fah Cheong, and Zhuwen Li. Motion segmentation by exploiting complementary geometric models. Cornell University - arXiv,Cornell University - arXiv, 2018. 2
- [65] Xun Xu, Loong Fah Cheong, and Zhuwen Li. Motion segmentation by exploiting complementary geometric models. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2859–2867, 2018. 3
- [66] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, and Yue Wang. Emernerf: Emergent spatialtemporal scene decomposition via self-supervision. arXiv preprint arXiv:2311.02077, 2023. 1
- [67] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 3, 4
- [68] Shichao Yang and Sebastian Scherer. Cubeslam: Monocular 3d object slam. IEEE Transactions on Robotics, page 925–938, 2019. 4
- [69] Shu Yang, Lu Zhang, Jinqing Qi, Huchuan Lu, Shuo Wang, and Xiaoxing Zhang. Learning motion-appearance coattention for zero-shot video object segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1564–1573, 2021. 3
- [70] Yanchao Yang, Antonio Loquercio, Davide Scaramuzza, and Stefano Soatto. Unsupervised moving object detection via contextual information separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 879–888, 2019. 4, 5, 6, 7, 2
- [71] Chao Yu, Zuxin Liu, Xin-Jun Liu, Fugui Xie, Yi Yang, Qi Wei, and Qiao Fei. Ds-slam: A semantic visual slam towards dynamic environments. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

2018. 4

- [72] Kaihua Zhang, Zicheng Zhao, Dong Liu, Qingshan Liu, and Bo Liu. Deep transport network for unsupervised video object segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8781–8790,

2021. 3

- [73] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. Particlesfm: Exploiting dense point trajectories for localizing moving cameras in the wild. In European conference on computer vision (ECCV), 2022. 4, 5
- [74] Tianfei Zhou, Shunzhou Wang, Yi Zhou, Yazhou Yao, Jianwu Li, and Ling Shao. Motion-attentive transition for zero-shot video object segmentation. In Proceedings of the AAAI conference on artificial intelligence, pages 13066– 13073, 2020. 3
- [75] Tianfei Zhou, Shunzhou Wang, Yi Zhou, Yazhou Yao, Jianwu Li, and Ling Shao. Motion-attentive transition for zero-shot video object segmentation. In Proceedings of the AAAI conference on artificial intelligence, pages 13066– 13073, 2020. 2

## Segment Any Motion in Videos Supplementary Material

##### 7. Pseudo-code for SAM2 Iterative Prompting

We present the pseudo-code for the first stage of SAM2 Iterative Prompting in Algorithm 1. The first stage focuses on grouping trajectories belonging to the same object and storing the trajectories of each distinct object in memory.

Algorithm 1 Process Invisible Trajectory with Memory

- 1: Initialize iteration to 0
- 2: Initialize memory dict as an empty dictionary

- 3: Set take all to False

- 4: if traj.shape[1] ≤ 5 then
- 5: Set take all to True

- 6: end if
- 7: while iteration < max iterations do

- 8: Set t to frame with maximum visible points
- 9: Extract visible points at frame t
- 10: Find densest point as nearest point

- 11: Reset predictor state and add new point
- 12: Reset predictor state
- 13: Set obj id to 1 and labels to [1]

- 14: Add new point using predictor to get mask
- 15: Dilate the mask and determine points within the mask: dilated mask

- 16: Determine points in prompt mask (visible + nondilated): prompt mask

- 17: if sufficient points in mask or take all is True then

- 18: Increment valid obj id

- 19: Store object information in memory dict

- 20: end if
- 21: Remove points included in the mask from traj, visiblemask, and confidences
- 22: Update traj, visiblemask, and confidences with remaining points
- 23: Increment iteration by 1
- 24: if traj.shape[1] < 6 then
- 25: Break the loop
- 26: end if
- 27: end while
- 28: return memory dict

##### 8. Additional Experiment Details

Training Details. We train the model for 5 epochs, with each epoch comprising approximately 8000 steps, using the Adam optimizer with a learning rate of 1e-4 and a weight decay of 1e-4.

Model Architecture. As shown in the Fig 9, for the trajectory motion pattern encoder, we employ 4 heads for

multi-head attention and a 64-dimensional feed-forward layer. For the tracks decoder, we use 8 heads for multi-head attention and a 512-dimensional feed-forward layer.

𝑁 ×

DINO

Featuredtracks

AttentionBlock

EncoderBlock

DecoderBlock

Featuredtracks

FeedForward

Add&Norm

Multi-Head Attention

Multi-Head Attention

Add&Norm

Embedding

Multi-Head

Attention

[Figure 143]

Figure 9. Architecture of tracks decoder.

Model efficiency and details. We parallelized the code during data processing. For a 50-frame video, processing takes 2 minutes, model inference 3 seconds, and object prompt generation requires 2 seconds per object only. For a dynamic object, 1-2 iterations are usually required. And the experimental settings are discussed in Sec 4.1 and Sec 7. Training time is about 60 hours, and the hardware used for training is an NVIDIA RTX A6000.

Point Trajectory. We utilize BootsTAP [15] to generate 2D tracks for query frames in video sequences. Specifically, query frames are selected at intervals defined by step, and 2D tracks are generated only for these frames. For training datasets, grid size specifies the sampling grid resolution, determining the spacing between sampled points, while step controls the temporal interval between query frames. During training, we randomly select one query frame and load all its associated tracks to accelerate the process. For the Kubric dataset with a resolution of 512×512, we set grid size to 8, generating 4096 points per frame, and step to 8, with the total number of tracks randomly sampled from [512,1024,2048,3000,4096]. For the HOI4D dataset with a resolution of 1920 × 1080, we set grid size to 15, generating 9216 points per frame, and step to 15, with total tracks number sampled from [1024,1536,2048,4096,5000,6000]. For the Stereo dataset with a resolution of 1280×720, we set grid size to 32, generating 920 points per frame, and step to 8, with track counts sampled from [256,512,768,920]. During inference, 2D tracks are also generated for each query frame. To ensure that dynamic objects appearing at different times are fully captured, tracks from all query frames are loaded, and 5000 tracks are randomly selected. For the FBMS-59 dataset [43], we set grid size to 7 and step to 30, because some datasets contain relatively long sequences, we select a larger step to accelerate the loading process. For SegTrack V2 [34], grid size is set to 5 and step to 8. For DAVIS-16 [47] and DAVIS-17 [48], grid size is set to 10 and step to 8.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

### OCLR-TTARGBABROursGT

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

- Figure 10. Our method demonstrates exceptional capability in generating fine-grained masks. Most previous approaches rely on the common fate assumption, where objects moving at the same speed are considered part of the same entity. Moreover, many methods lack the ability to produce fine-grained masks altogether. In contrast, our method can accurately distinguish and segment individual objects, even when they are closely positioned, moving simultaneously, or traveling at the same speed.

Failure case. We perform well on most sequences, but struggle on cases like the “penguin” in SegTrackv2 (Fig 11), where 90% of the content has similar motion. The lack of contrast and uniform motion patterns can cause the model to misinterpret object motion as camera motion, leading to a J metric of 0.014. Since SAM2 requires prompts, any failure in this process results in near-zero scores, whereas the baseline still achieves the 30-50 range even when it fails.

[Figure 164]

[Figure 165]

[Figure 166]

- Figure 11. From left to right: input, dynamic tracks and mask.

|Methods<br><br>|Supervision<br><br>|Davis16-m J ↑ F ↑|Davis16 J ↑ F ↑<br><br>|
|---|---|---|---|
|FlowSAM<br><br>|YES<br><br>|85.7 83.8|87.1 84.9|
|Ours|YES<br><br>|89.2 89.7|90.6 91.0|

Table 4. Comparison with FlowSAM on MOS task.

require human annotation—our method needs human annotation during training, but not during inference.

##### 10. Additional Visualizations

We present additional visualizations on the three main datasets that we benchmark our method on [35, 38, 60, 61, 70]. We visualize our methods on DAVIS2016 in Fig. 12, Fig. 13 on the task of moving object segmentation. And Fig. 10 shows the result of fine-grained moving object segmentation on DAVIS2017.

##### 9. Additional Experiments

Comparison with FlowSAM [63]. We further included a new baseline experiment (see Tab. 4) to demonstrate the superior performance of our model. It is worth noting that among all the baselines, only our method and FlowSAM

Additionally, we provide a video demonstration featuring featuring non-cherry-picked examples from DAVIS16Moving, showcasing both long-range trajectory label predictions and the final mask results.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

#### OCLR-TTAOursRCF-AllABPCISGTEM

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

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

- Figure 12. Our method effectively preserves the geometric integrity of articulated objects, such as human legs or camel limbs. At the same time, it can distinguish between dynamic backgrounds and foregrounds, focusing specifically on the object level. Additionally, it accurately identifies camouflage-like textures, such as a camel’s head blending with the wooden fence in the background.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

#### OCLR-TTAOursRCF-AllABPCISGTEM

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

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

- Figure 13. Our method handles occlusion scenarios more effectively. Thanks to long-range tracks, we can accurately follow a boy temporarily obscured by trees. Furthermore, our approach addresses complex situations, such as transparent glass, by including it in the mask to ensure the completeness of the moving object mask. Additionally, for highly intricate reflections, such as vehicle shadows, our method can accurately exclude them.

