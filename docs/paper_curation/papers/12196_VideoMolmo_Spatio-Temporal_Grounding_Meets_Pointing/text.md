arXiv:2506.05336v2[cs.CV]5Jul2025

# VIDEOMOLMO: Spatio-Temporal Grounding Meets Pointing

Ghazi Shazan Ahmad1∗ Ahmed Heakl1∗ Hanan Gani1 Abdelrahman Shaker1 Zhiqiang Shen1 Fahad Shahbaz Khan1,4 Salman Khan1,5

1Mohamed Bin Zayed University of Artificial Intelligence 4Linköping University 5Australian National University Correspondence: {ghazi.ahmad, ahmed.heakl, hanan.ghani} @mbzuai.ac.ae

###### https://mbzuai-oryx.github.io/VideoMolmo

[Figure 1]

- Figure 1: Given complex referring expressions in natural language, VIDEOMOLMO demonstrates improved spatio-temporal reasoning in visual grounding. By decomposing the visual grounding task into sequential steps—pointing (denoted by star) followed by generating masks (in red) -VIDEOMOLMO produces more accurate and coherent segmentation masks compared to prior approaches.

## Abstract

Spatio-temporal localization is vital for precise interactions across diverse domains, from biological research to autonomous navigation and interactive interfaces. Current video-based approaches, while proficient in tracking, lack the sophisticated reasoning capabilities of large language models, limiting their contextual understanding and generalization. We introduce VIDEOMOLMO, a large multimodal model tailored for fine-grained spatio-temporal pointing conditioned on textual descriptions. Building upon the Molmo architecture, VIDEOMOLMO incorporates

*Equal technical contribution

Preprint. Under review.

a temporal module utilizing an attention mechanism to condition each frame on preceding frames, ensuring temporal consistency. Additionally, our novel temporal mask fusion pipeline employs SAM2 for bidirectional point propagation, significantly enhancing coherence across video sequences. This two-step decomposition i.e., first using the LLM to generate precise pointing coordinates, then relying on a sequential mask-fusion module to produce coherent segmentation, not only simplifies the task for the language model but also enhances interpretability. Due to the lack of suitable datasets, we curate a comprehensive dataset comprising 72k video-caption pairs annotated with 100k object points. To evaluate the generalization of VideoMolmo, we introduce VPoS-Bench, a challenging out-of-distribution benchmark spanning five real-world scenarios: Cell Tracking, Egocentric Vision, Autonomous Driving, Video-GUI Interaction, and Robotics. We also evaluate our model on Referring Video Object Segmentation (Refer-VOS) and Reasoning VOS tasks. In comparison to existing models, VIDEOMOLMO substantially improves spatio-temporal pointing accuracy and reasoning capability. Our code and models are publicly available at https://github.com/mbzuai-oryx/VideoMolmo.

## 1 Introduction

Precise spatio-temporal localization underpins a wide range of applications, enabling fine-grained interactions and analysis in dynamic environments. For instance, accurately tracking cell nuclei movement is essential in biological research, aiding in understanding developmental processes and disease progression [22]. Similarly, autonomous vehicles rely on continuous tracking of pedestrians, vehicles and traffic lights to ensure safe navigation; robots manipulating objects must precisely localize contact points over time while avoiding collisions [41]. In egocentric videos, captured from wearable, first-person cameras, spatio-temporal localization is key for tasks such as daily activity recognition, assistive technologies for visually impaired users, and modeling hand-object interactions in real-world settings [8]. Counting objects across frames is also vital in scenarios such as surveillance and traffic monitoring, where assessing object quantities informs critical decisions (Fig. 4).

Prior efforts [24, 40, 44, 36, 14] have explored text-guided tracking and grounding in videos. However, these approaches often lack the reasoning depth offered by large language models (LLMs), limiting their generalization and contextual understanding. While recent video large multimodal models (LMMs) can generate dense, spatio-temporally coherent masks [3, 23], none support free-form, text-conditioned pointing in videos.

For instance, as illustrated in Fig. 1, the instruction is to identify “the white pigeon that has moved slightly from left to right” among five pigeons in close proximity in a video sequence. Solving such a task demands both temporal understanding and fine-grained reasoning to distinguish subtle object movements. Existing methods often struggle in such scenarios, frequently predicting multiple objects or localizing the wrong one, highlighting the limitations of models without integrated reasoning and fine-grained localization capabilities.

Image-based pointing LMMs have demonstrated strong single-frame performance [5, 41], but they cannot model the temporal dynamics essential for video tasks. To fill this gap, we introduce VIDEOMOLMO, an LMM that accepts natural-language queries and produces point-level predictions for target objects across entire video sequences, ensuring temporal consistency. In this way, VIDEOMOLMO decomposes visual grounding in videos into two simpler stages: first, generating precise pointing coordinates via the LLM, then sequentially fusing these points into coherent masks with a dedicated module. This decomposition simplifies the problem for the language model and improves overall interpretability.

VIDEOMOLMO is built on top of Molmo [5], which features a pre-processor that converts the input image into multiscale, multi-crop images, a vision encoder, an LLM decoder and a visual projector that pools and projects image features into the LLM’s embedding space. While Molmo processes images independently, we adapt Molmo architecture to process video data in a simplified manner. We first introduce a temporal module that handles the temporal nature of the video data by conditioning each frame on previous frames through an attention mechanism. Additionally, we propose a novel temporal mask fusion pipeline that leverages SAM2 for bidirectional point propagation, enhancing temporal coherence by utilizing local structural cues embedded in videos. Since there exists no spatio-temporal

pointing dataset to train such systems, we release a comprehensive dataset of 72k video-caption pairs and 100k object points. To evaluate the generalization of VideoMolmo, we introduce VPoS-Bench, a challenging out-of-distribution benchmark spanning five real-world scenarios: Cell Tracking, Egocentric Vision, Autonomous Driving, Video-GUI Interaction, and Robotics. We also assess our model on Referring Video Object Segmentation (Ref-VOS) and Reasoning VOS tasks, which require masks as output. For this, we leverage SAM2 to convert the predicted points into segmentation masks and propose a bidirectional temporal mask fusion technique that enhances mask consistency without further training. Experimental results show that VideoMolmo outperforms existing approaches across all benchmarks and task settings, offering a generalizable solution for fine-grained languageguided reasoning in dynamic visual environments. Specifically, on our challenging VPoS- Bench, VideoMolmo exhibits 5.4 pp average improvement compared to the strongest baseline (Table 4). Similarly, on the challenging referring segmentation on MeViS [6], VideoMolmo outperforms the strongest baseline by 9.5 pp (Table 1), highlighting its effectiveness in visual grounding and reasoning tasks.

## 2 Related work

Video-LMMs. Multi-modal LLMs such as [18, 45] have demonstrated notable advancements due to their strong zero-shot abilities, which have been possible due to their training on millions of image-text pairs. Typically, such models project visual information in the latent space of LLM via an encoder and a connector and thus align the information from vision and text modality. The work in LMMs paved the way for the development of Video-LMMs [13, 43, 15, 20, 37, 47, 2, 42], which unlike image-based LMMs, can reason on dynamic video content. While effective for overall video input comprehension, these methods fell short of fine-grained visual grounding in videos.

Visual grounding. Recent works in Grounded LMMs [30] have sparked tremendous interest among the research community. Visual grounding [17] seeks to identify the location of nouns or short phrases (such as a man with blue shirt) within an image. These models are trained on a huge dataset of image-caption pairs along with the dense segmentation mask associated with the objects in the caption. [3, 23] extended the grounding to video data by releasing a large dataset of grounded video-QA triplet pairs along with the masks associated with the objects. Training on such huge video grounded datasets allowed for video grounding. On the contrary, our proposed VideoMolmo model and dataset are designed for outputting precise object level points which are crucial for applications such as autonomous driving, counting tasks, robotics etc.

Language-assisted object tracking. Most text-based tracking methods are limited to tracking a single object only [40, 44, 36, 14]. However, real-world applications can feature multiple object trajectories, making it harder for the single object tracking methods. [24] propose Type-toTrack along with a tracking dataset ‘GroOT’ for multi-object tracking. However, they track objects via bounding boxes and not precise points which limits their applicability. Another work SAM-PT [29] proposes to use SAM [31] model along with a long-term point tracking mechanism for point-centric interactive video segmentation. However, since their method adapts a 2D model to handle video data, it faces challenges in temporal consistency, especially in the cases of occlusions and fast-moving objects. On the contrary, our proposed VIDEOMOLMO is trained end-to-end on our training dataset and maintains temporal consistency via a dedicated memory module.

## 3 VIDEOMOLMO

VIDEOMOLMO is trained end-to-end for spatio-temporal pointing conditioned on textual instructions. It features a visual encoder, a temporal module, an LLM, and a post-processing module. The visual encoder processes the video frames and outputs multi-crop features. To maintain temporal consistency, we introduce a temporal module which employs a cross-attention operation and ensures that the current frame attends to the information in previous frames. The resultant features are then passed to the LLM, which, along with the textual query, processes this information and outputs the points corresponding to the objects. Our post-processing module takes the predictions from VIDEOMOLMO and uses SAM2 to propagate the points across all video frames bidirectionally.

##### 3.1 Architecture

VIDEOMOLMO extends the Molmo architecture [5] from static image understanding to spatiotemporal video grounding. The model consists of four end-to-end trainable components: (1) a visual

[Figure 2]

###### LLM

[Figure 3]

Visual Projector

###### Textual Query

<x, y, alt text>

[Figure 4]

K

[Figure 5]

Point to the ball

Multi-Head Cross Attention

| | |
|---|---|
| | |

Q

Feature Pooling V

Temporal Module ( )

SAM2

+

| | |
|---|---|
| | |

Temporal Mask Fusion

| | |
|---|---|

| | |
|---|---|

[Figure 6]

Visual Encoder

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Past frames Current Frame

- Figure 2: VIDEOMOLMO Architecture. The visual encoder extracts multi-crop features from the current frame and the past l frames. These temporal features provide contextual cues and are processed by the Temporal Module M via multi-head cross-attention, where the query comes from the current frame, and key and value from the mean of previous frames. The output is fused with the original features to enrich temporal cues while preserving the spatial details of the current frame. The combined visual-textual representations are then passed to the LLM to predict grounded points. These points are converted into masks using our Bidirectional Temporal Mask Fusion module, ensuring temporally consistent pixel-level grounding.

encoder, (2) a temporal module, (3) visual projector (4) a decoder-only large language model (LLM); and a post-processing module (see Fig. 2).

We represent an input video as V ∈ R|T |×H×W×C where H, W, and C denote the height, width, and number of channels respectively of the frames and |T | are the number of frames in the video. For a frame Ti ∈ V , we generate N spatial overlapping multi-crops to capture both global context and fine-grained details in the frame. The multi-crops of the frame are processed independently using a visual encoder F. Following [5], we build patch features for each crop by concatenating features from third-to-last and tenth-to-last ViT layers. The features extracted from each crop of frame Ti are denoted by fTj

∈ RP×D, where j denotes the crop index, P is the number of patches and D is the latent dimension of the ViT. Since the components in our model are based on 2D modules, we inject temporal information through a dedicated temporal module denoted as M (Sec. 3.2). For each frame Ti, we compute a temporal context feature by aggregating features from the l most recent frames {Ti−l,Ti−l−1,...,Ti−1} such that (l < i) and denote their mean as fT ∗

i

. The pooled context features and the current frame features are then passed into the temporal module. The temporally enriched features predicted by the temporal module M correspond to 2 × 2 patch window features extracted from all the patches of the crops. These window-level features are added to the current frame features by reshaping fT

i−

from RN×P×D to RN.P4 ×4×D.

i

fˆT

(1)

+ M fT

= fT

,fT ∗

i

i

i

i−

These resultant features fˆT

are then aggregated into a single frame-level representation using an attention pooling mechanism. The pooled feature fˆT

i

is then projected into the embedding space of the language model using a learnable projection layer P. It is then concatenated with the tokenized query q and passed to the decoder-only LLM to obtain the final output text p which contains the coordinate {(x,y)}O where O is the number of objects grounded in the frame.

i

p = LLM P(fˆT

); q (2)

i

The model is trained end-to-end by minimizing the cross-entropy loss LCE between the predicted text p and the ground-truth one-hot labels p∗ in an auto-regressive manner:

LCE = −p∗ · log(p) (3)

##### 3.2 Temporal Module

The original MoLMo architecture [5] was developed for static images and cannot model the temporal dynamics inherent in video data. To address this, we introduce a dedicated temporal module M that infuses each frame with temporal context from the preceding l frames [24, 12]. For each

crop j of frame Ti, patch features fTj

∈ RP×D are extracted and reshaped into non-overlapping 2 × 2 windows of four vectors, fTj

i

∈ R4×D}P/p=14. We then flatten the window features across all N crops to obtain resized fetures called window vectors fT

= {fTj,p

i

i

and fT ∗

such that both lie

i

i−

in R(N·P/4)×4×D for both the current features and the context features. To capture fine-grained temporal correspondences, we apply multi-head cross-attention (MHCA) over each patch window, where the query comes from current frame window vector fT

, and key and value come from vector fT ∗

i

such that MHCA fT

denotes the final attended window features. By restricting

, fT ∗

, fT ∗

i

i−

i−

i−

attention to small 2 × 2 neighborhoods, the module selectively refines each local region of the current frame using only the most relevant fine-grained patches from its temporal context. This targeted cross-attention both preserves spatial locality and amplifies subtle motion cues that would be lost in coarse global pooling.

##### 3.3 Bidirectional Temporal Mask Fusion

While our method predicts grounded point coordinates corresponding to objects mentioned in the textual query, most existing evaluation protocols are designed to operate on segmentation masks. Therefore, for compatibility and consistent evaluation, we introduce Bidirectional Temporal Mask Fusion, a novel post-processing technique that leverages SAM2 to convert points to dense masks. Specifically, we use the predicted points as prompts to SAM2 [31] to obtain the segmentation masks of the objects. Due to computational constraints, we avoid dense frame-level inference across the entire video. Instead, we sparsely sample frames at a sampling rate k, and generate point predictions only on these sampled frames. Let the two consecutive sampled frames be denoted as Ti and Ti+k. The corresponding point predictions are transformed into segmentation masks mi and mi+k using SAM2. For frames lying between the two sampled frames (i.e., Ti+n for 0 < n < k), we propagate the neighboring masks mi and mi+k bidirectionally to estimate the intermediate masks. The leftward propagation from mi and the rightward propagation from mi+k are performed using SAM2’s temporal propagation module, defined as,

m→i+n = Prop→({Ti,mi},Ti+n) m←i+n = Prop←({Ti+k,mi+k},Ti+n),

(4)

where m→i+n and m←i+n denote the masks propagated from the left and right directions, respectively. To reconcile the two propagated masks at frame Ti+n, we compute their Intersection over Union (IoU). If the IoU exceeds a predefined threshold τ, we adopt the intersection as the final prediction.

IOU( m→i+n, m←i+n), if IOU(·) ≥ τ m→i+n ∪ m←i+n, otherwise

(5)

mi+n =

The OR operation in the second case accounts for cases with significant motion transitions, where the left and right predictions might deviate spatially or structurally. In such a case, two masks propagated from the left and the right considerably differ in either spatial location or shape.

In rare cases, SAM2 may fail to propagate a mask from one direction, resulting in an empty prediction. We handle this with a fallback mechanism. Specifically, if the mask being propagated from the left is empty, we assign the right mask to the prediction of the current frame. Otherwise, we assign the left mask. Mathematically, it can be written as,

mi+n =

m←i+n, if m→i+n = ∅ m→i+n, otherwise

(6)

[Figure 13]

[Figure 14]

[Figure 15]

|SAM2|
|---|

0.8

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

|IOU|
|---|

[Figure 20]

[Figure 21]

0.95

Sample points

[Figure 22]

Raw video

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

0.9

[Figure 28]

GT Mask

GT Mask

Candidate Masks

- Figure 3: VIDEOMOLMO annotation pipeline: We construct point-level supervision from framelevel masks using a semi-automatic process. For each frame, k points are sampled on the mask and passed to SAM2 to generate candidate masks. The point with the highest-IoU candidate mask (w.r.t. ground truth) is selected as the optimal annotation.

This bidirectional temporal fusion ensures temporally consistent masks with reduced compute overhead. It not only aligns with existing segmentation-based evaluation protocols but also introduces robustness in dynamic scenes by leveraging context from both past and future observations.

- 4 VIDEOMOLMO dataset

Training Data: Our dataset is essential for training the model’s spatio-temporal pointing capabilities. It features 72k Video-caption pairs along with 100k object points. Our benchmark video dataset comes from different sources: Refer-YTVOS [32], Refer-DAVIS [27], MeViS [6], GroOT [24], LaSOT [7], ViCaS-LGVIS [1], and Reason-VOS [38].To create fine-grained data grounded in point coordinates, we develop a semi-automated annotation pipeline (see Fig. 3) that ensures both high-quality and scalable annotations. Each of the above mentioned datasets features video-mask-expression triplets (V,M,T) such that V ∈ R|T |×H×W×C,M ∈ {0,1}|T |×|O|×H×W with |O| denoting the number of unique annotated objects in the frame Ti. For each object Oj ∈ O in frame Ti ∈ RH×W×C, with a binary mask mj ∈ {0,1}H×W, the goal is to extract a single highly representative point coordinate for the object. We sample k candidate points within the mask, assigning each point (x,y) a sampling probability proportional to its Euclidean distance to the nearest boundary pixel of the mask mj, i.e.,

P(x,y) ∝ min

(x′,y′)∈∂mj

∥(x,y) − (x′,y′)∥2 (7)

where ∂mj denotes the set of boundary pixels of the mask. For each sampled point, we use SAM2 to predict a segmentation mask. We then compute the Intersection-over-Union (IoU) between each

predicted mask and the corresponding ground truth mask mj. The point coordinate whose predicted mask achieves the highest IoU is selected as the representative ground truth point for the object:

p∗ = arg max (x,y)

IoU(SAM2(x,y), mj), (8) where SAM2(x,y) denotes the predicted mask obtained using point (x,y) as a prompt to SAM2.

VPoS-Bench: To evaluate the generalization capabilities of VIDEOMOLMO, we introduce Video Pointing and Segmentation (VPoS-Bench), a curated benchmark test set comprising of 100 videocaption pairs and 1k manually annotated object points. For mask-based evaluations, we employ the SAM2 model to convert these point annotations into segmentation masks. The test benchmark encompasses diverse real-world scenarios, sourced from both open datasets [4, 16, 8, 34, 21] and internal collections, spanning five categories: Cell Tracking, Egocentric Videos, Autonomous Driving, Video-GUI, and Robotics. Our benchmark also consists of a dedicated counting task sourced from [9] dataset. For additional details about the benchmark, refer to Sec. A.3 of the Appendix.

- 5 Experiments

Implementation details. VIDEOMOLMO follows the architecture of Molmo [5]. For the image encoder, we use a pretrained CLIP ViT-L/14 (336 × 336) [28] model. Our proposed Temporal Module is initialized from scratch. Our choice of LLM is the pretrained Qwen2 7B [39]. We train the model

[Figure 29]

- Figure 4: VIDEOMOLMO demonstrates robust generalization and fine-grained spatio-temporal grounding across diverse out-of-distribution scenarios from our proposed benchmark, for instance, correctly pointing to traffic lights (2nd row) in challenging driving scenes despite never encountering such scenarios during training. (Please refer to Appendix A.4 for additional qualitative results.)

on 8 NVIDIA A100 80GB GPUs. Learning rate of 1e−5 is used for the LLM, and 5e−6 for the vision encoder, visual projector and temporal module. We adopt a batch size of 1 with 256 gradient accumulation steps, and use AdamW optimizer [19] following the fine-tuning recipe from [5].

Tasks. We evaluate VIDEOMOLMO on four challenging tasks: point grounding, counting, referring segmentation, and reasoning video object segmentation. For point grounding, we report performance on our proposed VPoS-Bench. For the counting task, we utilize videos from the Kinetics dataset [9], where object counts range from 2−13. For referring video segmentation, we use MeViS validation set [6] Refer-DAVIS-17 [10] and Refer-Youtube-VOS [33] datasets. Finally, for reasoning segmentation, we evaluate our model on the ReasonVOS dataset [3].

Evaluation metrics. For the Point Grounding task, we follow the evaluation protocol of Molmo [5] and report Precision, Recall, and F1 score. For mask-based evaluations, we use Region Jaccard (J ), Boundary F-measure (F), and their average (J &F). For the Counting task, we report Exact Matching Accuracy (EMA) and Mean Absolute Error (MAE) [5, 44].

Baselines. For point grounding, we compare VIDEOMOLMO with three strong baselines: VideoLISA[3], VideoGLaMM [23], and Molmo+SAM2. To adapt Molmo for videos, we augment it with SAM2. For referring segmentation, we evaluate against VideoLISA, VideoGLaMM, and prior baselines. For counting, we compare VIDEOMOLMO with both closed-source (GPT-4o [25]) and open-source models [5, 2, 35]). For further experimentation details please refer to Appendix A.2.

##### 5.1 Main experimentation results

Point Grounding. The point grounding task focuses on accurately identifying the spatial coordinates of a queried object within video frames. As depicted in Fig. 5, VIDEOMOLMO demonstrates superior performance in localizing target points, as evidenced by its significantly higher Precision, Recall, and F1 scores compared to Molmo. This performance gap can be attributed to Molmo’s training on static frames, which limits its ability to handle temporal variations. In dynamic video inputs, where object presence and position may vary across frames, Molmo struggles, whereas VIDEOMOLMO effectively addresses this challenge by leveraging temporal context. Furthermore, VIDEOMOLMO outperforms all baseline models across each subtask from our benchmark, as evident from higher J , F, and the combined J &F metric (Table 4). Qualitative results in Fig.4 further validate the robustness of VIDEOMOLMO, showcasing its ability to accurately localize objects across diverse and out-of-distribution scenarios.

- Table 1: Performance comparison on Refer-DAVIS-17, Refer-Youtube-VOS, and MeViS benchmarks. VideoMolmo consistently improves referring video object segmentation across datasets.

Model Refer-DAVIS-17 Refer-Youtube-VOS MeViS

J F J &F J F J &F J F J &F

LISA-7B [11] 61.9 54.9 58.4 50.6 49.7 50.2 – – – LISA-13B [11] 64.6 56.8 60.7 53.0 52.1 52.6 – – – TrackGPT-7B [46] 67.0 59.4 63.2 57.4 55.3 56.4 – – – TrackGPT-13B [46] 70.4 62.7 66.5 60.8 58.1 59.5 – – – VideoLISA [3] 72.7 64.9 68.8 65.7 61.7 63.7 41.3 47.6 44.4 VideoGLaMM [23] 73.3 65.6 69.5 65.4 68.2 66.8 42.1 48.2 45.2 Molmo [5]+SAM2 [31] 65.3 72.2 68.8 61.0 66.2 63.6 44.4 49.4 46.9

VideoMolmo 71.3 73.6 72.5 65.6 69.1 67.3 51.2 56.6 53.9

- Table 2: Performance comparison of VideoMolmo on the ReasonVOS benchmark.

Table 3: Performance comparison of VideoMolmo on the counting benchmark.

Model J F J &F

Model MAE ↓ EMA ↑

LISA [11] 33.1 29.1 31.1 VideoLISA [3] 49.9 45.1 47.5 VideoGLaMM [23] 40.5 27.2 33.9 Molmo [5] + SAM2 [31] 43.5 47.8 45.7 VideoMolmo 48.7 53.4 51.1

GPT-4o [26] 0.76 60.0 Gemma3-12B [35] 0.96 43.3 Qwen2.5-VL-7B [2] 0.83 50.0 Molmo [5] 1.21 49.3 VideoMolmo 0.72 73.3

- Table 4: Performance of various models on five subtasks of VPoS-Bench (Ego4D, Robotics, Autonomous, Cells, VideoGUI)

Model Ego4D Robotics Autonomous Cells VideoGUI

J F J &F J F J &F J F J &F J F J &F J F J &F

|VideoLISA [3] VideoGLaMM [23] Molmo [5] +SAM2 [31]<br><br>|47.1 41.1 44.2<br><br>47.2 40.4 43.8 50.6 50.1 50.4<br><br><br>|3.9 2.0 2.9 15.3 10.3 12.8 27.8 25.6 26.7|34.7 22.0 28.4 31.4 17.7 24.8 49.5 47.5 48.5<br><br>|18.9 3.3 11.1<br><br>11.8 7.8 9.8<br><br>20.8 7.6 14.2|65.3 39.4 52.4 58.7 32.5 45.6 60.2 55.9 58.0<br><br>|
|---|---|---|---|---|---|
|VideoMolmo|55.5 54.3 54.9<br><br>|29.1 26.2 27.6<br><br>|57.1 57.7 57.4<br><br>|25.4 13.1 19.2<br><br>|68.9 62.4 65.7<br><br>|

72.5

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

| | |Molmo VideoMolmo<br><br>| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

60.0

72.0

57.5

71.5

55.0

Scores

71.0

###### J&F

52.5

70.5

50.0

70.0

47.5

69.5

45.0

Precision Recall F1

0 1 2 3 4 5

Metrics

l

- Figure 5: Performance comparison of VideoMolmo on point grounding.

Figure 6: Effect of temporal module context length on segmentation accuracy in ReferDAVIS benchmark.

Object Counting. Our benchmark introduces a dedicated object counting task, a capability essential to many real-world video understanding applications. The number of objects in our benchmark ranges from 2 to 13, therefore demanding enhanced temporal and spatial understanding. For this evaluation, we compare VIDEOMOLMO against both open-source and proprietary models, as shown in Table 3. VIDEOMOLMO achieves state-of-the-art performance, significantly outperforming all baselines in terms of both Mean Absolute Error (MAE) and Exact Match Accuracy (EMA). Notably, VIDEOMOLMO also surpasses advanced proprietary model such as GPT-4o, underscoring its strength

- as a specialized counting model. We attribute this success to the explicit training of VideoMolmo on our proposed training dataset, which includes diverse scenarios with 0 to 10 objects per video, enabling the model to develop a fine-grained understanding of multi-object scenes.

Referring Segmentation. For referring video segmentation, the goal is to localize specific object instances in a video based on a given phrase. Table 1 presents results across three standard datasets. On the MeViS benchmark, which involves motion-guided segmentation with multiple objects, VIDEOMOLMO outperforms all baselines by a notable margin, demonstrating its effectiveness in grounding

- Table 5: Effect of different temporal mask fusion strategies on Refer-DAVIS dataset.

Strategy J F J &F Prefer left 67.31 73.38 70.34 Prefer right 67.05 71.69 69.37 Intersection 60.20 63.58 61.89 Larger mask 70.40 72.91 71.65 Smaller mask 64.10 67.18 65.64 VideoMolmo 71.27 73.63 72.45

Table 6: Ablation on different variants of the temporal module on the Refer-DAVIS dataset.

Variant J F J &F Single frame 66.04 72.60 69.32 Addition 66.13 72.97 69.55 Concatenation 65.34 72.06 68.71 VideoMolmo 71.27 73.63 72.45

complex, multi-object scenes. This advantage stems in part from the simplicity and efficiency of VIDEOMOLMO’s point-based supervision as reflected in its superior J , F, and J &F scores, which contrasts with recent methods like VideoGLaMM [23] and VideoMolmo [3] that rely on dense, pixel-level mask prediction where precise delineation between objects becomes challenging (Fig. 1). VIDEOMOLMO also achieves superior performance on Refer-DAVIS-17 and Refer-Youtube-VOS. Notably, VideoGLaMM performs competitively on Refer-Youtube-VOS which features fast-moving objects, benefiting from its dual encoder architecture that integrates spatial and temporal features. However, VIDEOMOLMO, despite using a single encoder with point-based supervision, surpasses these strong baselines which could be attributed to its temporal module capturing past inference, novel post-processing technique, and point grounding training paradigm.

Reasoning Video Object Segmentation. Table 2 presents a comparative analysis of VIDEOMOLMO against existing approaches on the ReasonVOS benchmark, which emphasizes complex reasoning, temporal comprehension, and consistency across frames, making it particularly challenging. Prior methods perform noticeably worse, largely due to their limited temporal and fine-grained reasoning capabilities. While VideoLISA incorporates spatio-temporal cues, it still falls short of VIDEOMOLMO. This performance gap highlights VIDEOMOLMO’s architectural strengths, specifically its dedicated temporal module providing rich spatio-temporal contextual understanding.

5.2 Ablations and Analysis

Effect of Temporal Module. We conduct an ablation study to evaluate the effectiveness of different temporal module variants on the Refer-DAVIS benchmark (Table 6). Using a single frame or simple feature fusion methods such as addition or token-space concatenation yields relatively lower performance compared to our proposed cross-attention-based temporal module as it enables dynamic and selective integration of relevant features across frames, allowing the model to focus on temporally coherent and semantically meaningful cues critical for accurate grounding.

Ablation on Temporal Mask Fusion: To enable efficient and temporally consistent segmentation, we evaluate various strategies for combining masks propagated from the sampled frames. As shown in Table 5, naive strategies like preferring left/right predictions or computing mask intersections result in suboptimal performance, either due to loss of temporal context or overly conservative fusion. Our proposed bidirectional fusion strategy outperforms all baselines by adaptively reconciling forward and backward propagated masks based on their agreement (IoU). Our fallback mechanism ensures robustness against failure cases where one of the propagated masks is missing. This approach achieves a significant improvement in J &F score (72.45), demonstrating its effectiveness.

Effect of context-length in temporal module: To analyze the effect of context length in the temporal module, we evaluate VIDEOMOLMO on the Refer-DAVIS benchmark (Fig. 6). We observe that there is a consistent increase in J&F as the context length increases from 1 → 4, indicating that incorporating more temporal information enhances the model’s spatio-temporal reasoning. However, there is a slight drop in accuracy at l = 5, suggesting that adding more frames may introduce redundancy or noise rather than useful context. Please refer to Appendix A.1 for additional ablations.

- 6 Conclusion

We present VIDEOMOLMO, an LMM for fine-grained spatio-temporal pointing conditioned on textual queries. It leverages a temporal module that incorporates temporal cues from preceding frames and a novel bidirectional post-processing strategy for robust mask prediction. To enable training, we curate a large-scale dataset using a semi-automatic annotation pipeline. VIDEOMOLMO shows strong generalization and consistently outperforms state-of-the-art models across diverse and

out-of-distribution tasks, including point grounding, object counting, referring segmentation, and reasoning segmentation.

Limitations and Future Work. VIDEOMOLMO demonstrates strong spatio-temporal grounding performance, excelling in fine-grained localization without explicit pixel-level mask supervision. However, its performance is relatively limited on videos with fast-moving objects, such as those in Refer-Youtube-VOS (Table 1), due to single-frame processing during training—an efficiency-driven design constrained by architectural and computational limits. Additionally, VIDEOMOLMO relies on SAM2, making mask quality dependent on SAM2’s performance from point predictions. Predicting a single point per object can also yield suboptimal masks (see A.4). Future work may include joint multi-frame training with improved sampling strategies and extending VIDEOMOLMO to predict multiple points per object to enhance mask quality and further benefit downstream applications.

## 7 Acknowledgement

The computations were enabled by resources provided by NAISS at Alvis partially funded by Swedish Research Council through grant agreement no. 2022-06725, LUMI hosted by CSC (Finland) and LUMI consortium, and by Berzelius resource provided by the Knut and Alice Wallenberg Foundation

## at the NSC. References

- [1] Ali Athar, Xueqing Deng, and Liang-Chieh Chen. Vicas: A dataset for combining holistic and pixel-level video understanding using captions with grounded segmentation. arXiv preprint arXiv:2412.09754, 2024.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Zheng Zhang, and Mike Zheng Shou. One token to seg them all: Language instructed reasoning segmentation in videos. Advances in Neural Information Processing Systems, 37:6833–6859, 2024.
- [4] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.
- [5] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.
- [6] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. MeViS: A large-scale benchmark for video segmentation with motion expressions. In ICCV, 2023.
- [7] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5374–5383, 2019.
- [8] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.
- [9] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

- [10] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part IV 14, pages 123–141. Springer, 2019.
- [11] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692, 2023.
- [12] Zihang Lai, Erika Lu, and Weidi Xie. Mast: A memory-augmented self-supervised tracker. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6479–6488, 2020.
- [13] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv:2305.06355, 2023.
- [14] Yihao Li, Jun Yu, Zhongpeng Cai, and Yuwen Pan. Cross-modal target retrieval for tracking by natural language. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 4927–4936, 2022.
- [15] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [16] Kevin Qinghong Lin, Linjie Li, Difei Gao, Qinchen Wu, Mingyi Yan, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. Videogui: A benchmark for gui automation from instructional videos. arXiv preprint arXiv:2406.10227, 2024.
- [17] Fenglin Liu, Xian Wu, Shen Ge, Xuancheng Ren, Wei Fan, Xu Sun, and Yuexian Zou. Dimbert: Learning vision-language grounded representations with disentangled multimodal-attention. ACM Transactions on Knowledge Discovery from Data (TKDD), 16(1):1–19, 2021.
- [18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [19] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.
- [20] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024.
- [21] M. Maška, V. Ulman, P. Delgado-Rodriguez, E. Gómez-de Mariscal, T. Neˇcasová, F.A. Guerrero Peña, T.I. Ren, E.M. Meyerowitz, T. Scherr, K. Löffler, R. Mikut, T. Guo, Y. Wang, J.P. Allebach, R. Bao, N.M. Al-Shakarji, G. Rahmon, I.E. Toubal, K. Palaniappan, F. Lux, P. Matula, K. Sugawara, K.E.G. Magnusson, L. Aho, A.R. Cohen, A. Arbelle, T. Ben-Haim, T.R. Raviv, F. Isensee, P.F. Jäger, K.H. Maier-Hein, Y. Zhu, C. Ederra, A. Urbiola, E. Meijering, A. Cunha, A. Muñoz-Barrutia, M. Kozubek, and C. Ortiz-de Solórzano. The cell tracking challenge: 10years of objective benchmarking. Nature Methods, 20(7):1010–1020, July 2023. Epub 2023 May 18.
- [22] Erik Meijering, Oleksiy Dzyubachyk, and Ihor Smal. Methods for cell and particle tracking. In Methods in Enzymology, volume 504, pages 183–200. Academic Press, 2012.
- [23] Shehan Munasinghe, Hanan Gani, Wenqi Zhu, Jiale Cao, Eric Xing, Fahad Shahbaz Khan, and Salman Khan. Videoglamm: A large multimodal model for pixel-level visual grounding in videos. arXiv preprint arXiv:2411.04923, 2024.
- [24] Pha Nguyen, Kha Gia Quach, Kris Kitani, and Khoa Luu. Type-to-track: Retrieve any object via prompt-based tracking. Advances in Neural Information Processing Systems, 36:3205–3219, 2023.
- [25] OpenAI. Chatgpt: Large language model for human-style conversation. https://chat. openai.com, 2023.
- [26] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [27] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Computer Vision and Pattern Recognition, 2016.
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [29] Frano Rajiˇc, Lei Ke, Yu-Wing Tai, Chi-Keung Tang, Martin Danelljan, and Fisher Yu. Segment anything meets point tracking. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 9302–9311. IEEE, 2025.
- [30] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Erix Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. arXiv preprint arXiv:2311.03356, 2023.
- [31] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [32] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16, pages 208–223. Springer, 2020.
- [33] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 208–223, Cham,

2020. Springer International Publishing.

- [34] Harsh Singh, Rocktim Jyoti Das, Mingfei Han, Preslav Nakov, and Ivan Laptev. Malmm: Multi-agent large language models for zero-shot robotics manipulation. arXiv preprint arXiv:2411.17636, 2024.
- [35] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clément Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open models based on gemini research and technology, 2024.
- [36] Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, and Feng Wu. Towards more flexible and accurate object tracking with natural language: Algorithms and benchmark. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13763–13773, 2021.

- [37] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024.
- [38] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Visa: Reasoning video object segmentation via large language models. In European Conference on Computer Vision, pages 98–115. Springer, 2024.
- [39] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [40] Zhengyuan Yang, Tushar Kumar, Tianlang Chen, Jingsong Su, and Jiebo Luo. Groundingtracking-integration. IEEE Transactions on Circuits and Systems for Video Technology, 31(9):3433–3443, 2020.
- [41] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024.
- [42] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.
- [43] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv:2306.02858, 2023.
- [44] Haojie Zhao, Xiao Wang, Dong Wang, Huchuan Lu, and Xiang Ruan. Transformer visionlanguage tracking via proxy token guided cross-modal fusion. Pattern Recognit. Lett., 168:10– 16, April 2023.
- [45] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [46] Jiawen Zhu, Zhi-Qi Cheng, Jun-Yan He, Chenyang Li, Bin Luo, Huchuan Lu, Yifeng Geng, and Xuansong Xie. Tracking with human-intent reasoning. arXiv preprint arXiv:2312.17448, 2023.
- [47] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## A Appendix

- A.1 Additional Ablations

- A.1.1 Training Ablations

In addition to the ablation studies presented in the main paper, we conduct further investigations into the impact of language backbone choice, parameter tuning, and numerical precision on the Ref-DAVIS dataset, as summarized in Table 7.

In the first row, we assess the effect of replacing the Qwen-7B language model with Olmo-7B. The resulting drop in J &F score highlights Qwen-7B’s superior grounding capabilities which is consistent with the observational findings reported in [5]. This emphasizes the importance of selecting an Qwen-7b LLM with strong multimodal alignment for visual grounding tasks.

- Table 7: Performance comparison of different VIDEOMOLMO variants on Refer-DAVIS benchmark.

|Variant|J<br><br>|F|J&F|
|---|---|---|---|
|VideoMolmo-O-7B VideoMolmo (LoRA) VideoMolmo (16bit)<br><br>|66.25 67.82 67.74<br><br>|72.93 74.72 74.72|69.59 71.27 71.25<br><br>|
|VideoMolmo|71.27|73.63|72.45|

Next, we investigate the impact of end-to-end training. In the second row, we freeze the video encoder and train only the LLM’s projection layers by integrating LoRA adapters. This lightweight training strategy significantly underperforms compared to the fully fine-tuned model (last row), validating our hypothesis that joint optimization of all components is essential for capturing the temporal nuances required for precise point grounding.

Finally, we examine the effect of training precision. In the third row, we use 16-bit floating point precision which is commonly adopted to save memory and accelerate training. However, we find that this leads to a notable degradation in performance. In contrast, training with full 32-bit precision (last row) enhances the model’s capacity to learn fine-grained spatial and temporal cues, consistent with prior observations in [5].

Together, these ablations underline the significance of careful backbone selection, full end-to-end optimization, and high-precision training for achieving robust and fine-grained visual grounding in VideoMolmo.

- A.1.2 Inference Ablations

Sampling rate k: As described in the main paper, we adopt a frame sampling rate of k = |T | for the Molmo+SAM2 baseline during inference, which means that we take the first frame prediction and use SAM2 to propagate the mask across all the frames. This choice is motivated by performance on the Refer-DAVIS-17 dataset, where Molmo+SAM2 achieves its highest J &F score of 67.69 at this value. However, our analysis in Table 8 reveals that the optimal sampling rate is not universal, it varies across datasets. To ensure a fair and competitive comparison with our proposed VideoMolmo, we conduct additional ablations on the sampling rate k for Molmo+SAM2 across the Refer-YouTubeVOS and MeViS datasets. We find that a lower sampling rate of k = 3 yields the best performance on Refer-YouTube-VOS, while k = 1 proves optimal on MeViS. Despite this tuning, VideoMolmo consistently outperforms Molmo+SAM2 under each dataset’s optimal configuration. Interestingly, across all three datasets, we observe a consistent decline in performance as the sampling rate increases. This is particularly evident at k = 30, where the baseline performs starts dropping. These findings further highlight the robustness of VideoMolmo in leveraging temporal context, even when competing baselines are tuned to their best-performing configurations.

We further ablate the effect of sampling rate on our proposed VideoMolmo. While our main results on the Refer-YouTube-VOS benchmark in the main paper are reported using a sampling rate of k = 5, we acknowledge that this choice, although consistent with the baseline configuration, may not be optimal for our method. As seen from the Table 10, VideoMolmo benefits from careful selection of

- Table 8: Effect of sampling rate k on Refer-DAVIS-17, Refer-Youtube-VOS, and MeViS benchmarks using Molmo + SAM2

k Refer-DAVIS-17 Refer-Youtube-VOS MeViS J F J &F J F J &F J F J &F

1 58.32 64.60 61.46 60.08 64.69 62.38 45.53 51.06 48.30 3 58.77 63.85 61.31 60.11 65.04 62.58 – – – 10 59.61 64.77 62.19 60.24 64.88 62.56 45.63 50.93 48.28 30 63.31 69.31 66.31 59.48 64.02 61.75 45.51 50.65 48.08 |T | 65.29 70.09 67.69 59.48 64.07 61.78 44.37 49.37 46.87

- Table 9: Effect of varying threshold τ on the performance of VIDEOMOLMO evaluated on the Refer-DAVIS benchmark.

Table 10: Effect of sampling rate k on the performance of VIDEOMOLMO evaluated on the Refer-YouTube-VOS benchmark.

k J F J &F 3 64.39 67.68 66.03 10 65.69 69.39 67.54 15 66.26 69.95 68.11 20 66.34 69.93 68.14 30 65.80 69.32 67.56 VideoMolmo (k = 5) 65.55 69.11 67.33

τ J F J &F

- 0 67.31 73.38 70.34 0.3 69.05 75.51 72.28 0.5 68.90 75.26 72.08 0.9 68.85 75.24 72.05 VideoMolmo (τ = 0.7) 71.27 73.63 72.45

sampling rate as we observe that a sampling rate k = 20 yields the highest J &F score of 68.14, compared to 67.33 with k = 5.

Threshold τ: Our proposed post-processing strategy, Bidirectional Temporal Mask Fusion, enhances model performance by combining masks propagated from both right and left directions to achieve a robust temporal consensus. As described in Section 3.3 of the main paper, the fusion process is governed by a threshold hyperparameter τ, which determines how agreement between the two masks is evaluated. Specifically, when the Intersection-over-Union (IoU) between the forward and backward masks exceeds τ, their intersection is used as the final mask, enforcing stricter agreement. Conversely, if the IoU falls below τ, their union is taken, promoting flexibility in ambiguous regions. This mechanism balances precision and recall based on temporal consistency. We ablate different values of τ in Table 9 to identify the most effective setting. The results indicate that τ = 0.7 yields the best overall performance. However, the differences across values are relatively minor, underscoring the high quality and stability of the point predictions generated by VideoMolmo. This consistency highlights the robustness of our model in temporal point grounding, even under varying post-processing thresholds.

Effect of Post-processing on baselines: To assess the generalizability and effectiveness of our proposed Bidirectional Temporal Mask Fusion, we integrate it with the Molmo+SAM2 baseline, resulting in an enhanced variant denoted as Molmo†+SAM2. As illustrated in Fig. 7, this integration consistently improves performance across all three datasets in terms of J &F score. These results demonstrate that our post-processing strategy not only strengthens our own model but also benefits existing methods. The modular, plug-and-play nature makes it a valuable addition to any video grounding pipeline, improving temporal consistency and overall segmentation quality with minimal integration effort.

##### A.2 Additional Experimentation Details

VIDEOMOLMO follows the architecture of Molmo [5]. For the image encoder, we use a pretrained CLIP ViT-L/14 (336 × 336) [28] model. We initialize the temporal module with Xavier Normalized weights for stable training. Our choice of LLM is the pretrained Qwen2 7B [39]. We train the model on 8 NVIDIA A100 80GB GPUs. Learning rate of 1e−5 is used for the LLM, and 5e−6 for the viion encoder, visual projector and temporal module. We adopt a batch size of 1 with 256 gradient accumulation steps, and use AdamW optimizer with β = (0.9,0.95) and ϵ = 1e−6, . We train VIDEOMOLMO for 4 epochs on 8 NVIDIA A100 GPUs (80 GB each), consuming roughly 1,000 GPU-hours in total. Training runs in full 32-bit precision with a 10-step linear warmup, after which

80

70

60

#### &

50

Molmo+SAM2

40

Molmo +SAM2

30

MeViS YT-VOS Ref-DAVIS

### Datasets

Figure 7: Effect of Bidirectional temporal mask fusion on Molmo+SAM2 baseline.

we follow a cosine learning-rate schedule; we also clip gradients to a maximum norm of 1.0 to guard against unstable updates. For all inference and reported results, we use 4-bit precision.

##### A.3 VPoS Bench

As mentioned in the main paper, we introduce Video Pointing and Segmentation (VPoS-Bench), a curated benchmark test set comprising of 100 video-caption pairs and 1k manually annotated object points. To obtain mask annotations for evaluation, we use SAM2 to convert these point annotations into segmentation masks. For mask-based evaluations, we employ the SAM2 model to convert these point annotations into segmentation masks. The test benchmark encompasses diverse real-world scenarios, sourced from both open datasets [4, 16, 8] and internal collections, spanning five categories: Cell Tracking, Egocentric Videos, Autonomous Driving, Video-GUI, and Robotics. Our benchmark also consists of a dedicated counting task sourced from [9] dataset. Below, we present details about each subset in VPoS-Bench.

- 1) Cell Tracking: Features internally sourced 12 microscopic videos with dynamic cellular structures, where precise localization of individual cells is essential for tasks like tracking cell division or counting. These videos are partially sourced from [21] and the remaining are requested internally.
- 2) Egocentric Videos: Comprises 18 first-person videos capturing daily human-object interactions, enabling the assessment of grounded pointing in scenarios such as object manipulation and activity recognition. The egocentric videos in our test benchmark are derived from [8] dataset.
- 3) Autonomous Driving: Includes 13 urban driving scenes from nuScenes’s dataset [4], with complex environments, requiring accurate identification of specific road elements (e.g., traffic signals) to support navigation and safety systems.
- 4) Video-GUI: Consists of 13 screen recordings from software applications, focusing on tasks like identifying and interacting with user interface elements based on textual instructions. The VideoGUI videos are sampled from VideoGUI dataset [16].
- 5) Robotics: Encompasses 14 videos of robotic operations, emphasizing the need for precise object localization to execute commands such as "pick up the red block" or "press the top button." Few of the robotic videos in our benchmark are sourced from [34], and the remaining are sourced internally.

##### A.4 Additional qualitative results

General qualitative results. We also present some qualitative results in Figures 8,9,10, 11, and 12 on our proposed VPoS-Bench, MeViS, YT-VOS, Ref-DAVIS, and ReasonVOS, respectively. We observe that in each case, VideoMolmo generates fine-grained points and corresponding masks pertaining to the query objects. In fact, VideoMolmo performs well even in the cases of multi-object

queries (>2 objects) such as in VPoS-Bench counting task of Fig. 8 (1st row) and Fig. 9 (3rd row). Further, VideoMolmo also excels at grounding small and fine-grained objects. Fig. 8 (3rd row) shows VideoMolmo accurately points and grounds the far-away car on the road, although the car is too small to point at in some frames. Similarly, VideoMolmo is able to ground the helmet in Fig. 11 (2nd row) while avoiding to ground the entire biker.

Failure cases. While VideoMolmo demonstrates strong fine-grained pointing capabilities, it is not without limitations. As illustrated in Fig. 13, certain failure cases highlight areas for improvement. In the first row, the model is expected to point to the black harness but instead grounds a part of the adjacent bag. This misalignment stems from the limitations of SAM2, which struggles to accurately convert the predicted point coordinate into a meaningful mask. In the second row, the model points to only one of several visible paraglider lines, missing multiple lines. Such cases suggest a need for enhanced expressiveness, such as enabling the model to predict multiple points for a single query. Addressing these limitations opens new avenues for future work in improving the robustness and granularity of point grounding in complex scenes.

[Figure 30]

Point to everyone wearing a green shirt.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Point to the boat in front.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Point to the car in the front.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Point to the moving car.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Figure 8: VPoS-Bench qualitative examples.

[Figure 50]

Point to the horse running around.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Point to the cat going from right to left.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Point to the tigers circling around the bus.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Figure 9: MeVis qualitative examples.

[Figure 65]

Point to a yellow umbrella.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Point to a ball

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Point to a child walking to the bus with an adult

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Figure 10: Refer-YouTube-VOS qualitative examples.

[Figure 80]

Point to person at the back of the go-cart without a helmet.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Point to the helmet worn by the biker.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Point to a rope which the guy is hanging on.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Figure 11: Refer-DAVIS qualitative examples.

[Figure 95]

Point to who disturbs the dog from moving forward smoothly?

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Point to the vehicle that can accommodate more passengers.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Point to the musician playing drums.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Figure 12: ReasonVOS qualitative examples.

[Figure 110]

Point to a black harness with an airbag.

[Figure 111]

[Figure 112]

[Figure 113]

Point to the paraglider lines.

[Figure 114]

[Figure 115]

Figure 13: Qualitative failure cases of VIDEOMOLMO.

