## 3AM: 3egment Anything with Geometric Consistency in Videos

Yang-Che Sun1, Cheng Sun2, Chin-Yang Lin1, Fu-En Yang2, Min-Hung Chen2, Yen-Yu Lin1, and Yu-Lun Liu1

1 National Yang Ming Chiao Tung University, 2 NVIDIA Research jkarly.md11@nycu.edu.tw, yulunliu@cs.nycu.edu.tw

# arXiv:2601.08831v5[cs.CV]16Apr2026

Video / Unconstrained Photo Collection Consistent Tracking

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Mask Point Box

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

3AM

Fig. 1: Overview of 3AM. Given an input video or an unconstrained photo collection, 3AM takes a user-provided prompt, e.g., mask, point, or box, and produces a consistent object track across all views. Our method maintains cross-view correspondence even under large viewpoint changes, cluttered scenes, and variations in capture conditions, enabling robust object tracking from both videos and casual multi-view image sets.

Abstract. Video object segmentation methods like SAM2 achieve strong

performance through memory-based architectures but struggle under large viewpoint changes due to reliance on appearance features. Traditional 3D instance segmentation methods address viewpoint consistency but require camera poses, depth maps, and expensive preprocessing. We introduce 3AM, a training-time enhancement that integrates 3D-aware features from MUSt3R into SAM2. Our lightweight Feature Merger fuses multi-level MUSt3R features that encode implicit geometric correspondence. Combined with SAM2’s appearance features, the model achieves geometry-consistent recognition grounded in both spatial position and visual similarity. We propose a field-of-view aware sampling strategy ensuring frames observe spatially consistent object regions for reliable 3D correspondence learning. Critically, our method requires only RGB input at inference, with no camera poses or preprocessing. On challenging datasets with wide-baseline motion (ScanNet++, Replica), 3AM substantially outperforms SAM2 and extensions, achieving 90.6% IoU and 71.7% Tracking Recall on ScanNet++’s Selected Subset, improving over state-of-the-art VOS methods by +15.9 and +30.4 points. Project page: https://jayisaking.github.io/3AM-Page/

### 1 Introduction

Video object segmentation (VOS) identifies and tracks target objects throughout a video sequence with consistent masks across frames. This is fundamental

- (a) SAM2: loses track when viewpoint changes drastically
- (b) EmbodiedSAM: relys on camera poses and 3D mask merging
- (c) Ours: consistently tracks instances without 3D Ground Truth

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

| |
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- Fig. 2: Limitations of traditional VOS and 3D segmentation approaches, and an overview of our capability. (a) Traditional VOS methods such as SAM2 [63] lose track when the camera undergoes large viewpoint changes, causing masks to drift or disappear. (b) 3D segmentation approaches rely on accurate camera poses and explicit 3D mask merging; they often propagate errors when the 3D reconstruction is incomplete or noisy. (c) Our 3AM consistently tracks object instances across drastic viewpoint changes without requiring camera poses or 3D ground-truth masks, demonstrating robust cross-view correspondence purely from geometry-aware 2D tracking.

to autonomous driving, robotics, augmented reality, and video editing. A critical challenge is maintaining object identity under large viewpoint variations, temporary disappearances, or similar distractors. Traditional VOS methods rely on 2D appearance features and fail when the object appears dramatically different from varying viewpoints. For embodied AI in dynamic 3D environments, consistently recognizing objects across diverse viewpoints is essential for robust scene understanding.

Recent video object segmentation advances follow two parallel lines. First, 2D foundation models with memory-based architectures: SAM2 [63] introduced promptable segmentation with streaming memory and spatio-temporal attention. SAM2Long [18] employs memory trees for long-range identity preservation, while DAM4SAM [77] incorporates distractor-aware updates. However, purely 2D approaches fail under large viewpoint changes as appearance features cannot establish reliable correspondence. Second, 3D instance segmentation methods: proposal-centric approaches like Mask3D [65] and OneFormer3D [38] operate on point clouds, while projection-based methods like Open3DIS [54], SAM3D [102], and SAM2Object [116] lift 2D masks to 3D. These require camera poses, depth maps, preprocessing, and super-linear computational scaling.

Both paradigms have critical gaps (Fig. 2). 2D methods like SAM2 excel at efficiency but lack geometric awareness, failing under viewpoint variation (Fig. 2a). MOSEv2 [17] shows significant degradation under wide-baseline changes. Conversely, 3D methods achieve better consistency but require explicit 3D inputs (Fig. 2b) and struggle to generalize. 2D-to-3D lifting suffers

from view-inconsistent predictions [31,58,73,94,116]. End-to-end approaches like PanSt3R [123] require offline processing. Can we achieve 3D-aware, viewpointrobust segmentation while preserving promptability and efficiency, without explicit 3D supervision? (Fig. 2c)

We introduce 3AM (Fig. 1), a generalized 3D-aware tracker. Our key insight is that MUSt3R [7] encodes implicit geometric correspondence through features learned from multi-view consistency. By fusing multi-level MUSt3R features with SAM2’s appearance features via a lightweight Feature Merger, we enable geometry-consistent recognition based on spatial position, which at inference requires only RGB input and user prompts. We introduce a field-of-view aware sampling strategy ensuring frames observe spatially consistent object regions during training. On ScanNet++ [109] and Replica [69] with wide-baseline motion, 3AM achieves 90.6% IoU and 71.7% Positive IoU versus SAM2Long’s 74.7% and 41.3% (+15.9 and +30.4 points).

In summary, our contributions are:

- – A key observation on the gap between 2D tracking and 3D consistency. 2D tracking is fundamentally limited under large viewpoint variations. Existing 3D approaches require camera poses, depth, or 3D masks, often unavailable in practice. This motivates our geometry-aware framework, which achieves cross-view consistency without 3D supervision at inference.
- – A 3D-aware feature integration framework fusing geometric features from 3D reconstruction models with SAM2’s appearance features, and a field-of-view aware training strategy ensuring geometric consistency for effective 3D correspondence learning.
- – A generalized tracking framework that adaptively reverts to the original SAM2 for non-rigid or highly dynamic objects, along with extensive experimental evaluation demonstrating consistent improvements over stateof-the-art VOS methods on challenging datasets such as ScanNet++ and Replica, especially under object reappearance and significant viewpoint changes, while retaining competitive performance on conventional VOS benchmarks.

### 2 Related Work

- 2D Video Object Segmentation. Video object segmentation (VOS) segments and tracks target objects across video sequences with temporally coherent masks [8,51,59,78,87]. Early approaches relied on appearance propagation or optical flow but struggled with long-term consistency and occlusion [4,55,100]. Recent memory-based architectures leverage spatio-temporal attention and feature retrieval for improved object association [12–14,42,47,56,93,103–105], with subsequent work improving memory efficiency via restricted banks [118], gated linear matching [46], unified transformer frameworks [43,114], and dynamic anchor queries [120]. New benchmarks [26] and specialized settings such as panoramic video [97] and point-based prompting [49] further broaden the evaluation landscape.

SAM2 [63], a promptable model with streaming memory, achieves strong results across benchmarks. Follow-up efforts extend SAM2 with memory trees for

[Figure 37]

MUSt3R Enc Dec

[Figure 38]

[Figure 39]

|[Figure 40]| |
|---|---|
| | |

[Figure 41]

[Figure 42]

SAM2 Memories [0, 𝑖 − 1]

[Figure 43]

[Figure 44]

Memory Attention

[Figure 45]

Mask Decoder

[Figure 46]

Image Encoder

Memory Encoder

Feature Merger

𝑀 𝑀 𝑀

- Fig. 3: 3AM Pipeline Overview. Our Feature Merger fuses multi-level MUSt3R features, learned from multi-view consistency to encode implicit geometric correspondence, with SAM2’s appearance features via cross-attention and convolutional refinement. These merged geometry-aware representations then undergo memory attention with previous frames and mask decoding, enabling spatially-consistent object recognition that maintains identity across large viewpoint changes without requiring camera poses at inference.

long-range identity preservation [18], distractor-aware updates [77], on-device efficiency [117], text-driven understanding [15], and other improvements [98,101, 108]. Concurrent work further combines SAM2 with large language models for reasoning-based segmentation [2, 96, 112]. Despite these advances, large viewpoint changes, occlusion, and disappearance-reappearance events cause significant degradation [17, 61, 92]. Our method addresses these limitations by integrating 3D-aware representations into SAM2 for object-consistent segmentation under challenging viewpoint and appearance variations.

- 3D Instance Segmentation. Mainstream 3D instance segmentation follows two tracks. Proposal-centric methods predict instance masks from point clouds with learned proposals [25, 34, 38, 52, 60, 65, 71, 73, 80], with recent extensions adding promptable segmentation [121] and fast open-vocabulary recognition [6]. A complementary line lifts 2D results to 3D by projecting per-view masks and merging across views [3,5,27,33,35,36,41,54,76,79,90,94,102], employing superpoint affinity [110], zero-shot feature distillation [20,84], hierarchical contrastive learning [37, 111], and egocentric Gaussian lifting [24], but still depending on posed RGB-D for mask consolidation.

The rise of 3D Gaussian Splatting has further enabled semantic scene representations by distilling foundation-model features into Gaussians [57,62,88,119], assigning per-Gaussian identity encodings [107,113], solving 2D-to-3D label assignment optimally [66], and unifying lifting with segmentation end-to-end [122]. However, pure 2D promptable segmentation rarely suffices for 3D instances because cross-view consistency is not enforced, leading to view-inconsistent masks and fragmented instances [31,58,73,94,116]. Our approach instead imposes 3Daware self-consistency directly on video, preserving identity and geometry without 3D annotations or point-cloud mask merging.

End-to-End 3D-Aware Methods. Recent end-to-end 3D reconstruction models directly infer geometric structure from 2D inputs [7, 9, 11, 21, 39, 44, 48, 67, 70, 81, 83, 85, 86, 89, 99, 106]. The DUSt3R/MASt3R family has been extended to dynamic scenes [115], incremental reconstruction with spatial memory [82], feed-forward Gaussian splatting [68], efficient multi-view fusion [74], real-time dense SLAM [53], unconstrained structure-from-motion [19], and training-free

- 4D reconstruction [10]. Feed-forward models now also jointly predict geometry and semantics [23,32,72], with studies analyzing how different foundation models contribute complementary 3D capabilities [50].

This progress has fueled end-to-end 3D-aware segmentation frameworks [29, 30, 40]. PanSt3R [123] integrates 3D-aware features from MUSt3R [7] with a transformer decoder for joint geometry and panoptic segmentation via learnable quadratic fusion. However, it is incompatible with promptable backbones (e.g., SAM2), limiting interactivity, and requires offline access to entire sequences, preventing streaming.

- 3 Method

- 3.1 Problem Setting

Given an RGB image I and a user-provided prompt p (mask, box, or point), our goal is to identify the same object throughout a video sequence with consistent segmentation, avoiding false correspondences or identity switches across frames. Previous approaches often fail at this goal. 2D-based methods (e.g., SAM2 [63]) rely solely on appearance features, making them sensitive to viewpoint or distance changes and prone to confusion among visually similar objects [77], leading to frequent tracking failures over longer sequences. 3D-based methods (e.g., Mask3D [65], OpenYOLO3D [5], EmbodiedSAM [95]) localize objects using 3D mask proposals or dense 2D masks with camera poses, but require SfM or similar pipelines whose runtime and complexity grow rapidly with the number of frames, making them impractical for long sequences. One might instead apply a 3D segmentation model on predicted geometry (e.g., point clouds from MUSt3R). However, geometry quality, especially in dynamic scenes, limits downstream segmentation reliability. Moreover, current 3D segmentation models are mainly offline, unsuitable for real-time scenarios such as robotic navigation. More importantly, large-scale training data is far more abundant in the 2D video domain, and the SAM2 model we adopt provides a strong, promptable 2D prior difficult to match with current 3D alternatives. Our method leverages this advantage by using 3D information only as a lightweight association cue rather than a hard prerequisite, maintaining a 2D segmentation framework. Specifically, we encourage geometry-consistent feature matching, enabling the model to identify the same object by recognizing its correspondence in 3D space.

- 3.2 Architecture

The overall pipeline is shown in Fig. 3. For each video frame i, our model extracts two complementary feature streams that capture appearance cues and multi-view geometric consistency. First, the frame is processed by the SAM2 vision encoder, producing a 2D appearance feature map F2Di . In parallel, the same frame is passed into MUSt3R, which attends to its own multi-view memory through MUSt3R’s internal cross-attention mechanism, yielding a geometryaware feature map F3iD. Next, F2iD and F3iD are fused into a refined representation Fmergedi . The merged feature is processed by the memory attention module and the mask decoder, and is finally encoded by the memory encoder to produce

[Figure 47]

[Figure 48]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Feature Merger

- Fig. 4: Illustration of Features for Feature Merging. The heat map is computed using the cosine similarity between the red query point and the target frame. As illustrated in the lower row, vanilla SAM2 fails under large viewpoint changes. In contrast, as the MUSt3R feature hierarchy gradually shifts from semantic correspondence toward the point-cloud domain, we select intermediate layers to preserve both semantic relevance and geometric structure. By combining MUSt3R’s geometric cues with SAM2’s

visual semantics, the merged feature Fmerged provides a significantly more reliable localization of the target object.

memory features that will be referenced by future frames. We provide comparisons of different memory selection methods in Tab. 7.

Feature Merging. As shown in Fig. 4, SAM2 becomes unreliable when the object reappears or when the viewpoint changes drastically. The heat map in the lower row demonstrates that SAM2 cannot reliably associate the query point with its target under large angular differences, revealing its limited capacity to maintain object identity when the visual appearance changes substantially.

To address this issue, we enhance SAM2 with MUSt3R features F3iD. The upper row of Fig. 4 visualizes MUSt3R activations across depth. Early layers preserve a clearer correspondence pattern with the query point, while deeper layers produce more diffuse responses. This shift reflects MUSt3R’s progression toward point-cloud decoding: the representation becomes more geometry-oriented but less semantically aligned with the 2D query. Consequently, relying solely on either early or late layers is insufficient; early layers lack explicit geometric structure, whereas late layers lose semantic clarity. We therefore sample MUSt3R features from both early and late stages to capture complementary information.

These multi-level MUSt3R features are processed by our Feature Merger, which jointly performs cross-attention fusion and convolutional refinement. The cross-attention blocks first integrate the sampled MUSt3R layers into a single geometry-aware feature, aligning information from different depths. Specifically, as shown in Fig. 4, we take the MUSt3R encoder feature at the shallowest sampled depth, i.e., the most semantically rich layer, as the initial input to the Feature Merger. It first passes through a self-attention block to establish an initial semantic representation. The remaining sampled MUSt3R features are then incorporated one by one through a sequence of cross-attention layers, where the current merged representation acts as the query, and each additional MUSt3R feature provides the key–value pairs. In this way, the Feature Merger progressively integrates information from both shallow and deeper stages of MUSt3R. The output is then fed into a convolutional stage that merges it with SAM2’s

- 2D feature F2D for restoring fine spatial detail. We further compare different 3D

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

(a) Vanilla continuous sampling strategy

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

(b) Naive random sampling without considering field-of-view

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

(c) Field-of-view–aware sampling strategy (Ours)

- Fig. 5: Overview of our sampling strategy during training. (a) Continuous sampling provides densely spaced frames but offers limited viewpoint diversity. (b) Naïve random sampling introduces viewpoint variation but may select frames that observe spatially disjoint regions of the same object. For example, frame 0 shows the left side of the couch while frame 1 shows the right side. Because these regions are far apart in 3D space, treating them as the same supervisory signal forces the model to match inconsistent geometry and leads to ambiguous learning. (c) Our field-of-view–aware sampling retains only frames whose masked 3D points lie within the reference camera frustum over a threshold, ensuring consistent geometric overlap while preserving natural pose and occlusion variation.

foundation models in Tab. 8, and more architectural details and comparisons are provided in the supplementary materials.

The final merged feature, Fmerged, retains SAM2’s high-resolution, segmentation-

aware appearance cues while incorporating MUSt3R’s geometry-aware consistency, enabling robust object localization even under severe viewpoint changes.

#### 3.3 Training Frame Sampling

A key challenge in training our model is the limited capacity of the memory module (SAM2 maintains at most eight memory slots). To enable the model to learn robust object identification across diverse camera viewpoints, we must carefully design how training frames are sampled from each video sequence.

Naïve Random Sampling. A straightforward strategy is to randomly sample N frames from the video as in Fig. 5b. This exposes the model to a broad distribution of camera viewpoints and object appearances. Random sampling also naturally introduces distracting or visually similar objects in the scene , which encourages the model to rely on 3D-aware alignment rather than purely local 2D similarity.

Object Spanning Problem. However, naïve random sampling introduces a failure mode when an object occupies a large spatial extent. Consider a bed,

table, or cabinet: two randomly sampled frames may both contain the same instance, yet correspond to spatially distant regions (e.g., one frame at the headboard, the other at the footboard). Since our objective is to enforce 3D location consistency, which is to encourage the model to identify an object by recognizing that it lies in the nearby 3D position; thus, these wide-baseline views of a large object may contradict the intended training signal. Although the two frames belong to the same instance, their projected 3D locations are far apart, which can confuse the learning objective.

Field-of-View Sampling. For each training iteration, the first sampled frame is designated as the reference frame. To select the remaining N −1 frames, we apply a FOV-based filter using camera poses and depth. For every candidate frame, we back-project its object mask into 3D, transform the points into the reference frame, and re-project them onto the reference image. A frame is kept only if a sufficient fraction of its masked 3D points fall inside the reference camera frustum:

#{candidate masked points inside reference frustum} #{masked points in candidate frame}

> τ.

This ensures that selected frames observe overlapping physical regions of the object, avoiding degenerate cases where two views of the same instance correspond to distant, non-overlapping parts. We do not filter out partially occluded frames: frustum overlap reflects alignment in viewing direction, not full visibility, and retaining occlusion cases helps the model learn to differentiate viewpoint changes from true object absence.

Note that we apply FoV Sampling as a filter only when candidate masks contain an object. For no-object frames, which are also sampled during training, the model is simply required to predict target absence. In these cases, spatial distance does not affect the objective. That is, when views are far apart, the prediction is driven by geometric inconsistency, while for nearby views, it relies on appearance cues.

We describe our final sampling policy and its ablations in Sec. 4.

#### 3.4 Dynamic Object Fallback

An intuitive limitation of our method lies in dynamic objects, whose shape, location, or orientation may change over time. Such scenarios are prevalent in traditional VOS benchmarks [22,61,92]. Our approach relies heavily on relative appearance with respect to surrounding geometry (i.e., consistent geometrical locations across views). When an object undergoes significant motion, this assumption may break, potentially leading to tracking failure. To address this issue, we detect the presence of motion using [28] and revert to the original SAM2 pipeline when substantial motion is observed. Since the SAM2 image encoder remains frozen in our framework, we can reuse the visual feature and the additional computational overhead incurred by memory attention and mask decoder is negligible.

- Table 1: Datasets used for training. We apply FOV-aware sampling only when ground-truth geometry is available.

ScanNet++ [109] ASE [1] MOSE [16]

Real ✓ ✗ ✓ Dynamic ✗ ✗ ✓ FOV sampling prob 0.8 0.8 0.0 #Scenes / Videos 855 2612 1453

Empirically, we observed that dynamically toggling between 3AM and vanilla SAM2 mid-sequence (e.g., with a simple motion detection branch before mask decoder) corrupts the temporal memory bank, as the two pathways produce fundamentally different mask distributions. Therefore, we utilize a simple sequencelevel fallback to the original SAM2 pipeline when substantial motion is detected. This design choice is grounded in a practical categorization of tracking scenarios based on the object’s visibility with respect to the camera frustum: (1) Dynamic object with continuous visibility, where the object remains within the camera frustum and vanilla SAM2 performs well; (2) Static object with temporary disappearance, where the object leaves the camera frustum and later re-enters, resulting in a re-identification problem that 3AM is designed to address; and (3) Dynamic object with unobserved motion outside the frustum.

The third scenario represents an inherently ill-posed problem, as the object may move unpredictably while unobserved outside the camera frustum. Consequently, reliable tracking is fundamentally impossible. Since this extreme scenario cannot be effectively solved, engineering a complex mid-sequence fusion module is unnecessary. Hence a simple sequence-level switch is sufficient to cover the two tractable scenarios.

### 4 Experiments

Training Setup. We train our model for 1M iterations using the AdamW optimizer with a batch size of 1, where we only set the Memory Attention, the Mask Decoder, and the Feature Merger to trainable with learning rates of 5e-6, 5e-6, and 1e-5, respectively. All loss coefficients and memory frames, which are set to 8, follow the original SAM2 configuration.

Datasets. Intuitively, to equip the model with the ability to segment objects consistently across a scene, it is necessary to train on 3D-based datasets that record long video sequences covering diverse camera viewpoints, along with corresponding segmentation masks and RGB frames. However, the 2D masks in such

- 3D-based datasets are often projected from point cloud segmentations, which can be degraded due to the sparsity of the point cloud and projection errors.

To mitigate the imperfect 2D mask annotations common in 3D-based datasets,

we train on a hybrid mixture of synthetic, real, and video segmentation datasets. Specifically, the Aria Synthetic Dataset(ASE) [1] provides clean geometric supervision, ScanNet++ [109] offers realistic indoor 3D consistency, and MOSE [16] maintains the original temporally coherent masks. This combination enables balanced spatial and temporal learning. Note that to remain consistent with our

sampling strategy described in Sec. 3.3, we apply continuous sampling on MOSE, as it does not provide camera poses or depth maps. For ASE and ScanNet++, which include calibrated geometry, we adopt a mixed policy: continuous sampling with probability 0.2 and our FOV-aware sampling with probability 0.8. The FOV threshold τ is set to 0.25. To apply our sampling strategy, we first run MUSt3R inference on every scene in ScanNet++ and MOSE and store the resulting features. These precomputed features are then used during training to enable FOV-aware sampling. For continuous sampled batches, MUSt3R is executed on the fly.

- 4.1 2D Evaluation. To demonstrate the capability of our model, we evaluate its performance on

- 2D object tracking under challenging camera motion. Specifically, we focus on scenarios where the camera undergoes significant translation and rotation, and where objects may disappear and later reappear. Traditional VOS benchmarks such as LVOS [26], VOST [75], DAVIS [61], and YTOS [92] primarily assume a relatively fixed camera and stable scene surroundings, making them insufficient for testing robustness under large viewpoint changes.

To address this limitation, we use 3D-based datasets, ScanNet++ [109] and Replica [69], which naturally provide extensive camera trajectory variation due to their 3D reconstruction requirements. Their wide-baseline viewpoints make

- them suitable for evaluating consistency under severe pose changes. For each scene in both datasets, we use ground-truth camera poses to project 3D instance masks onto the 2D frames. We then select the frame with the largest visible mask area as the conditioning frame and perform forward and backward tracking from that reference.

Compared Methods. For comparison, we use SAM2 as the state-of-the-art VOS baseline. We also include the following works: SAM2Long [18] and DAM4SAM [77], which enhance SAM2 by improving its memory selection for more robust tracking. Note that our method, 3AM, uses the naive memory mechanism of the original SAM2 in all reported results unless specified otherwise.

Metrics. We use three complementary metrics to evaluate tracking quality. IoU is computed over all frames, including those where the object is absent, reflecting overall stability. Tracking Recall measures accuracy only on frames where a target object is present. Accuracy is further restricted to visible frames where the prediction has non-zero overlap with ground truth, capturing accuracy conditioned on successful localization. These metrics assess robustness to disappearance and accuracy when tracking succeeds.

ScanNet++. The results are shown in Tab. 2, and we provide visualization at Fig. 8. On the ScanNet++ dataset, we further evaluate performance on a Selected Subset specifically constructed to emphasize object reappearance. For each object track, we count the number of non-continuous visible segments, where a segment is defined as any contiguous span of visible frames whose length ℓ exceeds a minimal threshold ℓmin. Objects with frequent reappearance are typically

###### Table 2: Video Object Segmentation results on the ScanNet++ [109] and VOS datasets. Tracking Recall is computed over frames with object presence. Accuracy is computed over frames where a ground-truth object exists and the IoU of prediction and GT ̸= 0.

Mem ScanNet++ Whole Set ScanNet++ Selected Subset DAVIS17 LaSOT LaSOText DiDi G IoU ↑ Tracking Recall ↑ Accuracy ↑ IoU ↑ Tracking Recall ↑ Accuracy ↑ J&F ↑ AUC ↑ AUC ↑ Accuracy ↑

Method FPS

SAM2 [63] 14.4 3.6 0.4392 0.0235 0.0831 0.3397 0.0179 0.0395 89.0 70.0 56.9 0.720 SAM2Long [18] 7.1 6.8 0.8233 0.4166 0.6855 0.7474 0.4133 0.6382 88.3 73.9 - 0.719 DAM4SAM [77] 12.3 3.5 0.8205 0.4193 0.6783 0.7648 0.4356 0.6650 86.6 75.1 60.9 0.727 3AM 6.3 3.6 0.8898 0.5630 0.7155 0.9061 0.7168 0.7737 89.0 70.0 56.9 0.720

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

SAM2LongDAM4SAMSAM2Ours

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

- Fig. 6: Visual comparison of VOS methods. The leftmost frame is used as the conditioned frame and provides the reference mask.

associated with large viewpoint changes, since each disappearance–reappearance cycle often corresponds to observing the object from a substantially different angle or position. This subset, therefore, provides a focused evaluation of robustness under severe pose variation.

3AM achieves the best performance on both the Whole Set (IoU 0.8898, highest Recall and Accuracy) and the challenging Selected Subset (IoU 0.9061, Recall 0.7168, Accuracy 0.7737), highlighting its ability to maintain object identity even when the object reappears under large changes in viewpoint and spatial configuration. With a simple fallback mechanism, it also retains competitive results on conventional VOS benchmarks [22,61,77] as shown in Tab. 2.

Two-view Matching Comparison with SegMASt3R. We further compare our 3AM with SegMASt3R [30], a two-view matching approach built on MASt3R that associates segment masks between two images under large viewpoint changes, in Tab. 3. SegMASt3R is not directly prompable and does not natively support > 2 multiview tracking. Nevertheless, we include it as a strong two-view baseline by adopting the following protocol: we use the ground-truth initial mask as the source-frame candidate, and for each subsequent frame, we independently perform two-view mask matching against the source frame. To

###### Table 4: Video Object Segmentation results on the Replica [69] dataset.

Table 3: Quantitative comparison of twoview matching between 3AM and SegMASt3R on ScanNet++ Whole Set.

Method IoU ↑ Tracking Recall ↑ Accuracy ↑

SAM2 [63] 0.4424 0.1432 0.2188 SAM2Long [18] 0.7691 0.5195 0.6273 DAM4SAM [77] 0.7744 0.5135 0.6124 3AM (Ours) 0.8119 0.6381 0.6793

Method IoU ↑ Tracking Recall ↑ Accuracy ↑ SegMASt3R [30] 0.6800 0.3628 0.4053 3AM (Ours) 0.8915 0.5115 0.6405

fairly compare the two approaches, when tracking with our 3AM, we don’t use any memory other than the source frame.

Under this protocol, 3AM achieves IoU of 0.8915, Tracking Recall of 0.5115, and Accuracy of 0.6405, whereas SegMASt3R obtains 0.6800, 0.3628, and 0.4053, respectively. These results demonstrate 3AM not only provides stronger functionality, but also outperforms SegMASt3R even when restricted to two-view matching.

Replica. On the Replica dataset, 3AM achieves the best performance across all metrics (Tab. 4), reaching an IoU of 0.8119 and surpassing SAM2 (0.4424), SAM2Long (0.7691), and DAM4SAM (0.7744). It also attains the highest Tracking Recall (0.6381) and Accuracy (0.6793), demonstrating robust tracking under the large viewpoint variations characteristic of Replica.

These results demonstrate that 3AM consistently surpasses prior methods and is particularly effective in scenarios involving object reappearance and large viewpoint variation, despite not relying on the explicit memory-selection mechanisms used in previous approaches. Due to space limitations, more visualization is provided in the supplementary materials.

#### 4.2 3D Evaluation

We next evaluate our method on a 3D task. We adopt the class-agnostic 3D instance segmentation setting of ScanNet200, as it directly reflects a model’s ability to produce object-consistent predictions in 3D scenes. Prior approaches typically require explicit 3D fusion [95] or merging [102], regardless of whether the proposals originate from 2D masks [35] or 3D detectors [6, 54]. Our goal is to demonstrate that this additional 3D-space merging is unnecessary: if 2D tracking is geometrically consistent across views, then reliable 3D instances can be obtained simply by projecting the tracked 2D masks into 3D.

Concretely, we follow prior methods [116] and generate 2D proposals from SAM2 on keyframes selected by a stride-based sampling. These proposals are

- then propagated with 3AM, whose improved cross-view consistency enables stable object identification across multiple camera viewpoints. We perform merging using IoU and inter-mask precision to associate proposals over time, and project the resulting 2D tracks into 3D. In contrast, previous SAM2-based methods [116] rely heavily on 3D merging because their tracking often drifts or fails under large viewpoint changes. Further implementation details are provided in the supplementary material.

As shown in Table 5, 3AM achieves the highest performance among online methods, with an AP of 47.3 and strong AP50 and AP25 scores (59.7 and 75.3).

Table 5: Class-agnostic 3D instance segmentation results of different methods on ScanNet200 dataset.

Method Type 3D GT AP ↑ AP50 ↑ AP25 ↑

SAMPro3D [91] Offline Yes 18.0 32.8 56.1 Open3DIS [54] Offline Yes 34.6 43.1 48.5 SAI3D [110] Offline Yes 28.2 47.2 67.9 SAM2Object [116] Offline Yes 34.0 52.7 70.3

SAM3D [102] Online Yes 20.2 35.7 55.5 ESAM [95] Online Yes 42.2 63.7 79.6

3AM (Ours) Online No 47.3 59.7 75.3

##### Table 6: Component analysis on ScanNet++ selected subset.

Feature Source Feature Merger Sampling Strategy IoU Tracking Recall Accuracy

SAM2 & MUSt3R Addition & Convolution FoV Sampling 0.8771 0.6481 0.7149 SAM2 & MUSt3R Concatentation & Attention FoV Sampling 0.8831 0.6744 0.7462 SAM2 & MUSt3R Hierarchical Cross Attention (Final) Continuous Sampling 0.7925 0.6518 0.7513 SAM2 & MUSt3R Hierarchical Cross Attention (Final) Random Sampling 0.7363 0.5371 0.6311

MUSt3R - FoV Sampling 0.8461 0.5471 0.4129 SAM2 - FoV Sampling 0.6628 0.0038 0.0276 SAM2 & MUSt3R Hierarchical Cross Attention (Final) FoV Sampling 0.9061 0.7168 0.7737

These results highlight a central finding of our work: robust 3D instance segmentation can emerge from geometry-aware tracking without heavy 3D supervision.

#### 4.3 Ablation Study

Component Analysis. We provide ablation studies in Tab. 6 for Feature Source, Feature Merger Architecture, and Sampling Strategy. We evaluate variants using MUSt3R or SAM2 features alone under the same training protocol

- to demonstrate that both features are necessary. MUSt3R provides geometric grounding, while SAM2 captures appearance correspondence. Without MUSt3R, relying solely on appearance matching from SAM2 introduces ambiguity and leads to degraded performance. One thing to note is that SAM2 performs worse after finetuning. We attribute this degradation to the absence of grounding information from MUSt3R, which causes the memory attention to be mistrained due to insufficient reference cues for distinguishing different objects. Regarding the sampling strategy, naively applying random sampling results in worse performance than continuous sampling. When valid masks exist in both views, enforcing associations across large spatial discrepancies introduces ambiguity and leads to slower convergence.

Memory Selection. We further analyze the role of memory selection by combining 3AM with existing strategies proposed for SAM2. Table 7 reports results on the ScanNet++ Whole Set. 3AM, which uses the original SAM2 memoryselection mechanism, already achieves strong performance with an IoU of 0.8898. When we incorporate alternative memory-selection policies, such as those from DAM4SAM [77] or SAM2Long [18], we observe modest improvements: DAM4SAM3AM slightly increases Accuracy to 0.7204, while SAM2Long-3AM achieves

Table 7: Comparison of different memory selection methods on ScanNet++ Whole Set.

###### Table 8: Comparison of different 3D reconstruction backbones.

Method IoU ↑ Tracking Recall ↑ Accuracy ↑

###### Model Online Frame Num ScanNet++ Selected Subset Tracking Recall

VGGT × ∼200 – π3 × ∼300 – CUT3R ✓ > 10,000 0.2751 MUSt3R ✓ > 10,000 0.7168

3AM 0.8898 0.5630 0.7155 DAM4SAM-3AM [77] 0.8941 0.5471 0.7204 SAM2Long-3AM [18] 0.9004 0.5498 0.7361

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

3AM-MUSt3R3AM-CUT3R

| |
|---|

0 55

[Figure 107]

[Figure 108]

145 217

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

| |
|---|

0 55

[Figure 113]

[Figure 114]

145 217

- Fig. 7: Visual comparison of different 3D reconstruction backbones. (Top) CUT3R’s reconstruction lacks stable object alignment; the same table appears at inconsistent 3D locations. Such geometric instability weakens feature distinctiveness, making reliable tracking difficult. (Bottom) In contrast, MUSt3R provides coherent and stable object alignment across viewpoints, yielding features that preserve object identity and enable robust tracking

the highest IoU (0.9004) and Accuracy (0.7361). These results suggest that 3AM already delivers strong and stable performance without relying on specialized memory-selection policies. While alternative strategies such as those from SAM2Long or DAM4SAM offer small additional gains, their impact is relatively limited compared to the improvements brought by 3AM itself. We view the design of memory-selection mechanisms tailored specifically for 3AM as an interesting direction for future work.

- 3D Foundataion Models. We compare 3D foundation models in Table 8. VGGT and π3 operate offline on frame batches, unsuitable for online tracking. CUT3R supports online operation, but yields limited instance-level alignment as shown in Fig. 7, reflected by its Tracking Recall of 0.2751 on the ScanNet++ Selected Subset. MUSt3R is fully online with substantially stronger object alignment across viewpoints. This alignment is essential: consistent 3D alignment enables stable 2D cross-view correspondence, allowing reliable mask propagation without explicit 3D merging.

### 5 Conclusion

We introduce 3AM, integrating 3D-aware features into SAM2 for viewpointrobust video object segmentation. Through our Feature Merger and field-ofview aware sampling, 3AM achieves geometry-consistent tracking requiring only RGB at inference. On wide-baseline datasets, 3AM substantially outperforms state-of-the-art VOS methods, achieving gains of +15.9 and +30.4 points on the ScanNet++ Selected Subset, while automatically switching back to SAM2’s capability to handle dynamic video scenarios.

### Acknowledgements

This work was supported by NVIDIA Taiwan AI Research & Development Center (TRDC). This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628E-A49-023-. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

- 1. Avetisyan, A., Xie, C., Howard-Jenkins, H., Yang, T.Y., Aroudj, S., Patra, S., Zhang, F., Frost, D., Holland, L., Orme, C., et al.: Scenescript: Reconstructing scenes with an autoregressive structured language model. In: European Conference on Computer Vision. pp. 247–263. Springer (2024)
- 2. Bai, Z., He, T., Mei, H., Wang, P., Gao, Z., Chen, J., Zhang, Z., Shou, M.Z.: One token to seg them all: Language instructed reasoning segmentation in videos. Advances in Neural Information Processing Systems 37, 6833–6859 (2024)
- 3. Bautista, M.A., Guo, P., Abnar, S., Talbott, W., Toshev, A., Chen, Z., Dinh, L., Zhai, S., Goh, H., Ulbricht, D., et al.: Gaudi: A neural architect for immersive 3d scene generation. Advances in Neural Information Processing Systems 35, 25102– 25116 (2022)
- 4. Bhat, G., Lawin, F.J., Danelljan, M., Robinson, A., Felsberg, M., Van Gool, L., Timofte, R.: Learning what to learn for video object segmentation. In: European Conference on Computer Vision. pp. 777–794. Springer (2020)
- 5. Boudjoghra, M.E.A., Dai, A., Lahoud, J., Cholakkal, H., Anwer, R.M., Khan, S., Khan, F.S.: Open-yolo 3d: Towards fast and accurate open-vocabulary 3d instance segmentation. arXiv preprint arXiv:2406.02548 (2024)
- 6. Boudjoghra, M.E.A., Dai, A., Lahoud, J., Cholakkal, H., Anwer, R.M., Khan, S., Khan, F.S.: Open-YOLO 3d: Towards fast and accurate open-vocabulary 3d instance segmentation. In: The Thirteenth International Conference on Learning Representations (2025)
- 7. Cabon, Y., Stoffl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1050–1060 (2025)
- 8. Caelles, S., Maninis, K.K., Pont-Tuset, J., Leal-Taixé, L., Cremers, D., Van Gool, L.: One-shot video object segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 221–230 (2017)

- 9. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19457–19467 (2024)
- 10. Chen, X., Chen, Y., Xiu, Y., Geiger, A., Chen, A.: Easi3r: Estimating disentangled motion from dust3r without training. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9158–9168 (2025)
- 11. Chen, Z., Qin, M., Yuan, T., Liu, Z., Zhao, H.: Long3r: Long sequence streaming 3d reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5273–5284 (2025)
- 12. Cheng, H.K., Oh, S.W., Price, B., Lee, J.Y., Schwing, A.: Putting the object back into video object segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3151–3161 (2024)
- 13. Cheng, H.K., Schwing, A.G.: Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model. In: European conference on computer vision. pp. 640–658. Springer (2022)
- 14. Cheng, H.K., Tai, Y.W., Tang, C.K.: Rethinking space-time networks with improved memory coverage for efficient video object segmentation. Advances in neural information processing systems 34, 11781–11794 (2021)
- 15. Cuttano, C., Trivigno, G., Rosi, G., Masone, C., Averta, G.: Samwise: Infusing wisdom in sam2 for text-driven video segmentation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 3395–3405 (2025)
- 16. Ding, H., Liu, C., He, S., Jiang, X., Torr, P.H., Bai, S.: MOSE: A new dataset for video object segmentation in complex scenes. In: ICCV (2023)
- 17. Ding, H., Ying, K., Liu, C., He, S., Jiang, X., Jiang, Y.G., Torr, P.H., Bai, S.: Mosev2: A more challenging dataset for video object segmentation in complex scenes. arXiv preprint arXiv:2508.05630 (2025)
- 18. Ding, S., Qian, R., Dong, X., Zhang, P., Zang, Y., Cao, Y., Guo, Y., Lin, D., Wang, J.: Sam2long: Enhancing sam 2 for long video segmentation with a training-free memory tree. arXiv preprint arXiv:2410.16268 (2024)
- 19. Duisterhof, B.P., Zust, L., Weinzaepfel, P., Leroy, V., Cabon, Y., Revaud, J.: Mast3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. In: 2025 International Conference on 3D Vision (3DV). pp. 1–10. IEEE (2025)
- 20. Engelmann, F., Manhardt, F., Niemeyer, M., Tateno, K., Pollefeys, M., Tombari, F.: Opennerf: Open set 3d neural scene segmentation with pixel-wise features and rendered novel views. arXiv preprint arXiv:2404.03650 (2024)
- 21. Fan, C.D., Chang, C.W., Liu, Y.R., Lee, J.Y., Huang, J.L., Tseng, Y.C., Liu, Y.L.: Spectromotion: Dynamic 3d reconstruction of specular scenes. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21328–21338

(2025)

- 22. Fan, H., Lin, L., Yang, F., Chu, P., Deng, G., Yu, S., Bai, H., Xu, Y., Liao, C., Ling, H.: Lasot: A high-quality benchmark for large-scale single object tracking. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5374–5383 (2019)
- 23. Fan, Z., Zhang, J., Cong, W., Wang, P., Li, R., Wen, K., Zhou, S., Kadambi, A., Wang, Z., Xu, D., et al.: Large spatial model: End-to-end unposed images to semantic 3d. Advances in neural information processing systems 37, 40212–40229

(2024)

- 24. Gu, Q., Lv, Z., Frost, D., Green, S., Straub, J., Sweeney, C.: Egolifter: Open-world 3d segmentation for egocentric perception. In: European conference on computer vision. pp. 382–400. Springer (2024)

- 25. Han, L., Zheng, T., Xu, L., Fang, L.: Occuseg: Occupancy-aware 3d instance segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2940–2949 (2020)
- 26. Hong, L., Chen, W., Liu, Z., Zhang, W., Guo, P., Chen, Z., Zhang, W.: Lvos: A benchmark for long-term video object segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13480–13492

(2023)

- 27. Hsu, P.H., Zhang, K., Wang, F.E., Tu, T., Li, M.F., Liu, Y.L., Chen, A.Y., Sun, M., Kuo, C.H.: Openm3d: Open vocabulary multi-view indoor 3d object detection without human annotations. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8688–8698 (2025)
- 28. Huang, N., Zheng, W., Xu, C., Keutzer, K., Zhang, S., Kanazawa, A., Wang, Q.: Segment any motion in videos. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 3406–3416 (June 2025)
- 29. Jain, A., Katara, P., Gkanatsios, N., Harley, A.W., Sarch, G., Aggarwal, K., Chaudhary, V., Fragkiadaki, K.: Odin: A single model for 2d and 3d segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3564–3574 (2024)
- 30. Jayanti, R., Agrawal, S., Garg, V., Tourani, S., Khan, M.H., Garg, S., Krishna, M.: SegMASt3r: Geometry grounded segment matching. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https: //openreview.net/forum?id=DI2AAFnLrc
- 31. Jia, B., Chen, Y., Yu, H., Wang, Y., Niu, X., Liu, T., Li, Q., Huang, S.: Sceneverse: Scaling 3d vision-language learning for grounded scene understanding. In: European Conference on Computer Vision. pp. 289–310. Springer (2024)
- 32. Jiang, H., Liu, L., Cheng, T., Wang, X., Lin, T., Su, Z., Liu, W., Wang, X.: Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11960–11970 (2025)
- 33. Jiang, L., Shi, S., Schiele, B.: Open-vocabulary 3d semantic segmentation with foundation models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21284–21294 (2024)
- 34. Jiang, L., Zhao, H., Shi, S., Liu, S., Fu, C.W., Jia, J.: Pointgroup: Dual-set point grouping for 3d instance segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and Pattern recognition. pp. 4867–4876 (2020)
- 35. Jung, S., Zheng, J., Zhang, K., Qiao, N., Chen, A.Y., Xia, L., Liu, C., Sun, Y., Zeng, X., Huang, H.W., et al.: Details matter for indoor open-vocabulary 3d instance segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9627–9637 (2025)
- 36. Kerr, J., Kim, C.M., Goldberg, K., Kanazawa, A., Tancik, M.: Lerf: Language embedded radiance fields. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 19729–19739 (2023)
- 37. Kim, C.M., Wu, M., Kerr, J., Goldberg, K., Tancik, M., Kanazawa, A.: Garfield: Group anything with radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21530–21539 (2024)
- 38. Kolodiazhnyi, M., Vorontsova, A., Konushin, A., Rukhovich, D.: Oneformer3d: One transformer for unified point cloud segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20943– 20953 (2024)
- 39. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: European Conference on Computer Vision. pp. 71–91. Springer (2024)

- 40. Li, H., Qu, J., Zhang, L.: Ovseg3r: Learn open-vocabulary instance segmentation from 2d via 3d reconstruction. arXiv preprint arXiv:2509.23541 (2025)
- 41. Li, M.F., Ku, Y.F., Yen, H.X., Liu, C., Liu, Y.L., Chen, A.Y., Kuo, C.H., Sun, M.: Genrc: Generative 3d room completion from sparse image collections. In: European Conference on Computer Vision. pp. 146–163. Springer (2024)
- 42. Li, M., Li, S., Zhang, X., Zhang, L.: Univs: Unified and universal video segmentation with prompts as queries. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3227–3238 (2024)
- 43. Li, W., Guo, P., Zhou, X., Hong, L., He, Y., Zheng, X., Zhang, W., Zhang, W.: Onevos: unifying video object segmentation with all-in-one transformer framework. In: European Conference on Computer Vision. pp. 20–40. Springer (2024)
- 44. Lin, C.Y., Sun, C., Yang, F.E., Chen, M.H., Lin, Y.Y., Liu, Y.L.: Longsplat: Robust unposed 3d gaussian splatting for casual long videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27412–27422

(2025)

- 45. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2117–2125 (2017)
- 46. Liu, Q., Wang, J., Yang, Z., Li, L., Lin, K., Niethammer, M., Wang, L.: Livos: Light video object segmentation with gated linear matching. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 8668–8678 (2025)
- 47. Liu, Y., Yu, R., Yin, F., Zhao, X., Zhao, W., Xia, W., Yang, Y.: Learning qualityaware dynamic memory for video object segmentation. In: European Conference on Computer Vision. pp. 468–486. Springer (2022)
- 48. Liu, Y.L., Gao, C., Meuleman, A., Tseng, H.Y., Saraf, A., Kim, C., Chuang, Y.Y., Kopf, J., Huang, J.B.: Robust dynamic radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13–23 (2023)
- 49. Mahadevan, S., Zulfikar, I.E., Voigtlaender, P., Leibe, B.: Point-vos: Pointing up video object segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22217–22226 (2024)
- 50. Man, Y., Zheng, S., Bao, Z., Hebert, M., Gui, L., Wang, Y.X.: Lexicon3d: Probing visual foundation models for complex 3d scene understanding. Advances in Neural Information Processing Systems 37, 76819–76847 (2024)
- 51. Maninis, K.K., Caelles, S., Chen, Y., Pont-Tuset, J., Leal-Taixé, L., Cremers, D., Van Gool, L.: Video object segmentation without temporal information. IEEE transactions on pattern analysis and machine intelligence 41(6), 1515–1530 (2018)
- 52. Misra, I., Girdhar, R., Joulin, A.: An end-to-end transformer model for 3d object detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2906–2917 (2021)
- 53. Murai, R., Dexheimer, E., Davison, A.J.: Mast3r-slam: Real-time dense slam with 3d reconstruction priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16695–16705 (2025)
- 54. Nguyen, P., Ngo, T.D., Kalogerakis, E., Gan, C., Tran, A., Pham, C., Nguyen, K.: Open3dis: Open-vocabulary 3d instance segmentation with 2d mask guidance. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4018–4028 (2024)
- 55. Oh, S.W., Lee, J.Y., Xu, N., Kim, S.J.: Video object segmentation using spacetime memory networks. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9226–9235 (2019)

- 56. Park, K., Woo, S., Oh, S.W., Kweon, I.S., Lee, J.Y.: Per-clip video object segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1352–1361 (2022)
- 57. Peng, Q., Planche, B., Gao, Z., Zheng, M., Choudhuri, A., Chen, T., Chen, C., Wu, Z.: 3d vision-language gaussian splatting. arXiv preprint arXiv:2410.07577

(2024)

- 58. Peng, S., Genova, K., Jiang, C., Tagliasacchi, A., Pollefeys, M., Funkhouser, T., et al.: Openscene: 3d scene understanding with open vocabularies. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 815–824 (2023)
- 59. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., SorkineHornung, A.: A benchmark dataset and evaluation methodology for video object segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 724–732 (2016)
- 60. Piekenbrinck, J., Schmidt, C., Hermans, A., Vaskevicius, N., Linder, T., Leibe, B.: Opensplat3d: Open-vocabulary 3d instance segmentation using gaussian splatting. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5246–5255 (2025)
- 61. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbeláez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 davis challenge on video object segmentation. arXiv:1704.00675 (2017)
- 62. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: Langsplat: 3d language gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20051–20060 (2024)
- 63. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024), https://arxiv. org/abs/2408.00714
- 64. Ryali, C., Hu, Y.T., Bolya, D., Wei, C., Fan, H., Huang, P.Y., Aggarwal, V., Chowdhury, A., Poursaeed, O., Hoffman, J., Malik, J., Li, Y., Feichtenhofer, C.: Hiera: A hierarchical vision transformer without the bells-and-whistles. ICML

(2023)

- 65. Schult, J., Engelmann, F., Hermans, A., Litany, O., Tang, S., Leibe, B.: Mask3d: Mask transformer for 3d semantic instance segmentation. arXiv preprint arXiv:2210.03105 (2022)
- 66. Shen, Q., Yang, X., Wang, X.: Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally. In: European Conference on Computer Vision. pp. 456–472. Springer (2024)
- 67. Shih, M.L., Chen, Y.H., Liu, Y.L., Curless, B.: Prior-enhanced gaussian splatting for dynamic scene reconstruction from casual video. In: Proceedings of the SIGGRAPH Asia 2025 Conference Papers. pp. 1–13 (2025)
- 68. Smart, B., Zheng, C., Laina, I., Prisacariu, V.A.: Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912 (2024)
- 69. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., Mur-Artal, R., Ren, C., Verma, S., Clarkson, A., Yan, M., Budge, B., Yan, Y., Pan, X., Yon, J., Zou, Y., Leon, K., Carter, N., Briales, J., Gillingham, T., Mueggler, E., Pesqueira, L., Savva, M., Batra, D., Strasdat, H.M., Nardi, R.D., Goesele, M., Lovegrove, S., Newcombe, R.: The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019)

- 70. Su, C.H., Hu, C.Y., Tsai, S.R., Lee, J.Y., Lin, C.Y., Liu, Y.L.: Boostmvsnerfs: Boosting mvs-based nerfs to generalizable view synthesis in large-scale scenes. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–12 (2024)
- 71. Sun, J., Qing, C., Tan, J., Xu, X.: Superpoint transformer for 3d scene instance segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 2393–2401 (2023)
- 72. Sun, X., Jiang, H., Liu, L., Nam, S., Kang, G., Wang, X., Sui, W., Su, Z., Liu, W., Wang, X., et al.: Uni3r: Unified 3d reconstruction and semantic understanding via generalizable gaussian splatting from unposed multi-view images. arXiv preprint arXiv:2508.03643 (2025)
- 73. Takmaz, A., Fedele, E., Sumner, R.W., Pollefeys, M., Tombari, F., Engelmann, F.: Openmask3d: Open-vocabulary 3d instance segmentation. arXiv preprint arXiv:2306.13631 (2023)
- 74. Tang, Z., Fan, Y., Wang, D., Xu, H., Ranjan, R., Schwing, A., Yan, Z.: Mvdust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5283–5293 (2025)
- 75. Tokmakov, P., Li, J., Gaidon, A.: Breaking the" object" in video object segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22836–22845 (2023)
- 76. Tu, T., Chuang, S.P., Liu, Y.L., Sun, C., Zhang, K., Roy, D., Kuo, C.H., Sun, M.: Imgeonet: Image-induced geometry-aware voxel representation for multi-view 3d object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6996–7007 (2023)
- 77. Videnovic, J., Lukezic, A., Kristan, M.: A distractor-aware memory for visual object tracking with sam2. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24255–24264 (2025)
- 78. Voigtlaender, P., Chai, Y., Schroff, F., Adam, H., Leibe, B., Chen, L.C.: Feelvos: Fast end-to-end embedding learning for video object segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9481–9490 (2019)
- 79. Vora, S., Radwan, N., Greff, K., Meyer, H., Genova, K., Sajjadi, M.S., Pot, E., Tagliasacchi, A., Duckworth, D.: Nesf: Neural semantic fields for generalizable semantic segmentation of 3d scenes. arXiv preprint arXiv:2111.13260 (2021)
- 80. Vu, T., Kim, K., Luu, T.M., Nguyen, T., Yoo, C.D.: Softgroup for 3d instance segmentation on point clouds. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2708–2717 (2022)
- 81. Wang, H., Agapito, L.: 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 (2024)
- 82. Wang, H., Agapito, L.: 3d reconstruction with spatial memory. In: 2025 International Conference on 3D Vision (3DV). pp. 78–89. IEEE (2025)
- 83. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)
- 84. Wang, P., Fan, Z., Wang, Z., Su, H., Ramamoorthi, R., et al.: Lift3d: Zero-shot lifting of any 2d vision model to 3d. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21367–21377 (2024)
- 85. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10510–10522 (2025)

- 86. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20697–20709 (2024)
- 87. Wu, C.H., Chen, S.H., Hu, C.Y., Wu, H.Y., Chen, K.H., Chen, Y.Y., Su, C.H., Lee, C.K., Liu, Y.L.: Denver: Deformable neural vessel representations for unsupervised video vessel segmentation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15682–15692 (2025)
- 88. Wu, Y., Meng, J., Li, H., Wu, C., Shi, Y., Cheng, X., Zhao, C., Feng, H., Ding, E., Wang, J., et al.: Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems 37, 19114–19138 (2024)
- 89. Wu, Y., Zheng, W., Zhou, J., Lu, J.: Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863 (2025)
- 90. Xu, D., Jiang, Y., Wang, P., Fan, Z., Wang, Y., Wang, Z.: Neurallift-360: Lifting an in-the-wild 2d photo to a 3d object with 360deg views. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4479–4489 (2023)
- 91. Xu, M., Yin, X., Qiu, L., Liu, Y., Tong, X., Han, X.: Sampro3d: Locating sam prompts in 3d for zero-shot instance segmentation. In: 2025 International Conference on 3D Vision (3DV). pp. 1222–1232. IEEE (2025)
- 92. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327 (2018)
- 93. Xu, X., Wang, J., Li, X., Lu, Y.: Reliable propagation-correction modulation for video object segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 2946–2954 (2022)
- 94. Xu, X., Chen, H., Zhao, L., Wang, Z., Zhou, J., Lu, J.: Embodiedsam: Online segment any 3d thing in real time. arXiv preprint arXiv:2408.11811 (2024)
- 95. Xu, X., Chen, H., Zhao, L., Wang, Z., Zhou, J., Lu, J.: Embodiedsam: Online segment any 3d thing in real time. arXiv preprint arXiv:2408.11811 (2024)
- 96. Yan, C., Wang, H., Yan, S., Jiang, X., Hu, Y., Kang, G., Xie, W., Gavves, E.: Visa: Reasoning video object segmentation via large language models. In: European Conference on Computer Vision. pp. 98–115. Springer (2024)
- 97. Yan, S., Xu, X., Zhang, R., Hong, L., Chen, W., Zhang, W., Zhang, W.: Panovos: Bridging non-panoramic and panoramic views with transformer for video segmentation. In: European Conference on Computer Vision. pp. 346–365. Springer

(2024)

- 98. Yang, C.Y., Huang, H.W., Chai, W., Jiang, Z., Hwang, J.N.: Samurai: Adapting segment anything model for zero-shot visual tracking with motion-aware memory. arXiv preprint arXiv:2411.11922 (2024)
- 99. Yang, J., Sax, A., Liang, K.J., Henaff, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21924–21935 (2025)
- 100. Yang, L., Fan, Y., Xu, N.: Video instance segmentation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5188–5197 (2019)
- 101. Yang, Q., Yao, Y., Cui, M., Bo, L.: Mosam: Motion-guided segment anything model with spatial-temporal memory selection. arXiv preprint arXiv:2505.00739

(2025)

- 102. Yang, Y., Wu, X., He, T., Zhao, H., Liu, X.: Sam3d: Segment anything in 3d scenes. arXiv preprint arXiv:2306.03908 (2023)

- 103. Yang, Z., Miao, J., Wei, Y., Wang, W., Wang, X., Yang, Y.: Scalable video object segmentation with identification mechanism. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(9), 6247–6262 (2024)
- 104. Yang, Z., Wei, Y., Yang, Y.: Associating objects with transformers for video object segmentation. Advances in Neural Information Processing Systems 34, 2491–2502

(2021)

- 105. Yang, Z., Yang, Y.: Decoupling features in hierarchical propagation for video object segmentation. Advances in Neural Information Processing Systems 35, 36324–36336 (2022)
- 106. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207 (2024)
- 107. Ye, M., Danelljan, M., Yu, F., Ke, L.: Gaussian grouping: Segment and edit anything in 3d scenes. In: European conference on computer vision. pp. 162–179. Springer (2024)
- 108. Ye, M., Oh, S.W., Ke, L., Lee, J.Y.: Entitysam: Segment everything in video. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24234–24243 (2025)
- 109. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023)
- 110. Yin, Y., Liu, Y., Xiao, Y., Cohen-Or, D., Huang, J., Chen, B.: Sai3d: Segment any instance in 3d scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3292–3302 (2024)
- 111. Ying, H., Yin, Y., Zhang, J., Wang, F., Yu, T., Huang, R., Fang, L.: Omniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20612–20622 (2024)
- 112. Yuan, H., Li, X., Zhang, T., Sun, Y., Huang, Z., Xu, S., Ji, S., Tong, Y., Qi, L., Feng, J., et al.: Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001 (2025)
- 113. Zhai, H., Li, H., Li, Z., Pan, X., He, Y., Zhang, G.: Panogs: Gaussian-based panoptic segmentation for 3d open vocabulary scene understanding. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 14114–14124

(2025)

- 114. Zhang, J., Cui, Y., Wu, G., Wang, L.: Jointformer: A unified framework with joint modeling for video object segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)
- 115. Zhang, J., Herrmann, C., Hur, J., Jampani, V., Darrell, T., Cole, F., Sun, D., Yang, M.H.: Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825 (2024)
- 116. Zhao, J., Zhuo, J., Chen, J., Ma, H.: Sam2object: Consolidating view consistency via sam2 for zero-shot 3d instance segmentation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19325–19334 (2025)
- 117. Zhou, C., Zhu, C., Xiong, Y., Suri, S., Xiao, F., Wu, L., Krishnamoorthi, R., Dai, B., Loy, C.C., Chandra, V., et al.: Edgetam: On-device track anything model. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 13832–13842 (2025)
- 118. Zhou, J., Pang, Z., Wang, Y.X.: Rmem: Restricted memory banks improve video object segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18602–18611 (2024)

- 119. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., Kadambi, A.: Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21676–21685 (2024)
- 120. Zhou, Y., Zhang, T., Ji, S., Yan, S., Li, X.: Improving video segmentation via dynamic anchor queries. In: European Conference on Computer Vision. pp. 446–

463. Springer (2024)

- 121. Zhou, Y., Gu, J., Chiang, T.Y., Xiang, F., Su, H.: Point-sam: Promptable 3d segmentation model for point clouds. arXiv preprint arXiv:2406.17741 (2024)
- 122. Zhu, R., Qiu, S., Liu, Z., Hui, K.H., Wu, Q., Heng, P.A., Fu, C.W.: Rethinking end-to-end 2d to 3d scene segmentation in gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3656–3665 (2025)
- 123. Zust, L., Cabon, Y., Marrie, J., Antsfeld, L., Chidlovskii, B., Revaud, J., Csurka, G.: Panst3r: Multi-view consistent panoptic segmentation. arXiv preprint arXiv:2506.21348 (2025)

### A Overview

This supplementary material presents additional results to complement the main manuscript. First, we provide all the implementation details, including network architectures in Sec. B. Next, we elaborate on training details, such as the FoV sampling and datasets in Sec. C. Then, we describe how we perform classagnostic instance segmentation in Sec. D. Finally, we provide more qualitative comparisons on both video object tracking and class-agnostic instance segmentation in Sec. E. In addition to this document, we provide additional video results to compare with state-of-the-art methods.

### B Implementation Details

Through all experiments, we use SAM 2.1-Large as our baseline due to computational resource constraints. A Hiera [64] image encoder produces multi-scale features, where stride-16 and stride-32 outputs (Stages 3/4) are fused via an FPN [45] with convolutions to obtain frame embeddings, while shallow features (stride 4/8) from Stages 1/2 are injected only into the mask decoder to recover fine boundaries. Memory attention uses sinusoidal absolute positional embeddings together with 2D RoPE; object pointer tokens do not use RoPE, and we adopt L = 4 layers. The prompt encoder follows SAM, and the mask decoder uses the mask token as the object pointer stored in the memory bank. An additional token predicts object visibility via an MLP head, and occluded frames receive a learned occlusion embedding in memory. As in SAM 2, the decoder outputs multiple mask candidates and selects the one with the highest predicted IoU when ambiguity persists. The memory encoder reuses Hiera features without a separate backbone; memory features are projected to 64 dimensions, and the 256-dim pointer is split into four 64-d tokens for cross-attention. For multi-object videos, the image encoder is shared while each object maintains an independent memory bank and decoder.

#### B.1 Feature Merger

As illustrated in Fig. 4, our method uses multi-layer features from the MUSt3R decoder. We keep MUSt3R’s memory mechanism unchanged—i.e., we do not modify its view-coverage–based memory selection. We empirically select the layer indices [encoder,4,7,11] to incorporate both semantic cues from early layers and geometry-aware signals from later layers, as deeper decoder layers in MUSt3R increasingly capture 3D structure and directly decode point clouds.

After selecting the MUSt3R features, we merge them using a sequence of cross-attention operations. We first construct positional embeddings: PE3D is obtained by processing the point map and ray map produced by MUSt3R, while

- PE2D is taken from MUSt3R’s 2D positional encoding. The encoder feature is projected from 1024 to 768 channels using a 1×1 convolution, added with

- PE3D, and then passed through a self-attention layer with PE2D to form an initial coarse feature Fcoarse.

For each selected MUSt3R layer Fi with i ∈ {4,7,11}, we update Fcoarse using:

Fcoarse ← FFN CrossAttn SelfAttn(Fcoarse + PE3D), Fi + PE3D .

(1)

We omit normalization and skip connections for clarity.

After the attention fusion, Fcoarse is upsampled and refined by a 3×3 convolution, then concatenated with the Hiera image-encoder feature F2D from SAM 2, followed by an additional convolution block to match the feature dimensions. The resulting feature Fmerged is used for memory attention and memory encoding. We leave the shallow Hiera features from Stages 1 and 2 untouched so that the mask decoder can still rely on strong low-level visual cues for producing high-resolution segmentation outputs.

### C Training Details

#### C.1 Field-of-View Sampling

We do not use 100% FOV-aware sampling because we observe that applying FOV filtering to every batch degrades the model’s original feature-matching ability inherited from SAM2. When all training samples come from drastically different viewpoints, the model is over-regularized toward cross-view matching and loses its within-view correspondence skills, leading to a form of feature collapse.

#### C.2 Settings

Loss We follow SAM2’s loss design without modification. Specifically, mask prediction is supervised using a weighted combination of focal loss (20) and dice loss (1); the IoU head is trained with an ℓ1 loss (1); and the occlusion prediction head uses a cross-entropy loss (1).

Prompts Due to the degraded 2D mask and the large viewpoint variations introduced by the sampling strategy during training, we restrict the input modality to masks only for ScanNet++ and ASE. For MOSE, we enable all prompt types, including point, box, and mask inputs.

#### C.3 Datasets

Aria Synthetic Environments (ASE) [1] ASE is a large-scale synthetic dataset of 100,000 procedurally generated indoor scenes rendered with simulated Aria-glass sensors. Each scene contains realistic 3D object layouts, simulated trajectories, and aligned 2D/3D annotations. ASE enables large-scale training of 3D scene understanding, object detection, and tracking, particularly in data-hungry settings where real-world labelled data is scarce. We sample a total of 2,612 scenes whose

number of views falls within a practical range: scenes with too few views do not provide sufficient viewpoint variation for training, while scenes with excessively many views lead to memory overflow when running MUSt3R.

MOSE [16] The MOSE dataset (Complex Video Object Segmentation) targets video-object segmentation in cluttered and occluded real-world scenes. It comprises 2,149 video clips with 5,200 objects from 36 categories and 431,725 high-quality masks. Unlike previous VOS benchmarks that feature large, salient, isolated targets, MOSE features heavy occlusion, object disappearance and reappearance, and small/inconspicuous objects. MOSE is primarily used to preserve the core VOS capability of our model. While our geometric integration encourages strong cross-view consistency, we do not want the model to over-rely on geometry and “hallucinate” object tracks without sufficient visual evidence; MOSE helps anchor the model to appearance-driven cues.

Although MUSt3R supports dynamic scenes, we find that training on highly dynamic and diverse datasets such as SA-V Train leads to instability in practice. MOSE provides a more controlled level of motion and scene variation, allowing us to maintain stable learning while still leveraging dynamic-object supervision.

ScanNet++ [109] ScanNet++ is a high-quality indoor RGB–D reconstruction dataset that provides accurate camera poses, dense trajectories, and detailed geometric annotations. The public release contains over 1,000 reconstructed scenes, representing an expanded version of the initial smaller release [109]. Its combination of reliable geometry and viewpoint diversity makes it well suited for studying view-consistent object tracking under large camera motions. However, the 2D masks in ScanNet++ are generated by projecting 3D point-cloud instance labels into the image plane, which inevitably introduces reprojection noise from depth and pose inaccuracies. Besides, the pose annotations in ScanNet++ often omit long stretches of consecutive frames, as the SfM pipeline discards uncertain frames. Under such conditions, our FoV-based sampling becomes even more important. Despite this, ScanNet++ remains valuable because it provides realistic indoor scenes with rich geometric structure and large viewpoint variation. Together with the other two datasets, which supply clean, frame-aligned 2D masks, this forms a complementary training combination that balances realistic geometry with high-quality appearance supervision.

### D Class-Agnostic Instance Segmentation

For class-agnostic 3D instance segmentation, we proceed as follows. Given a set of keyframes, either sampled at a fixed interval or selected using a view-coverage criterion, we first generate 2D instance masks using SAM2’s automatic mask generator. For each keyframe i, we then propagate its instance masks forward to all subsequent frames using our model; no backward tracking is performed. The propagated masks are lifted into 3D by back-projecting their pixels using the depth map and camera pose. A practical issue is the presence of reprojection errors near object boundaries, caused by depth noise and view-dependent

misalignment. To mitigate this, we (1) compute a reprojection score using the agreement between projected depth and ground-truth depth, and (2) erode the

- 2D masks slightly before projection, which reduces artifacts concentrated near object edges.

After obtaining the per-frame 3D fragments, we merge them into consolidated

- 3D instances. Two scores are computed for every fragment: (1) a 3D overlap score in the point-cloud domain, and (2) a 2D temporal-overlap score based on the propagated masks along the video. Because our model maintains consistent object identity across the video, an object visible earlier will be recognized again whenever it reappears, providing reliable temporal evidence for merging.

Finally, after merging, duplicate assignments are resolved by majority voting at the superpoint level: each superpoint is assigned to the instance in which it is observed the greatest number of times.

### E Qualitative Comparison

Video Comparisoin We provide video object tracking visualization in Figs. 8 to 14.

Class-Agnostic Instance Segmentation. We provide additional visual results on class-agnostic instance segmentation in Fig. 15.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

SAM2LongDAM4SAMSAM2Ours

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

##### Fig. 8: Visual comparison of different VOS methods

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

##### Fig. 15: Visual results on class-agnostic instance segmentation.

