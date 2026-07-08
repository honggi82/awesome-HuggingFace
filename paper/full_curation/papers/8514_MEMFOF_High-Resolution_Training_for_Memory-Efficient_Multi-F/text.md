arXiv:2506.23151v1[cs.CV]29Jun2025

# MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation

Vladislav Bargatin1 Egor Chistov1 Alexander Yakovenko1,2 Dmitriy Vatolin1,2

1Lomonosov Moscow State University, Moscow, Russia 2MSU Institute for Artificial Intelligence, Moscow, Russia

{vladislav.bargatin, egor.chistov, alexander.yakovenko, dmitriy}@graphics.cs.msu.ru

### Abstract

Recent advances in optical flow estimation have prioritized accuracy at the cost of growing GPU memory consumption, particularly for high-resolution (FullHD) inputs. We introduce MEMFOF, a memory-efficient multi-frame optical flow method that identifies a favorable trade-off between multi-frame estimation and GPU memory usage. Notably, MEMFOF requires only 2.09 GB of GPU memory at runtime for 1080p inputs, and 28.5 GB during training, which uniquely positions our method to be trained at native 1080p without the need for cropping or downsampling.

We systematically revisit design choices from RAFT-like architectures, integrating reduced correlation volumes and high-resolution training protocols alongside multi-frame estimation, to achieve state-of-the-art performance across multiple benchmarks while substantially reducing memory overhead. Our method outperforms more resourceintensive alternatives in both accuracy and runtime efficiency, validating its robustness for flow estimation at high resolutions. At the time of submission, our method ranks first on the Spring benchmark with a 1-pixel (1px) outlier rate of 3.289, leads Sintel (clean) with an endpoint error (EPE) of 0.963, and achieves the best Fl-all error on KITTI-2015 at 2.94%. The code is available at: https: //github.com/msu-video-group/memfof.

### 1. Introduction

Optical flow (the dense per-pixel motion between frames) estimation is a fundamental task in low-level vision with wide applications from video action recognition and object detection [22, 29, 38] to video restoration and synthesis [3, 11, 16, 37]. Traditional variational methods formulated flow as an optimization problem [9, 17], but often struggled with large motions and real-time performance. The deep learning era sparked a leap in both accuracy and processing speed: FlowNet [5] pioneered this shift, with PWC-Net [28]

MEMFOF is memory efficient

MEMFOF outperforms The SotA on Spring

24

10

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | |Fl|ow|Ne|t2|RA|FT| | |
| | | | | | | | |M|em|Fl|ow|
| | | | | | | | | |PK|N|et|
| | |M|EM|FO|F| | | | | | |
| | | | | | | | | | | | |
| |4.|06×|le|ss|me|mo|ry| | | | |
| | |fo|r in|fe|ren|ce| | | | | |
| | | | | | | | | | | | |

Zero-Shot1pxError(%)

Two-frame

MemoryUsage(GB)

19.0

20

Multi-frame

8

16

6

StreamFlow

12

4

8.0 8.1 8.5

8

4.2

2

4

2.1

0

0 2 4 6 8 10 12

MEMFOFFlowNet2RAFTMemFlowRPKNetStreamFlow

Memory Usage (GB)

Figure 1. Comparison with state-of-the-art optical flow methods. Left: Quality-memory trade-off on the Spring [18] benchmark. MEMFOF demonstrates superior memory efficiency and the lowest zero-shot error among all methods. Right: GPU memory consumption for 1080p resolution inputs. MEMFOF outperforms state-of-the-art methods such as RPKNet [20] and StreamFlow [30]. For more details please see Tab. 2. StreamFlow [30] is omitted from the left plot due to space constraints.

introducing cost volume warping for efficiency. RAFT [31] later established state-of-the-art accuracy through iterative GRU-based refinement of a 4D all-pairs correlation volume. However, RAFT’s quadratic memory scaling with image size creates prohibitive costs for high-resolution inference (e.g., 8 GB at FullHD and 25+ GB at WQHD), forcing input downsampling that degrades motion boundary details.

Recent advances address these limitations through two complementary strategies: (1) enhancing correlation efficiency and (2) exploiting multi-frame temporal cues. Memory-efficient architectures reduce correlation costs via sparse candidate matching (SCV) [14], 1D motion decomposition (Flow1D) [35], or hybrid volumes (HCV) [39]. Transformer-based methods like GMFlow [36] and FlowFormer++ [27] enable global feature matching with fewer iterations. Multi-frame approaches such as VideoFlow [26] and MemFlow [4] leverage temporal consistency to resolve

occlusions, while StreamFlow [30] optimizes spatiotemporal processing efficiency.

Despite this progress, high-resolution benchmarks like Spring [18] remain a challenge. Some methods either downsample inputs [32] or employ tiling strategies [33] to reduce memory consumption, trading it for less accuracy or longer inference. And methods operating at native resolutions tend to use large amounts of memory, prohibiting their use on consumer grade hardware, see Fig. 1 for details.

In this work, we propose MEMFOF, the first multiframe optical flow method designed for memory efficiency at FullHD. MEMFOF can be trained and run on full 1080p frames without any downsampling or tiling, using only a few GB of memory at inference – all while achieving stateof-the-art accuracy. To achieve this, we extend SEA-RAFT, a two-frame optical flow architecture to incorporate a threeframe strategy. Crucially, we adjust the RAFT-style architecture to drastically cut memory usage (about 4× lower, down to just 2.09 GB) while enabling multi-frame input, allowing our model to run at 1080p on common GPUs. Which in turn allows for training at native 1080p using under 32GB of memory.

For better handling of large motions found at high resolutions, we devise a training regime that overcomes the mismatch between standard optical flow datasets (often limited in image size and motion magnitude) and the FullHD domain by upscaling existing datasets and training at higher resolutions, see Figure 4. An ablation study shows that this upsampling is critical to avoid underfitting on large-motion regions, leading to consistent performance gains on real high-resolution benchmarks. Notably, our method ranks first at zero-shot evaluation on the Spring benchmark, surpassing all other published work (both zero-shot and finetuned in Spring). To the best of our knowledge, we are the first to address the issues of memory consumption of multiframe methods at high-resolutions in a principled manner.

In summary, our key contributions are:

- • Memory-Efficient Multi-Frame Design. We propose a refined multi-frame RAFT-style architecture that processes FullHD inputs natively, reducing GPU memory needs by up to 3.9× compared to RAFT / SEA-RAFT, requiring only ∼2 GB of GPU memory at 1080p inference, well within the capacity of consumer-grade GPUs.
- • High-Resolution Training Strategy. A novel FullHDcentric data augmentation and multi-stage learning approach to accurately capture large motions, preventing the underfitting that commonly arises when transferring from low-resolution to high-resolution tasks.
- • State-of-the-Art Results on Multiple Benchmarks. MEMFOF achieves top accuracy on multiple benchmarks with substantially lower memory overhead. It leads on Spring [18], KITTI-2015 [19], and Sintel [2].

### 2. Related Works

Optical flow estimation is a fundamental problem in computer vision, with applications ranging from motion analysis to video compression. Over the years, various approaches have been proposed to address the challenges of accuracy and efficiency. In this section, we review the existing literature, categorizing it into three main areas: twoframe optical flow, multi-frame optical flow, and memoryefficient optical flow.

Two-frame optical flow. Classical approaches [6, 9, 17] optimize an energy function combining similarity and smoothness terms. With the advent of deep learning, methods like FlowNet [5] revolutionized the field by leveraging convolutional neural networks to directly predict optical flow from image pairs. PWC-Net [28] then introduced a pyramid, warping, and cost volume mechanism.

More recently, RAFT [31] has introduced a new paradigm, employing an iterative refinement process and an all-pair correlation volume. Building on RAFT’s success, several variants have been proposed to improve its efficiency and accuracy. One strategy is to introduce global receptive fields via transformers or attention. GMFlow [36] treats optical flow as a global feature matching problem, while FlowFormer [10] integrates a transformer into the cost volume processing. Beyond transformers, GMA [13] introduces global motion attention to focus the iterative updates on important regions. On the other hand, SEARAFT [32] aims to enhance RAFT by three simple tricks: using a mixture of Laplace loss, directly regressing initial flow, and pre-training on a rigid-flow dataset.

Unfortunately, all RAFT-like methods require substantial memory resources on high-resolution inputs. As a result, they are often applied to downscaled frames or with a tiling-based approach, compromising the quality of the estimated flow by losing fine details or global motion context, respectively.

Multi-frame optical flow. While two-frame methods have advanced significantly, they inherently ignore the rich temporal information available in video streams. Early multiframe attempts simply extended two-frame methods with flow propagation, for example, by fusing the backward warped past flow with current flow through a fusion module, as in PWC-Fusion [23], or by using a “warm start” initialization where the previous frame’s flow is used to initialize the next estimation, as in RAFT. Recent research has moved beyond pairwise estimation by explicitly modeling sequences of frames. VideoFlow [26] introduces a tri-frame optical flow (TROF) module to estimate forward and backward flows from a center frame to its neighboring frames. Multiple TROF modules can then be connected via a motion propagation module to extend to longer video sequences. Another approach, MemFlow [4], augments a RAFT-like architecture with a memory buffer that car-

Cached and reused

[Figure 1]

[Figure 2]

###### Image

ft→t−1

Feature extractor

It−1 ×N

###### 1080p

|[Figure 3]|
|---|

###### <·,·>

Correlation volume

Recurrentblock

[Figure 4]

Feature extractor

###### It

[Figure 5]

StreamFlow

|[Figure 6]|
|---|

18.97GB

<·,·>

Correlation volume

[Figure 7]

Feature extractor

It+1

ft→t+1

Hidden state

[Figure 8]

MEMFOF (ours)

| | |
|---|---|
| | |

[Figure 9]

[Figure 10]

2.09GB

[Figure 11]

Context network

Context features

[Figure 12]

[Figure 13]

- Figure 2. Overview of our method and FullHD inference results. Left: Outline of MEMFOF: when operating on videos we cache and reuse results of the feature extraction stage and correlation volume calculation. For each new frame we extract features and run the context network on the frame triplet, which returns the initial flow estimates, context features and hidden (recurrent) state. The flows are recurrently updated for N iterations and finally upsampled to get the final predictions. Right: Comparison of our method (MEMFOF) with StreamFlow [30] on FullHD images from the DAVIS dataset [21]. Our method correctly captures the tennis ball’s movement while requiring much less memory.

### 3. Method

ries forward motion features. StreamFlow [30] proposes a streamlined pipeline that processes multiple frames in one forward pass, avoiding redundant calculations of feature maps and correlation volumes. Unfortunately, all three of these approaches do not address the inherent limitations of the cost volume framework’s large memory consumption on modern high-resolution videos.

Our method introduces a novel approach to optical flow estimation that combines memory efficiency with multi-frame processing without sacrificing accuracy. The method consists of three key components: (1) extending SEA-RAFT to three frames, (2) resolution reduction of the correlation volume, and (3) performance optimization techniques. Below, we describe each component in detail.

Memory-efficient optical flow. Memory efficiency has become a critical concern in optical flow estimation since the introduction of RAFT. Methods like Flow1D [35] and MeFlow [34] have explored low-dimensional representations of the correlation volume. Similarly, Sparse Cost Volume (SCV) [14] restricts the correspondence search of RAFT to a few top matches. On the other hand, Deep Inverse Patchmatch (DIP) [40] uses a PatchMatch [1]-based approach to avoid building the all-pairs correlation volume. While these approaches achieve notable improvements in efficiency, they often sacrifice accuracy, falling short of the performance achieved by state-of-the-art methods in the RAFT family. This trade-off between memory efficiency and accuracy highlights the need for novel approaches that can bridge the gap between these competing objectives.

#### 3.1. Extending SEA-RAFT to three frames

To leverage temporal information, we extend the two-frame SEA-RAFT architecture to three frames. Following VideoFlow [26], we predict bidirectional flows, one between the current frame and the previous frame, and another between the current frame and the next frame. This involves calculating two correlation volumes instead of one. The update block is also modified to refine both flows at the same time, enabling the network to capture long-range dependencies. Similar to SEA-RAFT, to predict the initial flow, we pass all three frames into the context network. We will now formalize our method.

Approach. Given three consecutive frames It−1,It,It+1, we iteratively estimate a sequence of bidirectional flows f0,f1,...,fN ∈ (RH×W×2,RH×W×2); where N indicates the number of iterative refinements; fk includes a flow to the previous frame ftk→t−1 and a flow to the next frame

Notably, there has been little work that effectively applies memory-efficient techniques to multi-frame optical flow estimation. In this work, we address this gap by proposing a novel method that enables high-quality optical flow estimation without excessive memory demands.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Reference Frame Crop MemFlow StreamFlow SEA RAFT  M  MEMFOF  Ours)

[Figure 24]

| |
|---|

[Figure 25]

| |
|---|

[Figure 26]

- Figure 3. Qualitative comparison of MemFlow [4], StreamFlow [30], SEA-RAFT [32], and our method on Spring benchmark [18] crops, colorbar represents endpoint error. Our approach surpasses prior methods and demonstrates that: (1) multi-frame processing enhances temporal coherence, and (2) native Full HD resolution preserves local and global motion details. Crops are sourced from official leaderboard submissions.

ftk→t+1; H and W are the reduced height and width of the input images. We begin by extracting the input frame fea-

ture maps Ft,Ft−1,Ft+1 ∈ RH×W×D

f. To get the initial prediction f0, the hidden state h0 ∈ RH×W×D

c, and the context features g ∈ RH×W×D

c, we pass all three frames into the context network:

g,h0 = ContextNetwork(It−1,It,It+1), (1) f0 = FlowHead(h0). (2)

The dual correlation volumes Ct,t−1 and Ct,t+1 are computed as:

Ct,t−1(u,v) = ⟨Ft(u),Ft−1(v)⟩, (3) Ct,t+1(u,v) = ⟨Ft(u),Ft+1(v)⟩, (4)

where ⟨·,·⟩ denotes the dot product.

Iterative refinement. The correlation values ckt→t−1 and ckt→t+1 are retrieved from the dual correlation volumes based on the current flow predictions:

ckt→t−1 = LookUp(Ct,t−1,ftk→t−1), (5) ckt→t+1 = LookUp(Ct,t+1,ftk→t+1). (6)

These values are then fused and encoded into correlation and flow features, which are in turn transformed into a bidi-

rectional motion feature Fmk :

Fcorrk = CorrEncoder(ckt→t−1,ckt→t+1), (7) Fflowk = FlowEncoder(ftk→t−1,ftk→t+1), (8)

Fmk = MotionEncoder(Fcorrk ,Fflowk ). (9)

The hidden state hk is updated iteratively using the motion feature, context features g, and previous hidden state:

##### hk+1 = Updater(Fmk ,g,hk), (10)

and the residual flows ∆fk are decoded from the updated hidden state:

∆fk = FlowHead(hk+1). (11) The flow predictions are refined as:

ftk→+1t−1 = ftk→t−1 + ∆ftk→t−1, (12) ftk→+1t+1 = ftk→t+1 + ∆ftk→t+1. (13)

The final flow predictions are convexly upsampled to the input resolution as in RAFT.

#### 3.2. Resolution reduction of the correlation volume

A major bottleneck in modern optical flow methods, such as RAFT and SEA-RAFT, is the memory consumption of the correlation volume, which scales quadratically with the input resolution as O((HW)2). To address this, we propose

- Table 1. Details of our training procedure. Dataset abbreviations: T: Things, S: Sintel, K: KITTI-2015, H: HD1K. Following SEA-RAFT, the dataset distribution for the TSKH stage is S(.32), T(.31), K(.12), H(.24). N indicates the number of iterative refinements used in our method during training. Memory usage is stated per GPU.

Stage Weights Datasets Scale Crop size N Learning rate Batch size Steps Memory (GB)

TartanAir - TartanAir 2x [480, 960] 4 1.4e-4 64 225k 12.0 Things TartanAir T 2x [864, 1920] 4 7e-5 32 120k 17.1 TSKH Things T+S+K+H 2x [864, 1920] 4 7e-5 32 225k 17.1

Sintel-ft TSKH S 2x [872, 1920] 8 3e-5 32 12.5k 22.8 KITTI-ft TSKH K 2x [750, 1920] 8 3e-5 32 2.5k 19.6 Spring-ft TSKH Spring 1x [1080, 1920] 8 4.8e-5 32 60k 28.5

reducing the resolutions of the correlation volumes and the working flow predictions to 1/16 of the input frames, compared to the standard 1/8 resolution.

Our three-frame setup benefits from this reduction, decreasing the memory footprint for two correlation volumes from 10.4 GB to just 0.65 GB. While other components (e.g., feature maps and intermediate activations) also contribute to memory usage, preventing a sixteen-fold reduction in overall consumption, the total memory usage remains significantly lower than that of the original two-frame SEA-RAFT (8.19 GB vs. 2.09 GB for FullHD).

To account for the correlation volume size reduction, we adapt the ResNet34 [8] backbone used in SEA-RAFT. Specifically, to get 1/16 resolution features, we apply a strided convolution on the original 1/8 resolution feature maps. Additionally, to account for more information being stored in each pixel, we increase the feature map dimension Df from 256 to 1024 and the update block dimension Dc from 128 to 512.

This reduction in memory usage enables training our method in native FullHD, alleviating the need for cropping or downsampling of inputs. Memory consumption during different training stages can be seen in Table 1.

#### 3.3. Performance optimization techniques

To further enhance motion coherence, we reintroduce the GMA module [13]. To better adapt to different resolutions, similar to MemFlow [4], we modify the scale factor in attention from 1/√Dc to log3 (HW)/√Dc.

We additionally apply three inference-time speed and memory optimizations. Firstly, similar to StreamFlow [30], we note that when optical flow needs to be predicted for a video sequence, already calculated feature maps can be reused for future predictions. Secondly, following Flow1D [35], we use convex upsampling only on the last predictions. And finally, we reuse the previously computed correlation volume Ct,t+1 for overlapping frame pairs when moving to the next frame in video sequence, instead of recomputing it from scratch.

### 4. Experiments

We evaluate our method on three popular optical flow benchmarks: Spring [18] (modern high-resolution sequences), Sintel [2] (synthetic scenes with complex motion) and KITTI [19] (autonomous driving).

#### 4.1. Training Details

We follow the SEA-RAFT training protocol with some adjustments. We train our method on 32 A100 GPUs with automatic mixed precision. Our main changes with respect to SEA-RAFT are skipping FlyingChairs [5] due to its two-frame limitation, 2× upsampled frames and flows on datasets other than Spring, and in turn larger crop sizes. Training details are provided in Table 1. In cases when the crop size is bigger than the frame size or is not a multiple of 16, we pad the images with black pixels. Training our main model on all stages takes from 3 to 4 days.

Evaluation metrics. We adopt widely used metrics from established benchmarks [7, 18, 24] in this study: endpoint error (EPE), 1-pixel outlier rate (1px), Fl-score, and WAUC error. The 1px outlier rate measures the percentage of pixels where the flow error exceeds 1 pixel. The endpoint error (EPE) is defined as the average Euclidean distance between predicted and ground truth flow vectors. The Fl-score measures the percentage of pixels where the disparity or flow exceeds 3 pixels and 5% of its true value. Finally, the WAUC metric evaluates the inlier rates for a range of thresholds, from 0 to 5 px, and integrates these rates, giving higher weight to lower-threshold rates. Please refer to the supplementary for a formal definition of WAUC.

Mixture-of-Laplace Loss. Following SEA-RAFT [32], we use a mixture-of-Laplace (MoL) loss instead of an L1 loss. The MoL loss for T optical flow frame predictions with N iterative refinements is defined as:

T

N

1 T

γN−kLt,kMoL, (14)

L =

t=1

k=0

where Lt,kMoL is the MoL loss for the t-th optical flow frame prediction after k refinement iterations and γ is set to 0.85

- Table 2. Benchmark comparison of optical flow methods. Results are sourced from official leaderboard of the Spring benchmark, where minus (”-”) indicates the method has no published results. Speed (runtime) and peak GPU memory consumption were measured on a Nvidia RTX 3090 GPU (24 GB) without automatic mixed precision or memory efficient correlation volumes. Lower values are better (↓) except for WAUC (↑). The best results are indicated in bold, second-best are underlined. Method configurations are taken from submissions to the Spring benchmark if present, and from submissions to the Sintel benchmark otherwise.

Inference Cost (1080p) Spring (test) Memory, GB Runtime, ms 1px ↓ EPE ↓ Fl ↓ WAUC ↑

Method #Frames

Flow1D [35] 2 1.34 405 - - - MeFlow [34] 2 1.32 1028 - - - PWC-Net [28] 2 1.41 76 82.265 2.288 4.889 45.670 FlowNet2 [12] 2 4.16 167 6.710 1.040 2.823 90.907 RAFT [31] 2 7.97 557 6.790 1.476 3.198 90.920 GMA [13] 2 13.26 1185 7.074 0.914 3.079 90.722 FlowFormer [10] 2 OOM - 6.510 0.723 2.384 91.679 RPKNet [20] 2 8.49 295 4.809 0.657 1.756 92.638

NOFINE-TUNE

VideoFlow-BOF [26] 3 17.74 1648 - - - VideoFlow-MOF [26] 5 OOM - - - - MemFlow [4] 3 8.08 885 5.759 0.627 2.114 92.253 StreamFlow [30] 4 18.97 1403 5.215 0.606 1.856 93.253 MEMFOF (Ours) 3 2.09 472 3.600 0.432 1.353 94.481

CrocoFlow [33] 2 2.01 6524 4.565 0.498 1.508 93.660 SEA-RAFT (S) [32] 2 8.15 205 3.904 0.377 1.389 94.182 SEA-RAFT (M) [32] 2 8.19 286 3.686 0.363 1.347 94.534

FINE-TUNE

MemFlow [4] 3 8.08 885 4.482 0.471 1.416 93.855 StreamFlow [30] 4 18.97 1403 4.152 0.467 1.424 94.404 MEMFOF (Ours) 3 2.09 472 3.289 0.355 1.238 95.186

to add higher weights on later predictions following RAFT. Please refer to the supplementary for more details.

#### 4.2. Results

We will now state our results on established public benchmarks.

Results on Spring. Our approach fine-tunes on and processes native 1080p sequences, which allows it to preserve fine motion details as shown in Figure 3. This enables stateof-the-art accuracy — we outperform SEA-RAFT (M) by 10% in 1px outlier rate and 2% in EPE (Table 2). Additionally, our upsampled pre-train strategy also places us first among all non-fine-tuned submissions, even outperforming the fine-tuned SEA-RAFT (M) by 2.3% on the 1px metric. Crucially, our memory efficiency allows three-frame temporal processing at native 1080p even with a low memory budget, and our method is faster than other multi-frame competitors.

Results on Sintel and KITTI. Due to pre-training on 2x upsampled frames, for submissions to the Sintel and KITTI benchmarks, we bilinearly upscale all input images by a factor of two and bilinearly downscale all resulting flow maps by a factor of two. For Sintel submissions we use 16 update

iterations. Our method leads on Sintel clean split, surpassing the five-frame version of VideoFlow and outperforms SEA-RAFT (L) by 27% on the final pass (Table 3). On the KITTI benchmark, we achieve state-of-the-art performance among all non-scene flow methods. Please refer to the supplementary material for visual and zero-shot comparisons with other methods.

#### 4.3. Ablation Study

The ablation study is conducted on the Spring training set (only on the forward left 4K flow), as we mainly focus on FullHD performance. If not otherwise stated, we use the same training procedure and hyperparameters as in the experiments section — models after the TSKH stage but before Spring fine-tuning, and perform 8 iterative refinements. High-Resolution Training Analysis. Commonly used optical flow datasets come in relatively small resolutions, and methods trained on such data often generalize poorly to motion magnitudes seen in high-resolution inputs. This limits the practical use of optical flow methods, causing input downsampling to a resolution that better matches the training stage [15]. See Fig. 4 for motion vector histogram which illustrates this discrepancy between common datasets and

- Table 3. Evaluation of our method on the Sintel and KITTI-15 public benchmarks. The Sintel benchmark uses EPE as it’s metric for both splits, while KITTI-15 uses the Fl-all outliers metric.

Sintel KITTI-15 Clean ↓ Final ↓ Fl-all ↓

Method

PWC-Net [28] 3.86 5.04 9.60 FlowNet2 [12] 4.16 5.74 10.41 Flow1D [35] 2.238 3.806 6.27 MeFlow [34] 2.054 3.090 4.95 RAFT [31] 1.609 2.855 5.10 GMA [13] 1.388 2.470 5.15 SEA-RAFT (M) [32] 1.442 2.865 4.64 SEA-RAFT (L) [32] 1.309 2.601 4.30 FlowFormer [10] 1.159 2.088 4.68 RPKNet [20] 1.315 2.657 4.64 CrocoFlow [33] 1.092 2.436 3.64 DDVM [25] 1.754 2.475 3.26

StreamFlow [30] 1.041 1.874 4.24 MemFlow [4] 1.046 1.914 4.10 MemFlow-T [4] 1.081 1.840 3.88 VideoFlow-BOF [26] 1.005 1.713 4.44 VideoFlow-MOF [26] 0.991 1.649 3.65 MEMFOF (Ours) 0.963 1.907 2.94

Spring FullHD motion range. Plese refer to the supplementary for details on histogram creation.

We evaluate three strategies to bridge resolution gaps during training, see Table 4 for detailed metrics:

- • Native Resolution: Training on original data yields the worst performance (EPE: 0.430), as low-res motion magnitudes mismatch FullHD. We additionally test predicting the flow at half the resolution (like in SEA-RAFT [32]), which helps improve EPE at large displacements (s40+) but hinders the methods ability to predicts fine motions as shown by all the other metrics.
- • Upsampled (2×) with Crops: Training on upsampled data cropped to original resolution helps improve the quality but performs worse than full-frame training, likely due to cropped context limiting very large motion learning.
- • Upsampled (2×) Full Frames: This achieves the best FullHD results (EPE: 0.341), as full-frame upsampled training optimally aligns motion distributions with highres inference.

Three-Frame Flow Estimation Strategy. We compare bidirectional (current-to-previous & current-to-next) and unidirectional (previous-to-current & current-to-next) flow estimation. As shown in Table 4, bidirectional training improves EPE by 14.75% on Spring train data. We posit this stems from simplified motion boundary learning: bidi-

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

(a) TartanAir (b) FlyingThings (c) KITTI-2015 (d) HD1K

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

(e) Sintel (f) Spring (g) Combined at 1× resolution

(h) Combined at 2× resolution

Figure 4. We analyze motion patterns in optical flow datasets using 2D histograms. Each histogram uses the same bins, covering all possible motions at FullHD resolution. Color intensity corresponds to the number of motion vectors in each bin. Borders show the maximum motion range in each dataset. (a – e) Training datasets histograms at their native resolutions. (f) Motion histogram of the Spring training set, note the large motions, not covered by any of the datasets. (g – h) Combined motion histograms of training datasets without and with 2× upsampling.

rectional flows share consistent boundaries of the central frame, whereas unidirectional flows face distinct boundaries for each direction, which makes the task of predicting initial flow much harder for the context network.

Correlation volume resolution, feature dimension, and number of frames. We carefully examine the trade-off between the resolution of the correlation volume and the number of frames. We train 2-, 3- and 5- frame models with either 1/16 or 1/24 correlation volume resolution. We also ablate the use of GMA module alongside the hidden state / feature dimensions. The results are reported in Table 5. We see consistent gains when increasing the feature dimension and a favorable trade-off between 2- and 3-frame models. We also note the performance degradation when moving from 3 to 5 frames which we attribute to the insufficient size of the context network and recursive module, but leave further analysis to future work.

Inference-Time Optimizations. Due to the slidingwindow processing of video sequences and iterative nature of RAFT-derived methods we note a few calculation redundancies that can be used to further improve runtime.

- • Feature Network Reuse: When processing sequential data, parts of the computations may be reused. Notably in a three frame scenario we may reuse 2 out of 3 feature network results. We implement this by caching extracted features and only running the feature network on the new frame. This optimization cannot be applied to the context network as it takes all three frames as input.
- • Late Convex Upsampling: During training all flow predictions are upsampled for loss computation. However, during inference, we need to apply convex upsampling only on the final iteration, eliminating redundant computations.

- Table 4. Ablation. We validate our training design choices on the Spring training set after the TSKH stage. We compare training at original scales and inference at half scale (baseline) to inference at full resolution and training on either crops of or on full upsampled images. We also study the effect of uni-/bi-directional flow prediction. Our final method is highlighted in gray . For more details see Sec. 4.3.

Configuration EPE ↓ 1px ↓ Other Flow Train

Crop

Inf. Dir. Scale Scale avg s0-10 s10-40 s40+ avg s0-10 s10-40 s40+ WAUC ↑ Fl ↓

- Bi- 1x × 1/2 0.402 0.177 1.047 6.843 4.300 2.491 16.877 35.401 93.840 1.260

- Bi- 1x × 1 0.430 0.165 0.857 8.858 3.232 1.755 11.628 33.903 94.230 0.984

- Bi- 2x ✓ 1 0.378 0.166 0.811 6.960 3.195 1.815 11.231 31.933 94.192 0.873

- Bi- 2x × 1 0.341 0.133 0.818 6.592 3.061 1.739 11.156 29.423 95.604 0.823 Uni- 2x × 1 0.400 0.137 0.869 8.732 3.281 1.888 11.633 31.563 95.157 0.917

- Table 5. Ablation. Correlation volume resolution and number of frames. In all models, we set the feature dimension Df equal to 2Dc. Please refer to the supplementary material for additional metrics.

maps, correlation volumes can also be reused. By rearranging axes in Ct,t+1 and then pooling the result multiple times, we can get Ct+1,t without performing any matrix multiplications.

These optimizations reduce inference time by over 22% when compared to naive implementations of two variants of our method (Table 6).

Corr. scale #Frames Dc GMA 1px ↓ Mem

- 1/24 2 128 × 4.235 0.78

- 1/16 2 128 × 3.644 1.11

- 1/16 2 128 ✓ 3.547 1.29

- 1/16 2 256 × 3.420 1.12

- 1/16 2 512 × 3.375 1.30

1/24 3 512 ✓ 3.480 1.03

- 1/16 3 128 ✓ 3.560 1.78

- 1/16 3 256 ✓ 3.144 1.86

- 1/16 3 512 ✓ 3.061 2.09

- 1/16 3 512 × 3.151 1.82 1/24 5 512 ✓ 3.809 1.84

### 5. Conclusion

Table 6. Ablation. Inference time optimizations.

Time, ms 3 fr (1/16) 5 fr (1/24)

Method

Baseline 611 597 Only last Convex upsample 579 533 + Feature network reuse 483 341 + Fast correlation volume 478 334 + Correlation volume reuse 472 329

- • Fast correlation volume: The official SEA-RAFT implementation naively computes multi-scale correlation volumes between the feature map of the first image and pooled feature maps of the second image. Instead, we compute the correlation volume once and then pool it multiple times.
- • Correlation volume reuse: Similar to reusing feature

In this work, we introduced MEMFOF, a memory-efficient multi-frame optical flow method that achieves state-ofthe-art performance while maintaining a significantly reduced GPU memory footprint. By systematically revisiting RAFT-like architectures, we identified an optimal trade-off between multi-frame accuracy and memory efficiency, enabling training at native 1080p resolution without the need for cropping or downsampling. Our approach integrates reduced correlation volumes, multi-frame estimation, and high-resolution training strategies to deliver competitive accuracy across multiple benchmarks while operating with lower computational requirements. These findings position MEMFOF as a practical solution for large-scale, highresolution optical flow estimation, bridging the gap between accuracy and efficiency. Future work may further explore extending our approach to even higher resolutions and realtime applications.

### 6. Acknowledgments

This work was supported by the The Ministry of Economic Development of the Russian Federation in accordance with the subsidy agreement (agreement identifer 000000C313925P4H0002; grant No 139-15-2025-012). The research was carried out using the MSU-270 supercomputer of Lomonosov Moscow State University. We want to additionally thank Sergey Lavrushkin, Andrey Moskalenko, Ekaterina Shumitskaya and Vladislav Pyatov for proofreading and providing valuable feedback on the manuscript.

### References

- [1] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Trans. Graph., 28(3):24, 2009. 3
- [2] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In ECCV, pages 611–625. Springer,

2012. 2, 5

- [3] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In CVPR, pages 4947–4956, 2021. 1
- [4] Qiaole Dong and Yanwei Fu. Memflow: Optical flow estimation and prediction with memory. In CVPR, pages 19068– 19078, 2024. 1, 2, 4, 5, 6, 7
- [5] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learning optical flow with convolutional networks. In ICCV, pages 2758–2766, 2015. 1, 2, 5
- [6] Gunnar Farneb¨ack. Two-frame motion estimation based on polynomial expansion. In Image Analysis: 13th Scandinavian Conference, SCIA 2003 Halmstad, Sweden, June 29– July 2, 2003 Proceedings 13, pages 363–370. Springer, 2003. 2
- [7] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237,

2013. 5

- [8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 5
- [9] Berthold KP Horn and Brian G Schunck. Determining optical flow. Artificial intelligence, 17(1-3):185–203, 1981. 1, 2
- [10] Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer: A transformer architecture for optical flow. In ECCV, pages 668–685. Springer, 2022. 2, 6, 7
- [11] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In ECCV, pages 624–642. Springer, 2022. 1
- [12] Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox. Flownet 2.0: Evolution of optical flow estimation with deep networks. In CVPR, pages 2462–2470, 2017. 6, 7
- [13] Shihao Jiang, Dylan Campbell, Yao Lu, Hongdong Li, and Richard Hartley. Learning to estimate hidden motions with global motion aggregation. In ICCV, pages 9772–9781,

2021. 2, 5, 6, 7

- [14] Shihao Jiang, Yao Lu, Hongdong Li, and Richard Hartley. Learning optical flow from a few matches. In CVPR, pages 16592–16600, 2021. 1, 3
- [15] Wei-Sheng Lai, Yichang Shih, Lun-Cheng Chu, Xiaotong Wu, Sung-Fang Tsai, Michael Krainin, Deqing Sun, and

- Chia-Kai Liang. Face deblurring using dual camera fusion on mobile phones. ACM Transactions on Graphics (TOG), 41(4):1–16, 2022. 6
- [16] Xiaozhang Liu, Hui Liu, and Yuxiu Lin. Video frame interpolation via optical flow estimation with image inpainting. International Journal of Intelligent Systems, 35(12):2087– 2102, 2020. 1
- [17] Bruce D Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In IJCAI’81: 7th international joint conference on Artificial intelligence, pages 674–679, 1981. 1, 2
- [18] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In CVPR, pages 4981–4991, 2023. 1, 2, 4, 5
- [19] Moritz Menze and Andreas Geiger. Object scene flow for autonomous vehicles. In CVPR, pages 3061–3070, 2015. 2,

- 5

[20] Henrique Morimitsu, Xiaobin Zhu, Xiangyang Ji, and XuCheng Yin. Recurrent partial kernel network for efficient optical flow estimation. In AAAI, pages 4278–4286, 2024. 1,

- 6, 7

- [21] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, pages 724–732, 2016. 3
- [22] AJ Piergiovanni and Michael S Ryoo. Representation flow for action recognition. In CVPR, pages 9945–9953, 2019. 1
- [23] Zhile Ren, Orazio Gallo, Deqing Sun, Ming-Hsuan Yang, Erik B Sudderth, and Jan Kautz. A fusion approach for multiframe optical flow estimation. In IEEE Winter Conference on Applications of Computer Vision, pages 2077–2086. IEEE,

2019. 2

- [24] Stephan R Richter, Zeeshan Hayder, and Vladlen Koltun. Playing for benchmarks. In ICCV, pages 2213–2222, 2017. 5, 1
- [25] Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, and David J Fleet. The surprising effectiveness of diffusion models for optical flow and monocular depth estimation. NeurIPS, 36:39443– 39469, 2023. 7
- [26] Xiaoyu Shi, Zhaoyang Huang, Weikang Bian, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. In ICCV, pages 12469–12480, 2023. 1, 2, 3, 6, 7
- [27] Xiaoyu Shi, Zhaoyang Huang, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer++: Masked cost volume autoencoding for pretraining optical flow estimation. In CVPR, pages 1599–1610, 2023. 1
- [28] Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. In CVPR, pages 8934–8943, 2018. 1, 2, 6, 7
- [29] Shuyang Sun, Zhanghui Kuang, Lu Sheng, Wanli Ouyang, and Wei Zhang. Optical flow guided feature: A fast and robust motion representation for video action recognition. In CVPR, pages 1390–1399, 2018. 1

- [30] Shangkun Sun, Jiaming Liu, Huaxia Li, Guoqing Liu, Thomas Li, and Wei Gao. Streamflow: streamlined multiframe optical flow estimation for video sequences. NeurIPS, 37:9205–9228, 2025. 1, 2, 3, 4, 5, 6, 7
- [31] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 1, 2, 6, 7

- [32] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In ECCV, pages 36–

54. Springer, 2024. 2, 4, 5, 6, 7

- [33] Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Br´egier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, and J´erˆome Revaud. Croco v2: Improved cross-view completion pretraining for stereo matching and optical flow. In ICCV, pages 17969–17980, 2023. 2, 6, 7
- [34] Gangwei Xu, Shujun Chen, Hao Jia, Miaojie Feng, and Xin Yang. Memory-efficient optical flow via radius-distribution orthogonal cost volume. arXiv preprint arXiv:2312.03790,

2023. 3, 6, 7

- [35] Haofei Xu, Jiaolong Yang, Jianfei Cai, Juyong Zhang, and Xin Tong. High-resolution optical flow from 1d attention and correlation. In ICCV, pages 10498–10507, 2021. 1, 3, 5, 6, 7
- [36] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In CVPR, pages 8121–8130, 2022. 1, 2
- [37] Xiangyu Xu, Li Siyao, Wenxiu Sun, Qian Yin, and MingHsuan Yang. Quadratic video interpolation. NeurIPS, 32,

2019. 1

- [38] Yuxuan Zhao, Ka Lok Man, Jeremy Smith, Kamran Siddique, and Sheng-Uei Guan. Improved two-stream model for human action recognition. EURASIP Journal on Image and Video Processing, 2020:1–9, 2020. 1
- [39] Yang Zhao, Gangwei Xu, and Gang Wu. Hybrid cost volume for memory-efficient optical flow. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 8740– 8749, 2024. 1
- [40] Zihua Zheng, Ni Nie, Zhi Ling, Pengfei Xiong, Jiangyu Liu, Hao Wang, and Jiankun Li. Dip: Deep inverse patchmatch for high-resolution optical flow. In CVPR, pages 8925–8934,

2022. 3

# MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation

## Supplementary Material

### 7. Definitions

Here we will provide more detailed definitions used in the main text.

#### 7.1. WAUC

In optical flow, weighted area under curve (WAUC), originally from VIPER [24], is formally defined as the integral

2 5

5

5 − x 5

dx, (15)

f(x) ·

0

where f(x) is equal to the percentage of pixels where the flow error does not exceed x pixels. The metric ranges from 0 at worst to 100 at best.

#### 7.2. Mixture-of-Laplace Loss

For a single flow vector coordinate, the Mixture-of-Laplace (MoL) in SEA-RAFT is defined as:

α 2 · e−|µ

gt−µ|+

MixLap(µgt;α,β,µ) = −log

1 − α 2eβ · e−

|µgt−µ| eβ

+

, (16)

where µgt is the target flow coordinate, µ is the predicted flow coordinate, α is the predicted mixing coefficient, and β is the predicted scale parameter. For a single optical flow frame prediction, the MoL loss is defined as:

- 1

- 2HW u,v

MixLap µgt(u,v)d;

LMoL =

d∈{x,y}

α(u,v),β(u,v),µ(u,v)d . (17)

#### 7.3. 2D Motion histogram

In order to visually demonstrate the discrepancy in motion magnitudes between common training datasets and Spring, we construct 2D histograms of motion vectors. Final results can be seen in Figure 4. The histograms are constructed in the following way:

N

H

W

[u ≤ fn(h,w,0) ≤ u + 1]

H(u,v) =

n=1

w=1

h=1

·[v ≤ fn(h,w,1) ≤ v + 1],

where fn ∈ RH×W×2 is the nth flow field from a dataset, (u,v) is the motion vector (u ∈ [−H′,H′] and v ∈

- Table 7. Performance of our main model depending on the number of iterative refinements (N). Metrics are calculated on the Spring train dataset after the TSKH stage. Speed (runtime) was measured on an Nvidia RTX 3090 GPU (24 GB).

N 1px ↓ EPE ↓ WAUC ↑ Fl ↓ Speed, ms

- 0 6.170 0.893 90.898 2.625 71
- 1 3.752 0.397 94.731 1.212 172
- 2 3.300 0.350 95.322 0.979 215 4 3.133 0.339 95.565 0.863 299 6 3.081 0.340 95.603 0.835 385 8 3.061 0.341 95.604 0.823 472 10 3.050 0.342 95.601 0.820 557 12 3.045 0.342 95.598 0.819 642

- Table 8. FullHD, method configurations taken from leaderboard sumbissions. Speed (runtime) was measured on an Nvidia RTX 3090 GPU (24 GB).

Standard corr. Alt. corr. GB ms GB ms

Method

RAFT 7.97 557 1.32 1302 VideoFlow-BOF 17.74 1648 7.41 3275 MEMFOF 2.09 472 1.52 1235

[−W′,W′]) and [·] is the Iverson bracket. We set H′ = 1080, W′ = 1920, therefore our final histograms all have the same 2160 × 3840 resolution, for illustration purposes, we take the logarithm of bin counts. Maximum motion boundaries are derived as twice the size of images in the dataset, since the largest motion possible is to move diagonally from one corner of an image to the other one.

### 8. Additional ablations

In this section, we provide ablations or ablation data not included in the main text.

#### 8.1. Number of iterative refinements

We study our method’s behavior depending on the number of iterative refinements. The results are provided in Table 7. For a balance between speed and accuracy, we choose to perform 8 iterative refinements.

Table 9. Full correlation volume and number of frames ablation table.

1px ↓ EPE ↓ WAUC ↑ Fl ↓ Memory, GB avg s0-10 s10-40 s40+

Corr. scale #Frames Dc GMA

- 1/24 2 128 × 4.235 2.556 15.213 35.309 0.438 93.166 1.150 0.78

- 1/16 2 128 × 3.644 2.232 12.171 32.141 0.396 94.574 1.167 1.11

- 1/16 2 128 ✓ 3.547 2.132 12.101 32.025 0.408 94.617 1.035 1.29

- 1/16 2 256 × 3.420 2.072 11.440 30.941 0.372 94.761 1.018 1.12

- 1/16 2 512 × 3.375 2.047 11.201 30.614 0.350 95.130 0.888 1.30

1/24 3 512 ✓ 3.480 1.940 13.539 32.104 0.362 94.858 0.970 1.03

- 1/16 3 128 ✓ 3.560 2.154 12.176 31.543 0.380 94.859 1.094 1.78

- 1/16 3 256 ✓ 3.144 1.789 11.365 30.390 0.346 95.493 0.886 1.86

- 1/16 3 512 ✓ 3.061 1.739 11.156 29.423 0.341 95.604 0.823 2.09

- 1/16 3 512 × 3.151 1.833 10.988 30.165 0.332 95.623 0.896 1.82 1/24 5 512 ✓ 3.809 2.164 14.389 34.620 0.408 94.546 1.117 1.84

Table 10. Generalization performance of optical flow estimation on Sintel and KITTI-15 after the ”Things” stage. By default, all methods are trained on (FlyingChairs +) FlyingThings3D, additional datasets are listed in the ”Extra data” column.

Sintel (train) KITTI-15 (train) Clean ↓ Final ↓ Fl-epe ↓ Fl-all ↓

Extra data Method

PWC-Net 2.55 3.93 10.4 33.7 Flow1D 1.98 3.27 6.69 22.95 MeFlow 1.49 2.75 5.31 16.65 RAFT 1.43 2.71 5.04 17.40

TartanAir SEA-RAFT (S) 1.27 3.74 4.43 15.1 SEA-RAFT (M) 1.21 4.04 4.29 14.2 SEA-RAFT (L) 1.19 4.11 3.62 12.9

MemFlow 0.93 2.08 3.88 13.7 MemFlow-T 0.85 2.06 3.38 12.8 VideoFlow-BOF 1.03 2.19 3.96 15.3 VideoFlow-MOF 1.18 2.56 3.89 14.2 StreamFlow 0.87 2.11 3.85 12.6 MEMFOF (ours) 1.10 2.70 3.31 10.08

TartanAir MEMFOF (ours) 1.20 3.91 2.93 9.93

#### 8.2. Alternative correlation implementation

We additionally provide memory consumption and speed measurements for RAFT, VideoFlow and our method in Tab. 8 when using alternative correlation volume implementation that trades compute time for memory efficiency.

#### 8.3. Corr. volume resolution and number of frames

We provide the full version of Table 5 with additional metrics as Table 9.

### 9. Additional results

In this section, we provide some other results that are not included in the main text.

#### 9.1. Additional zero-shot evaluation

Following previous works, we evaluate the zero-shot performance of our method after the ”Things” training stage on Sintel (train) and KITTI (train). The results are provided in Table 10. Our method has the best zero-shot evaluation on KITTI and outperforms SEA-RAFT (L) on Sintel when trained on the same datasets.

Reference Frame MemFlow-T

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

SEA RAFT  L  MEMFOF  Ours)

- Figure 5. Qualitative comparison of MemFlow-T, SEA-RAFT (L), and our method on the Sintel benchmark. Sourced from official leaderboard submissions.

Reference Frame MemFlow-T

[Figure 39]

[Figure 40]

[Figure 41]

SEA RAFT  L  MEMFOF  Ours)

[Figure 42]

- Figure 6. Qualitative comparison of MemFlow-T, SEA-RAFT (L), and our method on the KITTI-2015 benchmark. Sourced from official leaderboard submissions.

#### 9.2. Qualitative comparison on Sintel and KITTI

We provide qualitative comparisons of our method on the Sintel and KITTI public benchmarks. As Figure 5 and Fig-

ure 6 show, our method has higher motion detail and coherence than our baseline or competitor.

