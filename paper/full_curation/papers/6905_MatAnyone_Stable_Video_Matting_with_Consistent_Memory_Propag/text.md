## MatAnyone: Stable Video Matting with Consistent Memory Propagation

Peiqing Yang1 Shangchen Zhou1 Jixin Zhao1 Qingyi Tao2 Chen Change Loy1 1S-Lab, Nanyang Technological University 2SenseTime Research, Singapore

[Figure 1]

[Figure 2]

https://pq-yang.github.io/projects/MatAnyone

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2501.14677v2[cs.CV]25Mar2025

[Figure 6]

[Figure 7]

|Input|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|---|---|---|---|---|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

RVMOursOurs

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

Figure 1. Our MatAnyone is capable of producing highly detailed and temporally consistent alpha mattes throughout a video. (a) It adapts to a variety of frame sizes and media types (e.g., films, games, smartphone videos), achieving fine-grained details at the image-matting level. (b) RVM [33], an auxiliary-free video matting method, struggles with complex or ambiguous backgrounds. In contrast, our method effectively isolates the target object from such distractors, preserving a clean background and complete foreground parts. (c) Our method also excels at consistently tracking the target (i.e., the lady in pink) even in scenes containing multiple salient objects (i.e., the man and the lady). It accurately distinguishes between them even during their interactions. (Zoom-in for best view)

### Abstract

module via region-adaptive memory fusion, which adaptively integrates memory from the previous frame. This ensures semantic stability in core regions while preserving fine-grained details along object boundaries. For robust training, we present a larger, high-quality, and diverse dataset for video matting. Additionally, we incorporate a novel training strategy that efficiently leverages large-scale segmentation data, boosting matting stability. With this new

Auxiliary-free human video matting methods, which rely solely on input frames, often struggle with complex or ambiguous backgrounds. To address this, we propose MatAnyone, a robust framework tailored for target-assigned video matting. Specifically, building on a memory-based paradigm, we introduce a consistent memory propagation

network design, dataset, and training strategy, MatAnyone delivers robust and accurate video matting results in diverse real-world scenarios, outperforming existing methods.

### 1. Introduction

Auxiliary-free human video matting (VM) is widely recognized for its convenience [24, 27, 33], as it only requires input frames without additional annotations. However, its performance often deteriorates in complex or ambiguous backgrounds, especially when similar objects, i.e., other humans, appear in the background (Fig. 2(b)). We consider auxiliary-free video matting to be under-defined, as their results can be uncertain without a clear target object.

In this work, we focus on a problem that is more applicable to real-world video applications: video matting focused on pre-assigned target object(s), with the target segmentation mask provided in the first frame. This enables the model to perform stable matting via consistent object tracking throughout the entire video, while offering better interactivity. The setting is well-studied in Video Object Segmentation (VOS), where it is referred to as “semisupervised” [10, 19, 38]. A common strategy is to use a memory-based paradigm [8, 12, 38, 51], encoding past frames and corresponding segmentation results into memory, from which a new frame retrieves relevant information for its mask prediction. This allows a lightweight network to achieve consistent and accurate tracking of the target object. Inspired by this, we adapt the memory-based paradigm for video matting, leveraging its stability across frames.

Video matting poses additional challenges compared to VOS, as it requires not only accurate semantic detection in core regions but also high-quality detail extraction along the boundary (e.g., hair), as defined in Fig. 2(a). A straightforward approach is to fine-tune matting details using matting data, based on segmentation priors from VOS. Recent approaches attempt to achieve both goals, either in a coupled or decoupled manner. For instance, AdaM [31] and FTPVM [21] refine the memory-based segmentation mask for each frame via a decoder to produce alpha mattes, while MaGGIe [22] devises a separate refiner network to process segmentation masks across all frames from an off-the-shelf VOS model. However, these methods often lead to suboptimal results due to limitations in the available video matting data: (i) the quality of VideoMatte240K [32], the most widely used human video matting dataset, is suboptimal. Its ground-truth alpha mattes exhibit problematic semantic accuracy in core areas (e.g., interior holes) and lack fine details along the boundaries (e.g., blurry hair); (ii) video matting datasets are much smaller in scale compared to VOS datasets; and (iii) video matting data are synthetic due to the extreme difficulty of human annotations, limiting their generalizability to real-world cases [33]. Consequently, finetuning a strong VOS prior for video matting with existing

video matting data usually disrupts this prior. While boundary details may show improvement compared to segmentation results, the matting quality in terms of semantic stability in core areas and details in boundary areas remain unsatisfactory, as shown by the results of MaGGIe in Fig. 2(b).

Producing matting-level details while maintaining semantic stability of a memory-based approach is challenging, especially training with suboptimal video matting data. To tackle this, we focus on several key aspects:

Network - we introduce a consistent memory propagation mechanism in the memory space. For each current frame, the alpha value change relative to the previous frame is estimated for every token. This estimation guides the adaptive integration of information from the previous frame. The “large-change” regions rely more on the current frame’s information queried from the memory bank, while “smallchange” regions tend to retain the memory from the previous frame. This region-adaptive memory fusion inherently stabilizes memory propagation throughout the video, improving matting quality with fine details and temporal consistency. Specifically, it encourages the network to focus on boundary regions during training to capture fine details, while “small-change” tokens in the core regions preserve internally complete foreground and clean background (see our results in Fig. 2(b)).

Data - we collect a new training dataset, named VM800, which is twice as large, more diverse, and of higher quality in both core and boundary regions compared to the VideoMatte240K dataset [32], greatly enhancing robust training for video matting. In addition, we introduce a more challenging test dataset, named YoutubeMatte, featuring more diverse foreground videos and improved detail quality. These new datasets offer a solid foundation for robust training and reliable evaluation in video matting.

Training Strategy - the lack of real video matting data remains a significant limitation, affecting both stability and generalizability. We address this problem by leveraging large-scale real segmentation data via a novel training strategy. Unlike common practices [21, 22, 33] that train with segmentation data on a separate prediction head parallel to the matting head, we propose using segmentation data within the same head as matting for more effective supervision. This is achieved by applying region-specific losses – for core regions, we apply a pixel-wise loss to ensure stability and generalization in semantics; for boundary regions, where segmentation data lacks alpha labels, we employ an improved DDC loss [35], scaled to make edges resemble matting rather than segmentation.

In summary, our main contributions are as follows:

• We propose MatAnyone, a practical human video matting framework supporting target assignment, with stable performance in both semantics of core regions and fine-grained boundary details. Target object(s) can be easily assigned using off-the-shelf segmentation methods, and reliable tracking is achieved even in long videos with

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

| |
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

| |
|---|

Core Areas Boundary Area Input

MaGGIe RVM Ours

|MaGGIe|
|---|

|RVM|
|---|

(a) Definitions for Matting

(b) Issues: Segmentation prior broken Confused by ambiguous background

- Figure 2. Definitions and motivations for MatAnyone. (a) In a matting frame, the image can be broadly divided into two areas based on the alpha value: the core (semantic) and the boundary (fine-details). The core includes the background (alpha values of 0) and the solid foreground (alpha values of 1), while the boundary (highlighted in pink) encompasses areas with alpha values between 0 and 1. (b) Due to the under-defined setting, auxiliary-free methods like RVM [33] are easily confused by ambiguous background. Meanwhile, mask-guided methods like MaGGIe [22] tend to break the segmentation prior they aim to leverage, due to the deficiency in video matting data.

multaneously train with real segmentation data for semantic supervision [21, 31, 33].

complex and ambiguous backgrounds.

- • We introduce a consistent memory propagation mechanism via region-adaptive memory fusion, improving stability in core regions and quality in boundary details.
- • We contribute larger and higher-quality datasets for training and testing, offering a solid foundation for robust training and reliable evaluation in video matting.
- • To overcome the scarcity of real video matting data, we leverage real segmentation data for core-area supervision, largely improving semantic stability over prior methods.

Memory-based VOS. Semi-supervised VOS segments the target object with a first-frame annotation across frames [8– 12, 18, 30, 37, 42]. The memory matching paradigm by Space-Time Correspondence Network (STCN) [10] is widely followed by current VOS methods [8, 12, 46, 51], and achieves good performance. We thus take the memorybased paradigm as our framework since it is similar to our setting except that our outputs are alpha mattes.

Video Consistency in Low-level Vision. To enhance temporal consistency across adjacent frames, the recurrent frame fusion [47, 59] and optical flow-guided propagation [4–6, 60] are commonly utilized in the video restoration networks. Recent methods also employ temporal layers such as 3D convolution [2, 48] and temporal attention [2, 7, 49, 61] during training, while other training-free methods resort to cross-frame attention [50, 53] and flowguided attention [13, 15] in the pretrained models. In this work, we find that the memory-based paradigm is effective enough to maintain video consistency for video matting.

### 2. Related Work

Video Matting. Due to the intrinsic ambiguity in the auxiliary-free setting [24, 27, 33, 39, 57, 62], such tasks generally are object-specific. Among them, human video matting [24, 27, 43, 62] without auxiliary inputs is popular due to its wide applications. Challenging as the auxiliaryfree setting, being in the video domain brings in additional difficulties in temporal coherency. MODNet [24] extends its portrait matting setting to video domain with a flickering reduction trick (non-learning) within a local sequence. RVM [33] steps further to design for videos specifically with ConvGRU [1] as its recurrent architecture. Robust as RVM, it is still easy to be confused by humans in the background. With the success of promptable segmentation [25, 40, 58, 63], obtaining segmentation mask for a target human object only requires minimal human efforts. Recent mask-guided image [3, 29, 55, 56] and video matting [21, 22, 28, 31] thus leverage this convenience for a more robust performance. Adam [31] propagates the first-frame segmentation mask across all frames while FTPVM [21] propagates the first-frame trimap. Taking the propagated mask as a rough result, their decoder serves for matting details refinement. MaGGIe [22] enjoys a stronger prior by taking the segmentation mask across all frames instead of the first one. Taking all the segmentation masks at a time, the network is able to perform bidirectional temporal fusion for coherency. To mitigate the poor generalizability of synthetic video matting data, a common practice is to si-

### 3. Methodology

Overview. Achieving matting-level details while preserving the semantic stability of a memory-based approach poses challenges, especially when training with suboptimal video matting data. To tackle this, we propose our MatAnyone, as illustrated in Fig. 3. Similar to semi-supervised VOS, MatAnyone only requires the segmentation mask for the first frame as a target assignment (e.g., the yellow mask in Fig. 3(a)). The alpha matte for the assigned object is then generated frame by frame in a sequential manner. Specifically, for an incoming frame t, it is first encoded into Ft as ×16 downsampled feature representation, which is then transformed into key and query for consistent memory propagation (Sec. 3.1), and output the pixel memory readout Pt. We employ the object transformer proposed by Cutie [12] to group the pixel memory by object-level semantics for robustness against noise brought by low-level pixel matching.

[Figure 37]

|[Figure 38]<br><br>#0<br><br>[Figure 39]<br><br>#t<br><br>[Figure 40]<br><br>#N<br><br>……|
|---|

Matting Data

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- (a)OverallFramework

(c)TrainingStrategy

- (b)ConsistentMemoryPropagation

|synthetic small scale w/ matting details<br><br>|
|---|

#### ……

[Figure 50]

Decoder

Encoder

$!

!! "!

%!

Consistent Memory Propagation

Object Transformer

[Figure 51]

Segment. Data

[Figure 52]

[Figure 53]

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

|real large scale w/o matting details<br><br>|
|---|

#!

Value Encoder

Update Alpha Memory

Update Alpha Memory !! (every r-th frame)

Matting Data (w/ GT alpha matte)

[Figure 64]

MattingLoss

[Figure 65]

|Attention|)"#|
|---|---|
| | |

| | |
|---|---|
| | |

|GT|
|---|

[Figure 66]

[Figure 67]

###### Alpha Memory Bank

[Figure 68]

'#,)#

Attention

| |
|---|

[Figure 69]

MatAnyone

*"

key value

Output

###### Current Frame

|Uncertainty Prediction|,"|
|---|---|
| | |

+"

| | |
|---|---|
| | |

[Figure 70]

'"

Uncertainty Prediction

&"

Uncertain Loss

key query

CertainLoss

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

| |
|---|

|Output| |
|---|---|
| | |

'"$%

Last Frame Memory

| |
|---|

MatAnyone

[Figure 75]

)"$%

GT

key value

Segmentation Data (w/o GT alpha matte)

Update Alpha Memory !! (every frame)

- Figure 3. An overview of MatAnyone. MatAnyone is a memory-based framework for video matting. Given a target segmentation map in the first frame, our model achieves stable and high-quality matting through consistent memory propagation, with a region-adaptive memory fusion module to combine information from the previous and current frame. To overcome the scarcity of real video matting data, we incorporate a new training strategy that effectively leverages matting data for fine-grained matting details and segmentation data for semantic stability, with designed losses separately.

masks [31] or trimaps [21] in memory and use a decoder to refine the matting details. Such approaches do not fully leverage the stability provided by the memory paradigm in boundary regions, leading to instability such as flickering. To address this, building on the memory-based framework [10], our MatAnyone stores the alpha matte in an alpha memory bank to enhance stability in boundary regions. Region-Adaptive Memory Fusion. Given the inherent difference between the segmentation map (values of 0 or 1) and the matting map (values between 0 and 1), the memorymatching approach needs to be adjusted. Specifically, in STCN [10], memory values for the query frame are based on the similarity between query and memory key, assuming equal importance for all query tokens. However, this assumption does not hold for video matting. As shown in Fig. 2(a), a query frame can be divided into core and boundary regions. When compared with frame t−1, only a small fraction of tokens in frame t change significantly in alpha values, with these “large-change” tokens mainly located in object boundaries, while the “small-change” tokens reside in the core regions. This highlights the need to treat core and boundary regions separately to enforce stability.

The refined memory readout Ot acts as the final feature to be sent into the decoder for alpha matte prediction. The predicted alpha matte Mt is then encoded to memory value V t, which is used to update the alpha memory bank.

Due to limitations in the quality and quantity of video matting data, training with such data makes it difficult to achieve satisfactory stability in core regions. To mitigate this, RVM [33] proposes a parallel head for real segmentation data alongside the matting head, guiding the network to be robust in real-world cases. However, this is not sufficient, as the matting head itself cannot receive supervision from real data. Inspired by the DDC loss [35] designed for alpha-free image matting, we devise a training strategy for core regions, which provides direct supervision to the matting head with segmentation data (Sec. 3.2), leading to substantial improvements in semantic stability.

We also propose a practical inference strategy that allow for flexible application: a recurrent refinement approach applied to the first frame, based on the memory-driven paradigm, enhancing robustness to the given mask and refining matting details (Sec. 3.3).

##### 3.1. Consistent Memory Propagation

Specifically, we introduce a boundary-area prediction module to estimate the change probability Ut of each query token for adaptive memory fusion, where higher Ut indicates “large-change” regions and lower Ut indicates “smallchange” regions. The prediction module is lightweight,

Alpha Memory Bank. In this study, we introduce a consistent memory propagation (CMP) module specifically designed for video matting, as illustrated in Fig.3(b). Existing memory-based VM methods store either segmentation

consisting of three convolution layers. We formulate the prediction as a binary segmentation problem with loss Lbin seg and use the actual alpha change between frame t − 1 and t as supervision. Specifically, we define UtGT : |MtGT−1 − MtGT| >= δ, where δ is a threshold. Using the output of the module Uˆt, we compute the binary cross entropy loss against UtGT. During the region-adaptive memory fusion process, we apply the sigmoid function on Uˆt to transform it as a probability. The final pixel memory readout is a soft merge:

Pt = Vtm ∗ Ut + Vt−1 ∗ (1 − Ut), (1)

where Ut ∈ [0,1], Vtm are current values queried from memory bank, and Vt−1 are values propagated from the last frame. This approach significantly improves stability in core regions by maintaining internal completeness and a clean background (Fig. 2(b) and Fig. 4). It also enhances stability in boundary regions, as it directs the network to focus on object boundaries with soft alpha values, while the memory-based paradigm inherently stabilizes the matched values (see Table 3(c)). A detailed analysis is provided in the ablation study of Sec. 5.2 and Sec. J.2.

##### 3.2. Core-area Supervision via Segmentation

New Training Scheme. Most recent video matting methods follow RVM’s approach of using real segmentation data to address the limitations of video matting data. In these methods, segmentation and matting data are fed to the main shared network, but are directed to produce outputs at separate heads. Although segmentation data do supervise the main network to empower generalizability and robustness to the matting model, the stability they provide falls short of what a VOS model could achieve. As shown in Fig. 2, both RVM and MaGGIe perform significantly worse than the VOS outputs (white masks on inputs) by XMem [8] in core areas, where semantic information is key. We believe the parallel head training scheme may not fully exploit the rich segmentation prior in the data. To address this, we propose to supervise the matting head directly with segmentation data. Specifically, we predict the alpha matte for segmentation inputs and optimize the matting outputs accordingly, as illustrated in Fig. 3(c).

Scaled DDC Loss. A natural challenge arises with the aforementioned approach: how can we compute the loss on matting outputs for segmentation data when there is no ground truth (GT) alpha matte? For core areas, the GT labels are readily available in the segmentation data, where an l1 loss suffices, and we denote it as Lcore. The real difficulty lies in the boundary region. A recent paper proposes DDC loss [35], which supervises boundary areas using the input image without requiring a GT alpha matte.

LDDC = N1

N

- i j

|αi − αj − ∥Ii − Ij∥2|,

- j ∈ argtopk{−∥Ii − Ij∥2}.

(2)

However, we find that the underlying assumption of this design, that ∥αi − αj∥2 = ∥Ii − Ij∥2 for αi > αj, does not always hold true. For two image pixels Ii and Ij, their difference is given by:

Ii −Ij = [αiFi +(1−αi)Bi]−[αjFj +(1−αj)Bj], (3)

where Fi, Bi represent the foreground and background values at pixel i, and similarly for Fj and Bj at pixel j. Since we impose the constraint j ∈ argtopk{−∥Ii − Ij∥2}, we can assume Fi = Fj = F, Bi = Bj = B within a small window. This simplifies Eq. (3) to:

Ii − Ij = (αi − αj)(F − B). (4)

This shows that the assumptions for DDC loss hold only when |F − B| = 1. To account for this, we devise a scaled version as our boundary loss Lboundary:

N

Lboundary = N1

|(αi − αj)(F − B) − ∥Ii − Ij∥2|, j ∈ argtopk{−∥Ii − Ij∥2},

i j

(5) where F is approximated by the average of the top k largest pixel values in the small window, and B by the average of the top k smallest pixel values. In the ablation study (Sec. 5.2), we show that training with our scaled DDC loss (Eq. (5)) yields more natural matting results than training with the original version (Eq. (2)), which tends to produce segmentation-like jagged and stair-stepped edges.

##### 3.3. Recurrent Refinement During Inference

The first-frame matte is predicted from the given first-frame segmentation mask, and its quality will affect the matte prediction for the subsequent frames. The sequential prediction in the memory-based paradigm enables recurrent refinement during inference. Leveraging this mechanism, we introduce an optional first-frame warm-up module for inference. Specifically, we repeat the first frame n times, treating each repetition as the initial frame, and use only the nth alpha output as the first frame to initialize the alpha memory bank. This (1) enhances robustness against the given segmentation mask and (2) refines matting details in the first frame to achieve image-matting quality (see Fig. 6 and Fig. 13 in the appendix).

### 4. Data

We briefly introduce our new training datasets and benchmarks for evaluation, including both synthetic and realworld. More details are provided in the appendix (Sec. I).

##### 4.1. Training Datasets

To address limitations in video matting datasets in both quality and quantity, we collect abundant green screen videos, process them with Adobe After Effects, and conduct

- Table 1. Quantitative comparisons on different video matting benchmarks from diverse sources. The best and second-best performances are marked in red and orange , respectively. † indicates that MaGGIe [22] requires the instance mask as guidance for each frame, while our method only requires it in the first frame.

|Metrics<br><br>|Auxiliary-free (AF) Methods|Mask-guided Methods|
|---|---|---|
| |MODNet [24] RVM [33] RVM-Large [33]|AdaM [31] FTP-VM [20] MaGGIe† [22] Ours|

VideoMatte (512 × 288) MAD↓ 9.41 6.08 5.32 5.30 6.13 5.49 5.15

- MSE↓ 4.30 1.47 0.62 0.78 1.31 0.60 0.93

- Grad↓ 1.89 0.88 0.59 0.72 1.14 0.57 0.67

- dtSSD↓ 2.23 1.36 1.24 1.33 1.60 1.39 1.18 Conn↓ 0.81 0.41 0.30 0.30 0.41 0.31 0.26 VideoMatte (1920 × 1080) MAD↓ 11.13 6.57 5.81 4.42 8.00 4.42 4.24

MSE↓ 5.54 1.93 0.97 0.39 3.24 0.40 0.33 Grad↓ 15.30 10.55 9.65 5.12 23.75 4.03 4.00

- dtSSD↓ 3.08 1.90 1.78 1.39 2.37 1.31 1.19 YoutubeMatte (512 × 288) MAD↓ 19.37 4.08 3.36 - 3.08 3.54 2.72 MSE↓ 16.21 1.97 1.04 - 1.29 1.23 1.01

- Grad↓ 2.05 1.34 1.03 - 1.16 1.10 0.97

dtSSD↓ 2.79 1.81 1.62 - 1.83 1.88 1.60 Conn↓ 2.68 0.60 0.50 - 0.41 0.49 0.39 YoutubeMatte (1920 × 1080)

MAD↓ 15.29 4.37 3.58 - 6.49 2.37 1.99 MSE↓ 12.68 2.25 1.23 - 4.58 0.98 0.71 Grad↓ 8.42 15.1 12.97 - 29.78 7.69 8.91 dtSSD↓ 2.74 2.28 2.04 - 2.41 1.77 1.65

manual selection to remove common artifacts also found in VideoMatte240K [32] (see Fig. 8). Compared to VideoMatte240K, our dataset, VM800, is (1) twice as large, (2) more diverse in terms of hairstyles, outfits, and motion, and (3) higher in quality. Ablation studies (Table 3(b) and Sec. J.1) further demonstrate the advantages of our dataset.

##### 4.2. Synthetic Benchmark

The standard benchmark, VideoMatte [32], derived from VideoMatte240K, includes only 5 unique foreground videos, which is under representative. Additionally, their foregrounds lack sufficient boundary details, limiting their ability to discern matting precision in boundary regions. To create a more comprehensive benchmark, we compile 32 distinct 1920 × 1080 green-screen foreground videos from YouTube, and process them similarly to our training dataset. Our benchmark, YouTubeMatte, provides enhanced detail representation, as reflected by higher Grad [41] values.

##### 4.3. Real-world Benchmark and Metric

Real-world benchmarks are essential to facilitate the practical use of video matting models. Although real-world videos lack ground truth (GT) alpha mattes, we can generate frame-wise segmentation masks as GT for core areas benefiting from the high capability of existing VOS methods.

Specifically, we select a subset of 25 real-world videos [33] (100 frames each) with high-quality core GT masks verified manually. MAD, MSE, and dtSSD [14] are then calculated at the core region as core region metrics, representing semantic stability that is critical for visual perception.

### 5. Experiments

Training Schedule. Stage 1. Following the practice of RVM [33], we start by training the entire model on our VM800 for 80k iterations. The sequence length is initially set to 3 and extended to 8 with increasing sampling intervals for more complex scenarios. Stage 2. As the key stage, we apply the core supervision training strategy introduced in Section 3.2. Real segmentation data COCO [34], SPD [45] and YouTubeVIS [52] are added for supervising the matting head. The loss function applied are specified in Section 3.2. Stage 3. Finally, we fine-tune the model with image matting data D646 [39] and AIM [26] for finer matting details.

##### 5.1. Comparisons

We compare MatAnyone with several state-of-the-art methods, including auxiliary-free (AF) methods: MODNet [24], RVM [33], and RVM-Large [33], and mask-guided methods: AdaM [31], FTP-VM [21], and MaGGIe [22].

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

| |
|---|

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

Frame t Frame t+5

Video Frame RVM FTP-VM MaGGIe Ours

- Figure 4. Qualitative comparisons on real-world videos. Our MatAnyone significantly outperforms existing auxiliary-free (RVM [33]) and mask-guided (FTP-VM [21] and MaGGIe [22]) approaches in both detail extraction and semantic accuracy. For the lowest row, while other methods all miss out on important body parts (i.e., head) and mistakenly take background pixels as foreground (due to similar colors), thus generating messy outputs, our method presents an accurate and visually clean output by even identifying the shadow near the boundary.

- Table 2. Quantitative comparisons on real-world benchmark [33]. The best and second performances are marked in red and

orange , respectively.

Methods MAD↓ MSE↓ dtSSD↓ Auxiliary-free

MODNet [24] 11.67 10.12 3.37 RVM [33] 1.21 0.77 1.43 RVM-Large [33] 0.95 0.50 1.30 Mask-guided

FTP-VM [21] 4.77 4.11 1.68 MaGGIe [22] 1.94 1.53 1.63 MatAnyone (Ours) 0.14 0.10 0.89

- 5.1.1 Quantitative Evaluations

Table 3. Ablation study of the new training dataset (New Data), consistent memory propagation module (CMP), and new training scheme (New Training) on real benchmark (about 1080p).

Exp. New Data CMP New Training MAD↓ MSE↓ dtSSD↓

- (a) 3.16 2.65 1.37

- (b) ✓ 2.55 2.25 1.36

- (c) ✓ ✓ 1.85 1.67 1.25

- (d) ✓ ✓ ✓ 0.42 0.34 0.94

###### 5.1.2 Qualitative Evaluations

Visual results on real-world videos are in Fig. 4 and Fig. 5. General Video Matting. MatAnyone outperforms existing auxiliary-free and mask-guided approaches in both detail extraction (boundary) and semantic accuracy (core). Fig. 4 shows that MatAnyone excels at fine-grained details (e.g., hair in the middle row) and differentiates full human body against complicated or ambiguous backgrounds when foreground and background colors are similar (e.g., last row).

Synthetic Benchmarks. For a comprehensive evaluation on synthetic benchmarks, we employ MAD (mean absolute difference) and MSE (mean squared error) for semantic accuracy, Grad (spatial gradient) [41] for detail extraction, Conn (connectivity) [41] for perceptual quality, and dtSSD [14] for temporal coherence. In Table 1, our method achieves the best MAD and dtSSD across all datasets at both high and low resolutions, demonstrating exceptional spatial accuracy for alpha mattes and remarkable temporal stability. Apart from accuracy and stability, our method achieves the best Conn on both benchmarks, indicating its superior visual quality (Fig. 4 and Sec. J.5 in the appendix).

Instance Video Matting. The assignment of target object at the first frame gives us flexibility for instance video matting. In Fig. 5, although MaGGIe [22] benefits from using instance masks as guidance for each frame, our method demonstrates superior performance in instance video matting, particularly in maintaining object tracking stability and preserving fine-grained details of alpha mattes.

##### 5.2. Ablation Study

Enhancement from New Training Data. In Table 3, by comparing (a) and (b), it is observed that training with new data noticeably improves the semantic performance with decreased MAD and MSE, showing that our newlycollected VM800 indeed contributes to robust training with its upgraded quantity, quality, and diversity.

Real Benchmark. For evaluation on real benchmarks, we use the core region metrics in Section 4.3. In Table 2, our method demonstrates superior generalizability on real cases, achieving the best metric values with a substantial margin over both auxiliary-free and mask-guided methods.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

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

[Figure 114]

| | | |
|---|---|---|
|#1| | |
| | | |
| | |#3|

MaGGIeOurs

MaGGIeOurs

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

#1

[Figure 120]

#3

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

#2

#2

Video Frame Instance #1 Instance #2 Instance #3

Video Frame Instance #1

Instance #2 Instance #3

Figure 5. Quantitative comparisons with MaGGIe [22] on instance video matting. Despite MaGGIe using instance mask as guidance for each frame, our method shows better performance, achieving better stability in object tracking and finer alpha matte details.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

|[Figure 130]<br><br>|[Figure 131]|
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

!! = 1 !! = 2 !! = 5 !! = 10

Video Frame Segmentation Mask

Figure 6. Improvement with Recurrent refinement. (Zoom-in for best view)

other methods in Table 2 without further fine-tuning.

[Figure 132]

[Figure 133]

[Figure 134]

| |
|---|

Scaled DDC Loss. We examine the merit of the scaled version of DDC loss by training with Lcore and Lboundary only to maximize its effect. In Fig. 7, training with vanilla DDC loss produces segmentation-like jaggedness, especially among the boundary region. Our scaled DDC loss yields more stable and natural matting results.

[Figure 135]

[Figure 136]

[Figure 137]

| |
|---|

Effectiveness of Recurrent Refinement. Fig. 6 shows the effectiveness of recurrent refinement in a progressive manner. Given a rough segmentation mask, our method can produce alpha matte with descent details within 10 iterations.

### 6. Conclusion

Video Frames DDC Loss

Scaled DDC loss

- Figure 7. Comparison of matting results training with original DDC loss [35] and with scaled DDC loss, where the latter gives more stable and natural matting results.

We introduce MatAnyone, a practical framework for targetassigned human video matting that ensures stable and accurate results across diverse real-world scenarios. Our method leverages a region-adaptive memory fusion approach, which combines memory from previous frames to maintain semantic consistency in core areas while preserving fine details along object boundaries. With a new training dataset that is larger, high-quality, and diverse and a novel training strategy that effectively leverages segmentation data, MatAnyone achieves robust and stable matting performance, even with complex backgrounds. These advancements position MatAnyone a practical solution for real-world video matting, also setting a solid foundation for future research in memory-based video processing.

Effectiveness of Consistent Memory Propagation. We further investigate the effectiveness of the consistent memory propagation (CMP) module. From Table 3 (b) to (c), improvement can be seen across all metrics with CMP added, indicating its effectiveness in improving semantic stability and temporal coherency. In particular, dtSSD in (c) is already lower than all the other methods in Table 2, showing the superiority of CMP in terms of temporal consistency.

Effectiveness of New Training Scheme. Our new training scheme brings our model to the next level with a noticeable improvement in all metrics. It already outperforms all the

Acknowledgement. This study is supported under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

### References

- [1] Nicolas Ballas, Li Yao, Christopher J Pal, and Aaron Courville. Delving deeper into convolutional networks for learning video representations. In ICLR, 2016. 3
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your Latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 3
- [3] Huanqia Cai, Fanglei Xue, Lele Xu, and Lili Guo. TransMatting: Enhancing transparent objects matting with transformers. In ECCV, 2022. 3
- [4] Kelvin C.K. Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. BasicVSR: The search for essential components in video super-resolution and beyond. In CVPR,

2021. 3

- [5] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Improving video super-resolution with enhanced propagation and alignment. In CVPR, 2022.
- [6] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In CVPR, 2022. 3
- [7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. VideoCrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 3
- [8] Ho Kei Cheng and Alexander G. Schwing. XMem: Longterm video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022. 2, 3, 5, 12, 13, 14, 16
- [9] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Modular interactive video object segmentation: Interaction-to-mask, propagation and difference-aware fusion. In CVPR, 2021.
- [10] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. In NeurIPs, 2021. 2, 3, 4, 12
- [11] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In ICCV, 2023.
- [12] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In CVPR, 2024. 2, 3, 12, 13, 14, 16
- [13] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. FLATTEN: optical flow-guided attention for consistent text-to-video editing. In ICLR, 2024. 3
- [14] Mikhail Erofeev, Yury Gitman, Dmitriy S Vatolin, Alexey Fedorov, and Jue Wang. Perceptually motivated benchmark for video matting. In BMVC, 2015. 6, 7, 16
- [15] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. TokenFlow: Consistent diffusion features for consistent video editing. In ICLR, 2024. 3

- [16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,

2016. 12

- [17] Qiqi Hou and Feng Liu. Context-aware image matting for simultaneous foreground and alpha estimation. In ICCV, 2019. 13
- [18] Li Hu, Peng Zhang, Bang Zhang, Pan Pan, Yinghui Xu, and Rong Jin. Learning position and target consistency for memory-based video object segmentation. In CVPR, 2021. 3
- [19] Yuan-Ting Hu, Jia-Bin Huang, and Alexander Schwing. MaskRNN: Instance level video object segmentation. In NeurIPS, 2017. 2
- [20] Wei-Lun Huang and Ming-Sui Lee. End-to-end video matting with trimap propagation. In CVPR, 2023. 6
- [21] Wei-Lun Huang and Ming-Sui Lee. End-to-end video matting with trimap propagation. In CVPR, 2023. 2, 3, 4, 6, 7, 13, 18, 20, 21, 22
- [22] Chuong Huynh, Seoung Wug Oh, , Abhinav Shrivastava, and Joon-Young Lee. MaGGIe: Masked guided gradual human instance matting. In CVPR, 2024. 2, 3, 6, 7, 8, 13, 18, 20, 21, 22
- [23] Zhanghan Ke, Chunyi Sun, Lei Zhu, Ke Xu, and Rynson WH Lau. Harmonizer: Learning to perform white-box image and video harmonization. In ECCV, 2022. 16
- [24] Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, and Rynson W.H. Lau. MODNet: Real-time trimap-free portrait matting via objective decomposition. In AAAI, 2022. 2, 3, 6, 7
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 3, 19
- [26] Jizhizi Li, Jing Zhang, and Dacheng Tao. Deep automatic natural image matting. In IJCAI, 2021. 6
- [27] Jiachen Li, Vidit Goel, Marianna Ohanyan, Shant Navasardyan, Yunchao Wei, and Humphrey Shi. VMFormer: End-to-end video matting with transformer. In WACV, 2024. 2, 3
- [28] Jiachen Li, Roberto Henschel, Vidit Goel, Marianna Ohanyan, Shant Navasardyan, and Humphrey Shi. Video instance matting. In WACV, 2024. 3
- [29] Jiachen Li, Jitesh Jain, and Humphrey Shi. Matting Anything. In CVPR, 2024. 3
- [30] Xiangtai Li, Haobo Yuan, Wenwei Zhang, Guangliang Cheng, Jiangmiao Pang, and Chen Change Loy. Tube-link: A flexible cross tube framework for universal video segmentation. In ICCV, 2023. 3
- [31] Chung-Ching Lin, Jiang Wang, Kun Luo, Kevin Lin, Linjie Li, Lijuan Wang, and Zicheng Liu. Adaptive human matting for dynamic videos. In CVPR, 2023. 2, 3, 4, 6, 13
- [32] Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian L Curless, Steven M Seitz, and Ira KemelmacherShlizerman. Real-time high-resolution background matting. In CVPR, 2021. 2, 6, 12, 14, 15, 16, 17
- [33] Shanchuan Lin, Linjie Yang, Imran Saleemi, and Soumyadip Sengupta. Robust high-resolution video matting with temporal guidance. In WACV, 2022. 1, 2, 3, 4, 6, 7, 13, 16, 18, 20, 21, 22

- [34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 6, 13
- [35] Wenze Liu, Zixuan Ye, Hao Lu, Zhiguo Cao, and Xiangyu Yue. Training matting models without alpha labels. arXiv preprint arXiv:2408.10539, 2024. 2, 4, 5, 8
- [36] I Loshchilov. Decoupled weight decay regularization. In ICLR, 2019. 12
- [37] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In ICCV, 2019. 3
- [38] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In ICCV, 2019. 2
- [39] Yu Qiao, Yuhao Liu, Xin Yang, Dongsheng Zhou, Mingliang Xu, Qiang Zhang, and Xiaopeng Wei. Attention-guided hierarchical structure aggregation for image matting. In CVPR,

2020. 3, 6

- [40] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3
- [41] Christoph Rhemann, Carsten Rother, Jue Wang, Margrit Gelautz, Pushmeet Kohli, and Pamela Rott. A perceptually motivated online benchmark for image matting. In CVPR,

2009. 6, 7

- [42] Hongje Seong, Junhyuk Hyun, and Euntai Kim. Kernelized memory network for video object segmentation. In ECCV,

2020. 3

- [43] Xiaoyong Shen, Xin Tao, Hongyun Gao, Chao Zhou, and Jiaya Jia. Deep automatic portrait matting. In ECCV, 2016. 3
- [44] Yanan Sun, Guanzhi Wang, Qiao Gu, Chi-Keung Tang, and Yu-Wing Tai. Deep video matting via spatio-temporal alignment and aggregation. In CVPR, 2021. 13
- [45] Pavel Tokmakov, Karteek Alahari, and Cordelia Schmid. Learning video object segmentation with visual memory. In ICCV, 2017. 6, 13
- [46] Haochen Wang, Xiaolong Jiang, Haibing Ren, Yao Hu, and Song Bai. Swiftnet: Real-time video object segmentation. In CVPR, 2021. 3
- [47] Xintao Wang, Kelvin C.K. Chan, Ke Yu, Chao Dong, and Chen Change Loy. EDVR: Video restoration with enhanced deformable convolutional networks. In CVPRW, 2019. 3
- [48] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional video synthesis with motion controllability. In NeurIPS, 2024. 3
- [49] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, and Ziwei Liu. LaVie: High-quality video generation with cascaded latent diffusion models. In IJCV,

2024. 3

- [50] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-A-Video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 3
- [51] Haozhe Xie, Hongxun Yao, Shangchen Zhou, Shengping Zhang, and Wenxiu Sun. Efficient regional memory network for video object segmentation. In CVPR, 2021. 2, 3
- [52] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In ICCV, 2019. 6, 13
- [53] Shuai Yang, Yifan Zhou, Ziwei Liu, , and Chen Change Loy. Rerender A Video: Zero-shot text-guided video-tovideo translation. In SIGGRAPH Asia, 2023. 3
- [54] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. In NeurIPS, 2021. 14
- [55] Jingfeng Yao, Xinggang Wang, Shusheng Yang, and Baoyuan Wang. ViTMatte: Boosting image matting with pre-trained plain vision transformers. Information Fusion,

2024. 3

- [56] Jingfeng Yao, Xinggang Wang, Lang Ye, and Wenyu Liu. Matte Anything: Interactive natural image matting with segment anything model. Image and Vision Computing, page 105067, 2024. 3, 17, 19
- [57] Yunke Zhang, Lixue Gong, Lubin Fan, Peiran Ren, Qixing Huang, Hujun Bao, and Weiwei Xu. A late fusion cnn for digital matting. In CVPR, 2019. 3
- [58] Chong Zhou, Xiangtai Li, Chen Change Loy, and Bo Dai. EdgeSAM: Prompt-in-the-loop distillation for on-device deployment of sam. arXiv preprint, 2023. 3
- [59] Shangchen Zhou, Jiawei Zhang, Jinshan Pan, Haozhe Xie, Wangmeng Zuo, and Jimmy Ren. Spatio-temporal filter adaptive network for video deblurring. In ICCV, 2019. 3
- [60] Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan, and Chen Change Loy. ProPainter: Improving propagation and transformer for video inpainting. In ICCV, 2023. 3
- [61] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-A-Video: Temporalconsistent diffusion model for real-world video superresolution. In CVPR, 2024. 3
- [62] Bingke Zhu, Yingying Chen, Jinqiao Wang, Si Liu, Bo Zhang, and Ming Tang. Fast deep matting for portrait animation on mobile phone. In ACMMM, 2017. 3
- [63] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. In NeurIPS,

2024. 3

## Appendix

In this supplementary material, we provide additional discussions and results to supplement the main paper. In Section G, we present the network details of our MatAnyone. In Section H, we discuss more training details, including training schedules, training augmentations, and loss functions. In Section I, we provide more details on our new training and testing datasets, including the generation pipeline and some examples for demonstration. We present comprehensive results in Section J to further show our performance, including those for ablation studies and qualitative comparisons. It is noteworthy that we also include a demo video (Section J.6) to showcase a Hugging Face demo and additional results on real-world cases in video format.

### Contents

- 1. Introduction 2
- 2. Related Work 3
- 3. Methodology 3

- 3.1. Consistent Memory Propagation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2. Core-area Supervision via Segmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.3. Recurrent Refinement During Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

4. Data 5

- 4.1. Training Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 5. Experiments 6

- 5.1. Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 5.1.1 Quantitative Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.1.2 Qualitative Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 5.2. Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 6. Conclusion 8

- 4.2. Synthetic Benchmark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.3. Real-world Benchmark and Metric . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

###### G. Architecture 12

- G.1. Network Designs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

H. Training 12

- H.1. Training Schedules . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- H.2. Training Augmentations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- H.3. Loss Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

###### I. Dataset 14

- I.1 . New Training Dataset - VM800 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- I.2 . New Test Dataset - YouTubeMatte . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- I.3 . Real Benchmark and Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

###### J. More Results 17

- J.1. Enhancement from New Training Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- J.2. Effectiveness of Consistent Memory Propagation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- J.3. Effectiveness of New Training Scheme . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- J.4. Effectiveness of Recurrent Refinement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- J.5. More Qualitative Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- J.6. Demo Video . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

### G. Architecture

- G.1. Network Designs

As illustrated in Fig. 3 in the main paper, our MatAnyone mainly has five important components: (1) an encoder for key and query transformation, (2) a consistent memory propagation module for pixel memory readout, (3) an object transformer [12] for memory grouping by object-level semantics, (4) a decoder for alpha matte decoding, (5) a value encoder for alpha matte encoding, which is used to update the alpha memory bank.

Encoder. We adopt ResNet-50 [16] for encoder following common practices in memory-based VOS [8, 10, 12]. Discarding the last convolution stage, we take ×16 downsampled feature as Ft for key and query transformation, while features at scales ×8, ×4, ×2, and ×1 are used as skip connections for the decoder.

Consistent Memory Propagation. The process of consistent memory propagation is detailed in Fig. 3(b) in the main paper. Alpha memory bank serves as the main working memory for past information query as in [8, 12], which is updated every rth frame across the whole time span. The query of the current frame to the alpha memory bank is implemented in an attention manner following [8, 12]. For the query QHW×C 1 and alpha memory bank KTHW×C, V THW×C

v 2, the affinity matrix A ∈ [0,1]HW×THW of the query to alpha memory is computed as:

Aij =

exp(d(Qi,Kj)) z exp(d(Qi,Kz))

, (6)

where d(·,·) is the anisotropic L2 function, H and W are the height and width at ×16 downsampled input scale, and T is the number of memory frames stored in alpha memory bank. The queried values Vtm in Fig. 3(b) in the main manuscript is obtained as:

Vtm = AVm. (7) In addition to that, we also maintain last frame memory solely for the uncertainty prediction module we propose, and it is updated every frame. The boundary-area prediction module is lightweight with one 1 × 1 convolution and two 3 × 3 convolutions. By taking the input of a concatenation of current frame feature Kt, last frame feature Kt−1, and last alpha matte prediction Mt−1, it outputs a one-channel change probability mask Ut of each query token, where higher Ut indicates such token is likely to change more in the alpha value compared with Mt−1. As mentioned in Sec. 3.1 in the manuscript, the ground truth Ut label is obtained by: UtGT : |MtGT−1 − MtGT| >= δ, where δ is set at 0 for segmentation data, and 0.001 for matting data as noise tolerance. Since Ut is predicted at a ×16 downsampled scale in the memory space, the ground truth mask UtGT is also downsampled in the mode of area.

Object Transformer. Our object transformer is derived from Cutie [12] with three consecutive object transformer blocks. Pixel memory readout Pt obtained from the consistent memory propagation module is then grouped through several attention layers and feed-forward networks. In this way, the noise brought by low-level pixel matching could be effectively reduced for a more robust matching against distractors. We do not claim contributions for this module.

Decoder. Our decoder is inspired by common practices in VOS [8, 12] with modified designs specifically for the matting tasks. The mask decoder is VOS generally consists of two interactive upsampling from ×16 to ×4, and then a bilinear interpolation is applied to the input scale. However, since the boundary region for an alpha matte requires much more precision than a segmentation mask, we enrich the decoder with two more upsampling layers until ×1, where skip connections from the encoder are applied at each scale to enhance the boundary precision.

Value Encoder. Similar to the encoder, we adopt ResNet-18 [16] for value encoder following common practices in memorybased VOS [8, 10, 12]. Different from the encoder for key and query, the value encoder takes the predicted alpha matte Mt as well as the image features as input, the encoded values are then used to update the alpha memory bank and last frame memory according to their updating rules.

- H. Training

##### H.1. Training Schedules

- Stage 1. To initialize our model on memory propagation learning, we train with our new video matting data VM800, which is of larger scale, higher quality, and better diversity than VideoMatte240K [32]. We use the AdamW [36] optimizer with a learning rate of 1 × 10−4 with a weight decay 0.001. The batch size is set to 16. We train with a short sequence length of

- 3 for 80K first, and then we train with a longer sequence length of 8 for another 5K for more complex scenarios. Video and

- 1We ignore the subscript t in Qt for simplicity
- 2We ignore the subscript m in Km and Vm for simplicity

- Table 4. Training settings and losses used in different training stages. † indicates that segmentation loss is computed as an auxiliary loss on a segmentation head, which will be abandoned during inference. Other than that, matting loss and core supervision loss are computed on the matting head for semantic stability in core regions and matting details in the boundary region.

Training Stage #Iterations Matting Data Segmentation Data Sequence Length Matting Loss Segmentation Loss† Core Supervision Loss

- Stage 1 85K video image & video 3 (80K) → 8 (5K) ✓ ✓

- Stage 2 40K video image & video 8 ✓ ✓ ✓

- Stage 3 5K image image & video 8 ✓ ✓ ✓

image segmentation data COCO [34], SPD [45] and YouTubeVIS [52] are used to train the segmentation head parallel to the matting head at the same time, as previous practices [21, 31, 33].

- Stage 2. We apply our key training strategy - core-area supervision in this stage. On the basis of the previous stage, we add additional supervision on the matting head with segmentation data to enhance the semantics robustness and generalizability towards real cases. In this stage, the learning rate is set to be 1 × 10−5, and we train with a sequence length of 8 for 40K for both matting and segmentation data.
- Stage 3. Due to the inferior quality of video matting data compared with image matting data annotated by humans, we finetune our model with image matting data instead for 5K with a 1 × 10−6 learning rate. Noticeable improvements in matting details, especially among boundary regions, could be seen after this stage.

##### H.2. Training Augmentations

Augmentations for Training Data. As discussed in the manuscript, video matting data are deficient in quantity and diversity. In order to enhance training data variety during the composition process, we follow RVM [33] to apply motion (e.g., affine translation, scale, rotation, etc.) and temporal (e.g., clip reversal, speed changes, etc.) augmentations to both foreground and background videos. Motion augmentations applied to image data also serve to synthesize video sequences from images, making it possible to fine-tune with higher-quality image data for details.

Augmentations for Given Mask. Since our setting is to receive the segmentation mask for the first frame and make alpha matte prediction for all the frames including the first one, it is important to have our model robust to the given mask. To generate the given mask in the training pipeline, we first obtain the original given mask. For segmentation data, it is just the ground truth (GT) for the first frame, while for matting data, it is the binarization result on the first-frame GT alpha matte, with a threshold of 50. Erosion or dilation is then applied with a probability of 40% each, with kernel sizes ranging from 1 to 5. In this way, we force the model to learn alpha predictions based on an inaccurate segmentation mask, also enhancing the model robustness towards memory readout if it is not so accurate during the predictions in following frames.

Augmentations for Assigned Object(s). The assignment of target object(s) as a segmentation mask for the first frame gives us flexibility for instance video matting. Given the strong prior, the model is still easy to be confused by other salient humans not assigned as target. To solve this, we find that a small modification in the video segmentation data pipeline has an obvious effect. In YouTubeVIS [52], for each video with human existence, suppose the number of human instances is H. Instead of combining all of them as one object (practice in previous auxiliary-free methods [33]), we randomly take h ≤ H instance as foreground, while unchosen instances are marked as background. In this way, we force the model to distinguish the target human object(s) even when other salient human object(s) exist, enhancing the robustness in object tracking for instance video matting even without instance mask for each frame as MaGGIe [22] has.

##### H.3. Loss Functions

Given that we take the first-frame segmentation mask alongside with input frames as input, our model needs to predict alpha matte starting from the first frame, which is different from VOS methods [8, 12]. In addition, since we also apply mask augmentation on the given segmentation mask, the prediction from the segmentation head should also start from the first frame. As a result, we need to apply losses on all t ∈ [0,N] frames for both matting and segmentation heads.

There are mainly three kinds of losses involved in our training: (1) matting loss Lmat; (2) segmentation loss Lseg; (3) core supervision (CS) loss Lcs, and their usages in different training stages are summarized in Table 4.

Matting Loss. For frame t, suppose we have the predicted alpha matte Mt w.r.t. its ground-truth (GT) MtGT. We follow RVM [33] to employ L1 loss for semantics Ll1, pyramid Laplacian loss [17] for matting details Llap, and temporal coherence loss [44] Ltc for flickering reduction:

Ll1 = ∥Mt − MtGT∥1, (8)

5

2s−1 5 ∥Lspyr(Mt) − Lspyr(MtGT)∥1, (9)

Llap =

s=1

dMt dt −

dMtGT

dt ∥2, (10) The overall matting loss is summarized as:

Ltc = ∥

Lmat = Ll1 + 5Llap + Ltc. (11)

Segmentation Loss. For frame t, suppose we have the predicted segmentation mask St w.r.t. its ground-truth (GT) StGT from the segmentation head. We employ common losses used in VOS [8, 12, 54], Lce and Ldice.

Lce = StGT(−log(St)) + (1 − StGT)(−log(1 − St)), (12)

2StStGT + 1 St + StGT + 1

. (13)

Ldice = 1 −

The overall segmentation loss is summarized as:

###### Lseg = Lce + Ldice. (14)

Core Supervision Loss. For core-area supervision, we combine the region-specific losses: Lcore for core region and Lboundary for boundary region as defined in Sec. 3.2 in the manuscript, and the overall core supervision loss is summarized as:

Lcs = Lcore + 1.5Lboundary. (15)

### I. Dataset

- Table 5. Comparison on Datasets. We compare our new training data and testing data with the old ones, in terms of the number of distinct foregrounds, sources, and whether harmonization is applied.

Datesets VideoMatte240K (old train) [32] VM800 (new train) VideoMatte (old test) [32] YouTubeMatte (new test) #Foregrounds 475 826 5 32 Sources - Storyblocks, Envato Elements, Motion Array - YouTube Harmonized - - x ✓

##### I.1. New Training Dataset - VM800

Overview. As summarized in Table 5, our new training dataset VM800 has almost twice the number of foreground videos than VideoMatte240K [32] in quantity. To enhance diversity and data distribution, our foreground green screen videos are downloaded from a total of three video footage websites: Storyblocks, Envato Elements, and Motion Array, and thus enjoy a diversity in hairstyles, outfits, and motion. In addition, we ensure the high quality of our VM800 dataset in fine detail and through careful manual selection.

Generation Pipeline. We employ Adobe After Effects in our data generation pipeline to extract alpha channels from green screen footage videos. Since the amount of green screen footage to be processed is huge, we would like to obtain the preliminary results with an automatic pipeline. We first use Keylight and set Screen Color to be the pixel value taken from the upper left corner for each frame. To obtain a clean alpha matte, we clip the values smaller than 20 to be 0 and those larger than 80 to be 255. To further enhance the alpha matte quality, we post-process with another two keying effects Key Cleaner and Advanced Spill Supressor, which are generally used together following Keylight. Since we are processing a video, we also turn on reduce chatter in Key Cleaner to reduce flickering in the boundary region. For batch processing, we compile the above process into a Javascript and XML file for After Effects to run with, and obtain a large batch of preliminary results for manual selection.

Keylight

- - Screen Color: pixel value of upper left corner
- - Screen Matte:
- - Clip Black: 20
- - Clip White: 80

Key Cleaner

- - radius: 1
- - reduce chatter: check

Advanced Spill Supressor

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

| | |
|---|---|
| | |

(a) Errors in reflective regions (e.g., glasses) (b) Inhomogeneous in core regions (e.g., shadow)

- Figure 8. Issues with VideoMatte240K [32]. (a) Errors in alpha values exist in reflective regions (e.g., “a hole” on glasses). (b) Inhomogeneous alpha values exist in core regions (e.g., caused by shadow), where the alpha value should be exactly 0 or 1.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

(a) Errors in reflective regions (e.g., glasses) (b) Inhomogeneous in core regions (e.g., shadow)

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

- Figure 9. Gallery for our new training dataset VM800. High-quality details in the boundary regions and diversity in terms of gender, hairstyles, and aspect ratios could be clearly observed.

Quality - Fine Details. The green screen foreground videos we downloaded are almost in a 4K quality, and we also place a higher priority on those videos with more details (e.g., hair) in our download choice. Fig. 9 shows the fine details in our VM800 dataset.

Quality - Careful Manual Selection. We notice that alpha mattes extracted with After Effects from green screen videos often encounter inhomogeneities in core regions. For example, reflective regions in the foreground will result in a near-zero value (i.e., a hole) in the alpha matte, as shown in Fig. 8(a). In addition, noise also exists in the green screen background, resulting in the fact the alpha values may not homogeneously equal 0, which should not be the case in the core region.

Similarly, for foregrounds, colors that are similar to the background green, or shadow in the foreground, may also result in the alpha values not homogeneously equal to 1 in the core foreground region, making the alpha matte look noisy, as shown in Fig. 8(b). Since VideoMatte240K [32] is also obtained with After Effects, we observe that alpha mattes with the above problems still exist, and thus taking such wrong ground truth for training will inevitably lead to problematic inference results (Fig. 11(a)). As a result, we conduct careful manual selection to examine all our processed alpha mattes, and leave out those with the above problems. As shown in Fig. 11(a), training with our VM800 will not lead to such problematic results.

##### I.2. New Test Dataset - YouTubeMatte

Overview. As summarized in Table 5, our new synthetic benchmark YouTubeMatte has over six times larger than the number of distinct foreground videos in VideoMatte [32], making it a much more representative benchmark for evaluation with better diversity. In addition, the green screen videos for foregrounds are downloaded from YouTube at a scale of 1920 × 1080 with rich boundary details, thus enhancing its ability to discern matting precision in boundary regions. While the generation pipeline for YouTubeMatte is almost the same as that for VM800, harmonization [23], however, is applied when compositing the foreground on a background. Such an operation effectively makes YouTubeMatte a more challenging benchmark that is closer to the real distribution. As shown in Fig. 10, while RVM [33] is confused by the harmonized frame, our method still yields robust performance.

[Figure 162]

[Figure 163]

[Figure 164]

BeforeAfter

Harmonization

[Figure 165]

[Figure 166]

[Figure 167]

Video Frame RVM Ours

Figure 10. Harmonization on synthetic benchmarks and its effect on model performance. Harmonization [23] is an operation that makes the composited frame more natural and realistic, which also effectively makes our YouTubeMatte a more challenging benchmark that is closer to the real distribution. It is observed that while RVM [33] is confused by the harmonized frame, our method still yields robust performance.

##### I.3. Real Benchmark and Evaluation

Overview. As a technique towards real-world applications (e.g., virtual background in the online meeting), the synthetic benchmark is not enough to test the generalizability of video matting models. Although there are countless of real human videos for testing in the wild, the lack of GT alpha mattes makes them hard to serve as a real benchmark. Here, we select a subset of 25 real-world videos from [33], where a consecutive of 100 frames for each video are selected with no scene transition, to form our real benchmark. According to our definitions in Fig. 2(a) in the manuscript, we could also divide the evaluation metrics for core regions and for boundary separately, making evaluation for real benchmarks feasible.

Evaluation on Core Regions. Thanks to the recent success of VOS methods [8, 12], frame-wise segmentation masks could be generated with high precision. Here, we employ Cutie [12] for video segmentation results. We first obtain the trimap for each segmentation mask by applying dilation and erosion (with kernel size 21), and then compute the core mask where trimap values equal 0 or 1. In this way, the values of a segmentation mask within its core region could be considered as the GT alpha values for the core region, where common metrics including MAD and MSE for semantic accuracy, and dtSSD [14] for temporal coherency could be applied for evaluation.

### J. More Results

##### J.1. Enhancement from New Training Data

As discussed in Sec. 4.1 in the manuscript and Section I.1 in the supplementary, our new training data VM800 is upgraded in quantity, quality, and diversity. In addition to the quantitative evaluation in Tab. 3 in the manuscript, we further show the enhancement from new training data by providing more results when comparing the model trained with VideoMatte240K [32] and the model trained with our VM800 in Fig. 11(a).

Video Frame Old Training Data New Training Data Video Frame w/o Core Supervision w/ Core Supervision

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

|Errors in reflective objects|
|---|

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

|Inhomogeneous core regions|
|---|

[Figure 186]

[Figure 187]

(a) Enhancement from New Training Data (b) Effectiveness of New Training Scheme

- Figure 11. (a) Comparison on results trained with old training data (VideoMatte240K [32]) and new training data (our VM800). It could be observed that training with old data will lead to errors in reflective objects (e.g., holes on the sunglasses) and inhomogeneous alpha values in the core regions. However, both issues are fixed when training with our new data, indicating a higher quality. (b) Comparison on results trained without and with core-area supervision. It could be observed that training without it will lead to semantics error due to the weak supervision from real segmentation data, while training with core supervision largely improves semantics accuracy thanks to the stronger supervision enabled.

##### J.2. Effectiveness of Consistent Memory Propagation

As one of our key designs, the consistent memory propagation (CMP) module improves both stability in core regions and quality in boundary details. In addition to the quantitative evaluation in Tab. 3 in the manuscript, we give more qualitative results and analysis in Fig. 12.

##### J.3. Effectiveness of New Training Scheme

Our new training scheme introduces core-area supervision, which largely enhances the semantic accuracy and stability, as shown in Tab. 3 in the manuscript. More qualitative results are shown in Fig. 11(b) for better visualization of its effects.

##### J.4. Effectiveness of Recurrent Refinement

As discussed in Sec. 3.3 in the manuscript, the sequential prediction in the memory-based paradigm enables recurrent refinement without the need for retraining during inference. By repeating the first frame n times and iteratively updating the first frame prediction based on the last-time prediction, the quality of the first frame alpha matte could be recurrently refined. We show in Fig. 13 that such recurrent refinement can not only (1) enhance the robustness to the given segmentation mask even when it is of low quality, but also (2) achieve matting details at an image-matting level when compared with an image matting method (i.e., Matte Anything [56] in the last column).

t t+40 t+80 t+120 t+160

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

w/CMPChangeProb.w/oCMPVideoFrames

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

| |
|---|

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|
|---|---|---|---|---|

- Figure 12. Comparison on results with and without Consistent Memory Propagation. It could be observed that when CMP is not applied, semantic errors constantly exist across a wide span of video frames. However, when training with CMP, we observe from the “Change Probability” mask that usually our model only takes pixels near the boundary as “changed”, and most of the inner regions (i.e., earring) will mainly take the memory values from the last frame. As we can see on the figure, while predictions are both correct at time t, the model with CMP successfully keeps the correctness and gives stable results, while the model without CMP quickly breaks the correctness and never recovers.

##### J.5. More Qualitative Comparisons

In this subsection, we provide additional visual comparisons of our method with the state-of-the-art methods, including auxiliary-free (AF) method: RVM [33] and mask-guided methods: FTP-VM [21], and MaGGIe [22]. Fig. 14 presents the general video matting results on real videos. To further demonstrate the superiority of our model, Fig. 15 and Fig. 16 both showcase a challenging case respectively, where other methods mostly fail. In addition, Fig. 17 demonstrates the instance matting results compared with MaGGIe [22], a method with instance mask for each frame is given as guidance, while our model only has the segmentation mask for the first frame as guidance.

##### J.6. Demo Video

We also offer a demo video. This video showcases more video matting results and a hugging face demo for applicability, both on real-world videos.

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

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

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

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

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

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Video Frame Segmentation Mask !! = 1 !! = 5 !! = 10 Image Matting (Matte Anything)

- Figure 13. Comparison on results with iterative refinement. A noticeable enhancement on details can be observed even with one iteration of refinement compared with the given segmentation mask. Within 10 iterations, our model is able to achieve matting details at an image-matting level, even better than Matte Anything [56], which is an image matting model also based on the results from SAM [25].

Video Frame RVM FTP-VM MaGGIe Ours

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

- Figure 14. More qualitative comparisons on general video matting with SOTA methods. We compare our MatAnyone with both auxiliary-free (AF) method: RVM [33] and mask-guided methods: FTP-VM [21], and MaGGIe [22]. It could be observed that our method significantly outperforms others in both detail extraction and semantic accuracy, across diverse and complex real scenarios. It is noteworthy that although sometimes MaGGIe [22] seems to give acceptable results when compositing with a green screen, its alpha matte turns out to be noisy (i.e., inhomogeneous in the core foreground region and blurry in the boundary region), while our alpha matte is clean with fine-grained details in the boundary region. As a result, we also include alpha mattes for a more comprehensive comparison. (Zoom in for best view)

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

VideoFramesMaGGIeOursRVMFTP-VM

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

- Figure 15. A challenging example of general video matting across a long time span. We compare our MatAnyone with both auxiliaryfree (AF) method: RVM [33] and mask-guided methods: FTP-VM [21], and MaGGIe [22]. It could be observed that our model is able to track the target object stably even when the object is moving fast in a highly complex scene, where all the other methods present noticeable failures. (Zoom in for best view)

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

VideoFramesMaGGIeOursRVMFTP-VM

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

- Figure 16. Another challenging example of general video matting across a long time span. We compare our MatAnyone with both auxiliary-free (AF) method: RVM [33] and mask-guided methods: FTP-VM [21], and MaGGIe [22]. This example showcases that our model is able to track the target objects even in a highly ambiguous background, where the colors for foreground and background are similar, and also multiple humans in the background. In addition, it also demonstrates when there is more than one target object, our model is still able to handle this challenging case well. (Zoom in for best view)

[Figure 312]

| |
|---|

[Figure 313]

| |
|---|

[Figure 314]

| |
|---|

[Figure 315]

| |
|---|

[Figure 316]

| |
|---|

[Figure 317]

| |
|---|

[Figure 318]

| |
|---|

[Figure 319]

| |
|---|

| |[Figure 320]|
|---|---|
| | |

|[Figure 321]| |
|---|---|

#1

#2

#1

#2

[Figure 322]

|#2|
|---|

#1

#3

Video Frame MaGGIe (#1) MaGGIe (#2) Ours (#1) Ours (#2) Video Frame Instance #1 Instance #2 Instance #3

[Figure 323]

| |
|---|

[Figure 324]

| |
|---|

[Figure 325]

| |
|---|

[Figure 326]

| |
|---|

[Figure 327]

| |
|---|

[Figure 328]

OursMaGGIe

- Figure 17. More qualitative comparisons on instance matting. We compare our MatAnyone with MaGGIe [22], a mask-guided method that requires the instance mask for each frame, while our method only requires the mask for the first frame. It could be observed that even with such strong given prior, MaGGIe still performs below our method in terms of semantic accuracy in the core regions. Moreover, in terms of the boundary regions, by examining the details there, we could clearly observe that the details generated by MaGGIe are blurry and far from fine-grained compared with our results. (Zoom in for best view)

