## VidEoMT: Your ViT is Secretly Also a Video Segmentation Model

Narges Norouzi1 Idil Esen Zulfikar2,† Niccol`o Cavagnero1,† Tommie Kerssies1 Bastian Leibe2 Gijs Dubbelman1 Daan de Geus1

1Eindhoven University of Technology 2RWTH Aachen University

# arXiv:2602.17807v3[cs.CV]4Mar2026

### Abstract

Existing online video segmentation models typically combine a per-frame segmenter with complex specialized tracking modules. While effective, these modules introduce significant architectural complexity and computational overhead. Recent studies suggest that plain Vision Transformer (ViT) encoders, when scaled with sufficient capacity and large-scale pre-training, can conduct accurate image segmentation without requiring specialized modules. Motivated by this observation, we propose the Video Encoderonly Mask Transformer (VidEoMT), a simple encoder-only video segmentation model that eliminates the need for dedicated tracking modules. To enable temporal modeling in an encoder-only ViT, VidEoMT introduces a lightweight query propagation mechanism that carries information across frames by reusing queries from the previous frame. To balance this with adaptability to new content, it employs a query fusion strategy that combines the propagated queries with a set of temporally-agnostic learned queries. As a result, VidEoMT attains the benefits of a tracker without added complexity, achieving competitive accuracy while being 5×–10× faster, running at up to 160 FPS with a ViT-L backbone. Code: https://www.tue-mps.org/videomt/.

### 1. Introduction

The video segmentation task involves segmenting and classifying objects in each frame, while also matching those objects across frames. As such, a video segmentation model should have the capability to localize objects, classify them, and track them across different frames. For this reason, great progress has been made through the introduction of specialized neural network components that are designed to improve one or more of these capabilities. Current methods obtain state-of-the-art performance by combining many such specialized components within increasingly complex models [21, 39–41], building upon years of prior work. This trend of increasing complexity motivates us to explore

†Equal contribution.

AveragePrecision(AP)

> 10× faster

70

62

54

ViT-L ViT-B ViT-S

CAVIS

46

EoMT + CAVIS

###### VidEoMT (Ours)

20 40 80 160 320

Frames per second (FPS) [log scale]

Figure 1. CAVIS vs. VidEoMT (Ours). VidEoMT is much faster than both CAVIS [21] and a combination of EoMT [19] and CAVIS, while maintaining competitive AP across different sizes of DINOv2 [30]. Evaluated on YouTube-VIS 2019 val [38].

whether such complexity is necessary, or if this task can be solved with similar accuracy using a simpler approach.

In this paper, we hypothesize that a simpler approach can match the accuracy of more complex models by making use of powerful vision foundation models (VFMs). These VFMs [14, 30, 33], which typically adopt the Vision Transformer (ViT) architecture [13], are pre-trained on large amounts of data, and have proven to be solid foundations for subsequent finetuning for downstream tasks. For video segmentation, VFMs like DINOv2 [30] are incorporated in most recent models [21, 40, 41] by being extended with many specialized components. However, we believe that these strong pre-trained ViTs can learn to take over many of the functionalities of the specialized components that are typically added on top, making these components redundant. This is inspired by Kerssies et al. [19], who showed that state-of-the-art image segmentation can be achieved by simply adding a few learnable queries to a large pre-trained ViT with a model called EoMT, without needing specialized components. EoMT demonstrates that a large, pretrained ViT encoder can learn to effectively localize and classify objects, which is also required for video segmentation. However, video segmentation has an additional requirement: temporally tracking objects across frames.

##### Current State-of-the-Art Video Segmentation Methods

##### VidEoMT (Ours)

5 × −10 × faster

Track Queries t − 1

Learnable Queries

Frame t

Frame t

FeaturesMasks

Track Queries t − 1

CA Feature Extractor

ReID Layers

Pixel Decoder

TF Decoder

TF Blocks

ViT Adapter

ViT

Learnable

Queries

tQueries

Seg.

Track Queries t

Track Queries t

Masks + Classes

Masks + Classes

Segmenter Tracker

- Figure 2. Current State-of-the-Art Video Segmentation Methods vs. VidEoMT (Ours). We compare the architectures of current state-of-the-art video segmentation methods – using CAVIS [21] as a representative example – and our encoder-only VidEoMT method. VidEoMT streamlines the video segmentation framework, relying on the power of large-scale pre-training with vision foundation models rather than handcrafted task-specific components. TF means Transformer and CA means context-aware.

We expect that the pre-trained ViT encoders from VFMs will also be able to learn to track objects because of the training objectives of these VFMs. For instance, DINOstyle models [2, 30, 33] employ training objectives that promote consistent feature representations for a given object across different views. Cross-view consistent features are crucial for tracking, as they allow for the identification of the same object in different frames. This makes pre-trained ViTs from VFMs highly suited for video segmentation.

We verify our hypothesis about the redundancy of complex components in video segmentation models by taking state-of-the-art video segmentation models [21, 40, 41] and evaluating the effect of removing their specialized modules. These existing models all follow roughly the same paradigm: they first employ a segmenter, which predicts frame-level segmentation masks and class labels and outputs object-level feature queries, and then apply a tracker to match object-level feature queries across different video frames. As shown in Fig. 2 (left) for the example of CAVIS [21], both the segmenter and the tracker consist of many specialized components. We first replace the complex segmenter with EoMT [15], followed by a step-by-step removal of specialized tracking modules.

Next, we move away from the conventional decoupling of segmenter and tracker, and instead explore whether temporal modeling can be conducted within a ViT encoder. To this end, we introduce a lightweight approach based on (1) query propagation, where object-level queries are carried across frames to enable temporal modeling in an encoderonly framework, and (2) query fusion, which combines propagated queries with learned queries to allow the identification of newly appearing objects. This leads to the design of the Video Encoder-only Mask Transformer (VidEoMT), which unifies segmentation and temporal association within the ViT, as illustrated in Fig. 2 (right).

By no longer requiring complex specialized components

and performing all computations within a single ViT-style model, VidEoMT is remarkably efficient. Through experiments, we find that VidEoMT with a ViT-Large backbone is over 10× faster than existing state-of-the-art methods on the YouTube-VIS [38] benchmarks, achieving processing speeds of up to 160 FPS, as shown in Fig. 1. Importantly, this speed is obtained while maintaining a comparable accuracy. These findings are further validated on the VIPSeg [28] and VSPW [27] benchmarks, where VidEoMT consistently achieves speedups of 5×–10× with negligible impact on accuracy. Such speed-up factors can be a veritable game changer for applications, enabling online video processing across a wide range of use cases. These results also validate our hypothesis that a large, extensively pretrained ViT can take over the functionalities of specialized components to conduct accurate video segmentation, without requiring additional complex components.

In summary, we make the following contributions:

- • We propose VidEoMT, a simple and highly efficient architecture for video segmentation, that unifies segmentation and temporal association within a single ViT encoder.
- • Using VidEoMT, we demonstrate that a sufficiently large, pre-trained ViT can learn to take over the functionality of specialized components for video segmentation.
- • We show that VidEoMT, with its simple encoder-only architecture, can achieve accuracies comparable to the state of the art while being up to 10× faster.

### 2. Related Work

Image Segmentation. Image segmentation requires that objects in an image are segmented and classified. Early image segmentation models treated this task as a per-pixel classification problem, predicting a class label for each pixel [4, 5, 23]. Later works propose an alternative mask classification approach, where a model predicts a segment –

consisting of a segmentation mask and class label – for each object in the image [8]. These mask classification methods typically make use of Mask Transformers, which use image features from a backbone and learnable queries to predict a segmentation mask and class label for each query with a Transformer decoder [3, 9, 18, 35]. Recently, EoMT [19] has demonstrated that it is possible to conduct accurate image segmentation without a decoder or other task-specific components, by simply feeding the learnable queries directly into a large, pre-trained ViT. In this work, inspired by EoMT, we investigate whether video segmentation models can be simplified in a similar manner, with the goal of improving efficiency while preserving high accuracy.

Video Segmentation. Video segmentation is a wellestablished computer vision task, encompassing video instance segmentation (VIS) [38], video panoptic segmentation (VPS) [20], and video semantic segmentation (VSS) [29], where the primary objective is to segment, classify, and track all objects of interest in a video. Current VIS, VPS, and VSS methods typically use Mask Transformerbased architectures [7, 16, 17, 21, 32, 37, 39–41]. They extend Mask Transformers for image segmentation [9] into the video domain by incorporating specialized tracking components or enhancing temporal representations. The most recent methods [17, 21, 39–41] are universal models, which can handle VIS, VPS, and VSS within a single framework. These models follow a decoupled paradigm, where the segmentation and tracking sub-tasks are separated. First, a segmenter conducts image segmentation for each frame, and then a tracker associates these segmented objects over time. Generally, both the segmenter and the tracker contain various specialized components, which increase accuracy but reduce efficiency. In this work, we analyze these universal video segmentation models and demonstrate that they can be simplified to an encoder-only design, significantly improving efficiency while achieving competitive accuracy.

### 3. Method

#### 3.1. Task Definition

We consider the task of online video segmentation, where the goal is to assign a class label and binary mask to each object in every frame, while also associating predictions of the same object across time steps to ensure temporal consistency. Here, we use the term “object” broadly to refer to either object instances (as in VIS), semantic classes (as in VSS), or both (as in VPS).

Formally, a video is a sequence of T frames V = {I1,I2,...,IT}. For each frame It ∈ R3×H×W with spatial resolution (H,W), a model should yield a set of Kt predictions Yt = {(mt,i,ct,i)}K

i=1, where mt,i ∈ {0,1}H×W is a binary segmentation mask, and ct,i ∈ {1,...,C} is a semantic category label from C classes. Additionally, these

t

per-frame predictions must be temporally associated across frames to maintain identity consistency. That is, each prediction (mt,i,ct,i) at time t should be matched to a corresponding prediction in a previous frame at t−1 if they refer to the same object. The task must be solved in an online manner: at timestep t, predictions Yt may only depend on the current frame It and earlier frames {I1,...,It−1}.

#### 3.2. Preliminaries

Current state-of-the-art online video segmentation models [21, 39–41] typically decompose the video segmentation pipeline into two distinct components: a segmenter, which is responsible for generating segmentation masks and class labels for each frame, and a tracker that ensures temporal association by linking the segmenter’s predictions across frames, associating object instances over time (see Fig. 2).

Segmenter. The segmenter S generates frame-level image segmentation predictions, yielding a class label and a binary mask for each object. To obtain these predictions, stateof-the-art methods combine a pre-trained ViT [13, 30], a ViT-Adapter [6], and a Mask Transformer segmentation decoder [9]. The ViT encoder embeds an input image It into non-overlapping patch tokens and processes them with L Transformer blocks. A CNN-based ViT-Adapter augments the encoder with multi-scale features, which are fused and refined in the Mask2Former [9] head by a pixel decoder, producing a set of enriched features {F4,F8,F16,F32}, with Fi ∈ RD×(H/i)×(W/i). A Transformer decoder then updates N learnable queries Qlrn = {qlrni ∈ RD}Ni=1 through cross- and self-attention, yielding refined queries QS = {qSi ∈ RD}Ni=1. These refined queries, in combination with the feature maps F4, are used to generate the final segmentation outputs: class labels are predicted via a linear layer applied to each query, while binary masks are produced by passing the queries through a three-layer MLP followed by a dot product with the pixel-level feature maps. Tracker. The objective of the tracker T is to associate the segmenter’s predictions across frames to maintain consistent object identities over time. Rather than relying on the predicted masks and class labels from the segmenter, the tracker performs association based on the per-object query embeddings QS. Formally, the tracker T aligns the query embeddings from the current frame, QSt , with the temporally updated queries from the previous frame, QTt−1, to achieve correspondence between object instances over time, producing temporally updated queries QTt :

QTt = T QSt ,QTt−1 . (1) In practice, T consists of L Transformer blocks with crossattention, self-attention, and feed-forward layers. During cross-attention, queries QTt−1 serve as queries and QSt as keys and values, allowing the tracker to align and update the representations of identical objects across consecutive

frames. The output of the cross-attention layer is then refined by a self-attention layer, which further enhances temporal coherence among the updated queries. These operations ensure that a consistent query ordering is obtained, meaning that QTt represents the same objects as QTt−1, in the same order. The refined queries QTt can then be used to predict temporally consistent masks and class labels, following the same procedure as in the segmenter.

Context-Aware Features. To enrich query embeddings QSt with information from the local neighborhood of each object, the state-of-the-art method CAVIS [21] introduces context-aware features. Concretely, given predicted masks Mt = {mt,i}Ki=1 and features F4,t at timestep t, binary boundary maps Bt,i ∈ {0,1}(H/4)×(W/4) are extracted using a Laplacian filter. Next, the features F4,t are smoothed with an average filter, yielding FA4,t. Finally, the contextaware features QAt = {qAt,i ∈ RD}Ni=1, extracted by pooling the smoothed features at boundary pixels, are concatenated with the per-frame query embeddings QSt . This process produces an enriched set of queries QCt = {qCt,i ∈ R2D}Ni=1, which are then fed into the tracker’s Transformer blocks in place of the segmenter queries QSt .

Re-identification Layers. To further improve robustness, recent methods [21, 40, 41] employ re-identification layers. These layers are paired with contrastive training objectives to enforce similarity between embeddings of the same instance while separating those of different instances. In practice, query embeddings QSt from the segmenter are usually fed to the re-identification layers, implemented as a 3-layer MLP. In CAVIS, this MLP is instead applied to the contextaware queries QCt :

QRt = MLP QCt . (2)

This yields enhanced queries QRt which are subjected to contrastive learning and are fed into the tracker’s Trans-

former blocks, in place of the context-aware queries QCt .

#### 3.3. Removing Task-specific Components

Recently, EoMT [19] has challenged the dominant paradigm in image segmentation that relies on many specialized components, showing that this task can be performed in an encoder-only fashion, given a sufficiently large ViT model and strong pre-training. This is also relevant in the video domain, as the segmenter modules of state-ofthe-art video segmentation models also use the same specialized components. In EoMT [19], learned queries Qlrn are injected into the last L2 (usually L2 = 4) layers of a ViT encoder and processed jointly with patch tokens, yielding updated queries QS which can be used to produce segmentation predictions {(ci,mi)}Ki=1. Despite its simplicity, EoMT performs competitively with complex frameworks while greatly improving efficiency.

Inspired by this result, we explore a similar simplifica-

tion for video segmentation, where inference speed is even more critical. Our hypothesis is that a strong ViT can handle not only segmentation but also temporal association within a unified encoder-only architecture, removing the need for explicit tracking modules. To verify this, starting from the state-of-the-art CAVIS model, we first replace its heavy segmenter with EoMT, and then we progressively remove video-specific components to evaluate whether the encoder can also learn to conduct temporal association.

Replacing the Segmenter. In the current state-of-the-art video segmentation models, such as CAVIS [21], the segmenter S is composed of an inefficient ViT-Adapter [6] and a complex and resource-intensive Mask2Former [9] pixel decoder and Transformer decoder. We replace the entire segmenter with EoMT [19], which integrates query tokens directly into the ViT and predicts object representations without specialized components. This greatly simplifies the pipeline and it is expected to consistently improve inference speed, similar to the original findings for EoMT [19].

Removing Context-Aware Features. The context-aware features in CAVIS [21] explicitly encode information from the spatial neighborhood of each instance to stabilize predictions under appearance changes or occlusion. Extracting these features requires convolutional filtering over highresolution features, repeated for every query in all frames of a video, making it inefficient. We hypothesize that the auxiliary context added by these features is not strictly necessary when leveraging a strong pre-trained ViT, as its features are already fine-grained enough to be easily fine-tuned to capture specific object identity and to maintain stability under appearance changes or occlusion.

Removing Re-identification Layers. While effective, reidentification layers add complexity at both inference and training time, where the associated contrastive losses are memory-intensive and slow to optimize. We hypothesize that with large-scale pre-training, the features of the ViT encoder already contain rich instance-level information. Since the segmentation queries explicitly cross-attend to these features, they effectively inherit this instance-discriminative knowledge and preserve it across frames. Therefore, we can eliminate these layers, to not only simplify the whole pipeline but also make training more affordable.

#### 3.4. VidEoMT

After the previously described simplifications, the model consists of EoMT [19] as the segmenter combined with a simplified tracker T . The tracker ensures that a given object is represented by the same query index across frames, preserving temporal consistency. However, this comes at the cost of considerable architectural complexity and significant computational overhead.

We hypothesize that strong pre-training, e.g., with DI-

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

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Pre-trained Vision Transformer Linear Patch Embedding

Pre-trained Vision Transformer Linear Patch Embedding

Frame at t = 0 Frame at t = 1

Pos. Emb.

Pos. Emb.

Query Fusion

…

[Figure 21]

Learn. Queries

- Encoder Blocks × L1

- Encoder Blocks × L2

× L1

Encoder Blocks

[Figure 22]

Learn. Queries

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Linear

× L2

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Encoder Blocks

[Figure 31]

[Figure 32]

Track Queries 0

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Track Queries

Class & Mask Logits

Class & Mask Logits

[Figure 41]

[Figure 42]

1

- Figure 3. VidEoMT architecture. For the initial video frame at t = 0, learnable queries are concatenated to the patch tokens after the first

L1 ViT blocks. Both sets of tokens are then jointly processed in the last L2 blocks, outputting predictions and track queries. For successive frames, learnable queries and previous-frame track queries are fed to the query fusion module before being processed by the ViT blocks.

NOv2 [30], already equips the ViT encoder with representations strong enough to enable temporal association within the encoder itself with only minimal changes, without the need for specialized tracking components. Hence, we move away from the conventional decoupling of segmenter and tracker and adopt a unified encoder-only design.

Enabling temporal modeling within an encoder-only framework presents two key challenges: (i) effectively integrating information from the previous frame to maintain temporal continuity, and (ii) preserving the model’s ability to detect and recognize newly appearing objects. To address the first challenge, we introduce a query propagation mechanism that carries object-level information across frames, enabling temporal continuity within the encoder. To handle the second challenge, we propose a lightweight query fusion strategy that combines propagated queries with learnable ones, allowing the model to better detect newly appearing objects. The resulting model, which we name Video Encoder-only Mask Transformer (VidEoMT), performs temporal association without specialized tracking components, as visualized in Fig. 3.

Query Propagation. When the tracker is entirely removed, the model reduces to a purely image-level EoMT architecture that processes each frame independently. In this case, queries QS are the model’s output, and there are no queries QT , as there is no longer a tracker.

We reintroduce temporal modeling through query propagation. At timestep t = 0, we follow the standard EoMT setup and feed learnable queries Qlrn into the last L2 layers of the ViT to produce object query embeddings QS0 and

the corresponding segmentation predictions. At subsequent timesteps t > 0, instead of reusing the learnable queries, we use the track queries, i.e., the propagated queries from the previous frame QSt−1, as input to the last L2 layers of the ViT. During these timesteps, the segmentation procedure remains identical to that of EoMT, the only difference is that the propagated queries replace the learnable ones.

This strategy enables information to flow across time without additional computational cost per frame, allowing for temporal consistency across frames. However, since the ViT only receives information from the previous frame, the influence of the learnable queries Qlrn gradually diminishes, causing the model to lose the ability to recognize newly appearing objects in the video.

Query Fusion. To address this limitation, we introduce query fusion, illustrated in Figure 3. In this design, queries from the previous frame QSt−1 are first transformed by a lightweight linear layer and then combined with the original learned queries Qlrn through element-wise addition:

QFt = Linear QSt−1 + Qlrn. (3) The element-wise addition is possible because the supervision strategy guarantees that the query order remains the same across frames. This fusion ensures that the model has access to temporal context from the past through QSt−1, as well as learnable queries Qlrn to enable adaptability to new objects. By balancing temporal context propagation and adaptability to new objects, query fusion allows our encoder-only framework to conduct accurate object tracking with negligible additional architectural complexity.

Training. VidEoMT is trained using the same objective

function as Mask2Former [9]. We use the cross-entropy loss for classification and the binary cross-entropy and Dice losses for segmentation predictions. To ensure temporally consistent supervision, we follow the ground-truth matching strategy of DVIS++ [40]. Here, a ground-truth object is only matched to a query in the frame where the object first appears. In the remaining frames, the ground-truth object stays matched to this query, ensuring temporal consistency.

### 4. Experiments

Datasets and Evaluation Metrics. We evaluate VidEoMT on six major benchmarks for video segmentation: OVIS [31] and YouTube-VIS 2019, 2021, and 2022 [38] for Video Instance Segmentation (VIS), VIPSeg [28] for Video Panoptic Segmentation (VPS), and VSPW [27] for Video Semantic Segmentation (VSS). We adopt Average Precision (AP) and Average Recall (AR) metrics [38] for VIS, Video Panoptic Quality (VPQ) [20] and Segmentation and Tracking Quality (STQ) [36] for VPS, and mean IoU (mIoU) and Video Consistency (mVC) [27] for VSS.

Implementation Details. Similar to the state-of-theart models CAVIS [21] and DVIS-DAQ [41], we use a DINOv2-pretrained ViT [30] as the default backbone of VidEoMT. We adopt a batch size of 8 with 5 frames as a temporal window, using mixed precision and the AdamW optimizer [24] with a learning rate of 10−4. Following EoMT [19], we apply layer-wise learning rate decay (LLRD) [12] with a factor of 0.6 and a polynomial learning rate decay with a power of 0.9. The number of iterations and training video resolutions follow the settings of CAVIS [21] for fair comparison. We refer to the supplementary material for additional implementation details.

To assess the computational efficiency, we measure both FPS and FLOPs. FPS is reported as the average number of video frames processed per second on the validation set with a batch size of 1, evaluated on an NVIDIA H100 GPU with FlashAttention v2 [10] and torch.compile [1] (default settings) enabled. FLOPs are calculated using fvcore [26], averaging over all images in the validation set. We report the results in GFLOPs, i.e., FLOPs ×109.

### 5. Results 5.1. Main Results

From CAVIS to VidEoMT. In Tab. 1, we report a stepwise transformation from state-of-the-art video segmentation method CAVIS [21] to our proposed VidEoMT. We gradually remove specialized tracking components to obtain the lightweight EoMT baseline, and we then introduce modifications to EoMT to support tracking. For details of the architectures at different steps, see the supplementary.

In step (1), we find that replacing the segmenter with EoMT [19] improves FPS by almost 3×, while AP drops

Step Method AP Params GFLOPs FPS

- (0) CAVIS [21] 68.9 358M 838 15
- (1) w/ EoMT as Segmenter 68.1 328M 699 42

- (2) w/o Context-aware Features 68.4 327M 581 72

- (3) w/o Re-identification Layers 68.0 326M 580 74

- (4) w/o Tracker = EoMT 61.3 316M 565 162

– EoMT [19] 61.3 316M 565 162

- (5) w/ Query Propagation 63.9 316M 565 162

- (6) w/ Query Fusion = VidEoMT 68.6 318M 566 160

Table 1. From CAVIS to VidEoMT. Stepwise removal of CAVIS modules toward EoMT, and modifications extending it to our VidEoMT. Evaluated on YouTube-VIS 2019 val [38].

by only 0.8. In steps (2)–(3), we observe that removing context-aware features and the re-identification layers further increases speed by 1.8× to 74 FPS, with almost no impact on accuracy. These results demonstrate that the DINOv2 ViT encoder can take over the functionality of these components without degrading performance. In step (4), we note that the elimination of the tracker, which results in the naive, per-frame application of EoMT, yields a speedup of more than 10× to 162 FPS compared to CAVIS’s 15 FPS, but also leads to a substantial 7.6 AP drop. Interestingly, though, even without any tracking modules and just relying on the queries, the model still retains reasonable accuracy. This shows that EoMT can learn to output objects in a somewhat consistent order across frames, despite processing them independently without temporal interaction.

Applying query propagation in step (5) is necessary to introduce temporal modeling in EoMT, improving the AP by +2.6 without increasing the computational cost. However, we find that the model struggles with identifying newly appearing objects over time. In the final step (6), we show that query fusion allows VidEoMT to recover nearly all of the original accuracy, while achieving a speedup of more than 10× compared to CAVIS. Notably, the gain in inference speed is much larger than in FLOPs. This is the case because VidEoMT almost purely consists of a plain ViT. As such, it can better leverage dedicated hardware and software optimizations for the Transformer architecture without being bottlenecked by inefficient specialized modules [19].

Overall, these results show that VidEoMT achieves an excellent balance between accuracy and efficiency, as heavy modules in CAVIS can be safely removed, while our encoder-only framework effectively restores performance with negligible computational cost. Moreover, these results confirm our hypothesis that a VFM-pretrained ViT can be trained to conduct both segmentation and tracking within the same encoder, without complex tracking modules.

#### 5.2. Comparison with State-of-the-Art Models

Video Instance Segmentation (VIS). We first compare VidEoMT with state-of-the-art VIS models across four datasets. The results, reported in Tabs. 2 and 3, demonstrate that VidEoMT consistently outperforms DVIS [39]

YouTube-VIS 2019 val [38] YouTube-VIS 2021 val [38] AP AP75 AR10 GFLOPs FPS AP AP75 AR10 GFLOPs FPS

Method Backbone Pre-training

MinVIS [17] Swin-L IN21K 61.6 68.6 66.6 401 29 55.3 62.0 60.8 255 30 DVIS [39] Swin-L IN21K 63.9 70.4 69.0 411 23 58.7 66.6 64.6 405 24 DVIS-DAQ [41] Swin-L IN21K 65.7 73.6 70.7 415 13 61.1 68.2 66.6 410 11 DVIS++ [40] ViT-L DINOv2 67.7 75.3 73.7 846 18 62.3 70.2 68.0 830 17 DVIS-DAQ [41] ViT-L DINOv2 68.3 76.1 73.5 851 10 62.4 70.8 68.0 836 10 CAVIS [21] ViT-L DINOv2 68.9 76.2 73.6 838 15 64.6 72.5 69.3 824 15

VidEoMT ViT-L DINOv2 68.6 75.6 73.9 566 160 63.1 69.3 68.1 560 160

###### Table 2. Online VIS on YouTube-VIS 2019 and 2021.

YouTube-VIS 2022 val [38] OVIS val [31] APL APL75 ARL10 GFLOPs FPS AP AP75 AR10 GFLOPs FPS

Method Backbone Pre-training

MinVIS [17] Swin-L IN21K 33.1 33.7 36.6 224 31 39.4 41.3 43.3 408 30 DVIS [39] Swin-L IN21K 39.9 42.6 44.9 401 23 45.9 48.3 51.5 419 24 DVIS-DAQ [41] Swin-L IN21K – – – – – 49.5 51.7 54.9 423 12 DVIS++ [40] ViT-L DINOv2 37.5 39.4 43.5 820 18 49.6 55.0 54.6 868 17 CAVIS [21] ViT-L DINOv2 39.5 40.5 44.9 815 15 53.2 59.1 58.2 863 15 DVIS-DAQ [41]† ViT-L DINOv2 42.0 43.0 48.4 826 10 54.3 60.2 59.8 1173 8

VidEoMT† ViT-L DINOv2 42.6 46.1 48.1 557 161 52.5 57.2 57.5 934 115 Table 3. Online VIS on YouTube-VIS 2022 and OVIS. †Input resolution of 544 (shortest image side) for OVIS, default for others.

VIPSeg val [28] VPQ STQ GFLOPs FPS

Method Backbone Pre-training

DVIS [39] Swin-L IN21K 54.7 47.7 879 20 DVIS++ [40] ViT-L DINOv2 56.0 49.8 2290 13 CAVIS [21] ViT-L DINOv2 56.9 51.0 2612 10 DVIS-DAQ [41] ViT-L DINOv2 57.4 52.0 2315 4

VidEoMT ViT-L DINOv2 55.2 48.9 1897 75

###### Table 4. Online VPS on VIPSeg.

VSPW val [27] mVC16 mIoU GFLOPs FPS

Method Backbone Pre-training

DVIS [39] Swin-L IN21K 94.3 61.3 879 22 DVIS++ [40] ViT-L DINOv2 94.2 62.8 2290 13

VidEoMT ViT-L DINOv2 95.0 64.9 1909 73

Table 5. Online VSS on VSPW.

and DVIS++ [40], while being 5×–8× faster. Compared to DVIS-DAQ [41], VidEoMT is over 14× faster while achieving higher accuracy on all benchmarks except OVIS, where the gap is within 2 AP points. Similarly, VidEoMT surpasses CAVIS on YouTube-VIS 2022, and achieves comparable accuracy on YouTube-VIS 2019 and OVIS, and remains within 2 AP on YouTube-VIS 2021, while being 7× to more than 10× faster. Finally, we note that VidEoMT is also both faster and more accurate than MinVIS [17], which was specifically designed for efficiency and simplicity. Overall, VidEoMT demonstrates a significantly superior trade-off between accuracy and efficiency compared to existing approaches.

Video Panoptic Segmentation (VPS). Tab. 4 compares VidEoMT with state-of-the-art methods for the VPS task on the VIPSeg benchmark. VidEoMT incurs only a minor VPQ drop compared to DVIS++ and CAVIS, while running

5×–7× faster. Compared to DVIS-DAQ, which obtains the highest VPQ of 57.4 but runs at the lowest FPS of 4, VidEoMT sacrifices just 2.2 VPQ while delivering nearly 19× higher speed. These results confirm that VidEoMT also provides a significantly better accuracy and efficiency balance for video panoptic segmentation.

Video Semantic Segmentation (VSS). Tab. 5 compares VidEoMT with state-of-the-art VSS methods on the VSPW benchmark. VidEoMT outperforms existing methods, improving the mIoU by +2.1 compared to DVIS++ while also achieving a higher temporal consistency with +0.8 mVC16 and being more than 5× faster. These results confirm the general applicability and strength of VidEoMT on yet another video segmentation task.

#### 5.3. Further Analyses

EoMT as a Segmenter. In this work, we propose a unified architecture that performs video segmentation in an encoder-only fashion. Alternatively, one could consider augmenting EoMT with recent trackers. In Tab. 6, we compare VidEoMT to alternative approaches where EoMT is used as a segmenter and existing trackers are applied on top. Compared to the best alternative approach, EoMT + CAVIS, VidEoMT achieves slightly better AP while being nearly 4× faster. These results show that our VidEoMT is not only more streamlined but also considerably faster and even more accurate than alternative strategies.

Query Propagation. In VidEoMT, we directly propagate object queries into the ViT encoder. To verify that this is just as effective as propagating them into a separate decoder, we take a DINOv2 + ViT-Adapter encoder and Mask2Former decoder [9], and apply query propaga-

Segmenter Tracker AP AP75 AR10 GFLOPs FPS

EoMT CAVIS [21] 68.1 76.3 73.6 699 42 EoMT DVIS++ [40] 67.0 73.8 73.2 683 69 EoMT DVIS-DAQ [41] 67.3 74.8 73.4 703 28

VidEoMT – 68.6 75.6 73.9 566 160

Table 6. Alternative approaches: EoMT as a segmenter. Comparison of EoMT equipped with state-of-the-art trackers and our proposed VidEoMT. Evaluated on YouTube-VIS 2019 val.

tion into the decoder. We evaluate two propagation variants: TrackFormer [25], which was originally introduced for object tracking rather than video segmentation, and our query fusion approach that combines propagated queries with learned queries. The results in Tab. 7 show that our encoder-only approach achieves an AP score comparable to the encoder–decoder design, validating the effectiveness of the proposed encoder-only architecture. Furthermore, our query fusion strategy slightly improves AP over TrackFormer [25] while also being significantly faster. Overall, VidEoMT is just as effective but considerably faster than both alternative approaches. See the supplementary material for additional experiments on temporal propagation and more details on these alternative methods.

Impact of Pre-training. In this work, we hypothesize that large-scale pre-training with VFMs enables the ViT encoder in VidEoMT to take over the functionalities of specialized components. To verify this hypothesis, in Tab. 8, we compare the performance of VidEoMT and CAVIS using largescale pre-training with DINOv3 [33], DINOv2 [30], and EVA-02 [14], as well as medium-scale ImageNet-21K and small-scale ImageNet-1K pre-training [11, 34]. We observe that, with strong pre-training (DINOv2, DINOv3, EVA02), VidEoMT attains performance comparable to CAVIS, while the performance gap increases in favor of CAVIS as the pre-training scale decreases. Notably, DINOv3 offers only marginal improvements over DINOv2, which we attribute to DINOv3 being designed to be kept frozen rather than fine-tuned. Nevertheless, VidEoMT remains highly effective under DINOv3 pre-training. Overall, these results support our hypothesis that large-scale pre-training is necessary to unleash the potential of VidEoMT. While EoMT [19] showed this effect for image segmentation, our results demonstrate that large-scale pre-training also enables the ViT encoder to take over the functionalities of specialized video segmentation components.

Impact of Model Size. Similarly, we hypothesize that increased model size positively impacts the ViT’s ability to conduct segmentation and tracking. In Tab. 9, we assess this by evaluating CAVIS and VidEoMT for ViT model sizes L, B, and S. The results show that the gap between the CAVIS baseline and VidEoMT decreases as model size increases, confirming our hypothesis. Additionally, while there is a moderate performance gap between CAVIS and VidEoMT

Encoder Decoder Temporal Modeling AP GFLOPs FPS

ViT-Adapter M2F [9] TrackFormer [25] 67.8 739 22 ViT-Adapter M2F [9] Query Fusion 68.0 718 32

VidEoMT – Query Fusion 68.6 566 160

- Table 7. Alternative approaches: Query propagation in the decoder. Comparison of ViT-Adapter + Mask2Former (M2F) equipped with TrackFormer or our query fusion strategy and the proposed VidEoMT. All methods use a ViT-L backbone with DINOv2 pre-training. Evaluated on YouTube-VIS 2019 val.

Model Pre-training AP GFLOPs FPS CAVIS

DINOv3

68.8 838 13 VidEoMT 68.9↑0.1 566 133 CAVIS

DINOv2

68.9 838 15 VidEoMT 68.6↓0.3 566 160 CAVIS

EVA-02

68.0 835 12

- VidEoMT 67.8↓0.2 564 119

CAVIS

IN21K

62.2 838 15 VidEoMT 60.8↓1.4 566 160 CAVIS

IN1K

59.4 838 15

VidEoMT 56.7↓2.7 566 160

Table 8. Impact of pre-training. VidEoMT performs better with larger-scale pre-training. Evaluated on YouTube-VIS 2019 val.

Method Size AP Params GFLOPs FPS CAVIS

L

68.9 358M 838 15

- VidEoMT 68.6↓0.3 318M 566 160

- Table 9. Impact of model size. VidEoMT performs better as ViT [13] size increases. Evaluated on YouTube-VIS 2019 val.

CAVIS

59.5 131M 390 18 VidEoMT 58.2↓1.3 95M 182 251 CAVIS

B

55.5 57M 251 19

S

VidEoMT 52.8↓2.7 25M 56 294

for smaller model sizes, VidEoMT with a large ViT-L backbone is still an order of magnitude faster than CAVIS with a small ViT-S backbone, while being much more accurate. This further highlights the effectiveness of VidEoMT.

### 6. Conclusion

We have introduced VidEoMT, an encoder-only video segmentation architecture that unifies segmentation and temporal association within a single ViT encoder. Through a step-by-step reduction of prior models, we showed that heavy specialized modules are no longer required. We replace them with a lightweight query propagation method, enhanced with an efficient query fusion mechanism. This design achieves an order-of-magnitude speedup while preserving or improving accuracy across multiple video segmentation benchmarks. Overall, our findings suggest that a sufficiently large and well-pretrained ViT can take over much of the functionality that was previously handled by complex downstream components in video segmentation. We hope that this work can serve as a foundation for applications with strict efficiency requirements.

###### Acknowledgments

This work was partly funded by the EU project MODI (grant no. 101076810), the KDT JU project EdgeAI (grant no. 101097300), and the BMFTR project WestAI (grant nos. 01IS22094D and 16IS22094D). The experiments utilized both the Dutch national infrastructure, supported by the SURF Cooperative under grant nos. EINF-14337, EINF-11307, and EINF-15136 and funded by the Dutch Research Council (NWO), and the computing resources provided by the Gauss Centre for Supercomputing e.V. through the John von Neumann Institute for Computing (NIC) on the GCS supercomputer JUWELS at J¨ulich Supercomputing Centre.

### References

- [1] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In ASPLOS, 2024. 6, 11
- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging Properties in Self-Supervised Vision Transformers. In ICCV, 2021. 2
- [3] Niccol`o Cavagnero, Gabriele Rosi, Claudia Cuttano, Francesca Pistilli, Marco Ciccone, Giuseppe Averta, and Fabio Cermelli. PEM: Prototype-based Efficient MaskFormer for Image Segmentation. In CVPR, 2024. 3
- [4] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE TPAMI, 40

(4):834–848, 2018. 2

- [5] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In ECCV, 2018. 2
- [6] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision Transformer Adapter for Dense Predictions. In ICLR, 2023. 3, 4, 11, 12
- [7] Bowen Cheng, Anwesa Choudhuri, Ishan Misra, Alexander Kirillov, Rohit Girdhar, and Alexander G. Schwing. Mask2Former for Video Instance Segmentation. arXiv preprint arXiv:2112.10764, 2021. 3
- [8] Bowen Cheng, Alex Schwing, and Alexander Kirillov. PerPixel Classification is Not All You Need for Semantic Segmentation. In NeurIPS, 2021. 3
- [9] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask

Transformer for Universal Image Segmentation. In CVPR,

2022. 3, 4, 6, 7, 8, 11, 12

- [10] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR, 2024. 6, 11
- [11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR, 2009. 8
- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL, 2019. 6, 11
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, 2021. 1, 3, 8, 11, 13
- [14] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA-02: A Visual Representation for Neon Genesis. Image and Vision Computing, 2024. 1, 8, 13
- [15] Dan Hendrycks and Kevin Gimpel. Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415, 2016. 2
- [16] Miran Heo, Sukjun Hwang, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. VITA: Video Instance Segmentation via Object Token Association. In NeurIPS, 2022. 3
- [17] De-An Huang, Zhiding Yu, and Anima Anandkumar. MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training. In NeurIPS, 2022. 3, 7
- [18] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. OneFormer: One Transformer To Rule Universal Image Segmentation. In CVPR, 2023. 3
- [19] Tommie Kerssies, Niccol`o Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus. Your ViT is Secretly an Image Segmentation Model. In CVPR, 2025. 1, 3, 4, 6, 8, 11
- [20] Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Video Panoptic Segmentation. In CVPR, 2020. 3, 6
- [21] Seunghun Lee, Jiwan Seo, Kiljoon Han, Minwoo Choi, and Sunghoon Im. Context-Aware Video Instance Segmentation. In ICCV, 2025. 1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16
- [22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In ECCV, 2014. 11
- [23] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation. In CVPR, 2015. 2
- [24] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In ICLR, 2019. 6, 11
- [25] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. TrackFormer: Multi-Object Tracking with Transformers. In CVPR, 2022. 8, 12, 13
- [26] Meta Research. fvcore, 2023. 6, 11
- [27] Jiaxu Miao, Yunchao Wei, Yu Wu, Chen Liang, Guangrui Li, and Yi Yang. VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild. In CVPR, 2021. 2, 6, 7, 11

- [28] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark. In CVPR, 2022. 2, 6, 7, 11, 13, 16
- [29] David Nilsson and Cristian Sminchisescu. Semantic Video Segmentation by Gated Recurrent Flow Propagation. In CVPR, 2018. 3
- [30] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR, 2024. 1, 2, 3, 5, 6, 8, 11, 12, 13
- [31] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded Video Instance Segmentation: A Benchmark. IJCV, 130(8):2022–2039, 2022. 6, 7, 11, 13, 15
- [32] Inkyu Shin, Dahun Kim, Qihang Yu, Jun Xie, Hong-Seok Kim, Bradley Green, In So Kweon, Kuk-Jin Yoon, and Liang-Chieh Chen. Video-kMaX: A Simple Unified Approach for Online and Near-Online Video Panoptic Segmentation. In WACV, 2024. 3
- [33] Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025. 1, 2, 8, 13
- [34] Hugo Touvron, Matthieu Cord, and Herv´e J´egou. DeiT III: Revenge of the ViT. In ECCV, 2022. 8
- [35] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers. In CVPR, 2021. 3
- [36] Mark Weber, Jun Xie, Maxwell Collins, Yukun Zhu, Paul Voigtlaender, Hartwig Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, et al. STEP: Segmenting and Tracking Every Pixel. In NeurIPS, 2021. 6
- [37] Yuetian Weng, Mingfei Han, Haoyu He, Mingjie Li, Lina Yao, Xiaojun Chang, and Bohan Zhuang. Mask Propagation for Efficient Video Semantic Segmentation. In NeurIPS,

2023. 3

- [38] Linjie Yang, Yuchen Fan, and Ning Xu. Video Instance Segmentation. In ICCV, 2019. 1, 2, 3, 6, 7, 11, 13, 14
- [39] Tao Zhang, Xingye Tian, Yu Wu, Shunping Ji, Xuebo Wang, Yuan Zhang, and Pengfei Wan. DVIS: Decoupled Video Instance Segmentation Framework. In CVPR, 2023. 1, 3, 6, 7
- [40] Tao Zhang, Xingye Tian, Yikang Zhou, Shunping Ji, Xuebo Wang, Xin Tao, Yuan Zhang, Pengfei Wan, Zhongyuan Wang, and Yu Wu. DVIS++: Improved Decoupled Framework for Universal Video Segmentation. PAMI, 2025. 1, 2, 4, 6, 7, 8, 11
- [41] Yikang Zhou, Tao Zhang, Shunping Ji, Shuicheng Yan, and Xiangtai Li. Improving Video Segmentation via Dynamic Anchor Queries. In ECCV, 2024. 1, 2, 3, 4, 6, 7, 8, 11

## VidEoMT: Your ViT is Secretly Also a Video Segmentation Model Supplementary Material

### Appendix

###### Table of contents

- • §A: Implementation Details
- • §B: Additional Experiments
- • §C: Qualitative Results

### A. Implementation Details

#### A.1. Training

Following state-of-the-art models CAVIS [21], DVISDAQ [41] and DVIS++ [40], we adopt a DINOv2pretrained ViT [13, 30] as the backbone of VidEoMT, and we train our model in two stages. In stage one, we train the model for image segmentation only. Concretely, we train on COCO [22] instance segmentation and the target video segmentation dataset without applying any temporal supervision. In the second stage, we introduce temporal modeling and fine-tune the model from stage one for video segmentation. Unlike CAVIS, DVIS-DAQ, and DVIS++, which freeze the DINOv2-initialized ViT encoder after stage one, we keep fine-tuning the ViT encoder for VidEoMT. We explore fine-tuning the ViT encoder for the CAVIS and DVIS++ baselines in Tabs. 1 and 2 as well, but find that the loss diverges or the memory increases beyond the GPUs’ limits. For our VidEoMT, note that fine-tuning the encoder is necessary because our model is encoder-only, meaning that the encoder weights need to be optimized to allow the model to be trained for video segmentation.

#### A.2. Evaluation

During evaluation, we process videos in a frame-by-frame fashion, as is required for online video segmentation. We evaluate efficiency in terms of FPS and GFLOPs. All metrics are measured on a single NVIDIA H100 GPU using PyTorch 2.7 and CUDA 12.6. We use a batch size of 1 frame to report mean values computed across all frames in the entire validation set. FPS is measured using FlashAttention v2 [10] and torch.compile [1] with default settings and automatic mixed precision, after 100 warm-up iterations. FLOPs are measured with fvcore [26], and reported in GFLOPs (FLOPs ×10−9).

#### A.3. Visualizations of Model Configurations

In Sec. 3.3 and Tab. 1, we gradually remove specialized components from the state-of-the-art video segmentation model CAVIS [21], which is visualized in Fig. 2 (left). To provide more details, we additionally illustrate the architectures at intermediate steps (1) to (4) in Fig. A. In the first

step, we replace CAVIS’s original segmenter – consisting of DINOv2 [30], ViT-Adapter [6], and Mask2Former’s pixel decoder and Transformer decoder [9] – with EoMT [19]. In the second step, we remove the context-aware features module and directly forward the segmenter’s output queries to the re-identification layers. In the third step, we also remove the re-identification layers, sending the segmenter’s output queries directly to the tracker’s Transformer blocks. Subsequently, in the fourth step, we discard the tracker altogether, and naively apply EoMT only on a per-frame basis. In this step, temporal association is then obtained in the simplest possible way: we assign all objects predicted from the same query across frames to the same track, without any additional post-hoc temporal matching.

In step (5), which we do not visualize here, we propagate queries by directly feeding the output segmentation queries from frame t − 1 into the encoder for frame t. Finally, in step (6), we introduce our query fusion design, where propagated queries are fused with learnable queries. The resulting architecture is visualized in Fig. 2 (right).

#### A.4. Hyperparameters

For step (0) in Tab. 1, we report results using the official CAVIS [21] model weights, which we were able to reproduce. For all subsequent steps, we train the models using the same settings as CAVIS with respect to input size, number of iterations, batch size, and number of sampled frames. Specifically, we use a batch size of 8, train on 8 NVIDIA H100 GPUs, and sample 5 frames from a video clip. We train for 160k iterations on YouTube-VIS [38] (all versions) and OVIS [31], for 40k iterations on VIPSeg [28], and for 20k iterations on VSPW [27]. Additionally, all VidEoMT models use N = 200 learnable queries with a feature dimension of D = 1024.

For all experiments that use EoMT as the segmenter, as well as for all experiments with VidEoMT, we keep the optimization strategy identical to that of EoMT. Concretely, we use automatic mixed precision and the AdamW optimizer [24] with a learning rate of 10−4. We apply layerwise learning rate decay (LLRD) [12] with a factor of 0.6 and polynomial learning rate decay with a power of 0.9. A two-stage linear warm-up strategy is used for all models, including the baselines. Specifically, we first warm up the randomly initialized parameters for 500 iterations while keeping the pre-trained parameters frozen. Then, after 500 iterations, we warm up the pre-trained parameters for 1000 iterations. In both stages, the initial learning rate is set to 0.

To supervise our models, we adopt the same loss functions as Mask2Former [9]. Across all tasks and datasets, we

Learnable Queries

Track Queries t − 1

Frame t

MasksFeatures

CA Feature Extractor

ReID Layers

TF Blocks

EoMT

tQueries

Seg.

Track Queries t

Segmenter Tracker Masks + Classes

###### Step (1): w/ EoMT as the Segmenter

Learnable Queries

Track Queries t − 1

Frame t

tQueries

ReID Layers

TF Blocks

Seg.

EoMT

Track Queries t

Segmenter Tracker

Masks + Classes

###### Step (2): w/o Context-aware Features

Learnable Queries

Track Queries t − 1

Frame t

| | |
|---|---|
| | |

tQueries

TF Blocks

Seg.

EoMT

Track Queries t

Segmenter Tracker

Masks + Classes

###### Step (3): w/o Re-identification Layers

Learnable Queries

Frame t

Masks+Classes

tQueries

Seg.

EoMT

Segmenter

###### Step (4): w/o Tracker Blocks

- Figure A. Removing specialized components. This figure visualizes the step-by-step removal of complex, specialized components from the CAVIS [21] model, as reported in the results in Tab. 6 of the main manuscript.

use the cross-entropy (CE) loss for the classification predictions, and the binary cross-entropy (BCE) loss together with Dice loss for segmentation predictions. The total loss is a weighted sum of these components:

Ltot = λbceLbce + λdiceLdice + λceLce. (4)

where λbce, λdice, and λce are set to 5.0, 5.0, and 2.0, respectively, following Mask2Former [9].

#### A.5. Architectures of Alternative Approaches

In Tab. 7, we compare VidEoMT with an alternative encoder–decoder architecture that performs temporal modeling in the decoder, with two different temporal modeling approaches: our proposed query fusion and a TrackFormerbased design [25]. As the encoder, we use DINOv2 + ViT-Adapter [6, 30], and as the decoder we use a Transformer that follows the architecture of the Mask2Former Transformer decoder for segmentation. [9]. Concretely, we adopt the standard Mask2Former decoder with 9 layers, each composed of cross-attention, self-attention, and feed-forward blocks, operating with a hidden dimension of 256. To introduce temporal modeling, we feed the track queries and learnable queries into the decoder instead of the encoder’s Transformer blocks, which we would do for VidEoMT. At the output of the decoder, the resulting queries are used to predict segmentation masks and classes in the same way as for VidEoMT. For query fusion, we adopt the same approach as described in Sec. 3.4.

The described encoder–decoder approach, which is much less efficient than the encoder-only VidEoMT method (see Tab. 7), somewhat resembles TrackFormer [25], a method for bounding-box multi-object tracking (MOT). TrackFormer also applies temporal modeling by propagating queries into the decoder, but follows a more complex approach to do so. To assess the effectiveness of our query fusion approach compared to TrackFormer’s approach, we therefore additionally implement TrackFormer’s temporal modeling strategy in the encoder–decoder setting, while staying as close as possible to the original implementation. Specifically, we make predictions for the first frame using a set of 400 learnable queries. Using these predictions, only the N queries with a classification score s > 0.8 are kept and converted into track queries. For the next frame, these track queries are concatenated with the 400 original learnable queries, which are then fed to the decoder for that frame. In subsequent frames, the decoder updates the propagated track queries such that they predict the masks for the same objects in the new frames. Again, newly detected queries with scores s > 0.8 are added as additional track queries, and non-maximum suppression (NMS) with an IoU threshold of σNMS = 0.9 is applied to remove near-duplicate predictions. Note that this NMS operation is the main reason for the TrackFormer approach’s inefficiency compared to VidEoMT’s query propagation mechanism. Finally, at

YouTube-VIS 2019 val [38] AP GFLOPs FPS

Method

No propagation 61.3 565 162 Propagation only 63.9 565 162 Non-object reset 67.8 565 157 TrackFormer 67.7 571 117

Fusion ⇒ VidEoMT 68.6 566 160

Table A. Query propagation methods. Comparison of alternative strategies for temporal propagation.

each frame, track queries are removed if their score remains below s < 0.8 for five consecutive frames, indicating that the object they are tracking has disappeared from the scene.

### B. Additional Experiments

#### B.1. Query Propagation Methods

VidEoMT propagates queries by fusing the learnable queries with the propagated track queries. In Tab. A, we compare this approach with alternative methods to propagate queries.

We start with the no propagation approach, the simplest variant, where the model receives only the learnable queries – similar to EoMT – but is fine-tuned for video segmentation. This setting performs the worst, as it lacks any form of explicit temporal modeling.

Next, in the propagation only variant, we introduce temporal modeling by directly propagating the output queries from the previous frame into the current frame’s encoder. This is step (5) in Tab. 1. However, this approach struggles to detect new objects effectively, as the influence of the learnable queries diminishes over time.

Non-object reset improves over this by replacing a propagated query with a learnable query if it did not predict an object in the previous frame, but this still underperforms the default fusion approach.

Finally, we evaluate the TrackFormer approach [25] of only propagating queries for detected objects and introducing new learnable queries to detect new objects. This approach performs slightly worse than our fusion approach, but most importantly it is considerably slower because it requires filtering out duplicate detections that should not be propagated. Overall, these results demonstrate that our fusion approach is the most accurate and efficient.

#### B.2. Impact of Model Size

In Tab. 9 of the main manuscript, we report the impact of model size for both VidEoMT and CAVIS. In this section, in Tab. B, we additionally report the results of the EoMT + CAVIS combination, which we also visualize in Fig. 1. Compared to the alternative approach of extending EoMT with a CAVIS tracker, VidEoMT consistently performs better in terms of both efficiency and accuracy across all back-

Method Size AP Params GFLOPs FPS

CAVIS 68.9 358M 838 15 EoMT + CAVIS L 68.1 328M 699 42 VidEoMT 68.6 318M 566 160

CAVIS 59.5 131M 390 18 EoMT + CAVIS B 57.4 103M 284 67 VidEoMT 58.2 95M 182 251

CAVIS 55.5 57M 251 19 EoMT + CAVIS S 50.3 34M 150 93 VidEoMT 52.8 25M 56 294

Table B. Impact of model size. VidEoMT performs better as ViT [13] size increases. Evaluated on YouTube-VIS 2019 val.

bones. This highlights the effectiveness of VidEoMT over the more naive approach of extending EoMT with a stateof-the-art tracker.

#### B.3. Impact of Pre-training

In Tab. 8 of the main manuscript, we observe that DINOv3 [33] and EVA-02 [14] are slower than DINOv2 [30], despite having a similar number of GFLOPs. Since both DINOv3 and EVA-02 use rotary positional embeddings (RoPE), we attribute a significant part of this slowdown to RoPE, as it introduces additional element-wise operations in the attention layers. When we disable RoPE in these models, we obtain faster models, confirming that RoPE is one of the main sources of the slowdown. Other implementation details may also play a secondary role.

### C. Qualitative Results

In Figs. B to D, we visualize the predictions of both CAVIS [21] and VidEoMT for VIS and VPS on the YouTube-VIS 2019 [38], OVIS [31], and VIPSeg [28] datasets.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

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

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

###### CAVIS (15 FPS) VidEoMT (160 FPS) CAVIS (15 FPS) VidEoMT (160 FPS)

- Figure B. Qualitative results for video instance segmentation. We compare CAVIS [21] to VidEoMT on the YouTube-VIS 2019 dataset [38].

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

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

###### CAVIS (15 FPS) VidEoMT (112 FPS) CAVIS (15 FPS) VidEoMT (112 FPS) Figure C. Qualitative results for video instance segmentation. We compare CAVIS [21] to VidEoMT on the OVIS dataset [31].

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

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

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

###### CAVIS (10 FPS) VidEoMT (75 FPS) CAVIS (10 FPS) VidEoMT (75 FPS) Figure D. Qualitative results for video panoptic segmentation. We compare CAVIS [21] to VidEoMT on the VIPSeg dataset [28].

