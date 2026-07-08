# arXiv:2504.14032v1[cs.CV]18Apr2025

## LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models

Haiwen Huang1,2 Anpei Chen1,2 Volodymyr Havrylov1 Andreas Geiger1,2 Dan Zhang3 1University of T¨ubingen 2 T¨ubingen AI Center 3 Bosch Center for Artificial Intelligence

#### Abstract

Vision foundation models (VFMs) such as DINOv2 and CLIP have achieved impressive results on various downstream tasks, but their limited feature resolution hampers performance in applications requiring pixel-level understanding. Feature upsampling offers a promising direction to address this challenge. In this work, we identify two critical factors for enhancing feature upsampling: the upsampler architecture and the training objective. For the upsampler architecture, we introduce a coordinate-based crossattention transformer that integrates the high-resolution images with coordinates and low-resolution VFM features to generate sharp, high-quality features. For the training objective, we propose constructing high-resolution pseudogroundtruth features by leveraging class-agnostic masks and self-distillation. Our approach effectively captures fine-grained details and adapts flexibly to various input and feature resolutions. Through experiments, we demonstrate that our approach significantly outperforms existing feature upsampling techniques across various downstream tasks. Our code is released at https://github.com/ andrehuang/loftup.

#### 1. Introduction

High-quality pretrained representations from Vision Foundation Models (VFMs) have become standard for a wide range of computer vision tasks [10, 16, 21, 37, 41, 43, 45, 54, 56, 62, 63]. However, because of the patrification or aggressive pooling operations in VFMs, the output features are typically 16 or even more times smaller in spatial resolution than the input images, limiting their utility for tasks that require fine-grained, pixel-level understanding.

To address the challenge of limited feature resolution in VFMs, one straightforward approach is to use larger image inputs to obtain higher-resolution features. However, processing high-resolution inputs incurs a quadratic increase in computational cost and can introduce severe artifacts if the VFMs are not trained for such resolutions. Moreover,

|LoftUp (ours)<br><br>Max(FeatUp, LiFT)<br><br>Backbone only|
|---|

Semantic Seg.

61.11

Depth Estimation

53.76 88.71

Interactive Seg.

91.35

78.49

65.89

26.61

69.83 44.30

27.82

72.61

Open-Voc Seg.

Normal Estimation

60.25

Video Object Seg. (Object Tracking)

Figure 1. LoftUp improves significantly across various tasks over the VFM backbone (DINOv2-S [37]) and current SoTA feature upsampling performance (FeatUp [12] and LiFT [52]). See experiment details in Sec. 5.

training or fine-tuning VFMs on high-resolution images demands substantial computational resources and meticulous tuning [37, 43, 45, 55]. An alternative strategy is to train task-specific decoders that leverage multi-layer intermediate features to upsample VFM outputs [3, 16, 24, 27, 62]. Yet, this approach often comes with significant training costs and necessitates retraining the decoder for each new application. In addition, many downstream tasks lack sufficient data needed to fine-tune a high-quality decoder.

More recently, FeatUp [12] and LiFT [52] have independently introduced task-agnostic feature upsamplers trained with general reconstruction losses, demonstrating that such methods can substantially enhance VFM performance across a variety of tasks. By upsampling VFM features and pairing them with a lightweight task-specific decoder, these approaches provide a promising alternative that avoids heavy, task-specific training while achieving a more generalizable solution. Nonetheless, as shown in Fig. 1, current feature upsamplers still fall short of reaching optimal performance.

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

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

(a) Original image (b) Low res (c) Bilinear (d) Resize-conv (e) LIIF [2] (f) LiFT [52] (g) FeatUp [12] (h) LoftUp (Ours)

Figure 2. Comparison of features from upsamplers. Backbone is DINOv2-S/14 [37].

In this work, we systematically explore the design space of feature upsamplers and identify two critical components: the upsampler architecture and the training objective. The architecture determines the capacity of the upsampler to learn effectively, while the training objective defines the upper performance limit. By optimizing both elements, our approach achieves substantially stronger results than previous state-of-the-art methods.

Regarding the upsampler architecture, to capture highresolution details while avoiding the cumulative artifacts from multiple layers of interpolation or deconvolution, we propose a simple coordinate-based transformer that directly predicts high-resolution features for each pixel. Specifically, our model takes image coordinates and RGB values as inputs and performs cross-attention with the lowresolution VFM feature map. This facilitates a fine-grained, content-aware mapping from coordinates to high-resolution features, effectively bypassing the constraints imposed by fixed local kernels and standard upsampling layers. Moreover, unlike the implicit approach in FeatUp [12], which requires test-time optimization, our method learns feature upsampling directly from the training dataset and generalizes to diverse scenes without additional test-time adjustments.

For the training objective, the primary challenge is the absence of groundtruth high-resolution feature annotations. Although downstream task labels such as depth or masks can be used, this approach risks compromising the taskagnostic nature of the upsampler and hinders its ability to generalize to unseen tasks. Due to this challenge, previous task-agnostic feature upsampling works, FeatUp [12] and LiFT [52], both compute the training loss at a low resolution. In this work, we address these limitations by constructing pseudo-groundtruth (pseudo-GT) features directly

at the input image resolution. Specifically, we leverage class-agnostic masks generated by off-the-shelf segmentation foundation models [21, 45] to ensure that the pseudoGT accurately reflects the underlying geometry and delineates object boundaries. We further refine the pseudo-GT using a self-distillation strategy that reduces noise and artifacts. This high-resolution pseudo-GT enables loss computation at a high resolution, empowering the upsampler to learn fine-grained details.

In Fig. 2, we qualitatively show that LoftUp yields markedly sharper and more detailed features than alternative upsampling methods. To further demonstrate the versatility and effectiveness of our approach, we evaluate it on a range of downstream tasks such as semantic segmentation, depth estimation, and video objectschannen2025siglip2. As shown in Fig. 1, our approach leads to performance gains of 10–20% over previous SoTA upsamplers for most tasks, and an impressive nearly 50% improvement on video object segmentation [39]. Furthermore, thanks to its coordinatebased design, our upsampler adapts seamlessly to various input and feature resolutions, catering to the diverse requirements of downstream applications. Overall, with less than a 20% increase in parameters compared to the original foundation models, our feature upsampler offers a taskagnostic, lightweight, and plug-and-play enhancement that significantly boosts VFM backbones across multiple tasks.

#### 2. Related Work

Feature upsampling refers to increasing the spatial resolution of a feature map. In this work, our goal is to increase the feature resolution to the original image resolution, that is, full resolution. Traditional non-learnable methods include various ways of interpolation [7, 33] and image-

adaptive filtering such as joint bilateral filtering (JBU) [22] and guided image filtering [15]. In modern deep learning, previous work has proposed various architecture- and downstream-task-specific feature upsamplers. For example, Index Networks [29] and A2U [5] are effective on image matting, but fall short in other tasks. PointRend [20] proposes a point-rendering method specifically for upsampling segmentation output. And CARAFE [57], SAPA [31], and FADE [30] are proposed specifically for encoder-decoder architectures. More recently, with the success of vision foundation models such as DINOv2 [37] and CLIP [41], there is a trend for feature upsamplers to be downstreamtask-agnostic so that they can be used with the VFM backbone together in various applications [12, 52]. Our work falls into this category. With this task-agnostic goal in mind, we further explain the architecture and training objective design as follows.

Architecture for Feature Upsamplers. Traditional upsampler architectures rely on multiple layers of interpolation or deconvolution to transform low-resolution features into higher resolutions. Examples include JBU [22], standard deconvolution [8, 35, 49], resize-convolution [36] and U-Net-style upsampling modules [47, 52]. However, the multi-layer design inevitably leads to error accumulation, resulting in increased blurriness as the resolution increases.

In this work, inspired by coordinate-based methods in 3D reconstruction [25, 53, 60, 61], we adopt a coordinatebased approach and view feature upsampling as a mapping from high-resolution coordinates to high-resolution features. This effectively bypasses the limitations of standard upsampling layers. Previously, FeatUp also proposed a coordinate-based network (MLP) for feature upsampling [12]. However, their approach requires per-image optimization and is therefore not scalable. Another related work, LIIF [2], also employs an MLP to parameterize highresolution outputs but is limited to local feature interactions. In contrast, our method does not need test-time optimization and enables global interactions between image inputs and low-resolution features through a cross-attention mechanism, leading to stronger upsampling performance.

Task-agnostic Training Objective for Feature Upsampling. Due to the absence of groundtruth high-resolution features, it is a challenge to create high-quality training objective for task-agnostic feature upsampling. FeatUp [12] and LiFT [52] were the first two works to propose a taskagnostic training pipeline for feature upsamplers. However, their training is at a low resolution, leaving the highresolution features severely under-constrained. We will explain them in more detail Sec. 4. In this paper, we propose a self-distillation approach to generate full-resolution pseudo-GT, which is then used to supervise feature upsampling training at full resolution. This approach fully un-

[Figure 25]

[Figure 26]

High-res Coordinates (Sinusoidal Positional Encodings)

[Figure 27]

RGB values

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

3 x 3 conv

CA FFN

x L

[Figure 33]

High-res Features

[Figure 34]

[Figure 35]

Sinusoidal Positional Encodings Low-res Features

Figure 3. Architecture of LoftUp. Our coordinate-based network with cross-attention mechanism effectively integrates the fine-grained details from image RGB values and semantically-rich low-res features to produce high-resolution feature maps.

locks the potential of the feature upsampler to capture finegrained details in high-resolution images.

#### 3. Coordinate-Based Feature Representation

Prior works typically address feature upsampling as a gradual resolution enhancement process, where features are progressively lifted to a higher resolution. In contrast, we adopt a coordinate-based representation of the high-resolution features and view feature upsampling as a mapping from a pixel coordinate (x,y) to the high-resolution features at that pixel. Specifically, we propose a cross-attention mechanism to effectively incorporate the low-resolution features and the high-resolution image inputs with coordinates.

As shown in Fig. 3, following prior work on 3D coordinate-based reconstruction [34, 61], our model encodes full-resolution coordinates with sinusoidal positional embeddings and concatenates them with RGB values. A convolutional layer then projects this combined input into the feature dimension. Next, an L-block cross-attention transformer uses these high-resolution features as queries and low-resolution VFM features as keys and values, producing the final high-resolution feature map.

Our cross-attention design enables high-frequency details to interact globally with semantically rich representations, facilitating global-content-aware upsampling. This overcomes the limitations of fixed or locally predicted kernels in prior feature upsamplers. As illsustrated in Fig. 2, multi-layer upsamplers such as resize-conv [36], LiFT (UNet) [52], and FeatUp (a modified JBU) [12] tend to produce blurry outputs with artifacts. Additionally, while LIIF [2] employs a coordinate-based design, its reliance on local interactions limits its upsampling quality. In contrast, our LoftUp accurately captures object boundaries and produces fine-grained feature maps with minimal artifacts.

Stage 1 Stage 2

ℝH×W

ℝH×W

ℝH×W

ℝH×W

ℝ

H t ×Wt

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

[Figure 46]

[Figure 47]

[Figure 48]

Upsampler

Student

|[Figure 49]|
|---|

🔥 Upsampler

[Figure 50]

[Figure 51]

🔥

[Figure 52]

ℒMask-Bicubic

ℒSelf-Distilled σ↓

[Figure 53]

[Figure 54]

EMA Update

SAM

[Figure 55]

[Figure 56]

[Figure 57]

Bicubic Upsample

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

|[Figure 62]|
|---|

Teacher Upsampler

[Figure 63]

[Figure 64]

Mask Reﬁne

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

ℝH×W

ℝH×W

FMask-Bicubic

ℝtH×tW

ℝH×W ℝH×W

- Figure 4. Our two-stage LoftUp training approach. Stage 1 trains an upsampler with class-agnostic masks to refine bicubic-upsampled features. Stage 2 employs self-distillation, initializing teacher and student upsamplers from Stage 1’s pre-trained model. All VFM image inputs share the resolution (H × W). For visual clarity, the VFM block is omitted from Stage 2’s teacher branch.

Per-image implicit 2x features (LiFT)

Mask-Bicubic Self-Distilled

Depth map

Metrics

High-Res Loss ✓ ✓ ✗ ✓ ✓ Task-Agnostic ✗ ✓ ✓ ✓ ✓ Geometry Fidelity ✓ ✓ ✗ ✓ ✓ Free of Noise/Artifacts ✓ ✗ ✗ ✗ ✓ Scalability ✓ ✗ ✓ ✓ ✓

COCO Seg. (IoU) 56.02 59.08 52.90 59.87 61.11 Cityscapes Seg. (IoU) 44.82 46.05 35.85 50.43 53.10

Table 1. Comparison of different pseudo-GT choices.

Furthermore, our coordinate-based approach allows flexible generation of feature maps at arbitrary resolutions by adjusting the input coordinate resolution—unlike traditional upsampling methods, which are restricted to fixed scaling factors due to their multi-layer structures. Finally, we note that while attention mechanisms can be computationally expensive, our cross-attention remains relatively efficient because it processes a much smaller set of lowresolution tokens as keys and values, rather than the progressively higher-resolution feature maps as in multi-layer upsampler architectures. In fact, as our experiments (Tab. 7) later show, our upsampler’s inference speed is comparable to bilinear upsampling.

#### 4. Training Objective

While the architecture of the upsampler establishes its capacity, the training objective ultimately sets the performance ceiling. A central challenge for training a taskagnostic feature upsampler is the absence of full-resolution groundtruth features. As a result, previous work has either

relied on proxy tasks or constructed pseudo-GT at lower resolutions. For instance, FeatUp [12] employs a multiview reconstruction task: the predicted high-resolution features, FˆHR, are first transformed via an affine mapping t, then downsampled, and finally compared against the lowresolution features extracted from images that go through the same transformation, i.e.,

LFeatUp = D f(t(I)),σ↓(t(FˆHR) .

In contrast, LiFT [52] directly uses features from 2× larger images I2× as pseudo-GT, constraining the upsampler to predict 2× upsampled features Fˆ2× via

LLiFT = D F ˆ2×,f(I2×) .

However, since both LFeatUp and LLiFT are at relatively low resolutions (only 1/16 or 1/8 of the target resolution), they provide only weak supervisory signals for capturing finegrained details inherent in high-resolution outputs, potentially leaving the upsampled features under-constrained.

In this work, we employ a self-distillation strategy to generate high-quality pseudo-GT for supervising fullresolution features, as shown in Fig. 4. First, we train an upsampler using high-resolution class-agnostic masks, which emphasize sharp boundaries and geometric awareness. In the second stage, we enhance training through selfdistillation. Specifically, we first initialize a teacher and a student upsampler both from the trained upsampler in Stage 1. Then, the teacher processes high-resolution image crops, and its outputs serve as supervision for the corresponding student upsampler outputs. By distilling more detailed and accurate feature maps from the teacher, Stage 2 further reduces noise and enhances sharpness in the student’s outputs. Overall, by constructing high-quality pseudo-GT fea-

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

(a) Original image (b) Per-image optimized (c) 2x features (LiFT) (d) Mask-Bicubic (Stage 1) (e) Self-Distilled (Stage 2)

- Figure 5. Visualization of different pseudo-GT. Both Mask-Bicubic and Self-Distilled are proposed by our work. We set α = 0.8 (in Eq. (1)) to balance sharp boundaries from masks and fine-grained details from high-res features.

ture maps at full-resolution, our approach enables the upsampler to capture more detailed structures and boundaries, ultimately pushing the limits of feature upsampling.

###### 4.1. Stage 1: Training with class-agnostic masks

Low-resolution features and their upsampled features using existing upsamplers are often too blurry and noisy to be used as pseudo-GT. In contrast, Segment Anything Model (SAM) [21] produces full-resolution, class-agnostic masks that capture fine-grained details such as small object parts and boundaries. Previous work has shown that training with these masks leads to strong task generalization [21, 44–46]. These qualities make them well suited for refining feature maps to reduce artifacts and noise while promoting smoothness in homogeneous regions.

To leverage these masks, we first upsample a lowresolution feature map to full resolution via bicubic interpolation, yielding FBicubic. Then, for each mask m ∈ M = {m1,m2,...,mN}, we compute the mean feature, FBicubic[m], and blend it with the original features, yielding a mask-refined feature map at pixels within mask m:

FMask-Bicubic[m] = α∗FBicubic[m]+(1−α)∗FBicubic[m], (1) where α ∈ [0,1] controls the degree of mask refinement.

We then supervise the upsampler using the loss:

LMask-Bicubic = ||FˆHR − FMask-Bicubic||2, (2)

which encourages high-resolution outputs with homogeneous features within each mask region, thereby leveraging the rich structural information from the class-agnostic masks. Note while these masks can also refine features from more complicated upsampling methods such as FeatUp [12]

Per-image implicit 2x features (LiFT)

Mask-Bicubic Self-Distilled

Depth map

Metrics

High-Res Loss ✓ ✓ ✗ ✓ ✓ Task-Agnostic ✗ ✓ ✓ ✓ ✓ Geometry Fidelity ✓ ✓ ✗ ✓ ✓ Free of Noise/Artifacts ✓ ✗ ✗ ✗ ✓ Scalability ✓ ✗ ✓ ✓ ✓

COCO Seg. (IoU) 56.02 59.08 52.90 59.87 61.11 Cityscapes Seg. (IoU) 44.82 46.05 35.85 50.43 53.10

Table 2. Comparison of different pseudo-GT choices.

or JBU [22], we observe that simple bicubic upsampling already yields strong results, making it the preferred choice for Stage 1 training.

###### 4.2. Stage 2: Training with self-distillation

We further enhance our upsampler training using selfdistillation. To distill high-quality features, we adopt a teacher-student dual-branch design. Both the teacher and student upsamplers are initialized with the upsampler trained in Stage 1. The teacher is updated via an Exponential Moving Average (EMA) during training, ensuring a stable and progressively improving target.

For the student branch, each image is resized to a fixed resolution I ∈ RH×W (224px or 336px), matching the VFM’s input requirements. For the teacher branch, the original image is resized to a larger size IHR ∈ RtH×tW with t ∈ [2,4]. The models then process on a crop from IHR that also matches the VFM’s resolution, i.e., crop(IHR) ∈ RH×W. Note our original images, sourced from the SA1B

Semantic Seg. Depth Estimation Normal Estimation Method COCO mIoU ↑ CS mIoU ↑ RMSE ↓ Recall ↑ RMSE ↓ Recall ↑

Low-res 51.21 36.54 0.1071 89.08 32.29 69.56 Bilinear 56.15 44.79 0.1132 87.68 32.27 70.03 LiFT 53.35 35.80 0.1078 88.71 32.31 69.78 FeatUp 56.30 44.19 0.1092 88.57 32.25 69.83 LoftUp 61.11 53.10 0.0921 91.35 30.79 72.76

- Table 3. Comparison of feature upsamplers across tasks of semantics segmentation, depth estimation, and normal estimation. For each metric, we boldface the best performance and underline the second best.

Video Obj. Seg. Open-Voc Seg. Interactive Seg. Method J Mean F Mean J & F Mean COCO [26] CS [4] ADE [64] GrabCut [48] Berkeley [32] DAVIS [38]

Low-res 42.05 31.27 36.66 25.70 34.48 19.50 64.90 55.77 52.82 Bilinear 42.62 33.90 38.26 25.78 34.56 19.53 65.04 55.83 54.26 LiFT 47.68 36.78 42.23 25.96 35.39 19.50 29.66 31.99 41.14 FeatUp 45.70 42.90 44.30 26.61 35.01 19.99 65.89 56.67 55.03 LoftUp 58.72 61.79 60.25 27.82 38.82 21.29 78.49 65.24 67.31

- Table 4. Comparison of feature upsamplers across tasks of video object segmentation, zero-shot open-vocabulary segmentation, and interactive segmentation. Following previous works [16, 17, 21, 23, 27, 52], we report J Mean, F Mean, and their average for video object segmentation, mIoU for open-vocabulary segementation, and IoU@1 Click for interactive segmentation.

dataset [21], have a minimum resolution of 1500px on the shortest side and is thus larger than tH × tW. While a higher t yields more fine-grained teacher outputs, it also lowers the supervision resolution on the student (H/t × W/t), reducing supervision effectiveness. In practice, we find t ∈ [2,4] strikes a good balance. The teacher’s output on this crop, fteacher crop(IHR) ∈ RH×W, is subsequently downsampled via σ↓(·) to match the resolution of the corresponding student output, crop fstudent(I) ∈ RH/t×W/t, and used as pseudo GT for the student. Formally, the selfdistillation loss is defined as:

LSelf-Distilled = D σ↓ fteacher(crop(IHR) ,crop fstudent(I) ,

where D denotes a discrepancy function between features. In our experiments, we adopt the affinity matrix loss [40, 58, 59], which consistently outperforms the standard L2 loss. This self-distillation strategy leverages the teacher’s ability to handle an easier task—generating high-quality features for cropped regions—thus providing a more reliable pseudo-GT to guide the student. Furthermore, to capitalize on the sharp boundary priors provided by the class-agnostic masks, we apply the same mask refinement described in Eq. (1) to the teacher’s features. For simplicity, we denote this refined output as FSelf-Distilled. As shown in Fig. 5, FSelf-Distilled reduces the blurry artifacts in FMask-Bicubic and captures the underlying geometry much better.

Further Discussion on Pseudo-GT Choices. Earlier in this section, we identified low-resolution loss as a bottle-

neck in prior training objectives. Here, we expand on that discussion by addressing additional key factors uncovered during our pseudo-GT exploration. In particular, Tab. 2 compares our approach with three alternative pseudo-GT choices: (1) depth maps, (2) per-image optimized high-resolution features from FeatUp-Implicit [12], and (3) features from 2× larger inputs, as in LiFT [52]. First, pseudo-GT should be task-agnostic to ensure generalizability across downstream tasks. For example, although depth maps contain fine-grained geometric details, they lack semantic information, potentially limiting the upsampler’s ability to maintain global semantic consistency. Furthermore, pseudo-GT must capture image geometry accurately, maintain sharp boundaries and clear color transitions, and minimize artifacts and noise. As shown in Fig. 5, per-image optimized features exhibit halo artifacts around objects and speckle noise in smooth regions, while 2× features introduces significant mosaic artifacts. FMask-Bicubic still suffers from blurriness and blobby artifacts inherited from bicubic upsampling. Among these, only FSelf-Distilled meets these requirements, demonstrating the effectiveness of our self-distillation framework. Finally, scalability is another crucial factor. Per-image optimization is computationally expensive, requiring 1 minute per HR feature map on an A100 GPU and 74MB for storage —equating to 35 GPU days and 3.5TB storage for just 50K images. In contrast, other methods generate pseudo-GT in real time with a single forward pass, taking less than 0.1 seconds per image. Overall, we find that FSelf-Distilled best satisfies these

requirements and delivers the strongest performance.

#### 5. Experiments

In this section, we compare LoftUp with other upsampling approaches on various tasks and conduct in-depth analysis on the strengths of LoftUp. By default, we use the DINOv2S/14 model [37] as VFM. Experiments with CLIP [41] and RADIO [43] show that DINOv2 produces the best downstream performance. Detailed results for CLIP and RADIO are provided in the supplements. Following previous works [12, 52], we resize input images to a resolution of 224 and upsample the VFM features to match the input resolution (14× for DINOv2-S/14). All upsamplers are trained on a 1M-image subset of the SA1B dataset [21] and additional implementation details are available in the supplements.

###### 5.1. Cross-Task Comparison

We compare LoftUp with two most recent task-agnostic feature upsamplers: FeatUp (JBU) [12] and LiFT [52], along with bilinear upsampling as a baseline. As demonstrated in Tab. 3 and Tab. 4, LoftUp consistently outperforms all alternatives on six downstream tasks.

Since we expect high-resolution features to enhance finegrained scene understanding, we first evaluate them on semantic segmentation following [12, 13]. We train a linear projection on upsampled features to predict coarse classes in COCO-Stuff [1] and Cityscapes [4]. LoftUp achieves 7.3% and 15.6% relative improvements over previous best methods. As expected, Cityscapes benefits more from feature upsampling due to its high number of small objects.

To assess the upsamplers’ generalizability to geometry prediction, we follow [10] and evaluate depth and normal estimation using a lightweight DPT decoder head [42] on the NAVI dataset [18]. Among the compared methods, LoftUp is the only upsampling method that significantly outperforms the low-res baseline, achieving 2.5% and 4.4% relative improvements in recall.

We further evaluate the temporal consistency of upsampled features through video object segmentation on DAVIS2017 [39], which requires consistent object tracking across video frames. Following prior works [17, 52], we use feature affinity maps to track objects across frames and report J Mean (mIoU, reflecting region similarity), F Mean (mean F-score, reflecting contour accuracy) and their average. LoftUp achieves significant performance improvements, with a 39.6% increase in J Mean and a 97.6% increase in F Mean over the low-resolution baseline, demonstrating a strong ability to identify object boundaries.

Finally, to assess the flexibility of usage across modalities like text and click embeddings, we evaluate on zeroshot open-vocabulary segmentation with ProxyCLIP [23] where DINOv2 features adjust CLIP features spatially for

Architecture COCO Cityscapes Depth Rec. Normal Rec.

Low-res 51.21 36.54 89.08 69.56 Resize-conv 56.05 45.94 88.38 65.59 LIIF 52.24 40.36 79.85 46.06 FeatUp-JBU 57.90 44.83 89.38 68.83 LoftUp 61.11 53.10 91.35 72.61

Table 5. Architecture comparison on semantic segmentation and depth and normal estimation. All upsamplers are trained using LoftUp training objective.

better vision-language alignment, and on interactive segmentation using a modified SimpleClick [27] where visual features are incorporated with click embeddings for segmentation. Results show that LoftUp is the only method to achieve consistent, significant improvements over the lowres and bilinear baselines, demonstrating its adaptability to multimodal alignment.

###### 5.2. More Analysis of LoftUp Design

Our pseudo-GT provides strong training signal. In Fig. 5 and Tab. 2, we already discuss and compare different pseudo-GT choices both qualitatively and quantitatively by training LoftUp architecture using different pseudo-GTs. In Fig. 7, we further show that our pseudo-GT can be used to enhance other upsamplers such as resize-conv and FeatUp (JBU), demonstrating broad applicability of the strong, full-resolution supervision signal of our pseudo-GT alone.

Our architecture significantly outperforms other alternatives when training under the same LoftUp training objective as in Sec. 4. As shown in Tab. 5, we compare LoftUp with resize-conv, LIIF, and FeatUp-JBU. We demonstrate that while other methods may bias towards certain tasks, LoftUp achieves the strongest performance across all of the tasks, indicating its stronger capacity to leverage the highquality pseudo-GT.

LoftUp benefits from the long-range, content-aware image-feature interaction thanks to the cross-attention mechanism. Instead of using a local kernel as in most previous feature upsampling works, our cross-attention mechanism effectively allows global attention and content-aware upsampling. In Fig. 8, we visualize the attended region of a pixel in high-res coordinate (in cross) in the corresponding low-resolution features (in dots) where the density of dots reflects the values of attention maps. As shown, the attended regions often exhibit semantically similar patterns across the image. Consequently, our upsampler leverages relevant information from the entire image, enhancing global semantic consistency and refining details at object boundaries.

LoftUp supports arbitrary upsampling scales via its coordinate-based design. As shown in Tab. 6, LoftUp maintains strong performance across varying input and feature resolutions. Here, “input res” refers to the image reso-

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

(a) Original image (b) Bilinear (c) LiFT (d) FeatUp (e) LoftUp (f) Groundtruth

- Figure 6. Visualization of predictions examples on semantic segmentation on COCO-Stuff [1] and depth estimation on NAVI [18]. We provide additional visualization in the supplements.

Resize-conv FeatUp-JBU LoftUp Arch

50

52

54

56

58

60

62

mIoU(%)

No Pseudo-GT LoftUp Pseudo-GT

- Figure 7. LoftUp pseudo-GT improves different upsamplers.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

- Figure 8. Visualization of attended regions (shown as dots) in the low-resolution features corresponding to a high-resolution pixel (marked as a cross). The density of dots represents the values in the attention map. LoftUp leverages relevant information from the global feature map to upsample features at each pixel.

Method Input res Feature res COCO Cityscapes

Low-res 224 16 51.21 36.54 2x-large input 448 32 55.42 50.29 4x-large input 896 64 56.79 58.45

56 59.87 48.20 112 60.75 51.04 224 61.11 53.10 448 60.61 52.29

224

LoftUp

56 60.16 51.57 112 61.50 56.46 224 61.78 58.55 448 62.26 61.24

448

Table 6. LoftUp maintains strong performance with diverse input sizes and upsampling scales.

curs nearly 4× the computational and memory cost. This highlights LoftUp’s efficiency and adaptability, making it well-suited for diverse scaling needs in downstream tasks.

LoftUp also enjoys relatively high efficiency. In Tab. 7, we compare the parameter size and inference time of each upsampler. Interestingly, while LoftUp has slightly more parameters than most alternatives (still less than 20% of the VFM), it is the second fastest upsampler after LiFT, which only does 2x upsampling. This efficiency arises because LoftUp directly maps coordinates to features, bypassing multiple intermediate upsampling layers used by other methods. Moreover, compared to the coordinatebased FeatUp-Implicit that requires per-image optimization, LoftUp is both significantly faster and more effective.

#### 6. Conclusion

In this work, we systematically explored upsampler architectures and training objectives for feature upsampling and introduced LoftUp, a coordinate-based crossattention transformer upsampler trained with our proposed self-distilled high-resolution pseudo-GT. Our experiments demonstrate that LoftUp generates sharp, fine-

lution fed into the VFM, while “feature res” denotes the final feature map size generated by LoftUp. Notably, LoftUp outperforms the original VFM backbone even when the latter processes images at 2× resolution—an approach that in-

Architecture COCO Cityscapes Params (M) Infer time (s/img) Bilinear 56.15 44.79 22.1 0.0604 Resize-conv 56.05 45.94 27.5 (+5.4) 0.0922 LIIF 52.24 40.36 24.0 (+1.9) 0.1757 LiFT 53.35 35.80 23.3 (+1.2) 0.0773 FeatUp-JBU 57.90 44.83 22.3 (+0.2) 0.1213 FeatUp-Implicit 51.12 44.81 22.5 (+0.4) 54.302 LoftUp 61.11 53.10 26.4 (+4.3) 0.0893

Table 7. Comparison of efficiency for different upsampler architectures. Inference time is computed on a single A100 GPU.

grained feature maps, enables global content-aware imagefeature interaction, supports arbitrary upsampling scales, and achieves fast inference. Notably, LoftUp outperforms existing methods across six diverse downstream tasks. We hope our work inspires further research in feature upsampling and contributes to the vision community by enhancing VFMs for a wide range of downstream applications.

#### References

- [1] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In CVPR, pages 1209–1218, 2018. 7, 8, 12, 15
- [2] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8628–8638,

2021. 2, 3

- [3] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. 2022. 1
- [4] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR,

2016. 6, 7, 12

- [5] Yutong Dai, Hao Lu, and Chunhua Shen. Learning affinityaware upsampling for deep image matting. In CVPR, pages 6841–6850, 2021. 3
- [6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 12
- [7] Claude E Duchon. Lanczos filtering in one and two dimensions. Journal of Applied Meteorology (1962-1982), pages 1016–1022, 1979. 2
- [8] Vincent Dumoulin and Francesco Visin. A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285, 2016. 3
- [9] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. NeurIPS, 27, 2014. 12
- [10] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun,

- Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3d awareness of visual foundation models. In CVPR, pages 21795–21806, 2024. 1, 7, 12
- [11] David F Fouhey, Wajahat Hussain, Abhinav Gupta, and Martial Hebert. Single image 3d without a single 3d image. In ICCV, pages 1053–1061, 2015. 12
- [12] Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. Featup: A model-agnostic framework for features at any resolution. In ICLR, 2024. 1, 2, 3, 4, 5, 6, 7, 12
- [13] Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T Freeman. Unsupervised semantic segmentation by distilling feature correspondences. arXiv preprint arXiv:2203.08414, 2022. 7, 12
- [14] Bharath Hariharan, Pablo Arbel´aez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In ICCV, pages 991–998. IEEE, 2011. 13
- [15] Kaiming He, Jian Sun, and Xiaoou Tang. Guided image filtering. PAMI, volume=35, number=6, pages=1397–1409, year=2012, publisher=IEEE. 3
- [16] Haiwen Huang, Songyou Peng, Dan Zhang, and Andreas Geiger. Renovating names in open-vocabulary segmentation benchmarks. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 1, 6
- [17] Allan Jabri, Andrew Owens, and Alexei Efros. Space-time correspondence as a contrastive random walk. NeurIPS, 33: 19545–19560, 2020. 6, 7, 12
- [18] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. NAVI: Categoryagnostic image collections with high-quality 3d shape and pose annotations. In NeurIPS, 2023. 7, 8, 12, 15
- [19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 13

- [20] Alexander Kirillov, Yuxin Wu, Kaiming He, and Ross Girshick. Pointrend: Image segmentation as rendering. In CVPR, pages 9799–9808, 2020. 3
- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pages 4015–4026, 2023. 1, 2, 5, 6, 7, 12, 13
- [22] Johannes Kopf, Michael F Cohen, Dani Lischinski, and Matt Uyttendaele. Joint bilateral upsampling. ACM TOG, 26(3): 96–es, 2007. 3, 5
- [23] Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Proxyclip: Proxy attention improves clip for open-vocabulary segmentation. In ECCV, 2024. 6, 7, 12
- [24] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In European conference on computer vision, pages 280–296. Springer, 2022. 1

- [25] Kai-En Lin, Yen-Chen Lin, Wei-Sheng Lai, Tsung-Yi Lin, Yi-Chang Shih, and Ravi Ramamoorthi. Vision transformer for nerf-based view synthesis from a single input image. pages 806–815, 2023. 3
- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755. Springer, 2014. 6, 12
- [27] Qin Liu, Zhenlin Xu, Gedas Bertasius, and Marc Niethammer. Simpleclick: Interactive image segmentation with simple vision transformers. In ICCV, pages 22290–22300, 2023. 1, 6, 7, 12, 13
- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 12
- [29] Hao Lu, Yutong Dai, Chunhua Shen, and Songcen Xu. Index networks. IEEE TPAMI, 44(1):242–255, 2020. 3
- [30] Hao Lu, Wenze Liu, Hongtao Fu, and Zhiguo Cao. Fade: Fusing the assets of decoder and encoder for task-agnostic upsampling. In ECCV, pages 231–247. Springer, 2022. 3
- [31] Hao Lu, Wenze Liu, Zixuan Ye, Hongtao Fu, Yuliang Liu, and Zhiguo Cao. Sapa: Similarity-aware point affiliation for feature upsampling. NeurIPS, 35:20889–20901, 2022. 3
- [32] David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In ICCV, pages 416–423. IEEE, 2001. 6, 13
- [33] Sky McKinley and Megan Levine. Cubic spline interpolation. College of the Redwoods, 45(1):1049–1060, 1998. 2
- [34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3, 13
- [35] Hyeonwoo Noh, Seunghoon Hong, and Bohyung Han. Learning deconvolution network for semantic segmentation. In ICCV, 2015. 3
- [36] Augustus Odena, Vincent Dumoulin, and Chris Olah. Deconvolution and checkerboard artifacts. Distill, 1(10):e3,

2016. 3

- [37] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. 2024. Featured Certification. 1, 2, 3, 7, 12, 14
- [38] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, pages 724–732, 2016. 6, 13
- [39] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alexander Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation.

arXiv:1704.00675, 2017. 2, 7, 12, 16

- [40] Gilles Puy, Spyros Gidaris, Alexandre Boulch, Oriane Sim´eoni, Corentin Sautier, Patrick P´erez, Andrei Bursuc, and Renaud Marlet. Three pillars improving vision foundation model distillation for lidar. In CVPR, pages 21519–21529,

2024. 6

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. pages 8748–8763. PmLR, 2021. 1, 3, 7, 12, 13
- [42] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, pages 12179–12188, 2021. 7
- [43] Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In CVPR, pages 12490– 12500, 2024. 1, 7, 13
- [44] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In CVPR, pages 13009–13018, 2024. 5
- [45] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 1, 2
- [46] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks,

2024. 5

- [47] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 3
- [48] Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. ” grabcut” interactive foreground extraction using iterated graph cuts. ACM TOG, 23(3):309–314, 2004. 6, 13
- [49] Wenzhe Shi, Jose Caballero, Lucas Theis, Ferenc Huszar, Andrew Aitken, Christian Ledig, and Zehan Wang. Is the deconvolution layer the same as a convolutional layer? arXiv preprint arXiv:1609.07009, 2016. 3
- [50] Konstantin Sofiiuk, Olga Barinova, and Anton Konushin. Adaptis: Adaptive instance selection network. In ICCV, pages 7355–7363, 2019. 13
- [51] Konstantin Sofiiuk, Ilya A Petrov, and Anton Konushin. Reviving iterative training with mask guidance for interactive segmentation. In ICIP, pages 3141–3145. IEEE, 2022. 13
- [52] Saksham Suri, Matthew Walmer, Kamal Gupta, and Abhinav Shrivastava. Lift: A surprisingly simple lightweight feature transform for dense vit descriptors. In ECCV, pages 110–

128. Springer, 2024. 1, 2, 3, 4, 6, 7, 12

- [53] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In CVPR, pages 10208–10217, 2024. 3
- [54] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 1
- [55] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 1
- [56] Narek Tumanyan, Assaf Singer, Shai Bagon, and Tali Dekel. Dino-tracker: Taming dino for self-supervised point tracking in a single video. In ECCV, pages 367–385. Springer, 2024. 1
- [57] Jiaqi Wang, Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy, and Dahua Lin. Carafe: Content-aware reassembly of features. In ICCV, pages 3007–3016, 2019. 3
- [58] Yangtao Wang, Xi Shen, Shell Xu Hu, Yuan Yuan, James L Crowley, and Dominique Vaufreydaz. Self-supervised transformers for unsupervised object discovery using normalized cut. In CVPR, pages 14543–14553, 2022. 6
- [59] Monika Wysocza´nska, Oriane Sim´eoni, Micha¨el Ramamonjisoa, Andrei Bursuc, Tomasz Trzci´nski, and Patrick P´erez. Clip-dinoiser: Teaching clip a few dino tricks for openvocabulary semantic segmentation. In ECCV, pages 320–

337. Springer, 2024. 6

- [60] Jianglong Ye, Naiyan Wang, and Xiaolong Wang. Featurenerf: Learning generalizable nerfs by distilling foundation models. In ICCV, pages 8962–8973, 2023. 3, 13
- [61] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. 2021 ieee. In CVPR, 2020. 3, 13
- [62] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional clip. Advances in Neural Information Processing Systems, 36:32215–32234,

2023. 1

- [63] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pages 11975–11986, 2023. 1
- [64] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, pages 633–641, 2017. 6, 12

## LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models

### Supplementary Material

This supplementary material to the main paper “LoftUp: earning a Coordinate-Based Feature Upsampler for Vision Foundation Models” is structured as follows:

- • In Appendix A, we explain more implementation details of LoftUp training and the downstream tasks.
- • In Appendix B, we show more quantitative results of LoftUp with CLIP and RADIO backbones and ablate the architecture and dataset size choices of training LoftUp.
- • In Appendix C, we provide more visualization of upsampled features, prediction results on various tasks, pseudoGT, and cross-attention regions.

#### A. More Implementation Details

###### A.1. Training details of LoftUp

Our LoftUp upsampler is a 2-block cross-attention transformer that incorporates high-res image inputs with coordinates using an additional convolutional layer and low-res VFM features as keys and values in the cross-attention layers. Each transformer block consists of 1 cross-attention layer and 1 feedforward layer as in ViT [6]. To train LoftUp, we use a batch size of 8 and AdamW [28] optimizer with a learning rate of 1e-3 in Stage 1 and 1e-4 in Stage 2 for more stable improvement during self-distillation. In Stage 2, we take 2 random crops per image to construct crop IHR , and update the teacher upsampler’s weights every 10 steps using the EMA of the student upsampler with a decay factor of 0.99. In both stages, we use α = 0.8 for mask refinement when constructing pseudo-GT to balance sharp boundaries from masks and the fine-grained details from high-resolution features within each mask region. For all upsamplers, including our compared ones, we train for 1 epoch on a 1M-image subset of SA1B dataset [21].

###### A.2. Task setups

Semantic segmentation. Following [12, 13], we perform semantic segmentation on coarse classes in COCO-Stuff [1] (27 classes) and Cityscapes [4] (19 classes) and report mean Intersection-over-Union (mIoU) for each dataset. We train a linear decoder layer on upsampled features with a batch size of 8 and AdamW optimizer [28] with a learning rate of 1e-4 for 10 epochs.

Depth and normal estimation. Following [10], we evaluate depth and normal estimation using NAVI dataset [18] and train a DPT decoder head with 7 convolutional layers on top of the VFM features. We use a batch size of 8 and

AdamW optimizer [28] with a learning rate of 5e-4 for 10 epochs. Following prior works [9–11], we report the rootmean-squared prediction error (RMSE) for both tasks and recall at δ3 for scale-invariant depth estimation and at 30◦ for normal estimation. Here δ3 is computed as the number of pixels whose ratio of depth prediction to groundtruth is less than 1.253:

1 N j∈N

δ3(dpr,dgt) =

max

dprj dgtj

,

dgtj dprj

< 1.253,

where dpr is predicted depth and dgt is groundtruth depth.

Video object segmentation. This task involves propagating an object segmentation mask across video frames, given the ground truth mask for the first frame. Following prior evaluation protocols [17, 52], we compute dense feature affinity maps between frames to track objects. Performance is assessed using three metrics: J Mean, F Mean, and J & F Mean. Specifically, J Mean denotes the average Intersection-over-Union (IoU) between predicted segmentations and groundtruth masks, while F Mean represents the average F-score, measuring contour accuracy via precision and recall against groundtruth boundaries.We evaluate our method on the DAVIS validation set [39], a popular benchmark for video object segmentation. The dataset comprises 30 videos of varying lengths, each containing between 1 and 4 objects.

Zero-shot Open-Vocabulary Segmentation. We incorporate upsampled VFM features into ProxyCLIP [23], a stateof-the-art method for zero-shot open-vocabulary segmentation (OVSeg), and evaluate on three popular OVSeg benchmarks: COCO [26], Cityscapes [4], and ADE20K [64]. ProxyCLIP enhances CLIP features by leveraging spatial feature correspondence from VFMs as proxy attention, effectively inheriting the strong local consistency of VFMs while retaining CLIP’s remarkable zero-shot transferability. Due to the high computational cost of proxy attention, we perform upsampling to 8× for all upsampling methods. We use CLIP ViT-B/16 [41] as the CLIP backbone, DINOv2S/14 [37] as the proxy VFM, and set the input resolution to 336px, matching the resolution of the CLIP backbone.

Interactive Segmentation. We adapt the SimpleClick [27] architecture to evaluate upsampled features. Specifically, we use a frozen VFM backbone and train a single-layer click encoder that directly adds to the image patch embedding, along with a three-layer convolutional decoder head

on top of the upsampled features for interactive segmentation. For training, we follow prior works [27, 51] and use the SBD dataset [14] to train for 20 epochs with the normalized focal loss [50, 51]. We employ the Adam optimizer [19] with a learning rate of 5e-5 and a batch size of 8. For evaluation, following common practice [21, 27, 51], we sample the first click point as the farthest point from the object boundary, and report the mean IoU of the predicted segmentation masks with the groundtruth, denoted as IoU@1 Click. We report results on three popular interactive segmentation benchmarks: GrabCut [48], Berkeley [32], and DAVIS [38].

#### B. More Quantitative Results

Resolution Upsampler COCO Cityscapes

NA 40.30 30.79 Bilinear 47.12 39.84 FeatUp 52.08 33.50

224

##### LoftUp 52.58 44.66

NA 42.14 37.36 Bilinear 48.32 45.58 FeatUp 52.55 40.00

448

##### LoftUp 53.87 50.14

- Table B.1. Comparison of feature upsampers when VFM is CLIP-B/16 [41].

Resolution Upsampler COCO Cityscapes

224

NA 51.00 34.42 Bilinear 56.77 43.09 FeatUp 56.59 42.23 LoftUp 58.36 46.98

448

NA 58.94 49.30 Bilinear 62.29 57.42 FeatUp 62.18 56.58 LoftUp 63.55 60.83

- Table B.2. Comparison of feature upsampers when VFM is RADIOv2.5-B [43].

Upsampling CLIP and RADIO. In Tab. B.1 and Tab. B.2, we compare LoftUp with FeatUp and a bilinear upsampling baseline using CLIP [41] and RADIO [43] as VFM backbones. The upsamplers are trained following the same procedure as on the DINOv2 backbone and evaluated at resolutions 224 and 448. As with DINOv2, LoftUp consistently

Feat PE Image conv # blocks # Train data COCO Cityscapes

- (a)

no 1x1 2 50k 56.46 43.13 learnable 1x1 2 50k 57.89 46.06 learnable 3x3 2 50k 58.40 47.35

Sine 1x1 2 50k 58.10 47.24 Sine 3x3 2 50k 58.65 48.63

- (b)

- Sine 3x3 1 50k 57.94 48.13

- Sine 3x3 2 50k 58.65 48.63

- Sine 3x3 3 50k 58.35 48.32

- (c)

Sine 3x3 2 50k 58.65 48.63 Sine 3x3 2 200k 59.40 49.84 Sine 3x3 2 1M 59.87 50.43

Table B.3. Ablation of architecture and training data size choices. Upsamplers are trained using only Stage 1 loss for convenience.

outperforms all baselines when using CLIP and RADIO as VFM backbone, demonstrating the general applicability of our approach across different VFMs.

Ablation on the architecture and training data size. In Tab. B.3, we conduct an ablation study on both the architecture components of LoftUp and the training data size. For convenience, the upsamplers are trained using only the Stage 1 training objective. Specifically, in experiment (a), we demonstrate that employing a sinusoidal positional encoding for the low-resolution features—combined with a 3x3 convolutional layer to process the high-resolution coordinates and image inputs—yields improved performance. This result is in line with prior work showing that sinusoidal positional encodings excel in coordinate-based methods [34, 60, 61] and that stronger image processing layers help better integrate high-resolution information. In experiment (b), we observe that two blocks of the cross-attention transformer are sufficient for optimal feature upsampling, with performance saturating at greater depths. Finally, in experiment (c), we find that training with a larger dataset improves performance, although the benefits begin to diminish as the dataset size increases. Consequently, we select a 1M-subset of the SA1B dataset [21] to achieve the best balance among data diversity, model performance, and training time.

#### C. More Visualization

We further provide more visualization examples of upsampled features of various methods in Fig. C.1, more prediction examples in semantic segmentation, depth estimation, and video object segmentation in Fig. C.2 and Fig. C.3, more examples of different pseduo-GT in Fig. C.4, and more examples of the attended regions of a high-resolution pixel in Fig. C.5.

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

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

(a) Original image (b) Low res (c) Bilinear (d) Resize-conv (e) LIIF (f) LiFT (g) FeatUp (h) LoftUp(Ours)

###### Figure C.1. More visualization of features from various upsamplers. Backbone is DINOv2-S/14 [37].

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

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

(a) Original image (b) Bilinear (c) LiFT (d) FeatUp (e) LoftUp (f) Groundtruth

- Figure C.2. More visualization of predictions examples on semantic segmentation on COCO-Stuff [1] and depth estimation on NAVI [18].

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

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

(a) Original image (b) Bilinear (c) LiFT (d) FeatUp (e) LoftUp (f) Groundtruth

- Figure C.3. Visualization of prediction examples of video object segmentation on the DAVIS 2017 dataset [39]. Each image displays its corresponding frame number in the top right corner. The groundtruth segmentation for the 0-th frame is provided, and dense feature affinity maps are employed to propagate its segmentation labels to subsequent frames. We can see that LoftUp outperforms all the other baselines in accurately tracking the objects across the frames.

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

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

(a) Original image (b) Per-image optimized (c) 2x features (LiFT) (d) Mask-Bicubic (e) Self-Distilled

###### Figure C.4. More visualization of different pseudo-GT.

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

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

###### Figure C.5. Visualization of attended region (in dots) in the low-res features of a high-res pixel (in cross). The density of dots reflects the value of the attention map. LoftUp is able to use relevant information across the global feature map for upsampling features at each pixel.

