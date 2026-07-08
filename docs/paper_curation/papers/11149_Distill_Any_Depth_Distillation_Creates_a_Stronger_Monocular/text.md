## Distill Any Depth: Distillation Creates a Stronger Monocular Depth Estimator

Xiankang He∗1,2 Dongyan Guo∗1 Hongji Li2,3 Ruibo Li4 Ying Cui1 Chi Zhang†2 1Zhejiang University of Technology 2 AGI Lab, Westlake University 3Lanzhou University 4Nanyang Technological University

{hexiankang577, 3420670269neon}@gmail.com {guodongyan,cuiying}@zjut.edu.cn ruibo001@e.ntu.edu.sg chizhang@westlake.edu.cn

# arXiv:2502.19204v2[cs.CV]21Apr2025

https://distill-any-depth-official.github.io/

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

[Figure 25]

[Figure 26]

Figure 1: Zero-shot prediction on in-the-wild images. Our model, distilled from Genpercept [49] and DepthAnythingv2 [51], outperforms other methods by delivering more accurate depth details and exhibiting superior generalization for monocular depth estimation on in-the-wild images.

### Abstract

which integrates both global and local depth cues to enhance pseudo-label quality. We also introduce an assistant-guided distillation strategy that incorporates complementary depth priors from a diffusion-based teacher model, enhancing supervision diversity and robustness. Extensive experiments on benchmark datasets demonstrate that our approach significantly outperforms state-of-the-art methods, both quantitatively and qualitatively.

Recent advances in zero-shot monocular depth estimation(MDE) have significantly improved generalization by unifying depth distributions through normalized depth representations and by leveraging large-scale unlabeled data via pseudo-label distillation. However, existing methods that rely on global depth normalization treat all depth values equally, which can amplify noise in pseudo-labels and reduce distillation effectiveness. In this paper, we present a systematic analysis of depth normalization strategies in the context of pseudo-label distillation. Our study shows that, under recent distillation paradigms (e.g., shared-context distillation), normalization is not always necessary—omitting it can help mitigate the impact of noisy supervision. Furthermore, rather than focusing solely on how depth information is represented, we propose Cross-Context Distillation,

### 1. Introduction

Monocular depth estimation (MDE) predicts scene depth from a single RGB image, offering greater flexibility compared to stereo or multi-view methods, and benefiting a wide range of applications, such as autonomous driving and robotic perception [11, 13, 18, 52, 30]. Recent research on zero-shot MDE models [37, 55, 47, 24] aims to handle diverse scenarios, but training such models requires largescale, diverse depth data, which is often limited by the need for specialized equipment [31, 54]. A promising solution is

*denotes co-first authorship. This work was done while Xiankang He was a visiting student at the AGI Lab, Westlake University.

† denotes corresponding author.

using large-scale unlabeled data, which has shown success in tasks like classification and segmentation [27, 63, 48]. Studies like DepthAnything [50] highlight the effectiveness of using pseudo labels from teacher models for training student models.

Global Depth GT

[Figure 27]

Error

[Figure 28]

Crop

Affine-invariant Depth Prediction

[Figure 29]

[Figure 30]

[Figure 31]

Global Leastsquare

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- ①

Error Map

- ②

Local Depth GT

[Figure 36]

[Figure 37]

[Figure 38]

To enable training on such a diverse, mixed dataset, most state-of-the-art methods [51, 40, 55] employ scale-and-shift invariant (SSI) depth representations for loss computation. This approach normalizes raw depth values within an image, making them invariant to scaling and shifting, and ensures that the model learns to focus on relative depth relationships rather than absolute values. The SSI representation facilitates the joint use of diverse depth data, thereby improving the model’s ability to generalize across different scenes [38, 5]. Similarly, during evaluation, the metric depth of the prediction is recovered by solving for the unknown scale and shift coefficients of the predicted depth using least squares, ensuring the application of standard evaluation metrics.

[Figure 39]

Local LeastCrop square

[Figure 40]

[Figure 41]

(a) Least-square strategy

AbsRel ① ②

[Figure 42]

Global Least-square Local Least-square

[Figure 43]

14.0

8.3

###### 7.2 6.2

3.8 3.1 3.73.2

3.9 3.1

[Figure 44]

KITTI NYU ScanNet DIODE

ETH3D

(b) Metric of different way of Least-square

Figure 2: Issue with Global Normalization (SSI). In (a), we compare two alignment strategies for the central w/2,h/2 region: (1) Global Least-Square, where alignment is applied to the full image before cropping, and (2) Local Least-Square, where alignment is performed on the cropped region. Metrics are computed on the cropped region. As shown in (b), the outperformed local strategy demonstrates that global normalization degrades local accuracy compared to local normalization.

Despite its advantages, using SSI depth representation for pseudo-label distillation in MDE models presents several issues. Specifically, the inherent normalization process in SSI loss makes the depth prediction at a given pixel not only dependent on the teacher model’s raw prediction at that location but also influenced by the depth values in other regions of the image. This becomes problematic because pseudo-labels inherently introduce noise. Even if certain local regions are predicted accurately, inaccuracies in other regions can negatively affect depth estimates after global normalization, leading to suboptimal distillation results. As shown in Fig. 2, we empirically demonstrate that normalizing depth maps globally tends to degrade the accuracy of local regions, as compared to only applying normalization within localized regions during evaluation.

approach yields more detailed and reliable depth predictions.

To harness the strengths of both, we propose using a diffusion-based model as the teacher assistant to generate pseudo-labels, which are then used to supervise the student model. This strategy enables the student model to learn from the detailed depth information provided by diffusionbased models, while also benefiting from the precision and efficiency of encoder-decoder models.

Building on this insight, in this paper, we investigate the issue of depth normalization in pseudo-label distillation. We analyze various depth normalization strategies, including global normalization, local normalization, hybrid globallocal approaches, and the absence of normalization. Through empirical experiments, we explore how each depth representation impacts the performance of different distillation designs, especially when using pseudo-labels for training.

To validate the effectiveness of our design, we conduct extensive experiments on various benchmark datasets. The empirical results show that our method significantly outperforms existing baselines qualitatively and quantitatively. The contributions can be summarized below: 1) We systematically analyze the role of different depth normalization strategies in pseudo-label distillation, providing insights into their effects on MDE performance. 2) To enhance the quality of pseudo-labels, we propose Cross-Context Distillation, a hybrid local-global framework that leverages fine-grained details and global depth relationships; a teacher assistant that harnesses the complementary strengths of diverse depth estimation models to further improve robustness and accuracy. 3) We conduct extensive experiments on benchmark datasets, demonstrating that our method outperforms state-of-the-art approaches both quantitatively and qualitatively.

Rather than focusing solely on pseudo-label representation, we introduce Cross-Context Distillation, a method that integrates both global and local depth cues to enhance pseudo-label quality. Our findings reveal that local regions, when used for distillation, produce pseudo-labels that capture higher-quality depth details, improving the student model’s depth estimation accuracy. However, relying solely on local regions may overlook broader contextual relationships in the image. To address this, we combine both local and global inputs within a unified distillation framework. By leveraging the context-specific advantages of local distillation alongside the broader understanding provided by global methods, our

### 2. Related Work

##### 2.1. Monocular Depth Estimation

Monocular depth estimation (MDE) has evolved from hand-crafted methods to deep learning, significantly improving accuracy [11, 29, 12, 16, 62, 38]. Architectural refinements, such as multi-scale designs and attention mechanisms, have further enhanced feature extraction [21, 6, 61]. However, most models remain reliant on labeled data and struggle to generalize across diverse environments. Zeroshot MDE improves generalization by leveraging largescale datasets, geometric constraints, and multi-task learning [37, 55, 57, 60, 58]. Metric depth estimation incorporates intrinsic data for absolute depth learning [2, 56, 22, 35, 46, 4], while generative models such as Marigold refine depth details using diffusion priors [24, 49, 17, ?]. Despite these advances, effectively utilizing unlabeled data remains a challenge due to pseudo-label noise and inconsistencies across different contexts. DepthAnything [51] explores large-scale unlabeled data but struggles with pseudo-label reliability. PatchFusion [9, 32] improves depth estimation by refining high-resolution image representations but lacks adaptability in generative settings. To address these issues, we propose Cross-Context and Multi-Teacher Distillation, which enhances pseudo-label supervision by leveraging diverse contextual information and multiple expert models, improving both accuracy and generalization ability.

##### 2.2. Semi-supervised Monocular Depth Estimation

Semi-supervised depth estimation has garnered increasing attention, primarily leveraging temporal consistency to utilize unlabeled data more effectively [28, 19]. Some approaches [1, 44, 7, 53, 15] integrate stereo geometric constraints, enforcing left-right consistency in stereo video to enhance depth accuracy. Others incorporate additional supervision, such as semantic priors [36, 20]or generative adversarial networks (GANs). For instance, DepthGAN [23] refines depth predictions through adversarial learning. However, these methods often rely on temporal cues, stereo constraints, or other auxiliary information, limiting their applicability to broader and more general scenarios. Recent work [34] has explored pseudo-labeling for semi-supervised monocular depth estimation (MDE), but it lacks generative modeling capabilities, restricting its generalization across diverse environments. DepthAnything [50] demonstrates the effectiveness of large-scale unlabeled data in improving generalization; however, pseudo-label reliability remains a challenge. In contrast, our approach focuses on singleimage depth estimation, improving pseudo-label reliability and maximizing its effectiveness. By relying solely on unlabeled data without additional constraints, our method achieves a more accurate and generalizable MDE model.

### 3. Method

In this section, we introduce a novel distillation framework designed to leverage unlabeled images for training zero-shot Monocular Depth Estimation (MDE) models. We begin by exploring various depth normalization techniques in Section 3.1, followed by detailing our proposed distillation method in Section 3.2, which combines predictions across multiple contexts. The overall framework is illustrated in Fig. 3. Finally, we introduce an assistant-guided distillation scheme, in which a diffusion-based model acts as an auxiliary teacher to provide additional supervision for student training.

##### 3.1. Depth Normalization

Depth normalization is a crucial component of our framework as it adjusts the pseudo-depth labels dt from the teacher model and the depth predictions ds from the student model for effective loss computation. To understand the influence of normalization techniques on distillation performance, we systematically analyze several approaches commonly employed in prior works. These strategies are visually illustrated in Fig. 4.

Global Normalization: The first strategy we examine is the global normalization [50, 51] used in recent distillation methods. Global normalization [37] adjusts depth predictions using global statistics of the entire depth map. This strategy aims to ensure scale-and-shift invariance by normalizing depth values based on the median and mean absolute deviation of the depth map. For each pixel i, the normalized depth for the student model and pseudo-labels are computed as:

dsi − med(ds) 1 M

d˜si = Nglo(ds) =

M j=1 dsj − med(ds)

(1)

dti − med(dt) 1 M

d ˜ti = Nglo(dt) =

,

M j=1 dtj − med(dt)

where med(ds) and med(dt) are the medians of the predicted depth and pseudo depth, respectively. The final regression loss for distillation is computed as the average absolute difference between the normalized predicted depth and the normalized pseudo depth across all valid pixels M:

M

1 M

LDis =

i=1

d ˜si − d˜ti . (2)

Hybrid Normalization: In contrast to global normalization, Hierarchical Depth Normalization [59] employs a hybrid normalization approach by integrating both global and local depth information. This strategy is designed to preserve both the global structure and local geometry in the depth map. The process begins by dividing the depth range into S

[Figure 45]

[Figure 46]

- (1) Shared-Context Distillation
- (2) Local-Global Distillation

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- S T
- T T

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Teacher

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

Student

Local-Global loss

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

T

Random Crop

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Teacher

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

S

[Figure 93]

[Figure 94]

Crop following Teacher stage

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Student

- Figure 3: Overview of Cross-Context Distillation. Our method combines local and global depth information to enhance the student model’s predictions. It includes two scenarios: (1) Shared-Context Distillation, where both models use the same image for distillation; and (2) Local-Global Distillation, where the teacher predicts depth for overlapping patches while the student

predicts the full image. The Local-Global loss Llg (Top Right) ensures consistency between local and global predictions, enabling the student to learn both fine details and broad structures, improving accuracy and robustness.

[Figure 100]

Global Norm.

Local Norm.

Hybrid Norm.

No Norm.

Norm. Area

Pixel

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

- Figure 4: Normalization Strategies. We compare four normalization strategies: Global Norm [37], Hybrid Norm [59], Local Norm, and No Norm. The figure visualizes how each strategy processes pixels within the normalization region (Norm. Area). The red dot represents any pixel within the region.

pixels as part of a single context, akin to global normalization. In the case of S = 2, the depth range is divided into two segments, with each pixel being normalized within one of these two local contexts. Similarly, for S = 4, the depth range is split into four segments, allowing normalization to be performed within smaller, localized contexts. By adapting the normalization process to multiple levels of granularity, hybrid normalization achieves a balance between global coherence and local adaptability. For each context u, the normalized depth values for the student model Nu(dsi) and pseudo-labels Nu(dti) are calculated within the corresponding depth range. The loss for each pixel i is then computed by averaging the losses across all contexts Ui to which the pixel belongs:

1 |Ui| u∈U

LiDis =

Nu(dsi) − Nu(dti) , (3)

i

where |Ui| denotes the total number of groups (or contexts) that pixel i is associated with. To obtain the final loss LDis, we average the pixel-wise losses across all valid pixels M:

M

1 M

LiDis. (4)

LDis =

i=1

Local Normalization: In addition to global and hybrid normalization, we investigate Local Normalization, a strategy

segments, where S is selected from {1,2,4}. When S = 1, the entire depth range is normalized globally, treating all

###### RGB Local Crop Global Depth Local Depth

that focuses exclusively on the finest-scale groups used in hybrid normalization. This approach isolates the smallest local contexts for normalization, emphasizing the preservation of fine-grained depth details without considering hierarchical or global scales. Local normalization operates by dividing the depth range into the smallest groups, corresponding to S = 4 in the hybrid normalization framework, and each pixel is normalized within its local context. The loss for each pixel i is computed using a similar formulation as in hybrid normalization, but with ui now representing the local context for pixel i, defined by the smallest four-part group:

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

M

1 M

Nui(dsi) − Nui(dti) . (5)

LDis =

Figure 5: Different Inputs Lead to Different Pseudo Labels. Global Depth: The teacher model predicts depth using the entire image, and the local region’s prediction is cropped from the output. Local Depth: The teacher model directly takes the cropped local region as input, resulting in more refined and detailed depth estimates for that area, capturing finer details compared to using the entire image.

i=1

No Normalization: As a baseline, we also consider a direct depth regression approach with no explicit normalization. The absolute difference between raw student predictions and teacher pseudo-labels is used for loss computation:

M

1 M

dsi − dti , (6)

LDis =

model across different spatial contexts, improving its ability to generalize to varying scene structures. For the loss of shared-context distillation, the teacher and student models receive identical inputs and produce each depth prediction, denoted as dtlocal and dslocal:

i=1

This approach eliminates the need for normalization, assuming pseudo-depth labels naturally reside in the same domain as predictions. It provides insight into whether normalization enhances distillation effectiveness or if raw depth supervision suffices.

Lsc = LDis dslocal,dtlocal , (7)

##### 3.2. Distillation Pipeline

This loss encourages the student model to refine its finegrained predictions by directly aligning with the teacher’s outputs at local scales.

In this section, we introduce an enhanced distillation pipeline that integrates two complementary strategies: CrossContext Distillation andassistant-guided distillation. Both strategies aim to improve the quality of pseudo-label distillation, enhance the model’s fine-grained perception.

2) Local-Global Distillation: In this approach, the teacher and student models operate on different input contexts. The teacher model processes local cropped regions, generating fine-grained depth predictions, while the student model predicts a global depth map from the entire image. To ensure knowledge transfer, the teacher’s local depth predictions supervise the corresponding overlapping regions in the student’s global depth map. This strategy allows the student to integrate fine-grained local details into its holistic depth estimation. Formally, the teacher model produces multiple depth predictions for cropped regions, denoted as dtlocal

Cross-context Distillation. A key challenge in monocular depth distillation is the trade-off between local detail preservation and global depth consistency. As shown in Fig. 5, providing a local crop of an image as input to the teacher model enhances fine-grained details in the pseudo-depth labels, but it may fail to capture the overall scene structure. Conversely, using the entire image as input preserves the global depth structure but often lacks fine details. To address this limitation, we propose Cross-Context Distillation, a method that enables the student model to learn both local details and global structures simultaneously. Cross-context distillation consists of two key strategies:

, while the student generates a global depth map, dsglobal. The loss for Local-Global distillation is computed only over overlapping areas between the teacher’s local predictions and the corresponding regions in the student’s global depth map:

n

1) Shared-Context Distillation: In this setup, both the teacher and student models receive the same cropped region of the image as input. Instead of using the full image, we randomly sample a local patch of varying sizes from the original image and provide it as input to both models. This encourages the student model to learn from the teacher

N

1 N

LDis Crop(dsglobal),dtlocal

, (8)

Llg =

n

n=1

where Crop(·) extracts the overlapping region from the student’s depth prediction, and N is the total number of sampled

patches. This loss ensures that the student benefits from the detailed local supervision of the teacher model while maintaining global depth consistency. The total loss function integrates both local and cross-context losses along with additional constraints, including feature alignment and gradient preservation, as proposed in prior works [51]:

Ltotal = Lsc + λ1 · Llg + λ2 · Lfeat + λ3 · Lgrad. (9)

Here, λ1, λ2, and λ3 are weighting factors that balance the different loss components. By incorporating cross-context supervision, this framework effectively allows the student model to integrate both fine-grained details from local crops and structural coherence from global depth maps.

Assistant-Guided Distillation. In addition to cross-context distillation, we propose an assistant-guided distillation strategy to further enhance the quality and robustness of the distilled depth knowledge. This approach pairs a primary teacher [51] with a single auxiliary assistant, selected as a diffusion-based depth estimator [49], which leverages generative priors to complement the primary teacher’s predictions. This design leverages their complementary strengths: the primary teacher excels in providing efficient and globally consistent supervision, while the assistant offers fine-grained depth cues derived from its generative modeling capabilities. By drawing supervision from two distinct architectures trained with different paradigms(e.g., optimization strategies or data distributions), the student benefits from diverse knowledge sources, effectively mitigating biases and limitations inherent to a single teacher model. Formally, let M and Ma denote the primary and assistant models, respectively. During training, a probabilistic sampling mechanism determines whether supervision for each iteration is drawn from M or Ma. This stochastic guidance encourages the student to adapt to multiple supervision styles, fostering richer and more generalizable depth representations. Overall, this assistant-guided scheme introduces complementary and diversified pseudo-labels, reducing over-reliance on any single teacher and improving both generalization and depth estimation performance.

### 4. Experiment 4.1. Experimental Settings

Datasets. We train our proposed distillation framework on a subset of 200,000 unlabeled images from the SA-1B dataset [26], following the training protocol of DepthAnythingv2 [51]. For evaluation, we assess the distilled student model on five widely used depth estimation benchmarks. All test datasets are kept unseen during training, enabling a zeroshot evaluation of generalization performance. The chosen benchmarks include: NYUv2 [43], KITTI [14], ETH3D [42], ScanNet [8], and DIODE [45]. Additional dataset details are provided in the Appendix.

Metrics. We assess depth estimation performance using two key metrics: the mean absolute relative error (AbsRel) and δ1 accuracy. Following previous studies [37, 56, 24] on zero-shot MDE, we align predictions with ground truth in both scale and shift before evaluation.

Implementation. Our experiments use state-of-the-art monocular depth estimation models as teachers to generate pseudo-labels, supervising various student models in a distillation framework with only RGB images as input. In shared-context distillation, both teacher and student receive the same global region, extracted via random cropping from the original image. The crop maintains a 1:1 aspect ratio and is sampled within a range from 644 pixels to the shortest side of the image, then resized to 560 × 560 for prediction. In global-local distillation, the global region is cropped into overlapping local patches, each sized 560 × 560, for the teacher model to predict pseudo-labels. For assistant-guided distillation, we adopt a probabilistic sampling strategy where the primary teacher and the assistant model are selected with a ratio of 7:3, respectively. We train our best student model using the distillation pipeline for 20,000 iterations with a batch size of 8 on a single NVIDIA V100 GPU, initialized with pre-trained DAv2-Large weights. The learning rate is in tune with that of the corresponding student model. For DAv2 [51], the decoder learning rate is set to 5 × 10−5. For the total loss function, we set the parameters as follows: λ1 = 0.5, λ2 = 1.0 and λ3 = 2.0.

##### 4.2. Analysis

For the ablation study and analysis, we sample a subset of 50K images from SA-1B [26] as our training data, with an input image size of 560 × 560 for the network. We conduct experiments on two of the most challenging benchmarks, DIODE [45] and ETH3D [42], which include both indoor and outdoor scenes.

Analysis of Normalization across Cross-Context Distillation. We evaluate the impact of different depth normalization strategies on Cross-Context Distillation, as shown in Table 1. The results reveal that the optimal normalization method depends on the specific distillation design. In shared-context distillation, where all pseudo-labels are generated by a single teacher model, hybrid normalization achieves the best performance, closely followed by no normalization. The consistent domain across supervision signals reduces the need for normalization, enabling the model to better preserve local depth relationships. Unlike ground-truth-based training—where normalization is essential to align depth distributions across datasets captured by heterogeneous sensors or represented in varying formats (e.g., sparse vs. dense, relative vs. absolute)—pseudo-labels from a single model are inherently more uniform. Therefore, direct L1 loss without normalization can more faithfully supervise pixel-level depth without distortion from global rescaling. In contrast, global

###### RGB MiDaS v3.1 DepthAnythingv2 Marigold Genpercept Ours

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

Figure 6: Qualitative Comparison of Relative Depth Estimations. We present visual comparisons of depth predictions from our method (”Ours”) alongside other classic depth estimators (”MiDaS v3.1” [3], and models using DINOv2 [33] or SD as priors (”DepthAnythingv2 [51]”, ”Marigold” [24], ”Genpercept” [49]). Compared to state-of-the-art methods, the depth map produced by our model, particularly at the position indicated by the black arrow, exhibits finer granularity and more detailed depth estimation.

Table 1: Analysis of Normalization Strategies. Performance comparison of different normalization strategies across Shared-Context Distillation and Local-Global Distillation.

normalization introduces undesirable inter-pixel dependencies, while local normalization discards global structural coherence. In local-global distillation, hybrid normalization again proves most effective, likely due to its hierarchical design that enforces consistency across both local and global depth predictions. The relatively small gap between hybrid and global normalization suggests that our framework, which uses local cues to refine global predictions, effectively mitigates the limitations of global normalization. However, no normalization leads to a notable performance drop compared to the shared-context setting, indicating that localized regions in this case come from distinct depth domains, making direct L1 supervision less reliable. Local normalization, as before, sacrifices global consistency and thus underperforms.

ETH3D DIODE AbsRel↓ AbsRel↓

Method Normalization

Global Norm. 0.064 0.259

Shared-Context Distillation

No Norm. 0.057 0.239 Local Norm. 0.070 0.245

Hybrid Norm. 0.057 0.238

Global Norm. 0.065 0.239

No Norm. 0.273 0.300 Local Norm. 0.076 0.244

Local-Global Distillation

Ablation Study of Cross-Context Distillation. To further validate the effectiveness of our distillation framework, we conduct ablation studies by removing Shared-Context Distillation and Local-Global Distillation in Table 2. Without both components, the model degrades to a conventional distillation setup, resulting in significantly lower performance. Introducing Shared-Context Distillation with Hybrid Normalization notably improves accuracy, highlighting the benefits of a better normalization strategy with consistent context supervision. When using only Local-Global Distillation, the model still performs well, showing the effectiveness of region-wise depth refinement even without global context information. Combining both strategies yields the best results,

Hybrid Norm. 0.064 0.238

confirming that both components contribute significantly to improving the student model’s ability to utilize pseudolabels, demonstrating the robustness of our approach.

Cross-Architecture Distillation. To highlight the limitations of previous state-of-the-art distillation approaches employing global normalization, we compare their performance against the Hybrid Normalization strategy, which we utilize in our distillation framework, across diverse model architectures. To demonstrate the generalizability of our approach,

- Table 2: Effect of Cross-context Distillation. Performance comparison of various combinations of Shared-Context Distillation and Local-Global distillation on the ETH3D [42] and DIODE [45] datasets. The baseline corresponds to a simple shared-context approach with no random cropping. When neither method is applied, the model defaults to this baseline.

Shared-Context Distillation

Local-Global Distillation

ETH3D DIODE AbsRel↓ AbsRel↓

- ✗ ✗ 0.075 0.270
- ✗ ✓ 0.064 (−14.6%) 0.238 (−13.3%)

✓ ✗ 0.058 (−22.6%) 0.237 (−12.2%) ✓ ✓ 0.056 (−25.3%) 0.232 (−14.1%)

- Table 3: Comparison in Cross-Architecture Distillation. Evaluation of our distillation pipeline in the context of CrossArchitecture Distillation. We adopt different architectures as teacher and student models, where the Base represents the previous distillation method [51]. Our method consistently improves the performance of the distilled student models.

Training Loss

DIODE ETH3D AbsRel↓ AbsRel↓

Teacher Student

Base 0.290 0.110 Ours 0.262 (−9.6%) 0.098 (−10.9%)

DA-L DA-S

Base 0.313 0.147 Ours 0.295(−5.7%) 0.126(−14.3%)

DA-L Midas-L

Base 0.303 0.150 Ours 0.272 (−10.2%) 0.120(−20.0%)

Midas-L Midas-S

we conduct cross-architecture distillation experiments on both the state-of-the-art DepthAnything [51] and the classic MiDaS [37] architecture. Experiments are conducted using MiDaS [37] and DepthAnything [51] in four configurations (DA-L, MiDaS-L, DA-S, MiDaS-S), as shown in Table 3. Our method consistently outperforms previous global normalization-based distillation on both the DIODE [45] and ETH3D [42] datasets. These results demonstrate superior performance both within and across architectures, underscoring the limitations of global normalization in pseudo-label distillation.

Effect of Assistant-Guided Distillation. To validate the effectiveness of our proposedassistant-guided distillation strategy, we design a comparative experiment that introduces an additional assistant model to the conventional teacher-student distillation framework. Specifically, we adopt DepthAnything v2 as the primary teacher and GenPercept—a diffusion-based model—as the assistant. The student model shares the same architecture as DAv2-Large and is initialized with its pre-trained weights. This setup allows us to investigate whether supervision from two diverse

Table 4: Effect of Assistant-Guided Distillation. Bold values indicate the best performance. Our method integrates a primary teacher, DepthAnything v2 (denoted as ‘D’), with a diffusion-based assistant, GenPercept (denoted as ‘G’), leveraging their complementary strengths to produce higher-quality pseudo-labels. The student model, trained under this assistant-guided distillation framework, consistently achieves better accuracy than when distilled from the DAv2Large teacher alone.

Assistant-Guided Strategy

ETH3D DIODE AbsRel↓ AbsRel↓

Method

DepthAnything v2 w/o 0.131 0.262

Genpercept(Disparity) w/o 0.096 0.226 D + G Avg. 0.228 0.371 D + G Select. 0.054 0.258

architectures—trained under different paradigms—can offer complementary guidance that enhances both generalization and depth estimation performance. To explore the most effective way to combine pseudo-labels from the primary teacher and the assistant, we compare two assistant-guided strategies: (1) a weighted averaging approach (Avg.), which assigns greater weight to pixels where the two teachers exhibit high agreement, and (2) a selection-based strategy (Select.), which probabilistically samples the supervision signal from either teacher. While the averaging strategy attempts to leverage consistency between teachers, it often performs poorly due to conflicting pseudo-labels, where averaging can amplify errors. In contrast, the selection-based strategy allows the student to selectively absorb the strengths of each teacher, avoiding error reinforcement. As shown in Table 4, the Select. strategy significantly outperforms both individual teachers and the averaging method on the ETH3D benchmark, demonstrating the effectiveness ofassistant-guided distillation in delivering robust and diverse supervision.

##### 4.3. Comparison with State-of-the-Art

Quantitative Analysis. As shown in Table 5, our method achieves state-of-the-art performance across a diverse range of zero-shot depth estimation benchmarks. These include both structured indoor scenes (e.g., NYUv2 [43], ScanNet [8]) and challenging outdoor environments (e.g., KITTI [14], DIODE [45], ETH3D [42]), demonstrating strong generalization across domains with varying scene structures, lighting conditions, and depth statistics. To further validate the effectiveness and scalability of our distillation framework, we conduct evaluations on two representative model architectures: DepthAnythingv2, a recent stateof-the-art model based on DINOv2, and MiDaS, a classic and widely adopted encoder-decoder framework. For each setup, the student model is initialized with the corresponding pre-trained encoder and distilled using pseudo-labels

Table 5: Quantitative comparison with other affine-invariant depth estimators on several zero-shot benchmarks. The bold values indicate the best performance, and underscored represent the second-best results.

NYUv2 KITTI DIODE ScanNet ETH3D AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

Method

DiverseDepth [55] 0.117 0.875 0.190 0.704 0.376 0.631 0.108 0.882 0.228 0.694 MiDaS [37] 0.111 0.885 0.236 0.630 0.332 0.715 0.111 0.886 0.184 0.752 LeReS [47] 0.090 0.916 0.149 0.784 0.271 0.766 0.095 0.912 0.171 0.777 Omnidata [10] 0.074 0.945 0.149 0.835 0.339 0.742 0.077 0.935 0.166 0.778 HDN [59] 0.069 0.948 0.115 0.867 0.246 0.780 0.080 0.939 0.121 0.833 DPT [39] 0.098 0.903 0.100 0.901 0.182 0.758 0.078 0.938 0.078 0.946 DepthAnything v2 [50] 0.045 0.979 0.074 0.946 0.262 0.754 0.042 0.978 0.131 0.865 GenPercept [49] 0.058 0.969 0.080 0.934 0.226 0.741 0.063 0.960 0.096 0.959 Marigold [25] 0.055 0.961 0.099 0.916 0.308 0.773 0.064 0.951 0.065 0.960 MiDaS v3.1 [3] - 0.980 - 0.949 - - - - 0.061 0.968

Ours† 0.046 0.985 0.063 0.972 0.142 0.788 0.049 0.980 0.057 0.976 Ours∗ 0.043 0.981 0.070 0.949 0.233 0.753 0.043 0.980 0.054 0.981

† Cross-Context distillation applied to MiDaS v3.1, using a pre-trained MiDaS v3.1 model as the teacher. ∗ Cross-Context distillation applied to DepthAnythingv2-Large, using a pre-trained DAv2 model as the teacher.

generated by the teacher model. Our approach yields consistent improvements over both teacher models across all benchmarks, highlighting its effectiveness in learning from pseudo-labels. Notably, it establishes new state-of-the-art results in most cases, outperforming existing affine-invariant depth estimators and demonstrating the robustness of our method in both DAv2 and MiDaS settings. These results confirm that our distillation framework is broadly applicable and can effectively transfer knowledge across model scales and depth distributions, enabling the student model to surpass its teacher in both accuracy and generalization under zero-shot evaluation.

Qualitative analysis. We present a qualitative comparison of depth estimations from different models in Fig. 6, including recent state-of-the-art approaches and our student model, which shares the same architecture as DAv2 but is trained using our distillation framework. Compared with DAv2 [51], our method clearly preserves finer structural details—particularly in regions highlighted by arrows—thanks to the proposed cross-context distillation strategy and the assistant model’s enhanced detail perception. While diffusionbased MDE methods such as Marigold [24] and GenPercept [49] generate visually rich depth maps by leveraging generative priors, they are trained on a limited amount of synthetic data, which hinders their ability to maintain correct relative depth ordering in real-world scenes. This issue arises from the inherent stochasticity and creativity of their generation paradigms, which, although capable of producing accurate depth ordering in certain regions, may introduce

inconsistencies in others. In contrast, our student model effectively balances detail preservation and structural consistency. with shared-context distillation and local-global distillation, it achieves more reliable and robust depth estimations that are both locally detailed and globally consistent.

### 5. Conclusion

In this work, we investigate pseudo-label distillation strategies for MDE. We observe that the commonly used global normalization scheme tends to amplify noise in teacher-generated pseudo-labels, thereby impairing local depth accuracy. To address this issue, we propose CrossContext Distillation, which combines local refinement with global consistency through a more effective normalization strategy. This enables the model to learn both finegrained details and high-level structural context. Furthermore, ourassistant-guided distillation framework integrates diffusion-based generative priors as complementary guidance to traditional encoder-decoder networks, achieving state-of-the-art performance across multiple benchmarks.

### References

- [1] Ali Jahani Amiri, Shing Yan Loo, and Hong Zhang. Semisupervised monocular depth estimation with left-right consistency using deep neural network. 2019 IEEE International Conference on Robotics and Biomimetics (ROBIO), pages 602–607, 2019. 3
- [2] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. Zoedepth: Zero-shot transfer by com-

- bining relative and metric depth. arXiv preprint arXiv: 2302.12288, 2023. 3
- [3] Reiner Birkl, Diana Wofk, and Matthias M¨uller. Midas v3.1 – a model zoo for robust monocular relative depth estimation,

2023. 7, 9

- [4] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv, 2024. 3, 13
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021. 2
- [6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017. 3
- [7] Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. A large rgb-d dataset for semi-supervised monocular depth estimation. arXiv preprint arXiv: 1904.10230, 2019. 3
- [8] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5828–5839. IEEE, 2017. 6, 8, 13
- [9] John Doe and Jane Smith. Patchfusion: Multi-scale feature fusion for enhanced depth estimation. International Journal of Computer Vision, 131:1234–1250, 2023. 3, 13
- [10] Adnan Eftekhar, Mate Balog, et al. Omnidata: A pipeline for building synthetic data of complex 3d scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 9, 13
- [11] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2366–2374, 2014. 1, 3
- [12] Huazhu Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2002–2011, 2018. 3
- [13] Ravi Garg, Vijay Kumar BG, Gustavo Carneiro, and Ian Reid. Unsupervised learning of depth and ego-motion from video. In European Conference on Computer Vision, pages 556–573. Springer, 2016. 1
- [14] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3354–3361. IEEE, 2012. 6, 8, 13
- [15] Cl´ement Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Monodepth2: Self-supervised monocular depth estimation with left-right consistency. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 168–176, 2019. 3

- [16] Cl´ement Godard, Oisin Mac Aodha, and Gabriel J Brostow. Unsupervised monocular depth estimation with left-right consistency. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 270–279, 2017. 3
- [17] Ming Gui, Johannes Schusterbauer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Depthfm: Fast monocular depth estimation with flow matching, 2024. 3
- [18] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2485–2494,

2020. 1

- [19] Vitor Guizilini, Jie Li, Rares Ambrus, Sudeep Pillai, and Adrien Gaidon. Robust semi-supervised monocular depth estimation with reprojected distances. In Conference on robot learning, pages 503–512. PMLR, 2020. 3
- [20] Lukas Hoyer, Dengxin Dai, Qin Wang, Yuhua Chen, and Luc Van Gool. Improving semi-supervised and domain-adaptive semantic segmentation with self-supervised depth estimation. International Journal of Computer Vision, 131(8):2070–2096,

2023. 3

- [21] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7132–7141,

2018. 3

- [22] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. arXiv preprint arXiv:2404.15506, 2024. 3
- [23] Rongrong Ji, Ke Li, Yan Wang, Xiaoshuai Sun, Feng Guo, Xiaowei Guo, Yongjian Wu, Feiyue Huang, and Jiebo Luo. Semi-supervised adversarial monocular depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42:2410–2422, 2020. 3
- [24] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. 2024. 1, 3, 6, 7, 9, 13
- [25] Qianli Ke, Hanxiao Lu, Yingcong Zhang, et al. Marigold: Multi-modal 3d perception with diffusion models. arXiv preprint arXiv:2402.04567, 2024. 9
- [26] Alexander Kirillov, Eric Mintun, et al. Sa-1b: Segment anything 1-billion mask dataset. https:// segment-anything.com, 2023. 6, 13
- [27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv preprint arXiv: 2304.02643, 2023. 2
- [28] Yevhen Kuznietsov, Jorg Stuckler, and Bastian Leibe. Semisupervised deep learning for monocular depth map prediction. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6647–6655, 2017. 3
- [29] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab. Deeper depth prediction

- with fully convolutional residual networks. In Proceedings of the Fourth International Conference on 3D Vision (3DV), pages 239–248, 2016. 3
- [30] Yanhua Li, Qixing Zhang, and Liqian Zhang. Ar shadow: Real-time 3d object tracking and shadow rendering for mobile augmented reality. IEEE Transactions on Visualization and Computer Graphics, 26(9):2871–2881, 2020. 1
- [31] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016. 1
- [32] S. M. H. Miangoleh, Sebastian Dille, Long Mai, Sylvain Paris, and Ya˘gız Aksoy. Boosting monocular depth estimation models to high-resolution via content-adaptive multi-resolution merging. Computer Vision and Pattern Recognition, 2021. 3
- [33] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. TMLR, 2024. 7
- [34] Andra Petrovai and Sergiu Nedevschi. Exploiting pseudo labels in a self-supervised learning framework for improved monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1578–1588, 2022. 3
- [35] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [36] Pierluigi Zama Ramirez, Matteo Poggi, Fabio Tosi, S. Mattoccia, and L. D. Stefano. Geometry meets semantics for semi-supervised monocular depth estimation. Asian Conference on Computer Vision, 2018. 3
- [37] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020. 1, 3, 4, 6, 8, 9, 13
- [38] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12179–12188, 2021. 2, 3
- [39] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 9
- [40] Rene Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(3):1623–1637, 2020. 2

- [41] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV,

2021. 13

- [42] Thomas Sch¨ops, Torsten Sattler, and Marc Pollefeys. A multiview stereo benchmark with high-resolution images and multicamera videos. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3260–3269. IEEE, 2017. 6, 8, 13
- [43] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European Conference on Computer Vision (ECCV), pages 746–760. Springer, 2012. 6, 8, 13
- [44] Nikolai Smolyanskiy, Alexey Kamenev, and Stan Birchfield. On the importance of stereo for accurate depth estimation: An efficient semi-supervised deep neural network approach. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 1007–1015, 2018. 3
- [45] Igor Vasiljevic, Ayan Chakrabarti, Vladlen Koltun, and Jack Tumblin. Diode: A dense indoor and outdoor depth dataset. In IEEE International Conference on Computer Vision (ICCV), pages 896–905. IEEE, 2019. 6, 8, 13
- [46] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision, 2024. 3, 13, 16
- [47] Zhenyu Wei, Andreas Geiger, et al. Leres: Learning-based monocular depth estimation for all scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2021. 1, 9
- [48] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10687–10698,

2020. 2

- [49] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? arXiv preprint arXiv:2403.06090,

2024. 1, 3, 6, 7, 9, 13

- [50] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. 2024. 2, 3, 9
- [51] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024. 1, 2, 3, 6, 7, 8, 9, 13, 14
- [52] Nan Yang, Rui Wang, J¨org St¨uckler, and Daniel Cremers. D3vo: Deep depth, deep pose and deep uncertainty for monocular visual odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1281–1292, 2020. 1
- [53] Nan Yang, Rui Wang, J. St¨uckler, and D. Cremers. Deep virtual stereo odometry: Leveraging deep depth prediction for monocular direct sparse odometry. European Conference on Computer Vision, 2018. 3

- [54] Weicong Yin, Jianping Shi, and Yao Feng. Learning to recover 3d scene shape from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2042–2051, 2021. 1
- [55] Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, and Renyin Dou. Diversedepth: Affine-invariant depth prediction using diverse data. arXiv preprint arXiv:2002.00569, 2020. 1, 2, 3, 9
- [56] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. IEEE International Conference on Computer Vision, 2023. 3, 6, 13
- [57] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. Computer Vision and Pattern Recognition, 2020. 3
- [58] Ge Yongtao, Xu Guangkai, Zhao Zhiyue, Huang zheng, Sun libo, Sun Yanlong, Chen Hao, and Shen Chunhua. Geobench: Benchmarking and analyzing monocular geometry estimation models. arXiv preprint arXiv:2406.12671, 2024. 3
- [59] Chi Zhang, Wei Yin, Billzb Wang, Gang Yu, Bin Fu, and Chunhua Shen. Hierarchical normalization for robust monocular depth estimation. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 3, 4, 9
- [60] Chi Zhang, Wei Yin, Gang Yu, Zhibin Wang, Tao Chen, Bin Fu, Joey Tianyi Zhou, and Chunhua Shen. Robust geometrypreserving depth estimation using differentiable rendering. IEEE International Conference on Computer Vision, 2023. 3
- [61] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2881–2890, 2017. 3
- [62] Tinghui Zhou, Matthew Brown, Noah Snavely, and David G Lowe. Unsupervised learning of depth and ego-motion from video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1851–1858,

2017. 3

- [63] Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, and Quoc V Le. Rethinking pre-training and self-training. In Advances in Neural Information Processing Systems, volume 33, pages 3833–3845, 2020. 2

- A. Appendix
- B. Dataset Details

##### B.1. Datasets.

Our model is trained on SA-1B [26], a large-scale dataset comprising high-quality RGB images of diverse indoor and outdoor scenes. These high-fidelity images facilitate the generation of more detailed pseudo-labels, enabling robust depth estimation and fine-grained detail learning for realworld applications. SA-1B [26] is also the dataset employed in DAv2 [51]. For evaluation, we use established monocular depth benchmarks:

- • NYUv2 [43]: Indoor depth estimation and semantic segmentation dataset. We evaluate on the official test split of 654 samples.
- • KITTI [14]: Autonomous driving dataset with outdoor scenes and high-quality LiDAR ground truth depth. Following prior work, we use the 697-image test split. (Corrected number of images based on standard KITTI benchmark)
- • ETH3D [42]: High-resolution stereo images for indoor and outdoor depth estimation and 3D reconstruction. We evaluate on all 454 images.
- • ScanNet [8]: Large-scale RGB-D dataset for 3D scene reconstruction and semantic segmentation. We use a test split of 1000 samples.
- • DIODE [45]: Dense, high-quality depth maps for both indoor and outdoor environments. However, as noted in MoGe [46], this dataset exhibits artifacts in the depth values of the ground truth near the boundaries of the object.

For visualization, we also use images from Dav2 [51], GenPercept [49], PatchFusion [9], Hypersim [41], Omnidata [10], Depth Pro [4] and Gen-3.

##### B.2. Metrics.

We evaluate depth estimation using mean absolute relative error (AbsRel) and δ1 accuracy. AbsRel is defined as:

M

|di − d∗i | d∗

1 M

(10)

AbsRel =

i

i=1

where di is the predicted depth, d∗i is the ground truth, and M is the total number of depth values. δ1 accuracy measures the percentage of pixels where:

- d∗i

- di

di d∗

< 1.25 (11)

δ1 = max

,

i

indicating prediction accuracy within a specific tolerance. Following Metric3D [37, 56, 24], we align predictions with ground truth in scale and shift before evaluation.

### C. More Experiments

##### C.1. Implementation Details.

For visualization, our model uses Dav2 as the student model and is fine-tuned with Dav2 parameters as the pretrained weights. Since our training iterations and dataset size are relatively small, leveraging the strong prior knowledge of Dav2 allows us to achieve significant visual improvements quickly. Regarding Table 5, the model is fine-tuned with Dav2 but only uses the backbone parameters from Dav2. Other components, such as the DPT head, are initialized randomly. We found that training entirely with Dav2’s pre-trained parameters does not directly demonstrate the effectiveness of our method. By retaining only the encoder and training the decoder from scratch, the accuracy clearly shows the improvement in pseudo-label utilization due to our normalization strategy, as well as the effectiveness of our cross-context distillation approach.

##### C.2. Effect of Data Scaling.

To investigate the impact of dataset size on model performance, we conducted experiments with progressively larger training sets and compared our method against the SSI Loss baseline across five popular benchmarks. The results are averaged over these benchmarks. As shown in Fig. 7, we report the Absolute Relative Error (AbsRel) as the dataset size increases from 10K to 200K images. Our distillation pipeline consistently outperforms the traditional SSI-based global normalization approach across all dataset sizes. Notably, the performance gap between our method and the baseline widens as more training data is introduced. Moreover, our approach enables the student model to surpass the teacher’s performance using significantly less training data, highlighting its data efficiency.

##### C.3. Distilling Generative Models vs. DepthAnythingv2.

Beyond distilling encoder-decoder depth models, we extend our approach to generative models, specifically GenPercept [49], aiming to transfer their superior detail preservation to a more efficient student model. While diffusion-based depth estimators achieve fine-grained depth reconstruction, their high computational cost limits practical applications. We investigate whether their depth estimation capability can be effectively distilled into a lightweight DPT-based model. Experimental results in Fig. 8 show that compared to using DepthAnythingv2 as the teacher, distilling from a diffusionbased model yields a student model with significantly enhanced fine-detail prediction.

##### C.4. Qualitative Comparison with Baseline Distillation.

We present a qualitative comparison between our method and the previous distillation method [51], where the Base model relies solely on global normalization. We analyze the depth map details and the distribution differences between predicted and ground truth depths. The red diagonal lines represent the ground truth, with results closer to these lines indicating better performance. As shown in Fig. 9, our method produces smoother surfaces, sharper edges, and more detailed depth maps.

##### C.5. Additional Results on 3D reconstruction in the Wild.

Benefiting from MoGe’s advances in geometrypreserving depth estimation, we align the relative depth predicted by our model with MoGe’s outputs. Using the camera parameters estimated by MoGe, we project the aligned depth into 3D space to obtain visualizable point clouds. As shown in Fig. 10, these visualizations demonstrate the effectiveness and practical applicability of our model in unconstrained, real-world scenarios. Remarkably, our method performs well even on stylized or synthetic content, such as animestyle images, making it potentially useful for downstream tasks like virtual character modeling. Similarly, our model generates high-quality reconstructions for images from game engines. In real-world photographs—captured by consumer devices such as smartphones—the reconstructed point clouds preserve meaningful geometric structures and fine details. Furthermore, even in abstract scenes like sketches with missing visual cues, our model can infer plausible relative depth and recover a semantically coherent 3D layout of the entire scene.

##### C.6. Qualitative Comparison: Additional Results on Depth Estimation in the Wild.

As shown in Fig. 11, our model demonstrates strong generalization and robustness across a wide range of scenarios, including real-world indoor and outdoor environments, stylized virtual content such as anime and game engine renders, and information-sparse inputs like sketches or line drawings. Even in unconventional perspectives such as bird’s-eye cityscapes, the model preserves accurate relative depth and structural coherence. These results highlight its ability to deliver detailed and semantically meaningful depth predictions in both natural and synthetic domains, enabling practical applications in 3D reconstruction, content creation, and downstream tasks across real and virtual worlds.

[Figure 156]

[Figure 157]

Figure 7: Comparison of Data Scaling . Performance comparison of our model with SSI Loss as the dataset size increases, measured by the average AbsRel. The results indicate that our method consistently outperforms the baseline method.

RGB Dav2 as Teacher Genpercept as Teacher

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

- Figure 8: Distilled Generative Models: Instead of just distilling classical depth models, we also apply distillation to diffusionbased generative models, aiming for the student model to learn the rich details inherent in these models, which are often not fully reflected in standard accuracy metrics.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

PredictedDepth

Inputs Base Ours Base Ours

Ground Truth Ground Truth

Ground Truth

Ground Truth

Ground Truth

Ground Truth

Ground Truth

Ground Truth

[Figure 176]

[Figure 177]

PredictedDepthPredictedDepthPredictedDepth

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

PredictedDepthPredictedDepthPredictedDepthPredictedDepth

- Figure 9: Qualitative Comparison with Baseline Distillation. We compare our method with the baseline as the previous distillation method, which uses only global normalization. The red diagonal lines represent the ground truth, with results closer to the lines indicating better performance. Our method produces smoother surfaces, sharper edges, and more detailed depth maps.

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

RGB Front View Top View Left View

- Figure 10: Additional results on 3D reconstruction from in-the-wild RGB images. We present point clouds generated from our model’s predicted depth maps, aligned with geometry-preserving depth from MoGe [46]. These visualizations demonstrate the effectiveness and practical applicability of our model in unconstrained, real-world scenarios.

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

#### RGB Depth Map RGB Depth Map RGB Depth Map

- Figure 11: Additional Results on Depth Estimation in the Wild. We showcase more depth maps generated by our model on in-the-wild scenes, highlighting its robustness and precision.

