arXiv:2507.01634v1[cs.CV]2Jul2025

# Depth Anything at Any Condition

Boyuan Sun∗1, Modi Jin∗1, Bowen Yin1, Qibin Hou†1 1VCIP, School of Computer Science, Nankai University ∗Equal Contribution. †Corresponding author.

Abstract: We present Depth Anything at Any Condition (DepthAnything-AC), a foundation monocular depth estimation (MDE) model capable of handling diverse environmental conditions. Previous foundation MDE models achieve impressive performance across general scenes but not perform well in complex open-world environments that involve challenging conditions, such as illumination variations, adverse weather, and sensor-induced distortions. To overcome the challenges of data scarcity and the inability of generating high-quality pseudo-labels from corrupted images, we propose an unsupervised consistency regularization finetuning paradigm that requires only a relatively small amount of unlabeled data. Furthermore, we propose the Spatial Distance Constraint to explicitly enforce the model to learn patch-level relative relationships, resulting in clearer semantic boundaries and more accurate details. Experimental results demonstrate the zero-shot capabilities of DepthAnything-AC across diverse benchmarks, including real-world adverse weather benchmarks, synthetic corruption benchmarks, and general benchmarks.

Project Page: https://ghost233lism.github.io/depthanything-AC-page Code: https://github.com/HVision-NKU/DepthAnythingAC

HVision@Nankai

## 1 Introduction

Monocular depth estimation (MDE) [96, 26, 25, 5] is a crucial task in computer vision as it enables the direct extraction of a depth map from a single image. Recently, foundation MDE models [46, 32, 34, 43, 66, 31], especially Depth Anything series [89, 90] and DepthPro [7], have emerged, demonstrating impressive zeroshot capabilities in complex and diverse scenarios. These models significantly reduce the cost of acquiring accurate depth maps in general scenarios. As a result, depth maps have not only propelled advancements in traditional perception-driven tasks, such as RGB-D segmentation [100, 92, 93], robotics [42, 41], autonomous driving [33, 81, 71], and 3D reconstruction [95, 35, 97, 77, 14], but have also been increasingly adopted in emerging domains like AI-generated content (AIGC) [98, 65] and large multi-modal models [11, 55, 103, 70] to enhance scene understanding capabilities.

Compared to traditional approaches [20, 16], recent foundation MDE models often utilize a massive amount of training data to achieve robust performance in general scenarios. However, since the main training data [24, 9, 13] are captured under normal lighting and weather conditions, these models often struggle to perform well in more challenging environments, particularly those involving complex illumination, diverse weather patterns, and sensor-induced distortions, as illustrated in Fig. 1. This limitation restricts the applicability of foundation MDE models in open-world scenarios.

In this paper, we aim to propose a foundation MDE model capable of extracting depth maps under complex conditions while maintaining its general abilities. The main challenge is the lack of image-depth pairs for complex weather scenarios. While some existing datasets [10, 40, 52] attempt to include complex weather scenarios, they often focus on a specific domain, which poses the potential risk of undermining the foundation model if we directly finetune on them. Meanwhile, as shown in Fig. 1, existing approaches [90, 7] struggle in complex real-world corrupted scenarios, making it difficult to obtain reliable pseudo-labels directly from such data. To address these issues, we propose a consistency regularization paradigm based on perturbation. Specifically, this paradigm utilizes a relatively small amount of unlabeled data from general scenes and applies augmentations such as lighting, blur, weather, and contrast on them, encouraging the model to predict consistent

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

V2

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

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

|[Figure 20]<br><br>[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]<br><br>[Figure 25]|
|---|

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 1 Zero-shot predictions. DepthAnything-AC achieves better details compared to other approaches, including Depth Anything V2 [90], DepthPro [7], and RobustDepth [61], under challenging lighting and climatic conditions.

results among different views. This aims to equip the model with robustness to complex lighting and weather conditions while maintaining its performance in general scenes.

Furthermore, existing models [58, 2, 89] focus solely on per-pixel depth differences while neglecting the relative positional relationships between pixels, which limits their ability to perceive object structures and makes them more susceptible to disruptions at object boundaries. The unsatisfactory results of DepthAnything V2 [90] and DepthPro [7] on object boundaries and details in Fig. 1 further support this point. To address this, we attempt to derive some insights from the underlying spatial geometric relationships. Specifically, we propose combining positional information with depth values to construct a more comprehensive representation for measuring patchwise spatial distances. As shown in Fig. 4, spatial distance maps can emphasize the semantic information of the object at the corresponding position with clear boundaries. Based on this, we further propose the Spatial Distance Constraint for monocular relative depth estimation to constrain the relative positioning between patches. By introducing spatial distance constraints between patches, the model can perceive the positional relationships and shapes of objects. This position-aware optimization method not only enhances the accuracy of object boundary localization but also reduces the prediction ambiguity caused by texture loss.

Our contributions are summarized as follows:

- • We propose a perturbation-based knowledge distillation paradigm that does not require any labeled data, which can expand the capabilities of foundation MDE models on real-world corrupted scenes while preserving their performance in general scenarios.
- • We show that the relative spatial positioning between patches is highly beneficial for modeling semantic boundaries. By extracting spatial distance as semantic priors from depth maps and spatial positions and using it as the spatial distance constraint, we can better reconstruct object structures and boundaries from corrupted texture information.
- • We propose the Depth Anything at Any Condition (DepthAnything-AC) model to improve depth estimation in more complex open-world scenarios, particularly under complex lighting and weather conditions. Experiments demonstrate that DepthAnything-AC outperforms state-of-the-art approaches in both realworld and synthetic noise scenarios.

[Figure 31]

[Figure 32]

[Figure 33]

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- Figure 2 Visualization of perturbation types. We apply a combination of darkness, weather, blur, and contrast perturbations during training. Note that different weather or blur perturbations do not appear simultaneously. For example, motion blur and zoom blur are mutually exclusive.

## 2 Related Work

### 2.1 Foundation Monocular Depth Estimation Model

Monocular depth estimation [3, 101, 57, 84, 4] is one of the most classic tasks in computer vision, and it has even been studied early before the advent of deep learning [72, 83, 62, 48]. Although deep learning-based methods [20, 16, 29] have achieved remarkable progress in in-domain scenarios [1, 24, 2, 82], their reliance on single-domain data limits their applicability in open-world environments.

MiDaS [6, 59] and Metric3D [94, 32] are pioneering works toward building a foundation MDE model, which integrates multiple depth datasets. The Depth Anything series [89, 90] establish a semi-supervised paradigm [91, 69] by leveraging large-scale unlabeled data for training, significantly enhancing the zero-shot performance of MDE models in diverse real-world scenarios. Some approaches tackle the MDE task with generative models [86, 63, 21, 30, 56]. Marigold [34] establishes a paradigm based on Stable Diffusion [60] to generate depth maps. DepthLab [50] leverages inpainting techniques to enhance structural details in depth predictions. DepthFM [28] introduces a flow matching strategy to enable fast and efficient sampling within diffusion-based frameworks.

Although these foundation MDE models have demonstrated impressive zero-shot performance in general “in-thewild” scenarios, they overlook the impact of real-world image degradations, such as various lighting, adverse weather conditions, and sensor-involved distortions, which commonly occur in open-world settings. As a result, their performance may deteriorate under such challenging conditions. Therefore, we aim to propose a more comprehensive foundation MDE model that not only maintains strong general capabilities but also exhibits robustness to more complex scenarios.

### 2.2 Robust Depth Estimation Models

Besides achieving zero-shot capability in general scenarios, obtaining robust monocular depth estimation models remains a critical objective. Some early self-supervised methods leverage GANs [107, 27] to improve monocular depth estimation in nighttime and adverse weather conditions. Works like ADFA [74], ADDS [49], and RNW [80] learn features from complex scenes, while ITDFA [102] generates nighttime pseudo-labels for supervision. Recently, many works utilize the diffusion-based pipeline [73, 53, 79] to enhance the robustness of depth estimation models.

To address dynamic ambiguity, methods like DeFeat-Net [68] and R4Dyn [23] propose joint learning and auxiliary modality integration, while others [75, 99, 18] focus on illumination variations with targeted constraints. Recently, increasing attention has shifted toward image-level analysis. For example, Robodepth [40] presents a depth enhancement scheme and robustness benchmark, while works like Robust-depth [61], Weatherdepth [78], STEPS [104], and Syn2Real [87] explore robustness under challenging lighting and weather through data perturbations, augmentations, or joint learning.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

teacher

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

Frozen Trained

[Figure 75]

[Figure 76]

student

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

perturbation

[Figure 93]

[Figure 94]

[Figure 95]

- Figure 3 Training pipeline for DepthAnyting-AC. The perturbation consistency framework encourages DepthAnythingAC to generate consistent predictions under augmentations while retaining generality via the frozen original model. To enhance semantic boundaries and details, the spatial distance constraint is used to strengthen the understanding of inter-patch relationships.

However, certain issues have been identified with these works. First, unlike foundation MDE models, they are primarily designed for specific domains such as autonomous driving, which restricts their applicability in open-world scenarios. Moreover, despite employing complex strategies to enhance model robustness, the lack of sufficient semantic perception capabilities often results in suboptimal boundary delineation and insufficient detail recovery.

## 3 Depth Anything at Any Condition

The overall framework of the proposed approach is depicted in Fig. 3. Our DepthAnything-AC consists of a perturbation-based consistency framework that encourages the model to produce consistent outputs for unlabeled images before and after different perturbations, thereby enhancing its robustness in complex scenarios, and a spatial distance constraint that enforces geometric relationships between patches to help recover object boundaries and details from corrupted images.

### 3.1 Perturbation-Based Consistency Framework

Considering the unsatisfactory performance of existing models on real-world corrupted images and the lack of densely annotated images, we propose a perturbation-based unsupervised consistency framework to extend the capabilities of the existing foundational MDE model in complex scenarios.

We first consider simulating common scenarios that degrade image quality in the real world. Given a normal image from a general scene, we focus on four typical scenarios: lighting, weather, blurriness, and contrast. Specifically, for each scenario, we apply perturbations with varying intensity levels as detailed in Fig. 2. For each image, one scenario is selected with a certain probability, and the corresponding perturbation is applied accordingly. Therefore, for an unlabeled image xi, the image augmented through regular transformations is denoted as xwi , and the image further processed with perturbations is denoted as xsi = A(xwi ).

Then, from the perspective of consistency regularization, we design the perturbation-based unsupervised consistency framework, as shown in Fig. 3. Specifically, given a mini-batch of unlabeled images {xi} and a foundation MDE model F, we define the consistency loss Lc as:

N

1 N

ℓ(F(xwi ),F(xsi)), (1)

Lc =

i=1

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

Image Spatial Distance Relation The Current Query Token

- Figure 4 Visualization of spatial distance relationships corresponding to different query tokens. The pentagram denotes the current query token, illustrating the spatial distance between the patch containing the pentagram and all other patches in the frame.

where xwi and xsi are with regular and perturbed images, respectively. This loss function enforces consistency between the model’s predictions under normal and perturbed scenarios, encouraging the model to provide consistent outputs across different corrupted conditions. The ℓ(y,yˆ) in Eqn. 1 is a measurement of the prediction disparity y and target disparity yˆ (Disparity is the inverse of relative depth, which is the direct output of the foundation MDE model). Inspired by DepthAnything [89], we define the affine-invariant loss ℓ as:

HW

yj − t(y) s(y) −

yj − t(ˆy)

1 HW

s(ˆy) |, (2) where t(y) and s(y) are the factors used to balance the scale between prediction and target.

|

ℓ =

j=1

1 HW

t(y) = Mean(y),s(y) =

HW

|yj − t(y)|. (3)

j=1

Also, while consistency regularization encourages the model to provide consistent predictions across images with different perturbations, the lack of precise label supervision introduces a potential risk: The model might prioritize consistency over correctness, which could degrade its performance in general scenes. To tackle this, we keep the initial foundation MDE model Fˆ frozen as guidance and use a knowledge distillation loss to guide the model’s predictions in the normal scene, as follows:

N

1 N

ℓ(F(xwi ),Fˆ(xwi )). (4)

Lkd =

i=1

### 3.2 Spatial Distance Constraint

Beyond the perturbation-based consistency framework, real-world corruptions often degrade image quality, making it challenging for existing foundation MDE models to accurately distinguish object boundaries and capture fine-grained details. To address this, we propose the Spatial Distance Relation (SDR), a geometric prior that captures relative distance among objects. It can serve as a complement to enhance the semantic perception capabilities of MDE models under adverse conditions.

The SDR SD consists of two parts, the position relation Sp and the depth relation Sd. Given an unlabeled image xi consisting of H×W patches, we denote the spatial coordinate of each patch as pi,j, where i ∈ {1,...,H} and j ∈ {1,...,W}. To characterize the spatial structure, we first compute the position relation Sp

by calculating the Euclidean distance between each pair of patches:

i,j

= {∥pi,j − pm,n∥2}, (5)

Sp

i,j

Dataset Indoor Outdoor Samples Used Ratio ADE20k [105, 106] ✓ ✓ 20K 20K 100% MegaDepth [44] ✓ 128K 100K 78% DIML [13, 37, 12, 38, 36] ✓ ✓ 927K 80K 8.6% VKITTI2 [9] ✓ 42K 42K 100% HRWSI [22, 85] ✓ 20K 20K 100% SA-1B [39] ✓ ✓ 11.1M 140K 1.3% COCO [47] ✓ ✓ 120K 120K 100% Pascal VOC 2012 [17] ✓ ✓ 10K 10K 100% AODRaw [45] ✓ ✓ 80K 8K 10%

- Table 1 Datasets used for training. In total, our DepthAnything-AC is fine-tuned on 540K unlabeled images, which is much less than the number used in the DepthAnything series [89, 90].

Method Encoder DA-2K DA-2K dark DA-2K fog DA-2K snow DA-2K blur

DynaDepth [99] ResNet 0.655 0.652 0.613 0.605 0.633 EC-Depth [67] ViT-S 0.753 0.732 0.724 0.713 0.701 STEPS [104] ResNet 0.577 0.587 0.581 0.561 0.577 robustdepth [61] ViT-S 0.724 0.716 0686 0.668 0.680 weather-depth [78] ViT-S 0.745 0.724 0.716 0.697 0.666 DepthPro[7] ViT-S 0.947 0.872 0.902 0.793 0.772 DepthAnything V1 [89] ViT-S 0.884 0.859 0.836 0.880 0.821 DepthAnything V2 [90] ViT-S 0.952 0.910 0.922 0.880 0.862 DepthAnything-AC ViT-S 0.953 0.923 0.929 0.892 0.880

- Table 2 Quantitative results on the enhanced multi-condition DA-2K benchmark, including complex light and climate conditions. The first and second ranked performances are indicated by bold and underline. The evaluation metric used is Accuracy ↑.

where m ∈ {1,2,...,H} and n ∈ {1,2,...,W}. This produces a position relation matrix Sp ∈ RHW×HW, which contains pairwise positional distances between each pair of patches. The position relation matrix is subsequently min-max normalized.

For the depth relation Sd

of patch pi,j, it is defined as the absolute difference in predicted disparities between patch pi,j and some patch pm,n:

ij

#### = {|F(xi)i,j − F(xi)m,n|}. (6)

#### Sd

i,j

Since the position relation Sp encodes the relative positional distance between any two patches in the image plane and the depth relation Sd represents the disparity distance between them, the relative geometric distance between each pair of patches can be formulated as:

#### SD = Sp2 + Sd2 ∈ RHW×HW. (7)

The physical interpretation of the spatial distance relation SD is a geometric distance between any pair of patches. Each row of SD reflects the degree of geometric proximity to a specific patch, inherently encoding rich semantic information. As illustrated in Fig. 4, the constructed spatial relations closely correlate with the semantic structure of objects within the image. This enables the model to leverage semantic priors in a training-free manner. Based on SDR, we propose the Spatial Distance Relation loss function as a constraint:

N

1 N

(SD(F(xsi)) − SˆD(Fˆ(xwi )))2. (8)

Ls =

i=1

This loss function encourages the model to maintain consistency between the spatial distance relations derived from the perturbed image and those from the original image predicted by the original model. It facilitates the recovery of semantic boundaries and fine details degraded by real-world corruptions.

NuScenes-night RobotCar-night DS-rain DS-cloud DS-fog AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

Method Encoder

DynaDepth [99] ResNet 0.381 0.394 0.512 0.294 0.239 0.606 0.172 0.608 0.144 0.901 EC-Depth [67] ViT-S 0.243 0.623 0.228 0.552 0.155 0.766 0.158 0.767 0.109 0.861 STEPS [104] ResNet 0.252 0.588 0.350 0.367 0.301 0.480 0.252 0.588 0.216 0.641 robustdepth [61] ViT-S 0.260 0.597 0.311 0.521 0.167 0.755 0.168 0.775 0.105 0.882 weather-depth [78] ViT-S - - - - 0.158 0.764 0.160 0.767 0.105 0.879 Syn2Real [87] ViT-S - - - - 0.171 0.729 - - 0.128 0.845 DepthPro [7] ViT-S 0.218 0.669 0.237 0.534 0.124 0.841 0.158 0.779 0.102 0.892 DepthAnything V1 [89] ViT-S 0.232 0.679 0.239 0.518 0.133 0.819 0.150 0.801 0.098 0.891 DepthAnything V2 [90] ViT-S 0.200 0.725 0.239 0.518 0.125 0.840 0.151 0.798 0.103 0.890 DepthAnything-AC ViT-S 0.198 0.727 0.227 0.555 0.125 0.840 0.149 0.801 0.103 0.889

- Table 3 Zero-shot relative depth estimation results on real complex benchmarks.

Method Encoder

Dark Snow Motion Gaussian

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

DynaDepth [99] ResNet 0.163 0.752 0.338 0.393 0.234 0.609 0.274 0.501 STEPS [104] ResNet 0.230 0.631 0.242 0.622 0.291 0.508 0.204 0.692 DepthPro [7] ViT-S 0.145 0.793 0.197 0.685 0.170 0.746 0.170 0.745 DepthAnything V2 [90] ViT-S 0.130 0.832 0.115 0.872 0.127 0.840 0.157 0.785 DepthAnything-AC ViT-S 0.130 0.834 0.114 0.873 0.126 0.841 0.153 0.793

- Table 4 Zero-shot relative depth estimation results on synthetic KITTI-C benchmarks.

Equipped with the perturbation-based consistency framework and the spatial distance constraint, the overall supervision is a combination of Lc, Lkd, and Ls:

L = λ1Lc + λ2Lkd + λ3Ls, (9) where λ1, λ2, and λ3 are weighting coefficients and set to [1/3, 1/3, 1/3] by default.

## 4 Experiments

### 4.1 Experiment Setup

Implementation details. DepthAnything-AC is fine-tuned based on DepthAnythingV2 [90]. Specifically, it adopts ViT-S [15] as the backbone and DPT [58] as the decoder. During training, we use the AdamW [51] optimizer with an initial learning rate of 5e-6, a weight decay of 0.01, a crop size of 518×518, and a batch size of 16. We train the model for 20 epochs with 4 NVIDIA RTX 3090 GPUs.

Training datasets. The datasets we collect for fine-tuning DepthAnything-AC model are summarized in Tab. 1. Our training set consists of 540K samples in total, which is relatively small compared to the Depth Anything series (around 63M) [89, 90], making up less than 1%. Among these datasets, except for AODRaw [45], which includes some scenes under complex natural conditions, the rest consist of images captured under normal lighting and weather conditions.

Evaluation benchmarks. We evaluate the zero-shot performance of DepthAnything-AC across various benchmarks. Specifically, we test on the enhanced multi-weather DA-2K benchmark, real-world scenarios with complex lighting and weather conditions, and synthetic benchmarks. Additionally, we assess performance on general benchmarks to ensure that DepthAnything-AC maintains its generalization ability. Specifically, we select the NuScenes-night [10], Robotcar-night [52], and Driving-Stereo [88] as the real-world benchmarks, and the KITTI-C [40] as the synthetic benchmark. As for general benchmarks, following previous methods, we use KITTI [24], NYU-D [54], Sintel [8], ETH3D [64], and DIODE [76]. All experiments are performed on a single NVIDIA RTX 3090 GPU.

###### KITTI NYU-D Sintel ETH3D DIODE

Method Encoder

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

DynaDepth [99] ViT-S 0.085 0.916 0.212 0.630 0.458 0.391 0.192 0.703 0.180 0.737 EC-Depth [67] ViT-S 0.075 0.931 0.220 0.606 0.447 0.405 0.181 0.779 0.172 0.760 STEPS [104] ResNet 0.204 0.683 0.220 0.606 0.458 0.375 0.197 0.699 0.214 0.652 robustdepth [61] ViT-S 0.076 0.929 0.174 0.718 0.411 0.436 0.185 0.776 0.182 0.741 weather-depth [78] ViT-S 0.075 0.932 0.167 0.733 0.419 0.443 0.181 0.771 0.170 0.760 DepthPro [7] ViT-S 0.068 0.947 0.052 0.972 0.240 0.682 0.057 0.963 0.060 0.952 DepthAnything V1 [89] ViT-S 0.082 0.936 0.053 0.972 0.211 0.736 0.064 0.966 0.078 0.940 DepthAnything V2 [90] ViT-S 0.083 0.934 0.051 0.973 0.240 0.712 0.064 0.969 0.073 0.943 DepthAnything-AC ViT-S 0.083 0.934 0.051 0.973 0.235 0.715 0.065 0.968 0.073 0.941

- Table 5 Zero-shot relative depth estimation results on general benchmarks.

Lc Lkd P Ls

DA-2K dark NuScenes-night Robotcar-night Motion Gaussian

Acc↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

✓ ✓ 0.911 0.196 0.727 0.239 0.519 0.126 0.840 0.157 0.786 ✓ ✓ ✓ 0.914 0.196 0.731 0.239 0.519 0.126 0.839 0.155 0.789

✓ ✓ ✓ 0.916 0.197 0.725 0.239 0.538 0.127 0.839 0.156 0.789 ✓ ✓ ✓ ✓ 0.923 0.198 0.727 0.227 0.555 0.126 0.841 0.153 0.793

- Table 6 Ablation study of different components. P denotes the perturbation.

### 4.2 Zero-Shot Relative Depth Estimation

Performance on multi-condition DA-2K benchmark. DA-2K is a large-scale and high-resolution modern benchmark introduced in DepthAnything v2 [90], which evaluates depth models based on pairwise relative depth comparisons. Beyond that, we further built the multi-condition DA-2K benchmark to evaluate the robust capability of foundation MDE models. Specifically, we augment the original DA-2K images with various types of perturbations, namely dark, fog, snow and blur. As shown in Tab. 2, DepthAnything-AC outperforms other models under complex lighting and climatic conditions (e.g., 0.862 → 0.880 on DA-2K blur).

Performance on real complex lighting and weather scenarios. We compare our DepthAnything-AC with others on five real complex benchmarks with realistic lighting and climate conditions. As shown in the Tab. 3, our results outperform both robust depth estimation models and foundation MDE models. For instance, compared to Depth Anything V2 [90], our DepthAnything-AC achieves 0.037 improvements of δ1 metric on the Robotcarnight [52] benchmark.

Performance on KITTI-C benchmarks. KITTI-C [40] is a widely used benchmark proposed by RoboDepth [40],

which introduces synthetic perturbations to KITTI [24] images to simulate various challenging scenarios. We select four subsets: darkness, snow, motion and Gaussian noise, which reflect complexities in lighting and climate conditions. The experimental results, as presented in Tab. 4, demonstrate consistent improvements across multiple performance metrics.

Performance on general benchmarks. In addition, comparisons are made on generalized datasets. As demonstrated in Tab. 5, the performance of our model on the generalized dataset is comparable to that of the existing foundation MDE models. This finding suggests that the proposed approach can enhance the robustness of the depth estimation model without compromising its generalizability.

Limited improvements on traditional benchmarks. Combining Tab. 3, Tab. 4, and Tab. 5, we observe that the improvements of DepthAnything-AC over other foundation MDE models are somewhat limited. Similar phenomena can also be seen in the DepthPro [7] and DepthAnything series [89, 90]. As discussed in DepthAnything V2 [90], existing benchmarks with ground truth collected in real-world environments often suffer from noise, limited diversity, and low resolution. Moreover, ground truths acquired from sensors such as LiDAR [24] or depth cameras [54] are typically sparse and lack fine-grained object details, thereby limiting the effectiveness of evaluating foundation monocular depth estimation (MDE) models.

Compared to these benchmarks, DepthAnything-AC demonstrates more pronounced improvements on the more

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Image Featuree e Image-perturbationa e ert r at DepthAnythingth n th n V2 Ours

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

- Figure 5 Visualization of feature representations. ‘Feature’ refers to the representation extracted from the original image, while the two columns on the right show the features generated from the perturbed images. Compared to Depth Anything V2 [90], our DepthAnything-AC demonstrates the ability to recover feature representations from perturbed images.

NuScenes-night Robotcar-night AbsRel↓ δ1↑ AbsRel↓ δ1↑

Method

KD 0.201 0.721 0.234 0.535 KD+Con. 0.200 0.724 0.233 0.538 Con. 0.198 0.727 0.227 0.555

NuScenes-night Robotcar-night AbsRel↓ δ1↑ AbsRel↓ δ1↑

Loss Function

Midas [58] 0.206 0.716 0.238 0.529 DepthAnything [89] 0.202 0.721 0.231 0.544 Ours 0.198 0.727 0.227 0.555

(a) Discussion of training strategy.

(b) Discussion of affine-invariant loss.

- Table 7 Ablation studies of DepthAnything-AC. ‘KD’ denotes the knowledge distillation paradigm. ‘Con.’ denotes the consistency regularization paradigm.

modern DA-2K-based benchmark, which provides precise depth relationships, high-resolution imagery, and extensive scene coverage. This highlights the advantage of evaluating on benchmarks with higher-quality ground truths and greater scene diversity.

### 4.3 Ablation Study

Effectiveness of components. We first conduct ablation studies on different components of our DepthAnythingAC to demonstrate their effectiveness in Tab. 6. It can be seen that on the DA-2K dark benchmark, both the perturbation module P and the spatial distance constraint Ls lead to improvements over the consistency regularization baseline. Notably, for the setting without perturbation, we adopt the strong augmentation strategies that are commonly used in semi-supervised learning paradigms [69, 91].

Different training paradigms. To demonstrate the effectiveness of the proposed perturbation-based consistency regularization framework, we compare different training paradigms in Tab. 7a. The experimental results show that our consistency-based paradigm outperforms the knowledge distillation approach. This suggests that the non-perturbed branch of the student model possesses sufficient generalization capability to provide effective supervision and can generate higher-quality pseudo labels compared to a frozen teacher model. Furthermore, it supports the ability of DepthAnything-AC to maintain competitive performance on general benchmarks, as evidenced in Tab. 5.

Different affine-invariant loss. As demonstrated in the Tab. 7b, a comparative analysis is conducted to ascertain the discrepancy in performance between the affine-invariant loss proposed by Midas [58], DepthAnything [89, 90], and DepthAnything-AC. The experimental findings demonstrate that the selected loss function is the most appropriate for the given conditions.

### 4.4 Qualitative analysis

In Fig. 5, we illustrate the impact of image perturbations on the feature representations of Depth Anything V2 [90] along with the restoration achieved by DepthAnything-AC. It is evident that perturbations significantly degrade the feature quality, which may explain the poor ability of the existing foundation MDE models to perceive semantic boundaries and fine details from corrupted images. Meanwhile, DepthAnything-AC expands the capability of foundation MDE models with the consistency-based framework and enhances the model’s awareness of spatial relationships throughout spatial distance constraint, thereby enabling it to recover finegrained features from corrupted inputs.

## 5 Conclusions

In this paper, we propose DepthAnything-AC, a foundation monocular depth estimation (MDE) model that extends the capabilities of existing models in complex scenarios, particularly under challenging lighting, diverse weather conditions, and distortions caused by devices, while preserving better object boundaries and details. We introduce an unsupervised perturbation-based consistency framework that enables the model to learn to handle complex scenes from a small amount of unlabeled general data. To address the inability of existing models to recover details from corrupted images, we incorporate a spatial distance constraint to enforce the geometric relative relationships between patches. Experimental results demonstrate that DepthAnything-AC provides robust and fine-grained depth predictions, achieving superior performance on various benchmarks, especially on the more modern DA-2K-based benchmark, while maintaining general scene capabilities.

## References

- [1] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4009–4018, 2021. 3
- [2] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 2, 3
- [3] Amlaan Bhoi. Monocular depth estimation: A survey. arXiv preprint arXiv:1901.09402, 2019. 3
- [4] Jiawang Bian, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, and Ian Reid. Unsupervised scale-consistent depth and ego-motion learning from monocular video. In Advances in Neural Information Processing Systems (NeurIPS), 2019. 3
- [5] Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jin Chin, Chunhua Shen, and Ian Reid. Auto-rectify network for unsupervised indoor depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. 1
- [6] Reiner Birkl, Diana Wofk, and Matthias Müller. Midas v3.1 – a model zoo for robust monocular relative depth estimation. arXiv preprint arXiv:2307.14460, 2023. 3
- [7] Aleksei Bochkovskii, AmaÃG, l Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024. 1, 2, 6, 7, 8, 20, 23
- [8] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 611–625. Springer, 2012. 7, 17, 18
- [9] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2, 2020. 1, 6
- [10] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631,

2020. 1, 7, 17

- [11] Wenxiao Cai, Yaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. arXiv preprint arXiv:2406.13642,

2024. 1

- [12] Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. Deep monocular depth estimation leveraging a large-scale outdoor stereo dataset. Expert Systems with Applications, 178:114877, 2021. 6
- [13] Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes. arXiv preprint arXiv:2110.11590, 2021. 1, 6
- [14] Junyuan Deng, Wei Yin, Xiaoyang Guo, Qian Zhang, Xiaotao Hu, Weiqiang Ren, Xiaoxiao Long, and Ping Tan. Boost 3d reconstruction using diffusion-based monocular camera calibration. arXiv preprint arXiv:2411.17240, 2024. 1
- [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 7
- [16] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014. 1, 3
- [17] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010. 6
- [18] Junkai Fan, Kun Wang, Zhiqiang Yan, Xiang Chen, Shangbing Gao, Jun Li, and Jian Yang. Depth-centric dehazing and depth-estimation from real-world hazy driving video. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2852–2860, 2025. 3
- [19] Alain Fournier, Don Fussell, and Loren Carpenter. Computer rendering of stochastic models. Communications of the ACM, 25(6):371–384, 1982. 18
- [20] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2002–2011, 2018. 1, 3
- [21] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In ECCV, 2024. 3
- [22] Adrien Gaidon, Qiao Wang, Yohann Cabon, and Eleonora Vig. Virtual worlds as proxy for multi-object tracking analysis. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 4340–4349, 2016. 6
- [23] Stefano Gasperini, Patrick Koch, Vinzenz Dallabetta, Nassir Navab, Benjamin Busam, and Federico Tombari. R4dyn: Exploring radar for self-supervised monocular depth estimation of dynamic scenes. In 2021 International Conference on 3D Vision (3DV), pages 751–760. IEEE, 2021. 3
- [24] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013. 1, 3, 7, 8, 17
- [25] Clément Godard, Oisin Mac Aodha, and Gabriel J Brostow. Unsupervised monocular depth estimation with left-right consistency. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 270–279, 2017. 1
- [26] Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3828–3838, 2019. 1
- [27] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139– 144, 2020. 3

- [28] Ming Gui, Johannes Schusterbauer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Björn Ommer. Depthfm: Fast monocular depth estimation with flow matching, 2024. 3
- [29] Haoyu Guo, He Zhu, Sida Peng, Haotong Lin, Yunzhi Yan, Tao Xie, Wenguan Wang, Xiaowei Zhou, and Hujun Bao. Multi-view reconstruction via sfm-guided monocular depth estimation. In CVPR, 2025. 3
- [30] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Liu, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024. 3
- [31] Xiankang He, Dongyan Guo, Hongji Li, Ruibo Li, Ying Cui, and Chi Zhang. Distill any depth: Distillation creates a stronger monocular depth estimator. arXiv preprint arXiv: 2502.19204, 2025. 1
- [32] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zeroshot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 3
- [33] Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. Drivingworld: Constructingworld model for autonomous driving via video gpt. arXiv preprint arXiv:2412.19505, 2024. 1
- [34] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3
- [35] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 1
- [36] Sunok Kim, Dongbo Min, Bumsub Ham, Seungryong Kim, and Kwanghoon Sohn. Deep stereo confidence prediction for depth estimation. In 2017 ieee international conference on image processing (icip), pages 992–996. IEEE, 2017. 6
- [37] Youngjung Kim, Bumsub Ham, Changjae Oh, and Kwanghoon Sohn. Structure selective depth superresolution for rgb-d cameras. IEEE Transactions on Image Processing, 25(11):5227–5238, 2016. 6
- [38] Youngjung Kim, Hyungjoo Jung, Dongbo Min, and Kwanghoon Sohn. Deep monocular depth estimation via integration of global and local predictions. IEEE transactions on Image Processing, 27(8):4131–4144,

2018. 6

- [39] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 6
- [40] Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit R. Cottereau, and Wei Tsang Ooi. Robodepth: Robust out-of-distribution depth estimation under corruptions. In Advances in Neural Information Processing Systems, 2023. 1, 3, 7, 8, 17
- [41] Jinming Li, Wanying Wang, Yaxin Peng, Chaomin Shen, Yichen Zhu, and Zhiyuan Xu. Visual robotic manipulation with depth-aware pretraining. In 2024 IEEE International Conference on Robotics and Biomimetics (ROBIO), pages 843–850. IEEE, 2024. 1
- [42] Xiaoqi Li, Mingxu Zhang, Yiran Geng, Haoran Geng, Yuxing Long, Yan Shen, Renrui Zhang, Jiaming Liu, and Hao Dong. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18061–18070, 2024. 1
- [43] Zhenyu Li, Shariq Farooq Bhat, and Peter Wonka. Patchfusion: An end-to-end tile-based framework for high-resolution monocular metric depth estimation. 2024. 1
- [44] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In Computer Vision and Pattern Recognition (CVPR), 2018. 6

- [45] Zhong-Yu Li, Xin Jin, Boyuan Sun, Chun-Le Guo, and Ming-Ming Cheng. Towards raw object detection in diverse conditions. arXiv preprint arXiv:2411.15678, 2024. 6, 7
- [46] Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, and Bingyi Kang. Prompting depth anything for 4k resolution accurate metric depth estimation. 2024. 1
- [47] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 6
- [48] Beyang Liu, Stephen Gould, and Daphne Koller. Single image depth estimation from predicted semantic labels. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 1253–1260. IEEE, 2010. 3
- [49] Lina Liu, Xibin Song, Mengmeng Wang, Yong Liu, and Liangjun Zhang. Self-supervised monocular depth estimation for all day images using domain separation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12737–12746, 2021. 3
- [50] Zhiheng Liu, Ka Leong Cheng, Qiuyu Wang, Shuzhe Wang, Hao Ouyang, Bin Tan, Kai Zhu, Yujun Shen, Qifeng Chen, and Ping Luo. Depthlab: From partial to complete. arXiv preprint arXiv:2412.18153, 2024. 3
- [51] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [52] Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman. 1 year, 1000 km: The oxford robotcar dataset. The International Journal of Robotics Research, 36(1):3–15, 2017. 1, 7, 8, 17
- [53] Yifan Mao, Jian Liu, and Xianming Liu. Stealing stable diffusion prior for robust monocular depth estimation. arXiv preprint arXiv:2403.05056, 2024. 3
- [54] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 7, 8, 17, 18
- [55] Xincheng Pang, Wenke Xia, Zhigang Wang, Bin Zhao, Di Hu, Dong Wang, and Xuelong Li. Depth helps: Improving pre-trained rgb-based policy with depth information injection, 2024. 1
- [56] Duc-Hai Pham, Tung Do, Phong Nguyen, Binh-Son Hua, Khoi Nguyen, and Rang Nguyen. Sharpdepth: Sharpening metric depth predictions using diffusion distillation. arXiv preprint arXiv:2411.18229, 2024. 3
- [57] Uchitha Rajapaksha, Ferdous Sohel, Hamid Laga, Dean Diepeveen, and Mohammed Bennamoun. Deep learning-based depth estimation methods from monocular image and videos: A comprehensive survey. ACM Computing Surveys, 56(12):1–51, 2024. 3
- [58] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on

- Pattern Analysis and Machine Intelligence, 44(3), 2022. 2, 7, 9

[59] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on

- Pattern Analysis and Machine Intelligence, 44(3), 2022. 3

- [60] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 3
- [61] Kieran Saunders, George Vogiatzis, and Luis J Manso. Self-supervised monocular depth estimation: Let’s talk about the weather. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8907–8917, 2023. 2, 3, 6, 7, 8
- [62] Ashutosh Saxena, Sung Chung, and Andrew Ng. Learning depth from single monocular images. Advances in neural information processing systems, 18, 2005. 3

- [63] Saurabh Saxena, Abhishek Kar, Mohammad Norouzi, and David J Fleet. Monocular depth estimation using diffusion models. arXiv preprint arXiv:2302.14816, 2023. 3
- [64] Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multicamera videos. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 7, 17, 18
- [65] Jaidev Shriram, Alex Trevithick, Lingjie Liu, and Ravi Ramamoorthi. Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion. 2025. 1
- [66] Ziyang Song, Zerong Wang, Bo Li, Hao Zhang, Ruijie Zhu, Li Liu, Peng-Tao Jiang, and Tianzhu Zhang. Depthmaster: Taming diffusion models for monocular depth estimation. arXiv preprint arXiv:2501.02576,

2025. 1

- [67] Ziyang Song, Ruijie Zhu, Chuxin Wang, Jiacheng Deng, Jianfeng He, and Tianzhu Zhang. Ec-depth: Exploring the consistency of self-supervised monocular depth estimation under challenging scenes. arXiv preprint arXiv:2310.08044, 2023. 6, 7, 8, 17
- [68] Jaime Spencer, Richard Bowden, and Simon Hadfield. Defeat-net: General monocular depth via simultaneous unsupervised representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020. 3
- [69] Boyuan Sun, Yuqi Yang, Le Zhang, Ming-Ming Cheng, and Qibin Hou. Corrmatch: Label propagation via correlation matching for semi-supervised semantic segmentation. IEEE Computer Vision and Pattern Recognition (CVPR), 2024. 3, 9
- [70] Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llava-scissor: Token compression with semantic connected components for video llms. arXiv preprint arXiv:2506.21862, 2025. 1
- [71] Libo Sun, Jia-Wang Bian, Huangying Zhan, Wei Yin, Ian Reid, and Chunhua Shen. Sc-depthv3: Robust self-supervised monocular depth estimation for dynamic scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023. 1
- [72] Antonio Torralba and Aude Oliva. Depth estimation from image structure. IEEE Transactions on pattern analysis and machine intelligence, 24(9):1226–1238, 2002. 3
- [73] Fabio Tosi, Pierluigi Zama Ramirez, and Matteo Poggi. Diffusion models for monocular depth estimation: Overcoming challenging conditions. In European Conference on Computer Vision (ECCV), 2024. 3
- [74] Madhu Vankadari, Sourav Garg, Anima Majumder, Swagat Kumar, and Ardhendu Behera. Unsupervised monocular depth estimation for night-time images using adversarial domain feature adaptation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16, pages 443–459. Springer, 2020. 3
- [75] Madhu Vankadari, Stuart Golodetz, Sourav Garg, Sangyun Shin, Andrew Markham, and Niki Trigoni. When the sun goes down: Repairing photometric losses for all-day depth estimation. arXiv preprint arXiv:2206.13850, 2022. 3
- [76] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, and Gregory Shakhnarovich. DIODE: A Dense Indoor and Outdoor DEpth Dataset. CoRR, abs/1908.00463, 2019. 7, 17, 18
- [77] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 1
- [78] Jiyuan Wang, Chunyu Lin, Lang Nie, Shujun Huang, Yao Zhao, Xing Pan, and Rui Ai. Weatherdepth: Curriculum contrastive learning for self-supervised depth estimation under adverse weather conditions. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4976–4982. IEEE, 2024. 3, 6, 7, 8
- [79] Jiyuan Wang, Chunyu Lin, Lang Nie, Kang Liao, Shuwei Shao, and Yao Zhao. Digging into contrastive learning for robust depth estimation with diffusion models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4129–4137, 2024. 3

- [80] Kun Wang, Zhenyu Zhang, Zhiqiang Yan, Xiang Li, Baobei Xu, Jun Li, and Jian Yang. Regularizing nighttime weirdness: Efficient self-supervised monocular depth estimation in the dark. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16055–16064, 2021. 3
- [81] Li Wang, Liang Du, Xiaoqing Ye, Yanwei Fu, Guodong Guo, Xiangyang Xue, Jianfeng Feng, and Li Zhang. Depth-conditioned dynamic message propagation for monocular 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 454–463, 2021. 1
- [82] Yiran Wang, Jiaqi Li, Chaoyi Hong, Ruibo Li, Liusheng Sun, Xiao Song, Zhe Wang, Zhiguo Cao, and Guosheng Lin. Tacodepth: Towards efficient radar-camera depth estimation with one-stage fusion. In CVPR, 2025. 3
- [83] Andreas Wedel, Uwe Franke, Jens Klappstein, Thomas Brox, and Daniel Cremers. Realtime depth estimation and obstacle detection from monocular video. In Joint Pattern Recognition Symposium, pages 475–484. Springer, 2006. 3
- [84] Cho-Ying Wu, Jialiang Wang, Michael Hall, Ulrich Neumann, and Shuochen Su. Toward practical monocular indoor depth estimation. In CVPR, 2022. 3
- [85] Ke Xian, Jianming Zhang, Oliver Wang, Long Mai, Zhe Lin, and Zhiguo Cao. Structure-guided ranking loss for single image depth prediction. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 6
- [86] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In CVPR, 2025. 3
- [87] Weilong Yan, Ming Li, Haipeng Li, Shuwei Shao, and Robby T Tan. Synthetic-to-real self-supervised robust depth estimation via learning with motion and structure priors. arXiv preprint arXiv:2503.20211,

2025. 3, 7

- [88] Guorun Yang, Xiao Song, Chaoqin Huang, Zhidong Deng, Jianping Shi, and Bolei Zhou. Drivingstereo: A large-scale dataset for stereo matching in autonomous driving scenarios. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 7, 17
- [89] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 1, 2, 3, 5, 6, 7, 8, 9, 18, 20, 22
- [90] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024. 1, 2, 3, 6, 7, 8, 9, 10, 17, 20, 21
- [91] Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, and Yinghuan Shi. Revisiting weak-to-strong consistency in semi-supervised semantic segmentation. In CVPR, 2023. 3, 9
- [92] Bowen Yin, Xuying Zhang, Zhong-Yu Li, Li Liu, Ming-Ming Cheng, and Qibin Hou. Dformer: Rethinking rgbd representation learning for semantic segmentation. In ICLR, 2024. 1
- [93] Bo-Wen Yin, Jiao-Long Cao, Ming-Ming Cheng, and Qibin Hou. Dformerv2: Geometry self-attention for rgbd semantic segmentation. In CVPR, 2025. 1
- [94] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9043–9053, 2023. 3
- [95] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Simon Chen, Yifan Liu, and Chunhua Shen. Towards accurate reconstruction of 3d scene shape from a single monocular image. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022. 1
- [96] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (CVPR), 2021. 1
- [97] Ze-Xin Yin, Peng-Yi Jiao, Jiaxiong Qiu, Ming-Ming Cheng, and Bo Ren. Ms-nerf: Multi-space neural radiance fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–18, 2025. 1

- [98] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 1
- [99] Sen Zhang, Jing Zhang, and Dacheng Tao. Towards scale-aware, robust, and generalizable unsupervised monocular depth estimation by integrating imu motion dynamics. In European Conference on Computer Vision, pages 143–160. Springer, 2022. 3, 6, 7, 8
- [100] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe, and Jian Yang. Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4106–4115, 2019. 1
- [101] Chaoqiang Zhao, Qiyu Sun, Chongzhen Zhang, Yang Tang, and Feng Qian. Monocular depth estimation based on deep learning: An overview. Science China Technological Sciences, 63(9):1612–1627, 2020. 3
- [102] Chaoqiang Zhao, Yang Tang, and Qiyu Sun. Unsupervised monocular depth estimation in highly complex environments. IEEE Transactions on Emerging Topics in Computational Intelligence, 6(5):1237–1246,

2022. 3

- [103] Jiaxing Zhao, Boyuan Sun, Xiang Chen, Xihan Wei, and Qibin Hou. Llava-octopus: Unlocking instructiondriven adaptive projector fusion for video understanding. arXiv preprint arXiv:2501.05067, 2025. 1
- [104] Yupeng Zheng, Chengliang Zhong, Pengfei Li, Huan-ang Gao, Yuhang Zheng, Bu Jin, Ling Wang, Hao Zhao, Guyue Zhou, Qichao Zhang, et al. Steps: Joint self-supervised nighttime image enhancement and depth estimation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 4916–4923. IEEE, 2023. 3, 6, 7, 8, 17
- [105] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017. 6
- [106] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127:302–321, 2019. 6
- [107] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Computer Vision (ICCV), 2017 IEEE International Conference on, 2017. 3

## Appendix

In order to provide a more complete illustration of the capabilities of DepthAnything-AC, a comprehensive appendix has been developed. This appendix includes detailed descriptions of the evaluation benchmarks, additional experiments analysis, a variety of visualizations, and the discussion among limitations and broader impacts.

## A Detailed Evaluation Benchmarks

### A.1 Multi-condition DA-2K benchmark

DA-2K [90] is a high-resolution depth estimation dataset featuring extensive and diverse scenes. Proposed by DepthAnything V2 [90], it evaluates a model’s capability by determining which of two given points lies closer to the camera plane. Beyond this modern benchmark, we further apply perturbations to the original images to simulate four common challenging conditions: dark, fog, snow, and blur. This results in an enhanced Multi-condition DA-2K benchmark, designed to evaluate the model’s robustness and effectiveness under complex visual scenarios.

### A.2 Real-world complex benchmarks

To evaluate the model’s capability in real-world complex environments, we adopt several widely used benchmarks in the robust depth estimation field [104, 67], including NuScenes-Night, RobotCar-Night, and DrivingStereo. These datasets present challenging conditions such as low illumination, motion blur, and diverse weather variations, thereby providing a comprehensive assessment of model robustness.

NuScenes-night [10] is a nighttime subset of the NuScenes dataset, which serves as a significant benchmark in the domain of autonomous driving. Following the split used in STEPS[104], we evaluate various methods on 500 real-world nighttime images, offering a realistic testbed for assessing depth estimation performance under low-light conditions.

Robotcar-night [52] is a nighttime subset of the RobotCar benchmark, which contains both daytime and nighttime data collected in autonomous driving scenarios. Following the splits of STEPS [104], we use 186 real-world nighttime images for evaluation.

DrivingStereo [88] is a large-scale dataset comprising 180K images collected from real-world driving scenes under diverse and challenging weather conditions. For evaluation, we follow the official subset partition and select the rain, fog, and cloud subsets, each consisting of 500 images. These subsets are used to assess the model’s performance under varying climatic scenarios, including adverse weather and reduced visibility.

### A.3 Synthetic complex benchmark KITTI-C

KITTI-C [40] is an evaluation benchmark designed to assess the robustness of depth estimation models. It augments the original KITTI dataset [24] with synthetic corruptions to simulate a variety of challenging scenarios, including adverse weather conditions and external disturbances such as motion blur and LiDAR beam dropout. For our evaluation, we select four representative subsets: darkness, snow, motion blur, and Gaussian noise, which cover common sources of degradation in visual perception systems.

### A.4 General benchmarks

To verify that DepthAnything-AC maintains its performance on general scenes after being enhanced for robustness in complex scenarios, we follow the evaluation protocol of Depth Anything V2 and select KITTI [24], NYUD [54], Sintel [8], DIODE [76], and ETH3D [64] as representative general benchmarks.

KITTI. The KITTI [24] dataset is the go-to benchmark for automated driving. Created in 2012 by the Karlsruhe Institute of Technology (KIT) and Toyota Technical Institute-Central America (TTI-C), it is used to evaluate computer vision algorithms like stereo vision, optical flow, depth estimation, and 3D target detection and tracking in real road scenarios. The dataset, collected from various sensors (including grayscale/color cameras, 64-line

Changes in light-Apply all

Brightness is decreased nonlinearly, with brighter regions decreasing more and darker regions less. Poisson noise is added to model random photon noise due to reduced photon counts in low-light environments, and Gaussian noise is added to model the read noise of the camera sensor in low-light conditions.

Dark

Blur , Weather and Color - Apply with certain probabilities

Randomly select a direction angle θ ∈ [−45◦, 45◦], set the blur kernel radius and Gaussian parameter σ based on severity, and construct a directional kernel via 1D Gaussian sampling with small displacements along the motion direction weighted by Gaussian coefficients.

Motion Blur

Generate scaling layers with incremental scaling factors using cropping and scaling techniques, and synthesize the blurring effect by averaging the equal weights.

Zoom Blur

Using the Diamond-Square algorithm [19] to generate a 2D self-similar noise field, where a severity parameter controls fog density and texture. Higher severity is associated with higher noise amplitude and reduced detail retention.

Fog

A Gaussian random field with parameters (µ, σ) is generated as the snow distribution, then snowflake size and density are adjusted via nonlinear transformation. Directional motion blur (θ ∈ [−135◦, −45◦]) simulates falling snow.

Snow

Contrast Using a centered scaling transform with coefficients by [0.05, 0.4].

- Table 8 Specific implementation of different perturbation methods

LIDAR, and GPS/IMU systems), contains 200,000+ 3D annotated images and 39.2 km of visual range sequences, along with annotations on occlusion and truncation of vehicles, pedestrians, bicycles, and more.

NYU-D. The NYUDepth V2 [54] dataset is an authoritative indoor scene understanding dataset released by New York University (NYU) in 2012. RGB-D video sequences containing 464 diverse indoor scenes were captured through the Microsoft Kinect device, providing 1,449 pairs of densely labeled RGB vs. depth image pairs, with each image labeled with object class, instance number, and depth information.

Sintel. The Sintel [8] dataset is a high-quality optical flow estimation benchmark dataset based on the opensource animated movie "Sintel" constructed by the Max-Planck-Institut (MPI) in 2012. The database contains 1,064 combinations of stereoscopic images and densely labeled ground-truth optical flow fields, covering complex motion patterns (e.g., occlusion, motion blur, and atmospheric effects) for 23 different scenes. The dataset is divided into a training set (40 sequences) and a test set (23 sequences), and is available in two versions, "Sintel Final" (with motion blur and lighting changes) and "Sintel Clean" (without effects), and we use "Sintel Final" for evaluation.

ETH3D. The ETH3D [64] dataset is a multimodal benchmark dataset introduced by ETH Zurich, designed for 3D reconstruction, stereo vision and evaluation of multi-view stereo matching algorithms. The dataset collects high-resolution RGB images and depth information of indoor and outdoor scenes through high-precision laser scanners and various cameras (including DSLR cameras and multi-view synchronized cameras), and provides sparse parallax annotations and dense point cloud ground truths, covering a subset of high-resolution multiviews, low-resolution multiviews, and binocular configurations.

DIODE. The DIODE [76] dataset is a high-precision RGB-D dataset covering both indoor and outdoor scenes. It is acquired by a unified sensor suite (e.g., Faro scanners), providing high-resolution, dense and wide range of depth measurements and plane normal annotations. The dataset contains 8,574 indoor and 16,884 outdoor training samples, as well as 325 indoor and 446 outdoor validation samples, supporting tasks such as monocular depth estimation and 3D reconstruction. Following the Depth Anything [89] , We use the indoor samples for validation.

###### NuScenes-night Robotcar-night Gaussian

Dark Weather Blur Contrast

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

✓ 0.199 0.728 0.239 0.517 0.159 0.785 ✓ 0.199 0.728 0.237 0.524 0.158 0.791 ✓ ✓ 0.200 0.727 0.233 0.540 0.154 0.792 ✓ ✓ ✓ 0.199 0.728 0.231 0.543 0.158 0.791 ✓ ✓ ✓ ✓ 0.198 0.727 0.227 0.555 0.153 0.793

- Table 9 Impact of each perturbation type.

Perturbation

NuScenes-night Robotcar-night AbsRel↓ δ1↑ AbsRel↓ δ1↑

0.7+0.8 0.202 0.718 0.232 0.544 0.5+0.6 0.202 0.721 0.231 0.545 0.3+0.4 0.201 0.725 0.235 0.533 0.1+0.2 0.198 0.727 0.227 0.555

(a) Discussion on the probability of occurrence of blur and weather perturbation.

Method

NuScenes-night Robotcar-night AbsRel↓ δ1↑ AbsRel↓ δ1↑

Con. (M) 0.198 0.728 0.237 0.523 Con. (E) 0.199 0.728 0.239 0.518 KD (M) 0.198 0.729 0.237 0.524 KD (E) 0.198 0.727 0.227 0.555

(b) Discussion of different training paradigms of spatial distance relations.

- Table 10 More ablation experiments. In (a), the left side of the figure signifies the probability of blur occurrence when each image is subjected to perturbation, whilst the right side of the figure denotes the probability of weather occurrence. In (b), ‘KD’ denotes the knowledge distillation paradigm. ‘Con.’ denotes the consistency regularization paradigm. ‘M’ refers to Manhattan distance while ‘E’ refers to Euclidean distance.

## B More Experimental Analysis

### B.1 Perturbations

Perturbation types. To assess the impact of each perturbation type on model performance, we conduct ablation experiments on DepthAnything-AC in Tab. 9. Results on the diverse Robotcar-Night benchmark reveal that each type of perturbation contributes positively to the overall performance, underscoring the necessity of incorporating each perturbation strategy.

Probability of different perturbations. In the Tab. 10a, we present the probabilities of applying different perturbations to unlabeled images. Specifically, illumination perturbation is applied to all images, while the Tab. 10a reports the probabilities used for blur and weather perturbations. Note that contrast perturbation is grouped under weather-related perturbations. As shown, the model achieves the best performance when the probability of blur augmentation is set to 0.1 and that of weather augmentation to 0.2.

### B.2 Training scheme of Spatial Distance Relation

As demonstrated in the Tab. 10b, we explore the impact of the spatial distance relation on the outcomes. In this analysis, M and E represent the utilization of Manhattan or Euclidean distance in the Eqn. 6 and Eqn. 7, respectively. The findings indicate that the spatial distance relation is generated using the Euclidean distance and optimized using the knowledge distillation paradigm.

### B.3 Encoder Frozen vs Unfrozen

As shown in Tab. 11, we investigate the impact of whether to freeze the encoder parameters during training. The results clearly indicate that including the encoder parameters in the training process leads to a decline in performance during evaluation. This suggests that freezing the encoder helps preserve its pre-trained feature representations, which proves more effective for consistent and robust performance, especially in the presence of perturbations.

###### NuScenes-night Robotcar-night DS-cloud Snow Gaussian

Method

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

Unfrozen 0.218 0.691 0.248 0.494 0.151 0.798 0.114 0.872 0.155 0.789 Frozen 0.198 0.727 0.227 0.555 0.149 0.801 0.114 0.873 0.153 0.793

- Table 11 The importance of maintaining the frozen state of the encoder.

λ1 λ2 λ3

NuScenes-night Robotcar-night Gaussian

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

- 0.5 0.3 0.2 0.202 0.720 0.227 0.559 0.156 0.790

- 0.2 0.5 0.3 0.200 0.727 0.233 0.540 0.153 0.792
- 0.3 0.2 0.5 0.202 0.721 0.233 0.539 0.155 0.788

- 1.0 1.0 1.0 0.200 0.726 0.230 0.547 0.155 0.791 1/3 1/3 1/3 0.200 0.727 0.227 0.555 0.153 0.793

- Table 12 Discussion about different combinations of loss weights λ.

### B.4 Different loss weight

In Tab. 12, we conduct more ablation experiments on different loss weights λ. The results demonstrate that our method is relatively insensitive to variations in the loss weight configuration. Overall, setting [λ1, λ2, λ3] to [1/3, 1/3, 1/3] yields the best performance across benchmarks.

## C More Visualization

- C.1 Qualitative comparison between Depth Anything V2 and DepthAnything-AC

- Please refer to Fig. 6. We provide more visualization of our DepthAnything-AC with comparison of Depth Anything V2 [90].

C.2 Qualitative comparison between Depth Anything V1 and DepthAnything-AC

- Please refer to Fig. 7. We provide more visualization of our DepthAnything-AC with comparison of Depth Anything V1 [89].

C.3 Qualitative comparison between DepthPro and DepthAnything-AC

- Please refer to Fig. 8. We provide more visualization of our DepthAnything-AC with comparison of DepthPro [7].

Image DepthAnything V2 Ours

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

- Figure 6 Comparison between DepthAnything-V2 [90] and our DepthAnything-AC

- Image DepthAnything V1 Ours
- Figure 7 Comparison between DepthAnything-V1 [89] and our DepthAnything-AC

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

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

- Image DepthPro Ours
- Figure 8 Comparison between DepthPro [7] and our DepthAnything-AC

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

