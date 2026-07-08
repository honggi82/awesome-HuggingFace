## How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach

Chirui Chang1 Jiahui Liu1 Zhengzhe Liu3 Xiaoyang Lyu1 Yi-Hua Huang1 Xin Tao2 Pengfei Wan2 Di Zhang2 Xiaojuan Qi1*

[Figure 1]

1The University of Hong Kong 2Kling Team, Kuaishou Technology 3Lingnan University

[Figure 2]

[Figure 3]

# arXiv:2406.19568v2[cs.CV]5Oct2025

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]|
|---|

[Figure 19]

|beh|ind| |
|---|---|---|
| | | |

###### penetrate ahead

[Figure 20]

[Figure 21]

Watch

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

| |
|---|

| |
|---|

|[Figure 38]|
|---|

| |
|---|

Human Check

| | | | | |
|---|---|---|---|---|
| |change| | | |

[Figure 39]

[Figure 40]

| |
|---|

[Figure 41]

[Figure 42]

[Figure 43]

| |
|---|

| |
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Runway Hailuo AI

Input

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Appearance Score: 0.26

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

GradCam

[Figure 64]

| |
|---|

[Figure 65]

| |
|---|

[Figure 66]

Geometry Appearance

[Figure 67]

KLING AI

[Figure 68]

GradCam

| |
|---|

| |
|---|

[Figure 69]

Motion

[Figure 70]

OpenAI SORA

[Figure 71]

[Figure 72]

| |
|---|

Geometry Score: 0.01

| |
|---|

Motion Score: 0.22

[Figure 73]

L3DE

GradCam

Generative Video Dataset

Figure 1. L3DE evaluates videos from any generative model based on 3D visual coherence, assessing appearance, motion, and geometry. Its scores align closely with human perception and can localize regions of 3D simulation failures, similar to human intuition. Examples highlight key failure cases: (1) incorrect occlusion between the basketball and hoop, disrupting geometric consistency, (2) abrupt texture transition in plant leaves, and (3) unnatural relative motion between the golf ball and the golf club, violating real-world motion dynamics.

#### Abstract

tensive experiments, demonstrating strong alignment with 3D reconstruction quality and human judgments. Our evaluations on leading generative models (e.g., Kling, Sora, and MiniMax) reveal persistent simulation gaps and subtle inconsistencies. Beyond generative video assessment, L3DE extends to broader applications: benchmarking video generation models, serving as a deepfake detector, and enhancing video synthesis by inpainting flagged inconsistencies.

Recent advancements in video diffusion models enable the generation of photorealistic videos with impressive 3D consistency and temporal coherence. However, the extent to which these AI-generated videos simulate the 3D visual world remains underexplored. In this paper, we introduce Learned 3D Evaluation (L3DE), an objective, quantifiable, and interpretable method for assessing AI-generated videos’ ability to simulate the real world in terms of 3D visual qualities and consistencies, without requiring manually labeled defects or quality annotations. Instead of relying on 3D reconstruction, which is prone to failure with in-the-wild videos, L3DE employs a 3D convolutional network, trained on monocular 3D cues of motion, depth, and appearance, to distinguish real from synthetic videos. Confidence scores from L3DE quantify the gap between real and synthetic videos in terms of 3D visual coherence, while a gradient-based visualization pinpoints unrealistic regions, improving interpretability. We validate L3DE through ex-

#### 1. Introduction

Video diffusion models, such as Sora [5], have recently shown remarkable capabilities in visual simulation, producing photorealistic videos with 3D consistency and temporal coherence that can even deceive human observers. This progress raises a fundamental question: how well do AIgenerated videos simulate the 3D visual world? While existing evaluations heavily rely on subjective user studies, a quantifiable and interpretable approach remains missing for assessing 3D visual coherence of generative videos. 3D scene reconstruction [18, 21, 34, 69, 71] is a natural

*Corresponding author.

way to assess whether generative videos preserve 3D visual coherence. The intuition is that if a video enables highquality 3D reconstruction, it should maintain 3D-consistent appearance, structure, and motion across frames. However, even state-of-the-art reconstruction methods struggle with in-the-wild videos due to challenges such as unreliable pose estimation [10, 66, 71] and the absence of multiview cues [10, 25, 64, 76], making large-scale evaluation based on reconstruction impractical. To overcome these limitations, inspired by [11], we turn to monocular 3D cues, such as depth and optical flow, which naturally emerge from videos and serve as strong proxies for 3D structure and motion. Thus, we explore leveraging monocular cues from foundation models [42, 62] as an alternative for assessing 3D realism. Specifically, we use RAFT [62] for optical flow estimation and UniDepth [42] for depth prediction, while utilizing DINOv2 [38] to capture high-level appearance features.

We collect real and synthetic videos from Pexels [41] and Stable Video Diffusion (SVD) [3], respectively. Pexels provides diverse real-world videos, while SVD is one of the most accessible video generator. We align their visual content by using real video frames as prompts to generate paired synthetic videos. This minimizes content disparities, isolating differences in 3D consistency and helping analyze how generative videos deviate from real ones.

Equipped with 3D proxies and data, the next challenge is measuring the gap between generative and real-world videos. To tackle this, we develop Learned 3D Evaluation (L3DE), a data-driven learning-based tool that uses monocular 3D cues to evaluate generative videos and identify 3D visual simulation failures. L3DE captures intrinsic differences between real and synthetic videos by training a 3D convolutional network with contrastive learning using 3D proxies as inputs. The confidence scores quantify the gap between synthetic and real videos regarding these 3D proxies. Additionally, L3DE enhances interpretability by highlighting key failure regions via a gradient-based method [54] (see Table 3). Finally, by integrating depth, motion, and appearance proxies through a feature fusion module, L3DE provides a stable and comprehensive evaluation of 3D visual coherence in generative videos.

To validate L3DE’s effectiveness, we conduct 3D scene reconstruction experiments and user studies. Our results in Sec. 5.1 show that L3DE scores highly correlate with reconstruction quality, with flagged areas aligning with regions of high 3D inconsistency, as confirmed by reconstruction errors. Human studies in Sec. 5.2 further reveal that L3DE scores align closely with human perceptual judgments, with flagged areas consistently rated high by annotators. These results demonstrate L3DE’s effectiveness in assessing and analyzing 3D visual coherence in generative videos. We conduct experiments applying L3DE to videos from leading

generative models, including Sora [5], MiniMax [35], Kling [24], and others to benchmark their 3D visual simulation capabilities and analyze their strengths and limitations. With L3DE validated through 3D reconstruction and human evaluation, these results provide insights into how well different models capture 3D realism. As shown in Table 5, models like Sora and Kling achieve higher L3DE scores, particularly in appearance simulation, while all models show room for improvement in motion and geometry consistency. Most generative videos still exhibit noticeable gaps from real ones in 3D visual coherence, as reflected in their lower L3DE scores. Beyond evaluating 3D visual coherence in AI-generated videos, L3DE can serve as a deepfake detector by applying a confidence score threshold. Despite not being trained on videos from specific sources, L3DE effectively identifies fake videos from Kling and others (see Table 1 in the appendix) with over 0.7 accuracy. Additionally, L3DE’s localized failure regions can help improve video synthesis. By inpainting flagged areas (see appendix), we can enhance the 3D visual coherence of generative videos.

Our contributions can be summarized as follows:

- • We take the first step in systematically investigating the 3D visual coherence of AI-generated videos across appearance, motion, and geometry—key factors in representing a dynamic 3D world. To facilitate quantitative analysis, we extract monocular clues from foundation models to disentangle these aspects.
- • We introduce Learned 3D Evaluation (L3DE) that quantifies the 3D visual coherence of a video using confidence scores from models trained on pairing data with contrastive loss. L3DE also highlights spatial and temporal regions as evidence for its assessment. Moreover, we integrate these three aspects to deliver a more robust assessment tool.
- • Through controlled user studies and 3D reconstruction experiments on diverse generative videos, we show that L3DE’s quantification scores and localized regions align well with user intent and reconstruction quality.
- • L3DE can be used for broader applications. Our experiments and studies provide valuable insights and findings about the capabilities of current video generation models.

#### 2. Related Work

Diffusion models for video generation. The success of diffusion models [15, 56] in image synthesis [7, 13, 30, 37, 43, 47, 49] has driven advancements in video generation [4, 12, 14, 16, 17, 22, 32, 70, 75, 79]. Stable Video Diffusion [3] leverages large-scale training for high-quality video synthesis. Sora [5] demonstrate the ability to simulate humans, animals, and environments, highlighting video generation as a potential path towards world simulation. Our work aim to help the community gain more understand-

ing about generative videos, especially their gap from realworld videos in terms of 3D viusal simulation capabilities.

AI-generated video evaluation. Existing metrics for evaluating AI-generated videos include Inception Score (IS) [50], Fr´echet Video Distance (FVD) [63], Perceptual Input Conformity(PIC) [70] and CLIPSIM [45], among others. Recent benchmarks, such as VBench [19] and EvalCrafter [29], establish standardized protocols by integrating automated metrics for comprehensive model comparisons. In contrast, our approach identifies differences between real and generative videos using a data-driven yet simple method, complemented by low-level statistical analysis to assess their 3D visual simulation capabilities.

Video feature extraction. Extracting appearance, motion, and geometry information is crucial for evaluating video realism. DINOv2 [38] shows strong image appearance representation, while optical flow estimation methods [8, 20, 62] provide robust motion features. Monocular depth cues encode rich geometric information, with recent methods like UniDepth [42] achieving precise metric depth estimation with excellent video consistency. We leverage these techniques to extract relevant features for our analysis.

3D scene reconstruction. Recent advancements in 3D reconstruction, such as NeRF-based [2, 26, 33, 34, 39, 44, 65, 73] and 3D-GS-based [18, 21, 69, 71] methods, have improved static and dynamic scene modeling. Despite the robustness of novel view synthesis (NVS) methods for inthe-wild scenes, unreliable camera pose estimation in such videos limits the feasibility of 3D scene reconstruction as a robust large-scale evaluation tool for assessing the 3D visual simulation capabilities of AI-generated videos.

#### 3. Data Curation

To gain a deeper understanding of the 3D visual simulation capabilities of AI-generated videos, we design a data curation process and compile a dataset that includes both realworld and AI-generated videos, as detailed in Table 1. Our model training, method validation, and subsequent analysis are all conducted using different subsets in this dataset.

In-the-wild real-world videos. We begin by collecting approximately 100,000 real-world, in-the-wild videos from Pexels [41]. These videos encompass a wide range of content, including animals, people, natural scenes, urban landscapes, indoor environments, and more. For raw video processing, we follow the method introduced in [3]. More details on the data processing can be found in the appendix.

Paired generative videos. We employ the open-source generative model Stable Video Diffusion (SVD) [3] to generate synthetic videos. To ensure the focus is on the 3D visual coherence, rather than potential biases in the generated content or color distribution, we condition SVD model us-

ing the first frames from real video clips. This enables SVD to generate paired synthetic samples that preserve the same semantic content and color distribution as their real video counterparts. Thus we create a paired generative video dataset, where the video clips share similar visual content to the real videos, minimizing the risk of model bias.

3D reconstruction verification set. To evaluate L3DE’s effectiveness, we curate a verification set using videos generated by the commercial model Kling [24], as SVDgenerated videos are typically of low quality, hindering 3D reconstruction and rendering. Our verification set consists of two parts: (1) Generated Videos for In-the-wild Scenes. Given the low success rate of pose estimation [10, 66, 71] on AI-generated videos, we generate diverse samples conditioned on keyframes from unseen real videos. We then screen the large pool of generated videos and retain 30 that successfully undergo 3D reconstruction. (2) Twin Videos for Public Scene Datasets. To analyze the correlation between 3D consistency and L3DE score, we iteratively generate twin videos for 15 scenes from public static datasets (i.e., Mip-NeRF360 [2], Tanks-and-Temples [23]) and dynamic datasets (i.e., Hyper-NeRF [39], Neural 3D Video Synthesis Dataset [26]), ensuring that each scene yields at least one video that successfully undergoes COLMAP and reconstruction. Each twin video pair is generated using one real frame as the start frame and another with sufficient overlap as the end frame, maintaining close alignment with the real 3D content. Videos from (1) and (2) form the 3D reconstruction verification set, totaling 3000 videos. For validation experiments, we use only videos that successfully undergo pose estimation, while the entire set is used in supplementary fake video detection experiments.

- 3D visual simulation benchmark. We conduct studies using L3DE on generated videos from recent commercial generative models, augmented with data from [74], to assess their ability to simulate the 3D visual world. The dataset includes videos from models such as Sora [5], Kling [24], Runway-Gen3 [48], Luma [31], MiniMax [35], Vidu [55], and CogVideoX [72]. To ensure relevance, we exclude videos with non-realistic content, such as animations. Since all videos are generated with the same set of image or text prompts, this dataset enables a direct and fair comparison of 3D visual simulation capabilities across different models by eliminating prompt-induced variability. Furthermore, we provide 14,000 unseen real video samples as references to establish an empirical upper bound for L3DE scores.
- 4. Learned 3D Evaluation

Below, we first discuss proxies for representing the 3D visual world, followed by a detailed explanation of the newly proposed Learned 3D Evaluation (L3DE) for assessing the 3D visual simulation capabilities of AI-generated videos.

Source Synthetic/Real Number of Videos Clip Length Resolution Frame Rate Prompt Type Paired Real/Synthetic Video Set

Pexels [41] Real 80,000 4s Variable Variable – Stable Video Diffusion [3] Synthetic 80,000 4s 1024*576 7 FPS I2V

###### 3D Reconstruction Verification Set

Kling 1.5 [24] Synthetic 3,000 5s Variable 30 FPS I2V & T2V

###### 3D Visual Simulation Benchmark

Pexels [41] Real 14,000 4s Variable Variable – Runway-Gen3 [48] Synthetic 539 5s 1280*768 24 FPS I2V & T2V MiniMax [35] Synthetic 539 5s 1280*720 25 FPS I2V & T2V Vidu [55] Synthetic 539 3s Variable 24 FPS I2V & T2V Luma Dream Machine 1.6 [31] Synthetic 539 Variable Variable 24 FPS I2V & T2V

- Kling 1.5 [24] Synthetic 539 5s Variable 30 FPS I2V & T2V CogVideoX-5B [72] Synthetic 539 6s 720*480 8 FPS I2V & T2V Sora [5] Synthetic 539 5s Variable 30 FPS I2V & T2V
- Kling 2.1 [24] Synthetic 539 5s Variable 30 FPS I2V & T2V

Table 1. Overview of our dataset, which consists of (1) Paired Real/Synthetic Video Set, designed to study the gap between real-world and AI-generated videos; (2) the 3D Reconstruction Verification Set, curated for validating L3DE through 3D reconstruction; and (3) the 3D Visual Simulation Benchmark, which includes videos from multiple generative models to evaluate their 3D visual simulation capabilities.

##### 4.1. Proxies for Representing 3D Visual World

Reconstructing and rendering in-the-wild videos to assess 3D world simulation is challenging, primarily due to issues such as unreliable camera pose estimation [52, 53, 71]. Beyond reconstructing a scene in 3D space, the realism of the 3D visual world is shaped by multiple perceptual factors. Inspired by [11, 51], we identify three key aspects: 1) Appearance: Visual attributes of video frames, including color, texture, and lighting; 2) Motion: Temporal dynamics and changes within the video; and 3) Geometry: The spatial structure and shape of objects in the frames. These cues reflect the consistency of a video’s 3D structure and can be reliably estimated from videos using foundation models, which we leverage as proxies for the 3D visual world. We extract these cues using the following foundation models:

- • Appearance representation: Instead of simply using the original RGB information, we extract per-frame visual feature with DINOv2 [38] as the appearance representation. Its features are capable of cross-image dense and sparse matching [9, 38], which enhances the potential to capture cross-frame appearance consistency.
- • Motion representation: We leverage optical flow, which is well-studied to represent motion, to examine the motion pattern differences between synthetic and real videos. To be more specific, we employ RAFT [62], a state-ofthe-art optical flow estimation model, to extract optical flow between the adjacent frames.
- • Geometry representation: To investigate the geometric properties of generative videos, we leverage the perframe depth as the geometry representation. Depth con-

veys many 2.5D geometric cues, such as occlusion, spatial relationships, scales, and so on. In detail, considering the cross-frame scale consistency, we adopt metric depth from UniDepth [42] as it has a uniform scale and provides better consistency across frames, which aids in perceiving changes in the geometric structure of the video.

##### 4.2. Design of L3DE

With the prepared data and extracted 3D visual proxies, we develop L3DE. The model first trains a classifier on the paired real/synthetic video dataset in Table 1, enabling it to learn to distinguish them based on the three proxies. This is achieved with a contrastive learning objective, which enhances the discriminative power of the learned features. Additionally, we integrate Grad-CAM [54] to enable L3DE to identify simulation traits. Finally, we design a fusion module that combines all three 3D proxies to produce a more comprehensive evaluation score for video assessment.

###### Classifier construction and training.

Based on the 3D proxies outlined in Sec. 4.1, we design a 3D convolutional network with multiple layers interleaved with ReLU activation functions. It predicts the confidence score evaluating whether a sample belongs to real or synthetic videos. Further details are provided in the appendix. The penultimate layer features are used to construct the contrastive loss. For any input generative video feature fgen, the loss encourages pushing apart its closest real video feature, thereby making real video feature more distinguishable. It

is computed as:

Lcontrastive =

i

exp − fgen(i) − freal(j(i))

2 2

, (1)

where freal(j(i)) is the closest real video feature to fgen(i) in Euclidean distance. The total loss function combines the clas-

sification loss Lcls and contrastive loss Lcontrastive as follows: L = Lcls + λLcontrastive. (2)

As the network learns to distinguish between real and synthetic videos, its confidence scores serve as a quantitative metric for assessing how closely an input video resembles real-world videos in 3D visual coherence. To interpret its predictions and understand the underlying evidence, we apply Grad-CAM [54], which generates a class-discriminative localization map by backpropagating gradients to the last convolutional layer. This map highlights the video regions that mostly influence the model’s decision (see Fig. 4).

Feature fusion for comprehensive scores. Since video content inherently combines appearance, motion, and geometry, we design a feature fusion module for a more robust and comprehensive evaluation. Within the network, we concatenate features from these three aspects:

ffused = Concat(fapp, fmot, fgeo), (3)

where fapp, fmot, and fgeo represent the features for appearance, motion, and geometry. The fused representation in our Fusion variant of L3DE produces an overall score, jointly accounting for all three aspects. This holistic evaluation provides a more comprehensive measure of 3D visual coherence, complementing single-aspect assessments.

#### 5. Validation of L3DE

To validate L3DE’s reliability in evaluating 3D visual coherence, we employ two complementary strategies: 3D reconstruction and human perceptual judgment. 3D reconstruction objectively assesses how well AI-generated videos preserve spatial structure and motion realism. However, pose estimation often fails on in-the-wild videos, meaning that only a subset of videos– those where camera parameters can be reliably estimated– can be reconstructed for validation. Within this subset, we use reconstruction to precisely verify L3DE’s predicted scores and detected regions. Beyond this subset, human perception provides a more flexible and perceptually grounded evaluation of 3D visual coherence, as it is not constrained by camera estimation failures. This allows us to confirm that L3DE remains effective across a wider range of generative videos.

##### 5.1. Validation using 3D Reconstruction

We conduct 3D reconstruction experiments in two controlled settings to assess the correlation between L3DE

Correlation with L3DE Fusion Appearance Motion Geometry

3D Reconstruction Quality 0.7566 0.7181 0.6669 0.3142 Human Ratings 0.6460 0.5643 0.4617 0.3479

Table 2. Spearman correlation between L3DE scores and different reference evaluations. The first row shows correlation with 3D reconstruction quality, while the second shows correlation with human ratings on the same verification dataset.

Appearance

Motion

Geometry

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

Pixel Count Proportion

Pixel Count Proportion

Pixel Count Proportion

Mean Pixel Error

Mean Pixel Error

Mean Pixel Error

Activation Value Range

Activation Value Range

Activation Value Range

Figure 2. Illustration of the statistics of activation value, pixel value error and the distribution of pixel number for each proxy.

scores and 3D rendering quality. Additionally, we examine whether L3DE’s detected inconsistencies align with reconstruction errors by comparing its localized regions to rendering-based discrepancy maps. These experiments utilize the 3D reconstruction verification dataset (Table 1).

L3DE score v.s. 3D rendering quality. We evaluate the correlation between L3DE scores and 3D reconstruction quality by optimizing a 3D representation, such as 3D-GS [21], across all video frames to reconstruct each scene. To ensure a more adaptive evaluation, we use the ‘Twin Videos for Public Scene Datasets’ from the 3D reconstruction verification set, which provides real and synthetic videos of the same content for fair comparisons. For static scenes, we assess L3DE’s appearance and geometry scores by measuring visual fidelity and spatial accuracy. For dynamic scenes, we focus on validating the motion score by analyzing temporal coherence and movement realism. Specifically, we use 3D-GS [21] for static scenes and SC-GS [18] for dynamic scenes. Rendering quality is quantified using Peak Signal-to-Noise Ratio (PSNR). To compensate for contentdependent variations in PSNR, we normalize the rendering quality of synthetic videos Qsynthetic relative to that of real videos Qreal. This normalization mitigates scene-specific biases, leading to a more robust assessment. The normalized quality difference is defined as :

∆Q = max(Qreal − Qsynthetic,0). (4)

To quantify the disparity between real and synthetic videos, we define the simulation gap G based on L3DE scores S:

###### G = 1 − S. (5)

We then evaluate L3DE’s ability to capture 3D rendering quality by computing the correlation between ∆Q and G. As shown in Table 2, L3DE scores are positively correlated

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

SyntheticRealSyntheticReal

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

[Figure 95]

[Figure 96]

- Figure 3. Frames and reconstruction results of twin videos. Even though synthetic videos appear plausible, they do not achieve the same level of 3D scene reconstruction accuracy as real videos (see the Shrunken Gaussians in the rightmost column). This discrepancy underscores a key limitation: current generative videos are not yet adept at faithfully simulating the world in terms of 3D visual coherence.

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

|[Figure 106]|
|---|

Mean Pixel Error

Pixel Count Proportion

Activation Value Range

|[Figure 107]|
|---|

Mean Pixel Error

Pixel Count Proportion

Activation Value Range

|[Figure 108]|
|---|

Mean Pixel Error

Pixel Count Proportion

Activation Value Range

CaseThreeCaseTwoCaseOne

(a) AI-generated Frame (b) Rendered Frame (c) Pixel Difference (d) Grad-CAM Result (e) Activation Alignment

[Figure 109]

[Figure 110]

[Figure 111]

- Figure 4. Illustration of 3D inconsistencies identified by L3DE. From left to right: (a) AI-generated video frame; (b) rendered frame with 3D reconstruction with pose aligned with the original view; (c) pixel-level difference between (a) and (b); (d) Grad-CAM result from the L3DE network, which closely aligns with (c); (e) Blue solid line: large (normalized) activation value in (d) is highly aligned with large mean pixel value error in (c). Green dashed line: areas with high (normalized) activation values cover only a small portion of the entire frame. L3DE identifies key artifacts in the cases: (1) unnatural hand motion in the first case, reflected in a low motion score of 0.4642; (2) abrupt geometric deformation of the marked object in the second case, with a geometry score of 0.637; and (3) sudden texture changes in the chair and table in the third case, resulting in an appearance score of 0.2578.

with 3D rendering quality, indicating that higher L3DE scores correspond to the superior rendering fidelity. Notably, our L3DE fusion model achieves the highest correlation of 0.7566, demonstrating strong alignment with the reconstruction-based evaluation.

L3DE localized region vs. inconsistent region. We as-

sess L3DE’s ability to localize 3D-inconsistent regions in AI-generated videos using the ‘Generated Videos for Inthe-wild Scenes’ from the 3D reconstruction verification dataset. Grad-CAM [54] highlights the regions L3DE focuses on for real-fake classification. To establish reference 3D-inconsistent regions, we split the dataset into training and test sets, ensuring discrepancies are measured only

from test viewpoints to mitigate overfitting effects in GSbased reconstruction. We then quantify the alignment between L3DE-detected regions and rendering-based discrepancy maps. Fig. 2 presents the quantitative correlation results, demonstrating strong alignment between L3DEdetected and rendering-inconsistent regions. Qualitative comparisons are shown in Fig. 4.

the videos used in the rendering quality experiments and compute their correlation with L3DE scores. It shows that L3DE consistently aligns with both reconstruction quality and human judgment on the same dataset (see Table 2).

Appearance Motion Geometry

Average score 0.8600 0.7200 0.7400 Spearman’s ρ 0.4894 0.4026 0.4317

##### 5.2. Validation using Human Judgment

To complement reconstruction-based validation, we conduct human evaluations to assess whether L3DE scores and detected regions align with human perception judgments. This ensures that L3DE not only correlates with objective reconstruction quality but also reflects subjective judgments of 3D visual coherence.

Table 3. Average human plausibility scores on the Grad-CAM visualization and correlation between L3DE localized region and human-annotated region from different aspects.

L3DE localized region v.s. human plausibility. We further validate L3DE’s localized regions through an additional user study. 10 volunteers are shown highlighted regions from both L3DE and randomly generated maps, without disclosure of their source to prevent bias. Each participant rates the plausibility of the highlighted regions on a 1–5 scale, with scores subsequently normalized. As shown in Table 3, L3DE achieves a significantly higher score (0.7–0.8) compared to random maps (average 0.21, minimum 0.2). To reinforce our validation, we conduct a second experiment where 10 participants annotate unrealistic regions in 30 unseen videos. The correlation between these annotations and L3DE-detected regions (Table 3) further confirms that L3DE effectively aligns with human perception of unrealistic content.

Fusion 𝜌 = 0.6484

Appearance 1.0 𝜌 = 0.5970

1.0

HighDensityLowDensity

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

0.5

0.5

0.0

0.0

0.5 1.0

0.5 1.0

Geometry

Motion 𝜌 = 0.4792

1.0

1.0

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

𝜌 = 0.5418

0.5

0.5

#### 6. Analysis and Applications of L3DE

##### 6.1. Comparison with Existing Metrics

0.0

0.0

While existing methods such as VBench [19] and EvalCrafter [29] provide general-purpose video evaluation, they do not specifically assess 3D visual coherence. To compare them with L3DE, we select relevant metrics from each benchmark that focus on spatial and temporal consistency. We evaluate them based on their correlation with human judgments, following the standard approach for validating evaluation methods [19, 29]. As shown in Table 4, L3DE achieves a stronger correlation with human ratings than existing metrics, demonstrating its effectiveness in assessing 3D realism in generative videos. Beyond correlation analysis, L3DE also introduces unique capabilities, such as identifying unrealistic areas—an aspect missing from existing metrics—which enhances interpretability and provides actionable insights for improving generative models.

0.5 1.0

0.5 1.0

Figure 5. The correlation between L3DE scores and human ratings. The X-axis represents the average human ratings and the Y-axis represents the L3DE scores.

L3DE scores v.s. human ratings. First, We validate the correlation between L3DE scores and human evaluations through a user study involving 15 participants who provided 4,500 annotations on 300 randomly selected AI-generated videos, rating their realism in terms of 3D visual coherence. More details on the study setup are provided in the appendix. For each video, we compute the average participant rating as the human rating. We then evaluate L3DE scores across appearance, geometry, motion, and their fusion. We then compute the Spearman correlation between these scores and the human ratings. As shown in Fig. 5, L3DE scores exhibit a strong positive correlation with human evaluations, confirming their reliability in assessing generative videos. Notably, the fusion score achieves the highest correlation, underscoring the effectiveness of our fusion strategy. Additionally, we analyze human ratings for

##### 6.2. Benchmarking Video Generation Models

Given that L3DE effectively evaluates the 3D visual coherence of generative videos, we expand video generation model benchmarking by introducing 3D visual simulation capabilities as a new assessment dimension, which has been

###### Metric Method Spearman’s ρ

Subject Consistency VBench [19] 3.90 Background Consistency VBench [19] 20.68 Motion Smoothness VBench [19] 19.99 Temporal Consistency EvalCrafter [29] 13.85 L3DE Fusion Score L3DE 64.84

- Table 4. Correlation of L3DE scores and automatic metrics from different baselines with human ratings.

Generators Fusion Appearance Motion Geometry

Runway-Gen3 [48] 0.7162 0.6946 0.5768 0.6739 MiniMax [35] 0.7932 0.7714 0.6098 0.7251 Vidu [55] 0.7052 0.6406 0.6228 0.7615 Luma 1.6 [31] 0.5062 0.4950 0.5853 0.6800

- Kling 1.5 [24] 0.7518 0.7247 0.5926 0.6927 CogVideoX-5B [72] 0.6104 0.5893 0.6203 0.7539 Sora [5] 0.8895 0.8394 0.6467 0.7458

- Kling 2.1 [24] 0.8904 0.8129 0.6735 0.7623 Real Videos 0.9999 0.9950 0.8321 0.8435

- Table 5. Benchmarking results of generative models. The Fusion column, highlighted as the primary L3DE ranking, represents the overall 3D visual coherence. Real videos achieve near-perfect scores, serving as an empirical upper bound for L3DE.

largely overlooked in existing benchmarks. Using the data outlined in Sec. 3, we evaluate leading generative models based on their ability to simulate the 3D visual world and present our findings below.

Quantitative Studies. To benchmark generative models, we compute the average L3DE score across all generated videos for each model. The fusion score represents the model’s overall evaluation, while individual scores for appearance, motion, and geometry are also reported. The evaluation results are shown in Table 5 and the model rankings strongly correlate with large-scale human-preference benchmarks [1] (see appendix), confirming the robustness and generalizability of L3DE. Based on the overall fusion score, Kling 2.1 [24] and Sora [5] produces the highestquality videos in terms of 3D visual simulation assessment. While these models excel in appearance simulation, their motion and geometry scores remain significantly lower, with minimal variation among models. As a reference, we calculate L3DE scores for a large set of 14,000 real video clips and they achieve an average fusion score of 0.9999, reaffirming the reliability of L3DE. Kling’s and Sora’s fusion and appearance scores exceed 0.8, but their motion and geometry scores are notably lower, indicating potential areas for improvement. These findings indicate that:

• While some videos generated by leading models achieve high L3DE scores, most still exhibit significant gaps in 3D visual coherence compared to real videos.

• The primary distinction among video generation models lies in their ability to simulate appearance, whereas their motion and geometry performance remains notably lower, lacking the fidelity of real-world videos.

Qualitative Studies. We analyze the Grad-CAM results from the fusion version of L3DE and observe that, while it provides less direct interpretability compared to individual aspects, it effectively captures more complex artifacts. For instance, Fusion Grad-CAM effectively identifies physically implausible interactions, such as issues with liquid, glass, and human scaling. For more qualitative studies, please refer to the supplementary. These findings indicate that integrating multiple cues in L3DE enhances its capability to detect higher-level inconsistencies beyond individual appearance, motion, or geometry assessments.

##### 6.3. Applications

We further demonstrate several downstream applications of L3DE, including fake video detection by applying a threshold on the prediction score and enhancing generative video quality by inpainting regions identified by L3DE. More details on these applications can be found in the appendix.

#### 7. Conclusion and Discussion

We present Learned 3D Evaluation (L3DE), a robust and interpretable framework for assessing the 3D visual coherence of generative videos. By leveraging monocular 3D cues—motion, depth, and appearance—from foundation models, L3DE provides an objective and quantifiable measure of discrepancies between real and synthetic videos. Extensive experiments demonstrate L3DE’s effectiveness in evaluating videos from generative models, revealing significant 3D simulation gaps and subtle inconsistencies that are often overlooked by human observers. L3DE aligns well with reconstruction quality and human judgment, validating its role as an analytical tool and deepfake detector. Beyond evaluation, L3DE’s insights can inform video synthesis improvements, offering a promising avenue for enhancing the realism of AI-generated content. Overall, L3DE presents a powerful tool for advancing our understanding of AI’s capabilities in simulating the 3D visual world, with broad applications in video generation and evaluation.

Acknowledgments: This work has been supported by Kuaishou Technology, Hong Kong Research Grant Council - Early Career Scheme (Grant No. 27209621), General Research Fund Scheme (Grant No. 17202422), RGC Matching Fund Scheme (RMGS), Lingnan University StartUp Grant fund code: SUG-001/2526, and Faculty Research Grant fund code:106106. Part of the research work is conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust.

#### References

- [1] Artificial Analysis. Video Arena Leaderboard. https: //artificialanalysis.ai/text-to-video/ arena?tab=Leaderboard. 8, 17, 18
- [2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 3
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 4, 13, 16
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 1, 2, 3, 4, 8, 18

- [6] Brandon Castellano. Pyscenedetect: Video cut detection and analysis tool. https://github.com/ Breakthrough/PySceneDetect. 13
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [8] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learning optical flow with convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 2758–2766, 2015. 3
- [9] Johan Edstedt, Qiyu Sun, Georg B¨okman, M˚arten Wadenb¨ack, and Michael Felsberg. Roma: Robust dense feature matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19790– 19800, 2024. 4
- [10] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309, 2, 2024. 2, 3
- [11] James A Ferwerda. Three varieties of realism in computer graphics. In Human vision and electronic imaging viii, pages 290–297. SPIE, 2003. 2, 4
- [12] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023. 2

- [13] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696–10706, 2022. 2
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [17] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [18] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4220–4230, 2024. 1, 3, 5
- [19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023. 3, 7, 8, 16
- [20] Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox. Flownet 2.0: Evolution of optical flow estimation with deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2462–2470, 2017. 3
- [21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 1, 3, 5

- [22] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023. 2
- [23] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 3
- [24] Kuaishou. Kling ai. https://klingai.kuaishou.com/, 2024.06. 2, 3, 4, 8, 15, 18
- [25] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421, 2024. 2
- [26] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt,

- Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5521–5531, 2022. 3
- [27] Jiahui Liu, Chirui Chang, Jianhui Liu, Xiaoyang Wu, Lan Ma, and Xiaojuan Qi. Mars3d: A plug-and-play motionaware model for semantic segmentation on multi-scan 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9372– 9381, 2023. 15
- [28] Jiahui Liu, Xin Wen, Shizhen Zhao, Yingxian Chen, and Xiaojuan Qi. Can ood object detectors learn from foundation models? In European Conference on Computer Vision, pages 213–231. Springer, 2024. 18
- [29] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22139–22149, 2024. 3, 7, 8, 16, 17
- [30] Zhengzhe Liu, Qing Liu, Chirui Chang, Jianming Zhang, Daniil Pakhomov, Haitian Zheng, Zhe Lin, Daniel Cohen-Or, and Chi-Wing Fu. Object-level scene deocclusion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [31] LumaLabs. Dream machine. https://lumalabs.ai/dreammachine, 2024.06. 3, 4, 8, 18
- [32] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10209–10218, 2023. 2
- [33] Xiaoyang Lyu, Chirui Chang, Peng Dai, Yang-Tian Sun, and Xiaojuan Qi. Total-decom: Decomposed 3d scene reconstruction with minimal interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20860–20869, 2024. 3
- [34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 1, 3
- [35] MiniMax. Hailuo ai. https://hailuoai.com/video, 2024.09. 2, 3, 4, 8, 18
- [36] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G Derpanis, Jonathan Kelly, Marcus A Brubaker, Igor Gilitschenski, and Alex Levinshtein. Spin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20669–20679,

2023. 13

- [37] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [38] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 3, 4, 14
- [39] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021. 3
- [40] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 16
- [41] Pexels. https://www.pexels.com/, 2023. 2, 3, 4, 13
- [42] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. arXiv preprint arXiv:2403.18913, 2024. 2, 3, 4, 14
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [44] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318–10327, 2021. 3
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [46] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 13
- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [48] Runway. Gen-3. https://runwayml.com/, 2024.06. 3, 4, 8, 18
- [49] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [50] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 3
- [51] Ayush Sarkar, Hanlin Mai, Amitabh Mahapatra, Svetlana Lazebnik, David A Forsyth, and Anand Bhattad. Shadows don’t lie and lines can’t bend! generative models don’t

- know projective geometry... for now. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28140–28149, 2024. 4
- [52] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 4
- [53] Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016. 4
- [54] Ramprasaath R Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and Dhruv Batra. Grad-cam: Why did you say that? arXiv preprint arXiv:1611.07450, 2016. 2, 4, 5, 6, 16
- [55] ShengShu-AI. Vidu. https://www.vidu.studio/, 2024.07. 3, 4, 8
- [56] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [57] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2149– 2159, 2022. 13
- [58] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28130–28139, 2024. 13, 14
- [59] Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, and Xiaojuan Qi. Data pruning via movingone-sample-out. In Neural Information Processing Systems (NeurIPS), 2023. 18
- [60] Haoru Tan, Sitong Wu, Zhuotao Tian, Yukang Chen, Xiaojuan Qi, and Jiaya Jia. Saco loss: Sample-wise affinity consistency for vision-language pre-training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [61] Haoru Tan, Sitong Wu, Wei Huang, Shizhen Zhao, and Xiaojuan Qi. Data pruning by information maximization. In International Conference on Learning Representations (ICLR),

2025. 18

- [62] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 2, 3, 4, 14

- [63] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 3
- [64] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin

- Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2024. 2
- [65] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 3
- [66] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 2, 3
- [67] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8695–8704, 2020. 13, 14
- [68] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for diffusion-generated image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22445–22455, 2023. 13, 14
- [69] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20310–20320, 2024. 1, 3
- [70] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023. 2, 3
- [71] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20331–20341, 2024. 1, 2, 3, 4
- [72] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3, 4, 8, 18
- [73] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems, 35:25018–25032, 2022. 3
- [74] Ailing Zeng, Yuhang Yang, Weidong Chen, and Wei Liu. The dawn of video generation: Preliminary explorations with sora-like models. arXiv preprint arXiv:2410.05227, 2024. 3, 13
- [75] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 2

- [76] Jason Zhang, Gengshan Yang, Shubham Tulsiani, and Deva Ramanan. Ners: Neural reflectance surfaces for sparse-view 3d reconstruction in the wild. Advances in Neural Information Processing Systems, 34:29835–29847, 2021. 2
- [77] Shizhen Zhao, Changxin Gao, Yuanjie Shao, Wei-Shi Zheng, and Nong Sang. Weakly supervised text-based person reidentification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 18
- [78] Shizhen Zhao, Xin Wen, Jiahui Liu, Chuofan Ma, Chunfeng Yuan, and Xiaojuan Qi. Learning from neighbors: Category extrapolation for long-tail learning. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025. 18
- [79] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2

## How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach

### Supplementary Material

#### H. Applications for L3DE

In this section, we mainly demonstrate two downstream applications for our proposed L3DE: 1.) Fake video detection and 2.) Generative video refinement.

##### H.1. Fake Video Detection

L3DE is designed to evaluate the 3D real world simulation capabilities of AI-generated videos, enabling it to distinguish low-quality AI-generated videos from real-world ones. Motivated by this capability, we conduct fake video detection experiments to assess how well L3DE performs on this task. This can be achieved by setting a threshold on the L3DE score, allowing us to classify videos as real or fake based on their ability to simulate the real 3D visual world.

Specifically, we use fake videos from our 3D reconstruction verification set together with those from [74] and an equal number of unseen in-the-wild real videos from Pexels [41] to build a fake video detection benchmark. As there is currently no open-source general fake video detector to the best of our knowledge, we adapt fake image detection methods for videos. To do this, we compare L3DE fusion scores with existing fake image detection methods [58, 67, 68] by averaging frame-wise predictions to produce a final prediction for each video. The results are presented in Table A6.

The results indicate that L3DE scores exhibit strong performance in fake video detection, even though L3DE is not specifically designed for this task. Across videos generated by different models, L3DE scores generally achieve higher accuracy than image-based fake detection methods. These results suggest that most synthesized videos still have significant gaps in 3D simulation capabilities. In conclusion, L3DE scores demonstrate strong performance in fake video detection, despite not being specifically designed for this task.

##### H.2. AI-Generated Video Refinement

In current generative videos with regional artifacts, such artifacts often necessitate discarding the entire video if it does not meet the criteria for downstream tasks. However, with L3DE’s ability to identify and localize artifact regions, we can achieve AI-generated video refinement by removing these artifacts in a 3D-consistent manner.

Specifically, we utilize L3DE activation values to localize the regions of artifacts in the keyframes of the downsampled clip. We then employ SAM-2 [46] to refine and propagate the masks across the entire original generative video.

[Figure 120]

[Figure 121]

[Figure 122]

OriginalVideoRefinedVideo

[Figure 123]

[Figure 124]

[Figure 125]

Figure A6. A qualitative result of generative video refinement. In this example, bounding boxes highlight the regions where artifacts are detected in the original video. After refinement, these artifacts are successfully removed across all frames of the video.

Inspired by [36], we implement a 3D-GS-based multi-view consistent inpainting iteratively using LaMa [57].

We demonstrate our results for video refinement in Figure A6. Based on our findings, the artifact-detection capability of L3DE can effectively guide the post-processing step of video refinement, helping to remove artifacts in generative videos.

#### I. Data Processing

In this section, we detail our data processing procedures, including raw video processing and video feature extraction.

##### I.1. Raw Video Processing

We follow the approach introduced in [3] for raw video processing. First, we collect an open-world, in-the-wild long video dataset from Pexels [41], covering a wide range of content with varying aspect ratios, resolutions, and frame rates. Figure A7 showcases the diversity of our dataset. To avoid biases caused by cuts and fades, we apply PySceneDetect [6] to the long videos.

Next, to prepare paired data, we slice these videos into equal-length clips of 4 seconds. For videos that do not match the 16:9 aspect ratio, we apply a center crop and resize them to a resolution of 1024×576 with 25 frames. Additionally, we use the first frame of these processed video clips as image prompts for stable video diffusion [3] to generate paired synthetic samples. Moreover, we provide visualizations of randomly sampled paired videos in Figure A12. As introduced in the main paper, we sample 160,000 paired videos for training the L3DE models.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Figure A7. Our collected real-world, in-the-wild videos encompass a wide range of visual content, from indoor to outdoor scenes, including people, animals, landscapes, food, and more.

Method Input MiniMax Kling 1.5 Runway-Gen3 Luma Dream Machine CogVideoX Vidu Sora Average

CNNDetection [67] Image 49.92 50.02 50.00 50.45 50.07 50.00 49.91 50.05 DIRE [68] Image 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 NPR [58] Image 60.19 67.91 64.99 54.06 35.79 36.04 60.82 54.25

L3DE Video 66.51 82.52 72.19 83.38 76.73 70.01 56.31 73.14

- Table A6. Fake video detection performance of L3DE scores and image-based approaches. The reported metric is accuracy with all values presented as percentages.

##### I.2. Video Feature Extraction

We extract video features using different foundation models following their official implementation: For appearance features, we extract frame-wise features from the DINOv2 ViT-G model [38]. For motion features, we input adjacent frames into RAFT [62] to obtain the optical flow sequence of the entire video. For geometry features, we extract per-frame metric depth using the UniDepth v2 ViT-S model [42].

To align the inputs from different proxies, we use the metric depth and DINOv2 features of the first 24 frames of the video clips, since the optical flow maps are calculated based on adjacent frames. This strategy ensures that L3DE simultaneously captures different modalities of 3D proxies.

##### I.3. The Impact of Data Diversity

Our goal is to construct a diverse training dataset to improve the robustness of L3DE. Diversity in training data plays a crucial role in enhancing generalization by exposing the model to a broad range of real-world and challenging scenarios. While data diversity can encompass various factors—such as object motion, scene complexity, and environmental variations—we focus on two key aspects in our analysis: (1) the role of object motion (static vs. mixed

[Figure 135]

Figure A8. Interface for human rating annotation. Users can provide a rating in the scoring section below after watching the video above.

static-dynamic scenes) and (2) the impact of scene diversity (indoor-only vs. mixed indoor-outdoor scenes). These controlled experiments illustrate how different types of training data contribute to model performance, reinforcing the im-

###### Experiment Training Data Test Data Accuracy

Static-only Static-scene videos Mixed-motion videos 69.55 Mixed-motion Static + dynamic videos Mixed-motion videos 77.70

Indoor-only Indoor videos Mixed indoor-outdoor 67.60 Indoor + Outdoor Indoor + outdoor videos Mixed indoor-outdoor 76.55

- Table A7. Impact of Data Diversity on Model Performance. Training on diverse data significantly improves accuracy.

portance of a diverse dataset.

Effect of Object Motion To assess the impact of object motion, we train two models using different datasets: one on 10,000 static-scene real and synthetic videos (1:1 ratio), and the other on an equally sized dataset that includes both static and dynamic scenes. Both models are evaluated on a 2,000-sample test set, which consists of an equal number of real and synthetic videos featuring mixed motion. The synthetic videos are generated using Kling [24]. As shown in Table A7, the model trained solely on static scenes underperforms compared to the one trained with motion diversity (69.55 vs. 77.70), confirming that incorporating object motion in training significantly improves generalization.

Effect of Scene Diversity To analyze the effect of scene diversity, we train one model using 10,000 indoor real and synthetic videos (1:1 ratio), and another using 10,000 mixed indoor-outdoor videos. Both models are evaluated on a 2,000-sample mixed indoor-outdoor test set, maintaining a 1:1 ratio of real to synthetic videos. As seen in Table A7, the model trained only on indoor data exhibits lower accuracy (67.60 vs. 76.55), demonstrating that exposure to a wider variety of environments enhances model robustness.

#### J. L3DE Architecture

In this section, we provide details about the L3DE architecture, including both the single-proxy and fusion versions.

##### J.1. Single-proxy Network

First, we illustrate our design of the single-proxy version of the L3DE network in Figure A9(a). Given a single aspect proxy, such as frame-wise appearance features of a video as input, the 3D ConvNet produces a corresponding confidence score for the video. Specifically, the single-proxy L3DE is a single-branch 3D convolutional network focusing on capturing spatiotemporal features from a single input modality.

The network begins with sequential 3D convolutional layers that progressively encode high-level representations of the input through non-linear activations and feature refinement. After the convolutional stages, the feature map is flattened into a 1D vector, which is passed through a fully

connected layer to reduce dimensionality. The final prediction is performed using another fully connected layer with a sigmoid activation, producing a confidence score.

##### J.2. Fusion Network

Next, we illustrate the design of the fusion version of L3DE in Figure A9(b). In detail, the fusion network is a 3D ConvNet integrating appearance, motion, and geometric features through a multi-branch architecture. Each input modality—appearance features, motion features, and geometric features—is processed separately using specialized 3D convolutional layers, which hierarchically encode spatiotemporal information through non-linear activations and down-sampling via strided convolutions.

The outputs of the three branches are concatenated along the channel dimension, enabling the model to jointly leverage complementary features from all modalities, in line with prior efforts [27]. The fused representation undergoes further refinement through additional convolutional layers that capture high-level correlations across the integrated features. The network concludes with two fully connected layers and a final sigmoid activation for score prediction.

We also provide the architecture details of the fusion network in Figure A10. Note that each single-branch model adopts the same architecture as its corresponding branch in the fusion network.

##### J.3. Ablation Study

In this section, we conduct an ablation study to analyze the impact of contrastive loss and feature fusion strategies on distinguishing real and synthetic videos in L3DE. As shown in Table A8, both contrastive loss and fusion strategies play a crucial role in model performance. We compare two feature fusion methods: (1) Element-wise Addition (Add), where features from different sources are summed component-wise; and (2) Feature Concatenation (Concat), where features are stacked along the channel dimension to retain independent information. First, comparing the Add and Concat fusion strategies, we observe that Concat consistently outperforms Add. Without contrastive loss, Concat achieves 68.77%, surpassing Add (66.01%), indicating that concatenation preserves richer feature representations. When contrastive loss is introduced, performance improves significantly in both fusion strategies (+3.25% for Add and +4.37% for Concat), confirming that the loss function enhances feature discrimination. Our L3DE setting (Concat + Contrastive Loss) achieves the highest accuracy (73.14%), as highlighted in Table A8. These results demonstrate that contrastive loss effectively boosts performance by improving the feature separation between real and synthetic videos. Additionally, the superior performance of Concat over Add suggests that maintaining richer feature representations is beneficial for this task. Thus, we adopt the Concat + Con-

|3D ConvNet<br><br>Frame 1<br><br>Appearance features<br><br>Appearance Score<br><br>time|
|---|

|3D ConvNet<br><br>Frame 1<br><br>Motion features<br><br>Motion Score<br><br>time|
|---|

|3D ConvNet<br><br>Frame 1<br><br>Geometry features<br><br>Geometry Score<br><br>time|
|---|

|3D Conv<br><br>Frame 1<br><br>Appearance features<br><br>time<br><br>3D Conv<br><br>Frame 1<br><br>Motion features<br><br>Fusion Score<br><br>time<br><br>3D Conv<br><br>Frame 1<br><br>Geometry features<br><br>time<br><br>+<br><br>3D ConvNet|
|---|

(a) Single-proxy Network (b) Fusion Network

Figure A9. The design of both single-proxy network shown in part (a), and fusion network illustrated in part (b).

|Conv3D (128)|
|---|

Appearance Feature (1536 channels)

|Conv3D (16)|
|---|

|Conv3D (32)|
|---|

|Conv3D (64)|
|---|

|Conv3D (128)|
|---|

|Conv3D (512)|
|---|

|Conv3D (256)|
|---|

|Conv3D (512)|
|---|

|FC (128)|
|---|

|FC (1)|
|---|

Motion Feature (2 channels)

Concatenate (384 channels)

Flatten

Sigmoid

|Conv3D (16)|
|---|

|Conv3D (32)|
|---|

|Conv3D (64)|
|---|

|Conv3D (128)|
|---|

Geometry Feature (1 channel)

- Figure A10. The detailed architecture of fusion network, a 3D convolutional neural network designed for multimodal feature fusion. The network takes three input streams: Appearance Features (1536 channels), Motion Features (2 channels), and Geometry Features (1 channel). Each stream undergoes a series of 3D convolutional layers with ReLU activations before being concatenated into a 384-channel fused representation. The concatenated features are further processed through additional convolutional layers, followed by flattening and fully connected layers.

#### K. User Study

###### Fusion Strategy Contrastive Loss Accuracy (%)

In this section, we provide detailed descriptions of the user studies mentioned in the main paper.

✗ 66.01 ✓ 69.26

Element-wise Addition

✗ 68.77 ✓ 73.14

##### K.1. User Study for Video Ratings

Feature Concatenation

We conduct a user study involving 15 volunteers, who provide a total of 4,500 annotations on 300 randomly selected generative videos from our dataset. Annotators are recruited via our internal platform. Participants are aged between 20 and 40, come from diverse educational backgrounds, and do not possess specialized computer vision knowledge, ensuring broad representativeness.

- Table A8. Ablation study on contrastive loss and feature fusion strategies (Concat vs. Add). The highlighted row represents our setting and results.

trastive Loss setting as the default configuration in L3DE.

To ensure annotation quality, volunteers complete a prelabeling task following previous work [19, 29] and only those showing consistent and accurate judgments qualify for the main study.

##### J.4. Implementation Details

Qualified participants receive clear scoring guidelines to ensure consistency. The guidelines explicitly instruct them to evaluate the realism of videos based on 3D visual coherence in appearance, motion, and geometry, rather than semantic content or other unrelated factors. Participants rate each video’s realism on a 1 to 5 scale, with clear definitions provided:

We implement our 3D ConvNet using PyTorch [40]. The models are trained with a learning rate of 1e-4 and a batch size of 20. For video generation with SVD-XT [3] and training of L3DE models, we utilize NVIDIA A100 GPUs. Additionally, NVIDIA 4090 GPUs are used for conducting 3D reconstruction experiments. We follow the official implementation for Grad-CAM [54] visualization.

• Score 1: Videos exhibit obvious visual artifacts, severe

geometry deformation, unnatural motion, or evident synthetic features.

- • Score 2: Videos have significant artifacts clearly distinguishable from real ones, significantly impacting realism.
- • Score 3: Videos contain noticeable but non-disruptive artifacts, moderately realistic overall.
- • Score 4: Videos closely resemble real-world footage with minor and infrequent artifacts.
- • Score 5: Videos are indistinguishable from real-world footage, exhibiting minimal to no noticeable artifacts or inconsistencies.

Participants rate all 300 videos through our internal annotation interface (Figure A8). After collecting annotations, we then compute the Spearman correlation coefficients between these human ratings and the L3DE scores across different modalities. Moreover, to further verify L3DE’s alignment with human perception, we conduct additional human evaluations on the subset ”Generated Videos for In-thewild Scenes.” These evaluations comprehensively validate our method’s performance on the same dataset, facilitating comparison with the reconstruction-based validation.

##### K.2. User Study for Grad-CAM Region Ratings

To evaluate the interpretability and effectiveness of the localized regions identified by L3DE (via Grad-CAM), we conduct an additional user study involving 10 qualified volunteers. Participants review 40 randomly selected generative videos from our dataset, each presented alongside visualizations highlighting artifact regions.

Among these 40 videos, for each modality (appearance, motion, and geometry), we randomly select 10 diverse videos. Additionally, we insert 10 videos with randomly generated Grad-CAM highlights serving as a control group to mitigate potential participant biases toward highlighted regions.

Participants view each video along with the corresponding visualization and rate the relevance of highlighted regions to the observed visual artifacts using the following scale:

- • Score 1: Highlighted regions are irrelevant or poorly match the perceived artifacts.
- • Score 2: Highlighted regions slightly match perceived artifacts but miss major inconsistencies.
- • Score 3: Highlighted regions partially match perceived artifacts.
- • Score 4: Highlighted regions generally reflect perceived artifacts with minor discrepancies.
- • Score 5: Highlighted regions accurately reflect major perceived artifacts.

Participants are unaware that 10 of the provided visualizations are randomly highlighted (random baseline) to minimize bias. We specifically evaluate these procedures on the subset ”Generated Videos for In-the-wild Scenes” to

Visual Quality Motion Quality Temporal Consistency

EvalCrafter 55.4 45.0 56.7 Ours 67.0 43.6 58.0

Table A9. Correlation between L3DE scores and human annotations from the ECTV dataset. Appearance, motion, and fusion scores correspond to visual quality, motion quality, and temporal consistency, respectively.

verify L3DE’s effectiveness in localizing artifacts under realistic conditions. Average scores across participants quantify human plausibility, as presented in the main paper. Additionally, 10 participants manually annotate regions they perceive as unrealistic in 30 unseen videos. This serves as a further validation step for Grad-CAM localization, allowing us to quantitatively evaluate pixel-level correlations between human annotations and Grad-CAM highlighted regions.

#### L. More Experiments for L3DE

##### L.1. Additional Comparison with Baselines

To further assess the generalizability of L3DE, we compare its performance against EvalCrafter [29] using correlation metrics on the EvalCrafter Text-to-Video (ECTV) Dataset. EvalCrafter evaluates video quality across multiple dimensions, among which visual quality, motion quality, and temporal consistency are the most relevant to L3DE’s evaluation criteria. As shown in Table A9, L3DE achieves a higher correlation with human annotations in terms of visual quality (+11.6%) and temporal consistency (+1.3%), demonstrating its strong ability to assess both appearance and temporal coherence. L3DE achieves a comparable correlation in motion quality (43.6% vs. 45.0%), indicating its effectiveness in capturing motion fidelity. These results suggest that L3DE provides a more comprehensive and robust evaluation, particularly in aspects that contribute to overall perceptual quality.

##### L.2. Comparison with External Human Preference Benchmark

To further validate the generalizability and robustness of our L3DE results, we compare the ranking of generative video models obtained by L3DE against the publicly available large-scale human preference leaderboard from Video Arena [1], which aggregates extensive user votes. Although the datasets and specific videos differ, the model rankings obtained by L3DE closely align with those in the Video Arena leaderboard as shown in Table A10. Notably, both assessments consistently identify similar high-performing and lower-performing generative models. This alignment further confirms that L3DE effectively captures general human perceptual judgments regarding video realism, strengthening the validity of our evaluation framework.

###### Generative Model L3DE Score ↑ Arena ELO ↑ Ranking (Ours / Arena)

Sora [5] 0.8895 1077 1 / 1 MiniMax [35] 0.7932 1067 2 / 2 Kling 1.5 [24] 0.7518 1058 3 / 3 Runway-Gen3 [48] 0.7162 1017 4 / 4 CogVideoX [72] 0.6104 811 5 / 6 Luma [31] 0.5062 997 6 / 5

Table A10. Comparison of generative model rankings obtained by L3DE and human preference judgments from Video Arena [1]. Rankings only consider models appearing in both our 3D visual simulation benchmark and the Video Arena leaderboard. Although datasets differ and there are minor discrepancies in model versions due to rapid iterations in commercial models, the consistent ranking demonstrates L3DE’s alignment with general human perceptual judgments.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

- Figure A11. Both clips are from Sora [5]. The first row highlights implausible liquid-glass-table interaction (Score: 0.7256), while the second reveals an incorrect human scale (Score: 0.0023).

##### L.3. More Qualitative Results

In this section, we provide additional qualitative results of L3DE for reference. Specifically, we illustrate the GradCAM results and analyses of L3DE’s appearance, motion, and geometry components in Figures A13, A14, and A15, respectively. We further include comprehensive qualitative examples from the Fusion Grad-CAM analysis, highlighting complex artifacts captured by integrating multiple cues. Figure A11 demonstrates cases involving physically implausible interactions, such as abnormal behaviors of liquids interacting with glass and tables, as well as incorrect human scaling. These examples emphasize the enhanced capability of the fusion model to detect high-level inconsistencies beyond individual appearance, motion, or geometry assessments..

#### M. Clarification on Research Scope

L3DE focuses explicitly on 3D visual coherence, specifically assessing appearance, motion, and geometry, as these dimensions are fundamental prerequisites for realistic simulations. It is important to clarify that our method does not comprehensively evaluate all the aspects related to

world simulation such as complex interactions (e.g., accurate physics-based interactions, fluid dynamics). Thus, L3DE provides a targeted assessment specifically related to foundational 3D visual coherence, forming a necessary basis for further advancements towards comprehensive world simulation.

#### N. Limitations

Although our study takes a very first step to assess the 3D simulation capabilities of AI-generated videos, several challenges remain: 1.) Dataset Size and Diversity: Currently, we use 160000 video clips to train L3DE model. However, the real-world patterns are very complicated and training on more videos will provide a more general and robust evaluation tool. 2.) Limited Generative Video Length: Due to the constraints of current open-source generative video models, which produce relatively short videos, it is challenging to evaluate long-range coherence and object permanence of the future generative videos. To address these limitations, we plan to continually update L3DE to adapt to the generative videos in the future, and further explore its potential in broader data-centric research [28, 59–61, 77, 78].

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

SyntheticSyntheticSyntheticRealRealSyntheticSyntheticRealRealReal

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

- Figure A12. Visualization of randomly sampled paired videos. The images on the left are the image prompts for the generated videos and their first frame. The remaining images show the subsequent frames of the real videos and the generated videos.

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

[Figure 204]

Reference Frame Inconsistent Region Analyzed Frame Inconsistent Region Detected Artifacts

- Figure A13. Appearance Grad-CAM results of L3DE. For the first video, appearance Grad-CAM detect unstable scene appearances in the connecting regions between the two scenes, such as objects suddenly appearing or disappearing. For the second video, Appearance Grad-CAM detect regions with inconsistent scene appearance styles. Specifically, the first half of the video depicts a realistic cowshed, but it generates cartoon-style cows inside. For the third video, Appearance Grad-CAM detect a sudden change in the texture of the wooden board and food in the video. More specifically, the color of the wooden board and the food in the marked area change significantly between consecutive frames.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Reference Frame Inconsistent Region Analyzed Frame Inconsistent Region Detected Artifacts

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

- Figure A14. Motion Grad-CAM results of L3DE. For the first video, Motion Grad-CAM detect unnatural motion patterns of the wolves. In the video, the movement of the wolves in the marked area is accompanied by an appearance-disappearance phenomenon, which does not conform to real-world motion patterns. For the second video, Grad-CAM detect regions where the wolf exhibits unnatural motion. Specifically, a wolf that appears with normal four legs in the reference frame experiences sudden disappearance of its legs when moving in subsequent frames. Such motion patterns are inconsistent with real-world ones. For the third video, Grad-CAM detect a sudden unnatural ’compression’ motion in the bus, which remain stationary in the first half of the video. This does not conform to real-world motion laws.

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

Reference Frame Inconsistent Region Analyzed Frame Inconsistent Region Detected Artifacts

- Figure A15. Geometry Grad-CAM results of L3DE. For the first video, Grad-CAM detect inconsistent geometric structures in the person’s feet, thereby highlighting the corresponding regions. Specifically, the foot region in the analyzed frame differs from that in the reference frame, exhibiting noticeable blurring and distortion. Such degradation of geometric structure does not conform to real-world patterns. For the second video, Grad-CAM detect an abnormal geometric change in the hammer. In the first half of the video, the elderly person holds a single hammer, but in the subsequent frame, the geometry of the hammer suddenly exhibits a ’cloning’ effect, splitting into two. Such geometric inconsistency does not conform to real-world geometry rules. For the third video, Grad-CAM detect regions where a chair suddenly appears in the video. Such sudden changes in scene geometry are inconsistent with real-world patterns.

