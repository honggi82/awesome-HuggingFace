## DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis

Cheng-You Lu1 , Yi-Shan Hung2* , Wei-Ling Chi3* , Hao-Ping Wang3* , Charlie Li-Ting Tsai1* , Yu-Cheng Chang1 , Yu-Lun Liu3 , Thomas Do1 ,

and Chin-Teng Lin1

# arXiv:2604.13416v2[cs.CV]18Jun2026

1 University of Technology Sydney 2 University of Sydney 3 National Yang Ming Chiao Tung University

[Figure 1]

[Figure 2]

###### (a) DF3DV-1K

[Figure 3]

- * 1048 Indoor/Outdoor Scenes Manual capture over 9 months
- * Cluttered & Clean Image per Scene
- * 128 Distractor Types
- * Full Benchmark across 10 Methods ~2.03 L4/L40 GPU Years

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

[Figure 10]

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

[Figure 14]

| |
|---|

[Figure 15]

### … …

… … …

[Figure 16]

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

Cluttered Images Distractor-Free Radiance Fields Synthesis Clean Images

###### (b) DF Dataset Comparison (c) DF3DV-1K Full Benchmark across 10 Methods

[Figure 20]

PSNR↑ LPIPS↓ SSIM↑

DF3DV-1K RobustNeRF On-the-go

No. of

Scenes* Difficulty

No. of Distractor

(3DGS LPIPS)

| | |
|---|---|
| | |

Types*

No. of

No. of Themes*

Outdoor Scenes*

No. of

No. of Images*

Indoor

No. of Scenes* Benchmarkable

Scenes*

Fig. 1: Features of the DF3DV-1K dataset and benchmark. (a) DF3DV-1K Dataset. We introduce DF3DV-1K, a large-scale distractor-free dataset comprising 1,048 manually captured indoor and outdoor scenes, with clean and cluttered images per scene spanning 128 distractor types. (b) Distractor-Free Dataset Comparison. * denotes log scale in the normalized radar chart. We compare DF3DV-1K with public datasets [67, 69], showing a larger scale (e.g., ∼100× more scenes), broader scene diversity (e.g., ∼10× more distractor types), and increased difficulty, reflected by higher 3DGS [30] LPIPS. (c) DF3DV-1K Benchmark. A comprehensive benchmark across nine recent distractor-free radiance field methods [20,33,36,47,54,68,83,86] and 3DGS [30] reveals varying levels of robustness to distractors.

Abstract. Advances in radiance fields have enabled photorealistic novel view synthesis. In several domains, large-scale real-world datasets have

* Equal contribution.

been developed to support comprehensive benchmarking and to facilitate progress beyond scene-specific reconstruction. However, for distractorfree radiance fields, a large-scale dataset with clean and cluttered images per scene remains lacking, limiting the development. To address this gap, we introduce DF3DV-1K, a large-scale real-world dataset comprising 1,048 scenes, each providing clean and cluttered image sets for benchmarking. In total, the dataset contains 89,924 images captured using consumer cameras to mimic casual capture, spanning 128 distractor types and 161 scene themes across indoor and outdoor environments. A curated subset of 41 scenes, DF3DV-41, is systematically designed to evaluate the robustness of distractor-free radiance field methods under challenging scenarios. Using DF3DV-1K, we benchmark nine recent distractor-free radiance field methods and 3D Gaussian Splatting, identifying the most robust methods and the most challenging scenarios. Beyond benchmarking, we demonstrate an application of DF3DV-1K by fine-tuning a diffusion-based 2D enhancer to improve radiance field methods, achieving average improvements of 0.96 dB PSNR and 0.057 LPIPS on the held-out set (e.g., DF3DV-41) and the On-the-go dataset. We hope DF3DV-1K facilitates the development of distractor-free vision and promotes progress beyond scene-specific approaches. The dataset and leaderboard are available at https://johnnylu305.github.io/df3dv1k_web/.

Keywords: Dataset · Benchmark · Radiance Field

##### 1 Introduction Radiance fields [30,56] have shown photorealistic novel view synthesis abilities.

Their variants such as in-the-wild (with large temporal gap) [3,11,12,15,19,33, 36, 40, 55, 57, 84, 86–88, 92, 96, 101] and dynamic radiance fields [8, 18, 39, 43, 73, 74, 81, 85, 90, 93, 94] have been introduced together with comprehensive benchmarks and large-scale datasets [7, 25, 29, 45, 48, 49, 52, 65, 79, 79, 99, 106] to identify remaining limitations and support further research. Despite recent advances, distractor-free radiance field methods, which aim to synthesize clean novel views from images containing visual distractors commonly observed in short-span casual image capture (see Fig. 2), still lack a large-scale, challenging dataset and benchmark with clean and cluttered images per scene (see Tab. 1). This absence leads to two major issues: (1) it becomes dif-

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- (a) Distractor-free task
- (b) In-the-wild task (with large temporal gap)
- (c) Dynamic task

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

|[Figure 29]|[Figure 30]|[Figure 31]|
|---|---|---|

[Figure 32]

Fig. 2: Radiance field variants. (a) Distractor-free tasks [67] use casually captured images within a short period. (b) In-the-wild tasks [27,55], with large temporal gaps, often target images collected across seasons. (c) Dynamic tasks [93] assume densely captured sequential data.

- Table 1: Comparison of the 3D datasets. We compare datasets based on the number of scenes, image resolution, capture density, scale of visual changes, camera intrinsic consistency within each scene, availability of clean and cluttered images, and indoor/outdoor coverage. Despite recent progress, distractor-free radiance field research still lacks a large-scale real-world dataset with clean and cluttered images per scene, limiting further development in this area.

Dataset Scene Resolution Capture Change Scale Intrinsics Clean Clutter Indoor Outdoor Vanilla

DTU [25] 124 2MP sparse — shared ✓ ✗ ✓ ✗ RealEstate10K [106] 10K 720p HD dense — shared ✓ ✗ ✓ ✗ ACID [49] 13K — dense — shared ✓ ✗ ✗ ✓ ScanNet++ [99] 1K ∼8K UHD dense — shared ✓ ✗ ✓ ✗ DL3DV-10K [48] 10K 4K UHD dense — shared ✓ ✗ ✓ ✓

In-the-wild (with large temporal gap)

Phototourism [27] 25 — sparse global multiple ✓ ✓ ✗ ✓ MegaScenes [79] 430K — sparse global multiple ✗ ✓ ✓ ✓

Dynamic

- D2NeRF [93] 5 <qHD dense local shared ✓ ✓ ✓ ✗ SEED4D [29] 10K — dense local shared ✗ ✓ ✗ ✓

Distractor-free

RobustNeRF [69] 5 12MP sparse local shared ✓ ✓ ✓ ✗ On-the-go [67] 12 12MP sparse local shared ✓ ✓ ✓ ✓ DF3DV-1K (Ours) 1K ∼12MP sparse local shared ✓ ✓ ✓ ✓

ficult to identify the strengths and limitations of recent approaches, and (2) progress from scene-specific methods toward generalizable solutions is hindered.

In fact, the first issue can be observed in recent works [20, 36, 47, 83]. For example, Fig. 10 in [47], Fig. 7 in [20], and Fig. 4 in [36] reveal subtle background differences often visible only under zoomed-in views. In other cases (see Fig. 4 in [83]), evaluations rely on cross-domain datasets supporting only qualitative analysis. The number of distractor-free radiance field papers targeting casually captured images doubled in 2025 (see Fig. 3), yet no systematically designed, large-scale public dataset or benchmark has emerged to keep pace with this rapid growth. These observations do not indicate shortcomings of existing benchmarks but rather reflect the rapid progress of the field, highlighting the need for a new, challenging, and systematic benchmark

? Data. & Bench.

On-the-go Data. & Bench.

[Figure 33]

[Figure 34]

[Figure 35]

RobustNeRF Data. & Bench.

1st Gen. DF-3DGS

Fig. 3: Number of distractor-free radiance field papers. The first distractor-free radiance field method [69], targeting images captured over short time spans, was introduced in 2023 together with a benchmark. The research area rapidly gained attention in 2024 with the release of the On-the-go benchmark [67]. Although the number of works doubled in 2025, a public, large-scale, challenging dataset and benchmark systematically designed for this area are still lacking.

- (see Fig. 4) better aligned with current method capabilities, which our work aims to address. Similarly, the second issue

- Table 2: Comparison of real-world distractor-free radiance field datasets. We compare datasets based on the number of scenes, indoor and outdoor environments, distractor types, themes, images, and image resolution. DF3DV-1K is a large-scale, diverse real-world dataset, providing clean and cluttered images for each scene.

Dataset Scene Indoor Outdoor Distractor Theme Image Resolution Clean Not published yet (2026 June) T-3DGS [54] 5 5 0 >2 4 — — ✓ Not released yet (2026 June)

Entity-NeRF [60] 3 0 3 >3 3 78 — ✗ UniVerse [9] 6 — >3 >3 >3 — — ✓ Wild3A [40] 3 2 1 >3 3 — — ✓ StaticNeRF [35] 8 8 0 — 2 — — ✓

Public

DeGauss [83] 4 4 0 >5 4 >11K — ✗ RobustNeRF [69] 5 5 0 4 4 1.6K 12MP ✓ On-the-go [67] 12 2 10 14 10 2.2K 12MP ✓ D-RE10K-iPhone [13] 50 50 0 >2 4 1.8K — ✓ RealX3D [50] 55 55 0 — — — 33MP ✓ DF3DV-1K (Ours) 1K 726 322 128 161 89.9K ∼12MP ✓

can be observed by examining the development history of distractor-free radiance fields. The first distractor-free radiance field method [69], explicitly targeting images captured over short time spans, was introduced in 2023 together with a benchmark, followed by several subsequent approaches [11,33,54,60,67,80,87]. Among them, [67] introduced a new paradigm and benchmark that influenced later research. Subsequently, many methods emerged [3,9,19,20,32,36,40,41,44, 47,68,83,86], yet most require per-scene optimization. Only recently, in 2026, a generalizable radiance field model fine-tuned on synthetic and small-scale realworld datasets [67] has been explored in this setting [2]. This delay highlights the need for a large-scale real-world dataset with clean and cluttered images per scene to enable research beyond scene-specific approaches.

###### To facilitate progress, we present DF3DV-1K (Distractor-Free 3D Vision

1K), a newly collected dataset designed for distractor-free novel view synthesis. The dataset comprises clean and cluttered images for each scene. In total, DF3DV-1K contains 1,048 indoor and outdoor scenes spanning 128 distractor types and 161 scene themes, with 89,924 images overall. To support and challenge distractor-free radiance field research focusing on casually captured images, we mimic real-world capture conditions by systematically collecting data using consumer cameras (see Figs. 4 and 6). DF3DV-1K represents a large-scale, real-world, and diverse dataset with clean and cluttered images per scene for this research area (see Tab. 2). This extensive effort establishes a large-scale benchmark for distractor-free novel view synthesis. Using this benchmark, we evaluate nine recent open-source distractor-free radiance field methods [20,33,36,47,54,68,83,86] together with 3DGS [30] across all 1,048 scenes. The large-scale evaluation identifies AsymGS [36] and RobustSplat [20] as the most robust methods, followed by OCSplats [47] and DeGauss [83] as the secondbest methods (see Tab. 3 and Fig. S.16). Interestingly, the ranking follows the

|[Figure 36]|[Figure 37]|[Figure 38]|
|---|---|---|
| | |[Figure 39]|

|[Figure 40]|[Figure 41]|[Figure 42]|
|---|---|---|
| | |[Figure 43]|

|[Figure 44]|[Figure 45]|[Figure 46]|
|---|---|---|
| | |[Figure 47]|

Fluid distractors: Distractors (e.g., water) exhibit non-rigid, irregular, or scattered visual patterns.

Frontal occlusion distractors: The distractor (e.g., fingers) occludes the view from the front.

###### Color-similar distractors: The distractor has

a color similar to the static parts.

|[Figure 48]|[Figure 49]|[Figure 50]|
|---|---|---|
| | |[Figure 51]|

|[Figure 52]|[Figure 53]|[Figure 54]|
|---|---|---|
| | |[Figure 55]|

|[Figure 56]|[Figure 57]|[Figure 58]|
|---|---|---|
| | |[Figure 59]|

Local air distractors: Air distractors (e.g., smoke)

Large-scale distractors: Distractors that are large in scale.

Highly reflective distractors: Distractors (e.g., CD) that exhibit strong specular reflections.

introduce transient, semi-transparent structures.

|[Figure 60]|[Figure 61]|[Figure 62]|
|---|---|---|
| | |[Figure 63]|

|[Figure 64]|[Figure 65]|[Figure 66]|
|---|---|---|
| | |[Figure 67]|

|[Figure 68]|[Figure 69]|[Figure 70]|
|---|---|---|
| | |[Figure 71]|

Semantically similar distractors: The distractor is semantically similar to the static parts.

###### Local appearance distractors: Distractors (e.g.,

###### Semi-transparent distractors: Distractors (e.g.,

lighting changes) that alter surface appearance without changing geometry.

windows) cause foreground–background mixing.

|[Figure 72]|[Figure 73]|[Figure 74]|
|---|---|---|
| | |[Figure 75]|

|[Figure 76]|[Figure 77]|[Figure 78]|
|---|---|---|
| | |[Figure 79]|

|[Figure 80]|[Figure 81]|[Figure 82]|
|---|---|---|
| | |[Figure 83]|

###### Semi-transient distractors: Distractors (e.g.,

Shadow distractors: Shadows cause appearance changes without altering the geometric structure.

###### Slow-motion distractors: The distractors (e.g.,

rolling shutter) start moving from a static state.

train) move slowly over time.

|[Figure 84]|[Figure 85]|[Figure 86]|
|---|---|---|
| | |[Figure 87]|

|[Figure 88]<br><br>[Figure 89]|[Figure 90]<br><br>[Figure 91]|[Figure 92]<br><br>[Figure 93]|
|---|---|---|
| | |[Figure 94]<br><br>[Figure 95]|

|[Figure 96]|[Figure 97]|[Figure 98]|
|---|---|---|
| | |[Figure 99]|

Common distractors as static parts: Common distractors (e.g., passer-by) appear as static parts.

Daily scenes: Dynamic interactions and object manipulations in everyday scenes.

Various distractors: Different distractors appear across views.

[Figure 100]

|[Figure 101]|[Figure 102]|[Figure 103]|
|---|---|---|
| | |[Figure 104]|

|[Figure 105]<br><br>[Figure 106]|[Figure 107]<br><br>[Figure 108]|[Figure 109]|
|---|---|---|
| | |[Figure 110]|

Nighttime scenes: Low-light conditions and

###### Other distractors/scenes: Regular large-scale

###### Other distractors/scenes: Regular scenes with

artificial illumination introduce noise and reduce the visibility of distractors.

scenes with common distractors (e.g., passer-by).

common distractors (e.g., bag).

- Fig. 4: Samples of systematically designed scenarios in DF3DV-41. Each example, where green and red boxes denote clean and cluttered images, respectively, illustrates a carefully designed scenario with curated distractor types or scene conditions, highlighting the benchmark’s diversity. Motivated by the observations of the potential limitations of prior methods, these 17 scenarios systematically evaluate the robustness of distractor-free 3D reconstruction methods under challenging conditions.

publication timeline, particularly in PSNR and SSIM, reflecting a consistent and reasonable progression in performance as more recent methods advance the field. These methods are further benchmarked on DF3DV-41, a subset of DF3DV-1K consisting of 41 systematically designed capture scenes covering 17 distractor and scene scenarios (see Fig. 4 and Figs. S.5 and S.6). This analysis identifies the most challenging cases (e.g., semantically similar and fluid distractors and nighttime scenes) and shows that AsymGS [36], RobustSplat [20], and OCSplats [47] remain comparatively robust, while other methods degrade under certain scenarios (see Tabs. S.2 to S.4). In addition, the results in Fig. 9 and Figs. S.1 to S.4 reveal that existing approaches, while demonstrating promising performance on prior datasets, still leave room for improvement in challenging scenarios. Finally, to promote progress beyond scene-specific methods, we apply DIFIX [91], a 2D enhancement framework for sparse-view radiance fields, to distractor-free radi-

Per-Image Performance Distribution of 3DGS

Numberofimages(logscale)

DF3DV-1K

On-the-go

RobustNeRF

| |
|---|

| |
|---|

| |
|---|

- 100

- 101

- 102

- 103

- 104

4

8

12 12 16 16 20 20 24 24 28 28 32 32 36 36 40

0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4 0.5 0.5 0.6 0.6 0.7 0.7 0.8 0.8 0.9 0.9 1.0

0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4 0.5 0.5 0.6 0.6 0.7 0.7 0.8 0.8 0.9 0.9 1.0

0

4

8

0.0

0.0

PSNR

SSIM

LPIPS

- Fig. 5: Dataset difficulty analysis via per-image performance distributions. DF3DV-1K shows wider LPIPS and SSIM distributions, indicating greater diversity and increased difficulty. In contrast, RobustNeRF [69] is relatively clean, where a vanilla 3DGS [30] frequently achieves PSNR values exceeding 30 dB.

ance fields by fine-tuning on DF3DV-1K* (e.g., 1,007 scenes) and evaluating on DF3DV-41 and the On-the-go dataset [67]. The resulting model, referred to as

- DI2FIX (Distractor-Free DIFIX), achieves improvements of 0.96 dB PSNR and 0.057 LPIPS. The contributions are: (1) DF3DV-1K, a large-scale, real-world, and diverse dataset for distractor-free novel view synthesis containing 1,048 indoor and outdoor scenes with clean and cluttered images spanning 128 distractor types and 161 scene themes. (2) A comprehensive benchmark evaluating nine recent distractor-free radiance field methods together with 3DGS, enabling systematic robustness analysis across methods and scenarios. (3) DI2FIX, an enhancer that leverages large-scale data to improve distractor-free radiance field rendering quality, achieving gains of 0.96 dB PSNR and a 0.057 reduction in LPIPS.

#### 2 Related Work

###### 2.1 Radiance Fields

Radiance Fields [30, 56] are widely used for novel view synthesis due to their photorealistic rendering capabilities. One line extends radiance fields from scenespecific to generalizable models [2,10,14,16,22,23,26,28,82,95,100,103,105,108]. Another line focuses on dynamic radiance fields [8, 18, 73, 81, 83, 93], enabling novel view synthesis for 4D scenes. A separate line investigates in-the-wild radiance fields with large temporal gaps [3, 11, 12, 15, 19, 33, 36, 40, 55, 57, 84, 86– 88,92,96,101], which typically aim to render novel views from images captured across seasons. Among the variants, distractor-free radiance fields [2, 3, 9, 11, 19, 20, 32, 33, 35, 36, 40, 41, 44, 47, 54, 60, 61, 67–69, 77, 80, 83, 86, 87, 102] demonstrate the ability to synthesize clean novel views from casually captured images collected over short time spans4. Although the boundaries between these lines [3,11,19,33,36,40,83,86,87] may overlap, we emphasize that each line typically targets a specific data domain (see Fig. 2). Most existing distractor-free

4 As this survey focuses on general scenarios, multi-modal [37,63,66], task-specific [17, 78,89,104], and domain-specific [1,24,42,51,75,107] approaches are excluded.

###### (a) Scene Design and Capture

Scene Design

Controllable Capture

Data Review

- Tool-Assisted Data Verification
- • instant-ngp

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- • Pass
- • Fail

• Clean capture • Cluttered capture

- • Theme: kitchen
- • Distractor: vegetable, ...
- • Coverage: 360°
- • Viewpoint: landscape
- • #Clean: 50
- • #Clutter: 50

(b) Data Curation

- • Device: iPhone 15
- • Resolution: 12MP
- • Image Format: JPEG
- • Exposure: auto
- • Focus: auto
- • Anti-shake: disabled

- • Remove low-quality images
- • Ensure clean images are free of distractors

Pose Estimation and Undistortion

[Figure 115]

- • COLMAP image undistortion

[Figure 116]

[Figure 117]

- • COLMAP pose estimation

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Capture Setup Non-Controllable Capture

• Capture • Clean

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Metadata Annotation

[Figure 128]

[Figure 129]

- • Theme: kitchen
- • Distractor: vegetable, ...
- • Environment: indoor

• Clutter

[Figure 130]

[Figure 131]

- Fig. 6: Overview of the data collection and curation pipeline. (a) Scene design and capture. We design scenes with predefined themes, distractor types, coverage (e.g., 180◦-360◦), viewpoint orientation (e.g., landscape or portrait), and the number of images per scene. Images are captured using consumer cameras (e.g., iPhone). Camera settings such as exposure and focus are set to automatic by default and adjusted only when necessary. Anti-shake settings are disabled to ensure consistent image resolution within each scene. For controllable setups, clean and cluttered scenes are captured separately. Otherwise, they are captured simultaneously and later separated by operators. (b) Data curation. Low-quality images (e.g., defocused samples) are removed through manual inspection by two experts per scene. Then, camera poses are jointly estimated using COLMAP [71, 72], followed by image undistortion. Next, scenes are reconstructed with instant-ngp [58] for verification. Failed cases are reprocessed with adjusted COLMAP [71, 72] parameters and discarded if reconstruction remains unsuccessful. Finally, metadata are manually annotated.

radiance field methods remain scene-specific and require per-scene optimization. Only recently, the first generalizable distractor-free 3DGS [2] was introduced by fine-tuning a pretrained generalizable model [14] on synthetic data and a smallscale real-world dataset [67]. This delayed emergence of generalizable methods underscores the need for a large-scale, real-world, distractor-free dataset to enable scalable training. Meanwhile, the rapid growth of distractor-free radiance field research (see Fig. 3) underscores the need for a large-scale, challenging dataset and benchmark for systematic evaluation across methods.

###### 2.2 Datasets for Distractor-Free Radiance Fields

RobustNeRF [69] introduces an indoor, tabletop-scale dataset consisting of five scenes spanning four themes and four distractor types. Each scene provides both clean and cluttered image sets, in which distractor objects are manually repositioned by operators between captures. Although it served as the first widely adopted dataset and benchmark, it has become less challenging for recent stateof-the-art methods, making it difficult to clearly distinguish among their performance differences. Another widely used dataset is the On-the-go dataset [67], which contains twelve scenes, including two indoor and ten outdoor scenes. The dataset spans ten scene themes and fourteen distractor types. Among these, six scenes provide clean images, enabling both qualitative and quantitative bench-

Scene Distribution by Resolution (W x H)

Scene Distribution by Month

Scene Distribution by Device Type

iPhone 15 iPhone 12

4032x3024 3024x4032 3056x4080 4000x1848 4080x3060 3060x4080 5712x4284 4080x3072 4000x3000

- 2025-05
- 2025-06
- 2025-07
- 2025-08
- 2025-09
- 2025-10
- 2025-11
- 2025-12 2026-01

Samsung Galaxy A15

Samsung Galaxy S22+ iPhone 16 iPhone 14

iPad Air Samsung Galaxy S7 iPhone 13 XREAL Beam Pro DJI Mini 3 Pro Samsung Galaxy Z Fold 5

0 1 2 3

0.0 0.5 1.0 1.5 2.0 2.5

0.0 0.5 1.0 1.5 2.0 2.5

Scene Count (log scale)

Scene Count (log scale)

Scene Count (log scale)

- Fig. 7: Distribution of DF3DV-1K by capture device, resolution, and month of data collection. The distributions highlight the diversity of acquisition settings.

marking. This dataset has been widely adopted because it extends the evaluation setting to outdoor environments and presents more challenging scenes

- (see Fig. 5). Despite being more challenging, as discussed in Sec. 1, it has also become saturated for recent methods, requiring many approaches to emphasize subtle visual differences in background regions to demonstrate performance gains. Recently, two amazing concurrent works, RealX3D [50] and DRE10K-iPhone [13], provide paired clean and cluttered images captured with rail-mounted professional cameras and sparse-view indoor settings, respectively. As shown in Fig. 1 and Tab. 2, DF3DV-1K stands out by providing a large-scale collection of data with clean and cluttered images per scene, greater diversity, support for both qualitative and quantitative benchmarking, and more challenging, systematically designed scenarios under casual capture settings.

#### 3 Data Acquisition and Curation

###### 3.1 Data Acquisition

Scene Design. The goal is to construct a large-scale dataset with clean and cluttered images per scene for distractor-free radiance fields. Operators (see Fig. 6) first predefine each scene by specifying the scene theme (e.g., kitchen), potential distractors (e.g., vegetables), viewpoint coverage ranging from 180◦ to 360◦, viewpoint orientation (e.g., landscape or portrait), and the approximate number of clean and cluttered images to be captured. The indoor and outdoor capture protocols follow prior works [67,69]. For indoor scenes, distractors are manually introduced. In contrast, for uncontrolled environments where distractors naturally exist, no or limited additional distractors are introduced.

Capture Setup. Operators select a device configured to capture ∼12MP JPEG images with automatic exposure and focus. Images are taken individually rather than in video mode, and anti-shake is disabled to maintain consistent resolution. Operators may adjust exposure or focus when needed to prevent defocusing.

Capture. For controllable scenes, operators capture clean and cluttered images separately. For uncontrollable scenes, both conditions are captured together and later manually separated. Operators are instructed to pause capture during windy conditions. This manual capture process closely mimics casual image acquisition and thereby aligns with our dataset design goals. Furthermore, col-

Scene Count by Clean Image Ratio

Scene Count by Cluttered Image Ratio

Scene Count by Total Image Count

Numberofscenes

Numberofscenes

Numberofscenes

300

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

200

100

0

120

140140+

40

60

80

100

20

20%

30%

40%

50%

60%

70%

80%

90% 90

20%

30%

40%

50%

60%

70%

80%

90% 90

100%

100%

10%

10%

0

20

40

60

80

100

120

- 0

0

10

20

30

40

50

60

70

80

10

20

30

40

50

60

70

80

Number of images

Image ratio per scene

Image ratio per scene

- Fig. 8: Scene count distribution. Clean images per scene skew slightly toward lower bins for efficient novel-view benchmarking, whereas cluttered images extend toward higher bins to capture diverse distractor conditions. Total images per scene remain comparable to typical non-sparse radiance field settings.

lecting data ourselves rather than sourcing images from existing data enables the customized design of challenging scenarios (see Fig. 4 and Figs. S.5 and S.6).

###### 3.2 Data Curation

Data Review. Operators manually remove low-quality images (e.g., motion blur in static regions) and ensure the clean image set contains no distractors (see Fig. 6). We allow distractors themselves to exhibit lower visual quality (see Fig. S.19), as such artifacts naturally occur in casual capture scenarios.

Pose Estimation and Undistortion. Following common practice [67,69], we use COLMAP [71,72] to jointly estimate camera poses from clean and cluttered images, leveraging reliable correspondences from clean views to improve robustness and enforce a shared camera coordinate system. We then use these camera parameters to undistort all images, producing data used in our benchmark.

Tool-Assisted Data Verification. We reconstruct each scene using instantngp [58] to verify data quality. Failure cases typically include missing geometry, degenerate solutions, or fragmented reconstructions. When failures occur, we adjust COLMAP [71,72] parameters (e.g., minimum inlier threshold, registration trials, and minimum model size) to improve pose registration and avoid degeneracy. Scenes that still fail after tuning are discarded. In practice, most scenes pass verification directly, with only a few requiring additional tuning.

Annotation. We annotate passed scenes with metadata that requires manual labeling, including the scene theme, distractor type, scenario, and environment.

###### 3.3 Data Statistics

Scale. DF3DV-1K contains 1,048 scenes (726 indoor, 322 outdoor) with clean and cluttered image sets, totaling 89,924 images (see Tab. 2). The clean image ratio (see Fig. 8) per scene skews slightly toward lower ratios for efficient benchmarking, while the cluttered image ratio shifts toward higher ratios to cover more diverse distractors. Overall, the total number of images per scene is centered around 50 images, comparable to commonly used radiance field datasets.

Semantic Diversity. DF3DV-1K covers 128 distractor types. The types span from common objects (e.g., cars) to less frequent or large-scale distractors (e.g.,

- Table 3: Benchmark results on RobustNeRF [69], On-the-go [67], DF3DV-

- 1K, and DF3DV-41. RobustNeRF [69] is the least challenging benchmark, where all methods achieve strong performance, with several methods exceeding 29 dB PSNR, comparable to results on clean benchmarks [34]. Compared with RobustNeRF [69], On-the-go [67] is more challenging due to the inclusion of outdoor scenes. DF3DV1K and DF3DV-41 are the most challenging benchmarks, featuring large-scale data with diverse distractor scenarios. On DF3DV-1K and DF3DV-41, AsymGS [36] and RobustSplat [20] are the most robust, followed by OCSplats [47] and DeGauss [83].

Method Venue

RobustNeRF [69] On-the-go [67] DF3DV-1K DF3DV-41

PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

3DGS [30] TOG’23 25.44 0.834 0.157 19.33 0.662 0.306 17.93 0.630 0.330 18.10 0.620 0.331 T-3DGS [54] arXiv 28.46 0.890 0.098 22.80 0.818 0.128 18.92 0.682 0.263 19.00 0.672 0.259 T-3DGS-TMR [54] arXiv 26.79 0.872 0.121 19.82 0.736 0.231 18.11 0.657 0.296 18.41 0.654 0.279 WildGaussians [33] NeurIPS’24 27.25 0.872 0.130 22.60 0.793 0.155 18.78 0.669 0.293 19.26 0.670 0.285 SLS [68] TOG’25 28.70 0.873 0.103 22.73 0.779 0.153 19.75 0.694 0.263 19.37 0.662 0.287 DeSplat [86] CVPR’25 28.57 0.880 0.096 22.74 0.805 0.121 19.26 0.681 0.249 19.31 0.662 0.248 DeGauss [83] ICCV’25 29.22 0.892 0.089 23.57 0.831 0.107 20.00 0.714 0.243 19.98 0.695 0.236 OCSplats [47] ICCV’25 28.91 0.884 0.088 22.98 0.817 0.108 20.02 0.710 0.247 19.84 0.689 0.249 RobustSplat [20] ICCV’25 29.19 0.889 0.089 23.09 0.825 0.112 20.13 0.714 0.226 19.95 0.696 0.232 AsymGS [36] NeurIPS’25 29.44 0.896 0.096 23.16 0.821 0.136 20.49 0.735 0.229 20.35 0.712 0.247

planes). In addition, DF3DV-41 (see Fig. 4) contains systematically designed distractor types (e.g., semantically similar distractors challenging feature-based methods) and scene themes. See Figs. S.17 and S.18 for detailed distributions.

Capture Diversity. DF3DV-1K is collected over nine months. The dataset includes data captured using 12 consumer cameras, ranging from iPhone to Samsung smartphones, covering nine different image resolutions (see Fig. 7).

- 4 Benchmarks & Experiments

###### 4.1 DF3DV-1K and DF3DV-41 Benchmarks

Benchmark Methods. We evaluate recent open-source distractor-free radiance field methods, including AsymGS [36], RobustSplat [20], OCSplats [47], DeGauss [83], SLS [68], DeSplat [86], WildGaussians [33], T-3DGS and T-3DGSTMR [54], together with 3DGS [30]. See Sec. S.2 for more details.

Most Robust Methods on Large-Scale Benchmarks. Tab. 3 shows that AsymGS [36] and RobustSplat [20] are the most robust methods on DF3DV-1K and DF3DV-41, followed by OCSplats [47] and DeGauss [83] as the second most robust. This trend aligns with the publication timeline, reflecting steady performance improvements as methods evolve and supporting benchmark reliability. Tabs. S.2 to S.4 present the relative improvements of each method compared to 3DGS [30] across scenarios. Among them, AsymGS [36], RobustSplat [20], and OCSplats [47] show stronger robustness, consistently outperforming 3DGS [30], while other methods degrade under specific scenarios. Fig. 9 and Figs. S.1 to S.4 show similar a trend, with AsymGS [36], RobustSplat [20], OCSplats [47], and DeGauss [83] performing relatively well across several scenarios.

###### DF3DV-1K and DF3DV-41 vs. Prior Benchmarks. Compared with the

[Figure 132]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Color-Similar

Distractors

GT

3DGS

WildGaussians

T-3DGS

SLS

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

AsymGS

DeSplat

OCSplats

RobustSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Similar

| |
|---|

| |
|---|

| |
|---|

Distractors

Semantically

WildGaussians

T-3DGS

3DGS

GT

SLS

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

| |
|---|

| |
|---|

| |
|---|

OCSplats

AsymGS

RobustSplat

DeSplat

DeGauss

- Fig. 9: Qualitative results of the radiance field methods on DF3DV-41. The benchmark introduces systematically challenging conditions that enable clear visual comparison across methods and support the evaluation of robustness differences. For instance, WildGaussian [33] is less robust to color-similar distractors and tends to blend the black distractors with the background, while DeSplat [86] produces a noisier background in the chess scene.

public benchmarks [67,69], DF3DV-1K and DF3DV-41 are larger in scale and more diverse in distractor types and scene themes, discouraging per-scene tuning, and posing greater challenges. Tab. 3 further confirms this trend. The RobustNeRF [69] benchmark appears less challenging, as improvements over 3DGS [30] are modest while many methods still achieve PSNR above 29 dB, comparable to 3DGS performance on clean-scene benchmarks [34]. In contrast, DF3DV-1K and DF3DV-41 are more challenging than On-the-go [67], exhibiting lower PSNR and SSIM, higher LPIPS, and smaller gains over 3DGS [30] across methods. Fig. S.16 shows the top-4 most robust methods on RobustNeRF [69], On-the-go [67], and DF3DV-1K. Rankings differ across datasets, underscoring the importance of large-scale and diverse benchmarking. Notably, rankings on DF3DV-1K roughly follow publication timelines, a trend less evident in prior benchmarks. Recent methods often rely on subtle background differences in qualitative comparisons on earlier benchmarks to highlight strengths or limitations (see Sec. 1). This is not a shortcoming of existing benchmarks, but rather reflects rapid methodological progress and the need for more challenging evaluations. Our benchmark addresses this gap by enabling “zoom-in no more” qualitative comparisons, where performance differences are directly observable, and providing significance analyses (see Sec. S.2).

Limitations of Current Methods. Tabs. S.2 to S.4 show the relative improvement of each method relative to 3DGS [30] across scenarios. The quantitative results show that semantically similar and fluid distractors are the most challenging distractor types, while nighttime scenes represent the most difficult scene type, as indicated by the lower 3DGS performance and smaller achievable

Table 4: Effect of enhancers. Vanilla is the rendering result of the methods [20, 30, 33, 36, 47, 54, 68, 83, 86]. DIFIX+RobustNeRF and DI2FIX are DIFIX [91] finetuned on RobustNeRF [69] and DF3DV-1K*, respectively. Results report mean performance relative to Vanilla on the On-thego [67] and DF3DV-41.

Table 5: Effect of training data scale and diversity on DI2FIX. We vary the number of DF3DV1K* training scenes from 250, 500, 750, to 1,007 and report the mean performance on the On-thego [67] and DF3DV-41. Performance consistently improves as the training set size increases, and begins to plateau at larger scales.

Table 6: Effect of training data degradation level on DI2FIX. We vary the LPIPS threshold used to select training pairs and report the mean performance on the On-thego [67] and DF3DV-41. Moderate thresholds yield the best performance, while overly strict or overly loose selection slightly reduces performance.

Method PSNR↑ SSIM↑ LPIPS↓

Method ≤LPIPS PSNR↑ SSIM↑ LPIPS↓

Method Scenes PSNR↑ SSIM↑ LPIPS↓

Vanilla 20.82 0.731 0.211 DIFIX [91] 20.16 0.663 0.223 DIFIX+RobustNeRF 20.54 0.695 0.203 DI2FIX (Ours) 21.78 0.737 0.154

0.1 21.55 0.734 0.166 0.3 21.64 0.737 0.160 0.5 21.78 0.737 0.154 0.7 21.70 0.734 0.161 0.9 21.73 0.731 0.165

250 21.42 0.723 0.169 500 21.57 0.731 0.163 750 21.69 0.736 0.159 1K 21.78 0.737 0.154

DI2FIX

DI2FIX

improvements. This difficulty arises because many methods rely on semantic features [31, 59, 64, 76] to identify distractors. When distractors share similar semantic meanings with static scene objects, distinguishing distractors becomes difficult (see Fig. 9 and Fig. S.2). For fluid distractors, many methods learn masks online to remove dynamic content. However, fluid phenomena (e.g., splashes) are often spatially scattered and semi-transparent, which causes masking strategies to inadvertently remove large valid scene or fail to fully suppress the distractors (see blending artifacts in Fig. S.1). For nighttime scenes, many approaches rely on fixed thresholds to filter distractors. Because these thresholds are typically tuned for daytime conditions, they become unreliable under low illumination, where distractors and static components are harder to distinguish (see Fig. S.4). In addition, distractor removal may occasionally affect view-dependent effects on static objects, although Sec. S.2 shows that the benefits generally outweigh this drawback.

###### 4.2 Beyond Scene-Specific Methods

Distractor-free radiance field methods show promising results but still exhibit limitations (see Fig. 9 and Figs. S.1 to S.4), motivating the question of whether performance can be improved by enhancing existing methods without model modifications or scene-specific tuning. We therefore conduct a pilot experiment demonstrating how DF3DV-1K facilitates generalizable solutions.

Experimental Details. We introduce DI2FIX (Distractor-Free DIFIX), a plug-and-play 2D enhancer built on DIFIX [91] to improve radiance field renderings. DIFIX [91] is a diffusion-based model [62,70] that enhances radiance field renderings under sparse input using a clean reference view and degraded target view. To adapt DIFIX [91] to distractor-free tasks, we replace the clean reference

1 4

[Figure 133]

| |
|---|

3.

| |
|---|

| |
|---|

Target (T-3DGS)

DI2FIX

DIFIX+RobustNeRF

GT

DIFIX

| |
|---|

| |
|---|

Target (DeGauss)

DI2FIX

DIFIX+RobustNeRF

DIFIX

GT

- Fig. 10: Qualitative results of enhancers. Leveraging DF3DV-1K*, a large-scale and diverse dataset, DI2FIX effectively removes distractor artifacts (e.g., dynamic chess pieces and vegetable artifacts) while inpainting occluded regions in target views.

[Figure 134]

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

| |
|---|

| |
|---|

GT

Target (RobustSplat)

DF3DV-250

DF3DV-500

DF3DV-750

DF3DV-1K

GT

Target (3DGS)

DF3DV-250

DF3DV-500

DF3DV-750

DF3DV-1K

- Fig. 11: Qualitative results of DI2FIX trained with different data scales. Increasing the amount and diversity of training data improves robustness. In particular, DI2FIX progressively removes distractor artifacts (e.g., the chess pieces) while avoiding incorrect modifications to static scene content (e.g., the animal toy).

view with a radiance field rendering (see Sec. S.3 for details). We construct paired training data by reconstructing scenes from cluttered images in DF3DV-1K* (excluding DF3DV-41) using radiance field methods [20,30,33,36,47,54,83,86] and rendering novel views at clean-image views. This yields 316,890 candidate pairs, from which those with LPIPS≤ γ are retained, where γ = 0.5 in the final setting. We then fine-tune DIFIX [91] into DI2FIX.

DI2FIX Results. Tab. 4 reports the performance of 2D enhancers, where DIFIX+Robust and DI2FIX denote DIFIX [91] fine-tuned on RobustNeRF [69] data and DF3DV-1K*, respectively. DI2FIX improves the average rendering quality of 3DGS [30] and distractor-free radiance-field methods, particularly in PSNR and LPIPS, achieving a 0.96 dB PSNR gain and a 0.057 reduction in LPIPS. In contrast, DIFIX [91] without domain-specific fine-tuning degrades performance on average, while DIFIX+RobustNeRF still results in notable performance drops. Tab. S.6 shows that DI2FIX consistently improves PSNR and reduces LPIPS for every method. The SSIM changes are relatively limited, which is unsurprising since DIFIX [91] does not use an SSIM loss, and the perceptual trade-off with respect to SSIM has been reported in prior studies [4, 5, 38, 46]. Benefiting from large-scale and diverse supervision, DI2FIX suppresses distractor artifacts and restores visual structures across several challenging scenarios, whereas DIFIX and DIFIX+RobustNeRF often retain distractors or mistakenly modify static components (see Fig. 10 and Figs. S.8 to S.13).

###### Importance of Dataset Scale and Diversity. We randomly split DF3DV-

[Figure 135]

| |
|---|

| |
|---|

| |
|---|

- Fig. 12: Qualitative results of DI2FIX trained using different LPIPS filtering thresholds. Overly strict thresholds (e.g., 0.1) exclude many challenging image pairs, making artifacts difficult to remove (e.g., floaters around the cutting board). Overly loose thresholds (e.g., 0.9) introduce excessive noisy training samples, which may lead to undesired modifications of scene content (e.g., disappearing game cards). A moderate threshold provides a better balance between data quality and diversity.

1K into training sets of 250, 500, and 750 scenes and fine-tune DIFIX [91] to obtain DI2FIX. Performance improves with larger training sets (see Tab. 5), and qualitative results (see Fig. 11) show that additional scenes help DI2FIX better identify distractors and avoid editing static regions.

Effect of Training Data Degradation Level. To analyze the effect, we select pairs with LPIPS ≤ γ, and fine-tune DIFIX [91] to obtain DI2FIX. Tab. 6 and Fig. 12 show that a moderate threshold promotes better performance. Stricter thresholds make the training set less diverse and challenging, weakening the model’s ability to handle distractors, while looser thresholds encourage unnecessary edits to non-distractor regions.

Out-of-Distribution Test. To evaluate the generalizability of DI2FIX to unseen radiance-field methods, we conduct a leave-one-method-out training and evaluation. Tab. S.7 shows stable out-of-distribution performance, with worstcase changes limited to 0.1 dB PSNR, 0.005 SSIM decrease, and a 0.009 LPIPS increase, indicating that DI2FIX generalizes well to unseen methods.

#### 5 Conclusion

We introduced DF3DV-1K, a large-scale real-world dataset and benchmark for distractor-free novel view synthesis. It contains 1,048 indoor and outdoor scenes with clean and cluttered images, spanning 161 scene themes and 128 distractor types, and includes DF3DV-41 as a systematically designed challenging subset for scenario-wise evaluation. We establish a comprehensive benchmark across 10 representative methods, enabling reliable large-scale comparisons and revealing remaining failure cases. Beyond benchmarking, we show that DF3DV-1K enables progress beyond scene-specific vision by fine-tuning DIFIX on DF3DV-1K* to obtain DI2FIX, a plug-and-play 2D enhancer for distractor-free radiance fields that delivers consistent improvements. We hope DF3DV-1K accelerates the development of more robust and generalizable distractor-free vision methods.

Limitations. DF3DV-1K is smaller than general datasets [48] due to higher capture costs and careful scenario design. Slight cloud movement may still occur in clean images, but it is minimal due to the short capture duration. DI2FIX

may fail when input views are severely corrupted or exhibit confirmation bias (see Fig. S.15). A multi-reference framework is a good future direction.

#### 6 Acknowledgements

This work was supported in part by the Australian Research Council (ARC) under discovery grant DP250103612 and DP260101395, ARC Research Hub for Human-Robot Teaming for Sustainable and Resilient Construction (ITRH) grant IH240100016, and Australian National Health and Medical Research Council (NHMRC) Ideas Grant APP2021183. We thank the NYCU Computational Photography Lab for insightful discussions related to this work, and Fu-Jung Liu, Li-Chen Liu, Su-Lan Liu, and Yu-Jun Huang for their assistance with data collection.

#### References

- 1. Bai, N., Yang, A., Chen, H., Du, C.: Satgs: Remote sensing novel view synthesis using multi-temporal satellite images with appearance-adaptive 3dgs. Remote Sensing 17(9), 1609 (2025) 6
- 2. Bao, Y., Liao, J., Huo, J., Gao, Y.: Distractor-free generalizable 3d gaussian splatting. In: The Fourteenth International Conference on Learning Representations

(2026) 4, 6, 7

- 3. Bao, Y., Tang, C., Wang, Y., Li, H.: Seg-wild: Interactive segmentation based on 3d gaussian splatting for unconstrained image collections. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 8567–8576 (2025) 2, 4, 6
- 4. Blau, Y., Michaeli, T.: The perception-distortion tradeoff. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6228–6237

(2018) 13, 27

- 5. Blau, Y., Michaeli, T.: Rethinking lossy compression: The rate-distortionperception tradeoff. In: International Conference on Machine Learning. pp. 675–

685. PMLR (2019) 13, 27

- 6. Bradski, G.: The opencv library. Dr. Dobb’s Journal: Software Tools for the Professional Programmer 25(11), 120–123 (2000) 24
- 7. Broxton, M., Flynn, J., Overbeck, R., Erickson, D., Hedman, P., Duvall, M., Dourgarian, J., Busch, J., Whalen, M., Debevec, P.: Immersive light field video with a layered mesh representation. ACM Transactions on Graphics (TOG) 39(4), 86–1 (2020) 2
- 8. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 130–141 (2023) 2, 6
- 9. Cao, J., Wu, H., Feng, Z., Bao, H., Zhou, X., Peng, S.: Universe: Unleashing the scene prior of video diffusion models for robust radiance field reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27031–27041 (2025) 4, 6
- 10. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19457–19467 (2024) 6

- 11. Chen, J., Qin, Y., Liu, L., Lu, J., Li, G.: Nerf-hugs: Improved neural radiance fields in non-static scenes using heuristics-guided segmentation. CVPR (2024) 2, 4, 6
- 12. Chen, X., Zhang, Q., Li, X., Chen, Y., Feng, Y., Wang, X., Wang, J.: Hallucinated neural radiance fields in the wild. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12943–12952 (2022) 2, 6
- 13. Chen, X., Zhou, W., Cheng, Z.: Wildrayzer: Self-supervised large view synthesis in dynamic environments. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2026) 4, 8
- 14. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In: European conference on computer vision. pp. 370–386. Springer (2024) 6, 7
- 15. Dahmani, H., Bennehar, M., Piasco, N., Roldao, L., Tsishkou, D.: Swag: Splatting in the wild images with appearance-conditioned gaussians. In: European Conference on Computer Vision. pp. 325–340. Springer (2024) 2, 6
- 16. Dey, A., Lu, C.Y., Comport, A.I., Sridhar, S., Lin, C.T., Martinet, J.: Hfgaussian: Learning generalizable gaussian human with integrated human features. IEEE Transactions on Artificial Intelligence (2025) 6
- 17. Du, S., Liu, J., Chen, Q., Chen, H.X., Mu, T.J., Yang, S.: Rge-gs: Reward-guided expansive driving scene reconstruction via diffusion priors. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 25756–25764

(2025) 6

- 18. Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A.: K-planes: Explicit radiance fields in space, time, and appearance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12479– 12488 (2023) 2, 6
- 19. Fu, C., Chen, G., Zhang, Y., Yao, K., Xiong, Y., Huang, C., Cui, S., Matsushita, Y., Cao, X.: Robustsplat++: Decoupling densification, dynamics, and illumination for in-the-wild 3dgs. arXiv preprint arXiv:2512.04815 (2025) 2, 4, 6
- 20. Fu, C., Zhang, Y., Yao, K., Chen, G., Xiong, Y., Huang, C., Cui, S., Cao, X.: Robustsplat: Decoupling densification and dynamics for transient-free 3dgs. In: ICCV (2025) 1, 3, 4, 5, 6, 10, 12, 13, 24, 25, 26, 27, 28, 30, 31, 32, 33, 40, 48, 49
- 21. Gatys, L.A., Ecker, A.S., Bethge, M.: A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576 (2015) 26
- 22. Hosseinzadeh, M., Chng, S.F., Xu, Y., Lucey, S., Reid, I., Garg, R.: G3splat: Geometrically consistent generalizable gaussian splatting. arXiv preprint arXiv:2512.17547 (2025) 6
- 23. Houchens, T., Lu, C.Y., Duggal, S., Fu, R., Sridhar, S.: Neuralodf: Learning omnidirectional distance fields for 3d shape representation. arXiv preprint arXiv:2206.05837 (2022) 6
- 24. Huang, X., Liu, X., Wan, Y., Zheng, Z., Zhang, B., Xiong, M., Pei, Y., Zhang, Y.: Skysplat: Generalizable 3d gaussian splatting from multi-temporal sparse satellite images. arXiv preprint arXiv:2508.09479 (2025) 6
- 25. Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 406–413 (2014) 2, 3
- 26. Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J., Zhao, F., et al.: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG) 44(6), 1–16 (2025) 6

- 27. Jin, Y., Mishkin, D., Mishchuk, A., Matas, J., Fua, P., Yi, K.M., Trulls, E.: Image matching across wide baselines: From paper to practice. International Journal of Computer Vision 129(2), 517–547 (2021) 2, 3
- 28. Kang, G., Nam, S., Yang, S., Sun, X., Khamis, S., Mohamed, A., Park, E.: ilrm: An iterative large 3d reconstruction model. arXiv preprint arXiv:2507.23277 (2025) 6
- 29. Kästingschäfer, M., Gieruc, T., Bernhard, S., Campbell, D., Insafutdinov, E., Najafli, E., Brox, T.: Seed4d: A synthetic ego-exo dynamic 4d data generator, driving dataset and benchmark. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 7752–7764. IEEE (2025) 2, 3
- 30. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023) 1, 2, 4, 6, 10, 11, 12, 13, 24, 25, 26, 27, 31, 32, 33, 40, 47, 48
- 31. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026

(2023) 12, 25

- 32. Kong, H., Yang, X., Wang, X.: Rogsplat: Robust gaussian splatting via generative priors. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 25735–25745 (2025) 4, 6
- 33. Kulhanek, J., Peng, S., Kukelova, Z., Pollefeys, M., Sattler, T.: WildGaussians: 3D gaussian splatting in the wild. In: Proceedings of the 38th International Conference on Neural Information Processing Systems (2024) 1, 2, 4, 6, 10, 11, 12, 13, 24, 26, 27, 31, 32, 33, 40, 48
- 34. Kulhanek, J., Sattler, T.: Nerfbaselines: Consistent and reproducible evaluation of novel view synthesis methods. In: Proceedings of the 39th International Conference on Neural Information Processing Systems (NeurIPS 2025) (2025) 10, 11
- 35. Lee, J., Yang, G., Ma, S., Cho, Y.: Freeze-frame with staticnerf: Uncertaintyguided nerf map reconstruction in dynamic scenes. IEEE Robotics and Automation Letters 11(1), 778–785 (2025) 4, 6
- 36. Li, C., Shi, Z., Lu, Y., He, W., Xu, X.: Robust neural rendering in the wild with asymmetric dual 3d gaussian splatting. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025) 1, 2, 3, 4, 5, 6, 10, 12, 13, 24, 25, 26, 27, 28, 30, 31, 32, 33, 39, 40, 48, 49
- 37. Li, D., Feng, J., Chen, J., Dong, W., Li, G., Shi, G., Jiao, L.: Egosplat: Openvocabulary egocentric scene understanding with language embedded 3d gaussian splatting. arXiv preprint arXiv:2503.11345 (2025) 6
- 38. Li, J., Cao, J., Guo, Y., Li, W., Zhang, Y.: One diffusion step to real-world superresolution via flow trajectory distillation. In: International Conference on Machine Learning. pp. 34044–34053. PMLR (2025) 13, 27
- 39. Li, L., Shen, Z., Wang, Z., Shen, L., Tan, P.: Streaming radiance fields for 3d video synthesis. Advances in Neural Information Processing Systems 35, 13485–13498

(2022) 2

- 40. Li, M., Zhai, S., Zhao, Z., Sun, L., Wang, X., Li, D., Liu, S., Wang, H.: Wild3a: Novel view synthesis from any dynamic images in seconds. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 7472–7480 (2025) 2, 4, 6
- 41. Li, R., Cheung, Y.m.: Modeling and identifying distractors with curriculum for robust 3d gaussian splatting. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 10122–10131 (2025) 4, 6

- 42. Li, Y., Wu, J., Zhao, L., Liu, P.: Derainnerf: 3d scene estimation with adhesive waterdrop removal. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 2787–2793. IEEE (2024) 6
- 43. Li, Z., Wang, Q., Cole, F., Tucker, R., Snavely, N.: Dynibar: Neural dynamic image-based rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4273–4284 (2023) 2
- 44. Lin, J., Gu, J., Fan, L., Wu, B., Lou, Y., Chen, R., Liu, L., Ye, J.: Hybridgs: Decoupling transients and statics with 2d and 3d gaussian splatting. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 788–797 (2025) 4, 6
- 45. Lin, K.E., Xiao, L., Liu, F., Yang, G., Ramamoorthi, R.: Deep 3d mask volume for view synthesis of dynamic scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1749–1758 (2021) 2
- 46. Lin, X., Yu, F., Hu, J., You, Z., Shi, W., Ren, J.S., Gu, J., Dong, C.: Harnessing diffusion-yielded score priors for image restoration. ACM Transactions on Graphics (TOG) 44(6), 1–21 (2025) 13, 27
- 47. Ling, H., Xu, X., Sun, Y., Sun, Q.: Ocsplats: Observation completeness quantification and label noise separation in 3dgs. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 25680–25689 (2025) 1, 3, 4, 5, 6, 10, 12, 13, 24, 25, 26, 27, 28, 30, 31, 32, 33, 40, 48, 49
- 48. Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al.: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22160–22169 (2024) 2, 3, 14
- 49. Liu, A., Tucker, R., Jampani, V., Makadia, A., Snavely, N., Kanazawa, A.: Infinite nature: Perpetual view generation of natural scenes from a single image. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14458–14467 (2021) 2, 3
- 50. Liu, S., Bao, C., Cui, Z., Liu, Y., Chu, X., Gu, L., Conde, M.V., Umagami, R., Hashimoto, T., Hu, Z., et al.: Realx3d: A physically-degraded 3d benchmark for multi-view visual restoration and reconstruction. arXiv preprint arXiv:2512.23437

(2025) 4, 8

- 51. Liu, S., Chen, X., Chen, H., Xu, Q., Li, M.: Deraings: Gaussian splatting for enhanced scene reconstruction in rainy environments. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 5558–5566 (2025) 6
- 52. Lu, C.Y., Zhou, P., Xing, A., Pokhariya, C., Dey, A., Shah, I.N., Mavidipalli, R., Hu, D., Comport, A.I., Chen, K., et al.: Diva-360: The dynamic visual dataset for immersive neural fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22466–22476 (2024) 2
- 53. Lu, C.Y., Zhuang, Z., Le, N.T.T., Xiao, D., Chang, Y.C., Do, T., Sridhar, S., Lin, C.T.: Hestia: Voxel-face-aware hierarchical next-best-view acquisition for efficient 3d reconstruction. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5302–5312 (2026) 28
- 54. Markin, A., Pryadilshchikov, V., Komarichev, A., Rakhimov, R., Wonka, P., Burnaev, E.: T-3dgs: Removing transient objects for 3d scene reconstruction. arXiv preprint arXiv:2412.00155 (2024) 1, 4, 6, 10, 12, 13, 24, 26, 27, 31, 32, 33, 39, 40, 48
- 55. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duckworth, D.: Nerf in the wild: Neural radiance fields for unconstrained photo collections. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7210–7219 (2021) 2, 6

- 56. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021) 2, 6
- 57. Mithun, N.C., Pham, T., Wang, Q., Southall, B., Minhas, K., Matei, B., Mandt, S., Samarasekera, S., Kumar, R.: Diffusion-guided gaussian splatting for largescale unconstrained 3d reconstruction and novel view synthesis. arXiv preprint arXiv:2504.01960 (2025) 2, 6
- 58. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41(4), 1–15 (2022) 7, 9, 23
- 59. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal (2024) 12, 25
- 60. Otonari, T., Ikehata, S., Aizawa, K.: Entity-nerf: Detecting and removing moving entities in urban scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20892–20901 (2024) 4, 6
- 61. Park, W., Nam, M., Kim, S., Jo, S., Lee, S.: Forestsplats: Deformable transient field for gaussian splatting in the wild. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 6978–6987 (2026) 6
- 62. Parmar, G., Park, T., Narasimhan, S., Zhu, J.Y.: One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036 (2024) 12, 26
- 63. Prabakaran, A., Shukla, P.: Semantic-guided 3d gaussian splatting for transient object removal. arXiv preprint arXiv:2602.15516 (2026) 6
- 64. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. In: The Thirteenth International Conference on Learning Representations (2025) 12, 25
- 65. Reizenstein, J., Shapovalov, R., Henzler, P., Sbordone, L., Labatut, P., Novotny, D.: Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10901–10911 (2021) 2
- 66. Rematas, K., Liu, A., Srinivasan, P.P., Barron, J.T., Tagliasacchi, A., Funkhouser, T., Ferrari, V.: Urban radiance fields. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12932–12942 (2022) 6
- 67. Ren, W., Zhu, Z., Sun, B., Chen, J., Pollefeys, M., Peng, S.: Nerf on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 29, 40, 47, 49
- 68. Sabour, S., Goli, L., Kopanas, G., Matthews, M., Lagun, D., Guibas, L., Jacobson, A., Fleet, D., Tagliasacchi, A.: Spotlesssplats: Ignoring distractors in 3d gaussian splatting. ACM Transactions on Graphics 44(2), 1–11 (2025) 1, 4, 6, 10, 12, 24, 26, 27, 31, 32, 33, 40, 48
- 69. Sabour, S., Vora, S., Duckworth, D., Krasin, I., Fleet, D.J., Tagliasacchi, A.: Robustnerf: Ignoring distractors with robust losses. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20626– 20636 (2023) 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 26, 29, 40, 49
- 70. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial diffusion distillation. In: European Conference on Computer Vision. pp. 87–103. Springer (2024) 12, 26

- 71. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016) 7, 9, 23, 24
- 72. Schönberger, J.L., Zheng, E., Pollefeys, M., Frahm, J.M.: Pixelwise view selection for unstructured multi-view stereo. In: European Conference on Computer Vision (ECCV) (2016) 7, 9, 23, 24
- 73. Song, L., Chen, A., Li, Z., Chen, Z., Chen, L., Yuan, J., Xu, Y., Geiger, A.: Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics 29(5), 2732–2742 (2023) 2, 6
- 74. Sun, D., Guan, H., Zhang, K., Xie, X., Zhou, S.K.: Sdd-4dgs: Static-dynamic aware decoupling in gaussian splatting for 4d scene reconstruction. arXiv preprint arXiv:2503.09332 (2025) 2
- 75. Tang, J., Gao, Y., Yang, D., Yan, L., Yue, Y., Yang, Y.: Dronesplat: 3d gaussian splatting for robust 3d reconstruction from in-the-wild drone imagery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 833–843 (2025) 6
- 76. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image diffusion. Advances in neural information processing systems 36, 1363–1389 (2023) 12, 25
- 77. Tang, Y., Xu, D., Hou, Y., Wang, Z., Jiang, M.: Nexussplats: Efficient 3d gaussian splatting in the wild. arXiv preprint arXiv:2411.14514 (2024) 6
- 78. Trevithick, A., Paiss, R., Henzler, P., Verbin, D., Wu, R., Alzayer, H., Gao, R., Poole, B., Barron, J.T., Holynski, A., et al.: Simvs: Simulating world inconsistencies for robust view synthesis. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16464–16474 (2025) 6
- 79. Tung, J., Chou, G., Cai, R., Yang, G., Zhang, K., Wetzstein, G., Hariharan, B., Snavely, N.: Megascenes: Scene-level view synthesis at scale. In: European conference on computer vision. pp. 197–214. Springer (2024) 2, 3
- 80. Ungermann, P., Ettenhofer, A., Nießner, M., Roessle, B.: Robust 3d gaussian splatting for novel view synthesis in presence of distractors. In: DAGM German Conference on Pattern Recognition. pp. 153–167. Springer (2024) 4, 6
- 81. Wang, F., Tan, S., Li, X., Tian, Z., Song, Y., Liu, H.: Mixed neural voxels for fast multi-view video synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19706–19716 (2023) 2, 6
- 82. Wang, Q., Wang, Z., Genova, K., Srinivasan, P.P., Zhou, H., Barron, J.T., MartinBrualla, R., Snavely, N., Funkhouser, T.: Ibrnet: Learning multi-view image-based rendering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4690–4699 (2021) 6
- 83. Wang, R., Lohmeyer, Q., Meboldt, M., Tang, S.: Degauss: Dynamic-static decomposition with gaussian splatting for distractor-free 3d reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6294–6303 (2025) 1, 3, 4, 6, 10, 12, 13, 24, 26, 27, 28, 30, 31, 32, 33, 40, 48, 49
- 84. Wang, S., Xu, H., Li, Y., Chen, J., Tan, G.: Ie-nerf: Exploring transient mask inpainting to enhance neural radiance fields in the wild. Neurocomputing 618, 129112 (2025) 2, 6
- 85. Wang, W., Shao, F., Li, L., Wang, Z., Xiao, J., Chen, L.: Uncertainty-aware 4d gaussian splatting for monocular occluded human rendering. arXiv preprint arXiv:2602.06343 (2026) 2
- 86. Wang, Y., Klasson, M., Turkulainen, M., Wang, S., Kannala, J., Solin, A.: Desplat: Decomposed gaussian splatting for distractor-free rendering. In: Proceedings of

- the Computer Vision and Pattern Recognition Conference. pp. 722–732 (2025) 1, 2, 4, 6, 10, 11, 12, 13, 24, 26, 27, 31, 32, 33, 40, 48
- 87. Wang, Y., Li, K., Chen, M., Wang, L., Zhou, S., Xue, K., Guo, Y.: Distractorfree novel view synthesis via exploiting memorization effect in optimization. In: European Conference on Computer Vision. pp. 477–493. Springer (2024) 2, 4, 6
- 88. Wang, Y., Wang, J., Qi, Y.: We-gs: An in-the-wild efficient 3d gaussian representation for unconstrained photo collections. arXiv preprint arXiv:2406.02407

(2024) 2, 6

- 89. Warburg*, F., Weber*, E., Tancik, M., HoÅ‚yÅ„ski, A., Kanazawa, A.: Nerfbusters: Removing ghostly artifacts from casually captured nerfs. In: International Conference on Computer Vision (ICCV) (2023) 6
- 90. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20310–20320 (2024) 2
- 91. Wu, J.Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M.Z., Fidler, S., Gojcic, Z., Ling, H.: Difix3d+: Improving 3d reconstructions with single-step diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 26024–26035 (2025) 5, 12, 13, 14, 26, 27, 40, 49
- 92. Wu, L., Zhang, T.: Wildsplatting: Unposed incremental 3d gaussian splatting reconstruction in the wild. In: Proceedings of the 2025 20th ACM SIGGRAPH International Conference on Virtual-Reality Continuum and its Applications in Industry. pp. 1–6 (2025) 2, 6
- 93. Wu, T., Zhong, F., Tagliasacchi, A., Cole, F., Oztireli, C.: Dˆ 2nerf: Selfsupervised decoupling of dynamic and static objects from a monocular video. Advances in neural information processing systems 35, 32653–32666 (2022) 2, 3, 6
- 94. Xiao, Y., Lin, Z., Lu, C., Zhai, D., Jiang, K., Zhao, W., Zhang, W., Jiang, J., Wang, H., Liu, X.: Vdegaussian: Video diffusion enhanced 4d gaussian splatting for dynamic urban scenes modeling. arXiv preprint arXiv:2508.02129 (2025) 2
- 95. Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A., Pollefeys, M.: Depthsplat: Connecting gaussian splatting and depth. In: CVPR (2025) 6
- 96. Xu, J., Mei, Y., Patel, V.: Wild-gs: Real-time novel view synthesis from unconstrained photo collections. Advances in Neural Information Processing Systems 37, 103334–103355 (2024) 2, 6
- 97. Yang, J., Lin, C.T.: Toward autonomous distributed clustering. IEEE Transactions on Emerging Topics in Computational Intelligence 9(2), 2065–2072 (2024) 29
- 98. Yang, J., Lu, C.Y., Wang, Z., Chen, H.T., Xu, G.K., Zhang, C., Dong, S., Liang, X., Jiang, B.: Multi-view clustering with granularity-aware pseudo supervision. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 27538–27546 (2026) 29
- 99. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023) 2, 3
- 100. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4578–4587 (2021) 6
- 101. Zhang, D., Wang, C., Wang, W., Li, P., Qin, M., Wang, H.: Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In: European Conference on Computer Vision. pp. 341–359. Springer (2024) 2, 6

- 102. Zheng, J., Zhu, Z., Bieri, V., Pollefeys, M., Peng, S., Armeni, I.: Wildgs-slam: Monocular gaussian splatting slam in dynamic environments. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11461–11471 (2025) 6
- 103. Zheng, S., Zhou, B., Shao, R., Liu, B., Zhang, S., Nie, L., Liu, Y.: Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19680–19690 (2024) 6
- 104. Zheng, W., Ou, L., He, J., Zhou, L., Yu, X., Wei, Y.: Up-slam: Adaptively structured gaussian slam with uncertainty prediction in dynamic environments. arXiv preprint arXiv:2505.22335 (2025) 6
- 105. Zhou, B., Zheng, S., Tu, H., Shao, R., Liu, B., Zhang, S., Nie, L., Liu, Y.: Gpsgaussian+: Generalizable pixel-wise 3d gaussian splatting for real-time humanscene rendering from sparse views. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025) 6
- 106. Zhou, T., Tucker, R., Flynn, J., Fyffe, G., Snavely, N.: Stereo magnification: Learning view synthesis using multiplane images. In: SIGGRAPH (2018) 2, 3
- 107. Zhu, C., Wan, R., Tang, Y., Shi, B.: Occlusion-free scene recovery via neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20722–20731 (2023) 6
- 108. Ziwen, C., Tan, H., Zhang, K., Bi, S., Luan, F., Hong, Y., Fuxin, L., Xu, Z.: Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4349–4359 (2025) 6

#### S.1 Author Statement

In this supplementary material, we provide detailed information on the experimental setup of the DF3DV-1K and DF3DV-41 benchmarks, a significance analysis of the top four methods, scenario-wise performance analyses including quantitative and qualitative results for each benchmarked radiance field method, a trade-off analysis of view-dependent static objects (see Sec. S.2), experimental details of DI2FIX, the performance of DI2FIX across different radiance field methods, out-of-distribution evaluations of DI2FIX, and failure cases of DI2FIX (see Sec. S.3). We also present all scenes in DF3DV-41, together with detailed descriptions of themes and distractor types (see Sec. S.4). Due to the page limitations of the main paper, we aim to closely connect the main paper and the supplementary material to facilitate a smoother reading experience.

Dataset Organization and File Structure. The dataset and leaderboard are available at https://johnnylu305.github.io/df3dv1k_web/. All data were collected in accordance with applicable local laws and regulations. The dataset is distributed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. The 1,007 scenes are divided into 25 folders, named from "0000" to "0024", where each folder contains up to 49 scene zip files. Each scene zip file follows the naming convention "series_numberscene_name" (e.g., "221225-Livingroom"). Every scene contains three subfolders: "series_number-scene_name-All", "series_number-scene_name-Clean", and "series_number-scene_name-Clutter". The "Clean" and "Clutter" folders each contain an "images" directory storing all clean and cluttered JPG images, respectively. The"series_number-scene_name-All" folder contains curated data prepared for distractor-free radiance field research. Specifically, it includes the following components:

- – images — stores the valid images selected from the "Clean" and "Clutter" folders. Cluttered images are named "clutter_xxx.JPG", while clean images are named "extra_xxx.JPG".
- – sparse — contains the COLMAP [71,72] sparse reconstruction results, including estimated camera poses, intrinsic parameters, and reconstructed sparse 3D points.
- – undistortion_images — stores undistorted images generated from the "images" folder after COLMAP [71,72] image undistortion.
- – undistortion_sparse — contains the corresponding COLMAP [71, 72] sparse reconstruction with undistorted camera parameters, where distortion coefficients have been removed.
- – instant-ngp [58] JSON files — camera parameters are additionally parsed into the official instant-ngp [58] JSON format, including "transforms.json", "transforms_clutter.json", and "transforms_extra.json", which contain data information for all images, cluttered images only, and clean images only, respectively.

The naming convention and directory organization of DF3DV-1K follow commonly adopted structures in distractor-free radiance field research, with minor

renaming for clarity (e.g., "sparse" → "undistortion_sparse" and "images" → "undistortion_images"). DF3DV-41 follows the same organization but is hosted in a separate folder. The annotation data are provided as JSON files for each scene. The total dataset size is approximately 2 TB. Overall, this design provides a user-friendly and standardized data layout that enables straightforward integration with existing radiance field training and evaluation pipelines.

#### S.2 DF3DV-1K and DF3DV-41 Benchmarks

In this section, we provide additional experimental details, significance analysis, scenario-wise qualitative and quantitative results, and a trade-off analysis of view-dependent static objects. We evaluate nine recent open-source distractorfree radiance field methods, including AsymGS [36], RobustSplat [20], OCSplats [47], DeGauss [83], SLS [68], DeSplat [86], WildGaussians [33], T-3DGS and T-3DGSTMR [54], together with 3DGS [30], on the DF3DV-1K and DF3DV-41 benchmarks. The goal is to identify the most robust methods under challenging scenarios and the most challenging scenarios in the benchmarks.

Experimental Details. Our evaluation is fully based on the official implementations of distractor-free radiance field methods [20,33,36,47,54,68,83,86], with the following minimal modifications to ensure a fair comparison. Some methods perform image undistortion on-the-fly using OpenCV [6], adopt scenedependent parameter settings, or downsample images prior to undistortion. To ensure a fair comparison, we standardize the data preprocessing pipeline across methods and the training configuration across all scenes. Specifically, images are first undistorted offline using COLMAP [71,72], followed by 8× downsampling after undistortion. The same parameter configuration (defaulting to the On-thego setting in the official implementation, when applicable) is used for all scenes to avoid advantages from scene-specific tuning. We further observe that some methods, such as DeGauss [83], may occasionally become unstable and crash on a small subset of scenes. In such cases, we slightly reduce the learning rate to stabilize optimization and retrain the affected scenes.

Significance analysis. We report pairwise significance comparisons among the top four methods for PSNR, SSIM, and LPIPS in Tab. S.1. For each metric block, the upper triangular entries show the paired t-test p-values between pairs of methods across the evaluated scenes, while the lower triangular entries show the corresponding average performance margins. Smaller p-values indicate stronger statistical evidence of a performance difference between two methods. Colors encode the ranking of pairwise differences within each metric. The consistent patterns between the upper triangular p-values and the lower triangular performance margins suggest that the observed method rankings are statistically meaningful, highlighting the effectiveness of DF3DV-1K as a benchmark

Scenario-wise Performance. DF3DV-41 includes 17 systematically designed

challenging scenarios. Details of these 41 scenes are provided in Figs. S.5 and S.6. In addition, scenario-wise improvements in PSNR, SSIM, and LPIPS of each benchmark method relative to 3D Gaussian Splatting (3DGS) [30] are shown in Tabs. S.2 to S.4. From the quantitative results per scenario, we observe that AsymGS [36], RobustSplat [20], and OCSplats [47] perform robustly across all scenarios, achieving consistent improvements across the benchmark. In contrast, other methods may fail under specific scenarios, occasionally underperforming compared to 3DGS [30].

Figs. S.1 to S.4 show that the challenging scenarios in DF3DV-41 enable clearly visible distinctions between benchmark methods. We hope that this challenging dataset and benchmark facilitate “zoom-in no more” qualitative comparisons, allowing recent state-of-the-art methods to be compared directly without relying on subtle background zoom-ins to highlight differences.

We identify semantically similar distractors, fluid distractors, and nighttime scenes as the most challenging scenarios in DF3DV-41, as reflected by their low 3DGS performance and the relatively limited improvements achieved by distractor-free radiance field methods compared to other scenarios. These challenges are largely related to the reliance of many existing approaches on semantic features [31,59,64,76] for distractor identification. When distractors share similar semantic meanings with static scene objects, distinguishing true distractors from valid scene content becomes inherently ambiguous. For instance, in the chess scene shown in Fig. S.2, chess pieces that should ideally be removed as distractors remain partially preserved in the reconstructed results across multiple methods, indicating that semantic similarity can mislead distractor detection. For fluid distractors, many approaches learn masks online to suppress dynamic content during training. However, fluid phenomena (e.g., splashes or water sprays) are typically spatially dispersed and semi-transparent, which violates the implicit assumption of coherent object boundaries. As a result, masking strategies may either remove valid scene regions or fail to completely suppress the distractors, producing noticeable blending artifacts. This behavior can be clearly observed in the washing-car scene in Fig. S.1, where artifacts introduced by fluid distractors spread across large portions of the reconstructed scene. For nighttime scenes, many approaches rely on fixed thresholds or appearance-based heuristics to filter distractors. Because these thresholds are commonly tuned under daytime illumination, their reliability degrades under low-light conditions, where distractors and static scene components exhibit reduced contrast and weaker semantic cues, making reliable separation significantly more difficult (see Fig. S.4).

View-dependent Static Objects Tradeoff DF3DV-1K contains 388 scenes with strong view-dependent static objects, enabling an investigation of the tradeoff between preserving view-dependent static objects and removing distractors. We further select 18 DF3DV-41 scenes with strong view-dependent static objects and evaluate them using three images per scene with human-annotated masks focused on view-dependent regions. View-dependent static objects may be misclassified as distractors because their appearance changes lie between those of

static and transient objects. Nevertheless, the benefits of removing distractors still outweigh the occasional misclassification, as shown in Tab. S.5 and Fig. S.7.

#### S.3 Beyond Scene-Specific Methods

In this section, we provide additional experimental details, as well as per-method quantitative results, additional qualitative results, and failure cases of DI2FIX. We demonstrate how DF3DV-1K, a large-scale dataset, facilitates the development of approaches beyond scene-specific methods by introducing DI2FIX, an enhancement module that improves the rendering quality of distractor-free radiance field methods [20,30,33,36,47,54,68,83,86] and 3DGS [30] under distractorfree settings.

Experimental Details. We introduce DI2FIX (Distractor-Free DIFIX), a plug-and-play 2D enhancement module designed to improve the visual quality of radiance field renderings. The method is built upon DIFIX [91], a diffusion-based image-to-image model [62,70] originally developed to enhance degraded renderings under sparse-view settings using a clean reference image together with a degraded target view. In distractor-free reconstruction scenarios, clean reference images are not available. To adapt DIFIX [91] to this setting, we replace the clean reference input with renderings produced by radiance field models themselves. This modification enables the model to learn corrections directly from reconstruction artifacts arising in cluttered real-world captures while preserving the original DIFIX [91] architecture. Training data are constructed from DF3DV-1K* (DF3DV-1K excluding DF3DV-41). Specifically, scenes are reconstructed from cluttered image collections using multiple radiance field methods [20, 30,33,36, 47,54, 83, 86], and novel views are rendered at camera poses corresponding to clean-image viewpoints. Each rendered image is paired with its corresponding clean validation image, resulting in 316,890 candidate training pairs. To improve training stability and remove severely corrupted samples, we retain only pairs whose perceptual similarity satisfies LPIPS ≤ γ, where γ = 0.5 in our final configuration. We then fine-tune DIFIX [91] to obtain DI2FIX using two NVIDIA L40 GPUs with a batch size of 1, requiring approximately 3-4 days to converge. During adaptation, only minimal modifications are introduced, including replacing the clean reference with a radiance field rendering and disabling the Gram-matrix loss [21], as we empirically observe negligible benefits from this term in distractor-free reconstruction scenarios. All remaining training settings follow the original DIFIX [91] configuration.

Per-method Improvement. Tab. S.6 reports the performance improvements of DIFIX [91], DIFIX [91] fine-tuned on the RobustNeRF dataset [69] (DIFIX+RobustNeRF), and DIFIX fine-tuned on DF3DV-1K* (DI2FIX), measured relative to each radiance field method being enhanced. DI2FIX consistently improves all methods in terms of PSNR and LPIPS, achieving non-marginal performance gains. Changes in SSIM are relatively limited, which is expected since

DIFIX [91] does not employ an SSIM loss, and similar perceptual trade-offs with respect to SSIM have been reported in prior studies [4,5,38,46]. In contrast, the original DIFIX [91] provides only marginal improvements across methods, while DIFIX+RobustNeRF shows slightly better performance than DIFIX [91] but still fails to produce consistent gains. These results highlight the importance and value of domain-specific large-scale datasets.

We present enhanced results for all methods except SLS [68] and AsymGS [36] in the main paper for qualitative comparisons. Therefore, for additional qualitative evaluation, we select SLS [68] and AsymGS [36], together with 3DGS [30], as target methods for enhancement. Figs. S.8 to S.13 show qualitative comparisons among DIFIX [91], DIFIX+RobustNeRF, and DI2FIX when applied to renderings produced by 3DGS [30], SLS [68], and AsymGS [36]. From a visual perspective, DI2FIX produces more effective corrections than DIFIX and DIFIX+RobustNeRF, successfully removing distractor artifacts and inpainting occluded regions in most scenes. In comparison, DIFIX tends to preserve the original rendering with limited corrections, while DIFIX+RobustNeRF often oversmooths fine details. Specifically, Fig. S.8 shows that DI2FIX identifies and corrects color-similar distractor artifacts. For fluid distractors, DI2FIX largely mitigates blending artifacts while preserving better background details, as illustrated by the AsymGS [36] row. In the frontal distractor scenario, DI2FIX also preserves the black hole on the middle wall, which is often lost in DIFIX+RobustNeRF. In Fig. S.9, DI2FIX more effectively removes CD artifacts in the highly reflective distractor scenario and eliminates floaters in front of the building in the large-scale distractor scenario. It also successfully recovers the ceiling soffit in the 3DGS [30] and SLS [68] rows in the local air distractor scenario. Fig. S.10 further shows that floaters in the local appearance distractor scenario are removed and the background is recovered. In the semantically similar distractor scenario, DI2FIX more accurately identifies the dynamic chess pieces. In Fig. S.11, although all methods fail to correct the rolling shutter artifact in front of the transparent window, DI2FIX better removes the remaining fragments in the semi-transient distractor scenario and more effectively corrects shadow artifacts in the shadow distractor scenario. In the slow-motion scenario, DIFIX+RobustNeRF smooths the pattern on the tissue box. Finally, Fig. S.12 shows that DI2FIX removes distractor artifacts across various distractor scenarios that remain in DIFIX+RobustNeRF results in the AsymGS [36] row, and also corrects vegetable artifacts in daily scene scenarios.

Our-of-distribution Test. To assess the generalization capability of DI2FIX to previously unseen radiance-field methods, we perform a leave-one-method-out cross-method training and evaluation protocol. For each of the ten distractorfree radiance field methods [20,30,33,36,47,54,83,86], we exclude one method at a time and construct training pairs using renderings generated by the remaining nine methods. DIFIX is then fine-tuned on each reduced training set to obtain a corresponding DI2FIX model, resulting in ten models in total, each trained without exposure to renderings from one specific method. During evaluation,

each model is tested on all methods, and performance differences are reported relative to the DI2FIX model trained using renderings from all ten methods.

As shown in Tab. S.7, the performance variations remain small across all settings, with worst-case changes bounded within 0.1 PSNR, a 0.005 decrease in SSIM, and a 0.009 increase in LPIPS. These results indicate stable out-ofdistribution behavior and demonstrate that DI2FIX generalizes effectively to radiance-field methods not observed during training. Interestingly, excluding training data from AsymGS [36] occasionally yields slightly higher PSNR and SSIM, though the differences remain within the expected variance range.

Limitations of DI2FIX. As illustrated in Fig. S.15, DI2FIX may fail under extreme degradation or in cases affected by confirmation bias. When the input views are heavily corrupted, the model lacks sufficient reliable visual evidence and therefore relies primarily on learned diffusion priors to reconstruct occluded regions, which can lead to reduced reconstruction fidelity. In addition, when multiple views consistently contain a strong and visually unambiguous distractor, the model may incorrectly interpret the artifact as a valid scene component, resulting in confirmation bias during enhancement. A potential solution is to incorporate multiple reference views to provide stronger supervision. Exploring such multi-reference frameworks or view-selection approaches [53] is left for future work, as the primary goal of this work is to demonstrate the value of DF3DV-1K.

#### S.4 DF3DV-1K and DF3DV-41

We introduce DF3DV-1K (Distractor-Free 3D Vision 1K), a newly collected dataset designed for distractor-free novel view synthesis. The dataset provides clean and cluttered images for each scene, enabling systematic evaluation under diverse distractor conditions. In total, DF3DV-1K contains 1,048 indoor and outdoor scenes covering 128 distractor types and 161 scene themes (see Fig. S.17), comprising 89,924 images overall. To better reflect practical usage scenarios, the data are captured under real-world conditions using consumer cameras, emphasizing casually captured imagery. Within DF3DV-1K, we further introduce DF3DV-41, a curated subset consisting of systematically designed challenging scenarios intended to systematically evaluate distractor-free radiance field methods. Details of the scene design and scenario categories are provided in Figs. S.5 and S.6. DF3DV-1K* follows a real-world long-tail distribution dominated by common scenes/objects (e.g., buildings, cars), and DF3DV-41 is more uniformly distributed by design. In addition, scenes can have multiple distractor types, so category counts are not mutually exclusive. We believe the long-tail distribution better reflects real-world conditions, whereas DF3DV-41 supports systematic evaluation. We also provide scenario annotations for each scene in DF3DV-1K to support future investigations (see Fig. S.18). Through comprehensive benchmarking, we identify AsymGS [36] and RobustSplat [20] as the most robust approaches, followed by OCSplats [47] and DeGauss [83].

As illustrated in Fig. S.16, the ranking trends observed on DF3DV-1K broadly align with the chronological progression of recent research developments, suggesting that performance improvements in distractor-free radiance field methods have steadily advanced alongside methodological innovations. This observation indicates that DF3DV-1K provides a realistic and challenging benchmark capable of reflecting progress in the field. We also identify the weaknesses of recent distractor-free radiance field methods, such as their reliance on semantic features for distractor identification, which can lead to ambiguous decisions when distractors share similar semantics or appearance with static scene elements. Finally, we present a generalizable model demonstrating that DF3DV-1K facilitates the transition from scene-specific methods toward more generalizable solutions through DI2FIX, a distractor-free enhancement module for radiance fields. DI2FIX shows consistent improvements across ten radiance field methods, achieving average gains of 0.96 dB in PSNR and a 0.057 reduction in LPIPS.

Annotation Logic. Although each data instance in DF3DV-1K is scene-level, theme annotations follow two configuration types. The first is scene-based, where the spatial environment is clearly identifiable and well defined (e.g., factory or construction site), similar to the naming logic of On-the-go [67]. The second is object-based, applied when the environment is less distinguishable (e.g., a restaurant dining scene), where themes are named according to the dominant object (e.g., a dish), similar to the naming strategy used in RobustNeRF [69]. For most themes, we adopt high-level categories to ensure clarity and reproducibility (e.g., forks, knives, and spoons are grouped as cutlery, while cooking tools are categorized as utensils). To reduce semantic ambiguity, objects with multiple possible functions (e.g., a bowl) in our dataset are labeled using neutral object names rather than context-dependent categories.

Regarding distractor annotation logic, we also control semantic ambiguity, and distractors are distinguished based on perceptual specificity rather than broad functional class membership. For example, a general label is used (e.g., toy) when an object does not possess defining visual features that warrant further specification, while a more specific label is used when morphology, material, or texture clearly indicates a subtype (e.g., plush toy). For transportation-related categories, following common computer vision conventions, semantically distinct categories (e.g., car, truck, and bus) are preserved rather than merged into a single label. For future studies, clustering-based approaches [97,98] represent a promising direction for automatically annotating the dataset with feature-based scenarios and reducing manual annotation effort.

###### Table S.1: DF3DV-1K significance analysis. We compare top-performing methods using paired t-tests across evaluated scenes for PSNR, SSIM, and LPIPS. In each metric block, upper triangular entries report p-values, while lower triangular entries report the corresponding average performance margins. Smaller p-values indicate stronger statistical significance, and colors encode the ranking.

|margin<br><br>p-value<br><br>|PSNR AsymGS [36] RobustSplat [20] OCSplats [47] DeGauss [83]<br><br>|
|---|---|
|AsymGS [36] RobustSplat [20] OCSplats [47] DeGauss [83]|– 9.3×10−14 7.4×10−36 4.7×10−19<br><br>0.3572 – 1.9×10−2 2.2×10−2 0.4708 0.1135 – 6.8×10−1 0.4908 0.1336 0.0200 –<br><br>|

|margin<br><br>p-value<br><br>|SSIM AsymGS [36] RobustSplat [20] DeGauss [83] OCSplats [47]|
|---|---|
|AsymGS [36] RobustSplat [20] DeGauss [83] OCSplats [47]<br><br>|– 5.9×10−33 4.5×10−57 8.6×10−139<br><br>0.0210 – 8.2×10−1 2.4×10−2 0.0214 0.0004 – 5.6×10−3 0.0249 0.0039 0.0035 –<br><br>|

|margin<br><br>p-value<br><br>|LPIPS RobustSplat [20] AsymGS [36] DeGauss [83] OCSplats [47]|
|---|---|
|RobustSplat [20] AsymGS [36] DeGauss [83] OCSplats [47]|– 5.4×10−2 2.5×10−14 4.5×10−32 0.0035 – 2.2×10−9 4.7×10−37 0.0168 0.0133 – 2.8×10−2 0.0213 0.0178 0.0045 –<br><br>|

- Table S.2: Per-scenario PSNR of each method. The first column shows the PSNR achieved by 3DGS [30], while all other entries indicate the performance difference relative to 3DGS [30]. AsymGS [36], RobustSplat [20], and OCSplats [47] show consistent robustness across scenarios, while other methods remain sensitive to specific conditions, which can result in performance degradation. In addition, we identify semantically similar distractors as the most challenging distractor scenario, as the PSNR achieved by 3DGS [30] is low and all methods obtain improvements of less than or equal to 1 dB compared to other distractor scenarios.

3DGS [30]

T-3DGS [54]

T-3DGS-TMR [54]

WildGaussian [33]

SLS [68]

DeSplat [86]

DeGauss [83]

OCSplats [47]

RobustSplat [20]

AsymGS [36]

Scenario

Color-similar distractors

17.47 +2.02 +0.81 +1.37 +1.60 +2.15 +3.53 +2.26 +3.18 +3.01

Fluid distractors

14.54 +0.02 −0.01 +0.52 +0.96 +0.82 +1.40 +1.30 +0.46 +1.71

Frontal occlusion distractors

18.95 +1.71 +0.15 +1.73 +2.07 +1.67 +2.09 +2.08 +2.28 +2.08

Highly reflective distractors

- 17.42 +1.27 +0.12 +1.59 +1.19 +1.41 +2.39 +1.85 +1.57 +2.19

Large-scale distractors

- 18.00 +0.97 +0.16 +1.68 +1.44 +2.72 +2.03 +2.81 +3.57 +2.49

Local air distractors

18.70 −0.38 −0.33 +0.33 +0.97 +1.24 +2.24 +0.83 +1.57 +2.34

Local appearance distractors

- 15.60 +0.05 −0.01 +1.70 +2.19 +2.54 +2.95 +2.43 +1.17 +3.29

Semantically similar distractors

- 16.24 +0.68 +0.32 +0.66 +0.24 +0.02 +0.80 +0.41 +0.63 +1.00

Semi-transparent distractors

19.10 +1.03 +0.29 +0.99 +0.78 +0.11 +0.90 +0.57 +1.10 +1.36

Semi-transient distractors

15.85 +0.00 −0.58 +1.31 −0.25 +0.86 −0.59 +2.50 +2.18 +1.72

Shadow distractors

14.38 +0.55 +0.27 +0.91 +2.65 +2.13 +1.70 +2.52 +2.23 +2.61

Slow-motion distractors

19.98 −0.09 −0.58 −0.14 +1.37 +1.32 +1.95 +2.55 +1.07 +2.09

Various distractors

23.06 +2.31 +0.53 +1.06 +2.86 +2.64 +3.11 +3.08 +3.11 +2.78

Common distractors as static parts

17.79 +1.60 +0.66 +2.13 +2.06 +2.13 +2.00 +2.37 +2.13 +2.95

Daily scenes

17.73 +2.39 −0.77 +2.98 +2.68 +2.14 +3.44 +3.04 +3.88 +3.67

Nighttime scenes

20.25 +0.60 +0.09 +0.04 +0.47 −0.63 +0.98 +0.45 +1.46 +1.37

Other distractors/scenes

19.32 +1.01 +0.87 +1.23 +1.08 +0.87 +2.04 +1.57 +1.72 +2.34

- Table S.3: Per-scenario SSIM of each method. The first column shows the SSIM achieved by 3DGS [30], while all other entries indicate the performance difference relative to 3DGS [30]. All methods except DeSplat [86] demonstrate consistent robustness across scenarios. In addition, we identify semantically similar and fluid distractors as the most challenging distractor scenarios, since 3DGS [30] achieves relatively low SSIM values and all methods show improvements of less than 0.08 compared to other distractor scenarios.

3DGS [30]

T-3DGS [54]

T-3DGS-TMR [54]

WildGaussian [33]

SLS [68]

DeSplat [86]

DeGauss [83]

OCSplats [47]

RobustSplat [20]

AsymGS [36]

Scenario

Color-similar distractors

0.657 +0.089 +0.042 +0.067 +0.078 +0.067 +0.150 +0.102 +0.139 +0.128

Fluid distractors

0.410 +0.004 +0.003 +0.029 +0.039 +0.006 +0.052 +0.054 +0.021 +0.079

Frontal occlusion distractors

0.678 +0.073 +0.045 +0.066 +0.049 +0.063 +0.069 +0.064 +0.078 +0.080

Highly reflective distractors

0.589 +0.101 +0.044 +0.108 +0.076 +0.110 +0.148 +0.128 +0.121 +0.146

Large-scale distractors

0.703 +0.037 +0.024 +0.046 +0.042 +0.058 +0.060 +0.073 +0.104 +0.069

Local air distractors

0.720 +0.017 +0.015 +0.032 +0.041 +0.067 +0.092 +0.053 +0.073 +0.097

Local appearance distractors

0.518 +0.012 +0.011 +0.040 +0.042 +0.045 +0.076 +0.055 +0.043 +0.092

Semantically similar distractors

0.457 +0.058 +0.043 +0.052 +0.024 −0.009 +0.053 +0.050 +0.061 +0.079

Semi-transparent distractors

0.689 +0.067 +0.043 +0.064 +0.037 +0.027 +0.065 +0.057 +0.070 +0.089

Semi-transient distractors

0.515 +0.033 +0.013 +0.080 +0.019 +0.077 +0.012 +0.112 +0.114 +0.114

Shadow distractors

0.457 +0.046 +0.038 +0.035 +0.073 +0.052 +0.069 +0.078 +0.094 +0.094

Slow-motion distractors

0.778 +0.038 +0.014 +0.024 +0.035 +0.053 +0.066 +0.074 +0.034 +0.070

Various distractors

0.811 +0.067 +0.043 +0.043 +0.058 +0.061 +0.074 +0.070 +0.071 +0.073

Common distractors as static parts

0.621 +0.076 +0.045 +0.079 +0.053 +0.083 +0.087 +0.092 +0.093 +0.122

Daily scenes

0.701 +0.097 +0.015 +0.105 +0.090 +0.074 +0.118 +0.112 +0.131 +0.137

Nighttime scenes

0.664 +0.030 +0.019 +0.007 +0.011 −0.016 +0.049 +0.027 +0.047 +0.055

Other distractors/scenes

0.645 +0.055 +0.049 +0.042 +0.034 +0.026 +0.074 +0.057 +0.063 +0.085

- Table S.4: Per-scenario LPIPS of each method. The first column shows the LPIPS achieved by 3DGS [30], while all other entries indicate the performance difference relative to 3DGS [30]. Lower LPIPS values indicate better perceptual similarity. All methods except T-3DGS-TMR [54] and WildGaussian [33] demonstrate consistent robustness across scenarios. In addition, we identify fluid distractors as the most challenging scenario, since 3DGS [30] achieves relatively high LPIPS values and all methods show improvements of less than 0.08 compared to other distractor scenarios.

Scenario

3DGS [30]

T-3DGS [54]

T-3DGS-TMR [54]

WildGaussian [33]

SLS [68]

DeSplat [86]

DeGauss [83]

OCSplats [47]

RobustSplat [20]

AsymGS [36]

Color-similar distractors

0.477 −0.109 −0.051 −0.076 −0.068 −0.073 −0.192 −0.114 −0.176 −0.176

Fluid distractors

0.444 −0.042 −0.041 −0.031 −0.017 −0.052 −0.076 −0.031 −0.056 −0.038

Frontal occlusion distractors

0.210 −0.071 −0.036 −0.058 −0.053 −0.082 −0.062 −0.070 −0.081 −0.043

Highly reflective distractors

0.333 −0.097 −0.036 −0.094 −0.068 −0.114 −0.141 −0.119 −0.118 −0.130

Large-scale distractors

0.283 −0.059 −0.040 −0.016 −0.048 −0.091 −0.090 −0.097 −0.133 −0.063

Local air distractors

0.244 −0.020 −0.021 −0.038 −0.035 −0.078 −0.106 −0.039 −0.085 −0.098

Local appearance distractors

0.358 −0.011 −0.008 −0.056 −0.036 −0.078 −0.078 −0.054 −0.061 −0.072

Semantically similar distractors

0.407 −0.093 −0.065 −0.030 −0.038 −0.061 −0.099 −0.068 −0.087 −0.078

Semi-transparent distractors

0.289 −0.085 −0.053 −0.058 −0.048 −0.067 −0.091 −0.070 −0.093 −0.091

Semi-transient distractors

0.341 −0.035 −0.014 −0.071 −0.017 −0.095 −0.029 −0.124 −0.132 −0.088

Shadow distractors

0.398 −0.071 −0.063 −0.042 −0.054 −0.111 −0.068 −0.084 −0.097 −0.032

Slow-motion distractors

0.177 −0.024 +0.005 −0.021 −0.024 −0.058 −0.063 −0.075 −0.024 −0.057

Various distractors

0.131 −0.060 −0.031 −0.038 −0.057 −0.058 −0.068 −0.066 −0.068 −0.062

Common distractors as static parts

0.268 −0.093 −0.058 −0.072 −0.063 −0.114 −0.112 −0.097 −0.113 −0.120

Daily scenes

0.409 −0.127 −0.018 −0.123 −0.106 −0.115 −0.166 −0.138 −0.177 −0.166

Nighttime scenes

0.428 −0.063 −0.081 +0.010 −0.005 −0.078 −0.066 −0.016 −0.051 −0.036

Other distractors/scenes

0.317 −0.090 −0.081 −0.042 −0.045 −0.083 −0.103 −0.099 −0.105 −0.090

- Table S.5: View-dependent static object benchmark. Removing distractors may accidentally affect the rendering quality of view-dependent effects on static objects. However, the benefits still outweigh this drawback, as evidenced by the better performance of distractor-free radiance field methods compared with 3DGS [30].

| |3DGS [30] T-3DGS [54] T-3DGS-TMR [54] WildGaussian [33] SLS [68] DeSplat [86] DeGauss [83] OCSplats [47] RobustSplat [20] AsymGS [36]<br><br>|
|---|---|
|PSNR SSIM LPIPS<br><br>|21.35 21.83 21.56 22.18 21.90 21.38 22.28 22.13 22.49 23.19 0.769 0.796 0.788 0.797 0.790 0.781 0.806 0.802 0.807 0.826 0.196 0.165 0.176 0.182 0.173 0.168 0.147 0.161 0.151 0.149<br><br>|

[Figure 136]

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

Color-Similar

Distractors

WildGaussians

GT

3DGS

T-3DGS

SLS

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

AsymGS

OCSplats

RobustSplat

DeSplat

DeGauss

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

Distractors

Fluid

3DGS

T-3DGS

GT

SLS

WildGaussians

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

AsymGS

RobustSplat

DeSplat

OCSplats

DeGauss

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

Occlusion

Distractors

GT

T-3DGS

3DGS

WildGaussians

SLS

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

Frontal

| |
|---|

| |
|---|

| |
|---|

RobustSplat

AsymGS

OCSplats

DeSplat

DeGauss

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

| |
|---|

| |
|---|

| |
|---|

Reflective

Distractors

3DGS

WildGaussians

GT

T-3DGS

SLS

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Highly

| |
|---|

| |
|---|

| |
|---|

DeSplat

AsymGS

OCSplats

RobustSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Large-Scale

Distractors

GT

3DGS

WildGaussians

T-3DGS

SLS

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

AsymGS

DeSplat

OCSplats

RobustSplat

DeGauss

###### Fig. S.1: Qualitative comparison of radiance field methods on DF3DV-41 across color-similar, fluid, frontal occlusion, highly reflective, and largescale distractors. The benchmark introduces systematically challenging conditions that enable clear visual comparison across methods and support the evaluation of robustness differences.

[Figure 137]

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

Distractors

Air

GT

3DGS

T-3DGS

SLS

WildGaussians

Local

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

AsymGS

DeSplat

OCSplats

RobustSplat

DeGauss

| |
|---|

| |
|---|

Appearance

Distractors

GT

3DGS

T-3DGS

SLS

WildGaussians

Local

O

| |
|---|

| |
|---|

DeSplat

OCSplats

AsymGS

RobustSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Similar

Distractors

Semantically

3DGS

GT

WildGaussians

SLS

T-3DGS

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

AsymGS

OCSplats

RobustSplat

DeSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Semi-Transparent

| |
|---|

| |
|---|

Distractors

3DGS

T-3DGS

GT

WildGaussians

SLS

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

| |
|---|

| |
|---|

RobustSplat

AsymGS

DeSplat

OCSplats

DeGauss

| |
|---|

| |
|---|

| |
|---|

Semi-Transient

WildGaussians

GT

3DGS

T-3DGS

SLS

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

AsymGS

RobustSplat

OCSplats

DeSplat

DeGauss

###### Fig. S.2: Qualitative comparison of radiance field methods on DF3DV-41 across local air, local appearance, semantically similar, semi-transparent, and semi-transient distractors. The benchmark introduces systematically challenging conditions that allow clear visual comparison across methods and support the evaluation of robustness differences.

[Figure 138]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Distractors

Shadow

GT

WildGaussians

3DGS

SLS

T-3DGS

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

AsymGS

OCSplats

DeSplat

RobustSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

Slow-Motion

Distractors

WildGaussians

SLS

3DGS

GT

T-3DGS

| |
|---|

| |
|---|

| |
|---|

| |
|---|

OCSplats

RobustSplat

AsymGS

DeSplat

DeGauss

| |
|---|

| |
|---|

Distractors

Various

3DGS

GT

T-3DGS

WildGaussians

SLS

| |
|---|

AsymGS

RobustSplat

DeSplat

OCSplats

DeGauss

| |
|---|

| |
|---|

| |
|---|

Distractors

| |
|---|

Parts

GT

SLS

T-3DGS

3DGS

WildGaussians

Static

Common

| |
|---|

| |
|---|

as

AsymGS

DeSplat

OCSplats

RobustSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Scenes

Daily

GT

T-3DGS

SLS

3DGS

WildGaussians

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

DeSplat

OCSplats

RobustSplat

AsymGS

DeGauss

###### Fig. S.3: Qualitative comparison of radiance field methods on DF3DV-41 across shadow, slow-motion, and various distractors, as well as common distractors treated as static parts and daily scenes. The benchmark introduces systematically challenging conditions that allow clear visual comparison across methods and support the evaluation of robustness differences.

[Figure 139]

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

| |
|---|

| |
|---|

| |
|---|

Nighttime

Scenes

T-3DGS WildGaussians

GT

3DGS

SLS

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

| |
|---|

AsymGS

RobustSplat

OCSplats

DeSplat

DeGauss

| |
|---|

Other

T-3DGS

GT

WildGaussians

3DGS

SLS

| |
|---|

AsymGS

DeSplat

RobustSplat

OCSplats

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Other

SLS

WildGaussians

3DGS

T-3DGS

GT

| |
|---|

| |
|---|

| |
|---|

AsymGS

RobustSplat

DeSplat

OCSplats

DeGauss

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

Other

3DGS

GT

T-3DGS

WildGaussians

SLS

| |
|---|

| |
|---|

| |
|---|

| |
|---|

AsymGS

OCSplats

DeSplat

RobustSplat

DeGauss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Other

3DGS

GT

T-3DGS

WildGaussians

SLS

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

RobustSplat

AsymGS

OCSplats

DeSplat

DeGauss

###### Fig. S.4: Qualitative comparison of radiance field methods on DF3DV-41 across nighttime scenes and other distractors and scenes. The benchmark introduces systematically challenging conditions that allow clear visual comparison across methods and support the evaluation of robustness differences.

|[Figure 140]<br><br>[Figure 141]|[Figure 142]|[Figure 143]|
|---|---|---|
| | |[Figure 144]|

|[Figure 145]|[Figure 146]|[Figure 147]|
|---|---|---|
| | |[Figure 148]|

|[Figure 149]|[Figure 150]|[Figure 151]|
|---|---|---|
| | |[Figure 152]|

Fluid distractors - WashCar: Distractors (e.g., water) exhibit non-rigid, irregular, or scattered visual patterns.

Color-similar distractors – BlackTable: The distractor’s (e.g., black objects) color is similar to the static parts

Color-similar distractors – BlackBackground: The distractor’s (e.g., passerby) color is similar to the static parts.

|[Figure 153]|[Figure 154]|[Figure 155]|
|---|---|---|
| | |[Figure 156]|

|[Figure 157]|[Figure 158]|[Figure 159]|
|---|---|---|
| | |[Figure 160]|

|[Figure 161]|[Figure 162]|
|---|---|
| |[Figure 163]|

|[Figure 164]|
|---|

Fluid distractors - Water: Distractors (e.g., water) exhibit non-rigid, irregular, or scattered visual patterns.

Frontal occlusion distractors - Island Factory: The distractor (e.g., fingers) occludes the view from the front.

Frontal occlusion distractors - TunnelEntrance: The distractor (e.g., sleeve) occludes the view from the front.

|[Figure 165]|[Figure 166]|[Figure 167]|
|---|---|---|
| | |[Figure 168]|

|[Figure 169]|[Figure 170]|[Figure 171]|
|---|---|---|
| | |[Figure 172]|

|[Figure 173]|[Figure 174]|[Figure 175]|
|---|---|---|
| | |[Figure 176]|

Large-scale distractors - LightRail: Distractor (e.g., light rail) that are large in scale.

Highly reflective distractors - CD: Distractors (e.g., CD) that exhibit strong specular reflections.

Highly reflective distractors - AluminumFoil: Distractors (e.g., aluminum foil) that exhibit strong specular reflections.

|[Figure 177]|[Figure 178]|[Figure 179]|
|---|---|---|
| | |[Figure 180]|

|[Figure 181]|[Figure 182]|[Figure 183]|
|---|---|---|
| | |[Figure 184]|

|[Figure 185]|[Figure 186]|[Figure 187]|
|---|---|---|
| | |[Figure 188]|

Large-scale distractors - Train: Distractor (e.g., ctrain) that are large in scale.

Local air distractors - BellIncense: Air distractors (e.g., smoke) introduce transient, semi-transparent structures.

Local air distractors - DrumIncense: Air distractors (e.g., smoke) introduce transient, semi-transparent structures.

|[Figure 189]|[Figure 190]|[Figure 191]|
|---|---|---|
| | |[Figure 192]|

|[Figure 193]|[Figure 194]|[Figure 195]|
|---|---|---|
| | |[Figure 196]|

|[Figure 197]|[Figure 198]|[Figure 199]|
|---|---|---|
| | |[Figure 200]|

Semantically similar distractors - Chess: The distractor (e.g., chess) is semantically similar to the static parts.

Local appearance distractors - Light: Distractors (e.g., lighting changes) that alter surface appearance without changing geometry.

Local appearance distractors - Light: Distractors (e.g., lighting changes) that alter surface appearance without changing geometry.

|[Figure 201]|[Figure 202]|[Figure 203]|
|---|---|---|
| | |[Figure 204]|

|[Figure 205]|[Figure 206]|[Figure 207]|
|---|---|---|
| | |[Figure 208]|

|[Figure 209]|[Figure 210]|[Figure 211]|
|---|---|---|
| | |[Figure 212]|

Semantically similar distractors TomatoPaste: The distractor (e.g., tomato paste) is semantically similar to the static parts.

###### Semi-transparent distractors - CarWindow:

Semi-transparent distractors – GlassyContainer: Distractors (e.g., glassy container) cause foreground–background mixing.

Distractors (e.g., windows) cause foreground– background mixing.

|[Figure 213]|[Figure 214]|[Figure 215]|
|---|---|---|
| | |[Figure 216]|

|[Figure 217]<br><br>[Figure 218]|[Figure 219]<br><br>[Figure 220]|[Figure 221]<br><br>[Figure 222]|
|---|---|---|
| | |[Figure 223]<br><br>[Figure 224]|

|[Figure 225]| |[Figure 226]|
|---|---|---|
| | |[Figure 227]|

|[Figure 228]|
|---|

Semi-transient distractors –RollingShutter : Distractors (e.g., rolling shutter) start moving from a static state.

Semi-transient distractors – RollingShutter2: Distractors (e.g., rolling shutter) start moving from a static state.

Shadow distractors - Shadow: Shadows cause appearance changes without altering the geometric structure.

|[Figure 229]|[Figure 230]|[Figure 231]|
|---|---|---|
| | |[Figure 232]|

|[Figure 233]|[Figure 234]|[Figure 235]|
|---|---|---|
| | |[Figure 236]|

|[Figure 237]|[Figure 238]|[Figure 239]|
|---|---|---|
| | |[Figure 240]|

Various distractors - TeaPot: Different distractors appear across views.

Shadow distractors – Shadow2: Shadows cause appearance changes without altering the geometric structure.

Slow-motion distractors - MusicBox: The distractors (e.g., train) move slowly over time.

|[Figure 241]|[Figure 242]|[Figure 243]|
|---|---|---|
| | |[Figure 244]|

|[Figure 245]<br><br>[Figure 246]|[Figure 247]<br><br>[Figure 248]|[Figure 249]<br><br>[Figure 250]|
|---|---|---|
| | |[Figure 251]<br><br>[Figure 252]|

|[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]|[Figure 257]<br><br>[Figure 258]|[Figure 259]<br><br>[Figure 260]|
|---|---|---|
| | |[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]|

Daily scenes - Cut: Dynamic interactions and object manipulations in everyday scenes.

Common distractors as static parts - Actor: Common distractors (e.g., passer-by) appear as

Common distractors as static parts – Actor2: Common distractors (e.g., passer-by) appear as static parts.

static parts.

|[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]|[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]|[Figure 275]|
|---|---|---|
| | |[Figure 276]|

|[Figure 277]<br><br>[Figure 278]|[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]|[Figure 282]|
|---|---|---|
| | |[Figure 283]|

|[Figure 284]|[Figure 285]|[Figure 286]|
|---|---|---|
| | |[Figure 287]|

Nighttime scenes – NightOperaHouse: Low-light conditions and artificial illumination introduce noise and reduce the visibility of distractors.

Nighttime scenes – NightOperaHouse2: Low-light conditions and artificial illumination introduce noise and reduce the visibility of distractors.

Nighttime scenes -Car: Low-light conditions and artificial illumination introduce noise and reduce the visibility of distractors.

###### Fig. S.5: Sample views of scenes in DF3DV-41. DF3DV-41 covers a wide range of specifically designed distractor types and scene scenarios. The systematic design enables evaluation and clear comparison of method robustness under diverse challenging conditions.

|[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]|[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]|[Figure 295]|
|---|---|---|
| | |[Figure 296]|

|[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]|[Figure 302]<br><br>[Figure 303]|[Figure 304]|
|---|---|---|
| | |[Figure 305]|

|[Figure 306]|[Figure 307]|[Figure 308]|
|---|---|---|
| | |[Figure 309]|

Other distractors/scenes – Operahouse: Outdoor scene with common distractors (e.g.,

Other distractors/scenes – Operahouse2: Outdoor scene with common distractors (e.g.,

Other distractors/scenes - Koala: Multiple distractors appear in views with random combination.

passer-by).

passer-by).

|[Figure 310]|[Figure 311]|[Figure 312]|
|---|---|---|
| | |[Figure 313]|

|[Figure 314]|[Figure 315]|[Figure 316]|
|---|---|---|
| | |[Figure 317]|

|[Figure 318]|[Figure 319]|[Figure 320]|
|---|---|---|
| | |[Figure 321]|

Other distractors/scenes - BoardGame: Multiple thin distractors (e.g., card) appear in different views with random combination.

Other distractors/scenes - Yarn: Slim distractors (e.g., yarn) appear in different views.

Other distractors/scenes - Flower: Fast moving distractors (e.g., petals) show in front of static object.

|[Figure 322]<br><br>[Figure 323]|[Figure 324]<br><br>[Figure 325]|[Figure 326]<br><br>[Figure 327]|
|---|---|---|
| | |[Figure 328]<br><br>[Figure 329]|

|[Figure 330]|[Figure 331]|[Figure 332]|
|---|---|---|
| | |[Figure 333]|

|[Figure 334]|[Figure 335]|[Figure 336]|
|---|---|---|
| | |[Figure 337]|

Other distractors/scenes -straw: Slim distractors (e.g., straw) appear in different views.

Other distractors/scenes - headset: Different distractors (e.g., headset) cover up part of the static object.

Other distractors/scenes - TrinketBox: Tiny distractors (e.g., accessories) show across different position around static object.

|[Figure 338]|[Figure 339]|[Figure 340]|
|---|---|---|
| | |[Figure 341]|

|[Figure 342]<br><br>[Figure 343]|[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]|[Figure 348]|
|---|---|---|
| | |[Figure 349]|

Other distractors/scenes - Statue: Regular outdoor scene with common distractors (e.g., bag).

Other distractors/scenes - Temple: Regular largescale scenes with common distractors (e.g., passerby).

###### Fig. S.6: Sample views of scenes in DF3DV-41. Other distractor and scene scenarios include diverse common distractors that vary in shape and characteristics, such as slim, small, or fast-moving objects.

[Figure 350]

###### Fig. S.7: Novel view synthesis of view-dependent static objects. Distractor-free radiance field methods [36, 54] produce higher-quality novel views around the television screen, showing that the benefits of removing distractors outweigh the potential negative impact on view-dependent effects. In addition, AsymGS [36] successfully renders the console reflected on the screen, demonstrating its ability to distinguish viewdependent static objects from distractors.

- Table S.6: Effect of enhancers on each radiance field method. Vanilla is the rendering results of the methods [20,30,33,36,47,54,68,83,86]. DIFIX+RobustNeRF and DI2FIX are DIFIX [91] fine-tuned on RobustNeRF [69] and DF3DV-1K*, respectively. Values report the mean performance change on On-the-go [67] and DF3DV-41 relative to the Vanilla. Positive values indicate improvement for PSNR and SSIM, while negative values indicate improvement for LPIPS.

3DGS [30]

T-3DGS [54]

T-3DGS-TMR [54]

WildGaussian [33]

SLS [68]

AsymGS [36] PSNR↑

DeSplat [86]

DeGauss [83]

OCSplats [47]

RobustSplat [20]

Vanilla

18.71 20.90 19.12 20.93 21.05 21.02 21.77 21.41 21.52 21.75 DIFIX

−0.19 −0.62 −0.27 −0.64 −0.66 −0.80 −0.94 −0.80 −0.83 −0.85 DIFIX+RobustNeRF

- +0.41 −0.12 +0.53 −0.37 −0.50 −0.29 −0.71 −0.46 −0.56 −0.65

DI2FIX

- +1.83 +1.26 +2.02 +0.77 +0.67 +0.86 +0.47 +0.63 +0.69 +0.39

SSIM↑ Vanilla

0.641 0.745 0.695 0.732 0.721 0.733 0.763 0.753 0.760 0.767 DIFIX

−0.047 −0.068 −0.061 −0.065 −0.060 −0.072 −0.075 −0.076 −0.076 −0.075 DIFIX+RobustNeRF

−0.014 −0.033 −0.019 −0.039 −0.035 −0.034 −0.047 −0.042 −0.045 −0.050 DI2FIX

+0.023 +0.011 +0.026 +0.001 +0.002 +0.008 −0.003 +0.000 +0.000 −0.005

LPIPS↓ Vanilla

0.319 0.193 0.255 0.220 0.220 0.184 0.172 0.178 0.172 0.192 DIFIX

−0.001 +0.012 +0.001 +0.009 +0.011 +0.019 +0.020 +0.018 +0.018 +0.014 DIFIX+RobustNeRF

###### −0.047 −0.005 −0.032 −0.007 −0.009 −0.000 +0.008 +0.003 +0.006 +0.001 DI2FIX −0.111 −0.050 −0.085 −0.058 −0.059 −0.041 −0.036 −0.038 −0.036 −0.049

| |
|---|

| |
|---|

[Figure 351]

DI2FIX

3DGS

GT

DIFIX

DIFIX+RobustNeRF

| |
|---|

Color-Similar

Distractors

DI²FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

| |
|---|

AsymGS

DI2FIX

GT

DIFIX

DIFIX+RobustNeRF

| |
|---|

| |
|---|

| |
|---|

DI2FIX

3DGS

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

Distractors

Fluid

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

| |
|---|

DI2FIX

AsymGS

DIFIX+RobustNeRF

GT

DIFIX

| |
|---|

| |
|---|

| |
|---|

G

DI2FIX

DIFIX+RobustNeRF

3DGS

Occlusion

DIFIX

| |
|---|

Distractors

| |
|---|

| |
|---|

Frontal

DI2FIX

SLS

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI2FIX

AsymGS

GT

DIFIX+RobustNeRF

DIFIX

###### Fig. S.8: Qualitative comparison of enhancers on radiance-field outputs under color-similar, fluid, and frontal-occlusion distractor scenarios. Although DI2FIX cannot always restore severely degraded regions, leveraging DF3DV-1K*, it shows promising results in mitigating distractor artifacts and improving visual quality across different methods.

[Figure 352]

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|

DI2FIX

GT

3DGS

DIFIX+RobustNeRF

DIFIX

Reflective

| |
|---|

| |
|---|

Distractors

| |
|---|

Highly

DI2FIX

DIFIX+RobustNeRF

GT

SLS

DIFIX

| |
|---|

| |
|---|

AsymGS

DI2FIX

DIFIX+RobustNeRF

GT

DIFIX

| |
|---|

| |
|---|

DI2FIX

3DGS

GT

DIFIX+RobustNeRF

DIFIX

Large-Scale

Distractors

| | |
|---|---|
| | |

| |
|---|

DI2FIX

DIFIX+RobustNeRF

SLS

GT

DIFIX

| |
|---|

| | |
|---|---|
| | |
| | |

| |
|---|

| |
|---|

AsymGS

DI2FIX

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI2FIX

GT

3DGS

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

Distractors

Air

Local

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

| |
|---|

DI2FIX

AsymGS

GT

DIFIX+RobustNERF

DIFIX

###### Fig. S.9: Qualitative comparison of enhancers on radiance-field outputs under highly reflective, large-scale, and local air distractor scenarios. Although

[Figure 353]

| |
|---|

| |
|---|

DI2FIX

GT

DIFIX+RobustNeRF

3DGS

Appearance

DIFIX

| |
|---|

| |
|---|

Distractors

Local

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

DI2FIX

AsymGS

GT

DIFIX+RobustNeRF

DIFIX

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

Similar

GT

3DGS

DI2FIX

DIFIX+RobustNeRF

DIFIX

| |
|---|

Distractors

| |
|---|

Semantically

| |
|---|

| |
|---|

| |
|---|

GT

DIFIX+RobustNERF

DI2FIX

SLS

DIFIX

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

DI2FIX

AsymGS

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI²FIX

GT

3DGS

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

Distractors

Semi-T

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

DI2FIX

AsymGS

GT

DIFIX

DIFIX+RobustNeRF

###### Fig. S.10: Qualitative comparison of enhancers on radiance-field outputs under local appearance, semantically similar, and semi-transparent distractor scenarios. Although DI2FIX cannot always restore severely degraded regions, it shows promising results in mitigating distractor artifacts and improving visual quality across different methods.

[Figure 354]

| |
|---|

| |
|---|

3DGS

DIFIX+RobustNeRF

DI FIX

GT

DIFIX

Semi-Transient

| |
|---|

Distractors

| |
|---|

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

| |
|---|

| |
|---|

DI2FIX

AsymGS

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI2FIX

DIFIX+RobustNeRF

3DGS

GT

DIFIX

Distractors

| |
|---|

Shadow

| |
|---|

DI2FIX

GT

SLS

DIFIX

DIFIX+RobustNeRF

| |
|---|

| |
|---|

DI2FIX

AsymGS

DIFIX

GT

DIFIX+RobustNeRF

| |
|---|

| |
|---|

DI2FIX

DIFIX+RobustNeRF

GT

DIFIX

3DGS

| |
|---|

Slow-Motion

Distractors

| |
|---|

DI2FIX

GT

SLS

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI²FIX

AsymGS

GT

DIFIX+RobustNeRF

DIFIX

###### Fig. S.11: Qualitative comparison of enhancers on radiance-field outputs under semi-transient, shadow, and slow-motion distractor scenarios. Although

[Figure 355]

| |
|---|

DI²FIX

GT

3DGS

DIFIX+RobustNeRF

DIFIX

Distractors

Various

DI2FIX

DIFIX+RobustNeRF

GT

DIFIX

SLS

| |
|---|

| |
|---|

AsymGS

DI2FIX

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

Distractors

DI²FIX

GT

3DGS

DIFIX+RobustNeRF

DIFIX

Parts

| |
|---|

Static

Common

as

DI2FIX

SLS

GT

DIFIX+RobustNeRF

DIFIX

AsymGS

GT

DI2FIX

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI2FIX

3DGS

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

Scenes

Daily

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

| |
|---|

DI2FIX

AsymGS

DIFIX+RobustNeRF

GT

DIFIX

###### Fig. S.12: Qualitative comparison of enhancers on radiance-field outputs across various distractor scenarios, as well as common distractors treated as static parts and daily scene scenarios. Although DI2FIX cannot always restore severely degraded regions, it shows promising results in mitigating distractor artifacts and improving visual quality across different methods.

[Figure 356]

| |
|---|

| |
|---|

DI2FIX

DIFIX+RobustNeRF

GT

DIFIX

3DGS

Nighttime

| |
|---|

| |
|---|

Scenes

GT

DI2FIX

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

DI2FIX

AsymGS

GT

DIFIX+RobustNeRF

DIFIX

| |
|---|

| |
|---|

DI2FIX

GT

DIFIX+RobustNeRF

3DGS

DIFIX

| |
|---|

Other

DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

| |
|---|

DI2FIX

AsymGS

DIFIX+RobustNeRF

GT

DIFIX

| |
|---|

| |
|---|

DI2FIX

3DGS

DIFIX+RobustNeRF

GT

DIFIX

|TR|
|---|

Other

RA

RA

RA

###### DI2FIX

GT

DIFIX+RobustNeRF

SLS

DIFIX

|RA|
|---|

RA

R

AsymGS

DI2FIX

GT

DIFIX+RobustNeRF

DIFIX

###### Fig. S.13: Qualitative comparison of enhancers on radiance-field outputs under nighttime and other scene scenarios. Although DI2FIX cannot always restore severely degraded regions, it shows promising results in mitigating distractor artifacts and improving visual quality across different methods.

[Figure 357]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

DI2FIX

GT

DIFIX+RobustNeRF

3DGS

DIFIX

Corner

DI2FIX

DIFIX+RobustNeRF

GT

SLS

DIFIX

AsymGS

DI2FIX

GT

DIFIX

DIFIX+RobustNeRF

| |
|---|

| |
|---|

| |
|---|

| |
|---|

DI2FIX

GT

DIFIX+RobustNeRF

3DGS

DIFIX

| |
|---|

High

| |
|---|

| |
|---|

| |
|---|

Patio

DI2FIX

DIFIX+RobustNeRF

GT

SLS

DIFIX

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

DI²FIX

AsymGS

GT

DIFIX

DIFIX+RobustNeRF

| |
|---|

| |
|---|

| |
|---|

| |
|---|

GT

3DGS

DI2FIX

DIFIX+RobustNeRF

DIFIX

| |
|---|

Fountain

| |
|---|

DIFIX+RobustNeRF

DI2FIX

SLS

GT

DIFIX

| |
|---|

AsymGS

GT

DIFIX+RobustNeRF

DI2FIX

DIFIX

- Fig. S.14: Qualitative comparison of enhancers on radiance-field outputs for the Corner, Patio High, and Fountain scenes from On-the-go [67]. DI2FIX significantly enhances the results of 3DGS [30]. Although recent state-of-the-art methods already perform well on the On-the-go [67] dataset, DI2FIX can further improve rendering quality when artifacts are present (e.g., the window in the background of the Patio High scene).

- Table S.7: Cross-method ablation of DI2FIX. Each “Without” row denotes a leave-one-method-out setting, where DI2FIX is trained without data from that method and evaluated on all methods. Values report the mean performance change relative to DI2FIX trained on all ten methods. Positive values indicate improvement for PSNR/SSIM, while negative values indicate improvement for LPIPS. Across all metrics, performance drops remain small, with maximum decreases of 0.1 dB in PSNR, 0.005 in SSIM, and 0.009 in LPIPS, demonstrating strong cross-method generalization. We do not observe reduced generalization to the held-out method, as indicated by the diagonal entries. Although the differences fall within the error bars, it is interesting to observe that excluding AsymGS [36] results in slightly better PSNR and SSIM.

3DGS [30]

T-3DGS [54]

T-3DGS-TMR [54]

WildGaussian [33]

AsymGS [36] PSNR↑

SLS [68]

DeSplat [86]

DeGauss [83]

OCSplats [47]

RobustSplat [20]

Without

DI2FIX 20.55 22.16 21.15 21.71 21.73 21.89 22.25 22.05 22.21 22.14 3DGS [30] −0.10 −0.02 −0.04 −0.07 −0.08 −0.01 −0.04 −0.04 −0.03 −0.04 T-3DGS [54] +0.01 +0.05 +0.03 +0.03 +0.03 +0.07 +0.02 +0.05 +0.04 +0.04 T-3DGS-TMR [54] +0.00 +0.00 +0.01 +0.01 −0.03 +0.00 −0.04 +0.00 +0.00 +0.02 WildGaussian [33] +0.07 +0.06 +0.10 +0.05 +0.05 +0.04 +0.03 +0.05 +0.05 +0.06 SLS [68] −0.08 −0.01 +0.00 −0.05 −0.05 −0.02 −0.05 +0.00 −0.03 +0.00 DeSplat [86] −0.10 +0.00 −0.02 −0.02 −0.04 −0.08 −0.07 +0.01 −0.01 −0.02 DeGauss [83] +0.02 −0.03 +0.04 −0.04 −0.08 +0.00 −0.08 −0.01 −0.01 −0.02 OCSplats [47] −0.05 −0.08 −0.04 −0.07 −0.04 −0.09 −0.09 −0.06 −0.05 −0.02 RobustSplat [20] −0.03 −0.05 +0.00 −0.03 −0.05 −0.02 −0.04 +0.00 +0.00 +0.00 AsymGS [36] +0.07 +0.05 +0.10 +0.05 +0.01 +0.10 +0.03 +0.05 +0.04 +0.07

###### SSIM↑

DI2FIX 0.664 0.756 0.722 0.733 0.724 0.742 0.759 0.753 0.760 0.761

3DGS [30] −0.002 +0.000 +0.000 −0.002 −0.002 +0.000 −0.001 −0.001 −0.001 −0.002 T-3DGS [54] +0.000 −0.001 +0.000 −0.002 −0.001 +0.000 −0.002 −0.001 −0.002 −0.002 T-3DGS-TMR [54] −0.002 −0.002 −0.002 −0.002 −0.002 −0.001 −0.003 −0.002 −0.002 −0.003 WildGaussian [33] +0.001 +0.000 +0.000 +0.000 +0.000 +0.000 +0.000 +0.000 +0.000 +0.000 SLS [68] −0.001 −0.003 −0.002 −0.003 −0.003 −0.003 −0.004 −0.003 −0.004 −0.004 DeSplat [86] −0.005 −0.003 −0.004 −0.004 −0.005 −0.004 −0.005 −0.004 −0.004 −0.005 DeGauss [83] −0.001 −0.001 +0.000 −0.002 −0.003 −0.001 −0.003 −0.002 −0.003 −0.003 OCSplats [47] −0.002 −0.002 −0.002 −0.003 −0.003 −0.002 −0.003 −0.002 −0.003 −0.003 RobustSplat [20] −0.002 −0.002 +0.000 −0.002 −0.003 −0.001 −0.003 −0.002 −0.002 −0.002 AsymGS [36] +0.000 +0.000 +0.001 +0.000 +0.000 +0.001 +0.000 +0.000 +0.000 +0.000

###### LPIPS↓

DI2FIX 0.208 0.142 0.169 0.161 0.161 0.143 0.135 0.140 0.136 0.142

3DGS [30] +0.007 +0.003 +0.004 +0.006 +0.005 +0.005 +0.004 +0.004 +0.004 +0.004 T-3DGS [54] +0.006 +0.004 +0.004 +0.006 +0.005 +0.005 +0.005 +0.005 +0.005 +0.006 T-3DGS-TMR [54] +0.003 +0.002 +0.002 +0.002 +0.003 +0.003 +0.002 +0.003 +0.002 +0.002 WildGaussian [33] +0.001 −0.000 −0.000 +0.005 −0.000 +0.001 +0.001 −0.000 +0.001 +0.003 SLS [68] +0.007 +0.002 +0.001 +0.004 +0.006 +0.003 +0.004 +0.004 +0.004 +0.004 DeSplat [86] +0.006 +0.005 +0.004 +0.008 +0.006 +0.006 +0.006 +0.006 +0.006 +0.007 DeGauss [83] +0.005 +0.004 +0.003 +0.008 +0.007 +0.006 +0.007 +0.006 +0.006 +0.007 OCSplats [47] +0.009 +0.005 +0.005 +0.009 +0.008 +0.007 +0.007 +0.007 +0.007 +0.007 RobustSplat [20] +0.006 +0.004 +0.003 +0.005 +0.006 +0.005 +0.005 +0.005 +0.005 +0.004 AsymGS [36] +0.004 +0.001 −0.000 +0.004 +0.004 +0.004 +0.003 +0.004 +0.004 +0.005

[Figure 358]

DI2FIX

Target

GT

Reference

DIFIX

DIFIX+RobustNeRF

Target

GT

DI2FIX

DIFIX+RobustNeRF

DIFIX

Reference

- Fig. S.15: Failure cases of DI2FIX. Each row shows the reference view, the target view to be fixed, the results of DIFIX [91], DIFIX+RobustNeRF, and DI2FIX, and the ground truth. While DI2FIX improves visual quality in many regions, it can struggle under extremely severe degradations affecting the reference or target views (e.g., large floaters in both the target and reference views in row 1), or under cross-view confirmation bias (e.g., the vegetable-like distractor appearing in both the target and reference views in row 2), where artifacts may persist or scene structures cannot be fully recovered.

[Figure 359]

PSNR↑ LPIPS↓ SSIM↑

- Fig. S.16: Top-4 performing methods on each dataset. Results on DF3DV-1K identify AsymGS [36] and RobustSplat [20] as the most robust methods, followed by OCSplats [47] and DeGauss [83]. This ranking differs from those observed on On-thego [67] and RobustNeRF [69], highlighting the importance of large-scale evaluation. Interestingly, performance trends on DF3DV-1K largely follow publication chronology, with AsymGS [36] (NeurIPS 2025) outperforming RobustSplat [20], OCSplats [47], and DeGauss [83] (ICCV 2025), while the three ICCV 2025 methods exhibit similar performance in PSNR and SSIM. Such trends are less evident on existing datasets.

0 10 20 30 40 50 60

0 50 100 150 200 250 300

building dish

|passer-by accessory stationery<br><br>container<br><br>bike 3C product<br><br>bottle decoration plush toy home appliance cup dish<br><br>drink mug tube<br><br>makeup tool basket lid trolley<br><br>key mat<br><br>figure stick<br><br>| | |No.ofscenesperdistractortype| |
|---|---|---|---|---|
|spice bedding<br><br>plastic bag tag pot<br><br>smoke bouquet eyedrop<br><br>train<br><br>bread light rail<br><br>rope<br><br>bucket cube door<br><br>flag headset mask roll<br><br>sauce shadow swatter<br><br>vase aluminium foil<br><br>| | | | |
|brochure<br><br>| | | | |
|car component<br><br>| | | | |
|coaster<br><br>| | | | |
|dummy<br><br>| | | | |
|glass bottle<br><br>| | | | |
|grocery<br><br>| | | | |
|incense<br><br>| | | | |
|menu<br><br>| | | | |
|pedal<br><br>| | | | |
|rack<br><br>| | | | |
|robot<br><br>| | | | |
|rubber<br><br>| | | | |
|shelf<br><br>| | | | |
|statue<br><br>| | | | |
|teapot<br><br>| | | | |
|watering pot<br><br>| | | | |
|window<br><br>| | | | |
| | | | | |

body part

container statue

No.ofscenespertheme

No.ofscenesperdistractortype

car pack

accessory decoration

pack fruit

3C product bottle

clothing

seat toy

toy cutlery animal utensil

clothing plant

plush toy house

drink tree

bike flower

jar shelf

cleaning supply book

sign vegetable

home appliance path

spice can

fruit can ball

pot table

book cleaning supply

cup makeup tool

bedding car

plane bus seat

entrance mug

pole stationery

art machine

jar motorbike vegetable

shop bowl

figure kitchen appliance

mat playground

plaza road

tool truck

bridge cutlery

factory rack

rock trolley

glass cup bowl umbrella lighter snack clip coin

truck animal

balcony bread

garage oil

platform snack

stall umbrella

utensil wall

basket bin

cafe courtyard

spray egg leaf

engine fountain

gate instrument

mailbox parking lot

plant pot rocking horse

bin helmet plant art candle

sauce stair

bedroom bell

candle cannon

construction site deck

eyedrop facility

field garden

doll duster

glass cup headset

kitchen living room

lounge motorbike

hanger

passer-by pond

light name card

rail roller

scent diffuser toiletry

train transformer box

rolling shutter scooter

tube tunnel

advertisement ball

straw toiletry vitamin battery

bar set bathroom

board body part

bouquet burner

carpet carriage

cathedral censer

cage chess cone

chess cone

counter cylinder

door drawer

drone egg

fire glass pot

elevator fence

fire exit fire hose

glass bottle hanger

helicopter instrument

helmet home gym

junction key

partition plant pot receipt

ladder offering

parking meter pavilion

plane restaurant

safe scale

rock scale

sink sky

spray square

stand table

station surfboard

teapot temple

tool tower

thread wheelchair

traffic light trophy

viewpoint window

yarn

yarn

- Fig. S.17: Distribution of scenes by theme and distractor type. Number of scenes per theme (left) and per distractor type (right). DF3DV-1K includes 161 themes and 128 distractor types.

###### No. of scenes per scenario type

0 50 100 150 200 250

Other. Frontal.

Various. Color.

Semantic.

Large. Reflective.

Semi-transparent. Nighttime.

Shadow.

Slow-motion. Daily.

Semi-transient. Common.

Fluid. Air. Appearance.

###### Fig. S.18: Distribution of scenes by scenario. In addition to DF3DV-41, which provides a more uniform distribution of scenes across scenarios, we further provide the scenario label for each scene in the full dataset.

[Figure 360]

###### Fig. S.19: Low-quality distractor examples. DF3DV-1K includes casually captured distractors with motion blur, partial visibility, defocus, and fast-moving objects, reflecting realistic capture conditions.

