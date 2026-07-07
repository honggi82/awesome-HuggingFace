## DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling

#### Kairun Wen1∗†, Yuzhi Huang1∗, Runyu Chen1, Hui Zheng1, Yunlong Lin1, Panwang Pan1, Chenxin Li2, Wenyan Cong3, Jian Zhang1, Junbin Lu4, Chenguo Lin5, Dilin Wang6, Zhicheng Yan6, Hongyu Xu6, Justin Theiss6, Yue Huang1, Xinghao Ding1 , Rakesh Ranjan6, Zhiwen Fan3

* Equal Contribution; † Project Lead; Corresponding Author

# arXiv:2512.03000v2[cs.CV]3Dec2025

1XMU 2CUHK 3UT Austin 4UW 5PKU 6Meta

Project Website: https://dynamic-verse.github.io/

Video Frames Dynamic Content (Moving Object & Camera)

Annotations

[Figure 1]

……………………

……………………

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Object Mask & Category

Object Caption

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Scene Caption

Camera Caption

[Figure 10]

[Figure 11]

[Figure 12]

Metric-scale Camera

Metric-scale PointMap

Object Category: Running Lady

Camera Caption: The camera moves forward, following the running lady from behind, resulting in slightly unsteady motion with

shaking. As the video concludes, the camera stops tracking, tilts up, and pans left to reveal the scene ahead.

Object Caption:

An elderly lady with short white hair, wearing a vibrant multicolored blouse and black pants, walks with a steady, rhythmic gait. Their arms are slightly bent, holding a small object. Maintaining an upright posture with head tilted forward, they move at a consistent pace, suggesting focus, purpose, or familiarity with their path.

Scene Caption:

A lively scene inside a spacious, well-lit restaurant, characterized by wooden floors, large windows with natural light, and a mix of modern and rustic decor including exposed brick walls and furniture. An elderly lady walks purposefully through the warm, inviting space, her stride steady, bustling with diverse patrons seated enjoying meals or conversations at many tables set with cutlery and glasses. The ambiance is cozy yet sophisticated.

Figure 1: The overview of physically-aware multi-modal world modeling framework DynamicVerse.

### Abstract

Understanding the dynamic physical world, characterized by its evolving 3D structure, real-world motion, and semantic content with textual descriptions, is crucial for human-agent interaction and enables embodied agents to perceive and act within real environments with human-like capabilities. However, existing datasets are often derived from limited simulators or utilize traditional Structurefrom-Motion for up-to-scale annotation and offer limited descriptive captioning, which restricts the capacity of foundation models to accurately interpret real-world dynamics from monocular videos, commonly sourced from the internet.

To bridge these gaps, we introduce DynamicVerse, a physical-scale, multimodal 4D world modeling framework for dynamic real-world video. We employ large vision, geometric, and multimodal models to interpret metric-scale static geometry, real-world dynamic motion, instance-level masks, and holistic descriptive captions. By integrating window-based Bundle Adjustment with global optimization, our method converts long real-world video sequences into a comprehensive 4D multimodal format. DynamicVerse delivers a large-scale dataset consisting of 100K+

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

videos with 800K+ annotated masks and 10M+ frames from internet videos. Experimental evaluations on three benchmark tasks, namely video depth estimation, camera pose estimation, and camera intrinsics estimation, demonstrate that our 4D modeling achieves superior performance in capturing physical-scale measurements with greater global accuracy than existing methods.

### 1 Introduction

Humans inhabit a dynamic 3D world where geometric structure and semantic content evolve over time, constituting a 4D reality (spatial with temporal dimension). Understanding this dynamic environment is fundamental for developing advanced AI applications in fields such as robotics [1, 2, 3, 4, 5, 6], extended reality [7, 8, 9], human–agent interaction [10, 11], and digital twins [12, 13]. However, building generalizable foundation models for these downstream tasks faces a longstanding challenge: acquiring high-quality, ground-truth 4D datasets from real-world environments, given that datadriven solutions increasingly demand 4D data while its collection using multiple sensors remains non-scalable. This raises the question: Can we develop an automated pipeline capable of generating a real-world 4D dataset at scale?

Current real-world 4D data primarily focus on indoor scenes [14, 15] or autonomous driving scenarios [16], where geometry capture is straightforward, but their diversity is limited. Even synthetic 4D data [17, 18, 19, 20], while controllable, often lack the fidelity and complexity required to truly represent the real world, resulting in a notable simulation-to-real gap. Moreover, physically-aware multimodal annotations, including metric-scale 3D geometry, detailed representations of non-rigid actors (e.g., object size, mask and bounding box, etc.), as well as descriptive captions of dynamic content (i.e., object, camera and scene), are often absent [21, 22]. This limited data landscape, especially when contrasted with the progress fueled by large-scale datasets in modalities like images, videos, and language, underscores the compelling need for a large-scale, diverse, physically-aware, and semantically rich annotated multi-modal dataset for 4D scene understanding.

Against this background, this paper aims to generate scalable, physically-aware, and multimodal annotations from massive monocular video data (Fig. 1) for numerous potential applications, such as enhancing 4D Vision-Language Models [23], facilitating advanced 3D-aware video generation [24], and enabling linguistic 4D Gaussian Splatting [25]. However, achieving this goal is not trivial. To the best of our knowledge, there is currently a significant lack of rich and diverse 4D datasets (see Tab. 1) adequate for these demanding tasks. To address this data scarcity, we introduce DynamicGen, a novel automated data curation pipeline (Fig. 3) designed to generate physically-aware multi-modal 4D data at scale. This pipeline contains two main stages: (1) metric-scale geometric and moving object recovery (i.e., object category and mask) from raw videos, and (2) hierarchical dynamic content (i.e., object, camera and scene) detailed caption generation. Specifically, the pipeline curates diverse real-world monocular video sources; employs a filtering strategy to remove outliers such as camera motion intensity; integrates multiple foundation models (i.e., VFMs, VLMs, LLMs, GFMs) for initial frame-wise annotation; applies dynamic bundle adjustment to jointly minimize global photometric error; and concludes with dynamic content captioning at three granularities and human-in-the-loop quality review to ensure annotation semantic accuracy.

The resulting multi-modal 4D dataset, termed DynamicVerse (Fig. 1), comprises over 100K distinct 4D scenes, 800K masklets, and 10M video frames. Each scene is extensively annotated with multiple modalities: metric-scale point maps, camera parameters, object masks with corresponding categories, and detailed descriptive captions. We evaluate DynamicGen through three benchmarks: video depth estimation, camera pose estimation, and camera intrinsics estimation. We demonstrate the generalization capability of DynamicGen to process web-scale video data and extract multi-modal information qualitatively. We also conduct human study and GPT-assited evaluation to validate the quality of generated captions.

Our main contributions are summarized as follows:

- • We develop DynamicGen, a novel automated data curation pipeline designed to generate physically-aware multi-modal 4D data at scale. This pipeline contains two main stages: (1) metric-scale geometric and moving object recovery from raw videos, and (2) hierarchical detailed semantic captions generation at three granularities (i.e., object, camera and scene). Powered by

- foundation models (i.e., VFMs, VLMs, LLMs, GFMs), DynamicGen efficiently generate 4D data at scale, thus addressing the critical scalability, physical reality and modality diversity limitations of traditional 4D data curation.
- • We introduce DynamicVerse, a large-scale 4D dataset featuring diverse dynamic scenes accompanied by rich multi-modal annotations including metric-scale point maps, camera parameters, object masks with corresponding categories, and detailed descriptive captions. DynamicVerse encompasses 100K+ 4D scenes coupled with 800K+ masklets, sourced through a combination of massive 2D video datasets and existing 4D datasets. This represents a significant improvement in terms of data scale, scene and modality diversity compared to prior 4D datasets.
- • We validate DynamicGen through three benchmarks: video depth estimation, camera pose and intrinsics estimation. We demonstrate the generalization capability of DynamicGen to process web-scale videos and extract multi-modal information qualitatively. We also conduct human study and GPT-assited evaluation to validate the quality of generated captions.

### 2 Related Work

#### Table 1: Comparison of DynamicVerse with large-scale 2D video datasets and existing 4D scene datasets. DynamicVerse expands the data scale and annotation richness compared to prior works.

Numerical Statistics Provided Annotations Detailed Features

ObjectCategory

CameraCaption

SemanticMask

ObjectCaption

DynamicType

SceneCaption

InstanceMask

Metric-scale?

Real-world?

SceneType

#Masklets

Depthmap

#Frames

#Videos

Camera

Dataset Name

- 2D Video Dataset DAVIS2017 [26] 0.2K 10.7K 0.4K - - - Youtube-VIS [27] 3.8K - 8,171 - - - UVO-dense [28] 1.0K 68.3K 10.2K - - - VOST [29] 0.7K 75.5K 1.5K - - - BURST [30] 2.9K 195.7K 16.1K - - - MOSE [31] 2.1K 638.8K 5.2K - - - SA-V [32] 50.9K 4.2M 642.6K - - - MiraDATA [33] 330K - - - - - 4D Scene Dataset T.Air Shibuya [34] 7 0.7K - Mixed Street Synthetic Yes MPI Sintel [35] 14 0.7K - - Scripted Synthetic FlyingThings3D [36] 220 2K - Mixed Objects Synthetic Waymo [16] 1,150 200K - Outdoor Driving Real-world Yes CoP3D [14] 4,200 600K - Mixed Pets Real-world Stereo4D [37] 110,000 10,000K - Mixed S. fisheye Real-world Yes PointOdyssey [17] 159 200K - Mixed Realistic Synthetic Yes Spring [18] 47 6K - Mixed Realistic Synthetic Yes Dynamic Replica [19] 524 145K - Indoor Realistic Synthetic Yes MVS-Synth [20] 120 12K - Outdoor Urban Synthetic Yes RealCam-Vid [21] 100K - - Mixed Realistic Synthetic Yes DynPose-100K [22] 100K 6,806K - Mixed Realistic Synthetic Yes DynamicVerse 100K+ 13.6M 800K+ Mixed Realistic Real-world Yes

Multi-modal foundation models. The development of numerous large foundation models in recent years has yielded remarkable performance across multiple tasks such as depth estimation [38, 39, 40, 41, 42], multi-view stereo [43, 44, 45], detection and segmentation [46, 47, 48, 32], human parsing [49], optical flow estimation [50, 51], and point tracking [52, 53, 40]. We propose that these models are highly applicable to achieving holistic 4D understanding, and unifying them within a single framework represents a promising direction for advancing tasks like nonrigid structure from motion. Our DynamicGen pipeline implements this idea by integrating the following pretrained components: UniDepthv2 [54] for geometry initialization, CoTracker3 [53] and UniMatch [51] for correspondence initialization, and Qwen2.5-VL [55] and SA2VA [56] for dynamic object segmentation. This integration, coupled with multi-stage optimization and regularization, allows us to extract accurate metric-scale camera poses and 4D geometry from monocular video. Similar to our method, the concurrently developed Uni4D [57] captures 4D geometry and pose, but it suffers from limited data modalities and discontinuous geometric estimates. In contrast, our DynamicGen pipeline not only produces globally refined dense 4D geometry but also supports moving object recovery (i.e., object category and mask) and provides fine-grained dynamic content (i.e., object, camera and scene) caption annotations.

[Figure 13]

[Figure 14]

[Figure 15]

(a) Moving objects type.

(b) Camera motion type.

[Figure 16]

[Figure 17]

(c) Object actions type. (e) Data source of DynamicVerse.

(d) Dynamic Scene (environments) type.

Figure 2: The statistics and data source of DynamicVerse.

Multi-modal datasets. The development of large-scale multi-modal datasets has proven essential for advancing model performance across numerous domains, including language, image-text (e.g., LAION [58, 59], Conceptual Captions [60], WebImageText [61]), and video understanding (e.g., DAVIS2017 [26], Youtube-VIS [27], UVO-dense [28], VOST [29], BURST [30], MOSE [31], SAV [32], MiraDATA [33]). Extending this success to holistic 4D understanding requires datasets that capture the dynamic 3D world with rich, multi-modal annotations. Existing 4D datasets, whether from early reconstruction efforts [17, 18, 19, 20] (limited diversity) or recent large-scale posed video collections like RealCam-Vid [21] and DynPose-100K [22] (lacking detailed geometry and semantics beyond pose), and even OBJAVERSE [62] (limited content), fall short of providing the comprehensive multi-modal information needed. Our DynamicVerse dataset bridges this gap by offering extensive multi-modal annotations, including metric-scale depth, camera parameters, instance segmentation with labels, and descriptive captions, specifically designed to facilitate advanced 4D research.

### 3 DynamicVerse

Overview DynamicVerse is a physical-scale, multi-modal 4D modeling framework for real-world video, which contains a novel automated data curation pipeline and corresponding large-scale 4D dataset. The DynamicGen pipeline (Fig. 3) contains two main stages: (1) metric-scale geometric and moving object recovery (i.e., object category and mask) from raw videos, and (2) hierarchical dynamic contents (i.e., object, camera and scene) detailed caption generation. This pipeline primarily consists of five steps: 4D scene curation (in Sec. 3.1), data filtering strategy (in Sec. 3.2), moving object recovery (in Sec. 3.3), dynamic bundle adjustment (in Sec. 3.4) and dynamic content caption generation (in Sec. 3.5). The resulting DynamicVerse dataset comprises over 100K distinct 4D scenes, 800K masklets, and 10M video frames. The data statistics and collection of DynamicVerse are illustrated in Fig. 2.

#### 3.1 4D scene curation

To address the scarcity of available 4D scene data, DynamicGen unifies video data from various real-world video datasets, including DAVIS2017 [26], Youtube-VIS [27], UVO-dense [28], VOST [29], BURST [30], MOSE [31] and SA-V [32], alongside existing synthetic 4D datasets from PointOdyssey [17], Spring [18], Dynamic Replica [19], MVS-Synth [20], RealCam-Vid [21] and DynPose-100K [22]. The inclusion of these datasets is mainly motivated by their potential as scalable data sources for 4D scene understanding.

#### 3.2 Data filtering strategy

Data filtering is a critical step for identifying video data suitable for subsequent dynamic bundle adjustment. This process presents challenges due to the noisy quality and inherent variability of video data, which impedes the precise selection of high-quality sequences. To address this, we developed a filtering strategy incorporating several distinct criteria: proximal depth, focal-length stability, video blur, camera motion smoothness, and non-perspective distortion. Each of these aspects is quantified

- Stage 1 : Generating Metric-scale Camera, Video Depth, Object Category, Object Mask and Size
- Stage 2 : Generating Fine-grained Captions For Moving Objects, Cameras and Dynamic Scenes

[Figure 18]

[Figure 19]

Data Filter Strategy Dynamic Bundle Adjustment

Powerful Foundation Models

Raw Data Collection

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Proximal Depth Verification

[Figure 31]

[Figure 32]

Dynamic Masking

[Figure 33]

[Figure 34]

Dynamic Object Coverage

[Figure 35]

[Figure 36]

[Figure 37]

SegAnyMo

UniDepth-V2

Focal-Length Stability

[Figure 38]

Coarse Camera Initialization

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

###### Random forest

Video Blur Degree Camera Motion Smoothness

[Figure 44]

[Figure 45]

[Figure 46]

Static Area Bundle Adjustment

Raw 2D video datasets with GT instance mask & semantic label

UniMatch CoTracker-V3

[Figure 47]

VLM Judgement

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Non-Rigid

[Figure 53]

[Figure 54]

Moving Object Recovery

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Bundle Adjustment

SAM2 & Variants

VGGT

[Figure 59]

[Figure 60]

VLM-based object recognition Semantic-based Segmentation Physical-aware Size Extraction

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Sliding Window Global Refinement

Raw dynamic 3D datasets

with GT camera & depth

Epipolar Errors

Language Models

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Dynamic Scene Caption

Moving Object Caption

[Figure 73]

###### Camera Motion Caption

[Figure 74]

[Figure 75]

Prompt: Given the <raw video> , <moving instance masks> , <instance label>

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Prompt: Given the <raw video> , <moving instance mask> and

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Prompt: Given the <raw video> , <instance label> , <moving instance caption> , and <dynamic scene caption> . Provide a description of the camera motion.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

and <moving instance caption> . Provide a description of the dynamic scene.

[Figure 90]

<instance label> . Provide a description of the moving instance. Focus on its action and attributes (e.g., color, shape, material).

[Figure 91]

[Figure 92]

Focus on the scene type, instance-scene interaction, instance’s motion pattern, etc.

[Figure 93]

[Figure 94]

[Figure 95]

Response (scene-level)

Response (object-level)

[Figure 96]

<think> Alright, let's break down the attributes step by step: First, identify the scene type: an indoor trampoline park, … . Then, note the primary interactive entities: a young boy, … . Next, observe the instance’s motion patterns: jumps, spins, …. Then consider the static background components: Padded Walls, … . Finally, take into account the spatial relationships and interactions among different moving instances and the scene: the boy actively interacts on the trampoline by jumping and performing various acrobatic actions. </think>. <answer> {...} </answer>

[Figure 97]

<think> Alright, let's break down the attributes step by

[Figure 98]

step: First, observe the boy’s action: stand, crouch, … . Then, see the boy's attire: light green shirt, … . Next, note the shape and movement: agile, … . Then, consider the material of the clothing: lightweight, … . Finally, take into account the setting and the boy's engagement: indoor trampoline park, … .</think>. <answer> {...} </answer>

[Figure 99]

Response

[Figure 100]

<answer> {

The camera smoothly pans right, tracking the subject from the side with minimal shaking. }</answer>

[Figure 101]

Raw Videos

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

LLM-rephrased

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

LLM-rephrased

[Figure 116]

<answer> {

[Figure 117]

A young boy in a vibrant green vest with red trim and dark pants performs a dynamic sequence. Beginning in profile, they turn, extending arms and balancing on ready legs. Fluidly shifting from

[Figure 118]

<answer> {

[Figure 119]

LLM-aligned

Inside a vibrant indoor trampoline park, a young boy in a turquoise shirt dynamically jumps, spins, and flips. Brightly lit, the park features multiple trampolines with blue padding and colorful play structures like orange and red climbing frames. Other children in white and pink shirts enjoy the lively, energetic atmosphere as the boy's joyful acrobatics highlight the scene's playful activity and vibrant setting. } </answer>

[Figure 120]

<answer> {

The camera smoothly pans right, tracking the young boy in the indoor trampoline park with minimal shaking. }</answer>

standing to a focused crouch, their movements

display agility and control. The energetic sequence concludes with them seated, relaxed yet purposeful, embodying athleticism and grace throughout the transitions. } </answer>

[Figure 121]

Instance Masks

“A Young Boy”

[Figure 122]

Instance Label

Figure 3: The physically-aware multi-modal 4D data generation pipeline DynamicGen.

by a normalized score. We combine these scores as features and employ a Random Forest model to predict a video quality score ranging from 0 to 5. For model training, we manually annotated approximately 1,000 videos, assigning scores between 0 (indicating largely unsuitable, poor quality or insufficient dynamics) and 5 (indicating highly suitable, good quality and sufficient dynamics). We further apply VLM-based judgment to automatically exclude unsuitable videos before reconstruction.

#### 3.3 Moving object recovery

To accurately identify the main dynamic objects within a video, we integrated multiple foundation models to achieve reliable segmentation. Specifically, our pipeline first employs Qwen2.5-VL [63] to identify moving objects and determine their semantic categories. These categories are then used to prompt SA2VA [56] for generating corresponding object masks. Leveraging the obtained object masks and geometric annotations, we can apply physical-aware size extraction to annotate the 3D bounding box for moving objects.

#### 3.4 Dynamic bundle adjustment

Leveraging the high-quality RGB filtered videos, we employed a robust dynamic bundle adjustment method for annotating metric-scale camera parameters and point maps. This task is challenging due to dynamic objects occluding the static scene and static scene appearance changes hindering correspondence estimation. To effectively addresses both difficulties, we design a multi-stage optimization framework, see Fig. 3, including: (1) dynamic masking, (2) coarse camera initialization, (3) tracking-based static area bundle adjustment, (4) tracking-based non-rigid bundle adjustment, and (5) flow-based sliding window global refinement. Compared with traditional Structure-from-Motion techniques [64] and DUSt3R-based methods [65], our framework not only can handle massive video data with different resolutions but also yield metric-scale results by leveraging the full power of various foundation models.

Formulation Given T video RGB frames I = (I1,...,IT) with resolution H × W, we aim to estimate for each timestep t = 1,...,T: per-frame point map Xt ∈ RH×W×3, camera intrinsics Kt, and camera pose Pt = [Rt|Tt], where Rt and Tt denote the t-th camera’s rotation and translation,

Video SegAnyMo GPT+DEVA DynamicGen

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

Figure 4: Qualitative Results of Moving Object Segmentation. We show qualitatively some of our segmentation results on the Youtube-VIS dataset compared with other methods.

respectively. Here, X contains static points Xstatic and dynamic points Xdyn. We assume all frames share the same intrinsics K where we optimize focal lengths fx and fy. The overall cost function is formulated as follows:

CBA(P,Xstatic) + Cflow(Xstatic) + CNR(Xdyn) + Cmotion(Xdyn) + Ccam(R) (1)

where CBA(P,Xstatic) and Cflow(Xstatic) are bundle adjustment terms measuring the reprojection error between static correspondences and the static 3D structure Xstatic. CNR(Xdyn) is a non-rigid structure-from-motion term evaluating the consistency of the dynamic point cloud with its tracklets. Regularization is applied to camera motion smoothness through Ccam(P) and to the dynamic structure and motion via Cmotion(Xdyn). Each term participates in different optimization stages, which are described below. Detailed explanations of the cost terms are provided in the supplementary material.

- Stage I: Dynamic masking We first extract dynamic masks to filter out the dynamic points for static area bundle adjustment. Specifically, we use semantic-based and motion-based method to obtain dynamic masks M = {Mt}Tt=1 = {Mtsem ∪ Mtflow}Tt=1. For the segmentation-based approach, we

use the generated moving object masks {Mtsem}Tt=1 in Sec. 3.3. For the flow-based approach, we employ Unimatch [51] to obtain dense optical flow predictions and compute per-frame epipolar error maps [66], which indicate the likelihood of pixels belonging to the dynamic foreground. Then we can obtain dynamic masks Mflow = [E1,E2,...,ET] by thresholding on these epipolar error maps.

- Stage II: Coarse camera initialization In this stage, we start camera initialization by obtaining

video depth D = {Dt}Tt=1 and dense pixel motion Z = {Zk}Kk=0. For video depth estimation, we use UniDepthV2 [54], a monocular depth estimation network, to estimate initial depth maps D and

initial camera intrinsics Kinit. For dense pixel motion estimation, we utilize Co-TrackerV3 [53] for its robustness. We apply Co-Tracker bi-directionally on a dense grid every 10 frames to ensure thorough coverage. We filter and classify tracklets using segmentation masks yielding a set of correspondent point trajectories {Zk ∈ RT×2}Kk=0 at visible time steps determined by Co-Tracker. Combining D and Z allows us to establish 2D-to-3D correspondences. This allows us to initialize and tune camera parameter P by minimizing the following cost function with respect to camera parameters only. Specifically, we can unproject each video frame’s depth at time t back to 3D and minimize the following cost function:

min

P

(t′,t) Zk∈¬M

∥Zk,t′ − πK(πK−1(Zk,t,Dt,ξt),ξt′)∥22 (2)

where πK−1 is the unprojection function that maps 2D coordinates into 3D world coordinates using estimated depth Dt. We perform this over all pairs within a temporal sliding window of 5 frames. Given camera initialization Pˆ, we unproject our depth prediction into a common world coordinate system, which provides an initial 4D structure Xˆ . This is used as initialization for later optimization.

- Stage III: Static area bundle adjustment We jointly optimizes camera pose and static geometry by minimizing the static component-related energy in a bundle adjustment fashion. Formally speaking, we solve the following:

min

P,Xstatic

CBA(P,Xstatic;Z,M) + Ccam(R) (3)

By enforcing consistency with each other, this improves both the static geometry and the camera pose quality. We perform a final scene integration by unprojecting correspondences into 3D using improved pose and filtering outlier noisy points in 3D.

- Stage IV: Non-rigid bundle adjustment Given the estimated camera pose, this stage focuses on inferring dynamic structure. Note that we freeze camera parameters in this stage, as we find that incorrect geometry and motion evidence often harm camera pose estimation rather than improve it. Additionally, enabling camera pose optimization introduces extra flexibility in this ill-posed problem, harming robustness. Formally speaking, we solve the following:

min

Xdyn

CNR(Xdyn;P,Z,M) + Cmotion(Xdyn) (4)

We initialize Xdyn using video depth and our optimized camera pose from last step. This energy optimization might still leave some high-energy noisy points, often from incorrect cues, motion boundaries, or occlusions. We filter these outliers based on their energy values in a final step. To further densify the global point cloud, enabling each pixel to correspond to a 3D point, we perform depth-based interpolation by computing a scale offset.

- Stage V: Sliding window global refinement Given the estimated optical flow, this stage focuses on refining static structure. Note that we freeze camera parameters in this stage. Formally speaking, we solve the following:

Cflow(Xstatic) (5)

min

Xstatic

With consideration for accuracy and efficiency, the sliding window global refinement is capable of significantly enhancing the multi-view consistency of static points and generalizing effectively to real-world 4D scenes. The detailed process can be found in the appendix.

#### 3.5 Dynamic Content Caption Generation

Drawing upon the emphasis placed by LEO [67] and SceneVerse [68] on the criticality of caption quality and granularity for comprehensive scene understanding, we design captions at three specific levels: object, scene, and camera. Object captioning focuses on detailed object motion, scene captioning describes object-scene interactions, and camera captioning conveys intricate camera movement. To augment the caption, Large Language Models (LLMs) are employed to automatically rephrase initial captions and align them with these three granularity levels. Finally, to ensure data quality, human verification is conducted to filter out low-quality caption annotations.

Moving object captioning. Moving object captions provide detailed descriptions crucial for object grounding. However, prior datasets often have incorrect temporal alignment [68] or insufficient detail [17, 69], while current video captioning methods yield only simple (e.g., Panda-70M [70]) or non-localized descriptions (e.g., Qwen2.5-VL [63]). To address these limitations and generate detailed, accurate captions for individual objects, we utilize DAM [71], known for its superior capabilities. Given RGB videos and corresponding object masks, DAM [71] generates detailed and temporally aligned object descriptions through carefully designed prompts, enabling precise grounding and richer scene understanding.

Dynamic scene captioning. Scene-level captions are designed to capture global information, depicting the key objects within the scene along with their associated actions, interactions, and functionalities. For a comprehensive understanding of the entire dynamic scene, we utilize Qwen2.5VL [63] for dynamic scene captioning. To obtain more detailed, fine-grained, and accurate captions, we propose the use of structured captions. This process involves leveraging the fine-grained moving object captions as auxiliary input and employing specific prompting to generate the final scene-level descriptions. In the design of the prompts, we discovered that an explicit Hierarchical Prompt Design [72] significantly aids the Qwen2.5-VL[63] in comprehending its role, its expected format, and its operational boundaries. This approach contributes to the stabilization of the output’s format and enhances the overall quality of the results.

Camera motion captioning. Camera Motion Captioning aims to describe the camera’s trajectory and movement patterns. Using the powerful VLM [73], we analyze the sequence of inter-frame transformations to identify key motion types like panning, tilting, zooming, and dolly movements. This kinematic information is then used to generate natural language descriptions, potentially leveraging template-based generation or LLM prompting, to convey how the viewpoint changes over time.

Caption rephrasing. Following the generation of three distinct caption types (object, scene, and camera motion), a Large Language Model (LLM) [63] is employed to jointly process them. This step aligns the dynamic content descriptions across caption types and refines their phrasing to enhance overall consistency and readability.

Human-in-the-loop quality review. To provide a faithful comparison against larger pretrained models, human evaluation was used. Addressing persistent errors from source annotation inaccuracies, we implemented an iterative human-in-the-loop verification during caption construction to identify errors, trace sources, and revise/remove problematic data.

### 4 Experiments

In this section , we present experimental results to evaluate the robustness of our DynamicGen pipeline. Due to the page limit, we direct readers to the appendix for implementation details, more qualitative results, and more experimental analyses.

#### 4.1 Video Depth Estimation

To evaluate video depth estimation accuracy, we assess several baseline methods, including metric depth predictors such as Metric3Dv2 [74], Depth-Pro [38], DepthCrafter [39], and Unidepth [41], which operate without scale or shift alignment. We also consider joint 4D modeling approaches, including MonST3R [65] and RCVD [75]. Evaluations are conducted on the Sintel [35] and KITTI [77] datasets, following standard protocols [39] by applying global shift and scale alignment to the predicted depth maps. We report absolute relative error (Abs Rel) and the percentage of inlier points (δ < 1.25), with all methods undergoing least-squares alignment in disparity space. As shown in Tab. 2, DynamicGen achieves the best overall performance across all datasets and evaluation metrics. In particular, it consistently outperforms prior approaches in both absolute accuracy and geometric consistency, demonstrating strong generalization to diverse and dynamic scenes. As illustrated in Fig. 5, MonST3R consistently struggles with object geometry reconstruction, producing distorted

All research undertaken at Meta AI was limited to general guidance on model architectural design. Meta did not participate in any model training activities. Fan, Z. contributed to this project prior to the NeurIPS submission deadline.

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

Monst3r Uni4D Ours Monst3r Uni4D Ours

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

| |
|---|

Monst3r Uni4D Ours

Monst3r Uni4D Ours

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

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Monst3r Uni4D Ours Monst3r Uni4D Ours

Figure 5: Visual comparisons of 4D reconstruction on in-the-wild data.

- Table 2: Video depth evaluation on Sintel and KITTI datasets. Bold and underlined values indicate best and second best results, and blue-shaded cells denote our method.

Sintel KITTI Alignment Category Method Abs↓ δ1.25↑ Abs↓ δ1.25↑ Per-sequence scale

Monst3r [65] 0.344 55.9 0.089 91.4 Uni4D [57] 0.289 64.9 0.086 93.3

Joint depth & pose

Depth-pro [38] 0.280 60.5 0.080 94.2 Metric3D [74] 0.205 71.9 0.039 98.8

Single-frame depth

Video depth DepthCrafter [39] 0.231 69.0 0.112 88.4

Per-sequence scale & shift

Robust-CVD [75] 0.358 49.7 0.182 72.9 CasualSAM [76] 0.292 56.9 0.113 88.3 Uni4D [57] 0.216 72.5 0.098 89.7 DynamicGen(Ours) 0.205 72.9 0.091 91.2

Joint video depth & pose

shapes and noisy dynamic masks. Uni4D also exhibits mask imprecision. DynamicGen, however, achieves the cleanest dynamic segmentations and the strongest dynamic/static reconstructions.

#### 4.2 Camera Pose Estimation

We evaluate our method against recent dynamic scene pose estimation approaches, including learningbased visual odometry (e.g., LEAP-VO [78], DPVO [79]) and joint depth-pose optimization methods (e.g., Robust-CVD [75], CasualSAM [76], MonST3R [65]). Experiments are conducted on the Sintel [35] and TUM-dynamics [80] datasets, following LEAP-VO’s split for Sintel and subsampling the first 270 frames of TUM-dynamics, as done in MonST3R. Camera trajectories are aligned using Umeyama alignment [81], and we report Absolute Trajectory Error (ATE), Relative Translation Error (RPE trans), and Relative Rotation Error (RPE rot). As shown in Tab. 3, DynamicGen consistently achieves state-of-the-art results across all metrics and datasets, outperforming existing methods in both translation and rotation accuracy.

#### 4.3 Camera Intrinsics Estimation

Camera intrinsics are typically unavailable for most casual videos, especially those sourced from the Internet. However, accurate intrinsics are critical for reliable pose estimation and 3D reconstruction.

- Table 3: Camera Pose Evaluation on Sintel and TUM-dynamic datasets. Bold and underlined values indicate best and second best results, and blue-shaded cells denote our method.

Sintel TUM-dynamics Category Method ATE↓ RPE trans↓ RPE rot↓ ATE↓ RPE trans↓ RPE rot↓

DPVO [79] 0.171 0.063 1.291 0.019 0.014 0.406 LEAP-VO [78] 0.035 0.065 1.669 0.025 0.031 2.843

Pose only

Robust-CVD [75] 0.368 0.153 3.462 0.096 0.027 2.590 CasualSAM [76] 0.137 0.039 0.630 0.036 0.018 0.745 Monst3r [65] 0.108 0.043 0.729 0.108 0.022 1.371 Uni4D [57] 0.110 0.032 0.338 0.012 0.004 0.335 DynamicGen(Ours) 0.108 0.029 0.282 0.012 0.004 0.331

Joint depth & pose

To assess this, we evaluate focal length estimation accuracy on the Sintel dataset, with results summarized in Tab. 4. UniDepth predicts depth and focal length from a single image, while Dust3r processes sequential frames but is trained under classical multi-view settings and fails to generalize well to dynamic scenes. In contrast, DynamicGen demonstrates strong generalization to dynamic content and achieves the best performance in both Absolute Focal Error (AFE) and Relative Focal Error (RFE), setting a new state-of-the-art for focal length estimation in unconstrained video scenarios.

Table 5: Dynamic Scene Caption evaluation.

Table 4: Camera intrinsics estimation.

Method Acc.↑ Com.↑ Con.↑ Rel.↑ Avg.↑ Direct Output 79.28 76.65 73.23 80.33 77.37 + SAKFE 80.23 77.46 74.01 81.45 78.29 + HP 82.57 81.42 71.17 82.56 79.43 + Rephrasing 82.48 80.50 71.86 83.27 79.53 + COT 84.38 82.09 75.87 85.56 81.97

Method AFE(px)↓ RFE(%)↓

UniDepth [41] 447.4 0.357 Dust3r [43] 434.0 0.364 DynamicGen(Ours) 413.1 0.241

#### 4.4 Caption Quality Evaluation

To assess caption quality, we sampled 100 videos from the SA-V dataset [32]. As presented in Table 5, our experimental results indicate that integrating semantic-aware key frame extraction (SAKFE), hierarchical prompting (HP), caption rephrasing, and Chain-of-Thought (CoT) prompting [82] significantly enhances the quality of dynamic scene captions generated by Vision-Language Models (VLMs). We evaluated caption quality using the LLM-as-Judge metric G-VEval [83], conducting ten independent evaluations to ensure robust average results. The resulting captions exhibited notable improvements across accuracy, completeness, conciseness, and relevance, confirming the effectiveness of these strategies for improving caption quality in this task.

### 5 Conclusion

In this work, we address key limitations in traditional 4D data curation regarding scalability, physical realism, and modality diversity. We introduce DynamicGen, an automated pipeline leveraging foundation models for video filtering, metric-scale geometry and motion recovery, and hierarchical semantic captioning from raw videos. DynamicGen’s capabilities are validated through standard benchmarks on video depth and camera pose/intrinsics estimation, qualitative analyses on diverse web videos, and human/LLM-based evaluations confirming caption quality. Utilizing DynamicGen, we construct DynamicVerse, a large-scale 4D dataset with over 100K dynamic scenes and rich physically grounded multimodal annotations. Together, this work offers a scalable 4D data generation methodology and a comprehensive new resource to advance 4D scene understanding.

### References

- [1] Richard A Newcombe, Dieter Fox, and Steven M Seitz. Dynamicfusion: Reconstruction and tracking of non-rigid scenes in real-time. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 343–352, 2015.
- [2] Chao Yu, Zuxin Liu, Xin-Jun Liu, Fugui Xie, Yi Yang, Qi Wei, and Qiao Fei. Ds-slam: A semantic visual slam towards dynamic environments. In 2018 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 1168–1174. IEEE, 2018.
- [3] Berta Bescos, José M Fácil, Javier Civera, and José Neira. Dynaslam: Tracking, mapping, and inpainting in dynamic scenes. IEEE robotics and automation letters, 3(4):4076–4083, 2018.
- [4] Linhui Xiao, Jinge Wang, Xiaosong Qiu, Zheng Rong, and Xudong Zou. Dynamic-slam: Semantic monocular visual localization and mapping based on deep learning in dynamic environment. Robotics and Autonomous Systems, 117:1–16, 2019.
- [5] Jiahui Huang, Sheng Yang, Zishuo Zhao, Yu-Kun Lai, and Shi-Min Hu. Clusterslam: A slam backend for simultaneous rigid body clustering and motion estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5875–5884, 2019.
- [6] Jesse Morris, Yiduo Wang, and Viorela Ila. The importance of coordinate frames in dynamic slam. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 13755–13761. IEEE, 2024.
- [7] Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, and Chuang Gan. Tesseract: Learning 4d embodied world models. arXiv preprint arXiv:2504.20995, 2025.
- [8] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
- [9] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023.
- [10] Pascale Fung, Yoram Bachrach, Asli Celikyilmaz, Kamalika Chaudhuri, Delong Chen, et al. Embodied ai agents: Modeling the world. arXiv preprint arXiv:2506.22355, 2025.
- [11] Ying Zheng, Lei Yao, Yuejiao Su, Yi Zhang, Yi Wang, Sicheng Zhao, Yiyi Zhang, and Lap-Pui Chau. Embodied ai: A survey on the evolution from perceptive to behavioral intelligence. arXiv preprint arXiv:2502.04809, 2025.
- [12] Panwang Pan, Zhuo Su, Chenguo Lin, Zhen Fan, Yongjie Zhang, Zeming Li, Tingting Shen, Yadong Mu, and Yebin Liu. Humansplat: Generalizable single-image human gaussian splatting with structure priors. Advances in Neural Information Processing Systems, 37:74383–74410, 2024.
- [13] Hezhen Hu, Zhiwen Fan, Tianhao Wu, Yihan Xi, Seoyoung Lee, Georgios Pavlakos, and Zhangyang Wang. Expressive gaussian human avatars from monocular rgb video. arXiv preprint arXiv:2407.03204, 2024.
- [14] Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ignacio Rocco, Natalia Neverova, Andrea Vedaldi, and David Novotny. Common pets in 3d: Dynamic new-view synthesis of real-life deformable categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4881–4891, 2023.
- [15] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400, 2024.

- [16] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.
- [17] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19855–19865, 2023.
- [18] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn. Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4981–4991, 2023.
- [19] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13229–13239, 2023.
- [20] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view stereopsis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2821–2830, 2018.
- [21] Guangcong Zheng, Teng Li, Xianpan Zhou, and Xi Li. Realcam-vid: High-resolution video dataset with dynamic scenes and metric-scale camera movements. arXiv preprint arXiv:2504.08212, 2025.
- [22] Chris Rockwell, Joseph Tung, Tsung-Yi Lin, Ming-Yu Liu, David F Fouhey, and Chen-Hsuan Lin. Dynamic camera poses and where to find them. arXiv preprint arXiv:2504.17788, 2025.
- [23] Hanyu Zhou and Gim Hee Lee. Llava-4d: Embedding spatiotemporal prompt into lmms for 4d scene understanding, 2025.
- [24] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025.
- [25] Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, and Hanspeter Pfister. 4d langsplat: 4d language gaussian splatting via multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [26] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.
- [27] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5188–5197, 2019.
- [28] Weiyao Wang, Matt Feiszli, Heng Wang, and Du Tran. Unidentified video objects: A benchmark for dense, open-world segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10776–10785, 2021.
- [29] Pavel Tokmakov, Jie Li, and Adrien Gaidon. Breaking the" object" in video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22836–22845, 2023.
- [30] Ali Athar, Jonathon Luiten, Paul Voigtlaender, Tarasha Khurana, Achal Dave, Bastian Leibe, and Deva Ramanan. Burst: A benchmark for unifying object recognition, segmentation and tracking in video. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1674–1683, 2023.

- [31] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. Mose: A new dataset for video object segmentation in complex scenes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 20224–20234, 2023.
- [32] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [33] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024.
- [34] Yuheng Qiu, Chen Wang, Wenshan Wang, Mina Henein, and Sebastian Scherer. Airdos: Dynamic slam benefits from articulated objects. In 2022 International Conference on Robotics and Automation (ICRA), pages 8047–8053. IEEE, 2022.
- [35] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 611–625. Springer, 2012.
- [36] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016.
- [37] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. arXiv preprint arXiv:2412.09621, 2024.
- [38] Aleksei Bochkovskii, AmaÃG, l Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024.
- [39] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. arXiv preprint arXiv:2409.02095, 2024.
- [40] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492–9502, 2024.
- [41] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024.
- [42] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024.
- [43] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024.
- [44] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024.
- [45] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

- [46] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.
- [47] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026,

- 2023.

[48] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer,

- 2024.

- [49] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. In European Conference on Computer Vision, pages 206–228. Springer, 2024.
- [50] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8121–8130, 2022.
- [51] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
- [52] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision, pages 18–35. Springer, 2024.
- [53] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. arXiv preprint arXiv:2410.11831, 2024.
- [54] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110, 2025.
- [55] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [56] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025.
- [57] David Yifan Yao, Albert J Zhai, and Shenlong Wang. Uni4d: Unifying visual foundation models for 4d modeling from a single video. arXiv preprint arXiv:2503.21761, 2025.
- [58] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [59] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [60] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018.

- [61] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 2443–2449, 2021.
- [62] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023.
- [63] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [64] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. Particlesfm: Exploiting dense point trajectories for localizing moving cameras in the wild. In European Conference on Computer Vision, pages 523–542. Springer, 2022.
- [65] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024.
- [66] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13–23, 2023.
- [67] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023.
- [68] Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. Sceneverse: Scaling 3d vision-language learning for grounded scene understanding. In European Conference on Computer Vision, pages 289–310. Springer, 2024.
- [69] Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, and Hanspeter Pfister. 4d langsplat: 4d language gaussian splatting via multimodal large language models. arXiv preprint arXiv:2503.10437, 2025.
- [70] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024.
- [71] Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, and Yin Cui. Describe anything: Detailed localized image and video captioning. arXiv preprint arXiv:2504.16072, 2025.
- [72] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472– 19495, 2024.
- [73] Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, Rushikesh Zawar, Xue Bai, Yilun Du, Chuang Gan, and Deva Ramanan. Towards understanding camera motions in any video. arXiv preprint arXiv:2504.15376, 2025.

- [74] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [75] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1611–1621, 2021.
- [76] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In European Conference on Computer Vision, pages 20–37. Springer, 2022.
- [77] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11):1231–1237, 2013.
- [78] Weirong Chen, Le Chen, Rui Wang, and Marc Pollefeys. Leap-vo: Long-term effective any point tracking for visual odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19844–19853, 2024.
- [79] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch visual odometry. Advances in Neural Information Processing Systems, 36:39033–39051, 2023.
- [80] Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573–580. IEEE, 2012.
- [81] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis & Machine Intelligence, 13(04):376–380, 1991.
- [82] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [83] Tony Cheng Tong, Sirui He, Zhiwen Shao, and Dit-Yan Yeung. G-veval: A versatile metric for evaluating image and video captions using gpt-4o. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7419–7427, 2025.
- [84] Olga Sorkine and Marc Alexa. As-rigid-as-possible surface modeling. In Symposium on Geometry processing, volume 4, pages 109–116. Citeseer, 2007.
- [85] Wei-Chiu Ma, Shenlong Wang, Rui Hu, Yuwen Xiong, and Raquel Urtasun. Deep rigid instance scene flow. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3614–3622, 2019.
- [86] Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, and Hanbyul Joo. Banmo: Building animatable 3d neural models from many casual videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2863–2873, 2022.
- [87] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764, 2024.
- [88] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021.

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

Figure 6: DynamicVerse dataset.

### A Appendix

In the appendix, we provide more results and analysis and summarize them as follows:

- • In Section A.1, we introduce the broader impact of our DynamicVerse framework.

- • In Section A.2, we supplement details of dynamic bundle adjustment.

- • In Section A.3, we ablate the different components for dynamic bundle adjustment.

- • In Section A.4, we provide additional experiments on generated hierarchical captions.

- • In Section A.5, we provide more qualitative results of dynamic bundle adjustment.

- • In Section A.6, we provide inference speed and computational cost for DynamicGen.

- • In Section A.7, we provide the limitation.

#### A.1 Broader Impact

The introduction of DynamicVerse, with its large-scale, physically-aware, and multimodally annotated 4D dataset derived from real-world videos, is set to significantly influence several advanced research areas. Our framework’s unique ability to capture metric-scale geometry, real-world motion, instancelevel semantics, and descriptive captions offers an unparalleled resource that can catalyze progress in the following domains:

- • Dynamic 4D Scene Generation: DynamicVerse offers a paradigm shift for Dynamic 4D Scene Generation. Current methods often rely on limited simulators or struggle to realistically portray complex real-world physics and motion from internet-sourced content. By accurately interpreting real-world dynamics from monocular videos and integrating window-based Bundle Adjustment with global optimization, DynamicVerse converts long video sequences into a comprehensive 4D multimodal format, capturing fine-grained dynamic information. This rich, real-world data provides an unparalleled training ground for generative models, leading to the creation of highly

- realistic, physically plausible, and semantically coherent dynamic 4D scenes. This has profound implications for high-fidelity content creation in entertainment (e.g., movies, games), realistic virtual environments for training and simulation (e.g., disaster response, architectural visualization), and the synthetic generation of diverse data for further AI research, helping to overcome privacy and data collection limitations.
- • 4D Vision-Language Models (4D-VLM): DynamicVerse will greatly accelerate the development of sophisticated 4D Vision-Language Models that can reason about space, time, and semantics concurrently. Existing VLMs often operate on 2D images or short video clips with limited 3D awareness. Our framework provides a unique combination of metric-scale 4D geometry, real-world dynamic motion, and comprehensive textual descriptions for long video sequences, allowing 4D-VLMs to learn intricate relationships between evolving 3D scenes and natural language narratives. Such models could enable more advanced human-agent interaction, where agents can provide detailed textual explanations of complex dynamic events they perceive in 4D, or understand nuanced, temporally extended instructions involving interactions within a 3D space. This could revolutionize areas like AI-powered video captioning, temporal question answering in 3D, and the development of embodied AI agents that communicate their understanding of the dynamic world with human-like richness.
- • 4D Language-Grounded Gaussian Splatting (4D-LangSplat): DynamicVerse offers a foundational dataset for advancing 4D-LangSplat methodologies. While current 4D Gaussian Splatting techniques excel at novel view synthesis of dynamic scenes, their integration with language for semantic understanding and manipulation is still nascent. Our dataset, rich with 800K+ instance masks and holistic descriptive captions directly linked to evolving 3D structures and motions at a physical scale, empowers 4D-LangSplat models. This will enable the development of systems that can not only reconstruct dynamic scenes with high fidelity but also allow users to query, edit, and interact with these 4D representations using natural language. For instance, users could ask an agent to "track the red car that just turned left" or "remove the person walking in front of the fountain," with the model understanding both the spatial dynamics and the semantic context. This can significantly enhance applications in robotics, augmented reality, and interactive content creation by bridging the gap between visual perception and linguistic instruction in dynamic 3D environments.

In summary, DynamicVerse is poised to serve as a crucial catalyst, providing the data and framework necessary to bridge the gap between 2D understanding and true 4D world modeling, thereby fostering advancements in semantic scene understanding, dynamic object interaction, multimodal reasoning, and realistic content generation.

#### A.2 Details of Dynamic Bundle Adjustment

Camera parameterization In Eq. (2), ξ ∈ SE(3) represents the camera poses as rigid transformations. Rotations are parameterized using so(3) rotation vectors, which offer a minimal representation facilitating direct optimization.

Static Area Bundle Adjustment Term In Eq. (3), the bundle adjustment energy CBA(P,Xstatic) measures the consistency between the pixel-level correspondences and the 3D structure of static

scene elements. Given the input pixel tracks Z = {Zk}Kk=0 and video segmentation M = {Mt}Tt=1, we filter all tracks corresponding to static areas and minimize the distance between the projected

pixel location and the observed pixel location:

wk,t∥Zk,t − πK(Xk,ξt)∥2 (6)

CBA(P,Xstatic;Z,M) =

Zk∈M t

where Xk is the k-th 3D point, Zk,t is the k-th 3D point’s corresponding pixel track’s 2D coordinates at time t, wk,t ∈ {0,1} is a visibility indicator and πK is the perspective projection function.

Camera Smoothness Prior In Eq. (3), given the video input, a temporal smoothness prior is imposed on camera poses. This prior penalizes abrupt changes in relative pose, defined as ξt→t+1 = ξt−+11 ·ξt. We adaptively reweight this term based on the magnitude of the relative motion. Specifically, a larger relative motion results in a reduced penalty on its change rate, while a smaller relative motion

incurs a higher penalty. Formally, this is expressed as:

Ccam(P) =

Crot(Rt−1,t,t+1) +

Ctrans(Tt−1,t,t+1)

t

t

where Crot(Rt−1,t,t+1) = 2||rad(R

t→t+1)−rad(Rt−1→t)||

||rad(Rt−1→t)||+||rad(Rt→t+1)|| and Ctrans(tt−1,t,t+1) = 2||tt→t+1−tt−1→t||

||tt−1→t||+||tt→t+1||; rad converts the rotation matrix into absolute radians.

Non-Rigid Bundle Adjustment Term In Eq. (4), for dynamic objects, we impose a nonrigid bundle adjustment term, ENR(Xdyn), which measures the discrepancy between the dynamic point cloud and pixel tracklets. Here, each pixel tracklet corresponds to a dynamic 3D point sequence, {Xk,t}, optimized for each observed tracklet:

wk,t∥Zk,t − πK(Xk,t,ξt)∥2 (7)

CNR(Xdyn,P,Z,M) =

zk∈M t

where Xk,t ∈ R3 is the k-th dynamic point’s location at t.

Dynamic Motion Prior In Eq. (4), Cmotion(Xdyn) is a regularization term that encodes the characteristics of the dynamic structure. It contains two prior terms that are used to regularize the dynamic structure, both of which have demonstrated effectiveness in previous work.

Cmotion(Xdyn) = Carap(Xdyn) + Csmooth(Xdyn). (8)

Carap represents an as-rigid-as-possible (ARAP) prior [84] designed to penalize extreme deformations that compromise local rigidity. Specifically, for each dynamic control point k, its nearest neighbors are identified using k-Nearest Neighbors (KNN) on the remaining tracks. We then enforce that the relative distances among these neighboring pairs remain consistent, preventing sudden changes

Carap =

wkm∥d(Xk,t,Xm,t) − d(Xk,t+1,Xm,t+1)∥2 (9)

t (k,m)

where d(,) is the L2 distance and wkm,t = 1 if all relevant points are visible. Csmooth is a simple smoothness term that promotes temporal smoothness for the dynamic point cloud:

wk,t∥Xk,t − Xk,t+1∥2. (10)

Csmooth =

t Xk∈Xdyn

Despite simplicity, both motion terms are crucial in our formulation, as they significantly reduce ambiguities in 4D dynamic structure estimation, which is highly ill-posed. Unlike other methods, we do not assume strong modelbased motion priors, such as rigid motion [85], articulated motion [86], or a linear motion basis [87].

Optical Flow Prior In Eq. (5), we also use a flow projection loss to encourage the global point maps to be consistent with the estimated flow for the confident, static regions of the actual frames. More precisely, given two frames t,t′, using their global point maps, camera extrinsics and intrinsics, we compute the flow fields from taking the global point map Xt, assuming the scene is static, and then moving the camera from t to t′. We denote this value Fglobal: t→t

′

cam , similar to the term defined in the confident static region computation above. Then we can encourage this to be close to the estimated flow, Ft→t

′

est , in the regions which are confidently static Xglobal: t→t

′

staic according to the global parameters:

′

′

′

∥Xglobal: t→t

· (Fglobal: t→t

cam − Ft→t

est )∥1, (11)

Cflow(Xstatic) =

Wi∈W t′∈Wi

where · indicates element-wise multiplication. Note that the confident dynamic mask is initialized using the foundation models as described in Sec. 3.3. During the optimization, we use the global

static point maps and camera parameters to compute Fglobalcam and update the confident dynamic mask. eving an average score of 4.3.

#### A.3 Ablation Study on Different Components for Dynamic Bundle Adjustment

Our dynamic BA pipeline introduces three key components absent in prior work like Uni4D [57], which systematically improve the decomposition of static/dynamic elements and global consistency:

- • (a) Epi-Mask-Based Dynamics Filtering: We introduce a geometric filtering step using an epipolar-based mask ("Epi-mask") to achieve a cleaner separation between static background and dynamic foreground pixels before bundle adjustment. This leads to more stable camera pose estimation and background reconstruction.
- • (b) VLM-Based Semantic Dynamics Analysis: We leverage a Vision-Language Model (VLM) for a high-level, semantic understanding of motion. This enables intelligent, motion-aware keyframe extraction and provides robust masks for dynamic objects, a significant improvement over purely geometric or flow-based segmentation.
- • (c) Optical Flow-Based Sliding Window Global Refinement: To address error accumulation and temporal drift common in long videos, we implement a global refinement strategy over a sliding window. This enforces long-range temporal consistency, correcting errors that a frame-by-frame or local BA approach would miss.

Table 6: Components Ablation on Sintel. Ablations (a) (b) (c) ATE↓ RPEtrans↓ RPErot↓ Abs↓ δ1.25↑ Baseline 0.114694 0.032125 0.347920 0.216433 0.725167

- Ablation-1 ✓ 0.114065 0.032250 0.335198 0.215058 0.726943
- Ablation-2 ✓ 0.11053 0.033122 0.334005 0.210339 0.722999
- Ablation-3 ✓ 0.114694 0.032125 0.347920 0.214282 0.724084
- Ablation-4 ✓ ✓ 0.108459 0.028906 0.281979 0.205892 0.727616
- Ablation-5 ✓ ✓ 0.114065 0.032250 0.335198 0.214143 0.725534
- Ablation-6 ✓ ✓ 0.110530 0.033122 0.334005 0.207329 0.725784 DynamicGen (Ours) ✓ ✓ ✓ 0.108459 0.028906 0.281979 0.204574 0.728961

#### A.4 Additional experiments on generated hierarchical captions.

We performed three distinct experiments to validate the high quality of our hierarchical semantic annotations:

- • (a) Object-Level Semantics via 4D-LangSplat [69]: To validate the annotations produced by our DynamicGen framework, we performed a time-sensitive querying experiment using a 4D-LangSplat model. For this evaluation, we trained the model on the "americano" scene from the HyperNeRF dataset and benchmarked it against a re-implemented 4D-LangSplat* baseline. The results, presented in Tab. 7, demonstrate that our approach yields substantial gains in Accuracy and volumetric Intersection over Union (vIoU). This superior performance confirms that our precise object masks and labels are highly effective for demanding multi-modal applications.

Table 7: Quantitative comparisons of time-sensitive querying on the HyperNeRF [88] dataset.

Method americano Acc(%) vIoU(%)

4D-LangSplat* [69] 53.84 27.55 DynamicGen 64.42 51.65

- • (b) Scene-Level Semantics via G-VEval [83]: To rigorously assess our scene-level captions, we moved beyond single-score metrics and employed a more granular evaluation using the ACCR framework in G-VEval benchmark. This approach provides a comprehensive, multi-dimensional assessment of caption quality across four key axes: Accuracy, Completeness, Conciseness, and Relevance. On a random sample of 100 videos from SA-V data, our generated captions demonstrated high performance across all four criteria, as detailed in the Tab. 8. The strong

performance across these metrics confirms that our captions are not only factually accurate and relevant to the video content, but also complete in their coverage of events and efficiently concise. This robust, multi-faceted quality makes them highly suitable and reliable for demanding downstream applications.

Table 8: Evaluation of generated captions using the ACCR framework from G-VEval. Evaluation Criteria Accuracy↑ Completeness↑ Conciseness↑ Relevance↑ Average↑ Scene-Level Captions 84.38 82.09 75.87 85.56 81.97

• (c) Camera-Level Semantics via Human Study: We conducted a formal human study to quantitatively analyze the quality of the final camera motion captions. Following prior work [73], we asked human evaluators to rate our captions on three criteria: (1) Clearness (clarity of information), (2) Conciseness (brevity without losing clarity), and (3) Grammar & Fluency. On a sub-sample of 88 videos from our dataset (i.e., filtered DAVIS), our captions performed excellently. The results, presented in Tab. 9 showed that over 60.22% of the captions were rated as both clear and fluent, while also receiving high scores for conciseness. This confirms the effectiveness of our generation and quality control process.

Table 9: Human evaluation results for the generated camera captions. Scores indicate the percentage of captions that met each quality criterion.

Human Evaluation Rated as Clear Rated as Fluent Rated as Concise Camera Captions 85.22% 89.77% 67.04%

- A.5 More qualitative results of dynamic bundle adjustment

We present additional qualitative reconstruction results in Fig. 8, demonstrating the generalizability and performance of our pipeline on real-world data.

- A.6 Inference Speed and Computational Cost for DynamicGen

For a reproducible analysis of computational performance, we processed the entire Sintel training set (23 videos) on NVIDIA H20 GPUs. A detailed breakdown of the average processing time and peak VRAM consumption for each component of our pipeline is provided in Table 10.

Table 10: Computational Cost Analysis.

Module Hardware Used Avg. Time / Sintel Peak VRAM Notes Video (mins) (GB)

- 1. Motion-aware Keyframe Extraction 1x H20 GPU ∼0.1 ∼10 Selects representative frames
- 2. VLM-Based Semantic Analysis (Qwen-VL) 2x H20 GPU ∼1.6 ∼60 Identifies dynamic elements
- 3. Moving Object Segmentation (SA2VA) 1x H20 GPU ∼0.8 ∼30 Per-object video segmentation
- 4. Dynamic Bundle Adjustment 1x CPU Core + 1x H20 GPU ∼12.2 ∼30 Main time bottleneck
- 5. Moving Object Captioning 2x H20 GPU ∼2.0 ∼24 Object-level descriptions
- 6. Dynamic Scene Captioning 2x H20 GPU ∼3.0 ∼40 Scene-level descriptions
- 7. Camera Motion Captioning 2x H20 GPU ∼2.0 ∼40 Camera-level descriptions
- 8. Caption Rephrasing 1x H20 GPU ∼2.0 ∼24 LLM-based refinement for consistency and conciseness

Total (per video) H20 GPU ∼23.7 ∼60 Peak VRAM, not sum

- A.7 Limitations

Despite its considerable capabilities, DynamicVerse exhibits several inherent limitations. First, its reliance on in-the-wild internet videos introduces significant noise and quality variance. This can compromise the fidelity of metric-scale geometry and motion recovery, particularly in complex, cluttered, or occluded scenes that fall outside the typical distribution of the foundation models’ training

[Figure 252]

Figure 7: Examples captions on DAVIS dataset.

data. Second, the substantial computational overhead required to process long video sequences with large-scale models presents a practical barrier to real-time performance and scalable deployment. Finally, while extensive, the dataset cannot exhaustively capture the long tail of real-world phenomena. Consequently, the model’s generalization to truly novel environments is fundamentally tethered to the intrinsic biases and capabilities of its underlying foundation models.

These limitations raise AI-safety concerns: (i) privacy and security risks, since metric-scale reconstructions from web videos can expose sensitive interiors or critical infrastructure and facilitate covert mapping or surveillance; and (ii) miscalibrated confidence under distribution shift, producing plausible but erroneous geometry and dynamics that misguide downstream robotic or AR planners. Biases and licensing gaps in foundation models and web data may further perpetuate representational harms and legal or IP issues. A practical mitigation is to prefilter ineligible videos using policy rules and automated detectors (e.g., content with PII, sensitive interiors or infrastructure, minors, or restricted licenses).

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 260]

[Figure 261]

[Figure 268]

[Figure 272]

| |
|---|

[Figure 275]

[Figure 276]

Monst3r Uni4D Ours

Monst3r Uni4D Ours

[Figure 277]

[Figure 282]

[Figure 283]

[Figure 291]

[Figure 294]

[Figure 296]

[Figure 297]

Monst3r Uni4D Ours

Monst3r Uni4D Ours

[Figure 299]

[Figure 304]

[Figure 305]

[Figure 313]

[Figure 314]

[Figure 316]

[Figure 318]

| |
|---|

Monst3r Uni4D Ours

Monst3r Uni4D Ours

[Figure 321]

[Figure 326]

[Figure 327]

[Figure 333]

[Figure 335]

[Figure 336]

[Figure 337]

Monst3r Uni4D Ours

Monst3r Uni4D Ours

- Figure 8: Qualitative Results on in-the-wild data. We show qualitatively some of our reconstruction results on in-the-wild data. For full reconstruction, please refer to our attached supplementary webpage.

Video SegAnyMo GPT+DEVA DynamicGen

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

- Figure 9: Qualitative Results of moving object Segmentation. We show qualitatively some of our segmentation results on the Youtube-VIS dataset compared with other baselines.

