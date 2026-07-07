[Figure 1]

[Figure 2]

2025-9-16

## OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling

Yang Zhou1, Yifan Wang1, Jianjun Zhou1,2, Wenzheng Chang1, Haoyu Guo1, Zizun Li1, Kaijing Ma1, Xinyue Li1, Yating Wang1, Haoyi Zhu1, Mingyu Liu1,2, Dingning Liu1, Jiange Yang1, Zhoujie Fu1, Junyi Chen1, Chunhua Shen2, Jiangmiao Pang1, Kaipeng Zhang1 and Tong He1 1Shanghai AI Lab, 2ZJU

# arXiv:2509.12201v2[cs.CV]24Sep2025

The field of 4D world modeling—aiming to jointly capture spatial geometry and temporal dynamics—has witnessed remarkable progress in recent years, driven by advances in large-scale generative models and multimodal learning. However, the development of truly general 4D world models remains fundamentally constrained by the availability of high-quality data. Existing datasets and benchmarks often lack the dynamic complexity, multi-domain diversity, and spatial-temporal annotations required to support key tasks such as 4D geometric reconstruction, future prediction, and camera-controlled video generation. To address this gap, we introduce OmniWorld, a large-scale, multi-domain, multi-modal dataset specifically designed for 4D world modeling. OmniWorld consists of a newly collected OmniWorld-Game dataset and several curated public datasets spanning diverse domains. Compared with existing synthetic datasets, OmniWorld-Game provides richer modality coverage, larger scale, and more realistic dynamic interactions. Based on this dataset, we establish a challenging benchmark that exposes the limitations of current state-of-the-art (SOTA) approaches in modeling complex 4D environments. Moreover, fine-tuning existing SOTA methods on OmniWorld leads to significant performance gains across 4D reconstruction and video generation tasks, strongly validating OmniWorld as a powerful resource for training and evaluation. We envision OmniWorld as a catalyst for accelerating the development of general-purpose 4D world models, ultimately advancing machines’ holistic understanding of the physical world.

GitHub | Data | Homepage

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

###### MULTI-DOMAIN MULTI-MODAL 300M+ FRAMES

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

The environment is a simple indoor setting. A hand reached forward and grasped the bottle in the wooden cabinet …

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

The environment is a kitchen setting. The person’s hand extends forward to grasp the bowl with the egg …

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

Figure 1. We introduce OmniWorld, a large-scale, multi-domain, and multi-modal dataset. OmniWorld provides a rich resource for 4D world modeling by integrating high-quality data from multiple domains and offers a variety of data types, including depth maps, camera poses, text captions, optical flow and foreground masks. OmniWorld is designed to accelerate the development of more general models for modeling the real physical world.

Corresponding author: Tong He, tonghe90@gmail.com © 2025 Shanghai Artificial Intelligence Laboratory. All rights reserved.

Data modality Depth Camera Text Optical flow Fg. masks

Dataset Scene Type Motion Resolution # Frames

MPI Sintel (Butler et al., 2012) Mixed Dynamic 1024 × 436 1K ✔ ✔ ✗ ✔ ✔ FlyingThings++ (Harley et al., 2022; Mayer et al., 2016) Outdoor Dynamic 960 × 540 28K ✔ ✗ ✗ ✔ ✔ TartanAir (Wang et al., 2020) Mixed Dynamic 640 × 480 1,000K ✔ ✔ ✗ ✔ ✔ BlendedMVS (Yao et al., 2020) Mixed Static 768 × 576 17K ✔ ✔ ✗ ✗ ✗ HyperSim (Roberts et al., 2021) Indoor Static 1024 × 768 77K ✔ ✔ ✗ ✗ ✔ Dynamic Replica (Karaev et al., 2023) Indoor Dynamic 1280 × 720 169K ✔ ✔ ✗ ✔ ✔ Spring (Mehl et al., 2023) Mixed Dynamic 1920 × 1080 23K ✔ ✔ ✗ ✔ ✗ EDEN (Le et al., 2021) Outdoor Static 640 × 480 300K ✔ ✔ ✗ ✔ ✔ PointOdyssey (Zheng et al., 2023) Mixed Dynamic 960 × 540 216K ✔ ✔ ✗ ✗ ✔ SeKai-Game (Li et al., 2025) Outdoor Dynamic 1920 × 1080 4,320K ✗ ✔ ✔ ✗ ✗ OmniWorld-Game (Ours) Mixed Dynamic 1280 × 720 18,515K ✔ ✔ ✔ ✔ ✔

- Table 1. Comparisons between OmniWorld-Game and existing synthetic datasets. OmniWorldGame surpasses existing public synthetic datasets in modal diversity and data scale.

### 1. Introduction

The development of world models (Agarwal et al., 2025; DeepMind, 2025; Ha and Schmidhuber, 2018; Hafner et al., 2023; LeCun, 2022) has become a central pursuit in visual intelligence systems, aiming to build systems that can simulate and reason about the physical world. This capability goes beyond simple static perception, demanding models that can simulate dynamic environments, predict object motion, infer causality, and generate content that adheres to physical laws. Such spatio-temporal modeling is a cornerstone for effective world models, with its development critically dependent on large-scale, multi-domain, and multi-modal datasets (Chen et al., 2025; Feng et al., 2024; He et al., 2025b; Team et al., 2025a,b; Yu et al., 2025a,b).

Two fundamental tasks that reflect a model’s world modeling capability have drawn widespread attention: 3D geometric foundation models (Leroy et al., 2024; Tang et al., 2024; Wang et al., 2025a,b, 2024c, 2025d; Yang et al., 2025; Zhang et al., 2024, 2025), and camera-controlled video generation models (Bahmani et al., 2024; Bai et al., 2025; He et al., 2024; Wang et al., 2024d; YU et al., 2025; Zheng et al., 2024). The former aims to extract comprehensive 3D geometric information from 2D image inputs, while the latter focuses on generating dynamic video content that follows precise spatio-temporal instructions. Both tasks heavily rely on large-scale, high-quality datasets with rich modalities, including RGB images, depth maps, and camera poses.

However, existing benchmarks and datasets for evaluating and training these models have significant limitations. In the domain of 3D geometric foundation models, existing benchmarks suffer from short sequence lengths, which constrain the evaluation of a model’s long-term robustness. For example, Sintel (Butler et al., 2012), which is a widely used dataset, consists of videos with an average length of only 50 frames. Furthermore, the limited motion amplitude and single-action types within these datasets (e.g., Bonn’s (Palazzolo et al., 2019) focuses on indoor human motion, Kitti’s (Geiger et al., 2013) focuses on outdoor street scenes) fail to comprehensively evaluate model performance in complex, dynamic environments. Similarly, in the field of camera-controlled video generation, mainstream datasets like RealEstate10K (Zhou et al., 2018) primarily consist of static scenes with smooth camera trajectories. This lack of diverse object motion and complex camera operations results in a noticeable gap between the dataset’s content and real-world scenarios, thereby hindering a comprehensive assessment of a model’s true capabilities.

From the perspective of training data, there is a critical scarcity of high-quality, multi-domain, multi-modal datasets that include rich geometric annotations. For instance, in image or video generation, while there are numerous image-text (Gadre et al., 2023; Schuhmann et al., 2022) or video-text datasets (Chen et al., 2024; Ju et al., 2024; Nan et al., 2024), they often lack critical geometric modalities such as depth maps, camera poses, and optical flow. Similarly, the demand for large-scale, diverse datasets with accurate geometric annotations is increasingly urgent for 3D

Data modality Depth Camera Text Opt. flow Fg. masks

Dataset Domain # Seq. FPS Resolution # Frames

OmniWorld-Game Simulator 96K 24 1280×720 18,515K AgiBot (Bu et al., 2025) Robot 20K 30 640×480 39,247K ✔ ✔ ✗ DROID (Khazatsky et al., 2024) Robot 35K 60 1280×720 26,643K ✔ RH20T (Fang et al., 2024) Robot 109K 10 640×360 53,453K ✗ ✔ RH20T-Human (Fang et al., 2024) Human 73K 10 640×360 8,875K ✗ ✔ ✗ ✗ HOI4D (Liu et al., 2022) Human 2K 15 1920×1080 891K ✔ Epic-Kitchens (Damen et al., 2018) Human 15K 30 1280×720 3,635K ✗ ✗ ✗ Ego-Exo4D (Grauman et al., 2024) Human 4K 30 1024×1024 9,190K ✗ ✔ ✗ HoloAssist (Wang et al., 2023) Human 1K 30 896×504 13,037K ✗ ✗ Assembly101 (Sener et al., 2022) Human 4K 60 1920×1080 110,831K ✗ ✔ EgoDex (Hoque et al., 2025) Human 242K 30 1920×1080 76,631K ✗ ✔ ✗ ✗ CityWalk (Li et al., 2025) Internet 7K 30 1280×720 13,096K ✗ ✔ ✗ ✗

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

- Table 2. OmniWorld structure. A smiling face ( ) indicates the modality is newly (re-)annotated by us, a green check (✔) denotes ground-truth data that already exists in the original dataset, and a red cross (✗) marks missing modalities.

[Figure 121]

geometric foundation models.

To address these shortcomings, we introduce OmniWorld, a large-scale, multi-domain, and multimodal dataset composed of a self-collected high-quality OmniWorld-Game synthetic dataset and several public datasets. Its core characteristics are: 1) High-Quality 4D Data. OmniWorld-Game is a massive synthetic video dataset comprising over 96K clips and more than 18M frames, with a total duration of over 214 hours. It is captured from diverse game environments with 720P RGB images, dense ground truth depth maps, accurate camera poses, and annotations for text captions, optical flow and foreground masks. As shown in Table 1, the dataset significantly surpasses existing public synthetic datasets in modal diversity and scale. 2) Multi-Domain Coverage. By integrating datasets from four key domains including simulator, robot, human, and the internet, OmniWorld covers a wide range of real-world and virtual scenarios, greatly enhancing data diversity. 3) Multi-Modality Annotations. OmniWorld provides a rich suite of multi-modal annotations, crucial for detailed world modeling, as shown in Table 2.

Based on OmniWorld-Game, we propose a new benchmark for both 3D geometric foundation models and camera-controlled video generation models. Our OmniWorld-Game benchmark provides challenging, complex scenarios and dynamics that accurately reflect a model’s true world capabilities, revealing the limitations of current SOTAs. By fine-tuning existing SOTAs (e.g., DUSt3R (Wang et al., 2024c), CUT3R (Wang et al., 2025b), Reloc3r (Dong et al., 2024), AC3D (Bahmani et al., 2024)) with OmniWorld, we demonstrate significant performance improvements on public benchmarks. This strongly validates OmniWorld as a powerful training resource for enhancing world modeling capabilities.

In summary, our contributions are as follows:

- 1. We introduce OmniWorld, a multi-domain and multi-modal dataset designed to address the lack of diversity in existing datasets. Its self-collected subset, OmniWorld-Game, surpasses current synthetic datasets in both modality diversity and data volume.
- 2. We establish a comprehensive benchmark for 3D geometric foundation models and cameracontrolled video generation models based on OmniWorld-Game, providing a unified platform for evaluation.
- 3. We fine-tune several SOTAs on OmniWorld and observe significant performance gains, underscoring its value as a training resource.

[Figure 122]

###### Data Collection Video Slicing

[Figure 123]

Text Captions

[Figure 124]

QWenVL

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Depth Maps

DepthPipe

###### Clean

[Figure 130]

Camera Poses

PosePipe

Simulator Robot

[Figure 131]

[Figure 132]

Motion blur Insufficient features Excessive motion … Filtered RGB

[Figure 133]

Fg. Mask

SAM

Multi-Domain Multi-Modal

Human Internet Raw Data

[Figure 134]

[Figure 135]

DPFlow

Optical Flow

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

PosePipe

DepthPipe

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

Coarse Refine

Depth Maps

[Figure 152]

[Figure 153]

ReShade

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

CoTracker with BA

DroidCalib or VGGT

Prior Depth

Fused Point Cloud

[Figure 166]

RGB

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

Foundation Stereo

###### Fg. Mask

- Figure 2. OmniWorld acquisition and annotation pipeline. We collect raw data from diverse domains and apply a video slicing filter to obtain high-quality RGB sequences. These sequences are then processed through a suite of specialized pipelines to generate multi-modal annotations, including text captions, depth maps, camera poses, foreground masks, and optical flow.

### 2. OmniWorld Dataset

To advance comprehensive spatio-temporal modeling of the real physical world, we curate OmniWorld, a large-scale, multi-domain and multi-modal dataset that mirrors the complexity of the physical world. We design and implement a detailed data acquisition and annotation pipeline to ensure high-quality multi-modal annotations, as illustrated in Figure 2.

##### 2.1. Data Acquisition

To address the scarcity of high-precision, temporally consistent, and dynamically rich data, we develop a sophisticated data acquisition pipeline. Our approach is centered on a novel self-collected dataset, OmniWorld-Game, which we supplement with data from three other domains: robot, human, and internet. This strategy allows us to integrate the strengths of diverse data sources to comprehensively capture real-world complexity.

Simulator Domain. To acquire the high-precision and temporally consistent multimodal data that is hard to obtain in the real world, we collect OmniWorld-Game from game environments. Following prior works (Feng et al., 2024; Richter et al., 2016; Team et al., 2025a; Yang et al., 2024a), we utilize ReShade (ReShade Contributors, 2024) to access depth information during the rendering process, and simultaneously capture synchronized RGB images from the screen using OBS (Contributors, 2024). This approach offers significant advantages: 1) High-Precision Modal Data. We can precisely control the environment and acquire accurate depth data, which is often unattainable in real-world settings and is crucial for spatio-temporal modeling. 2) Rich Real-World Scene Simulation. Modern virtual environments provide highly realistic graphics and diverse simulations of real-world scenarios, encompassing complex settings from wilderness to urban areas, and from day to night.

Robot Domain. We integrate public datasets from robot manipulation and human-robot interaction tasks, including AgiBot (Bu et al., 2025), DROID (Khazatsky et al., 2024), and RH20T (Fang et al., 2024). These datasets provide valuable sequences of robot-environment interactions and navigation, which are essential for tasks involving robotic manipulation and physical world understanding.

Human Domain. We incorporate public datasets describing various human activities, including RH20T-Human (Fang et al., 2024), HOI4D (Liu et al., 2022), Epic-Kitchens (Damen et al., 2018), Ego-Exo4D (Grauman et al., 2024), HoloAssist (Wang et al., 2023), Assembly101 (Sener et al., 2022), and EgoDex (Hoque et al., 2025). These datasets capture diverse human behaviors, ranging from daily activities to complex assembly tasks, from both egocentric and exocentric perspectives.

Internet Domain. To acquire large-scale, realistic, and diverse in-the-wild scene data, we utilize the CityWalk dataset (Li et al., 2025). This dataset offers rich real-world street view videos from the internet. We specifically focus on supplementary camera pose annotation for this data, providing valuable real-world information for 3D geometry and camera pose estimation tasks.

To prepare raw data for our annotation pipeline, we first perform video slicing to ensure all clips are of high quality and temporal coherence. This process has two main objectives: first, to remove frames unsuitable for geometric or motion analysis, such as those with motion blur, insufficient feature points, or excessively large dynamic areas; and second, to segment long videos into shorter, manageable clips. After this preprocessing step, the filtered, high-quality video segments are then passed to our multi-modal annotation pipeline.

##### 2.2. Data Annotation

To provide high-quality multi-modal annotation information, we design an innovative data processing pipeline. We primarily annotate the following key modalities: depth maps, camera poses, text captions, optical flow, and foreground masks (see Figure 2 for the overall pipeline). These modalities are crucial for models to achieve comprehensive spatio-temporal modeling. Here we briefly introduce the annotation method of each modality, please refer to supplementary material for more details.

Depth maps. Accurate depth information is paramount for geometric modeling. To ensure the quality and consistency of depth maps, we adopt a tailored approach based on the data source. For the self-collected dataset OmniWorld-Game, as mentioned in Section 2.1, we directly access depth information during the rendering process using tools like ReShade (ReShade Contributors, 2024). For public datasets AgiBot (Bu et al., 2025) and HOI4D (Liu et al., 2022), these datasets typically provide raw depth maps that are often noisy and sparse. We employ Prior Depth Anything (Wang et al., 2025e) to optimize these noisy depth maps, generating denser and more accurate depth maps. For the public stereo dataset DROID (Khazatsky et al., 2024), we leverage FoundationStereo (Wen et al., 2025) for stereo depth estimation on this dataset.

Foreground masks. To provide precise, temporally consistent masks of primary subjects for tasks like subject-environment interaction and behavior analysis, we develop specialized automated pipelines. For robot domain data, we use RoboEngine (Yuan et al., 2025) to generate initial masks for keyframes, followed by temporal tracking and fusion with SAM 2 (Ravi et al., 2024). For OmniWorld-Game (e.g., player characters in third-person view), we leverage Grounding DINO (Liu et al., 2023) to detect initial bounding boxes within predefined regions of keyframes, which then serve as prompts for SAM (Kirillov et al., 2023). These generated masks can be used as dynamic foreground masks to guide camera pose estimation, as detailed in the following section.

Camera poses. Accurate camera pose annotation in dynamic videos is highly challenging due to transitions, weakly textured areas, and abrupt movements that hinder traditional Structure-fromMotion methods (Li et al., 2024; Rockwell et al., 2025). Following prior work (Team et al., 2025a), we develop a robust, automated, two-stage pipeline for dynamic camera pose annotation, whose principles are validated across diverse data types.

The pipeline leverages the pre-computed foreground masks to focus on static background regions.

Assembly101

HoloAssist

EgoExo4D

EpicKitchens

EgoDex HOI4D

Human

RH20T-Human

Internet

Simulator

CityWalk

OmniWorldGame AgiBot

Robot

RH20T

DROID

(a) OmniWorld Compositional Distribution

|Terrain<br><br>Building<br><br>Vehicle<br><br>Mixed<br><br>Dominant Object|OutdoorUrban<br><br>OutdoorNatural<br><br>Indoor Mixed<br><br>Scene Type|
|---|---|
|Ancient<br><br>Modern<br><br>Futuristic<br><br>Historical Era|FirstPerson<br><br>ThirdPerson<br><br>Camera Perspective|

(b) OmniWorld-Game Internal Composition

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

Mean: 200.259

Percentage(%)

100 150 200 250

(c) Caption Tokens Distribution

- Figure 3. Statistical information of OmniWorld. (a) displays compositional distribution of data from different domains within OmniWorld, (b) presents internal composition of OmniWorld-Game. (c) shows caption tokens distribution of OmniWorld.

The stages include: 1) Coarse camera pose estimation leveraging VGGT (Wang et al., 2025a) for videos without depth or DroidCalib (Hagemann et al., 2023) with depth constraints; 2) Camera pose refinement through dense point tracking (SIFT (Lowe, 2004), SuperPoint (DeTone et al., 2018) with CoTracker3 (Karaev et al., 2024)) on static regions and subsequent bundle adjustment to minimize reprojection errors, optionally enhanced by forward-backward reprojection with depth information (Chen et al., 2019).

Text captions. We generate high-quality text descriptions for video sequences using a semi-automated approach primarily driven by Qwen2-VL-72B-Instruct model (Wang et al., 2024a). We design specific prompting strategies tailored to different data domains. For robot and human domain data, we first annotate overall video tasks, then annotate in units of 81-frame segments. For OmniWorld-Game, we develop distinct prompts for various viewpoints (e.g., first-person, third-person), encompassing types such as short caption, player character caption, background caption, camera caption, video caption, and key tags, utilizing 81-frame segments.

Optical flow. Optical flow, as a dense motion vector field, is crucial for capturing pixel-level motion information in videos and serves as a fundamental modality for accurate spatio-temporal modeling. We select DPFlow (Morimitsu et al., 2025) for optical flow annotation. Unlike mainstream models such as RAFT (Teed and Deng, 2020) which require downsampling inputs when processing high-resolution videos, DPFlow can directly perform predictions on the original resolution. Given that our dataset includes various resolutions, the choice of DPFlow ensures that the optical flow annotation accurately reflects subtle movements within the videos.

##### 2.3. Data Statistics

OmniWorld comprises 12 heterogeneous datasets from four domains: simulators, robots, humans, and the internet. Table 2 summarizes the key metadata for these datasets. OmniWorld collectively contains over 600 thousand video sequences and more than 300 million frames. Notably, our collection includes a significant portion of high-resolution videos, with more than half of the data having a resolution of 720P or higher. We meticulously annotate the data with multiple modalities, including depth, camera poses, text, optical flow, and foreground masks.

- Figure 3a illustrates the compositional distribution of data from different domains within Omni-

World. Notably, data from the human domain constitutes the largest share, underscoring the dataset’s richness in reflecting real-world human activities and interactions.

- Figure 3b further elaborates on the internal composition of OmniWorld-Game, showcasing its high

diversity across multiple dimensions. For scene type, OmniWorld-Game encompasses outdoor-urban, outdoor-natural, indoor, and mixed scenes, with outdoor-urban scenes having the highest proportion. For camera perspective, OmniWorld-Game includes both first-person and third-person-following perspectives, predominantly featuring first-person views. Regarding the historical era, OmniWorldGame covers diverse styles, including ancient, modern, and futuristic sci-fi periods. In terms of dominant object, OmniWorld-Game includes various types such as natural terrain, architecture, vehicles, and mixed elements. Most scenes incorporate multiple object types, significantly enhancing the data’s challenge and complexity. These statistics collectively demonstrate that OmniWorld-Game exhibits an exceptionally diverse and challenging scene distribution.

For the text modality, we provide comprehensive and detailed annotations. As shown in Figure 3c, our text captions primarily contain between 150 and 250 tokens per description. This rich annotation density significantly surpasses that of most existing video-text datasets, such as OpenVid-1M (Nan

- et al., 2024) and Panda-70M (Chen et al., 2024).

3. OmniWorld-Game Benchmark

To comprehensively evaluate and advance world modeling, we construct OmniWorld-Game benchmark, providing a comprehensive and challenging evaluation platform for two critical tasks: 3D geometric prediction and camera-controlled video generation.

3.1. 3D Geometric Prediction Benchmark

Benchmark design and motivation. Existing benchmarks for 3D geometric foundation models (GFMs) suffer from significant limitations. Specifically, many current benchmarks have the following drawbacks: First, sequence lengths are generally short, which restricts evaluating models’ ability in long sequences reconstruction. For instance, Sintel (Butler et al., 2012) video sequences average only 50 frames. Second, the dynamic motion in these datasets is relatively small in amplitude and uniform in type. For example, Bonn (Palazzolo et al., 2019) focuses on human dynamics in indoor scenes, NYU-v2 (Silberman et al., 2012) focuses on indoor static objects, and KITTI (Geiger et al., 2013) datasets only include outdoor street views, making it challenging to comprehensively test model performance in complex dynamic environments.

To address this, OmniWorld-Game offers an advanced evaluation environment featuring extended temporal sequences (up to 16 seconds with 384 frames), rich and diverse motion, extreme scenarios with environmental diversity (e.g., mixed scene types), and high-resolution realistic data (720P). These characteristics allow for a deeper and more comprehensive assessment of GFMs capabilities.

Evaluated baselines and experiment details. We thoroughly assess current GFMs, including DUSt3R (Wang et al., 2024c), MASt3R (Leroy et al., 2024), MonST3R (Zhang et al., 2024), Fast3R (Yang

- et al., 2025), CUT3R (Wang et al., 2025b), FLARE (Zhang et al., 2025), VGGT (Wang et al., 2025a), and MoGe (Wang et al., 2024b, 2025c), within the OmniWorld-Game benchmark. These models are evaluated on two core tasks: monocular depth estimation and video depth estimation. All images are consistently resized to a long side of 512 pixels while preserving aspect ratio.

Quantitative analysis. Our quantitative analysis on OmniWorld-Game reveals key performance insights and bottlenecks. For monocular depth estimation, MoGe-2 achieves the best results, though

Mono-Depth Video-Depth

Method

scale scale scale&shift

###### FPS

Abs Rel ↓ 𝛿<1.25 ↑ Abs Rel ↓ 𝛿<1.25 ↑ Abs Rel ↓ 𝛿<1.25 ↑

DUSt3R (Wang et al., 2024c) 0.742 0.460 0.709 0.447 0.379 0.560 0.96 MASt3R (Leroy et al., 2024) 0.485 0.560 0.482 0.579 0.217 0.724 0.79 MonST3R (Zhang et al., 2024) 0.670 0.493 0.669 0.505 0.272 0.648 0.95 Fast3R (Yang et al., 2025) 0.755 0.404 0.741 0.384 0.464 0.531 14.99 CUT3R (Wang et al., 2025b) 0.624 0.518 0.690 0.479 0.429 0.603 10.75 FLARE (Zhang et al., 2025) 0.664 0.475 0.757 0.453 0.511 0.527 4.24 VGGT (Wang et al., 2025a) 0.531 0.554 0.440 0.625 0.194 0.755 18.75

- MoGe-1 (Wang et al., 2024b) 0.459 0.586 – – – – –

- MoGe-2 (Wang et al., 2025c) 0.401 0.589 – – – – –

##### Table 3. Monocular Depth & Video Depth Estimation on OmniWorld-Game benchmark.

FVD

Method TransErr↓ RotErr↓ CamMC↓

VideoGPT↓ StyleGAN↓

AC3D (T2V) (Bahmani et al., 2024) 6.2788 0.8867 6.6965 1745.778 1594.885 MotionCtrl (I2V) (Wang et al., 2024d) 7.8633 1.1402 8.2710 694.342 745.652 CamCtrl (I2V) (He et al., 2024) 1.2882 0.2022 1.3856 615.417 637.574 CAMI2V (I2V) (Zheng et al., 2024) 5.9626 0.5087 6.2010 837.185 742.594

##### Table 4. Camera-Controlled Video Generation Evaluation on OmniWorld-Game benchmark.

significant room for improvement remains across models, underscoring the benchmark’s challenge on single-frame geometric understanding (Table 3). In the more demanding video depth estimation task, VGGT demonstrated superior performance across all metrics under both scale-only and scale-and-shift alignments, with significantly higher FPS than competitors. While MASt3R also showed competitive metrics, its low FPS due to global alignment limits its practicality (Table 3). Overall, no single GFM achieves top-tier performance across all metrics, indicating that current SOTAs still face considerable challenges in handling the high-dynamic, long-sequence 3D geometric understanding and consistency problems introduced by OmniWorld-Game.

Visual Results. In Figure 4, we provide a visual comparison of the monocular depth prediction results from various methods on the OmniWorld-Game benchmark. As a model specifically designed for monocular geometry tasks, MoGe-2 (Wang et al., 2025c) achieves superior accuracy and produces visually sharp depth maps, surpassing the performance of other multi-view methods.

To show the challenges of video depth estimation on the OmniWorld-Game benchmark, we present a qualitative comparison of feed-forward reconstruction methods using point cloud visualizations in Figure 5. The video-depth estimation task demands high temporal consistency. Our visualizations show that VGGT (Wang et al., 2025a) generates more coherent 3D structures than other methods in dynamic scenes. However, even VGGT shows noticeable artifacts, revealing limitations in capturing complex details.

These observations indicate that the robustness of current methods needs improvement on OmniWorld-Game. Our benchmark provides a clear direction for advancing the next generation of GFMs with stronger spatio-temporal consistency.

RGB Ground Truth DUSt3R MASt3R MonST3R Fast3R CUT3R FLARE VGGT MoGe-1 MoGe-2

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

- Figure 4. Qualitative comparison of Monocular Depth Estimation on OmniWorld-Game benchmark.

Input Images Ground Truth Fast3R CUT3R FLARE VGGT

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

- Figure 5. Qualitative comparison of multi-view 3D reconstruction on OmniWorld-Game benchmark.

- 3.2. Camera-Controlled Video Generation Benchmark

Benchmark design and motivation. Existing benchmarks for camera-controlled video generation often rely on static datasets with smooth camera trajectories (e.g., RealEstate10K (Zhou et al., 2018)), which do not reflect real-world complexity. OmniWorld-Game benchmark addresses this by providing a challenging testing environment with rich dynamic content (e.g., diverse motions, complex interactions), extremely diverse scenes and environments (e.g., varied geographical, weather, lighting conditions), complex camera trajectories reflecting real patterns, and multi-modal input with diverse subjects (e.g., various perspectives, characters, vehicles). This enables a rigorous evaluation of models’ ability to handle complex spatio-temporal dynamics and adhere to precise control instructions.

Evaluated baselines and experiment details. We benchmark mainstream SOTAs, including AC3D (Bahmani et al., 2024) (T2V), CamCtrl (He et al., 2024), MotionCtrl (Wang et al., 2024d), and CAMI2V (Zheng et al., 2024) (all I2V). These models represent different conditioned video generation models and are evaluated adhering to their default configurations. Following CAMI2V (Zheng et al., 2024), metrics include Camera Parameter Metrics (RotError, TransError, and CamMC) to quantify adherence to camera commands, and Fréchet Video Distance (FVD) (Unterthiner et al., 2018) to assess perceptual realism.

Quantitative analysis. Our quantitative analysis on OmniWorld-Game reveals key insights and

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

AC3D

“A man in a suit walks along tram tracks in an urban setting, passing stationary trams and observing his surroundings…”

|[Figure 263]|
|---|

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

MotionCtrl

|[Figure 278]|
|---|

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

[Figure 292]

CamCtrl

|[Figure 293]|
|---|

[Figure 294]

[Figure 295]

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

CAMI2V

- Figure 6. Qualitative comparison of Camera-Controlled Video Generation on OmniWorld-Game benchmark. In T2V setting, AC3D takes the text as a condition signal. In I2V setting, MotionCtrl, CamCtrl, CAMI2V takes the image as a condition signal. Condition images are the first images of each row.

challenges. In the Text-to-Video task, AC3D showed basic camera control but high FVD, indicating the difficulty of generating high-fidelity, dynamic content with camera control and text prompts in complex scenes (Table 4). For Image-to-Video models, CamCtrl achieves superior performance in both camera-controlled accuracy and video quality. However, all evaluated SOTAs still exhibit significant room for improvement across OmniWorld-Game, especially in simultaneously ensuring video generation quality and precise camera control. This highlights ongoing challenges and future research directions.

Visual results. To visually demonstrate the challenges posed by the OmniWorld-Game benchmark, we present the qualitative results of various camera-controlled video generation models in Figure 6. In the T2V setting, although AC3D (Bahmani et al., 2024) generates semantically coherent video content, the depicted human motion is minimal, and the model fails to accurately follow the input camera trajectory. This highlights a fundamental limitation of current models in understanding and generating complex dynamic motions from abstract text instructions. In the I2V setting, while the camera trajectory of CamCtrl’s (He et al., 2024) generated video aligns well with the input conditions, the visual quality of moving characters is blurry, and the overall video quality is poor. Similar quality degradation issues are observed in the outputs of MotionCtrl (Wang et al., 2024d) and CAMI2V (Zheng et al., 2024). These results reveal the unique challenges of the OmniWorld-Game benchmark.

Sintel Bonn KITTI NYU-v2 Abs Rel↓ 𝛿 < 1.25 ↑ Abs Rel↓ 𝛿 < 1.25 ↑ Abs Rel↓ 𝛿 < 1.25 ↑ Abs Rel↓ 𝛿 < 1.25 ↑

Method

DUSt3R (Wang et al., 2024c) 0.488 0.532 0.139 0.831 0.109 0.873 0.081 0.909 MonST3R (Zhang et al., 2024) 0.402 0.525 0.069 0.954 0.098 0.895 0.094 0.887 DUSt3R* 0.370 0.529 0.067 0.948 0.088 0.932 0.089 0.902

CUT3R (Wang et al., 2025b) 0.420 0.520 0.058 0.967 0.097 0.914 0.081 0.914 CUT3R* 0.408 0.522 0.075 0.944 0.087 0.935 0.075 0.920

- Table 5. Comparison of Original and Fine-tuned Models for Monocular Depth Estimation on Sintel (Butler et al., 2012), Bonn (Palazzolo et al., 2019), KITTI (Geiger et al., 2013) and NYUv2 (Silberman et al., 2012). The notation * denotes models that have been fine-tuned on OmniWorld.

Method Align

Sintel Bonn KITTI Abs Rel ↓ 𝛿<1.25 ↑ Abs Rel ↓ 𝛿<1.25 ↑ Abs Rel ↓ 𝛿<1.25 ↑ DUSt3R (Wang et al., 2024c)

scale

0.652 0.436 0.151 0.839 0.143 0.814 DUSt3R* 0.512 0.456 0.083 0.920 0.135 0.800 CUT3R (Wang et al., 2025b)

scale

0.417 0.510 0.078 0.937 0.123 0.875 CUT3R* 0.396 0.516 0.078 0.938 0.107 0.907 DUSt3R (Wang et al., 2024c)

scale&shift

0.570 0.493 0.152 0.835 0.135 0.818 DUSt3R* 0.520 0.480 0.084 0.914 0.136 0.808 CUT3R (Wang et al., 2025b)

scale&shift

0.537 0.556 0.075 0.944 0.111 0.884 CUT3R* 0.314 0.574 0.067 0.964 0.103 0.912

- Table 6. Comparison of Original and Fine-tuned Models for Video Depth Estimation on Sintel (Butler et al., 2012), Bonn (Palazzolo et al., 2019) and KITTI (Geiger et al., 2013). The notation * denotes models that have been fine-tuned on OmniWorld.

### 4. Model Fine-tuning and Efficacy Validation

Through comprehensive experiments, we systematically validate OmniWorld as a training source. We select baselines for two core tasks: 3D geometric foundation models and camera-controlled video generation models, and fine-tuned them using OmniWorld. The experimental results clearly demonstrate that models fine-tuned with OmniWorld consistently achieve significant performance improvements over their original published versions, powerfully confirming OmniWorld’s capabilities in spatio-temporal modeling.

##### 4.1. Improving 3D Geometric Prediction with OmniWorld

We select DUSt3R (Wang et al., 2024c), CUT3R (Wang et al., 2025b), and Reloc3r (Dong et al., 2024) as our primary baselines and conduct fine-tuning experiments on subsets of OmniWorld.

The quantitative results confirm that models fine-tuned with OmniWorld consistently surpass their original performance across multiple critical tasks: monocular depth estimation (Table 5), video depth estimation (Table 6), and camera pose estimation. This outcome strongly demonstrates that OmniWorld’s scale and diversity enable it to serve as a valuable large-scale training source, effectively enhancing the generalization capabilities and robustness of 3D geometric foundation models.

For monocular depth estimation (Table 5), fine-tuned DUSt3R significantly outperformed its original baseline performance, even surpassing MonST3R, which is fine-tuned on multiple dynamic datasets (Mehl et al., 2023; Sun et al., 2020; Wang et al., 2020; Zheng et al., 2023). Similarly, CUT3R

FVD

Method Benchmark TransErr↓ RotErr↓ CamMC↓

VideoGPT↓ StyleGAN↓

3.4433 0.6308 3.6615 479.320 409.795 AC3D* 2.8648 0.5314 3.0518 472.683 416.948 AC3D (Bahmani et al., 2024)

AC3D (Bahmani et al., 2024)

RealEstate10K

6.2788 0.8867 6.6965 1745.778 1594.885 AC3D* 4.1428 0.7610 4.4854 1437.247 1249.1858

OmniWorld-Game

- Table 7. Comparison of Original and Fine-tuned Models for Camera-Controlled Video Generation Evaluation on RealEstate10K (Zhou et al., 2018) and OmniWorld-Game benchmark. The notation * denotes models that have been fine-tuned on OmniWorld.

also showed improved performance after fine-tuning compared to the original baseline.

For video depth estimation (Table 6), both DUSt3R and CUT3R exhibited enhanced performance after fine-tuning on OmniWorld, demonstrating OmniWorld’s utility in improving temporal consistency.

For camera pose estimation, please refer to supplementary materials.

- 4.2. Enhancing Camera-Controlled Video Generation with OmniWorld

Current public datasets for camera-controlled video generation models have significant limitations. For example, most datasets like RealEstate10K (Zhou et al., 2018) primarily consist of static scenes and relatively smooth camera movements, which hinders models’ ability to generate dynamic video content.

To address this data bottleneck and validate OmniWorld’s effectiveness, we select AC3D (Bahmani

- et al., 2024) as our baseline and fine-tune it. Our experimental results further verify the finding from prior work (e.g., CAMERACTRL II (He et al., 2025a)), which highlight the critical importance of dynamic data for improving a model’s camera-controlled capabilities.

The fine-tuned model is evaluated on two distinct benchmarks: a random subset of 150 video samples from the RealEstate10K test set and OmniWorld-Game benchmark, which consists of 200 video samples. For a fair comparison, all models are configured to output videos at a uniform resolution of 720 × 480 with a sequence length of 25 frames.

As shown in Table 7, the model fine-tuned on OmniWorld significantly outperforms the original baseline model on both the RealEstate10K (Zhou et al., 2018) and OmniWorld-Game benchmarks. This outcome provides strong evidence that OmniWorld serves as an effective training resource, substantially enhancing the ability of controllable video generation models to follow precise cameracontrolled instructions in complex and dynamic scenarios.

- 5. Related Work

- 5.1. World Model Datasets

The ability of models to perform world modeling is intrinsically linked to the availability of large-scale, high-quality spatio-temporal datasets.

Static 3D datasets, such as ScanNet (Dai et al., 2017), NYU-v2 (Silberman et al., 2012), and MegaDepth (Li and Snavely, 2018), have advanced 3D reconstruction by providing precise geometric information. However, their static nature limits their utility for modeling motion and dynamic interactions. In video generation, large-scale video-text datasets (Bain et al., 2021; Chen et al., 2024;

Ju et al., 2024; Nan et al., 2024) offer rich semantic annotations but lack geometric information (e.g., depth, camera poses, optical flow), making them unsuitable for applications requiring precise 3D world modeling.

To bridge this gap, researchers have created dynamic real-world datasets like KITTI (Geiger et al., 2013) and Waymo (Sun et al., 2020) for autonomous driving, and Bonn (Palazzolo et al., 2019), HOI4D (Liu et al., 2022), RH20T (Fang et al., 2024), and EPIC-Kitchens (Damen et al., 2018) for human-robot interaction. While valuable, these datasets often suffer from a lack of scene diversity and noisy/sparse geometric annotations.

The sim-to-real gap has been significantly reduced due to the advancement of modern rendering technology (Wang et al., 2020). Synthetic datasets have emerged as a valuable alternative, providing rich and precise ground-truth annotations. Pioneers like MPI Sintel (Butler et al., 2012) are instrumental in optical flow research, but their small scale (e.g., an average sequence length of less than 50 frames) is insufficient for training large-scale foundation models. Other recent synthetic datasets, such as FlyingThings++ (Harley et al., 2022; Mayer et al., 2016), TartanAir (Wang et al., 2020), Dynamic Replica (Karaev et al., 2023) and Spring (Mehl et al., 2023), have made progress but still fall short in terms of scale, diversity, and modal richness compared to our self-collected OmniWorld-Game dataset, as shown in Table 1.

The design of OmniWorld aims to systematically address these limitations. By integrating selfcollected OmniWorld-Game dataset and several public datasets from various domains, we provide highprecision geometric annotations and rich spatio-temporal dynamics, enabling a more comprehensive evaluation and enhancement of world modeling.

##### 5.2. 3D Geometric Foundation Models

Recently, 3D geometric foundation models have emerged as a data-driven alternative to traditional methods like Structure-from-Motion (SfM), capable of directly predicting a scene’s 3D structure in a single feed-forward pass. Early works like DUSt3R (Wang et al., 2024c) and MonST3R (Zhang et al.,

- 2024) operate on image pairs, requiring expensive global alignment for larger scenes. To overcome this, Fast3R (Yang et al., 2025) enables simultaneous inference on thousands of images.

Other methods explore simplifying the learning task. FLARE (Zhang et al., 2025) decomposes the problem into separate pose and geometry prediction steps. CUT3R (Wang et al., 2025b) is an online model that continuously updates its state from an image stream. VGGT (Wang et al.,

- 2025a) achieves superior performance through multi-task learning, while 𝜋3 (Wang et al., 2025d) employs a permutation-equivariant architecture to remove the dependency on a fixed reference view. For monocular inputs, MoGe (Wang et al., 2024b, 2025c) achieves accurate monocular geometry estimation by predicting affine-invariant point maps.

The performance of these methods is highly dependent on being trained on large-scale, multi-modal spatio-temporal datasets. When evaluated on the OmniWorld-Game benchmark, these methods show room for improvement, particularly when handling long sequences with highly dynamic, complex motions. By fine-tuning these models on OmniWorld, we achieve significant performance gains, powerfully demonstrating OmniWorld’s value as an effective training resource for enhancing models’ spatio-temporal modeling capabilities.

##### 5.3. Camera-Controlled Video Generation

Camera-controlled video generation aims to empower users with the ability to control the camera within a generated video. Most methods in this field inject camera parameters (such as extrinsics or

Plücker embeddings) into a pre-trained video diffusion model (Blattmann et al., 2023; Chen et al., 2023; Yang et al., 2024b) with representative works including MotionCtrl (Wang et al., 2024d), CameraCtrl (He et al., 2024), CAMI2V (Zheng et al., 2024), and AC3D (Bahmani et al., 2024).

Despite this progress, these methods still struggle to generate dynamic content with complex camera control. They are typically trained on datasets like RealEstate10K (Zhou et al., 2018) or DL3DV-10K (Ling et al., 2024), which consist of static scenes with smooth camera motions. This data limitation inherently restricts a models’ ability to handle dynamic scenes (He et al., 2025a).

Our experiments confirm this limitation. When evaluated on OmniWorld-Game benchmark, which features rich dynamics and complex camera movements, these methods show considerable room for improvement in both visual quality and camera-controlled accuracy. By fine-tuning them on OmniWorld, their performance in dynamic scenes is significantly enhanced, demonstrating our dataset’s value for improving models’ spatio-temporal modeling capabilities.

### 6. Conclusion

In this work, we introduce OmniWorld, a large-scale, multi-domain, and multi-modal dataset designed to address the critical data bottleneck for world modeling. By integrating self-collected OmniWorldGame dataset and several public datasets from various domains, we create a comprehensive data resource for world modeling. We demonstrate that OmniWorld-Game serves as a challenging benchmark for 3D geometric foundation models and camera-controlled video generation models, revealing the limitations of current SOTAs. Furthermore, we provide strong evidence that fine-tuning with OmniWorld significantly boosts the performance of these models, underscoring its value as a powerful training resource. We believe that OmniWorld will serve as a crucial data resource for the community, accelerating the development of more general and robust models for understanding and interacting with the real physical world.

### References

- N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

S. Bahmani, I. Skorokhodov, G. Qian, A. Siarohin, W. Menapace, A. Tagliasacchi, D. B. Lindell, and

- S. Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2024.

J. Bai, M. Xia, X. Fu, X. Wang, L. Mu, J. Cao, Z. Liu, H. Hu, X. Bai, P. Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025.

- M. Bain, A. Nagrani, G. Varol, and A. Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.

- G. Baruch, Z. Chen, A. Dehghan, T. Dimry, Y. Feigin, P. Fu, T. Gebauer, B. Joffe, D. Kurz, A. Schwartz, and E. Shulman. ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. URL https://openreview.net/forum?id=tjZjv_qh_CE.

A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, X. He, X. Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025.

D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In European Conference on Computer Vision, 2012. URL https://api. semanticscholar.org/CorpusID:4637111.

H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.

- J. Chen, H. Zhu, X. He, Y. Wang, J. Zhou, W. Chang, Y. Zhou, Z. Li, Z. Fu, J. Pang, et al. Deepverse: 4d autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025.

T.-S. Chen, A. Siarohin, W. Menapace, E. Deyneka, H.-w. Chao, B. E. Jeon, Y. Fang, H.-Y. Lee, J. Ren,

- M.-H. Yang, and S. Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

- Y. Chen, C. Schmid, and C. Sminchisescu. Self-supervised learning with geometric constraints in monocular video: Connecting flow, depth, and camera. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7063–7072, 2019.

O. Contributors. Obs studio, 2024. URL https://obsproject.com/. A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner. Scannet: Richly-annotated

- 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price, and M. Wray. Scaling egocentric vision: The epic-kitchens dataset. In European Conference on Computer Vision (ECCV), 2018.

- G. DeepMind. Genie 3. https://deepmind.google/discover/blog/ genie-3-a-new-frontier-for-world-models/, 2025. Accessed: 2025-08-27.

D. DeTone, T. Malisiewicz, and A. Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 224–236, 2018.

S. Dong, S. Wang, S. Liu, L. Cai, Q. Fan, J. Kannala, and Y. Yang. Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization. arXiv preprint arXiv:2412.08376, 2024.

- H.-S. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 653–660. IEEE, 2024.

- R. Feng, H. Zhang, Z. Yang, J. Xiao, Z. Shu, Z. Liu, A. Zheng, Y. Huang, Y. Liu, and H. Zhang. The matrix: Infinite-horizon world generation with real-time moving control, 2024. URL https: //arxiv.org/abs/2412.03568.
- S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh,

- J. Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36:27092–27112, 2023.

- A. Geiger, P. Lenz, C. Stiller, and R. Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013.
- K. Grauman, A. Westbury, L. Torresani, K. Kitani, J. Malik, T. Afouras, K. Ashutosh, V. Baiyya, S. Bansal,

- B. Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400, 2024.

D. Ha and J. Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018. D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse domains through world models.

arXiv preprint arXiv:2301.04104, 2023. A. Hagemann, M. Knorr, and C. Stiller. Deep geometry-aware camera self-calibration from video. In

Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3438–3448, 2023. A. W. Harley, Z. Fang, and K. Fragkiadaki. Particle video revisited: Tracking through occlusions using

point trajectories. In European Conference on Computer Vision, pages 59–75. Springer, 2022. H. He, Y. Xu, Y. Guo, G. Wetzstein, B. Dai, H. Li, and C. Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

H. He, C. Yang, S. Lin, Y. Xu, M. Wei, L. Gui, Q. Zhao, G. Wetzstein, L. Jiang, and H. Li. Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models. arXiv preprint arXiv:2503.10592, 2025a.

- X. He, C. Peng, Z. Liu, B. Wang, Y. Zhang, Q. Cui, F. Kang, B. Jiang, M. An, Y. Ren, et al. Matrixgame 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025b.

- R. Hoque, P. Huang, D. J. Yoon, M. Sivapurapu, and J. Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

- X. Ju, Y. Gao, Z. Zhang, Z. Yuan, X. Wang, A. Zeng, Y. Xiong, Q. Xu, and Y. Shan. Miradata: A large-scale video dataset with long durations and structured captions, 2024. URL https: //arxiv.org/abs/2407.06358.

N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. CVPR, 2023.

N. Karaev, I. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. arXiv preprint arXiv:2410.11831, 2024.

- A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

- A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick. Segment anything. arXiv:2304.02643, 2023.

H. Le, P. Das, T. Mensink, S. Karaoglu, and T. Gevers. EDEN: Multimodal Synthetic Dataset of Enclosed garDEN Scenes. In Proceedings of the IEEE/CVF Winter Conference of Applications on Computer Vision (WACV), 2021.

- Y. LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.

- V. Leroy, Y. Cabon, and J. Revaud. Grounding image matching in 3d with mast3r, 2024.

- Z. Li and N. Snavely. Megadepth: Learning single-view depth prediction from internet photos. In Computer Vision and Pattern Recognition (CVPR), 2018.

- Z. Li, R. Tucker, F. Cole, Q. Wang, L. Jin, V. Ye, A. Kanazawa, A. Holynski, and N. Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In arxiv, 2024.

Z. Li, C. Li, X. Mao, S. Lin, M. Li, S. Zhao, Z. Xu, X. Li, Y. Feng, J. Sun, Z. Li, F. Zhang, J. Ai, Z. Wang,

- Y. Wu, T. He, J. Pang, Y. Qiao, Y. Jia, and K. Zhang. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025.

- L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

- S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.

- Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21013–21022, June 2022.

- D. G. Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004.

N. Mayer, E. Ilg, P. Hausser, P. Fischer, D. Cremers, A. Dosovitskiy, and T. Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

L. Mehl, J. Schmalfuss, A. Jahedi, Y. Nalivayko, and A. Bruhn. Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

H. Morimitsu, X. Zhu, R. M. Cesar, X. Ji, and X.-C. Yin. Dpflow: Adaptive optical flow estimation with a dual-pyramid framework. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17810–17820, 2025.

K. Nan, R. Xie, P. Zhou, T. Fan, Z. Yang, Z. Chen, X. Li, J. Yang, and Y. Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.

- E. Palazzolo, J. Behley, P. Lottes, P. Giguère, and C. Stachniss. ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. arXiv, 2019. URL https: //arxiv.org/abs/1905.02082.

N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson,

- E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

J. Reizenstein, R. Shapovalov, P. Henzler, L. Sbordone, P. Labatut, and D. Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021.

ReShade Contributors. ReShade, 2024. URL https://reshade.me/.

- S. R. Richter, V. Vineet, S. Roth, and V. Koltun. Playing for data: Ground truth from computer games. In European conference on computer vision, pages 102–118. Springer, 2016.

M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, R. Webb, and J. M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021.

Rockstar Games. Policy on Posting Copyrighted Rockstar Games Material, 2024. URL https://support.rockstargames.com/articles/7bNaeoMFTV0iUDGhStTXvz/ policy-on-posting-copyrighted-rockstar-games-material.

C. Rockwell, J. Tung, T.-Y. Lin, M.-Y. Liu, D. F. Fouhey, and C.-H. Lin. Dynamic camera poses and where to find them. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12444–12455, 2025.

C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. URL https://arxiv.org/abs/2210.08402.

F. Sener, D. Chatterjee, D. Shelepov, K. He, D. Singhania, R. Wang, and A. Yao. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. CVPR, 2022.

N. Silberman, D. Hoiem, P. Kohli, and R. Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.

J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573–580. IEEE, 2012.

P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.

Z. Tang, Y. Fan, D. Wang, H. Xu, R. Ranjan, A. Schwing, and Z. Yan. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. arXiv preprint arXiv:2412.06974, 2024.

- A. Team, H. Zhu, Y. Wang, J. Zhou, W. Chang, Y. Zhou, Z. Li, J. Chen, C. Shen, J. Pang, and T. He. Aether: Geometric-aware unified world modeling. arXiv preprint arXiv:2503.18945, 2025a.

H. Team, Z. Wang, Y. Liu, J. Wu, Z. Gu, H. Wang, X. Zuo, T. Huang, W. Li, S. Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025b.

Z. Teed and J. Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020.

- T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025a.

- P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du,

- X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

- Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025b.
- R. Wang, S. Xu, C. Dai, J. Xiang, Y. Deng, X. Tong, and J. Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision, 2024b. URL https://arxiv.org/abs/2410.19115.

- R. Wang, S. Xu, Y. Dong, Y. Deng, J. Xiang, Z. Lv, G. Sun, X. Tong, and J. Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details, 2025c. URL https://arxiv.org/abs/ 2507.02546.
- S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024c.

- W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, and S. Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020.
- X. Wang, T. Kwon, M. Rad, B. Pan, I. Chakraborty, S. Andrist, D. Bohus, A. Feniello, B. Tekin,

- F. V. Frujeri, N. Joshi, and M. Pollefeys. Holoassist: an egocentric human interaction dataset for interactive ai assistants in the real world. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20270–20281, October 2023.

- Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He. 𝜋3: Scalable permutation-equivariant visual geometry learning, 2025d. URL https://arxiv.org/abs/2507. 13347.
- Z. Wang, Z. Yuan, X. Wang, Y. Li, T. Chen, M. Xia, P. Luo, and Y. Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024d.

Z. Wang, S. Chen, L. Yang, J. Wang, Z. Zhang, H. Zhao, and Z. Zhao. Depth anything with any prior,

#### 2025e. URL https://arxiv.org/abs/2505.10565.

- B. Wen, M. Trepte, J. Aribido, J. Kautz, O. Gallo, and S. Birchfield. Foundationstereo: Zero-shot stereo matching. CVPR, 2025.

H. Xia, Y. Fu, S. Liu, and X. Wang. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22378–22389, 2024.

J. Xing, M. Xia, Y. Zhang, H. Chen, W. Yu, H. Liu, X. Wang, T.-T. Wong, and Y. Shan. Dynamicrafter:

Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023. H. Yang, D. Huang, W. Yin, C. Shen, H. Liu, X. He, B. Lin, W. Ouyang, and T. He. Depth any video

with scalable synthetic data. arXiv preprint arXiv:2410.10815, 2024a.

J. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025.

Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024b.

- Y. Yao, Z. Luo, S. Li, J. Zhang, Y. Ren, L. Zhou, T. Fang, and L. Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1790–1799, 2020.

M. Ye, P. Yin, W.-C. Lee, and D.-L. Lee. Exploiting geographical influence for collaborative point-ofinterest recommendation. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval, pages 325–334, 2011.

- C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

J. Yu, J. Bai, Y. Qin, Q. Liu, X. Wang, P. Wan, D. Zhang, and X. Liu. Context as memory: Scene-consistent

interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025a. J. Yu, Y. Qin, X. Wang, P. Wan, D. Zhang, and X. Liu. Gamefactory: Creating new games with

generative interactive videos, 2025b. M. YU, W. Hu, J. Xing, and Y. Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025.

C. Yuan, S. Joshi, S. Zhu, H. Su, H. Zhao, and Y. Gao. Roboengine: Plug-and-play robot data augmentation with semantic robot segmentation and background generation. arXiv preprint arXiv:2503.18738, 2025.

J. Zhang, C. Herrmann, J. Hur, V. Jampani, T. Darrell, F. Cole, D. Sun, and M.-H. Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arxiv:2410.03825, 2024.

- L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models, 2023.

- S. Zhang, J. Wang, Y. Xu, N. Xue, C. Rupprecht, X. Zhou, Y. Shen, and G. Wetzstein. Flare: Feedforward geometry, appearance and camera estimation from uncalibrated sparse views, 2025. URL https://arxiv.org/abs/2502.12138.

- G. Zheng, T. Li, R. Jiang, Y. Lu, T. Wu, and X. Li. Cami2v: Camera-controlled image-to-video diffusion model. arXiv preprint arXiv:2410.15957, 2024.

Y. Zheng, A. W. Harley, B. Shen, G. Wetzstein, and L. J. Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In ICCV, 2023.

- T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely. Stereo magnification: Learning view synthesis using multiplane images. In SIGGRAPH, 2018.

### A. Overview

Section B discusses more details of OmniWorld. Section C and Section D discuss more details of our benchmark and fine-tuning experiments.

### B. OmniWorld Dataset

##### B.1. Data Statistics

[Figure 308]

- Figure 7. The OmniWorld-Game distribution of scene category (the primary POI locations).

To quantitatively analyze the scene diversity of OmniWorld-Game, we adopt the methodology from DL3DV (Ling et al., 2024) to classify and count scenes across 16 Point-of-Interest (POI) categories (Ye et al., 2011). The statistical results are shown in Figure 7. OmniWorld-Game encompasses a wide variety of scene categories, including "Nature & Outdoors," "Tourist Attractions," "Parks and Recreation," and "Hotels and Accommodations." "Nature & Outdoors" represents the largest share, reflecting its dominant presence in the dataset. The distribution of these scene categories aligns with their prevalence in the real world and the characteristics of the games themselves. For instance, scenes related to "Government & Civic Services" and "Events & Conferences" are typically less frequent in games, leading to their lower representation in our dataset. These statistics further validate the richness and real-world attributes of OmniWorld-Game.

To provide a more detailed analysis of the dominant "Nature & Outdoors" scenes in OmniWorldGame, we further subdivide this category into 5 second-level and 40 third-level categories. The detailed distribution is shown in Figure 8. Our statistics reveal that "Natural Landforms & Ecosystems" is the dominant second-level category. Within this category, scenes depicting "Forests & Rainforests"

[Figure 309]

- Figure 8. Scene Diversity within the "Nature & Outdoors" Category. A quantitative breakdown of second- and third-level scene categories in OmniWorld-Game dataset, demonstrating the high internal diversity and distribution of natural environments.

and "Cliffs & Rock Formations" are the most prevalent. "Outdoor Sports & Scenic Routes" is the second-largest category, with scenes of "Rock-Climbing Areas" and "Scenic Drives & Viewpoints" being particularly prominent. Additionally, "Urban Outdoor Spaces & Activities" and "Agricultural & Rural Landscapes" also make up a small portion of the data. These detailed statistics confirm that the "Nature & Outdoors" scenes in OmniWorld-Game are not only abundant but also internally diverse. This rich composition provides a diverse data source for world modeling in complex natural environments.

- B.2. Ethics Statements

To ensure compliance, we strictly adhere to the terms of use for relevant game content (e.g. Rockstar Games (Rockstar Games, 2024)), including usage for non-commercial purposes only and avoiding story spoilers. We also automatically remove UI elements and text information via a ReShade (ReShade Contributors, 2024) plugin and manually filter specific scenes to ensure no unauthorized content is disclosed.

- C. OmniWorld-Game Benchmark

- C.1. 3D Geometric Prediction

Experiment Details. We adhere to the default configurations of each evaluated model. The entire evaluation process is conducted on a single A800 GPU.

For the monocular depth Estimation, we evaluate the first 200 frames of 18 test sequence from the OmniWorld-Game benchmark. Following the evaluation protocols of prior works (Wang et al., 2025b,d; Zhang et al., 2024), we focus on scale-invariant monocular depth accuracy. The primary evaluation metrics are Absolute Relative Error (Abs Rel) and threshold accuracy (𝛿 < 1.25). Under this setting, the depth map of each frame is independently aligned with its corresponding ground truth.

Sintel TUM-dynamics ScanNet ATE↓ RPE trans↓ RPE rot↓ ATE↓ RPE trans↓ RPE rot↓ ATE↓ RPE trans↓ RPE rot↓

Method

CUT3R (Wang et al., 2025b) 0.210 0.071 0.627 0.045 0.014 0.441 0.096 0.022 0.733 CUT3R* 0.178 0.055 0.651 0.041 0.013 0.374 0.095 0.022 0.604

- Table 8. Comparison of Original and Fine-tuned Models for Camera Pose Estimation on Sintel (Butler et al., 2012), TUM-dynamics (Sturm et al., 2012) and ScanNet (Dai et al., 2017). The notation * denotes models that have been fine-tuned on OmniWorld.

Method

DynPose-100K OmniWorld-CityWalk

AUC@5↑ AUC@10↑ AUC@20↑ AUC@5↑ AUC@10↑ AUC@20↑

Reloc3r (Dong et al., 2024) 6.9 15.4 27.1 33.3 49.4 63.1 Reloc3r* 14.4 25.5 37.8 42.5 58.0 70.3

- Table 9. Comparison of Original and Fine-tuned Models for Relative Camera Pose Evaluation on DynPose-100K (Rockwell et al., 2025), OmniWorld-CityWalk(Li et al., 2025). The notation * denotes models that have been fine-tuned on OmniWorld.

For the video depth estimation, we select the first 100 frames of the same test sequence from the OmniWorld-Game benchmark. To ensure a fair comparison across all models, we cap the input sequence length at 100 frames, as some models (e.g., FLARE (Zhang et al., 2025)) cannot handle longer sequences without errors. Similar to the mono depth estimation, we report Abs Rel and 𝛿 < 1.25. To more comprehensively evaluate depth consistency across video sequences, we provide results under two different alignment settings: (i) scale-only alignment (scale) and (ii) combined scale and translation alignment (scale & shift). These settings test a model’s depth estimation capabilities under different constraints, particularly in handling motion and viewpoint changes.

It is important to note that since the benchmark data is included in the training set of 𝜋3 (Wang

- et al., 2025d), we did not evaluate it in our benchmark.

- C.2. Camera-Controlled Video Generation

Experiment Details. AC3D (Bahmani et al., 2024) uses CogVideoX-5B (Yang et al., 2024b) as base T2V model, it generates 25 frames per inference at a resolution of 480 × 720. CamCtrl (He et al., 2024) and MotionCtrl (Wang et al., 2024d) use Stable Video Diffusion (SVD) (Blattmann et al., 2023) as base I2V model and generate 14-frame video sequences at a resolution of 320 × 512. CAMI2V (Zheng

- et al., 2024) uses DynamiCrafter (Xing et al., 2023) as base I2V model. It generates 16-frame video sequences at a resolution of 320 × 512. For a fair comparison with CamCtrl and MotionCtrl, we use the first 14 frames of its generated videos for evaluation. We use 𝜋3 (Wang et al., 2025d) to get camera poses of the generated videos. All methods are evaluated on an A800 GPU.

- D. Model Fine-tuning

- D.1. Camera Pose Estimation.

Following (Wang et al., 2025b,d), we report the Absolute Trajectory Error (ATE), Relative Pose Error for translation (RPE trans), and Relative Pose Error for rotation (RPE rot) on Sintel (Butler et al., 2012), TUM-dynamics (Sturm et al., 2012) and ScanNet (Dai et al., 2017). The results in Table 8 show that CUT3R’s performance notably improved after fine-tuning on OmniWorld in camera pose estimation.

Following (Dong et al., 2024), we assess performance with three indicators: AUC@5/10/20, which measure the area under the pose accuracy curve. This curve is based on minimum thresholds of

- 5, 10, and 20 degrees for rotation and translation angular errors. Reloc3r demonstrated substantial improvements in its ability to estimate dynamic camera poses after fine-tuning on OmniWorld in relative camera pose evaluation (Table 9).

D.2. Implementation Details We conduct comprehensive fine-tuning experiments on several SOTAs to validate the efficacy of our OmniWorld as a training resource. All experiments are performed on 8 NVIDIA A800 GPUs.

DUSt3R (Wang et al., 2024c). For fine-tuning, we use OmniWorld-Game alongside a portion of DUSt3R’s original training sets, including ARKitScenes (Baruch et al., 2021), MegaDepth (Li and Snavely, 2018), and Waymo (Sun et al., 2020). We load the pre-trained weights of DUSt3R and performed full fine-tuning. The model is fine-tuned on images with random resolutions (e.g., 288×512, 384×512, 336×512). The training runs for 40 epochs, with each epoch consisting of 800 iterations. We use the AdamW optimizer with an initial learning rate of 2.5 × 10−5 and a weight decay of 0.05. Each GPU had a batch size of 7, with each batch containing two images.

CUT3R (Wang et al., 2025b). We fine-tune CUT3R using OmniWorld-Game and a subset of its original training data, including CO3Dv2 (Reizenstein et al., 2021), WildRGBD (Xia et al., 2024), ARKitScenes (Baruch et al., 2021), Waymo (Sun et al., 2020), and TartanAir (Wang et al., 2020). We load the pre-trained weights and follow the training strategy from CUT3R’s training stage 3. We fine-tune on higher-resolution images with varied aspect ratios, setting the maximum side to 512 pixels. The encoder is frozen, with only the decoder and heads being trained on longer sequences of 4 to 64 views. The model is fine-tuned for 2,000 iterations with a total batch size of 96 and a learning rate of 1.0 × 10−6, optimized by AdamW with a weight decay of 0.05.

Reloc3r (Dong et al., 2024). For fine-tuning Reloc3r, we utilize OmniWorld-Game, OmniWorldCityWalk, OmniWorld-HoloAssist, and OmniWorld-EpicKitchens, along with a portion of its original training sets, including CO3Dv2 (Reizenstein et al., 2021), ARKitScenes (Baruch et al., 2021), Scannet++ (Yeshwanth et al., 2023), BlendedMVS (Yao et al., 2020), and MegaDepth (Li and Snavely, 2018). We load the pre-trained weights, freeze the ViT encoder, and only update the weights for the decoder and pose regression head. Fine-tuning is performed on images of random resolutions, including 288 × 512, 384 × 512, and 336 × 512. The model is trained for 80 epochs, with each epoch comprising 400 iterations. We use the AdamW optimizer with a learning rate of 5.0 × 10−6 and a weight decay of 0.05. Each GPU has a batch size of 32, with each batch containing two images.

AC3D (Bahmani et al., 2024). We fine-tune AC3D using OmniWorld-Game, OmniWorld-EpicKitchens, OmniWorld-HOI4D, OmniWorld-HoloAssist, OmniWorld-EgoExo4D, and OmniWorld-EgoDex, as well as the original training set, RealEstate10K (Zhou et al., 2018). We load the pre-trained weights of the AC3D ControlNet (Zhang et al., 2023), which is based on CogVideoX-5B (Yang et al., 2024b). Only the ControlNet model is fine-tuned, with other network structures frozen. The fine-tuning is performed on video clips of 49 frames with a resolution of 352 × 640. The model is fine-tuned for

- 6,000 iterations with a total batch size of 8 and a learning rate of 5.0 × 10−5, optimized by AdamW with a weight decay of 0.0001.

##### D.3. Visual Results.

- Figure 9 provides a qualitative comparison of DUSt3R (Wang et al., 2024c) and CUT3R (Wang

- et al., 2025b) on the Sintel (Butler et al., 2012) subset of the Video Depth Estimation benchmark,

RGB DUSt3R DUSt3R* CUT3R CUT3R*

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

- Figure 9. Qualitative comparison of Original and Fine-tuned Models for Video Depth Estimation on the Sintel (Butler et al., 2012). * denotes models that have been fine-tuned on OmniWorld. After fine-tuning, both models recover finer geometric details and produce more accurate depth maps, highlighting the efficacy of OmniWorld as a geometric supervision source.

"A character rides a horse along a dirt path through a lush forested area, surrounded by tall pine trees and distant snow-covered mountains under a partly cloudy sky."

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

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

AC3D

AC3D*

- Figure 10. Qualitative comparison of Original and Fine-tuned Models for Camera-Controlled Video Generation. * denotes models that have been fine-tuned on OmniWorld. The visualizations show that fine-tuning with our dataset significantly improves the model’s ability to generate videos that more accurately follow camera trajectories and maintain higher temporal consistency for moving objects.

evaluated both before and after fine-tuning on OmniWorld. After fine-tuning, both models recover finer geometric details and generate more accurate depth maps. These results indicate that OmniWorld offers strong geometric supervision and can substantially enhance a model’s geometric prediction capability.

Figure 10 presents a visual comparison of AC3D (Bahmani et al., 2024) on the OmniWorld-Game benchmark before and after fine-tuning on the OmniWorld dataset for the camera-controlled video generation task. The visualizations clearly show that after fine-tuning, the generated videos more

###### closely follow the desired camera trajectory and exhibit higher temporal consistency for moving objects. This demonstrates that OmniWorld can significantly enhance a model’s ability to model dynamics.

