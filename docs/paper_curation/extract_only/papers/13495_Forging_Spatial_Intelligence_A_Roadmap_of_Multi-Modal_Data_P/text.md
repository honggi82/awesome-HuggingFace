# arXiv:2512.24385v2[cs.CV]8Jan2026

[Figure 1]

## Forging Spatial Intelligence: A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems

Song Wang1, , Lingdong Kong2, , Xiaolu Liu1, , Hao Shi1, Wentong Li3, Jianke Zhu1, , Steven C. H. Hoi4,5

1Zhejiang University 2National University of Singapore 3Nanjing University of Aeronautics and Astronautics 4Alibaba Group 5Singapore Management University

[Figure 2]

WorldBench Team

[Figure 3]

[Figure 4]

Equal Contributions Corresponding Author

The rapid advancement of autonomous systems, including self-driving vehicles and drones, has intensified the need to forge true Spatial Intelligence from multi-modal onboard sensor data. While foundation models excel in single-modal contexts, integrating their capabilities across diverse sensors like cameras and LiDAR to create a unified understanding remains a formidable challenge. This paper presents a comprehensive framework for multi-modal pre-training, identifying the core set of techniques driving progress toward this goal. We dissect the interplay between foundational sensor characteristics and learning strategies, evaluating the role of platform-specific datasets in enabling these advancements. Our central contribution is the formulation of a unified taxonomy for pre-training paradigms: ranging from single-modality baselines to sophisticated unified frameworks that learn holistic representations for advanced tasks like 3D object detection and semantic occupancy prediction. Furthermore, we investigate the integration of textual inputs and occupancy representations to facilitate open-world perception and planning. Finally, we identify critical bottlenecks, such as computational efficiency and model scalability, and propose a roadmap toward general-purpose multi-modal foundation models capable of achieving robust Spatial Intelligence for real-world deployment.

[Figure 5]

GitHub Repo: https://github.com/worldbench/awesome-spatial-intelligence

1 Introduction

With the rapid proliferation of autonomous platforms, ranging from self-driving vehicles [72, 121, 165] and aerial drones [161, 302] to unmanned surface vehicles [52, 271], rail-based systems [113, 282, 325], and legged robots [55, 57], the challenge of endowing machines with the capability to perceive and act in the real world has reached an unprecedented level of complexity. As these systems are required to navigate diverse and dynamically evolving scenarios, they demand a robust and deeply nuanced understanding of their environment to support critical downstream functions, including navigation [4, 155], interaction [81], and planning [72, 142].

Central to these platforms is a sophisticated suite of onboard sensors, primarily comprising cameras, LiDAR, radar, and emerging event cameras, which collectively serve as the foundation for perception [17, 21, 47, 69]. Each modality contributes a unique and complementary stream of information, where cameras provide rich visual semantics [121, 244], LiDAR delivers precise 3D geometry [107, 125], and radar captures essential motion cues [178, 272], while event cameras offer microsecond-level temporal precision for high-speed dynamics [56, 103, 151, 266]. The effective integration of these heterogeneous data streams is paramount for achieving the holistic perception required for safe and generalizable autonomy [61, 98, 115, 188, 246]. In response to this

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Self-Supervised Learning

LiDAR

Background

IMU/GPS

Transfer Learning

[Figure 13]

Cross-Modality Interaction

Foundation Methods Model

Onboard Sensors

Camera Radar

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Platforms & Datasets

Vehicles

###### Drones

###### Others

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

UAV-3D UAVScenes

KITTI nuScenes

Rail3D Water-VG

[Figure 24]

Single Modality Pre-training Multi Modality Pre-training Incorporating Additional Sensors

Pre-Training

Camera-Only LiDAR-Only Camera-Centric LiDAR-Centric Radar Event Camera

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Applications

Detection Segmentation Occupancy Planning Grounding

- Figure 1 Overview of the paper structure. We systematically structure the landscape of multi-modal data pre-training for forging Spatial Intelligence. This work is organized into four key pillars: (1) Background, introducing onboard sensors and foundational learning paradigms; (2) Platforms & Datasets, analyzing benchmarks across autonomous vehicles, drones, and other robotic systems; (3) Pre-Training Methodologies, categorized into single-modality, cross-modal (Camera/LiDAR-centric), and unified frameworks; and (4) Applications, highlighting downstream tasks from 3D perception to open-world planning.

demand, the research community has curated numerous large-scale and sensor-centric datasets [22, 47, 198], alongside specialized benchmarks for drones [161, 302] and robotic agents [57].

While these datasets provide an invaluable empirical foundation, they simultaneously highlight a fundamental challenge that constitutes the central focus of this work. Most existing datasets heavily rely on costly manual annotations to support supervised learning paradigms [17, 47, 52], creating significant bottlenecks for scalability and generalization [97, 99, 204, 222, 245]. Consequently, there has been a growing interest in representation learning methods that aim to distill meaningful features directly from raw sensor data, alleviating the dependence on extensive human supervision [62–64, 239]. Particularly noteworthy is the emergence of foundation models, which facilitate large-scale, transferable pre-training across various domains, including vision [95, 181], 3D geometry [139, 298], and multi-modal scenarios [105, 264, 323]. Such foundation models provide a unified paradigm for extracting general-purpose representations from diverse sensor inputs [100, 157, 270], significantly enhancing cross-domain adaptability and paving the way for next-generation world models [71, 106, 124, 125, 145, 165, 256, 305].

As a result, pre-training strategies tailored specifically to sensor modalities have become an essential research frontier. As depicted in Fig. 1, these strategies form the core techniques to forge what we define as Spatial Intelligence – a capability that transcends simple detection to encompass holistic scene understanding, reasoning, and future prediction [126, 142, 270]. Current approaches include single-modality methods (e.g., LiDAR-only or camera-only) [3, 65, 107, 288], cross-modal knowledge transfer (e.g., distillation between camera and LiDAR) [104, 188, 270], and unified multi-modal pre-training frameworks [264, 323]. Understanding the overall landscape of these methods, as well as their connections to sensor characteristics, platform constraints, and the development of foundation models, is crucial for advancing robust and efficient perception capabilities in intelligent systems.

In this work, we systematically analyze the state-of-the-art techniques in representation learning from onboard sensor data, emphasizing multi-modal interactions and integration with foundation models. We first dissect foundational methodologies such as self-supervised learning, transfer learning, and multimodal learning, evaluating their respective strengths and limitations across various autonomous platforms, including self-driving vehicles, drones, robotic dogs, and rail transportation systems. By structuring and characterizing representative pre-training approaches according to modality composition, sensor interactions, and targeted applications, we highlight their adaptability to diverse sensor configurations and practical scenarios. Furthermore, we investigate key challenges in sensor representation learning, such as data sparsity, sensor noise, multi-modal alignment, and real-time processing demands. Finally, we propose promising directions for future research toward generative world models and embodied reasoning suitable for dynamic real-world environments.

- 1.1 Scope of the Work

Multi-modal representation learning from onboard sensors encompasses various related areas, such as singlemodality pre-training, cross-modal fusion, and foundation model integration. Given the breadth of these topics, it is impractical to exhaustively analyze all relevant methods within a single manuscript. Therefore, this work specifically concentrates on recent advances in foundation models for multi-modal representation learning, primarily focusing on onboard camera and LiDAR sensors [191, 264, 270]. Representative methods involving additional sensors such as radar and event cameras are also examined [37, 266, 267, 272].

We emphasize significant progress from the past five years, particularly highlighting influential works published in top-tier conferences and journals. In addition to technical approaches, we analyze widely adopted datasets, evaluation metrics, and sensor configurations. Finally, we investigate key challenges and outline promising future research directions.

- 1.2 Relation to Previous Studies

Several existing studies [237, 239, 260, 318, 321] have recently explored representation learning in autonomous systems, typically focusing on individual sensor modalities, specific autonomous platforms, or particular downstream tasks. While these efforts offer valuable insights into targeted aspects of the field, they often discuss sensors and tasks independently, lacking an integrated perspective on how single-modality and cross-modal approaches collectively advance multi-modal representation learning.

In contrast, our work presents a comprehensive framework emphasizing the role of foundation models within multi-modal representation learning from onboard sensors. We systematically analyze modality-specific pretraining as well as cross-modal interactions and unified frameworks, clearly highlighting how these methods interconnect and contribute to robust, generalizable perception across various platforms and tasks. By bridging single-modality strategies with unified multi-modal paradigms, this study uniquely facilitates an in-depth understanding of recent advances, emerging trends, and future directions in multi-modal pre-training for autonomous systems.

- 1.3 Organization

To provide a clear roadmap of the field, we present a comprehensive taxonomy encompassing datasets, pre-training paradigms, and downstream applications, as illustrated in Fig. 3. The remainder of this paper is structured as follows:

- • Section 2 introduces the foundations of data representation learning for onboard sensors, covering sensor characteristics, pre-training paradigms, and the role of foundation models.
- • Section 3 analyzes platform-specific datasets, including those collected from autonomous vehicles, aerial drones, and other robotic systems.
- • Section 4 provides a comprehensive analysis of pre-training methods, categorized by sensor modality, interaction level, and downstream application.
- • Section 5 investigates recent progress in open-world perception and planning, focusing on text-assisted understanding and the shift toward generative world models for end-to-end autonomy.

- • Section 6 outlines key challenges in current research and highlights promising future directions.
- • Section 7 concludes the paper with a summary of major insights and takeaways.

### 2 Background

Multi-modal pre-training from onboard sensors serves as the bedrock for forging Spatial Intelligence in autonomous systems. It aims to transcend simple feature extraction by distilling compact, discriminative, and semantically rich representations from diverse sensory inputs. By effectively integrating complementary information from modalities such as cameras, LiDAR and radar, these methods enable foundation models to not only perceive geometry and semantics but also reason about dynamics and affordances [126, 270]. In the context of autonomous deployment, developing scalable and reliable multi-modal pre-training approaches is essential for achieving robust open-world generalization and bridging the gap between passive perception and active embodied reasoning.

- 2.1 Onboard Sensors and Data Characteristics

The sensory apparatus of intelligent agents, primarily comprising cameras, LiDAR, radar, and event cameras, presents a heterogeneous data landscape characterized by distinct modalities and formats. Cameras provide dense semantic and textural information essential for scene understanding [121, 270], yet they remain susceptible to environmental variations such as illumination changes and adverse weather conditions. In contrast, LiDAR sensors deliver precise 3D geometric structures via point clouds [107, 156, 243], which offer robustness against lighting variations but suffer from inherent sparsity and limited semantic richness. Radar provides robust Doppler velocity cues even in adverse weather, albeit at lower spatial resolution [37, 272]. Complementing these with microsecond-level temporal precision, event cameras capture asynchronous brightness changes to handle high-speed dynamics and motion blur inherent in standard vision [101–103, 266, 308]. Understanding the inherent properties of these sensors, specifically the trade-off between the semantic richness of vision and the geometric precision of ranging sensors, is fundamental. Effective representation learning must address these disparities to construct a unified and coherent world model.

- 2.2 Paradigms of Representation Learning

To forge spatial intelligence from the heterogeneous data streams described above, a robust methodological framework is required. The evolution of this framework has followed a clear logical progression, as chronologically illustrated in Fig. 2. Initially, to overcome the immense cost of manual annotation, the field turned to Self-Supervised Learning to extract meaningful features directly from vast quantities of unlabeled data. A natural next step was to exploit the complementary nature of different sensors through Cross-Modality Interaction, creating a more holistic representation than any single sensor could provide. Concurrently, Knowledge Distillation and Transfer Learning emerged as a powerful technique to leverage priors from well-established vision foundation models, accelerating progress in 3D domains. Ultimately, these distinct paradigms are being synthesized under a unified vision: the development of Foundation Models and Generative World Models.

- 2.2.1 Self-Supervised Learning

Self-supervised learning (SSL) has emerged as the dominant paradigm for representation learning from unlabeled sensor data [20, 62, 63, 170]. By defining suitable pretext tasks, models leverage supervisory signals inherently present within the data. Classic strategies include Contrastive Learning, which discriminates between augmented views of the same instance, and Masked Modeling, which reconstructs obscured portions of inputs [63, 96, 107]. More recently, Generative Modeling (e.g., next-token prediction or video generation) has gained prominence [8, 305]. By learning to predict future frames or occupancy states, these methods enable models to internalize the physics and dynamics of the environment, serving as a precursor to world models.

- 2.2.2 Cross-Modality Interaction

#### Cross-modal interaction methods aim to fuse disparate sensor modalities into a unified representation space, enhancing both robustness and semantic depth [165, 191, 255, 270]. For instance, projecting dense visual

2020 - 2021

Jan 2025 - Present

EventFly

AutoOcc TT-Occ

4D-ROLLS

TurboTrain

SPOT

PointContrast

SimIPU

AssociationNet DD3D

LLSC-LF

LiDAR-Only Pre-Training

LRS4Fusion

NOMAE Hermes

Gaussian2Scene

AD-L-JEPA

FCOS3D

ContrastiveSceneContexts

GCC-3D

DepthContrast

Camera-Only Pre-Training

DriveX UniFuture GASP BALViT

LiMA

2022

World4Drive

WOTE

OccVLA OccTens FastTopoWM

LiDAR-Centric Pre-Training

SSPC-Im

AV-Det SSRLD

SLidR

OpenScene

I2P-MAE

July 2024 - Dec 2024

ProposalContrast SegContrast

MAELi

BEV-MAE

4DContrast

GaussTR

BEVWorld

PICTURE

GPICTURE

TREND

UNIT

DEPT

Occupancy-MAE GD-MAE

ALSO

Multi-Modal Pre-Training forAutonomous Systems

OccLLama

Affinity3D

RenderWorld

S3PT

GS3

CooPre

Jan 2023 - Aug 2023

[Figure 31]

LOcc

GaussianPretrain

OccVAR

U-MLPNet

CLAP

ConDense

4D-Occ

AD-PT

PatchContrast INoD

MV-JAR

TempO

EUCA-3DP

VisionPAD

SuperFlow Cross Modal SSL

Rel

TriCC

Seal

OccNet GAPretrain

GeoMIM

UniScene

AdaCo

STP DMM

LangOcc VEON

IntraCorr3D

ZPSC-MM

CLIP2Scene

CLIP-FO3D

OVO

ST-SLidR

FSF-Net

GaussianOcc

Drive-OccWorld

Camera-Centric Pre-Training

ZPCS-MM

CNS

UniM2AE

CLIP2

Jan 2024 - June 2024

Sep 2023 - Dec 2023

Unified Pre-Training

MIM4D

DriveWorld

RadarContrast

RiCL

UOV

DistillNeRF

More Modalities

UP-VL 3DOV-VLD

BEVContrast Copilot4D T-MAE

PRED

OccWorld

POP-3D

LAW

GenAD

OccSora

SAM LiDAR

ImageTo360

ScaLR

GPC

ViDAR

SelfOcc

CLIP-ZSPCS

OccFeat

VLM2Scene

NS-MAE

CM3D

LetsMap NeRF-MAE

Radar-Rep SS-RODNet

Radical

PonderV2

UniPAD

MVS3D

E-SSL

GroupContrast

HVDistill

OLIVINE

CSC

Core ECDDP EventBind MEM ECDP

RCAD-SSL

UnO

LSV-MAE

MVRAE

R-MAE

RSLM

- Figure 2 Chronological evolution of representative pre-training methods (2020–2025). The timeline illustrates the paradigm shift in representation learning for autonomous systems. Early approaches predominantly focused on single-modality self-supervision (e.g., LiDAR-only contrastive learning). In contrast, recent advancements (2023–present) demonstrate a surge in cross-modal synergy, characterized by Camera/LiDAR-centric methods and Unified pre-training frameworks, ultimately paving the way for generative world models and comprehensive spatial intelligence.

#### features from cameras onto sparse LiDAR point clouds allows models to achieve superior spatial-semantic reasoning [6, 249, 275]. Key challenges addressed by these approaches include spatio-temporal alignment, handling modality-specific noise, and maintaining robustness when one modality is degraded or missing.

- 2.2.3 Knowledge Distillation and Transfer Learning

While transfer learning traditionally involves adapting pre-trained weights to new domains [144, 277], in the context of multi-modal autonomous systems, it increasingly takes the form of Knowledge Distillation. Here, powerful 2D vision foundation models (teachers) are used to guide the training of 3D sensor backbones (students). This allows 3D models to inherit the open-vocabulary capabilities and rich semantics of large-scale vision models [95, 170, 181] without requiring massive annotated 3D datasets [157, 188]. This paradigm effectively bridges the data scale gap between the 2D image domain and the 3D robotics domain.

- 2.2.4 Foundation Models

Foundation models represent a paradigm shift from specialized pipelines to unified, scalable representation learning [13, 195]. In the vision domain, the trajectory from CNNs [60] to Transformers [38] and generalpurpose encoders like DINO [20, 170] and SAM [95, 183] has established robust, transferable perceptual priors. Recent research integrates these visual priors into non-visual modalities (e.g., LiDAR and radar) via cross-modal alignment, enriching 3D perception with open-world semantics [104, 255, 272]. Crucially, the field is now advancing beyond perception towards Generative World Models [106, 305] and Vision-Language-Action (VLA) models [39, 142, 216]. These next-generation foundation models integrate vision, language, and action into a unified reasoning framework [71, 117], enabling systems not just to recognize objects, but to simulate future scenarios and plan actions in complex, dynamic environments.

Autonomous Vehicles

- (Section 3.2.1)

KITTI[47]; SemanticKITTI [9]; nuScenes[17]; Lyft L5[68]; Waymo[198]; Argoverse[22, 234]; ONCE[160]; nuPlan[18, 88]; MAN TruckScenes[44]; OminiHD-Scenes[304];AevaScenes [2]; PhysicalAI-AV [169]

Drones (UAVs)

- (Section 3.2.2)

Campus[187]; CarFusion[184]; VisDrone[317]; DOTA[36, 238]; AU-AIR[16]; SensatUrban[69]; MAVREC[41]; BioDrone[302]; PDT[309]; UAVScenes [219]

Platform-Specific Datasets

(Section 3)

RailSem19[282]; SRLC[208]; Rail-DB[113]; Rail3D[92]; OSDaR23[92]; WHU-Railway3D[180]; Flow[31]; DartMouth[79]; SeaSAW[89]; WaterScenes[271]; LW3DR[162]; WaterVG[52]; LAQR[57];M3ED[21]; Pi3DET[123]

Other Robotic Platforms

(Section 3.2.3)

- 1.LiDAR-Only: (1) PointContrast[243]; DepthContrast [299]; GCC-3D [127]; ContrastiveSceneContexts [67]; SegContrast[168]; GroupContrast [211]; PatchContrast [193]; AD-PT [280]; E-SSL [64]; ProposalContrast [274]; BEVContrast [189]; 4DContrast [29];

- (2) MV-JAR [248]; Occupancy-MAE [163]; Core [210]; MAELi [107]; R-MAE [202]; T-MAE [232]; GD MAE [263]; LSV-MAE [30]; BEV-MAE [134]; AD-L-JEPA [315]; TurboTrain [312]; NOMAE [1]
- (3) ALSO [14]; 4D Occ [94]; Copilot4D [292]; UnO [3]; PICTURE [251]; UNIT [190]; GPICTURE [252]; CooPre [301]; TREND [24];

- 2.Camera Only: INoD[65]; TempO[108]; LetsMap[50]; NeRF-MAE[77]; VisionPAD[288]

Single Modality Pre-Training

Multi-ModalDataPre-TrainingforAutonomousSystems

(Section 4.1)

SimIPU [120]; SLidR [188]; SSPC-Im[78]; ST-SLidR [156]; Rel [157]; I2P-MAE [293]; TriCC [174]; Seal [144]; PRED [262]; LiMA [255]; ScaLR [179]; CSC [23]; GPC [173]; Cross Modal SSL [19]; SuperFlow [253]; HVDistill [294]; CM3D [93]; ImageTo360 [185]; OLIVINE [297]; EUCA-3DP[199]; GASP [150]; BALViT [66]

LiDAR-Centric Pre-Training

(Section 4.2.1)

Pre-Training Techniques for Sensor Perception

DD3D [175]; DEPT [122]; OccNet [206]; GeoMIM [139]; GAPretrain [73]; ViDAR [270]; MIM4D [322]; SelfOcc [75]; UniScene [164]; MVS3D [229]; DriveWorld [165]; GaussianPretrain [250]; OccFeat [196]; S3PT [235]; UniFuture [126]; GaussianOcc [45]; GaussTR [83]; DistillNeRF [214];

(Section 4)

Camera-Centric Pre-Training

(Section 4.2.2)

PonderV2 [316]; UniPAD [264]; UniM2AE [323]; ConDense [295]; NS-MAE [257]; BEVWorld [298]; CLAP [25]; GS3 [137]; Hermes [310]; LRS4Fusion [172]; Gaussian2Scene [140]

Unified Pre-Training

(Section 4.2.3)

- 1.Radar: RadarContrast [225]; AssociationNet [37]; MVRAE [314]; SSRLD [194]; Radar-Rep[272]; U-MLPNet [258]; 4D-ROLLS [141]; SS-RODNet [319, 320]; Radical [58]; RiCL [33]; RSLM [178];
- 2.Event Camera: ECDP [266]; MEM [96]; DMM [76]; STP [128]; ECDDP [267]; EventBind [308]; EventFly [103]

Incorporating Additional Sensors (Section 4.3)

CLIP2Scene [26]; OpenScene [176]; CLIP-ZSPCS[227]; CLIP-FO3D [291]; POP-3D [209]; VLM2Scene [129]; IntraCorr3D[87]; SAL [171]; Affinity3D [136]; UOV [197]; OVO [201]; LangOcc [10]; VEON [303]; LOcc [279]; UP-VL [167]; ZPCS-MM[152]; CNS[27]; 3DOV-VLD[241]; CLIP2 [283]; AdaCo [324]; TT-Occ [286]; AutoOcc [313]

Text-Grounded Understanding

(Section 5.2)

Open-World Perception and Planning

(Section 5)

OccWorld [305]; GenAD [265]; OccSora [213]; OccLLaMA [231]; OccVAR [85]; RenderWorld [261]; Drive-OccWorld [269]; LAW [116]; FSF-Net [53]; DriveX [191]; SPOT [259]; WoTE [118]; FASTopoWM [268]; OccTens [86]; OccVLA [142]; World4Drive [306]

Unified World Representation for Action (Section 5.3)

- Figure 3 Taxonomy of multi-modal pre-training methodologies. We structure the landscape into three pillars: (1) Platformspecific datasets, (2) Core pre-training techniques classified by sensor interaction (single-modality, cross-modal, and unified), and (3) Advanced open-world perception and planning tasks.

##### Table 1 Summary of representative autonomous vehicle datasets. Region: “AS” (Asia), “EU” (Europe), “NA” (North America). Sensor Configuration: “Camera”, “LiDAR”, and “Radar” denote the count of equipped sensors. Data Statistics: Scenes refers to the number of dataset clips/sequences; Frames indicates the total annotated frames. Conditions: Weather captures adverse scenarios; “d&n” denotes day and night coverage. Symbol “-” indicates that the specific modality or statistic is unavailable/unsupported.

Sensor Data

Annotation

Dataset Year Region

Weather Time Scenes Camera LiDAR Radar 3D Det. 3D Occ. HD-Map E2E Plan

Frames

KITTI [47] 2012 EU 22 4× 1×64-Beam - 15k ✓ ✗ ✗ ✗ ✗ day ApolloScape [74] 2018 AS 103 2× 2×VUX-1HA - 144k ✓ ✗ ✓ ✗ ✓ day

nuScenes [17] 2019 NA/AS 1000 6× 1×32-Beam 5×3d 40k ✓ ✓ ✓ ✓ ✓ d&n SemanticKITTI [9] 2019 EU 22 4× 1×64-Beam - - ✓ ✓ ✗ ✗ ✗ day

Waymo [198] 2019 NA 1150 5× 5×64-Beam - 230k ✓ ✗ ✗ ✗ ✓ d&n Argoverse [22] 2019 NA 113 7× 2×32-Beam - 22k ✓ ✓ ✓ ✓ ✓ d&n

Lyft L5 [68] 2019 NA 366 7× 1×64 & 2×40-Beam 5×3d 46k ✓ ✗ ✗ ✗ ✓ day

A*3D [177] 2019 AS - 2× 1×64-Beam - 39k ✓ ✗ ✗ ✗ ✓ d&n KITTI-360 [130] 2020 EU 11 4× 1×64-Beam - 80k ✓ ✓ ✗ ✗ ✗ day

A2D2 [48] 2020 EU - 6× 5×16-Beam - 12.5k ✓ ✗ ✗ ✗ ✓ day

PandaSet [240] 2020 NA 179 6× 2×64-Beam - 14k ✓ ✗ ✗ ✗ ✗ d&n Cirrus [228] 2020 - 12 1× 2×64-Beam - 6285 ✓ ✗ ✗ ✗ ✗ d&n ONCE [160] 2021 AS - 7× 1×40-Beam - 15k ✓ ✗ ✗ ✗ ✓ d&n

Shifts [158] 2021 AS - - - - - ✓ ✗ ✓ ✗ ✓ d&n nuPlan [18] 2021 NA/AS 3098 8× 3×40 & 2×20-Beam - - ✓ ✗ ✓ ✓ ✓ d&n

Argoverse2 [234] 2022 NA 1000 7× 2×32-Beam - 150k ✓ ✓ ✓ ✓ ✓ d&n MONA [51] 2022 EU 3 3× - - - ✓ ✗ ✓ ✗ ✓ day

Dual Radar [296] 2023 AS 151 1× 1×80-Beam 2×4d 10k ✓ ✗ ✗ ✗ ✓ d&n MAN TruckScenes [44] 2024 EU 747 4× 6×64-Beam 6×4d 30k ✓ ✗ ✗ ✗ ✓ d&n

OmniHD-Scenes [304] 2024 AS 1501 6× 1×128-Beam 6×4d 11.9k ✓ ✓ ✓ ✗ ✓ d&n

AevaScenes [2] 2025 NA 100 6× 6× - - 10k ✓ ✗ ✓ ✗ ✗ d&n PhysicalAI-AV [169] 2025 NA/EU 310,895 7 × 1× - 11× - ✗ ✗ ✗ ✓ ✓ d&n

### 3 Platform-specific Datasets

The efficacy of multi-modal representation learning is intrinsically linked to the scale, diversity, and fidelity of the underlying data. As the field transitions from supervised learning to self-supervised pre-training and foundation models, the role of datasets has evolved from static benchmarks to dynamic engines for forging Spatial Intelligence. In this section, we systematically evaluate prominent datasets across autonomous vehicles, aerial drones, and other robotic platforms. We analyze not only their sensor configurations and annotation richness but also their suitability for emerging tasks such as open-vocabulary perception and generative world modeling.

- 3.1 Overview of Sensor Modalities and Datasets

Multimodal perception systems integrate a heterogeneous suite of onboard sensors, primarily including RGB cameras, LiDAR, radar, event camera, and inertial measurement units (IMUs). Each modality offers distinct perceptual affordances and IMUs enable high-rate ego-motion estimation. Beyond raw sensing, the utility of a dataset for modern pre-training is defined by several critical attributes:

- • SensorConfiguration&Coverage: The spatial arrangement and field-of-view (FoV) determine the system’s ability to construct holistic 360-degree world representations.
- • Spatio-TemporalSynchronization: Precise calibration is non-negotiable for learning unified representations, especially for fusing high-frequency visual streams with sparse geometric points.
- • Annotation Granularity & Modality: The shift from bounding boxes to dense occupancy grids, and recently to natural language descriptions, reflects the community’s move towards reasoning-centric tasks.
- • Domain Diversity: Variations in weather, lighting, and geography are essential for training robust foundation models capable of zero-shot generalization.

##### Table 2 Chronological overview of state-of-the-art UAV-based datasets (2016–Present). Region: “Multi” denotes data collected across multiple regions/platforms; “Sim” indicates synthetic simulation data. Viewpoint: “G” (Ground-view), “A” (Aerial-view), and “AG” (Aerial & Ground joint view). Annotations lists the supported downstream tasks.

Dataset Year Region Viewpoint Sensor Configuration Frames Sensor Resolution Annotations

Campus [187] 2016 NA Single (A) 1× Camera 929,499 1400 × 2019 Target Forecasting/ Tracking UAV123 [166] 2016 AS Multi (A) 1×Camera 110,000 720 × 720 UAV Tracking

CarFusion[184] 2018 NA Multi 22×Camera 53,000 1,280 × 720 3D Vehicle Reconstruction UAVDT [40] 2018 AS Single 1×Camera 80,000 1080 × 540 2D object Detection/ Tracking DOTA[238] 2018 Multi Single (A) Multi-Source 2,806 4000 × 4000 2D Object Detection

VisDrone[317] 2019 AS Single (A) 1×Camera 179,264 3840 × 2160 2D Object Detection/ Tracking DOTA V2.0 [36] 2021 Multi Single (A) Multi-Source 11,268 4000 × 4000 2D Object Detection MOR-UAV [159] 2020 AS Single 1×Camera 10,948 1280 × 720, 1920 × 1080 Moving Object Recognition AU-AIR [16] 2020 EU Multi 1×Camera 32,823 1920 × 1080 2D Object Detection UAVid [154] 2020 EU Single 1×Camera 300 3840 × 2160, 5472 × 3078 2D Semantic Segmentation

5472 × 43078, 7360 × 4192

MOHR [287] 2021 AS Multi (A) 3×Camera 10,631

2D Object Detection

8688 × 5792

SensatUrban [69] 2021 EU Single (A) 1×Camera - - 3D Segmentation UAVDark135 [109] 2023 AS Single 1×Camera 125,466 1920 × 1080 2D Object Tracking MAVREC [41] 2023 EU Multi (AG) 2×Camera 537,030 2700 × 1520 2D Sup/Semi-Sup Object Detection BioDrone [302] 2024 AS Single (A) 1×Camera 304,000 1400 × 1080 2D Object Tracking

PDT [309] 2024 AS Single (A) 1×Camera, 1×LiDAR 5,775 5472 × 3648, 640 × 640 2D Object Detection UAV3D [273] 2024 Sim Multi (A) 5×Camera 20,000 800 × 450 3D Object Detection/ Tracking IndraEye [54] 2024 AS Multi (A) 1×Camera 2,000 1280 × 720, 640 × 480 2D Object Detection/ Semantic Segmentation

UAVScenes [219] 2025 AS Multi (A) 1×Camera, 1×LiDAR 120,000 2448×2048 2D/3D Semantic Segmentation; 6-DoF Visual Localization

The following subsections examine datasets from specific platforms, revealing how platform-specific constraints shape data characteristics and subsequent learning paradigms.

- 3.2 Datasets Acquired from Various Platforms

- 3.2.1 Autonomous Vehicles

Autonomous driving serves as the primary testbed for multi-modal spatial intelligence. Vehicles typically deploy a redundant sensor suite consisting of surround-view cameras, high-beam LiDARs, and radars to ensure safety-critical perception [98, 221, 244]. The continuous collection of synchronized sensor streams has produced massive-scale datasets [7, 17, 47, 169, 198], which act as the fuel for self-supervised pre-training. Current methodologies leverage these unlabeled streams for pretext tasks such as temporal future prediction [165, 270], cross-modal masked reconstruction [1, 134, 264], and contrastive distillation [188, 255], effectively turning raw data into transferable representations without human labeling.

- Table 1 summarizes representative datasets. Notably, the evolution from early perception-centric benchmarks (e.g., KITTI [47]) to modern reasoning-centric datasets (e.g., nuPlan [18] and Argoverse 2 [234]) highlights a crucial trend: the integration of high-definition maps, long-horizon trajectories, and increasingly, languagebased scenario descriptions [142, 218]. These rich annotations are pivotal for training next-generation End-to-End planners and Vision-Language-Action (VLA) models [71, 117, 118].

3.2.2 Drones (UAVs)

Unmanned Aerial Vehicles (UAVs) present unique perception challenges due to their bird’s-eye viewpoints, six degrees-of-freedom (6-DoF) motion, and rapid scale changes [69, 154, 219]. While RGB cameras and IMUs remain standard, advanced datasets now incorporate LiDAR to capture 3D structural information for complex environments [69, 273].

- Table 2 details key UAV datasets. Unlike ground vehicles, UAV data is characterized by significant perspective distortion and motion blur [154, 302]. Consequently, pre-training in this domain heavily utilizes transfer learning from ground-level or satellite imagery [317], adapting visual foundation models to aerial domains. Recent efforts also explore cross-view geo-localization and self-supervised flow estimation to handle the dynamic nature of flight. The emergence of multi-modal UAV datasets [219, 273, 302] is crucial for extending Spatial Intelligence from 2D ground planes to 3D volumetric spaces.

##### Table3 Overviewofmulti-modaldatasetsfordiverseroboticplatforms. This table categorizes datasets into three specialized domains: Railways, Unmanned Surface Vehicles (USVs), and Legged Robots. These benchmarks extend spatial intelligence research to constrained tracks, maritime environments, and complex terrains.

Dataset Year Region Platform Sensors Frames Annotations

RailSem19 [282] 2019 EU Railway Camera 8,500 Image Classification, Semantic Segmentation

FRSign [59] 2020 EU Railway 2×Camera 105,352 Railway Signaling Reading RAWPED [207] 2020 EU, AS Railway 1×Camera 26,000 2D Object Detection

SRLC [208] 2021 EU Railway LiDAR - Point Cloud Generation, Semantic Segmentation Rail-DB [113] 2022 AS Railway Camera 7,432 Rail Detection

RailSet [325] 2022 EU Railway 1×Camera 6,600 Railway Anomaly Detection OSDaR23 [200] 2023 EU Railway 9×Camera, 6×LiDAR, 1×Radar 1,534 Rail and Object Detection, LiDAR Segmentation

Rail3D [92] 2024 EU Railway 4×Camera, 1×LiDAR - LiDAR Semantic Segmentation WHU-Railway3D [180] 2024 AS Railway 1×LiDAR 40 tiles LiDAR Segmentation

FloW [31] 2021 AS Unmanned Surface Vehicle 2×Camera, 1×4D Radar 2,000 2D Object Detection DartMouth [79] 2021 NA Unmanned Surface Vehicle 3×Camera, 1×LiDAR - 2D Object Detection, Semantic Segmentation MODS [15] 2022 EU Unmanned Surface Vehicle 2×Camera, 1×LiDAR 8,175 2D Object Detection

SeaSAW [89] 2022 EU, NA Unmanned Surface Vehicle 5×Camera 1,900,000 2D Object Detection, Tracking, Classification WaterScenes [271] 2023 AS Unmanned Surface Vehicle 1×Camera, 1×4D Radar 54,120 2D Object Detection, Semantic/ Panoptic Segmentation

MVDD13 [215] 2024 AS Unmanned Surface Vehicle Camera x1 - 2D Object Detection SeePerSea [80] 2024 AS, NA Unmanned Surface Vehicle 1×Camera, 1×LiDAR 10,906 2D & 3D Object Detection

WaterVG [52] 2024 AS Unmanned Surface Vehicle 1×Camera, 1×4D Radar 11,568 Multi-Task Visual Grounding Han et al. [57] 2024 AS Legged Robots Depth Camera - Animal Motions

Luo et al. [153] 2025 AS Legged Robots Panoramic Camera 1,920s 2D Object Tracking QuadOcc [192] 2025 AS Legged Robots Panoramic Camera, 1×LiDAR 8,000 3D Occupancy

M3ED [21] 2023 NA Car, UAV, Legged Robots 3× Camera, 2× Event Camera,1×LiDAR - Depth Estimation, Semantic Segmentation Pi3DET [123] 2025 NA Car, UAV, Legged Robots 3× Camera, 2× Event Camera,1×LiDAR 51,545 3D Object Detection

- 3.2.3 Other Robotic Platforms

Beyond cars and drones, diverse robotic platforms such as Unmanned Surface Vehicles (USVs) [52, 271], railway systems [113, 282, 325], and legged robots [21, 55, 57] operate in highly constrained or unstructured environments. These domains challenge pre-training models with unique noise patterns (e.g., water reflections for USVs) and motion dynamics (e.g., non-linear locomotion for quadrupeds).

- Table 3 lists representative datasets. For instance, datasets for legged robots [57, 153] emphasize egocentric perception under severe camera shake, motivating research into robust, motion-aware representation learning. Similarly, rail and USV datasets focus on long-range, track-constrained perception [180, 271, 325]. A growing trend in these specialized domains is the use of Simulation-to-Real transfer and domain adaption [21, 123]. Engines like QuaDreamer [236] generate synthetic training data to supplement scarce real-world samples, training models that can generalize to physical robots via domain randomization. This highlights the increasing role of synthetic data in democratizing foundation models for varied robotic form factors [208].

- 3.3 Key Dataset Trends and Implications

Analyzing the landscape of platform-specific datasets identifies three evolutionary trends that are reshaping multi-modal pre-training:

FromPerceptiontoReasoningandAction. Modern datasets are moving beyond bounding boxes. Benchmarks like nuPlan [18] and OmniDrive [218] introduce planning trajectories, logic-based scenarios, and open-vocabulary language labels. This shift enables the training of models that do not just see but reason and act, laying the groundwork for Embodied AI and VLA models [71, 117, 142, 176].

The Rise of Synthetic and Generative Data. Recognizing the long-tail limitations of real-world data, there is a surge in high-fidelity synthetic datasets and simulation environments [161, 208, 236]. This supports the development of Generative World Models, which can simulate infinite what-if scenarios for robust policy learning, effectively closing the loop between perception and simulation [106, 124].

Scale and Diversity for Foundation Models. The explosion in data volume and modality diversity (from LiDAR to Event cameras) has rendered manual annotation obsolete [21, 107, 266]. This reality firmly establishes Self-Supervised Pre-Training as the necessary paradigm. Future progress will depend on data engines that can automatically curate, label, and align these massive multi-modal streams to feed hungry foundation models [104, 264].

These trends collectively signal a transition: datasets are no longer just static benchmarks for performance evaluation, but active components in the loop of training generative, reasoning-capable Spatial Intelligence

##### Table 4 Comprehensive summary of LiDAR-based pre-training techniques. The table categorizes methods into LiDAR-only (single-modality) and LiDAR-centric (cross-modal) paradigms. Input Modality: “L” denotes LiDAR input; “SC” and “MC” refer to Single-Camera and Multi-Camera data used for cross-modal distillation or alignment.

Method Venue Input Modality Proxy Task Downstream Task Dataset Key Contribution

PointContrast [243] ECCV’20 L Spatial Contra. (Point) Sem-Seg. ScanNet, SemanticKITTI Point-wise contrastive learning on augmentations DepthContrast [299] ICCV’21 L Spatial Contra. Sem-Seg./Det. Waymo, nuScenes Frame-wise depth consistency learning

GCC-3D [127] ICCV’21 L Spatial Contra. Det. Waymo Geometry-aware contrast with clustering SimIPU [120] AAAI’22 L + SC Spatial Contra. Sem-Seg. SemanticKITTI Simple 2D-3D spatial alignment ProposalContrast [274] ECCV’22 L Spatial Contra. (Region) Det. Waymo, nuScenes Contrastive learning on detection proposals GD-MAE [263] CVPR’23 L MAE Sem-Seg./Det. Waymo MAE with generative decoder ALSO [14] CVPR’23 L Occupancy Estimation Occ. nuScenes Occupancy-based self-supervision

BEV-MAE [134] AAAI’24 L BEV MAE Det. Waymo Masked BEV feature learning MAELi-MAE [107] WACV’24 L MAE Det. Waymo MAE for large-scale LiDAR representation learning BEVContrast [189] 3DV’24 L BEV Contra. Sem-Seg./Det. nuScenes Contrastive learning in BEV space

PPKT [146] arXiv’21 L + MC Spatial Contra. Sem-Seg. nuScenes Pixel-to-point contrastive transfer learning SLidR [188] CVPR’22 L + MC Spatial Contra. Sem-Seg. nuScenes Superpixels to guide the image-to-LiDAR pre-training

ST-SLidR [156] CVPR’23 L + MC Spatial Contra. Sem-Seg. nuScenes Class-balanced cross-modal contrastive learning TriCC [174] CVPR’23 L + MC Spatial & Temp. Contra. Sem-Seg. nuScenes Triangle-constrained spatiotemporal contrastive

Seal [144] NeurIPS’23 L + MC Spatial Contra. Sem-Seg. nuScenes Transfer knowledge from foundation models to 3D

CSC [23] CVPR’24 L + MC Spatial Distill. Sem-Seg. nuScenes Unified baseline for large-scale pretraining OLIVINE [297] NeurIPS’24 L + MC Spatial Distill. Sem-Seg. nuScenes Fine-grained contrast with vision features HVDistill [294] IJCV’24 L + MC Spatial Distill. Sem-Seg. nuScenes Hybrid-view distillation from images to 3D

ScaLR [179] CVPR’24 L + MC Spatial Distill. Sem-Seg./Det. nuScenes, KITTI, PandaSet Directly distill knowledge from image to LiDAR SuperFlow [253] ECCV’24 L + MC Spatial & Temp. Contra. Sem-Seg. nuScenes Spatiotemporal contrastive for knowledge transfer LargeAD [104] arXiv’25 L + MC Spatial Contra. Sem-Seg./Det. nuScenes, KITTI, Waymo Large-scale multi-dataset pre-training LiMoE [254] CVPR’25 L + MC Spatial & Temp. Distill. Sem-Seg./Det. nuScenes MoE-based multi-representation pre-training LiMA [255] ICCV’25 L + MC Spatial & Temp. Distill. Sem-Seg./Det. nuScenes Cross-view and long-horizon distillation for pre-training

agents [142, 218, 304]. By providing rich multi-modal contexts, these data engines facilitate the transition from passive perception to active world modeling and decision-making [71, 106].

### 4 Pre-Training Techniques for Perception

In this section, we critically examine the methodologies that empower autonomous systems to learn robust representations from raw sensor data. As depicted in the taxonomy (Fig. 3), we structure the landscape based on sensor interaction paradigms: Single-Modality baselines, Multi-Modality synergy (including Camera-Centric and LiDAR-Centric distillation), and Unified frameworks that jointly optimize cross-modal encoders.

Beyond the modality-based categorization, we emphasize a crucial trend: the integration of Foundation Models and Generative Objectives. Recent approaches are shifting from simple discriminative tasks to generative reconstruction [91, 264](e.g., NeRF, 3DGS) and future forecasting [126, 165, 270], leveraging the rich semantic priors of large-scale vision models to enhance geometric reasoning [188, 279]. We also briefly discuss complementary sensors such as radar and event cameras [178, 266, 267, 272]. Finally, we synthesize benchmark performance to offer a holistic evaluation of how these pre-training techniques translate to downstream perception tasks.

- 4.1 Single-Modality Pre-Training

Single-modality pre-training serves as the bedrock of perception, aiming to extract intrinsic semantic and geometric features from individual sensor streams without the aid of cross-modal supervision. Given their ubiquity in autonomous systems, we primarily focus on Camera and LiDAR modalities in this subsection. Mastering these single-modality representations is a prerequisite for effective sensor fusion and interaction, as it ensures that each branch of a multi-modal system contributes robust, high-quality features to the unified world model.

- 4.1.1 LiDAR-Only Pre-Training

#### LiDAR sensors provide precise and metric-accurate 3D measurements, making them indispensable for tasks requiring fine-grained geometric perception, such as object detection and occupancy prediction. Unlike cameras, LiDAR data is inherently sparse, unordered, and lacks texture, necessitating specialized pre-training objectives to capture underlying topological structures and temporal dynamics. As illustrated in Fig. 4, current research focuses on three primary paradigms to forge robust 3D representations from unlabeled point

clouds: Masked Reconstruction for structural understanding, Contrastive Learning for spatial invariance, and Temporal Forecasting for dynamic world modeling.

Masked Reconstruction and Structural Completion.

- (a) MAE-based Pre-Training
- (b) Contrastive Pre-Training
- (c) Forecasting-based Pre-Training

Drawing inspiration from Masked Autoencoders (MAE) in general vision [63] and NLP [35], this paradigm forces the network to infer unseen geometric structures from partial observations, thereby learning holistic spatial priors. To handle the irregularity of point clouds, approaches such as GD-MAE [263] and BEV-MAE [134] leverage regular 2D/3D grids for structured masking, while MAELi [107] explicitly reconstructs intensity values to incorporate surface reflectivity properties. MVJAR [248] and Occupancy-MAE [163] also operate on voxelized features to enforce spatial consistency. Recent advances extend this concept to the temporal dimension. T-MAE [232] and LSV-MAE [30] reconstruct sequence-level motion patterns. Furthermore, AD-L-JEPA [315] moves beyond voxel reconstruction to latent space prediction, focusing on learning abstract relational reasoning rather than low-level details.

|LiDAR<br><br>[Figure 32]|
|---|

|LiDAR<br><br>[Figure 33]|
|---|

LiDAR Decoder

LiDAR Encoder

or

Current Input Masking Reconstructed Input

BEV/3D Features

|LiDAR<br><br>[Figure 34]|
|---|

| |
|---|

LiDAR Encoder

Contrastive Learning

- View 1
- View 2

|LiDAR<br><br>[Figure 35]|
|---|

| |
|---|

LiDAR Encoder

| |
|---|

BEV/3D Features/Proposals

|LiDAR<br><br>[Figure 36]|
|---|

|LiDAR<br><br>[Figure 37]|
|---|

LiDAR Decoder

LiDAR Encoder

Contrastive Learning and Spatial Invariance. Contrastive learning aims to learn discriminative feature spaces where semantically similar points or scenes are pulled together. This paradigm has evolved from point-level discrimination to multi-scale hierarchical understanding. PointContrast [243] pioneered this direction by optimizing point-level invariance across augmented views, while DepthContrast [299] utilized singleview depth maps to construct informative pairs. Subsequent research has scaled this objective to various spatial hierarchies: Patch/Proposal-level methods [168, 193, 274] focus on object-centric features; BEV-level approaches [64, 189, 280] align features in the bird’s-eye view for downstream perception tasks; and Scene-level methods [29, 67] capture global context. This hierarchical evolution demonstrates the versatility of contrastive objectives in encoding geometry at different granularities.

Current Input Future Frames

BEV/3D Features

Figure 4 Schematic illustration of representative LiDAR-only pre-training paradigms. To learn robust geometric representations from sparse point clouds without annotations, methods typically adopt three strategies: (a) Masked Autoencoding (MAE), which reconstructs missing structures to learn local geometry; (b) Contrastive Learning, which enforces view-invariant feature discrimination; and (c) Temporal Forecasting, which predicts future frames to capture dynamic scene evolution.

Temporal Forecasting and Predictive Modeling. Moving beyond static perception, forecasting-based pre-training leverages the sequential nature of LiDAR streams to anticipate future states, serving as a precursor to predictive world models. Early works like ALSO [14] and 4D-Occ [94] formulate pre-training as occupancy or flow prediction, enabling the model to fill in future geometric voids. Recent frameworks such as Copilot4D [292] and UnO [3] explicitly predict point cloud sequences, fostering temporally consistent representations. Advanced methods further incorporate complex interactions: PICTURE [251] and UNIT [190] introduce mutual information maximization and spatio-temporal clustering, while CooPre [301] and TREND [24] extend forecasting to multiagent cooperative scenarios. These approaches equip models with the predictive capacity essential for planning in dynamic environments.

- 4.1.2 Camera-Only Pre-Training

#### Visual data from camera offers the rich semantic information for scene understanding. While supervised pre-training on generic datasets like ImageNet [34] and MS-COCO [132] remains a standard initialization strategy for common vision backbones (e.g., ResNet [60] and ViT [38]), it suffers from a domain gap when

applied to the complex, 3D-centric tasks of autonomous systems. Consequently, the field has pivoted towards Self-Supervised Learning (SSL) on domain-specific onboard data, evolving through three key paradigms:

Domain and Temporal Consistency. Handling domain shifts and exploiting temporal continuity are fundamental for robust vision. INoD [65] addresses the domain generalization challenge by formulating a dataset affiliation prediction pretext task, interleaving feature maps from disjoint domains to learn invariant representations. Capitalizing on the sequential nature of driving videos, TempO [108] treats region-level feature ordering as a sequence prediction problem. By modeling the temporal evolution of features, it enables the visual encoder to capture motion dynamics and causality, which are critical for planning.

Geometric Lifting to BEV. Bridging the gap between 2D images and 3D perception is a core objective. LetsMap [50] pioneers a label-efficient approach for semantic Bird’s-Eye-View (BEV) mapping. It leverages the spatial constraints inherent in monocular sequences to enforce consistency between perspective and BEV representations, effectively lifting 2D semantics into a metric space without relying on expensive dense annotations or LiDAR depth.

Neural Fields and Volumetric Reasoning. The most recent frontier involves incorporating implicit 3D representations into visual pre-training. NeRF-MAE [77] represents a paradigm shift, adapting Masked Autoencoders (MAE) to Neural Radiance Fields (NeRF). By using posed RGB images to reconstruct masked volumetric tokens, it forces the transformer to internalize 3D spatial layouts and view-dependent effects. Similarly, VisionPAD [288] introduces a voxel-centric framework that combines voxel warping with multi-frame photometric consistency. This allows the model to learn fine-grained motion and geometry directly from image streams, offering a scalable alternative to depth-supervised methods.

Collectively, these methods illustrate a trajectory from learning 2D semantics to mastering 3D geometry and temporal dynamics, enabling cameras to function as standalone sensors for spatial intelligence.

- 4.2 Multi-Modality Pre-Training

While single-modality pre-training establishes the foundational feature space, forging true Spatial Intelligence requires the synergy of heterogeneous sensors. The physical world manifests in diverse signals: cameras capture dense semantic texture, while LiDAR and radar provide sparse but metric-accurate geometry and kinematics. Multi-modality pre-training aims to bridge the semantic-geometric gap by learning unified representations that leverage the complementary strengths of these modalities.

We categorize these approaches based on the information flow direction: LiDAR-Centric (distilling visual semantics into 3D geometry), Camera-Centric (injecting geometric priors into 2D features), and Unified Frameworks (jointly optimizing modality-agnostic representations). This taxonomy highlights how crossmodal interactions evolve from simple alignment to unified world modeling.

- 4.2.1 LiDAR-Centric Pre-Training

LiDAR sensors excel at capturing precise 3D structures but suffer from inherent semantic sparsity and lack of texture. Conversely, the computer vision community has cultivated powerful foundation models [20, 95, 181] that encapsulate rich, open-world semantic knowledge. LiDAR-centric pre-training aims to bridge this asymmetry by treating visual signals as Privileged Information during training. The goal is to transfer the semantic richness of 2D images into 3D point cloud networks, enabling them to hallucinate semantic features even when cameras are absent during inference. As illustrated in Fig. 5, this paradigm has evolved through four key strategies:

Masked Reconstruction with Visual Guidance. Integrating cross-modal cues from camera images into the Masked Autoencoder (MAE) framework [63] enhances structural learning for LiDAR point cloud. I2P-MAE [293] and CM3D [93] condition the reconstruction of masked LiDAR tokens on visible image patches, forcing the network to infer 3D geometry from 2D semantic context. ImageTo360 [185] and EUCA-3DP [199] extend this to full-scene scales, leveraging BEV context to promote holistic spatial reasoning that fuses visual texture with geometric occupancy.

Cross-Modal Contrastive Alignment. The foundational approach involves aligning 2D and 3D feature spaces through contrastive learning. By maximizing the similarity between corresponding image pixels and projected LiDAR points, models learn to associate geometric clusters with visual concepts. SimIPU [120] and SLidR [188] pioneered this by constructing point-pixel pairs to enforce local semantic consistency. Recent extensions like ST-SLidR [156] and Cross-ModalSSL [19] incorporate temporal constraints and region-aware affinity, improving the robustness of alignment against calibration errors and dynamic objects.

- (a) Cross-modal MAE-based Pre-Training
- (b) Cross-modal Contrastive/Distillation Pre-Training
- (c) Temporal Pre-Training

|LiDAR<br><br>[Figure 38]|
|---|

|LiDAR<br><br>[Figure 39]|
|---|

LiDAR Decoder

LiDAR Encoder

2D-Guided Masking

Current Input BEV/3D Features Reconstructed Input

|LiDAR<br><br>[Figure 40]|
|---|

LiDAR Encoder

Contrastive Learning

Knowledge Distillation

|Image<br><br>[Figure 41]| |
|---|---|
| | |

Camera Encoder

Knowledge Distillation from Foundation Models.

Pre-Trained VFM / Super-Pixel

Moving beyond simple alignment, recent works leverage 2D foundation models as teachers to distill open-vocabulary semantics into 3D students. Seal [144] and ScaLR [179] utilize the segmentation capability of SAM and vision transformers to generate high-quality pseudo-labels or soft feature targets for point clouds. CSC [23] and OLIVINE [297] further refine this process by incorporating hierarchical clustering and class-aware gating, ensuring that the distilled knowledge respects the geometric boundaries of 3D objects. This strategy effectively imparts sight to blind LiDAR networks.

|LiDAR<br><br>[Figure 42]|
|---|

LiDAR Encoder

LiDAR Encoder

LiDAR Encoder

LiDAR Encoder

......

|Image<br><br>[Figure 43]|
|---|

Camera Encoder

Camera Encoder

Camera Encoder

Camera Encoder

T1 T2 Tn-1 Tn

Figure5 TaxonomyofLiDAR-centricpre-trainingmethodologies.

To bridge the semantic gap of point clouds, these approaches leverage images as privileged information during training. The main paradigms involve: (a) Cross-modal MAE-based Pre-Training, which incorporates 2D-guided masking strategies to enhance geometric reconstruction and structural understanding; (b) Cross-modal Contrastive/Distillation PreTraining, which either enforces feature alignment between modalities or directly transfers rich open-vocabulary semantics from pre-trained Vision Foundation Models (VFMs) to 3D encoders; and (c) Temporal Pre-Training, which exploits video-LiDAR sequences to capture motion dynamics and enforce spatiotemporal consistency.

Temporal Dynamics and Motion Transfer. Static cross-modal alignment is insufficient for dynamic autonomous systems in real world. SuperFlow [253] and PRED [262] introduce temporal supervision by transferring motion knowledge from video to point cloud sequences. By aligning the temporal evolution of features across modalities, these methods enable LiDAR backbones to capture long-horizon dynamics [255], serving as a stepping stone towards predictive world models.

In summary, LiDAR-centric pre-training transforms point cloud networks from pure geometric processors into semantically aware perception engines, significantly enhancing performance in detection and semantic segmentation tasks, particularly in data-scarce regimes. Table 4 provides a comprehensive taxonomy of these LiDAR-based techniques, categorizing them by input modality, proxy tasks, and downstream applications.

- 4.2.2 Camera-Centric Pre-Training

Camera-centric pre-training addresses the ill-posed nature of monocular perception: recovering 3D structures from 2D projections. While cameras are cost-effective and ubiquitous, they lack intrinsic and accurate depth. To overcome this, recent methods utilize LiDAR data as a Geometric Supervisor during pre-training. By injecting precise depth and structural priors into visual backbones, these models learn to hallucinate 3D geometry from images alone, retaining efficient camera-only inference while benefiting from LiDAR-grade supervision. As visually taxonomized in Fig. 6, this domain bifurcates into two primary streams: Geometric Perception (via explicit depth or feature distillation) and Predictive World Modeling (via forecasting or neural rendering). A detailed overview of these vision-centric methodologies, including their proxy tasks and key contributions, is summarized in Table 5.

##### Table5 Overviewofcamera-centricandunifiedpre-trainingmethodologies. This table summarizes representative approaches that leverage visual data as the primary input. Input Modality: “MC” denotes Multi-Camera setups; “L” indicates the use of LiDAR; “T” signifies the integration of temporal information for dynamic modeling.

Method Venue Input Modality Proxy Task Downstream Task Dataset Key Contribution

GeoMIM [139] ICCV’23 MC Reconstruction Det./Map/Occ. nuScene Leveraging the knowledge of a pretrained LiDAR model OccNet [206] ICCV’23 MC Forecasting Det./Map nuScene Utilizing the semantic occupancy as the latent feature supervision

UniScene [164] RA-L’24 MC Forecasting Det./Occ. nuScene Utilizing the geometric occupancy as the latent feature supervision DriveWorld [165] CVPR’24 MC Forecasting Det./Map/Occ./E2E nuScene Utilizing 4D occupancy as the latent feature supervision

ViDAR [270] CVPR’24 MC-T Forecasting Det./Map/Occ./E2E nuScene Visual point cloud forecasting

MIM4D [322] IJCV’25 MC-T Rendering Det./Map/Vec. Map nuScene Investigating spatial and temporal relations with video GaussianPretrain [250] arXiv’24 MC-T Rendering Det./Occ./Vec. Map nuScene Leveraging the Gaussian representation

VisionPAD [288] CVPR’25 MC-T Rendering Det./Occ. nuScene Vision-only pre-training with temporal constraint UniPAD [264] CVPR’24 MC & L Rendering Det./Seg. nuScene Multi-modality pre-training with MAE UniM2AE [323] ECCV’24 MC & L Rendering Det./Map nuScene Multi-modality pre-training with MAE and extra alignment

NS-MAE [257] CASE’25 MC & L Rendering Det./Map nuScene Multi-modality pre-training with differential neural volume rendering BEVWorld [298] arXiv’24 MC-T & L Rendering Det./Motion nuScene Multi-modality with temporal information

LRS4Fusion [172] ICCV’25 MC-T & L Forecasting Det./Depth LR & nuScenes Self-supervised sparse sensor fusion for long range perception

Geometric Perception: From Depth to Distillation. The primary goal here is to equip vision models with spatial awareness by aligning 2D features with 3D structural constraints. Explicit Depth Pre-Training (Fig. 6 (a)) serves as the foundational approach. Early works like DD3D [175] and DEPT [122] leverage pseudo-depth supervision from LiDAR to initialize 3D object detectors, effectively grounding visual features in metric space. Moving beyond simple depth maps, Distillation-based Pre-Training (Fig. 6 (b)) aligns latent representations. OccNet [206] and SelfOcc [75] advance this by learning to predict dense

- (a) Traditional Depth Pre-Training
- (b) Distillation-based Pre-Training
- (c) Forecasting-based Pre-Training
- (d) Rendering-based Pre-Training

|Supervise|LiDAR<br><br>[Figure 44]|
|---|---|
| | |

|Depth Map<br><br>[Figure 45]|
|---|

|Image<br><br>[Figure 46]|
|---|

Camera Encoder

Depth Estimation

|LiDAR<br><br>[Figure 47]|
|---|

|Image<br><br>[Figure 48]| |
|---|---|
| | |

Supervise

Pretrained LiDAR Model

Camera Encoder

View Transform

BEV Feature

|LiDAR<br><br>[Figure 49]|
|---|

|Image<br><br>[Figure 50]| |
|---|---|
| | |

Future Decoder

Supervise

Camera Encoder

View Transform

Future 3D Frame

|Image<br><br>[Figure 51]|
|---|

[Figure 52]

- 3D occupancy grids, utilizing LiDAR occupancy as a ground-truth supervisor. Furthermore, Masked Image Modeling (MIM) has been adapted for geometric consistency: GeoMIM [139] and MIM4D [322] reconstruct masked image patches by cross-referencing with projected LiDAR points, forcing the network to internalize 3D spatial correspondences within the feature extraction process.

|Image<br><br>[Figure 53]| |
|---|---|
| | |

[Figure 54]

NeRF

Supervise

Camera Encoder

View Transform

|[Figure 55]<br><br>3D Gauss.|
|---|

|LiDAR<br><br>[Figure 56]|
|---|

2D and 3D Info.

Figure 6 Overview of camera-centric pre-training paradigms (LiDARto-Vision). These methods aim to inject 3D geometric priors into 2D visual backbones using LiDAR as a supervisor. Key approaches include: (a) Depth Estimation for explicit geometry learning; (b) Feature Distillation to align 2D-3D latent spaces; and (c) Forecasting and (d) GenerativeRendering, which empower vision models to hallucinate 3D structures and predict future dynamics from monocular inputs.

Predictive World Modeling: Forecasting and Rendering. This stream represents the transition from static perception to dynamic simulation, requiring models to understand temporal evolution and photorealistic synthesis. Forecasting-based Pre-Training (Fig. 6 (c)) compels models to predict future states from current video streams, thereby internalizing the physics of the environment. ViDAR [270] pioneers "Visual Point Cloud Forecasting," treating future LiDAR points as a supervision signal for historical visual inputs. Extensions like DriveWorld [165] and UniScene [164] scale this to 4D occupancy, learning spatio-temporal abstractions that facilitate long-term planning. Complementing this, Rendering-based Pre-Training (Fig. 6 (d)) exploits the differentiability of neural fields. Frontier methods like GaussianPretrain [250] and GaussianOcc [45] incorporate 3D Gaussian Splatting (3DGS) [91] into the pre-training loop. By enforcing photometric consistency through differentiable rendering, these models learn continuous, high-fidelity geometric representations that surpass discrete voxels in precision. Finally, generative approaches such as GenAD [265] and OccSora [213] integrate these concepts to function as neural simulators, paving the way for end-to-end agents capable of reasoning about future consequences [231, 269].

- 4.2.3 Unified Pre-Training

#### Unified pre-training represents the convergence of multi-modal learning. Unlike asymmetric distillation (LiDAR-centric or Camera-centric), which treats one modality as primary, unified frameworks jointly optimize encoders for heterogeneous modalities within a shared latent space. As explicitly illustrated in Fig. 7, a canonical unified framework processes data through a cohesive pipeline encompassing masking, alignment, and reconstruction. This paradigm can be deconstructed into three critical phases:

Multi-Modal Masking and Encoding. The pipeline begins by treating raw sensor inputs as discrete tokens. As depicted in the MultiModal Masking stage of Fig. 7, methods like UniPAD [264] and UniM2AE [323] apply randomized masking to both LiDAR points and image patches. This forces the encoders to learn robust local features rather than relying on redundant shortcuts. Specifically, the visual branch typically employs a Camera Encoder, while the geometric branch utilizes a LiDAR Encoder to extract high-dimensional primitives from sparse inputs.

|LiDAR<br><br>[Figure 57]| |
|---|---|
| | |

[Figure 58]

Multi-ModalFusionandInteraction

LiDAR Encoder Depth Map

3D-to-BEV Flatten

Multi-ModalMasking

BEV Features

|LiDAR<br><br>[Figure 59]|
|---|

|Image<br><br>[Figure 60]| |
|---|---|
| | |

|Image<br><br>[Figure 61]|
|---|

Camera Encoder

View Transform

Volumetric Features

Unified Representation

Figure 7 Illustration of unified multi-modal pre-training frameworks. Unlike asymmetric distillation, unified approaches jointly optimize Camera and LiDAR encoders within a shared representation space. This paradigm facilitates the learning of modality-agnostic features that integrate both semantic richness and geometric precision, forging a holistic basis for Spatial Intelligence.

View Transformation and Unified Fusion. To bridge the dimensional gap between 2D images and 3D points, the framework transforms heterogeneous features into a common coordinate system. As shown in the center of Fig. 7, visual features undergo a View Transform [110, 121, 264], while point features are processed via 3D-to-BEV Flattening. These streams converge at the Multi-Modal Fusion and Interaction stage, resulting in a Unified Representation – manifesting typically as BEV Features or dense Volumetric Features. Approaches like BEVWorld [298] and GS3 [137] leverage this shared latent space to enforce strict geometric consistency between modalities.

Generative Reconstruction. The final objective is to validate the understanding of the scene by reconstructing the masked or missing information. The right side of Fig. 7 demonstrates that the unified representation is decoded to simultaneously reconstruct the original LiDAR geometry, Image texture, and often auxiliary Depth Maps. By optimizing for this holistic reconstruction objective, the model learns modality-agnostic features that integrate semantic richness with geometric precision, ensuring robustness even when individual sensors are compromised during inference [25, 257, 295].

In conclusion, unified pre-training moves beyond simple sensor fusion; it forges a holistic understanding of the physical world that is independent of the specific sensing apparatus, a key characteristic of true Spatial Intelligence.

- 4.3 Incorporating Additional Sensors

In complex open-world environments, reliance solely on cameras and LiDAR can lead to perceptual failures under adverse conditions, such as severe weather, high-speed motion, or extreme lighting changes [98, 100, 103, 244]. To forge robust Spatial Intelligence, incorporating complementary sensors becomes imperative. Millimeter-wave radar offers resilience against fog and rain via Doppler signatures, while Event Cameras (neuromorphic sensors) capture microsecond-level dynamics with high dynamic range. Integrating these modalities into pre-training frameworks not only enhances system reliability but also extends the operational design domain of autonomous agents. In this subsection, we analyze representation learning paradigms specialized for these sensors.

- 4.3.1 Radar Pre-Training

Radar point clouds differ significantly from LiDAR in their scarcity, noise characteristics (clutter), and unique velocity channels. Pre-training methods in this domain focus on suppressing noise and extracting meaningful structural features through three key approaches:

Cross-Modal Alignment and Supervision. Due to the semantic sparsity of radar returns, aligning them with richer modalities is a standard strategy. AssociationNet [37] utilizes well-structured LiDAR point clouds to supervise radar feature learning, enhancing geometric consistency. RadarContrast [225] and RiCL [33] employ contrastive learning to enforce invariance between radar representations and their multi-view or temporal counterparts, effectively grounding radar features in a stable metric space.

Masked Modeling for Sparse Signals. Adapting masked reconstruction to radar involves dealing with extreme sparsity. MVRAE [314] and RSLM [178] introduce autoencoding frameworks that reconstruct raw radar signals, enabling the model to learn spatiotemporal priors and filter out multi-path noise. Radar-Rep [272] and Radical [58] further refine this by designing radar-specific masking strategies and curriculum learning to handle the high variance in signal quality.

Domain Adaptation and Simulation. To bridge the gap between synthetic and real-world radar data, domainadaptive strategies are crucial. SS-RODNet [319, 320] facilitates transfer learning across domains, while U-MLPNet [258] explores lightweight inductive biases to enable efficient radar perception on edge devices.

- 4.3.2 Event Camera Pre-Training

Event cameras capture asynchronous brightness changes, offering a paradigm shift for high-speed perception. Pre-training methodologies here must address the non-grid, asynchronous nature of event streams:

SpatiotemporalReconstruction. Reconstructing dense signals from sparse events forces the model to understand scene dynamics. MEM [96] and DMM [76] adapt masked modeling to event streams, reconstructing spatial structures from fragmented temporal triggers. ECDP [266] and ECDDP [267] focus on future frame prediction, leveraging the high temporal resolution of events to forecast motion with exceptional precision. STP [128] introduces specialized transformer architectures to simultaneously model the spatial sparsity and temporal continuity inherent in event data.

Cross-Modal Synergy. Integrating events with standard RGB frames combines high dynamic range with semantic texture. EventBind [308] aligns asynchronous event streams with synchronous RGB frames in a shared latent space, enabling semantic understanding even in high-motion blur scenarios. EventFly [103] further demonstrates the utility of this synergy for agile navigation in aerial robotics, where latency is a critical bottleneck.

- 4.3.3 Auxiliary Modalities

Beyond primary perception sensors, other onboard instruments serve as critical sources of weak supervision or geometric constraints during pre-training [17, 22, 47], rather than just as input modalities:

- • Inertial Measurement Units (IMU): Instead of learning IMU representations in isolation, recent works utilize IMU data to enforce ego-motion consistency. By providing accurate acceleration and orientation priors, IMUs supervise the temporal alignment of vision and LiDAR backbones, essential for learning physically plausible world models.
- • GPS and Localization Signals: Global positioning data provides coarse-grained location context. In large-scale pre-training, GPS traces are often used to retrieve topologically neighboring scenes or to enforce trajectory consistency in long-horizon prediction tasks.
- • Thermal/Infrared Sensors: In safety-critical applications, these sensors provide distinct signatures for living beings (e.g., pedestrians and animals) that are invisible to standard cameras at night. Pretraining on thermal data typically follows domain adaptation paradigms to transfer RGB-based semantic knowledge to the thermal domain.

##### Table 6 Comparative analysis of 3D object detection on the nuScenes benchmark [17]. The table reports the mean Average Precision (mAP) and NuScenes Detection Score (NDS) of various pre-training frameworks. The values in parentheses denote the performance gains relative to the corresponding baseline methods.

Auxiliary Data Performance

Method Venue Backbone Image Size Baseline

Temporal Pre-Training Others mAP NDS

FCOS3D [224] ICCVW’21 ResNet101 [60] 1600 × 900 BEVFormer [121] ✓ ImageNet [34] - 41.6(+3.9) 51.7 (+4.0) GeoMIM [139] ICCV’23 Swin-B [147] 1408 × 512 BEVDepth [119] ✗ ImageNet [34] - 52.3(+5.7) 60.5 (+5.0)

OccNet [206] ICCV’23 ResNet101 [60] 1600 × 900 BEVFormer [121] ✓ ImageNet [34] - 43.6(+2.0) 53.2 (+1.5) UniScene [164] RA-L’24 ResNet101 [60] 1600 × 900 BEVFormer [121] ✓ FCOS3D [224] - 43.8(+2.2) 53.4 (+1.7)

DriveWorld [165] CVPR’24 ResNet101 [60] 1600 × 900 BEVFormer [121] ✓ FCOS3D [224] - 44.2(+2.6) 53.6 (+1.9)

ViDAR [270] CVPR’24 ResNet101 [60] 1600 × 900 BEVFormer [121] ✓ FCOS3D [224] - 45.8(+4.3) 54.8 (+4.3) UniPAD [264] CVPR’24 ConNeXt-S [149] 1600 × 900 UVTR-CS [114] ✗ FCOS3D [224] - 42.8(+3.6) 50.2 (+1.4)

MIM4D [322] IJCV’25 ResNet50 [60] 704 × 256 Sparse4Dv3 [133] ✓ ImageNet [34] - 46.4(+0.1) 57.0 (+0.6) GaussianPretrain [250] arXiv’24 ResNet50 [60] 1600 × 900 StreamPETR [217] ✗ ImageNet [34] - 38.6(+0.6) 48.8 (+0.9)

VisionPAD [288] CVPR’25 ResNet101 [60] 1600 × 900 UVTR-CS [114] ✓ FCOS3D [224] - 43.1(+3.9) 50.4 (+1.6)

SQS [289] NeurIPS’25 ResNet101 [60] 1408 × 512 SparseBEV [135] ✓ FCOS3D [224] - 50.9(+0.8) 60.2 (+1.0) UniPAD [264] CVPR’24 VoxelNet [311] - UVTR-L [114] ✗ - - 65.0(+4.1) 70.6 (+2.9)

UniM2AE [323] ECCV’24 SST [42] - TransFusion [6] ✗ - - 65.7(+0.7) 70.4 (+0.5) NS-MAE [257] arXiv’24 VoxelNet [311]+Swin-T [147] 704 × 256 BEVFusion [148] ✗ - - 63.0(+2.2) 65.5 (+1.4)

UniPAD [264] CVPR’24 VoxelNet [311]+ConNeXt-S [149] 1600 × 900 UVTR-M [114] ✗ FCOS3D [224] - 69.9(+4.5) 73.2 (+3.0) UniM2AE [323] ECCV’24 SST [42]+Swin-T [147] 1600 × 900 FocalFormer3D [28] ✗ MMIM [323] - 71.1(+0.6) 73.8 (+0.7)

- 4.4 Empirical Analysis and Benchmark Performance

To empirically substantiate the efficacy of the discussed pre-training paradigms, we evaluate their impact on core 3D perception tasks: 3D Object Detection and LiDAR Semantic Segmentation. These tasks serve as the definitive litmus test for Spatial Intelligence, assessing whether learned representations can translate pretext objectives (e.g., reconstruction and forecasting) into precise geometric localization and fine-grained semantic understanding. In this subsection, we synthesize key findings from major benchmarks, highlighting how different pre-training strategies reshape the performance landscape.

- 4.4.1 3D Object Detection

- 3D object detection requires the model to identify and localize objects within a metric space, a task that demands both high-level semantics and low-level geometric precision. Quantitative results on the nuScenes benchmark (Table 6) provide compelling evidence for the superiority of Unified Pre-Training.

As shown in the comparative analysis, frameworks that jointly optimize multi-modal encoders consistently outperform camera-only baselines. Notably, UniM2AE [323] achieves state-of-the-art performance with 71.1 mAP and 73.8 NDS, representing a significant gain over the strong FocalFormer3D [28] baseline. Similarly, UniPAD [264] demonstrates remarkable robustness, boosting the UVTR-M [114] baseline by +4.5 mAP to reach 69.9 mAP. This suggests that learning a shared latent space for vision and geometry allows the model to capture complementary features that are otherwise lost in late-fusion pipelines, proving that unified multi-modal masking is superior to disjoint training strategies.

- 4.4.2 LiDAR Segmentation

LiDAR semantic segmentation, involving dense point-level classification, is the rigorous testing ground for the Semantic-Geometric Gap. Since point clouds inherently lack texture, performance on this task directly reflects a model’s ability to hallucinate semantics from geometry. The comparisons in Table 7 reveal a decisive trend: Camera-to-LiDAR Distillation is indispensable, particularly for Data Efficiency.

Approaches utilizing visual priors consistently surpass training-from-scratch baselines, with advantages magnified in data-scarce regimes. For instance, with only 1% of labeled data, the random baseline yields a poor mIoU of 30.30. In stark contrast, distillation-based methods like OLIVINE [297] and LiMoE [254] achieve 50.58 and 49.60 mIoU respectively, effectively doubling the performance of the baseline. This indicates that self-supervised pre-training effectively unlocks the latent geometric structure of unlabeled data, significantly reducing the dependency on costly manual annotations.

Crucially, the results uncover a Scaling Law Transfer phenomenon. Advanced distillation methods like LiMoE [254] not only achieve state-of-the-art results on the full dataset (77.27 mIoU) but also demonstrate

##### Table 7 Benchmark of cross-modal pre-training for LiDAR semantic segmentation on nuScenes [17]. We evaluate the transferability of visual semantics to 3D point clouds via knowledge distillation. The results highlight performance gains across varying data regimes (e.g., 1% vs. 100% labeled data), underscoring the data efficiency of LiDAR-centric pre-training.

Backbone Backbone nuScenes KITTI Waymo (2D) (3D) LP 1% 5% 10% 25% Full 1% 1%

Method Venue

Random - None MinkUNet-34 8.10 30.30 47.84 56.15 65.48 74.66 39.50 39.41 PointContrast [243] ECCV’20

1.90 32.50 - - - - 41.10 DepthContrast [299] ICCV’21 2.10 31.70 - - - - 41.50 -

MinkUNet-34 [32]

None

ALSO [14] CVPR’23 - 37.70 - 59.40 - 72.00 - BEVContrast [189] 3DV’24 - 38.30 - 59.60 - 72.30 - -

PPKT [146] arXiv’21

35.90 37.80 53.74 60.25 67.14 74.52 44.00 47.60 SLidR [188] CVPR’22 38.80 38.30 52.49 59.84 66.91 74.79 44.60 47.12

ST-SLidR [156] CVPR’23 40.48 40.75 54.69 60.75 67.70 75.14 44.72 44.93 TriCC [174] CVPR’23 38.00 41.20 54.10 60.40 67.60 75.60 45.90 Seal [144] NeurIPS’23 44.95 45.84 55.64 62.97 68.41 75.60 46.63 49.34

ResNet-50 [60]

MinkUNet-34 [32]

CSC [23] CVPR’24 46.00 47.00 57.00 63.30 68.60 75.70 47.20 OLIVINE [297] NeurIPS’24 50.09 50.58 60.19 65.01 70.13 76.54 49.38 HVDistill [294] IJCV’24 39.50 42.70 56.60 62.90 69.30 76.60 49.70 LargeAD [104] arXiv’25 46.13 47.08 56.90 63.74 69.34 76.03 49.55 50.29

PPKT [146] arXiv’21

38.60 40.60 52.06 59.99 65.76 73.97 43.25 47.44 SLidR [188] CVPR’22 44.70 41.16 53.65 61.47 66.71 74.20 44.67 47.57

Seal [144] NeurIPS’23 45.16 44.27 55.13 62.46 67.64 75.58 46.51 48.67 ScaLR [179] CVPR’24 42.40 40.50 - - - - - -

ViT-S [170]

MinkUNet-34 [32]

SuperFlow [253] ECCV’24 46.44 47.81 59.44 64.47 69.20 76.54 47.97 49.94 LargeAD [104] arXiv’25 46.58 46.78 57.33 63.85 68.66 75.75 50.07 50.83 LiMoE [254] CVPR’25 48.20 49.60 60.54 65.65 71.39 77.27 49.53 51.42

that 3D backbones can inherit the rich, open-world semantics of large-scale 2D Foundation Models (utilizing ViT-S teachers). It validates the hypothesis that forging Spatial Intelligence does not require reinventing semantic understanding, but rather effectively transferring it from the vision domain to the 3D physical world.

### 5 Open-World Perception and Planning

The ultimate goal of Spatial Intelligence is not merely to perceive closed-set categories but to generalize to the open world and make robust decisions in unseen scenarios. Traditional perception systems, constrained by fixed ontologies and supervised data, struggle with the long-tail unpredictability of real-world environments. In this section, we explore how multi-modal pre-training is evolving to address these challenges. We first analyze the demands of Open-World Perception (Section 5.1). We then discuss how Text-Grounded Understanding leverages Vision-Language Models (VLMs) to bridge the semantic gap and automate supervision (Section 5.2). Finally, we examine the culmination of these efforts in Unified World Representations, where generative world models and Vision-Language-Action (VLA) architectures are redefining end-to-end planning (Section 5.3).

- 5.1 Open-World Challenges

Open-world deployment introduces complexity vectors that exceed the capacity of traditional representation learning:

- • Open-Vocabulary Recognition: Systems must identify novel objects (e.g., "overturned truck", "debris") that were never explicitly annotated during training, requiring a shift from ID-based classification to language-driven reasoning.
- • Domain Shifts and Anomalies: Robustness against changing weather, lighting, and sensor degradation is critical. Models must quantify epistemic uncertainty to handle "unknown unknowns" safely.
- • Data Scalability: The combinatorial explosion of corner cases makes manual annotation infeasible. Learning from vast, unlabeled, diverse data streams is the only viable path to coverage.

##### Table 8 Performance comparison for self-supervised 3D occupancy prediction on Occ3D-nuScenes [204]. This table assesses the capability of methods to learn dense volumetric representations without manual 3D labels. “FM” denotes the specific 2D Foundation Model utilized for pseudo-label generation or feature distillation.

Performance

Method Venue Representation Foundation Model Used Supervision Other Supported Task

IoU mIoU

SimpleOcc [46] TIV’24 NeRF - Video Sequence Depth Estimation - 7.99 OccNeRF [285] TIP’25 NeRF Grounding DINO [143] Video Sequence & FM Depth Estimation 22.81 9.53

SelfOcc [75] CVPR’24 BEV/TPV Feature OpenSeeD [290] Video Sequence & FM Novel Depth Synthesi/Depth Estimation 45.01 9.30 DistillNeRF [214] NeurIPS’24 NeRF CLIP [181] & DINOv2 [170] FM Novel View Synthesi/Depth Estimation 29.11 8.93 GaussianOcc [45] ICCV’25 Gaussians Grounding DINO [143] Video Sequence & FM Depth Estimation - 9.94

GaussTR [75] CVPR’25 Gaussians Metric3D [276] & CLIP [181] & SAM [95] FM Open-Vocabulary Occupancy Prediction 45.19 11.70

LangOcc [10] 3DV’25 NeRF MaskCLIP [307] Video Sequence & FM 3D Open Vocabulary Retrieval 51.76 11.84 VEON-L [303] ECCV’24 Occ MiDAS [182] & SAN [247] & CLIP [181] FM & LiDAR 3D Open Vocabulary Retrieval - 15.14

TT-OccLiDAR [286] arXiv’25 Gaussians VGGT [212] & OpenSeeD [290] Video Sequence & FM & LiDAR Progressive Occupancy Estimation - 23.60 GaussianFlowOcc [11] arXiv’25 Gaussians GroundedSAM [186] & Metric3D [276] Video Sequence & FM Depth Estimation 46.91 17.08

ShelfOcc [12] arXiv’25 Voxel MapAnything [90] & GroundedSAM [186] Video Sequence & FM - 56.14 22.87 ShelfGaussian [300] arXiv’25 Gaussians DINOv2 [170] & Metric3D [276] Video Sequence & FM & LiDAR BEV Segmentation / Trajectory Planning 63.25 19.07

QueryOcc [131] arXiv’25 Query Metric3D [276] & GroundedSAM [186] & DinoV3 [195] Video Sequence & FM & LiDAR Depth Estimation 55.00 21.30

Addressing these challenges necessitates a paradigm shift: from learning specific tasks to learning generalizable world knowledge.

- 5.2 Text-Grounded Understanding

Language serves as the universal interface for open-world knowledge. By aligning 3D sensor data with rich textual semantics, foundation models can read the scene, unlocking zero-shot capabilities. This paradigm manifests in two key directions: Auto-Labeling Data Engines and Open-Vocabulary Representation Learning.

Auto-Labeling as a Scalable Data Engine. The most immediate impact of foundation models is breaking the annotation bottleneck. Instead of relying on human labelers, recent works utilize pre-trained VLMs [181, 216] to generate high-quality pseudo-labels for sensor data. CLIP2Scene [26] and OpenScene [176] pioneered the distillation of 2D vision-language features into 3D point clouds, effectively automating semantic segmentation. Advanced frameworks like Affinity3D [136] and VLM2Scene [129] further refine this process by enforcing multi-view consistency, ensuring that the hallucinated labels are geometrically coherent for downstream supervised training.

Text-Assisted Representation Learning. Beyond generating discrete labels, recent research focuses on SelfSupervised 3D Occupancy Prediction, treating text-aligned 2D features as continuous supervision signals. Methods like LangOcc [10] and LOcc [279] leverage knowledge distillation from diverse teacher models [5, 170, 183] to directly guide the learning of dense volumetric semantics. As shown in Table 8, these self-supervised approaches now rival supervised baselines, proving that foundation model-driven supervision can replace manual effort. Furthermore, the trend towards 3D Gaussian Splatting [11, 83, 300] illustrates the push for representations that are not only semantically rich but also geometrically continuous and renderable, facilitating better alignment with 2D VLMs.

- 5.3 Unified World Representation for Action

Perception serves as the foundation for decision-making, while the ultimate manifestation of Spatial Intelligence is Action. The field is transitioning from modular perception-planning pipelines to unified World Models that can simulate future states and plan end-to-end within a shared space.

From Discriminative to Generative Planning. Traditional end-to-end planning often relied on explicit perception outputs (e.g., bounding boxes and vectorized maps) or decoupled feature maps. Recent breakthroughs, however, are driven by Generative World Models [106, 269, 306]. Moving beyond discrete label prediction, models like OccWorld [305] and GenAD [265] learn to predict the future evolution of the 3D world (e.g., 4D Occupancy flow) conditioned on ego-actions. This predictive learning objective forces the model to internalize scene dynamics, causal relationships, and object interactions. As evidenced in Table 9, these generative planners significantly outperform discriminative baselines in both collision rates and open-loop planning metrics.

Unified End-to-End Architectures: VA and VLA. The convergence of generative modeling and autonomous driving has bifurcated into two powerful paradigms for action generation: Vision-Action (VA) latent models and Vision-Language-Action (VLA) reasoning frameworks. The first paradigm focuses on pure decision-making

##### Table 9 Evaluation of end-to-end planning on the nuScenes benchmark [17]. The table compares the planning fidelity of state-of-the-art methods, contrasting traditional pipelines with emerging Generative World Models. Performance is measured by planning L2 error (L2) and Collision Rate (CR), where lower values indicate better safety and precision.

Performance

Method Venue Input Representation Supported Task Auxiliary Supervision

L2 Avg. (m) CR Avg. FPS ST-P3 [70] ECCV’22 Image • BEV Feature BEV Seg. Map & Box & Depth 2.11 0.71 1.6

UniAD [72] CVPR’23 Image • BEV Feature Track./Map/Motion Fore./Occ. Map & Box & Motion & Tracklets & Occ 1.03 0.31 1.8 VAD-Tiny [82] ICCV’23 Image • Vectorized BEV Scene Vectorized Map Map & Box & Motion 1.30 0.72 16.8 VAD-Base [82] ICCV’23 Image • Vectorized BEV Scene Vectorized Map Map & Box & Motion 1.22 0.53 4.5

OccNet [206] ICCV’23 Image • 3D Occupancy Semantic Occ. Pred. 3D-Occ & Map & Box 2.14 0.72 2.6

OccWorld [305] ECCV’24 Image • 3D Occupancy 4D Occ. Fore. 3D-Occ 1.34 0.73 2.8 OccWorld [305] ECCV’24 Image • 3D Occupancy 4D Occ. Fore. None 1.83 2.02 2.8

OccWorld[305] ECCV’24 Occ • 3D Occupancy 4D Occ. Fore. None 1.17 0.60 18.0

RenderWorld [261] ICRA’25 Image • 3D Occupancy 4D Occ. Fore. None 1.48 0.97 RenderWorld [261] ICRA’25 Occ • 3D Occupancy 4D Occ. Fore. None 1.03 0.61 -

OccLLaMA [231] arXiv’24 Image • 3D Occupancy 4D Occ. Fore./VQA 3D-Occ 1.20 0.70 OccLLaMA [231] arXiv’24 Occ • 3D Occupancy 4D Occ. Fore./VQA None 1.14 0.49 -

OccVAR [85] arXiv’24 Image • 3D Occupancy 4D Occ. Fore. 3D-Occ 1.35 0.83 OccVAR [85] arXiv’24 Occ • 3D Occupancy 4D Occ. Fore. None 1.21 0.78 -

LAW [116] ICLR’25 Image • Latent Feature Latent Prediction Fore. None 0.61 0.30 19.5

SSR [111] ICLR’25 Image • BEV Feature BEV Feature Fore. None 0.39 0.06 19.5 FSF-Net [53] arXiv’24 Occ • 3D Occupancy 4D Occ. Fore. None 0.82 0.01 -

Drive-OccWorld [269] AAAI’25 Image • 3D Occupancy 4D Occ. Fore./Generation 3D-Occ 0.85 0.29 OccTens [86] arXiv’25 Occ • 3D Occupancy 4D Occ. Fore./Generation 3D-Occ 1.12 0.48 -

OccVLA [142] arXiv’25 Occ • 3D Occupancy 3D Occ. Generation 3D-Occ 0.28 - World4Drive [306] ICCV’25 Image • Latent Feature Latent Prediction Fore. Open-vocabulary Semantics 0.50 0.16 -

efficiency by constructing Latent World Models. Unlike traditional pipelines that rely on explicit perception supervision, methods like LAW [116] and SSR [111] bypass human annotations entirely. By abstracting the environment into high-dimensional latent states, these models learn to predict future rewards and control signals directly from sensor inputs without the need for perception labels.

Parallel to pure latent modeling, the integration of Large Language Models (LLMs) has catalyzed the emergence of VLA frameworks that emphasize interpretability and open-world reasoning [5, 138, 223]. Approaches like OccVLA [142] and DriveVLA-W0 [117] tokenize visual input and project them into the LLM’s context window alongside text. This enables the system to not only generate control actions but also to perform causal reasoning (“Why is the car stopping?”) and handle complex social interactions (“Yield to the aggressive merger”) in a unified autoregressive process.

In summary, the trajectory is clear: from detecting objects to simulating latent futures (VA), and finally to reasoning with language (VLA). This evolution underscores the pivotal role of multi-modal pre-training in constructing the next generation of embodied intelligent systems.

### 6 Challenges and Future Directions

As demonstrated in this work, the pursuit of Spatial Intelligence has evolved from task-specific supervision to a paradigm dominated by large-scale, multi-modal pre-training. While the techniques analyzed in Section 4 and Section 5 demonstrate immense progress, the rapid emergence of generative AI and foundation models introduces new frontiers. In this section, we synthesize critical remaining obstacles and outline a forwardlooking research agenda centered on generative world modeling and embodied reasoning.

- 6.1 Current Challenges

The Semantic-Geometric Gap. A fundamental dissonance remains between the rich semantic knowledge encapsulated in Vision-Language Models (VLMs) and the precise metric requirements of autonomous control. While VLMs excel at open-vocabulary recognition [5, 138, 181, 216], they often lack the fine-grained spatial grounding necessary to localize it with centimeter-level accuracy. Bridging the gap between high-level semantic reasoning and low-level geometric constraints without compromising either remains a formidable theoretical and engineering challenge [142, 231].

Data-Centric Bottlenecks and Corner Cases. The scaling laws of foundation models are increasingly hitting diminishing returns regarding data quality. The primary challenge has shifted from acquiring more data to mining valuable data—specifically, long-tail corner cases and safety-critical scenarios [49, 203, 281]. Current pre-training objectives treat all data samples equally, often wasting computation on repetitive driving patterns

while under-weighting rare, high-value events [264, 270]. Furthermore, utilizing foundation models for auto-labeling introduces epistemic uncertainty that is difficult to filter from the training pipeline.

Real-Time Inference of Foundation Models. There is a growing disparity between the computational demands of state-of-the-art pre-trained models and the strict latency/power constraints of onboard edge devices [233, 242]. While cloud-based pre-training leverages unlimited resources, distilling these massive teacher models into lightweight, real-time student networks without catastrophic performance drops is an ongoing bottleneck for deployment [253, 255, 294].

- 6.2 Future Directions

While recent advancements have laid the foundation for spatial intelligence, several critical frontiers remain to be conquered to achieve robust, human-level autonomy.

Physically Consistent World Simulators. Although emerging Generative World Models [106, 117, 306] can synthesize plausible futures, they often suffer from hallucinations that violate physical laws [124]. A key future direction is to enforce Physical Consistency within the pre-training objective. By integrating differentiable physics engines or explicit geometric constraints into the generation process [8, 117, 142, 226], future models must evolve from merely generating visual pixels to simulating realistic physical interactions, thereby serving as reliable training environments for safety-critical policies.

Trustworthy and Real-Time Embodied VLA. Current Vision-Language-Action (VLA) models [71, 84, 117, 142] demonstrate promise but face significant hurdles in real-world deployment: high inference latency and lack of interpretability. Future research should bridge the gap between heavy foundation models and the millisecondlevel reaction requirements of autonomous systems. This necessitates exploring lightweight VLA architectures, efficient tokenization strategies, and mechanisms for uncertainty quantification to ensure that end-to-end decision-making is not only intelligent but also trustworthy and verifiable [43, 112, 220, 278].

- 4D Semantic-Geometric Unification. The transition from discrete voxels to continuous representations like 3D Gaussian Splatting (3DGS) [83, 91, 140] is underway. However, current 3DGS methods largely focus on visual rendering quality rather than semantic understanding. The next frontier lies in Semantic Lifting—imbuing these continuous geometric primitives with dense semantic and instance-level attributes over time. Pre-training tasks that enforce spatiotemporal consistency on Gaussian attributes [11, 250] will be pivotal for enabling agents to not just view the scene, but to manipulate and interact with specific objects in a dynamic 4D world.

System 2 Reasoning for Long-Tail Safety. Existing pre-training paradigms excel at pattern recognition (System 1) but struggle with rare, complex scenarios requiring logical deduction. Future systems will integrate System 2 capabilities [71, 205], potentially via Chain-of-Thought (CoT) distillation from LLMs [218, 223, 230, 284]. The goal is to move beyond passive explanation to active Causal Reasoning—enabling the vehicle to counterfactually simulate what if scenarios and override reactive policies when facing novel, long-tail safety hazards.

### 7 Conclusion

In this paper, we have presented a systematic analysis of multi-modal pre-training for autonomous systems, characterizing the evolution from modality-specific pre-training to unified foundation models as the cornerstone of Spatial Intelligence. By structuring datasets and methodologies across autonomous vehicles, drones, and other robotic systems, we demonstrated how integrating complementary sensor modalities (specifically camera and LiDAR) creates representations that are both semantically rich and geometrically precise. Our analysis confirms that leveraging pre-trained foundation models is no longer optional but essential for achieving open-world generalization and mitigating the scarcity of annotated 3D data.

Looking ahead, the field stands at a critical inflection point. As demonstrated, the paradigm is shifting from passive perception to active, embodied reasoning. Future breakthroughs will likely stem from bridging the semantic-geometric gap through Generative World Models that serve as neural simulators, and from the development of end-to-end Vision-Language-Action (VLA) frameworks that unify perception with decisionmaking. Furthermore, equipping these systems with explicit reasoning capabilities will be pivotal for handling the long-tail unpredictability of real-world environments. Ultimately, the transition from seeing to acting and

reasoning represents the next frontier. Continued advancements in these generative and embodied pre-training paradigms will be instrumental in forging autonomous systems that are not only robust and scalable but possess true Spatial Intelligence for safe and real-world deployment.

References

- [1] Mohamed Abdelsamad, Michael Ulrich, Claudius Gläser, and Abhinav Valada. Multi-scale neighborhood occupancy masked autoencoder for self-supervised learning in LiDAR point clouds. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 22234–22243, 2025.
- [2] Inc. Aeva Technologies. AevaScenes: Open-access 4D FMCW LiDAR and camera dataset, 2025. URL https: //scenes.aeva.com.
- [3] Ben Agro, Quinlan Sykora, Sergio Casas, Thomas Gilles, and Raquel Urtasun. UnO: Unsupervised occupancy fields for perception and forecasting. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 14487– 14496, 2024.
- [4] Raghad Alqobali, Maha Alshmrani, Reem Alnasser, Asrar Rashidi, Tareq Alhmiedat, and Osama Moh’D. Alia. A survey on robot semantic navigation systems for indoor environments. Applied Sciences, 14(1):89, 2023.
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.
- [6] Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, and Chiew-Lan Tai. TransFusion: Robust LiDAR-camera fusion for 3D object detection with transformers. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 1090–1099, 2022.
- [7] Bifta Sama Bari, Deepak Puthal, and Kumar Yelamarthi. Datasets in vehicular communication systems: A review of current trends and future prospects. SN Computer Science, 6(3):1–25, 2025.
- [8] Florent Bartoccioni, Elias Ramzi, Victor Besnier, Shashanka Venkataramanan, Tuan-Hung Vu, Yihong Xu, Loick Chambon, Spyros Gidaris, Serkan Odabas, David Hurych, Renaud Marlet, Alexandre Boulch, Mickael Chen, Éloi Zablocki, Andrei Bursuc, Eduardo Valle, and Matthieu Cord. VaViM and VaVAM: Autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672, 2025.
- [9] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall. SemanticKITTI: A dataset for semantic scene understanding of lidar sequences. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 9297–9307, 2019.
- [10] Simon Boeder, Fabian Gigengack, and Benjamin Risse. LangOcc: Self-supervised open vocabulary occupancy estimation via volume rendering. arXiv preprint arXiv:2407.17310, 2024.
- [11] Simon Boeder, Fabian Gigengack, and Benjamin Risse. GaussianFlowOcc: Sparse and weakly supervised occupancy estimation using Gaussian splatting and temporal flow. arXiv preprint arXiv:2502.17288, 2025.
- [12] Simon Boeder, Fabian Gigengack, Simon Roesler, Holger Caesar, and Benjamin Risse. ShelfOcc: Native 3D supervision beyond LiDAR for vision-based occupancy estimation. arXiv preprint arXiv:2511.15396, 2025.
- [13] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181, 2025.
- [14] Alexandre Boulch, Corentin Sautier, Björn Michele, Gilles Puy, and Renaud Marlet. ALSO: Automotive LiDAR self-supervision by occupancy estimation. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 13455–13465, 2023.
- [15] Borja Bovcon, Jon Muhovič, Duško Vranac, Dean Mozetič, Janez Perš, and Matej Kristan. MODS—a USVoriented object detection and obstacle segmentation benchmark. IEEE Trans. Intell. Transport. Sys., 23(8): 13403–13418, 2021.
- [16] Ilker Bozcan and Erdal Kayacan. AU-Air: A multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance. In Proc. IEEE Int. Conf. Robot. Automat., pages 8504–8510, 2020.

- [17] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 11621–11631, 2020.
- [18] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810, 2021.
- [19] Mu Cai, Chenxu Luo, Yong Jae Lee, and Xiaodong Yang. Cross-modal self-supervised learning with effective contrastive units for LiDAR point clouds. arXiv preprint arXiv:2409.06827, 2024.
- [20] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 9650–9660, 2021.
- [21] Kenneth Chaney, Fernando Cladera, Ziyun Wang, Anthony Bisulco, M Ani Hsieh, Christopher Korpela, Vijay Kumar, Camillo J Taylor, and Kostas Daniilidis. M3ED: Multi-robot, multi-sensor, multi-environment event dataset. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog. Worksh., pages 4016–4023, 2023.
- [22] Ming-Fang Chang, John W Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays. Argoverse: 3D tracking and forecasting with rich maps. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019.
- [23] Haoming Chen, Zhizhong Zhang, Yanyun Qu, Ruixin Zhang, Xin Tan, and Yuan Xie. Building a strong pre-training baseline for universal 3D large-scale perception. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 19925–19935, 2024.
- [24] Runjian Chen, Hyoungseob Park, Bo Zhang, Wenqi Shao, Ping Luo, and Alex Wong. TREND: Unsupervised 3D representation learning via temporal forecasting for LiDAR perception. arXiv preprint arXiv:2412.03054, 2024.
- [25] Runjian Chen, Hang Zhang, Avinash Ravichandran, Wenqi Shao, Alex Wong, and Ping Luo. CLAP: Unsupervised 3D representation learning for fusion 3D perception via curvature sampling and prototype learning. arXiv preprint arXiv:2412.03059, 2024.
- [26] Runnan Chen, Youquan Liu, Lingdong Kong, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, Yu Qiao, and Wenping Wang. CLIP2Scene: Towards label-efficient 3D scene understanding by CLIP. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 7020–7030, 2023.
- [27] Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge Zhu, Yuexin Ma, Tongliang Liu, and Wenping Wang. Towards label-free scene understanding by vision foundation models. In Adv. Neural Inf. Process. Syst., volume 36, 2024.
- [28] Yilun Chen, Zhiding Yu, Yukang Chen, Shiyi Lan, Anima Anandkumar, Jiaya Jia, and Jose M. Alvarez. FocalFormer3D: Focusing on hard instance for 3D object detection. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 8394–8405, 2023.
- [29] Yujin Chen, Matthias Nießner, and Angela Dai. 4DContrast: Contrastive learning with dynamic correspondences for 3D scene understanding. In Proc. Eur. Conf. Comput. Vis., pages 543–560. Springer, 2022.
- [30] Nuo Cheng, Chuanyu Luo, Xinzhe Li, Ruizhi Hu, Han Li, Sikun Ma, Zhong Ren, Haipeng Jiang, Xiaohan Li, Shengguang Lei, et al. Rethinking masked-autoencoder-based 3D point cloud pretraining. In IEEE Intell. Veh. Symposium, pages 2763–2768, 2024.
- [31] Yuwei Cheng, Jiannan Zhu, Mengxin Jiang, Jie Fu, Changsong Pang, Peidong Wang, Kris Sankaran, Olawale Onabola, Yimin Liu, Dianbo Liu, et al. Flow: A dataset and benchmark for floating waste detection in inland waters. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 10953–10962, 2021.
- [32] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D spatio-temporal ConvNets: Minkowski convolutional neural networks. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 3075–3084, 2019.
- [33] Colin Decourt, Rufin VanRullen, Didier Salle, and Thomas Oberlin. Leveraging self-supervised instance contrastive learning for radar object detection. arXiv preprint arXiv:2402.08427, 2024.
- [34] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 248–255, 2009.

- [35] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proc. Conf. of the North American Chapter of the Association for Computational Linguistics, pages 4171–4186, 2019.
- [36] Jian Ding, Nan Xue, Gui-Song Xia, Xiang Bai, Wen Yang, Michael Ying Yang, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, et al. Object detection in aerial images: A large-scale benchmark and challenges. IEEE Trans. Pattern Anal. Mach. Intell., 44(11):7778–7796, 2021.
- [37] Xu Dong, Binnan Zhuang, Yunxiang Mao, and Langechuan Liu. Radar camera fusion via representation learning in autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 1672–1681, 2021.
- [38] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In Int. Conf. Learn. Represent., 2021.
- [39] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. PaLM-E: An embodied multimodal language model. In Proc. Int. Conf. Mach. Learn., pages 8469–8488. PMLR, 2023.
- [40] Dawei Du, Yuankai Qi, Hongyang Yu, Yifan Yang, Kaiwen Duan, Guorong Li, Weigang Zhang, Qingming Huang, and Qi Tian. The unmanned aerial vehicle benchmark: Object detection and tracking. In Proc. Eur. Conf. Comput. Vis., pages 370–386. Springer, 2018.
- [41] Aritra Dutta, Srijan Das, Jacob Nielsen, Rajatsubhra Chakraborty, and Mubarak Shah. Multiview aerial visual recognition (MAVREC): Can multi-view improve aerial visual perception? In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 22678–22690, 2024.
- [42] Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Embracing single stride 3D object detector with sparse transformer. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 8458–8468, 2022.
- [43] Sicheng Feng, Song Wang, Shuyi Ouyang, Lingdong Kong, Zikai Song, Jianke Zhu, Huan Wang, and Xinchao Wang. Can mllms guide me home? a benchmark study on fine-grained visual reasoning from transit maps. arXiv preprint arXiv:2505.18675, 2025.
- [44] Felix Fent, Fabian Kuttenreich, Florian Ruch, Farija Rizwin, Stefan Juergens, Lorenz Lechermann, Christian Nissler, Andrea Perl, Ulrich Voll, Min Yan, et al. MAN TruckScenes: A multimodal dataset for autonomous trucking in diverse conditions. arXiv preprint arXiv:2407.07462, 2024.
- [45] Wanshui Gan, Fang Liu, Hongbin Xu, Ningkai Mo, and Naoto Yokoya. GaussianOcc: Fully self-supervised and efficient 3D occupancy estimation with gaussian splatting. arXiv preprint arXiv:2408.11447, 2024.
- [46] Wanshui Gan, Ningkai Mo, Hongbin Xu, and Naoto Yokoya. A comprehensive framework for 3D occupancy estimation in autonomous driving. IEEE Trans. Intell. Veh., 9(12):7852–7864, 2024.
- [47] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 3354–3361, 2012.
- [48] Jakob Geyer, Yohannes Kassahun, Mentar Mahmudi, Xavier Ricou, Rupesh Durgesh, Andrew S Chung, Lorenz Hauswald, Viet Hoang Pham, Maximilian Mühlegg, Sebastian Dorn, et al. A2D2: Audi autonomous driving dataset. arXiv preprint arXiv:2004.06320, 2020.
- [49] Anurag Ghosh, Shen Zheng, Robert Tamburo, Khiem Vuong, Juan Alvarez-Padilla, Hailiang Zhu, Michael Cardei, Nicholas Dunn, Christoph Mertz, and Srinivasa G. Narasimhan. ROADWork dataset: Learning to recognize, observe, analyze and drive through work zones. arXiv preprint arXiv:2406.07661, 2024.
- [50] Nikhil Gosala, Kürsat Petek, B Ravi Kiran, Senthil Yogamani, Paulo Drews-Jr, Wolfram Burgard, and Abhinav Valada. LetsMap: Unsupervised representation learning for label-efficient semantic BEV mapping. In Proc. Eur. Conf. Comput. Vis., pages 110–126. Springer, 2025.
- [51] Luis Gressenbuch, Klemens Esterle, Tobias Kessler, and Matthias Althoff. Mona: The Munich motion dataset of natural driving. In Proc. IEEE Int. Conf. Intell. Transport. Syst., pages 2093–2100, 2022.

- [52] Runwei Guan, Liye Jia, Fengyufan Yang, Shanliang Yao, Erick Purwanto, Xiaohui Zhu, Eng Gee Lim, Jeremy Smith, Ka Lok Man, Xuming Hu, et al. WaterVG: Waterway visual grounding based on text-guided vision and mmWave radar. arXiv preprint arXiv:2403.12686, 2024.
- [53] Erxin Guo, Pei An, You Yang, Qiong Liu, and An-An Liu. FSF-Net: Enhance 4D occupancy forecasting with coarse BEV scene flow for autonomous driving. arXiv preprint arXiv:2409.15841, 2024.
- [54] Prajwal Gurunath, Sumanth Udupa, Aditya Gandhamal, Shrikar Madhu, Aniruddh Sikdar, Suresh Sundaram, et al. IndraEye: Infrared electro-optical UAV-based perception dataset for robust downstream tasks. arXiv preprint arXiv:2410.20953, 2024.
- [55] Sehoon Ha, Joonho Lee, Michiel van de Panne, Zhaoming Xie, Wenhao Yu, and Majid Khadiv. Learning-based legged locomotion; state of the art and future perspectives. arXiv preprint arXiv:2406.01152, 2024.
- [56] Haiqian Han, Lingdong Kong, Jianing Li, Ao Liang, Chengtao Zhu, Jiacheng Lyu, Lai Xing Ng, Xiangyang Ji, Wei Tsang Ooi, and Benoit R. Cottereau. Learning to remove lens flare in event camera. arXiv preprint arXiv:2512.09016, 2025.
- [57] Lei Han, Qingxu Zhu, Jiapeng Sheng, Chong Zhang, Tingguang Li, Yizheng Zhang, He Zhang, Yuzhen Liu, Cheng Zhou, Rui Zhao, et al. Lifelike agility and play in quadrupedal robots using reinforcement learning and generative pre-trained models. Nature Machine Intelligence, pages 1–12, 2024.
- [58] Yiduo Hao, Sohrab Madani, Junfeng Guan, Mohammed Alloulah, Saurabh Gupta, and Haitham Hassanieh. Bootstrapping autonomous driving radars with self-supervised learning. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 15012–15023, 2024.
- [59] Jeanine Harb, Nicolas Rébéna, Raphaël Chosidow, Grégoire Roblin, Roman Potarusov, and Hatem Hajri. FRSign: A large-scale traffic light dataset for autonomous trains. arXiv preprint arXiv:2002.05665, 2020.
- [60] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 770–778, 2016.
- [61] Kaiming He, Ross Girshick, and Piotr Dollár. Rethinking ImageNet pre-training. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 4918–4927, 2019.
- [62] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 9729–9738, 2020.
- [63] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 16000–16009, 2022.
- [64] Deepti Hegde, Suhas Lohit, Kuan-Chuan Peng, Michael J Jones, and Vishal M Patel. Equivariant spatio-temporal self-supervision for LiDAR object detection. In Proc. Eur. Conf. Comput. Vis., pages 475–491. Springer, 2024.
- [65] Julia Hindel, Nikhil Gosala, Kevin Bregler, and Abhinav Valada. INOD: Injected noise discriminator for self-supervised representation learning in agricultural fields. IEEE Robotics Autom. Letters, 2023.
- [66] Julia Hindel, Rohit Mohan, Jelena Bratulić, Daniele Cattaneo, Thomas Brox, and Abhinav Valada. Label-efficient LiDAR scene understanding with 2D-3D vision transformer adapters. In IEEE Int. Conf. Robotics Autom. Worksh., 2025.
- [67] Ji Hou, Benjamin Graham, Matthias Nießner, and Saining Xie. Exploring data-efficient 3D scene understanding with contrastive scene contexts. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 15587–15597, 2021.
- [68] John Houston, Guido Zuidhof, Luca Bergamini, Yawei Ye, Long Chen, Ashesh Jain, Sammy Omari, Vladimir Iglovikov, and Peter Ondruska. One thousand and one hours: Self-driving motion prediction dataset. In Conf. Robot Learn. PMLR, 2020.
- [69] Qingyong Hu, Bo Yang, Sheikh Khalid, Wen Xiao, Niki Trigoni, and Andrew Markham. Towards semantic segmentation of urban-scale 3D point clouds: A dataset, benchmarks and challenges. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 4977–4987, 2021.
- [70] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. ST-P3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In Proc. Eur. Conf. Comput. Vis., pages 533–549. Springer, 2022.

- [71] Tianshuai Hu, Xiaolu Liu, Song Wang, Yiyao Zhu, Ao Liang, Lingdong Kong, Guoyang Zhao, Zeying Gong, Jun Cen, Zhiyu Huang, Xiaoshuai Hao, Linfeng Li, Hang Song, Xiangtai Li, Jun Ma, Shaojie Shen, Jianke Zhu, Dacheng Tao, Ziwei Liu, and Junwei Liang. Vision-language-action models for autonomous driving: Past, present, and future. arXiv preprint arXiv:2512.16760, 2025.
- [72] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 17853–17862, 2023.
- [73] Linyan Huang, Huijie Wang, Jia Zeng, Shengchuan Zhang, Liujuan Cao, Junchi Yan, and Hongyang Li. LiDARguided geometric pretraining for vision-centric 3D object detection. Int. J. Comput. Vis., 133(7):3877–3890, 2025.
- [74] Xinyu Huang, Peng Wang, Xinjing Cheng, Dingfu Zhou, Qichuan Geng, and Ruigang Yang. The ApolloScape open dataset for autonomous driving and its application. IEEE Trans. Pattern Anal. Mach. Intell., 42(10): 2702–2719, 2019.
- [75] Yuanhui Huang, Wenzhao Zheng, Borui Zhang, Jie Zhou, and Jiwen Lu. SelfOcc: Self-supervised vision-based 3D occupancy prediction. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 19946–19956, 2024.
- [76] Zhenpeng Huang, Chao Li, Hao Chen, Yongjian Deng, Yifeng Geng, and Limin Wang. Data-efficient event camera pre-training via disentangled masked modeling. arXiv preprint arXiv:2403.00416, 2024.
- [77] Muhammad Zubair Irshad, Sergey Zakharov, Vitor Guizilini, Adrien Gaidon, Zsolt Kira, and Rares Ambrus. NeRF-MAE: Masked autoencoders for self-supervised 3D representation learning for neural radiance fields. In Proc. Eur. Conf. Comput. Vis., pages 434–453. Springer, 2024.
- [78] Andrej Janda, Brandon Wagstaff, Edwin G Ng, and Jonathan Kelly. Self-supervised pre-training of 3D point cloud networks with image data. arXiv preprint arXiv:2211.11801, 2022.
- [79] Mingi Jeong and Alberto Quattrini Li. Efficient LiDAR-based in-water obstacle detection and segmentation by autonomous surface vehicles in aquatic environments. In Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., pages 5387–5394, 2021.
- [80] Mingi Jeong, Arihant Chadda, Ziang Ren, Luyang Zhao, Haowen Liu, Monika Roznere, Aiwei Zhang, Yitao Jiang, Sabriel Achong, Samuel Lensgraf, et al. Multi-modal perception dataset of in-water objects for autonomous surface vehicles. arXiv preprint arXiv:2404.18411, 2024.
- [81] Aleksandar Jevtić et al. Comparison of interaction modalities for mobile indoor robot guidance: Direct physical interaction, person following, and pointing control. IEEE Trans. Human-Machine Sys., 45(6):653–663, 2015.
- [82] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. VAD: Vectorized scene representation for efficient autonomous driving. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 8340–8350, 2023.
- [83] Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, and Xinggang Wang. GaussTR: Foundation model-aligned gaussian transformer for self-supervised 3D spatial understanding. arXiv preprint arXiv:2412.13193, 2024.
- [84] Sicong Jiang, Zilin Huang, Kangan Qian, Ziang Luo, Tianze Zhu, Yang Zhong, Yihong Tang, Menglin Kong, Yunlong Wang, Siwen Jiao, Hao Ye, Zihao Sheng, Xin Zhao, Tuopu Wen, Zheng Fu, Sikai Chen, Kun Jiang, Diange Yang, Seongjin Choi, and Lijun Sun. A survey on vision-language-action models for autonomous driving. arXiv preprint arXiv:2506.24044, 2025.
- [85] Bu Jin, Xiaotao Hu, Yupeng Zheng, Xiaoyang Guo, Qian Zhang, Yao Yao, Diming Zhang, Xiaoxiao Long, Wei Yin, et al. OccVAR: Scalable 4D occupancy prediction via next-scale prediction. 2024.
- [86] Bu Jin, Songen Gu, Xiaotao Hu, Yupeng Zheng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Wei Yin. OccTENS: 3D occupancy world model via temporal next-scale prediction. arXiv preprint arXiv:2509.03887, 2025.
- [87] Xin Kang, Lei Chu, Jiahao Li, Xuejin Chen, and Yan Lu. Hierarchical intra-modal correlation learning for labelfree 3D semantic segmentation. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 28244–28253, 2024.

- [88] Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, et al. Towards learningbased planning: The nuPlan benchmark for real-world autonomous driving. arXiv preprint arXiv:2403.04133, 2024.
- [89] Parneet Kaur, Arslan Aziz, Darshan Jain, Harshil Patel, Jonathan Hirokawa, Lachlan Townsend, Christoph Reimers, and Fiona Hua. Sea situational awareness (SeaSaw) dataset. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 2579–2587, 2022.
- [90] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. MapAnything: Universal feed-forward metric 3D reconstruction. arXiv preprint arXiv:2509.13414, 2025.
- [91] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
- [92] Abderrazzaq Kharroubi, Zouhair Ballouch, Rafika Hajji, Anass Yarroudh, and Roland Billen. Multi-context point cloud dataset and machine learning for railway semantic segmentation. Infrastructures, 9(4):71, 2024.
- [93] Mehar Khurana, Neehar Peri, James Hays, and Deva Ramanan. Shelf-supervised cross-modal pre-training for 3D object detection. In Conf. Robot Learn. PMLR, 2024.
- [94] Tarasha Khurana, Peiyun Hu, David Held, and Deva Ramanan. Point cloud forecasting as a proxy for 4D occupancy forecasting. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 1116–1124, 2023.
- [95] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 4015–4026, 2023.
- [96] Simon Klenk, David Bonello, Lukas Koestler, Nikita Araslanov, and Daniel Cremers. Masked event modeling: Self-supervised pretraining for event cameras. In Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., pages 2378–2388, 2024.
- [97] Lingdong Kong, Youquan Liu, Runnan Chen, Yuexin Ma, Xinge Zhu, Yikang Li, Yuenan Hou, Yu Qiao, and Ziwei Liu. Rethinking range view representation for LiDAR segmentation. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 228–240, 2023.
- [98] Lingdong Kong, Youquan Liu, Xin Li, Runnan Chen, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Robo3D: Towards robust and reliable 3D perception against corruptions. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 19994–20006, 2023.
- [99] Lingdong Kong, Jiawei Ren, Liang Pan, and Ziwei Liu. LaserMix for semi-supervised LiDAR semantic segmentation. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 21705–21715, 2023.
- [100] Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit R. Cottereau, and Wei Tsang Ooi. RoboDepth: Robust out-of-distribution depth estimation under corruptions. In Adv. Neural Inf. Process. Syst., volume 36, pages 21298–21342, 2023.
- [101] Lingdong Kong, Youquan Liu, Lai Xing Ng, Benoit R. Cottereau, and Wei Tsang Ooi. OpenESS: Event-based semantic scene understanding with open vocabularies. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 15686–15698, 2024.
- [102] Lingdong Kong, Dongyue Lu, Ao Liang, Rong Li, Yuhao Dong, Tianshuai Hu, Lai Xing Ng, Wei Tsang Ooi, and Benoit R. Cottereau. Talk2Event: Grounded understanding of dynamic scenes from event cameras. In Adv. Neural Inf. Process. Syst., volume 38, 2025.
- [103] Lingdong Kong, Dongyue Lu, Xiang Xu, Lai Xing Ng, Wei Tsang Ooi, and Benoit R Cottereau. EventFly: Event camera perception from ground to the sky. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 1472–1484, 2025.
- [104] Lingdong Kong, Xiang Xu, Youquan Liu, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Liu Ziwei. LargeAD: Large-scale cross-sensor data pretraining for autonomous driving. arXiv preprint arXiv:2501.04005, 2025.

- [105] Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang, Liang Pan, Kai Chen, Wei Tsang Ooi, and Ziwei Liu. Multi-modal data-efficient 3D scene understanding for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell., 47(5):3748–3765, 2025.
- [106] Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi, Steven C. H. Hoi, and Ziwei Liu. 3D and 4D world modeling: A survey. arXiv preprint arXiv:2509.07996, 2025.
- [107] Georg Krispel, David Schinagl, Christian Fruhwirth-Reisinger, Horst Possegger, and Horst Bischof. MAELi: Masked autoencoder for large-scale LiDAR point clouds. In Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., pages 3383–3392, 2024.
- [108] Christopher Lang, Alexander Braun, Lars Schillingmann, Karsten Haug, and Abhinav Valada. Self-supervised representation learning from temporal ordering of automated driving sequences. IEEE Robotics Autom. Letters, 2024.
- [109] Bowen Li, Changhong Fu, Fangqiang Ding, Junjie Ye, and Fuling Lin. All-day object tracking for unmanned aerial vehicle. IEEE Trans. Mobile Comput., 22(8):4515–4529, 2022.
- [110] Hongyang Li, Chonghao Sima, Jifeng Dai, Wenhai Wang, Lewei Lu, Huijie Wang, Jia Zeng, Zhiqi Li, Jiazhi Yang, Hanming Deng, et al. Delving into the devils of bird’s-eye-view perception: A review, evaluation and recipe. IEEE Trans. Pattern Anal. Mach. Intell., 46(4):2151–2170, 2023.
- [111] Peidong Li and Dixiao Cui. Does end-to-end autonomous driving really need perception tasks? arXiv preprint arXiv:2409.18341, 2024.
- [112] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. TokenPacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, pages 1–19, 2025.
- [113] Xinpeng Li and Xiaojiang Peng. Rail detection: An efficient row-based network and a new benchmark. In Proc. ACM Int. Conf. Multimedia, pages 6455–6463, 2022.
- [114] Yanwei Li, Yilun Chen, Xiaojuan Qi, Zeming Li, Jian Sun, and Jiaya Jia. Unifying voxel-based representation with transformer for 3D object detection. In Adv. Neural Inf. Process. Syst., volume 35, pages 18442–18455, 2022.
- [115] Ye Li, Lingdong Kong, Hanjiang Hu, Xiaohao Xu, and Xiaonan Huang. Is your LiDAR placement optimized for 3D scene understanding? In Adv. Neural Inf. Process. Syst., volume 37, pages 34980–35017, 2024.
- [116] Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481, 2024.
- [117] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, and Zhaoxiang Zhang. DriveVLA-W0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025.
- [118] Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via BEV world model. arXiv preprint arXiv:2504.01941, 2025.
- [119] Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, and Zeming Li. BEVDepth: Acquisition of reliable depth for multi-view 3D object detection. In Proc. AAAI Conf. Artifi. Intell., volume 37, pages 1477–1485, 2023.
- [120] Zhenyu Li, Zehui Chen, Ang Li, Liangji Fang, Qinhong Jiang, Xianming Liu, Junjun Jiang, Bolei Zhou, and Hang Zhao. SimIPU: Simple 2D image and 3D point cloud unsupervised pre-training for spatial-aware visual representations. In Proc. AAAI Conf. Artifi. Intell., volume 36, pages 1500–1508, 2022.
- [121] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In Proc. Eur. Conf. Comput. Vis., pages 1–18. Springer, 2022.
- [122] Zhuoling Li, Chuanrui Zhang, En Yu, and Haoqian Wang. Delving into the pre-training paradigm of monocular 3D object detection. arXiv preprint arXiv:2206.03657, 2022.
- [123] Ao Liang, Lingdong Kong, Dongyue Lu, Youquan Liu, Jian Fang, Huaici Zhao, and Wei Tsang Ooi. Perspectiveinvariant 3D object detection. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 27725–27738, 2025.

- [124] Ao Liang, Lingdong Kong, Tianyi Yan, Hongsi Liu, Wesley Yang, Ziqi Huang, Wei Yin, Jialong Zuo, Yixuan Hu, Dekai Zhu, Dongyue Lu, Youquan Liu, Guangfeng Jiang, Linfeng Li, Xiangtai Li, Long Zhuo, Lai Xing Ng, Benoit R. Cottereau, Changxin Gao, Liang Pan, Wei Tsang Ooi, and Ziwei Liu. WorldLens: Full-spectrum evaluations of driving world models in real world. arXiv preprint arXiv:2512.10958, 2025.
- [125] Ao Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, and Wei Tsang Ooi. LiDARCrafter: Dynamic 4D world modeling from LiDAR sequences. arXiv preprint arXiv:2508.03692, 2025.
- [126] Dingkang Liang, Dingyuan Zhang, Xin Zhou, Sifan Tu, Tianrui Feng, Xiaofan Li, Yumeng Zhang, Mingyang Du, Xiao Tan, and Xiang Bai. Seeing the future, perceiving the future: A unified driving world model for future generation and perception. arXiv preprint arXiv:2503.13587, 2025.
- [127] Hanxue Liang, Chenhan Jiang, Dapeng Feng, Xin Chen, Hang Xu, Xiaodan Liang, Wei Zhang, Zhenguo Li, and Luc Van Gool. Exploring geometry-aware contrast and clustering harmonization for self-supervised 3D object detection. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 3293–3302, 2021.
- [128] Quanmin Liang, Qiang Li, Xinzi Cao, Jinyi Lu, Mingyue Cui, Feidiao Yang, Wei Zhang, Kai Huang, and Yonghong Tian. Enhancing event camera data pretraining via prompt-tuning with visual models. 2024.
- [129] Guibiao Liao, Jiankun Li, and Xiaoqing Ye. VLM2Scene: Self-supervised image-text-LiDAR learning with foundation models for autonomous driving scene understanding. In Proc. AAAI Conf. Artifi. Intell., volume 38, pages 3351–3359, 2024.
- [130] Yiyi Liao, Jun Xie, and Andreas Geiger. KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2D and 3D. arXiv preprint arXiv:2109.13410, 2021.
- [131] Adam Lilja, Ji Lan, Junsheng Fu, and Lars Hammarstrand. QueryOcc: Query-based self-supervision for 3D semantic occupancy. arXiv preprint arXiv:2511.17221, 2025.
- [132] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In Proc. Eur. Conf. Comput. Vis., pages 740–755. Springer, 2014.
- [133] Xuewu Lin, Zixiang Pei, Tianwei Lin, Lichao Huang, and Zhizhong Su. Sparse4D v3: Advancing end-to-end 3D detection and tracking. arXiv preprint arXiv:2311.11722, 2023.
- [134] Zhiwei Lin, Yongtao Wang, Shengxiang Qi, Nan Dong, and Ming-Hsuan Yang. BEV-MAE: Bird’s eye view masked autoencoders for point cloud pre-training in autonomous driving scenarios. In Proc. AAAI Conf. Artifi. Intell., volume 38, pages 3531–3539, 2024.
- [135] Haisong Liu, Yao Teng, Tao Lu, Haiguang Wang, and Limin Wang. SparseBEV: High-performance sparse 3D object detection from multi-camera videos. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 18580–18590, 2023.
- [136] Haizhuang Liu, Junbao Zhuo, Chen Liang, Jiansheng Chen, and Huimin Ma. Affinity3D: Propagating instancelevel semantic affinity for zero-shot point cloud semantic segmentation. In Proc. ACM Int. Conf. Multimedia, pages 9019–9028, 2024.
- [137] Hao Liu, Minglin Chen, Yanni Ma, Haihong Xiao, and Ying He. Point cloud unsupervised pre-training via 3D gaussian splatting. arXiv preprint arXiv:2411.18667, 2024.
- [138] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Adv. Neural Inf. Process. Syst., volume 36, pages 34892–34916, 2023.
- [139] Jihao Liu, Tai Wang, Boxiao Liu, Qihang Zhang, Yu Liu, and Hongsheng Li. GeoMIM: Towards better 3D knowledge transfer via masked image modeling for multi-view 3D understanding. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 17839–17849, 2023.
- [140] Keyi Liu, Weidong Yang, Ben Fei, and Ying He. Gaussian2Scene: 3D scene representation learning via self-supervised learning with 3D gaussian splatting. arXiv preprint arXiv:2506.08777, 2025.
- [141] Ruihan Liu, Xiaoyi Wu, Xijun Chen, Liang Hu, and Yunjiang Lou. 4D-ROLLS: 4D radar occupancy learning via LiDAR supervision. arXiv preprint arXiv:2505.13905, 2025.
- [142] Ruixun Liu, Lingyu Kong, Derun Li, and Hang Zhao. OccVLA: Vision-language-action model with implicit 3D occupancy supervision. arXiv preprint arXiv:2509.05578, 2025.

- [143] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In Proc. Eur. Conf. Comput. Vis., pages 38–55. Springer, 2024.
- [144] Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. Segment any point cloud sequences by distilling vision foundation models. Adv. Neural Inf. Process. Syst., 36, 2024.
- [145] Youquan Liu, Lingdong Kong, Weidong Yang, Xin Li, Ao Liang, Runnan Chen, Ben Fei, and Tongliang Liu. La La LiDAR: Large-scale layout generation from LiDAR data. arXiv preprint arXiv:2508.03691, 2025.
- [146] Yueh-Cheng Liu, Yu-Kai Huang, Hung-Yueh Chiang, Hung-Ting Su, Zhe-Yu Liu, Chin-Tang Chen, Ching-Yu Tseng, and Winston H Hsu. Learning from 2D: Contrastive pixel-to-point knowledge transfer for 3D pretraining. arXiv preprint arXiv:2104.04687, 2021.
- [147] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 10012–10022, 2021.
- [148] Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela Rus, and Song Han. BEVFusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. In IEEE Int. Conf. Robotics Autom., pages 2774–2781, 2023.
- [149] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 11976–11986, 2022.
- [150] William Ljungbergh, Adam Lilja, Adam Tonderski Ling, Carl Lindström, Willem Verbeke, Junsheng Fu, Christoffer Petersson, Lars Hammarstrand, and Michael Felsberg. GASP: Unifying geometric and semantic self-supervised pre-training for autonomous driving. arXiv preprint arXiv:2503.15672, 2025.
- [151] Dongyue Lu, Lingdong Kong, Gim Hee Lee, Camille Simon Chane, and Wei Tsang Ooi. FlexEvent: Towards flexible event-frame object detection at varying operational frequencies. In Adv. Neural Inf. Process. Syst., volume 38, 2025.
- [152] Yuhang Lu, Qi Jiang, Runnan Chen, Yuenan Hou, Xinge Zhu, and Yuexin Ma. See more and know more: Zero-shot point cloud segmentation via multi-modal visual data. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 21674–21684, 2023.
- [153] Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, and Kailun Yang. Omnidirectional multi-object tracking. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 21959–21969, 2025.
- [154] Ye Lyu, George Vosselman, Gui-Song Xia, Alper Yilmaz, and Michael Ying Yang. UAVid: A semantic segmentation dataset for UAV imagery. ISPRS J. Photogrammetry Remote Sensing, 165:108–119, 2020.
- [155] Hichem Maaref and Claude Barret. Sensor-based navigation of a mobile robot in an indoor environment. Robotics Autonom. Sys., 38(1):1–18, 2002.
- [156] Anas Mahmoud, Jordan SK Hu, Tianshu Kuai, Ali Harakeh, Liam Paull, and Steven L Waslander. Self-supervised image-to-point distillation via semantically tolerant contrastive loss. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 7102–7110, 2023.
- [157] Anas Mahmoud, Ali Harakeh, and Steven Waslander. Image-to-LiDAR relational distillation for autonomous driving data. In Proc. Eur. Conf. Comput. Vis., pages 459–475. Springer, 2024.
- [158] Andrey Malinin, Neil Band, German Chesnokov, Yarin Gal, Mark JF Gales, Alexey Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan Provilkov, Vatsal Raina, et al. SHIFTS: A dataset of real distributional shift across multiple large-scale tasks. arXiv preprint arXiv:2107.07455, 2021.
- [159] Murari Mandal, Lav Kush Kumar, and Santosh Kumar Vipparthi. MOR-UAV: A benchmark dataset and baselines for moving object recognition in UAV videos. In Proc. ACM Int. Conf. Multimedia, pages 2626–2635, 2020.
- [160] Jiageng Mao, Minzhe Niu, Chenhan Jiang, Xiaodan Liang, Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, Jie Yu, Chunjing Xu, et al. One million scenes for autonomous driving: ONCE dataset. arXiv preprint arXiv:2106.11037, 2021.

- [161] Johannes Meier, Luca Scalerandi, Oussema Dhaouadi, Jacques Kaiser, Araslanov Nikita, and Daniel Cremers. CARLA Drone: Monocular 3D object detection from a different perspective. GCPR, 2024.
- [162] Takahiro Miki, Joonho Lee, Lorenz Wellhausen, and Marco Hutter. Learning to walk in confined spaces using 3D representation. arXiv preprint arXiv:2403.00187, 2024.
- [163] Chen Min, Liang Xiao, Dawei Zhao, Yiming Nie, and Bin Dai. Occupancy-MAE: Self-supervised pre-training large-scale LiDAR point clouds with masked occupancy autoencoders. IEEE Trans. Intell. Veh., 9(7):5150–5162, 2024.
- [164] Chen Min, Liang Xiao, Dawei Zhao, Yiming Nie, and Bin Dai. Multi-camera unified pre-training via 3D scene reconstruction. IEEE Robotics Autom. Letters, 9(4):3243–3250, 2024.
- [165] Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, et al. DriveWorld: 4D pre-trained scene understanding via world models for autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 15522–15533, 2024.
- [166] Matthias Mueller, Neil Smith, and Bernard Ghanem. A benchmark and simulator for UAV tracking. In Proc. Eur. Conf. Comput. Vis., pages 445–461. Springer, 2016.
- [167] Mahyar Najibi, Jingwei Ji, Yin Zhou, Charles R Qi, Xinchen Yan, Scott Ettinger, and Dragomir Anguelov. Unsupervised 3D perception with 2D vision-language distillation for autonomous driving. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 8602–8612, 2023.
- [168] Lucas Nunes, Rodrigo Marcuzzi, Xieyuanli Chen, Jens Behley, and Cyrill Stachniss. SegContrast: 3D point cloud feature representation learning through self-supervised segment discrimination. IEEE Robotics Autom. Letters, 7(2):2116–2123, 2022.
- [169] NVIDIA Corporation. PhysicalAI autonomous vehicles dataset, 2025. URL https://huggingface.co/datasets/ nvidia/PhysicalAI-Autonomous-Vehicles. Dataset release.
- [170] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Trans. Machine Learn. Research, 2024.
- [171] Aljoša Ošep, Tim Meinhardt, Francesco Ferroni, Neehar Peri, Deva Ramanan, and Laura Leal-Taixé. Better call SAL: Towards learning to segment anything in LiDAR. In Proc. Eur. Conf. Comput. Vis., pages 71–90. Springer, 2024.
- [172] Edoardo Palladin, Samuel Brucker, Filippo Ghilotti, Praveen Narayanan, Mario Bijelic, and Felix Heide. Selfsupervised sparse sensor fusion for long range perception. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 27498–27509, 2025.
- [173] Tai-Yu Pan, Chenyang Ma, Tianle Chen, Cheng Perng Phoo, Katie Z Luo, Yurong You, Mark Campbell, Kilian Q Weinberger, Bharath Hariharan, and Wei-Lun Chao. Pre-training LiDAR-based 3D object detectors through colorization. In Int. Conf. Learn. Represent., 2024.
- [174] Bo Pang, Hongchi Xia, and Cewu Lu. Unsupervised 3D point cloud representation learning by triangle constrained contrast for autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 5229–5239, 2023.
- [175] Dennis Park, Rares Ambrus, Vitor Guizilini, Jie Li, and Adrien Gaidon. Is pseudo-LiDAR needed for monocular 3D object detection? In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 3142–3152, 2021.
- [176] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. OpenScene: 3D scene understanding with open vocabularies. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 815–824, 2023.
- [177] Quang-Hieu Pham, Pierre Sevestre, Ramanpreet Singh Pahwa, Huijing Zhan, Chun Ho Pang, Yuda Chen, Armin Mustafa, Vijay Chandrasekhar, and Jie Lin. A*3D dataset: Towards autonomous driving in challenging environments. In Proc. IEEE Int. Conf. Robot. Automat., pages 2267–2273, 2020.
- [178] Mariia Pushkareva, Yuri Feldman, Csaba Domokos, Kilian Rambach, and Dotan Di Castro. Radar spectralanguage model for automotive scene parsing. arXiv preprint arXiv:2406.02158, 2024.
- [179] Gilles Puy, Spyros Gidaris, Alexandre Boulch, Oriane Siméoni, Corentin Sautier, Patrick Pérez, Andrei Bursuc, and Renaud Marlet. Three pillars improving vision foundation model distillation for lidar. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 21519–21529, 2024.

- [180] Bo Qiu, Yuzhou Zhou, Lei Dai, Bing Wang, Jianping Li, Zhen Dong, Chenglu Wen, Zhiliang Ma, and Bisheng Yang. WHU-RailWay3D: A diverse dataset and benchmark for railway point cloud semantic segmentation. IEEE Trans. Intell. Transport. Sys., 25(12):20900–20916, 2024.
- [181] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Int. Conf. Machine Learn., pages 8748–8763. PMLR, 2021.
- [182] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Trans. Pattern Anal. Mach. Intell., 44(3):1623–1637, 2020.
- [183] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [184] N Dinesh Reddy, Minh Vo, and Srinivasa G Narasimhan. CarFusion: Combining point tracking and part detection for dynamic 3D reconstruction of vehicles. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 1906–1915, 2018.
- [185] Laurenz Reichardt, Nikolas Ebert, and Oliver Wasenmüller. 360° from a single camera: A few-shot approach for LiDAR segmentation. In Proc. IEEE/CVF Int. Conf. Comput. Vis. Worksh., pages 1067–1075, 2023.
- [186] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded SAM: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.
- [187] Alexandre Robicquet, Amir Sadeghian, Alexandre Alahi, and Silvio Savarese. Learning social etiquette: Human trajectory understanding in crowded scenes. In Proc. Eur. Conf. Comput. Vis., pages 549–565. Springer, 2016.
- [188] Corentin Sautier, Gilles Puy, Spyros Gidaris, Alexandre Boulch, Andrei Bursuc, and Renaud Marlet. Imageto-LiDAR self-supervised distillation for autonomous driving data. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 9891–9901, 2022.
- [189] Corentin Sautier, Gilles Puy, Alexandre Boulch, Renaud Marlet, and Vincent Lepetit. BEVContrast: Selfsupervision in BEV space for automotive LiDAR point clouds. In Proc. IEEE Int. Conf. 3D Vis., pages 559–568, 2024.
- [190] Corentin Sautier, Gilles Puy, Alexandre Boulch, Renaud Marlet, and Vincent Lepetit. UNIT: Unsupervised online instance segmentation through time. arXiv preprint arXiv:2409.07887, 2024.
- [191] Chen Shi, Shaoshuai Shi, Kehua Sheng, Bo Zhang, and Li Jiang. DriveX: Omni scene modeling for learning generalizable world knowledge in autonomous driving. arXiv preprint arXiv:2505.19239, 2025.
- [192] Hao Shi, Ze Wang, Shangwei Guo, Mengfei Duan, Song Wang, Teng Chen, Kailun Yang, Lin Wang, and Kaiwei Wang. OneOcc: Semantic occupancy prediction for legged robots with a single panoramic camera. arXiv preprint arXiv:2511.03571, 2025.
- [193] Oren Shrout, Ori Nitzan, Yizhak Ben-Shabat, and Ayellet Tal. PatchContrast: Self-supervised pre-training for 3D object detection. arXiv preprint arXiv:2308.06985, 2023.
- [194] Lingyu Si, Gang Li, Changwen Zheng, and Fanjiang Xu. Self-supervised representation learning for the object detection of marine radar. In Proc. Int. Conf. Comput. Artifi. Intell., pages 751–760, 2022.
- [195] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025.
- [196] Sophia Sirko-Galouchenko, Alexandre Boulch, Spyros Gidaris, Andrei Bursuc, Antonin Vobecky, Patrick Pérez, and Renaud Marlet. OccFeat: Self-supervised occupancy feature prediction for pretraining BEV segmentation networks. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 4493–4503, 2024.

- [197] Boyi Sun, Yuhang Liu, Xingxia Wang, Bin Tian, Long Chen, and Fei-Yue Wang. 3D unsupervised learning by distilling 2D open-vocabulary segmentation models for autonomous driving. arXiv preprint arXiv:2405.15286, 2024.
- [198] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Chen Zhifeng, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 2446–2454, 2020.
- [199] Tianfang Sun, Zhizhong Zhang, Xin Tan, Yanyun Qu, and Yuan Xie. Exploring the untouched sweeps for conflict-aware 3D segmentation pretraining. arXiv preprint arXiv:2407.07465, 2024.
- [200] Rustam Tagiew, Pavel Klasek, Roman Tilly, Martin Köppel, Patrick Denzler, Philipp Neumaier, Tobias Klockau, Martin Boekhoff, and Karsten Schwalbe. Osdar23: Open sensor data for rail 2023. In Int. Conf. Robotics Autom. Engineer., pages 270–276. IEEE, 2023.
- [201] Zhiyu Tan, Zichao Dong, Cheng Zhang, Weikun Zhang, Hang Ji, and Hao Li. OvO: Open-vocabulary occupancy. arXiv preprint arXiv:2305.16133, 2023.
- [202] Sina Tayebati, Theja Tulabandhula, and Amit R Trivedi. Sense less, generate more: Pre-training LiDAR perception with masked autoencoders for ultra-efficient 3D sensing. arXiv preprint arXiv:2406.07833, 2024.
- [203] Ran Tian, Boyi Li, Xinshuo Weng, Yuxiao Chen, Edward Schmerling, Yue Wang, Boris Ivanovic, and Marco Pavone. Tokenize the world into object-level knowledge to address long-tail events in autonomous driving. In Conf. Robot Learn., pages 3656–3673. PMLR, 2025.
- [204] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3D: A large-scale 3D occupancy prediction benchmark for autonomous driving. In Adv. Neural Inf. Process. Syst., volume 36, 2024.
- [205] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. DriveVLM: The convergence of autonomous driving and large vision-language models. In Conf. Robot Learn., pages 4698–4726. PMLR, 2025.
- [206] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 8406–8415, 2023.
- [207] Tugce Toprak, Burak Belenlioglu, Burak Aydın, Cuneyt Guzelis, and M Alper Selver. Conditional weighted ensemble of transferred models for camera based onboard pedestrian detection in railway driver support systems. IEEE Trans. Veh. Tech., 69(5):5041–5054, 2020.
- [208] Gustaf Uggla and Milan Horemuz. Towards synthesized training data for semantic segmentation of mobile laser scanning point clouds: Generating level crossings from real and synthetic point cloud samples. Automation in Construction, 130:103839, 2021.
- [209] Antonin Vobecky, Oriane Siméoni, David Hurych, Spyridon Gidaris, Andrei Bursuc, Patrick Pérez, and Josef Sivic. POP-3D: Open-vocabulary 3D occupancy prediction from images. Adv. Neural Inf. Process. Syst., 36, 2024.
- [210] Binglu Wang, Lei Zhang, Zhaozhong Wang, Yongqiang Zhao, and Tianfei Zhou. Core: Cooperative reconstruction for multi-agent perception. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 8710–8720, 2023.
- [211] Chengyao Wang, Li Jiang, Xiaoyang Wu, Zhuotao Tian, Bohao Peng, Hengshuang Zhao, and Jiaya Jia. GroupContrast: Semantic-aware self-supervised representation learning for 3D understanding. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 4917–4928, 2024.
- [212] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 5294–5306, 2025.
- [213] Lening Wang, Wenzhao Zheng, Yilong Ren, Han Jiang, Zhiyong Cui, Haiyang Yu, and Jiwen Lu. OccSora: 4D occupancy generation models as world simulators for autonomous driving. arXiv preprint arXiv:2405.20337, 2024.

- [214] Letian Wang, Seung Wook Kim, Jiawei Yang, Cunjun Yu, Boris Ivanovic, Steven L Waslander, Yue Wang, Sanja Fidler, Marco Pavone, and Peter Karkus. DistillNeRF: Perceiving 3D scenes from single-glance images by distilling neural fields and foundation model features. arXiv preprint arXiv:2406.12095, 2024.
- [215] Ning Wang, Yuanyuan Wang, Yi Wei, Bing Han, and Yuan Feng. Marine vessel detection dataset and benchmark for unmanned surface vehicles. Applied Ocean Research, 142:103835, 2024.
- [216] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [217] Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3D object detection. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 3621–3631, 2023.
- [218] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M. Alvarez. OmniDrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 22442–22452, 2025.
- [219] Sijie Wang, Siqi Li, Yawei Zhang, Shangshu Yu, Shenghai Yuan, Rui She, Quanjiang Guo, JinXuan Zheng, Ong Kang Howe, Leonrich Chandra, Shrivarshann Srijeyan, Aditya Sivadas, Toshan Aggarwal, Heyuan Liu, Hongming Zhang, Chujie Chen, Junyu Jiang, Lihua Xie, and Wee Peng Tay. UAVScenes: A multi-modal dataset for UAVs. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 28946–28958, 2025.
- [220] Song Wang, Zhongdao Wang, Jiawei Yu, Wentong Li, Bailan Feng, Junbo Chen, and Jianke Zhu. ReliOcc: Towards reliable semantic occupancy prediction via uncertainty learning. arXiv preprint arXiv:2409.18026, 2024.
- [221] Song Wang, Jiawei Yu, Wentong Li, Wenyu Liu, Xiaolu Liu, Junbo Chen, and Jianke Zhu. Not all voxels are equal: Hardness-aware semantic scene completion with self-distillation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14792–14801, 2024.
- [222] Song Wang, Jiawei Yu, Wentong Li, Hao Shi, Kailun Yang, Junbo Chen, and Jianke Zhu. Label-efficient semantic scene completion with scribble annotations. In Proc. Int. Joint Conf. Artifi. Intell., pages 1398–1406, 2024.
- [223] Song Wang, Gongfan Fang, Lingdong Kong, Xiangtai Li, Jianyun Xu, Sheng Yang, Qiang Li, Jianke Zhu, and Xinchao Wang. PixelThink: Towards efficient chain-of-pixel reasoning. arXiv preprint arXiv:2505.23727, 2025.
- [224] Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. FCOS3D: Fully convolutional one-stage monocular 3D object detection. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 913–922, 2021.
- [225] Wenpeng Wang, Bradford Campbell, and Sirajum Munir. Self-supervised contrastive learning for camera-to-radar knowledge distillation. In Int. Conf. Distributed Comput. in Smart Sys. Internet of Things, pages 154–161. IEEE, 2024.
- [226] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. DriveDreamer: Towards real-world-drive world models for autonomous driving. In Proc. Eur. Conf. Comput. Vis., pages 55–72. Springer, 2024.
- [227] Yuanbin Wang, Shaofei Huang, Yulu Gao, Zhen Wang, Rui Wang, Kehua Sheng, Bo Zhang, and Si Liu. Transferring CLIP’s knowledge into zero-shot point cloud semantic segmentation. In Proc. ACM Int. Conf. Multimedia, pages 3745–3754, 2023.
- [228] Ze Wang, Sihao Ding, Ying Li, Jonas Fenn, Sohini Roychowdhury, Andreas Wallin, Lane Martin, Scott Ryvola, Guillermo Sapiro, and Qiang Qiu. Cirrus: A long-range bi-pattern LiDAR dataset. In Proc. IEEE Int. Conf. Robot. Automat., pages 5744–5750, 2021.
- [229] Zichen Wang, Zhuokun Yao, Jianwei Zhang, Ye Zheng, Zhengyuan Zhang, Shuang Deng, Yajing Liu, and Hao Liu. Focus on your geometry: Exploiting the potential of multi-frame stereo depth estimation pre-training for 3D object detection. In Int. Joint Conf. Neural Networks, pages 1–8, 2024.
- [230] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Adv. Neural Inf. Process. Syst., volume 35, pages 24824–24837, 2022.

- [231] Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, and Wenchao Ding. OccLLaMA: An occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272, 2024.
- [232] Weijie Wei, Fatemeh Karimi Nejadasl, Theo Gevers, and Martin R Oswald. T-MAE: temporal masked autoencoders for point cloud representation learning. In Proc. Eur. Conf. Comput. Vis., pages 178–195. Springer, 2024.
- [233] Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. PARA-Drive: Parallelized architecture for real-time autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 15449–15458, 2024.
- [234] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and James Hays. Argoverse 2: Next generation datasets for self-driving perception and forecasting. In Adv. Neural Inf. Process. Syst., 2021.
- [235] Maciej K Wozniak, Hariprasath Govindarajan, Marvin Klingner, Camille Maurice, Ravi Kiran, and Senthil Yogamani. S3PT: Scene semantics and structure guided clustering to boost self-supervised pre-training for autonomous driving. arXiv preprint arXiv:2410.23085, 2024.
- [236] Sheng Wu, Fei Teng, Hao Shi, Qi Jiang, Kai Luo, Kaiwei Wang, and Kailun Yang. QuaDreamer: Controllable panoramic video generation for quadruped robots. In Conf. Robot Learn., pages 1777–1789. PMLR, 2025.
- [237] Xiongfei Wu, Mingfei Cheng, Qiang Hu, Jianlang Chen, Yuheng Huang, Manabu Okada, Michio Hayashi, Tomoyuki Tsuchiya, Xiaofei Xie, and Lei Ma. Foundation models for autonomous driving system: An initial roadmap. arXiv preprint arXiv:2504.00911, 2025.
- [238] Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. DOTA: A large-scale dataset for object detection in aerial images. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 3974–3983, 2018.
- [239] Aoran Xiao, Jiaxing Huang, Dayan Guan, Xiaoqin Zhang, Shijian Lu, and Ling Shao. Unsupervised point cloud representation learning with deep neural networks: A survey. IEEE Trans. Pattern Anal. Mach. Intell., 45(9): 11321–11339, 2023.
- [240] Pengchuan Xiao, Zhenlei Shao, Steven Hao, Zishuo Zhang, Xiaolin Chai, Judy Jiao, Zesong Li, Jian Wu, Kai Sun, Kun Jiang, et al. PandaSet: Advanced sensor suite dataset for autonomous driving. In IEEE Int. Conf. Intell. Transport. Sys., pages 3095–3101, 2021.
- [241] Zihao Xiao, Longlong Jing, Shangxuan Wu, Alex Zihao Zhu, Jingwei Ji, Chiyu Max Jiang, Wei-Chih Hung, Thomas Funkhouser, Weicheng Kuo, Anelia Angelova, et al. 3D open-vocabulary panoptic segmentation with 2D-3D vision-language distillation. In Proc. Eur. Conf. Comput. Vis., pages 21–38. Springer, 2025.
- [242] Jihong Xie, Xiang Zhou, and Lu Cheng. Edge computing for real-time decision making in autonomous driving: Review of challenges, solutions, and future trends. Int. J. Adv. Comput. Sci. & Appl., 15(7), 2024.
- [243] Saining Xie, Jiatao Gu, Demi Guo, Charles R Qi, Leonidas Guibas, and Or Litany. PointContrast: Unsupervised pre-training for 3D point cloud understanding. In Proc. Eur. Conf. Comput. Vis., pages 574–591. Springer, 2020.
- [244] Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. RoboBEV: Towards robust bird’s eye view perception under corruptions. arXiv preprint arXiv:2304.06719, 2023.
- [245] Shaoyuan Xie, Lingdong Kong, Yuhao Dong, Chonghao Sima, Wenwei Zhang, Qi Alfred Chen, Ziwei Liu, and Liang Pan. Are VLMs ready for autonomous driving? an empirical study from the reliability, data, and metric perspectives. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 6585–6597, 2025.
- [246] Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Benchmarking and improving bird’s eye view perception robustness in autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell., 47(5):3878–3894, 2025.
- [247] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 2945–2954, 2023.
- [248] Runsen Xu, Tai Wang, Wenwei Zhang, Runjian Chen, Jinkun Cao, Jiangmiao Pang, and Dahua Lin. MV-JAR: Masked voxel jigsaw and reconstruction for LiDAR-based self-supervised pre-training. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 13445–13454, 2023.

- [249] Shaoqing Xu, Dingfu Zhou, Jin Fang, Junbo Yin, Zhou Bin, and Liangjun Zhang. FusionPainting: Multimodal fusion with adaptive attention for 3D object detection. In IEEE Int. Conf. Intell. Transport. Sys., pages 3047–3054, 2021.
- [250] Shaoqing Xu, Fang Li, Shengyin Jiang, Ziying Song, Li Liu, and Zhi-xin Yang. GaussianPretrain: A simple unified 3D gaussian representation for visual pre-training in autonomous driving. arXiv preprint arXiv:2411.12452, 2024.
- [251] Weichen Xu, Jian Cao, Tianhao Fu, Ruilong Ren, Zicong Hu, Xixin Cao, and Xing Zhang. Point cloud reconstruction is insufficient to learn 3D representations. In Proc. ACM Int. Conf. Multimedia, pages 8471–8479, 2024.
- [252] Weichen Xu, Tianhao Fu, Jian Cao, Xinyu Zhao, Xinxin Xu, Xixin Cao, and Xing Zhang. Mutual informationdriven self-supervised point cloud pre-training. Knowledge-Based Systems, page 112741, 2024.
- [253] Xiang Xu, Lingdong Kong, Hui Shuai, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, and Qingshan Liu. 4D contrastive superflows are dense 3D representation learners. In Proc. Eur. Conf. Comput. Vis., pages 58–80. Springer, 2024.
- [254] Xiang Xu, Lingdong Kong, Hui Shuai, Liang Pan, Liu Ziwei, and Qingshan Liu. LiMoE: Mixture of LiDAR representation learners from automotive scenes. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 27368–27379, 2025.
- [255] Xiang Xu, Lingdong Kong, Song Wang, Chuanwei Zhou, and Qingshan Liu. Beyond one shot, beyond one perspective: Cross-view and long-horizon distillation for better LiDAR representations. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 25506–25518, 2025.
- [256] Xiang Xu, Ao Liang, Youquan Liu, Linfeng Li, Lingdong Kong, Ziwei Liu, and Qingshan Liu. U4D: Uncertaintyaware 4D world modeling from LiDAR sequences. arXiv preprint arXiv: 2512.02982, 2025.
- [257] Xiaohao Xu, Ye Li, Tianyi Zhang, Jinrong Yang, Matthew Johnson-Roberson, and Xiaonan Huang. Learning shared RGB-D fields: Unified self-supervised pre-training for label-efficient LiDAR-camera 3D perception. arXiv preprint arXiv:2405.17942, 2024.
- [258] Hang Yan, Yongji Li, Luping Wang, and Shichao Chen. Learning omni-dimensional spatio-temporal dependencies for millimeter-wave radar perception. Remote Sensing, 16(22):4256, 2024.
- [259] Xiangchao Yan, Runjian Chen, Bo Zhang, Hancheng Ye, Renqiu Xia, Jiakang Yuan, Hongbin Zhou, Xinyu Cai, Botian Shi, Wenqi Shao, Ping Luo, Yu Qiao, Tao Chen, and Junchi Yan. SPOT: Scalable 3D pre-training via occupancy prediction for learning transferable 3D representations. IEEE Trans. Pattern Anal. Mach. Intell., 47

(11):9609–9625, 2025.

- [260] Xu Yan, Haiming Zhang, Yingjie Cai, Jingming Guo, Weichao Qiu, Bin Gao, Kaiqiang Zhou, Yue Zhao, Huan Jin, Jiantao Gao, et al. Forging vision foundation models for autonomous driving: Challenges, methodologies, and opportunities. arXiv preprint arXiv:2401.08045, 2024.
- [261] Ziyang Yan, Wenzhen Dong, Yihua Shao, Yuhang Lu, Liu Haiyang, Jingwen Liu, Haozhe Wang, Zhe Wang, Yan Wang, Fabio Remondino, et al. RenderWorld: World model with self-supervised 3D label. arXiv preprint arXiv:2409.11356, 2024.
- [262] Hao Yang, Haiyang Wang, Di Dai, and Liwei Wang. PRED: pre-training via semantic rendering on LiDAR point clouds. Adv. Neural Inf. Process. Syst., 36, 2024.
- [263] Honghui Yang, Tong He, Jiaheng Liu, Hua Chen, Boxi Wu, Binbin Lin, Xiaofei He, and Wanli Ouyang. GD-MAE: generative decoder for MAE pre-training on LiDAR point clouds. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 9403–9414, 2023.
- [264] Honghui Yang, Sha Zhang, Di Huang, Xiaoyang Wu, Haoyi Zhu, Tong He, Shixiang Tang, Hengshuang Zhao, Qibo Qiu, Binbin Lin, et al. UniPAD: A universal pre-training paradigm for autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 15238–15250, 2024.
- [265] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 14662–14672, 2024.
- [266] Yan Yang, Liyuan Pan, and Liu Liu. Event camera data pre-training. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 10699–10709, 2023.

- [267] Yan Yang, Liyuan Pan, and Liu Liu. Event camera data dense pre-training. In Proc. Eur. Conf. Comput. Vis., pages 292–310. Springer, 2024.
- [268] Yiming Yang, Hongbin Lin, Yueru Luo, Suzhong Fu, Chao Zheng, Xinrui Yan, Shuqi Mei, Kun Tang, Shuguang Cui, and Zhen Li. FASTopoWM: Fast-slow lane segment topology reasoning with latent world models. arXiv preprint arXiv:2507.23325, 2025.
- [269] Yu Yang, Jianbiao Mei, Yukai Ma, Siliang Du, Wenqing Chen, Yijie Qian, Yuxiang Feng, and Yong Liu. Driving in the occupancy world: Vision-centric 4D occupancy forecasting and planning via world models for autonomous driving. arXiv preprint arXiv:2408.14197, 2024.
- [270] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024.
- [271] Shanliang Yao, Runwei Guan, Zhaodong Wu, Yi Ni, Zile Huang, Ryan Wen Liu, Yong Yue, Weiping Ding, Eng Gee Lim, Hyungjoon Seo, et al. WaterScenes: A multi-task 4D radar-camera fusion dataset and benchmarks for autonomous driving on water surfaces. IEEE Trans. Intell. Transport. Sys., 25(11):16584–16598, 2024.
- [272] Shanliang Yao, Runwei Guan, Zitian Peng, Chenhang Xu, Yilu Shi, Yong Yue, Eng Gee Lim, Hyungjoon Seo, Ka Lok Man, Xiaohui Zhu, et al. Exploring radar data representations in autonomous driving: A comprehensive review. IEEE Trans. Intell. Transport. Sys., 26(6):7401–7425, 2025.
- [273] Hui Ye, Rajshekhar Sunderraman, and Shihao Ji. UAV3D: A large-scale 3D perception benchmark for unmanned aerial vehicles. arXiv preprint arXiv:2410.11125, 2024.
- [274] Junbo Yin, Dingfu Zhou, Liangjun Zhang, Jin Fang, Cheng-Zhong Xu, Jianbing Shen, and Wenguan Wang. ProposalContrast: Unsupervised pre-training for LiDAR-based 3D object detection. In Proc. Eur. Conf. Comput. Vis., pages 17–33. Springer, 2022.
- [275] Tianwei Yin, Xingyi Zhou, and Philipp Krähenbühl. Multimodal virtual point 3D detection. In Adv. Neural Inf. Process. Syst., volume 34, pages 16494–16507, 2021.
- [276] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3D: Towards zero-shot metric 3D prediction from a single image. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 9043–9053, 2023.
- [277] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In Adv. Neural Inf. Process. Syst., volume 27, pages 3320–3328, 2014.
- [278] Hanxun Yu, Wentong Li, Song Wang, Junbo Chen, and Jianke Zhu. Inst3D-LMM: Instance-aware 3D scene understanding with multi-modal instruction tuning. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 14147–14157, 2025.
- [279] Zhu Yu, Bowen Pang, Lizhe Liu, Runmin Zhang, Qihao Peng, Maochun Luo, Sheng Yang, Mingxia Chen, Si-Yuan Cao, and Hui-Liang Shen. Language driven occupancy prediction. arXiv preprint arXiv:2411.16072, 2024.
- [280] Jiakang Yuan, Bo Zhang, Xiangchao Yan, Botian Shi, Tao Chen, Yikang Li, and Yu Qiao. AD-PT: Autonomous driving pre-training with large-scale point cloud dataset. Adv. Neural Inf. Process. Syst., 36, 2023.
- [281] Mahmut Yurt, Xin Ye, Yunsheng Ma, Jingru Luo, Abhirup Mallik, John Pauly, Burhaneddin Yaman, and Liu Ren. LTDA-Drive: LLMs-guided generative models based long-tail data augmentation for autonomous driving. arXiv preprint arXiv:2505.18198, 2025.
- [282] Oliver Zendel, Markus Murschitz, Marcel Zeilinger, Daniel Steininger, Sara Abbasi, and Csaba Beleznai. RailSem19: A dataset for semantic rail scene understanding. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Worksh., 2019.
- [283] Yihan Zeng, Chenhan Jiang, Jiageng Mao, Jianhua Han, Chaoqiang Ye, Qingqiu Huang, Dit-Yan Yeung, Zhen Yang, Xiaodan Liang, and Hang Xu. CLIP2: Contrastive language-image-point pretraining from real-world point cloud data. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 15244–15253, 2023.
- [284] Mingliang Zhai, Cheng Li, Zengyuan Guo, Ningrui Yang, Xiameng Qin, Sanyuan Zhao, Junyu Han, Ji Tao, Yuwei Wu, and Yunde Jia. World knowledge-enhanced reasoning using instruction-guided interactor in autonomous driving. In Proc. AAAI Conf. Artifi. Intell., volume 39, pages 9842–9850, 2025.
- [285] Chubin Zhang, Juncheng Yan, Yi Wei, Jiaxin Li, Li Liu, Yansong Tang, Yueqi Duan, and Jiwen Lu. OccNeRF: Self-supervised multi-camera occupancy prediction with neural radiance fields. arXiv preprint arXiv:2312.09243, 2023.

- [286] Fengyi Zhang, Huitong Yang, Zheng Zhang, Zi Huang, and Yadan Luo. TT-Occ: Test-time compute for self-supervised occupancy via spatio-temporal gaussian splatting. arXiv preprint arXiv:2503.08485, 2025.
- [287] Haijun Zhang, Mingshan Sun, Qun Li, Linlin Liu, Ming Liu, and Yuzhu Ji. An empirical study of multi-scale object detection in high resolution UAV images. Neurocomputing, 421:173–182, 2021.
- [288] Haiming Zhang, Wending Zhou, Yiyao Zhu, Xu Yan, Jiantao Gao, Dongfeng Bai, Yingjie Cai, Bingbing Liu, Shuguang Cui, and Zhen Li. VisionPAD: A vision-centric pre-training paradigm for autonomous driving. arXiv preprint arXiv:2411.14716, 2024.
- [289] Haiming Zhang, Yiyao Zhu, Wending Zhou, Xu Yan, Yingjie Cai, Bingbing Liu, Shuguang Cui, and Zhen Li. SQS: Enhancing sparse perception models via query-based splatting in autonomous driving. arXiv preprint arXiv:2509.16588, 2025.
- [290] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 1020–1031, 2023.
- [291] Junbo Zhang, Runpei Dong, and Kaisheng Ma. CLIP-FO3D: Learning free open-world 3D scene representations from 2D dense clip. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 2048–2059, 2023.
- [292] Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Copilot4D: Learning unsupervised world models for autonomous driving via discrete diffusion. In Int. Conf. Learn. Represent., 2024.
- [293] Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, and Hongsheng Li. Learning 3D representations from 2D pre-trained models via image-to-point masked autoencoders. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 21769–21780, 2023.
- [294] Sha Zhang, Jiajun Deng, Lei Bai, Houqiang Li, Wanli Ouyang, and Yanyong Zhang. HVDistill: Transferring knowledge from images to point clouds via unsupervised hybrid-view distillation. Int. J. Comput. Vis., pages 1–15, 2024.
- [295] Xiaoshuai Zhang, Zhicheng Wang, Howard Zhou, Soham Ghosh, Danushen Gnanapragasam, Varun Jampani, Hao Su, and Leonidas Guibas. ConDense: Consistent 2D/3D pre-training for dense and sparse features from multi-view images. In Proc. Eur. Conf. Comput. Vis., pages 19–38. Springer, 2024.
- [296] Xinyu Zhang, Li Wang, Jian Chen, Cheng Fang, Lei Yang, Ziying Song, Guangqi Yang, Yichen Wang, Xiaofei Zhang, and Jun Li. Dual radar: A multi-modal dataset with dual 4D radar for autononous driving. arXiv preprint arXiv:2310.07602, 2023.
- [297] Yifan Zhang and Junhui Hou. Fine-grained image-to-LiDAR contrastive distillation with visual foundation models. arXiv preprint arXiv:2405.14271, 2024.
- [298] Yumeng Zhang, Shi Gong, Kaixin Xiong, Xiaoqing Ye, Xiao Tan, Fan Wang, Jizhou Huang, Hua Wu, and Haifeng Wang. BEVWorld: A multimodal world model for autonomous driving via unified bev latent space. arXiv preprint arXiv:2407.05679, 2024.
- [299] Zaiwei Zhang, Rohit Girdhar, Armand Joulin, and Ishan Misra. Self-supervised pretraining of 3D features on any point-cloud. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 10252–10263, 2021.
- [300] Lingjun Zhao, Yandong Luo, James Hay, and Lu Gan. ShelfGaussian: Shelf-supervised open-vocabulary gaussian-based 3D scene understanding. arXiv preprint arXiv:2512.03370, 2025.
- [301] Seth Z Zhao, Hao Xiang, Chenfeng Xu, Xin Xia, Bolei Zhou, and Jiaqi Ma. CooPre: Cooperative pretraining for V2X cooperative perception. arXiv preprint arXiv:2408.11241, 2024.
- [302] Xin Zhao, Shiyu Hu, Yipei Wang, Jing Zhang, Yimin Hu, Rongshuai Liu, Haibin Ling, Yin Li, Renshu Li, Kun Liu, et al. BioDrone: A bionic drone-based single object tracking benchmark for robust vision. Int. J. Comput. Vis., 132(5):1659–1684, 2024.
- [303] Jilai Zheng, Pin Tang, Zhongdao Wang, Guoqing Wang, Xiangxuan Ren, Bailan Feng, and Chao Ma. VEON: Vocabulary-enhanced occupancy prediction. In Proc. Eur. Conf. Comput. Vis., pages 92–108. Springer, 2024.
- [304] Lianqing Zheng, Long Yang, Qunshu Lin, Wenjin Ai, Minghao Liu, Shouyi Lu, Jianan Liu, Hongze Ren, Jingyue Mo, Xiaokai Bai, et al. OmniHD-Scenes: A next-generation multimodal dataset for autonomous driving. arXiv preprint arXiv:2412.10734, 2024.

- [305] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. OccWorld: Learning a 3D occupancy world model for autonomous driving. arXiv preprint arXiv:2311.16038, 2023.
- [306] Yupeng Zheng, Pengxuan Yang, Zebin Xing, Qichao Zhang, Yuhang Zheng, Yinfeng Gao, Pengfei Li, Teng Zhang, Zhongpu Xia, Peng Jia, XianPeng Lang, and Dongbin Zhao. World4Drive: End-to-end autonomous driving via intention-aware physical latent world model. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 28632–28642, 2025.
- [307] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from CLIP. In Proc. Eur. Conf. Comput. Vis., pages 696–712. Springer, 2022.
- [308] Jiazhou Zhou, Xu Zheng, Yuanhuiyi Lyu, and Lin Wang. EventBind: Learning a unified representation to bind them all for event-based open-world understanding. In Proc. Eur. Conf. Comput. Vis., pages 477–494. Springer, 2024.
- [309] Mingle Zhou, Rui Xing, Delong Han, Zhiyong Qi, and Gang Li. PDT: UAV target detection dataset for pests and diseases tree. In Proc. Eur. Conf. Comput. Vis., pages 56–72. Springer, 2025.
- [310] Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. HERMES: A unified self-driving world model for simultaneous 3D scene understanding and generation. arXiv preprint arXiv:2501.14729, 2025.
- [311] Yin Zhou and Oncel Tuzel. VoxelNet: End-to-end learning for point cloud based 3D object detection. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 4490–4499, 2018.
- [312] Zewei Zhou, Seth Z. Zhao, Tianhui Cai, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. TurboTrain: Towards efficient and balanced multi-task learning for multi-agent perception and prediction. In Proc. IEEE/CVF Int. Conf. Comput. Vis., pages 4391–4402, 2025.
- [313] ZXiaoyu Zhou, Jingqi Wang, Yongtao Wang, Yufei Wei, Nan Dong, and Ming-Hsuan Yang. AutoOcc: Automatic open-ended semantic occupancy annotation via vision-language guided gaussian splatting. arXiv preprint arXiv:2502.04981, 2025.
- [314] Haoran Zhu, Haoze He, Anna Choromanska, Satish Ravindran, Binbin Shi, and Lihui Chen. Multi-view radar autoencoder for self-supervised automotive radar representation learning. In IEEE Intell. Veh. Symposium, pages 1601–1608, 2024.
- [315] Haoran Zhu, Zhenyuan Dong, Kristi Topollai, and Anna Choromanska. AD-L-JEPA: Self-supervised spatial world models with joint embedding predictive architecture for autonomous driving with LiDAR data. arXiv preprint arXiv:2501.04969, 2025.
- [316] Haoyi Zhu, Honghui Yang, Xiaoyang Wu, Di Huang, Sha Zhang, Xianglong He, Tong He, Hengshuang Zhao, Chunhua Shen, Yu Qiao, et al. Ponderv2: Pave the way for 3D foundation model with a universal pre-training paradigm. arXiv preprint arXiv:2310.08586, 2023.
- [317] Pengfei Zhu, Longyin Wen, Dawei Du, Xiao Bian, Heng Fan, Qinghua Hu, and Haibin Ling. Detection and tracking meet drones challenge. IEEE Trans. Pattern Anal. Mach. Intell., 44(11):7380–7399, 2021.
- [318] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, et al. Is Sora a world simulator? a comprehensive survey on general world models and beyond. arXiv preprint arXiv:2405.03520, 2024.
- [319] Long Zhuang and Tiezhen Jiang. Pre-training for mmWave radar object detection through masked image modeling. In Int. Conf. Image Process., Comput. Vis. Machine Learn., pages 546–550. IEEE, 2023.
- [320] Long Zhuang, Tiezhen Jiang, Jianhua Wang, Qi An, Kai Xiao, and Anqi Wang. Effective mmwave radar object detection pre-training based on masked image modeling. IEEE Sensors J., 2023.
- [321] Yongshuo Zong, Oisin Mac Aodha, and Timothy Hospedales. Self-supervised multimodal learning: A survey. arXiv preprint arXiv:2304.01008, 2023.
- [322] Jialv Zou, Bencheng Liao, Qian Zhang, Wenyu Liu, and Xinggang Wang. MIM4D: Masked modeling with multi-view video for autonomous driving representation learning. arXiv preprint arXiv:2403.08760, 2024.
- [323] Jian Zou, Tianyu Huang, Guanglei Yang, Zhenhua Guo, Tao Luo, Chun-Mei Feng, and Wangmeng Zuo. UniM2AE: Multi-modal masked autoencoders with unified 3D representation for 3D perception in autonomous driving. In Proc. Eur. Conf. Comput. Vis., pages 296–313. Springer, 2024.

- [324] Pufan Zou, Shijia Zhao, Weijie Huang, Qiming Xia, Chenglu Wen, Wei Li, and Cheng Wang. AdaCo: Overcoming visual foundation model noise in 3D semantic segmentation via adaptive label correction. arXiv preprint arXiv:2412.18255, 2024.
- [325] Arij Zouaoui, Ankur Mahtani, Mohamed Amine Hadded, Sébastien Ambellouis, Jacques Boonaert, and Hazem Wannous. RailSet: A unique dataset for railway anomaly detection. In Proc. IEEE Int. Conf. Image Process. Appl. Sys., pages 1–6, 2022.

