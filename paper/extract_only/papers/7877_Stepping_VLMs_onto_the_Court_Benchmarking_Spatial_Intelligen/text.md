[Figure 1]

[Figure 2]

[Figure 3]

# arXiv:2603.09896v1[cs.CV]10Mar2026

## Stepping VLMs onto the Court: Benchmarking Spatial Intelligence in Sports

Yuchen Yang1,2,∗, Yuqing Shao4,2,∗, Duxiu Huang5,∗, Linfeng Dong6,2,∗, Yifei Liu7,2, Suixin Tang4, Xiang Zhou4, Yuanyuan Gao8,2, Wei Wang2, Yue Zhou9, Xue Yang3, Yanfeng Wang3, Xiao Sun2,

Zhihang Zhong3,

∗Equal Contribution; Corresponding Authors

1Fudan University, 2Shanghai Artificial Intelligence Laboratory, 3Shanghai Jiao Tong University, 4East China University of Science and Technology, 5Southeast University, 6Zhejiang University, 7Beihang University, 8Hong Kong University of Science and Technology, 9East China Normal University

Sports have long attracted broad attention as they push the limits of human physical and cognitive capabilities. Amid growing interest in spatial intelligence for vision-language models (VLMs), sports provide a natural testbed for understanding high-intensity human motion and dynamic object interactions. To this end, we present CourtSI, the first large-scale spatial intelligence dataset tailored to sports scenarios. CourtSI contains over 1M QA pairs, organized under a holistic taxonomy that systematically covers spatial counting, distance measurement, localization, and relational reasoning, across representative net sports including badminton, tennis, and table tennis. Leveraging well-defined court geometry as metric anchors, we develop a semi-automatic data engine to reconstruct sports scenes, enabling scalable curation of CourtSI. In addition, we introduce CourtSI-Bench, a high-quality evaluation benchmark comprising 3,686 QA pairs with rigorous human verification. We evaluate 25 proprietary and open-source VLMs on CourtSI-Bench, revealing a remaining human–AI performance gap and limited generalization from existing spatial intelligence benchmarks. These findings indicate that sports scenarios expose limitations in spatial intelligence capabilities captured by existing benchmarks. Further, fine-tuning Qwen3-VL-8B on CourtSI improves accuracy on CourtSI-Bench by 23.5 percentage points. The adapted model also generalizes effectively to CourtSI-Ext, an evaluation set built on a similar but unseen sport, and demonstrates enhanced spatial-aware commentary generation. Together, these findings demonstrate that CourtSI provides a scalable pathway toward advancing spatial intelligence of VLMs in sports.

Website: https://visionary-laboratory.github.io/CourtSI Code: https://github.com/Visionary-Laboratory/CourtSI Email: zhongzhihang95@gmail.com

[Figure 4]

Focus SpatialCapabilities CourtSI-Bench

[Figure 5]

[Figure 6]

Extra Evaluation

<player> <ball> <player> <court> <ball> <player> <player> <player> <ball>

<camera>

[Figure 7]

3,686 QA Pairs

Distance Measurement

[Figure 8]

<camera>

[Figure 9]

verification

Unseen Sport

[Figure 10]

[Figure 11]

[Figure 12]

### CourtSI

<court>

Human-AI Gap

Spatial-Aware Commentary

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

1M QA Pairs

[Figure 17]

[Figure 18]

[Figure 19]

Relational Reasoning

[Figure 20]

25 VLMs

###### ...

Counting

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

Semi-Automatic Data Engine

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

- Figure 1 | Overview. We introduce a semi-automatic data engine that reconstructs sports scenes in 3D with court, player, and ball locations. Built upon this pipeline, we present CourtSI and CourtSI-Bench, the first large-scale spatial intelligence dataset and benchmark for sports scenarios. In addition, we provide extra evaluation protocols to validate applicability on an unseen sport and spatial-aware commentary.

#### 1. Introduction

As Vision-Language Models (VLMs) continue to achieve strong performance in semantic understanding and 2D visual reasoning, researchers have begun to explore VLMs’ ability to perceive and reason about the 3D world. This shift has led to the emergence of spatial intelligence [1] as a focused research direction, aiming to equip models with foundational capabilities required for effective interaction with the physical world in the pursuit of AGI.

Current efforts [2, 3, 4, 5, 6, 7] primarily focus on boosting the spatial understanding of modern VLMs, along with developing diverse benchmarks for evaluation across multiple spatial dimensions. However, the datasets proposed in these works concentrate on static scenes and rigid objects, resulting in a relatively narrow coverage of spatial subjects. In contrast, humans, critical subjects in real-world environments characterized by non-rigid deformations and articulated body constraints, remain underexplored. Sports scenarios, characterized by high-intensity human motion and dynamic object interactions, provide a natural but challenging testbed for investigating spatial intelligence at a fine-grained level.

Motivated by the nature of sports scenarios, as illustrated in fig. 1, we present CourtSI and CourtSI-Bench, the first large-scale dataset and benchmark dedicated to spatial intelligence in sports. Our work introduces sports as a new and challenging scenario for spatial intelligence, while simultaneously extending existing VLM benchmarks for sports understanding [8, 9, 10, 11] beyond activity-centric to fine-grained spatial reasoning.

To obtain data at scale, we design a semi-automatic reconstruction data engine that recovers 3D scene information from monocular images. Unlike general in-the-wild environments, sports

courts provide well-defined geometric structures with fixed metric scales. Leveraging this property, we jointly optimize camera intrinsics and extrinsics from court corner correspondences using a Perspective-n-Point (PnP) solver, thereby establishing a unified world coordinate system anchored to the court geometry. Locating players and balls into this geometry-aligned space ensures consistent and physically grounded spatial reasoning across scenes. Specifically, for players, we adopt PromptHMR [12] to recover human meshes in the SMPL-X [13] representation within the camera coordinate system, capturing fine-grained pose and shape information. Ball positions in the images are manually annotated. We observe that existing monocular depth estimation methods fail to produce reliable metric reconstruction. Instead, we manually estimate object heights relative to the court plane to enable accurate camera-to-world transformation. With strict 3D quality control over a multi-view set, our pipeline achieves 𝑐𝑚-level accuracy, providing a reliable foundation for subsequent data curation.

Building upon the reconstruction engine, we construct CourtSI by converting 3D sports states into large-scale question-answer (QA) pairs under a holistic taxonomy. Specifically, we filter data from the well-organized sports dataset, RacketVision [14], which includes badminton, tennis, and table tennis in broadcast views. The camera viewpoints in broadcast footage mitigate unnecessary viewpoint variance, allowing models to focus on learning spatial relationships. We design QA templates that systematically cover (i) spatial counting, (ii) distance measurement, (iii) localization, and (iv) relational reasoning, instantiated over players, balls, and the court. Answers are automatically derived from the reconstructed 3D states, resulting in over 1M QA pairs for training. To enable rigorous evaluation, we further curate CourtSI-Bench, comprising 3,686 high-quality QA pairs with careful human verification.

We comprehensively evaluate 25 state-of-the-art proprietary and open-source VLMs on CourtSI-Bench. Even the strongest baseline remains a gap behind humans, particularly on distance measurement tasks. Furthermore, models trained on existing spatial intelligence benchmarks generalize poorly to CourtSI-Bench, suggesting that current datasets fail to sufficiently capture the challenges posed by dynamic sports scenarios. To assess the training utility of CourtSI, we conduct supervised fine-tuning of Qwen3-VL-8B [15], improving accuracy by 23.5 percentage points on CourtSI-Bench, with particularly significant gains in distance measurement. To expand the evaluation, we introduce CourtSI-Ext, a benchmark constructed from pickleball, a similar yet unseen net sport. The fine-tuned model demonstrates strong generalization to this new sport, indicating that CourtSI fosters transferable spatial reasoning capabilities. Additionally, we explore spatial-aware commentary generation by prompting VLMs to incorporate spatial relationships into commentary for CourtSI-Bench samples. User studies demonstrate improved spatial understanding while preserving overall linguistic quality after fine-tuning on CourtSI. Collectively, these results validate the effectiveness of CourtSI and highlight its potential as a scalable pathway toward advancing spatial intelligence of VLMs in sports.

Our contributions are summarized in threefold:

- • We introduce CourtSI and CourtSI-Bench, the first large-scale spatial intelligence dataset and benchmark in sports, establishing a testbed for fine-grained, human-centric spatial reasoning beyond static object-centric datasets.
- • We develop a semi-automatic data engine that recovers accurate 3D scene states from broadcast net sports, enabling scalable data curation.
- • We conduct a comprehensive evaluation of 25 state-of-the-art VLMs and examine the impact of fine-tuning, along with cross-sport generalization on CourtSI-Ext and spatialaware commentary generation.

#### 2. Related Work

- 2.1. Spatial Intelligence of VLMs

Along with the development of Vision-Language Models (VLMs), researchers have increasingly questioned their ability to reason about relationships of perceived 3D objects when trained primarily on web-scale data on the image plane [1]. This limitation has motivated the emergence of spatial intelligence, a term used to characterize models’ capabilities in 3D spatial reasoning. Such capabilities are widely regarded as the foundation of reliable interaction with the physical world, in the broader pursuit of general intelligence [16, 17, 18].

To better characterize and advance these capabilities, the research community has developed both dedicated benchmarks and specialized approaches. From a benchmarking perspective, VSI [2] collects data with camera browsing inside indoor environments, requiring models to perceive, memorize, and recall spatial layouts. Subsequent works extend evaluation across different dimensions of spatial understanding [19, 20, 21, 3, 22]. MindCube [7] focuses on sparse-view reasoning, while ViewSpatial [23] emphasizes allocentric spatial reasoning. The underlying data sources have also expanded from structured indoor datasets such as ScanNet [24] to more diverse and less constrained 3D collections [25, 26]. From a methodological perspective, a common approach is to enhance spatial intelligence through supervised fine-tuning or reinforcement learning strategies [27, 28, 29, 30, 31, 6, 32]. In addition, several works [5, 33, 34, 35, 36] improve spatial reasoning by modifying the visual backbone, incorporating stronger geometric priors. In contrast, our work focuses on sports scenarios, with a particular emphasis on human-centric spatial reasoning.

- 2.2. Sport Understanding

Sports understanding has long been an active research area, encompassing tasks such as action recognition [37, 38, 39] and analysis [40, 41, 42]. The advent of language models has substantially accelerated progress in this domain via stronger end-to-end reasoning capability, especially in captioning and commentary generation [43, 44, 45]. More recently, unified benchmarks [46, 8, 9, 10, 47, 48] have been proposed to integrate diverse sports-related tasks under a common evaluation framework in the Question-Answer format. Existing efforts remain largely actioncentric, primarily focusing on basic sport rules or high-level semantics in events. In contrast, our work shifts the focus toward spatial intelligence in sports, emphasizing metrically grounded and human-centric spatial reasoning beyond conventional activity-based evaluation.

- 3. CourtSI Dataset

In this section, we first present the semi-automatic reconstruction data engine that enables scalable dataset construction. We then describe the CourtSI and CourtSI-Bench, which are built upon explicit 3D scene reconstruction.

##### 3.1. Data Engine

To construct spatial intelligence QA pairs from sports images, we adopt an explicit pipeline that first reconstructs the 3D scene and then formulates questions and derives answers based on the recovered spatial states. This design enables scalable QA generation, as answers can be computed through deterministic rules grounded in the reconstructed 3D information. In practice, the primary challenge lies in accurate scene reconstruction, particularly in estimating

Raw Image

Scene Reconstruction

[Figure 63]

[Figure 64]

Player Mesh Recovery

[Figure 65]

[Figure 66]

table tennis

PromptHMR

[Figure 67]

[Figure 68]

[Figure 69]

tennis

[Figure 70]

| | |
|---|---|
| | |
| | |

SAM3

Bounding Box

Height

[Figure 71]

badminton

Correction

[Figure 72]

[Figure 73]

Court Annotation Ball Annotation

Camera

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

ground point

ball 2D location

Param.

(metric-aware)

height point

projection line

ball ground location

PnP Solver

[Figure 78]

human-involved

- Figure 2 | Overview of the data engine. It consists of court annotation for metric-aware camera parameter estimation, ball annotation, and player mesh recovery. By leveraging court geometry and incorporating human-in-the-loop supervision, the system enables accurate and worldgrounded reconstruction in sports scenarios.

camera parameters at metric scale and recovering reliable depth for players and balls. We investigate state-of-the-art monocular methods, including WildCamera [49] and DepthAny-

thingV3 [50], but find them insufficiently robust (section A.2 for detailed comparisons). Unlike previous benchmarks [20, 5, 1, 26], as illustrated in fig. 2, we develop a human-involved pipeline that exploits court geometry for reliable metric reconstruction. The pipeline consists of the following components:

Court Annotation. Sports courts follow standardized geometric layouts, where the real-world dimensions of key structures (e.g., boundary lines and net height) are fixed for each sport. This property allows us to determine the 3D coordinates of predefined court keypoints in a metric world space. We manually annotate corresponding 2D court keypoints in images, including four ground corner points and two height points on the net. Given these 2D–3D correspondences, camera parameter calibration naturally becomes a Perspective-n-Point (PnP) problem, in which camera intrinsics and extrinsics are metric-accurately optimized via a PnP solver. This design defines a unified world coordinate system anchored to the court, while the additional height points on the net stabilize focal length estimation for more reliable reconstruction. For subsequent spatial intelligence learning, the resulting coordinate system standardizes spatial references across samples, reducing cross-scene variability and enabling consistent localization.

Ball Annotation. The ball is typically small, making it difficult for monocular depth estimation models to capture reliably. Moreover, as previously discussed, these models generally lack metric-scale accuracy. However, as a critical object in sports scenes, precise localization of the ball is essential. Inspired by [51], we design a tool that converts depth estimation into ground projection estimation, which is more intuitive for human annotators. With known camera parameters, a 2D pixel p corresponds to a 3D ray in world coordinates, parameterized as:

X(𝜆) = −R𝑇t + 𝜆R𝑇K−1p, 𝜆 > 0, (1)

where K denotes the camera intrinsics, and R,t are the extrinsics. 𝜆 is the depth parameter that varies along the ray. The projection line with the court plane 𝑍 = 0 intersection is obtained by solving

𝑍(𝜆) = 0. (2)

Based on this, annotators are instructed to click the 2D position of the ball and its corresponding ground projection along an assistive projection line rendered in the image. Then the depth parameter 𝜆 of the original ball pixel can be analytically solved, allowing us to recover the 3D location of the ball.

Player Mesh Recovery. We adopt the state-of-the-art human mesh recovery method PromptHMR [12] to estimate SMPL-X [13] parameters in the camera coordinate system. The model takes player bounding boxes and camera parameters as input to produce plausible human pose and shape reconstructions. To obtain reliable bounding boxes, we employ SAM3 [52] with text prompts and manually refine incorrect detections. However, we observe that the reconstructed human meshes frequently exhibit inaccurate depth estimation (e.g., foot penetration or floating). Therefore, we adopt a strategy similar to ball annotation, by annotating the height of the lowest mesh vertex. The entire mesh is then re-aligned to the correct depth using a perspective transformation based on the annotation.

As introduced above, a sports scene can be reconstructed in a world-grounded manner using our data engine. Please refer to section A.1 for additional details.

##### 3.2. Dataset Curation

Data Preparation. We build our dataset and benchmark upon broadcast-view images collected from RacketVision [14], a large-scale benchmark containing 1,672 professional net sports clips, including badminton, tennis, and table tennis. To ensure data quality, we first filter out frames with extreme viewing angles and then apply our data engine to reconstruct 3D scenes from the remaining.

Question-Answer Generation. QA pairs are automatically constructed using predefined question templates together with the corresponding 3D reconstruction outputs. As illustrated in fig. 3, we organize the QA pairs under a unified taxonomy comprising four categories: spatial counting, distance measurement, localization, and relational reasoning. The questions target core sports entities, including the ball, players, and the court, across camera and world views. In addition to semantic categorization, the QA pairs cover numerical and multiple-choice questions (MCQs).

To enhance question diversity, we design multiple templates for each question category, resulting in a total of 94 templates. Following [2], each question is accompanied by a general description and an example of the expected answer format to provide clear instructions. Details are provided in section B.1.

The generated QA pairs exhibit the following characteristics: (i) Metric-aware. Since accurate

- 3D positions of players and the ball are available, precise metric distance measurement can be performed in real-world units. (ii) Human-centric. Leveraging recovered human meshes, we formulate fine-grained body-part-level questions. Examples include locating a player’s foot or measuring inter-player distance using the pelvis as a reference point, which is commonly treated as the human body center in biomechanics. Both egocentric and allocentric perspectives

[Figure 79]

[Figure 80]

[Figure 81]

Ball-Player

[Figure 82]

Cam.-Obj.

###### MCQ

[Figure 83]

Numerical

| | |
|---|---|
| | |

Which player is closest to the ball?

Calculate the 3D distance between the camera and the <pelvis> of <Player X> / <ball>?

ego. / allo.

From the <camera> / <Player X> perspective, the ball is on which side of the <Player X>?

[Figure 84]

Obj.-Line

[Figure 85]

[Figure 86]

Input: This is a snapshot from a <badminton> match. [Coordinate Descriptions] [Question] [Answer Format]

###### Numerical

###### Cam.-Player

[Figure 87]

[Figure 88]

MCQ

What is perpendicular distance of <Player X>’s <pelvis> to the <near service line>?

Which player is closest to the camera?

[Figure 89]

Player-Zone

[Figure 90]

[Figure 91]

[Figure 92]

Obj.-Obj.

###### Ball Cnt.

MCQ

[Figure 93]

[Figure 94]

Numerical

MCQ

In which zone of the court is the <Player X> located?

What is the distance between the <ball> and the <right hand> of <Player X> ?

Is the ball visible?

[Figure 95]

###### Player-Player

[Figure 96]

[Figure 97]

###### MCQ

###### Player Cnt.

[Figure 98]

[Figure 99]

MCQ

Which player is closest to the <Player X> ?

###### Loc.

[Figure 100]

ego. / allo.

Numerical

How many players are visible on the court?

From the <camera> / <Player X> perspective, the <Player Y> is on which side of the <Player X> ?

Locate the <left foot> of <Player X> within the defined 3D coordinate system.

[Figure 101]

[Figure 102]

Height

###### Player-Line

[Figure 103]

[Figure 104]

[Figure 105]

Numerical

###### Ball-Zone

MCQ

[Figure 106]

MCQ

How high above the court is the <head> of <Player X>

Which player is closest to the <left doubles sideline>?

Is the <ball> above the <net>?

- Figure 3 | Taxonomy and examples of CourtSI. The questions are categorized into: spatial counting , distance measurement , localization , and relational reasoning . Cnt. de-

notes counting. Obj. refers to object, including the ball and players. Cam. denotes camera. Ego. and Allo. denote to ego-centric and allo-centric views.

%

[Figure 107]

[Figure 108]

CourtSI Distribution CourtSI-Bench Distribution

Distance Measurement

Cam. - Obj. Height

Obj. - Line Obj. - Obj.

Spatial Counting

Ball Player

Localization

Obj.

Relational Reasoning

Ball-Zone Ball-Player Cam. -Player Player-Zone Player-Player Player-Line

CourtSI 76k CourtSI 51k

CourtSI 102k CourtSI 179k

7.5 5.1

10.1 17.7

CourtSI-Bench 277 CourtSI-Bench 229 CourtSI-Bench 317 CourtSI-Bench 663

7.5 6.2 8.6 18.0

% % % %

CourtSI 23k CourtSI 23k

2.3 2.3

% %

CourtSI-Bench 028 CourtSI-Bench 034

0.8 0.9

% %

CourtSI 102k 10.1 CourtSI-Bench 368 10.0%

CourtSI 62k 6.1 CourtSI-Bench 255 6.9 % CourtSI 72k 7.2 CourtSI-Bench 297 8.1 % CourtSI 58k 5.8 CourtSI-Bench 248 6.7 % CourtSI 29k 2.9 CourtSI-Bench 082 2.2 % CourtSI 105k 10.4 CourtSI-Bench 393 10.7% CourtSI 127k 12.6

% % % % % % CourtSI-Bench 495 13.4%

Total Total

1M 3,686

% % % %

- Figure 4 | Distribution of CourtSI and CourtSI-Bench. Obj. refers to object, including the ball and players. Cam. denotes camera.

are involved, and all answers are generated automatically based on directional cues from the human mesh.

We construct CourtSI with 1,008,941 QA pairs generated from 52,481 images spanning 1,057 unique scenes. In addition, we introduce CourtSI-Bench as a dedicated benchmark, comprising 3,686 QA pairs sampled from 1,988 images across 382 distinct scenes. The dataset and benchmark have no scene overlap, preventing potential information leakage.

The distribution of CourtSI and CourtSI-Bench is illustrated in fig. 4. We carefully balance the categories by considering both their practical importance in sports scenarios and their relative difficulty. For CourtSI-Bench, we maintain a relatively balanced distribution of items across different sports to ensure reliable evaluation, as detailed in section B.4.

- Table 1 | Quantitative error analysis of the data engine. MPJPE denotes Mean Per Joint Position Error for human skeletons.

##### Camera Ball Player

𝑓𝑥 𝑓𝑦 𝑋 𝑌 𝑍 Pelvis MPJPE 2.2% 2.4% 22cm 9cm 9cm 23cm 17cm

Quality Control. To evaluate the reliability of CourtSI and CourtSI-Bench, we first conduct a quality assessment of the data produced by our data engine. Since ground-truth 3D annotations are unavailable for monocular broadcast videos, we instead leverage a purpose-built multiview dataset collected by our team, capturing professional matches with camera configurations similar to the source data in RacketVision. This dataset contains a total of 6,505 frames for each synchronized view. We use chessboard calibration for camera parameters and apply triangulation to obtain 3D location from annotated 2D ball and player keypoints (details are provided in section A.2). As shown in table 1, the focal length estimation error is approximately 2%, while both ball and player localization errors remain at the centimeter level. These results indicate that our data engine produces plausible world-grounded reconstructions. Furthermore, the errors are set as a reference for evaluation. For distance measurement, predictions with errors below a predefined threshold are considered correct.

For CourtSI-Bench, we additionally conduct human verification. Two annotators independently review all QA pairs with access to visualizations of the reconstructed scenes. This allows them to identify potential reconstruction failures that may lead to incorrect answers. Annotators assess the correctness of each QA pair, and any pair flagged by either annotator is removed. The process acts as a post-validation for the data engine, ensuring that occasional reconstruction failures do not compromise the overall QA data quality in CourtSI-Bench.

#### 4. Experiment

4.1. Evaluation Setup

Baseline Models. We conduct a comprehensive evaluation of 25 state-of-the-art vision-language models (VLMs), spanning diverse model families and parameter scales. For proprietary models, we include GPT-5.2, Gemini-3-Pro, Seed1.8, Claude-Sonnet4.5, Grok4, and Qwen3-Max. For open-source models, we evaluate the Qwen3-VL series [15], InternVL3.5 series [53], Kimi-VL [54], and the LLaVA-OneVision series [55, 56]. In addition, we benchmark models fine-tuned on prior spatial intelligence datasets, including SpaceR [29], VST [30], SpatialLadder [27], SenseNovaSI [6], and Cambrain-S [31], together with their corresponding base models [57, 58]. Human performance is reported as a reference for the benchmark. Finally, to assess the task-specific learning potential of CourtSI, we further conduct supervised fine-tuning (SFT) on Qwen3-VL-8B. The model is trained for one epoch using a global batch size of 2048 and a learning rate of

###### 5 × 10−6 in LLaMA Factory environment [59]. Please refer to section C for more details.

Evaluation Metrics. Following VSI [2], we use Accuracy based on exact matching as the main metric. For numerical answer tasks in distance measurement and localization, we report Threshold Mean Relative Accuracy(T-MRA) to allow for a certain error:

- Table 2 | Quantitative results on CourtSI-Bench. Dark orange and light orange highlight the best and second-best results within each group of models (proprietary and open-source).

—parsed denotes results obtained by using a LLM to extract answers from the original model outputs. Dist. Means., Cnt., Loc., and Rel. denote Distance Measurement, Counting, Localization, and Relational tasks, respectively.

Dist. Meas. Cnt. Loc. Rel. Reasoning

Ball-ZoneBall-PlayerCam.-PlayerPlayer-ZonePlayer-PlayerPlayer-Line

Overall Cam.-Obj.

Height Obj.-Line Obj.-Obj.

Models

Player Obj.

Ball

Baseline [5% Set] Human 64.4 92.7 67.8 70.0 100 100 11.9 85.7 75.0 100 83.3 90.3 88.9 73.6 Proprietary Models

GPT-5.2 27.9 78.4 31.8 49.2 32.1 100 1.1 68.2 67.0 75.4 50.0 67.7 77.4 53.7 Gemini-3-Pro 0.0 8.7 0.0 0.0 21.4 97.1 0.0 10.6 3.0 37.5 34.1 20.4 5.3 8.7

—parsed 40.4 81.4 50.5 67.8 60.7 100 5.2 71.4 70.7 89.5 73.2 85.0 79.8 64.6 Seed1.8 3.0 72.7 43.8 45.2 75.0 100 0.5 65.5 69.0 82.3 40.2 71.0 77.4 52.7 Claude-Sonnet4.5 0.0 0.0 0.0 0.0 85.7 88.2 0.0 26.3 1.7 11.7 14.6 2.5 1.4 5.0

—parsed 19.4 80.7 44.5 49.2 85.7 97.1 0.3 58.4 55.2 61.3 47.6 61.3 60.2 49.1 Grok4 12.2 60.0 30.9 36.7 50.0 97.1 0.0 44.7 38.7 37.5 34.1 46.8 48.9 36.2 Qwen3-Max 9.5 72.4 23.4 35.0 7.1 91.2 0.0 52.9 48.1 55.2 13.4 50.4 51.3 38.2

###### Open-source General Models

Qwen3-VL-8B 3.1 49.3 21.3 27.1 39.3 97.1 0.0 56.9 57.9 71.8 30.5 52.9 50.1 37.7 Qwen3-VL-32B 4.1 60.7 5.1 22.6 39.3 100 0.0 64.7 56.6 76.2 48.8 57.8 64.2 39.8 Qwen3-VL-235B-A22B 1.2 58.9 24.3 34.9 42.9 100 0.0 67.5 70.0 84.3 35.3 71.0 71.1 47.2 InternVL3.5-8B 0.0 0.0 0.0 0.0 78.6 67.6 0.0 50.2 55.6 69.8 20.7 51.4 60.0 27.9 InternVL3.5-38B 0.0 0.0 0.0 0.5 42.9 100 0.0 58.4 64.6 79.8 32.9 63.1 67.7 32.5 InternVL3.5-241B-A28B 0.7 51.9 16.5 16.0 39.3 100 0.0 58.4 66.0 80.2 56.1 64.1 65.3 40.0 Kimi-VL-16B-A3B 0.0 56.4 19.5 16.7 46.4 100 0.0 56.5 57.6 60.5 32.9 51.1 47.9 34.7 LLaVA-OneVision-7B 0.0 45.2 14.9 16.0 46.4 100 0.0 56.0 50.5 73.4 41.5 53.2 51.5 34.6 LLaVA-OneVision-72B 13.5 67.1 24.7 29.5 28.6 100 0.3 54.9 61.6 72.2 54.9 55.7 55.6 42.0 LLaVA-OneVision1.5-8B 3.7 49.0 21.2 26.9 10.7 100 0.3 44.7 44.1 56.0 34.1 45.3 46.9 33.3

###### Open-source Spatial Intelligence Models

[Base] Qwen2.5-VL-7B 4.8 50.5 20.2 9.3 35.7 100 0.0 54.9 60.3 74.2 58.5 61.6 54.9 37.0 SpaceR-7B 0.4 47.5 3.9 1.9 39.2 100 0.0 59.6 58.6 72.2 40.2 59.2 52.3 32.8 VST-7B-SFT 0.0 55.2 19.3 19.7 35.7 100 0.0 51.8 57.6 78.6 48.8 65.9 61.0 39.6 VST-7B-RL 0.0 50.3 22.2 20.3 35.7 100 0.0 54.9 59.6 75.4 53.6 64.9 61.8 40.0

[Base] Qwen2.5-VL-3B 4.6 51.4 20.3 20.1 35.7 97.1 0.0 52.9 49.8 68.5 40.2 51.7 46.6 35.0 SpatialLadder 0.0 56.7 22.3 12.4 57.1 97.1 0.0 53.7 50.5 63.7 48.8 55.7 49.5 34.7

[Base] InternVL3-8B 0.0 0.0 0.0 0.0 46.4 14.7 0.0 57.3 57.9 71.0 31.7 52.4 56.6 27.8 SenseNova-SI-8B 0.7 40.0 21.3 17.5 67.9 47.1 0.0 43.5 53.5 49.2 26.8 49.4 48.9 31.5

Cambrain-S-7B 0.0 3.2 0.2 0.0 17.9 85.3 0.0 63.5 44.8 58.1 7.3 55.2 47.5 25.5 Ours Qwen3-VL-8B 60.2 94.2 47.6 68.4 92.9 100 7.9 65.1 63.6 78.2 85.4 56.7 68.5 61.2 Improvement 57.1 44.9 26.3 41.3 53.6 2.9 7.9 8.2 5.7 6.4 54.9 3.8 18.4 23.5

∑︁

1 10

|𝑦ˆ − 𝑦| −𝑇 𝑦

T-MRA =

< 1 − 𝜃 , (3)

𝜃∈C

where 𝑦 and 𝑦ˆ denote ground truth and prediction, respectively. The confidence thresholds span {0.5,0.55, ...,0.95}, consistent with VSI. The distance threshold 𝑇 is set to 15cm according to table 1.

##### 4.2. Evaluation on CourtSI-Bench

We evaluate baseline models on CourtSI-Bench. Each input consists of a question paired with a single image annotated with bounding boxes and corresponding instructions to differentiate among players [20]. The results are summarized in table 2. We provide a detailed analysis below.

Human Level Performance. We recruit two volunteers to complete the evaluation on a uniformly sampled 5% subset of CourtSI-Bench. Human evaluators achieve the strongest performance compared to all existing models across all metrics. However, even with court geometry as a reference, human performance drops noticeably on metric-sensitive tasks, particularly distance measurement and localization. This limitation is also observed in several 3D vision tasks, where humans are required to estimate absolute distances and tend to underperform state-of-the-art specialized models. The current state of spatial intelligence in sports scenarios motivates the development of more general models capable of accurate 3D perception and reasoning under flexible language instructions, thereby assisting humans in metric-level spatial understanding.

Proprietary Models. Several proprietary models demonstrate strong performance, in some cases approaching human-level results. Among them, Gemini-Pro achieves the best overall performance across most metrics, with the exception of ball counting. However, we observe notable issues with instruction compliance in Gemini3-Pro and Claude-Sonnet-4.5. Although the models are required to produce final answers in a specified format, they frequently generate uncontrolled intermediate reasoning or extended explanations, violating the output constraints. Notably, their competitive performance is largely achieved only after applying an additional LLM to parse answers from the original outputs. Without this post-processing step, the performance drops significantly, indicating substantial room for improvement in controllable response generation.

Open-source General Models. Among the open-source general models, Qwen3-VL-235BA22B achieves the strongest performance, with only a limited gap compared to the bestperforming proprietary models. However, most open-source models perform poorly on CourtSIBench, with overall accuracy below 40%. Moreover, in distance measurement tasks, some models even exhibit near-total failure under the loose T-MRA metric.

Open-source Spatial Intelligence Models. For spatial intelligence models, although they are specifically fine-tuned for spatial relationship understanding and metric distance measurement, we do not observe consistent improvements over their respective base models on CourtSI-Bench. This suggests that sports scenarios introduce additional spatial reasoning challenges that are not sufficiently captured by existing large-scale spatial intelligence benchmarks.

SFT on CourtSI. After conducting SFT on CourtSI, the Qwen3-VL-8B model gains consistent improvement across all evaluation metrics, achieving a gain of 23.5 percentage points in accuracy. Notably, performance on the challenging distance measurement task improves by more than 25 percentage points. These results demonstrate the effectiveness of CourtSI in enhancing the spatial intelligence of VLM in sports.

[Figure 109]

###### <Relational Reasoning>

###### Question

###### VLM’s Explanation

[General Descriptions]…Players are identified by bounding boxes labeled with serial numbers.

Logic Chain: Identify the shuttlecock; Locate each player’s pelvis;

[Figure 110]

[Figure 111]

( Assess court length); Compare relative distance.

Measuring from the pelvis of each player, which of these players is closest to the shuttlecock?

[Human] clues: Player 1 is about to serve.

Specific Reasoning: The shuttlecock is visible as a small white object high in the air, positioned over the far court, toward the left side of the image.

(A)Player1√(B)Player2 (C)Player3 (D)Player4

[Figure 112]

[Figure 113]

The shuttlecock is visible slightly above the net on the far court, toward the left side of the image.

[Figure 114]

answer: D answer: D

[Figure 115]

Error in estimating 3D position of the ball

[Figure 116]

[Figure 117]

###### <Relational Reasoning>

###### Question

###### VLM’s Explanation

[General Descriptions]…Players are identified by bounding boxes labeled with serial numbers.

Logic Chain: ldentify the players; Determine Camera Perspective;

[Figure 118]

1

Locate Player 2 to Player 1; Perspective Conversion.

2

Based on Player 1's perspective, is Player 2 located to their left or right?

[Human] clues: Player 1 is facing camera sideways.

Specific Reasoning: Player 1 is on the far side of the table, facing the camera. Player 2 is on the near side of the table, facing away from the camera.

√(A)Left-side (B)Right-side

[Figure 119]

[Figure 120]

We switch to Player 1’s perspective (facing toward the camera). The camera’s left corresponds to Player 1’s right.

[Figure 121]

answer: B answer: B

[Figure 122]

Error in estimating orientation of a player

[Figure 123]

[Figure 124]

###### <Localization>

###### Question

###### VLM’s Explanation

[Figure 125]

[General Descriptions][Coordinate Descriptions]… [Player Descriptions]

Logic Chain: Calculate court geometry based on the coordinate system; Localize player using court as a reference.

What is the 3D coordinate (x, y, z) of the left foot of Player 2 in meters?

[Human] clues: Player 2 is outside the far baseline.

Specific Reasoning: Player 2 is in the far court, standing inside, in front of the far baseline ($X=0$)…the X coordinate is positive.

√(-1.55,3.87,0.03)

[Figure 126]

[Figure 127]

Player 2 is 1 to 1.5 meters inside the far baseline. Player 2 is on the right of the camera…Y coordinate is from the left sideline.

[Figure 128]

answer: (1.0, 4.4, 0.0) answer: (1.2, 6.1, 0.0)

[Figure 129]

Error in player localization

[Figure 130]

- Figure 5 | Error Analysis. The VLMs are prompted to provide detailed step-by-step reasoning. Correct and incorrect reasoning steps are highlighted in green and red, respectively. Questions and VLM’s explanations are simplified for demonstration.

##### 4.3. In-depth Error Analysis on CourtSI-Bench

To better understand VLM performance on CourtSI-Bench, we conduct case studies on the categories with low accuracy, including relational reasoning and localization. Specifically, we prompt the strongest-performing Gemini-3-Pro and GPT-5.2 to explain the reasoning behind their predictions on failure cases.

We summarize the representative cases in fig. 5. From top to bottom, the cases involve: the relative distance between sport-specific objects, ball and player; reasoning about player–player relationships under an allo-centric perspective; metric-aware localization for absolute distance measurement.

In many instances, the VLMs produce human-like and logically structured reasoning chains. For example, they first localize relevant objects before comparing their spatial relationships. Furthermore, VLMs perform well under some challenging instructions: they identify the tiny ball (“small white object”, fig. 5, top), handle ego-centric to allo-centric perspective conversion (“switch to ...from camera“, fig. 5, mid), and leverage court geometry as a reference based on general sport knowledge (“the far baseline (X=0)”, fig. 5, bottom). These observations suggest that VLMs can interpret spatial descriptions from instructions and demonstrate a basic level of structured reasoning.

However, the models struggle with accurate 3D localization from 2D imagery and finegrained relational understanding. In the top and bottom cases of fig. 5, the VLMs incorrectly estimate object relationships with respect to the court geometry. In the middle case, a counterfactual configuration, where a player stands on the far side of the court while facing sideways relative to the camera, leads to erroneous results. These failure modes stem from the distinctive characteristics of the curated CourtSI-Bench, which introduce spatial ambiguities that challenge current VLMs and highlight substantial room for improvement.

- Table 3 | Evaluation on CourtSI-Ext. Annotation conventions follow table 2.

Dist. Meas. Cnt. Loc. Rel. Reasoning

Ball-ZoneBall-PlayerCam.-PlayerPlayer-ZonePlayer-PlayerPlayer-Line

Overall Cam.-Obj.

Height Obj.-Line Obj.-Obj.

Models

Player Obj.

Ball

GPT-5.2 61.8 70.0 44.2 53.6 33.3 100 0.0 72.7 66.7 90.0 21.4 73.5 76.2 55.0 Gemini-3-Pro 0.0 15.4 0.0 0.0 33.3 100 0.0 9.1 11.1 30.0 64.3 17.6 0.0 13.5

—parsed 75.5 83.1 66.3 56.4 83.3 100 0.0 90.9 100 90.0 85.7 70.6 76.2 66.8 Qwen3-VL-235B-A22B 0.0 50.8 30.4 27.6 50.0 100 0.0 63.6 88.9 100 64.3 67.6 71.4 47.9 LLaVA-OneVision-72B 19.1 53.8 28.3 21.8 50.0 100 0.0 45.5 55.6 90.0 71.4 61.8 57.1 43.3

[Base] Qwen3-VL-8B 0.9 50.0 31.3 15.2 33.3 100 0.0 63.6 77.8 100 21.4 52.9 52.4 38.2 Ours 70.0 83.1 34.6 62.7 83.3 100 0.0 63.6 66.7 70.0 28.6 44.1 66.7 51.4

[Figure 131]

In addition, we further demonstrate VLMs’ limitations in handling spatial ambiguity caused by perspective projection. Specifically, we rank the cases of distance measurement in CourtSI-Bench by a ratio of 3D distance to 2D distance, to reflect the level of perspective ambiguity. A higher ratio indicates that objects are distant in 3D space but appear close in the image plane due to perspective effects. As shown in fig. 6, we evaluate VLMs’ performance on the top-percentage subsets. The results show a clear performance degradation as the ambiguity increases, revealing a factor behind the erroneous relational reasoning, particularly for distance measurement when precise estimates are required.

Figure 6 | The impact of perspective ambiguity.

##### 4.4. Expanding the Scope of CourtSI-Bench

To broaden the evaluation scope of CourtSI-Bench, we extend it toward more applicationoriented settings. Specifically, we construct two additional scenarios: (i) an unseen-sport evaluation set, CourtSI-Ext, designed to assess the generalization capability of spatial intelligence models fine-tuned on CourtSI data; and (ii) a spatial-aware commentary generation task, which serves as a potential downstream application of models equipped with sports spatial reasoning ability.

CourtSI-Ext. Following the taxonomy of CourtSI-Bench, we leverage the data engine in section 3.1 to construct an extended evaluation set, CourtSI-Ext. It is built on pickleball, a net sport with court geometry similar to tennis and badminton. CourtSI-Ext contains 215 QA pairs from 111 images across 35 distinct scenes for cross-sport evaluation. We report results of topperforming VLMs from CourtSI-Bench on this extension. The evaluation process is consistent with CourtSI-Bench. Image examples are presented in section C.3.

As shown in table 3, our fine-tuned model achieves 13.2 percentage points improvements in overall accuracy compared to its base model, further validating the effectiveness of the curated CourtSI data. However, the cross-sport generalization challenge in spatial intelligence remains. The improvement of SFT shrinks on CourtSI-Ext. Specifically, in the localization task, although

[Figure 132]

Primary Task: Generate a vivid and natural sports commentary describing the scene.

###### CourtSI-SFT vs Base

1

| |
|---|

better similar worse

Distance between the <ball> and the <right foot> of <Player 1>

[Figure 133]

2

SI Task: Incorporate the spatial

GT: 19.03 meters

| |
|---|

relationship into the commentary.

(not accessible in generation)

[Figure 134]

[Figure 135]

###### Base Model

CourtSI-SFT Model

linguistic quality

… Over on the far end, Player 1....his eyes locked on the trajectory of that ball, which is now just a few meters from the baseline. The distance between the ball and player’s right foot is a mere 18.2 meters — a testament to the speed of the shot and the urgency of the chase.

…. Player 1’s right foot is planted firmly on the clay, a stable anchor as he prepares to unleash his own powerful return. The distance between the ball and his right foot? Just 2.1 meters. That’s the kind of splitsecond precision that defines champions.

spatial-awareness

- Figure 7 | Evaluation on the spatial-aware commentary generation. Comparison between Qwen3-VL-8B fine-tuned on CourtSI and its base model. The left panel shows user study results assessing the quality of generated commentaries in both linguistic and spatial-awareness dimensions. The right panel presents an illustrative example.

our model reduces the average error to 3.9 meters compared to about 6 meters from other baselines, it does not yield corresponding gains in accuracy. As an initial study, we highlight the cross-sport challenge and curate CourtSI-Ext to serve as a small yet valuable benchmark for broader community validation.

Spatial-aware Commentary Generation. As shown in fig. 7 (right), we extract spatial relationships from CourtSI-Bench and instruct models to generate sports commentary that incorporates these spatial relationships. We compare the Qwen3-VL-8B model fine-tuned on CourtSI with its base model. A total of 100 generated commentaries are evaluated through a user study involving three volunteers across both linguistic quality and spatial awareness dimensions.

The results show that fine-tuning on CourtSI significantly improves spatial awareness, while preserving overall linguistic quality, highlighting that the model is able to transfer the spatial capability to downstream commentary generation tasks. It illustrates the potential of CourtSI to enhance spatial reasoning in sports understanding and to serve as supervision for general VLM post-training.

#### 5. Conclusion

In this paper, we present CourtSI, the first large-scale spatial intelligence dataset for sports, comprising over 1M QA pairs, along with the high-quality CourtSI-Bench for evaluation. By leveraging court geometry as metric anchors, we develop a semi-automatic data engine to produce accurate and scalable supporting data. A comprehensive evaluation across 25 stateof-the-art VLMs reveals a clear human–AI performance gap and limited generalization from existing spatial intelligence benchmarks. Furthermore, through fine-tuning, cross-sport evaluation, and commentary generation, we broaden the evaluation scope of CourtSI-Bench and demonstrate that CourtSI serves as an effective pathway for advancing the spatial intelligence of VLMs in sports.

#### References

- [1] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.
- [2] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.
- [3] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025.
- [4] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976, 2025.
- [5] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.
- [6] Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, et al. Scaling spatial intelligence with multimodal foundation models. arXiv preprint arXiv:2511.13719, 2025.
- [7] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025.
- [8] Haotian Xia, Zhengbang Yang, Junbo Zou, Rhys Tracy, Yuqing Wang, Chi Lu, Christopher Lai, Yanjun He, Xun Shao, Zhuoqing Xie, et al. Sportu: A comprehensive sports understanding benchmark for multimodal large language models. arXiv preprint arXiv:2410.08474, 2024.
- [9] Haotian Xia, Haonan Ge, Junbo Zou, Hyun Woo Choi, Xuebin Zhang, Danny Suradja, Botao Rui, Ethan Tran, Wendy Jin, Zhen Ye, et al. Sportr: A benchmark for multimodal large language model reasoning in sports. arXiv preprint arXiv:2511.06499, 2025.
- [10] Xusheng He, Wei Liu, Shanshan Ma, Qian Liu, Chenghao Ma, and Jianlong Wu. Finebadminton: A multi-level dataset for fine-grained badminton video understanding. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12776–12783, 2025.
- [11] Rong Gao, Xin Liu, Zhuozhao Hu, Bohao Xing, Baiqiang Xia, Zitong Yu, and Heikki Kälviäinen. Fsbench: A figure skating benchmark for advancing artistic sports understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13595–13605, 2025.
- [12] Yufu Wang, Yu Sun, Priyanka Patel, Kostas Daniilidis, Michael J Black, and Muhammed Kocabas. Prompthmr: Promptable human mesh recovery. In Proceedings of the computer vision and pattern recognition conference, pages 1148–1159, 2025.
- [13] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019.
- [14] Linfeng Dong, Yuchen Yang, Hao Wu, Wei Wang, Yuenan Hou, Zhihang Zhong, and Xiao Sun. Racketvision: A multiple racket sports benchmark for unified ball and racket analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

- [15] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [16] Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 346–355, 2024.
- [17] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.
- [18] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780, 2025.
- [19] Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, and Weidi Xie. Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv e-prints, pages arXiv–2505, 2025.
- [20] Nianchen Deng, Lixin Gu, Shenglong Ye, Yinan He, Zhe Chen, Songze Li, Haomin Wang, Xingguang Wei, Tianshuo Yang, Min Dou, et al. Internspatial: A comprehensive dataset for spatial reasoning in vision-language models. arXiv preprint arXiv:2506.18385, 2025.
- [21] Wenqi Wang, Reuben Tan, Pengyue Zhu, Jianwei Yang, Zhengyuan Yang, Lijuan Wang, Andrey Kolobov, Jianfeng Gao, and Boqing Gong. Site: towards spatial intelligence thorough evaluation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9058–9069, 2025.
- [22] Jingli Lin, Runsen Xu, Shaohao Zhu, Sihan Yang, Peizhou Cao, Yunlong Ran, Miao Hu, Chenming Zhu, Yiman Xie, Yilin Long, et al. Mmsi-video-bench: A holistic benchmark for video-based spatial intelligence. arXiv preprint arXiv:2512.10863, 2025.
- [23] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025.
- [24] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.
- [25] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.
- [26] Peiwen Sun, Shiqiang Lang, Dongming Wu, Yi Ding, Kaituo Feng, Huadai Liu, Zhen Ye, Rui Liu, Yun-Hui Liu, Jianan Wang, et al. Spacevista: All-scale visual spatial reasoning from mm to km. arXiv preprint arXiv:2510.09606, 2025.
- [27] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. In The Fourteenth International Conference on Learning Representations, 2026.
- [28] Zhipeng Cai, Ching-Feng Yeh, Hu Xu, Zhuang Liu, Gregory P. Meyer, Xinjie Lei, Changsheng Zhao, Shang-Wen Li, Vikas Chandra, and Yangyang Shi. DepthLM: Metric depth from vision language models. In The Fourteenth International Conference on Learning Representations, 2026.

- [29] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.
- [30] Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, et al. Visual spatial tuning. arXiv preprint arXiv:2511.05491, 2025.
- [31] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025.
- [32] Yuanyuan Gao, Hao Li, Yifei Liu, Xinhao Ji, Yuning Gong, Yuanjun Liao, Fangfu Liu, Manyuan Zhang, Yuchen Yang, Dan Xu, Xue Yang, Huaxi Huang, Hongjie Zhang, Ziwei Liu, Xiao Sun, Dingwen Zhang, and Zhihang Zhong. Holi-spatial: Evolving video streams into holistic 3d spatial intelligence. arXiv preprint arXiv:2603.07660, 2026.
- [33] Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, et al. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7395–7408, 2025.
- [34] Gongjie Zhang, Wenhao Li, Quanhao Qian, Jiuniu Wang, Deli Zhao, Shijian Lu, and Ran Xu. On the generalization capacities of MLLMs for spatial intelligence. In The Fourteenth International Conference on Learning Representations, 2026.
- [35] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025.
- [36] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625, 2025.
- [37] Silvio Giancola, Mohieddine Amine, Tarek Dghaily, and Bernard Ghanem. Soccernet: A scalable dataset for action spotting in soccer videos. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 1711–1721, 2018.
- [38] Mostafa S Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori. A hierarchical deep temporal model for group activity recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1971–1980, 2016.
- [39] Yuchen Yang, Wei Wang, Yifei Liu, Linfeng Dong, Hao Wu, Mingxin Zhang, Zhihang Zhong, and Xiao Sun. Sga-interact: A 3d skeleton-based benchmark for group activity understanding in modern basketball tactic. arXiv preprint arXiv:2503.06522, 2025.
- [40] Zhe Wang, Petar Velicˇkovic´, Daniel Hennes, Nenad Tomašev, Laurel Prince, Michael Kaisers, Yoram Bachrach, Romuald Elie, Li Kevin Wenliang, Federico Piccinini, et al. Tacticai: an ai assistant for football tactics. Nature communications, 15(1):1906, 2024.
- [41] Linfeng Dong, Wei Wang, Yu Qiao, and Xiao Sun. Lucidaction: A hierarchical and multi-model dataset for comprehensive action quality assessment. Advances in neural information processing systems, 37:96468–96482, 2024.
- [42] Jiayuan Rao, Haoning Wu, Hao Jiang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards universal soccer video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8384–8394, 2025.
- [43] Hassan Mkhallati, Anthony Cioppa, Silvio Giancola, Bernard Ghanem, and Marc Van Droogenbroeck. Soccernet-caption: Dense video captioning for soccer broadcasts commentaries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5074–5085, 2023.

- [44] Zeyu Xi, Ge Shi, Xuefen Li, Junchi Yan, Zun Li, Lifang Wu, Zilin Liu, and Liang Wang. A simple yet effective knowledge guided method for entity-aware video captioning on a basketball benchmark. Neurocomputing, 619:129177, 2025.
- [45] Zeyu Xi, Haoying Sun, Yaofei Wu, Junchi Yan, Haoran Zhang, Lifang Wu, Liang Wang, and Changwen Chen. Player-centric multimodal prompt generation for large language model based identity-aware basketball video captioning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24330–24339, 2025.
- [46] Haotian Xia, Zhengbang Yang, Yuqing Wang, Rhys Tracy, Yun Zhao, Dongdong Huang, Zezhi Chen, Yan Zhu, Yuan-fang Wang, and Weining Shen. Sportqa: A benchmark for sports understanding in large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5061–5081, 2024.
- [47] Jiayuan Rao, Zifeng Li, Haoning Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Multi-agent system for comprehensive soccer understanding. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 3654–3663, 2025.
- [48] Junbo Zou, Haotian Xia, Zhen Ye, Shengjie Zhang, Christopher Lai, Vicente Ordonez, Weining Shen, and Hanjie Chen. Deepsport: A multimodal large language model for comprehensive sports video reasoning via agentic reinforcement learning. arXiv preprint arXiv:2511.12908, 2025.
- [49] Shengjie Zhu, Abhinav Kumar, Masa Hu, and Xiaoming Liu. Tame a wild camera: In-the-wild monocular camera calibration. Advances in Neural Information Processing Systems, 36:45137–45149, 2023.
- [50] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [51] Gabriel Van Zandycke and Christophe De Vleeschouwer. 3d ball localization from a single calibrated image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3472–3480, 2022.
- [52] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.
- [53] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [54] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.
- [55] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [56] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.
- [57] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [58] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [59] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410, 2024.
- [60] Daniel Kienzle, Katja Ludwig, Julian Lorenz, Shin’ichi Satoh, and Rainer Lienhart. Uplifting table tennis: A robust, real-world application for 3d trajectory and spin estimation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026.
- [61] Shengjie Shen, Yunzhi Hua, Zhipeng Jiang, Zhicheng Li, Mingze Gao, Zequn Ge, Zhiqiang Han, Fan Zhong, and Xinggang Chen. Tame a wild camera: In-the-wild monocular camera calibration. In Advances in Neural Information Processing Systems, 2023.

#### Contents

- A. Data Engine Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .1
- B. CourtSI Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .4
- C. Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- D. Ethical Considerations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .13

#### A. Data Engine Details

##### A.1. Pipeline Details

The proposed data engine comprises court annotation, ball annotation, and player mesh recovery. We detail each in the following.

Court Annotation. We estimate the camera parameters using a PnP solver with annotated court keypoints and their corresponding 3D positions derived from the fixed court geometry. We develop an interactive panel to assist annotators in selecting keypoints.

The annotation is performed on the raw videos, and a frame in which all court keypoints are clearly visible is selected for calibration. For scenes with a static camera view, the estimated calibration parameters are reused across all frames. For a few cases with dynamic camera views, we propagate the court keypoints to adjacent frames and apply the proposed calibration method to estimate the camera parameters of each neighboring frame using the transformed keypoints. Specifically, we utilize DepthAnythingV3 to estimate both per-pixel depth and relative camera parameters between frames. Given the annotated reference frame and an adjacent frame, it provides the depth map as well as the camera intrinsics and extrinsics for each frame. Using the estimated depth, 2D keypoints in the reference frame are first back-projected into 3D space. The 3D points are then transformed to the coordinate system of the adjacent frame using the relative camera pose, and finally re-projected onto the image plane of the adjacent frame to obtain the corresponding 2D positions.

With reprojection-based verification, we observe that this propagation process is effective across frames. Although DepthAnythingV3 supports metric depth estimation and multi-frame camera parameter estimation, we find that directly using the calibrated reference frame as input and relying on DepthAnythingV3 to propagate camera parameters to subsequent frames introduces significant calibration errors. In practice, the accumulated pose and scale inconsistencies lead to noticeable reprojection deviations. Therefore, instead of directly adopting the propagated camera parameters, we use DepthAnythingV3 primarily for depth-guided geometric transfer and perform camera parameter estimation independently to ensure calibration stability.

The calibration results are illustrated in fig. 8. The world coordinate system is defined in a right-handed format as follows: the origin is located at the far corner point of the court from the camera’s perspective; the x-axis is aligned with the court length and is positive toward the camera; the y-axis is aligned with the court width and is positive toward the camera; and the z-axis is perpendicular to the court plane.

Ball Annotation. In the main text, we describe the ball annotation process within a single frame by converting depth estimation into ball projection estimation on the court ground plane. For the raw video data, we model the ball trajectory during each rally. While the ball is airborne, until it

[Figure 136]

- Figure 8 | Calibration examples. A 3D court box with real-world dimensions is reprojected onto the image using the estimated camera parameters. The close alignment between the projected court structure and the image clues indicates strong reprojection consistency, validating the accuracy of the calibration.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

2D Projection

image depth from DepthAnything V3

depth from PromptHMR

Ours (side view)

Ours (camera view)

- Figure 9 | Illustration of player depth estimation using PromptHMR, DepthAnythingV3, and our method. All baselines take metric-scale camera intrinsics as input.

is struck by a player or contacts the court, it is primarily influenced by gravity, aerodynamic lift generated by spin, and air resistance. We approximate this motion as a constant-acceleration problem. Annotators label the start point, midpoint, and end point of each trajectory segment, from which the 3D acceleration and initial velocity can be estimated. This approach significantly reduces the annotation effort required for airborne ball tracking. When the estimated trajectory does not meet quality standards, we revert to per-frame annotation to ensure accuracy. For table tennis, we adopt the 2D-to-3D lifting approach proposed in [60] to estimate the ball position. This method takes the court corner positions as input and is trained on large-scale, high-fidelity simulation data, which enhances robustness under our experimental conditions.

Player Mesh Recovery. PromptHMR estimates the human mesh in camera coordinates, conditioned on the bounding box and camera parameters. We first employ SAM3 to track target players using the prompt “player.” Although SAM3 performs well in most cases, we develop an interactive refinement panel to manually correct a small number of inaccurate detections. Regarding camera parameters, PromptHMR assumes simplified camera intrinsics, where the focal lengths along the x- and y-axes are identical and the principal point is located at the image center. We therefore optimize this simplified camera model using the previously annotated

- Table 4 | Quantitative error analysis of camera intrinsic parameters.

Baseline

Badminton Tennis Table Tennis 𝑒𝑓𝑥(%) 𝑒𝑓𝑦(%) 𝑒𝑓𝑥(%) 𝑒𝑓𝑦(%) 𝑒𝑓𝑥(%) 𝑒𝑓𝑦(%) WildCamera 67.20 70.86 17.48 12.16 13.66 15.20 DepthAnythingV3 38.89 38.24 5.42 4.16 0.85 0.90 Ours 0.55 0.72 4.23 4.31 0.01 1.45

- Table 5 | Quantitative error analysis of ball localization. ∗ denotes using ground truth 2D ball locations and camera parameters.

Badminton Tennis Table Tennis

Baseline

𝑋 𝑌 𝑍 𝑋 𝑌 𝑍 𝑋 𝑌 𝑍

DepthAnythingV3∗ 1227cm 252cm 241cm 3168cm 2045cm 1833cm 249cm 199cm 26cm Ours 10cm 4cm 6cm 29cm 11cm 9cm 0.3cm 0.3cm 0.5cm

court keypoints. Based on quality control, this simplification does not involve much error in final localization.

As discussed in the main text, we observe that the estimated depths of the recovered human meshes are often inaccurate. fig. 9 provides an qualitative example. To address this issue, annotators manually estimate the depth of the lowest mesh vertex using the same strategy as in ball annotation, by labeling its height above the court surface. The entire mesh is then re-aligned according to the corrected depth. Instead of directly translating the mesh by the depth offset, which would distort its 3D scale, we apply a similarity transformation centered at the camera location 𝐶:

𝑋′ = 𝑠𝑋 + (1 − 𝑠)𝐶, (4)

where the depth scale factor 𝑠 is computed from the depth correction of the lowest vertex. This transformation uniformly rescales the mesh along rays emanating from the camera center: the mesh is enlarged when the corrected depth is closer to the camera (i.e., smaller depth), and shrunk when the corrected depth is farther away (i.e., larger depth).

##### A.2. Comparison with Monocular Scene Reconstruction Methods

In this section, we present a detailed comparison with monocular scene reconstruction methods using the multi-view evaluation dataset introduced in section 3.1.

Camera Calibration. Following [61], we measure the accuracy of the estimated camera intrinsics using the relative focal length errors 𝑒𝑓𝑥 and 𝑒𝑓𝑦, defined as:

, 𝑒𝑓𝑦 = | 𝑓𝑦pred − 𝑓𝑦gt|

𝑒𝑓𝑥 = | 𝑓𝑥pred − 𝑓𝑥gt|

. (5)

𝑓𝑥gt

𝑓𝑦gt

Both metrics are reported as percentages and averaged over all video sequences within each sport category, as shown in table 4. The results indicate that our calibration method achieves the best performance across the three sports scenarios by explicitly leveraging court geometry.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

- Figure 10 | Original broadcast frames (top) and depth maps estimated by DepthAnythingV3 (bottom). Yellow arrows indicate the ball positions.

- Table 6 | Quantitative error analysis of player localization.

Pelvis

Baseline

Badminton Tennis Table Tennis

PromptHMR 134cm 1144cm 33cm DepthAnythingV3 2191cm 2744cm 62cm Ours 21cm 27cm 16cm

Ball Localization. For ball localization, we employ DepthAnythingV3 as the metric depth estimator, following a two-stage pipeline of detection followed by lifting. As shown in table 5, even when provided with ground-truth 2D locations and camera parameters, DepthAnythingV3 fails to produce accurate metric depth estimates. Qualitatively, as illustrated in fig. 10, this failure likely arises because the ball occupies only a small image region, and the predicted depth map cannot capture such subtle details.

Player Localization. In table 6, we report the quantitative evaluation of player localization. For all baseline methods, the estimated camera parameters are used as input, and the predicted depth is employed to transform the human mesh into the 3D world coordinate system. We then compute the 3D pelvis position error as the localization metric. The results demonstrate that our localization method outperforms the baseline approaches.

#### B. CourtSI Details

##### B.1. Question Template

All query templates evaluated in our benchmark follow a systematic structure composed of three components: pre-prompt, question, and post-prompt. Specifically, each query is formed by combining a pre-prompt, a question, and a post-prompt. The diversity of queries in our benchmark primarily arises from the use of different question types.

Pre-prompt establishes the contextual background of the problem and mitigates any potential ambiguities inherent in the query. Its standardized content is defined as follows:

This is a snapshot from a {sport name} match view from a high angle. The court closer to the camera is the ‘near court’, and the opposite one is the ‘far court’. All references to ‘left’ or ‘right’ in the questions describing the court or relative positions are based on the camera’s perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., ‘left wrist’,‘right knee’) follow the player’s anatomical perspective (the player’s own left/right).

Post-prompt delineates the explicit formatting rules for the model’s output. Depending on the inquiry category, the model must follow one of two high-level output groups: numerical or multiple-choice (MCQ). Numerical outputs include floating-point numbers, integers, or 3D spatial coordinates, while MCQ output is a single multiple-choice option. Its content is defined as follows:

floating-point number: Answer with a single float number representing meters. Example: 2.54 3D coordinate: Answer strictly in the format (x, y, z) with no units. Example: (1.2, 3.4, 0.0)

integer: Answer with a single integer number. Example: 3 multiple-choice option: Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

Question can be broadly classified into 13 primary categories. When further categorized by the generated templates, they expand into 20 distinct types, comprising a total of 94 unique templates. In this section, we present a selection of the template generation results, organized according to the classification methodology detailed in the paper.

Within these examples below, bold text denotes the question category. Italicized text indicates interchangeable variables or elements requiring additional specification within the template. Numerical indices represent varied phrasing or content variations within the same question category (an exhaustive list is omitted due to space limitations). Based on our quantitative assessment, this question generation strategy is capable of producing 4,403 entirely distinct questions across three different ball sports.

Distance Measurement camera-object:

- 1. How far apart are the camera and object in 3D space?
- 2. Calculate the 3D Euclidean distance between the camera and object in meters. object-object:

- 1. What is the distance between object1 and object2 in meters?
- 2. If a line were drawn directly from object1 to object2, what would be its length in meters?

##### object-line:

- 1. What is the perpendicular distance from object to line?
- 2. Mapping object’s position to the court zone/the table surface, what is its perpendicular distance to line in meters? height:

- 1. What is the height of object in meters at this moment?
- 2. How high above the court surface is object currently positioned?

Spatial Counting player:

- 1. How many players are visible on the court in this image?
- 2. Count the total number of players currently playing in the match. ball:

- 1. Can you see the tennis ball/the ping pong ball/the shuttlecock in the snapshot? (A)Yes (B)No
- 2. Is the tennis ball/the ping pong ball/the shuttlecock visible in this image? (A)Yes (B)No

Localization Object:

Using a coordinate system where the origin (0,0,0) is the intersection of the far baseline and the left doubles sideline/the top-left corner of the table surface. The X-axis extends along the sideline towards the camera, the Y-axis extends along the far baseline/the far endline to the right, and the Z-axis is vertical.

- 1. What is the 3D coordinate (x, y, z) of object in meters?
- 2. Locate object within the defined coordinate system and return its (x, y, z) values. Relational Reasoning

##### player-player:

- 1. Measuring from the pelvis of each player, which of these players is closest to player? (A)Player 1 (B)Player 2 (C)Player 4 (set options according to the situation)
- 2. Based on player’s perspective, is player located to their left or right? (A)Left side (B)Right side
- 3. Is player positioned to the left or to the right of player from the camera’s view? (A)Left (B)Right (C)Directly in front or behind ball-zone:

- 1. In which longitudinal zone of the court is the tennis ball currently located? (A)The forecourt (between the net and the service line) (B)The midcourt (between the service line and the baseline) (C)The backcourt (outside the baseline)
- 2. Is the shuttlecock currently positioned above or below the top edge of the net? (A)Above the net (B)Below the net
- 3. Is the ping pong ball on the left or right side of the table center line? (A)Left side (B)Right side (C)On the center line ball-player:

- 1. Measuring from the pelvis of each player, which player has the smallest Euclidean distance to the tennis ball/the ping pong ball/the shuttlecock? (A)Player 1 (B)Player 2 (C)Player 4 (set options according to the situation)

- 2. Imagine you are player. Is the tennis ball/the ping pong ball/the shuttlecock currently to your left-hand side or right-hand side? (A)Left side (B)Right side
- 3. From the camera’s perspective, which side is player on relative to the tennis ball/the ping pong ball/the shuttlecock? (A)Left (B)Right (C)Directly in front or behind cam-player:

- 1. Measuring from the pelvis of each player, which of these players is closest to the camera? (A)Player 1 (B)Player 2 (C)Player 4 (set options according to the situation)
- 2. From the ego-centric view of player, which side is the camera on? (A)Left side (B)Right side
- 3. Is player positioned to the left or to the right of the camera from the camera’s view? (A)Left (B)Right (C)Directly in front or behind player-zone:

- 1. Classify the position of player into one of the three court zones: forecourt, midcourt, or backcourt. (A)The forecourt (between the net and the service line) (B)The midcourt (between the service line and the baseline) (C)The backcourt (outside the baseline)
- 2. Where is object1 standing relative to the length of the court? (A)Front court (B)Mid court (C)Rear court player-line:

- 1. Considering the pelvis position of each player, which player has the smallest perpendicular distance to line? (A)Player 1 (B)Player 2 (C)Player 3 (D)Player 4
- 2. Based on the pelvis positions, which player is nearest to line in terms of perpendicular distance? (A)Player 1 (B)Player 2 (C)Player 3 (D)Player 4 Is player positioned to the left or to the right of player from the camera’s view?

##### B.2. QA examples

[Figure 149]

###### Question:

This is a snapshot from a table tennis match view from a high angle. The table half closer to the camera is the 'near half', and the opposite one is the 'far half'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

###### Calculate the 3D Euclidean distance between the camera and the right foot of Player 2 in meters.

Answer with a single float number representing meters. Example: 2.54

Task type: Distance Measurement Category: Camera - Object

###### Answer:

24.26

[Figure 150]

###### Question:

This is a snapshot from a badminton match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

###### How high above the court surface is the right hand of Player 1 currently positioned?

Answer with a single float number representing meters. Example: 2.54

Task type: Distance Measurement Category: Height

###### Answer:

0.98

[Figure 151]

###### Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

What is the perpendicular distance from the pelvis of Player 1 to the net? Answer with a single float number representing meters. Example: 2.54

Task type: Distance Measurement Category: Object - Line

###### Answer:

16.45

[Figure 152]

###### Question:

This is a snapshot from a badminton match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

###### What is the distance between the pelvis of Player 1 and the pelvis of Player 3 in meters?

Answer with a single float number representing meters. Example: 2.54

Task type: Distance Measurement Category: Object - Object

###### Answer:

8.04

[Figure 153]

###### Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

Can you see the tennis ball in the snapshot? Choices:

- (A)Yes
- (B)No

Task type: Spatial Counting Category: Ball

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

###### Answer:

A

[Figure 154]

Question:

This is a snapshot from a badminton match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

###### Count the total number of athletes currently playing in the match.

Answer with a single integer number. Example: 3

Task type: Spatial Counting Category: Player

###### Answer:

2

[Figure 155]

###### Question:

This is a snapshot from a table tennis match view from a high angle. The table half closer to the camera is the 'near half', and the opposite one is the 'far half'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers. Using a coordinate system where the origin (0,0,0) is the top-left corner of the table surface (intersection of far endline and left sideline). The X-axis extends along the sideline towards the camera, the Y-axis extends along the far endline to the right, and the Z-axis is vertical (0 is table surface).

What is the 3D coordinate (x, y, z) of the right hand of Player 1 in meters? Answer strictly in the format (x, y, z) with no units. Example: (1.2, 3.4, 0.0)

Task type: Localization

###### Answer:

(3.44, -0.72, 0.35)

[Figure 156]

###### Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

Is Player 2 positioned to the left or to the right of the tennis ball from the camera's view? Choices:

- (A)Left
- (B)Right
- (C)Directly in front or behind

Task type: Relational Reasoning Category: Ball - Player

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

###### Answer:

- A

[Figure 157]

###### Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

Is the ping pong ball on the left or right side of the table center line? Choices:

- (A)Left side
- (B)Right side
- (C)On the center line

Task type: Relational Reasoning Category: Ball - Zone

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

###### Answer:

- B

[Figure 158]

###### Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

Is Player 4 positioned to the left or to the right of the camera from the camera's view? Choices:

- (A)Left
- (B)Right
- (C)Directly in front or behind

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

###### Answer:

- B

Task type: Relational Reasoning Category: Camera - Player

[Figure 159]

Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

Where is Player 2 standing relative to the length of the court? Choices:

- (A)Front court
- (B)Mid court
- (C)Rear court

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

Answer:

- C

Task type: Relational Reasoning Category: Player - Zone

[Figure 160]

###### Question:

This is a snapshot from a tennis match view from a high angle. The court closer to the camera is the 'near court', and the opposite one is the 'far court'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

From the ego-centric view of Player 4, which side is Player 1 on? Choices:

- (A)Left side
- (B)Right side

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

Task type: Relational Reasoning Category: Player - Player

###### Answer:

B

[Figure 161]

###### Question:

This is a snapshot from a table tennis match view from a high angle. The table half closer to the camera is the 'near half', and the opposite one is the 'far half'. All references to 'left' or 'right' in the questions describing the court or relative positions are based on the camera's perspective, corresponding to the left and right sides of the image frame. However, references to specific body parts (e.g., 'left wrist', 'right knee') follow the player's anatomical perspective (the player's own left/right). Players are identified by bounding boxes labeled with serial numbers.

Based on the pelvis positions, which player is nearest to the net in terms of perpendicular distance? Choices:

- (A)Player 1
- (B)Player 2

Task type: Relational Reasoning Category: Player - Line

Select the best option. Output only the single uppercase letter corresponding to the choice. Example: B

###### Answer:

A

##### B.3. Human Review

As described in the main text, all QA pairs in CourtSI-Bench undergo a final round of manual verification. Any pair flagged by either annotator is removed to ensure annotation quality and consistency. After this filtering process, we resample the remaining questions according to task categories and per-sport distribution to maintain a balanced benchmark. The final CourtSIBench contains 3,686 QA pairs, selected from 4,356 raw samples. Most discarded instances are due to ambiguous questions or the resampling procedure. For example, because players occupy a non-negligible physical width, certain left/right spatial relationships can be inherently unclear, leading to potential ambiguity for evaluation. In CourtSI, we introduce task-specific thresholds for each sport to mitigate this issue.

##### B.4. Data Distribution

- Table 7 | Detailed Data Distribution. B, T, and TT denote badminton, tennis, and table tennis, respectively.

CourtSI-Bench CourtSI Count B T TT Count B T TT

Category Name

Camera-Object 277 27.80% 35.74% 36.46% 75,783 33.93% 24.41% 41.66% Height 229 23.58% 37.12% 39.30% 51,154 31.43% 25.00% 43.56% Object-Line 317 24.61% 44.16% 31.23% 102,054 31.09% 25.20% 43.71% Object-Object 663 25.34% 41.18% 33.48% 178,878 29.95% 25.55% 44.50%

Distance Measurement

Spatial Counting

Ball 28 25.00% 42.86% 32.14% 23,015 31.02% 25.10% 43.88% Player 34 23.53% 32.35% 44.12% 22,897 31.03% 25.33% 43.63%

Localization - 368 31.25% 39.67% 29.08% 101,698 31.02% 25.24% 43.74%

Ball-Zone 255 25.88% 32.16% 41.96% 61,997 20.05% 22.35% 57.60% Ball-Player 297 24.24% 40.40% 35.35% 72,232 24.92% 27.26% 47.82% Camera-Player 248 25.40% 43.15% 31.45% 58,280 26.54% 28.61% 44.85% Player-Zone 82 51.22% 48.78% - 28,769 55.31% 44.69% Player-Player 393 44.27% 28.24% 27.48% 104,961 45.42% 17.83% 36.75% Player-Line 495 32.32% 40.00% 27.68% 127,223 31.00% 25.37% 43.63%

Relational Reasoning

In table 7, we present the detailed sample counts and per-sport percentages for both CourtSIBench and CourtSI. Overall, the data distribution in CourtSI-Bench across different sports is relatively balanced.

Notably, the Player-Zone subtask under Relational Reasoning primarily describes a player’s relative position within the near or far zones of the court. Since table tennis players do not stand

[Figure 162]

- Figure 11 | Court-Ext examples.

on the table surface itself, these instances are excluded to maintain the validity and consistency of the annotations.

#### C. Experiment Details

##### C.1. Evaluation on CourtSI-Bench Details

For data parsing, we use the Qwen3-8B model to extract answers from the original model outputs. The detailed prompt is provided below.

Please extract the answer from the following VLM response. Only provide the answer without any explanation. If the answer cannot be found in the VLM response, please output “None”. We will give you the original question and the VLM response. Please strictly follow the format to answer. <Original Question>: {Question} <VLM Response>: {VLM Answer} <Extracted Answer>:

For human evaluation, evaluators are provided with the image and the corresponding question through an interactive panel. The information provided to evaluators is identical to that given to the VLMs. In addition, the court size of each sport is provided as a reference.

In the localization task, the output is represented as 3D coordinates, which prevents the use of T-MRA for computing relative distance error. Therefore, we adopt a binary accuracy metric with a smooth threshold of 30cm. Notably, this threshold is greater than the combined 3D distance threshold, 15 ×

√3. If the 3D localization error exceeds this threshold, the prediction is assigned an accuracy of 0; otherwise, it is assigned an accuracy of 1.

##### C.2. In-depth Error Analysis Details

We specifically select the object-object and object-line subtasks in CourtSI-Bench, as these tasks are particularly susceptible to perspective ambiguity affecting the target entities. For the target

subjects in each QA pair, we compute the ratio between their 3D distance (in meters) and their 2D projected distance (in pixels) as a quantitative metric. A higher ratio indicates that two subjects are far apart in 3D space but appear close in the 2D image plane, reflecting stronger perspective distortion. Note that not all reasoning processes are equally affected by perspective. Nevertheless, this ratio serves as a useful proxy for measuring the degree of perspective-induced ambiguity. Overall, model performance degrades as the perspective effect increases. In certain points of fig. 6, performance temporarily improves, likely because some reasoning instances are less dependent on perspective cues, as previously mentioned.

##### C.3. CourtSI-Ext Details

Similar to RacketVision, we collect pickleball videos from YouTube. To ensure a relatively balanced image distribution, the collected videos include both men’s and women’s matches, as well as singles and doubles matches. All videos are provided at 1080p resolution and 25 fps, consistent with the CourtSI source data. Examples are illustrated in fig. 11.

##### C.4. Spatial-aware Commentary Generation Details

We extract spatial relationships from distance measurement cases in CourtSI-Bench, as these cases can be reliably evaluated using ground-truth numerical distance annotations. We use the following prompt to instruct baseline models to incorporate spatial relationships into sports commentary generation.

You are a professional live sports commentator with 3D spatial intelligence. Primary Task: Generate a vivid and natural sports commentary describing the scene. Spatial Intelligence Task: You need to smoothly incorporate the following spatial relationship into the commentary. Here is the relationship: {Relationship description of objects}. Remember to make the commentary engaging and informative, as if you are describing the scene to an audience watching the game live. Avoid directly answering the question or explaining the reasoning steps; instead, focus on creating a rich and immersive commentary that captures the essence of the moment. But the commentary must include the numerical value from the relationship.

We sample 100 commentaries from all distance measurement cases for the user study involving three volunteers. For each instance, volunteers compare the outputs from the fine-tuned model and the base model, and evaluate their relative quality in terms of linguistic quality and spatial awareness. Linguistic quality refers to the overall fluency, expressiveness, and natural integration of spatial information into the commentary. Spatial awareness refers to the correctness and proper use of the numerical distance information described in the spatial relationship.

#### D. Ethical Considerations

Our dataset is primarily constructed from images derived from RacketVision, which collects data from publicly available YouTube videos of international net sports games. We only use image frames for research purposes and do not attempt to identify individuals or infer sensitive

###### personal attributes. We encourage responsible use of our dataset and recommend that future users comply with ethical research standards in RacketVision.

