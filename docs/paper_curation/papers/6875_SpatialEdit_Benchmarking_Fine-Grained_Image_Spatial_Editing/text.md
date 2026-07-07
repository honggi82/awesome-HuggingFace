# arXiv:2604.04911v2[cs.CV]8Apr2026

## SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing

Yicheng Xiao1,2,3⋆, Wenhu Zhang4,2 ⋆, Lin Song2 , Yukang Chen5, Wenbo Li2, Nan Jiang2, Tianhe Ren1, Haokun Lin2, Wei Huang1, Haoyang Huang2,

Xiu Li3, Nan Duan2, and Xiaojuan Qi1

1 The University of Hong Kong

- 2 JD Explore Academy
- 3 Tsinghua University

4 The Hong Kong University of Science and Technology 5 The Chinese University of Hong Kong

Camera-Level Manipulation

[Figure 1]

[Figure 2]

Rotate the camera 90 degrees to the left

Rotate the camera 60 degrees downward

Zoom out from / Zoom in on the scene

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

Pitch Yaw Zoom In/Out

Object-Level Manipulation

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Rotate the car to show the <specific> view

Move the man into the red box

left rear-left rear rear-right

Input Output Input Output

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

front-left front front-right

right

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Rotation Translation & Scaling

Fig. 1: Illustration for image spatial editing. It comprises two components: (1) camera-centric view manipulation, including pitch, yaw, and zoom transformations; and (2) single-object manipulation, encompassing object rotation while preserving the background, as well as translation and scaling of objects specified via user-defined bounding boxes.

⋆ Equal contribution. Corresponding author.

Abstract. Image spatial editing performs geometry-driven transformations, allowing precise control over object layout and camera viewpoints. Current models are insufficient for fine-grained spatial manipulations, motivating a dedicated assessment suite. Our contributions are listed: (i) We introduce SpatialEdit-Bench, a complete benchmark that evaluates spatial editing by jointly measuring perceptual plausibility and geometric fidelity via viewpoint reconstruction and framing analysis. (ii) To address the data bottleneck for scalable training, we construct SpatialEdit-500k, a synthetic dataset generated with a controllable Blender pipeline that renders objects across diverse backgrounds and systematic camera trajectories, providing precise ground-truth transformations for both objectand camera-centric operations. (iii) Building on this data, we develop SpatialEdit-16B, a baseline model for fine-grained spatial editing. Our method achieves competitive performance on general editing while substantially outperforming prior methods on spatial manipulation tasks. All resources will be made public at https://github.com/EasonXiao888/SpatialEdit.

Keywords: Fine-Grained Spatial Editing · Geometry-Aware Benchmark · Synthetic Dataset Pipeline

### 1 Introduction

Modern image editing [16, 35, 40] is quickly moving beyond what to change (e.g., add, remove, replace and style) toward where and how to change it in 3D space. We refer to this capability as image spatial editing: editing an image by applying geometry-driven transformations rather than appearance edits. Concretely, spatial editing spans two complementary axes (Fig. 1): camera-centric view manipulation (e.g., yaw, pitch, and zoom) and object-centric manipulation (e.g., translate/scale an object within a user-specified box, or rotate an object to a desired canonical view). This functionality is increasingly central to world-modeling and embodied perception pipelines, where controllable viewpoint change and object reconfiguration are prerequisites for interactive content creation, simulation, and downstream 3D reasoning.

Despite rapid progress in image generation following user instructions [16, 36,46,56], precise spatial control remains brittle. Existing systems fall into three common failure modes. First, many spatially-conditioned world-model or videogeneration pipelines require expert inputs such as full 6-DoF camera trajectories [34,54], creating a steep usability barrier for typical image-editing scenarios.

Object Object Object Camera VLM Precise Translation Scaling Rotation Manipulation Metric Metric

Benchmark

ImgEdit [65] ✗ ✗ ✗ ✗ ✔ ✗ GEdit [32] ✔ ✗ ✗ ✗ ✔ ✗ CEdit [46] ✔ ✗ ✗ ✔ ✔ ✗ SpatialEdit-Bench ✔ ✔ ✔ ✔ ✔ ✔

###### Table 1: Characteristics comparison with other editing benchmarks.

Second, strong general-purpose instruction-based editors often excel at semantic edits [32, 56, 60], but frequently miss metric or viewpoint intent– e.g., “rotate the camera 90◦ to the right” or “rotate the object to show its front-right side”– yielding outputs that look plausible but are spatially incorrect. Third, some methods [3,30,46] incorporate spatial reasoning but are typically narrow (one operation or one setting) and do not generalize across the diverse operation set demanded by real users. Together, these limitations suggest a gap between “semantic alignment” and faithful geometric compliance.

A key reason this gap persists is that evaluation for spatial editing remains underdeveloped as shown in Tab. 1. When metrics cannot reliably distinguish “looks right” from “is right,” model iteration becomes noisy and progress is hard to measure. To address this, we introduce SpatialEdit-Bench, a benchmark that covers both object-level and camera-level spatial editing, together with geometry-aware evaluation tailored to viewpoint changes. Beyond detectordriven composition and framing analysis (our Framing Error: FE), we quantify Viewpoint Error (VE) by reconstructing the camera pose in 3D space, enabling a direct check of whether the edited result matches the intended geometric transformation. In controlled validation with fine-grained pose variations, these metrics show substantially higher reliability than vision-language-based judging used in prior work [32,46], underscoring the necessity of geometry-sensitive evaluation for diagnosing true spatial capability.

Benchmarking alone, however, is not enough– training data is the real bottleneck for fine-grained spatial editing. Such data is difficult to obtain at scale because it requires paired images with known geometric transformations, consistent object identity across edits, and faithful, unambiguous instructions, all while covering a wide range of scenes, object categories, and camera configurations. To address this, we build a scalable and controllable data engine in Blender [23] to synthesize paired supervision together with corresponding textual instructions. For object-level spatial editing, we render a large collection of GLB assets from eight preset viewpoints to generate source images. We then use VLMs [1, 11] to verify the availability of a front view and assign object names, while SAM3 [10] segments each object to produce mask labels. Next, we generate diverse backgrounds with a high-quality text-to-image model [16] and inpaint the rendered object into these backgrounds, producing realistic edited images with ground-truth spatial intent. For camera-level editing, we curate a rich set of indoor and outdoor scenes, select salient objects as focal targets, and systematically sample camera poses around them by varying yaw, pitch, and zoom. We further use a VLM to generate paired images with accurate viewpoint changes and to produce natural language instructions that support flexible camera-centric edits. As shown in Fig. 2, the resulting SpatialEdit-500k achieves high diversity and a well-balanced distribution across task types. Building on this data, we develop SpatialEdit-16B, a fine-grained spatial editing model that combines a pretrained multimodal encoder [4] with an MM-DiT decoder [13]. We first ensure strong general editing behavior via pretraining on open-source editing data [53], and then specialize using parameter-efficient fine-tuning (LoRA) on

SpatialEdit-500k. Empirically, this strategy yields a model that is competitive on general editing benchmarks while substantially improving on spatial tasks across object manipulation and camera control-precisely the regime where prior systems most often fail.

To evaluate our approach, we use both the publicly available GEdit [32] and our newly introduced SpatialEdit-Bench for general and spatial editing tasks. Specifically, while maintaining comparable performance on general editing (7.52 on GEdit-Bench), our method acquires precise editing capabilities through continued training (as shown in Fig. 1), surpassing the current open-source stateof-the-art model, LongCatImage-Edit, by 0.300 and 0.127 points on moving and rotation scores, respectively, while achieving the lowest error in camera control. Furthermore, video-based world models [26,50] remain significantly inferior to image-based spatial editing models in performing fine-grained spatial manipulation guided by text instructions. Additionally, our model can also serve as a practical enhancement tool for single-view reconstruction.

### 2 Related Work

Image Editing and Generative Models. Diffusion-based generative models [9,41,56] have greatly improved the fidelity and controllability of image editing. Instruction-based editing [12, 42, 59] modifies images according to natural language instructions while preserving overall semantics, but relies on large-scale instruction-faithful supervision. Early pipelines such as InstructP2P [8] combine prompt engineering with diffusion editing operators like Prompt-to-Prompt [22] built on latent diffusion [38], while later works expand training data through human edits, inpainting-based synthesis, compositing, and expert-model orchestration [15,55,60,67]. Recent unified frameworks further support richer instructions and multi-task editing [58,60]. Beyond text-only conditioning, referencebased methods improve controllability [14,39], whereas later methods introduce lightweight visual adapters [29, 64]. More recent systems encode reference images as visual tokens within DiT [5,24,37,44,56,60,61], with transformer-based backbones such as DiT improving scalability and conditioning flexibility [37].

Spatially-Aware Visual Manipulation. Recent progress in spatially conditioned generative modeling shows that explicit spatial control can be learned at scale. Prior work has explored controllable viewpoint manipulation through camera-motion or 6-DoF conditioning (MotionCtrl [54], CameraCtrl [21], CVD [27]), camera-aware DiT architectures (AC3D [2]), and more general control interfaces (OminiControl [45]). Others incorporate geometric cues such as dense point tracks [25,63], enabling geometry-aware generation in GS-DiT [7] and Diffusionas-Shader [19]. Camera trajectory manipulation and novel-view synthesis using simulated data (Kubric [18]) are explored in GCD [49], while Recapture [66] studies adaptation-based camera control for real images. However, evaluation of spatial manipulation remains limited, as existing benchmarks rely on coarse metrics or semantic alignment checks. We address this by introducing a unified benchmark with ground-truth geometric annotations and geometry-aware metrics that explicitly measure transformation accuracy and viewpoint correctness.

     %)       

 (        (        

In door

Out door

Total

 %    )       

 (        

       )       

[Figure 39]

     ( %  )       

  %     %  

    %         

- (a) Distribution of Camera Operation Tasks.
- (b) Statistics of the Object Moving Task. (c) Distribution of the Overall Dataset

[Figure 40]

[Figure 41]

[Figure 42]

  (   

  )   %       

    (            ( )             )          )             )       

- Fig. 2: Statistics of SpatialEdit-500k. (a) Distribution of camera-level data across seven sub-tasks in outdoor and intdoor scenes, where Y, P, and D denote Yaw, Pitch, and Distance, respectively. (b) Aspect ratio distribution of bounding boxes for the moving task at the object level. (c) Object category statistics across the entire dataset.

### 3 Image Spatial Editing

#### 3.1 Revisiting Image Spatial Manipulation

Image spatial manipulation has traditionally been formulated with explicit geometric constraints (e.g., view synthesis and pose-conditioned generation), but modern instruction-following editing models are expected to perform it from language supervision alone, exposing a key mismatch between semantic alignment and geometric compliance. In practice, outputs often look plausible yet violate metric intent, especially for fine-grained camera operations (yaw/pitch/zoom) and canonical object reorientation. Camera-centric view manipulation requires globally coherent re-projection and framing consistency, whereas object-centric manipulation demands localized, identity-preserving transformations disentangled from the background. This motivates revisiting spatial manipulation as a first-class image editing capability with (i) a unified task definition spanning camera- and object-level control, (ii) geometry-aware evaluation that can distinguish “looks right” from “is right” (e.g., viewpoint- and framing-sensitive metrics), and (iii) scalable paired supervision with unambiguous transformation intent. Consequently, it directly guides our benchmark, data engine, and model design.

#### 3.2 Task Definition

To bridge the gap between semantic intent and geometric precision, we categorize spatial editing into two primary axes: object-centric manipulation and cameracentric view control.

Object-Level Spatial Manipulation. We aim to edit individual objects within the image, including translation (repositioning), scaling (resizing), and rotation

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

VLM Post-Processing VLM Post-Processing

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Zoom Pitch Yaw

Rotation Branch Moving Branch

Model Pool

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Rendering Engine

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Pre-Processing & Filter

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

VLM SAM3

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

Inpaint

Applied

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Rendering Engine

[Figure 89]

[Figure 90]

[Figure 91]

…

[Figure 92]

[Figure 93]

[Figure 94]

Just for Better display

Scene Pool

Camera Position

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

Approved Bounding

[Figure 105]

[Figure 106]

…

[Figure 107]

[Figure 108]

Background Generator

Box Generator

(a) Object-Level Data Engine (b) Camera-Level Data Engine

- Fig. 3: SpatialEdit-500k data generation pipeline. We leverage Blender to synthesize both objects and scenes, while preprocessing 3D assets using SAM3 and a vision-language model. The object-level engine constructs two inpainting-based data branches to generate object transformations, including rotation, translation, and scaling. The camera-level engine produces viewpoint transformation data by sampling different camera poses, resulting in variations in yaw, pitch, and zoom.

(orientation change) of specific entities while maintaining scene coherence. To enable granular control, we employ red bounding boxes to define the target translation and scaling operations. Users can constrain object movement and resizing either through textual instructions or by directly drawing a target rectangle on the canvas. For orientation, we discretize object orientation into eight canonical viewpoints: right, front-right, front, front-left, left, rear-left, rear, and rear-right.

Camera-Level View Control. This task involves manipulating the global imaging perspective to synthesize novel viewpoints without altering the underlying scene content. We parameterize this space through three degrees of freedom:(i) Pitch & Yaw: We discretize vertical tilt (pitch) at 15° intervals and horizontal panning (yaw) at 45° increments. This grid provides comprehensive coverage of practical camera trajectories. (ii) Zoom: Focal length variations are modeled to simulate movement toward or away from the focal point. By unifying these parameters, our framework elevates the editing task from a 2D image-toimage mapping to a geometry-aware transformation, effectively modeling the scene as a 3D environment with explicitly defined camera and object states.

#### 3.3 SpatialEdit-500k Dataset

In this section, we will mainly introduce how we collect and create such a dataset to support image spatial editing.

Object-Centric Data Generation Pipeline. To construct a high-quality object-centric dataset with controllable viewpoints and spatial variations, we design a multi-stage data curation pipeline. As shown in the left of Fig. 3, our pipeline progressively filters, augments, and annotates 3D assets to ensure geometric consistency, recognizability, and spatial diversity. The overall procedure consists of multiple stages, as described in the following.

We begin with variant GLB assets curated by TexVerse [48] and render them in Blender [23] under a predefined canonical front-facing camera configuration, fixing camera intrinsics and object alignment to ensure consistent nominal frontal views. To guarantee view correctness and remove ambiguous assets, we employ an advanced Vision-Language Model [11] to verify that each rendered image corresponds to a valid frontal view and exhibits minimal side-view characteristics, discarding assets that fail these criteria. For each retained GLB model, we render eight uniformly distributed viewpoints around the object while maintaining consistent camera intrinsics. We then apply the Segment Anything Model (SAM3) to obtain object masks for each view, using the first sentence of the TexVerse caption as the textual prompt to verify correct object localization and segmentation; views failing this check are removed. To introduce spatial diversity, we generate eight additional renderings per valid view with randomized translations and scaling factors in Blender, and re-apply SAM3 to verify that the perturbed objects remain visible and properly contained within the image frame, retaining samples with at least one valid rendering. For each canonical front view, we employ Nano-Pro [16] to synthesize a semantically compatible background image conditioned on the object’s appearance. We composite these backgrounds with the validated multi-view images and their spatially perturbed variants, blending foreground objects while preserving geometric consistency. Finally, we project the known ground-truth 3D bounding boxes into the image plane to obtain precise 2D bounding box annotations for each transformed object.

Camera-Centric Data Generation Pipeline. As depicted in the right pane of Fig. 3, we established a high-fidelity 3D simulation environment to systematically sample camera trajectories for viewpoint manipulation. We first build a large-scale pool of high-quality 3D scenes containing diverse indoor and outdoor layouts with semantically coherent object arrangements and physically plausible geometry. For each scene, we manually select Nscenetarget visually salient objects as camera focus targets, ensuring that the selected objects are recognizable and sufficiently exposed within the scene. These objects serve as anchors for defining camera viewpoint changes. We parameterize camera motion relative to the focus object using three degrees of freedom: yaw, pitch, and distance, corresponding to common camera operations such as horizontal orbiting, vertical tilting, and zooming. Using Blender [23], we systematically sample camera poses around each focus object by traversing predefined ranges of these parameters while keeping camera intrinsics fixed. Each sampled pose is used to render a candidate scene image, producing diverse viewpoints while preserving the underlying scene configuration and object layout.

To ensure dataset reliability, we apply a dual-branch quality filtering pipeline to remove invalid renderings. One branch employs a YOLO-based [51] detector to verify the visibility of the focus object and discard images where the object is missing, severely occluded, or truncated. The other branch uses the visionlanguage model QwenVL-30B to assess semantic and geometric plausibility, filtering out renderings that exhibit mesh interpenetration, extreme or unnatural viewpoints, or visually meaningless scene compositions. For each valid rendering, we record the raw camera parameters (θy,θp,d) and construct viewpoint pairs by sampling two poses associated with the same focus object, computing their relative transformation (∆θy,∆θp,∆d). These transformations are first converted into templated camera-editing instructions describing the viewpoint change. We further provide human-like instructions alongside a concise target scene description, which serves as a prompt enhancer to reduce the difficulty of geometric editing. The resulting dataset contains high-quality image pairs with controlled camera transformations, associated camera parameters, and diverse natural language instructions, enabling systematic evaluation of camera-centric spatial editing capabilities.

#### 3.4 SpatialEdit-Benchmark

To rigorously evaluate image spatial editing, we construct a benchmark focusing on spatial transformation tasks, jointly measuring geometric accuracy, semantic consistency, and structural preservation. Unlike conventional editing benchmarks centered on appearance changes, ours emphasizes spatial operations such as object scaling, translation, rotation, and camera viewpoint adjustment.

Evaluation Metric of Object-Level Task. Since we divide object-level manipulation into object moving and rotation, for the object moving branch, we can accurately evaluate it using only the detection model.

Moving Score. To quantitatively verify whether the predicted object location satisfies the prescribed absolute spatial constraint, we first employ a detection model to calculate the IoU. However, geometric correctness alone is insufficient, as spatial translation may introduce semantic artifacts or contextual inconsistencies. Therefore, we further introduce a VLM to compute an object consistency score Soc ∈ [0,1], which evaluates both subject integrity and environmental coherence after transformation. Then the moving score is defined as:

MS = IoU(bgt,bpred) · Soc. (1)

The geometric mean formulation enforces a multiplicative coupling between spatial accuracy and semantic fidelity, thereby penalizing imbalance.

Rotation Score. For the object rotation branch, Ipred denotes the image after applying a viewpoint transformation parameterized by yaw θ and pitch ϕ. Since

rotation correctness is inherently viewpoint-sensitive and difficult to localize geometrically, we employ an advanced closed-source VLM [11] to estimate the viewpoint correctness score Sview, which measures whether the rendered perspective matches the specified angular configuration. To further prevent appearance drift or structural distortion introduced during rotation, we additionally compute a consistency score Scons by the same VLM. The final rotation score is defined as:

RS = Sview · Scons. (2) This multiplicative design also enforces simultaneous viewpoint correctness and semantic continuity, preventing trivial viewpoint hallucination that disregards object identity or scene plausibility.

Evaluation Metric of Camera-Level Task. Given a triplet of source, groundtruth, and predicted views (Isrc,Igt,Ipred), we aim to evaluate camera-level editing from two complementary aspects: (i) Framing Error (FE) in the image plane (the focus object should remain visible with correct composition), and (ii) Viewpoint Error (VE) in terms of camera extrinsics (the predicted camera pose should match the target pose up to scene scale). We therefore adopt a dual-metric protocol, reporting a detector-based metric with YOLO [51] and a geometry-aware metric with VGGT [52].

Viewpoint Error. Specifically, to measure viewpoint correctness in a geometryaware manner, we employ VGGT [52], which is a feed-forward transformer that directly infers key 3D attributes of a scene, including camera parameters (intrinsics/extrinsics), depth maps, point maps, and 3D point tracks, with the input of a single or a set of images from various viewpoints. It models the scene in a globally consistent 3D representation rather than relying on purely 2D appearance cues. In our evaluation, VGGT takes (Isrc,Igt,Ipred) as inputs and returns estimated world-to-camera extrinsics:

( Rsrc,tsrc), ( Rgt,tgt), ( Rpred,tpred) = fVGGT(Isrc,Igt,Ipred)), (3) from which we compute camera centers in world coordinates:

C = − R⊤t, R = ( Rsrc, Rgt, Rpred), t = (tsrc,tgt,tpred). (4)

We then calculate a baseline-normalized translation error (to remove sensitivity to global scene scale) and and a rotation error based on the geodesic distance on SO(3), which are formulated as:

ϵxyz = ∥Cpred − Cgt∥2 ∥Cgt − Csrc∥2 + ε

,

1 90

dgeo Rpred, Rgt ,

ϵrot =

Tr(x⊤1 x2) − 1 2 ·

180 π

dgeo(x1,x2) = arccos

.

(5)

Finally, we aggregate them into a single pose error: VE =

- 1

- 2

(ϵxyz + ϵrot), (6) where lower VE indicates more accurate camera viewpoint editing. Framing Error. Calculating with Viewpoint Error alone may not reflect whether

the edited output preserves meaningful spatial layout under camera motion (e.g., objects might drift to incorrect positions or the scene structure may become distorted). We thus introduce an object-centric spatial consistency metric (FE: Framing Error) based on detection. We first introduce angle consistency. Let

Ogt = {ogt1 ,...,ogtn } and Opred = {opred1 ,...,opredm } denote the sets of detected objects in Igt and Ipred, respectively. For each object, we compute a ray direction [20, 43] from the image center to the object’s bounding box center (u,v):

[(u − cx)/f, (v − cy)/f, 1]⊤ ∥[(u − cx)/f, (v − cy)/f, 1]∥

, (7)

r(u,v) =

where (cx,cy) is the image center and f is the focal length. We establish correspondences between Ogt and Opred via the Hungarian Matching [28] algorithm, minimizing the sum of ray angles and area ratios. Let M = {(i,j)} be the set of matched object pairs. We compute the average ray angle difference ϵrag as:

1 |M|

180 π

arccos rgti · rpredj ×

, (8)

ϵrag =

(i,j)∈M

where a lower value indicates better spatial alignment between the predicted and target object layouts. Additionally, we verify whether the predicted image exhibits the correct scale change relative to the source for zoom editing commands. Let Mzoom be matched object pairs between Isrc and Ipred. Given commandspecified distance change ∆d (negative for zoom-in, positive for zoom-out), we compute the zoom direction error as follows:

log |bpredj | |bsrci |

- 1

- 2

ϵzde = I Fi,jmed

× ∆d > 0 , (9)

where |b| denotes bounding box area, I[·] is the indicator function and Fi,jmed indicates the median function. This binary metric ensures that zoom-in commands

(∆d < 0) produce larger objects (slog > 0) and vice versa. Finally, the framing error can be formulated as:

- 1

- 2

FE =

(ϵrag + ϵzde). (10)

### 4 Image Spatial Editing Model

As shown in Fig. 4, we adopt a cascaded editing pipeline [46,56,62]. Given an instruction and a reference image, a vision language model produces semantic embeddings as global conditioning. The image is encoded into a VAE latent, and an

Object Camera Object Camera

Method

Moving Rotation Viewpoint Framing Overall Overall Score ↑ Score ↑ Error↓ Error↓ Score ↑ Error ↓

World Model

ViduQ2-Turbo [50] – – 1.022 0.771 – 0.897 Kling-V2.5 [26] – – 1.051 0.733 – 0.892

Closed-Source Image Model

Nano-Banana [16] 0.099 0.420 0.845 0.708 0.260 0.777 Seedream4 [40] 0.163 0.482 0.839 0.701 0.323 0.770

Open-Source Image Model

QwenImageEdit [56] 0.311 0.531 0.922 0.692 0.421 0.807 Edit-R1 [30] 0.306 0.562 0.959 0.688 0.434 0.824 LongCatImage-Edit [46] 0.373 0.505 0.802 0.684 0.439 0.743 SpatialEdit-PT (Baseline) 0.186 0.489 0.890 0.719 0.338 0.804 SpatialEdit 0.673 0.632 0.243 0.527 0.653 0.385

Table 2: Performance comparison on proposed SpatialEdit-Bench.

MMDiT [13] denoises it under multimodal guidance to obtain the edited latent, which is decoded by the VAE to the final output.Training proceeds in two stages:

- (1) adapt the model to image editing via fine-tuning on public editing data, and

[Figure 109]

[Figure 110]

Qwen3VL

ViT Features

Time Step

Noise Tokens

[Figure 111]

[Figure 112]

MMDiT Block

[Figure 113]

MMDiT Block

…

C

[Figure 114]

[Figure 115]

VAE Features

[Figure 116]

[Figure 117]

VAE Encoder

Text Features

[Figure 118]

Replace the image style with fantasy art.

[Figure 119]

×𝑁

VAE Decode

Fig. 4: Overview of SpatialEdit.

- (2) specialize in image spatial editing scenario with LoRA post-tuning on our curated dataset, improving transformation control while preserving general priors.

### 5 Experiments

#### 5.1 Training Details

We pre-train the model on open-source editing datasets [53] and proprietary internal data, explicitly excluding spatial editing samples (see supplementary for details). Training uses the AdamW [33] optimizer with β1 = 0.9, β2 = 0.95, a learning rate of 1 × 10−4, and a linear warmup over the first 1,000 iterations.

#### 5.2 Quantitative Results

Image Spatial Editing Performance. As shown in Tab. 2, our SpatialEdit achieves the best overall performance across both object-level and camera-level metrics. For object-level tasks, our SpatialEdit significantly surpasses all baselines in object moving score with 0.673, while maintaining a competitive object rotation score (0.632). On the other hand, SpatialEdit yields the lowest viewpoint error (0.243) and framing error (0.527), boosting the current SOTA

###### SpatialEdit (Ours)

ViduQ2-Turbo Nano-Banana Seedream4.0 QwenImageEdit Edit-R1 LongCat-Edit

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Please zoom out and adjust the camera to a higher vantage point, maintaining a topdown perspective

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Rotate the camera 135 degrees to the left to shift the view from the rear of the cars to the front

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Please rotate the camera to a topdown view by increasing the pitch by 60 degrees

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Please rotate the camera 45 degrees to the right and tilt it up by 30 degrees

Multi-Modal User Input Various Model Output

- Fig. 5: Comparison of camera view manipulation across various methods.

method, LongCatImage-Edit by 0.358 in the overall camera error metric. We further evaluate closed-source world models on precise camera viewpoint control from text instructions by sampling the final frame of the generated videos. The results show that their performance is weaker than that of mainstream image editing models, likely due to the challenge of maintaining consistent camera motion during video generation.

General Editing Performance. To validate the general editing capability of our model during pre-training, we evaluate it on GEdit, as shown in Tab. 5. Among open-source models, SpatialEdit achieves competitive performance, providing a strong foundation for subsequent fine-tuning on spatial editing tasks.

#### 5.3 Ablation Studies

Training data combinations. As shown in Tab. 3, multi-task mixed training converges more reliably than single-task training. Mov+Rot boosts both object metrics (0.657/0.632), and adding Cam further improves Mov and lowers camera error (Mov+Cam: 0.665, Cam: 0.402). Training on all three tasks yields the best overall trade-off (Mov: 0.673, Rot: 0.632, Cam: 0.385), indicating positive transfer from shared spatial supervision.

Camera evaluation metrics. We render n fine-grained views of the same scene, fix one as the ground-truth, and treat the remaining views as pseudo edits with a known ordering. Each metric scores and ranks these views; we then compute Spearman correlation between the predicted and true rankings (Table 4). VE attains the highest correlation, followed by FE, and both substantially outperform GPT, supporting the reliability of VE for camera evaluation.

###### SpatialEdit (Ours)

###### Nano-Banana Seedream4.0 QwenImageEdit

Edit-R1 LongCat-Edit

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Move an antique key into the red box and finally remove the red box.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Move an orange basketball into the red box and finally remove the red box.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Move the white safety helmet into the red box and finally remove the red box.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Rotate the red car to show the right side view.

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Rotate the red car to show the front view.

\

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Rotate the black and white sneaker to show the rear view.

\

Multi-Modal User Input Various Model Output

- Fig. 6: Comparison of object-level manipulation across various methods.

#### 5.4 Qualitative Results

- Fig. 5 compares camera view manipulations (zoom-out, yaw/pitch changes, and rotation/tilt). SpatialEdit follows the requested viewpoint shifts more faithfully while better preserving scene geometry and reducing distortions (e.g., texture stretching, boundary drift, and hallucinations) than prior baselines. Moreover,
- Fig. 6 evaluates object-level edits (moving items into a target region and rotating objects to specified views). Competing methods often leave artifacts or alter the background, whereas SpatialEdit performs cleaner edits with higher object fidelity and stronger background preservation, yielding a better accuracy– preservation trade-off.

#### 5.5 Enhancement Tool for Single-view Reconstruction

We propose a pipeline that uses SpatialEdit to improve 3D reconstruction when multi-view observations are unavailable. As shown in Fig. 7, single-view inputs suffer from depth-scale ambiguity and missing geometric cues. By editing the camera to synthesize novel viewpoints, our approach adds geometric constraints, leading to more accurate, structurally consistent, and detailed reconstructions.

GEdit-Bench-EN↑

Mov. Rot. Cam. Score.↑ Score.↑ Error.↓

Model

Mov. Rot. Cam.

SC PQ O Closed Source

✓ 0.653 - ✓ - 0.628 -

Gemini 2.0 [11] 6.73 6.61 6.32 GPT Image 1 [53] 7.85 7.62 7.53 Nano Banana [16] 7.86 8.33 7.54 Seedream 4.0 [40] 8.24 8.08 7.68

✓ - - 0.395

✓ ✓ 0.657 0.632 -

✓ ✓ 0.665 - 0.402 ✓ ✓ ✓ 0.673 0.632 0.385

- Table 3: The impact of training with different data combinations. Mov, Rot, and Cam represent object moving, object rotation, and camera operation tasks, respectively.

FE VE GPT4.1 Spearman Score 0.659 0.932 0.445

- Table 4: The effectiveness comparison across three type of metrics in camera evaluation using Spearman Correlation.

Open Source

UniWorld-v1 [31] 4.93 7.43 4.85 MindOmni [62] 6.53 6.93 5.98 OmniGen2 [57] 7.16 6.77 6.41 FLUX.1 Kontext [6] 6.52 7.38 6.00 BAGEL [12] 7.36 6.83 6.52 Step1X-Edit [32] 7.66 7.35 6.97 Qwen-Image-Edit [56] 8.00 7.86 7.56 LongCat-Edit [46] 8.18 8.00 7.64 SpatialEdit 8.09 7.80 7.52

Table 5: Performance comparison on GEdit-Bench [32].

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

3D cloud map with one camera view

[Figure 200]

[Figure 201]

[Figure 202]

3D cloud map with our augmented views

Fig. 7: Serving as an enhancement tool for single-view reconstruction.

### 6 Conclusion

We proposed a fine-grained spatial image editing paradigm, where edits are composed by object manipulations and explicit geometric control of camera viewpoints. For rigorous assessment, we introduce SpatialEdit-Bench, which improves evaluation reliability by jointly measuring perceptual plausibility and geometric fidelity via viewpoint reconstruction and compositional analysis. To mitigate the data bottleneck, we build SpatialEdit-500k, a controlled Blender-based dataset with diverse scenes and 3D assets. Leveraging this data, we develop SpatialEdit16B, a strong baseline that remains competitive on general editing while substantially advancing performance on challenging spatial manipulation tasks. We hope our benchmark, dataset, and model will support reproducible progress and motivate future work that more tightly couples geometric inference with highquality image synthesis.

Supplemental Materials

- A Implementation Details

We pre-trained the model using the open-source general image editing dataset gpt-image-edit and proprietary internal data, explicitly excluding all spatially edited samples. During pre-training, we used the AdamW optimizer with parameters β1 = 0.9 and β2 = 0.95, and a learning rate of 1 × 10−4. A linear warm-up schedule was applied for the first 1000 iterations before transitioning to standard decay. In the post-training phase, we fine-tuned the model on our SpatialEdit-500k dataset using LoRA with rank 16 and α = 16, initializing the LoRA parameters with a Gaussian distribution.

Method

Camera Camera Viewpoint Framing Overall

Error↓ Error↓ Error ↓ World Model

Veo3.1 [17] 1.351 0.749 1.050 ViduQ2-Turbo [50] 1.022 0.771 0.897 Kling-V2.5 [26] 1.051 0.733 0.892 ReCamMaster [3] 0.755 0.720 0.738 LingBot-World [47] 0.696 0.701 0.699

Our Image Spatial Editing Model SpatialEdit 0.243 0.527 0.385

Table 6: Performance comparison on proposed SpatialEdit-Bench.

- B More World Model Results

We compare the performance of multiple video world models on precise camera viewpoint editing, including three closed-source models (ViduQ2-Turbo [50], Kling-V2.5 [26], Veo3.1 [17]) and two open-source models (LingBot-World [47] and ReCamMaster [3]), as summarized in Tab. 6.

- C Metrics in SpatialEdit-Bench

Algorithm 1 and Algorithm 2 provide a clearer illustration of the metric calculation process used in SpatialEdit-Bench.

- D More Qualitative

We provide more additional visualizations, as shown in Algorithm 1 and Algorithm 2. For object-level manipulation tasks, we compare various open-source and closed-source editing models, while for camera-level manipulation tasks, we focus on evaluating the performance of different world models.

##### Algorithm 1 Object-Level Spatial Editing Evaluation

Input: Images Isrc, Ipred; BBox bgt; Detector Y; VLM V Detection: bpred ← Detect(Y, Ipred)

Moving Score:

2: IoU ← IoU(bgt, bpred)

Soc ← V(Isrc, Ipred) (object consistency) 4: MS ←

√IoU · Soc Rotation Score:

6: Sview ← V(Ipred, θ, ϕ) (view correctness)

Scons ← V(Iscr, Ipred) (appearance consistency) 8: RS ←

√Sview · Scons Object Overall Score ← MS+2 RS

Output: MS (moving score), RS (rotation score), Object Overall Score

##### Algorithm 2 Camera-Level Spatial Editing Evaluation

Input: Images Isrc, Igt, Ipred; command ∆d; detector Y; VGGT model fVGGT Pose Estimation: ( R∗, t∗) ← fVGGT(Isrc, Igt, Ipred) for ∗ ∈ {src, gt, pred} Detection: O∗ ← Y(I∗) for ∗ ∈ {src, gt, pred}

###### Viewpoint Error:

2: Camera Center: C∗ ← − R⊤∗ t∗ Translation Error: ϵxyz ← ∥∥CCpred−Cgt∥2

gt−Csrc∥2+ε

4: Rotation Error: ϵrot ← 901 dgeo( Rpred, Rgt)

VE ← 12(ϵxyz + ϵrot) 6: Framing Error:

Ray: r(u, v) ← norm([(u − cx)/f, (v − cy)/f, 1]) 8: Match: Hungarian matching between Ogt and Opred cost cij = ∠(ri, rj) + λ| ln(ai/aj)| 10: ϵrag ← |M|1 (i,j)∈M arccos(rgti ·rpredj )180π

Mzoom ← Match(Osrc, Opred) 12: ϵzde ← I med 2 1 ln |b

pred j |

|bsrci | ·∆d > 0 FE ← 21(ϵrag + ϵzde)

14: Camera Overall Error ← VE+2FE Output: VE (viewpoint error), FE (framing error), Camera Overall Error

###### SpatialEdit (Ours)

###### Nano-Banana Seedream4.0 QwenImageEdit

Edit-R1 LongCat-Edit

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Move the white floral arrangement into the red box and finally remove the red box.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Move the baby polar bears into the red box and finally remove the red box.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Move the moon into the red box and finally remove the red box.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Move the light gray chair into the red box and finally remove the red box.

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

Move the man into the red box and finally remove the red box.

\

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Move the red ship into the red box and finally remove the red box.

\

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Rotate dancing clay figure to show the left side view.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Rotate an armored warrior to show the front right side view.

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Rotate blue armchair to show the rear left side view.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Rotate the teal armchair to show the right side view.

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Rotate the cruise ship to show the rear right side view.

\

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Rotate the seated man to show the rear right side view.

\

Multi-Modal User Input Various Model Output

###### Fig. 8: Comparison of object-level manipulation across various methods.

QwenImageEdit LongCat-Edit

###### SpatialEdit (Ours)

###### Veo3.1 LingBot-World ReCamMaster

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Please rotate the camera 135 degrees to the left to shift the view

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Rotate the camera 135 degrees to the right and tilt it upwards by 60 degrees

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Rotate the camera 180 degrees to the right and tilt it up by 45 degrees

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

Please zoom in and focus on the coffee table and the sofa

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Please zoom out and rotate the camera 90 degrees to the right

\

Please rotate the camera 45 degrees downward to transition from a topdown view to an eyelevel perspective

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

\

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Please rotate the camera upwards by 45 degrees

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Please rotate the camera 45 degrees to the left

Multi-Modal User Input Various Model Output

###### Fig. 9: Comparison of camera-level manipulation across various methods.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Bahmani, S., Skorokhodov, I., Qian, G., Siarohin, A., Menapace, W., Tagliasacchi, A., Lindell, D.B., Tulyakov, S.: Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673 (2024)
- 3. Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14834–14844 (2025)
- 4. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025)

- 5. Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., et al.: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints pp. arXiv–2506 (2025)
- 6. Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., et al.: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints pp. arXiv–2506 (2025)
- 7. Bian, W., Huang, Z., Shi, X., Li, Y., Wang, F.Y., Li, H.: Gs-dit: Advancing video generation with pseudo 4d gaussian fields through efficient dense 3d point tracking. arXiv preprint arXiv:2501.02690 (2025)
- 8. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)
- 9. Cao, S., Chen, H., Chen, P., Cheng, Y., Cui, Y., Deng, X., Dong, Y., Gong, K., Gu, T., Gu, X., et al.: Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951 (2025)
- 10. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)
- 11. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- 12. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025)
- 13. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 14. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)
- 15. Ge, Y., Zhao, S., Li, C., Ge, Y., Shan, Y.: Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007 (2024)

- 16. Google: Gemini 2.5 flash & 2.5 flash image model card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-2-5Flash-Model-Card.pdf, 2025. (2025)
- 17. Google: Introducing veo 3, our video generation model with expanded creative controls – including native audio and extended videos. https://deepmind.google/models/veo/ (2025)
- 18. Greff, K., Belletti, F., Beyer, L., Doersch, C., Du, Y., Duckworth, D., Fleet, D.J., Gnanapragasam, D., Golemo, F., Herrmann, C., Kipf, T., Kundu, A., Lagun, D., Laradji, I., Liu, H.T.D., Meyer, H., Miao, Y., Nowrouzezahrai, D., Oztireli, C., Pot, E., Radwan, N., Rebain, D., Sabour, S., Sajjadi, M.S.M., Sela, M., Sitzmann, V., Stone, A., Sun, D., Vora, S., Wang, Z., Wu, T., Yi, K.M., Zhong, F., Tagliasacchi, A.: Kubric: a scalable dataset generator. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- 19. Gu, Z., Yan, R., Lu, J., Li, P., Dou, Z., Si, C., Dong, Z., Liu, Q., Lin, C., Liu, Z., et al.: Diffusion as shader: 3d-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847 (2025)
- 20. Hartley, R., Zisserman, A.: Multiple view geometry in computer vision. Cambridge university press (2003)
- 21. He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., Yang, C.: Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101

(2024)

- 22. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)
- 23. Hess, R.: Blender foundations: The essential guide to learning blender 2.5. Routledge (2013)
- 24. Huang, L., Wang, W., Wu, Z.F., Shi, Y., Dou, H., Liang, C., Feng, Y., Liu, Y., Zhou, J.: In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775

(2024)

- 25. Karaev, N., Rocco, I., Graham, B., Neverova, N., Vedaldi, A., Rupprecht, C.: Cotracker: It is better to track together. In: European Conference on Computer Vision. pp. 18–35. Springer (2024)
- 26. Kling: Kling. Kling. Accessed Sept.30, 2024 [Online] https://kling.kuaishou. com/en (2024), https://kling.kuaishou.com/en
- 27. Kuang, Z., Cai, S., He, H., Xu, Y., Li, H., Guibas, L., Wetzstein, G.: Collaborative video diffusion: Consistent multi-video generation with camera control. arXiv preprint arXiv:2405.17414 (2024)
- 28. Kuhn, H.W.: The hungarian method for the assignment problem. Naval Research Logistics Quarterly 2(1–2), 83–97 (1955)
- 29. Li, D., Li, J., Hoi, S.: Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. NeurIPS (2023)
- 30. Li, Z., Liu, Z., Zhang, Q., Lin, B., Wu, F., Yuan, S., Yan, Z., Ye, Y., Yu, W., Niu, Y., et al.: Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888 (2025)
- 31. Lin, B., Li, Z., Cheng, X., Niu, Y., Ye, Y., He, X., Yuan, S., Yu, W., Wang, S., Ge, Y., et al.: Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147 (2025)
- 32. Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., et al.: Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761 (2025)

- 33. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 34. Luo, Y., Shi, X., Bai, J., Xia, M., Xue, T., Wang, X., Wan, P., Zhang, D., Gai, K.: Camclonemaster: Enabling reference-based camera control for video generation. In: Proceedings of the SIGGRAPH Asia 2025 Conference Papers. pp. 1–10 (2025)
- 35. OpenAI: Gpt-4o image generation. https://openai.com/index/introducing-4oimage-generation/ (2025)
- 36. OpenAI: Gpt-image-1. URL https://openai.com/index/introducing-4o-imagegeneration/ (2025)
- 37. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: ICCV (2023)
- 38. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)
- 39. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: CVPR (2023)
- 40. Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al.: Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427 (2025)
- 41. Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al.: Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427 (2025)
- 42. Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., Taigman, Y.: Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089 (2023)
- 43. Sturm, P.: Pinhole camera model. In: Computer Vision: A Reference Guide, pp. 983–986. Springer (2021)
- 44. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098 (2024)
- 45. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098 (2024)
- 46. Team, M.L., Ma, H., Tan, H., Huang, J., Wu, J., He, J.Y., Gao, L., Xiao, S., Wei, X., Ma, X., et al.: Longcat-image technical report. arXiv preprint arXiv:2512.07584

(2025)

- 47. Team, R., Gao, Z., Wang, Q., Zeng, Y., Zhu, J., Cheng, K.L., Li, Y., Wang, H., Xu, Y., Ma, S., et al.: Advancing open-source world models. arXiv preprint arXiv:2601.20540 (2026)
- 48. Thippeswamy, B., Ramachandra, H., Rohan, S., et al.: Textverse: A streamlit web application for advanced analysis of pdf and image files with and without language models. In: 2024 Asia Pacific Conference on Innovation in Technology (APCIT). pp. 1–6. IEEE (2024)
- 49. Van Hoorick, B., Wu, R., Ozguroglu, E., Sargent, K., Liu, R., Tokmakov, P., Dave, A., Zheng, C., Vondrick, C.: Generative camera dolly: Extreme monocular dynamic novel view synthesis. In: European Conference on Computer Vision. pp. 313–331. Springer (2024)
- 50. Vidu Team: Vidu: Ai video generator. https://www.vidu.cn/ (2024)
- 51. Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., et al.: Yolov10: Real-time end-to-end object detection. Advances in neural information processing systems 37, 107984–108011 (2024)
- 52. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)

- 53. Wang, Y., Yang, S., Zhao, B., Zhang, L., Liu, Q., Zhou, Y., Xie, C.: Gptimage-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033 (2025)
- 54. Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., Shan, Y.: Motionctrl: A unified and flexible motion controller for video generation. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–11 (2024)
- 55. Wei, C., Xiong, Z., Ren, W., Du, X., Zhang, G., Chen, W.: Omniedit: Building image editing generalist models through specialist supervision. In: ICLR (2024)
- 56. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)
- 57. Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al.: Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871 (2025)
- 58. Xia, B., Liu, J., Zhang, Y., Peng, B., Chu, R., Wang, Y., Wu, X., Yu, B., Jia, J.: Dreamve: Unified instruction-based image and video editing. arXiv preprint arXiv:2508.06080 (2025)
- 59. Xia, B., Wang, S., Tao, Y., Wang, Y., Jia, J.: Llmga: Multimodal large language model based generation assistant. In: ECCV (2024)
- 60. Xia, B., Zhang, Y., Li, J., Wang, C., Wang, Y., Wu, X., Yu, B., Jia, J.: Dreamomni: Unified image generation and editing. In: CVPR (2025)
- 61. Xiao, S., Wang, Y., Zhou, J., Yuan, H., Xing, X., Yan, R., Li, C., Wang, S., Huang, T., Liu, Z.: Omnigen: Unified image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13294–13304 (2025)
- 62. Xiao, Y., Song, L., Chen, Y., Luo, Y., Chen, Y., Gan, Y., Huang, W., Li, X., Qi, X., Shan, Y.: Mindomni: Unleashing reasoning generation in vision language models with rgpo. arXiv preprint arXiv:2505.13031 (2025)
- 63. Xiao, Y., Wang, Q., Zhang, S., Xue, N., Peng, S., Shen, Y., Zhou, X.: Spatialtracker: Tracking any 2d pixels in 3d space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20406–20417 (2024)
- 64. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023)
- 65. Ye, Y., He, X., Li, Z., Lin, B., Yuan, S., Yan, Z., Hou, B., Yuan, L.: Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275

(2025)

- 66. Zhang, D.J., Paiss, R., Zada, S., Karnad, N., Jacobs, D.E., Pritch, Y., Mosseri,

I., Shou, M.Z., Wadhwa, N., Ruiz, N.: Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. arXiv preprint arXiv:2411.05003 (2024)

- 67. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36, 31428–31449 (2023)

