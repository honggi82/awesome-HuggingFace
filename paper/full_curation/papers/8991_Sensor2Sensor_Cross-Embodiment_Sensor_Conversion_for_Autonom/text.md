## Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving

Jiahao Wang1,2†, Bo Sun1, Yijing Bai1, Vincent Casser1, Songyou Peng3, Zehao Zhu1, Meng-Li Shih1,4†, Xander Masotto1, Shih-Yang Su1, Kanaad Parvate1, Tiancheng Ge1, Linn Bieske1, Dragomir Anguelov1, Mingxing Tan1, Chiyu Max Jiang1

1Waymo, 2Johns Hopkins University, 3Google DeepMind, 4University of Washington

Input Generated Multi-view Image/Video Generated LiDAR Frames

# arXiv:2605.22809v2[cs.CV]22May2026

|[Figure 1]|
|---|

|[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

[Figure 11]

###### Sensor2Sensor

[Figure 12]

Dashcam

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Internet Driving Video

[Figure 20]

[Figure 21]

[Figure 22]

ADAS Recording

Phone Recording

Phone

|[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]|
|---|

|[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|
|---|

Figure 1. Sensor2Sensor is a novel generative paradigm for translating in-the-wild monocular videos from varied sources such as dashcams, internet driving videos, phones, and even other Autonomous Driving Systems (ADS), Advanced Driver-Assistance Systems (ADAS) and vehicle platforms into high-fidelity, multi-modal, multi-sensor Autonomous Vehicle (AV) logs specific to a target vehicle embodiment. This enables cross-embodiment sensor conversion to maximally leverage real-world long-tail data for AV system validation.

### Abstract

ing paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds. A core challenge is the lack of paired training data. We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering. Sensor2Sensor then utilizes a diffusion architecture to perform the generative conversion. We perform comprehensive quantitative evaluations on the fidelity and realism of the generated sensor data. We demonstrate Sensor2Sensor’s practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.

Robust training and validation of Autonomous Driving Systems (ADS) require massive, diverse datasets. Proprietary data collected by Autonomous Vehicle (AV) fleets, while high-fidelity, are limited in scale, diversity of sensor configurations, as well as geographic and long-tail-behavioral coverage. In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments. However, this unstructured, in-the-wild video data is incompatible with ADS expecting structured, multi-modal sensor inputs for validation and training. To bridge this data gap, we propose Sensor2Sensor, a novel generative model-

†Work done during an internship at Waymo.

### 1. Introduction

The validation of Autonomous Driving Systems (ADS) against the full spectrum of real-world driving scenarios remains a paramount challenge in the field [6]. While generalist policies trained on aggregated data from diverse embodiments have shown promise, they do not obviate the need for rigorous, per-embodiment evaluation. This evaluation is non-negotiable for safety-critical systems, and its efficacy is fundamentally constrained by the profound scarcity of longtail data [16, 29, 58]. These long-tail scenarios encompass statistically rare yet safety-critical events, including erratic driving, sudden pedestrian maneuvers, and extreme weather or environmental conditions. Collecting such data organically is prohibitively expensive, requiring fleet-scale operations of immense cost and duration [6].

Two main avenues have been explored to address this data deficiency. The first is de novo scenario synthesis using generative models [5, 21]. While this can create novel events, the generated data often suffers from a critical plausibility gap (non-physical dynamics) and a realism problem (low sensor fidelity) unsuitable for ADS validation.

The second avenue seeks to leverage the immense scale and diversity of “in-the-wild” third-party data, sourced from internet videos or partner dashcam fleets (Original Equipment Manufacturers, OEMs) [28]. These data are, by construction, grounded in physical reality, thus eliminating concerns of event plausibility. It is also naturally skewed towards the long-tail, as mundane events are less likely to be recorded or shared. This approach, however, suffers from a severe embodiment gap [11]. This in-the-wild data is sensorially and geometrically misaligned with the target ADS platforms: it typically consists of a single monocular video, lacks the 360-degree multi-camera perspectives, and is devoid of critical modalities like LiDAR. This frames the problem as a highly complex, unpaired domain translation task. Unfortunately, classical unpaired translation methods are ill-equipped to bridge such a vast domain gap, as they lack the strong geometric priors and modal capacity to generate a coherent, temporally-consistent, multi-modal sensor suite from a single, uncalibrated video stream [9].

In this work, we propose Sensor2Sensor, a novel generative paradigm for cross-embodiment sensor conversion that synthesizes the advantages of both paths. As shown in Figure 1, Sensor2Sensor inherits the real-world plausibility of in-the-wild data while generatively re-rendering it into the precise, multi-modal format of a target AV embodiment.

The central challenge in training Sensor2Sensor is the absence of large-scale, paired (dashcam, AV log) training data. We circumvent this limitation by proposing a novel synthetic data-pairing pipeline. We leverage existing AV logs, which, by design, contain rich 3D information and 360-degree coverage. This high-fidelity data enables us to first reconstruct a 4D scene representation via dynamic

3D Gaussian Splatting (3DGS) [22, 52]. From this reconstructed scene, we can render novel, synthetic-yet-realistic dashcam views, complete with augmentations of intrinsic and extrinsic parameters sampled from real-world dashcam distributions. This process yields the required paired training corpus: (synthetic dashcam, real AV log).

With this paired dataset, we design Sensor2Sensor as a conditional diffusion model for multi-sensor (eight cameras) and multi-modal (camera and LiDAR) output, conditioned on the input dashcam video. This use of diffusion for geometrically-aware domain adaptation aligns with recent successes in cross-domain transfer [18, 36, 54].

We validate Sensor2Sensor through a comprehensive evaluation strategy. Quantitative fidelity is assessed using a bespoke, manually-collected ground-truth dataset. Concurrently, a broad qualitative analysis demonstrates the model’s efficacy in converting challenging, real-world in-the-wild videos into realistic and usable sensor logs. Our results affirm that Sensor2Sensor achieves state-of-the-art (SOTA) fidelity, further unlocking vast, previously-incompatible data sources for AV development.

In summary, our contributions are:

- • We introduce Sensor2Sensor, a novel generative paradigm for translating in-the-wild monocular videos into high-fidelity, multi-modal, and multi-sensor AV logs specific to a target vehicle embodiment.
- • We propose a pipeline using dynamic 3D Gaussian Splatting to reconstruct scenes from raw AV logs, rendering paired realistic dashcam views as high-quality training data for diffusion models.
- • We develop a conditional diffusion architecture, designed to be multi-sensor multi-modal, capable of geometricallyaware cross-embodiment sensor conversion.
- • We demonstrate, through comprehensive evaluation, that our method further unlocks the vast scale and diversity of in-the-wild video, converting challenging internet footage into realistic, usable data for AV development.

### 2. Related Works

Generative World Models and High-Fidelity Sensor Synthesis. Generative World Models [3, 4, 13–15, 27, 43, 47, 51], often built upon diffusion architectures [18, 30], are now foundational for physical AI, enabling the synthesis of photorealistic, physics-based data [23, 25]. Prominent examples, such as Wayve’s GAIA-1 [19] and the NVIDIA Cosmos [2] platform, primarily target scenario generation, future prediction, and planning for closed-loop simulation [55]. While powerful, their objective is orthogonal to our goal of data conversion. However, the success of conditional diffusion in intra-embodiment sensor translation validates its use for our complex, multi-modal task. Specifically, Camera-to-LiDAR generation using models like LiDMs [32] successfully navigates the spatial and modal

V3 (new)

[Figure 41]

mismatch between camera views and 3D point clouds. More recent cross-modality frameworks [12, 26, 34, 38, 41] like X-Drive [53] further demonstrate the ability to generate consistent multi-sensor data. Sensor2Sensor extends this conditional diffusion capability to the more challenging cross-embodiment setting, translating a single monocular stream into a geometrically-accurate, multi-sensor AV log. This complex translation necessitates a geometricallyanchored training corpus, which motivates our integration of reconstructive techniques.

Optimization Rendering

[Figure 42]

Diverse dashcam parameters

4DGS

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

Reconstructive World Models and 4D Scene Representation. Reconstructive World Models are essential for high-fidelity 4D (spatio-temporal) scene representation [20, 31, 33, 40, 44, 48], enabling closed-loop evaluation and novel view synthesis [57]. Advances in explicit representations [52], particularly 3D Gaussian Splatting (3DGS) [22], have allowed for real-time, photorealistic rendering and dynamic scene modeling in autonomous driving [1]. Methods like PAGS [1] and Driv3R [8] focus on decomposing the scene or achieving fast, dense 4D reconstruction from multi-view inputs, ensuring geometric accuracy and temporal consistency. These models serve as powerful “data machines” to augment viewpoints, as seen in works like DriveDreamer4D [57]. Sensor2Sensor critically repurposes this reconstructive capability to resolve the training data bottleneck [46]. We reconstruct scenes from existing AV logs via 4DGS, treating the reconstruction as a geometric oracle. This allows us to render a synthetic dashcam view from a novel, external viewpoint [57]. This process yields a perfectly paired training corpus, transforming the cross-embodiment challenge into a fully supervised, geometrically-anchored generation task.

,

AV Sensor Data Synthetic dashcam from 4DGS

Figure 2. Synthetic paired-data curation pipeline. We reconstruct 4DGS from 8-view cameras and render a diverse set of synthetic third-party cameras (e.g. popular dashcam models).

cal object model to achieve more complete object coverage. Once a scene is optimized, it can be rendered using virtual cameras with augmented intrinsic and extrinsic parameters to mimic the optics and placement of dashcams found inthe-wild. Note that due to the purely reconstructive nature of 3DGS, the best rendering quality is achieved within a bounded region around the original camera poses. Unlike the original 3DGS formulation, we use a ray-tracing-based rendering approach to better support fish-eye optics.

Third-party Camera Synthesis. We leverage high-fidelity 4DGS representations to synthesize a large, paired training corpus by rendering virtual cameras (Figure 2). This process explicitly bridges the domain gap between the source sensor data and the target third-party sensors (e.g., dashcams). The synthesis pipeline models two primary sources of sensor variation found in off-the-shelf dashcam systems: Intrinsic Parameters (pi): Generated by sampling realistic focal lengths, principal points, and distortion coefficients (κ). This step emulates the diverse optical profiles of low-cost, wide-angle lenses prone to significant distortion. Extrinsic Parameters (pe): Sampled as 6-DoF poses, pe = [R|t], relative to the vehicle frame. This accounts for variations in vehicle type, diverse mounting locations (e.g., driver-side), and minor rotational perturbations (θp,θy,θr) simulating imperfect camera installation. This rendering approach creates a vast dataset where each dashcam-style frame is perfectly time-synchronized and spatially aligned with the ground truth sensors.

### 3. Method

Our approach consists of two key stages: (1) a scalable data curation pipeline using 4DGS to synthesize paired training data (Section 3.1), and (2) a diffusion model that generates synchronized multi-view imagery and LiDAR point clouds conditioned on a single camera input (Section 3.2). We further extend this to temporally consistent video generation via auto-regressive modeling (Section 3.3).

#### 3.1. Synthetic Sensor Simulation via 4DGS

4DGS for Autonomous Driving. We use a variant of 3D Gaussian Splatting (3DGS) [22] with support for dynamic rigid (e.g. vehicles) and deformable (e.g. pedestrian) objects to construct 4D representations of diverse AV scenarios. In total, approximately 100,000 scenes of 10s duration were chosen for reconstruction. Each scene contains multiview camera data spanning 360 degrees as well as LiDAR data, which is used to initialize and regularize the geometry of the 3D Gaussian Splats, though optional. Splats belonging to moving objects are accumulated using a canoni-

#### 3.2. Multi-modal Diffusion Model for Sensors

To enable sensor conversion from third-party data, we first develop a multi-sensor, multi-view generation model. This model simultaneously generates multi-view images C = {ci}Ni=1 and the LiDAR point cloud L. Each sensor modality has its own VAE and U-Net branch for diffusion. The key attributes of this model are multi-view (Section 3.2.1) and multi-sensor (Section 3.2.3) consistency.

[Figure 54]

Denoising U-Net

[Figure 55]

LiDAR VAE Decoder

[Figure 56]

|[Figure 57]|
|---|

[Figure 58]

SelfAttn Cross-view

LiDAR Range Image Latent

Prev. Frame Conditioning

Cross-sensor

… …

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Attn

[Figure 65]

[Figure 66]

###### …

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

…

Attn

Generated LiDAR

[Figure 72]

Refer. Latent

Multi-view Image Latents

Image VAE Decoder

|: View concat. : Channel concat.<br><br>|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

Cross-view Attn Cross-sensor Attn

Inference: Real dashcam images.

Training: 3DGS renderings.

… …

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

Flatten and Concat

Flatten and Concat

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Self Attn

Self Attn

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Third-party Camera Condition

Split and Reshape

Split and Reshape

Generated Images

- Figure 3. Our multi-modal, multi-view sensor generation model architecture. Based on Latent Diffusion, the model simultaneously generates multi-view images (C) and LiDAR point clouds (L) using modality-specific VAEs and U-Net towers. Multi-sensor consistency is enforced via cross-sensor attention, and multi-view consistency is maintained with 3D attention blocks.

##### 3.2.1. Multi-view Image Generation

The image branch builds on a multi-view diffusion model [10] that enables view consistency and camera pose control over the image generation. Given the camera parameters for each camera, this model learns a joint distribution of all images. To achieve multi-view consistency, the model replaces the 2D attention modules in the original LDM to 3D (1D cross views and 2D in spatial) and computes attentions on all images.

Furthermore, to precisely control the poses of generated images, this model accepts camera parameters as conditions. The camera parameters are represented via raymaps [10, 39], which encode the ray origin and direction at each spatial location. All raymaps are normalized with regard to the first camera and concatenated channelwise onto the image features.

##### 3.2.2. LiDAR Generation

LiDAR Representation. To effectively leverage the capabilities of 2D generative models, we utilize the LiDAR point cloud’s native representation as range-view spin images—a tensor with shape [HL,WL,DL], where the DL = 4 channels correspond to (1) range (depth in meters), (2) intensity (amount of light reflected), (3) elongation (to what extent the waveform has been “flattened”), and (4) validity (1 for a return, 0 otherwise). The image rows and columns map to the sensor’s elevation and azimuth angles, respectively. Each (row, col, range) value can be projected to and from 3D Euclidean space (x,y,z) given the vehicle trajectory and sensor calibration. For normalization, range values are clamped at 150 meters and linearly scaled to the [0,1] interval. Intensity and elongation are similarly normalized

to fit within [0,1].

LiDAR VAE. We introduce a VAE architecture for generating LiDAR spin images, jointly encoding depth, intensity, and elongation. The encoder and decoder are both convolutional, and we optimize the VAE via

LTOTAL =LL1range + LL1elongation + LL1intensity + LBCEvalidity + LLPIPSnormals

+ LLPIPSelongation + LLPIPSintensity + LLPIPSvalidity + LKL. (1) Additional training details are provided in the supplemental. LiDAR Diffusion. We first project the raw LiDAR range images into a latent space using the LiDAR VAE. A LiDAR U-Net branch then performs diffusion on this latent, operating similarly to a standard single-view image diffusion model. Each layer in the LiDAR U-Net is designed to output a feature with the same channel dimension as its corresponding layer in the multi-view image branch, enabling our cross-sensor feature fusion.

##### 3.2.3. Cross-Sensor Attention Module

As shown in Figure 3, to simultaneously generate consistent images and LiDAR, we introduce a cross-sensor attention module within each U-Net block. We inject this module after convolutional layers to promote continuous information interchange. In detail, at a given block i, we flatten the image features fCi and LiDAR features fLi into token sequences TiC ∈ RK

L×di, where KC = N × hiC × wCi and KL = hiL × wLi . The shared U-Net architecture for both modalities ensures their feature dimension di is identical. These tokens are then concatenated into a unified sequence TiU ∈ R(K

C×di and TiL ∈ RK

C+KL)×di, and the module computes self-attention over this sequence, allowing features from both sensors to interact directly.

##### 3.2.4. Third-party Camera Condition

To directly leverage the visual context of the third-party data (e.g., dashcams), we introduce it as an additional, conditional ninth view, distinct from the N = 8 views targeted for generation. This conditional input is processed by the encoder to generate a latent representation, which is then concatenated with (1) a corresponding raymap [10, 39] and (2) a binary conditioning mask. This mask explicitly signals to the model that this view is a known, noise-free condition, distinguishing it from the N noisy latents to be denoised. This augmented latent is then concatenated along the view dimension with the latents from the original eight views, and the resulting (N +1)×H ×W ×C tensor is processed by the diffusion layers. This allows the features from the 8 target views to interact with the conditional view through attention, effectively conditioning the synthesis of the surrounding scene on the dashcam’s context. This 9th view is excluded from the loss computation, ensuring its role as a conditioning input and that the network’s capacity is focused on accurately generating the eight target views.

#### 3.3. Auto-regressive Video Generation

To convert third-party videos to driving logs, we extend our model for auto-regressive generation. Given the third-party camera frame xt at time step t > 0, we aim to model the conditional probability distribution of the multi-view images Ct and LiDAR point cloud Lt, conditioning on the self-generations at step t − 1:

P(Ct,Lt|xt,Ct−1,Lt−1). (2) When t = 0, sensor data is generated conditioning only on x0. Vanilla auto-regressive generation suffers from drifting, as models trained on ground-truth (GT) context must generate sequences conditioned on their own imperfect generations during inference. This causes errors to accumulate over long rollouts. To mitigate this, we introduce the DAgger algorithm [35], which augments the training context with the model’s own generations. We gradually shrink this train-test mismatch by iteratively generating rollout videos and training a new model on the resulting context. To maintain robustness, we set a 0.2 probability of training on the original GT context.

### 4. Experiments

Our experiments are designed to: (1) quantify the fidelity of our generated images, video, and LiDAR point clouds against strong baselines; (2) test model’s generalizability on challenging, in-the-wild driving footage; and (3) validate key architectural and training choices via ablation studies.

#### 4.1. Experiment Settings

Evaluation metrics. We evaluate our results using Fr´echet Inception Distance (FID) (↓) [17] for image realism and

Fr´echet Video Distance (FVD) (↓) [42] for video realism. For paired ground-truth comparisons, we use Peak Signal-to-Noise Ratio (PSNR) (↑), Structural Similarity Index Measure (SSIM) (↑) [50], and the Learned Perceptual Image Patch Similarity (LPIPS) (↓) [56]. These are supplemented by Human Evaluation (↑), where raters choose the more realistic result in side-by-side comparisons.

Dataset. Since paired, third-party-to-AV sensor generation is a novel task, no public datasets with such synchronized data exist for evaluation. We therefore curated an evaluation dataset comprising two key components: (a) A dataset of 1,000 paired “Fixed-Camera-to-AV” log sequences (each 3 seconds long). The fixed-camera is a bumper camera positioned at the front-left bumper of the AV, and the 8-view surrounding cameras and the LiDAR are on top of the AV. (b) An “in-the-wild” dataset, including manually-collected real dashcam recordings, driving videos available on the internet, phone recordings and footage from other ADAS, for showing the in-the-wild generalizability.

Baselines. End-to-end conversion of a monocular thirdparty video to a full AV sensor suite (multi-view cameras and LiDAR) has not been fully explored in previous work. Thus, no direct baselines exist for our specific task. To benchmark Sensor2Sensor, we adapted several state-of-theart methods for comparison. Reconstruction-based: We compare against state-of-the-art feedforward 3D scene reconstruction models VGGT [45] and π3 [49] for the multicamera generation task. Generative models: We adapt two SOTA generative models. X-Drive [53], an imageLiDAR co-generation model, was modified to condition on the dashcam input via attention. We also adapted CAT3D [10] by (1) enabling LiDAR generation using the same VAE as our method and (2) conditioning it on the dashcam via channel-concatenation (CC) instead of view-concatenation (VC). We refer to this baseline as “Ours without (wo) VC”, which also serves as a key ablation against our approach.

#### 4.2. Multi-view Image Generation

We first evaluate the task of multi-view image generation. To quantitatively measure performance, we curate a “FixedCamera-to-AV” dataset. The input for this task comes from a real, front-left facing camera fixed on the AV near the bumper. This input camera is synchronized and calibrated with the target 8 surrounding views, to provide an accurate quantitative benchmark, as shown in Table 1.

On this “Fixed-Camera-to-AV” generation task, our method outperforms all baselines with an FID of 6.47 and LPIPS of 0.316, demonstrating the superior generative quality. Figure 4 shows that images generated by Sensor2Sensor are clear, geometrically plausible, and maintain consistent appearance of objects as they appear between camera views. In contrast, baseline methods often produce blurry results, distorted geometry, or noticeable artifacts.

[Figure 98]

[Figure 99]

[Figure 100]

Input Camera VGGT X-Drive Ours wo VC Ours GT

- Figure 4. Image comparison. Our method Sensor2Sensor produces results largely faithful to the ground truth, while the baselines either fail to preserve the scene and object structures, or cannot create plausible generations of the unobserved areas.

Table 1. Evaluation on multi-view image generation from a fixed bumper camera. We compare our method against baselines on our paired dataset. ↓: Lower is better. ↑: Higher is better. VC means concatenating dashcam in the view dimension.

Method FID ↓ PSNR ↑ SSIM ↑ LPIPS ↓

VGGT [45] 250.93 14.73 0.433 0.491 π3 [49] 246.27 14.93 0.470 0.458 X-Drive [53] 8.30 18.61 0.536 0.345 Ours without VC 6.88 18.69 0.531 0.346 Ours 6.47 19.06 0.539 0.316

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

|[Figure 105]|
|---|

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

|[Figure 110]|
|---|

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

|[Figure 115]|
|---|

Ours wo VC

Ours wo DAgger

Ours

time

0s 0.4s 0.8s 1.2s 1.6s

- Figure 5. Temporal video rollout comparison (only showing front view for compactness). DAgger training significantly improves temporal stability of generated videos through the rollout.

Table 2. Evaluation on multi-view video generation. We compare on the “Fixed-Camera-to-AV” dataset.

Method FVD ↓ PSNR ↑ SSIM ↑ LPIPS ↓

VGGT [45] 2373.15 14.73 0.433 0.491 π3 [49] 2007.35 14.93 0.470 0.458 Ours without VC 293.73 22.07 0.622 0.204 Ours 278.12 22.42 0.623 0.186

all baselines, such as Ours wo VC (293.73) and the feedforward reconstruction models π3 (2007.35) and VGGT (2373.15). The feedforward models’ high FVD scores are expected, as their reconstructive-only design cannot produce coherent novel views. This indicates that we not only generate realistic individual frames but also ensure they are coherent over time. The strong per-frame metrics (PSNR 22.42, SSIM 0.623, LPIPS 0.186) further support this, reinforcing the high fidelity seen in our static image evaluation.

Moreover, as shown in Figure 5, while baselines exhibit noticeable flickering or inconsistent object appearance across frames, our model produces smooth and coherent video sequences for all views, which is critical for downstream consumption by perception or simulation systems.

#### 4.4. LiDAR Generation

A key contribution of Sensor2Sensor is its multi-modal capability to co-generate LiDAR point clouds along with multi-view videos. Qualitatively, Figure 6 provides a direct comparison against baseline methods. Our model shows a superior ability to reconstruct plausible 3D geometry for both nearby actors (like the truck) and the static environment. Our results are cleaner, with fewer noise artifacts and more accurate intensity rendering compared to X-Drive and Ours wo VC. Furthermore, Figure 7 highlights our model’s strength in producing jointly consistent image and LiDAR outputs. The generated LiDAR points correctly align with their corresponding objects in the generated camera views, demonstrating that the model has learned a coherent underlying 3D representation of the scene.

#### 4.3. Video Generation

Beyond static images, we evaluate the temporal consistency of our generated multi-view videos. We report quantitative results on our paired “Fixed-Camera-to-AV” dataset in Table 2. We use Fr´echet Video Distance (FVD) (↓) as the primary metric for overall video quality, supplemented by frame-wise PSNR (↑), SSIM (↑), and LPIPS (↓). XDrive is excluded from this comparison, as it is an imageonly model and does not generate video. Furthermore, the reconstruction-based methods (VGGT and π3) only generate complete results for the front view, as their other views suffer from large empty regions. For a better comparison, all metrics in this table are computed exclusively on the generated front-view videos.

Quantitatively, we report the Chamfer Distance for generated LiDAR in Table 3. Moreover, human evaluation of LiDAR generation in Table 4 also demonstrates a clear preference for our generated LiDAR over the baselines.

Our model shows superior temporal stability, achieving the best FVD of 278.12. This significantly outperforms

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Fixed Camera

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Dashcam

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Internet

Input Camera X-Drive Ours wo VC Ours

- Figure 6. Qualitative LiDAR Comparison. Our method correctly renders the truck’s shape and has less noise in the surrounding objects, while the other methods produce distortions and incorrect intensity. All methods use the same LiDAR VAE for a fair comparison.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

- Figure 7. Visualization of joint image and LiDAR generation. Sensor2Sensor achieves cross-modal consistency between image and LiDAR, faithfully generating safety-critical objects, including signage, road markings, and vehicles.

Table 4. Human evaluation for in-the-wild generation. We show top-rank (top half) and pair-wise (bottom half) preference rates. Participants were asked to rank results of all three methods based on realism and alignment to the input.

Dashcam Internet Method Image ↑ LiDAR ↑ Image ↑ LiDAR ↑ X-Drive [53] 3.08% 8.08% 1.54% 7.69% Ours without VC 13.46% 23.85% 13.85% 33.85% Ours 83.46% 68.08% 84.62% 58.46%

Ours without VC > X-Drive 67.69% 69.62% 84.62% 73.46% Ours > Ours without VC 85.77% 73.46% 85% 63.46% Ours > X-Drive 94.62% 87.31% 95.38% 85%

characteristics and challenging, unseen environments (such as night-time near collisions, accidents, and active incidents), our model converts monocular inputs into coherent multi-sensor AV logs while preserving critical scene elements. This highlights its robustness for mining long-tail scenarios from vast, previously incompatible data sources.

Table 3. LiDAR Generation Accuracy. Comparison of Chamfer Distance (↓) between the baseline and our proposed method.

Method Chamfer Distance ↓ Improvement (%) X-Drive [53] 10.02 Ours 8.68 13.37%

Quantitatively, a comprehensive human evaluation is shown in Table 4. 26 participants evaluated 40×3 generated image and LiDAR samples based on realism and alignment with the input image. After training and qualification on the protocol, they ranked each triplet as best, middle, or worst, from which we computed top-rank and pairwise preference rates. On dashcam data, Sensor2Sensor is top-preferred in 83.46% of image cases and 68.08% for LiDAR; on internet data, 84.62% and 58.46%, respectively. Pairwise comparisons show Sensor2Sensor is preferred over X-Drive in over 94% of image cases and 85% for LiDAR.

#### 4.5. Generalization on in-the-wild driving data

The primary motivation for Sensor2Sensor is to further unlock “in-the-wild” data. We test this by applying our model, trained only on our paired dataset, to a diverse set of uncurated videos from internet, dashcams, and other third-party sources. These videos feature camera intrinsics, extrinsics, weather conditions and content unseen during training.

As shown in Fig. 8, Sensor2Sensor demonstrates strong qualitative generalization. Despite facing unknown sensor

[Figure 137]

[Figure 138]

[Figure 139]

Input Input Input

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

- Figure 8. Qualitative generalization to in-the-wild internet videos. Sensor2Sensor successfully converts diverse and challenging monocular inputs, including long-tail crashes, night-time scenes with low visibility, and active incidents, into full, coherent AV sensor suites.

[Figure 158]

[Figure 159]

- Table 5. Ablation on model architecture. We compare input conditioning (CC vs. VC) and the impact of joint LiDAR training, evaluated on the “Fixed-Camera-to-AV” dataset. CC is channel concatenation, VC is view concatenation.

Method FID ↓ PSNR ↑ SSIM ↑ LPIPS ↓

CAT3D + CC (image-only) 6.63 18.91 0.542 0.314 CAT3D + VC (image-only) 6.20 19.12 0.543 0.307

CAT3D + CC + LiDAR 6.88 18.69 0.531 0.346 CAT3D + VC + LiDAR (ours) 6.47 19.06 0.539 0.316

- Table 6. Ablation on DAgger finetuning for video generation.

Real Generated

- Figure 9. LiDAR detection. We tested a vehicle detection model using real and generated LiDAR. Comparable results confirm the fidelity of our generation.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Real Generated

- Figure 10. Image segmentation. Panoptic-DeepLab [7] produces consistent predictions on real and generated images.

Method Front-view FVD ↓ Front-view FID ↓

Without DAgger 288.90 24.65 With DAgger (ours) 278.12 21.54

#### 4.6. Ablation Study

Model Architecture. Table 5 analyzes key architectural choices. First, we compare input conditioning via channel concatenation (CC) and view concatenation (VC). In the image-only setting, VC achieves better FID (6.20 vs. 6.63). Second, we study joint image-LiDAR training. Our full model achieves LPIPS 0.316, outperforming the CC variant (0.346) while remaining competitive with image-only VC (0.307). This confirms that our design enables joint LiDAR generation without obvious image quality degradation.

### 5. Conclusion

Sensor2Sensor is a novel generative paradigm that bridges the embodiment gap between consumer driving videos and the complex, multi-modal sensor suites required for AV validation. Leveraging a 4DGS-based data pairing pipeline and a conditional diffusion architecture, Sensor2Sensor converts monocular third-party videos into synchronized multiview camera streams and LiDAR point clouds, achieving state-of-the-art performance in cross-embodiment sensor generation. Crucially, the model co-generates consistent LiDAR and demonstrates strong generalization to real-world footage. By unlocking large-scale driving videos for AV development, our approach provides a scalable solution to data scarcity for safety-critical validation and deployment of safety-critical autonomous systems. Future work will explore improved scalability, generalization to more sensors, and a more scalable evaluation protocol.

DAgger Finetuning. Table 6 shows that DAgger finetuning improves video quality. With DAgger, FVD and FID improve to 278.12 and 21.54. This demonstrates improved temporal consistency and fidelity.

#### 4.7. Downstream Tasks

We aim to build a high-fidelity simulation environment. To assess realism, we apply perception models trained on real data directly to our generated data without finetuning. Comparable performance on real and generated data in LiDAR detection (Fig. 9) and image segmentation (Fig. 10) indicates strong alignment with real-world distributions.

### References

- [1] Ying A, Wenzhang Sun, Chang Zeng, Chunfeng Wang, Hao Li, and Jianxun Cui. PAGS: Priority-adaptive gaussian splatting for dynamic driving scenes. arXiv preprint arXiv:2510.12282, 2025. 3
- [2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2
- [3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Selfsupervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 2
- [4] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024. 2
- [5] Xuan Cai, Xuesong Bai, Zhiyong Cui, Danmu Xie, Daocheng Fu, Haiyang Yu, and Yilong Ren. Text2Scenario: Text-driven scenario generation for autonomous driving test. arXiv preprint arXiv:2503.02911, 2025. 2
- [6] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. IEEE TPAMI, 2024. 2
- [7] Bowen Cheng, Maxwell D Collins, Yukun Zhu, Ting Liu, Thomas S Huang, Hartwig Adam, and Liang-Chieh Chen. Panoptic-deeplab: A simple, strong, and fast baseline for bottom-up panoptic segmentation. In CVPR, 2020. 8
- [8] Xin Fei, Wenzhao Zheng, Yueqi Duan, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Jiwen Lu. Driv3R: Learning dense 4d reconstruction for autonomous driving. arXiv preprint arXiv:2412.06777, 2024. 3
- [9] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, Kun Zhang, and Dacheng Tao. Geometryconsistent generative adversarial networks for one-sided unsupervised domain mapping. In CVPR, 2019. 2
- [10] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. In NeurIPS, 2024. 4, 5, 1
- [11] Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, et al. Foundation models in autonomous driving: A survey on scenario generation and scenario analysis. IEEE Open Journal of Intelligent Transportation Systems, 2026. 2
- [12] Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, et al. Genesis: Multimodal driving scene generation with spatio-temporal and cross-modal consistency. arXiv preprint arXiv:2506.07497, 2025. 3
- [13] David Ha and J¨urgen Schmidhuber. World models. arXiv

preprint arXiv:1803.10122, 2018. 2

- [14] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019.
- [15] Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025. 2
- [16] Deepti Hegde, Rajeev Yasarla, Hong Cai, Shizhong Han, Apratim Bhattacharyya, Shweta Mahajan, Litian Liu, Risheek Garrepalli, Vishal M. Patel, and Fatih Porikli. Distilling multi-modal large language models for autonomous driving. In CVPR, 2025. 2
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [19] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 2
- [20] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. S3Gaussian: Self-Supervised Street Gaussians for Autonomous Driving. arXiv preprint arXiv:2405.20323, 2024. 3
- [21] Pin Ji, Yang Feng, Zongtai Li, Xiangchi Zhou, Jia Liu, Jun Sun, and Zhihong Zhao. Txt2Sce: Scenario generation for autonomous driving system testing based on textual reports. arXiv preprint arXiv:2509.02150, 2025. 2
- [22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics,

2023. 2, 3

- [23] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, MingYu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026. 2
- [24] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 1
- [25] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 2022. 2
- [26] Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. In CVPR, 2025. 3
- [27] Taiming Lu, Tianmin Shu, Junfei Xiao, Luoxin Ye, Jiahao Wang, Cheng Peng, Chen Wei, Daniel Khashabi, Rama Chellappa, Alan Yuille, et al. Genex: Generating an explorable world. arXiv preprint arXiv:2412.09624, 2024. 2
- [28] Yan Miao, Georgios Fainekos, Bardh Hoxha, Hideki Okamoto, Danil Prokhorov, and Sayan Mitra. From dashcam videos to driving simulations: Stress testing automated vehicles against rare events. arXiv preprint arXiv:2411.16027,

2024. 2

- [29] Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. VLP: Vision language planning for autonomous driving. In CVPR, 2024. 2
- [30] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2
- [31] Chensheng Peng, Chengwei Zhang, Yixiao Wang, Chenfeng Xu, Yichen Xie, Wenzhao Zheng, Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan. Desire-gs: 4d street gaussians for static-dynamic decomposition and surface reconstruction for urban driving scenes. In CVPR, 2025. 3
- [32] Haoxi Ran, Vitor Guizilini, and Yue Wang. Towards realistic scene generation with LiDAR diffusion models. In CVPR,

2024. 2

- [33] Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, and Jiahui Huang. Scube: Instant large-scale scene reconstruction using voxsplats. In NeurIPS, 2024. 3
- [34] Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, et al. Cosmos-drivedreams: Scalable synthetic driving data generation with world foundation models. arXiv preprint arXiv:2506.09042,

2025. 3

- [35] St´ephane Ross, Geoffrey Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In AISTATS, 2011. 5, 6
- [36] Chinmay Samak, Tanmay Samak, Bing Li, and Venkat Krovi. Sim2real diffusion: Leveraging foundation vision language models for adaptive automated driving. RA-L,

2025. 2

- [37] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 6
- [38] Bharat Singh, Viveka Kulharia, Luyu Yang, Avinash Ravichandran, Ambrish Tyagi, and Ashish Shrivastava. Genmm: Geometrically and temporally consistent multimodal data generation for video and lidar. arXiv preprint arXiv:2406.10722, 2024. 3
- [39] Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. In NeurIPS, 2021. 4, 5
- [40] Rui Song, Chenwei Liang, Yan Xia, Walter Zimmer, Hu Cao, Holger Caesar, Andreas Festag, and Alois Knoll. Coda-4dgs: Dynamic gaussian splatting with context and deformation awareness for autonomous driving. In ICCV, 2025. 3
- [41] Tao Tang, Enhui Ma, Xia Zhou, Letian Wang, Tianyi Yan, Xueyang Zhang, Kun Zhan, Peng Jia, Xianpeng Lang, JiaWang Bian, et al. Omnigen: Unified multimodal sensor generation for autonomous driving. In ACM MM, 2025. 3
- [42] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. ICLR Workshop,

2019. 5

- [43] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video

- generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [44] Jingkang Wang, Henry Che, Yun Chen, Ze Yang, Lily Goli, Sivabalan Manivasagam, and Raquel Urtasun. Flux4d: Flow-based unsupervised 4d reconstruction. arXiv preprint arXiv:2512.03210, 2025. 3
- [45] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 5, 6
- [46] Jiahao Wang, Zhenpei Yang, Yijing Bai, Yingwei Li, Yuliang Zou, Bo Sun, Abhijit Kundu, Jose Lezama, Luna Yue Huang, Zehao Zhu, et al. Drive&gen: Co-evaluating endto-end driving and video generation models. In IROS, 2025. 3
- [47] Jiahao Wang, Luoxin Ye, TaiMing Lu, Junfei Xiao, Jiahan Zhang, Yuxiang Guo, Xijun Liu, Rama Chellappa, Cheng Peng, Alan Yuille, et al. Evoworld: Evolving panoramic world generation with explicit 3d memory. arXiv preprint arXiv:2510.01183, 2025. 2
- [48] Linhan Wang, Kai Cheng, Shuo Lei, Shengkun Wang, Wei Yin, Chenyang Lei, Xiaoxiao Long, and Chang-Tien Lu. Dcgaussian: Improving 3d gaussian splatting for reflective dash cam videos. In NeurIPS, 2024. 3
- [49] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347,

2025. 5, 6

- [50] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. TIP, 2004. 5
- [51] Waymo Team. The waymo world model: A new frontier for autonomous driving simulation. https://waymo. com/blog/2026/02/the-waymo-world-modela-new-frontier-for-autonomous-drivingsimulation/, 2026. Waymo Blog. 2
- [52] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024. 2, 3
- [53] Yichen Xie, Chenfeng Xu, Chensheng Peng, Shuqi Zhao, Nhat Ho, Alexander T. Pham, Mingyu Ding, Masayoshi Tomizuka, and Wei Zhan. X-drive: Cross-modality consistent multi-sensor data synthesis for driving scenarios. In ICLR, 2025. 3, 5, 6, 7
- [54] Zheyuan Zhan, Defang Chen, Jian-Ping Mei, Zhenghe Zhao, Jiawei Chen, Chun Chen, Siwei Lyu, and Can Wang. Conditional image synthesis with diffusion models: A survey. TMLR, 2025. 2
- [55] Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal M Patel, Paul Pu Liang, et al. World-in-world: World models in a closed-loop world. arXiv preprint arXiv:2510.18135, 2025. 2
- [56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5, 6

- [57] Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, Wenjun Mei, and Xingang Wang. Drivedreamer4d: World models are effective data machines for 4D driving scene representation. In CVPR, 2025. 3
- [58] Zehao Zhu, Yuliang Zou, Chiyu Max Jiang, Bo Sun, Vincent Casser, Xiukun Huang, Jiahao Wang, Zhenpei Yang, Ruiqi Gao, Leonidas Guibas, et al. Scenecrafter: Controllable multi-view driving scene editing. In CVPR, 2025. 2

## Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving Supplementary Material

### A. Extended Qualitative Results

In this section, we provide an in-depth visual analysis to complement the quantitative results presented in the main paper. These figures are specifically designed to highlight the efficacy and generalization capabilities of our Sensor2Sensor pipeline across different output modalities. We present additional qualitative results covering image generation (Figure 11 and Figure 12), LiDAR generation (Figure 13), and image–LiDAR alignment (Figure 14). Finally, to best illustrate the temporal coherence and realism of our full pipeline, more video generation results are presented in the accompanying supplementary video.

### B. Implementation Details

#### B.1. Training Pipeline

Our model is trained in a four-step pipeline to progressively incorporate increasingly complex conditioning information.

- • Step 1: Base Conditioning Single Frame Generation. The model is first trained on single frame generation, given conditional dashcam images.
- • Step 2: Previous Frame Conditioning. The model is then fine-tuned with dense conditioning signals, including the latent representations of the previous frame’s camera and LiDAR data, as well as an additional dashcam view.
- • Step 3: DAgger Data Generation. We use the model from Step 2 to generate a new dataset in a Dataset Aggregation (DAgger) fashion. The model is unrolled for multiple steps to create long-term simulations, which may include drifted data.
- • Step 4: DAgger Fine-tuning. Finally, the model is finetuned on the DAgger-generated dataset from Step 3. This step involves training with augmented latent representations from the previous frame, which helps the model learn to correct its own errors and improves long-term simulation stability.

#### B.2. Model Architecture

The core of our generative model is a conditional diffusion model with a multi-stream UNet backbone designed for multi-modal sensor data.

Backbone. We employ a UNet architecture with temporal attention connections. It features separate processing streams for camera and LiDAR data, allowing the model to learn modality-specific representations while fusing information through shared attention layers. The UNet processes inputs from 8 surrounding camera views and one dashcam

view, along with the top-mounted LiDAR. The architecture uses a block structure with output channels of (320, 640, 1280, 1280).

Variational Autoencoders (VAEs) [24] We use separate, pre-trained VAEs to encode the raw sensor data into a compact latent space.

- • Image VAE: A VAE [10] is used to encode the camera views into 8-channel latent representations.
- • LiDAR VAE: A dedicated VAE encodes the raw LiDAR spin image into a 16-dimensional latent space. The UNet’s LiDAR stream is configured with 16 input and output channels to match this latent space. More details about LiDAR VAE training is shown in Section B.7.

Conditioning Mechanisms The generation process is guided by conditioning inputs.

- • Dashcam Conditions: Current frame of dashcam is conditioned into the diffusion blocks by concatenate the feature with the denoising latents in the view dimension. During training, we incorporate random spatial masking (with a probability of 0.2) on the conditional dashcam frames. At inference time, we can leverage this capability to apply targeted masks over distractor elements (e.g., dashcam watermarks or the ego-vehicle hood), ensuring the model focuses solely on the relevant scene context.
- • Previous frame Conditions: To ensure temporal consistency, the model is conditioned on the latent representations of the previous frame’s camera images and LiDAR scan. We achieve this by concatenating the latents from the previous timestep to the current frame’s latents along the channel dimension. Additionally, during training, we randomly drop this temporal conditioning with a probability of 0.5 to facilitate the learning of initial frame generation and improve robustness.

#### B.3. Training Details

The model is trained on 128 TPUs. We use the AdamW optimizer with a learning rate of 5e-5. We clip the global norm of gradients at 1.0. For regularization, we randomly drop conditioning signals during training. For evaluation, we use an exponential moving average (EMA) of the model weights with a decay of 0.999. The training follows the multi-step pipeline described above, with each step finetuning from the checkpoint of the previous step. For step 1, 2, and 4, we train with 80k, 40k, and 20k steps, respectively. The number of model parameters is around 250M.

[Figure 164]

Input image: A white car and a black car in front

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

X-Drive

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Ours wo VC

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Ours

[Figure 189]

Input image: Two cars in front

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

X-Drive

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Ours wo VC

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Ours

[Figure 214]

Input image: A red and a white car in front left

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

X-Drive

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Ours wo VC

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Ours

[Figure 239]

Input image: A white car in front right

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

X-Drive

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Ours wo VC

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Ours

[Figure 264]

Input image: A tilted car

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

X-Drive

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Ours wo VC

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

Ours

[Figure 289]

Input image: A silver sedan heading toward our car

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

X-Drive

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Ours wo VC

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Ours

[Figure 314]

[Figure 315]

Input image:

Input image:

[Figure 316]

[Figure 317]

[Figure 318]

X-Drive Ours wo VC Ours

X-Drive Ours wo VC Ours

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

Input image: Input image:

[Figure 329]

X-Drive Ours wo VC Ours X-Drive Ours wo VC Ours

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

Input image: Input image:

[Figure 337]

[Figure 338]

X-Drive Ours wo VC Ours X-Drive Ours wo VC Ours

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

- Figure 13. Additional qualitative results for LiDAR generation. Our method yields more accurate geometry in the synthesized point clouds, resulting in a less noisy output and a better correspondence with the accompanying image data. This improved fidelity allows for a more accurate preservation of the underlying spatial relationships of the scene.

[Figure 343]

Input image:

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

Input image:

[Figure 354]

- Figure 14. Additional qualitative results showcasing the Image-LiDAR alignment and cross-modal consistency achieved by our method. These visualizations confirm that the generated LiDAR point cloud accurately reflects the geometric details observed in the synthesized image view, demonstrating precise spatial registration between the two modalities.

#### B.4. Dataset Details

For training, we use a proprietary dataset of 100k 10s clips (8 cameras + top LiDAR) for 4DGS reconstruction. The resulting rendered images and synchronized sensor logs constitute the paired training data for diffusion models. For evaluation, quantitative analysis uses 1K paired 3s sequences from proprietary Fixed-Camera-to-AV logs. For in-the-wild evaluation, we collect diverse unconstrained in-

puts: internet videos, ADAS logs, and manually captured dashcam (e.g., Nexar) and smartphone footage.

#### B.5. Dashcam Parameter Distribution

Camera parameters are sampled via a two-stage process: (1) Extrinsics: We first select a vehicle category (e.g., Sedan, SUV), then sample 6-DoF poses from category-specific distributions (e.g., for Sedans: height 1.1–1.3m, forward translation 2.0–2.5m, pitch ±10◦). (2) Intrinsics: Parameters are

drawn from a set of calibrated real-world dashcams (e.g., Nexar, VIOFO) and augmented with uniform noise (e.g., ±5% focal length). Final outputs undergo exposure compensation and gamma correction for lighting normalization.

#### B.6. Different Target Camera Configurations

Sensor2Sensor is designed for multi-sensor flexibility via its raymap-conditioning architecture, which encodes camera intrinsics and extrinsics into the generation process. While the current results focus on our large-scale proprietary platform, the raymap ensures the model is not limited to a single configuration, as it learns the fundamental mapping between 3D rays and pixel intensities. To adapt to new platforms, our paradigm simply requires 4DGS-based paired data generation for the target sensor configurations.

#### B.7. LiDAR VAE Training

We introduce a VAE architecture for generating LiDAR spin images, jointly encoding depth, intensity, and elongation. The encoder and decoder are both convolutional, and its latent space are regularized with a KL divergence loss. The normalized range, intensity, and elongation use an L1 reconstruction loss, while the validity reconstruction loss uses cross entropy. In addition, we add an LPIPS loss on surface normals (derived from predicted point cloud), intensity, elongation, and validity. The total loss, which we seek to minimize, is a weighted sum of all components, shown in Equation (1), with terms: LL1range + LL1elongation + LL1intensity + LBCEvalidity + LLPIPSnormals + LLPIPSelongation + LLPIPSintensity + LLPIPSvalidity + LKL. In this formulation, the LLPIPSnormals term uses normals fnormalsL = ComputeNormals(frangeL ) that are computed based on finite differences using the projected 3D lidar points. We now define each loss term individually.

L1 Reconstruction Loss. For the signal components using L1 loss (range, elongation, and intensity), the loss is defined as:

LL1signal = λsignal||fsignalL − ˆfsignalL ||1 (3) where “signal” represents range, elongation, or intensity. In this equation, fsignalL is the ground truth LiDAR feature map and ˆfsignalL is its corresponding reconstruction from the VAE. The term λsignal is a scalar hyperparameter that weights the contribution of this specific loss component.

Binary Cross-Entropy Loss. The cross-entropy loss on the validity mask is calculated by:

LBCEvalidity = −λBCE[fvalidL log(ˆfvalidL )

(4)

+ (1 − fvalidL )log(1 − ˆfvalidL )]

where fvalidL is the ground truth binary validity mask (with values 1 for valid returns and 0 otherwise) and ˆfvalidL is the predicted validity probability map output by the decoder. The λBCE is its corresponding loss weight.

LPIPS Perceptual Loss. The LPIPS (Learned Perceptual Image Patch Similarity) [56] loss measures the perceptual distance between a reference image x and a distorted image xˆ. Unlike traditional metrics like L1 or MSE, LPIPS leverages features extracted from a pre-trained deep neural network (e.g., VGG [37]). The loss, presented in the equation

1 HiWi h,w

wi ⊙ (yhwi − yˆhwi ) 22 ,

LLPIPS(x,xˆ) =

i

(5) is computed by feeding both images through the network and calculating a weighted distance between their internal activations. In this formulation, i indexes the network layers used for the comparison. At a given layer i, the terms yˆhwi and yˆ0i,hw represent the feature activation vectors at spatial position (h,w) for images x and x0, respectively, which have been unit-normalized along the channel dimension. The total height and width of the feature map at this layer are given by Hi and Wi, allowing the H1

iWi h,w operation to compute the spatial average of the distances. The difference between activations is scaled by wi, a learned channel-wise weight vector optimized to match human perceptual judgments, via the element-wise product (⊙). The squared L2 norm (∥·∥22) is then used to compute the distance between these weighted vectors. Finally, the total LLPIPS is the sum of these spatially-averaged distances across all included layers i.

The LPIPS loss on the signals (normals, elongation, intensity, and validity) is calculated by:

LLPIPSsignal = λsignalLLPIPS(fsignalL ,ˆfsignalL ) (6)

Here, λsignal is the corresponding weighting factor for each specific signal type.

KL Divergence Regularization. The KL divergence loss, which regularizes the latent space to follow a standard normal distribution, is calculated by:

LKL =

D

- 1

- 2

λKL

j=1

µ2j + σj2 − log(σj2) − 1 (7)

This term represents the Kullback-Leibler divergence between the encoder’s output distribution, N(µ,σ2), and the prior, N(0,I). Here, D is the dimensionality of the VAE’s latent space. For each latent dimension j, the encoder outputs a mean µj and a variance σj2. Finally, λKL is the hyperparameter that balances this regularization term against the reconstruction losses.

#### B.8. DAgger Training

Dataset Aggregation (DAgger) [35] is originally an imitation learning algorithm designed to mitigate the compounding errors of behavioral cloning. It iteratively collects and

aggregates data by querying an expert π∗ for optimal actions a∗ on states s visited by the current policy πi. A new policy πi+1 is then trained on this aggregated dataset.We adapt DAgger to autoregressive video generation, treating it as a sequential decision-making process to combat temporal inconsistency.

In our case, we introduce DAgger for Video Generation. We map the components as follows. (a) Policy π: the video generation model, which predicts the next frame. (b) State st: the sequence of previously generated frames, st = {f1,f2,...,ft}. (c) Action at: the generated next frame, at = ft+1. (d) Expert π∗: a mechanism (e.g., human evaluator, critic model, or ground-truth data) that provides a “correct” next frame a∗t given a policy-generated state st.

First, we train a base model to generate the current frame conditioned on the ground-truth previous frame. We then utilize the base model to auto-regressively generate rollout frames for all segments in the training set. These generated frames serve as a “degraded” dataset for augmentation. We train an improved model by randomly substituting the ground-truth history with these generated frames during training. This exposes the model to its own accumulation errors, making this model significantly more robust than the base model. While this process can be repeated iteratively, we find that a single iteration yields satisfactory rollout quality. For the DAgger training phase, we set the rollout horizon to 6 steps. Although each training segment contains approximately 35 frames, we find that training on this shorter rollout window is empirically sufficient to achieve robust performance, avoiding the computational cost of full-sequence training.

### C. Limitations and Potential Solutions

Our approach achieves state-of-the-art per-frame generative quality, with our multi-modal diffusion model serving as a high-fidelity backbone for static scenes. We then leverage this powerful single-frame model for video synthesis by extending it auto-regressively, conditioning each new frame on the previously generated one. A limitation is, while our DAgger finetuning strategy effectively mitigates shortterm error accumulation, temporal drift remains a known challenge for long-horizon sequences (e.g., > 30 seconds). Over extended rollouts, minor prediction errors such as small geometric drifts in LiDAR or slight visual inconsistencies, can compound. This may lead to a gradual loss of long-range temporal coherence or a perceived drift in sensor calibration. However, this limitation could be addressed by incorporating a more robust longer video generative backbone designed for long-range consistency. A complementary, and more immediate, solution would be to expand the auto-regressive conditioning window. Instead of conditioning only on the single prior frame (t − 1), the model could attend to a richer temporal context (e.g., t − k,...,t − 1).

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

Figure 15. Visualization of synthetic dashcam images rendered from 4DGS across diverse camera settings. The renders demonstrate high visual fidelity and realism, effectively simulating the characteristics of in-the-wild footage used for training.

This would provide stronger priors for maintaining objectlevel and scene-level consistency over time. While outside the primary scope of this work, we leave these promising directions for future exploration.

### D. Synthetic Cameras from 4DGS

One important component of our pipeline is the utilization of 4D Gaussian Splatting (4DGS) to synthesize paired training data by simulating third-party camera views. As illustrated in Figure 15, the synthetic dashcam images rendered via our 4DGS pipeline exhibit high photorealism, faithfully mimicking the optical characteristics and environmental complexity of real-world dashcam footage.

Crucially, our diffusion model is trained to map these synthetic inputs (Isynth), which may contain minor reconstruction artifacts such as floaters or slight blur, to pristine, ground-truth real sensor data (Oreal). This training objective effectively functions as a denoising task, forcing the network to learn robust spatial and semantic mappings between the monocular view and the target sensor suite, rather than overfitting to low-level input artifacts. Consequently, at inference time, the model demonstrates significant robustness when presented with sub-optimal or noisy realworld dashcam inputs, successfully generating coherent and geometrically consistent AV logs.

