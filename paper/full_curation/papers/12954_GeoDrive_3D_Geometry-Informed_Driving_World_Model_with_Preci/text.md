# GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control

Anthony Chen∗1,2, Wenzhao Zheng∗3, Yida Wang∗2 Xueyang Zhang2, Kun Zhan2, Peng Jia2, Kurt Keutzer3, Shanghang Zhang†1

1State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University 2Li Auto Inc. 3UC Berkeley

arXiv:2505.22421v2[cs.CV]29May2025

Code: https://github.com/antonioo-c/GeoDrive

3D Geometry Consistent Generation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

GTOurs

Input Fame

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

+ GT trajectory

Novel Trajectory Generalization Object Editing

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

+ left@3m trajectory

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Object Trajectory Control

[Figure 28]

+ up@3m trajectory

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

+ reverse trajectory

- Figure 1: GeoDrive enables precise trajectory following, correct novel view synthesis, and dynamic scene editing in autonomous driving scenarios. Our method integrates robust 3D conditions into driving world models, enhancing spatial understanding and action controllability.

### Abstract

Recent advancements in world models have revolutionized dynamic environment simulation, allowing systems to foresee future states and assess potential actions. In autonomous driving, these capabilities help vehicles anticipate the behavior of other road users, perform risk-aware planning, accelerate training in simulation,

*Equal Contribution. †Corresponding Author.

Preprint. Under review.

and adapt to novel scenarios, thereby enhancing safety and reliability. Current approaches exhibit deficiencies in maintaining robust 3D geometric consistency or accumulating artifacts during occlusion handling, both critical for reliable safety assessment in autonomous navigation tasks. To address this, we introduce GeoDrive, which explicitly integrates robust 3D geometry conditions into driving world models to enhance spatial understanding and action controllability. Specifically, we first extract a 3D representation from the input frame and then obtain its 2D rendering based on the user-specified ego-car trajectory. To enable dynamic modeling, we propose a dynamic editing module during training to enhance the renderings by editing the positions of the vehicles. Extensive experiments demonstrate that our method significantly outperforms existing models in both action accuracy and 3D spatial awareness, leading to more realistic, adaptable, and reliable scene modeling for safer autonomous driving. Additionally, our model can generalize to novel trajectories and offers interactive scene editing capabilities, such as object editing and object trajectory control.

### 1 Introduction

Driving world models simulating 3D dynamic environments enable critical capabilities including trajectory-consistent view synthesis [73], physics-compliant motion prediction [26], and safety-aware scenario reconstruction [73] and generation [13, 42]. Particularly, generative video models have emerged as effective tools for ego-motion forecasting and dynamic scene reconstruction [4, 21, 58]. Their ability to synthesize trajectory-faithful visual sequences proves crucial for developing autonomous systems that anticipate environmental interactions while maintaining physical plausibility.

Despite these advancements, most existing methods lack sufficient 3D geometric awareness due to their reliance on 2D space optimization [13]. This shortcoming results in structural incoherence across novel views and physically implausible object interactions [75, 60], which is particularly detrimental for safety-critical tasks like collision avoidance in dense traffic. Also, existing methods usually depend on dense annotations (e.g., HD-map sequences and 3D bounding box tracks) for controllability [60, 34], which only reproduce prescribed motions without understanding vehicle dynamics. A more flexible approach is to infer dynamic priors from single (or few) images while conditioning on the desired ego-trajectory. However, current methods that fine-tune on numerical camera parameters lack 3D geometry awareness, compromising their action controllability and consistency [13, 1, 23]. A reliable driving world model should satisfy three criteria: 1) rigid spatiotemporal coherence across static infrastructure and dynamic agents; 2) 3D controllability over ego-vehicle trajectories; and 3) kinematically constrained motion patterns for non-ego agents.

We achieve these demands through a hybrid neural-geometric framework that explicitly enforces 3D geometry consistency across generated sequences. We first build a 3D structural prior from monocular input and then perform projective rendering along user-specified camera trajectories to generate geometrically grounded conditioning signals. We further employ cascaded video diffusion to refine these projections through 3D-attentive denoising, jointly optimizing photometric quality and geometric fidelity. For dynamic objects, we introduce a physics-guided editing module which transforms agent appearances under explicit motion constraints to ensure physically plausible interactions. Our experiments demonstrate that GeoDrive significantly enhances the performance of controllable driving world models. Specifically, our method improves ego-car action controllability, reducing trajectory following errors by 42% compared to the Vista [13] model. Additionally, it achieves notable enhancements in video quality metrics, including LPIPS, PSNR, SSIM, FID, and FVD. Furthermore, our model generalizes effectively to novel view synthesis tasks, surpassing StreetGaussian in generated video quality. Beyond trajectory conditioning, GeoDrive offers interactive scene editing capabilities, such as dynamic object insertion, replacement, and motion control. Additionally, by integrating real-time visual input with predictive modeling, we enhance the decision-making process of vision-language models, providing an interactive simulation environment that enables safer and more effective trajectory planning.

### 2 Related Work

Driving World Models. World models have become a cornerstone for enabling intelligent agents to anticipate and act in complex, dynamic environments, with autonomous driving presenting unique challenges due to its large field of view, highly dynamic scenes, and the need for robust generalization. Recent research has explored a variety of generative frameworks for future prediction, leveraging representations such as point clouds [11, 38, 30, 82, 28, 80, 66, 27, 77], occupancy grids [37, 6, 41, 88, 15, 44, 92, 65, 59, 74], and images [60, 86, 85, 47, 61, 89, 9]. Point-cloud-based methods leverage the detailed geometric information captured by LiDAR to predict future states and enable precise modeling of spatial geometry and dynamic interactions [82, 28, 80]. Occupancy-grid-based methods further discretize the environment into voxel grids for more fine-grained and geometrically consistent modeling of scene evolutions [41, 88, 15, 44, 92, 65, 59, 74].

Image-based world models stand out for holding more promise for scaling up due to sensor flexibility and data accessibility [60, 86, 85, 47, 61, 12, 34, 40]. They usually leverage powerful generative models to capture the complex visual dynamics of real-world environments, making them particularly valuable for perception and planning tasks [25, 24, 54, 67, 84, 55, 62, 76, 72, 14, 13, 91, 35, 50]. Although existing generative models (e.g., DriveDreamer [60] and DrivingDiffusion [34]) achieve accurate scene control by conditioning on dense annotations (e.g., HD-map sequences and long tracks of 3D bounding boxes), they can only reproduce prescribed motions without truly understanding vehicle dynamics. A more flexible alternative is to infer dynamic priors directly from a single (or a few) images while simultaneously conditioning on the desired ego-trajectory. Recent systems such as Vista [13], Terra [1], and GAIA 1&2 [23, 52] enable action-conditioned generation by injecting raw numerical control vectors directly into the generative backbone. However, since control vectors are not explicitly grounded in the visual latent space, the resulting action signals are weak and often lead to unstable control, demanding larger training datasets for convergence [13, 23]. Differently, our approach renders action commands as visual conditioning inputs that are naturally aligned with the generative latent space, which delivers a substantially stronger control signal and yields significantly more stable and reliable generation results.

Conditional Video Generation. Generative diffusion models have evolved from text-to-image systems into fully multimodal engines that can synthesize entire video sequences on demand. Throughout this progression, the research focus has steadily shifted toward conditional generation—giving users explicit levers to guide the output. Milestones such as ControlNet [83], T2I-Adapter [45], and GLIGEN [36] first embedded conditioning signals into text-to-image pipelines; follow-up studies extended the idea to video, allowing control with RGB key-frames [3, 70, 68], depth maps [69, 10], object trajectories [79, 48], or semantic masks [49, 8]. Yet steering the 6-DoF camera path remains difficult. Coarse LoRA-based motion classes [16, 3], numeric-matrix conditioning [64], depth-warping schemes [46], and Plücker-coordinate encodings [71, 19] each fall short—either through imprecise control, limited domain coverage, or an indirect mapping from numbers to pixels.

Planners and safety modules require frame-level accuracy, so generators such as DriveDreamer [60] and DrivingDiffusion [34] lean on dense HD-map sequences and long 3D box tracks to lock the scene to a predefined route. Other systems like Vista [13], GAIA 1&2 [23, 52], add control vectors directly into the backbone features, but the mismatch between numerical commands and visual features weakens the signal, slows optimisation, and often produces drift. In this work, we propose to use explicit visual conditions for accurate ego trajectory control.

### 3 Methodology

Given an initial reference image I0 ∈ RH×W×3 and ego-vehicle trajectory {Ct}Lt=1, our framework synthesizes realistic future frames that follow the input trajectory. We leverage 3D geometric information from the reference image to guide world modeling. First, we reconstruct a 3D representation (Sec. 3.1), then render video sequences along user-specified trajectories with dynamic object handling (Sec. 3.2). The rendered video provides geometric guidance for generating spatio-temporally consistent videos that follow the input trajectory (Sec. 3.3). See Figure 2 for illustration.

[Figure 38]

❄ 🔥 🔥 🔥 🔥

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Gaussian noise

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

❄ 🔥

[Figure 55]

[Figure 56]

Trainable ❄ Frozen

[Figure 57]

❄

[Figure 58]

VAE

|Video|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

❄

[Figure 63]

[Figure 64]

###### ❄ C

|MonST3R|
|---|

VAE

o Diffusion Transformers

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

❄

[Figure 70]

VAE

Dynamic Aware Rendering

[Figure 71]

[Figure 72]

vehicle positions

[Figure 73]

[Figure 74]

🔥

[Figure 75]

[Figure 76]

static renders dynamic renders

Dynamic Aware Renders

❄

[Figure 77]

[Figure 78]

| | |
|---|---|
| |C|

[Figure 79]

[Figure 80]

Condtion Encoder

VAE

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

First Frame PointCloud camera trajectories

- Figure 2: Overview of our training pipeline. We use a pretrained dense stereo model to obtain 3D point clouds and camera trajectories. A dynamic video is rendered from the first-frame point cloud using our dynamic editing technique. The noisy latent representation and rendered video are encoded via a VAE and concatenated as input for our condition encoder, modulating the DiT model’s features. The DiT then generates photorealistic video that accurately follows the specified action conditions.

#### 3.1 Extracting 3D Representations from Reference Image

[Figure 85]

Gaussian noise

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

❄ 🔥

[Figure 95]

[Figure 96]

Trainable ❄ Frozen

[Figure 97]

❄

[Figure 98]

[Figure 99]

To utilize 3D information for 3D-consistent generation, we first construct a 3D representation from the single input image I0. We employ MonST3R [81], an off-the-shelf dense stereo model that

[Figure 100]

VAE

|with|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

❄

[Figure 104]

w our training paradigm. During inference, we duplicate the reference image to satisfy MonST3R’s cross-view matching requirements. Given RGB frames {It}Tt=0, MonST3R predicts per-pixel 3D coordinates {Ot}Tt=0 and confidence scores {Dt}Tt=0 via cross-view feature matching across frames

3D geometry and camera poses, aligning

[Figure 105]

[Figure 106]

###### ❄ C

|Monst3r<br><br>simultaneously predicts|
|---|

VAE

###### Video Diffusion Transformers

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

❄

VAE

Dynamic Aware Rendering

[Figure 113]

[Figure 114]

[Figure 115]

🔥

[Figure 116]

[Figure 117]

Dynamic Aware Renders

[Figure 118]

❄

|T|=|
|---|---|
| |C|

{Ot}Tt=0,{Dt}Tt=0 MonST3R {It}Tt=0 , (1)

Condtion Encoder

VAE

[Figure 119]

[Figure 120]

First Frame PointCloud camera trajectories

where Oi,jt ∈ R3 denotes the metric-space position of pixel (i,j) in the tth reference frame, and Di,jt ∈ [0,1] measures reconstruction reliability. By thresholding D0 at τ (typically τ = 0.65), the colored point cloud for the tth reference frame yields as

[Figure 121]

Pt = (Oi,jt , Iti,j) | Di,jt > τ . (2) To counteract the imbalance between valid and invalid matches across the sequence, the confidence map D0 is trained with a focal loss. Further, to disentangle static scene geometry from moving objects, MonST3R employs a transformer-based decoupler. This module processes the initial features of the reference frame (enriched by cross-view context) and separates them into static and dynamic components. The decoupler uses learnable prompt tokens to split attention maps: static tokens attend to large planar surfaces, and dynamic tokens attend to compact, motion-rich regions. By excluding dynamic correspondences, we obtain a robust camera pose estimate

[Figure 122]

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

π(Ct Oi,jt ) − pi,jt 22, (3)

Cˆt = arg min

Ct

(i,j)∈Fstatic

where π denotes the perspective projection operator, and only static feature matches are used. Compared to conventional structure-from-motion [51], this strategy reduces pose error by 38% in dynamic urban scenes [81]. The resulting point cloud P0 then serves as our geometric scaffold.

#### 3.2 Rendering 3D Videos with Dynamic Editing

To achieve precise input trajectory following, our model renders a video that serves as a visual guide for the generation process. We project the reference point cloud P0 through each user-provided camera configuration Ct = (Rt,Tt,ft) using standard projective geometry techniques. Each 3D point Pwi ∈ P0 undergoes a rigid transformation into the camera coordinate system Pci = RtPwi +Tt, followed by perspective projection using the camera’s intrinsic matrix Kt, yielding image coordinates pi = ftP

cy i

cx i

Pciz + W2 , ftP

Pciz + H2 . We only consider valid projections within a depth range of Pc

i ∈ [0.1,100.0] meters and use z-buffering to handle occlusions, ultimately producing the rendered view I˜t for each camera position.

z

###### Condition Frame Following Frames

|[Figure 133]|
|---|

[Figure 134]

[Figure 135]

[Figure 136]

###### GroundTruth

[Figure 137]

[Figure 138]

First frame 3D representation

|Misaligned vehicle motion|
|---|

[Figure 139]

[Figure 140]

[Figure 141]

Camera poses

Default rendering (ViewCrafter, Gen3C)

[Figure 142]

Vehicle boudingboxes

|Aligned vehicle motion|
|---|

[Figure 143]

[Figure 144]

[Figure 145]

+

(Required during training; Optional during inference)

Dynamic rendering (Ours)

- Figure 3: Illustration of dynamic edit design. Compared with default rendering, it effectively reduces disparity between static rendering and dynamic real-world scenarios.

#### Table 1: Quantitative results of generation quality and action fidelity on NuScenes [7] validation subset. We outperform baseline methods on every metric while requiring much less training data.

Prediction Quality Action Fidelity LPIPS ↓ PSNR ↑ SSIM ↑ FID ↓ FVD ↓ ADE×102 ↓ FDE×102 ↓

Method Data Scale

Vista [13] 1740h 0.351 20.086 0.621 8.35 163.7 2.77 5.28 Terra [1] 1740h 0.455 18.42 0.553 26.73 787.54 5.57 11.8

GeoDrive (Ours) 5h 0.303 21.979 0.6535 7.17 85.22 1.62 3.1

Limitations of Static Rendering. Since we utilize only the first frame point cloud, the rendered scene remains static throughout the sequence. This creates a significant discrepancy with real-world autonomous driving contexts, where vehicles and other dynamic objects are in constant motion. The static nature of our rendering fails to capture the dynamic essence that distinguishes autonomous driving datasets from traditional static scenes.

Dynamic Editing. To address this limitation, we propose dynamic editing to produce renderings R with static backgrounds and moving vehicles. Specifically, when users provide a sequence of 2D bounding box information for moving vehicles in the scene, we dynamically adjust their positions to create the illusion of motion in the rendering. This approach not only guides the ego-vehicle’s trajectory during the generation process but also directs the movement of other vehicles in the scene. Fig. 3 provides an illustration of this process. Such a design significantly reduces the disparity between static rendering and dynamic real-world scenarios while enabling flexible control over other vehicles—a capability that existing methods like Vista [13] and GAIA [23] fail to achieve.

#### 3.3 Dual-Branch Control for Spatio-Temporal Consistency

While the point cloud-based rendering accurately preserves geometric relationships between views, it suffers from several visual quality issues. The rendered views often contain substantial occlusions, missing areas due to limited sensor coverage, and reduced visual fidelity compared to real camera images. To enhance the quality, we adapt a latent video diffusion model [5] to refine projected views while preserving 3D structural fidelity through specialized conditioning.

Building on this, we further refine the integration of contextual features into a pre-trained diffusion transformer (DiT), drawing inspiration from the methodology introduced by VideoPainter [2]. However, we introduce key distinctions tailored to our specific needs. We employ dynamic renderings to capture temporal and contextual nuances, providing a more adaptive representation for the generation process. Let δϕ(zt,t,C) represent the feature output at layer i of our modified DiT backbone δϕ, where zR denotes the dynamic renderings latent via VAE encoder E and zt is the noisy latent at timestep t.

Ground Truth

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Condition Frame

[Figure 151]

Fail to precisly follow condition

Vista

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Condition Trajectory

|[Figure 157]|
|---|

Ours

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Ground Truth

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Condition Frame

[Figure 168]

Completely misaligned

Vista

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Condition Trajectory

[Figure 174]

Ours

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Reference Line

Frame 0 Frame 10 Frame 15 Frame 20 Frame 25

- Figure 4: Qualitative comparison of action fidelity under the same conditional frame and action control. Our model precisely follows desired trajectory, while Vista [13] produce misaligned results.

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Vista Zoom in Ours Vista Condition frame Ours

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Prediction Quality World Understanding

Moving backward

Car crash

Wait for traffic

Turning Le 

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

| |
|---|
| |

[Figure 199]

[Figure 200]

| |
|---|

[Figure 201]

Vista Zoom in Ours Vista Condition frame Ours

[Figure 202]

[Figure 203]

Prediction Quality World Understanding

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

| |
|---|

[Figure 212]

[Figure 213]

| |
|---|

[Figure 214]

| |
|---|

[Figure 215]

[Figure 216]

| |
|---|

Crash into front car

Wait until front car moves

- Figure 5: Qualitative Comparisons: Left - Enhanced visual fidelity in our predictions; Right Superior scene dynamics understanding. These renderings are processed through a lightweight condition encoder, which extracts essential background cues without duplicating extensive portions of the backbone architecture. The integration of features from the condition encoder into the frozen DiT is formulated as follows:

[Figure 217]

[Figure 218]

##### δϕ(zt,t,C)i = δϕ(zt,t,C)i + W γϕenc([zt,zR],t)i//M

, (4)

2

where γϕenc denotes the condition encoder processing the concatenated input of noisy latent zt and renderings latent zR, with M representing the total number of layers in the DiT backbone. W is a learnable linear transformation initialized to zero to prevent noise collapse in early training. The extracted features are selectively fused into the frozen DiT in a structured manner, ensuring that only relevant contextual information guides the generation process. The final video sequence is decoded

via the frozen VAE decoder D as Iˆt = D(zt(0)).

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

GroundTruth

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

StreetGaussians

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

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Leftshift@3mRightshift@3mLeftshift@3mRightshift@3m

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

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

Ours

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

- Figure 6: Qualitative comparison on novel-view synthesis on Waymo validation subset. Our model generates sharp results for deviated trajectories in a zero-shot manner, whereas the reconstruction-based method StreetGaussian [73] produces significant artifacts.

By limiting training to condition encoder gϕ alone (6% of total parameters), we maintain the pretrained model’s photorealism and gain precise camera control. Temporal coherence arises naturally from the video transformer’s dynamics modeling and the geometric consistency of {I˜t} features across frames, enabling trajectory-faithful video synthesis.

### 4 Experiments and Applications

- 4.1 Experimental Settings

Training Configuration. We train exclusively on nuScenes [7], processing each clip through MonST3R to obtain metric-scale 3D reconstructions and camera trajectories. 3D reconstructions of the initial frame P0 undergoes projective rendering along estimated trajectories via a differentiable rasterizer, where Dynamic Editing leverages 2D bounding box annotations to edit vehicle positions. We curate 25,109 video-condition pairs for training. We freeze the base diffusion model (CogVideo5B-I2V [22]) while training the condition encoder for 28,000 steps at learning rate 1 × 10−5 for 4 days. More details on training, inference and model configuration are in Appendix B.

- 4.2 Trajectory Following

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

Table 2: Quantitative results of generation quality on NuScenes validation fullset.

Benchmark and Baselines. We compare GeoDrive to two most-relevant baselines that condition on single image and ego action (Vista [13], Terra [1]), as well as several other driving world models. We adhere to Vista’s protocol by computing trajectory from sensor and calibration data that spans the 25-frame clip, as their condition input. We estimate our condition camera poses by running MonST3R on GT video. While we condition on different modalities, the trajectories for all methods are extracted from the same ground-truth video clip, ensur-

Models Data Scale FID ↓ FVD ↓

DriveGAN [31] 160h 73.4 502.3 DriveDreamer [60] 5h 14.9 340.8 DriveDreamer-2 [86] 5h 25.0 105.1 WoVoGen [39] 5h 27.6 417.7 Drive-WM [44] 5h 15.8 122.7 GenAD [75] 1740h 15.4 184.0

Vista [13] 1740h 6.6 167.7 GEM [18] 4000h 10.5 158.5 GeoDrive (Ours) 5h 4.1 61.6

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

|[Figure 295]<br><br>[Figure 296]|
|---|

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

|[Figure 303]<br><br>| |
|---|
<br><br>[Figure 304]|
|---|

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Condition Frame Vehicle Bboxes Action Condition Model Output

- Figure 7: Qualitative Results on Vehicle Manipulation. Our approach allows for the manipulation of vehicle movement directions within a scene by specifying different bounding boxes.

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Table 4: Ablation Studies on NuScenes validation subset. D.E. represents dynamic editing. w/o dual-branch means we adjust input channels to adapt renders and finetune backbone.

Table 3: Quantitative results for NVS on Waymo validation subset. GeoDrive is trained solely on the NuScenes dataset, yet it can generalize to Waymo scenes in a zero-shot, feed-forward manner.

Remove

Left@3m Right@3m FID ↓ FVD ↓ FID ↓ FVD ↓

Method

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Method FID FVD ADE×102

w/o D.E. 7.01 88.68 3.68 w/o dual-branch 8.83 74.76 3.45 GeoDrive 7.17 85.22 1.62

Edit

StreetGS [73] 63.84 1438.89 69.55 1526.62 Ours 67.13 1245.23 65.67 1422.63

Original Scene

Edited Scene

ing aligned action conditions. We evaluate all methods on NuScenes validation set. For trajectory control precision evaluation, we sample a subset of 1087 videos with balanced driving trajectories. Visual quality is quantified through PSNR, SSIM [63], LPIPS [29], FID [20], and FVD [57]. While trajectory fidelity metrics employ Average Displacement Error (ADE) and Final Displacement Error (FDE).

Quantitative Results. The quantitative results on nuScenes validation subset are presented in Table 1. GeoDrive outperforms baselines in all metrics. Specifically, our trajectory-following ability is significantly better than the baselines, yet requiring 99.7% less data, showing the effectiveness of our method. In Table 2, GeoDrive outperforms all baselines on FID & FVD results.

Qualitative Results. As shown in Figure 4, our method produce agents adhering more precisely to the specified paths compared to baseline methods, which often deviate from the intended trajectory. In addition, our generated videos exhibit a better understanding of the driving environment. As illustrated in Figure 5, our results are noticeably sharper and contain fewer artifacts. Furthermore, our model demonstrates a stronger grasp of scene dynamics compared to baseline methods. For example, in Figure 5, our model correctly anticipates that the car behind should wait for the bus ahead to move, whereas Vista erroneously accelerates the car forward, resulting in a collision. Moreover, we examine model performance on reverse trajectory, which is not included in the training data. Demonstrated in Figure 10 in Appendix C, GeoDrive successfully generalizes to the unseen trajectory, while Vista fails to perform. Moreover, thanks to our 3D geometric condition, the building structure remains exactly the same across generation.

#### 4.3 Novel View Synthesis

Benchmark and Baseline. We compare GeoDrive to scene reconstruction method, StreetGaussians [73]. We evaluate on the Waymo validation set and filter out 5 scenes for testing. The novel trajectory is created by horizontally shifting from the original trajectory of the frontal camera. We assess generation quality using FID and FVD since there is no ground truth for novel trajectories.

|<br><br>|
|---|

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

|[Figure 348]<br><br>| |
|---|
<br><br>[Figure 349]|
|---|

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

Condition Frame Vehicle Bboxes Action Condition Model Output

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

Remove

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

Edit

Original Scene

Edited Scene

- Figure 8: Qualitative Results on Scene Editing. Our approach enables the removal or replacement of vehicles within a scene, allowing for the prediction of seamless future scenarios.

[Figure 364]

[Figure 365]

<Thinking…> Here are three possible action you could take,

let me verify if they are paussible Planning 1:

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

<Thinking…> We are too close to the front car. Let’s try another one. Planning 2:

[Figure 370]

[Figure 371]

[Figure 372]

<Thinking…> Ops! We are driving onto the lane. Here’s another solution. Planning 3:

[Figure 373]

What should I do?

Input

Iterative Planning and Verifying

Output

VLA:

GeoDrive:

<Thinking…> This seems safe. Let’s go with this one!

VLA: VLA:

VLA & GeoDrive

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

[Figure 385]

GeoDrive: GeoDrive:

[Figure 386]

Driver: VLA:

- Figure 9: Illustration of application to VLA planning. By simulating each possible planned trajectory, we can assist the VLA model in refining its decisions until it reaches the optimal decision.

Quantitative Results. As demonstrated in Table 3, our method achieves lower (and thus better) FID & FVD scores compared to the reconstruction-based baselines. This is due to the difficulty reconstruction methods face in recovering scene structures from the sparsely observed views.

Qualitative Results. Figure 6 illustrates that while StreetGaussians [73] generates projectively correct renderings along given trajectories, it exhibits severe geometric distortions under viewpoint shifts. Such degrades reveal fundamental limitations in 3D scene reconstruction from sparse observational data, particularly the inability to resolve occlusion boundaries and low-textured supervisions. GeoDrive maintains trajectory adherence while preserving photorealistic rendering.

#### 4.4 Applications

In this section, we demonstrate the versatility and practical impact of GeoDrive across several key applications in autonomous driving. It not only supports advanced scene editing, such as object editing and object trajectory control, but also it can serve as an interactive environment that assists high-level planning models, such as the VLA, in making safer and more informed decisions.

Object Trajectory Control. As illustrated in Figure 7, given extra condition input (i.e., a sequence of 2D bounding boxes) for other vehicles, we can apply dynamic augmentation (Sec. 3.2) and thus control the trajectory of vehicles appearing in the condition frame.

Object Editing. GeoDrive supports intuitive and powerful scene editing capabilities, including modifying existing objects or removing unwanted elements from the environment. This is achieved by using off-the-shelf image editing models [33] on the condition frame. As shown in Fig. 8, our method offers significant flexibility in manipulating driving scenes. This level of control is invaluable for generating targeted training data, analyzing the influence of specific objects or configurations, and creating diverse scenarios from a limited amount of captured data.

Assistance for VLA planning. GeoDrive enhances the decision-making of VLA (Visual Language Action) models by providing an interactive simulation environment for evaluating driving actions

( [90, 56, 87, 17]). As illustrated in Figure 9, it integrates real-time visual input with predictive modeling, allowing the VLA system to simulate outcomes of planned trajectories. This helps foresee hazards, such as proximity to other vehicles or lane deviations, and assess action safety. Consequently, the VLA model refines its decisions by selecting actions predicted to be safe and effective, which ensures driving decisions are contextually appropriate and prioritize safety.

#### 4.5 Ablation Studies

Impact of Dynamic Editing. To quantify the impact of our dynamic editing strategy, we report the performance of our model trained with and without this component on key evaluation metrics as shown in Table 4.

Impact of Dual-branch Architecture. We further investigate the effectiveness of adopting a dualbranch architecture compared with single-branch architecture. We evaluate the performance of different architectures on key evaluation metrics. The results are shown in Table 4.

### 5 Conclusions and Limitations

We presented GeoDrive, a video diffusion world model for autonomous driving, enhancing action controllability and spatial accuracy via explicit meter-level trajectory control and direct visual conditioning. Our method reconstructs 3D scenes, renders along desired trajectories, and refines the output with video diffusion. Evaluations demonstrate superior performance over existing models in visual realism and action adherence, enabling applications like non-ego view generation and scene editing, thus setting a new benchmark. However, our performance depends on the accuracy of depth and pose estimation from MonST3R, and the reliance solely on image and trajectory input for world prediction poses a challenge. Future work will explore incorporating text conditions and VLA understanding to further improve realism and consistency.

### References

- [1] Hidehisa Arai, Keishi Ishihara, Tsubasa Takahashi, and Yu Yamaguchi. Act-bench: Towards action controllable world models for autonomous driving, 2024.
- [2] Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Any-length video inpainting and editing with plug-and-play context control, 2025.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023.
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. CVPR, 2023.
- [6] Daniel Bogdoll, Yitian Yang, and J Marius Zöllner. Muvo: A multimodal generative world model for autonomous driving with geometric representations. arXiv preprint arXiv:2311.11762, 2023.
- [7] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yuxin Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A Multimodal Dataset for Autonomous Driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2020.
- [8] Anthony Chen, Jianjin Xu, Wenzhao Zheng, Gaole Dai, Yida Wang, Renrui Zhang, Haofan Wang, and Shanghang Zhang. Training-free regional prompting for diffusion transformers, 2024.
- [9] Xiaowei Chi, Hengyuan Zhang, Chun-Kai Fan, Xingqun Qi, Rongyu Zhang, Anthony Chen, Chi min Chan, Wei Xue, Wenhan Luo, Shanghang Zhang, and Yike Guo. Eva: An embodied world model for future video anticipation, 2024.
- [10] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proc. IEEE Int. Conf. Comput. Vis., 2023.

- [11] Hehe Fan and Yi Yang. Pointrnn: Point recurrent neural network for moving point cloud processing. arXiv preprint arXiv:1910.08287, 2019.
- [12] Dechen Gao, Shuangyu Cai, Hanchu Zhou, Hang Wang, Iman Soltani, and Junshan Zhang. Cardreamer: Open-source learning platform for world model based autonomous driving. arXiv preprint arXiv:2405.09111, 2024.
- [13] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. Proc. Adv. Neural Inf. Process. Syst., 2024.
- [14] Anant Garg and K Madhava Krishna. Imagine-2-drive: High-fidelity world modeling in carla for autonomous vehicles. arXiv preprint arXiv:2411.10171, 2024.
- [15] Songen Gu, Wei Yin, Bu Jin, Xiaoyang Guo, Junming Wang, Haodong Li, Qian Zhang, and Xiaoxiao Long. Dome: Taming diffusion model into high-fidelity controllable occupancy world model. arXiv preprint arXiv:2410.10429, 2024.
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In Proc. Int. Conf. Learn. Represent., 2024.
- [17] Ziang Guo, Konstantin Gubernatorov, Selamawit Asfaw, Zakhar Yagudin, and Dzmitry Tsetserukou. Vdt-auto: End-to-end autonomous driving with vlm-guided diffusion transformers, 2025.
- [18] Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Pedro Rezende, Yasaman Haghighi, David Brüggemann, Isinsu Katircioglu, Lin Zhang, Xiaoran Chen, Suman Saha, et al. Gem: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control. arXiv preprint arXiv:2412.11198, 2024.
- [19] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.
- [20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video Diffusion Models. arXiv preprint arXiv:2204.03458, 2022.
- [22] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [23] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. Technical Report arXiv:2309.17080, 2023.
- [24] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A Generative World Model for Autonomous Driving. arXiv preprint arXiv:2309.17080, 2023.
- [25] Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. Drivingworld: Constructingworld model for autonomous driving via video gpt. arXiv preprint arXiv:2412.19505, 2024.
- [26] Bencheng Huang, Shaoyu Liu, Tianheng Chen, Xinggang Shen, Zeming Zhu, Zhe Wang, et al. Vad: Vectorized scene representation for autonomous driving. arXiv preprint arXiv:2303.12077, 2023.
- [27] Shengyu Huang, Zan Gojcic, Zian Wang, Francis Williams, Yoni Kasten, Sanja Fidler, Konrad Schindler, and Or Litany. Neural lidar fields for novel view synthesis. In Proc. IEEE Int. Conf. Comput. Vis., pages 18236–18246, 2023.
- [28] Zanming Huang, Jimuyang Zhang, and Eshed Ohn-Bar. Neural volumetric world models for autonomous driving. In Proc. Eur. Conf. Comput. Vis., pages 195–213. Springer, 2025.
- [29] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Proc. Eur. Conf. Comput. Vis., 2016.

- [30] Tarasha Khurana, Peiyun Hu, Achal Dave, Jason Ziglar, David Held, and Deva Ramanan. Differentiable raycasting for self-supervised occupancy forecasting. In Proc. Eur. Conf. Comput. Vis., pages 353–369. Springer, 2022.
- [31] Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. DriveGAN: Towards a Controllable High-Quality Neural Simulation. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021.
- [32] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [33] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [34] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scene video generation with latent diffusion model. arXiv preprint arXiv:2310.07771, 2023.
- [35] Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481, 2024.
- [36] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023.
- [37] Xinhao Liu, Moonjun Gong, Qi Fang, Haoyu Xie, Yiming Li, Hang Zhao, and Chen Feng. Lidar-based 4d occupancy completion and forecasting. In Proc. IEEE/RSJ Int. Conf. Intell. Robot. Syst., pages 11102–11109. IEEE, 2024.
- [38] Fan Lu, Guang Chen, Zhijun Li, Lijun Zhang, Yinlong Liu, Sanqing Qu, and Alois Knoll. Monet: Motion-based point cloud prediction network. IEEE Trans. Intell. Transp. Syst., 23(8):13794–13804, 2021.
- [39] Jiachen Lu, Ze Huang, Zeyu Yang, Jiahui Zhang, and Li Zhang. Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation. In Proc. Eur. Conf. Comput. Vis., pages 329–345. Springer, 2025.
- [40] Enhui Ma, Lijun Zhou, Tao Tang, Zhan Zhang, Dong Han, Junpeng Jiang, Kun Zhan, Peng Jia, Xianpeng Lang, Haiyang Sun, et al. Unleashing generalization of end-to-end autonomous driving with controllable long video generation. arXiv preprint arXiv:2406.01349, 2024.
- [41] Junyi Ma, Xieyuanli Chen, Jiawei Huang, Jingyi Xu, Zhen Luo, Jintao Xu, Weihao Gu, Rui Ai, and Hesheng Wang. Cam4docc: Benchmark for camera-only 4d occupancy forecasting in autonomous driving applications. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pages 21486–21495, 2024.
- [42] Jiageng Mao, Boyi Li, Boris Ivanovic, Yuxiao Chen, Yan Wang, Yurong You, Chaowei Xiao, Danfei Xu, Marco Pavone, and Yue Wang. Dreamdrive: Generative 4d scene modeling from street view images. arXiv preprint arXiv:2501.00601, 2024.
- [43] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shapeguided generation of 3d shapes and textures. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023.
- [44] Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, et al. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pages 15522–15533, 2024.
- [45] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI Conf. Artif. Intell., 2024.
- [46] Norman Müller, Katja Schwarz, Barbara Rössle, Lorenzo Porzi, Samuel Rota Bulò, Matthias Nießner, and Peter Kontschieder. Multidiff: Consistent novel view synthesis from a single image. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024.
- [47] Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida Wang, Xueyang Zhang, et al. Recondreamer: Crafting world models for driving scene reconstruction via online restoration. arXiv preprint arXiv:2411.19548, 2024.
- [48] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. arXiv preprint arXiv:2405.20222, 2024.

- [49] Elia Peruzzo, Vidit Goel, Dejia Xu, Xingqian Xu, Yifan Jiang, Zhangyang Wang, Humphrey Shi, and Nicu Sebe. Vase: Object-centric appearance and shape manipulation of real videos. arXiv preprint arXiv:2401.02473, 2024.
- [50] Alexander Popov, Alperen Degirmenci, David Wehr, Shashank Hegde, Ryan Oldja, Alexey Kamenev, Bertrand Douillard, David Nistér, Urs Muller, Ruchi Bhargava, et al. Mitigating covariate shift in imitation learning for autonomous vehicles using latent space generative world models. arXiv preprint arXiv:2409.16663, 2024.
- [51] Jérôme Revaud, Vincent Leroy, Philippe Weinzaepfel, Boris Chidlovskii, and Gabriela Csurka. Dust3r: Geometric 3d vision made easy. arXiv preprint arXiv:2312.14132, 2023.
- [52] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving, 2025.
- [53] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proc. Int. Conf. Learn. Represent., 2021.
- [54] Shiva Sreeram, Tsun-Hsuan Wang, Alaa Maalouf, Guy Rosman, Sertac Karaman, and Daniela Rus. Probing multimodal llms as world models for driving. arXiv preprint arXiv:2405.05956, 2024.
- [55] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-view image generation from a bird’s-eye view layout. IEEE Robot. Autom. Lett., 2024.
- [56] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models, 2024.
- [57] Thomas Unterthiner, Sjoerd Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. 2018.
- [58] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. ModelScope Text-to-Video Technical Report. arXiv preprint arXiv:2308.06571, 2023.
- [59] Lening Wang, Wenzhao Zheng, Yilong Ren, Han Jiang, Zhiyong Cui, Haiyang Yu, and Jiwen Lu. Occsora: 4d occupancy generation models as world simulators for autonomous driving. arXiv preprint arXiv:2405.20337, 2024.
- [60] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. Proc. Eur. Conf. Comput. Vis., 2024.
- [61] Xiaofeng Wang, Zheng Zhu, Guan Huang, Boyuan Wang, Xinze Chen, and Jiwen Lu. Worlddreamer: Towards general world models for video generation via predicting masked tokens. arXiv preprint arXiv:2401.09985, 2024.
- [62] Yuqi Wang, Ke Cheng, Jiawei He, Qitai Wang, Hengchen Dai, Yuntao Chen, Fei Xia, and Zhaoxiang Zhang. Drivingdojo dataset: Advancing interactive and knowledge-enriched driving world model. arXiv preprint arXiv:2410.10738, 2024.
- [63] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.
- [64] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. arXiv preprint arXiv:2312.03641, 2023.
- [65] Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, and Wenchao Ding. Occllama: An occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272, 2024.
- [66] Lemeng Wu, Dilin Wang, Chengyue Gong, Xingchao Liu, Yunyang Xiong, Rakesh Ranjan, Raghuraman Krishnamoorthi, Vikas Chandra, and Qiang Liu. Fast point cloud generation with straight flows. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pages 9445–9454, 2023.
- [67] Zehuan Wu, Jingcheng Ni, Xiaodong Wang, Yuxin Guo, Rui Chen, Lewei Lu, Jifeng Dai, and Yuwen Xiong. Holodrive: Holistic 2d-3d multi-modal street scene generation for autonomous driving. arXiv preprint arXiv:2412.01407, 2024.

- [68] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. arXiv preprint arXiv:2405.17933, 2024.
- [69] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE Trans. Vis. Comput. Graph., 2024.
- [70] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.
- [71] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Camera-controllable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024.
- [72] Tianyi Yan, Dongming Wu, Wencheng Han, Junpeng Jiang, Xia Zhou, Kun Zhan, Cheng-zhong Xu, and Jianbing Shen. Drivingsphere: Building a high-fidelity 4d world for closed-loop simulation. arXiv preprint arXiv:2411.11252, 2024.
- [73] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In ECCV, 2024.
- [74] Ziyang Yan, Wenzhen Dong, Yihua Shao, Yuhang Lu, Liu Haiyang, Jingwen Liu, Haozhe Wang, Zhe Wang, Yan Wang, Fabio Remondino, et al. Renderworld: World model with self-supervised 3d label. arXiv preprint arXiv:2409.11356, 2024.
- [75] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, Jun Zhang, Andreas Geiger, Yu Qiao, and Hongyang Li. Generalized Predictive Model for Autonomous Driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024.
- [76] Xuemeng Yang, Licheng Wen, Yukai Ma, Jianbiao Mei, Xin Li, Tiantian Wei, Wenjie Lei, Daocheng Fu, Pinlong Cai, Min Dou, et al. Drivearena: A closed-loop generative simulation platform for autonomous driving. arXiv preprint arXiv:2408.00415, 2024.
- [77] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pages 14673–14684, 2024.
- [78] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [79] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.
- [80] Junge Zhang, Feihu Zhang, Shaochen Kuang, and Li Zhang. Nerf-lidar: Generating realistic lidar point clouds with neural radiance fields. In AAAI Conf. Artif. Intell., volume 38, pages 7178–7186, 2024.
- [81] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arxiv:2410.03825, 2024.
- [82] Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Copilot4d: Learning unsupervised world models for autonomous driving via discrete diffusion. In Proc. Int. Conf. Learn. Represent., 2024.
- [83] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proc. IEEE Int. Conf. Comput. Vis., 2023.
- [84] Yumeng Zhang, Shi Gong, Kaixin Xiong, Xiaoqing Ye, Xiao Tan, Fan Wang, Jizhou Huang, Hua Wu, and Haifeng Wang. Bevworld: A multimodal world model for autonomous driving via unified bev latent space. arXiv preprint arXiv:2407.05679, 2024.
- [85] Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, et al. Drivedreamer4d: World models are effective data machines for 4d driving scene representation. arXiv preprint arXiv:2410.13571, 2024.
- [86] Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation. arXiv preprint arXiv:2403.06845, 2024.

- [87] Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Chengyuan Zheng, and Fei Gao. Sce2drivex: A generalized mllm framework for scene-to-drive learning, 2025.
- [88] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In Proc. Eur. Conf. Comput. Vis., pages 55–72. Springer, 2025.
- [89] Wenzhao Zheng, Zetian Xia, Yuanhui Huang, Sicheng Zuo, Jie Zhou, and Jiwen Lu. Doe-1: Closed-loop autonomous driving with large world model. arXiv preprint arXiv: 2412.09627, 2024.
- [90] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C. Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model, 2025.
- [91] Yunsong Zhou, Michael Simon, Zhenghao Peng, Sicheng Mo, Hongzi Zhu, Minyi Guo, and Bolei Zhou. Simgen: Simulator-conditioned driving scene generation. arXiv preprint arXiv:2406.09386, 2024.
- [92] Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Gaussianworld: Gaussian world model for streaming 3d occupancy prediction. arXiv preprint arXiv:2412.10373, 2024.

## Appendix

### A Preliminary Details

#### A.1 Video Diffusion Models

- A diffusion model [53] is built from two stochastic chains: a forward (noising) process q and a reverse (denoising) process pθ. Starting with a clean sample x0∼q0(x0), the forward process incrementally injects Gaussian noise, producing xt = αtx0 + σtϵ, ϵ ∼ N(0,I), where the schedule satisfies

αt2 + σt2 = 1. The reverse process seeks to remove this noise using a neural predictor ϵθ, trained by minimizing

LDM = Et∼U(0,1),ϵ∼N(0,I) ϵθ(xt,t) − ϵ 22 . (5)

To ease the heavy computation of pixel-space generation, latent diffusion models (LDMs) [43] compress each RGB video x∈RL×3×H×W into a lower-dimensional tensor z = E(x)∈RL×C×h×w via a frozen VAE encoder E. Both the forward and reverse chains operate in this latent space, after which a decoder xˆ = D(z) reconstructs the final video.

In this paper, we adopt the pretrained CogVideo-I2V[78] model as our backbone. Its ability to animate a single reference image aligns naturally with our objective of predicting future scenarios based on a single input.

- B Experimental Details

#### B.1 Model Configuration

GeoDrive is built upon a pre-trained image-to-Video Diffusion Transformer CogVideo-5B-I2V [22]. Our lightweight condition encoder clones only the first two layers of pre-trained DiT, taking up only

- 6% of the backbone parameters.

#### B.2 Training Details

We train GeoDrive at 480 × 720 resolution, with learning rate 1 × 10−5 and batch size 1 with AdamW [32] optimizer. During training, only the parameters of our condition encoder are optimized for 28000 steps. The model is trained with 8xA100 80GB GPUs for 4 days.

#### B.3 Inference Details

During inference, given a condition frame input and a trajectory (i.e. camera poses), we first build a 3D representation from the single image. In order to utilize MonST3R [81], we duplicate the input image and get a video input for MonST3R. Next, we render a sequence of images along user-provided camera poses. GeoDrive can optionaly accept object bounding boxes to achieve control over object’s trajectory in the scene, via dynamic editing module (Section 3.2.

For Trajectory Following experiments (Section 4.2), we use the first frame of each video as condition frame, and we estimate trajectory with MonST3R from the video. For fair comparision with baseline methods, We do not include object control as the baseline methods do not accept object bounding box information.

For Novel View Synthesis experiments (Section 4.3), we use the first frame of each video as condition frame, and we estimate the original trajectory with MonST3R from the video. Next, we align the original trajectory with the depth scale from the Lidar point cloud. Then we shift the trajectory left, right and up to obtain novel trajectory input for the experiment.

#### B.4 Benchmark Details

Evaluation Metrics. During evaluation, the visual quality is quantified through PSNR, SSIM [63], LPIPS [29], FID [20], and FVD [57], and the trajectory fidelity is quantified by Average Displacement

Error (ADE) and Final Displacement Error (FDE) upon the trajectory pair {yt,yˆt} estimated via MonST3R:

1 T

ADE =

T

∥yt − yˆt∥2,FDE = ∥yT − yˆT∥2 , (6)

t=1

where yt denotes ground truth poses and yˆt predicted positions, and T is the total number of frames.

### C Additional Results

#### C.1 Generalization to Unseen Trajectory

n Section 4.2, we highlight the superior trajectory-following capabilities of GeoDrive compared to baseline methods. GeoDrive’s controllability naturally extends to unseen trajectories, such as shifted and reverse trajectories (Section 4.3). Figure 10 illustrates this, where both Vista and GeoDrive were provided with a reverse trajectory. Unlike Vista, which only moves forward, GeoDrive accurately follows the instructions, moving first forward and then backward. This discrepancy arises because reverse trajectories are rare in the training data, causing Vista’s implicit action control to struggle with generalization. In contrast, GeoDrive’s design, which utilizes renderings as conditions, allows it to generalize effectively to any trajectory.

Driving Forward

Driving Backward

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

VistaOurs

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

###### Figure 10: Our model can faithfully follow given trajectory and predict consistent future, even when such trajectory is out of training distribution. Yet previous work fail to follow the given action.

