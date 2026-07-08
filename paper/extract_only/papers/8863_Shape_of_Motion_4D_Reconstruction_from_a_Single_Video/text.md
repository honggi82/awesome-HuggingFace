## Shape of Motion: 4D Reconstruction from a Single Video

Qianqian Wang1,2∗, Vickie Ye1∗, Hang Gao1∗, Weijia Zeng3∗, Jake Austin1, Zhengqi Li4, Angjoo Kanazawa1

1UC Berkeley 2Google DeepMind 3UC San Diego 4Adobe Research

# arXiv:2407.13764v2[cs.CV]16Oct2025

|[Figure 1]<br><br>[Figure 2]|
|---|

|[Figure 3]<br><br>[Figure 4]|
|---|

Figure 1. Shape of Motion. Our method enables joint long-range 3D tracking and novel view synthesis from a monocular video of a complex dynamic scene. We render moving elements at a fixed viewpoint across time and visualize estimated 3D motion as colorful trajectories. These trajectories reveal distinct geometric patterns, which leads to the term “Shape of Motion”.

### Abstract

Monocular dynamic reconstruction is a challenging and long-standing vision problem due to the highly ill-posed nature of the task. Existing approaches depend on templates, are effective only in quasi-static scenes, or fail to model 3D motion explicitly. We introduce a method for reconstructing generic dynamic scenes, featuring explicit, persistent 3D motion trajectories in the world coordinate frame, from casually captured monocular videos. We tackle the problem with two key insights: First, we exploit the low-dimensional structure of 3D motion by representing scene motion with a compact set of SE(3) motion bases. Each point’s motion is expressed as a linear combination of these bases, facilitating soft decomposition of the scene into multiple rigidly-moving groups. Second, we take advantage of offthe-shelf data-driven priors such as monocular depth maps and long-range 2D tracks, and devise a method to effectively consolidate these noisy supervisory signals, resulting in a globally consistent representation of the dynamic scene. Experiments show that our method achieves state-of-the-art performance for both long-range 3D/2D motion estimation and novel view synthesis on dynamic scenes. Project Page: https://shape-of-motion.github.io/

∗Equal contribution

### 1. Introduction

Reconstructing the persistent geometry and their 3D motion across a video is crucial for understanding and interacting with the underlying physical world. While recent years have seen impressive progress in modeling static 3D scenes [46, 71], recovering the geometry and motion of complex dynamic 3D scenes, especially from a single video, remains an open challenge. A number of prior dynamic reconstruction and novel view synthesis approaches have attempted to tackle this problem. However, most methods rely on synchronized multi-view videos [9, 23, 23, 69, 111] or additional LIDAR/depth sensors [29, 65, 80, 101, 109]. Recent monocular approaches can operate on regular dynamic videos, but they typically model 3D scene motion as shortrange scene flow between consecutive times [24, 59, 60] or deformation fields that map between canonical and view space [76, 77, 122], failing to capture 3D motion trajectories persistent over a video.

The longstanding challenge for general in-the-wild videos lies in the poorly constrained nature of the 4D reconstruction problem. In this work, we tackle this challenge with two key insights. The first is that, while the image space dynamics can be complex and discontinuous, the underlying 3D motion is a composition of continuous simple rigid motions. Our second insight is that data-driven priors provide complementary, though noisy cues, that aggregate

well into a globally coherent representation of the 3D scene geometry and motion.

Motivated by these two insights, we represent the dynamic scene as a set of persistent 3D Gaussians, and represent their motion across the video in terms of a compact set of shared SE(3) motion bases. Unlike traditional scene flow, which computes 3D correspondences between consecutive frames, our representation recovers a persistent 3D trajectory over the whole video, enabling long-range 3D tracking across the entire video. As the 3D trajectories produced by our method capture the geometric patterns that trace each point’s movement through 3D space and time as shown in Figure 1, we refer to our approach as “Shape of Motion”. We show how to fit our explicit scene representation to a general video in-the-wild, by fusing together complementary cues from two main sources: monocular depth estimates per-frame, and 2D track estimates across frames. We conduct extensive evaluations on both synthetic and realworld dynamic video datasets, and show that our proposed approach significantly outperforms prior methods in both long-range 2D/3D tracking and novel view synthesis.

In summary, our key contributions are: (1) A 4D scene representation enabling both real-time novel view synthesis and globally consistent 3D tracking for any point at any time. (2) An optimization framework that fuses learned motion and geometry priors into a unified 4D representation from a single, casually captured monocular video.

### 2. Related Work

Correspondences and Tracking. While monocular 3D long range tracking remains largely unexplored, numerous approaches track in 2D image space, typically using optical flow for point correspondences. This involves estimating dense motion fields between image pairs [4, 7, 8, 18, 33, 36, 36, 40, 42–44, 68, 82, 90, 96, 97, 115]. Sparse keypoint matching methods can enable long trajectory generation [2, 14, 63, 66, 84], but these methods are primarily intended for sparse 3D reconstruction. Longrange 2D trajectory estimation for arbitrary points has been explored in earlier works, which relied on hand-crafted priors to generate motion trajectories [3, 83, 86, 87, 91, 102]. Recently, there has been a resurgence of interest in this problem, with several works showcasing impressive long-range 2D tracking results on challenging, in-the-wild videos. These approaches employ either test-time optimization where models consolidate noisy short-range motion estimates into long-term correspondences [72, 93, 104], or data-driven strategies [16, 17, 30, 45], where neural networks learn long-term correspondence estimates from synthetic data [15, 129]. While these methods effectively track any 2D point throughout a video, they lack the knowledge of underlying 3D scene geometry and motions.

To model 3D scene motion, recent work in monocu-

lar settings has explored self-supervised learning and testtime optimization [38, 59, 60, 117–119]. More recent approaches estimate 3D trajectories in general scenes in a feedforward manner [52, 74, 113], but the predicted motion remains in the frame space, entangling object and camera motion. In contrast, our method recovers persistent, longrange 3D trajectories in the world coordinate space.

Dynamic Reconstruction and View Synthesis. Our work also relates to dynamic 3D scene reconstruction and novel view synthesis. In non-rigid reconstruction, early methods often required RGB-D sensors [5, 19, 41, 73, 130] or strong hand-crafted priors [54, 81, 85]. A family of works [6, 13, 75] apply low-rank assumptions to the underlying motion model. Recent work has demonstrated progress toward general dynamic reconstruction either by integrating monocular depth priors [51, 58, 62, 70, 127, 128], or directly learning to perform dynamic reconstruction [67, 105, 106, 126] from data. However, these methods do not perform long-range 3D tracking, and do not focus on novel view synthesis.

Neural Radiance Fields (NeRF) [71] and Gaussian Splatting [47] have shown strong performance in novel view synthesis. For dynamic scenes, many methods [1, 9, 10, 23, 23, 57, 61, 92, 95, 103] still require simultaneous multi-view video observations or predefined templates [39, 56, 110]. Template-free monocular approaches model dynamic scenes with different types of representations such as video depth maps [125], time-dependent NeRFs [20, 59, 60, 76, 77, 79, 100, 112], and dynamic 3D Gaussians [21, 111, 122, 123]. Although significant progress has been made, as DyCheck [25] pointed out, many approaches focus on scenarios with camera teleportation [53, 122] or quasi-static scenes [76, 77], which are effectively multiview and do not represent real-world monocular videos. In this work, we focus on casually-captured monocular videos, a more practical and challenging setup. Recent and concurrent efforts [12, 55, 64, 94, 107] have also investigated similar settings, using strong off-the-shelf data-driven priors.

### 3. Method

Our method takes as input a sequence of T video frames {It ∈ RH×W×3} of a dynamic scene, the camera intrinsics Kt ∈ R3×3, and world-to-camera extrinsics Et ∈ SE(3) of each input frame It. From these inputs, we aim to recover the geometry of the entire dynamic scene and the full-length 3D motion trajectory of every point in the scene. Unlike most prior dynamic NeRFs methods [25, 58, 60, 100, 112] which render the scene contents through volumetric ray casting and represent the motion implicitly at fixed 3D locations, we model the dense scene elements as a set of canonical 3D Gaussians, that translate and rotate over entire video as persistent motion trajectories. We adopt explicit pointbased representation because it simultaneously allows for both (1) high-fidelity rendering in real-time and (2) full-

[Figure 5]

Figure 2. System Overview. Given a single RGB video sequence with known camera poses, monocular depth maps and 2D tracks computed from off-the-shelf models [16, 121] as input, we optimize a dynamic scene representation as a set of persistent 3D Gaussians that translate and rotate over time. To capture the low-dimensional nature of scene motion, we model the motion with a set of compact SE(3) motion bases shared across all scene elements. Each 3D Gaussian’s motion is represented as a linear combination of these motion bases, weighted by motion coefficients specific to each Gaussian. We supervise our scene representation (canonical 3D Gaussians, per-Gaussian motion coefficients, and global motion bases) by comparing the rendered outputs (RGB, depths and 2D tracks) with the corresponding input signals. This results in a globally coherent dynamic 3D scene representation in the world coordinate with long-range 3D trajectories.

length 3D tracking of any surface point from any input time.

Optimizing an explicit representation of dynamic 3D Gaussians from a single video is severely ill-posed — at each point in time, the moving subjects in the scene are observed from only a single viewpoint. In order to overcome this ambiguity, we make two insights: First, while the projected 2D dynamics might be complex in the video, the underlying 3D motion in the scene is low-dimensional, and composed of simpler units of rigid motion. Second, powerful data-driven priors, namely monocular depth estimates and long-range 2D tracks, provide complementary but noisy signals of the underlying 3D scene. We propose a system that fuses these noisy estimates together into a globally coherent representation of both the scene geometry and motion. We show a schematic of our pipeline in Figure 2.

#### 3.1. Preliminaries: 3D Gaussian Splatting

We represent the appearance and geometry of a dynamic scene with a global set of 3D Gaussians, an explicit and expressive differentiable scene representation [46] for efficient optimization and rendering. We define parameters of each 3D Gaussian in the canonical frame t0 as ⃗g0 ≡ (µ0,R0,s,o,c), where µ0 ∈ R3, R0 ∈ SO(3) are the 3D mean and orientation in the canonical frame, and s ∈ R3 the scale, o ∈ R the opacity, and c ∈ R3 the color. Σ0 is the canonical frame covariance matrix. These properties are persistent and shared across time. To render a set of 3D

Gaussians from a camera with world-to-camera extrinsics E and intrinsics K, the projections of the 3D Gaussians in the image plane are obtained by 2D Gaussians parameterized as µ′0 ∈ R2 and Σ′0 ∈ R2×2 via affine approximation

µ′0(K,E) = Π(KEµ0) ∈ R2, Σ′0(K,E) = JKEΣ0J⊤KE ∈ R2×2, (1)

where Π is perspective projection, and JKE is the Jacobian of perspective projection with K and E at the point µ0. The

- 2D Gaussians can then be efficiently rasterized into RGB image and depth map via alpha compositing as

ˆI(p) =

i∈H(p)

Tiαici, Dˆ (p) =

i∈H(p)

Tiαidi, (2)

where αi = oi · exp − 12(p − µ′0)T Σ′0 (p − µ′0) , and Ti = ij−=11(1 − αj). H(p) is the set of Gaussians that intersect the ray shoot from the pixel p. This process is fully differentiable, and enables direct optimization of the

- 3D Gaussian parameters. 3.2. Dynamic Scene Representation

Scene Motion Parameterization. To model a dynamic 3D scene, we keep track of N canonical 3D Gaussians and vary their positions and orientations over time with a

per frame rigid transformation. In particular, for a moving 3D Gaussian at time t, its pose parameters (µt,Rt) are rigidly transformed from the canonical frame t0 to t via T0→t = R0→t t0→t ∈ SE(3):

µt = R0→tµ0 + t0→t, Rt = R0→tR0. (3)

Rather than modeling the 3D motion trajectories independently for each Gaussian, we define a set of B ≪ N learn-

able basis trajectories {T(0b→) t}Bb=1 that are globally shared across all Gaussians [53]. The transformation T0→t at each time t is then computed by a weighted combination of this global set of basis trajectories through per-point basis coefficients w(b) via

T0→t =

B

w(b) T(0b→) t, (4)

b=0

where ∥w(b)∥ = 1. In our implementation, we parameterize T(0b→) t as 6D rotation [32] and translation, and perform the weighted combination separately on each with the same weight w(b). During optimization, we jointly learn the set of global motion bases and motion coefficients of each 3D Gaussian. These compact motion bases explicitly regularize the trajectories to be low-dimensional, encouraging the 3D Gaussians that move similarly to each other to be represented by similar motion coefficients.

Rasterizing 3D Trajectories. Given this representation, we now describe how we obtain pixelwise 3D motion trajectory at any query frame It. we take a similar approach to Wang et al. [104] and rasterize the motion trajectories of 3D Gaussians into query frame It. Namely, for a query camera at time t with intrinsics Kt and extrinsics Et, we perform rasterization to obtain a map wXˆ t→t′ ∈ RH×W×3 that contains the expected 3D world coordinates of the surface points corresponding to each pixel at target time t′

wXˆ t→t′(p) =

Tiαiµi,t′, (5)

i∈H(p)

where H(p) is the set of Gaussians that intersect the pixel p at query time t. The 2D correspondence location at time t′ for a given pixel p, Uˆ t→t′(p), and the corresponding depth value at time t′, Dˆ t→t′(p) can then be written as

Uˆ t→t′(p) = Π Kt′cXˆ t→t′(p) , Dˆ t→t′(p) = cXˆ t→t′(p)

(6) where cXˆ t→t′(p) = Et′wXˆ t→t′(p), Π is a perspective projection operation, and (·)[3] is the third element of a vector.

#### 3.3. Optimization

We prepare the following estimates using off-the-shelf methods in our optimization: 1) camera parameters estimated by MegaSaM [62], 2) masks for the moving objects

for each frame {Mt}, which can be easily obtained using Track-Anything [50, 120] with a few user clicks, 3) monocular depth maps {Dt} computed using state-of-the-art relative depth estimation method such as Depth Anything [121] and 4) long-range 2D tracks {Ut→t′} for foreground pixels from state-of-the-art point tracking method TAPIR [16]. We align the relative depth maps with the metric depth maps by computing a per-frame global scale and shift and use them for our optimization, as we found relative depth maps tend to contain finer details. We treat the lifted 2D tracks unprojected with the aligned depth maps as noisy initial 3D track observations {Xt} for the moving objects. For the static part of the scene, we model them using standard static 3D Gaussians and initialize their 3D locations by unprojecting them into the 3D space using the aligned depth maps. The static and dynamic Gaussians are jointly optimized and rasterized together to form an image. We describe the optimization process below.

Initialization. We first select the canonical frame t0 to be the frame in which the most 3D tracks are visible, and ini-

tialize the Gaussian means in the canonical frame µ0 as N randomly sampled 3D tracks locations from this set of initial observations. We then perform k-means clustering on the vectorized velocities of the noisy 3D tracks from ev-

ery frame {Xt}, and initialize the motion bases {T(0b→) t}Bb=1 from these B clusters of tracks. Specifically, for the set

of trajectories {Xt}b belonging to cluster b, we initialize the basis transform T(0b→) τ using weighted Procrustes alignment between the point sets {X0}b and {Xτ}b for all τ = 0,...,T, where the weights are computed using uncertainty and visibility scores from TAPIR predictions. We initialize w(b) of each Gaussian to decay exponentially with its distance from the center of cluster b in the canonical frame. We then optimize µ0, w(b), and set of basis functions {T(0b→) t}Bb=1 to fit the observed 3D tracks with an ℓ1-loss under temporal smoothness constraints.

Training. We supervise the dynamic Gaussians with two sets of losses. The first set comprise our reconstruction loss to match the per-frame pixelwise color, depth, and masks inputs. The second set enforce consistency of correspondences across time. Specifically, during each training step, we render the image ˆIt, depth Dˆ t, and mask Mˆ t from their corresponding training cameras (Kt,Et) according to Equation 2. We supervise these predictions with a reconstruction loss applied independently per-frame

,

[3]

Lrecon = ∥ˆI−I∥1+λdepth∥Dˆ −D∥1+λmask∥Mˆ −M∥1. (7)

The second set of losses supervises the motion of the Gaussians between frames. Specifically, we additionally render the 2D tracks uˆt→t′ and reprojected depths Dˆ t→t′, for a pair of randomly sampled query time t and target time

t′. We supervise these rendered correspondences with the

- long-range 2D track estimates via

Ltrack-2d = ∥Ut→t′ − Uˆ t→t′∥1, (8) Ltrack-depth = ∥dˆt→t′ − Dˆ (Ut→t′)∥1. (9)

Finally, we leverage physics motion prior by enforcing a distance-preserving loss between randomly sampled dynamic Gaussians and their k-nearest neighbors. Let Xˆ t and Xˆ t′ denote the location of a Gaussian at time t and t′, and Ck(Xˆ t) denote the set of k-nearest neighbors of Xˆ t, we define

2 2

Lrigidity = dist(Xˆ t,Ck(Xˆ t)) − dist(Xˆ t′,Ck(Xˆ t′))

,

(10) where dist(·,·) measures Euclidean distance.

Implementation Details. For in-the-wild videos, MegaSaM [62] is used to estimate initial camera poses, which are further refined as learnable parameters during our optimization, updated via the camera gradient function implemented in the gsplat [124] CUDA kernel. For evaluation on public benchmarks, we use the camera annotations provided with each dataset (e.g., from COLMAP [89] or simulation).

We optimize our model using Adam Optimizer [49]. We perform 1000 iterations of optimization for the initialization phase and 500 epochs for training phase, respectively. The number of SE(3) bases B is set to 10 for all of our experiments. We initialize 40k dynamic Gaussians for the dynamic part and 100k static Gaussians for the static part of the scene, respectively. We perform the same adaptive Gaussian controls for dynamic and static Gaussians as per 3D-GS [47]. Training on a sequence of 300 frames of 960 × 720 resolution takes about 2 hours to finish on an A100 GPU. Our rendering FPS is around 140 fps. We use the same set of hyper-parameters for all experiments.

### 4. Experiment

We evaluate our performance quantitatively and qualitatively on a broad range of tasks: long range 3D point tracking, long-range 2D point tracking, and novel view synthesis. We focus our evaluation on datasets that exhibit substantial scene motion. In particular, the iPhone dataset [25] features casual captures of real-world scenes that closely match our target scenarios. It provides comprehensive annotations, including simultaneous validation views, lidar depth, sparse 2D point correspondences across the entire video, and can be used to evaluate our performance on all three tasks. Given the challenge of obtaining precise 3D track annotations for real data, we also evaluate performance using the scenes from the synthetic MOVi-F Kubric dataset [28].

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

BACKPACK

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

P-APERWINDMILL

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

SPIN

Ours TAPIR + DA∗ D-3DGS HyperNeRF

Figure 3. 3D Track visualization on iPhone dataset. We render novel views for each method and overlay their predicted 3D tracks on top of the rendered images. For clarity, we only display a segment of the trails spanning 50 frames for a specific set of grid query points. However, it is important to note that our method can generate dense, full-length 3D tracks. ∗Note that TAPIR + DA cannot produce novel view rendering, and we overlay their tracking results with our rendering to make it easier to interpret.

#### 4.1. Task Specification

Long-Range 3D Tracking. Our primary task is estimating 3D scene motion for any pixel over the entire video. For this purpose, we extend the metrics for scene flow evaluation introduced in RAFT-3D [99] to evaluate long-range 3D tracking. Specifically, we report the 3D end-point-error (EPE), which measures the Euclidean distance between the ground truth 3D tracks and predicted tracks at each target time step. In addition, we report the percentage of points that fall within a given threshold of the ground truth 3D location δ3.05D = 5cm and δ3.10D = 10cm in metric scale.

Long-Range 2D Tracking. Our 3D motion representation can be easily projected onto image plane to get long-range 2D tracks. Thus we also evaluate 2D tracking performance in terms of both position and occlusion accuracy. Following the metrics introduced in the TAP-Vid benchmark [15], we report the Average Jaccard (AJ), average position accuracy (<δavg), and Occlusion Accuracy (OA).

Novel View Synthesis. We measure our method’s novel view synthesis quality as a comprehensive assessment for geometry, appearance, and motion. We evaluate on the iPhone dataset [25] which provides validation views and report co-visibility masked image metrics [25]: mPSNR, mSSIM [108] and mLPIPS [26, 37]. Additionally, we evaluate on NVIDIA dataset for more comparison.

3D Tracking 2D Tracking View Synthesis

Method

EPE ↓ δ3.05D ↑ δ3.10D ↑ AJ ↑ <δavg ↑ OA ↑ PSNR ↑ SSIM ↑ LPIPS ↓ T-NeRF [25] - - - - - - 15.60 0.55 0.55 HyperNeRF [77] 0.182 28.4 45.8 10.1 19.3 52.0 15.99 0.59 0.51 DynIBaR [60] 0.252 11.4 24.6 5.4 8.7 37.7 13.41 0.48 0.55 Deformable-3D-GS [122] 0.151 33.4 55.3 14.0 20.9 63.9 11.92 0.49 0.66 DynMF [53] 0.188 22.9 53.8 5.5 9.5 60.5 16.54 0.59 0.49 CoTracker [45]+DA [121] 0.202 34.3 57.9 24.1 33.9 73.0 - - TAPIR [16]+DA [121] 0.114 38.1 63.2 27.8 41.5 67.4 - - DELTA (world) [74] 0.159 32.5 55.3 24.7 34.1 68.9 - - SpatialTracker (world) [113] 0.125 37.7 63.9 24.9 36.9 73.5 - - Ours 0.082 43.0 73.3 34.4 47.0 86.6 16.72 0.63 0.45 Ours + 2DGS[34] 0.097 47.3 71.3 35.8 47.0 87.3 16.75 0.65 0.40

Table 1. Evaluation on iPhone dataset. Our method achieves SOTA performance all tasks of 3D point tracking, 2D point tracking, and novel view synthesis. The baselines that perform best on 2D and 3D tracking (TAPIR [16]+DA [121], CoTracker [45]+DA [121], DELTA [74], SpatialTracker [113]) are unable to synthesize novel views of the scene, while the methods that perform best in novel view synthesis struggle with or fail to produce 2D and 3D tracks. Our method achieves a significant boost in all three tasks above baselines. We include training details about “Ours + 2DGS [34]” in the supplement.

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

P-APERWINDMILL

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

SPIN

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

TEDDY

Train View GT Ours HyperNeRF TNeRF DynIBaR D-3DGS

- Figure 4. Qualitative comparison of novel view synthesis on iPhone dataset. The leftmost image in each row shows the training view at the same time step as the validation view. The regions highlighted in green indicate areas excluded from evaluation due to the lack of co-visibility between training and validation views. Please see the supplemental video for more qualitative results and comparisons.

#### 4.2. Baselines

Our method represents dynamic 3D scene comprehensively with explicit long-range 3D scene motion estimation, which also allows for novel view synthesis. While no existing method achieves the exact same goals as ours, there are methods that focus on sub-tasks related to our problem, such as dynamic novel view synthesis, 2D tracking, or monocular depth estimation. We therefore adapt existing

methods as our baselines which are introduced below.

While dynamic novel view synthesis approaches focus primarily on the photometric reconstruction quality and do not explicitly output 3D point tracks, we can adapt some of their representations to derive 3D point tracks for our evaluation. For HyperNeRF [77], we compose the learned inverse mapping (from view space to canonical space) and a forward mapping solved via root-finding [11, 25] to produce 3D tracks at query points. DynIBaR [60] produces

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

Input PCA Coef.

Pred. Depth

Input PCA Coef.

Pred. Depth

- Figure 5. Visualization of motion coefficients after PCA and predicted depth maps on iPhone dataset. The motion coefficients encode information regarding rigid moving components. For instance, our motion coefficient PCA produces constant color for the block in the second example which exhibits rigid motion.

short-range view-to-view scene flow, which we chain into

- long-range 3D tracks for our evaluation. Deformable-3DGS (D-3DGS) [122] represents dynamic scenes using 3D Gaussians [46] in the canonical space and a deformation MLP network that deforms the canonical 3D Gaussians into each view, which naturally allows 3D motion computation. T-NeRF [25] models dynamic scenes using time as MLP input in addition to 3D locations, which does not provide a method for extracting 3D motion, and hence they are not considered for 2D/3D tracking evaluation. Finally, we implemented our own version of DynMF [53] with neural network motion bases. Implementation details are provided in the supplemental material. We evaluate the 3D tracks of these methods only on the iPhone dataset, because we found none can handle the Kubric scene motion.

Alongside dynamic view synthesis baselines, we include 3D tracking baselines. Specifically, we take state-of-the-art long-range 2D tracks from TAPIR [16] and CoTracker [45] and lift them into 3D scene motion using monocular depth maps produced by Depth Anything (DA) [121]. We compute the correct scale and shift for each relative depth map from Depth Anything to align them with the scene. The two resulting baselines are called “CoTracker [45] + DA [121]” and “TAPIR [16] + DA [121]”. Note that these baselines can only produce 3D tracks for visible regions, as the depth values for occluded points are unknown from such a 2.5D representation. In contrast, our global representation allows for modeling 3D motion through occlusions. In addition, we include frame-space 3D tracking methods such as DELTA [74] and SpatialTracker [113], and transform their trajectories into world coordinates using annotated camera poses for evaluation.

#### 4.3. Evaluation on iPhone Dataset

iPhone dataset [25] contains 14 sequences of 200-500 frames featuring various types of challenging real world scene motion. All sequences are recorded using a handheld moving camera in a casual manner, with 7 of them additionally featuring two synchronized static cameras with a large baseline for novel view synthesis evaluation. For 3D tracking evaluation, we generate the groundtruth 3D tracks

by lifting the 2D keypoint annotations into 3D using lidar depth, and masking out points that are occluded or with invalid lidar depth values. All experiments are conducted with the original instead of half resolution as in [25] given that our method can handle high-res video input. We also discard scenes SPACE-OUT and WHEEL due to camera and lidar error. We find that the original camera annotations from ARKit is not accurate enough, and refine them using global bundle adjustment from COLMAP [88]. The refined poses are used for all methods.

We report the quantitative comparison in Tab. 1 which shows that our method outperforms all baselines on all tasks by a substantial margin. For tracking, we observe significant improvements over naive baselines that combine 2D tracks with depth maps, especially “TAPIR+DA”, which also serves as input to our system, highlighting the effectiveness of our consolidation process. The improvement stems from three factors: 1) a low-dimensional SE(3) motion representation, which allows inferring occluded motion via nearby visible regions; 2) 3D regularizations, such as low acceleration constraints that promote coherent trajectories; 3) the consolidation of all pairwise tracks into a single 4D representation with persistent 3D geometry, which serves as a structural prior to correct noisy 2D tracks. Our method also achieves the best novel view synthesis quality across dynamic NeRF and 3D-GS-based baselines.

Fig. 3 shows qualitative comparison of the 3D tracking results. We render the novel views and plot the predicted 3D tracks of the given query points onto the novel views. Since “TAPIR + DA” cannot perform novel view synthesis, we overlay their track predictions onto our renderings to aid interpretation. D-3DGS [122] and HyperNeRF [77] fail to capture the significant scene motion in PAPER-WINDMILL and SPIN, resulting in structure degradation and blurry rendering. “TAPIR + DA” can track large motions, but their 3D tracks tend to be noisy and erroneous. In contrast, our method not only generates high-quality novel views but also the most smooth and accurate 3D tracks. Fig. 4 provides additional novel view synthesis comparison on the validation views. In Fig. 5, we visualize the rendering of the first three components of PCA decomposition for the motion coefficients, which correlates well with the rigid groups of the moving objects.

#### 4.4. Evaluation on Kubric Dataset

Method EPE↓ δ3.05D ↑ δ3.10D ↑

CoTracker [45]+DA [121] 0.19 34.4 56.5 TAPIR [16]+DA [121] 0.20 34.0 56.2 Ours 0.16 39.8 62.2

Table 2. 3D Tracking evaluation on Kubric dataset.

The Kubric MOVi-F dataset contains short 24-frame videos

of scenes of 10-20 objects, rendered with linear camera movement and motion blur. Multiple rigid objects are tossed onto the scene, at a speed that far exceeds the speed of the moving camera, making it a similar capture scenario to in-the-wild capture scenarios. It provides dense comprehensive annotations, including ground truth depth maps, camera parameters, segmentation masks, and point correspondences that are dense across time. We use 30 scenes from the MOVi-F validation set to evaluate the accuracy of our long-range 3D point tracks for all points in time.

We demonstrate our method on the synthetic Kubric MOVi-F dataset in long-range 3D point tracking across time. Of the above baselines, only “CoTracker [45]+DA [121]” and “TAPIR [45]+DA [121]” yield 3D tracks for these scenes. For all baselines, we provide the ground truth camera intrinsics and extrinsics, and monocular depth estimates that have been aligned to the ground truth depth map. We obtain point tracks for all non-background pixels for each method.

We report our quantitative 3D point tracking metrics in Table 2, and find that across all metrics, our method outperforms the baselines. Please see the supplement for figures of the optimized motion coefficients, which coherently group moving objects.

#### 4.5. Evaluation on NVIDIA Dataset

We conduct experiments on seven scenes from the NVIDIA dataset [125], following the evaluation protocol of Dynamic Gaussian Marbles [94], which evaluates the task from a single static camera instead of interleaving different camera as discussed in [25]. We adopt the same image resolution, camera parameters and depths maps as input. Please see the supplement for more details. For each scene, we compute covisibility masks between each evaluation view and the training view at each time step to exclude unobservable regions during evaluation. Results are reported in Tab. 3.

Balloon1 Balloon2 Jumping Truck

Method

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

DGM [94] 23.83 0.81 0.066 23.62 0.81 0.088 20.02 0.73 0.134 27.31 0.86 0.055 Ours 23.90 0.79 0.058 23.91 0.83 0.074 20.08 0.69 0.151 27.65 0.88 0.048

Playground Umbrella Skating Mean

Method

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

DGM [94] 16.68 0.60 0.177 24.99 0.70 0.083 27.79 0.90 0.052 23.46 0.77 0.093 Ours 16.87 0.55 0.168 23.27 0.57 0.167 27.91 0.92 0.047 23.37 0.75 0.102

Table 3. Evaluation on NVIDIA dataset. Our method is comparable with Dynamic Gaussian Marbles (DGM) [94].

#### 4.6. Ablation studies

We ablate various components of our method on the iPhone dataset in 3D tracking in Tab. 4. We first validate our choices of motion representation, namely the SE(3) motion bases parameterization, with three ablations: 1) “PerGaussian Transl.”: replacing our motion representation with naive per-Gaussian translational motion trajectories,

Methods SE(3) Motion Basis 2D tracks Initialization EPE↓ δ3.05D ↑ δ3.10D ↑

Ours (Full) ✓ ✓ ✓ ✓ 0.082 43.0 73.3 Transl. Bases ✓ ✓ ✓ 0.093 42.3 69.9 Per-Gaussian SE(3) ✓ ✓ ✓ 0.083 43.6 70.2 Per-Gaussian Transl. ✓ ✓ 0.087 41.2 69.2 No SE(3) Init. ✓ ✓ ✓ 0.111 39.3 65.7 No 2D Tracks ✓ ✓ 0.141 30.4 57.8

Table 4. Ablation Studies on iPhone dataset.

2) “Per-Gaussian SE(3)”: replacing our motion representation with naive per-Gaussian SE(3) motion trajectories, and 2) “Transl. Bases”: keeping the motion bases representation but only using translational bases instead of SE(3). We find that SE(3) bases significantly improve 3D tracking over both translational bases and per-Gaussian translational motion. It also outperforms per-Gaussian SE(3) in overall accuracy and offers noticeably better visual quality, as the latter exhibits significantly more popping and jitter artifacts.

Next, we ablate our training strategies including initialization and supervision signal. We conduct an ablation of “No SE(3) Init.”, where instead of performing our initial SE(3) fitting stage, we initialize the translational part of the motion bases with randomly selected noisy 3D tracks formed by directly lifting input 2D tracks using depth maps into 3D, and the rotation part as identity. We find that skipping this initialization noticeably hurts performance. Lastly, we remove the 2D track supervision entirely (“No

- 2D Tracks”) and find it to lead to significant drop in performance, which verifies the importance of the 2D track supervision for 3D tracking.

5. Discussion

Limitations. Like most prior monocular view synthesis methods, it still requires per-scene test-time optimization, hindering streamable applications. Recent feed-forward methods for joint reconstruction and tracking [22, 114, 116] offer a promising alternative, showing encouraging progress toward overcoming this limitation. Our method relies on off-the-shelf predictions for camera poses, geometry, and motion, which may degrade in textureless regions or under large motions. However, it also naturally benefits from advances in these components. Finally, it also requires user input to mask moving objects. Recent advances in moving object segmentation [27, 35] offer promising directions beyond our current manual solution and could be integrated into our pipeline.

Conclusion. We introduce a method for joint long-range

- 3D tracking and novel view synthesis from monocular video, representing dynamic elements via global 3D Gaussians with time-varying translations and rotations. Motion trajectories are regularized using low-dimensional rigid motion bases. By consolidating noisy observations into globally consistent scene estimates, our approach outperforms state-of-the-art methods in 2D/3D tracking and novel view synthesis across synthetic and real benchmarks.

Acknowledgement. We thank Ruilong Li, Noah Snavely, Brent Yi and Aleksander Holynski for helpful discussion. We are in memory of our beloved cat Sriracha, who will always be missed and loved. This project is supported in part by DARPA No. HR001123C0021. and IARPA DOI/IBC No. 140D0423C0035. The views and conclusions contained herein are those of the authors and do not represent the official policies or endorsements of these institutions.

### References

- [1] Aayush Bansal, Minh Vo, Yaser Sheikh, Deva Ramanan, and Srinivasa Narasimhan. 4d visualization of dynamic events from unconstrained multi-view videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 2
- [2] Herbert Bay, Tinne Tuytelaars, and Luc Van Gool. Surf: Speeded up robust features. In European Conference on Computer Vision, 2006. 2
- [3] Stanley T Birchfield and Shrinivas J Pundlik. Joint tracking of features and edges. In 2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2008. 2
- [4] Michael J Black and Padmanabhan Anandan. A framework for the robust estimation of optical flow. In 1993 (4th) International Conference on Computer Vision, 1993. 2
- [5] Aljaz Bozic, Michael Zollh¨ofer, Christian Theobalt, and Matthias Nießner. Deepdeform: Learning non-rigid rgb-d reconstruction with semi-supervised data. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2
- [6] C. Bregler, A. Hertzmann, and H. Biermann. Recovering non-rigid 3d shape from image streams. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition. CVPR 2000 (Cat. No.PR00662), 2000. 2
- [7] Thomas Brox, Andr´es Bruhn, Nils Papenberg, and Joachim Weickert. High accuracy optical flow estimation based on a theory for warping. In European Conference on Computer Vision, 2004. 2
- [8] Thomas Brox, Christoph Bregler, and Jitendra Malik. Large displacement optical flow. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009. 2
- [9] Michael Broxton, John Flynn, Ryan Overbeck, Daniel Erickson, Peter Hedman, Matthew Duvall, Jason Dourgarian, Jay Busch, Matt Whalen, and Paul Debevec. Immersive light field video with a layered mesh representation. ACM Transactions on Graphics (TOG), 39(4), 2020. 1, 2
- [10] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2023. 2

- [11] Xu Chen, Yufeng Zheng, Michael J Black, Otmar Hilliges, and Andreas Geiger. Snarf: Differentiable forward skinning for animating non-rigid neural implicit shapes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021. 6
- [12] Wen-Hsuan Chu, Lei Ke, and Katerina Fragkiadaki.

- Dreamscene4d: Dynamic multi-object scene generation from monocular videos. NeurIPS, 2024. 2
- [13] Yuchao Dai, Hongdong Li, and Mingyi He. A simple priorfree method for non-rigid structure-from-motion factorization. Int. J. Comput. Vision, 107(2), 2014. 2
- [14] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW),

2017. 2

- [15] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adri`a Recasens, Lucas Smaira, Yusuf Aytar, Jo˜ao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 2022. 2, 5
- [16] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 2, 3, 4, 6, 7, 15
- [17] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Joao Carreira, et al. Bootstap: Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference on Computer Vision, 2024. 2
- [18] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip H¨ausser, Caner Hazirbas, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learning optical flow with convolutional networks. 2015 IEEE International Conference on Computer Vision (ICCV), 2015. 2
- [19] Mingsong Dou, S. Khamis, Yu.G. Degtyarev, Philip L. Davidson, S. Fanello, Adarsh Kowdle, Sergio Orts, Christoph Rhemann, David Kim, Jonathan Taylor, Pushmeet Kohli, Vladimir Tankovich, and Shahram Izadi. Fusion4d. ACM Transactions on Graphics (TOG), 35, 2016. 2
- [20] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, 2021. 2
- [21] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers,

2024. 2

- [22] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. arXiv preprint arXiv:2504.13152, 2025. 8
- [23] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. Kplanes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 1, 2

- [24] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In Proceedings of the IEEE International Conference on Computer Vision, 2021. 1
- [25] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Dynamic novel-view synthesis: A reality check. In NeurIPS, 2022. 2, 5, 6, 7, 8, 16
- [26] Leon A Gatys, Alexander S Ecker, Matthias Bethge, Aaron Hertzmann, and Eli Shechtman. Controlling perceptual factors in neural style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2017. 5
- [27] Lily Goli, Sara Sabour, Mark Matthews, Marcus Brubaker, Dmitry Lagun, Alec Jacobson, David J Fleet, Saurabh Saxena, and Andrea Tagliasacchi. Romo: Robust motion segmentation improves structure from motion. arXiv preprint arXiv:2411.18650, 2024. 8
- [28] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 5, 16
- [29] Xiuye Gu, Yijie Wang, Chongruo Wu, Yong Jae Lee, and Panqu Wang. Hplflownet: Hierarchical permutohedral lattice flownet for scene flow estimation on large-scale point clouds. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 1
- [30] Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In European Conference on Computer Vision, 2022. 2
- [31] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction, 2024. 16
- [32] Thorsten Hempel, Ahmed A Abdelrahman, and Ayoub AlHamadi. 6d rotation representation for unconstrained head pose estimation. In 2022 IEEE International Conference on Image Processing (ICIP), 2022. 4
- [33] Berthold K. P. Horn and Brian G. Schunck. Determining optical flow. In Other Conferences, 1981. 2
- [34] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024. 6, 16
- [35] Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo Kanazawa, and Qianqian Wang. Segment any motion in videos. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR),

2025. 8

- [36] Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer: A transformer architecture for optical flow. In European Conference on Computer Vision. Springer, 2022. 2

- [37] Minyoung Huh, Richard Zhang, Jun-Yan Zhu, Sylvain Paris, and Aaron Hertzmann. Transforming and projecting images into class-conditional generative networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16,

2020. 5

- [38] Junhwa Hur and Stefan Roth. Self-supervised monocular scene flow estimation. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
- [39] Mustafa I¸sık, Martin R¨unz, Markos Georgopoulos, Taras Khakhulin, Jonathan Starck, Lourdes Agapito, and Matthias Nießner. Humanrf: High-fidelity neural radiance fields for humans in motion. ACM Transactions on Graphics (TOG), 2023. 2
- [40] Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox. Flownet 2.0: Evolution of optical flow estimation with deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2017. 2
- [41] Matthias Innmann, Michael Zollh¨ofer, Matthias Nießner, Christian Theobalt, and Marc Stamminger. Volumedeform: Real-time volumetric non-rigid reconstruction. In European Conference on Computer Vision, 2016. 2
- [42] Joel Janai, Fatma Guney, Anurag Ranjan, Michael Black, and Andreas Geiger. Unsupervised learning of multi-frame optical flow with occlusions. In Proceedings of the European conference on computer vision (ECCV), 2018. 2
- [43] Shihao Jiang, Dylan Campbell, Yao Lu, Hongdong Li, and Richard Hartley. Learning to estimate hidden motions with global motion aggregation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.
- [44] Shihao Jiang, Yao Lu, Hongdong Li, and Richard Hartley. Learning optical flow from a few matches. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021. 2
- [45] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European conference on computer vision, 2024. 2, 6, 7, 8
- [46] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics,

- 2023. 1, 3, 7

[47] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics,

- 2023. 2, 5, 15

- [48] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 15

- [49] Diederik P Kingma, J Adam Ba, and J Adam. A method for stochastic optimization. arxiv 2014. arXiv preprint arXiv:1412.6980, 2020. 5
- [50] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar,

- and Ross B. Girshick. Segment anything. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 4
- [51] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 2
- [52] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Joao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. Advances in Neural Information Processing Systems, 2024. 2
- [53] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. ECCV, 2024. 2, 4, 6, 7, 17
- [54] Suryansh Kumar, Yuchao Dai, and Hongdong Li. Monocular dense 3d reconstruction of a complex dynamic scene from two perspective frames. 2017 IEEE International Conference on Computer Vision (ICCV), 2017. 2
- [55] Jiahui Lei, Yijia Weng, Adam W Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2
- [56] Ruilong Li, Julian Tanke, Minh Vo, Michael Zollh¨ofer, J¨urgen Gall, Angjoo Kanazawa, and Christoph Lassner. Tava: Template-free animatable volumetric actors. In European Conference on Computer Vision, 2022. 2
- [57] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 2
- [58] Zhengqi Li, Tali Dekel, Forrester Cole, Richard Tucker, Noah Snavely, Ce Liu, and William T Freeman. Learning the depths of moving people by watching frozen people. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019. 2
- [59] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2021. 1, 2

- [60] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2023. 1, 2, 6

- [61] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2
- [62] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and

- robust structure and motion from casual dynamic videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2, 4, 5, 15
- [63] Ce Liu, Jenny Yuen, and Antonio Torralba. Sift flow: Dense correspondence across scenes and its applications. IEEE Transactions on Pattern Analysis and Machine Intelligence,

2011. 2

- [64] Qingming Liu, Yuan Liu, Jiepeng Wang, Xianqiang Lv, Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dynamic gaussian splatting from causually-captured monocular videos. ICLR, 2025. 2
- [65] Xingyu Liu, Charles R Qi, and Leonidas J Guibas. Flownet3d: Learning scene flow in 3d point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019. 1
- [66] David G. Lowe. Distinctive image features from scaleinvariant keypoints. International Journal of Computer Vision, 2004. 2
- [67] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3r: Aligned monocular depth estimation for dynamic videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2
- [68] Bruce D. Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In International Joint Conference on Artificial Intelligence,

1981. 2

- [69] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), 2024. 1
- [70] Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. Consistent video depth estimation. ACM Transactions on Graphics (ToG), 39(4), 2020. 2
- [71] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 2, 17
- [72] Michal Neoral, Jon´aˇs Ser´ˇ ych, and Jiˇr´ı Matas. Mft: Longterm tracking of every pixel. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024. 2
- [73] Richard A. Newcombe, Dieter Fox, and Steven M. Seitz. Dynamicfusion: Reconstruction and tracking of non-rigid scenes in real-time. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 2
- [74] Tuan Duc Ngo, Peiye Zhuang, Chuang Gan, Evangelos Kalogerakis, Sergey Tulyakov, Hsin-Ying Lee, and Chaoyang Wang. Delta: Dense efficient long-range 3d tracking for any video. In ICLR, 2025. 2, 6, 7
- [75] David Novotny, Nikhila Ravi, Benjamin Graham, Natalia Neverova, and Andrea Vedaldi. C3dpo: Canonical 3d pose networks for non-rigid structure from motion. In ICCV,

2019. 2

- [76] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields.

- In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021. 1, 2
- [77] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. SIGGRAPH Asia, 2021. 1, 2, 6, 7
- [78] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 15
- [79] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 2
- [80] Gilles Puy, Alexandre Boulch, and Renaud Marlet. Flot: Scene flow on point clouds guided by optimal transport. In European Conference on Computer Vision, 2020. 1
- [81] Rene Ranftl, Vibhav Vineet, Qifeng Chen, and Vladlen Koltun. Dense monocular depth estimation in complex dynamic scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2
- [82] Zhile Ren, Orazio Gallo, Deqing Sun, Ming-Hsuan Yang, Erik B Sudderth, and Jan Kautz. A fusion approach for multi-frame optical flow estimation. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV),

2019. 2

- [83] Michael Rubinstein and Ce Liu. Towards longer long-range motion trajectories. In British Machine Vision Conference,

2012. 2

- [84] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary R. Bradski. Orb: An efficient alternative to sift or surf. 2011 International Conference on Computer Vision, 2011. 2
- [85] Chris Russell, Rui Yu, and Lourdes Agapito. Video popup: Monocular 3d reconstruction of dynamic scenes. In European conference on computer vision. Springer, 2014. 2
- [86] P Sand. Long-range video motion estimation using point trajectories. PhD thesis, Ph. D. dissertation, Cambridge, MA, USA, 2006, adviser-Teller, Seth, 2006. 2
- [87] Peter Sand and Seth Teller. Particle video: Long-range motion estimation using point trajectories. International journal of computer vision, 2008. 2
- [88] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 7, 15, 16, 17
- [89] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016. 5
- [90] Xiaoyu Shi, Zhaoyang Huang, Weikang Bian, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. In ICCV, 2023. 2

- [91] Josef Sivic, Frederik Schaffalitzky, and Andrew Zisserman. Object level grouping for video shots. International Journal of Computer Vision, 2004. 2
- [92] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5), 2023. 2
- [93] Yunzhou Song, Jiahui Lei, Ziyun Wang, Lingjie Liu, and Kostas Daniilidis. Track everything everywhere fast and robustly. In European Conference on Computer Vision, 2024. 2
- [94] Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost, Federico Tombari, Gordon Wetzstein, and Leonidas Guibas. Dynamic gaussian marbles for novel view synthesis of casual monocular videos. Siggraph Asia, 2024. 2, 8, 16, 17
- [95] Timo Stich, Christian Linz, Georgia Albuquerque, and Marcus Magnor. View and time interpolation in image space. In Computer Graphics Forum. Wiley Online Library,

2008. 2

- [96] Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017. 2
- [97] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, 2020. 2
- [98] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 2021. 15
- [99] Zachary Teed and Jia Deng. Raft-3d: Scene flow using rigid-motion embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 5
- [100] Chaoyang Wang, Ben Eckart, Simon Lucey, and Orazio Gallo. Neural trajectory fields for dynamic novel view synthesis. arXiv preprint arXiv:2105.05994, 2021. 2
- [101] Chaoyang Wang, Xueqian Li, Jhony Kaesemodel Pontes, and Simon Lucey. Neural prior for trajectory estimation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1
- [102] Heng Wang and Cordelia Schmid. Action recognition with improved trajectories. 2013 IEEE International Conference on Computer Vision, 2013. 2
- [103] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Jingyi Yu, and Lan Xu. Fourier plenoctrees for dynamic radiance field rendering in real-time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2022. 2

- [104] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In International Conference on Computer Vision, 2023. 2, 4
- [105] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d

- perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2
- [106] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 2
- [107] Shizun Wang, Xingyi Yang, Qiuhong Shen, Zhenxiang Jiang, and Xinchao Wang. Gflow: Recovering 4d world from monocular video. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025. 2
- [108] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 2004. 5
- [109] Zirui Wang, Shuda Li, Henry Howard-Jenkins, Victor Adrian Prisacariu, and Min Chen. Flownet3d++: Geometric losses for deep scene flow estimation. 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 2019. 1
- [110] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 2022. 2
- [111] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2023. 1, 2
- [112] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 2
- [113] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 6, 7
- [114] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: 3d point tracking made easy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 8
- [115] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022. 2
- [116] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv. 4dgt: Learning a 4d gaussian transformer using real-world monocular videos. arXiv preprint arXiv:2506.08015, 2025. 8
- [117] Gengshan Yang, Deqing Sun, V. Jampani, Daniel Vlasic, Forrester Cole, Huiwen Chang, Deva Ramanan, William T. Freeman, and Ce Liu. Lasr: Learning articulated shape reconstruction from a monocular video. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2

- [118] Gengshan Yang, Deqing Sun, Varun Jampani, Daniel Vlasic, Forrester Cole, Ce Liu, and Deva Ramanan. Viser: Video-specific surface embeddings for articulated 3d shape reconstruction. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2021.
- [119] Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, and Hanbyul Joo. Banmo: Building animatable 3d neural models from many casual videos. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2
- [120] Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023. 4, 15
- [121] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024. 3, 4, 6, 7, 8, 15, 17
- [122] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024. 1, 2, 6, 7
- [123] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. ICLR, 2024. 2
- [124] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34), 2025. 5
- [125] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 2, 8, 16
- [126] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arxiv:2410.03825, 2024. 2
- [127] Zhoutong Zhang, Forrester Cole, Richard Tucker, William T Freeman, and Tali Dekel. Consistent depth of moving objects in video. ACM Transactions on Graphics (TOG), 40(4), 2021. 2
- [128] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In European Conference on Computer Vision. Springer, 2022. 2
- [129] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 2
- [130] Michael Zollh¨ofer, Matthias Nießner, Shahram Izadi, Christoph Rhemann, Christopher Zach, Matthew Fisher,

Chenglei Wu, Andrew William Fitzgibbon, Charles T. Loop, Christian Theobalt, and Marc Stamminger. Realtime non-rigid reconstruction using an rgb-d camera. ACM Transactions on Graphics (TOG), 33, 2014. 2

### Appendix A. Additional Preprocessing Details

Obtaining Camera Poses. Our method takes video sequences with known camera poses as input. To obtain camera poses for in-the-wild videos with moving objects, we adopt one of these two approaches depending on the type of input camera motion: (1) if there is sufficient camera motion parallax, we use COLMAP [88]’s SfM pipeline to obtain the camera poses and the sparse point clouds for the static regions, where we exclude keypoints in the foreground masks during the feature extraction stage. The foreground masks are generated using TrackAnything [120], a flexible and interactive tool for video object tracking and segmentation. The static point clouds produced by COLMAP [88] can then be used for aligning the affine-invariant monocular depth maps from Depth Anything [121]. (2) If the video is captured by a roughly stationary camera (small interframe camera baseline), COLMAP tends to fail catastrophically. Initial results on DAVIS use camera poses estimated with Unidepth [78] and DroidSLAM [98]. Specifically, we first employ Unidepth to predict metric depth and camera intrinsics. Using these predictions, we then estimate camera poses with DroidSLAM. (3) Camera parameters of in-the-wild videos in supplemental videos are estimated with MegaSaM [62].

Aligning Monocular Depth Maps. The dispairty output from Depth Anything [121] model is affine-invariant, so we need to align them with the cameras and reconstruction. To do so, we solve for a per-frame global scale and shift parameters that minimizes ℓ2 distance between the monocular disparity from Depth Anything and the disparity derived from SfM sparse point clouds or depth outputs from DroidSLAM or MegaSaM.

Computing Long-Range 2D Tracks. We utilize TAPIR [16] to compute long-range 2D tracks for a video. While the ideal scenario would involve computing fulllength tracks for every pixel in every frame (i.e., exhaustive pairs of correspondences), this approach is prohibitively computationally expensive. We therefore only compute full-length tracks for pixels located on a grid for foreground moving objects in each frame. In our experiment, we set grid interval to 4 (i.e., sampling a query point every 4 pixels). Due to our low-dimensional motion representation, we observe that our method can effectively operate with semidense 2D tracks without significant performance degradation During training, we filter out correspondences that TAPIR predicts with high uncertainty or those that are occluded.

### B. Additional Training Details

#### B.1. Initialization Details

During initialization stage, we first solve a Procrustes alignment problem for each cluster b, where we estimate SE(3) transformation between point sets {X0}b and {Xτ}b for all τ = 0,...,T . We exclude point pair when either one of them is occluded, and weight the point pair using the uncertainty score predicted by TAPIR [16] when solving Procrustes. This process produces the initialization for the set of basis functions {T(0b→) t}Bb=1. To initialize the motion coefficient w(b) for each track, we compute the distance between the 3D location of the track in the canonical frame and the 3D location of each cluster center, and initialize each of the corresponding motion coefficient value to be exponentially decay with the distance.

We then optimize the µ0, w(b), and set of basis functions {T(0b→) t}Bb=1 to fit the observed 3D tracks lifted by monocular depths and 2D tracks. Specifically, we enforce an ℓ1 loss between each of our predicted 3D track and its corresponding observed 3D track. In addition, we enforce the motion bases to be temporally smooth by adding an ℓ2 regularization on the acceleration of both the quaternion and the translation vector. We optimize the parameters using Adam [48] optimizer for 2k steps, where the initial learning rates for µ0, w(b), and {T(0b→) t}Bb=1 are 1 × 10−3, 1 × 10−2 and 1 × 10−2, respectively. All learning rates are exponentially decayed to 101 of their initial values during the optimization process.

#### B.2. Training Details

Gaussian Initialization. The aforementioned initialization stage gives the initialization of the mean of each Gaussian in the canonical frame. We follow the original 3D-GS [47] paper to initialize the scale s, rotation R0, and opacity o of each Gaussian in the canonical frame. The color c of each Gaussian is initialized as the pixel color at the projected location in the canonical frame.

Optimization. We use Adam [48] Optimizer to optimize all scene parameters. The learning rates for Gaussian’s canonical mean µ0, opacity o, scale s, rotation R0 (parameterized as quaternion) and color c are set to 1.6 × 10−4, 1 × 10−2, 5×10−3, 1×10−3, and 1×10−2, respectively. The learning rates for the SE(3) motion bases and the motion coefficients are set to 1.6×10−4 and 1×10−2 respectively. During each training iteration, we randomly select a batch of 8 query frames. For each query frame, we render the color, mask, depth, and the 3D track locations for 4 randomly selected target frames.

Loss weights and details. Our first set of loss function enforces our rendered results to match the per-frame pixelwise

color, depth, and masks inputs. The coefficients for depth loss λdepth and mask loss λmask are set to 0.5 and 1.0, respectively. We additionally add regularization to the estimated surface geometry via a depth gradient loss and per-Gaussian scale penalty. In particular, We add a pixelwise ℓ1 loss of spatial gradient between the depth renderings Dˆ t and corresponding reference depth maps D, with a weight set to 1.0. In addition, we enforce the foreground Gaussian to be isotropic by incorporating a regularization term that penalizes standard deviations of the scale s along all three axes.

Our second set of losses supervise the motion of the Gaussian. The weights for the 2D tracking loss λtrack-2d and the track depth loss λtrack-depth are set to 2.0 and 0.1, respectively. The Ltrack-2d is applied on normalized pixel coordinates (i.e., divided by the maximum image edge length). For the rigidity loss Lrigidity, we use foreground part masks from SAM automatic segmentation independently on each frame. For each training iteration, we sample 4 part masks for each query frame. For each mask, we sample 32 center points and find their 16 nearest neighbors within the mask. We compute the 3D track locations for all point samples in 4 target frames, and regularize the distances between the center points and their neighbors in the target frames to be similar to their distances in the query frame. We let λrigidity = 0.1. We weight the neighbor distances with an exponential kernel exp(−β∥x−x¯∥) with β = 2, where x¯ is the center point and x is one of its neighbors. We additionally add motion smoothness regularization that enforces an ℓ2 loss on acceleration of the motion translation bases and motion quaternion bases, with a weight set to 0.1 and an ℓ2 loss on the acceleration along z-axis (in the camera frame) of the µt, both through finite difference approximation.

Training with 2D Gaussian Splatting (2DGS) To enhance scene geometry estimation, we replace 3D Gaussian Splatting with 2D Gaussian Splatting [34]. Specifically, we employ an off-the-shelf monocular normal estimator [31] to generate monocular normal maps, N˜, which are then used to supervise both the rendered normals and the normals derived from the rendered depth. The supervision is achieved through the following losses:

Ln =

i

ωi(1 − nTi N˜) (11)

LN = 1 − NTN˜ (12)

where the summation is over the splats intersect the current ray, ωi represents the blending weight for i-th splat, and ni denotes the normal of i-th splat. Here, N represents the normal derived from rendered depth map. This supervision ensures that the orientation and depth of the 2D splats are aligned with the monocular normal prediction.

### C. Additional Evaluation Details

In our evaluation, since the synthetic Kubric dataset [28] comes with groundtruth camera poses, we directly use the groundtruth camera poses for our experiments. For iPhone dataset [25], we observed that the provided camera poses from ARKit are not accurate, we thus perform an additional global bundle adjustment using COLMAP [88] to refine the camera poses while fixing the camera intrinsics. To maintain metric scale after refinement, we compute a global SIM(3) transformation for each scene to align the refined camera poses with the original metric-scale camera poses. This allows us to evaluate 3D tracking performance in metric scale.

### D. Visualization of Kubric Experiment

We find qualitatively that the optimized motion coefficients of the scene representation are coherently grouped with each moving object in the scene. We demonstrate this in Figure 6, where we show the first 3 PCA components of our optimized motion coefficients of evaluation scenes.

|[Figure 45]|
|---|

|[Figure 46]|
|---|

Input PCA Coef.

- Figure 6. First three PCA components of the optimized motion coefficients.

E. Visualization of Complex Dynamic Scene

Our method is able to handle scenes with multiple moving objects. We visualize novel views and motion PCA of a real world complex scene in Fig. 7.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

- Figure 7. Novel view and motion coefficient PCA visualizations at time steps 0 and 54 of the school-girl sequence from the DAVIS dataset.

### F. NVIDIA Dataset Evaluation

We conduct experiments on seven scenes from the NVIDIA Dynamic Scenes dataset [125], following the evaluation protocol of Dynamic Gaussian Marbles [94]. Specifically, we use video footage from the static camera 4 for training and employ videos from cameras 3, 5, and 6 for evaluation. For consistency with Dynamic Gaussian Marbles [94]’s experiments, we use images at half-resolution for all seven experiments, though our method is compatible with

high-resolution images as well. Camera poses are estimated using COLMAP [88], and depths are predicted with Depth Anything [121] and subsequently aligned with the COLMAP point cloud. The quantitative results reported in this paper differ from those in DGM [94]. Specifically, we compute covisibility masks between training and test views and apply them during evaluation, whereas DGM [94] inpaints unobserved regions in test views for evaluation. This difference in evaluation procedure accounts for the variation in reported performance. We will make the covisibility masks publicly available to facilitate future research.

### G. DynMF Implementation Detail

DynMF [53] shares similar design choices with our method. Since they don’t have public code available, we implement our own version of it and provide implementation details here. Follow equation (4) in DynMF [53], each motion basis is represented as a MLP:

bj(t) = MLPj

t T

(13)

where bj is the j − th motion basis, t is the queried time step, and T is the number of time frames. We use a total of 10 motion bases. We apply positional encoding with 10 frequency bandwidths [71] to the input Tt . Following DynMF [53], we apply displacement motion to Gaussian’s means and rotation quaternions as follows:

µ(t) = µc +

10

cjbµj (t) (14)

j=1

where µc is the Gaussian means in the canonical frame and {bµj (t)}10j=1 are the mean motion bases represented as MLPs. and

10

cjbRj (t) (15)

q(t) = qc +

j=1

where qc is the Gaussian rotation represented as quaternion in the canonical frame and {bRj (t)}10j=1 are the rotation motion bases represented as MLPs.The motion coefficients {cj}10j=1 are shared between mean and rotation motions.

