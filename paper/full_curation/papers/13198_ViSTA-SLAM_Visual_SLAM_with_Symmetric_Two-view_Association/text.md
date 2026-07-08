### ViSTA-SLAM: Visual SLAM with Symmetric Two-view Association

Ganlin Zhang 1,2 Shenhan Qian 1,2 Xi Wang 1,2,3 Daniel Cremers 1,2 1 TU Munich 2 MCML 3 ETH Zurich

# arXiv:2509.01584v2[cs.CV]6Jan2026

[Figure 1]

|[Figure 2]<br><br>[Figure 3]|
|---|
|[Figure 4]<br><br>[Figure 5]|

[Figure 6]

[Figure 7]

[Figure 8]

Frontend: Symmetric Two-view Association

|𝑇𝑖𝑗|
|---|

## +

[Figure 9]

[Figure 10]

view 2 view 3

Backend:

##### …

[Figure 11]

Sim(3) Pose Graph

view 1

|[Figure 12]<br><br>view 𝑖|
|---|

|[Figure 13]<br><br>view 𝑗|
|---|

Optimization

###### Running Speed: 78 FPS

Figure 1. ViSTA-SLAM Results on a Multi-room Scene [8]. By combining the proposed lightweight frontend Symmetric Two-view Association (STA) model with Sim(3) pose graph optimization and loop closuring as the backend, ViSTA-SLAM achieves high-quality reconstruction and accurate trajectory estimation on challenging scenes while running in real time.

#### Abstract

We present ViSTA-SLAM as a real-time monocular visual SLAM system that operates without requiring camera intrinsics, making it broadly applicable across diverse camera setups. At its core, the system employs a lightweight symmetric two-view association (STA) model as the frontend, which simultaneously estimates relative camera poses and regresses local pointmaps from only two RGB images. This design reduces model complexity significantly, the size of our frontend is only 35% that of comparable state-ofthe-art methods, while enhancing the quality of two-view constraints used in the pipeline. In the backend, we construct a specially designed Sim(3) pose graph that incorporates loop closures to address accumulated drift. Extensive experiments demonstrate that our approach achieves superior performance in both camera tracking and dense 3D reconstruction quality compared to current methods. Github repository: https://github.com/zhangganlin/ vista-slam

#### 1. Introduction

###### 1.1. Real-time Monocular Dense SLAM

Simultaneous Localization and Mapping (SLAM) jointly estimates an agent’s pose and the surrounding 3D scene from sensor observations. In the monocular dense setting,

a single RGB camera is used to reconstruct a continuous 3D map, enabling both geometric accuracy and visual realism. The camera poses and dense reconstruction outputs underpin downstream tasks [2, 19, 31, 32, 42] such as semantic perception, object interaction, and scene editing, and are crucial for applications in VR/AR, robotics, and autonomous driving, where accurate, low-latency 3D perception is essential.

###### 1.2. Related Work

Classical Visual SLAM Classical visual SLAM methods can be broadly categorized into two types. The first, similar to incremental SfM [1, 26, 41], is feature-based SLAM [6, 10, 34, 35], relying on keypoint extraction and descriptor matching to provide constraints for triangulation and PnP [21, 22] pose estimation. The second, known as direct methods, such as LSD-SLAM [15] and DSO [16], optimizes camera poses by minimizing photometric error from pixel intensities while estimating a per-frame depth map. Both categories typically adopt a frontend (feature-based or direct) and a backend for optimization, most often bundle adjustment [49] to jointly refine poses and structure. However, they rely heavily on accurate camera calibration and are generally limited to sparse 3D reconstructions.

Dense Visual SLAM To enable denser reconstructions, recent works have incorporated deep learning into either

the frontend or the scene representation. For example, DROID-SLAM [48] uses RAFT [47] for dense optical flow in the frontend and performs dense bundle adjustment on the GPU, while methods such as SuperPrimitive [30] and COMO [12] leverage monocular priors (e.g., surface normals, depth distributions) from pretrained models with direct photometric optimization. BA-Track [7] augments point-based tracking with a scale-grid deformation of monocular depth priors. Neural scene representations have also been adopted, with NICER-SLAM [61] and MonoGS [29] optimizing camera poses and 3D structure using NeRF [33] and 3D Gaussian Splatting [20], respectively. Tracking robustness and geometric accuracy have been further improved by integrating depth priors and auxiliary trackers [40, 57–60]. However, most approaches still require accurate camera intrinsics and many struggle to achieve true real-time performance due to the computational demands of dense optimization and neural rendering. SLAM with 3D Foundation Models All aforementioned methods require known and accurate camera intrinsics. With the advent of 3D foundation models [23, 51, 53], several intrinsic-free SLAM frameworks have emerged, aiming to produce dense outputs without calibration. Methods such as Spann3R [50] and others [5, 27, 52] extend the two-view DUSt3R [53] model to sequential inputs, directly regressing point clouds in a unified global coordinate system. Reloc3r [13] instead regresses only relative poses and performs offline global optimization using SfM techniques [37, 46, 56]. MASt3R-SLAM [36] extracts dense correspondences from MASt3R [23] and feeds them into a classical optimization pipeline, while submap-based methods [11, 28] employs the multiview model VGGT [51] to regress local submaps before stitching them via pose graph optimization. While these approaches address some classical limitations, they still face notable drawbacks:

- 1. Current two-view models [23, 53] use asymmetric architectures that regress pointmaps of both views to the first view’s coordinates, making it difficult to decouple views for backend optimization (e.g. loop closure).
- 2. Pure regression methods [27, 50, 52] predict incoming frames with previous memory, but suffer from drift and start forgetting once the trajectory gets longer.
- 3. Methods [27, 36, 50, 52] built on current two-view models inherit the asymmetric architecture with two separate decoders, resulting in large model size. Submapbased methods [11, 28] employ an even larger multiview model [51] to build submaps, which further increases the size of the frontend model.

###### 1.3. Contributions of ViSTA-SLAM

To address these concerns, we propose ViSTA-SLAM, a novel real-time monocular visual SLAM pipeline based on symmetric two-view association. At its core is a lightweight

Symmetric Two-view Association (STA) model frontend, which takes two RGB images as input and simultaneously regresses two pointmaps in their respective local coordinate frames, along with the relative camera pose between them. During training, we enforce cycle consistency on relative poses and geometric consistency on pointmaps to improve accuracy and stability. Unlike prior 3D models [23, 51, 53], STA is fully symmetric with respect to its inputs: neither view is designated as a reference, and the same encoder–decoder architecture is applied to both. In the backend, we perform Sim(3) pose graph optimization with loop closures to mitigate drift and ensure global consistency. To further enhance robustness, each view is represented by multiple nodes rather than a single one, which are connected by scale-only edges to handle scale inconsistencies across different forward passes.

This symmetric design makes our frontend substantially more lightweight than existing methods, with STA being only 64% the size of MASt3R [23] and 35% the size of VGGT [51]. Unlike prior approaches [11, 28] that group multiple views into a single submap node, our method assigns each view its own nodes in the pose graph. Leveraging the local pointmaps produced by the STA frontend, each node can be represented independently, connected to others solely through relative transformations. Compared to submap-based methods [28], this design yields a more flexible graph structure and greater robustness. The combination of this flexibility and lightweight architecture underpins our choice of a symmetric two-view model as the frontend.

In summary, our main contributions are as follows:

- • We design and train a lightweight, symmetric two-view association network as the frontend, which takes only two RGB images as input and regresses their pointmaps in local coordinates along with the relative camera pose.
- • We construct a robust Sim(3) pose graph with loop closures, optimize it using the Levenberg–Marquardt algorithm for fast and stable convergence.
- • By integrating these components, we present a real-time monocular dense visual SLAM framework that operates without requiring any camera intrinsics.
- • Our method achieves state-of-the-art performance on the real-world 7-Scenes [43] and TUM-RGBD [45] datasets for both camera trajectory estimation and dense 3D reconstruction.

#### 2. ViSTA-SLAM Pipeline

As a monocular dense SLAM pipeline (Fig. 2), our aim is to simultaneously track camera poses and reconstruct the recorded scene online using a dense pointcloud. To achieve this, we propose a lightweight and novel symmetric twoview association model as the frontend of our pipeline, which extracts the relative pose and local point maps of two neighboring input frames (Sec. 2.1 and Sec. 2.2). In

|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
|---|
|……|
|[Figure 17]<br><br>|
|[Figure 18]<br><br>|

|[Figure 19]<br><br>[Figure 20]| | |
|---|---|---|
| |image 𝑖| |

[Figure 21]

[Figure 22]

𝑇𝑖𝑗

RelPose Head

[Figure 23]

Decoder

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Decoder-

DecoderBlock1

Encoder ...

[Figure 28]

BlockB

[Figure 29]

[Figure 30]

Local Pointmap

Head

Two-view Output

Information Sharing

𝑇𝑗𝑖

[Figure 31]

RelPose Head

[Figure 32]

|[Figure 33]<br><br>[Figure 34]| | |
|---|---|---|
| |image 𝑗| |

Decoder

[Figure 35]

[Figure 36]

[Figure 37]

view 2 view 3

[Figure 38]

[Figure 39]

view 1 …

DecoderBlockB

Decoder-

[Figure 40]

Encoder ...

Block1

|[Figure 41]<br><br>[Figure 42]|
|---|

|[Figure 43]<br><br>view 𝑖|
|---|

[Figure 44]

Local Pointmap Head

|[Figure 45]<br><br>view 𝑗|
|---|

Reconstructed Pointcloud

Frontend: Symmetric Two-view Association (STA) with Shared Weights

Backend: Pose Graph Optimization

& Estimated Camera Trajectory

with Loop Closure

Figure 2. ViSTA-SLAM Overview. Given sequential video frames without intrinsics as the input, our frontend model takes in view pairs and predicts local pointmaps and relative poses within each pair. We then use the pair-wise predictions to construct a Sim(3) pose graph with loop closure and optimize it via Levenberg–Marquardt algorithm. The frontend model employs a fully symmetric design, making the model lightweight and supporting more flexible pose graph optimization. The blue edges in the pose graph and final results represent connections between neighboring nodes (views), while the orange edges correspond to loop closures.

the backend, a sparse and efficient Sim(3) pose graph optimization with loop closure is performed to mitigate drift accumulation (Sec. 2.3).

###### 2.1. Symmetric Two-view Association Model

In classical monocular SLAM pipelines, the two-view estimation is one of the most critical building block, as it establishes geometric constraints that allow for further optimization. In this work, we follow the same principle; however, instead of relying on traditional methods, we propose a deep learning based Symmetric Two-view Association (STA) model that eliminates the need for camera intrinsics in the SLAM process.

Encoder Our STA model takes in two images Ii, Ij as input. It uses a shared ViT encoder [14] to divide input images into patches and encode them into features

Ei/j = Encoder Ii/j ∈ RK×C,

where K represents tokens and C denotes the dimensions. Subsequently, we insert a camera pose embedding p to the encoding features of each view, forming

Ei/j′ = p,Ei/j ∈ R(K+1)×C. Decoder The decoder further processes and fuses information between the encoding features Ei′ and Ej′. It contains a sequence of B decoder blocks, each conducting a self-attention operation followed by a cross-attention operation, producing decoding features

′

Di/j(b) = DecoderBlock(b) Di/j(b−1),Dj/i(b−1) ∈ R(K+1)×C

,

where b ∈ {1,2,...,B} is the index of a decoder block, and Di/j(0) = Ei/j′ .

Symmetric Formulation Prior approaches [23, 53] regress both point maps into the coordinate frame of the

first view, thus requiring two separate decoders. In contrast, our model predicts only local point maps and relative poses between views. This fully symmetric formulation makes it possible to use only one decoder. As a result, the number of parameters for decoding is effectively reduced by half (shown in Fig. 3), forming a more compact model for realtime applications. Moreover, producing local view outputs in their own coordinate systems is better suited for the subsequent pose graph optimization, see Sec. 2.3 for details.

Local Point Maps Given decoding features Di/j(b), we use a DPT head [38] to regress the local point maps P and corresponding confidence maps W:

###### Pi/j,Wi/j = PointHead Di/j(b)

Relative Poses Given the first embedding of the decoding feature Di(B), i.e., the camera pose embedding p(iB) ∈ R1×C

′

, we use an MLP to regress the relative transformation from view i to j . Specifically, the MLP outputs a matrix Mij ∈ R3×3 for rotation, a translation vector tij ∈ R3×1, and a confidence score wij ∈ [0,1]:

Mij,tij,wij = PoseHead(p(iB))

Since Mij is not guaranteed to lie on the SO(3) manifold, we apply SVD orthogonalization [24] to it to obtain a valid rotation matrix Rij. Then, the relative transformation is Tij = [Rij|tij]. With our symmetric formulation, we could also input pj into the pose head to regress Tji. But in practice we only regress one of them for building pose graph.

###### 2.2. Training Objective

There are three loss terms to supervise the training of our STA model. Pointmap Loss compares the regressed local pointmap with ground-truth points. Relative Pose Loss

penalizes errors in relative rotation and translation, with a cycle-consistency term ensuring the two predicted poses are mutual inverses. Geometric Consistency Loss enforces alignment of the two local point maps after applying the predicted relative transformation, improving local reconstruction consistency.

Local Pointmap Loss Following DUSt3R [53], we apply the confidence-weighted regression loss for all predicted point maps with valid ground truths. Since the reconstruction is up-to-scale, we also normalize the regressed pointmap and the ground-truth pointmap according to their mean Euclidean distance to the origin, n and nˆ:

P ˆv(x)

Pv(x) n

nˆ −

Wv(x) ·

Lpmap =

v∈{i,j} x∈Iv

− αpoint log (Wv(x)), (1)

where x is the pixel coordinate. Note that all the points are regressed in the local coordinate space of each view.

Relative Pose Loss Relative pose loss consists of three parts: rotation loss, translation loss and identity loss. The rotation loss LR evaluates the angle between the regressed rotation Rij and the ground-truth rotation Rˆij,

LR(R,Rˆ) = arccos

tr(R−1Rˆ) − 1 2

. (2)

The translation loss Lt evaluates the euclidean distance between the predicted translation tij and the ground truth tˆij, which are normalized by the same factors n and nˆ as for the pointmap loss in Eq. (1):

Lt(t,tˆ) =

tˆ nˆ

t n −

2

. (3)

The pose identity loss Lid minimizes the difference of TijTji and the identity transformation I, essentially constraining Tij and Tji to be the inverse of each other to improves the consistency of our pose prediction:

Lid = LR(RijRji,I) + Lt(Rijtji + tij,0). (4) Then the complete relative pose loss is defined as

Lpose =wij LR(Rij,Rˆij) + Lt(tij,tˆij) + Lid − α log(wij), (5)

weighted by a separate confidence score wij for pose regression.

Geometrical Consistency Loss To ensure that the predicted point maps of each pair are spatially consistent when placed in the same coordinate space, we introduce a geometric consistency loss. Given a pair of images with groundtruth intrinsics, depths, and relative poses, each pixel x in

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

- Figure 3. Asymmetric vs. Symmetric Architectures. Asymmetric architectures [23, 53] use two decoders to regress point maps in a shared coordinate space. our symmetric formulation regresses relative pose and local point maps with only a single decoder, reducing over 36% of the parameters (∼ 0.4 vs. 0.7 billion), while achieving higher accuracy and enabling pose graph optimization in the backend.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

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

- Figure 4. An Example Pose Graph for 5 views (N = 2). Nodes of the same view are grouped and connected to the first processed node of that view via scale edges. Our two-type-edge design enhances optimization robustness, yielding more accurate poses.

view i can be accurately projected into view j, yielding its ground-truth corresponding pixel Cij(x) in view j. Then the geometric consistency loss Lgc is defined as:

∥TijPi(x) − Pj (Cij(x))∥/n, (6)

Lgc =

x∈Ii

where Tij is the predicted relative transformation and n is the same normalization factor as in Eq. (1).

The total training objective function L is the weighted sum of the above three losses,

L = λpmapLpmap + λposeLpose + λgcLgc, (7) where λpmap, λpose and λgc are weighting factors.

###### 2.3. Backend Pose Graph Optimization

Notation In the backend, a Sim(3) pose graph G = (V,E) is maintained and optimized to mitigate accumulated errors introduced by the two-view estimations. The vertex set V and edge set E are defined as

V = {vij | i,j ∈ N}, E = {eij | i,j ∈ N}, (8)

where vij,eij ∈ Sim(3), each vertex vij represents an absolute camera pose with scale, of view i, with view i and

viewj as the input of STA; each edge eij encodes the relative transformation between a pair of connected vertices vij and vji. Both vij and eij have a rigid transformation part T ∈ SE(3) and a scale part s ∈ R+.

Graph Construction To construct the pose graph, for a given view i, the STA model performs 2N forward passes

Method chess fire heads office pumpkin kitchen stairs Avg.

|NICER-SLAM [61] DROID-SLAM [48]<br><br>Calib.<br><br>GlORIE-SLAM [57]|0.033 0.069 0.042 0.108 0.200 0.039 0.108 0.036 0.027 0.025 0.066 0.127 0.040 0.026 0.036 0.029 0.014 0.094 0.144 0.053 0.020|0.086 0.049 0.056|
|---|---|---|
|CUT3R [52]<br><br>SLAM3R [27]<br><br>MASt3R-SLAM [36]<br><br>VGGT-SLAM [28]<br><br>Uncalib.<br><br>ViSTA-SLAM|0.743 0.226 0.363 0.664 0.546 0.381 0.413 0.089 0.048 0.036 0.088 0.196 0.102 0.126 0.063 0.046 0.029 0.103 0.112 0.074 0.032 0.037 0.026 0.022 0.103 0.147 0.063 0.095 0.073 0.035 0.028 0.055 0.129 0.035 0.029<br><br>|0.477 0.098 0.066 0.070 0.055<br><br>|

Calib.Uncalib.

Table 1. Camera Trajectory Estimation (ATE RMSE) on 7-Scenes [43]. ViSTA-SLAM performs the best on average.

with neighboring views j ∈ [i − N,i − 1] ∪ [i + 1,i + N]. The predicted pointmap for each view in each forward pass corresponds to a node in the graph, resulting in multiple nodes per view since each view is processed multiple times by the frontend model. As shown in Fig. 4, we define two types of edges in the graph. Pose edges connect two nodes generated from the same forward pass, using the estimated relative camera pose and an identity relative scale, based on the assumption that the two local point maps from a single forward pass share the same scale. Scale edges connect nodes belonging to the same view but obtained in different forward passes (paired with different neighboring views), with the rigid transformation component set to identity, and the scale component solved via weighted least squares between the predicted point maps of different forward passes. Among all nodes of the same view, scale edges are constructed only between the first processed node and the others for sparsity and simplicity.

Loop Closure We use Bag of Words [17] to detect loop candidates, which form new pairs. Then we can feed each candidates pair into our STA model to confirm these proximity. If the predicted confidence score of the relative pose is higher than a predefined threshold τp, this pair is accepted as a valid loop, and two new nodes connected by a pose edge are added. Each new node is also connected to the first processed node of its corresponding view via a scale edge.

Optimization We perform pose graph optimization using the Levenberg–Marquardt algorithm in the space of Lie algebra sim(3).

2 Ωij

logSim(3) eij · (vij)−1 · vji

(9)

min

{vij∈V} eij∈E

where Ωij represents the covariance matrix, derived from the confidence score predicted by the STA model. The optimization process takes less than 5 iterations to converge in most cases.

Using the optimized camera pose and scale, the reconstructed pointcloud P˜ij in global coordinate is,

P˜ij = sjiRijPij + tji, (10)

where Rij, tji and sji are the orientation, position, scale of vij respectively. To avoid redundancy, for each view i, we

only keep the pointcloud with largest confidence among all P˜i∗ in the final result.

#### 3. Experiments

Evaluation Datasets and Metrics Following VGGTSLAM [28], we evaluate our method on standard monocular SLAM benchmarks for camera tracking accuracy and reconstruction quality. We report root mean square error (RMSE) of absolute trajectory error (ATE, in meters) on real-world 7-Scenes [43] and TUM-RGBD [45] datasets using the evo toolkit [18]. Reconstruction quality on 7-Scenes is assessed via RMSE of accuracy, completion, and Chamfer distance (meters), leveraging its ground-truth 3D scenes. Implementation Details The frontend STA model is initialized from the weights of DUSt3R [53], and trained on ScanNet [8], ScanNet++[54], ARKitScenes[4], CO3D [39], Aria Synthetic Environments [3], and Replica [44] for 7 days using 8 NVIDIA H100 GPUs. We use AdamW optimizer to train our STA model with learning rate 1.5e−5, weight decay 0.01, B = 12, αpoint = 0.2, αpose = 0.05, λpmap = 1, λpose = 1, λgc = 1, and τp = 0.75. We conducted evaluations on a machine with an NVIDIA RTX 4090 GPU and an Intel i9-14900KF CPU, with N = 2 for 7-Scenes and N = 3 for TUM-RGBD.

Baselines ViSTA-SLAM is primarily compared with state-of-the-art (SOTA) learning-based SLAM methods in uncalibrated scenarios: VGGT-SLAM [28], MASt3RSLAM [36], SLAM3R [27], and CUT3R [52]. To reduce randomness from VGGT-SLAM’s RANSAC, we run it 5 times per scene as suggested in their paper; results for other methods, including ours, are deterministic. We also compare with SOTA methods using known camera intrinsics [6, 25, 48, 57, 61]. Some results are taken from [28, 36]. For MASt3R-SLAM and VGGT-SLAM, we keep their original keyframe selection; for ours, CUT3R, and SLAM3R, we use frame strides of 5 (7-Scenes) and 3 (TUM-RGBD). Calibrated methods are shown in gray. Best results are highlighted as first , second , and third .

###### 3.1. Camera Trajectory Evaluation

In Tab. 1 and Tab. 2, we report ATE RMSE. ViSTA-SLAM achieves the best average performance on both datasets, out-

Method 360 desk desk2 floor plant room rpy teddy xyz Avg.

|ORB-SLAM3 [6] DPV-SLAM [25]<br><br>DPV-SLAM++ [25] DROID-SLAM [48]<br><br>Calibrated<br><br>GlORIE-SLAM [57]|× 0.017 0.210 × 0.034 × × × 0.009 0.112 0.018 0.029 0.057 0.021 0.330 0.030 0.084 0.010 0.132 0.018 0.029 0.050 0.022 0.096 0.032 0.098 0.010 0.111 0.018 0.042 0.021 0.016 0.049 0.026 0.048 0.012 0.128 0.016 0.028 0.021 0.021 0.042 0.020 0.035 0.010|N/A 0.076 0.054 0.038 0.036|
|---|---|---|
|CUT3R [52]<br><br>SLAM3R [27]<br><br>MASt3R-SLAM [36]<br><br>VGGT-SLAM [28]<br><br>Uncalibrated<br><br>ViSTA-SLAM|0.174 0.592 0.546 0.662 0.467 0.911 0.051 0.845 0.129 0.211 0.861 0.967 0.790 0.755 1.013 0.063 0.986 0.185 0.070 0.032 0.055 0.056 0.035 0.118 0.041 0.116 0.020 0.063 0.031 0.048 0.152 0.023 0.133 0.038 0.039 0.020 0.104 0.030 0.030 0.070 0.052 0.067 0.023 0.080 0.015<br><br>|0.486 0.648 0.060 0.061 0.052<br><br>|

CalibratedUncalibrated

Table 2. Camera Trajectory Estimation (ATE RMSE) on TUM-RGBD [45]. ViSTA-SLAM performs the best on average.

performing current SOTA [36] by 17% (0.055 vs. 0.066) and 13% (0.052 vs. 0.060), and surpasses some calibrated methods [25, 61]. ViSTA-SLAM performs less effectively on TUM-RGBD 360 scene due to predominantly rotational camera motion that leads to frontend ambiguity and degrading performance. Other methods [28, 36] use either heavier multi-view frontend or more intensive optimization to reduce the influence. Pure regression-based methods [27, 52] struggle to maintain consistent registration over long sequences with large camera motion due to forgetting effects.

- In Fig. 5, we show the estimated trajectories from

different methods on 7-Scenes [43] office and TUMRGBD [45] room. CUT3R [52] suffers from severe forgetting issues on long sequences; SLAM3R [27] has bad point registration on the challenged scene TUM-RGBD room, thus, does not produce correct camera poses. Compared to pure regression-based methods, MASt3R-SLAM [36] and VGGT-SLAM [28] work well, while ViSTA-SLAM achieves even higher trajectory accuracy.

3.2. Dense Reconstruction Evaluation

In Tab. 3, we evaluate reconstruction quality across methods. Leveraging accurate camera poses and consistent local point clouds, ViSTA-SLAM achieves the best Chamfer distance among all approaches. Despite using a lightweight two-view frontend, ViSTA-SLAM, combined with tailored Sim(3) pose graph optimization, significantly outperforms multi-view-frontend methods [27, 28] in accuracy (0.45 vs. 0.52) while matching or exceeding completeness. To demonstrate the effectiveness of our lightweight frontend, we add another strong baseline, replacing our STA model with a two-view VGGT [51] as the frontend and conducting the same pose graph optimization. ViSTA-SLAM still achieves better performance in Chamfer distance, completeness, and absolute trajectory error, highlighting the effectiveness of our lightweight symmetric frontend over larger multiview models like VGGT for SLAM tasks

- In Fig. 6, we show qualitative reconstruction results

on 7-Scenes redkitchen, TUM-RGBD [45] room, and BundleFusion [9] apt1. CUT3R [52] fails to reconstruct

Method ATE ↓ Acc. ↓ Comp. ↓ Chamfer ↓

|DROID-SLAM [48] MASt3R-SLAM [36]|0.049 0.047|0.141 0.048 0.094 0.089 0.085 0.087<br><br>|
|---|---|---|
|Spann3R @20 [50] Spann3R @2 [50] CUT3R [52] SLAM3R [27] MASt3R-SLAM [36] VGGT-SLAM [28] 2-view VGGT w/ PGO ViSTA-SLAM|N/A N/A 0.477 0.098 0.066 0.070 0.065 0.055<br><br>|0.069 0.047 0.058 0.124 0.043 0.084 0.276 0.303 0.290 0.053 0.059 0.056 0.059 0.056 0.057 0.052 0.060 0.056 0.039 0.077 0.058 0.045 0.056 0.051<br><br>|

- Table 3. Tracking and Reconstruction Evaluation on 7Scenes [43]. @n indicates a keyframe every n images. 2-view VGGT w/ PGO uses the 2-view VGGT frontend with the same pose graph optimization as ours. ViSTA-SLAM achieves the best trajectory estimation and reconstruction performance on 7-Scenes.

CUT3R SLAM3R MASt3R VGGT 2-view VGGT ViSTA [52] [27] SLAM [36] SLAM [28] w/ PGO SLAM

Size ↓ 0.79 0.76 0.69 1.26 1.26 0.44 FPS ↑ 34.2 45.8 30.3 93.3 12.6 78.0

- Table 4. Model Size and Running Time Evaluation on 7Scenes [43] redkitchen. Model sizes are reported in billions of parameters. FPS indicates the average number of frames processed per second over three runs. ViSTA-SLAM shows highly competitive real-time speed with the smallest model size among baselines.

correctly due to forgetting issues, while SLAM3R [27] struggles in scenes with large camera perspective changes. MASt3R-SLAM [36] and VGGT-SLAM [28] produce artifacts on object boundaries, failing to clearly separate foreground from background, and show misalignment across views. In contrast, ViSTA-SLAM overcomes these challenges through geometric consistency constraints during training. Notably, VGGT-SLAM fails midway through the apt1 scene as backend optimization diverges, which stems from the unstable RANSAC-based 3D homography estimation, which can sample planar regions and cause ambiguity in their proposed SL(4) pose graph optimization.

CUT3R

###### SLAM3R

MASt3R-SLAM

###### VGGT-SLAM

Ours

0.198

y(m)

| | |
|---|---|
|[Figure 88]| |

0.5

0.102

1.0

1.5

groundtruth

groundtruth

groundtruth

groundtruth

groundtruth

0.007

1.0 0.5 0.0 0.5 1.0

1.0 0.5 0.0 0.5 1.0

1.0 0.5 0.0 0.5 1.0

1.0 0.5 0.0 0.5 1.0

1.0 0.5 0.0 0.5 1.0

CUT3R

###### SLAM3R

MASt3R-SLAM

###### VGGT-SLAM

Ours

x (m)

y(m)

1.0

0.277

|[Figure 89]| |
|---|---|
| | |

0.5

0.0

0.143

0.5

1.0

groundtruth

groundtruth

groundtruth

groundtruth

groundtruth

0.009

1.5

1 0 1

1 0 1

1 0 1

1 0 1

1 0 1

x (m)

- Figure 5. Trajectory estimation results on 7-Scenes office (top) and TUM-RGBD room (bottom). Estimated camera trajectories are projected onto the x–y plane, with ground-truth shown as dashed lines. The trajectory color encodes ATE RMSE: higher errors in red, lower in blue. For MASt3R-SLAM [36] and VGGT-SLAM [28], only the poses of their selected keyframes are estimated.

once every 32 keyframes, reducing the total number of inference steps. When replacing our STA model with a twoview VGGT that takes two views as input at a time like STA, the running speed is significantly slower, further demonstrating the effectiveness of our lightweight frontend.

load encoder decoder detect construct optimize data loop graph graph

time (s) 1.88 0.89 3.38 0.41 2.93 3.31 % 14.7% 7.0% 26.4% 3.18% 22.9% 25.9%

- Table 5. Time Spent on Each Component of ViSTA-SLAM for 7-Scenes [43] redkitchen (in seconds and percentage).

Settings ATE ↓ Chamfer ↓ STA Model

w/o Lgc 0.056 0.057 w/o Lid 0.058 0.059

Pose Graph

w/o pose graph opt. 0.105 0.070 w/o loop closure 0.103 0.072 w/o two edge types 0.057 0.051

ViSTA-SLAM with full features 0.055 0.051

- Table 6. Ablation Study on 7-Scenes [43]. w/o pose graph opt. simply accumulates relative poses for absolute poses. w/o two edge types uses the classical pose graph in which each view is represented by a single node.

Tab. 5 shows the percentage of runtime spent on major pipeline components. Decoding two-view information and pose graph optimization dominate the processing time.

###### 3.4. Ablation Study

In Tab. 6, we present ablations by selectively disabling components of ViSTA-SLAM. Incorporating all proposed features yields the best performance on both camera trajectory estimation and 3D reconstruction.

Both Lgc and Lid contribute substantially to reconstruction quality. Lgc improves consistency between the reconstructed local pointmaps of the two-view input pair, while Lid further refines the estimated relative camera poses by enforcing a cycle consistency constraint on the model.

###### 3.3. Model Size and Speed

Pose graph optimization with loop closure is also highly effective, bringing 48% improvement in the trajectory accuracy (0.105 → 0.055), as it introduces simple yet powerful constraints through the edges of the pose graph, preventing error accumulation from two-view estimations as the trajectory grows longer. These findings further support the symmetric formulation of our frontend to regress local pointclouds and relative poses, which maximizes the effectiveness of pose graph optimization since the pointmaps of each view are tightly coupled with their corresponding camera poses in the graph. On the contrary, if pointmaps were regressed in a shared coordinate system inside a submap, as in previous works [5, 23, 27, 50, 52, 53], pose updates would not be able to fix misalignments inside submaps.

We compare the frontend model size and processing speed across methods in Tab. 4. Owing to our symmetric design, the decoder and regression heads use only half the parameters of existing feedforward models [23, 27, 50, 53]. Consequently, our model is far more compact: only 64% the size of MASt3R [23] (used in MASt3R-SLAM [36]) and 35% the size of VGGT [51] (used in VGGT-SLAM [28]).

The speed evaluation further confirms that ViSTASLAM achieves real-time performance. Benefiting from both the compact frontend and the sparse pose graph, our approach is highly competitive in runtime—faster than the pure regression-based methods CUT3R [52] and SLAM3R [27], and comparable to VGGT-SLAM [28]. It is worth noting that VGGT-SLAM performs inference only

Our two-edge-type pose graph design improves camera

[Figure 90]

[Figure 91]

[Figure 92]

CUT3R[52]SLAM3R[27] MASt3R-

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

SLAM[36] VGGT-

[Figure 99]

[Figure 100]

[Figure 101]

SLAM[28] ViSTA-SLAM

[Figure 102]

[Figure 103]

[Figure 104]

- Figure 6. Reconstruction results on 7-Scenes redkitchen (left), TUM-RGBD room (middle), and BundleFusion apt1 (right). Purple boxes highlight reconstruction artifacts near the edges (background points wrongly mapped to the edge of the foreground). Red boxes indicate misalignments. Green boxes highlights ViSTA-SLAM’s competitive results. VGGT-SLAM fails to complete reconstruction on apt1 due to divergence in pose graph optimization.

trajectory estimation by representing each view with multiple nodes connected by scale and pose edges, rather than a single node with standard edges. This structure better averages out uncertainty, particularly relative scale variations in pointmaps from different forward passes, improving the robustness of pose graph optimization and the accuracy of trajectory estimation.

#### 4. Conclusion

We propose a novel monocular intrinsics-free SLAM pipeline, ViSTA-SLAM, which features a lightweight frontend (Symmetric Two-View Association) and a Sim(3) pose graph optimization with loop closure as the backend. Ex-

perimental results demonstrate the superior camera tracking accuracy and 3D reconstruction quality of ViSTA-SLAM. Meanwhile, it is significantly more lightweight and operates at a faster or comparable speed comparing current state-ofthe-art methods.

Limitation and Future Work Our method omits optimization of point clouds in the backend for efficiency consideration. Therefore, it can suffer from misalignments caused by imperfect pointmap prediction by the frontend model. Future work could explore incorporating implicit camera information from previous estimates or aligning latent features across views to enhance local consistency across forward passes.

#### References

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54

(10):105–112, 2011. 1

- [2] Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R. Zamir, Martin Fischer, Jitendra Malik, and Silvio Savarese. 3D scene graph: A structure for unified semantics, 3D space, and camera. In ICCV, 2019. 1
- [3] Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan Frost, Luke Holland, Campbell Orme, et al. Scenescript: Reconstructing scenes with an autoregressive structured language model. In ECCV, pages 247–263. Springer,

2024. 5

- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes - a diverse real-world dataset for 3D indoor scene understanding using mobile RGB-D data. In NeurIPS, 2021. 5
- [5] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. MUSt3R: Multi-view network for stereo 3D reconstruction. In CVPR, pages 1050–1060, 2025. 2, 7
- [6] Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez, Jos´e MM Montiel, and Juan D Tard´os. ORB-SLAM3: An accurate open-source library for visual, visual–inertial, and multimap SLAM. IEEE transactions on robotics, 37(6): 1874–1890, 2021. 1, 5, 6
- [7] Weirong Chen, Ganlin Zhang, Felix Wimbauer, Rui Wang, Nikita Araslanov, Andrea Vedaldi, and Daniel Cremers. Back on track: Bundle adjustment for dynamic scene reconstruction. arXiv preprint arXiv:2504.14516, 2025. 2
- [8] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In CVPR, 2017. 1, 5, 2
- [9] Angela Dai, Matthias Nießner, Michael Zoll¨ofer, Shahram Izadi, and Christian Theobalt. BundleFusion: Real-time globally consistent 3D reconstruction using on-the-fly surface re-integration. ACM TOG, 2017. 6, 4
- [10] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. MonoSLAM: Real-time single camera SLAM. IEEE TPAMI, 29(6):1052–1067, 2007. 1
- [11] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. VGGT-Long: Chunk it, loop it, align it–pushing VGGT’s limits on kilometer-scale long RGB sequences. arXiv preprint arXiv:2507.16443, 2025. 2
- [12] Eric Dexheimer and Andrew J Davison. COMO: Compact mapping and odometry. In ECCV, pages 349–365. Springer,

2024. 2

- [13] Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, and Yanchao Yang. Reloc3r: Largescale training of relative camera pose regression for generalizable, fast, and accurate visual localization. In CVPR, pages 16739–16752, 2025. 2

- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3
- [15] Jakob Engel, Thomas Sch¨ops, and Daniel Cremers. LSDSLAM: Large-scale direct monocular SLAM. In ECCV, pages 834–849. Springer, 2014. 1
- [16] Jakob Engel, Vladlen Koltun, and Daniel Cremers. Direct sparse odometry. IEEE TPAMI, 40(3):611–625, 2017. 1
- [17] Dorian G´alvez-L´opez and J. D. Tard´os. Bags of binary words for fast place recognition in image sequences. IEEE Transactions on Robotics, 28(5):1188–1197, 2012. 5
- [18] Michael Grupp. evo: Python package for the evaluation of odometry and SLAM. https://github.com/ MichaelGrupp/evo, 2017. 5
- [19] Anna-Maria Halacheva, Yang Miao, Jan-Nico Zaech, Xi Wang, Luc Van Gool, and Danda Pani Paudel. Holistic understanding of 3D scenes as universal scene description. arXiv preprint arXiv:2412.01398, 2024. 1
- [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2

- [21] Laurent Kneip, Davide Scaramuzza, and Roland Siegwart. A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation. In CVPR, pages 2969–2976. IEEE, 2011. 1
- [22] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua. EPnP: An accurate O(n) solution to the PnP problem. IJCV, 81(2):155–166, 2009. 1
- [23] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3D with MASt3R, 2024. 2, 3, 4, 7
- [24] Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia. An analysis of SVD for deep rotation estimation. NeurIPS, 33:22554–22565, 2020. 3
- [25] Lahav Lipson, Zachary Teed, and Jia Deng. Deep patch visual SLAM. In ECCV, pages 424–440. Springer, 2024. 5, 6
- [26] Shaohui Liu, Yidan Gao, Tianyi Zhang, R´emi Pautrat, Johannes L Sch¨onberger, Viktor Larsson, and Marc Pollefeys. Robust incremental structure-from-motion with hybrid features. In ECCV, pages 249–269. Springer, 2024. 1
- [27] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. SLAM3R: Real-time dense scene reconstruction from monocular RGB videos. In CVPR, pages 16651–16662, 2025. 2, 5, 6, 7, 8, 1
- [28] Dominic Maggio, Hyungtae Lim, and Luca Carlone. VGGTSLAM: Dense RGB SLAM optimized on the SL(4) manifold. arXiv preprint arXiv:2505.12549, 2025. 2, 5, 6, 7, 8
- [29] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting SLAM. In CVPR, pages 18039– 18048, 2024. 2
- [30] Kirill Mazur, Gwangbin Bae, and Andrew J Davison. SuperPrimitive: Scene reconstruction at a primitive level. In CVPR, pages 4979–4989, 2024. 2

- [31] Yang Miao, Iro Armeni, Marc Pollefeys, and Daniel Barath. Volumetric semantically consistent 3d panoptic mapping. In IROS, pages 12924–12931. IEEE, 2024. 1
- [32] Yang Miao, Francis Engelmann, Olga Vysotska, Federico Tombari, Marc Pollefeys, and D´aniel B´ela Bar´ath. Scenegraphloc: Cross-modal coarse visual localization on 3D scene graphs. In ECCV, pages 127–150. Springer, 2024. 1
- [33] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [34] Raul Mur-Artal and Juan D Tard´os. ORB-SLAM2: An opensource SLAM system for monocular, stereo, and RGB-D cameras. IEEE transactions on robotics, 33(5):1255–1262,

2017. 1

- [35] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. ORB-SLAM: A versatile and accurate monocular SLAM system. IEEE transactions on robotics, 31(5):1147– 1163, 2015. 1
- [36] Riku Murai, Eric Dexheimer, and Andrew J Davison. MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors. In CVPR, pages 16695–16705, 2025. 2, 5, 6, 7, 8
- [37] Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Johannes L Sch¨onberger. Global structure-from-motion revisited. In ECCV, pages 58–77. Springer, 2024. 2
- [38] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, pages 12179–12188, 2021. 3
- [39] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3D: Large-scale learning and evaluation of real-life 3D category reconstruction. In ICCV, 2021. 5
- [40] Erik Sandstr¨om, Ganlin Zhang, Keisuke Tateno, Michael Oechsle, Michael Niemeyer, Youmin Zhang, Manthan Patel, Luc Van Gool, Martin Oswald, and Federico Tombari. Splat-SLAM: Globally optimized rgb-only SLAM with 3D gaussians. In CVPRW, pages 1680–1691, 2025. 2
- [41] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, pages 4104–4113, 2016. 1
- [42] Sunando Sengupta, Eric Greveson, Ali Shahrokni, and Philip HS Torr. Urban 3D semantic modelling using stereo vision. In ICRA, pages 580–585. IEEE, 2013. 1
- [43] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in RGBD images. In CVPR, pages 2930–2937, 2013. 2, 5, 6, 7, 1, 4
- [44] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael

- Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. 5, 2
- [45] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A benchmark for the evaluation of RGB-D SLAM systems. In IROS, 2012. 2, 5, 6, 4
- [46] Christopher Sweeney, Tobias Hollerer, and Matthew Turk. Theia: A fast and scalable structure-from-motion library. In ACM MM, pages 693–696, 2015. 2
- [47] Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In ECCV, pages 402–419. Springer, 2020. 2
- [48] Zachary Teed and Jia Deng. DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras. NeurIPS, 34:16558–16569, 2021. 2, 5, 6
- [49] Bill Triggs, Philip F McLauchlan, Richard I Hartley, and Andrew W Fitzgibbon. Bundle adjustment—a modern synthesis. In International workshop on vision algorithms, pages 298–372. Springer, 1999. 1
- [50] Hengyi Wang and Lourdes Agapito. 3D reconstruction with spatial memory. In 3DV, 2025. 2, 6, 7
- [51] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025. 2, 6, 7
- [52] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3D perception model with persistent state. In CVPR, pages 10510– 10522, 2025. 2, 5, 6, 7, 8
- [53] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D vision made easy. In CVPR, pages 20697–20709, 2024. 2, 3, 4, 5, 7
- [54] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3D indoor scenes. In ICCV, 2023. 5
- [55] Jie Yin, Ang Li, Tao Li, Wenxian Yu, and Danping Zou. M2DGR: A multi-sensor and multi-scenario SLAM dataset for ground robots. IEEE Robotics and Automation Letters, 7

(2):2266–2273, 2021. 4

- [56] Ganlin Zhang, Viktor Larsson, and Daniel Barath. Revisiting rotation averaging: Uncertainties and robust losses. In CVPR, pages 17215–17224, 2023. 2
- [57] Ganlin Zhang, Erik Sandstr¨om, Youmin Zhang, Manthan Patel, Luc Van Gool, and Martin R Oswald. GlORIE-SLAM: Globally optimized RGB-only implicit encoding point cloud SLAM. arXiv preprint arXiv:2403.19549, 2024. 2, 5, 6
- [58] Wei Zhang, Tiecheng Sun, Sen Wang, Qing Cheng, and Norbert Haala. HI-SLAM: Monocular real-time dense mapping with hybrid implicit fields. IEEE Robotics and Automation Letters, 9(2):1548–1555, 2023.
- [59] Wei Zhang, Qing Cheng, David Skuddis, Niclas Zeller, Daniel Cremers, and Norbert Haala. HI-SLAM2: Geometryaware gaussian SLAM for fast monocular scene reconstruction. arXiv preprint arXiv:2411.17982, 2024.
- [60] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. GO-SLAM: Global optimization for consistent 3D instant reconstruction. In ICCV, pages 3727–3737, 2023. 2

[61] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. NICER-SLAM: Neural implicit scene encoding for RGB SLAM. In 3DV, pages 42–52. IEEE, 2024. 2, 5, 6

### ViSTA-SLAM: Visual SLAM with Symmetric Two-view Association Supplementary Material

Method Metric chess fire heads office pumpkin redkitchen stairs Avg

Accuracy 0.274 0.102 0.106 0.326 0.452 0.325 0.345 0.276 Completeness 0.303 0.081 0.093 0.441 0.459 0.358 0.389 0.303 Chamfer 0.289 0.091 0.100 0.384 0.456 0.342 0.367 0.290 ATE 0.743 0.226 0.363 0.664 0.546 0.381 0.413 0.477

CUT3R

Accuracy 0.043 0.022 0.020 0.035 0.072 0.062 0.116 0.053 Completeness 0.030 0.013 0.015 0.030 0.055 0.061 0.209 0.059 Chamfer 0.037 0.018 0.017 0.033 0.064 0.061 0.162 0.056 ATE 0.089 0.048 0.036 0.088 0.196 0.102 0.126 0.098

SLAM3R

Accuracy 0.090 0.037 0.027 0.047 0.097 0.070 0.045 0.059 Completeness 0.055 0.024 0.021 0.053 0.054 0.036 0.149 0.056 Chamfer 0.073 0.031 0.024 0.050 0.075 0.053 0.097 0.057 ATE 0.063 0.046 0.029 0.103 0.112 0.074 0.032 0.066

MASt3R-SLAM

Accuracy 0.029 0.014 0.031 0.041 0.128 0.036 0.087 0.052 Completeness 0.052 0.064 0.021 0.066 0.054 0.057 0.110 0.061 Chamfer 0.040 0.039 0.026 0.054 0.091 0.047 0.098 0.056 ATE 0.037 0.026 0.022 0.103 0.147 0.063 0.095 0.070

VGGT-SLAM

Accuracy 0.065 0.015 0.031 0.036 0.061 0.035 0.074 0.045 Completeness 0.063 0.022 0.040 0.048 0.037 0.030 0.154 0.056 Chamfer 0.064 0.019 0.035 0.042 0.049 0.033 0.114 0.051 ATE 0.073 0.035 0.028 0.055 0.129 0.035 0.029 0.055

Ours

- Table 7. Per Scene Evaluation on 7-Scenes [43]. Comparison of accuracy, completeness, Chamfer distance, and trajectory error on the 7-Scenes dataset. Lower is better. Best results are bold, second best are underlined.

#### 5. Relative Scale in Pose Graph

As mentioned in Sec. 2.3, the scale edges connect nodes corresponding to the same view but obtained from different forwarding passes. Since the training supervision of the frontend STA model uses only normalized pointmaps, the scales of the same view across different passes are not consistent. Therefore, estimating the relative scale is crucial for pose graph construction. Given two pointmaps Pij and Pik of the same view i (obtained from forwarding passes with input view i j and view i k, respectively), along with their confidence maps Cij and Cik, we first get the confidence score wx of the point pair for pixel x,

wx =Cij(x) · Cik(x),

then, the relative scale sjki can be computed as,

sjki =min

wx∥Pij(x) − sPik(x)∥2

s

x

wx(Pij(x) · Pik(x))

. (11)

= x

x wx∥Pik(x)∥2

#### 6. Additional Quantitative Results

###### 6.1. Per Scene Evaluation Results on 7-Scenes

In Sec. 3.2, only the average reconstruction evaluation is provided. In Tab. 7, we present more detailed per-scene results on 7-Scenes [43] to offer deeper insights. The pure regression method SLAM3R [27] performs well in scenes where the camera primarily focuses on a single corner,

such as chess, fire, and heads. However, in scenes involving longer camera trajectories, like pumpkin and redkitchen, its performance degrades due to difficulties in accurately registering points. Our method, ViSTASLAM, achieves the best performance in average across all four metrics.

###### 6.2. Additional Trajectory Results on TUM-RGBD

In Sec. 3.1, to align with previous methods, only results for the freiburg1 partition of the TUM RGB-D dataset [45] are reported. In Tab. 8, we also report results for the freiburg2 and freiburg3 partitions.

Sequence ATE RMSE (m)

freiburg2 360 hemisphere 0.2037 freiburg2 360 kidnap 0.4617

- freiburg2 desk 0.0577

- freiburg2 large with loop 0.2170

- freiburg2 rpy 0.0222

- freiburg2 xyz 0.0155

- freiburg3 cabinet 0.3869

- freiburg3 large cabinet 0.1334

- freiburg3 long office household 0.1013

- freiburg3 teddy 0.0789

Table 8. Trajectory ATE results on the freiburg2 and freiburg3 partitions of the TUM RGB-D dataset [45].

- 6.3. Additional Evaluation on More Datasets

In Tab. 9 and Tab. 10, we additionally report camera trajectory evaluation results on Replica [44] and ScanNet [8], as well as reconstruction evaluation results on Replica for several commonly used SLAM testing scenes.

- 7. Additional Qualitative Results

###### 7.1. Pose Graph Optimization

In Fig. 7, we compare the reconstruction and trajectory estimation results with and without pose graph optimization on ScanNet [8] scene0000 00. Pose graph optimization effectively corrects misaligned areas and averages out the errors from the frontend.

###### 7.2. Wrong Loop Filtering

In Sec. 2.3, we describe feeding each loop candidate pair into our STA model to verify their spatial proximity. This is necessary because Bag of Words loop detection can produce false positives, which may significantly degrade performance by introducing misleading edges into the pose graph. As shown in Fig. 8, rejecting incorrect loop candidates using the relative pose confidence score provided by STA results in a much more stable performance.

###### Scene ATE Acc. Comp. Chamfer

- office0 0.0744 0.0595 0.0226 0.0410
- office1 0.1934 0.2614 0.1833 0.2223
- office2 0.1177 0.0914 0.0316 0.0615
- office3 0.0485 0.0623 0.0221 0.0422
- office4 0.1302 0.1338 0.0688 0.1013 room0 0.0688 0.0766 0.0209 0.0488 room1 0.0934 0.1105 0.0552 0.0828 room2 0.1363 0.1194 0.0223 0.0709 Table 9. Per-scene evaluation results on Replica [44].

###### Sequence ATE RMSE (m)

scene0000 00 0.0483 scene0059 00 0.0391 scene0106 00 0.0559 scene0169 00 0.0526 scene0181 00 0.0520 scene0207 00 0.0479

Table 10. Per-scene camera trajectory evaluation results on ScanNet [8].

###### 7.3. More Results

In Fig. 9, we present the results of ViSTA-SLAM across various datasets. ViSTA-SLAM demonstrates stable performance despite differing camera motions in these scenes. As before, light blue frustums represent camera poses, blue lines connect neighboring views, while orange lines indicate loop closures.

[Figure 105]

[Figure 106]

result w/o PGO result w/ PGO

Figure 7. Qualitative Comparison for Pose Graph Optimization. Red boxes highlight regions with misalignments, while green boxes indicate areas where these misalignments have been corrected after pose graph optimization.

[Figure 107]

Figure 8. Qualitative Comparison of Wrong Loop Filtering. Keeping wrong loop candidates decreases the performance a lot.

BundleFusion [9] apt2 TUM-RGBD [45] floor

[Figure 108]

[Figure 109]

BundleFusion [9] apt0 7-Scenes [43] pumpkin

[Figure 110]

[Figure 111]

M2DGR [55] room 03 M2DGR [55] room 01

[Figure 112]

[Figure 113]

Figure 9. More Qualitative Results. Reconstructions and camera trajectories from different datasets.

