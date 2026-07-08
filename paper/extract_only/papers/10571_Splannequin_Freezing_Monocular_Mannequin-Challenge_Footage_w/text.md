# arXiv:2512.05113v2[cs.CV]8Dec2025

## Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting

Hao-Jen Chien1 Yi-Chuan Huang1 Chung-Ho Wu1 Wei-Lun Chao2 Yu-Lun Liu1 1National Yang Ming Chiao Tung University 2The Ohio State University

[Figure 1]

Figure 1. Splannequin converts imperfect Mannequin-Challenge footage into a true freeze-time video. (Top) A monocular Mannequin-Challenge video is intended to resemble large-scale frozen frames, yet real-world recordings inevitably contain slight body movements. The red crops across successive frames with the corresponding camera pose (ci) and timestamp (ti) highlight these noticeable movements. (Bottom) After our processing, every crop (green boxes) of successive frames remains static. Splannequin analyzes the entire video and resynthesizes a temporally consistent sequence of the same camera poses at t⋆ while preserving overall visual fidelity.

### Abstract

Synthesizing high-fidelity frozen 3D scenes from monocular Mannequin-Challenge (MC) videos is a unique problem distinct from standard dynamic scene reconstruction. Instead of focusing on modeling motion, our goal is to create a frozen scene while strategically preserving subtle dynamics to enable user-controlled instant selection. To achieve this, we introduce a novel application of dynamic Gaussian splatting: the scene is modeled dynamically, which retains nearby temporal variation, and a static scene is rendered by fixing the model’s time parameter. However, under this usage, monocular capture with sparse temporal supervision introduces artifacts like ghosting and blur for Gaussians that become unobserved or occluded at

weakly supervised timestamps. We propose Splannequin, an architecture-agnostic regularization that detects two states of Gaussian primitives, hidden and defective, and applies temporal anchoring. Under predominantly forward camera motion, hidden states are anchored to their recent wellobserved past states, while defective states are anchored to future states with stronger supervision. Our method integrates into existing dynamic Gaussian pipelines via simple loss terms, requires no architectural changes, and adds zero inference overhead. This results in markedly improved visual quality, enabling high-fidelity, user-selectable frozen-time renderings, validated by a 96% user preference. Project page: https://chien90190.github.io/ splannequin/

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

### 1. Introduction

| |
|---|

| |
|---|

Freeze-time videos, also known as Mannequin Challenge clips, allow cameras to move freely through scenes while subjects appear perfectly frozen mid-action. This can: (1) generate pseudo-ground-truth for training dynamic scene models with cleaner signals [39]; and (2) enable artistic control for creators in visual effects (VFX) production to select precise frozen moments. As shown in Figure 1, we synthesize freeze-time sequences from casual monocular recordings with slight, unintended motion.

###### Input frames

[Figure 11]

SC-GS [Huang et al. 2024]

4DGaussians [Wu et al. 2024b]

| |
|---|

| |
|---|

D-3DGS [Yang et al. 2024a]

Figure 2. Existing Gaussian splatting frameworks cannot produce plausible results from casually captured MannequinChallenge videos. (Top) A short clip of the hand-held input sequence exhibits unintentional subject motion. (Right) State-ofthe-art methods containing SC-GS [23], 4DGaussians [90], and D-3DGS [97]. They all leave noticeable blur and double contours around the woman’s face (blue frames). (Left) Our Splannequin reconstruction (red frame) is crisp and temporally consistent, revealing fine hair strands and facial detail with no ghosting.

Traditional production with multi-camera bullet-time rigs [76] is costly and complex, requiring heavy postproduction with single shots costing over $750k. While the single-camera Mannequin-Challenge is a low-cost alternative, its monocular 3D reconstruction is prone to artifacts from minor subject motion and sparse observational data.

Recent advances in dynamic Gaussian splatting [23, 26, 40, 87, 90, 97, 98] and monocular dynamic reconstruction [42] model spatio-temporal scenes. In principle, they enable freeze-time synthesis by fixing the temporal coordinate. In practice, frozen renders often show ghosting and blur (Figure 2). This is common under predominantly forward camera motion as the camera moves away from objects. Existing objectives and benchmarks aim to preserve motion, keeping subjects centered and well supervised in action-focused clips. This favors motion fidelity. In contrast, imperfect Mannequin-Challenge videos are dynamic scene exploration and we focus on artifacts to recover the underlying frozen scene, so a new test set is needed.

are the first to formally address synthesizing high-fidelity, freeze-time videos from monocular MC footage, providing a new benchmark and evaluation protocol.

- • A Targeted Regularization Framework. We propose a novel method to identify and regularize hidden and defective Gaussians, the primary sources of temporal artifacts, and anchor them to reliable past or future states.
- • State-of-the-art performance with zero inference overhead. We improve visual quality and stability in existing methods without architectural changes. As the deformation runs only once for a target instant, we achieve inference speeds exceeding 280 FPS on an RTX 4090.

These visual artifacts stem from inconsistent temporal observations. In monocular sequences, Gaussians lack reliable data when they are occluded or move outside the camera’s view. When rendering a single frozen instant, inference for these unconstrained Gaussians creates ghosting and blur, corrupting the final reconstruction [35].

### 2. Related Work

The MANNEQUIN-CHALLENGE dataset popularized ”frozen-people” clips for depth learning [39], establishing the foundational benchmark for synthesizing freeze-time effects from casual captures. Modern dynamic human renderers such as HumanNeRF [89], HumanNeRF-SE [53], FlexNeRF [25], FloRen [68], and Holoported-Characters [69] reproduce articulated motion rather than suppress it. Scene motion-aware IBR methods like DynIBaR [41] and the recent BTimer feed-forward bullet-time pipeline [43, 76] generalize to arbitrary camera paths but retain micro-motions, with BTimer being the first to use 3D Gaussian Splatting for motion-aware bullet-time synthesis. Shape of Motion [86] addresses fast motion and occlusions in monocular 4D reconstruction relevant to imperfect Mannequin Challenge scenarios. Diffusion-based pipelines (4Real [100], CAT4D [92], Deblur-Avatar [52], GAUSSIANFlow [18], [8]) hallucinate photorealistic 4D geometry yet model sub-frame dynamics. We detect and regularize poorly supervised Gaussians[21] to erase erroneous placements and appearances while preserving

We propose Splannequin, a dual-detection regularization method that stabilizes dynamic Gaussians for freeze-time rendering. It identifies problematic Gaussians as hidden or defective, anchoring hidden ones to their well-observed past states and defective ones to future states with stronger supervision. Implemented as simple loss terms without architectural changes, Splannequin integrates into existing dynamic Gaussian pipelines with minimal overhead. This enables cleaner static views from single-camera captures and improves tolerance to small motions for consumer media and VR/AR tours.

We validate Splannequin on a new benchmark of 10 realworld Mannequin Challenge videos, demonstrating strong gains over state-of-the-art baselines. Our method improves compositional quality assessment (CQA) by up to 243.8% and technical quality (COVER) by 339.85% when applied to D-3DGS [97]. Our contributions are:

• A Novel Problem Formulation and Benchmark. We

rendering quality.

NeRFs and Gaussian Splatting for Static Novel-View Synthesis. Neural Radiance Fields (NeRF)[5] model radiance using MLPs[1, 2, 56], while hash-grid encodings reduce training time [57]. 3D Gaussian Splatting achieves real-time rendering [28], with recent developments surveyed in [94]. Extensions remove SfM preprocessing (COLMAP-Free 3DGS [17]), refine camera poses (Look Gauss, No Pose [67], LongSplat [45], and joint poseradiance optimization [9]), and incorporate directional information (6DGS [19]). To handle large-scale casual captures, progressive optimization significantly improves robustness [54]. Recent artifact removal methods include EFA-GS [83] which eliminates floating artifacts through frequency-domain analysis, VRSplat [79] which addresses temporal popping during viewpoint changes, and 3DGSHD [77] which removes unrealistic artifacts. Sparseview robustness is enhanced by depth priors [32, 38], multi-view priors (BoostMVSNeRFs [74], [85, 96]), coregularization[104], neural field integration [55], structural dropout [59], and few-shot convergence strategies without learned priors (FrugalNeRF [46]). Techniques like BOGausS prune redundant splats [62], while 2DGS uses oriented disks for improved fidelity [22]. Because these static pipelines assume fully rigid scenes, minor dynamics cause issues, making new solutions necessary.

Dynamic NeRF and Gaussian Representations. Early dynamic NeRFs embed time as an additional coordinate or latent code: D-NeRF models a deformation field [63], Nerfies incorporates a deformation MLP [60], HyperNeRF lifts the canonical space [61], and NR-NeRF targets non-rigid objects [78]. HexPlane provides foundational six-plane decomposition for 4D spacetime [7]. To scale to longer sequences, TiNeuVox employs a time-aware voxel grid [15], NeRFPlayer streams sub-fields [73], and RoDynRF refines poses with static and dynamic fields [51]. On the explicit side, 4D Gaussian Splatting enables real-time rendering of deformations [14, 90], with recent advances including uncertainty-aware regularization for handling poorly supervised regions [30], spatial-temporal consistency through motion-aware regularization [36], deblurring from blurry monocular videos [93], and depth-prior integration for casual captures (MoDGS [47]). MotionGS explores explicit motion guidance [106], while Grid4D uses 4D decomposed hash encoding [95]. Efficiency improvements include GaussianVideo [6], GIFStream [37], VeGaS [72], DynMF’s compact motion basis [31] and Animatable 3D Gaussians at 60 fps [99]. Recent benchmarking [44] reveals the brittleness of monocular dynamic reconstruction. Crucially, all aim to preserve observed motion. In contrast, our nearstatic setting addresses ghosting [84] by optimizing Gaussians for improved rigid scene and novel-view reconstruction.

Video Stabilization and Temporal Coherence. 2D stabilizers smooth footage via mesh warps. Bundled Camera Paths [64] optimize homographies, SteadyFlow enforces smooth optical flow [48, 49], and gyroscope filtering corrects rolling-shutter distortion [3]. Learning-based methods re-render frames or predict warp fields [50, 101]. Causal variants reduce latency [70, 105], while depth-aware schemes reconstruct geometry [34]. Recent work addresses spatiotemporally inconsistent observations through residual compensation [102] and exposure completion for temporal consistency [12]. LeanVAE provides efficient reconstruction for temporal coherence [11]. A recent survey highlights challenges like border completion [65]. These output warped 2D frames. Our pipeline unifies 3D stabilization and novel-view synthesis [4, 66] via Gaussian splatting, removing Gaussian artifacts [103] while enabling freeviewpoint playback [107].

Micro-Motion Detection and Repair. Eulerian video magnification reveals imperceptible shifts by amplifying per-pixel traces [91], with phase-based variants improving robustness [80]. Methods handle large displacements through layer-wise alignment [13], learned filters [58], and Transformer-based denoising [81]. All remain 2D and amplify motion. In 3D, Feng et al. model subtle dynamics via time-varying radiance fields [16], while early geometryaware editing hinted at sub-pixel motion control [33]. Recent work handles unobserved views through refinement and fusion [71], and applies static restoration priors for dynamic regularization [24]. Recent reconstructions capture fine dynamics without suppression [86]. We operate within the Gaussian representation: a confidence-weighted classifier identifies and improves hidden and defective primitives, removing sub-pixel jitter and enabling robust freeze-time synthesis in real-world, near-static scenes.

### 3. Background

Dynamic Gaussian splatting extends 3DGS to model timevarying scenes. Given a set of N images, camera rotations and translations, and timestamps, {(In,Rn,bn,tn)}Nn=1, the scene is represented by a set of canonical 3D Gaussians {Gk}Kk=1. Each Gaussian Gk is defined by a static mean µk and covariance Σk. A deformation network fθ, typically an MLP, predicts the temporal evolution of each Gaussian:

###### (∆µk,t,∆Σk,t) = fθ(µk,t). (1)

The time-dependent state of the Gaussian is then Gk(t) = (µk + ∆µk,t,Σk + ∆Σk,t). The model is optimized by minimizing a photometric loss between the rendered image Iˆn and the ground truth image In for each observation:

Lrecon =

N

ℓ(Iˆ(Rn,bn,tn),In). (2)

n=1

[Figure 12]

###### Camera Trajectory / Frame Time

[Figure 13]

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

[Figure 14]

- 0

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

- 1.0

Defective Area

[Figure 15]

Rendering(t*)

[Figure 16]

Time

Supervised

[Figure 17]

Hidden Area

[Figure 18]

Figure 4. Illustration of hidden Gaussians. Given a timestamp, hidden Gaussians (Left) lie outside the camera frustum, receiving no supervision, while visible Gaussians (Right) are rasterized to form the image. Our method targets ill-supervised hidden Gaussians to prevent visual artifacts.

Reconstruction Rendering

Figure 3. Time-Camera Conceptualization. Assuming forward camera motion, the diagonal dashed line represents standard dynamic rendering, while the horizontal line shows freeze-time rendering at a fixed timestamp t⋆. Along this freeze-time line, unsupervised Gaussians are either hidden (red points, as the camera has passed them) or defective (blue points, not yet well-observed). Our approach regularizes these problematic Gaussians by anchoring them to their supervised counterparts from other timestamps: hidden (red) Gaussians use past states, and defective (blue) Gaussians use future states. The right panel shows a bird’s-eye view of a hallway, illustrating how the camera’s path creates defective and hidden regions.

- 4.1. Identification of Ill-supervised Gaussians In dynamic Gaussian splatting, primitives only receive supervision from the reconstruction loss when they contribute to the rendered image. We formally identify two failure cases of missing supervision for a Gaussian k at time t. Our regularization strategy identifies these cases and applies targeted temporal anchoring (Figure 4).

- • Hidden Gaussian: A primitive k at time t is hidden if the primitive center projects outside the camera frustum, typically after the camera has moved past it. Such primitives are unobservable and receive no supervision, defined as shidden(k,t) = {1, visibility is 0; 0, otherwise}.
- • Defective Gaussian: A primitive k at time t is defective if its center is within the camera frustum while the rendered contribution is negligible, resulting in a zero-gradient up-

date, defined as sdefective(k,t) = {1, visibility is 1 with gradient ≤ 1e-9; 0, otherwise}. This often happens for distant, highly transparent, or occluded primitives.

A primitive k at time t is therefore well-supervised only if shidden(k,t) = 0 and sdefective(k,t) = 0. In practice, we get visibility from the differentiable rasterizer outputs [29].

- 5. Approach

This framework enables high-fidelity modeling of dynamic scenes. However, as we will detail, its reliance on direct supervision at each timestamp makes it prone to artifacts when applied to our specific problem.

### 4. Problem Definition

We address the problem of frozen scene reconstruction from a monocular Mannequin Challenge (MC) style video. The goal is to synthesize a high-fidelity static rendering Iˆ(R,b,t⋆) from any input camera view (R,b) at a single, user-selected timestamp t⋆, effectively freezing all motion.

The dynamic framework is a natural fit, as rendering a view of an instant seems as simple as fixing the time parameter to t⋆. However, this fails for monocular MC-style videos. The reconstruction loss, Lrecon, only provides a training signal for a Gaussian Gk at timestamps tn where it is visible and receives gradients in the image In. The Gaussian is also unobserved at time tn (due to occlusion or leaving the camera frustum), and its deformation fθ(µk,tn) receives no gradient. This sparse supervision leads to unstable state estimates at unobserved times.

To recover a frozen scene from an MC-style video, we address the parameter drift of ill-supervised Gaussians. Splannequin is a regularization framework that identifies and stabilizes these Gaussians (Figure 5). We classify each ill-supervised Gaussian at a given timestamp as either hidden or defective. We then apply a targeted temporal consistency loss that anchors the parameters of each illsupervised Gaussian to a nearby, well-supervised reference state, weighted by temporal distance to respect genuine motion while suppressing artifacts.

To analyze this supervision gap, we use a space-time diagram (Figure 3) that plots time against the camera’s spatial path. The video, or the standard dynamic reconstruction, is then captured along a diagonal trajectory, but the desired freeze-time render lies along a horizontal line at t⋆. This visualization reveals that artifacts arise directly from rendering primitives at timestamps when they were ill-supervised.

#### 5.1. Temporally-Anchored Regularization

For each ill-supervised Gaussian, we enforce temporal consistency by regularizing its parameters θk(t) toward a stable reference state, including position µk(t), covariance Σk(t),

A monocular video with accidental movements

θhidden(t),t‘

t’

Hidden Loss

timestamps

θhidden(t),t

t

1

2

Let θhidden(t),t = θhidden(t),t’ with point-wise distance confidence. t’< t

x,y,z

Dynamic Gaussian Splatting Model

θdefective(t),t

t

Point cloud Initialization

Defective Loss

3

θdefective(t),t’

t’

Render at t* for all Gaussians

Let θdefective(t),t = θdefective(t),t’ with point-wise distance confidence. t< t’

A freeze-time video

- Figure 5. Splannequin Pipeline Overview. The pipeline: (1) extracts point clouds from input video, (2) use dynamic Gaussian splatting with dual-detection losses that anchor hidden Gaussians to earlier frames (t’ < t) and defective Gaussians to later frames (t < t’), and (3) renders freeze-time videos at any timestamp t⋆. Temporal distance-based confidence weighting ensures appropriate regularization strength, with closer reference frames providing stronger anchoring than distant ones for robust temporal consistency and artifact elimination.

##### 5.1.3. Total Objective Function

opacity αk(t), and spherical harmonic coefficients c. During each training iteration, for an ill-supervised Gaussian k at time t, we randomly select a timestamp tref, described in the next paragraphs, from the set of all timestamps and check if it is a well-supervised state. This avoids creating explicit anchor pools or performing expensive searches.

The final training objective combines the standard reconstruction loss with our regularization terms, summed over all instances where a valid anchor was found:

L = Lrecon + λhidden Lhidden(k,t) (5) + λdefective Ldefective(k,t). (6)

##### 5.1.1. Anchoring Hidden Gaussians

If a Gaussian k at time t is hidden (shidden(k,t) = 1), its state should be ideally constrained by its recent appearance. In practice, a randomly sampled reference time tref is a valid anchor if it is in the past (tref < t) and the Gaussian k is well-supervised at tref. When these conditions are met, we apply the consistency loss defined as:

where Lhidden and Ldefective are instances of Lconsistency, and λhidden and λdefective are weighting hyperparameters.

#### 5.2. Fixed-Time Rendering

After training, our primary application is fixed-time rendering. A freeze-time video can be synthesized by rendering all N training poses {(Rn,bn)}Nn=1 at a single, user-selected timestamp t⋆. Because our method preserves temporal variation, it grants users the flexibility to scrub through time, select the exact desired moment, and generate a perfectly static video.

Lconsistency(k,t) = ϕ(t,tref) · D(θk(t),θk(tref)), (3) where the discrepancy measure D is the L1 or L2 norm:

D(θA,θB) = ∥θA − θB∥1, for L1 norm ∥θA − θB∥22, for L2 norm

(4)

### 6. Experiments

The term ϕ(t,tref) is weighting. For well-supervised anchors, we conservatively down-weight their influence using an exponential decay based on temporal distance, ϕ = e−τ|t−t

#### 6.1. Implementation Details

We implemented in PyTorch, extending the official dynamic Gaussian splatting framework, with all experiments run on a single NVIDIA RTX 4090 GPU. We follow standard adaptive densification practices and train each model for 30,000 iterations. The regularization is introduced in stages to allow the base geometry to stabilize. Our main losses, Lhidden and Ldefective, begin at iteration 10,000, initially using an L2 norm for the discrepancy measure D and switching to an L1 norm at iteration 20,000. We use loss weights λhidden = λdefective = 10 and a confidence decay factor of

ref|.

- 5.1.2. Anchoring Defective Gaussians Conversely, if a Gaussian k at time t is defective

(sdefective(k,t) = 1), it typically lacks supervision because the camera has not yet reached it. In practice, a randomly sampled reference time tref is a valid anchor only if it is in the future (tref > t) and the Gaussian k is well-supervised at tref. If so, we apply the same consistency loss.

Baseline Baseline + Splannequin Baseline Baseline + Splannequin

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

4dGaussiansSC-GSD-3DGS

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 6. Qualitative Comparison across Our Real-World Benchmark. Each column shows freeze-time renderings from all methods at a viewpoint. Rows correspond to direct comparisons of identical viewpoints with baselines: 4DGaussians (top), D-3DGS (middle), and SC-GS (bottom). Adding Splannequin consistently produces sharper, more temporally coherent results, exhibiting reduced ghosting and artifact suppression compared to baseline methods.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

| |
|---|

[Figure 40]

Camera Trajectory

t* =0t* =80

- Figure 7. User-Selectable Freeze-Time Instants. Splannequin empowers users to select the precise moment to freeze, allowing for artistic control over the final scene. Both rows show high-fidelity freeze-time videos generated from the same input sequence but frozen at two different, user-selected timestamps. Top: At Timestamp 0, the subject in the inset is looking down. Bottom: At Timestamp 80, captured seconds later, the subject has turned their head. Our method successfully reconstructs both moments with sharp detail and stability, preserving these subtle differences and enabling creative selection based on pose and expression.

τ = 5. These regularization losses are computed every 10 iterations by sampling two random view-timestamp pairs and applying the anchoring logic only when valid supervised anchor states are available for the target primitives.

#### 6.2. Experimental Setup

To ensure comprehension and robustness, we test across multiple timestamps for each video. After training a model on a full sequence, we render multiple freeze-time videos by using timestamps from every 8th frame of the input video as the target freeze-time t⋆. This process yields a diverse

set of “frozen” clips per scene for a holistic assessment of temporal consistency and artifact reduction across the entire duration of the original capture.

Benchmark. Our evaluation uses a challenging custom dataset of 10 640×360 Mannequin Challenge-style videos (2,869 frames, 361 fixed-time renderings) from the public Google Mannequin Challenge collection [39]. Spanning diverse scenes, including seven indoor scenes and three outdoor fields, with subjects exhibiting natural, unscripted micro-motions, the dataset is characterized by sparse tem-

[Figure 41]

[Figure 42]

- Table 1. Validation on our synthetic dataset. We report absolute scores for reference metrics and percentage improvements for nonreferenced quality assessment metrics.

Referenced Metrics Method PSNR↑ SSIM↑ LPIPS↓ FVD↓

4DGaussians 28.03 0.81 0.09 98.93 4DGaussians + Splannequin 28.85 0.83 0.08 82.73

Average Image Quality Assessment (IQA) Metric Improvements CQA ↑ TOPIQ-NR ↑ CLIP-IQA ↑ MUSIQ ↑ HyperIQA ↑ 26.43% 2.08% 1.18% 1.74% 2.13%

Video Quality Assessment (VQA) Metric Improvements COVER ↑

Semantic Technical Aesthetic Overall 1.29% 3.11% 95.60% 6.56%

- Table 2. Quantitative comparison on our real-world dataset. The values represent the percentage improvement Splannequin provides when added to each baseline method (higher is better). Our method consistently enhances all baselines, with the most gains in technical artifact suppression (COVER Technical) and on the lowest-quality frames (IQA Bottom 25%). Methods are abbreviated as: (1) 4DGaussians+, (2) D-3DGS+, and (3) SC-GS+. W.F. is the worst frame.

[Figure 43]

[Figure 44]

| |
|---|

| |
|---|

[Figure 45]

[Figure 46]

| |
|---|

| |
|---|

[Figure 47]

[Figure 48]

4DGaussians 4DGaussians + Splannequin

- Figure 8. Validation on Simulated Dataset. Qualitative comparison between 4DGaussians (left) and 4DGaussians + Splannequin (right) on synthetic scenes with ground truth. With Splannequin, the results better preserve geometric details and suppressed artifacts, as validated against static reference frames. Insets highlight regions of improved structural fidelity.

poral supervision. On average, the benchmark has less than 10% consistent visibility across the full videos.

Metrics. Since the ground-truth frames contain the very motions we aim to suppress, we primarily rely on a suite of no-reference (NR) perceptual quality metrics that correlate well with human judgments, rather than full-reference metrics. In all our tables, improvements are reported as relative gains computed as (x - baseline) / baseline × 100.

Video Quality Assessment (VQA) Metric Improvement Metric / Method Semantic Technical Aesthetic Overall

- • Composition (CQA): Adapted from View Evaluation Net (VEN) [88] to assess compositional clarity by comparing rendering quality at identical viewpoints.
- • TOPIQ-NR [10]: A unified model leveraging multiple feature types to provide a general-purpose quality score.
- • CLIP-IQA [82]: A metric assessing quality by measuring the similarity between rendered image features and textual quality descriptors in CLIP.
- • MUSIQ [27]: A multi-scale, Transformer-based model evaluating both fine-grained detail and global quality.
- • HyperIQA [75]: An adaptive model generating contentspecific weights to assess quality across diverse scenes.
- • COVER [20]: A holistic video quality assessor that evaluates semantic, technical, and aesthetic dimensions.

- (1) 2.23% 73.03% 20.53% 68.25%
- (2) 2.97% 339.85% 75.24% 183.68%
- (3) 1.84% 81.53% 30.46% 121.70%

COVER Image Quality Assessment (IQA) Metric Improvement Metric / Method Average

Bottom Percentage

W.F. 75% 50% 25%

- (1) 121.33% 60.97% 51.54% 101.44% 18.48%
- (2) 243.80% 65.94% 103.11% 98.64% 17.39%
- (3) 48.88% 38.45% 404.08% 36.31% 26.11%

CQA

- (1) 2.53% 3.61% 4.98% 6.76% 13.65%
- (2) 7.10% 9.78% 13.40% 17.82% 28.35%
- (3) 8.26% 11.87% 16.20% 22.62% 47.81%

TOPIQ-NR

- (1) 2.42% 3.70% 5.27% 7.44% 15.99%
- (2) 6.96% 8.99% 11.51% 14.46% 24.21%
- (3) 8.72% 12.61% 16.37% 20.19% 34.43%

CLIP-IQA

- (1) 1.29% 1.82% 2.49% 3.51% 8.75%
- (2) 6.62% 9.23% 13.10% 18.60% 30.45%
- (3) 6.43% 9.22% 12.92% 18.50% 35.75%

MUSIQ

Compared Methods. We integrated Splannequin with three state-of-the-art dynamic Gaussian Splatting methods: D-3DGS [97], SC-GS [23], and 4DGaussians [90]. To ensure a fair comparison, all baselines were trained under identical conditions using their default hyperparameters. We excluded other recent methods from our primary comparison for specific reasons: MoDGS [47] relies on precomputed depth maps that conflate camera and object motion, and STG [40] requires a multi-camera setup.

- (1) 4.60% 5.79% 7.23% 9.01% 12.12%
- (2) 7.14% 9.53% 12.04% 15.60% 21.33%
- (3) 5.90% 8.30% 10.88% 14.82% 30.50%

HyperIQA

plying Splannequin to the 4DGaussians yielded consistent improvements across all reference metrics. These results verify that Splannequin successfully regularizes artifacts for reconstructions measurably closer to the ideal static geometry while preserving motions, providing a strong foundation for our primary evaluation on real-world data.

#### 6.3. Validation on Simulated Benchmark

To validate our approach in a controlled environment, we first tested it on a synthetic dataset of 10 Blendergenerated scenes (2,400 frames, 300 fixed-time renderings) with known static ground truth, allowing for referencebased evaluation. As shown in Table 1 and Figure 8, ap-

#### 6.4. Real-World Evaluation

As an architecture-agnostic framework, Splannequin integrates into existing pipelines with substantial improvements (Figure 6). Using temporal anchoring (Section 4),

Table 3. Ablation study. We report the percentage degradation in performance when removing the hidden loss and the defective loss individually. The results show that both components are critical.

COVER (VQA Metrics) Average (IQA Metrics) Semantic Technical Aesthetic Overall CQA TOPIQ-NR CLIP-IQA MUSIQ HyperIQA

Method

No Hidden Loss -2.27% -94.22% -42.45% -1072.41% -162.23% -3.02% -2.65% -2.33% -4.31% No Defective Loss -4.79% -197.87% -56.25% -1027.33% -779.13% -12.52% -4.05% -13.32% -10.21%

[Figure 49]

[Figure 50]

- Figure 9. Effect of Confidence Weighting. Visualization comparing results with (right) and without (left) confidence weighting. Without confidence, regularization can over-smooth the frame.

it achieves large gains over baselines (Table 2) while maintaining high efficiency at over 280 FPS, as deformation only needs to be run once for a timestamp. Interestingly, improvement patterns reflect anchor availability: frames with the worst CQA scores suffer from compositional constraints tied to viewpoint rather than temporal supervision, yielding smaller relative gains than the overall average. In contrast, frames with the worst perceptual scores, which directly reflect artifacts, benefit more from temporal anchoring.

Although 4DGaussians + Splannequin (4DGaussians+) achieves performance comparable to static 3DGS reconstruction (Table 4), our approach uniquely enables motion preservation and user-controlled timestamp selection that 3DGS cannot provide. This allows users to select the precise instant to freeze and preserve subtle artistic details like a subject’s pose or expression, as shown in Figure 7.

#### 6.5. Ablation Studies

Effect of Regularization Losses. In Table 3, removing either component results in a performance drop as they play distinct and complementary roles. Removing the hidden loss indicates that anchoring past, out-of-view primitives maintains overall scene stability. In contrast, removing the defective loss suggests that anchoring future, not-yet-clear primitives reconstructs the initial geometry and composition of the scene. Together, these results confirm that both regularization terms are critical.

Effect of Confidence Weighting. Figure 9 shows that the confidence weighting plays a role in preserving overall fidelity, whereas the unweighted version can over-regularize and blur frames.

User Study. The ultimate goal of this work is to create a convincing “freeze-time illusion.” We conducted a user study where 23 participants judged videos on visual quality and perceived stillness. Our method was strongly pre-

[Figure 51]

[Figure 52]

Figure 10. User study results. Our method was preferred in 96% of comparisons for better visual appeal and fewer artifacts. 80% of our results were perceived as more ”perfectly frozen” than the original captures, validating our approach.

Table 4. Comparison to 3DGS. 4DGaussians+ is comparable to 3DGS when treating scenes as static. However, 3DGS cannot preserve motions or allow per-timestamp video evaluation.

Bottom Percentage

Metric Average

W.F. 75% 50% 25%

CQA 23.18% 46.31% 27.80% 22.09% 4.77% TOPIQ-NR -0.11% 0.77% 1.89% 3.80% 20.68% CLIP-IQA -2.17% -0.33% 1.73% 3.85% 16.95% MUSIQ -0.59% 0.10% 0.89% 2.64% 11.38% HyperIQA 2.95% 3.89% 5.33% 7.46% 14.14%

ferred, with 96% preference for fewer artifacts and better visual appeal (Figure 10). Crucially, in 80% of our generated results, participants reported a more ”perfectly frozen” effect compared to the original captures. This validates that our approach successfully captures motions and eliminates artifacts to deliver on its target application.

### 7. Conclusion

We present Splannequin, a novel strategy for synthesizing high-fidelity frozen scenes from monocular MannequinChallenge footage. Its architecture-agnostic regularization uses temporal anchoring to identify and stabilize Gaussians, eliminating artifacts. Evaluations show Splannequin delivers superior perceptual quality with real-time rendering. This work makes high-fidelity, user-selectable freeze-time reconstruction accessible from consumer video captures.

Limitations. Our method assumes nearly static scenes and fails under rapid, non-rigid changes. Fast-moving shadows, illumination shifts, or large motions lack reliable temporal anchors, causing artifacts. Quantitative analysis of motion thresholds and frame-position dependence remains for future work, along with more adaptive anchoring strategies for challenging cases.

Acknowledgements. This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-EA49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

- [1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855–5864,

2021. 3

- [2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 3
- [3] Steven Bell, Alejandro Troccoli, and Kari Pulli. A nonlinear filter for gyroscope-based video stabilization. In European Conference on Computer Vision, pages 294–308. Springer, 2014. 3
- [4] Alexander Berian and Abhijit Mahalanobis. Modern novel view synthesis: From neural fields to diffusion models. SPIE Defense+Commercial Sensing, 13459:33, 2025. Fewshot view synthesis survey. 3
- [5] Sandika Biswas, Qianyi Wu, Biplab Banerjee, and Hamid Rezatofighi. Template-free nerf for dynamic scene reconstruction. In NeurIPS, 2024. Deformable entity interactions. 3
- [6] Andrew Bond, Jui-Hsien Wang, Long Mai, Erkut Erdem, and Aykut Erdem. Gaussianvideo: Efficient video representation via hierarchical gaussian splatting. arXiv preprint arXiv:2501.04782, 2025. 3
- [7] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130–141, 2023. 3
- [8] Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. Generative novel view synthesis with 3d-aware diffusion models. In ICCV, pages 14386–14396, 2023. Diffusionbased view synthesis from single images. 2
- [9] Bo-Yu Chen, Wei-Chen Chiu, and Yu-Lun Liu. Improving robustness for joint optimization of camera pose and decomposed low-rank tensorial radiance fields. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 990–1000, 2024. 3
- [10] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing, 33:2404–2418, 2024. 7

- [11] Yu Cheng and Fajie Yuan. Leanvae: An ultra-efficient reconstruction vae for video diffusion models. arXiv preprint arXiv:2503.14325, 2025. 3
- [12] Jiahao Cui, Wei Jiang, Zhan Peng, Zhiyu Pan, and Zhiguo Cao. Exposure completing for temporally consistent neural high dynamic range video rendering. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10027–10035, 2024. 3
- [13] Mohamed Elgharib, Mohamed Hefeeda, Fredo Durand, and William T Freeman. Video magnification in presence of large motions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4119– 4127, 2015. 3
- [14] Cheng-De Fan, Chen-Wei Chang, Yi-Ruei Liu, Jie-Ying Lee, Jiun-Long Huang, Yu-Chee Tseng, and Yu-Lun Liu. Spectromotion: Dynamic 3d reconstruction of specular scenes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21328–21338, 2025. 3
- [15] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9, 2022. 3
- [16] Brandon Y Feng, Hadi Alzayer, Michael Rubinstein, William T Freeman, and Jia-Bin Huang. 3d motion magnification: Visualizing subtle motions from time-varying radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9837–9846,

- 2023. 3

[17] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20796–20805,

- 2024. 3

- [18] Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, and Ulrich Neumann. Gaussianflow: Splatting gaussian dynamics for 4d content creation. arXiv preprint arXiv:2403.12365, 2024. 2
- [19] Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa Choudhuri, Terrence Chen, and Ziyan Wu. 6dgs: Enhanced direction-aware gaussian splatting for volumetric rendering. arXiv preprint arXiv:2410.04974, 2024. 3
- [20] Chenlong He, Qi Zheng, Ruoxi Zhu, Xiaoyang Zeng, Yibo Fan, and Zhengzhong Tu. Cover: A comprehensive video quality evaluator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5799–5809, 2024. 7
- [21] Hao-Yu Hou, Chia-Chi Hsu, Yu-Chen Huang, Mu-Yi Shen, Wei-Fang Sun, Cheng Sun, Chia-Che Chang, Yu-Lun Liu, and Chun-Yi Lee. 3d gaussian splatting with grouped uncertainty for unconstrained images. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025. 2
- [22] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 3

- [23] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4220–4230, 2024. 2, 7
- [24] Berk Iskender, Sushan Nakarmi, Nitin Daphalapurkar, Marc L Klasky, and Yoram Bresler. Rsr-nf: Neural field regularization by static restoration priors for dynamic imaging. arXiv preprint arXiv:2503.10015, 2025. 3
- [25] Vinoj Jayasundara, Amit Agrawal, Nicolas Heron, Abhinav Shrivastava, and Larry S Davis. Flexnerf: Photorealistic free-viewpoint rendering of moving humans from sparse views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21118– 21127, 2023. 2
- [26] In-Hwan Jin, Haesoo Choo, Seong-Hun Jeong, Park Heemoon, Junghwan Kim, and Kyeongbo Kong. Dynamic scene reconstruction from single image via 4d gaussians. In ECCV Workshops, 2024. Single-image dynamic reconstruction. 2
- [27] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 7
- [28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 3

- [29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023. 4
- [30] Mijeong Kim, Jongwoo Lim, and Bohyung Han. 4d gaussian splatting in the wild with uncertainty-aware regularization. Advances in Neural Information Processing Systems, 37:129209–129226, 2024. 3
- [31] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. In European Conference on Computer Vision, pages 252–269. Springer,

2024. 3

- [32] Raja Kumar and Vanshika Vats. Few-shot novel view synthesis using depth aware 3d gaussian splatting. arXiv preprint arXiv:2410.11080, 2024. 3
- [33] Rakesh Kumar, Harpreet S Sawhney, Yanlin Guo, Steve Hsu, and Supun Samarasekera. 3d manipulation of motion imagery. In Proceedings 2000 International Conference on Image Processing (Cat. No. 00CH37101), pages 17–20. IEEE, 2000. 3
- [34] Yao-Chih Lee, Kuan-Wei Tseng, Yu-Ta Chen, ChienCheng Chen, Chu-Song Chen, and Yi-Ping Hung. 3d video stabilization with depth estimation by cnn-based optimization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10621–10630,

2021. 3

- [35] Marc Levoy and Pat Hanrahan. Light field rendering. In SIGGRAPH, pages 31–42, 1996. Seminal work on light field representation. 2

- [36] Deqi Li, Shi-Sheng Huang, Zhiyuan Lu, Xinran Duan, and Hua Huang. St-4dgs: Spatial-temporally consistent 4d gaussian splatting for efficient dynamic scene rendering. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,

2024. 3

- [37] Hao Li, Sicheng Li, Xiang Gao, Abudouaihati Batuer, Lu Yu, and Yiyi Liao. Gifstream: 4d gaussian-based immersive video with feature stream. arXiv preprint arXiv:2505.07539, 2025. 3
- [38] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20775–20785,

2024. 3

- [39] Zhengqi Li, Tali Dekel, Forrester Cole, Richard Tucker, Noah Snavely, Ce Liu, and William T Freeman. Learning the depths of moving people by watching frozen people. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4521–4530, 2019. 2, 6
- [40] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. arXiv preprint arXiv:2312.16812, 2023. 2, 7
- [41] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4273–4284, 2023. 2
- [42] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Monocular dynamic scene reconstruction via probabilistic integration. PAMI, 2024. Monocular dynamic reconstruction. 2
- [43] Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, et al. Feed-forward bullet-time reconstruction of dynamic scenes from monocular videos. arXiv preprint arXiv:2412.03526, 2024. 2
- [44] Yiqing Liang, Mikhail Okunev, Mikaela Angelina Uy, Runfeng Li, Leonidas Guibas, James Tompkin, and Adam W Harley. Monocular dynamic gaussian splatting: Fast, brittle, and scene complexity rules. arXiv preprint arXiv:2412.04457, 2024. 3
- [45] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed 3d gaussian splatting for casual long videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27412–27422, 2025. 3
- [46] Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, and Yu-Lun Liu. Frugalnerf: Fast convergence for extreme few-shot novel view synthesis without learned priors. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11227–11238,

2025. 3

- [47] Qingming Liu, Yuan Liu, Jiepeng Wang, Xianqiang Lyv, Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dynamic gaussian splatting from casually-captured monocular

videos with depth priors. arXiv preprint arXiv:2406.00434,

2024. Accepted as a poster at ICLR. 3, 7

- [48] Shuaicheng Liu, Lu Yuan, Ping Tan, and Jian Sun. Bundled camera paths for video stabilization. ACM transactions on graphics (TOG), 32(4):1–10, 2013. 3
- [49] Shuaicheng Liu, Lu Yuan, Ping Tan, and Jian Sun. Steadyflow: Spatially smooth optical flow for video stabilization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4209–4216,

2014. 3

- [50] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Hybrid neural fusion for fullframe video stabilization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2299– 2308, 2021. 3
- [51] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13–23,

2023. 3

- [52] Xianrui Luo, Juewen Peng, Zhongang Cai, Lei Yang, Fan Yang, Zhiguo Cao, and Guosheng Lin. Deblur-avatar: Animatable avatars from motion-blurred monocular videos. arXiv preprint arXiv:2501.13335, 2025. 2
- [53] Caoyuan Ma, Yu-Lun Liu, Zhixiang Wang, Wu Liu, Xinchen Liu, and Zheng Wang. Humannerf-se: A simple yet effective approach to animate humannerf with diverse poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1460– 1470, 2024. 2
- [54] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16539– 16548, 2023. 3
- [55] Marko Mihajlovic, Sergey Prokudin, Siyu Tang, Robert Maier, Federica Bogo, Tony Tung, and Edmond Boyer. Splatfields: Neural gaussian splats for sparse 3d and 4d reconstruction. In European Conference on Computer Vision, pages 313–332. Springer, 2024. 3
- [56] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3
- [57] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022. 3
- [58] Tae-Hyun Oh, Ronnachai Jaroensri, Changil Kim, Mohamed Elgharib, Fr’edo Durand, William T Freeman, and Wojciech Matusik. Learning-based video motion magnification. In Proceedings of the European conference on computer vision (ECCV), pages 633–648, 2018. 3
- [59] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting.

- In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21600–21609, 2025. 3
- [60] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5865–5874, 2021. 3
- [61] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021. 3
- [62] St´ephane Pateux, Matthieu Gendrin, Luce Morin, Th´eo Ladune, and Xiaoran Jiang. Bogauss: Better optimized gaussian splatting. arXiv preprint arXiv:2504.01844, 2025. 3
- [63] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10318–10327, 2021. 3
- [64] Marcos Roberto, Helena Maia, and Helio Pedrini. Digital video stabilization: From feature trajectories to camera path optimization. CSUR, 55(3):1–37, 2022. Comprehensive stabilization survey. 3
- [65] Marcos Roberto e Souza, Helena de Almeida Maia, and Helio Pedrini. Survey on digital video stabilization: concepts, methods, and challenges. ACM Computing Surveys (CSUR), 55(3):1–37, 2022. 3
- [66] Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Luˇci´c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6229–6238, 2022. 3
- [67] Christian Schmidt, Jens Piekenbrinck, and Bastian Leibe. Look gauss, no pose: Novel view synthesis using gaussian splatting without accurate pose initialization. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8732–8739. IEEE, 2024. 3
- [68] Ruizhi Shao, Liliang Chen, Zerong Zheng, Hongwen Zhang, Yuxiang Zhang, Han Huang, Yandong Guo, and Yebin Liu. Floren: Real-time high-quality human performance rendering via appearance flow using sparse rgb cameras. In SIGGRAPH Asia 2022 Conference Papers, pages 1–10, 2022. 2
- [69] Ashwath Shetty, Marc Habermann, Guoxing Sun, Diogo Luvizon, Vladislav Golyanik, and Christian Theobalt. Holoported characters: Real-time free-viewpoint rendering of humans from sparse rgb cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1206–1215, 2024. 2
- [70] Zhenmei Shi, Fuhao Shi, Wei-Sheng Lai, Chia-Kai Liang, and Yingyu Liang. Deep online fused video stabilization. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1250–1258, 2022. 3

- [71] Gyumin Shim, Minsoo Lee, and Jaegul Choo. Refu: Refine and fuse the unobserved view for detail-preserving singleimage 3d human reconstruction. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6850–6859, 2022. 3
- [72] Weronika Smolak-DyLL’ewska,´ Dawid Malarz, Kornel Howil, Jan Kaczmarczyk, Marcin Mazur, PrzemysL´ Spurek, et al. Vegas: Video gaussian splatting. arXiv preprint arXiv:2411.11024, 2024. 3
- [73] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5):2732–2742,

2023. 3

- [74] Chih-Hai Su, Chih-Yao Hu, Shr-Ruei Tsai, Jie-Ying Lee, Chin-Yang Lin, and Yu-Lun Liu. Boostmvsnerfs: Boosting mvs-based nerfs to generalizable view synthesis in largescale scenes. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
- [75] Shaolin Su, Qingsen Yan, Yu Zhu, Cheng Zhang, Xin Ge, Jinqiu Sun, and Yanning Zhang. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 7
- [76] Andreas Sudmann. Bullet time and the mediation of postcinematic temporality. 2016. 2
- [77] Hao Sun, Junping Qin, Lei Wang, Kai Yan, Zheng Liu, Xinglong Jia, and Xiaole Shi. 3dgs-hd: Elimination of unrealistic artifacts in 3d gaussian splatting. In 2024 6th International Conference on Data-driven Optimization of Complex Systems (DOCS), pages 696–702. IEEE, 2024. 3
- [78] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollh¨ofer, Christoph Lassner, and Christian Theobalt. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12959– 12970, 2021. 3
- [79] Xuechang Tu, Lukas Radl, Michael Steiner, Markus Steinberger, Bernhard Kerbl, and Fernando de la Torre. Vrsplat: Fast and robust gaussian splatting for virtual reality. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 8(1):1–22, 2025. 3
- [80] Neal Wadhwa, Michael Rubinstein, Fr´edo Durand, and William T Freeman. Phase-based video motion processing. ACM Transactions on Graphics (ToG), 32(4):1–10, 2013. 3
- [81] Fei Wang, Dan Guo, Kun Li, and Meng Wang. Eulermormer: Robust eulerian motion magnification via dynamic filtering within transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5345– 5353, 2024. 3
- [82] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023. 7
- [83] Jianchao Wang, Peng Zhou, Cen Li, Rong Quan, and Jie Qin. Low-frequency first: Eliminating floating artifacts in

3d gaussian splatting. arXiv preprint arXiv:2508.02493,

2025. 3

- [84] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Temporally consistent gaussian splatting for real-time dynamic scene representation. SIGGRAPH, 2024. Recent GS temporal coherence. 3
- [85] Qianqian Wang, Ziyi Yang, Xinyu Gao, and Xiaogang Jin. Multi-view stabilization for free-viewpoint video. In Eurographics, 2024. Temporal consistency across viewpoints.

- 3

[86] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion:

- 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764, 2024. 2, 3

- [87] Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei Zhou. Freetimegs: Free gaussian primitives at anytime anywhere for dynamic scene reconstruction. In CVPR,

2025. 2

- [88] Zijun Wei, Jianming Zhang, Xiaohui Shen, Zhe Lin, Radom´ır Mech, Minh Hoai, and Dimitris Samaras. Good view hunting: Learning photo composition from dense view pairs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3047– 3055, 2018. 7
- [89] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, pages 16210–16220, 2022. 2
- [90] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310–20320, 2024. 2, 3, 7
- [91] Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Fr´edo Durand, and William Freeman. Eulerian video magnification for revealing subtle changes in the world. ACM transactions on graphics (TOG), 31(4):1–8, 2012. 3
- [92] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. arXiv preprint arXiv:2411.18613, 2024. 2
- [93] Renlong Wu, Zhilu Zhang, Mingyang Chen, Zifei Yan, and Wangmeng Zuo. Deblur4dgs: 4d gaussian splatting from blurry monocular video. arXiv preprint arXiv:2412.06424,

2024. 3

- [94] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, YanPei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in 3d gaussian splatting. Computational Visual Media, 10(4): 613–642, 2024. 3
- [95] Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. Grid4d: 4d decomposed hash encoding for high-fidelity dynamic gaussian splatting. Advances in Neural Information Processing Systems, 37:123787–123811, 2024. 3

- [96] Wangze Xu, Huachen Gao, Shihe Shen, Rui Peng, Jianbo Jiao, and Ronggang Wang. Mvpgs: Excavating multi-view priors for gaussian splatting from sparse input views. In European Conference on Computer Vision, pages 203–220. Springer, 2024. 3
- [97] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331–20341, 2024. 2, 7
- [98] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Realtime photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations (ICLR), 2024. 2
- [99] Keyang Ye, Tianjia Shao, and Kun Zhou. Animatable 3d gaussians for high-fidelity synthesis of human motions. arXiv preprint arXiv:2311.13404, 2023. 3
- [100] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, L´aszl´o Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. Advances in Neural Information Processing Systems, 37: 45256–45280, 2024. 2
- [101] Jiyang Yu and Ravi Ramamoorthi. Learning video stabilization using optical flow. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8159–8167, 2020. 3
- [102] Youngsik Yun, Jeongmin Bae, Hyunseung Son, Seoha Kim, Hahyun Lee, Gun Bang, and Youngjung Uh. Compensating spatiotemporally inconsistent observations for online dynamic 3d gaussian splatting. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–9, 2025. 3
- [103] Jianfeng Zhang, Xiaolong Wang, and Zhuang Liu. Gaussian splatting for real-time motion tracking. In ICCV, pages 22130–22139, 2023. GS motion analysis. 3
- [104] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In European Conference on Computer Vision, pages 335–352. Springer, 2024. 3
- [105] Zhuofan Zhang, Zhen Liu, Ping Tan, Bing Zeng, and Shuaicheng Liu. Minimum latency deep online video stabilization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23030– 23039, 2023. 3
- [106] Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang, Tianzhu Zhang, and Yongdong Zhang. Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:101790–101817, 2024. 3
- [107] C. Lawrence Zitnick, Sing Bing Kang, Matthew Uyttendaele, Simon Winder, and Richard Szeliski. High-quality video view interpolation. In SIGGRAPH, pages 600–608,

2004. Early view synthesis. 3

