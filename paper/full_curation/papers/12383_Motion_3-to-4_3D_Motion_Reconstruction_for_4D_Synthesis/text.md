#### Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis

# arXiv:2601.14253v1[cs.CV]20Jan2026

Hongyuan Chen1 Xingyu Chen1 Youjia Zhang1,2 Zexiang Xu3 Anpei Chen1 1Westlake University 2HUST 3Hillbot

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

## Motion 3-to-4

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

Figure 1. From a single glance, Motion 3-to-4 unfolds: weaving time, shape, and movement into living 4D reality.

###### Abstract

We present Motion 3-to-4, a feed-forward framework for synthesising high-quality 4D dynamic objects from a single monocular video and an optional 3D reference mesh. While recent advances have significantly improved 2D, video, and 3D content generation, 4D synthesis remains difficult due to limited training data and the inherent ambiguity of recovering geometry and motion from a monocular viewpoint. Motion 3-to-4 addresses these challenges by decomposing 4D synthesis into static 3D shape generation and motion reconstruction. Using a canonical reference mesh, our model learns a compact motion latent representation and predicts per-frame vertex trajectories to recover complete, temporally coherent geometry. A scalable frame-wise transformer further enables robustness to varying sequence lengths. Evaluations on both standard benchmarks and a new dataset with accurate ground-truth geometry show that Motion 3-to-4 delivers superior fidelity and spatial consis-

tency compared to prior work. Project page is available at https://motion3-to-4.github.io/.

###### 1. Introduction

The creation of high-fidelity 4D assets–which comprehensively capture both the static shape and the dynamic motion of an object over time–is a critical and highly soughtafter capability in fields like virtual reality, cinematography, robotics, and simulation. Recent advancements in 2D image, video and 3D content synthesis have revolutionized the computer graphics and computer vision communities, achieving high fidelity through scalable datasets and learning frameworks. However, 4D reconstruction and generation remain significantly more challenging due to the larger solution space and the inherent complexity of modeling spatial-temporal evolution.

To address these challenges, recent research has advanced along several directions. One line of work [2, 29,

42, 82, 89, 101, 112] first generates multi-view videos from text or single-image inputs and then reconstructs 4D assets using dynamic NeRF [53] or Gaussian Splatting [30]. While conceptually straightforward, these pipelines rely on lengthy per-instance optimization and are fundamentally constrained by the view inconsistencies inherited from 2D generative models. Another line of work adopts pretrained

- 3D generative priors as a foundation for 4D synthesis. For example, V2M4 [7] produces per-frame meshes followed by iterative mesh alignment to recover coherent temporal structure. However, similar to multi-view reconstruction methods, the optimization remains slow and susceptible to temporal artifacts. Alternatively, several approaches such as GVFD [100] and AnimateAnyMesh [86] build motion latent spaces via VAEs and apply the predicted motion to an initial geometry in a feed-forward manner. These methods are efficient and can perform inference in seconds. However, VAE-based modeling typically requires large-scale and diverse training data to form a well-structured latent distribution. When trained solely on the limited and narrow 4D datasets, these models struggle to capture complex motion patterns and exhibit poor generalization. Given the scarcity of high-quality 4D training data, our central idea is to reformulate 4D generation as a combination of 3D shape generation and motion reconstruction.

In this work, we introduce Motion 3-to-4, a feed-forward framework that synthesizes 4D dynamic objects from a single monocular video and an optional reference mesh. To efficiently tackle this inherently ill-posed problem, we decompose 4D generation into two more tractable components: static 3D shape encoding and dynamic motion reconstruction. Our key insight is to leverage a static mesh, either provided or generated, as a stable reference geometry, and to estimate per-frame 3D motion flow relative to this canonical state. Given a monocular video and an optional initial mesh as input, Motion 3-to-4 first performs motion latent learning, jointly encoding the static object shape and the video context into a compact motion representation. Based on this representation, a motion decoder predicts vertex trajectories for queries sampled from the reference mesh, enabling accurate recovery of complete geometry and temporally coherent motion throughout the entire sequence.

We highlight our contributions as follows:

- • A feed-forward 4D synthesis framework. We propose a novel feed-forward pipeline that reformulates 4D object synthesis as a motion reconstruction using only monocular video guidance, achieving strong generalization despite the scarcity of high-quality 4D training data.
- • A scalable architecture. We design a frame-wise transformer architecture that is robust to input meshes of varying resolution and supports flexible processing of videos of arbitrary length.
- • Benchmarks. Existing motion reconstruction and 4D

generation benchmarks typically provide only multi-view renderings without accurate 3D geometry, making rendering alignment and motion evaluation difficult. To address this limitation, we introduce a new motion-80 benchmark with ground-truth motion, realistic renderings, and geometry. We demonstrate the effectiveness of Motion 3-to-4 on both this new dataset and a widely used benchmark [29].

- 2. Related work
- 3D Object Generation. Early progress in 3D generation largely relied on GAN-based models [5, 6, 18, 55, 61], allows for fast object crafting for a specific category. Subsequently, optimization-driven methods distill 2D generative priors, typically guided by CLIP scores [27, 90] or Score Distillation Sampling (SDS) [44, 57, 71, 74, 80, 97], or use multi-view generation followed by reconstruction [17, 26, 34, 65, 70, 91]. While conceptually simple, these methods are often time-consuming and struggle to maintain consistent geometry and appearance across views. Later methods [48, 50, 58, 64] reduce this issue through multi-view fine-tuning on large 3D datasets [12, 13], though reconstruction is still needed to obtain 3D representation.

With the rapid expansion of high-quality 3D data [12, 13], recent work shifts toward direct 3D generation using diffusion-transformers that output explicit 3D representations. These models either tokenize 3D shapes into diffusable latent sets by encoding unstructured point clouds into unordered latent vectors [32, 35, 99, 104, 110], or adopt voxel-based structured latents that can be decoded into explicit 3D representations [41, 81, 84, 87]. In particular, Hunyuan3D 2.0 [110] achieves high-quality generation of

- 3D assets with rich geometry and appearance, forming a strong foundation for our 4D asset creation.
- 4D Reconstruction. Structure-from-Motion (SfM) [1, 62, 66] and Simultaneous Localization and Mapping (SLAM) [11, 54, 72] have long been the foundation for 3D structure and camera pose estimation. Although effective, these approaches often struggle with dynamic sequences, leading to performance degeneracy. To improve robustness in dynamic scenarios, recent work has demonstrated progress toward general dynamic reconstruction by integrating semantic segmentation [22, 98], optical flows [79, 108], geometric constraints [31, 40, 43, 51, 52, 106], generative priors [23, 63]. Alternatively, point-map-based approaches [10, 77, 102] such as CUT3R [77] extend the feedforward 3D foundation model DUSt3R [78] to dynamic scenes by fine-tuning with dynamic datasets. This yields feedforward 4D reconstruction and can be further scaled to longer sequences [9, 85, 113], novel-view synthesis [33, 37, 76, 93], and 3D tracking [15, 46, 49, 88]. Despite recent progress, both reconstruction-based paradigms

remain inherently non-generative: they cannot hallucinate geometry for occluded or unseen regions, often resulting in incomplete surfaces or missing structures.

- 4D Generation. Similar to 3D object generation, early 4D approaches rely on 2D generative priors from multi-view or video models [107], or introduce intermediate rigging structures [3, 14, 21, 47, 68, 69, 92, 103] with skinning-based animation [3, 8, 19, 20, 25, 39, 67, 83]. Score Distillation Sampling (SDS) [57] is also widely adopted to optimize 4D representations from multi-view or video diffusion models [2, 4, 29, 36, 38, 45, 109, 112], but often introduces artifacts. More recent works use generated multi-view videos as explicit supervision [28, 42, 82, 89, 94, 95, 101], yet still require slow per-instance optimization. L4GM [60] enables faster feed-forward 4D regression from generated multiview data, but its performance remains limited by scarce 4D data and view-inconsistent 2D priors.

Current 4D generation methods primarily adopt a twostage pipeline that extends powerful 3D generative models into the temporal dimension. Notable, V2M4 [7] generates meshes for each frame via pre-trained 3D generative models [87, 110], followed by a post-processing step to align these meshes to form a 4D mesh. This generate-then-align strategy is slow and prone to topology drift due to independently conditioned frame generations. To improve temporal consistency, ShapeGen4D [96] fine-tunes a 3D generator [35] on 4D data and alleviates the misalignment issue, but still needs time-consuming post-processing. Parallel efforts such as GVFD [100] animate Trellis-based [87]

- 3D Gaussians via a latent diffusion model that learns global temporal changes, enabling motion generation conditioned on monocular videos. However, since the training depends on rendering supervision of the scarcity of 4D data, it yields weak geometry and 3D structure.

In contrast, we take an orthogonal path by combining the strengths of both 3D generation and reconstruction: we reformulate 4D generation as a combination of 3D shape synthesis and motion reconstruction. By generating the entire object and taking motion synthesis as an alignment problem between surface points and video pixels, our method enables efficient motion representation learning and feedforward prediction without requiring the post-processing alignment steps.

###### 3. Motion 3-to-4

In this paper, we aim to efficiently craft 4D assets that encompass complete shape and motion. Our key idea is to decompose the ill-posed 4D generation problem into static shape generation and dynamic motion reconstruction, enabling the recovery of complete motion flow and geometry, including both visible and unseen surfaces.

To this end, our method takes a monocular video as input and, optionally, an existing mesh asset of the first frame.

Method Solution FF Motion Mesh Retarget FPS

Consistent4D [29] MV Gen. + 3D Rec. ✗ ✗ ✗ ✗ 0.1 SV4D [89] MV Gen. + 3D Rec. ✗ ✓ ✗ ✗ 0.1 L4GM [60] MV Gen. + 3D Rec. ✓ ✗ ✗ ✗ 7.8 DreamMesh4D [38] MV Gen. + 3D Rec. ✗ ✓ ✓ ✗ 0.1 V2M4 [7] 3D Gen. + 4D Align ✗ ✗ ✓ ✗ 0.1 ShapeGen4D [96] 3D Gen. + 4D Align ✗ ✗ ✓ ✗ 0.1 GVFD [100] 3D Gen. + Motion Gen. ✓ ✓ ✗ ✗ 0.8

Motion 3-to-4 3D Gen. + Motion Rec. ✓ ✓ ✓ ✓ 6.5

Table 1. An overview of 4D synthesis methods from monocular video. “FF” denotes feed-forward. “FPS” is averaged over 512 frames. To address the ill-pose Video-to-4D problem, early pipelines generate multi-view images or videos but suffer from view inconsistency. Following frame-wise 3D generative models avoid this issue yet require time-consuming 4D alignment. Motion-generation–based methods animate 3D generation, but their generalizability is fundamentally constrained by the limited availability of 4D training data. We incorporate static generation and motion reconstruction to learn local surface-to-pixel correspondences for efficient novel shapes and complex motions.

If such a mesh is unavailable, we generate one from the initial frame using a pretrained 3D generative model [110]. We then estimate per-frame 3D motion flow relative to the first video frame, yielding temporally consistent 4D assets that encapsulate both shape and motion in their entirety, as shown in Figure 2.

Our framework consists of two main components: 1) motion latent learning that encodes the static mesh and video frames into a compact representation (Section 3.1); and 2) motion decoding that regresses per-frame point locations from queries sampled on the static mesh (Section 3.2).

###### 3.1. Motion Latent Learning

In the following, we will introduce a simple yet efficient representation learning framework for 3D motion.

Geometric Features. To efficiently capture 3D geometry, we first encode the reference mesh M ⊂ {V,F,T ∈ R} with vertices V, faces F, and texture T into a compact latent representation. The mesh can either be user-provided or lifted directly from the video using recent image-to3D generation techniques [87, 110]. To encode shape and appearance, we uniformly sample N surface points X0 = (xi,ni,ci)Ni=1, where xi ∈ R3 is the 3D coordinate, ni ∈ R3 is the surface normal, and ci ∈ R3 denotes the RGB color.

Our shape encoder is inspired by 3DShape2VecSet [99], and we compress the sampled points into a compact 1D latent representation by performing cross attention to aggregate shape information. Specifically, we employ a learnable query set A ∈ RK×C of fixed length K, where each query token is a C-dimensional latent vector. These query tokens act as anchors that attend to and gather information from their neighboring shape samples in X0, producing the shape latent representation.

= CrossSelfAttn(A,PointEmb(X0)) (1)

###### ZX

0

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Input Video

### Image ··· ··· ···

Learnable Token

[Figure 24]

Q

[Figure 25]

| |[Figure 26]<br><br>[Figure 27]<br><br>DIN<br><br>|O Encoder<br><br>| | |
|---|---|---|---|---|
| | | | | |

Shape

Encoder

[Figure 28]

[Figure 29]

[Figure 30]

Image to 3D

Token Motion 3-to-4

Shape

KV

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|K|V| | | |
|---|---|---|---|---|
| | |Motion Deco|[Figure 37]<br><br>[Figure 38]<br><br>der| |

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Q

[Figure 43]

Mesh

Sampled Points

[Figure 44]

Figure 2. An overview of our Motion 3-to-4 framework for 4D synthesis. At the core of the framework is a motion–latent learning module consisting of a geometry encoder and a video encoder, which jointly process the input video and sampled points. The resulting latent tokens are decoded into a frame-wise 3D motion flow relative to the first video frame, producing temporally consistent 4D assets.

where PointEmb(X0) : R9 → RC maps the point label into a high-dimensional positional embedding using an MLP. The aggregated tokens are further refined through a few layers of self-attention transformer blocks to exchange context.

This process embeds the mesh geometry into a lowdimensional latent space, yielding a shape latent ZX

0 ∈ RK×C that retains the essential geometric and semantic structure required for motion reconstruction.

###### Modulation with Video Features. Next, we aggregate the geometric token ZX

with the spatial-temporal video sequence to obtain a motion representation.

0

Specifically, we take a monocular video V ∈ RT×H×W×3 with T frames as input and extract patch-level features using a pretrained DINOv2 [56] encoder. These semantic features facilitate robust correspondence matching and strong generalization across diverse frames, which is crucial for maintaining consistent motion throughout the sequence. Additionally, we inject temporal embeddings into the patch tokens to make them explicitly aware of frame ordering.

To reconstruct the motion for each frame, a straightforward approach is to represent the entire motion as a single 1D latent sequence and decode frame-wise tokens using positional and temporal queries. While this design is efficient for motions with a fixed length, it cannot naturally handle motion sequences of arbitrary duration. To address this limitation, we propose to append the global shape token ZX

0

to each frame token as the frame-wise motion representa-

tion, enabling flexible input lengths, as shown in Figure 2. Furthermore, we add a reference positional token to explicitly distinguish the reference frame from the others, ensuring that the attention mechanism can properly leverage the reference information during propagation.

With this design, we derive the final latent representation Zt for each frame t, which jointly encodes the shared geometric structure and frame-specific motion information. To distinguish motion features across frames, inspired by VGGT [75], we adopt an Alternating-Attention architec-

ture. Let Z(0)t ∈ R(K+P)×C denote the initial aggregated shape and visual latent representation for frame t. For L alternating attention blocks, the updates are defined as:

(Global update) [Z(ℓ−

- 1

- 2 )

- 1

- 2 )

0 ,...,Z(ℓ−

T−1 ]=GlobalAttn(Z(0ℓ−1),...,Z(Tℓ−−11)) (Frame-wise update) Z(tℓ) = FrameAttn(Z(ℓ−

- 1

- 2 )

t ), ∀t = 0,...,T − 1 (2) After L blocks, we obtain the final motion-aware repre-

sentation for each frame as Zt = Z(tL). We take the first K tokens as per-frame motion representation.

###### 3.2. Motion decoding

With the learned shape and motion latent representations, we decode them into explicit per-frame 3D point flows. Rather than predicting the full shape independently per-

instance [96] or attribute offsets per-frame [100], we predict per-frame motion flows relative to the reference shape, which preserves surface correspondences and ensures temporal consistency over long sequences.

To achieve this, we adopt a cross-attention decoder: we take a set of points sampled from the reference mesh as queries and predict their positions at each time step. Specifically, we resample M points from the reference mesh Pˆ0 = {(xi,ni,ci)}Mi=1. These points are embedded using the same PointEmb module employed in shape encoder (Section 3.1).

The motion decoder then predicts the per-frame positions independently:

Xˆ t = MotionDecoder(Xˆ 0,Zt) (3) where Zt is the motion-aware latent for frame t.

This design enables motion prediction at any spatial location and arbitrary time step, providing a flexible and fully feed-forward 4D reconstruction framework. The decoded point features are subsequently processed through a shared fully connected layer to predict the final 3D coordinates.

- 3.3. Training

We train Motion 3-to-4 with straightforward direct supervision by minimizing the mean squared error (MSE) between the predicted and ground-truth point positions,

L =

1 M × T

M

i=1

T

t=1

∥Xˆ it − Xit∥22 (4)

Here, Xt and Xˆ t denote the ground-truth and predicted point positions, respectively. To capture continuous motion fields, we densely sample points during training. This dense supervision encourages the model to learn fine-grained shape correspondences and ensures coherent motion across the entire mesh.

- 4. Experiments

- 4.1. Implementation Details

Dataset. We curate a high-quality animation dataset by filtering 16,000 objects from a pool of approximately 50,000 models sourced from Objaverse [13] and ObjaverseXL [12]. Our filtering policy excludes objects with simplistic geometry (e.g., cubes, spheres) and employs Iterative Closest Point (ICP) analysis to discard sequences exhibiting trivial motion. To ensure scale consistency, we normalize each object to fit within a bounding cube defined by the dimensions [−0.5,0.5]. For the video data, we render each asset at a resolution of 256 × 256 from fixed viewpoints with uniformly sampled azimuth angles. Both the curated assets and their renderings will be publicly released.

Training Strategy. Our model is trained on 12-frame sequences. For each mesh, we sample N = 4096 points as the shape input, which are encoded into K = 64 shape latents. These latents are processed through L = 16 transformer blocks to produce motion latents for decoding, and M = 4096 for densely sampled ground-truth points. Training is conducted with a total batch size of 256 using 8 H100 GPUs, using a learning rate of 4 × 10−4. The model is trained with 60,000 steps in roughly 1.5 days.

###### 4.2. Evaluation

Evaluation Datasets. We evaluate our method on two datasets. (1) We collect a held-out set of 80 subjects from Objaverse, termed Motion-80, featuring objects with rich textures and diverse motion, which contains 64 short sequences and 16 long sequences exceeding 128 frames rendered from four orthogonal views. (2) The Consistent4D benchmark [29], which includes 7 videos of 32 frames each. Since ground-truth meshes are not available for the Consistent4D dataset, we report rendering-based metrics computed at four target novel views following the evaluation protocol in Consistent4D.

Baselines. We conduct comprehensive comparisons with state-of-the-art video-to-4D methods. Our baselines include feedforward approaches that predict 3D Gaussians, i.e., L4GM [60] and GVFD [100]. We also evaluate against the optimization-based method V2M4 [7], which first generates a 3D mesh for each frame and then performs temporal alignment to obtain a 4D mesh.

Metrics. For geometric evaluation, we follow the protocol of Shape2VecSet [99] to compute the Chamfer Distance (CD) and F-Score. This involves sampling 50,000 points from each mesh surface for comparison. However, due to the inherent ambiguity of 4D synthesis from a monocular view, the scale and orientation of the generated meshes may not align with the ground truth. To address this, we apply an Iterative Closest Point (ICP) algorithm for alignment prior to evaluation. Specifically, we register the first frame of the generated sequence to the ground-truth mesh to estimate a rigid transformation (rotation, scale, and translation), which is then applied to all subsequent frames in the sequence. For Gaussian-based methods, we extract the centers of all Gaussians as the surface point set and uniformly sample 50,000 points from it to ensure a fair comparison.

For appearance evaluation, we render the textured mesh from target viewpoints. For the Motion-80 evaluation set, we use the front view as input and the remaining 3 views for eval. For the Consistent4D dataset, we follow its original camera settings. We adopt LPIPS [105], CLIP [59], FVD [73], and DreamSim [16] to assess overall quality and temporal consistency, which are widely used in video-to-4D tasks. We adopt the V2M4 [7] evaluation protocol and fur-

| |Short Sequence| |Long Sequence| |
|---|---|---|---|---|
| |Geometry<br><br>|Appearance<br><br>|Geometry|Appearance|
|Method<br><br>|CD ↓ F-Score ↑|LPIPS ↓ CLIP ↑ FVD ↓ DreamSim ↓|CD ↓ F-Score ↑<br><br>|LPIPS ↓ CLIP ↑ FVD ↓ DreamSim ↓|
|L4GM [60] GVFD [100] V2M4 [7] Ours|0.3561 0.1269 0.1970 0.2608 0.3437 0.2318 0.1113 0.3171<br><br>|0.1487 0.8182 1120.67 0.1941 0.1664 0.7933 1414.21 0.2147 0.1769 0.8080 1516.47 0.1974 0.1495 0.8428 1175.89 0.1682|0.3648 0.0997 OOM OOM 0.3719 0.1652 0.1495 0.2347<br><br>|0.1467 0.7988 1070.72 0.2175 OOM OOM OOM OOM 0.2031 0.7872 1534.16 0.2292 0.1688 0.8352 1264.36 0.1967<br><br>|
|Ours w/m<br><br>|0.0437 0.6774<br><br>|0.0921 0.9251 497.43 0.0614<br><br>|0.0929 0.4322<br><br>|0.1057 0.9224 673.03 0.0781<br><br>|

- Table 2. Quantitative evaluation on our Motion-80 set. Results are reported for both short and long sequences. “Ours w/m” denotes our method initialized with the ground-truth static mesh from the first frame. Thanks to the disentangled mesh representation and sceneflow–based motion modeling, our approach capable of transforming artist-created static 3D meshes into fully dynamic 4D sequences.

|Method<br><br>|LPIPS ↓ CLIP ↑ FVD ↓ DreamSim ↓|
|---|---|
|L4GM [60] GVFD [100] V2M4 [7] Ours|0.1468 0.8457 1207.79 0.1830 0.1789 0.8278 1340.78 0.2009 0.1611 0.8482 1471.58 0.1832 0.1455 0.8609 1260.06 0.1691<br><br>|

- Table 3. Quantitative evaluation on Consist4D benchmark. We evaluate rendering performance across 7 test cases, each containing 32 frames, rendered from 4 target novel views.

GVFD L4GM V2M4 Ours

|[Figure 45]|
|---|

|[Figure 46]<br><br>[Figure 47]| |
|---|---|
|[Figure 48]<br><br>[Figure 49]| |

| |[Figure 50]<br><br>[Figure 51]|
|---|---|
| |[Figure 52]<br><br>[Figure 53]|

|[Figure 54]|[Figure 55]|
|---|---|
| |[Figure 56]<br><br>[Figure 57]|

|[Figure 58]<br><br>[Figure 59]| |
|---|---|
|[Figure 60]<br><br>[Figure 61]| |

[Figure 62]

| |
|---|

|[Figure 63]<br><br>[Figure 64]| |
|---|---|
| |[Figure 65]<br><br>[Figure 66]|

| |[Figure 67]<br><br>[Figure 68]|
|---|---|
| |[Figure 69]<br><br>[Figure 70]|

|[Figure 71]|[Figure 72]|
|---|---|
|[Figure 73]| |

|[Figure 74]|[Figure 75]|
|---|---|
| |[Figure 76]<br><br>[Figure 77]|

[Figure 78]

ther extend it by assessing performance from novel views instead of the input viewpoint. This provides a more comprehensive evaluation, as it avoids cases where a method appears satisfactory from the input view but exhibits artifacts from other views.

Figure 3. Geometric comparison on the Consistent4D benchmark [29]. Through spatially consistent motion reconstruction, we obtain plausible and high-quality 3D geometry.

of our motion reconstruction. Notably, Motion 3-to-4 is the only approach capable of converting artist-created static 3D meshes into dynamic 4D sequences, owing to its disentangled mesh representation and scene-flow–based motion modeling.

Quantitative Evaluation. On geometry. As demonstrated in Table 2 and Figure 3, our method achieves superior

- 4D geometric accuracy compared to all baselines, as indicated by consistent gain across both CD and F-Score metrics. Among Gaussian-based methods, L4GM fails to recover accurate geometry. Because the 3DGS representation is tailored for novel-view synthesis, the predicted Gaussians are not constrained to lie exactly on the surface [24], and the resulting point clouds exhibit floating artifacts, as shown in Figure 3. GVFD can generate reasonable surface point clouds. However, it struggles to accurately reconstruct the motion of these points over time, leading to degraded overall performance. The optimizationbased method V2M4 refines the generated 3D mesh from each frame and reconstructs a plausible surface, thereby outperforming Gaussian-based approaches. Nonetheless, it still suffers from temporal inconsistency, leading to flickering and physically implausible spatial motion, resulting in low CD scores. In contrast, we animate the generated explicit 3D mesh with the reconstructed scene flow, producing temporally coherent motion through alignment between the mesh and video observations and thus achieving high geometric accuracy. This advantage is particularly evident in “Ours w/m”, which drives a ground-truth static mesh from the first frame using our reconstructed motion and significantly outperforms all baselines, highlighting the fidelity

On appearance. As shown in Table 2 and Table 3, our approach achieves better 3D content fidelity and consistency, quantitatively outperforms baselines in CLIP and DreamSim metrics. Note that L4GM is trained and evaluated on the orthogonal view, thus its rendering is biased to the evaluation protocol and may have more advantages than

- our method on the specific rendering perspective. However, when viewed from non-orthogonal novel viewpoints, L4GM exhibits noticeable ghosting artifacts, whereas our method continues to produce plausible and stable results. We invite the reviewer to our supplemental material for more results.

Qualitative Evaluation. Figure 4 further illustrates the benefits of our reformulation. L4GM suffers from error accumulation during multi-view generation, resulting in ghosting artifacts when viewed from angles different from the input views. GVFD generates jittery, temporally inconsistent motions due to limitations in its VAE-based motion modeling that relies on large datasets to learn motion latent distribution, leading to weak generalization and discontinu-

- ous appearance in the Gaussian during movement. V2M4,

###### GT GVFD L4GM V2M4 Ours Ours w/m

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

- Figure 4. Qualitative Comparisons. We compare our method with strong baselines including GVFD [100], L4GM [60], and V2M4 [7] on our proposed Motion-80 benchmark. For fair evaluation, we render the generated 4D assets from all methods into an orthogonal novel view. Our approach produces more temporally coherent and structurally consistent motion. We invite reviewers to consult the supplemental material for animation visualization.

relying on per-frame optimization, generates plausible results from the input view, but suffers from spatial discontinuities when observed from other viewpoints, failing to capture true motion. In contrast, our method combines a strong pretrained 3D generator with feed-forward, generalizable motion reconstruction. This design preserves both spatial and temporal coherence, produces smooth, physically plausible motion, and achieves superior visual fidelity.

and the robust geometry encoder, our method generalizes well to in-the-wild video inputs, including both real-world footage and generated animation sequences.

Motion Transfer. Although our model is trained using paired videos and corresponding 3D subjects, we observe that it inherently generalizes to transferring the motion from an input video to a 3D object with different shape and appearance. As shown in Figure 6, we feed the dragon video into the DINO encoder and use the chicken and robotdragon meshes as inputs to the geometry encoder. Remarkably, our method successfully transfers the neck, body and leg motions from the video to these new target objects. For shapes with geometric differences, our method is still able to successfully transfer the leg movement from the source to the target model.

###### 4.3. More Results

Wild4D. We show a few in-the-wild testing samples Figure 5. To handle out-of-domain inputs, we first apply BiRefNet [111] to automatically remove background regions on a per-frame basis. We then generate an initial 3D shape using the first video frame with Hunyuan2.0 [110]. Thanks to the strong visual features from the DINO encoder

7

Video 4D Reconstruction Static Scene Dynamic Object

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- Figure 5. In-the-Wild Video-to-4D Synthesis. Our method generalizes to diverse in-the-wild inputs, including real-world videos (top row) and generated animations (bottom row). By formulating motion reconstruction as surface-to-pixel alignment, we achieve robust local correspondence reasoning across varied shapes and motion patterns.

[Figure 107]

Source Target

[Figure 108]

[Figure 109]

|[Figure 110]<br><br>A<br><br>|[Figure 111]|
|---|
<br><br>|[Figure 112]|
|---|
<br><br>|[Figure 113]<br><br>B<br><br>|[Figure 114]|
|---|
<br><br>|[Figure 115]|
|---|
|
|---|---|

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

- Figure 6. Motion Transfer Example. By disentangling 4D synthesis into 3D mesh generation and motion reconstruction, our framework can animate static articulated objects with motion retargeted from videos of different sources.

Figure 7. Failure cases. (A) Vertex sticking in challenging cases. (B) Initial mesh topology not able to adapt to later motion.

real-world shapes as well as diverse motion patterns. Flexibility: Our framework also supports lifting static articulated objects into dynamic 4D driven by video conditions, including motion retargeting from entirely different sources.

###### 5. Conclusion

We presented a 4D synthesis pipeline that decomposes the inherently ill-posed 4D generation problem into two tractable components: 3D shape generation and motion reconstruction. This decomposition offers several key advantages. Efficiency: Off-the-shelf 3D generative models can be directly reused for high-quality shape synthesis, while the motion branch remains lightweight, substantially reducing the scale requirements for 4D training data. Generalizability: By formulating motion reconstruction as an alignment task between surface points and video pixels, our method performs robust local correspondence reasoning, enabling strong generalization to both synthesized and

Limitations. Our method exhibits several limitations. First, the geometry encoder operates on a dense point cloud without explicitly modeling the mesh topology. As a result, when different parts of the object are not clearly separated in the reference mesh, the model may produce vertex sticking artifacts, as illustrated in Figure 7(A). Second, when reconstructing motion from monocular video, our pipeline relies on the mesh generated from the first frame as reference geometry. This makes it difficult to accommodate topology changes that occur in later frames, leading to failure cases, as shown in Figure 7(B).

###### Acknowledgments

We would like to thank the members of Inception3D Lab for their helpful discussions, and Isabella Liu for sharing the Blender scripts used in rendering the teaser. This work is done with the sponsorship of TeleAI.

###### References

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. ACM Communications,

2011. 2

- [2] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In CVPR, 2024. 1, 3
- [3] Ilya Baran and Jovan Popovi´c. Automatic rigging and animation of 3d characters. ACM Trans. on Graphics, 2007. 3
- [4] Yarin Bekor, Gal Michael Harari, Or Perel, and Or Litany. Gaussian see, gaussian do: Semantic 3d motion transfer from multiview video. In SIGGRAPH Asia, pages 1–10,

2025. 3

- [5] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In CVPR, 2022. 2
- [6] Anpei Chen, Ruiyang Liu, Ling Xie, Zhang Chen, Hao Su, and Jingyi Yu. Sofgan: A portrait image generator with dynamic styling. ACM Trans. on Graphics, 2022. 2
- [7] Jianqi Chen, Biao Zhang, Xiangjun Tang, and Peter Wonka. V2m4: 4d mesh animation reconstruction from a single monocular video. In ICCV, 2025. 2, 3, 5, 6, 7
- [8] Ling-Hao Chen, Yuhong Zhang, Zixin Yin, Zhiyang Dou, Xin Chen, Jingbo Wang, Taku Komura, and Lei Zhang. Motion2motion: Cross-topology motion transfer with sparse correspondence. In SIGGRAPH Asia, 2025. 3
- [9] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645, 2025. 2
- [10] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. In ICCV, 2025. 2
- [11] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. Monoslam: Real-time single camera slam. PAMI, 2007. 2
- [12] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. NeurIPS,

2023. 2, 5

- [13] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2, 5

- [14] Yufan Deng, Yuhao Zhang, Chen Geng, Shangzhe Wu, and Jiajun Wu. Anymate: A dataset and baselines for learning 3d object rigging. In SIGGRAPH, 2025. 3
- [15] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. In ICCV, 2025. 2
- [16] Stephanie Fu, Netanel Y Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: learning new dimensions of human visual similarity using synthetic data. NeurIPS, 2023. 5
- [17] Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, and Bernhard Sch¨olkopf. Graphdreamer: Compositional 3d scene synthesis from scene graphs. In CVPR, 2024. 2
- [18] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. NeurIPS, 2022. 2
- [19] Zhiyang Guo, Jinxu Xiang, Kai Ma, Wengang Zhou, Houqiang Li, and Ran Zhang. Make-it-animatable: An efficient framework for authoring animation-ready 3d characters. In CVPR, 2025. 3
- [20] Zhiyang Guo, Ori Zhang, Jax Xiang, Alan Zhao, Wengang Zhou, and Houqiang Li. Make-it-poseable: Feed-forward latent posing model for 3d humanoid character animation. arXiv preprint arXiv:2512.16767, 2025. 3
- [21] Guangzhao He, Chen Geng, Shangzhe Wu, and Jiajun Wu. Category-agnostic neural object rigging. In CVPR, 2025. 3
- [22] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask R-CNN. In ICCV, 2017. 2
- [23] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025. 2
- [24] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH, 2024. 6
- [25] Zehuan Huang, Haoran Feng, Yangtian Sun, Yuanchen Guo, Yanpei Cao, and Lu Sheng. Animax: Animating the inanimate in 3d with joint video-pose diffusion models. arXiv preprint arXiv:2506.19851, 2025. 3
- [26] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. In ICCV, 2025. 2
- [27] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In CVPR, 2022. 2
- [28] Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. Animate3d: Animating any 3d model with multi-view video diffusion. NeurIPS, 2024. 3
- [29] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360◦ dynamic object generation from monocular video. In ICLR, 2024. 1, 2, 3, 5, 6
- [30] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. on Graphics, 2023. 2

- [31] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In CVPR, 2021. 2
- [32] Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, et al. Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint arXiv:2506.16504, 2025. 2
- [33] Jiahui Lei, Yijia Weng, Adam W Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In CVPR, 2025. 2
- [34] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In ICLR, 2024. 2
- [35] Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, et al. Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747, 2025. 2, 3
- [36] Xuan Li, Qianli Ma, Tsung-Yi Lin, Yongxin Chen, Chenfanfu Jiang, Ming-Yu Liu, and Donglai Xiang. Articulated kinematics distillation from video diffusion models. In CVPR, 2025. 3
- [37] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In CVPR, 2021. 2
- [38] Zhiqi Li, Yiming Chen, and Peidong Liu. Dreammesh4d: Video-to-4d generation with sparse-controlled gaussianmesh hybrid representation. NeurIPS, 2024. 3
- [39] Zizhang Li, Dor Litvak, Ruining Li, Yunzhi Zhang, Tomas Jakab, Christian Rupprecht, Shangzhe Wu, Andrea Vedaldi, and Jiajun Wu. Learning the 3d fauna of the web. In CVPR,

2024. 3

- [40] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. MegaSaM: accurate, fast, and robust structure and motion from casual dynamic videos. In CVPR, 2025. 2
- [41] Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. Sparc3d: Sparse representation and construction for high-resolution 3d shapes modeling. arXiv preprint arXiv:2505.14521, 2025. 2
- [42] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024. 2, 3
- [43] Ting-Hsuan Liao, Haowen Liu, Yiran Xu, Songwei Ge, Gengshan Yang, and Jia-Bin Huang. Pad3r: Pose-aware dynamic 3d reconstruction from casual videos. In SIGGRAPH Asia, pages 1–11, 2025. 2
- [44] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: Highresolution text-to-3d content creation. In CVPR, 2023. 2

- [45] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In CVPR, 2024. 3
- [46] Isabella Liu, Hao Su, and Xiaolong Wang. Dynamic gaussians mesh: Consistent mesh reconstruction from dynamic scenes. arXiv preprint arXiv:2404.12379, 2024. 2
- [47] Isabella Liu, Zhan Xu, Wang Yifan, Hao Tan, Zexiang Xu, Xiaolong Wang, Hao Su, and Zifan Shi. Riganything: Template-free autoregressive rigging for diverse 3d assets. ACM Trans. on Graphics, 2025. 3
- [48] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In ICCV, 2023. 2
- [49] Xinhang Liu, Yuxi Xiao, Donny Y Chen, Jiashi Feng, YuWing Tai, Chi-Keung Tang, and Bingyi Kang. Trace anything: Representing any video in 4d via trajectory fields. arXiv preprint arXiv:2510.13802, 2025. 2
- [50] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. In ICLR, 2024. 2
- [51] Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. Consistent video depth estimation. ACM Trans. on Graphics, 2020. 2
- [52] Ziqiao Ma, Xuweiyi Chen, Shoubin Yu, Sai Bi, Kai Zhang, Chen Ziwen, Sihan Xu, Jianing Yang, Zexiang Xu, Kalyan Sunkavalli, et al. 4d-lrm: Large space-time reconstruction model from and to any view at any time. arXiv preprint arXiv:2506.18890, 2025. 2
- [53] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 2
- [54] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics, 2015. 2
- [55] Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In CVPR, 2021. 2
- [56] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research. 4, 1
- [57] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR,

2023. 2, 3

- [58] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d. In CVPR, 2024. 2
- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICCV, 2021. 5

- [60] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. NeurIPS, 2024. 3, 5, 6, 7, 2
- [61] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. Graf: Generative radiance fields for 3d-aware image synthesis. NeurIPS, 2020. 2
- [62] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In CVPR, 2016. 2
- [63] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. In CVPR, 2025. 2
- [64] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2
- [65] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In ICLR, 2024. 2
- [66] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In SIGGRAPH.

2006. 2

- [67] Chaoyue Song, Xiu Li, Fan Yang, Zhongcong Xu, Jiacheng Wei, Fayao Liu, Jiashi Feng, Guosheng Lin, and Jianfeng Zhang. Puppeteer: Rig and animate your 3d models. NeurIPS, 2025. 3
- [68] Chaoyue Song, Jianfeng Zhang, Xiu Li, Fan Yang, Yiwen Chen, Zhongcong Xu, Jun Hao Liew, Xiaoyang Guo, Fayao Liu, Jiashi Feng, et al. Magicarticulate: Make your 3d models articulation-ready. In CVPR, 2025. 3
- [69] Keqiang Sun, Dor Litvak, Yunzhi Zhang, Hongsheng Li, Jiajun Wu, and Shangzhe Wu. Ponymation: Learning articulated 3d animal motions from unlabeled online videos. In ECCV, 2024. 3
- [70] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In ECCV, 2024. 2
- [71] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In ICLR, 2024. 2
- [72] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. NeurIPS, 2021. 2
- [73] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 5
- [74] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR,

- 2023. 2

- [75] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 4, 1

- [76] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In ICCV, 2025. 2
- [77] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 2
- [78] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: geometric 3d vision made easy. In CVPR, 2024. 2
- [79] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In ECCV, 2024. 2
- [80] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: Highfidelity and diverse text-to-3d generation with variational score distillation. NeurIPS, 2023. 2
- [81] Guanjun Wu, Jiemin Fang, Chen Yang, Sikuang Li, Taoran Yi, Jia Lu, Zanwei Zhou, Jiazhong Cen, Lingxi Xie, Xiaopeng Zhang, et al. Unilat3d: Geometry-appearance unified latents for single-stage 3d generation. arXiv preprint arXiv:2509.25079, 2025. 2
- [82] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In CVPR, 2025. 2, 3
- [83] Shangzhe Wu, Ruining Li, Tomas Jakab, Christian Rupprecht, and Andrea Vedaldi. Magicpony: Learning articulated 3d animals in the wild. In CVPR, 2023. 3
- [84] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025. 2
- [85] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863, 2025. 2
- [86] Zijie Wu, Chaohui Yu, Fan Wang, and Xiang Bai. Animateanymesh: A feed-forward 4d foundation model for text-driven universal mesh animation. In ICCV, 2025. 2
- [87] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In CVPR, 2025. 2, 3
- [88] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: 3d point tracking made easy. In ICCV, 2025. 2
- [89] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. In ICLR,

2025. 2, 3

- [90] Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. Dream3d: Zero-shot text-to-3d synthesis using 3d shape prior and text-to-image diffusion models. In CVPR, 2023. 2
- [91] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d

mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

- 2024. 2

- [92] Zhan Xu, Yang Zhou, Evangelos Kalogerakis, Chris Landreth, and Karan Singh. Rignet: Neural rigging for articulated characters. ACM Trans. on Graphics, 2020. 3
- [93] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In CVPR, 2024. 2
- [94] Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. Diffusion2: Dynamic 3d content generation via score composition of video and multi-view diffusion models. In ICLR, 2025. 3
- [95] Chun-Han Yao, Yiming Xie, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d 2.0: Enhancing spatio-temporal consistency in multi-view video diffusion for high-quality 4d generation. In ICCV, 2025. 3
- [96] Jiraphon Yenphraphai, Ashkan Mirzaei, Jianqi Chen, Jiaxu Zou, Sergey Tulyakov, Raymond A Yeh, Peter Wonka, and Chaoyang Wang. Shapegen4d: Towards high quality 4d shape generation from videos. arXiv preprint arXiv:2510.06208, 2025. 3, 5
- [97] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In CVPR, 2024. 2
- [98] Chao Yu, Zuxin Liu, Xin-Jun Liu, Fugui Xie, Yi Yang, Qi Wei, and Fei Qiao. DS-SLAM: A semantic visual SLAM towards dynamic environments. In IROS, 2018. 2
- [99] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Trans. on Graphics, 2023. 2, 3, 5, 1
- [100] Bowen Zhang, Sicheng Xu, Chuxin Wang, Jiaolong Yang, Feng Zhao, Dong Chen, and Baining Guo. Gaussian variation field diffusion for high-fidelity video-to-4d synthesis. In ICCV, 2025. 2, 3, 5, 6, 7
- [101] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. 4diffusion: Multi-view video diffusion model for 4d generation. NeurIPS, 2024. 2, 3
- [102] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. MonST3R: a simple approach for estimating geometry in the presence of motion. In ICLR, 2025. 2
- [103] Jia-Peng Zhang, Cheng-Feng Pu, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. One model to rig them all: Diverse skeleton rigging with unirig. ACM Trans. on Graphics,

2025. 3

- [104] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Trans. on Graphics,

2024. 2

- [105] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5

- [106] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Noah Snavely, Michael Rubinstein, and William T. Freeman. Structure and motion from casual videos. In ECCV, 2022. 2
- [107] Mingrui Zhao, Sauradip Nag, Kai Wang, Aditya Vora, Guangda Ji, Peter Chun, Ali Mahdavi-Amiri, and Hao Zhang. Advances in 4d representation: Geometry, motion, and interaction. arXiv preprint arXiv:2510.19255, 2025. 3
- [108] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. Particlesfm: Exploiting dense point trajectories for localizing moving cameras in the wild. In ECCV, 2022. 2
- [109] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023. 3
- [110] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 2, 3, 7, 1
- [111] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI AIR, 3:1–12, 2024. 7
- [112] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for textand image-guided 4d scene generation. In CVPR, 2024. 2, 3
- [113] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025. 2

#### Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis Supplementary Material

###### A. Implementation Details

###### A.1. Training and Inference Details

Network Architecture. Our network architecture consists of two primary components: a shape encoder and a motion latent block. For the shape encoder, we adopt the architecture from 3DShape2VecSet [99], which provides strong representational capacity and high compression efficiency. Specifically, we employ a point cloud encoder to extract features from the canonical shape representation. The encoder takes N = 4096 sampled points as input. First, the point coordinates are projected through a Fourier feature embedding layer to obtain position embeddings. These embeddings are then concatenated with surface normals and color attributes, and subsequently projected to the model dimension d = 768 via a linear layer. To aggregate point features into a compact representation, we employ a QK-Norm crossattention block with 64 learnable query tokens. The resulting tokens are further processed through 4 transformer layers with self-attention mechanisms to capture inter-point relationships, ultimately producing a semantically rich canonical shape representation Z ∈ R64×768.

For video encoding, we adopt a frozen DINOv2-ViTB/14 model [56] to extract spatial features from each frame. Input videos are resized to 224 × 224 resolution and processed frame-by-frame through the DINOv2 encoder, which extracts 256 patch tokens per frame with dimension 768. For a video with T frames, we obtain image tokens V ∈ RT×256×768. These tokens are combined with fixed positional embeddings generated using sinusoidal encoding over both temporal and spatial dimensions. To further aggregate spatio-temporal information from the video, we process these embeddings through Motion 3-to-4 blocks. Following the design of VGGT [75], we adopt an alternating global-frame attention architecture consisting of 16 layers (8 global and 8 frame). Both global and frame attention layers use QK-normalization. This alternating design effectively captures both spatial and temporal dependencies while maintaining computational efficiency.

To predict per-point motion trajectories, we extract 64 motion tokens from the processed video sequence. For each point in M = 4096 output point clouds, we construct a query using its position, normal, and color through the same embedding layer used in the shape encoder. We then apply a QK-Norm cross-attention layer where the per-point queries attend to the extracted motion tokens from the corresponding frame. This cross-attention mechanism produces perpoint features F ∈ RM×768 that encode motion information. Finally, a two-layer MLP with GELU activation de-

codes these features into motion trajectories Xt ∈ RM×3 for time t. During inference, when we need to animate mesh vertices, we process them in chunks of 4096 to maintain memory efficiency. For temporal processing, we adopt a chunk size of 256 frames. When dealing with videos exceeding 256 frames, we use a sliding window approach with a stride of 255 frames, where each window consists of the first frame concatenated with 255 subsequent frames to maintain temporal consistency.

Training Configuration. We employ the AdamW optimizer with learning rate η = 4×10−4, β1 = 0.9, β2 = 0.95, and weight decay 0.05. The learning rate follows a cosine annealing schedule with 1000 warm-up steps. We train for 60,000 parameter update steps with gradient clipping at norm 1.0. To reduce memory consumption and accelerate training, we apply gradient checkpointing, FlashAttentionv2 in the xFormers library, and a mixed precision strategy with BF16.

Training Data and Strategy. For training data, we use videos at 256×256 resolution with black backgrounds. The point clouds are uniformly sampled on the mesh surface, and crucially, we sample points at consistent barycentric coordinates within each face across all frames. This ensures temporal correspondence between points across different frames, enabling us to track each point’s trajectory and supervise the model using MSE loss.

For the training strategy, we train on 12-frame sequences and apply temporal data augmentation. Specifically, we randomly select a starting frame and then sample 12 consecutive frames with stride intervals of 1, 2, or 4 frames. This augmentation strategy enables the model to handle different initial poses and learn to predict larger motion displacements across varying temporal scales.

Inference with video. For cases where only video input is provided without any mesh, we leverage Hunyuan 2.0 [110] to generate a mesh based on the first frame. It is worth noting that the directly generated mesh is non-watertight and includes texture. To address the watertight issue, we apply a vertex mapping technique to convert it into a watertight mesh while preserving the original texture, which is essential for subsequent video-driven animation. For cases where a mesh is already provided, we directly drive the mesh using generated or existing videos.

###### A.2. Evaluation Details

We utilize the official release code for evaluation. For our held-out dataset, we use the front view as the input view and the remaining three orthogonal views for evaluation.

We exclude the front view from evaluation because including it would be unfair to generation-based methods, since L4GM [60] can perform lossless reconstruction of the input view (i.e., the front view).

In the Table 2, ”OOM” refers to ”out of memory.” For the GVFD [100], the official released code does not provide scripts for processing long sequences; it only includes scripts for single-video inference. Consequently, when the sequence length exceeds 128 frames, our machine encounters the out-of-memory (OOM) problem, preventing us from obtaining quantitative results for this method.

###### A.3. Ablation Study

We conduct an ablation study to explore different model architecture choices. Due to limited training resources, we train each model variant for 30,000 steps under identical settings. To evaluate the trajectory prediction capability of each variant, we report the mean squared error (MSE) of point trajectories over time on the held-out dataset in Table 4, which provides a more direct measure of the model’s ability to track accurate motion trajectories.

Frame Attn Global Attn Ref Token Rec MSE ↓

✗ ✓ ✓ 0.0055 ✓ ✗ ✓ 0.0033 ✓ ✓ ✗ 0.0021 ✓ ✓ ✓ 0.0018

Table 4. Ablation studies of the modules of Motion 3-to-4. Rec MSE denotes the squared error averaged across time steps within the [−0.5, 0.5] bounding box.

###### B. More Results

We provide several categories of additional visualization results below. We also include a local video webpage in the supplementary materials to better present the results.

###### B.1. More Results from Synthesis Video

- As illustrated in Figure 8, our model demonstrates strong generalization capability across diverse test cases from synthetic videos, achieving high-quality results that showcase its ability to handle various object types and motion patterns.

B.2. More Results from Real-World Video

- As illustrated in Figure 9, despite being trained exclusively on synthetic data, our model generalizes well to real-world videos. This demonstrates the model’s robustness and its ability to bridge the synthetic-to-real domain gap, highlighting the effectiveness of our method. B.3. Results of Existing 3D Model
- As illustrated in Figure 10, our model has strong practical value as it can extend existing static 3D assets to dynamic 4D content. This capability enables flexible application scenarios, such as animating existing 3D models from various

Time 1 Time 2 Time 3 Time 1 Time 2 Time 3

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Input

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- View1
- View2
- View3

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

Figure 8. Additional visual results from synthesis video.

Time 1 Time 2 Time 3 Time 1 Time 2 Time 3

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Input

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

- View1
- View2
- View3

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

Figure 9. Additional visual results from real-world video.

sources, thereby significantly broadening the potential use cases of our method.

Time 1 Time 2 Time 3 Time 1 Time 2 Time 3

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Input

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- View1
- View2
- View3

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

Figure 10. Results of existing 3D model condition on generated video.

###### B.4. Visual Comparison on Consist4D

The Consist4D dataset [29] provides input views from various angles that differ from the frontal views in our heldout dataset, which better demonstrates the model’s robust-

###### Time 1 Time 2 Time 3 Time 1 Time 2 Time 3

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

###### Input

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

###### View1

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

###### View2

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

View3

Figure 11. More results from the held-out objaverse dataset.

ness to different viewpoints. This is especially relevant for method L4GM, which is designed to work well under orthogonal views. When the input view is not orthogonal, L4GM’s multi-view diffusion module fails to generate consistent results across views, leading to severe ghosting artifacts as shown in Figure 12.

###### B.5. More Results from the Held-Out Dataset

- As illustrated in Figure 11, we present additional results on our held-out dataset to demonstrate that our model achieves superior visual fidelity and temporal motion coherence. These results further validate the effectiveness of our approach in generating high-quality 4D content with both realistic appearance and consistent motion over time.

##### GT GVFD L4GM V2M4 Ours

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

- View1

- View3

- View1
- View2

View2

- View4

- View3
- View4

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

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Figure 12. Visual comparison with SOTA methods on Consist4D.

