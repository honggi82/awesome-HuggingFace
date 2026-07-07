# arXiv:2512.19678v1[cs.CV]22Dec2025

## WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion

Hanyang Kong1 Xingyi Yang2* Xiaoxu Zheng1 Xinchao Wang1*

1National University of Singapore 2The Hong Kong Polytechnic University

hanyang.k@u.nus.edu, xingyi.yang@polyu.edu.hk, xinchao@nus.edu.sg https://hyokong.github.io/worldwarp-page/

[Figure 1]

[Figure 2]

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

Starting Image 1st Frame 200th Frame 3DGS Reconstruction

Figure 1. WorldWarp: Long-range novel view synthesis from a single image. Given only a single starting image (left) and a specified camera trajectory, our method generates a long and coherent video sequence. The core of our approach is to generate the video chunk-bychunk, where each new chunk is conditioned on forward-warped “hints" from the previous one. A novel diffusion model then generates the next chunk by correcting these hints and filling in occlusions using a spatio-temporal varying noise schedule. The high geometric consistency of our 200-frame generated sequence is demonstrated by its successful reconstruction into a high-fidelity 3D Gaussian Splatting (3DGS) [25] model (right). This highlights our model’s robust understanding of 3D geometry and its capability to maintain long-term consistency.

### Abstract

Generating long-range, geometrically consistent video presents a fundamental dilemma: while consistency demands strict adherence to 3D geometry in pixel space, stateof-the-art generative models operate most effectively in a camera-conditioned latent space. This disconnect causes current methods to struggle with occluded areas and complex camera trajectories. To bridge this gap, we propose WorldWarp, a framework that couples a 3D structural anchor with a 2D generative refiner. To establish geometric grounding, WorldWarp maintains an online 3D geometric cache built via Gaussian Splatting (3DGS). By explicitly

*Corresponding author.

warping historical content into novel views, this cache acts as a structural scaffold, ensuring each new frame respects prior geometry. However, static warping inevitably leaves holes and artifacts due to occlusions. We address this using a Spatio-Temporal Diffusion (ST-Diff) model designed for a "fill-and-revise" objective. Our key innovation is a spatio-temporal varying noise schedule: blank regions receive full noise to trigger generation, while warped regions receive partial noise to enable refinement. By dynamically updating the 3D cache at every step, WorldWarp maintains consistency across video chunks. Consequently, it achieves state-of-the-art fidelity by ensuring that 3D logic guides structure while diffusion logic perfects texture. Project page: https://hyokong.github.io/worldwarp-page/.

### 1. Introduction

Novel View Synthesis (NVS) has emerged as a cornerstone problem in computer vision and graphics, with transformative applications in virtual reality, immersive telepresence, and generative content creation. While traditional NVS methods excel at view interpolation, which generates new views within the span of existing camera poses [2, 25, 40], the frontier of the field lies in view extrapolation [16, 32, 33, 37, 55, 67]. This far more challenging task involves generating long, continuous camera trajectories that extend significantly beyond the original scene, effectively synthesizing substantial new content and structure [37, 55]. The ultimate goal is to enable interactive exploration of dynamic, 3D-consistent worlds from only a limited set of starting images.

The central challenge in generating long-range, cameraconditioned video lies in finding an effective 3D conditioning. Existing works have largely followed two main strategies. The first is camera pose encoding, which embeds abstract camera parameters as a latent condition [16, 29, 39, 54, 55, 67]. This approach, however, relies heavily on the diversity of the training dataset and often fails to generalize to Out-Of-Distribution (OOD) camera poses, while also providing minimal information about the underlying 3D scene content [11, 20, 32, 41, 44, 77]. The second strategy, which uses an explicit 3D spatial prior, was introduced to solve this OOD issue [11, 20, 32, 77]. While these priors provide robust geometric grounding, they are imperfect, suffering from occlusions (blank regions) and distortions from 3D estimation errors [55, 77]. This strategy typically employs standard inpainting or video generation techniques [11, 20, 32], which are ill-suited to simultaneously handle the severe disocclusions and the geometric distortions present in the warped priors, leading to artifacts and inconsistent results.

To address this critical gap, we propose WorldWarp, a novel framework that generates long-range, geometricallyconsistent novel view sequences. Our core insight is to break the strict causal chain of AR models and the static nature of explicit 3D priors. Instead, WorldWarp operates via an autoregressive inference pipeline that generates video chunkby-chunk (see Fig. 3). The key to our system is a SpatioTemporal Diffusion (ST-Diff) model [49, 60], which is trained with a powerful bidirectional, non-causal attention mechanism. This non-causal design is explicitly enabled by our core technical idea: using forward-warped images from future camera positions as a dense, explicit 2D spatial prior [9]. At each step, we build an "online 3D geometric cache" using 3DGS [25], which is optimized only on the most recent, high-fidelity generated history. This cache then renders high-quality warped priors for the next chunk, providing ST-Diff with a rich, geometrically-grounded signal that guides the generation of new content and fills occlusions.

The primary advantage of WorldWarp is its ability to

avoid the irreversible error propagation that plagues prior work [55, 77]. By dynamically re-estimating a short-term 3DGS cache at each step, our method continuously grounds itself in the most recent, accurate geometry, ensuring highfidelity consistency over extremely long camera paths. We demonstrate the effectiveness of our approach through extensive experiments on challenging real-world and synthetic datasets for long-sequence view extrapolation, achieving state-of-the-art performance in both geometric consistency and visual fidelity.

In summary, our main contributions are:

- • WorldWarp, a novel framework for long-range novel view extrapolation that generates video chunk-by-chunk using an autoregressive inference pipeline.
- • Spatio-Temporal Diffusion (ST-Diff), a non-causal diffusion model that leverages bidirectional attention conditioned on forward-warped images as a dense geometric prior.
- • An online 3D geometric cache mechanism, which uses test-time optimized 3DGS [25] to provide high-fidelity warped priors while preventing the irreversible error propagation of static 3D representations.
- • State-of-the-art performance on challenging view extrapolation benchmarks, demonstrating significantly improved geometric consistency and image quality over existing methods.

### 2. Related Works

Novel view synthesis. Novel view synthesis (NVS) is a challenging problem that can be categorized into two aspects: view interpolation [2, 3, 25, 31, 40, 42, 46, 51, 52, 62, 71, 78, 79, 85] and extrapolation [16, 30, 32, 33, 37, 48, 55, 67, 76, 83]. View interpolation task aims to generate novel views within the distributions of the training views [2, 3, 25, 40, 78] even if the training views are sparse [31, 42, 62, 85] or the training views are captured in the wild with occlusions [46, 51, 52]. View extrapolation tasks [16, 30, 32,

- 33, 37, 48, 55, 67, 76, 83] focus on generating novel views which are extended significantly beyond the original scenes, introducing substantial new contents, by leveraging powerful pre-trained video diffusion models [21, 35, 49, 60, 73].

Auto-regressive video diffusion models. The field of video generation has seen a prominent trend towards either diffusion-based or autoregressive (AR) methodologies. Parallel (non-autoregressive) video diffusion systems often employ bidirectional attention to process and denoise all frames concurrently [4–6, 10, 14, 15, 18, 19, 28, 43, 59, 61, 74]. Conversely, AR-based techniques produce content in a sequential manner. This category encompasses several architectures, such as models based on pure next-token prediction [7, 23, 27, 45, 65, 68, 72], more recent hybrid systems integrating AR and diffusion principles [8, 12, 13, 22, 24,

- 34, 38, 69, 75, 82], and rolling diffusion variants that em-

ploy progressive noise schedules [26, 50, 53, 58, 70, 80]. However, these AR strategies are ill-suited for this work’s specific task. Learning an effective camera embedding for them is non-trivial, and their causal structure is incompatible with using warped images from future camera positions as conditional hints. Consequently, this work employs a non-autoregressive framework [57] to leverage this future information.

Camera pose encoding and 3D explicit spatial priors. Spatially consistent view generation relies on conditioning. One method, camera pose encoding, models camera geometry using absolute extrinsics [16, 39, 54, 55, 67] or relative representations like CaPE [29]. While useful for viewpoint control, these encodings lack 3D scene content. An alternative, explicit 3D spatial priors, builds 3D models (e.g., meshes, point clouds, 3DGS [25]) [11, 20, 32, 41, 44, 77] for re-projection and inpainting. This provides geometric grounding but suffers from error propagation from the initial 3D estimation [55, 77] and high computational cost. Instead, we utilize forward-warped images from future camera positions as a distinct explicit prior. These warped images serve as a dense, geometrically-grounded 2D hint, bypassing the error-prone and costly 3D reconstruction pipeline while offering a richer conditional signal than mere pose encoding.

### 3. Preliminaries

##### 3.1. Camera-Conditioned Video Generation

One major challenge in adding precise camera control to video diffusion models is finding a good way to represent 3D camera movement. Simply using raw camera intrinsics K and extrinsics E is often suboptimal, as their numerical values (e.g., translation t) are unconstrained and difficult for a network to correlate with visual content.

A more effective paradigm is to translate these abstract parameters into a dense, pixel-wise representation that provides a clearer geometric interpretation. For example, Plücker embeddings [56] define a 6D ray vector for each pixel. This transforms the abstract matrices into a dense tensor P ∈ Rn×6×h×w, which is much more informative for the diffusion model. This principle of using dense, geometrically-grounded priors is a key consideration for enabling fine-grained camera control.

##### 3.2. Diffusion Forcing and Non-Causal Priors

The Diffusion Forcing Transformer (DFoT) [57] paradigm reframes the noising operation as progressive masking, where each frame xt in a video is assigned an independent noise level kt ∈ [0,1]. This contrasts with conventional models that use a single noise level k for all frames. The model ϵˆθ is then trained on a per-frame noise prediction loss:

#### L = E

kT ,X,E

T

t=1

ϵ ˆθ(Xk,kT )t − ϵt 22 (1)

The critical advantage of this per-frame noise approach is that it enables a model to be trained with non-causal attention, learning to denoise a frame by conditioning on an arbitrary, partially-masked set of other frames.

This non-causal paradigm is particularly well-suited for our task. In typical video generation, a causal architecture is necessary as the future is unknown. However, in cameraconditioned novel view synthesis, we can generate a strong, geometry-consistent prior for all future frames simultaneously via forward-warping. These warped images provide a powerful non-causal conditioning signal. This insight is the foundation of our ST-Diff model, allowing us to discard restrictive causal constraints and employ a bidirectional, spatio-temporal diffusion strategy.

### 4. Method

##### 4.1. Spatio-Temporal Diffusion with Warped Priors

We address the task of novel view synthesis, where the goal is to generate a target view xt given a source view xs and corresponding camera poses {ps,pt}. To this end, we introduce Spatio-Temporal Diffusion with Warped Priors (ST-Diff), a bidirectional diffusion model designed for this task. Unlike causal, autoregressive video generation, where future frames are unknown, the camera-conditioned setting allows us to form a strong geometric prior for the target frame by projecting the source view. This key insight allows us to discard causal constraints and employ a more powerful bidirectional attention mechanism across all frames.

Our method first prepares geometric priors in the pixel space and then performs all diffusion, compositing, and noising operations in the latent space using a pre-trained VAE encoder E(·) and decoder D(·) [49, 60]. We use x to denote data in pixel-space and z for latent-space data.

One-to-all pixel-space warping. Given a training video sequence X = {xi}Ti=1, we first sample a single source frame xs from the sequence. We then create a full sequence of warped priors by warping this single source frame xs to every other frame’s viewpoint, including its own. To do this, we use pre-estimated depth maps Di and camera parameters (extrinsics Ei and intrinsics Ki) for all frames, obtained from a 3D geometry foundation model [9]. First, the source image xs and its depth Ds are unprojected into a 3D RGB point cloud Ps:

p(camu,v) = Ds(u,v) · K−s 1[u,v,1]T (2)

Ps = {(Esp(camu,v),xs(u,v))}u,v (3)

This single point cloud Ps is then rendered into all T target viewpoints using a differentiable point-based renderer. This "one-to-all" warping process yields two new sequences: a

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

!'→$

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

[Figure 66]

[Figure 67]

[Figure 68]

!!

[Figure 69]

!"#$%&$,"

[Figure 70]

[Figure 71]

[Figure 72]

)#," = !"#$%&$," ⊙ )$→" + RGB Point Cloud 1 − !"#$%&$," ⊙ )"

!$

###### 1) Spatially temporally-varying noisy latent. 2) Training ST-Diffusion.

- Figure 2. Training pipeline of our ST-Diff model. 1) Spatially temporally-varying noisy latent: The process begins by rendering a warped image and a validity mask from an RGB point cloud (images are shown for illustration, as operations are in latent space). The warped image is encoded to get zs→t, and the ground-truth image is encoded to get zt. A "clean composite" latent zc,t is created by combining the valid warped regions from zs→t with the blank regions from zt, using the downsampled mask Mlatent. 2) Training ST-Diffusion: This composite latent sequence is noised according to our spatio-temporal schedule, resulting in a noisy latent sequence (visualized as a stack) where the noise level for each latent varies across different frames and spatial regions. The resulting noisy latents are fed into our model Gθ, which is trained to predict the target velocity (defined as ϵt − zt), forcing it to learn the flow from the noisy composite latent back towards the original ground-truth latent sequence Z.

warped prior sequence, Xs→V = {xs→t}Tt=1, and a corresponding validity mask sequence, M = {Mt}Tt=1. Each mask Mt indicates which pixels in xs→t were successfully rendered from Ps.

Zc gets a different, independently sampled noise schedule. Second, at a spatial level, we apply different noise levels *within* each frame, distinguishing between the "warped" and "filled" regions. For each frame t, we therefore sample a pair of noise levels, (σwarped,t,σfilled,t). A spatially-varying noise map Σt is constructed using the latent-space mask:

Latent-space composite sequence. The training pipeline of our WorldWarp is illustrated in Fig. 2. With the pixelspace assets prepared, we move entirely to the latent space. We separately encode the new warped sequence Xs→V and the original ground-truth sequence X. We encode both:

Σt = Mlatent,t ⊙ σwarped,t + (1 − Mlatent,t) ⊙ σfilled,t (6)

We then generate the final noisy input sequence Znoisy = {znoisy,t}Tt=1 by sampling a noise sequence E = {ϵt}Tt=1 ∼ N(0,I):

Zs→V = {E(xs→t)}Tt=1 and Z = {E(xt)}Tt=1. (4)

znoisy,t = (1 − Σt) ⊙ zc,t + Σt ⊙ ϵt (7)

We also downsample the mask sequence M to match the latent dimensions, yielding Mlatent = {Mlatent,t}Tt=1. A clean composite latent sequence Zc is then created in the latent space. For each frame t, the composite zc,t takes its features from the warped latent zs→t in valid ("warped") regions and fills the remaining ("filled") regions with features from the ground-truth latent zt (which is the t-th element of Z):

A key architectural modification is required to process this spatiallyand temporally varying noise. Standard diffusion models [60] typically accept a single timestep embedding (e.g., shape B × 1) for an entire image or video chunk. Our ST-Diff model, however, is adapted to process a unique noise level for every token. We broadcast the noise map sequence ΣV = {Σt}Tt=1 to the full latent sequence dimensions (B × T × H′ × W′) and pass it through the time embedding network, thus generating a unique time-axis and spatial-axis embedding for each corresponding token.

zc,t = Mlatent,t ⊙zs→t +(1−Mlatent,t)⊙zt for t = 1...T

(5)

This entire sequence Zc = {zc,t}Tt=1 serves as the x0equivalent (clean signal) for the diffusion model.

Training objective. We train our ST-Diff model Gθ which takes the entire noisy sequence Znoisy, the sequence of noise maps ΣV, and other conditioning c (e.g., text, camera poses) as input. Critically, the model is trained to denoise the composite sequence Znoisy while regressing towards a target defined by the original ground-truth latent sequence Z. The

Spatially and temporally-varying noise. Our noising strategy extends the per-frame independent noise concept with a new, region-specific dimension, as shown in Fig. 2. The noise applied is varied at two levels simultaneously. First, at a temporal level, each frame t in the sequence

[Figure 73]

[Figure 74]

: Estimated camera poses from current video chunk : Extrapolated novel camera poses for next video chunk

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Describe what you see and what you may see later?

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

VLM

[Figure 86]

[Figure 87]

Prompt

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

[Figure 121]

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

[Figure 133]

!!

!!!"#

Initial Image or History Frames

Generated Novel Views

Forward-warping Images ST-Diff

[Figure 134]

3D Gaussian Splatting

[Figure 135]

- Figure 3. The autoregressive inference pipeline of WorldWarp. At each iteration k, the available history (either the initial images or the previously generated k − 1 chunk) is processed. First, TTT3R estimates camera poses and an initial 3D point cloud. This geometry is used to optimize a 3D Gaussian Splatting (3DGS) representation, which serves as a high-fidelity 3D cache. Concurrently, a VLM generates a descriptive text prompt, and novel camera poses are extrapolated for the next chunk. The optimized 3DGS renders forward-warped images at

these new poses. These warped priors, along with the VLM prompt, are fed into our non-causal ST-Diff model (Gθ) to denoise and generate the k-th chunk of novel views. The process then repeats, using the newly generated chunk as the history for the next iteration.

target velocity sequence is Vtarget = {ϵt − zt}Tt=1. Our training objective is the L2 loss, summed over the entire sequence:

L = EZ,Z

c,E,ΣV,c

T

t=1

∥vθ,t − (ϵt − zt)∥22 (8)

where Vθ = {vθ,t}Tt=1 = Gθ(Znoisy,ΣV,c). This loss forces the model to learn the complex relationship between the warped, GT-filled, and final target latents across the entire video.

- 4.2. Autoregressive Inference Pipeline

ing training, this 3DGS provides significantly higher-quality features for the non-blank (warped) regions, which is critical for maintaining geometric consistency.

Chunk-based Generation with ST-Diff. With the geometric and semantic conditioning prepared, we first render the sequence of prior images, Xs→V, from the 3DGS cache. These are encoded into latents Zs→V = {zs→t}Tt=1, and we also obtain the corresponding latent-space masks Mlatent = {Mlatent,t}Tt=1. Our goal is twofold: to fill in the blank (occluded) regions and to revise the non-blank (warped) regions, which may suffer from blur or distortion.

We achieve this by initializing the reverse diffusion process from a spatially-varying noise level, analogous to imageto-video translation. Let the full reverse schedule consist of N timesteps, from TN = 1000 down to T1 = 1. We define a strength parameter τ ∈ [0,1], which maps to an intermediate timestep Tstart and its corresponding noise level σstart. We set the noise level for the blank (filled) regions to σfilled = σT

The inference process is illustrated in Fig. 3. Our inference process generates novel view sequences autoregressively, producing a video chunk-by-chunk in a for-loop manner. Unlike training, which uses a fixed-radius point cloud representation, our inference pipeline leverages a dynamic, testtime optimized 3D representation as an explicit geometric cache. This process, illustrated in Fig. 3, integrates 3D Gaussian Splatting [25] (3DGS) for high-fidelity warping and a Vision-Language Model (VLM) [1] for semantic guidance.

, which corresponds to pure noise.

N

For each frame t, we construct a spatially-varying noise map Σstart,t using the latent-space mask:

Online 3D Geometric Cache. At the beginning of each iteration k of the generation loop, we take the available history (either the initial source views for k = 1 or the video chunk generated in the previous iteration k − 1). We first process these frames using a 3D geometry model (TTT3R) [9] to estimate their camera poses and an initial 3D point cloud. This point cloud is then used to initialize a 3D Gaussian Splatting (3DGS) representation, which we optimize for a few hundred steps (e.g., 200 steps) using the history frames and their estimated poses. This resulting online-optimized 3DGS model serves as an explicit, high-fidelity 3D representation cache. Compared to the fixed-radius point clouds used dur-

Σstart,t = Mlatent,t ⊙ σstart + (1 − Mlatent,t) ⊙ σfilled (9)

We then generate the initial noisy latent sequence Zstart = {zstart,t}Tt=1 for the reverse process. This is done by applying the noise map Σstart,t to the warped latent zs→t, using a sampled Gaussian noise ϵt:

zstart,t = (1 − Σstart,t) ⊙ zs→t + Σstart,t ⊙ ϵt (10)

This formulation effectively initializes the blank regions with pure noise (as σfilled ≈ 1.0) while applying a partial, strength-controlled noising to the warped regions.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

OursCameraCtrlGenWarpVMemOursCameraCtrlGenWarpVMem RealEstate10KDL3DV

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

1st frame 50th frame 100th frame 200th frame 1st frame 50th frame 100th frame 200th frame

Figure 4. Qualitative comparisons on the RealEstate10K [84] and DL3DV [36] datasets. We visualize videos generated by our method against those by GenWarp [55], CameraCtrl [16], and VMem [32]. Our WorldWarp generalizes to diverse camera motion, showcasing the spatial and temporal consistency.

Our ST-Diff model (Gθ) then takes this spatially-mixed latent sequence Zstart, the VLM text prompt, and the corresponding spatially-varying time embeddings as input. It denoises the sequence beginning from its spatially-varying timesteps (e.g., Tstart for warped regions and TN for blank regions) down to T1 to generate the k-th chunk of novel views. This newly generated chunk is then used as the history for the next iteration (k + 1), and the entire process repeats.

### 5. Experiments 5.1. Implementation Details.

We fine-tune WorldWarp based on Wan2.1-T2V-1.3B [60] model, with resolution 720x480 and batch size 8, on 8 H200 GPUs for 10k iterations. We apply TTT3R [9] as the 3D reconstruction foundation model for estimating camera parameters and depth maps. Please refer to the supplementary material for more details.

Datasets and evaluation metrics. We conduct experiments on two public scene-level datasets: RealEstate10K (Re10K) [84] and DL3DV [36] datasets. Our evaluation of

novel view synthesis quality comprises three main components: 1) Perceptual quality: We measure the distributional similarity between generated views and the test set using the Fréchet Image Distance (FID) [17]. 2) Detail preservation: Following [47], we assess the model’s ability to preserve image details across views by computing PSNR, SSIM [66], and LPIPS [81]. 3) Geometric alignment: We evaluate camera pose accuracy against the ground truth (Rgt, tgt), following [67]. We use DUST3R [64] to extract poses (Rgen, tgen) from generated views. We then compute the rotation distance (Rdist) and translation distance (tdist):

Rdist = arccos 0.5(tr(RgenRTgt) − 1) tdist = ∥tgt − tgen∥2,

where tr stands for the trace of a matrix. Per [16], estimated poses are expressed relative to the first frame, and translation is normalized by the furthest frame.

##### 5.2. Comparisons on the RealEstate10K Dataset

We present a comprehensive quantitative evaluation on the RealEstate10K dataset in Table 1, assessing generation quality (PSNR, LPIPS) and camera pose accuracy (Rdist, Tdist)

- Table 1. Quantitative comparison for single-view NVS on the RealEstate10K [84] dataset. We report performance for both short-term (50th frame) and long-term (200th frame) synthesis. For each metric, the best , second best , and third best results are highlighted. Our method significantly outperforms all baselines across most metrics, demonstrating superior quality and temporal consistency.

Short-term (50thframe) Long-term (200thframe)

PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Rdist ↓ Tdist ↓ PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Rdist ↓ Tdist ↓ InfiniteNature [37] 14.12 0.192 0.428 27.35 0.988 0.521 10.07 0.109 0.658 46.12 1.234 0.902 InfiniteNature-Zero [33] 14.31 0.201 0.409 26.98 0.931 0.502 10.22 0.117 0.647 45.71 1.201 0.885 GeoGPT [48] 13.54 0.186 0.437 26.52 0.732 0.413 9.67 0.089 0.664 41.52 1.112 0.876 Lookout [47] 14.63 0.216 0.372 28.67 1.221 0.864 10.73 0.163 0.647 57.82 1.331 0.912 PhotoNVS [76] 15.76 0.247 0.324 21.62 0.735 0.464 11.76 0.176 0.573 41.67 1.273 0.628 GenWarp [55] 13.21 0.252 0.428 29.51 0.553 0.059 9.72 0.192 0.601 36.12 1.136 0.446 MotionCtrl [67] 14.14 0.258 0.327 19.12 0.336 0.353 9.26 0.187 0.593 35.21 1.134 0.697 CameraCtrl [16] 14.97 0.271 0.311 20.07 0.308 0.267 11.16 0.183 0.584 35.07 1.206 0.704 ViewCrafter [77] 17.23 0.279 0.367 22.21 1.242 0.201 9.96 0.157 0.578 33.82 1.571 0.814 SEVA [83] 18.67 0.394 0.281 17.14 0.259 0.116 13.24 0.227 0.443 28.47 1.112 0.731 VMem [32] 18.19 0.403 0.273 16.97 0.221 0.043 14.91 0.223 0.471 25.17 1.132 0.494 DFoT [57] 18.53 0.439 0.265 17.27 0.326 0.318 15.21 0.245 0.418 24.85 1.643 0.835

Ours 20.32 0.527 0.216 15.56 0.188 0.039 17.13 0.281 0.352 21.75 0.697 0.203

for short-term (50th frame) and long-term (200th frame) synthesis. Our method achieves state-of-the-art results, outperforming all baselines across all 12 metrics. This advantage is most pronounced in the challenging long-term setting: while most methods suffer significant quality degradation, our model maintains the highest PSNR (17.13) and LPIPS

consistency and mitigating severe camera drift on complex, long-range trajectories. Visualizations are in Fig. 4 and the supplementary material.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

WarpedValidityOursGT

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

- (0.352), surpassing strong competitors like SEVA, VMem, and DFoT. This high fidelity is crucial, as pose estimation (using Master3R) fails on the low-quality or blurry outputs from baselines. Consequently, our model achieves the lowest

long-term pose error (Rdist 0.697, Tdist 0.203). This highlights a clear distinction: camera-embedding methods (MotionCtrl, CameraCtrl) suffer severe pose drift, and while 3D-aware methods (GenWarp, VMem) are more stable, our spatial-temporal noise diffusion strategy significantly surpasses both, proving its superior ability to mitigate cumulative camera drift. Qualitative results are in Fig. 4 and the supplementary.

- 5.3. Comparisons on the DL3DV Dataset

We further validate our model on the more challenging DL3DV dataset in Tab. 2. Despite the complex trajectories degrading performance for all methods, our model maintains a commanding lead in all 12 metrics, demonstrating superior robustness. In the demanding long-term (200th frame) setting, our model’s PSNR (14.53) decisively outperforms the next-best competitors, DFoT (13.51) and VMem (12.28). This fidelity is again proven critical for pose accuracy. On this complex dataset, our model remains the most stable, achieving the lowest Rdist (1.007) and Tdist (0.412). The weaknesses of competing approaches are magnified here, as 3D-aware methods like GenWarp (1.351Rdist) and VMem

- (1.419Rdist) lose stability. This proves our spatial-temporal noise diffusion strategy is more effective at preserving 3D

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

Figure 5. Illustration of the ST-Diff’s generating process. We illustrate the GT images, the warped images which serve as the condition for ST-Diff, the corresponding validity mask, and our final generated frames. The comparisons show that our ST-Diff successfully fills in the blank areas (initialized from a full noise level) while simultaneously revising distortions and enhancing details in the non-blank regions (initialized from a partial noise level) during the diffusion process.

##### 5.4. Ablation Study

We conduct ablation studies on the RealEstate10K dataset in Table 3 to validate our two core design choices: the 3DGSbased cache and the spatial-temporal noise diffusion model. Caching Mechanism. We first analyze the effect of our caching module. The "No Cache" baseline, which relies only on the initial image, fails completely in long-term generation, with PSNR dropping to 9.22. This confirms the necessity of a 3D cache for long-range synthesis. We then compare our full model, "Caching by online optimized 3DGS," against "Caching by RGB point cloud." Although our model is trained on warped point clouds (with unoptimized, uniform radii), using a simple point cloud cache at

- Table 2. Single-view NVS on DL3DV dataset [36] Short-term evaluation is on the 50th frame, and long-term is on frames 200th. This dataset is significantly more challenging due to complex camera trajectories and diverse environments. All methods show a noticeable performance drop compared to RealEstate10K [84]. For each metric, the best , second best , and third best results are highlighted.

Short-term (50thframe) Long-term (200thframe)

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Rdist ↓ Tdist ↓ PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Rdist ↓ Tdist ↓ InfiniteNature [37] 10.05 0.112 0.662 51.45 1.478 1.131 8.98 0.100 0.695 54.32 1.561 1.522 InfiniteNature-Zero [33] 10.21 0.121 0.648 51.05 1.432 1.109 9.12 0.107 0.685 53.95 1.528 1.501 GeoGPT [48] 9.83 0.096 0.688 50.12 1.553 1.407 8.52 0.081 0.773 55.24 1.851 1.703 Lookout [47] 11.14 0.131 0.609 69.53 1.252 1.058 9.91 0.117 0.678 75.06 1.354 1.552 PhotoNVS [76] 12.02 0.147 0.558 48.03 1.404 1.306 10.83 0.132 0.609 52.51 1.708 1.602 GenWarp [55] 12.87 0.201 0.677 44.04 0.952 0.381 8.63 0.092 0.749 48.13 1.351 0.953 MotionCtrl [67] 13.34 0.192 0.698 43.11 0.863 0.724 8.12 0.087 0.779 47.54 1.452 1.161 CameraCtrl [16] 13.62 0.212 0.573 32.53 0.921 0.832 10.24 0.127 0.623 46.92 1.523 0.924 ViewCrafter [77] 16.17 0.226 0.598 31.02 1.304 0.953 8.97 0.112 0.649 45.23 1.651 1.052 SEVA [83] 16.63 0.331 0.469 31.04 1.203 0.851 12.16 0.181 0.508 36.03 1.422 0.954 VMem [32] 16.98 0.348 0.458 31.52 0.854 0.352 12.28 0.197 0.502 35.52 1.419 0.858 DFoT [57] 16.13 0.372 0.402 32.76 1.139 0.570 13.51 0.233 0.471 33.58 1.685 1.144

Ours 18.10 0.432 0.315 28.03 0.433 0.086 14.53 0.241 0.413 29.21 1.007 0.412

- Table 3. Ablation studies on the RealEstate10K [84] dataset. We analyze the impact of our caching mechanism (top) and the spatialtemporal noise design (bottom).

Short-term (50thframe) Long-term (200thframe) PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Rdist ↓ Tdist ↓ PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Rdist ↓ Tdist ↓

No Cache 14.19 0.255 0.331 19.24 — — 9.22 0.175 0.598 35.30 — Caching by RGB point cloud 15.12 0.374 0.269 16.95 0.192 0.045 11.12 0.245 0.412 28.98 0.703 0.252 Caching by online optimized 3DGS 20.32 0.527 0.216 15.56 0.188 0.039 17.13 0.281 0.352 21.75 0.697 0.203

Full sequence noise 17.08 0.282 0.364 22.31 1.235 0.208 9.92 0.160 0.580 33.89 1.574 0.817 Spatial-varying noise 18.23 0.375 0.305 18.91 0.232 0.094 13.95 0.210 0.492 31.12 1.040 0.595 Temporal-varying noise 18.09 0.317 0.298 19.75 0.258 0.112 13.20 0.196 0.513 32.01 1.209 0.701 Spatial-temporal-varying noise 20.32 0.527 0.216 15.56 0.188 0.039 17.13 0.281 0.352 21.75 0.697 0.203

inference ("Caching by RGB point cloud") yields significantly lower performance (11.12 PSNR) than our full model (17.13 PSNR). This demonstrates that using an online optimized 3DGS as the cache provides a much more robust and high-fidelity 3D representation. Notably, this 3DGS optimization is highly efficient, requiring only 500 steps per chunk. This result confirms that despite the modality gap between training (point clouds) and inference (3DGS), the superior representation quality of 3DGS leads to a substantial improvement in both generation quality and pose accuracy.

Noise Diffusion Model. The bottom half of the table validates our spatial-temporal noise diffusion design. Using a "Full sequence noise" (i.e., a standard video diffusion model) results in poor generation quality (9.92 long-term PSNR) and, critically, a catastrophic loss of camera control (1.574 Rdist). When using only "Spatial-varying noise," we observe a dramatic improvement in camera accuracy (Rdist improves from 1.574 to 1.040), confirming that spatial noise is key for precise camera conditioning. Conversely, using only "Temporal-varying noise" improves generation quality (13.20 long-term PSNR) but fails to control the camera (1.209 Rdist). Our full "Spatial-temporal-varying noise"

model successfully combines both benefits, achieving the best generation quality (17.13 PSNR) and the best camera accuracy (0.697 Rdist), demonstrating the necessity and efficacy of our proposed noise diffusion strategy.

Table 4. Breakdown of latency and model size for each component in our pipeline. Times are in seconds (s).

Estimating 3D (TTT3R)

Optimizing 3DGS

Forward warping

ST-Diff 50 steps Total

VLM Prompting

Inference time (s) 3.5 5.8 2.5 0.2 42.5 54.5

Inferencing efficiency. We provide a detailed breakdown of the inference latency per video chunk in Table 4. The average total time to generate one chunk (49 frames) is 54.5 seconds. The primary computational bottleneck is the iterative denoising process of our spatial-temporal diffusion model (ST-Diff), which requires 42.5 seconds for 50 steps, accounting for approximately 78% of the total time. In contrast, all 3D-related components are highly efficient: estimating the initial 3D representation with TTT3R takes 5.8s, optimizing the 3DGS cache takes only 2.5s, and forward warping is near-instant at 0.2s. This analysis demonstrates that the 3D-aware caching and conditioning, while critical

for quality and consistency, add only a minimal computational overhead (8.5s total) compared to the main generative backbone.

### 6. Conclusion

In this work, we propose WorldWarp, a novel autoregressive framework for long-range, geometrically-consistent novel view extrapolation. Our method is designed to overcome the key limitation of prior work: the inability of standard generative models to handle imperfect 3D-warped priors. We introduce the ST-Diff model, a non-causal diffusion model trained with a spatially-temporally-varying noise schedule. This design explicitly trains the model to solve the fill-andrevise problem, simultaneously filling blank regions from pure noise while revising distorted content from a partiallynoised state. By coupling this model with an online 3D geometric cache to avoid irreversible error propagation, WorldWarp achieves state-of-the-art performance, setting a new bar for long-range, camera-controlled video generation.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5
- [2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855–5864, 2021. 2
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697– 19705, 2023. 2
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. 2
- [7] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack ParkerHolder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. 2024. 2
- [8] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion

- forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024. 2
- [9] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645, 2025. 2, 3, 5, 6, 1
- [10] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025. 2
- [11] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. Advances in Neural Information Processing Systems, 36:39897– 39914, 2023. 2, 3
- [12] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024. 2
- [13] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 2
- [14] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. In ECCV, 2024. 2
- [15] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 2
- [16] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 3, 6, 7, 8
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [18] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey A. Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. ArXiv, abs/2210.02303, 2022. 2
- [19] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 2
- [20] Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7909–7920, 2023. 2, 3
- [21] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868,

2022. 2

- [22] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and

- Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024. 2
- [23] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009,

2025. 2

- [24] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025. 2
- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 1, 2, 3, 5
- [26] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. In NeurIPS, 2024. 3
- [27] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. 2024. 2
- [28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [29] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9503–9513, 2024. 2, 3
- [30] Xin Kong, Daniel Watson, Yannick Strümpler, Michael Niemeyer, and Federico Tombari. Causnvs: Autoregressive multi-view diffusion for flexible 3d novel view synthesis. arXiv preprint arXiv:2509.06579, 2025. 2
- [31] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20775–20785, 2024. 2
- [32] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025. 2, 3, 6, 7, 8
- [33] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European conference on computer vision, pages 515–534. Springer,

2022. 2, 7, 8

- [34] Zongyi Li, Shujie Hu, Shujie Liu, Long Zhou, Jeongsoo Choi, Lingwei Meng, Xun Guo, Jinyu Li, Hefei Ling, and Furu Wei. Arlon: Boosting diffusion transformers with autoregressive models for long video generation. In ICLR, 2025. 2
- [35] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 2

- [36] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160– 22169, 2024. 6, 8, 1, 2
- [37] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14458–14467, 2021. 2, 7, 8
- [38] Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan C Pérez, Ding Liu, Kumara Kahatapitiya, Menglin Jia, et al. Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280, 2024. 2
- [39] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298– 9309, 2023. 2, 3
- [40] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [41] Norman Müller, Katja Schwarz, Barbara Rössle, Lorenzo Porzi, Samuel Rota Bulo, Matthias Nießner, and Peter Kontschieder. Multidiff: Consistent novel view synthesis from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10258–10268, 2024. 2, 3
- [42] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5480–5490, 2022. 2
- [43] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 2

- [44] Stefan Popov, Amit Raj, Michael Krainin, Yuanzhen Li, William T Freeman, and Michael Rubinstein. Camctrl3d: Single-image scene exploration with precise 3d camera control. In 2025 International Conference on 3D Vision (3DV), pages 649–658. IEEE, 2025. 2, 3
- [45] Shuhuai Ren, Shuming Ma, Xu Sun, and Furu Wei. Next block prediction: Video generation via semi-auto-regressive modeling. arXiv preprint arXiv:2502.07737, 2025. 2
- [46] Weining Ren, Zihan Zhu, Boyang Sun, Jiaqi Chen, Marc Pollefeys, and Songyou Peng. Nerf on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8931–8940, 2024. 2

- [47] Xuanchi Ren and Xiaolong Wang. Look outside the room: Synthesizing a consistent long-term 3d scene video from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3563–3573,

2022. 6, 7, 8

- [48] Robin Rombach, Patrick Esser, and Björn Ommer. Geometryfree view synthesis: Transformers and no 3d priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14356–14366, 2021. 2, 7, 8
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3
- [50] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models. 2024. 3
- [51] Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin, David J Fleet, and Andrea Tagliasacchi. Robustnerf: Ignoring distractors with robust losses. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20626–20636, 2023. 2
- [52] Sara Sabour, Lily Goli, George Kopanas, Mark Matthews, Dmitry Lagun, Leonidas Guibas, Alec Jacobson, David Fleet, and Andrea Tagliasacchi. Spotlesssplats: Ignoring distractors in 3d gaussian splatting. ACM Transactions on Graphics, 44

(2):1–11, 2025. 2

- [53] Sand-AI. Magi-1: Autoregressive video generation at scale,

2025. 3

- [54] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360degree view synthesis from a single real image. 2023. 2, 3
- [55] Junyoung Seo, Kazumi Fukuda, Takashi Shibuya, Takuya Narihira, Naoki Murata, Shoukang Hu, Chieh-Hsin Lai, Seungryong Kim, and Yuki Mitsufuji. Genwarp: Single image to novel views with semantic-preserving generative warping. Advances in Neural Information Processing Systems, 37: 80220–80243, 2024. 2, 3, 6, 7, 8
- [56] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34:19313– 19325, 2021. 3
- [57] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 3, 7, 8
- [58] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In CVPR, 2025. 3
- [59] R Villegas, H Moraldo, S Castro, M Babaeizadeh, H Zhang, J Kunze, PJ Kindermans, MT Saffar, and D Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In ICLR, 2023. 2
- [60] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao

- Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 4, 6, 1
- [61] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [62] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9065–9076,

2023. 2

- [63] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 2
- [64] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024. 6
- [65] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 2
- [66] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [67] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2, 3, 6, 7, 8
- [68] Dirk Weissenborn, Oscar Täckström, and Jakob Uszkoreit. Scaling autoregressive video models. In ICLR, 2020. 2
- [69] Wenming Weng, Ruoyu Feng, Yanhui Wang, Qi Dai, Chunyu Wang, Dacheng Yin, Zhiyuan Zhao, Kai Qiu, Jianmin Bao, Yuhui Yuan, et al. Art-v: Auto-regressive text-to-video generation with diffusion models. In CVPR, 2024. 2
- [70] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. arXiv preprint arXiv:2410.08151,

2024. 3

- [71] Jiacong Xu, Yiqun Mei, and Vishal Patel. Wild-gs: Realtime novel view synthesis from unconstrained photo collections. Advances in Neural Information Processing Systems, 37:103334–103355, 2024. 2
- [72] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2
- [73] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2

- [74] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In ICLR, 2025. 2
- [75] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 2
- [76] Jason J Yu, Fereshteh Forghani, Konstantinos G Derpanis, and Marcus A Brubaker. Long-term photometric consistent novel view synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7094–7104, 2023. 2, 7, 8
- [77] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 2, 3, 7, 8
- [78] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456,

2024. 2

- [79] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li, Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, pages 341–359. Springer, 2024. 2
- [80] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025. 3
- [81] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [82] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025. 2
- [83] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489, 2025. 2, 7, 8
- [84] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 6, 7, 8, 1, 2
- [85] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145–163. Springer, 2024. 2

## WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion Supplementary Material

### 7. Implementation Details

Training. We implement our WorldWarp using the Wan2.1T2V-1.3B diffusion transformer [60] as the generative backbone. All parameters of the model are fully fine-tuned in an end-to-end manner. The model is trained for 10k steps with a global batch size of 64 (8 per GPU) on 8 NVIDIA H200 GPUs. We utilize the AdamW optimizer with a learning rate of 5 × 10−5 and apply a 1,000-step linear warmup. The training video resolution is set to 480 × 720.

Inference. The video generation process is initiated from a single source image. Subsequent content is synthesized auto-regressively in chunks of 49 frames. To ensure temporal continuity, we utilize a fixed context overlap of 5 frames for every iteration following the initial chunk. To establish the global coordinate system, we first estimate camera poses and intrinsics from the reference video using TTT3R [9]. For generation lengths exceeding the reference trajectory, we employ a velocity-based extrapolation strategy, computing linear velocity for translation and Spherical Linear Interpolation (SLERP) for rotation based on a 20-frame smoothing window. During the generation of each chunk, we optimize the online 3DGS cache for 500 iterations with a learning rate of 1.6 × 10−3 to render the warped priors Xs→V.

We utilize the spatially-temporally schedule described in Fig. 3 in the main text for the reverse diffusion process. Specifically, the latent representations of the 5 context overlap frames are enforced as hard constraints. For the target frames, we set the strength parameter τ = 0.8. Consequently, regions with valid geometric warps (Mlatent = 1) are initialized with a reduced noise level σstart corresponding to τ, preserving the structural integrity of the 3D cache. Conversely, occluded or blank regions (Mlatent = 0) are initialized with standard Gaussian noise (σfilled = σT

N ≈ 1.0) to facilitate generative inpainting. The diffusion model employs a Flow Match Euler Discrete Scheduler with 50 denoising steps.

### 8. More Experiment Results 8.1. Visualization Results

We illustrate more results on the RealEstate10K [84] and DL3DV [36] datasets in Fig. 6 and Fig. 7. Please refer to video supplementary material for more results.

To further demonstrate the robust generalization capacity of ST-Diff, we extend our evaluation beyond standard photorealistic benchmarks to scenes rendered in diverse artistic styles in Fig. 8. By prompting the model with specific stylistic descriptors, such as "Van Gogh style" or "Studio Ghibli style," we generate a variety of stylized video sequences.

The results illustrate that our method successfully synthesizes these highly stylized scenes while strictly preserving the underlying 3D geometric consistency. These qualitative results validate that our proposed training strategy effectively integrates fine-grained geometric control without sacrificing the rich semantic and aesthetic generalization capabilities inherent in the pre-trained model. This confirms that adapting the foundation model into an asynchronous diffusion framework does not compromise its ability to interpret opendomain text prompts.

##### 8.2. Analysis of Spatially-Adaptive Noise Dynamics

To validate the effectiveness of our geometry-aware inference strategy, we visualize the evolution of the noise schedule matrix ΣV throughout the reverse diffusion process. As shown in Fig. 9, the visualization is structured as a spatiotemporal grid where each row corresponds to a latent temporal token t (derived from the 49 video frames via VAE encoding) and columns progress through the denoising steps from left (T = 999) to right (T = 0).

The map explicitly corroborates our dual-schedule formulation. The top two rows, representing the 5 history context frames (corresponding to ∼2 latent tokens), remain fully constrained with zero noise (dark purple) throughout the process, ensuring seamless transitions from previous chunks. In the subsequent 11 rows (the generated tokens), we observe a distinct spatial modulation. The valid geometric regions, projected from the 3DGS cache, are maintained at a reduced noise level τ (intermediate green/teal) to preserve high-fidelity structural details. In contrast, occluded or blank regions are initialized with high-variance noise (yellow) to facilitate the generative hallucination of new content. This confirms that our model effectively balances geometric preservation with generative inpainting during the autoregressive process.

### 9. Limitations

Despite the effectiveness of ST-Diff in generating geometrically consistent long-term videos, our method is subject to certain limitations common to autoregressive video generation frameworks.

Error Accumulation in Long-horizon Generation. Although our model is trained in an asynchronous diffusion manner, where we apply varying noise strengths to different frames and spatial regions to mimic inference conditions, generating infinite-length video sequences with perfect fidelity remains an unresolved challenge. In our autoregressive

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

1st frame 50th frame 100th frame 150th frame 200th frame 225th frame

Figure 6. Qualitative results on the RealEstate10K [84] datasets.

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

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

1st frame 50th frame 100th frame 150th frame 200th frame 225th frame

Figure 7. Qualitative results on the DL3DV [36] datasets.

Dependency on Geometric Priors. Our method operates on the premise that forward-warped images provide strong geometric hints for generation. Therefore, our performance is heavily dependent on the accuracy of the upstream 3D geometry foundation models, such as TTT3R [9] or VGGT [63], used for depth and camera pose estimation. In scenarios where these pre-trained estimators struggle, including complex outdoor environments with extreme lighting,

pipeline, the generated output of one chunk serves as the historical context for the next. Consequently, minor visual artifacts or geometric inconsistencies can propagate and accumulate over time. For extremely long sequences, such as those exceeding 1000 frames, this drift can eventually lead to degradation in visual quality or geometric stability. This remains a persistent issue shared by state-of-the-art video generation methods.

[Figure 272]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 278]

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

[Figure 293]

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

1st frame 50th frame 100th frame 150th frame 200th frame 225th frame

- Figure 8. Generalization to Out-of-Distribution Artistic Styles. We evaluate the robustness of ST-Diff by generating video sequences conditioned on diverse artistic text prompts (e.g., “Van Gogh style”, “Studio Ghibli style”). The visualized chunks demonstrate that our method successfully synthesizes high-quality stylized content while maintaining rigorous 3D geometric consistency across the autoregressive generation. This confirms that our geometry-aware fine-tuning strategy effectively incorporates structural control without compromising the semantic and aesthetic generalization capabilities of the pre-trained diffusion backbone.

transparency, or lack of texture, the estimated depth maps and poses may be inaccurate. This results in incorrect warping results that deviate significantly from the true geometry. While ST-Diff is designed to correct artifacts, it may fail to recover high-quality frames when the geometric guidance is fundamentally flawed or contains excessive noise.

[Figure 308]

- History tokens 0
- History tokens 1

Generated tokens 0

Generated tokens 11

Denoising Step

- Figure 9. Visualization of the Spatially-Adaptive Noise Schedule. We visualize the schedule matrix ΣV across the reverse diffusion process. The horizontal axis represents the progression of denoising steps (from T = 999 to 0), while the vertical axis corresponds to the sequence of temporal tokens (13 tokens derived from 49 frames). The top two rows represent the History Tokens (context), which are enforced as hard constraints (dark purple, σ = 0). The subsequent rows represent the Generated Tokens, where the noise levels are spatially modulated: (1) Valid warped regions are held at a reduced noise level τ (intermediate green) to preserve explicit geometry; and (2) Occluded regions are initialized with maximal noise (yellow, σ ≈ 1.0) to enable the generative synthesis of novel content.

