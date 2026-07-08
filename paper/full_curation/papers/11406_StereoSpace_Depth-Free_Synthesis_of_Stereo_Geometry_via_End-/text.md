## StereoSpace: Depth-Free Synthesis of Stereo Geometry via End-to-End Diffusion in a Canonical Space

# arXiv:2512.10959v3[cs.CV]29Apr2026

Tjark Behrens1 Anton Obukhov3 Bingxin Ke1 Fabio Tosi2 Matteo Poggi2 Konrad Schindler1

1ETH Zurich 2University of Bologna 3HUAWEI Bayer Lab Project page: https://hf.co/spaces/prs-eth/stereospace

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

Figure 1. StereoSpace for generating stereo from monocular images. Left: Built on a foundational LDM, our framework efficiently leverages learned priors for end-to-end view synthesis. The target baseline in world units acts as conditioning for precise view control. Right: Implicit scene understanding allows us to tackle the most complex cases where geometry cues alone are insufficient for novel view synthesis. Best viewed zoomed-in. Legend: □ (warping), ⃝ (breaks), — (bends), → (ghosting). StereoSpace consistently outperforms recent monocular competition, including generative 3DGS models like Lyra [2].

#### Abstract

We introduce StereoSpace, a diffusion-based framework for monocular-to-stereo synthesis that models geometry purely through viewpoint conditioning, without explicit depth or warping. A canonical rectified space and the conditioning guide the generator to infer correspondences and fill disocclusions end-to-end. To ensure fair and leakage-free evaluation, we introduce an end-to-end protocol that excludes any ground truth or proxy geometry estimates at test time. The protocol emphasizes metrics reflecting downstream relevance: iSQoE for perceptual comfort and MEt3R for geometric consistency. StereoSpace surpasses other methods from the warp & inpaint, latent-warping, and warpedconditioning categories, achieving sharp parallax and strong robustness on layered and non-Lambertian scenes. This establishes viewpoint-conditioned diffusion as a scalable, depth-free solution for stereo generation.

#### 1. Introduction

Stereo imaging provides a principled way to infer threedimensional structure from two-dimensional observations

captured from slightly displaced viewpoints.

As this mechanism mirrors human depth perception, stereo imaging also represents a key technology for spatially immersive entertainment. In 3D cinema, virtual reality, and augmented reality, presenting slightly offset stereo images to human eyes stimulates binocular disparity and produces a convincing illusion of depth for the viewer. This exploitation of a biologically grounded perception mechanism enhances visual realism and audience engagement, making stereo imaging not only a tool for 3D reconstruction but also a cornerstone in spatially immersive visual media.

However, acquiring high-quality stereo imagery for entertainment purposes can be both costly and technically demanding, as it requires precise camera alignment, synchronization, and calibration, with even minor mismatches possibly leading to visual discomfort or distortion in the perceived 3D effect. Consequently, generating stereo pairs out of single monocular images has emerged as an attractive alternative for reducing production costs and potentially enable conversion of any 2D content into 3D media.

The simplest and most intuitive way to achieve this is to recover the depth of the single image and use it to project its pixels over another camera frame placed on its right, e.g., by a simple forward warping operation [59, 86, 90].

This approach assumes the availability of a reliable and generalizable depth estimator, which is a reasonable requirement given the latest advances in single-image depth estimation [88, 99]. As a result, the stereo generation process becomes akin to an inpainting task, aimed at filling empty pixels after warping rather than a 3D generative task. However, we argue that this shortcut exposes the entire generation process to failure cases inherited from the depth estimator itself. Among these, the presence of multiple depth layers within a scene – occurring, for instance, in the presence of glass or transparent surfaces – represents a particularly severe challenge, given the inability of depth estimation models to deal with such complex structures.

Driven by these arguments, in this paper we reformulate the stereo generation problem as an image-conditioned 3D generative task by getting rid of depth estimation as a preliminary requirement. Inspired by the recent success achieved by the simple repurposing of diffusion-based image generators for dense predictive tasks [37, 38], we follow the same path and design a novel depth-free framework for stereo image generation. The key to compensating for the lack of structural guidance provided by depth is a proper encoding of relative displacements between viewpoints in a metric, posecanonicalized frame. Relative extrinsics and dense intrinsics encodings define a canonical StereoSpace, which suggests the name for our framework. This lets users set the stereo baseline B directly in physical units at inference, yielding predictable control and generalization across baselines, as shown in Fig. 1 on the left.

We build StereoSpace around the rich text-to-image generative prior of Stable Diffusion [67]. A mixture of off-theshelf single-baseline stereo datasets and custom-made multibaseline datasets, rendered by means of novel view synthesis techniques [43, 83], ensures efficient transfer of the generative prior to the downstream task. To assess StereoSpace’s generation quality, we establish a novel evaluation protocol based on iSQoE [78], aimed at quantifying perceptual comfort, and MEt3R [1], to evaluate geometric consistency of the generated image with the input. This evaluation is carried out on four real-world stereo datasets, covering both indoor scenes and outdoor driving scenarios, as well as multi-layered structures where conventional, depth-based approaches struggle, as shown in the right part of the teaser figure. Our results confirm the effectiveness of StereoSpace in generating stereo images despite the lack of explicit depth estimation performed in advance.

Our main contributions can be summarized as follows:

- • StereoSpace for single-image conditional generation of counterpart views, free from explicit geometric shortcuts
- • End-to-end training procedure for efficient transfer of the rich task-agnostic foundation prior to the task at hand
- • A novel perceptual and geometry-aware evaluation

#### 2. Related Work

Novel View Synthesis (NVS). Recent breakthroughs in novel view synthesis are rooted in neural rendering and implicit and explicit 3D scene representations. Neural Radiance Fields (NeRFs) introduced implicit neural representations by mapping spatial coordinates to density and radiance, rendered via differentiable volume integration. Despite capturing fine geometry and view-dependent effects, NeRFs require slow per-scene optimization and dense ray marching. Subsequent work improved efficiency [13, 55], antialiasing [5, 6], sparse-view reconstruction [12, 102], and dynamic scene handling [20, 56, 58]. NeRFs trained from monocular sequences have also been exploited to generate stereo training data by rendering rectified stereo pairs with controllable baselines [83]. 3D Gaussian Splatting (3DGS) [39], instead, introduced an explicit representation based on anisotropic Gaussian primitives rendered by efficient rasterization, enabling real-time performance. Building on this, recent works enhanced geometry [28], appearance [31], dynamics [49], and generalization [11, 15]. While these methods aim at general novel view synthesis from arbitrary views in a multi-view setting, our work focuses on synthesizing a single translationally-shifted viewpoint (the stereo pair) from a monocular image using a diffusion model.

Diffusion Models for Vision Tasks. Diffusion Models (DM) [25, 98] have transformed generative modeling, achieving state-of-the-art results in image [17] and video synthesis [7, 26]. Latent Diffusion Models (LDMs) [66] further improve efficiency by operating in compressed latent space, while conditional variants such as ControlNet [105] and T2IAdapter [54] enable fine-grained control through structural inputs (e.g., depth or edges). Beyond generation, DMs have been adapted for structured prediction, including monocular depth or surface normals [19, 37, 38, 60], segmentation [4, 62, 80, 96], object detection [14, 23], and inpainting [48]. For multi-view generation, existing methods generate multiple consistent views that span object-centric [45, 73] and scene-level [68, 79] settings. Some approaches [21] reconstruct multi-view images into explicit 3D representations via expensive per-scene optimization, while cameracontrolled video diffusion models [3, 22, 91, 95, 108] enable viewpoint conditioning, with recent works adopting Pl¨ucker coordinates for pixel-wise control; feed-forward 3D models [2, 44] then distill scene knowledge from such video priors. Following the paradigm of camera-conditioned diffusion, we adapt LDMs for stereo generation, synthesizing geometrically consistent stereo pairs through a dual-stream architecture conditioned on Pl¨ucker ray embeddings.

Monocular-to-Stereo Generation. Research on monocularto-stereo synthesis has evolved from explicit geometry-based pipelines to diffusion-driven generation. Early work, such as Deep3D [94], introduced end-to-end neural architectures

trained on stereo pairs to predict disparity-like maps for direct view synthesis. Recent diffusion-based methods, instead, can be categorized into distinct paradigms. Warp-andinpaint approaches [16, 29, 50, 74, 90, 104, 106] estimates disparity from monocular depth, forward-warp the input image, and inpaint disoccluded regions using diffusion priors, with [103] focusing specifically on tiny-baseline setups. Latent warping [72, 86] performs disparity shifts directly in diffusion latent space without explicit training, while Warped conditioning [59] applies warping to canonical coordinate embeddings that condition the denoising process. Trainingfree variants [16], instead, operate without fine-tuning. Other extensions include stereo matching through synthesis [90] and multi-baseline depth estimation [46]. Our work differs fundamentally by learning stereo geometry directly via viewpoint conditioning in canonicalized space, enabling metric baseline control through viewpoint conditioning, and achieving depth-free handling of complex multi-layer scenes with strong cross-baseline generalization.

#### 3. Method

We consider a standard rectified pinhole stereo setting with known intrinsics f and a metric baseline B. In rectified stereo, epipolar lines are horizontal and disparities vary only along the image x-axis. Hence, the mapping from source to target view is fully determined by the relative calibration (f,B), rather than by the absolute camera poses in the world. Put differently, in stereo view synthesis, the critical variable is the inter-camera geometry, not absolute position in space.

To exploit this, we introduce StereoSpace that canonicalizes any rectified stereo pair by re-expressing its extrinsics in a shared frame. The teaser figure (Fig. 1, left) visualizes our formulation of StereoSpace: the center of the stereo rig is fixed at the origin and the two cameras are constrained to lie on the x-axis, separated by the baseline B. This creates a common metric baseline along which all cameras move horizontally. Canonicalization in this space concentrates the training distribution: the model no longer needs to explain variation due to arbitrary world poses and can instead focus on the stereo-induced appearance changes and epipolar-consistent correspondences.

Unlike warping-based pipelines, we avoid building an intermediate depth volume. We argue that the representational capacity of a diffusion model is sufficient to learn stereo view synthesis directly in our StereoSpace. In practice, the generative model is conditioned on the source image and the relative calibration (f,B) and predicts the target view. This yields a depth-free formulation: geometry is injected through the conditioning variables and the canonical frame, not through an explicit 3D reconstruction.

##### 3.1. Viewpoint Conditioning

We formulate stereo view generation as conditional diffusion: given a rectified source view Is, the model synthesizes the paired target view It at the horizontally displaced viewpoint, conditioned on the known stereo configuration. We follow the standard latent diffusion setup and train with the velocity parameterization used in recent work [36, 67]. Concretely, for a noisy latent zt of the target, we minimize:

s,It),ϵ,t ∥v − vθ (zt,t;Is,Φ)∥2 , (1) where v is the ground truth velocity for the chosen noise schedule, vθ is the predicted velocity, and Φ denotes the viewpoint-conditioning signal.

Lvel = E(I

The purpose of Φ is to impose our canonical StereoSpace by providing exact camera control. We add this viewpoint conditioning Φ as Pl¨ucker embeddings in the diffusion process. Each camera ray is represented as a normalized 6D Pl¨ucker vector (moment and direction) that encodes the intrinsics and extrinsics along that sightline while being invariant to translations along the ray [30]. Distances in the StereoSpace are kept metric. Hence, Pl¨ucker rays retain a metric structure, and the diffusion process receives a geometry-aware, scale-preserving description of the target camera. Although we use Pl¨ucker embeddings as our default parameterization, the method is not tied to it: other cameraaware parameterizations, as well as mixtures thereof, can be substituted without substantial differences (Sec. 4.4).

Through the imposed canonicalization into StereoSpace via viewpoint conditioning, a single model can operate across multiple baselines and focal lengths observed during training and can handle datasets containing several stereo rigs, without being tied to a specific calibrated setup, which in turn allows generalization beyond those configurations at inference time.

##### 3.2. Architecture

Our model adopts a dual U-Net diffusion backbone for stereo view generation [59]: a reference U-Net encodes the source view into semantically rich features, while a denoising U-Net synthesizes the target view conditioned on these features, providing a natural trade-off between semantic preservation and geometric adaptation [27, 71]. Both U-Nets are initialized from Stable Diffusion checkpoints [65], transferring strong semantic and structural priors from large-scale image pretraining to our stereo setting. Viewpoint information Φ is injected via pixel-wise Pl¨ucker rays [57] computed for both source and target images in the canonical StereoSpace. This dense per-pixel pose representation is injected via Adaptive Layer Normalization into the ResNet blocks of both U-Nets [107], and additionally concatenated to the input latents [21], so that the diffusion process can attend directly to the underlying 3D ray configuration. The complete architecture is depicted in Fig. 2.

[Figure 40]

- Figure 2. Architecture overview. The model uses a dual U-Net initialized from Stable Diffusion v2.0. The top branch operates on the source view latent as well as the viewpoint condition. The target baseline is encoded similarly and is concatenated with the latent code of the counterpart view. Latent and pixel-space losses supervise fine-tuning, wherein target view synthesis leverages source view features through end-to-end cross attention. Red arrows denote operations at training time only. Refer to Sec. 3 for details on conditioning and warping.

- 3.3. Warping Loss

DDIM [75] sampling admits a closed-form expression for the clean sample at t = 0, which we interpret as the predicted target-view image It. We first supervise It with a photometric loss that combines structural similarity and perpixel ℓ1:

Lpix = α 1 − SSIM( It,It) + (1 − α)∥ It − It∥1. (2)

Unlike prior warping-based approaches, our model is not conditioned on disparity. Hence, the ground truth sourceview disparity ds is used purely as a supervision, injecting explicit scene geometry into learning. We define a differentiable backward-warping operator Wds

that maps It into the source frame using ds and known camera geometry. A binary validity mask M removes pixels that are out-of-bounds or fail a left–right consistency check, thus avoiding penalties on occlusions and invalid regions. The warp-consistency loss is then given by the masked ℓ1 residual:

Lwarp =

1 ∥M∥1

M ⊙ Wds

( It) − Is 1, (3)

where ⊙ denotes element-wise multiplication and the norms are taken over spatial dimensions. Our final training objective combines the velocity, photometric, and warpconsistency terms:

Ltotal = Lvel + λpixLpix + λwarp Lwarp. (4)

- 4. Experimental Results We now introduce our experimental evaluation.

- 4.1. Training Datasets

We train StereoSpace on a mixed-dataset strategy combining ∼750K single-baseline stereo pairs from 12 synthetic and photorealistic datasets. The main data sources include TartanAir [89] (306K pairs), Dynamic Replica [35] (145K), IRS [87] (103K), Falling Things [84] (61K), and LayeredFlow [93] (31K), among others [8, 9, 32, 33, 51, 81, 82], covering indoor and outdoor scenes. Importantly, we incorporate multi-baseline data to help the model understand baseline effects, a key distinction from prior work. Specifically, we use 27K multi-view tuples from NeRF-Stereo [83] and 5K from SceneSplat-7K [43] (Hypersim [64], Replica [76], ScanNet++ [101] subsets). For the Gaussian Splats provided in SceneSplat-7K, we recover the training cameras, constrain virtual viewpoints to their geometric hull, and cluster cameras by orientation to obtain locally consistent epipolar groups from which we render short virtual-baseline stacks along stereo directions. Examples are shown in Fig. 3. Each tuple contains 5-7 rectified views along a shared baseline direction, generating 10-21 stereo pairs per tuple. Smaller datasets are resampled to 10% of the largest dataset size for balance. Multi-baseline tuples are weighted by 10×#tuples to account for multiple pairs per tuple.

- 4.2. Implementation Details

Architecture. For our proposed StereoSpace architecture, we fine-tune both streams of the dual U-Net on top of a Stable Diffusion 2.0 checkpoint. The first convolution of each branch is modified to accept 10 channels: 4-channel VAE latent concatenated with 6D Pl¨ucker ray embeddings. New channels are zero-initialized to preserve pretrained weights.

[Figure 41]

[Figure 42]

- Figure 3. Multi-baseline training data. Samples from SceneSplat-7K (top two rows) and NeRF-Stereo (bottom row). Scenes span indoor and outdoor environments and are observed under multiple, controlled baselines, providing explicit cues on how stereo geometry changes with baseline length; numbers below each view indicate the distance (cm) to the left-most view in the corresponding tuple.

All parameters are trainable except the highest-resolution up-block of the reference U-Net, frozen to stabilize appearance features. Semantic conditioning uses a frozen CLIP ViT-H/14 (LAION-5B [70]) providing 1024-dim features matching the SD 2.0’s cross-attention dimension.

Training and Inference. We train on mixed synthetic and photorealistic stereo data (Sec. 4.1). Images are resized to 768 × 768 by short-side scaling and center cropping to preserve the rectified geometry. Paired crops are generated onthe-fly for consistent left-right transforms. We use AdamW [47] with learning rate 1×10−5, (β1,β2)=(0.9,0.999), weight decay=1×10−2, ϵ=1×10−8, constant schedule, and gradient clipping at 1.0. For the total loss in Eq. 4, we set the loss weights to λpix = 1.0 and λwarp = 0.3. Per-device batch size is 1 with gradient accumulation 6, yielding an effective batch size of 6×NGPU, with NGPU = 12. We employ the diffusers [85] DDIM scheduler with velocity parameterization, enabling a deterministic mapping and closed-form solution at (t=0) for pixel-space reconstruction. The scheduler uses 1000 steps with a scaled-linear (β) schedule, zeroSNR (γSNR = 5.0) and a 0.05 noise offset. Classifier-free guidance is applied on CLIP and reference U-Net features with 0.1 unconditional drop ratio. Inference uses 50 steps and guidance scale of 1.5.

- 4.3. Setup

ours. Moreover, these metrics are not expressive for ranking perceptual realism, as they penalize pixel misalignment between generated and ground-truth images more heavily than blur or unrealistic hallucinations.

We therefore propose an evaluation protocol that combines MEt3R [1] and iSQoE [78], two complementary metrics that operate on orthogonal axes: MEt3R measures stereo consistency by lifting per-pixel semantic features (DINO+FeatUp [10, 18]) into a common 3D frame predicted by a pretrained geometry model (MASt3R [41]), reprojecting them into both views, and computing symmetric cosinesimilarity maps S:

MEt3R(I1,I2) = 1 − 21 S(I1,I2) + S(I2,I1) . (5) The perceptual estimate of stereo fidelity and viewing comfort is provided with iSQoE [78], a learned stereoscopic QoE predictor that maps a left–right pair to a single scalar trained on VR preference data. In Sec. 4.5 and 4.6, we demonstrate that iSQoE and MEt3R provide more informative assessments than PSNR and SSIM, both qualitatively and quantitatively.

Furthermore, previous evaluation protocols [59, 86] for stereo generation suffer from test-time leakage, as they condition inference on ground truth disparity, and thus remove the need to predict geometry. We instead evaluate all methods strictly end-to-end, without access to ground truth geometry, and account for monocular scale ambiguity via per-scene calibration: for each method and scene, we choose the camera baseline (ours) or depth-to-disparity scale that best aligns the generated stereo pair with the ground truth, using a coarse-tofine search that minimizes the RMSE between SGBM [24] disparity maps on real versus synthesized views over jointly valid pixels. The selected scale is then fixed for all metrics reported, equalizing difficulty across methods while avoiding any target-side information during generation.

Evaluation Protocol. Assessing geometric consistency in stereo generation remains challenging: most pipelines are still evaluated solely with photometric metrics (PSNR, SSIM, LPIPS), which often assign the highest scores to over-smoothed, averaged predictions that visibly wash out high-frequency detail and depth edges [1, 100]. Recent work has sought to standardize these metrics in the context of tiny-baseline stereo [103], but this evaluation protocol remains tailored to a setting that is markedly different from

Baselines. We compare against state-of-the-art monocularto-stereo view generators: ZeroStereo’s Stereo-Gen [90] inpainting module (warp-and-inpaint, DAv2 [99] backbone), StereoDiffusion [86] (latent warping), and GenStereo [59] (warped conditioning). Within our end-to-end evaluation pipeline, we perform the coarse-to-fine search over the baseline and scale parameter in [0.025,1], respectively. To probe the behavior of large open-world generative models on this task, we additionally evaluate Lyra [2], where we approximate the right view by simulating a small camera translation along its x-axis and selecting the frame that minimizes the same RMSE criterion used for scale calibration.

Datasets. We perform architectural ablations on Middlebury 2014 [69] and evaluate the final model on DrivingStereo [97], which we select over KITTI [52] due to its more favorable aspect ratio, as well as on Booster [63] and LayeredFlow [93]. This suite collectively balances canonical stereo evaluation with diverse, real-world conditions (multilayered depth, weather changes, non-Lambertian surfaces). For all datasets, we rescale the shorter image side to each model’s native input resolution and then apply a square center crop to match the required aspect ratio while preserving stereo geometry. To ensure a fair comparison, all generated stereo pairs are finally downsampled to a common resolution of 512×512 before computing metrics.

##### 4.4. Ablation Study

Before comparing with state-of-the-art models, we study the impact of the different components in StereoSpace.

Viewpoint Conditioning. Our dual U-Net architecture exposes multiple injection points for viewpoint conditioning. We compare three alternatives on Middlebury (Tab. 1): (i) a CLIP [61] text embedding that encodes the desired stereo baseline as a prompt (“Stereo baseline xcm to the left/right”) into the denoising U-Net, (ii) PRoPE-style [42] projective attention that injects full camera frustums into the cross-attention between the two U-Nets, and (iii) a dense Pl¨ucker ray embedding of the target camera,. Each conditioning alone already surpasses GenStereo on both iSQoE and MEt3R, indicating that our viewpoint-conditioned diffusion framework is effective. Adding PRoPE on top of Pl¨ucker does not yield further improvements, indicating that stacking multiple conditioning signals is not beneficial in this regime. Pl¨ucker conditioning achieves the best scores and is used as our default, it further exposes camera intrinsics explicitly and avoids an additional text encoder.

Training Data and Disparity Loss. To assess the impact of additional multi-view stereo data (NeRF-Stereo, SceneSplat), we train a variant of our model without these sources, using effectively the same training corpus as GenStereo. Performance drops noticeably on both iSQoE and MEt3R compared to our full dataset, confirming the benefit of the extra multi-view supervision, yet this reduced-data variant

Middlebury 2014 [69] Method iSQoE(↓) MEt3R(↓)

GenStereo [59] 0.6933 0.1339 StereoSpace w/ text 0.6841 0.0907 StereoSpace w/ Pl¨ucker 0.6823 0.0901 StereoSpace w/ PRoPe 0.6865 0.0937 StereoSpace w/ Pl¨ucker+PRoPE 0.6828 0.0945 StereoSpace w/ Pl¨ucker 0.6823 0.0901 StereoSpace wo/ multi-baseline 0.6907 0.1095 StereoSpace w/ warping loss 0.6829 0.0893

Table 1: Ablation study on Middlebury dataset. Metrics are iSQoE (↓) and MEt3R (↓). Best and second-best scores highlighted over subcategories.

still outperforms GenStereo, supporting the robustness of our viewpoint conditioning design. Finally, we enable an auxiliary disparity loss during training. This variant achieves a modest improvement in MEt3R at the cost of a slight degradation in iSQoE, consistent with our hypothesis that disparity supervision encourages stronger geometric alignment while mildly trading off perceived viewing comfort.

##### 4.5. Results on Single-Layer Geometries

We first compare with state-of-the-art methods on two datasets: Middlebury 2014 [69] (indoor) and DrivingStereo [97] (outdoor). Table 2 collects both iSQoE and MEt3R metrics achieved by the four baselines and our StereoSpace framework on the two datasets.

On Middlebury 2014, StereoSpace achieves the lowest iSQoE (0.6829) and MEt3R (0.0893), improving upon the scores of GenStereo [59] and Lyra [2] in the latter metric by more than 30% and 20%, respectively. ZeroStereo [90] and StereoDiffusion [86] trail on this benchmark, with the highest errors in both metrics. For DrivingStereo, our method again yields the best iSQoE and attains a MEt3R score of 0.0717, while ZeroStereo is competitive in geometry (better than StereoDiffusion), but remains behind GenStereo (0.0728) and ours. We can notice how the margins on this dataset are lower due to the simpler geometry characteristic of driving environments.

These trends align with the qualitative behavior of the baselines. StereoDiffusion often produces overly smooth, monochrome completions: such artifacts can inflate pixel similarity measures yet harm stereo viewing comfort. This is reflected by poorer iSQoE and notably higher MEt3R scores, especially on Middlebury. ZeroStereo preserves warped structure where visible, but its inpainted regions are frequently inconsistent, which elevates iSQoE and, on Middlebury, also degrades MEt3R. It fares better on DrivingStereo where geometry is simpler, yet still lags behind GenStereo and StereoSpace. GenStereo’s warped conditioning is less susceptible to these issues: benefiting from a strong monocular depth prior, it provides accurate disparity

Middlebury 2014 [69] Drivingstereo [97] Method Category Depth iSQoE(↓) MEt3R(↓) iSQoE(↓) MEt3R(↓)

StereoDiffusion [86] Warp-and-inpaint DAv2 [99] 0.7475 0.1933 0.7887 0.1015 ZeroStereo [90] Latent warping DAv2 [99] 0.7423 0.2057 0.7964 0.0798 GenStereo [59] Warped conditioning DAv2 [99] 0.6933 0.1339 0.7850 0.0728 Lyra [2] 3DGS generative model MoGe-2 [88] 0.7184 0.1163 0.7891 0.0949

StereoSpace (ours) Depth-free - 0.6829 0.0893 0.7829 0.0717

- Table 2: Results on Middlebury and Drivingstereo. Metrics are iSQoE (↓) and MEt3R (↓). For each model, we report its category, followed by the depth estimator it uses at inference time. Best , second- and third-best scores highlighted.

Booster [63] LayeredFlow [93] Method Category Depth iSQoE(↓) MEt3R(↓) iSQoE(↓) MEt3R(↓)

StereoDiffusion [86] Warp-and-inpaint DAv2 [99] 0.7248 0.2011 0.8046 0.3074 ZeroStereo [90] Latent warping DAv2 [99] 0.7503 0.3171 0.8108 0.3630 GenStereo [59] Warped conditioning DAv2 [99] 0.6901 0.1457 0.7678 0.2275 Lyra [2] 3DGS generative model MoGe-2 [88] 0.6989 0.1293 0.7802 0.1877

StereoSpace (ours) Depth-free - 0.6764 0.1013 0.7489 0.1619

- Table 3: Results on Booster and LayeredFlow. Metrics are iSQoE (↓) and Met3r (↓). Best , second- and third-best scores highlighted. For each model, we report the category it belongs to, followed by the depth model it uses at inference time.

Ground truth GenStereo [59] StereoSpace (ours)

Ground truth GenStereo [59] StereoSpace (ours)

[Figure 43]

[Figure 44]

[Figure 45]

| |
|---|

| |
|---|

[Figure 46]

[Figure 47]

| |
|---|

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

PSNR = 19.9080 SSIM = 0.5933

PSNR = 19.9534 SSIM = 0.5697

[Figure 55]

[Figure 56]

[Figure 57]

Figure 5. Qualitative results on LayeredFlow [93]. While GenStereo (left) fails at modeling layered structures in the presence of transparent surfaces, StereoSpace (right) excels at this.

[Figure 58]

[Figure 59]

[Figure 60]

PSNR = 18.3323 SSIM = 0.6061

PSNR = 14.3611 SSIM = 0.4211

Figure 4. Qualitative results on Middlebury 2014 [69]. Compared

[Figure 61]

[Figure 62]

[Figure 63]

|layers.|
|---|

GenStereo (left), StereoSpace (right) preserves realistic image details, as shown on top. We also report PSNR and SSIM to highlight the inability of such metrics to account for it.

|to|
|---|

geometry and transparency-induced depth

lay Table 3 collects both iSQoE and MEt3R metrics achieved by the four baselines StereoSpace

|methods exdepth, trans-<br><br>violate|
|---|

o datasets.

|on the two<br><br>benchmarks, with<br><br>These|
|---|

|and<br><br>these more degradation<br><br>and irregular|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

guidance in latent space and achieves the best DrivingStereo MEt3R

On challenging rks, all hibit d consistent multi-level parency, ar reflectance. phenomena the single-surface assumption inherent in warping pipelines. A single disparity map cannot model semi-transparent layers, specular highlights, or view-dependent reflections simultaneously, leading to geometric drift and compensatory blur. As a consequence, StereoSpace attains the best scores on both metrics across both datasets by a large margin. ZeroStereo is the most affected, with its reliance on explicit warping and impainting yielding the highest errors. While StereoDiffusion performs slightly better but still suffers from oversmoothing, GenStereo remains competitive, but trails our method on both iSQoE and MEt3R, mostly competing with Lyra for the second place.

the baselines. Nevertheless, our formulation surpasses GenStereo overall qu ering th t iSQoE o datasets, as best geometric consistency. Notably, Lyra mostly competes with GenStereo for the second place on Middlebury, probably a consequence of its open-world capabilities avoiding specialization on the considered stereo setting. These results suggest that incorporating geometry directly into the generative model yields stereo pairs that are both perceptually appealing and geometrically accurate in typical settings.

|end-to-end quality, delivwell as the|
|---|

|among the strongest|
|---|

|in on both|
|---|

Ground Truth GenStereo StereoSpace

##### 4.6. Results on Multi-Layer Geometries

As a final comparison, we evaluate against the same baselines on Booster [63] and LayeredFlow [93], showcasing complex

[Figure 67]

###### (a)

Human preference ranking (Bradley–Terry; 95% CI)

###### Model-implied pairwise preference probabilities

1.0

| |0.58|0.67|0.90|0.93|
|---|---|---|---|---|
|[Figure 68]<br><br>0.42| |0.59|0.87|0.91|
|0.33|0.41| |0.82|0.87|
|0.10|0.13|0.18| |0.60|
|0.07|0.09|0.13|0.40| |

[Figure 69]

0.73

StereoSpace

Ours

P(rowpreferredtocolumn)

0.8

0.40

Lyra

Lyra

0.6

0.03

GenStereo

GenStereo

0.4

ZeroStereo

-1.50

ZeroStereo

0.2

StereoDiff

-1.91

StereoDiff

0.0

Ours LyraGenStereoZeroStereoStereoDiff

−2.5 −2.0 −1.5 −1.0 −0.5 0.0 0.5 1.0

Bradley–Terry log-strength (mean-centered)

###### (b) (c)

Figure 6. User study through pairwise comparisons. 70 participants were involved, for a total of 1.4K pairwise comparisons. (a) StereoSpace winrate against other methods; (b) Bradley-Terry win probabilities, (c) ranking according to Bradley-Terry scores.

This confirms that embedding stereo reasoning into the generator confers robustness to non-ideal optics and layered geometry, which are ubiquitous outside controlled settings.

##### 4.7. Qualitative Results

- Figure 4 shows qualitative samples from Storage scene, reporting the results rendered by GenStereo and StereoSpace. Our method accurately deals with the complex overlaying in the background, faithfully recreating the shadows effect and realistic occlusions between objects. In contrast, GenStereo produces unrealistic overlaps between foreground and background objects, mainly trying to inpaint the occlusions caused by the warping process.
- Figure 5 shows qualitative samples from the LayeredFlow

dataset, which depict in particular a glass railing together with the results rendered by GenStereo and StereoSpace. It is clearly evident how the reliance on depth priors and warping makes GenStereo unable to handle the two different layers in the scene, i.e., the glass railing itself and the wall in the background visible through it. Indeed, we can observe how the painting on the wall is split in the rendered image, with its bottom part being moved further to the left as if it belongs to the first depth layer. In contrast, StereoSpace is not affected by this issue and can render the painting faithfully. We refer the reader to the supplementary material for higher-resolution qualitative results.

###### The Ineffectiveness of Conventional Metrics. Both Figures

- 4 and 5 also reports PSNR and SSIM computed on images generated by GenStereo and StereoSpace with respect to the ground truth right image – an exhaustive evaluation is reported in the supplementary material. We can observe that, in the former case, GenStereo achieves better scores than our method, despite the evident artifacts and its inability to properly handle complex geometries and shadows. We attribute

this behavior to the tighter pixel-wise alignment between rendered and ground-truth views achieved by depth-warping methods. As a result, these metrics tend to penalize our approach, which does not rely on depth priors, yet produces more realistic and geometry-consistent images, as reflected by iSQoE and MEt3R, which we consider more appropriate for benchmarking this task.

##### 4.8. User Study Evaluation

To confirm the effectiveness of the proposed evaluation protocol, we conduct a study among 70 participants instructed to vote for the best generated image in pairwise comparisons. Each subject received 20 random pairs of different methods’ output images alongside the corresponding left image as a reference. We report the outcome in Figure 6, showing StereoSpace winrate against any other method (a), as well as aggregated scores using the Bradley–Terry preference model (b), used then to rank methods (c). These results align well with the trend highlighted by MEt3R and iSQoE: we outperform Lyra about 60-40, while others lag behind.

#### 5. Conclusion

In this paper, we presented StereoSpace, an end-to-end approach to stereo view synthesis. While existing solution to this task [2, 59, 86, 90] heavily rely on depth priors to exploit warping and turn the generative task into an inpainting problem, our framework removes this requirement and only relies on view conditioning to effectively learn stereo geometry. StereoSpace generates more realistic novel views, in particular when running on images depicting multi-layered geometry, as confirmed by both perceptual and geometry based metrics. Future developments will aim at extending StereoSpace to stereo video generation.

Acknowledgments. We acknowledge the EuroHPC Joint Undertaking for awarding us access to Leonardo at CINECA, Italy, under Proposal ID EHPC-DEV-2025D05-054. The authors also acknowledge the CINECA award under the ISCRA initiative.

#### References

- [1] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. MEt3R: Measuring multi-view consistency in generated images. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 2, 5, 16, 17, 18
- [2] Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B Lindell, Zan Gojcic, Sanja Fidler, et al. Lyra: Generative 3d scene reconstruction via video diffusion model self-distillation. preprint arXiv:2509.19296, 2025. 1, 2, 6, 7, 8, 17, 18, 19, 20
- [3] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. AC3D: Analyzing and improving 3d camera control in video diffusion transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 2
- [4] Dmitry Baranchuk, Andrey Voynov, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Label-efficient semantic segmentation with diffusion models. In International Conference on Learning Representations (ICLR), 2022. 2
- [5] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2
- [6] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2022. 2
- [7] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2023. 2
- [8] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In European Conference on Computer Vision (ECCV), 2012. 4, 14
- [9] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual Kitti 2. preprint arXiv:2001.10773, 2020. 4, 14
- [10] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 5
- [11] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelSplat: 3d gaussian splats from

- image pairs for scalable generalizable 3d reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 2
- [12] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. MVSNeRF: Fast generalizable radiance field reconstruction from multi-view stereo. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2
- [13] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial radiance fields. In European Conference on Computer Vision (ECCV), 2022. 2
- [14] Shoufa Chen, Peize Sun, Yibing Song, and Ping Luo. DiffusionDet: Diffusion model for object detection. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2
- [15] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. MVSplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision (ECCV), 2024. 2
- [16] Peng Dai, Feitong Tan, Qiangeng Xu, David Futschik, Ruofei Du, Sean Fanello, Xiaojuan Qi, and Yinda Zhang. SVG: 3d stereoscopic video generation via denoising frame matrix. In International Conference on Learning Representations (ICLR), 2025. 3
- [17] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 2
- [18] Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. FeatUp: A model-agnostic framework for features at any resolution. In International Conference on Learning Representations (ICLR), 2024. 5
- [19] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision (ECCV), 2024. 2
- [20] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2
- [21] Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole*. CAT3D: Create anything in 3d with multi-view diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 3
- [22] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation. preprint arXiv:2404.02101, 2024. 2
- [23] Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, and Lei Xie. A diffusion-based framework for multi-class anomaly detection. In AAAI Conference on Artificial Intelligence,

2024. 2

- [24] Heiko Hirschmuller. Stereo processing by semiglobal matching and mutual information. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(2):328–341, 2008. 5
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 2020. 2
- [26] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 2022. 2
- [27] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), pages 8153–8163, 2024. 3
- [28] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SigGraph, pages 1–11,

2024. 2

- [29] Xingchang Huang, Ashish Kumar Singh, Florian Dubost, Cristina Nader Vasconcelos, Sakar Khattar, Liang Shi, Christian Theobalt, Cengiz Oztireli, and Gurprit Singh. Restereo: Diffusion stereo video generation and restoration. preprint arXiv:2506.06023, 2025. 3
- [30] Yan-Bin Jia. Pl¨ucker coordinates for lines in the space. COMS 4770/5770 Notes, Iowa State University, 2024. Lecture notes. 3, 15
- [31] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. GaussianShader: 3d gaussian splatting with shading functions for reflective surfaces. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 2
- [32] Junpeng Jing, Ye Mao, Anlan Qiu, and Krystian Mikolajczyk. Match stereo videos via bidirectional alignment. preprint arXiv:2409.20283, 2024. 4, 14
- [33] Laurent Jospin, Allen Antony, Lian Xu, Hamid Laga, Farid Boussaid, and Mohammed Bennamoun. Active-passive simstereo-benchmarking the cross-generalization capabilities of deep learning-based stereo methods. Advances in Neural Information Processing Systems (NeurIPS), 2022. 4, 14
- [34] Yash Kant, Aliaksandr Siarohin, Ziyi Wu, Michael Vasilkovsky, Guocheng Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey Tulyakov, and Igor Gilitschenski. SPAD: Spatially aware multi-view diffusers. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 15
- [35] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. DynamicStereo: Consistent dynamic depth from stereo videos. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2023. 4, 14
- [36] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 3
- [37] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth

- estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 2
- [38] Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger, Shengyu Huang, Bo Li, Anton Obukhov, and Konrad Schindler. Marigold: Affordable adaptation of diffusionbased image generators for image analysis. preprint arXiv:2505.09358, 2025. 2
- [39] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42(4):139–1, 2023. 2
- [40] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. EscherNet: A generative model for scalable view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 15
- [41] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024. 5, 17
- [42] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. preprint arXiv:2507.10496, 2025. 6, 15
- [43] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, et al. SceneSplat: Gaussian splatting-based scene understanding with vision-language pretraining. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2, 4, 14
- [44] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N Plataniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navigating 3d scenes from a single image. IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 2
- [45] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2
- [46] Zihua Liu, Yizhou Li, Songyan Zhang, and Masatoshi Okutomi. DMS: Diffusion-based multi-baseline stereo generation for improving self-supervised depth estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 3
- [47] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. preprint arXiv:1711.05101, 2017. 5
- [48] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2022. 2
- [49] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In International Conference on 3D Vision (3DV), 2024. 2
- [50] Zhen Lv, Yangqi Long, Congzhentao Huang, Cao Li, Chengfei Lv, Hao Ren, and Dian Zheng. SpatialDreamer:

- Self-supervised stereo video synthesis from monocular input. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 3
- [51] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2023. 4, 14
- [52] Moritz Menze and Andreas Geiger. Object scene flow for autonomous vehicles. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2015. 6

- [53] Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. GTA: A geometry-aware attention mechanism for multi-view transformers. In International Conference on Learning Representations (ICLR), 2024. 15
- [54] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2I-Adapter: Learning adapters to dig out more controllable ability for text-toimage diffusion models. In AAAI Conference on Artificial Intelligence, 2024. 2
- [55] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG), 41(4):1–15, 2022. 2
- [56] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2
- [57] Julius Pl¨ucker. On a new geometry of space. Philosophical Transactions of the Royal Society of London, 155:725–791,

1865. 3

- [58] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural radiance fields for dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2020. 2
- [59] Feng Qiao, Zhexiao Xiong, Eric Xing, and Nathan Jacobs. GenStereo: Towards open-world generation of stereo images and unsupervised matching. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 1, 3, 5, 6, 7, 8, 14, 17, 18, 19, 20
- [60] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. RichDreamer: A generalizable normal-depth diffusion model for detail richness in textto-3d. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 2
- [61] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 6
- [62] Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, and Vishal M Patel. Ambiguous medical image segmentation using diffusion models. In IEEE/CVF Confer-

ence on Computer Vision and Pattern Recogition (CVPR),

2023. 2

- [63] Pierluigi Zama Ramirez, Fabio Tosi, Matteo Poggi, Samuele Salti, Stefano Mattoccia, and Luigi Di Stefano. Open challenges in deep stereo: the booster dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR),

2022. 6, 7, 15, 16, 20, 21

- [64] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021. 4, 14
- [65] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), pages 10684–10695, 2022. 3
- [66] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR),

2022. 2

- [67] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations (ICLR), 2022. 2, 3
- [68] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, and Jiajun Wu. ZeroNVS: Zero-shot 360-degree view synthesis from a single image. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 2
- [69] Daniel Scharstein, Heiko Hirschm¨uller, York Kitajima, Greg Krathwohl, Nera Neˇsi´c, Xi Wang, and Porter Westling. Highresolution stereo datasets with subpixel-accurate ground truth. In Pattern Recognition, pages 31–42, 2014. 6, 7, 15, 16, 17, 19, 20
- [70] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 5
- [71] Junyoung Seo, Kazumi Fukuda, Takashi Shibuya, Takuya Narihira, Naoki Murata, Shoukang Hu, Chieh-Hsin Lai, Seungryong Kim, and Yuki Mitsufuji. GenWarp: Single image to novel views with semantic-preserving generative warping. Advances in Neural Information Processing Systems (NeurIPS), 2024. 3
- [72] Jian Shi, Qian Wang, Zhenyu Li, Ramzi Idoughi, and Peter Wonka. StereoCrafter-Zero: Zero-shot stereo video generation with noisy restart. preprint arXiv:2411.14295, 2024. 3
- [73] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3d generation. In International Conference on Learning Representations (ICLR), 2024. 2

- [74] Nina Shvetsova, Goutam Bhat, Prune Truong, Hilde Kuehne, and Federico Tombari. M2SVid: End-to-end inpainting and refinement for monocular-to-stereo video conversion. preprint arXiv:2505.16565, 2025. 3
- [75] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 4
- [76] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. 4, 14
- [77] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 15

- [78] Netanel Tamir, Shir Amir, Ranel Itzhaky, Noam Atia, Shobhita Sundaram, Stephanie Fu, Ron Sokolovsky, Phillip Isola, Tali Dekel, Richard Zhang, and Miriam Farber. What makes for a good stereoscopic image? In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 2, 5
- [79] Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. MVDiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. Advances in Neural Information Processing Systems (NeurIPS), 2023. 2
- [80] Junjiao Tian, Lavisha Aggarwal, Andrea Colaco, Zsolt Kira, and Mar Gonzalez-Franco. Diffuse attend and segment: Unsupervised zero-shot segmentation using stable diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 2
- [81] Joshua Tokarsky, Ibrahim Abdulhafiz, Satya Ayyalasomayajula, Mostafa Mohsen, Navya G Rao, and Adam Forbes. PLT-D3: A high-fidelity dynamic driving simulation dataset for stereo depth and scene flow. preprint arXiv:2406.07667,

2024. 4, 14

- [82] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. SMD-Nets: Stereo mixture density networks. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2021. 4, 14
- [83] Fabio Tosi, Alessio Tonioni, Daniele De Gregorio, and Matteo Poggi. NeRF-supervised deep stereo. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR),

2023. 2, 4, 14

- [84] Jonathan Tremblay, Thang To, and Stan Birchfield. Falling things: A synthetic dataset for 3d object detection and pose estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recogition Workshops (CVPRW), 2018. 4, 14
- [85] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven

- Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers, 2022. 5
- [86] Lezhong Wang, Jeppe Revall Frisvad, Mark Bo Jensen, and Siavash Arjomand Bigdeli. StereoDiffusion: Training-free stereo image generation using latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 1, 3, 5, 6, 7, 8, 17, 18, 19, 20
- [87] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. IRS: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In IEEE International Conference on Multimedia and Expo (ICME), pages 1–6, 2021. 4, 14
- [88] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: Accurate monocular geometry with metric scale and sharp details. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2, 7
- [89] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A dataset to push the limits of visual slam. In IEE/RSJ International Conference on Intelligent Robotics and Systems, 2020. 4, 14
- [90] Xianqi Wang, Hao Yang, Gangwei Xu, Junda Cheng, Min Lin, Yong Deng, Jinliang Zang, Yurui Chen, and Xin Yang. ZeroStereo: Zero-shot stereo matching from single images. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 1, 3, 6, 7, 8, 17, 18, 19, 20
- [91] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A unified and flexible motion controller for video generation. In ACM SigGraph, 2024. 2
- [92] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. Foundationstereo: Zeroshot stereo matching. CVPR, 2025. 14
- [93] Hongyu Wen, Erich Liang, and Jia Deng. LayeredFlow: A real-world benchmark for non-lambertian multi-layer optical flow. In European Conference on Computer Vision (ECCV),

2024. 4, 6, 7, 14, 15, 16, 18, 19, 20, 21

- [94] Junyuan Xie, Ross Girshick, and Ali Farhadi. Deep3D: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks. In European Conference on Computer Vision (ECCV), 2016. 2
- [95] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. CamCo: Camera-controllable 3d-consistent image-to-video generation. preprint arXiv:2406.02509, 2024. 2
- [96] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2023. 2
- [97] Guorun Yang, Xiao Song, Chaoqin Huang, Zhidong Deng, Jianping Shi, and Bolei Zhou. DrivingStereo: A large-scale dataset for stereo matching in autonomous driving scenarios. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2019. 6, 7, 15, 16, 19, 20

- [98] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and MingHsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4): 1–39, 2023. 2
- [99] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 6, 7
- [100] Xianghui Yang, Yan Zuo, Sameera Ramasinghe, Loris Bazzani, Gil Avraham, and Anton van den Hengel. ViewFusion: Towards multi-view consistency via interpolated denoising. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 5
- [101] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the International Conference on Computer Vision (ICCV), 2023. 4, 14
- [102] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelNeRF: Neural radiance fields from one or few images. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2021. 2
- [103] Songsong Yu, Yuxin Chen, Zhongang Qi, Zeke Xie, Yifan Wangand Lijun Wang, Ying Shan, and Huchuan Lu. Mono2Stereo: A benchmark and empirical study for stereo conversion. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2025. 3, 5
- [104] Jiale Zhang, Qianxi Jia, Yang Liu, Wei Zhang, Wei Wei, and Xin Tian. SpatialMe: Stereo video conversion using depthwarping and blend-inpainting. preprint arXiv:2412.11512,

2024. 3

- [105] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2
- [106] Sijie Zhao, Wenbo Hu, Xiaodong Cun, Yong Zhang, Xiaoyu Li, Zhe Kong, Xiangjun Gao, Muyao Niu, and Ying Shan. StereoCrafter: Diffusion-based generation of long and highfidelity stereoscopic 3d from monocular videos. In preprint arXiv:2407.00367, 2024. 3
- [107] Chuanxia Zheng and Andrea Vedaldi. Free3D: Consistent novel view synthesis without 3d representation. In IEEE/CVF Conference on Computer Vision and Pattern Recogition (CVPR), 2024. 3, 15
- [108] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. preprint arXiv:2503.14489,

2025. 2

## StereoSpace: Depth-Free Synthesis of Stereo Geometry via End-to-End Diffusion in a Canonical Space

### Supplementary Material

Dataset Baseline Setting #Samples Year NeRF-Stereo [83] 27K 2023 SceneSplat [43] 5K 2025 TartanAir [89] 25 cm 306K 2020 Dynamic Replica [35] 145K 2023 IRS [87] 10 cm 103K 2021 Falling Things [84] 6 cm 61K 2018 LayeredFlow [93] 31K 2024 VKITTI2 [9] 53.3 cm 21K 2020 InfinigenSV [32] 17K 2024 SimStereo [33] 16 cm 14K 2022 UnrealStereo4K [82] 8K 2021 Spring [51] 5K 2023 PLT-D3 [81] 12 cm 3K 2024 Sintel [8] 10 cm 1K 2012

- Table 4: Training data. Stereo datasets used for mixed training of StereoSpace. For each dataset, we report the baseline (fixed value when available, indicates a variable baseline), scene type (indoor or outdoor ), the number of samples, and the release year. Datasets above the dotted line . . . . . are multi-baseline.

Section 6 reports additional details concerning the training of StereoSpace, Section 7 describes the main conditioning mechanisms used to allow for camera control, Section 8 describes the evaluation split composition in detail and finally Section 10 shows further qualitative results.

#### 6. Training Details

Dataset Composition. Our training data is drawn from 14 publicly available data sources. Here, we provide additional statistics in Tab. 4 for the stereo datasets. We report the nominal baseline, whether a dataset primarily contains indoor or outdoor scenes, the number of training stereo pairs and the release year, respectively. Together, these sources cover a diverse range of camera baselines, scene scales, and photorealism levels.

Disparity Supervision. Most datasets provide ground-truth left-to-right and right-to-left disparity maps. When some direction is missing, we infer the respective disparity using a strong off-the-shelf stereo matcher (FoundationStereo [92]), and treat the resulting pair as pseudo-ground-truth. Having both directions is required to compute the left–right consistency mask used in the warping loss (Sec. 3.3 in the main paper), which restricts supervision to pixels that are co-visible in the source and target views. The synthetic LayeredFlow dataset is a notable exception: its multi-layer depth represen-

[Figure 70]

Figure 7. Pl¨ucker coordinates of line ℓ are given by the 6D homogeneous vector (d, m).

tation violates the single-surface assumption underlying our view warping formulation. Therefore, for LayeredFlow samples, we disable the warping loss and re-weight the training batches accordingly.

Rendered Multi-Baseline Tuples. For SceneSplat-7K [43], we derive multi-baseline, rectified stacks from the preoptimized Gaussian splats. We restrict to the Hypersim [64], Replica [76], and ScanNet++ [101] subsets, filtering scenes using the dataset-provided PSNR, SSIM, LPIPS, and depth ℓ1 metrics to discard photometrically or geometrically unstable reconstructions. Within each retained scene, we form candidate stacks with moderate baselines and subsamples up to 20 diverse bundles via k-means over stack centroids per splat. Each selected stack is rendered to RGB and depth with a small global focal-length scaling jitter, and stacks with insufficient geometric support are removed based on simple depth- and opacity-based heuristics.

Training Schedule. StereoSpace, including all its tested variations, is trained for 3 epochs, corresponding to approximately 48.6 K optimizer steps. This schedule matches the GenStereo [59] setup and ensures a comparable training budget between variants in the ablation study (Sec. 4.4). We also experimented with the extension of training to 5 epochs, but observed saturated performance and no consistent improvement relative to added computation, so all reported results adhere to the 3-epoch schedule. In contrast to prior work, StereoSpace is trained in both stereo directions (left→right and right→left). Empirically, this bidirectional supervision did not degrade performance, while enabling inference-time stereo generation in either direction, whereas competing approaches typically support only the left→right direction.

###### Dataset Metric GenStereo Lyra StereoSpace (ours) StereoDiffusion ZeroStereo

PSNR ↑ 18.54 18.06 17.04 17.06 17.48 SSIM ↑ 0.5989 0.5793 0.5410 0.5246 0.5372 LPIPS ↓ 0.184 0.247 0.234 0.304 0.264

Middlebury

PSNR ↑ 24.95 24.25 23.44 22.89 23.64 SSIM ↑ 0.7939 0.7792 0.7424 0.7365 0.7511 LPIPS ↓ 0.136 0.172 0.164 0.193 0.170

DrivingStereo

PSNR ↑ 21.74 23.21 21.91 20.42 18.69 SSIM ↑ 0.7348 0.7813 0.7351 0.6986 0.5733 LPIPS ↓ 0.237 0.196 0.202 0.322 0.443

Booster

PSNR ↑ 17.44 18.67 17.67 17.08 15.83 SSIM ↑ 0.5912 0.6327 0.5940 0.5957 0.4421 LPIPS ↓ 0.370 0.321 0.335 0.433 0.521

LayeredFlow

- Table 5: Evaluation with conventional metrics – PSNR, SSIM, LPIPS. Experiments on Middlebury [69], DrivingStereo [97], Booster

- [63] and LayeredFlow [93].

#### 7. Viewpoint Conditioning

In this section, we recall the principles behind the conditioning mechanisms used by StereoSpace.

Pl¨ucker rays. A 3D line ℓ can be represented by a point o ∈ R3 and a (unit) direction d ∈ R3 (Fig. 7). Its Pl¨ucker (Grassmann) coordinates are the homogeneous 6D vector

ℓ ≡ (d, m), m = o × d. (6)

By construction d·m = 0 (the Pl¨ucker constraint), and for any other point o′ = o + λd on the same line we have (o′ × d) = m. Hence, Pl¨ucker coordinates are invariant to sliding o along the line, which makes them a natural parametrization of camera rays [30].

For a pinhole camera with center c (in world coordinates), the ray through the pixel (i,j) has Pl¨ucker coordinates ℓij = (dij,mij) with

mij = c × dij. (7)

We form dense Pl¨ucker embeddings Fplucker ∈ R6×H×W by concatenating (dij,mij) for each pixel of an image of size (H,W). Because (d,m) are homogeneous, s(d,m) with s̸=0 represents the same (unoriented) line. For rays, we fix this gauge by normalizing ∥d∥ = 1 and choosing the sign so that d points from the camera into the scene.

This representation encodes the camera geometry in a distributed way: instead of a single global pose vector, each pixel’s ray is tagged with a 6D vector that implicitly contains the camera intrinsics and extrinsics for that viewline. Thus, the diffusion model can, in principle, attend to the 3D configuration of rays when conditioning on the view.

Pl¨ucker coordinates also admit simple expressions for line-line relations. Given two rays ℓk = (dk,mk), the reciprocal product

⟨ℓ1,ℓ2⟩ = d1·m2 + d2·m1 (8)

vanishes iff the two rays are coplanar (i.e., they intersect or are parallel) [30]. For non-parallel rays, their shortest distance is

d1·m2 + d2·m1 ∥d1 × d2∥

. (9)

d(ℓ1,ℓ2) =

Rays that image the same 3D point are coplanar and intersect, and rays to nearby points have small reciprocal product and line-line distance. Importantly, these quantities are bilinear in (d,m), so they can be implemented by linear projections and dot products on Fplucker.

Although the computation of such expressions is not enforced explicitly in the network, Pl¨ucker ray embeddings provide an inductive bias that makes cross-view geometric consistency easy to test in the input space. Empirically, such per-pixel ray conditioning has been shown to improve camera pose accuracy and 3D consistency in generative models [34, 107], and we observe similar benefits. By contrast, global pose encodings (e.g., Euler angles) offer no direct notion of pixel-to-pixel correspondences across views and often suffer from symmetries such as front-back ambiguity. PRoPE attention. Besides Pl¨ucker-ray conditioning, we also evaluate an attention-level camera encoding based on Geometric Transform Attention (GTA) [53], CaPE [40], RoPE [77], and the recent PRoPE [42]. In this variant, each token t from camera i(t) is associated with (i) a projective transform derived from the camera’s projection matrix and (ii) a rotary position embedding of its 2D image coordinates (xt,yt). Following the GTA framework, these per-token transforms Dt are multiplied into Q,K,V so that attention logits depend on the relative projective transform between the cameras of the query and key tokens, while RoPE handles within-image spatial relations.

We apply PRoPE-style attention in all cross-attention layers where the denoising stream attends to reference views. The PRoPE mechanism complements Pl¨ucker embeddings:

[Figure 71]

-0.4m -0.2m Source 0.2m 0.4m

- Figure 8. Qualitative results of multiple inferences with varying baseline. StereoSpace naturally supports rendering images captured with arbitrary baselines, including viewpoints located to the left (negative baseline) and to the right (positive baseline) of the source image.

both encode the full camera frustum (intrinsics + extrinsics) but at different stages: Pl¨ucker rays provide per-pixel geometry in the input channels and ResNet blocks, while PRoPE aligns token features via the relative projective transform inside attention. However, as reported in our ablation study, the combination of the two does not produce any significant improvement, but Pl¨ucker embeddings alone achieve both the lowest iSQoE and MEt3R scores.

#### 8. Evaluation Sets Composition

We evaluate on four real-world stereo benchmarks: Middlebury 2014 [69], DrivingStereo [97], Booster [63], and LayeredFlow [93], none of these datasets are used during training. For Middlebury 2014, we follow the official evaluation protocol and use the same split as GenStereo and StereoDiffusion. For DrivingStereo, we uniformly sample 50 stereo pairs from the 500 released pairs for each weather condition, yielding 200 evaluation images in total. For Booster, we report results on the union of the official train and test splits. For LayeredFlow, we evaluate on the 300 real-world stereo pairs from the official validation and evaluation splits of the benchmark. In our mixed training set, we only include the separate synthetic stereo split of LayeredFlow rendered

in Blender.

#### 9. Evaluation with Conventional Metrics

For the sake of completeness, we report classical metrics such as PSNR, SSIM, and LPIPS in Tab. 5. In most cases, StereoSpaces ranks 2nd, just behind either Lyra or GenStereo. However, as discussed in the paper, these metrics fail at faithfully assessing both geometric consistency and perceptual comfort.

#### 10. Qualitative Results

We conclude by reporting additional qualitative results.

Native Multi-baseline Inference. Thanks to our viewpoint conditioning mechanism and training schedule, StereoSpace natively supports the generation images at arbitrary horizontal baselines on either side of the input view (Fig. 8). Warping-based frameworks can also be extended to this setting, but require either manually rescaling the monocular disparity used for warping or flipping the image to synthesize views on the opposite side of the reference image.

MEt3R Score Map Visualization. To better illustrate the margin in MEt3R [1] scores across methods, we visualize

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Adirondack

MEt3R scores: 0.1670 0.2045 0.1311 0.1016 0.0747

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Piano

MEt3R scores: 0.1241 0.1285 0.0709 0.0755 0.0582

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Pipes

MEt3R scores: 0.2342 0.2486 0.1683 0.1573 0.1137

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Shopvac

MEt3R scores: 0.2318 0.1919 0.1471 0.1135 0.0732

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Sticks

MEt3R scores: 0.2394 0.3091 0.1938 0.1568 0.1118

- Figure 9. Visualization of MEt3R score [1] maps on Middlebury dataset [69]. We report, from left to right, the original left image for four samples in the dataset, followed by the MEt3R score maps computed between it and the right images generated by different methods. The coloring is according to the magma colormap, with green regions representing occlusions (discarded by MEt3R when computing the average score). Under each score map, we report the global score computed by MEt3R (the lower, the better).

the per-pixel score maps produced by MEt3R before averaging, which makes local differences between methods directly observable. Fig. 9 shows results on five samples from the Middlebury dataset. Overall, warping-based methods produce higher MEt3R scores, particularly near depth discontinuities, whereas both Lyra and StereoSpace exhibit substantially lower scores. We also observe that the overall

quality of the generated images affects the ability of MEt3R (through its MASt3R [41] backbone) to accurately estimate the relative transformation between the original and synthesized images. This is visible in the green regions, which represent areas of the scene that are not overlapping between the two images. Warping-based solutions show larger nonoverlapping regions at both the top and bottom of the frame,

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

215

MEt3R scores: 0.1553 0.3044 0.3088 0.1901 0.1367

- 249

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

MEt3R scores: 0.5380 0.2258 0.3114 0.1697 0.1351

- 250

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

MEt3R scores: 0.1542 0.4504 0.1523 0.1923 0.1449

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

252

MEt3R scores: 0.0893 0.4622 0.0942 0.1717 0.1114

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

285

MEt3R scores: 0.2926 0.1685 0.1493 0.1454 0.0852

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

297

MEt3R scores: 0.4125 0.2727 0.3697 0.2200 0.1936

- Figure 10. Visualization of MEt3R score [1] maps on LayeredFlow dataset [93]. We report, from left to right, the original left image for four samples in the dataset, followed by the MEt3R score maps computed between it and the right images generated by different methods. The coloring is according to the magma colormap, with green regions representing occlusions (discarded by MEt3R when computing the average score). Under each score map, we report the global score computed by MEt3R (the lower, the better).

[Figure 138]

Figure 11. Qualitative results on Middlebury [69] and DrivingStereo [97] datasets.

although these areas should overlap in a stereo setup. By contrast, such regions are often much narrower (or absent) in the Lyra and StereoSpace score maps.

Fig. 10 collects six samples from the LayeredFlow dataset

[93]. The multi-layered geometry peculiar to these scenes leads to a significant increase in the MEt3R scores, in particular for warping-based solutions. As discussed, this effect is mostly related to the use of estimated depth, which fails to

[Figure 139]

Figure 12. Qualitative results on Booster [63] and LayeredFlow [93] datasets.

account for the multiple layers in the scene. On the contrary, StereoSpace shines in these cases, as it does not inherit the limitations inherent to warping-based methods.

Further Qualitative Comparisons. Finally, to highlight the superior realism of StereoSpace, we present extensive

qualitative comparisons with images generated by all competing methods considered in our evaluation. Fig. 11 reports two samples from Middlebury [69] and two from the DrivingStereo [97] datasets. In the Middlebury examples, we highlight the bleeding artifacts introduced by ZeroStereo,

as well as the interpolation effects between foreground and background objects produced by StereoDiffusion, clearly visible in the Shopvac scene, and both caused by warping. We can also appreciate how Lyra itself, despite its high realism, tends to introduce oversmoothing. On DrivingStereo, we observe fewer artifacts due to simpler scene geometry and smaller disparities from larger depths. Nevertheless, StereoSpace still demonstrates superior reconstruction of thin structures.

Fig. 12 shows two scenes from Booster [63] and two from LayeredFlow [93]. At the top, we can notice how reflective and transparent objects in general are a significant challenge for methods relying on estimated depth, as highlighted by the artifacts produced in correspondence with the mirror or the deformations visible in the jars. At the bottom, depthbased approaches again struggle to deal with transparent surfaces that induce multiple depth layers, while StereoSpace maintains more faithful geometry and appearance.

