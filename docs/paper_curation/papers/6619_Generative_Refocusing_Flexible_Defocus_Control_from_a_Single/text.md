## Generative Refocusing: Flexible Defocus Control from a Single Image

##### Chun-Wei Tuan Mu1, Cheng-De Fan1, Jia-Bin Huang2, and Yu-Lun Liu1

###### 1 National Yang Ming Chiao Tung University, 2 University of Maryland, College Park

# arXiv:2512.16923v3[cs.CV]18Mar2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Input Image Focus to middle Focus to backgeound

Input Image Medium bokeh No bokeh (all-in-focu)

(a) Aperture size control (b) Focus plane control

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Input Image

Medium bokeh, Focus to backgeound

Input Image heart-shape bokeh

(c) Aperture shape control

(d) Composite control

Fig. 1: Generative refocusing with controllable bokeh. Our model turns a single input image into a virtual camera that users can control. This enables a range of adjustments after image capture. (a) demonstrates aperture size control, allowing the user to change the depth of field from strong bokeh to an all-in-focus image. (b) demonstrates focus plane control by shifting the sharp region from the middle subject to the background. (c) highlights aperture-shape control, synthesizing a creative heartshaped bokeh from point lights in the scene. (d) shows composite control, where both the focus plane and aperture size are adjusted simultaneously to reframe the subject.

Abstract. Depth-of-field control is essential in photography, but achieving perfect focus often requires multiple attempts or specialized equipment. Single-image refocusing is still difficult. It involves recovering sharp content and creating realistic bokeh. Current methods have significant drawbacks. They require all-in-focus inputs, rely on synthetic data from simulators, and have limited control over the aperture. We introduce Generative Refocusing, a two-step process that uses DeblurNet to recover all-in-focus images from diverse inputs and BokehNet to create controllable bokeh. This method combines synthetic and real bokeh images to achieve precise control while preserving authentic optical characteristics. Our experiments show we achieve top performance in defocus deblurring, bokeh synthesis, and refocusing benchmarks. Additionally, our Generative Refocusing allows custom aperture shapes. Project page: https://generative-refocusing.github.io/

Keywords: Refocus · Defocus Deblur · Bokeh Synthesis

### 1 Introduction

Controlling depth of field and focus is essential for artistic photography. It helps guide the viewer’s eye through selective focus and bokeh effects. Achieving perfect focus often requires multiple shots or specialized equipment, which can be challenging for everyday photographers. Single-image refocusing removes these obstacles. It allows for focus and bokeh adjustments after the photo is taken. The main challenge is recovering sharp details from blurred areas while creating natural-looking bokeh, while still giving users fine control.

Existing research has addressed various aspects of this problem. Defocus deblurring methods [1,8,32,53,65,71,82] aim to recover sharp images. Diffusion approaches [36, 37] show promise for addressing spatially variant blur. Bokeh synthesis methods [9,27,43,57] focus on rendering realistic depth-of-field. Recent efforts include physics-based [33,34,45], neural [43,44], and diffusion-based [19, 69, 80] approaches. However, most methods handle either deblurring or bokeh synthesis individually. End-to-end refocusing often requires specialized capture setups [2,35,41].

Despite these advances, three main limitations remain (Tab. 1). First, most methods rely on all-in-focus inputs and accurate depth maps. This restricts their use to images that already have defocus blur. Second, while synthetic data [43, 69, 87] can be used for training, its realism is limited by the fidelity of the simulators. This approach often fails to capture detailed visual appearances. On the other hand, real datasets [57,64] suffer from a critical limitation: they are restricted to isolated one-dimensional variations (i.e., solely aperture sweeps or focus plane sweeps). Consequently, they fail to provide the joint distribution of focus and aperture changes required to train a model for flexible, dual-parameter refocusing. Third, current methods typically support only aperture size, not its shape. This restriction limits the possibilities for creative bokeh effects.

To tackle these issues, we present Generative Refocusing (GenRefocus), a flexible single-image refocusing system built on a two-stage design. Our DeblurNet module produces a sharp, in-focus image from a variety of inputs. It uses a diffusion model guided by initial deblurring predictions. Our BokehNet generates fully customizable bokeh, accounting for user-defined focus planes, bokeh intensity, and aperture shapes (Fig. 1). A significant advancement of our method is the proposed training scheme. While we utilize synthetic data to maintain geometric consistency, our key contribution lies in handling real-world data. By leveraging auxiliary information and dedicated processing, we complete the missing input or control signals (i.e., defocus maps) in existing real-world datasets. This strategy enables our model to effectively learn authentic lens characteristics that simulators fail to capture. Our approach achieves top performance in defocus deblurring, bokeh generation, and refocusing, while maintaining the natural consistency of scenes.

Our main contributions are:

### Table 1: Comparison across methods. ✓ = supported; ✗ = not supported;

△ = supported with caveat. Flexible input indicates compatibility with both all-infocus (AIF) and defocused images. BokehMe and Bokehdiff require AIF inputs, while Learn2refocus strictly uses defocused inputs. ⋆For BokehMe, only the classical renderer variant supports aperture-shape control. †Bokehlicious typically assumes an all-in-focus input and requires fine-tuning for different input types. ‡Learn2Refocus generates focal stacks at discrete depth planes rather than supporting point-based focus selection.

Real training data Supported user control types

Synthetic training data

Flexible input?

Method

Focus point

Focussweep

Single images

Aperture size

Aperture shape

Aperturesweep

Defocus map

BokehMe [43] ✗ ✓ ✗ ✗ ✗ ✓ △⋆ ✓ ✓ Bokehlicious† [57] ✓ ✗ ✗ ✓ ✗ ✓ ✗ ✗ ✗ BokehDiff [87] ✗ ✓ ✗ ✗ ✗ ✓ ✗ ✓ ✓ DiffCamera [69] ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✓ ✗ Learn2Refocus [64] ✗ ✗ ✓ ✗ ✗ ✗ ✗ △‡ ✗ GenRefocus (Ours) ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

- – A flexible refocusing pipeline accepts images in any focus state. It provides users with control over the focus plane, bokeh intensity, and aperture shape via two-stage decomposition.
- – A novel training scheme that overcomes the limitations of existing real-world datasets by completing absent input or control signals. This strategy successfully unites the geometric consistency of synthetic data with the authentic aberrations of real lenses.
- – The system performs well across defocus deblurring, bokeh synthesis, and refocusing. This has been validated on existing benchmarks (DPDD [1], RealDOF [32]) and our new light-field datasets (LF-Bokeh, LF-Refocus). It also has applications in creating aperture shapes.

### 2 Related Work

Diffusion Models for Image Restoration. Diffusion models [22,61,66,79] advance image restoration via generative priors, extending to zero-shot restoration and artifact removal. Methods evolved from pixel-space [54] to efficient latentspace approaches [51,74], enabling faster inference. ResShift [81] cuts steps from 1000 to 4–15 via residual shifting. Deblurring has progressed from merging diffusion priors with regression tasks, such as matting [72] or hierarchical restoration [12], to learning spatially varying kernels in latent space [30]. Cascaded pipelines [23] refine at multiple resolutions, while principled frameworks [39,40] treat degradations as stochastic processes. Training-free methods [28,88] enable plug-and-play restoration. Unlike these general methods, we target spatiallyvarying defocus blur with a two-stage pipeline explicitly separating deblurring from bokeh synthesis for controllable refocusing.

Defocus Deblurring. Defocus deblurring recovers sharp images from spatially varying blur caused by limited depth of field. Early deconvolution [4,

| |[Figure 18]|
|---|---|
| | |

🔥: Trainable Bokeh Level 𝐾

[Figure 19]

𝐷 = |𝐷 - 𝐷 | ∗ 𝐾 Defocus map estimation

Focus Plane 𝐷

Depth Map 𝐷

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

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

❌ ✅ ❌ All-in-focus Image

Latent 𝑋

Latent 𝑋

Input 𝐼

Defocus map 𝐷

Aperture Shape 𝐴 (Optional)

𝐼

Encoder 𝐸

Encoder 𝐸 Encoder 𝐸

Encoder 𝐸

Concat

Concat

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

LoRA 🔥 LoRA 🔥

[Figure 39]

[Figure 40]

DiT BokehNet Iterative denoising

DiT DeblurNet Iterative denoising

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Decoder 𝐷

Decoder 𝐷

|[Figure 49]|
|---|

|[Figure 50]|
|---|

All-in-focus Image 𝐼

Output 𝐼

(a) Defocus Deblurring (b) Shallow Depth-of-Field Synthesis

- Fig. 2: Pipeline Overview. Our method decomposes single-image refocusing into two stages: (a) Defocus Deblurring and (b) Shallow Depth-of-Field Synthesis. (a) The noisy latent Xt and the encoded blurry input Iin are concatenated into a unified token sequence St. DeblurNet iteratively denoises this sequence to reconstruct a high-quality all-in-focus image Iaif. (b) Following the same unified-sequence design, BokehNet conditions on Iaif and the defocus map Ddef to synthesize the refocused output Iout, with an optional aperture-shape condition A. The VAE encoder E and decoder D map images to latent representations for the DiT backbone. The defocus map Ddef is computed from the estimated depth map D [7] together with the userspecified focus plane Dfocus and bokeh level K.

89] introduced artifacts at depth transitions. Dual-pixel sensors [1], quad-pixel data [8], and disparity-aware techniques [32,68,77] improved blur–depth modeling and the recovery of all-in-focus images from focal stacks. Architectures shifted from CNNs [10] to transformers [65,71], with gains from implicit representations [17, 47] and multi-scale attention [85]. Recent work includes visionlanguage fusion [75] for semantic estimation and a shift from supervised [49] to diffusion-based learning [36,37], including unpaired data. While these excel as standalone solutions, we position deblurring as the first stage of refocusing, using FLUX’s [58] generative prior to enable controllable bokeh synthesis from any input.

Bokeh Rendering. Bokeh rendering progressed from physically based scattering [33,34,45] and differentiable rendering [60] to neural methods combining ray tracing with learned enhancements [43,44]. Learning-based approaches evolved from fixed apertures [27] to variable f-stops [9,57], extending to video [78] and

- 3D scenes [26, 59, 70]. Diffusion models enabled bokeh control [19, 80], mainly for text-to-image synthesis. These methods require all-in-focus inputs and ac-

curate depth. We relax this constraint via flexible input handling, which accepts defocused or fully in-focus images.

Single-Image Refocusing. Post-capture refocusing evolved from deconvolution [4,84] and GANs [55] to recent diffusion models [46], yet significant limitations remain (see Tab. 1). Constrained by conditioning on explicit focus coordinates, DiffCamera [69] requires fixed-resolution inputs (512 × 512) and suffers performance drops on arbitrary aspect ratios. Furthermore, it inherits simulator artifacts from its purely synthetic training. Learn2Refocus [64] trains on real data but lacks aperture-size control and generates a focal stack at predefined depth planes that may not align with the user’s target focus. Additionally, 3D representations [31,38,59,70] offer controllable DoF but require multi-view capture. To address these gaps, we combine three capabilities: (1) Hybrid training scheme mixing synthetic and real data for authentic optics, (2) single-image flexibility accepting any input without preprocessing, and (3) comprehensive control over focus, intensity, and aperture shape.

Camera-Conditioned Diffusion. Conditioning diffusion models on camera parameters progressed from extrinsic control for view synthesis [13, 24] and video generation [21,73], to intrinsic control of focal length, aperture, and exposure [11,18]. Direct conditioning often yields inconsistent content. Recent work uses temporal modeling [80], explicit calibration [6, 15], or transformer-based pose handling [3,56]. These focus on generation; we operate in editing, manipulating images with scene consistency. By decoupling deblurring and bokeh stages, we achieve efficient single-image manipulation without multi-frame overhead.

### 3 Method

Single-image refocusing couples two operations: recovering sharp content in outof-focus regions and applying controllable bokeh to originally sharp areas. Blur magnitude is dictated by a defocus map parameterized by a user-specified focus point and the scene depth. When the input is blurry, monocular depth estimation is brittle, undermining precise control over the defocus map and often yielding misfocus or artifacts. We propose Generative Refocusing (GenRefocus), which decomposes the task into defocus deblurring and bokeh synthesis, and comprises two models, DeblurNet and BokehNet, that jointly enable precise and controllable refocusing (see Fig. 2 for an overview of the pipeline); Such a flexible pipeline allows us to exert greater control over input conditions (see Tab. 1).

#### 3.1 Stage 1: Defocus Deblurring

Given a blurry input image Iin, we follow the conditioning design of [62,63]. At each denoising step t, DeblurNet processes a single unified token sequence

St = Xt ; E(Iin) , (1)

(a) Synthetic data

(c) LFDOF & RealBokeh

(b) ITW dataset

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

Pixel ratio F-number 𝐹 Focal length 𝑓

Bokeh level K Focus Plane S

Real AIF image

Real bokeh image

Real bokeh image Real AIF image

Random parameters

EXIF metadata

Data format for learning defocus blur control

[Figure 55]

[Figure 56]

[Figure 57]

Depth maps D

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

𝐷 = |𝐷 - 𝐷 | ∗ 𝐾

Bokeh level K

Focus Plane D

Bokeh images AIF images Defocus maps 𝐷

- Fig. 3: Training data generation. Each training sample consists of five components: (i) a bokeh image, (ii) an all-in-focus (AIF) image, (iii) a depth map D, (iv) a bokeh

level K, and (v) a focus plane Dfocus. We construct these samples via three routes: (a) Synthetic data. Given real AIF images and depth maps D, we compute a defocus map Ddef parameterized by randomly selected bokeh level K and focus plane Dfocus, and feed it into a bokeh renderer [43] to synthesize corresponding bokeh images. (b) ITW dataset [19]. Given real bokeh images, DeblurNet recovers an AIF image. We then estimate depth and extract a foreground mask [86] to define the estimated focus plane Dfocus. The bokeh level K is computed from the EXIF metadata and the estimated Dfocus following the formulation in [19]. (c) LFDOF [52] and RealBokeh. For real pairs, we obtain Dfocus as in (b), and follow Eq. (2) to estimate the bokeh level K.

where Xt denotes the noisy image latent tokens at timestep t, E is a pretrained VAE encoder that maps the input image to latent condition tokens, and [·; ·] denotes token concatenation. After iterative denoising, we decode the image latent part with the VAE decoder D to obtain the all-in-focus output Iaif (Fig. 2(a)). We train DeblurNet in a supervised manner using real paired data [1,57].

#### 3.2 Stage 2: Shallow Depth-of-Field Synthesis

Given an all-in-focus image Iaif and a user-specified focus plane Dfocus with bokeh level K, we define the defocus map Ddef as

Ddef = K · |D − Dfocus| , (2)

where D is the monocular depth map estimated from Iaif using an off-the-shelf depth estimator [7]. Following the unified-token conditioning design in Stage 1, BokehNet concatenates the condition tokens of Iaif and Ddef with the noisy latent tokens at each denoising step. BokehNet then iteratively denoises Xt to synthesize the refocused output Iout, as illustrated in Fig. 2(b).

Training data and supervision. Training BokehNet requires (Iaif,Iout,Ddef). Because accurately paired real supervision is scarce, we construct supervision via

three complementary sources (Fig. 3): (a) synthetic data, (b) ITW dataset [19], and (c) LFDOF [52] and RealBokeh [57].

- (a) Synthetic data. We pretrain with synthetic data: starting from real all-infocus images and their estimated depth map D, we randomly sample a focus

plane Dfocus and a target bokeh level K (Fig. 3(a)). We then construct Ddef using Eq. 2 and use a simulator [43] to render the corresponding target bokeh image consistent with Ddef. This synthetic pretraining helps the network modulate the circle of confusion according to Ddef; however, it is constrained by renderer bias and may introduce unrealistic artifacts.

- (b) ITW dataset. To capture real optics, we leverage ITW dataset [19] contains real bokeh images (see Fig. 3(b)). For each real image, we use DeblurNet to produce the AIF input and compute an approximate bokeh levelK following [19]:

K ≈

f2 Dfocus

2 F (Dfocus − f) × pixel_ratio, (3) where f is the focal length and F is the aperture f-number, both directly obtained from EXIF metadata. The pixel_ratio term accounts for differences in camera sensor size and image resolution. Although some devices may provide a focus-distance field in EXIF, it is frequently missing or noisy; therefore, we do not rely on EXIF for Dfocus. Instead, we compute the focus plane based on an in-focus mask and a monocular depth estimate, following the approach of [19]. Specifically, we estimate an in-focus mask M using BiRefNet [86] and produce a depth map D using a monocular depth estimator. The focus plane is then determined as the median depth within the masked in-focus area:

Dfocus = median D[M] . (4)

We then generate Ddef using Eq. 2, enabling BokehNet to effectively train on real single images. While these estimated AIF images and defocus maps contain inevitable noise, training on them inherently aligns with our single-image refocusing pipeline, making the model more robust to depth and deblurring artifacts.

- (c) LFDOF and RealBokeh. These datasets provide pairs but omit EXIF metadata or provide insufficient fields to estimate K (see Fig. 3(c)). To address

this, we first compute the focus-plane proxy Dfocus. Similar to (b), we employ BiRefNet [86] to obtain an initial in-focus mask M. However, due to the increased diversity and complexity of the scenes in these datasets, the initial estimate of M is sometimes unreliable. Rather than simply verifying and discarding unreliable cases, we introduce a manual refinement step. Specifically, we re-select a small yet reliable in-focus region to correct M, thereby accurately extracting Dfocus. This strategy preserves challenging samples rather than excluding them, thereby enabling the model to learn from a wider range of real-world scenarios.

With Dfocus established, we adopt a simulator-in-the-loop calibration of the bokeh level. Given an AIF image Iaif and estimated depth D, we sweep K and

- Table 2: Defocus deblurring benchmark on RealDOF [32] and DPDD [1] datasets. We report reference (LPIPS, DISTS) and no-reference (CLIP-IQA, MANIQA, MUSIQ) perceptual metrics. Best , second best , and third best results are highlighted.

RealDOF [32] DPDD [1] Method LPIPS ↓ DISTS ↓ CLIP-IQA ↑ MANIQA ↑ MUSIQ ↑ LPIPS ↓ DISTS ↓ CLIP-IQA ↑ MANIQA ↑ MUSIQ ↑ Input 0.5241 0.2865 0.3562 0.2213 28.7087 0.3485 0.1827 0.4337 0.3325 45.5376 DRBNet [53] 0.2550 0.1312 0.3889 0.2609 38.9014 0.1819 0.1063 0.4142 0.3254 46.7229 Restormer [82] 0.2863 0.1573 0.4062 0.2642 40.8355 0.1762 0.1204 0.4332 0.3307 47.9872 INIKNet [47] 0.2851 0.1615 0.3984 0.2580 38.9959 0.1860 0.1235 0.4290 0.3265 47.1961 Bokehlicious [57] 0.2080 0.1070 0.4281 0.2746 42.2314 0.1598 0.0918 0.4322 0.3237 47.6057 DiffCamera [69] 0.4523 0.2140 0.3861 0.2667 37.9951 0.4048 0.2141 0.4438 0.3350 47.1452 GenRefocus (Ours) 0.2408 0.1126 0.4595 0.2884 43.5222 0.1440 0.0772 0.4755 0.3452 49.4122

##### choose the value whose rendered result best matches the real bokeh target Ireal:

K⋆ = argmax

SSIM R(Iaif, D; Dfocus, K), Ireal , (5)

K∈ (Kmin, Kmax)

where R denotes our physically guided renderer. The selected K⋆ is then used as the pseudo-bokeh-level label for training, provided that its corresponding SSIM exceeds a predefined threshold to ensure reliable supervision. Finally, the defocus map Ddef is constructed accordingly following Eq. 2.

#### 3.3 Bokeh-Shape Aware Synthesis

State-of-the-art learning-based bokeh synthesis methods deliver compelling results but do not expose aperture-shape control. Physics-based renderers, in principle, support arbitrary shapes, yet public implementations typically omit this functionality. We therefore explore explicit shape-aware control using only a bokeh shape image within a trainable framework.

Simulator and Data. Real photographs exhibiting diverse bokeh shapes are rare, and paired AIF–bokeh examples are even scarcer. We thus rely on simulation. A key observation is that when an all-in-focus (AIF) image lacks pointlight stimuli, the simulated bokeh carries weak shape cues, making the model reluctant to learn shape-conditioned responses. To address this, we synthesize a PointLight-1K dataset designed to reveal aperture shape. We also extend the classical BokehMe renderer to scatter through a given shape: given an AIF image Iaif, depth D, focus-plane proxy Dfocus, bokeh level K, and a shape kernel s (binary/raster PSF), the simulator R renders

Isyn = R Iaif, D; Dfocus, K, s , (6)

##### yielding paired supervision for shape-aware training.

- (a) Input Image INIKNet [45] Bokehlicious [55] DiffCamera [67] GenRefocus (Ours)

[Figure 64]

Input GT DRBNet [51] Restormer [80]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- (b) Input Image INIKNet [45] Bokehlicious [55] DiffCamera [67] GenRefocus (Ours)

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Input GT DRBNet [51] Restormer [80]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

- Fig. 4: Qualitative comparison on defocus deblurring. Visual results on (a) RealDOF [32] and (b) DPDD [1] datasets. Blue boxes on the left indicate cropped regions shown in detail. In (a), most methods fail to resolve the text, whereas both our method and Bokehlicious [57] successfully recover the words “WELCOME” and “OPEN HOURS”. However, Bokehlicious produces noticeable distortions in the right-side characters (e.g., “ME” and “RS”), whereas our approach faithfully restores the original text structure. For the challenging example (b) with severe defocus blur, all baseline methods remain highly blurry and fail to recover meaningful details. In contrast, our method synthesizes a relatively clear and visually compelling output.

Shape-conditioned fine-tuning. To incorporate the aperture shape A, we append its tokens directly to the unified sequence. To ensure that shape edits do not regress the learned bokeh synthesis, we freeze all original LoRA weights and introduce a new, trainable LoRA module. Only this dedicated LoRA is fine-tuned to handle the shape conditioning.

#### 3.4 Pre-deblur Module

We also study a variant that augments DeblurNet with a conservative pre-deblur prior. We train this variant using the same data as the base DeblurNet. Specifically, it condition on Iin and Ipd, where Ipd is generated by an off-the-shelf defocus deblurring model. In the implementation, we use a non-generative defocus model [53], whose outputs are typically content-faithful but overly smooth. Qualitative results for this variant are shown in the supplementary material.

#### 3.5 Implementation Details

While our backbone [58] intrinsically supports arbitrary resolutions and aspect ratios without resizing, and our conditioning setup is completely coordinateindependent, ultra-high-resolution inputs (e.g., 4K) still incur prohibitive VRAM consumption and quality degradation. We implement a tiling strategy inspired by [5] during inference. Specifically, we process the image as overlapping patches

- Table 3: Bokeh synthesis benchmark. Evaluation on LF-Bokeh. Following [43,87], we use per-image binary search for optimal K. †Since Bokehlicious [57] uses f-stop rather than K as its control input, we perform a binary search over f-stop values. Due to this fundamentally different control mechanism, we exclude it from the LVCorr evaluation. ‡Note that Bokeh diffusion [19] targets the text-to-image task. To adapt it for image editing, we use textual inversion [61].

Fidelity Controllability Method LPIPS ↓ DISTS ↓ CLIP-I ↑ LVCorr ↑ BokehMe [43] 0.1228 0.0744 0.9511 0.9940 Bokehlicious† [57] 0.1799 0.1062 0.9304 BokehDiff [87] 0.1708 0.0933 0.9192 0.8976 Bokeh Diffusion‡ [19] 0.4999 0.2529 0.7569 0.8954 DiffCamera [69] 0.1429 0.0780 0.9441 0.5632 GenRefocus (Ours) 0.0833 0.0487 0.9713 0.9368

and blend the tile-wise denoised results. This strategy enables us to process images exactly at their original resolutions, fully preserving image fidelity.

- 4 Experiments

#### 4.1 Setup

Backbone and training details. Our backbone is FLUX-1-dev [58], finetuned via LoRA [25] with a conditioning scheme following [62, 63]. DeblurNet employs LoRA rank r=128, while BokehNet uses r=64. Both models are trained with a per-GPU batch size of 1 and gradient accumulation over 8 steps on 4× RTX A6000 GPUs. DeblurNet is trained for 60K steps. BokehNet is trained in two stages: (i) 40K steps on synthetic data, and (ii) 60K steps on real data.

Runtime and computational cost. During inference, each stage employs 28 denoising steps. Without the tiling strategy, our pipeline consumes approximately 40 GB of VRAM and takes approximately 50 seconds per image on a single NVIDIA RTX A6000 GPU.

Datasets. We train DeblurNet on 3.5K pairs sourced from DPDD [1] and a subset of RealBokeh [57]. For BokehNet, to simultaneously maintain realism and controllability, we utilize a hybrid dataset comprising ∼70K synthetic pairs derived from [27,80], combined with approximately 26K real examples sourced from ITW dataset [19], RealBokeh [57], and LFDOF [52].

Benchmarks and Metrics We evaluate our method on three tasks using reference-based fidelity and no-reference image quality assessment metrics.

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

Input GT BokehMe Bokehlicious BokehDiff Diffcamera GenRefocus [43] [57] [87] [69] (Ours)

- Fig. 5: Qualitative comparison on bokeh synthesis benchmark. Results on LFBokeh with zoomed patches (blue and orange boxes) highlighting detail quality. Our method synthesizes bokeh effects that better match ground truth with realistic blur gradients and natural occlusion handling.

- (i) Defocus Deblurring. We conduct deblurring experiments on RealDOF and DPDD datasets. To assess reconstruction fidelity and perceptual naturalness, we report LPIPS [83], DISTS [16], and no-reference metrics including CLIPIQA [67], MANIQA [76], and MUSIQ [29].
- (ii) Bokeh Synthesis. We introduce LF-Bokeh, featuring 200 images with diverse focus planes and aperture sizes synthesized from light-field captures [14, 50]. In real-world scenarios, the absence of ground-truth depth maps and reference bokeh levels K precludes the computation of accurate defocus maps, making the direct evaluation of fidelity and controllability challenging. To assess fidelity, following prior works [43,87], we conduct a per-image binary search over K and select the value that maximizes SSIM with the target, reporting LPIPS, DISTS, and CLIP-I [48]. To quantify controllability under these constraints, we adopt the LVCorr metric following [19,69,80]. Specifically, for images generated from the same all-in-focus input with a fixed focus plane across varying bokeh levels K, we compute the Pearson correlation coefficient (CorrCoef) between these K values and the Laplacian variance trend, thereby reflecting the model’s accuracy in rendering the physical blur progression.
- (iii) Refocusing. The LF-Refocus dataset consists of 400 source–target pairs from LF-Bokeh. Similar to bokeh synthesis, we employ an optimal K search for each pair. We report LPIPS, DISTS, and CLIP-I for fidelity, alongside CLIP-IQA and MUSIQ to evaluate perceptual quality across varying depths of field.

- Table 4: Quantitative comparison on refocusing benchmark. Results on our LF-Refocus dataset. We compare our method against current state-of-the-art refocusing pipelines [64,69]. Optimal bokeh level K is determined via per-image binary search following [43,87].†Learn2Refocus [64] takes a specific focal position as the input condition to synthesize the remaining images in the focal stack. To ensure coverage of the full range of focal planes, we perform inference twice separately: once conditioned on the minimum focal position (1) and once on the maximum (9). We then evaluate all outputs from both generated stacks and report the one yielding the highest SSIM.

Method LPIPS ↓ DISTS ↓ CLIP-I ↑ CLIP-IQA ↑ MUSIQ ↑

DiffCamera [69] 0.1426 0.0782 0.9401 0.4669 40.5869 Learn2Refocus† [64] 0.1706 0.0967 0.9322 0.4709 38.3440 GenRefocus (Ours) 0.1324 0.0759 0.9571 0.5803 52.9706

#### 4.2 Comparison

Defocus deblurring. As reported in Tab. 2, DeblurNet outperforms competing methods across most referenced and non-referenced metrics. While our method yields slightly lower reference scores than Bokehlicious [57] on RealDOF [52], it consistently produces more visually appealing results and demonstrates greater robustness to diverse input conditions, such as grayscale photographs and lowquality, blurry images. Extensive visual comparison supporting this flexibility is provided in the supplementary material. Qualitatively, this perceptual superiority is evident in Fig. 4. As shown in Fig. 4(a), our approach achieves faithful recovery of text details with well-preserved edges. Furthermore, the scene in Fig. 4(b) presents a particularly challenging case: while competing methods recover only a blurry dark stripe and introduce background distortions, DeblurNet reconstructs more consistent content and delivers compelling results.

Bokeh synthesis. As presented in Tab. 3, our approach outperforms all other methods across fidelity metrics. In terms of controllability, which measures how well the generated bokeh aligns with the conditioned bokeh level K, BokehMe [43] achieves near-perfect control owing to its hybrid design that incorporates a classical, physically motivated renderer. Nevertheless, our method outperforms all diffusion-based baselines, demonstrating that our learning strategy effectively preserves both realism and accuracy. Furthermore, it is worth noting the specific limitations of competing methods: DiffCamera [69] requires inputs with a fixed resolution and aspect ratio due to its reliance on explicit focus point conditioning, rendering it incompatible with tiling strategies; consequently, resizing operations severely degrade its controllability. Meanwhile, Bokeh Diffusion [19] relies on an inversion [61] process that not only prolongs inference time but also strips the image of its original high-frequency details, thereby lowering overall fidelity. The qualitative results shown in Fig. 5.

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

Input GT DiffCamera Learn2Refocus GenRefocus [69] [64] (Ours)

- Fig. 6: Qualitative comparison on the refocusing benchmark. Results on the LF-Refocus dataset with zoomed patches (blue and orange boxes) highlight the fidelity of focal and defocused regions. We observe that DiffCamera [69] struggles to reconstruct sharp structural details due to its strict input conditions and often assigns incorrect blurriness levels to background regions. Similarly, Learn2Refocus [64] fails to achieve accurate focus-plane placement, resulting in unintended blur of the target object. In contrast, our method achieves precise refocusing at the target plane while synthesizing photorealistic bokeh, closely matching the ground truth across diverse scenes.

Refocusing. As shown in Tab.4, our approach achieves the best results on both fidelity and perceptual metrics, outperforming other baselines. Qualitatively, Fig. 6 shows that competing methods tend to produce spatially ambiguous blur and fail to place the focal plane correctly, resulting in uniformly soft content. In contrast, our approach restores fine details at the intended focus while synthesizing physically consistent bokeh elsewhere.

#### 4.3 Ablation Study

Direct refocusing vs. deblurring then bokeh synthesis. We evaluate a baseline that directly predicts a refocused image from a blurry input and a defocus map. Trained on identical synthetic data using the BokehNet backbone, this direct approach significantly underperforms our two-stage pipeline (Tab. 5). We attribute this gap to two factors: (1) control fidelity: estimating depth from blurry inputs degrades defocus map accuracy, causing misaligned blur fields; and (2) data leverage: a direct formulation precludes injecting tailored real-data supervision into separate deblurring and synthesis subtasks.

- Table 5: Comparison of pipeline designs. Comparing direct refocusing with our proposed deblurring, then the bokeh synthesis approach on LF-Refocus.

Method LPIPS ↓ DISTS ↓ CLIP-I ↑ Direct refocusing 0.1723 0.0976 0.9345 Deblurring then bokeh synthesis (Ours) 0.1324 0.0759 0.9571

- Table 6: Ablation on BokehNet training datasets. Comparing combinations of datasets: (a) Synthetic data, (b) ITW dataset [19], and (c) LFDOF [52] and RealBokeh [57] on LF-Bokeh. The results show that combining all three yields the best optical characteristics.

Datasets Metrics (a) Synthetic (b) ITW (c) LFDOF & RealBokeh LPIPS ↓ DISTS ↓ CLIP-I ↑

✓ - - 0.1289 0.0738 0.9461 ✓ ✓ - 0.1156 0.0665 0.9529 ✓ - ✓ 0.0972 0.0560 0.9666 ✓ ✓ ✓ 0.0833 0.0487 0.9713

BokehNet training ablation. To quantify the impact of different training data distributions, we evaluate BokehNet training under various dataset combinations on LF-Bokeh. We consider three primary data sources: (a) synthetic data, (b) ITW dataset [19], and (c) LFDOF [52] and RealBokeh [57]. As shown in Tab. 6, a baseline trained purely on (a) provides a foundational performance but leaves room for improvement. Introducing either (b) or (c) to the synthetic baseline noticeably enhances performance. Ultimately, the comprehensive setting that combines all three sources yields the most substantial improvements, delivering the best optical characteristics and real-world generalization.

Aperture shape control. As illustrated in Fig. 7, after we fine-tune BokehNet on our point-light 1K dataset, the model can synthesize bokeh with diverse, userspecified aperture shapes.

### 5 Conclusion

We present Generative Refocusing, a two-stage diffusion framework turning single images into controllable virtual cameras. By decoupling into DeblurNet (deblurring) and BokehNet (bokeh synthesis), our approach handles arbitrary inputs with intuitive controls over focus plane, bokeh intensity, and aperture shape. Through training on various data sources, GenRefocus learns authentic optical behavior beyond simulators. Experiments on all benchmarks show consistent improvements, enabling aperture-shape editing.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Input Triangle Heart Star

- Fig. 7: Controllable aperture shape synthesis. Given an input image, our BokehNet can synthesize bokeh effects with custom aperture shapes (triangle, heart, star) by conditioning on shape exemplars. Background point lights exhibit the specified aperture geometry while maintaining scene consistency.

InputInputImage DepthDepthMapMap DefocusDefocusMapMap OutputOutput

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

- Fig. 8: Failure case: transparent surfaces. With a transparent glass, focusing on the foreground may fail to correctly blur the background due to unreliable defocus maps.

[Figure 116]

[Figure 117]

[Figure 118]

Input Ours GT

Fig. 9: Failure case: hallucinated details. With severely blurred inputs, our model may hallucinate incorrect results.

Limitations and future work. GenRefocus relies on the defocus map; Imperfections in the depth estimate can propagate to the defocus map and yield incorrect blur assignment. Transparent surfaces are particularly challenging, as monocular depth typically captures the depth of the front transparent surface while ignoring the background seen through it (see Fig. 8). In cases of severe blur, our diffusion-based method may fail to recover fine details and instead hallucinate textures (see Fig. 9), rendering it unsuitable for precision-critical fields such as scientific imaging or safety-sensitive applications. Complex aperture shapes require simulator-driven training data. Future work should generalize to richer aperture vocabularies, including user-drawn designs.

### References

- 1. Abuolaim, A., Brown, M.S.: Defocus deblurring using dual-pixel data. In: ECCV

(2020) 2, 3, 4, 6, 8, 9, 10, 21

- 2. Alzayer, H., Abuolaim, A., Chan, L.C., Yang, Y., Lou, Y.C., Huang, J., Kar, A.: Dc2: Dual-camera defocus control by learning to refocus. In: CVPR (2023) 2
- 3. Bahmani, S., Skorokhodov, I., Siarohin, A., Menapace, W., Qian, G., Vasilkovsky, M., Lee, H.Y., Wang, C., Zou, J., Tagliasacchi, A., et al.: Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781

(2024) 5

- 4. Bando, Y., Nishita, T.: Towards digital refocusing from a single photograph. In: PG (2007) 3, 5
- 5. Bar-Tal, O., Yariv, L., Lipman, Y., Dekel, T.: Multidiffusion: Fusing diffusion paths for controlled image generation. In: ICML (2023) 9
- 6. Bernal-Berdun, E., Serrano, A., Masia, B., Gadelha, M., Hold-Geoffroy, Y., Sun, X., Gutierrez, D.: Precisecam: Precise camera control for text-to-image generation. In: CVPR (2025) 5
- 7. Bochkovskii, A., Delaunoy, A., Germain, H., Santos, M., Zhou, Y., Richter, S.R., Koltun, V.: Depth pro: Sharp monocular metric depth in less than a second. In: ICLR (2025) 4, 6
- 8. Chen, H., Xie, Y., Peng, X., Sun, L., Su, W., Yang, X., Liu, C.: Quad-pixel image defocus deblurring: A new benchmark and model. In: CVPR (2025) 2, 4
- 9. Chen, K., Yan, S., Jiang, A., Li, H., Wang, Z.: Variable aperture bokeh rendering via customized focal plane guidance. arXiv preprint arXiv:2410.14400 (2024) 2, 4
- 10. Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: ECCV (2022) 4
- 11. Chen, S.K., Yen, H.L., Liu, Y.L., Chen, M.H., Hu, H.N., Peng, W.H., Lin, Y.Y.: Learning continuous exposure value representations for single-image hdr reconstruction. In: ICCV (2023) 5
- 12. Chen, Z., Zhang, Y., Liu, D., Gu, J., Kong, L., Yuan, X., et al.: Hierarchical integration diffusion model for realistic image deblurring. In: NeurIPS (2023) 3
- 13. Cheng, T.Y., Gadelha, M., Groueix, T., Fisher, M., Mech, R., Markham, A., Trigoni, N.: Learning continuous 3d words for text-to-image generation. In: CVPR

(2024) 5

- 14. Dansereau, D.G., Girod, B., Wetzstein, G.: Liff: Light field features in scale and depth. In: CVPR (2019) 11
- 15. Deng, J., Yin, W., Guo, X., Zhang, Q., Hu, X., Ren, W., Long, X.X., Tan, P.: Boost 3d reconstruction using diffusion-based monocular camera calibration. In: ICCV (2025) 5
- 16. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44(5), 2567–2581 (2020) 11
- 17. Fan, C.D., Chang, C.W., Liu, Y.R., Lee, J.Y., Huang, J.L., Tseng, Y.C., Liu, Y.L.: Spectromotion: Dynamic 3d reconstruction of specular scenes. In: CVPR (2025) 4
- 18. Fang, I.S., Han, Y.H., Chen, J.C.: Camera settings as tokens: Modeling photography on latent diffusion models. In: SIGGRAPH Asia (2024) 5
- 19. Fortes, A., Wei, T., Zhou, S., Pan, X.: Bokeh diffusion: Defocus blur control in text-to-image diffusion models. In: SIGGRAPH Asia (2025) 2, 4, 6, 7, 10, 11, 12, 14, 21
- 20. Google DeepMind: Gemini models: Product overview. https://deepmind.google/ technologies/gemini/ (2025), accessed: November 21, 2025 23, 26
- 21. He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., Yang, C.: Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101

(2024) 5

- 22. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS

(2020) 3

- 23. Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M., Salimans, T.: Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research 23(47), 1–33 (2022) 3

- 24. Höllein, L., Božič, A., Müller, N., Novotny, D., Tseng, H.Y., Richardt, C., Zollhöfer, M., Nießner, M.: Viewdiff: 3d-consistent image generation with text-to-image models. In: CVPR (2024) 5
- 25. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR (2022) 10
- 26. Huang, Y., Luo, X., Wang, Q., Shen, L., Li, J., Sun, H., Huang, Z., Jiang, W., Cao, Z.: Bokehflow: Depth-free controllable bokeh rendering via flow matching. arXiv preprint arXiv:2511.15066 (2025) 4
- 27. Ignatov, A., Patel, J., Timofte, R.: Rendering natural camera bokeh effect with deep learning. In: CVPRW (2020) 2, 4, 10, 21
- 28. Kawar, B., Elad, M., Ermon, S., Song, J.: Denoising diffusion restoration models. In: NeurIPS (2022) 3
- 29. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: ICCV (2021) 11
- 30. Kong, L., Zou, D., Wang, F.L., Ren, J., Wu, X., Dong, J., Pan, J., et al.: Deblurdiff: Real-world image deblurring with generative diffusion models. In: NeurIPS (2025) 3
- 31. Lee, B., Lee, H., Sun, X., Ali, U., Park, E.: Deblurring 3d gaussian splatting. In: ECCV (2024) 5
- 32. Lee, J., Son, H., Rim, J., Cho, S., Lee, S.: Iterative filter adaptive network for single image defocus deblurring. In: CVPR (2021) 2, 3, 4, 8, 9
- 33. Lee, S., Eisemann, E., Seidel, H.P.: Depth-of-field rendering with multiview synthesis. ACM TOG 28(5), 1–6 (2009) 2, 4
- 34. Lee, S., Kim, G.J., Choi, S.: Real-time depth-of-field rendering using point splatting on per-pixel layers. In: Computer Graphics Forum. vol. 27, pp. 1955–1962 (2008) 2, 4
- 35. Levoy, M., Hanrahan, P.: Light field rendering. In: Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pp. 441–452 (2023) 2
- 36. Li, Y., Fang, H., Lei, X., Wang, Q., Hu, G., Dong, J., Li, Z., Lin, J., Liu, Q., Song, X.: Real-world defocus deblurring via score-based diffusion models. Scientific Reports 15(1), 22942 (2025) 2, 4
- 37. Liang, H., Chai, S., Zhao, X., Kan, J.: Swin-diff: a single defocus image deblurring network based on diffusion model. Complex & Intelligent Systems 11(3), 170

(2025) 2, 4

- 38. Liu, Y.L., Gao, C., Meuleman, A., Tseng, H.Y., Saraf, A., Kim, C., Chuang, Y.Y., Kopf, J., Huang, J.B.: Robust dynamic radiance fields. In: CVPR (2023) 5
- 39. Luo, W., Qin, H., Chen, Z., Wang, L., Zheng, D., Li, Y., Liu, Y., Li, B., Hu, W.: Visual-instructed degradation diffusion for all-in-one image restoration. In: CVPR

(2025) 3

- 40. Luo, Z., Gustafsson, F.K., Zhao, Z., Sjölund, J., Schön, T.B.: Image restoration with mean-reverting stochastic differential equations. In: ICML (2023) 3
- 41. Ng, R., Levoy, M., Brédif, M., Duval, G., Horowitz, M., Hanrahan, P.: Light field photography with a hand-held plenoptic camera. Ph.D. thesis, Stanford university

(2005) 2

- 42. OpenAI: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024) 22
- 43. Peng, J., Cao, Z., Luo, X., Lu, H., Xian, K., Zhang, J.: Bokehme: When neural rendering meets classical rendering. In: CVPR (2022) 2, 3, 4, 6, 7, 10, 11, 12, 21
- 44. Peng, J., Cao, Z., Luo, X., Xian, K., Tang, W., Zhang, J., Lin, G.: Bokehme++: Harmonious fusion of classical and neural rendering for versatile bokeh creation. IEEE TPAMI (2024) 2, 4

- 45. Potmesil, M., Chakravarty, I.: A lens and aperture camera model for synthetic image generation. SIGGRAPH (1981) 2, 4
- 46. Qin, X., Wang, Z., Li, F., Chen, H., Pei, R., Li, W., Cao, X.: Camedit: Continuous camera parameter control for photorealistic image editing. In: NeurIPS (2025) 5
- 47. Quan, Y., Yao, X., Ji, H.: Single image defocus deblurring via implicit neural inverse kernels. In: ICCV (2023) 4, 8
- 48. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021) 11
- 49. Ren, D., Shu, X., Li, Y., Wu, X., Li, J., Zuo, W.: Reblurring-guided single image defocus deblurring: A learning framework with misaligned training pairs. IJCV pp. 1–18 (2025) 4
- 50. Rěřábek, M., Ebrahimi, T.: New light field image dataset. In: QoMEX (2016) 11
- 51. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022) 3
- 52. Ruan, L., Chen, B., Li, J., Lam, M.L.: Aifnet: All-in-focus image restoration network using a light field-based dataset. IEEE TCI 7, 675–688 (2021) 6, 7, 10, 12, 14
- 53. Ruan, L., Chen, B., Li, J., Lam, M.L.: Learning to deblur using light field generated and real defocus images. In: CVPR (2022) 2, 8, 9
- 54. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image superresolution via iterative refinement. IEEE TPAMI 45(4), 4713–4726 (2022) 3
- 55. Sakurikar, P., Mehta, I., Balasubramanian, V.N., Narayanan, P.: Refocusgan: Scene refocusing using a single image. In: ECCV (2018) 5
- 56. Saxena, S., Hur, J., Herrmann, C., Sun, D., Fleet, D.J.: Zero-shot metric depth with a field-of-view conditioned diffusion model. arXiv preprint arXiv:2312.13252

(2023) 5

- 57. Seizinger, T., Vasluianu, F.A., Conde, M.V., Timofte, R.: Bokehlicious: Photorealistic bokeh rendering with controllable apertures. In: ICCV (2025) 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 21, 23, 25
- 58. Shakker Labs: Flux.1-dev-controlnet-union-pro. https : / / huggingface . co / Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro (2024), accessed: 2025-11-13 4, 9, 10, 22
- 59. Shen, L., Liu, T., Sun, H., Li, J., Cao, Z., Li, W., Loy, C.C.: Dof-gaussian: Controllable depth-of-field for 3d gaussian splatting. In: CVPR (2025) 4, 5
- 60. Sheng, Y., Yu, Z., Ling, L., Cao, Z., Zhang, X., Lu, X., Xian, K., Lin, H., Benes, B.: Dr. bokeh: Differentiable occlusion-aware bokeh rendering. In: CVPR (2024) 4
- 61. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: ICLR (2021) 3, 10, 12
- 62. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. In: ICCV (2025) 5, 10
- 63. Tan, Z., Xue, Q., Yang, X., Liu, S., Wang, X.: Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280 (2025) 5, 10
- 64. Tedla, S., Zhang, Z., Zhang, X., Xin, S.: Learning to refocus with video diffusion models. In: SIGGRAPH Asia (2025) 2, 3, 5, 12, 13
- 65. Tsai, F.J., Peng, Y.T., Lin, Y.Y., Tsai, C.C., Lin, C.W.: Stripformer: Strip transformer for fast image deblurring. In: ECCV (2022) 2, 4
- 66. Tsai, S.R., Chang, W.C., Lee, J.Y., Su, C.H., Liu, Y.L.: Lightsout: Diffusion-based outpainting for enhanced lens flare removal. In: ICCV (2025) 3
- 67. Wang, J., Chan, K.C., Loy, C.C.: Exploring clip for assessing the look and feel of images. In: AAAI (2023) 11

- 68. Wang, N.H., Wang, R., Liu, Y.L., Huang, Y.H., Chang, Y.L., Chen, C.P., Jou, K.: Bridging unsupervised and supervised depth from focus via all-in-focus supervision. In: ICCV (2021) 4
- 69. Wang, Y., Chen, X., Xu, X., Liu, Y., Zhao, H.: Diffcamera: Arbitrary refocusing on images. In: SIGGRAPH Asia (2025) 2, 3, 5, 8, 10, 11, 12, 13, 21, 22
- 70. Wang, Y., Chakravarthula, P., Chen, B.: Dof-gs: Adjustable depth-of-field 3d gaussian splatting for post-capture refocusing, defocus rendering and blur removal. In: CVPR (2025) 4, 5
- 71. Wang, Z., Cun, X., Bao, J., Zhou, W., Liu, J., Li, H.: Uformer: A general u-shaped transformer for image restoration. In: CVPR (2022) 2, 4
- 72. Wang, Z., Li, B., Wang, J., Liu, Y.L., Gu, J., Chuang, Y.Y., Satoh, S.: Matting by generation. In: SIGGRAPH (2024) 3
- 73. Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., Shan, Y.: Motionctrl: A unified and flexible motion controller for video generation. In: SIGGRAPH

(2024) 5

- 74. Xia, B., Zhang, Y., Wang, S., Wang, Y., Wu, X., Tian, Y., Yang, W., Van Gool, L.: Diffir: Efficient diffusion model for image restoration. In: ICCV (2023) 3
- 75. Yang, H., Pan, L., Yang, Y., Hartley, R., Liu, M.: Ldp: Language-driven dual-pixel image defocus deblurring network. In: CVPR (2024) 4
- 76. Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., Yang, Y.: Maniqa: Multi-dimension attention network for no-reference image quality assessment. In: CVPR (2022) 11
- 77. Yang, Y., Pan, L., Liu, L., Liu, M.: K3dn: Disparity-aware kernel estimation for dual-pixel defocus deblurring. In: CVPR (2023) 4
- 78. Yang, Y., Zheng, S., Chen, J., Wu, B., He, X., Cai, D., Li, B., Jiang, P.T.: Any-tobokeh: One-step video bokeh via multi-plane image guided diffusion. arXiv preprint arXiv:2505.21593 (2025) 4
- 79. Yeh, C.H., Lin, C.Y., Wang, Z., Hsiao, C.W., Chen, T.H., Shiu, H.S., Liu, Y.L.: Diffir2vr-zero: Zero-shot video restoration with diffusion-based image restoration models. arXiv preprint arXiv:2407.01519 (2024) 3
- 80. Yuan, Y., Wang, X., Sheng, Y., Chennuri, P., Zhang, X., Chan, S.: Generative photography: Scene-consistent camera control for realistic text-to-image synthesis. In: CVPR (2025) 2, 4, 5, 10, 11, 21
- 81. Yue, Z., Wang, J., Loy, C.C.: Resshift: Efficient diffusion model for image superresolution by residual shifting. In: NeurIPS (2023) 3
- 82. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Efficient transformer for high-resolution image restoration. In: CVPR (2022) 2, 8
- 83. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018) 11
- 84. Zhang, W., Cham, W.K.: Single-image refocusing and defocusing. IEEE TIP 21(2), 873–882 (2011) 5
- 85. Zhang, Y., Zheng, P., Yan, W., Fang, C., Cheng, S.S.: A unified framework for microscopy defocus deblur with multi-pyramid transformer and contrastive learning. In: CVPR (2024) 4
- 86. Zheng, P., Gao, D., Fan, D.P., Liu, L., Laaksonen, J., Ouyang, W., Sebe, N.: Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research 3(9150038), 1–12 (2024) 6, 7
- 87. Zhu, C., Fan, Q., Zhang, Q., Chen, J., Zhang, H., Xu, C., Shi, B.: Bokehdiff: Neural lens blur with one-step diffusion. In: ICCV (2025) 2, 3, 10, 11, 12
- 88. Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., Van Gool, L.: Denoising diffusion models for plug-and-play image restoration. In: CVPR (2023) 3

###### 89. Zhuo, S., Sim, T.: Defocus map estimation from a single image. PR 44(9), 1852– 1858 (2011) 3

### A Overview

In the supplementary material, we provide additional details regarding the training datasets used in our framework and the construction pipeline of PointLight1K dataset. We also present extended comparisons with DiffCamera [69], Bokehlicious [57], and Vision-Language Models. Furthermore, we provide qualitative results demonstrating the controllability of our approach, along with further analysis on ITW dataset [19] and a demonstration of the pre-deblur module’s effectiveness. We also present additional in-the-wild results on the accompanying HTML page.

### B Additional Training Details

In this section, we detail how we process datasets for training DeblurNet and BokehNet.

#### B.1 DeblurNet Training Data

DeblurNet is trained using a combination of paired defocus-blur data and highquality sharp images. Specifically, we adopt all images from the official training split of the DPDD [1] dataset, which provides supervised pairs of defocused and all-in-focus images. To supplement this, we incorporate data from RealBokeh_3MP [57]. Since this dataset contains images with varying degrees of focus, we apply a quality filter by computing the Laplacian variance of each image as a focus measure. By ranking the dataset based on this metric, we retain the top 3,000 sharpest images to serve as additional supervision for the deblurring task.

#### B.2 BokehNet Training Data

Training BokehNet requires diverse, high-resolution all-in-focus inputs to simulate realistic bokeh effects. We draw candidate images from [80] and the EBB [27] collections, and similar to the DeblurNet stage, we use Laplacian variance to filter out blurry examples. This yields a refined pool of approximately 1.7K sharp images. Furthermore, we utilize a comprehensive dataset of 26K real bokeh images. This collection comprises 13K previously filtered and verified images from the ITW dataset [19], alongside 13K images newly curated for this work. The newly collected data consists of focus-consistent series captured with varying apertures, containing 2 to 4 images per set. To establish accurate ground truth for these series, we first conducted a manual verification and refinement process on the in-focus masks. This annotation step required 4 to 8 seconds per image, amounting to approximately 8 hours of manual effort in total. Utilizing these masks, we then optimized the parameter K using simulator [43]. After obtaining the optimized K values, we applied a Structural Similarity (SSIM) threshold to filter out sub-optimal results, ensuring that only reliable pairs are used for supervision.

- Table 7: Quantitative comparison on the DPDD dataset using 512×512 central crops for the defocus deblurring task. Both methods are evaluated without any tiling strategy to ensure a strictly fair comparison of core architectural capacity. The best results are highlighted in bold.

Method LPIPS ↓ DISTS ↓ CLIP-IQA ↑ MANIQA ↑ MUSIQ ↑ DiffCamera [69] 0.2833 0.1793 0.4602 0.3502 43.2997 Ours 0.1285 0.1045 0.4637 0.3765 47.2070

### C PointLight-1K Collection Pipeline

We construct the PointLight-1K dataset to better model scenes with strong point-light bokeh. The pipeline consists of the following steps:

- 1. Keyword mining. We first mine keywords from photographs on Flickr1 that are tagged with terms such as “night” or “bokeh”. These keywords capture typical compositions and scene elements with prominent point lights.
- 2. Prompt expansion. The collected keywords are then expanded into diverse, natural-language prompts using GPT-4o [42]. This step enriches the textual descriptions while preserving point-light characteristics.
- 3. Image generation with fine-tuned FLUX. We use a fine-tuned version of FLUX.1-Dev [58] (FLUX.1-dev-LoRA-AntiBlur LoRA) to generate images from the prompts. The fine-tuning is designed to encourage sharper scenes.
- 4. Deblurring with DeblurNet. Despite fine-tuning, we observe that images involving point lights may still exhibit mild residual blur. To further reduce this blur, we pass all generated images through our DeblurNet module, optimizing each image toward an all-in-focus appearance.
- 5. Final dataset. After this process, we obtain a curated set of 1k night-time images with prominent point lights and minimal residual blur. We refer to this dataset as PointLight-1K and use it as a specialized training set to pretrain and refine our learning strategy for controllable bokeh shape.

### D Additional Comparison with DiffCamera

To ensure a strictly fair comparison of core model performance, we evaluate both methods under identical input conditions. Because DiffCamera [69] is constrained to its native 512 × 512 training resolution and a fixed aspect ratio, we extract 512 × 512 central crops from the DPDD benchmark. This guarantees that both models are evaluated at DiffCamera’s optimal setting.

As shown in Table 7, our method consistently outperforms DiffCamera across all full-reference and no-reference metrics. Notably, our approach achieves a significant reduction in LPIPS (0.1285 vs. 0.2833) and DISTS (0.1045 vs. 0.1793).

1 https://www.flickr.com/

This comprehensive superiority under a strictly controlled baseline confirms our robust generative capacity and highlights the critical advantage of leveraging real-world data during training.

### E Qualitative comparison with Bokehlicious

In the main paper, we state that our model demonstrates greater robustness to diverse input conditions. To further substantiate this claim, we provide additional qualitative comparisons with Bokehlicious [57] in Fig. 10.

A primary real-world application of defocus deblurring lies in restoring invaluable legacy photographs that can no longer be recaptured, as well as rescuing modern in-the-wild images suffering from accidental misfocus. As shown in the first three rows of Fig. 10, we evaluate the models on historical images, including monochrome photographs. Under these conditions, Bokehlicious either yields negligible deblurring effects or introduces severe structural artifacts. Similarly, in the in-the-wild scenarios (the last three rows), Bokehlicious fails to effectively recover the sharp latent images and suffers from visual degradation.

In contrast, our proposed method successfully restores details across all these highly diverse conditions. From aging black-and-white photos to everyday digital captures, our model consistently produces visually appealing results, thereby validating its strong generalization capabilities and practical applicability.

### F Qualitative comparison with VLM model

We also compare our method with the vision–language model Nano Banana Pro [20], which explicitly supports text-driven refocusing. For this model, we use the prompt “focus on the man on the right” to specify the desired focus point. As shown in Fig. 11, Nano Banana Pro can change the focus to some extent, but it also alters the facial expressions and appearances of people in both the foreground and the background. In contrast, our method preserves the original content while producing a realistic bokeh effect and accurate refocusing.

### G Qualitative Results for Controllability

In the main paper, we quantitatively demonstrated that our framework outperforms all other diffusion-based baselines in terms of controllability. In this section, we qualitatively evaluate this controllability by examining the influence of the blur parameter K and the defocus map. As shown in figure 12, the blurriness of the scene increases monotonically as the given K becomes larger. This monotonic transition effectively proves that our model achieves precise controllability.

### H ITW Dataset Analysis

This dataset is utilized to estimate the image bokeh level K via EXIF metadata. We provide further analysis regarding the dataset’s diversity. Crucially, all data consists of real optical bokeh images captured directly by digital cameras, rather than artificial defocus synthesized through computational aperture techniques. In addition to the hardware diversity represented by the various camera models (Fig. 13), we analyze the distributions of the key parameters used in our K estimation. Specifically, we provide the distributions for focal length (Fig. 14), F-number (Fig. 15), and pixel ratio (Fig. 16).

### I The effect of the pre-deblur module.

We introduce a variant that conditions DeblurNet on a pre-deblurred estimate. This provides a structural prior to disambiguate severe blur and improve fidelity. As shown in Fig. 17, even if the pre-deblurred image remains distorted, it helps the model recover fine details (e.g., the text “NEW YORK”).

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

[Figure 134]

[Figure 135]

[Figure 136]

Input Bokehlicious GenRefocus(Ours)

- Fig. 10: Comparison with Bokehlicious [57]. Results are shown for historical images (top three rows) and in-the-wild images (bottom three rows). Bokehlicious [57] either fails to remove the blur entirely or introduces severe structural artifacts. Conversely, our proposed method exhibits strong generalization capabilities, robustly handling diverse blur conditions to synthesize crisp and visually pleasing outputs.

[Figure 137]

[Figure 138]

[Figure 139]

Input Nano Banana Pro GenRefocus (Ours)

###### Fig. 11: Qualitative comparison with VLM model. Qualitative refocusing results comparing our Generative Refocusing framework with Gemini 3 Nano Banana Pro [20] given prompt "focus on the man on the right".

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

K=0(Input) K=5 K=10 K=15

###### Fig. 12: Qualitative evaluation of controllability. Refocusing results across different blur levels (K ∈ {0, 5, 10, 15}). The blue boxes highlight specific text regions to illustrate the defocus effect. As K increases, the blurriness of the scene increases monotonically, demonstrating our model’s precise control over the bokeh effect.

[Figure 144]

###### Fig. 13: Camera model distribution. Percentage of images captured by the top 30 camera models in our dataset. Less frequent models are aggregated into the “Others” category.

[Figure 145]

###### Fig. 14: Focal length distribution. The percentage distribution of focal lengths across ITW dataset.

[Figure 146]

###### Fig. 15: Aperture (F-number) distribution. The percentage distribution of aperture settings across ITW dataset. Labels are presented as vertical fractions (Nf ) for clarity.

[Figure 147]

###### Fig. 16: Pixel ratio distribution. The percentage distribution of the pixel-to-sensor ratio across ITW dataset. This ratio, defined as the image’s largest edge length divided by the physical sensor width (px/mm).

[Figure 148]

[Figure 149]

[Figure 150]

|[Figure 151]<br><br>[Figure 152]|
|---|

GT

Pre-deblur

[Figure 153]

[Figure 154]

Input

w/o pre-deblur

w pre-deblur

- Fig. 17: The effect of the pre-deblur module. Even when the pre-deblurred image is still distorted and blurry, using it as a condition yields higher-fidelity results (e.g., recovering “NEW YORK”) than removing this condition, which leads to distorted characters.

