## AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction

# arXiv:2601.00796v1[cs.CV]2Jan2026

Jiewen Chan1 Zhenjun Zhao2 Yu-Lun Liu1 1National Yang Ming Chiao Tung University 2University of Zaragoza

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

| |
|---|

###### Ours

GT

CoDeF

Splatter A Video (NeurIPS’24)

[Figure 5]

[Figure 6]

CoDeF (CVPR’24)

RoDynRF

Ours

Splatter A Video

Car-roundabout

OmniMotion (ICCV’23)

(CVPR’23)

[Figure 7]

[Figure 8]

[Figure 9]

Deformable

Sprites (CVPR’22)

GT

CoDeF

[Figure 10]

[Figure 11]

| |
|---|

4DGS (CVPR’24)

Splatter A Video Ours

Dancing

Figure 1. State-of-the-art video reconstruction quality on DAVIS [65] dataset. Our Adaptive Gabor representation achieves superior rendering quality (PSNR: 35.49 dB, SSIM: 0.9433) while preserving fine details and temporal consistency. (Left) Qualitative comparisons demonstrate sharper textures in challenging regions (car windows, drum surface) compared to CoDeF [62] and Splatter A Video [76]. (Right) Our method (red point) significantly outperforms recent baselines across all metrics, achieving 6.86 dB PSNR improvement over the second-best method with reasonable training time (circle size indicates training duration: 30 mins to 24 hours).

### Abstract

Reconstructing dynamic 3D scenes from monocular videos requires simultaneously capturing high-frequency appearance details and temporally continuous motion. Existing methods using single Gaussian primitives are limited by their low-pass filtering nature, while standard Gabor functions introduce energy instability. Moreover, lack of temporal continuity constraints often leads to motion artifacts during interpolation. We propose AdaGaR, a unified framework addressing both frequency adaptivity and temporal continuity in explicit dynamic scene modeling. We introduce Adaptive Gabor Representation, extending Gaussians through learnable frequency weights and adaptive energy compensation to balance detail capture and stability. For temporal continuity, we employ Cubic Hermite Splines with Temporal Curvature Regularization to ensure smooth motion evolution. An Adaptive Initialization mechanism combining depth estimation, point tracking, and foreground masks establishes stable point cloud distributions in early training. Experiments on Tap-Vid DAVIS demonstrate state-of-the-art performance

(PSNR 35.49, SSIM 0.9433, LPIPS 0.0723) and strong generalization across frame interpolation, depth consistency, video editing, and stereo view synthesis. Project page: https://jiewenchan.github.io/AdaGaR/

### 1. Introduction

Reconstructing dynamic 3D scenes from monocular videos is a fundamental challenge in computer vision with wide applications in VR, AR, and film production. The key difficulty lies in jointly achieving temporal continuity and rich frequency representation: real-world scenes demand smooth motion over time while preserving high-frequency textures that define appearance.

Existing approaches [6, 30, 41, 66, 75, 102] fall into two camps. Gaussian-based primitives provide fast, explicit modeling but suffer from strong low-pass filtering, which suppresses high-frequency detail. Introducing frequency modulation [87] (e.g., Gabor-like representations) can enhance texture fidelity but often destabilizes energy balance and rendering quality. Moreover, many methods lack explicit

temporal constraints, leading to motion discontinuities and geometric tearing, especially under rapid motion or occlusions.

To address these gaps, we propose AdaGaR (Adaptive Gabor Representation for Dynamic Scene Reconstruction), a unified framework that jointly optimizes time and frequency in explicit dynamic representations. Our core idea is to separate and yet tightly couple two orthogonal aspects: (i) frequency adaptivity via a learnable Adaptive Gabor Representation that balances high- and low-frequency components while maintaining energy stability; and (ii) temporal continuity via Cubic Hermite Splines with Temporal Curvature Regularization, constraining motion trajectories for smooth evolution. An Adaptive Initialization further bootstraps stable, temporally coherent geometry at early training.

We validate AdaGaR on Tap-Vid [65], achieving stateof-the-art video reconstruction and strong generalization to frame interpolation, depth consistency, video editing, and stereo view synthesis, as shown in Fig. 1. This work provides a compact, end-to-end solution for modeling both time and frequency in explicit dynamic representations, with potential to guide future developments in frequency-aware dynamic modeling.

Our main contributions are summarized as follows:

- • We propose a novel Adaptive Gabor Representation that extends traditional Gaussians to the frequency domain, Fig. 2, which is (i) frequency-adaptive, (ii) energystable, and (iii) capable of capturing high-frequency texture details while automatically adjusting between high and low-frequency components according to scene requirements;
- • We introduce Temporal Curvature Regularization with Cubic Hermite Spline interpolation, which accurately and effectively ensures geometric and motion continuity in the temporal dimension, achieving smooth temporal evolution and avoiding interpolation artifacts;
- • We present an Adaptive Initialization mechanism that combines depth estimation, point tracking, and foreground masks to establish stable and temporally consistent point cloud distributions, significantly improving training efficiency and final reconstruction quality.

### 2. Related Work

Dynamic 3D Gaussian Splatting. 3D Gaussian Splatting (3DGS)[38] has inspired extensive research on dynamic scene extensions. Early work[59] used timedependent MLPs for deformation. Recent canonical space approaches [2, 19, 27, 32, 53, 58, 85, 96] employ deformation networks to handle compression and specular dynamics. Temporal modeling strategies include flow-guided methods [105], neural features [49], temporal slicing [18], spatial-temporal regularization [13, 45], and hash encoding [89]. Specialized applications target autonomous driv-

ing [71, 91, 104], sparse reconstruction [61], unconstrained capture [40, 72], acceleration [77, 101], and motion blur [86]. Most similar to our work, SplineGS [63] applies Cubic Hermite splines in multi-view settings. In contrast, we combine Cubic Hermite splines with Gabor-based primitives for monocular videos without camera pose estimation, introducing Temporal Curvature Regularization for physically plausible motion.

Frequency-Adaptive Rendering. Traditional 3D Gaussian kernels act as low-pass filters, limiting high-frequency detail representation. Anti-aliasing methods for 3DGS include multi-scale filtering [92, 98], analytical integration [51], and opacity field derivation [99]. NeRF frequency-aware approaches employ cone-tracing [3], frequency regularization [88, 93], frequency decomposition [25], and structurenoise separation [67]. Gabor representations in neural rendering [1, 83, 87] build on procedural graphics foundations [21, 44]. Alternative primitives include exponential functions [23], surfels [31], and Beta kernels [54]. However, existing Gabor approaches target static scenes with fixed frequencies. Our Adaptive Gabor Representation extends to dynamic videos with learnable frequency weights and graceful degradation to standard Gaussians.

Temporal Modeling and Spline Representations. Classical splines [20, 55] provide smooth temporal interpolation. Recent neural rendering incorporates splines through Hermite formulations [15], B-splines [63, 82], and timemodulated weights [22]. Coarse-fine decomposition methods [2, 90, 95] separate temporal scales. Flow-guided approaches [50, 60, 103, 105] leverage optical flow constraints. Alternative temporal models include Kalman filtering [100], neural trajectories [42], frame interpolation [33, 69], and robust dynamic fields [57]. Unlike implicit smoothness from architecture or training, we explicitly enforce smoothness through Temporal Curvature Regularization based on secondorder derivatives, ensuring physically plausible motion with geometric interpretability.

Video Representations and Canonical Spaces. Canonical space methods enable temporally consistent processing through layered atlases [36], deformation fields [11, 62], and canonical volumes [80]. Implicit neural video representations [8, 9, 39, 43, 47, 70, 73] achieve compression through image-wise functions. Explicit representations employ 4D Gaussians [85], 2D feature streams [78], layer decomposition [74], learned quantization [46], hash encoding [10], and scene inpainting [84]. Video Gaussian Splatting methods target monocular [6, 28, 30, 41, 52, 66, 75, 102] and multi-view [12, 56] settings. Our approach operates in an orthographic camera coordinate system [76], eliminating pose estimation while maintaining explicit 3D structure through Gabor primitives for high-frequency preservation and versatile applications.

Gaussian Primitives

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

| |
|---|

| |
|---|

| |
|---|

Gabor Primitives Gabor Primitives Gabor Primitives

- Figure 2. Hierarchical frequency adaptation. Our primitives adaptively transition from Gaussian (topleft) to Gabor (bottom), enabling coarse-to-fine reconstruction. Each primitive learns its optimal frequency response via learnable weights ωi, achieving both geometric stability and texture detail in a unified framework.

Monocular Depth and Motion Estimation. Foundation models provide robust monocular priors. Depth estimation methods [5, 29, 37, 48, 64, 68, 79, 94] achieve zero-shot generalization through synthetic training, diffusion repurposing, and multi-dataset learning. Point tracking methods [14, 16, 17, 24, 34, 35, 81] enable dense correspondence through pseudo-labeling, self-supervision, and local correlation. Unlike prior work using these signals independently, our adaptive initialization jointly reasons about depth, motion, and segmentation for geometrically and temporally consistent initialization.

### 3. Preliminary: 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [38] represents a 3D scene as a collection of parameterized Gaussian primitives {Gk | k = 1,...,N}. Each Gk has center µk ∈ R3, covariance Σk ∈ R3×3, opacity αk ∈ [0,1], and color ck. The density is

Gk(x) = exp −12(x − µk)⊤Σ−k 1(x − µk) , with Σk = RkSkS⊤k R⊤k .

Rendering projects Gaussians onto the image plane and accumulates color via front-to-back blending:

C(x) =

K

(1 − αj).

Tkαkck, Tk =

k=1

j<k

A key limitation is that a single Gaussian acts as a lowpass filter, constraining high-frequency textured detail. To address this, we introduce Gabor kernels as periodic extensions of Gaussians to enhance spatial frequency representation.

### 4. Method

#### 4.1. Overview

We present AdaGaR, an explicit 3D video representation that preserves high-frequency appearance while ensuring temporally smooth motion. As illustrated in Fig. 3, The video

is modeled as a set of dynamic Adaptive Gabor primitives in an orthographic camera coordinate system, where spatial texture and structure are encoded by the primitives and temporal evolution is interpolated with Cubic Hermite Splines to guarantee geometric and temporal consistency. Adaptive Gabor Representation extends Gaussian primitives with learnable frequency weights and energy compensation, enabling frequency-adaptive detail capture while maintaining energy stability. Coupled with temporal curvature regularization and multi-supervision losses, our approach delivers high visual quality and robust temporal consistency, with strong applicability to frame interpolation, depth consistency, video editing, and related tasks.

#### 4.2. Adaptive Gabor Video Representation

Camera Coordinate Space. Inspired by [80] and [76], we adopt an orthographic camera coordinate system that maps width, height, and depth to the X, Y , and Z axes, enabling a direct orthogonal representation of the 3D video structure. This avoids costly camera pose estimation and motion disentangling, treating camera motion and object motion as a single type of dynamic variation. The video is represented as a collection of dynamic adaptive Gabor primitives, each encoding spatial position, temporal variation, and frequency response, rendered from a fixed identity pose.

Adaptive Gabor Representation. To introduce highfrequency details on the image plane, the Gabor function can be viewed as a periodic extension of the Gaussian function. Its general 2D form can be defined as:

- 1

- 2||x − µ||2Σ−1 cos(f⊤x + ϕ), (1)

GGabor(x) = exp −

where x = (x,y)⊤ denotes the image plane coordinates, f = (fx,fy)⊤ is the center frequency vector, and ϕ represents the phase offset. This structure introduces a sinusoidal modulation within the Gaussian envelope, enabling the distribution to simultaneously capture local directional textures and high-frequency detail variations.

To model richer frequency components, multiple Gabor waves can be combined into a weighted superposition:

S(x) =

N

ωi cos(fi⟨di,x⟩ + ϕi), (2)

i=1

where ωi ∈ R denotes the amplitude weight, fi ∈ R+ represents the frequency magnitude, and di ∈ R2 with ||di||2 = 1 is the frequency direction unit vector. This structure generates spatially periodic texture variations, which produce richer textural details when combined with Gaussians.

While the Gabor structure enhances detail representation, fixed-amplitude cosine modulation disrupts the energy stability of the Gaussian. To address this, we propose Adaptive

𝐼𝑛𝑝𝑢𝑡

𝑂𝑝𝑡𝑖𝑚𝑖𝑧𝑎𝑡𝑖𝑜𝑛 𝐴𝑝𝑝𝑙𝑖𝑐𝑎𝑡𝑖𝑜𝑛

[Figure 26]

[Figure 27]

𝐿𝑜𝑠𝑠 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝑋

A𝑑𝑎𝑝𝑡𝑖𝑣𝑒 𝑀𝑜𝑡𝑖𝑜𝑛

[Figure 28]

[Figure 29]

Orthographic

𝜇 𝑁 − 1 ,𝑞(𝑁 − 1)

𝑦

|𝜇 0 ,𝑞(0)<br><br>Primitive|
|---|

Camera Coordinate

|𝐼𝑚𝑎𝑔𝑒 𝐼𝑡|
|---|

Interpolated Primitive

[Figure 30]

|𝜇 𝑡 , 𝑞(𝑡)<br><br>|
|---|

|𝐹𝑟𝑎𝑚𝑒 𝐼𝑛𝑡𝑒𝑟𝑝𝑜𝑙𝑎𝑡𝑖𝑜𝑛|
|---|

[Figure 31]

𝐿𝑐𝑢𝑟𝑣

|𝑚0|
|---|

…

𝑇

𝑥

𝑡 − 1 𝑡 𝑡 + 1

[Figure 32]

[Figure 33]

|Gr𝑜𝑢𝑛𝑑<br><br>𝑇𝑟𝑢𝑡ℎ|
|---|

3𝐷 𝑡𝑟𝑎𝑐𝑘

|𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑<br><br>𝑅𝑒𝑠𝑢𝑙𝑡|
|---|

[Figure 34]

Cubic Hermite Spline

Control Points

𝑧

|𝐷𝑒𝑝𝑡ℎ 𝑑𝑡|
|---|

𝑡 ∈ [0,𝑁 − 1]

[Figure 35]

[Figure 36]

[Figure 37]

𝐴𝑑𝑎𝑝𝑡𝑖𝑣𝑒 𝐺𝑎𝑏𝑜𝑟 𝑅𝑒𝑝𝑟𝑒𝑠𝑒𝑛𝑡𝑎𝑡𝑖𝑜𝑛

[Figure 38]

|𝐷𝑒𝑝𝑡ℎ 𝑐𝑜𝑛𝑠𝑖𝑠𝑡𝑒𝑛𝑐𝑦|
|---|

|𝐺𝑎𝑢𝑠𝑠𝑖𝑎𝑛 𝑃𝑟𝑖𝑚𝑖𝑡𝑖𝑣𝑒𝑠| |
|---|---|
| | |

|𝐼መ𝑡|
|---|

|𝐼𝑡|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

|2D 𝑇𝑟𝑎𝑐𝑘 𝑇𝑡|
|---|

[Figure 45]

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

[Figure 50]

|[Figure 51]| |
|---|---|
|𝐺𝑎𝑏𝑜𝑟 𝑃𝑟𝑖𝑚𝑖𝑡𝑖𝑣𝑒𝑠| |
| | |

| | |
|---|---|
|𝐺𝑎𝑏𝑜𝑟 𝑃𝑟𝑖𝑚𝑖𝑡𝑖𝑣𝑒𝑠| |
| | |

[Figure 52]

[Figure 53]

|𝑑መ𝑡|
|---|

|𝑑𝑡|
|---|

|𝐺𝑎𝑏𝑜𝑟 𝑃𝑟𝑖𝑚𝑖𝑡𝑖𝑣𝑒𝑠| |
|---|---|
| | |

|[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>|𝜔𝑘 ∈ W|
|---|
|
|---|

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

|𝑉𝑖𝑑𝑒𝑜 𝐸𝑑𝑖𝑡𝑖𝑛𝑔|
|---|

|𝐹෠𝑡|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|𝑀𝑎𝑠𝑘 𝑀𝑡|
|---|

|𝐹𝑡|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

- Figure 3. Method overview. Our approach represents dynamic videos as Adaptive Gabor primitives with temporally smooth motion. (Input) Multi-modal supervision from RGB, depth, tracking, and masks. (Optimization) Two core components: (1) Adaptive Motion: Cubic Hermite splines model primitive trajectories with control points µ(t), q(t) in orthographic camera space, ensuring C1 continuity. (2) Adaptive Gabor Representation: Learnable frequency weights ωk enable primitives to adaptively span from Gaussian (low-freq) to Gabor (high-freq), achieving hierarchical detail reconstruction. (Loss) Joint optimization via RGB, depth, flow supervision, and curvature regularization Lcurv. (Application) Supports frame interpolation, depth consistency, and video editing.

Gabor, which automatically adjusts the intensity based on the wave energy and naturally degrades to a Gaussian in extreme cases. We extend the original opacity expression to αGabor = G(x) · S(x). In practice, we set the phase terms ϕi = 0, yielding:

N

ωi cos(fi⟨di,x⟩), (3)

S(x) =

i=1

where we fix the frequency parameters fi ∈ {1,2}, corresponding to two orthogonal base frequency waveforms. The amplitude weights ωi ∈ [0,1] are the introduced learnable parameters for the Gabor structure, adjusting the energy weights of different frequency components. The direction unit vectors di are shared with the spatial orientation of the original Gaussian, ensuring consistency between frequency modulation and Gaussian shape orientation.

To prevent overall intensity attenuation when i ωi < 1, we introduce a compensation term b:

1 N

Sadap(x) = b +

N

ωi cos(fi⟨di,x⟩), (4)

i=1

N

1 N

ωi , (5)

b = γ + (1 − γ) 1 −

i=1

where γ ∈ [0,1] is a fixed hyperparameter controlling the degradation smoothness, and the factor 1/N normalizes the weighted average of multiple waves to a stable range. When ωi → 0, we have b → 1, and the formulation naturally degrades to a traditional Gaussian.

#### 4.3. Temporally Dynamic Adaptive Gabor

Cubic Hermite Spline Interpolation. We use Cubic Hermite Splines [7, 26] to interpolate the temporal evolution of dynamic primitives. Given M temporal keyframes at times {t0,t1,...,tM−1} with corresponding control point positions {y0,y1,...,yM−1} ⊂ R3, we define the time interval between adjacent keyframes as ∆k = tk+1 − tk, and the slope as δk = (yk+1 − yk)/∆k. To avoid unnecessary oscillations between keyframes, we introduce an auto-slope mechanism with a monotone gate:

mk =

β · δ

k−1+δk

2 , if sign(δk−1) = sign(δk), 0, otherwise,

(6)

where β ∈ (0,1] is a smoothness coefficient controlling the flatness of the interpolation curve. This design prevents reverse oscillations at keyframes and ensures visually stable interpolation.

The Hermite basis functions are defined as:

- H00(s) = 2s3 − 3s2 + 1, H10(s) = s3 − 2s2 + s,
- H01(s) = −2s3 + 3s2, H11(s) = s3 − s2,

(7)

where s = (t − tk)/∆k ∈ [0,1] is the normalized time within the interval [tk,tk+1]. The interpolated displacement at time t is:

∆(t) =H00(s)yk + H10(s)∆kmk

+ H01(s)yk+1 + H11(s)∆kmk+1.

(8)

To ensure consistent geometric continuity, the final position is obtained by adding the interpolated displacement to a

[Figure 73]

|=𝜔(0.0,0.0)| |
|---|---|
| | |
|=𝜔(0.5,0.5)| |
| | |
|=𝜔(1.0,1.0)| |
| |𝐺(𝑥)|
| | |

[Figure 74]

- 0.75=𝜔1

|1.00=𝜔1|
|---|

- 1.00=𝜔1

===𝜔𝜔𝜔(0.0,0.0)(0.5,0.5)(1.0,1.0)

|0.75=𝜔1|
|---|

|0.50=𝜔1|
|---|

0.50=𝜔1

|0.25=𝜔1|
|---|

0.25=𝜔1

|0.00=𝜔1|
|---|

0.00=𝜔1

|𝜔2 = 0.00|
|---|

|𝜔2 = 0.25|
|---|

|𝜔2 = 0.50|
|---|

|𝜔2 = 1.00|
|---|

|𝜔2 = 0.75|
|---|

(b) The effect of different combinations of Gabor wave coefficients (ω0 and ω1) on spatial frequency textures.

|𝑆(𝑥)|1 + 𝑆(𝑥)|
|---|---|

|𝑆𝑎𝑑𝑎𝑝 𝑥<br><br>|
|---|

(a) G(x),Gaussian, S(x), sinusoid part

- Figure 4. Adaptive Gabor formulation. (a) Smooth transition between Gaussian and Gabor kernels. Our method (rightmost column, Sours(x)) uses a compensation term b to maintain energy stability while transitioning from pure Gaussian (ω = 0, top) to frequencymodulated Gabor (ω = 1, bottom). Naive combination 1 + S(x) (third column) suffers from intensity artifacts. (b) Frequency weight combinations. Different (ω0, ω1) pairs generate diverse spatial patterns, from smooth (low ω) to high-frequency textures (high ω), enabling adaptive detail capture in different scene regions.

base position:

##### µ(t) = µbase + ∆(t). (9)

Rotation Interpolation. We extend the same principle to temporal interpolation of rotations. For rotation parameters, we first interpolate in the so(3) Lie algebra space, then convert to unit quaternions via the exponential map:

q(t) = normalize normalize(qbase)⊗exp(∆q(t)) , (10)

where ⊗ denotes quaternion multiplication, and angle wrapping ensures rotation angles remain within (−π,π].

Temporal Curvature Regularization. To enforce smooth temporal evolution, we introduce a curvature penalty on the trajectory at each keyframe. For non-uniform keyframes, the second-order derivative is estimated as

2(d+k − d−k ) hk−1 + hk

yk′′ =

, (11)

with hk−1 = tk − tk−1, hk = tk+1 − tk, d+k = (yk+1 − yk)/hk, d−k = (yk − yk−1)/hk−1, and D = 3. The curvature loss is

M−2 k=1 wk∥yk′′∥22 M−2 k=1 wkD + ε

, (12)

Lcurve =

where wk = 12(hk−1 + hk) and ε > 0 is a small constant. This term enforces smoothness by penalizing the second-

order energy along time.

#### 4.4. Optimization

To maintain both realistic appearance and temporal stability in dynamic scenes, we employ a multi-objective loss function that constrains appearance fidelity, motion consistency, depth geometry, and temporal smoothness.

Rendering Reconstruction Loss. We combine L1 and SSIM to preserve both pixel-level accuracy and structural features:

Lrgb(It,Iˆt) = (1 − λssim)Lrgb1 (It,Iˆt) + λssimLrgbssim(It,Iˆt),

(13) where It and Iˆt denote the ground-truth and predicted images at frame t, respectively.

Optical Flow Consistency Loss. We leverage CoTracker [34] to provide cross-frame supervision. The projected positions of Adaptive Gabor primitives are aligned with 2D trajectories using a visibility-weighted L1 loss:

wj x ˆjt

− xjt

1,t2) = j

2 1 j wj + ε

2

Lflow(Fˆt

, (14)

1,t2,Ft

where xjt

and xˆjt

are the ground-truth and predicted pixel positions of the j-th tracked point at frame t2, and wj denotes its visibility weight.

2

2

Depth Loss. We use monocular depth estimates from DPT [68] as geometric priors with scale- and shift-invariant

alignment:

Ldepth(Dt,Dˆt) = γ(Dt) − γ(Dˆt)

, (15)

1

where γ(Dt) = (Dt − ct(Dt))/∥Dt − ct(Dt)∥1 with ct(Dt) = median(Dt). Total Loss. The overall optimization objective combines all components:

Ltotal = λrgbLrgb + λflowLflow + λdepthLdepth + λcurvLcurv.

(16) This multi-faceted supervision enables AdaGaR to achieve both high-fidelity rendering and temporally stable dynamic scene representation.

#### 4.5. Adaptive Initialization

We propose an adaptive initialization to initialize a temporally coherent 3D point distribution early in training. It fuses multi-modal cues to generate a dense, dynamic initial point cloud from the input video, forming the geometric basis for subsequent explicit representations. Unlike random sampling or single-frame methods, our approach adaptively adjusts sampling density according to scene motion and depth distribution, ensuring balanced foreground/background coverage.

Temporal–Spatial Adaptive Sampling. For each candidate point pi, the sampling probability is

1 τi + ϵ

1 ρi + ϵ

, (17)

Π(pi) ∝

+ λτ

where τi is the temporal support, ρi the local density, λτ ∈ [0,1] balances temporal stability and spatial uniformity, and ϵ > 0.

Grid-Based Uniform Coverage. To ensure global coverage, we partition the image into a fixed grid G = {Gu,v} and modulate per-cell sampling by

Π(pi) 1 + λgCu,v

Π′(pi | Gu,v) =

, (18)

with Cu,v the cell’s cumulative samples and λg > 0. Boundary-Aware Compensation. We further adjust for motion boundaries via

Π′′(pi) = Π′(pi | Gu,v)(1 + λb∥∇Mt(pi)∥), (19) where Mt is the foreground mask and λb > 0.

This scheme yields a dense, temporally coherent initial point cloud and reduces early-stage flickering.

### 5. Experiment

#### 5.1. Evaluation

Dataset and Metrics. We evaluate on Tap-Vid DAVIS [65], featuring diverse dynamic scenes and occlusions. Quantitative metrics include PSNR, SSIM, and LPIPS to assess pixel accuracy, structural fidelity, and perceptual quality.

Table 1. Quantitative results on Tap-Vid DAVIS [65]. Our method achieves state-of-the-art performance across all metrics, with 6.86 dB PSNR improvement over the previous best method [76], validating our frequency-adaptive primitives with smooth temporal modeling.

Method PSNR↑ SSIM↑ LPIPS↓ 4DGS [86] 18.12 0.5735 0.5130 RoDynRF [57] 24.79 0.7230 0.3940 Deformable Sprites [97] 22.83 0.6983 0.3014 Omnimotion [80] 24.11 0.7145 0.3713 CoDeF [62] 26.17 0.8160 0.2905 Splatter A Video [76] 28.63 0.8373 0.2283 Ours 35.49 0.9433 0.0723

Implementation Details. The training consists of two stages: a 500-iteration warm-up and 10K iterations for main optimization, with control points updated every 100 iterations. Experiments run on an NVIDIA RTX 4090, 90 minutes per video sequence.

Video Reconstruction. As shown in Tab. 1, our method outperforms the baselines across PSNR/SSIM/LPIPS on TapVid DAVIS [65]. Compared with MLP-based representations, ours yields sharper textures and coherent motion in Fig. 5.

#### 5.2. Applications

Depth Consistency. We achieve stable depth distributions over time, substantially reducing depth flicker and boundary misalignment, and outperforming per-frame optimizers, as shown in Fig. 6.

Frame Interpolation. We generate smooth intermediate frames between keyframes using cubic Hermite splines with curvature regularization, preserving texture detail and avoiding boundary artifacts, as shown in Fig. 7.

Video Editing. In canonical space, style transfers remain temporally coherent by acting on shared Adaptive Gabor primitives, reducing style drift and flicker, as shown in Fig. 8.

Stereo View Synthesis. Our explicit representation supports stereo synthesis from monocular input, with improved disparity consistency and plausible geometry, as shown in Fig. 9.

#### 5.3. Ablation Study

Adaptive Gabor Representation. We compare Adaptive Gabor Representation (AGR) to Gaussian and standard Gabor, using the same 1M primitives. As shown in Tab. 2, AGR improves high-frequency detail and energy stability, yielding the best PSNR/SSIM/LPIPS among the three configurations.

Spline Interpolation. We ablate curve interpolation on 50 frames in Tab. 3. B-Spline and Cubic Spline provide some temporal continuity but struggle with nonlinear motion. In contrast, the proposed Cubic Hermite Spline achieves the

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

CoDeF Splatter A Video Ours

GT

- Figure 5. Visual comparison on DAVIS dataset. Our method preserves finer details (fur, vehicle edges, wheel structures) and sharper motion boundaries compared to CoDeF [62] and Splatter A Video [76]. Red boxes highlight key regions demonstrating our superior texture reconstruction and temporal consistency. Best viewed zoomed in.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

- t1

Ours

- t2

Marigold

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

- Figure 6. Depth consistency across time. (Left) Our 3D primitive representation maintains consistent depth for static elements across frames. (Right) While per-frame estimation (Marigold [37]) shows temporal flickering (red boxes). Explicit 3D geometry with smooth motion modeling ensures temporal coherence essential for depthbased video applications.

- Table 2. Gabor primitive ablation. Our Adaptive Gabor with compensation term b outperforms standard Gaussian, naive Gabor variants, validating that energy-aware formulation (Eq. (5)) is crucial for stable frequency modeling.

Method PSNR↑ SSIM↑ LPIPS↓ Gaussian 36.66 0.9423 0.0421 Standard Gabor (b = 0) 36.65 0.9543 0.0345 1 + S(x) 36.50 0.9511 0.0322 Adaptive Gabor (Ours) 37.43 0.9620 0.0242

- Table 3. Spline method ablation. Our Cubic Hermite Spline with monotone gate outperforms B-Spline and significantly surpasses standard Cubic Spline, which suffers from trajectory oscillations. Explicit velocity control (Eq. (6)) is essential for smooth, artifactfree motion modeling.

Methods PSNR↑ SSIM↑ LPIPS↓ B-Spline 36.68 0.9573 0.0368 Cubic Spline 32.42 0.9073 0.0818 Cubic Hermite Spline (Ours) 38.98 0.9697 0.0259

best performance across metrics, with smoother trajectories and preserved dynamic details.

Curvature Regularization. We compare with/without the temporal curvature term Lcurv in Fig. 10. Without Lcurv, mo-

tion artifacts and tearing appear, while, with Lcurv, interpolation is smoother and more stable, confirming the necessity

[Figure 115]

[Figure 116]

1𝑠𝑡, 2𝑛𝑑, 3𝑟𝑑 and 4𝑡ℎ interpolated frame

Left Input Right Input

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

| |
|---|

| |
|---|

t t + 1

- Figure 7. Frame interpolation results. Our method generates temporally smooth intermediate frames between input keyframes t and t + 1 by querying Cubic Hermite splines at fractional timestamps. The interpolated sequence (1st through 4th frames) maintains consistent fur texture details and natural motion without ghosting artifacts. Red boxes show the preservation of high-frequency details throughout the interpolation. This demonstrates our method’s ability to produce continuous motion with C1 smoothness via curvature-regularized spline trajectories. Please refer to the supplementary video for full temporal coherence.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Per-frameEditingOurs

t = 0 t = 10

[Figure 127]

Style Reference Image

| |
|---|

| |
|---|

| |
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

| |
|---|

|[Figure 131]|
|---|

- Figure 8. Temporally consistent video editing. (Top) Per-frame editing causes temporal flickering with inconsistent styles between frames. (Bottom) Our canonical space editing maintains temporal consistency by applying style transfer to shared Adaptive Gabor primitives, ensuring identical treatment of scene elements across time while preserving motion dynamics. Red boxes highlight key differences. Please see the supplementary video.

[Figure 132]

[Figure 133]

- Figure 9. Stereo view synthesis. Our 3D representation enables novel view synthesis for stereo visualization from monocular video. This demonstrates that Adaptive Gabor primitives in orthographic camera coordinate space capture accurate 3D geometry, enabling immersive applications.

[Figure 134]

Interpolated Frame

[Figure 135]

[Figure 136]

|[Figure 137]|
|---|

| |
|---|

|w/o 𝐿𝑐𝑢𝑟𝑣|
|---|

Ours

- Figure 10. Curvature regularization ablation. (Left) Without

Lcurv, interpolated frames show motion artifacts from trajectory oscillations. (Right) Our method produces smooth, artifact-free interpolation by constraining second-order derivatives, validating the necessity of explicit curvature control for temporal consistency.

[Figure 138]

[Figure 139]

| |
|---|

| |
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|PSNR: 32.19 SSIM: 0.9412|
|---|

|PSNR: 38.97 SSIM: 0.9806|
|---|

w/o Adaptive Initialization Ours

- Figure 11. Adaptive initialization ablation. (Left) Without motionaware initialization, primitives are poorly distributed, causing blurred details. (Right) Our adaptive initialization based on depth, tracking, and masks (Eqs. (17) to (19)) provides better initial geometry, yielding 6.78 dB improvement and sharp reconstruction.

Representation and employing Cubic Hermite Splines with Temporal Curvature Regularization, our approach captures high-frequency details while ensuring geometric and motion continuity. Experiments demonstrate state-of-the-art performance on Tap-Vid DAVIS with strong generalization across frame interpolation, depth consistency, video editing, and stereo synthesis.

of explicit curvature control.

Adaptive Initialization. We compare random initialization with our adaptive initialization. As shown in Fig. 11, the adaptive approach yields denser, temporally coherent initial geometry, reducing flicker and improving early reconstruction quality.

Limitations. Despite superior performance, AdaGaR has limitations. The spline-based motion modeling assumes smooth trajectories, potentially causing misalignment under abrupt or highly nonlinear motion. Additionally, Adaptive Gabor Representation may exhibit oscillations in highfrequency regions due to energy constraints. Future work could introduce adaptive temporal control points and motionaware frequency modulation.

### 6. Conclusion

We present AdaGaR, a unified framework for temporal continuity and frequency adaptivity in dynamic scene modeling. By extending Gaussian primitives to Adaptive Gabor

Acknowledgements. This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628E-A49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

- [1] Ahmad AlMughrabi, Ricardo Marques, and Petia Radeva. Momentsnerf: Leveraging orthogonal moments for few-shot neural rendering. arXiv preprint arXiv:2407.02668, 2024. 2
- [2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung Uh. Per-gaussian embeddingbased deformation for deformable 3d gaussian splatting. In European Conference on Computer Vision, pages 321–335. Springer, 2024. 2
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697–19705, 2023. 2
- [4] Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. 14
- [5] Aleksei Bochkovskii, AmaA˜ ¸Gl Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024. 3
- [6] Minh-Quan Viet Bui, Jongmin Park, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong Oh, and Munchurl Kim. Mobgs: Motion deblurring dynamic 3d gaussian splatting for blurry monocular video. arXiv preprint arXiv:2504.15122, 2025. 1, 2
- [7] AKB Chand and P Viswanathan. Cubic hermite and cubic spline fractal interpolation functions. In AIP conference Proceedings, pages 1467–1470. American Institute of Physics,

2012. 4

- [8] Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser Nam Lim, and Abhinav Shrivastava. Nerv: Neural representations for videos. Advances in Neural Information Processing Systems, 34:21557–21568, 2021. 2
- [9] Hao Chen, Matthew Gwilliam, Ser-Nam Lim, and Abhinav Shrivastava. Hnerv: A hybrid neural representation for videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10270– 10279, 2023. 2
- [10] Jie Chen, Zhangchi Hu, Peixi Wu, Huyue Zhu, Hebei Li, and Xiaoyan Sun. Dash: 4d hash encoding with self-supervised decomposition for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26349–26359, 2025. 2
- [11] Ting-Hsuan Chen, Jie Wen Chan, Hau-Shiang Shiu, ShihHan Yen, Changhan Yeh, and Yu-Lun Liu. Narcan: Natural

- refined canonical image with integration of diffusion prior for video editing. Advances in Neural Information Processing Systems, 37:36097–36120, 2024. 2
- [12] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pages 370–386. Springer, 2024. 2
- [13] Hao-Jen Chien, Yi-Chuan Huang, Chung-Ho Wu, Wei-Lun Chao, and Yu-Lun Liu. Splannequin: Freezing monocular mannequin-challenge footage with dual-detection splatting. arXiv preprint arXiv:2512.05113, 2025. 2
- [14] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In ECCV, pages 306–325, 2024. 3
- [15] Ilya Chugunov, David Shustin, Ruyu Yan, Chenyang Lei, and Felix Heide. Neural spline fields for burst image fusion and layer separation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 25763–25773, 2024. 2
- [16] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10061– 10072, 2023. 3
- [17] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Joao Carreira, et al. Bootstap: Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference on Computer Vision, pages 3257–3274, 2024. 3
- [18] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,

2024. 2

- [19] Cheng-De Fan, Chen-Wei Chang, Yi-Ruei Liu, Jie-Ying Lee, Jiun-Long Huang, Yu-Chee Tseng, and Yu-Lun Liu. Spectromotion: Dynamic 3d reconstruction of specular scenes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21328–21338, 2025. 2
- [20] Gerald E Farin. Curves and surfaces for CAGD: a practical guide. Morgan Kaufmann, 2002. 2
- [21] Bruno Galerne, Ares Lagae, Sylvain Lefebvre, and George Drettakis. Gabor noise by example. ACM Transactions on Graphics (ToG), 31(4):1–9, 2012. 2
- [22] Ivan Grega, William F Whitney, and Vikram S Deshpande. Neural rendering enables dynamic tomography. arXiv preprint arXiv:2410.20558, 2024. 2
- [23] Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl Vondrick, Bernard Ghanem, and Andrea Vedaldi. Ges: Generalized exponential splatting for efficient radiance field rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19812–19822, 2024. 2
- [24] Adam W Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan

- Chu, Achal Dave, Suya You, et al. Alltracker: Efficient dense point tracking at high resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5253–5262, 2025. 3
- [25] Yisheng He, Weihao Yuan, Siyu Zhu, Zilong Dong, Liefeng Bo, and Qixing Huang. Freditor: High-fidelity and transferable nerf editing by frequency decomposition. In European Conference on Computer Vision, pages 73–91. Springer,

2024. 2

- [26] Niels T Hintzen, Gerjan J Piet, and Thomas Brunel. Improved estimation of trawling tracks using cubic hermite spline interpolation of position registration data. Fisheries research, 101(1-2):108–115, 2010. 4
- [27] Cheng-Yuan Ho, He-Bi Yang, Jui-Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng. Ted-4dgs: Temporally activated and embedding-based deformation for 4dgs compression. arXiv preprint arXiv:2512.05446, 2025. 2
- [28] Hao-Yu Hou, Chia-Chi Hsu, Yu-Chen Huang, Mu-Yi Shen, Wei-Fang Sun, Cheng Sun, Chia-Che Chang, Yu-Lun Liu, and Chun-Yi Lee. 3d gaussian splatting with grouped uncertainty for unconstrained images. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025. 2
- [29] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 3
- [30] Shoukang Hu, Tao Hu, and Ziwei Liu. Gauhuman: Articulated gaussian splatting from monocular human videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20418–20431, 2024. 1, 2
- [31] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 2
- [32] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4220–4230, 2024. 2
- [33] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In European Conference on Computer Vision, pages 624–642. Springer, 2022. 2
- [34] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In ECCV, pages 18–35,

- 2024. 3, 5

[35] Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudolabelling real videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6013–6022,

- 2025. 3

- [36] Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021. 2
- [37] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9492–9502,

2024. 3, 7

- [38] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

- 2023. 2, 3

[39] Jina Kim, Jihoo Lee, and Je-Won Kang. Snerv: Spectrapreserving neural representation for video. In European Conference on Computer Vision, pages 332–348. Springer,

- 2024. 2

- [40] Mijeong Kim, Jongwoo Lim, and Bohyung Han. 4d gaussian splatting in the wild with uncertainty-aware regularization. Advances in Neural Information Processing Systems, 37: 129209–129226, 2024. 2
- [41] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian splats. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 505–515,

2024. 1, 2

- [42] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. In European Conference on Computer Vision, pages 252–269. Springer, 2024. 2
- [43] Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, and David Bull. Nvrc: Neural video representation compression. Advances in Neural Information Processing Systems, 37: 132440–132462, 2024. 2
- [44] Ares Lagae, Sylvain Lefebvre, George Drettakis, and Philip Dutr´e. Procedural noise using sparse gabor convolution. ACM Transactions on Graphics (TOG), 28(3):1–10, 2009. 2
- [45] Deqi Li, Shi-Sheng Huang, Zhiyuan Lu, Xinran Duan, and Hua Huang. St-4dgs: Spatial-temporally consistent 4d gaussian splatting for efficient dynamic scene rendering. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [46] Jiahao Li, Bin Li, and Yan Lu. Neural video compression with feature modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26099–26108, 2024. 2
- [47] Zizhang Li, Mengmeng Wang, Huaijin Pi, Kechun Xu, Jianbiao Mei, and Yong Liu. E-nerv: Expedite neural video representation with disentangled spatial-temporal context. In European Conference on Computer Vision, pages 267–

284. Springer, 2022. 2

- [48] Zhenyu Li, Shariq Farooq Bhat, and Peter Wonka. Patchfusion: An end-to-end tile-based framework for highresolution monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10016–10025, 2024. 3
- [49] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis.

- In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8508–8520, 2024. 2
- [50] Feng Liang, Bichen Wu, Jialiang Wang, Licheng Yu, Kunpeng Li, Yinan Zhao, Ishan Misra, Jia-Bin Huang, Peizhao Zhang, Peter Vajda, et al. Flowvid: Taming imperfect optical flows for consistent video-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8207–8216, 2024. 2
- [51] Zhihao Liang, Qi Zhang, Wenbo Hu, Lei Zhu, Ying Feng, and Kui Jia. Analytic-splatting: Anti-aliased 3d gaussian splatting via analytic integration. In European conference on computer vision, pages 281–297. Springer, 2024. 2
- [52] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed 3d gaussian splatting for casual long videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27412–27422, 2025. 2
- [53] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21136– 21145, 2024. 2
- [54] Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. Deformable beta splatting. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11,

2025. 2

- [55] Songrun Liu, Alec Jacobson, and Yotam Gingold. Skinning cubic b´ezier splines and catmull-clark subdivision surfaces. ACM Transactions on Graphics (TOG), 33(6):1–9, 2014. 2
- [56] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen, Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu. Mvsgaussian: Fast generalizable gaussian splatting reconstruction from multi-view stereo. In European Conference on Computer Vision, pages 37–53. Springer, 2024. 2
- [57] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13–23, 2023. 2, 6
- [58] Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen, Min Yang, Xiao Tang, Feng Zhu, and Yuchao Dai. 3d geometry-aware deformable gaussian splatting for dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8900–8910, 2024. 2
- [59] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), pages 800–809. IEEE, 2024. 2
- [60] Wan-Duo Kurt Ma, John P Lewis, and W Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 2
- [61] Marko Mihajlovic, Sergey Prokudin, Siyu Tang, Robert Maier, Federica Bogo, Tony Tung, and Edmond Boyer. Splat-

- fields: Neural gaussian splats for sparse 3d and 4d reconstruction. In European Conference on Computer Vision, pages 313–332. Springer, 2024. 2
- [62] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8089–8099, 2024. 1, 2, 6, 7
- [63] Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong Oh, and Munchurl Kim. Splinegs: Robust motion-adaptive spline for real-time dynamic 3d gaussians from monocular video. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26866–26875, 2025. 2
- [64] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024. 3
- [65] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 1, 2, 6
- [66] LIU Qingming, Yuan Liu, Jiepeng Wang, Xianqiang Lyu, Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dynamic gaussian splatting from casually-captured monocular videos with depth priors. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2
- [67] Zefan Qu, Ke Xu, Gerhard Petrus Hancke, and Rynson WH Lau. Lush-nerf: Lighting up and sharpening nerfs for lowlight scenes. arXiv preprint arXiv:2411.06757, 2024. 2
- [68] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 3, 5
- [69] Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. Film: Frame interpolation for large motion. In European Conference on Computer Vision, pages 250–266. Springer, 2022. 2
- [70] Jens Eirik Saethre, Roberto Azevedo, and Christopher Schroers. Combining frame and gop embeddings for neural video representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9253–9263, 2024. 2
- [71] Mu-Yi Shen, Chia-Chi Hsu, Hao-Yu Hou, Yu-Chen Huang, Wei-Fang Sun, Chia-Che Chang, Yu-Lun Liu, and Chun-Yi Lee. Driveenv-nerf: Exploration of a nerf-based autonomous driving environment for real-world performance validation. arXiv preprint arXiv:2403.15791, 2024. 2
- [72] Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless. Prior-enhanced gaussian splatting for dynamic scene reconstruction from casual video. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–13,

2025. 2

- [73] Seungjun Shin, Suji Kim, and Dokwan Oh. Efficient neural video representation with temporally coherent modulation.

In European Conference on Computer Vision, pages 179–

195. Springer, 2024. 2

- [74] Gaurav Shrivastava, Ser-Nam Lim, and Abhinav Shrivastava. Video decomposition prior: A methodology to decompose videos into layers. arXiv preprint arXiv:2412.04930, 2024. 2
- [75] Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost, Federico Tombari, Gordon Wetzstein, and Leonidas Guibas. Dynamic gaussian marbles for novel view synthesis of casual monocular videos. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 1, 2
- [76] Yang-Tian Sun, Yihua Huang, Lin Ma, Xiaoyang Lyu, YanPei Cao, and Xiaojuan Qi. Splatter a video: Video gaussian representation for versatile processing. Advances in Neural Information Processing Systems, 37:50401–50425, 2024. 1, 2, 3, 6, 7
- [77] Allen Tu, Haiyang Ying, Alex Hanson, Yonghan Lee, Tom Goldstein, and Matthias Zwicker. Speedy deformable 3d gaussian splatting: Fast rendering and compression of dynamic scenes. arXiv preprint arXiv:2506.07917, 2025. 2
- [78] Liao Wang, Kaixin Yao, Chengcheng Guo, Zhirui Zhang, Qiang Hu, Jingyi Yu, Lan Xu, and Minye Wu. Videorf: Rendering dynamic radiance fields as 2d feature video streams. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 470–481, 2024. 2
- [79] Ning-Hsu Albert Wang and Yu-Lun Liu. Depth anywhere: Enhancing 360 monocular depth estimation via perspective distillation and unlabeled data augmentation. Advances in Neural Information Processing Systems, 37:127739–127764,

2024. 3

- [80] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In ICCV, pages 19795–19806, 2023. 2, 3, 6
- [81] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In European Conference on Computer Vision, pages 36–54. Springer, 2024. 3
- [82] Yikai Wang, Xinzhou Wang, Zilong Chen, Zhengyi Wang, Fuchun Sun, and Jun Zhu. Vidu4d: Single generated video to high-fidelity 4d reconstruction with dynamic gaussian surfels. Advances in Neural Information Processing Systems, 37:131316–131343, 2024. 2
- [83] Haato Watanabe, Kenji Tojo, and Nobuyuki Umetani. 3d gabor splatting: Reconstruction of high-frequency surface texture using gabor noise. arXiv preprint arXiv:2504.11003,

2025. 2

- [84] Chung-Ho Wu, Yang-Jung Chen, Ying-Huan Chen, Jie-Ying Lee, Bo-Hsu Ke, Chun-Wei Tuan Mu, Yi-Chuan Huang, Chin-Yang Lin, Min-Hung Chen, Yen-Yu Lin, et al. Aurafusion360: Augmented unseen region alignment for referencebased 360deg unbounded scene inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16366–16376, 2025. 2
- [85] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering.

- In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310–20320, 2024. 2
- [86] Renlong Wu, Zhilu Zhang, Mingyang Chen, Zifei Yan, and Wangmeng Zuo. Deblur4dgs: 4d gaussian splatting from blurry monocular video. arXiv preprint arXiv:2412.06424,

2024. 2, 6

- [87] Skylar Wurster, Ran Zhang, and Changxi Zheng. Gabor splatting for high-quality gigapixel image representations. In ACM SIGGRAPH 2024 Posters, pages 1–2. 2024. 1, 2
- [88] Shuxiang Xie, Shuyi Zhou, Ken Sakurada, Ryoichi Ishikawa, Masaki Onishi, and Takeshi Oishi. G 2 f r: Frequency regularization in grid-based feature encoding neural radiance fields. In European Conference on Computer Vision, pages 186–203. Springer, 2024. 2
- [89] Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. Grid4d: 4d decomposed hash encoding for high-fidelity dynamic gaussian splatting. Advances in Neural Information Processing Systems, 37:123787–123811, 2024. 2
- [90] Hao Yan, Zhihui Ke, Xiaobo Zhou, Tie Qiu, Xidong Shi, and Dadong Jiang. Ds-nerv: Implicit neural video representation with decomposed static and dynamic codes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23019–23029, 2024. 2
- [91] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on Computer Vision, pages 156–173. Springer, 2024. 2
- [92] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20923–20931, 2024. 2
- [93] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8254–8263,

2023. 2

- [94] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024. 3
- [95] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023. 2
- [96] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331–20341, 2024. 2
- [97] Vickie Ye, Zhengqi Li, Richard Tucker, Angjoo Kanazawa, and Noah Snavely. Deformable sprites for unsupervised video decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2657–2666, 2022. 6

- [98] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456,

2024. 2

- [99] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (ToG), 43(6):1–13, 2024. 2
- [100] Yifan Zhan, Zhuoxiao Li, Muyao Niu, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, and Yinqiang Zheng. Kfdnerf: Rethinking dynamic nerf with kalman filter. In European Conference on Computer Vision, pages 1–18. Springer,

2024. 2

- [101] Yu-Ting Zhan, Cheng-Yuan Ho, Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng. Cat-3dgs: A context-adaptive triplane approach to ratedistortion-optimized 3dgs compression. arXiv preprint arXiv:2503.00357, 2025. 2
- [102] Tingyang Zhang, Qingzhe Gao, Weiyu Li, Libin Liu, and Baoquan Chen. Bags: Building animatable gaussian splatting from a monocular video with diffusion priors. arXiv preprint arXiv:2403.11427, 2024. 1, 2
- [103] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-a-video: Temporalconsistent diffusion model for real-world video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2535– 2545, 2024. 2
- [104] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21634– 21643, 2024. 2
- [105] Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang, Tianzhu Zhang, and Yongdong Zhang. Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:101790–101817, 2024. 2

### Contents

- 1. Introduction 1
- 2. Related Work 2
- 3. Preliminary: 3D Gaussian Splatting 3
- 4. Method 3

- 4.1. Overview . . . . . . . . . . . . . . . . . . . 3
- 4.2. Adaptive Gabor Video Representation . . . . 3
- 4.3. Temporally Dynamic Adaptive Gabor . . . . 4
- 4.4. Optimization . . . . . . . . . . . . . . . . . 5
- 4.5. Adaptive Initialization . . . . . . . . . . . . 6

- 5. Experiment 6

- 5.1. Evaluation . . . . . . . . . . . . . . . . . . 6
- 5.2. Applications . . . . . . . . . . . . . . . . . 6
- 5.3. Ablation Study . . . . . . . . . . . . . . . . 6

- 6. Conclusion 8

- A. Activation for Gabor Coefficients 14 A.1. Straight-Through Hard Sigmoid for Fre-

quency Weights . . . . . . . . . . . . . . . 14

- B. Proof of Adaptive Degradation to Gaussian 14

- B.1. Mathematical Formulation . . . . . . . . . . 14
- B.2. Degradation to Gaussian . . . . . . . . . . . 14
- B.3. Implication for Opacity . . . . . . . . . . . . 15
- B.4. Conclusion . . . . . . . . . . . . . . . . . . 15

- C. Additional Visual Comparisons and Results 15

### A. Activation for Gabor Coefficients

#### A.1. Straight-Through Hard Sigmoid for Frequency Weights

In Gabor primitives, the frequency coefficients ωi must satisfy two requirements: (1) values must be constrained within a learnable range, and (2) gradients must flow back through the activation to enable end-to-end optimization.

To achieve this, we employ a Straight-Through Estimator (STE) [4] with a hard sigmoid activation. During the forward pass, we apply hard sigmoid to clip ωi into the range [0,1]:

ωˆ = clip

ω + 1 2

,0,1 . (20)

This ensures that the Gabor kernel’s frequency modulation remains bounded, preventing unbounded growth that could destabilize energy balance.

However, since the hard clipping operation is nondifferentiable, we cannot directly backpropagate through

it. Instead, during the backward pass, we use the gradient of the sigmoid function as a surrogate:

∂L ∂ω

∂L ∂ωˆ · σ(ω)(1 − σ(ω)), (21)

=

where σ(·) is the standard sigmoid function. This provides a smooth, bounded gradient signal.

The combination of bounded forward pass and smooth backward pass achieves stable training: the forward pass prevents artifacts by constraining frequency weights, while the backward pass enables effective gradient-based optimization. This approach avoids exploding gradients that can arise from unbounded activations.

### B. Proof of Adaptive Degradation to Gaussian

We prove that our Adaptive Gabor representation naturally degrades to a traditional Gaussian when all frequency weights vanish, demonstrating its adaptive capability between Gaussian and Gabor modes.

#### B.1. Mathematical Formulation

Recall from Eq. (4) and Eq. (5) in the main paper, the adaptive modulation function is defined as:

1 N

Sadap(x) = b +

N

ωi cos(fi⟨di,x⟩), (22)

i=1

where the compensation term b is given by:

1 N

b = γ + (1 − γ) 1 −

N

ωi , (23)

i=1

with γ ∈ [0,1] as a fixed hyperparameter controlling degradation smoothness, and 1/N normalizing the weighted average of multiple waves.

#### B.2. Degradation to Gaussian

Consider the limiting case where all frequency weights approach zero: ωi → 0 for all i ∈ {1,...,N}.

In this case:

N

ωi → 0. (24) Substituting into the compensation term:

i=1

1

N · 0 = γ +(1−γ)·1 = 1. (25) And the modulation term becomes:

b → γ +(1−γ) 1 −

1 N

Therefore:

N

ωi cos(fi⟨di,x⟩) → 0. (26)

i=1

Sadap(x) → 1 + 0 = 1. (27)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

CoDeF Splatter A Video Ours GT

Figure 12. Visual comparison on DAVIS dataset.

- B.3. Implication for Opacity Since the Gabor-modulated opacity is defined as:

αGabor(x) = G(x) · Sadap(x), (28) when Sadap(x) = 1, we recover:

αGabor(x) = G(x) · 1 = G(x), (29)

which is exactly the traditional Gaussian primitive without frequency modulation.

#### B.4. Conclusion

This proof demonstrates that our Adaptive Gabor representation gracefully degrades to a standard Gaussian when frequency content is not needed (ωi → 0), while smoothly transitioning to frequency-enhanced Gabor modes when high-frequency details are required (ωi > 0). This adaptive behavior is crucial for maintaining energy stability across diverse scene regions with varying frequency characteristics.

### C. Additional Visual Comparisons and Results

For comprehensive visual comparisons with baseline methods across various dynamic scenes, please refer to Figs. 12

to 14. These figures demonstrate our method’s superior performance in preserving high-frequency texture details and maintaining temporal consistency across challenging scenarios including fast motion, occlusions, and complex deformations.

For interactive visualization of downstream application results, including frame interpolation, video editing, and stereo view synthesis, please refer to the supplementary HTML page (index.html). The interactive viewer allows frameby-frame inspection and video playback to better appreciate the temporal coherence and visual quality of our method.

[Figure 175]

[Figure 176]

[Figure 177]

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

CoDeF Splatter A Video Ours GT

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

CoDeF Splatter A Video Ours GT

###### Figure 13. Visual comparison on DAVIS dataset.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

| |[Figure 258]|
|---|---|
| | |

| |[Figure 259]|
|---|---|
| | |

| |[Figure 260]|
|---|---|
| | |

| |[Figure 261]|
|---|---|
| | |

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

CoDeF Splatter A Video Ours GT

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

CoDeF Splatter A Video Ours GT

###### Figure 14. Visual comparison on DAVIS dataset.

