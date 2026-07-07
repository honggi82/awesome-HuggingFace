# arXiv:2509.15130v3[cs.GR]21Mar2026

## Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control

Chenxi Song1 Yanming Yang1 Tong Zhao1 Ruibo Li2 Chi Zhang1,∗ 1AGI Lab, Westlake University 2Nanyang Technological University Project Page: https://worldforge-agi.github.io

[Figure 1]

[Figure 2]

[Figure 3]

SourceVideo

|[Figure 4]|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|---|---|

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]|[Figure 21]|
|---|---|---|

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

|[Figure 43]|[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|[Figure 51]|
|---|---|---|

|[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]|[Figure 59]|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]|
|---|---|---|

[Figure 64]

[Figure 65]

[Figure 66]

CamRe-CamRe-SourceVideo

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

###### 180° 180°

[Figure 80]

[Figure 81]

[Figure 82]

Outpainting Arc Orbit

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

|[Figure 91]|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]|[Figure 99]|
|---|---|---|

|[Figure 100]|[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]|[Figure 108]|
|---|---|---|

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Unprojection

|[Figure 116]|[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]|[Figure 124]|
|---|---|---|

|[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]|[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]|[Figure 133]|
|---|---|---|

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

local close-up

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Free Exploration

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

3D Scene Generation from Single Input View

Dynamic 4D Scene Re-Rendering

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Figure 1. We present WorldForge, a fully training-free framework leveraging a pre-trained video diffusion model for various 3D/4D tasks, such as monocular 3D scene generation (left) and dynamic 4D scene re-rendering (right), enabling precise camera trajectory control and high-quality outputs.

### Abstract

fectively neutralizing artifacts caused by misaligned structural inputs. Together, these components inject precise, trajectory-aligned control without model retraining, achieving accurate motion guidance and photorealistic synthesis. As a plug-and-play, model-agnostic solution, WorldForge demonstrates highly versatile generalizability. Beyond robust zero-shot 3D/4D generation, it readily empowers over a dozen diverse downstream applications, seamlessly enabling tasks like video editing, stabilization, and virtual try-on. Extensive experiments confirm state-of-the-art performance in trajectory adherence and perceptual quality, outperforming both training-dependent and inference-only baselines.

Video diffusion models have rich world priors, but their use in spatial tasks is limited by poor control, spatial-temporal inconsistent results, and entangled scene-camera dynamics. Current approaches, such as per-task fine-tuning or post-process warping, often introduce visual artifacts, fail to generalize, or incur high computational costs. We introduce WorldForge, a novel, training-free framework that operates purely at inference time to resolve these issues. Our method comprises three synergistic components. First, an intra-step refinement loop injects fine-grained motion guidance during the denoising process, iteratively correcting the output to ensure strict adherence to the target camera path. Second, an optical flow-based analysis identifies and isolates motion-related channels within the latent space. This allows our framework to selectively apply guidance, thereby decoupling motion from appearance and preserving visual fidelity. Third, a dual-path guidance strategy adaptively corrects for drift by comparing the guided generation against an unguided, reference denoising path, ef-

### 1. Introduction

Recent video diffusion models (VDMs) [8, 19, 75, 89] have significantly advanced spatial intelligence [11] tasks like 3D/4D understanding [1, 2], reconstruction [65, 76, 82], and generation [94, 95]. Trained on vast video datasets, these models encode rich spatiotemporal priors, enabling realistic spatial transformations for applications like novel view

*Corresponding author.

synthesis [86, 91], panoramic video [53, 79], and dynamic scene reconstruction [3, 74, 94]. Furthermore, VDMs are increasingly used to build “world models” [6, 9, 14], which are structured internal representations that support predictive reasoning in embodied AI.

Despite their strong priors, VDMs face fundamental limitations, including limited controllability, spatial–temporal consistency, and geometric fidelity, particularly when applied to 3D or 4D tasks [21, 44, 80, 87]. They struggle to follow precise motion constraints, such as a 6-DoF camera trajectory [26, 54], which undermines spatial consistency in tasks such as novel view synthesis and trajectory control. These models also entangle scene and camera motion, causing unintended object deformations and scene instability when viewpoint changes are desired [46, 95]. These limitations hinder their use in applications requiring structured spatial reasoning or controllable generation.

To handle these issues, prior works [30, 63, 94, 97] have pursued two main directions. The first, fine-tuning on motion-conditioned data [3, 4, 84], can improve control but is computationally costly, generalizes poorly, and risks degrading pretrained priors. The second, a “warpingand-repainting” strategy [48, 52, 55, 91], re-projects frames along a new camera path and uses a generative model to fill occlusions. Although this approach is more flexible, it lacks robustness because pretrained models handle warped, outof-distribution (OOD) [92] inputs poorly, often producing artifacts and fragmented geometry; for example, a bias toward dynamic training data can cause hallucinated motion in static scenes. Consequently, balancing fine-grained controllability with generation quality and generalization remains a challenging open problem.

To address this challenge, we aim to inject precise control into VDMs while preserving their valuable priors. For this purpose, we propose WorldForge, a general inferencetime guidance paradigm that leverages the rich priors of VDMs in spatial intelligence tasks, such as geometryaware 3D scene generation and video trajectory control. Our method uses a warping-and-repainting pipeline, in which input frames are warped along a reference trajectory and then used as conditional inputs in the repainting stage. Building on this, we develop a unified, trainingfree framework composed of three complementary mechanisms, each designed to address a specific challenge in trajectory-controlled generation, and work synergistically to effectively address the aforementioned OOD challenge.

First, to ensure the generated motion follows the target trajectory derived from depth-based rendering [60, 78], we introduce Intra-Step Recursive Refinement (IRR). It embeds a micro-scale predict–correct loop within each denoising step: before the next timestep, predicted content in observed regions is replaced with the corresponding groundtruth (GT) observations. This incremental correction allows

trajectory control signals to be injected at every step, enabling fine-grained aligned with the target trajectory.

Second, we observe that different channels of the VAEencoded [16, 37] latent representation encode different information, with some channels specializing in appearance and others in motion. This observation is also noted in [85], where they remove content correlation and use statistical methods to find channels that contribute most to the principal motion components. Directly overwriting all channels when injecting trajectory signals can inadvertently degrade visual details. To address this, we propose Flow-Gated Latent Fusion (FLF), which leverages optical-flow similarity to selectively inject trajectory information into channels highly correlated with motion, while leaving appearancerelevant channels unmodified. This process effectively decouples appearance from motion, allowing for precise viewpoint manipulation while preserving content fidelity.

Finally, while warping-based rendering effectively enforces user-defined trajectories, it inevitably introduces noise and visual artifacts stemming from imperfect depth, occlusions, or misalignments [91]. To balance control with quality, we propose Dual-Path Self-Corrective Guidance (DSG). Inspired by CFG [23], DSG uses two parallel denoising paths during inference: A non-guided path that relies on the model’s priors to produce high-fidelity but uncontrolled results, and a guided path that follows the warped trajectory, ensuring camera control but risking artifacts. At each step, DSG computes the difference between these paths to create a dynamic correction term. This term adjusts the guided path toward the higher perceptual quality of the non-guided path. This self-corrective mechanism mitigates artifacts from the warped trajectory while maintaining camera control, improving the video’s overall structure and visual quality.

Together, these three mechanisms form a cohesive inference-time guidance framework for robust and precise trajectory control while preserving VDM priors. Our method is training-free and plug-and-play, enabling broad applicability across tasks without model retraining. It is also model-agnostic and readily adapts to backbones such as Wan 2.1 [75] and SVD [8]. Comprehensive experiments on multiple tasks and benchmarks confirm that our approach achieves state-of-the-art (SOTA) results, improving trajectory adherence, geometric consistency, and perceptual quality over leading baselines. Our main contributions are:

- • WorldForge, a novel, training-free paradigm for leveraging VDM priors in spatial intelligence tasks, enabling precise and stable 3D/4D trajectory control without retraining or fine-tuning.
- • A synergistic inference-time guidance framework integrating Intra-Step Recursive Refinement (IRR) and FlowGated Latent Fusion (FLF), achieving accurate trajectory adherence while decoupling motion from content.

- • Dual-Path Self-Corrective Guidance (DSG), a selfreferential correction mechanism that enhances spatial alignment and perceptual fidelity without auxiliary networks or retraining.
- • Extensive experiments on diverse datasets and tasks show our approach achieves SOTA controllability and visual quality, even compared to training-intensive pipelines.

### 2. Related Works

We review prior work in three relevant areas: 3D static scene generation, 4D trajectory-controlled video generation, and guidance strategies for generative models.

3D Static Scene Generation. While 3D reconstruction [17, 33, 57, 59, 66, 90, 96] and object generation [41, 61, 81, 83] are advanced, they often lack scene-level priors. VDM [8, 40, 75] provide these priors and are leveraged by decoding scenes from images [43], fine-tuning on warped inputs [52, 95], or embedding camera parameters [80, 86]. Unlike costly fine-tuning which can corrupt priors, training-free strategies [46, 91] preserve them but must ensure geometric coherence. Our work takes this training-free approach to enhance both consistency and control.

Trajectory-Controlled Dynamic Video Generation. One paradigm for controllable video synthesis is to fine-tune lightweight adapters [55, 58, 80, 93] like LoRA [25] or ControlNet [98] on video-trajectory data, conditioning on diverse inputs [3, 20, 74, 94]. Another is the warp-andrepaint strategy [28, 48, 55, 72], which projects and inpaints frames but is prone to artifacts from noisy warps [91]. Our work uses inference-time guidance to directly steer the diffusion process for precise motion control.

Guidance and Control for Generative Models. Guidance strategies steer diffusion models toward desired outputs. While Classifier-Free Guidance (CFG) [23] is common, high weights can cause artifacts. More advanced techniques use auxiliary models [29, 32, 88] or iterative refinement [5] to improve sampling. In 3D/4D synthesis, guidance is used to enforce viewpoint consistency, but warp-based methods often suffer from noise-induced artifacts [10, 77]. To address this, we propose DSG. It derives a correction signal from the difference between guided and unguided predictions at each step, enhancing trajectory adherence and stability without retraining.

### 3. Proposed Methods

We propose an inference-time guidance strategy to balance controllability with visual fidelity for VDMs in 3D/4D tasks. Our method is a training-free framework that steers a pretrained model along a user-defined trajectory while preserving its generative priors. As shown in Fig. 2, our framework has three key components. First, Intra-Step Recursive Refinement (IRR) injects trajectory guidance from observed regions at each denoising step for consistent control

(Sec. 3.2). Second, Flow-Gated Latent Fusion (FLF) decouples motion from appearance in the latent space to prevent content drift and preserve fidelity (Sec. 3.3). Finally, DualPath Self-Corrective Guidance (DSG) uses the difference between guided and unguided paths as a corrective signal to suppress artifacts and improve stability (Sec. 3.4). Together, these modules enable fine-grained trajectory control and unlock the model’s latent 3D/4D awareness without any retraining.

#### 3.1. Preliminaries

Before detailing our method, we introduce the necessary preliminaries: diffusion models, guidance strategies, and trajectory-controlled video synthesis.

##### 3.1.1. Denoising Diffusion Models and Guidance

Diffusion Solvers. Generative models are largely diffusion [24] or flow-based [45]. Under the SDE view, diffusion models have a deterministic ODE limit that connects to flow models via reparameterization [18] (The detailed derivation is provided in Supplementary). We use the popular DDIM sampler [67] as an example to illustrate the sampling process: it recovers the clean sample x0 by reversing the forward noising of a Gaussian prior xT. Given a noiseprediction network ϵθ(xt,t), the sampler estimates an intermediate signal xˆ0 from the current state xt:

√1 − α¯t ϵθ(xt,t) √α¯t

xt −

, (1)

xˆ0(xt,t) =

where α¯t denotes cumulative noise attenuation. The term xˆ0(xt,t) is a key intermediate variable: at each step, it is the one-step denoised estimate from ϵθ, evolving from a coarse prediction to a sharp final output. The next sample xt−1 is then obtained by blending xˆ0 with the predicted noise ϵθ:

xt−1 = √α¯t−1 xˆ0(xt,t) + 1 − α¯t−1 ϵθ(xt,t). (2) Iterating from t = T to t = 0 yields the final sam-

ple x0. Our method intervenes at this stage by modifying xˆ0 to enforce trajectory control. Notably, other popular solvers, such as UniPC [99], EDM [31], and PNDM [47], also compute xˆ0 directly or can recover it via a parameterized transformation, so our framework is broadly compatible and can be flexibly applied to current mainstream diffusion-based and flow-based models. Since our experiments primarily use the flow-based Wan2.1 [75] model, we will subsequently detail our algorithm using a flow-based formulation.

Classifier Free Guidance. To improve fidelity to the condition, CFG [23] adjusts the network prediction during sampling as:

ϵ˜θ(xt,t) = ϵθ(xt,t,c)+ωCFG·[ϵθ(xt,t,c) − ϵθ(xt,t,ϕ)], (3)

[Figure 164]

|[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>Input<br><br>xො0𝑁<br><br>x𝑁<br><br>[Figure 173]<br><br>FLF<br><br>[Figure 174]<br><br>IRR<br><br>…<br><br>x1<br><br>|x0|
|---|
<br><br>Result<br><br>[Figure 175]<br><br>IRR<br><br>[Figure 176]<br><br>DSG<br><br>[Figure 177]<br><br>|v𝑜𝑟𝑖|
|---|
<br><br>[Figure 178]<br><br>DSG<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>FLF(ොx0𝑁)<br><br>N<br><br>[Figure 190]<br><br>[Figure 191]<br><br>Inference-Time Guidance<br><br>[Figure 192]<br><br>[Figure 193]<br><br>x𝑡𝑟𝑎𝑗 𝑀<br><br>[Figure 194]<br><br>IRR<br><br>[Figure 195]<br><br>DSG<br><br>[Figure 196]<br><br>Flow-base Gate<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]|
|---|

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Inference-Time Guidance

Single Image Video Frame

###### OR

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

“A truck…“ /“A girl…“

|[Figure 212]<br><br>|[Figure 213]<br><br>[Figure 214]|
|---|
|
|---|

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Image-to-Video Diffusion Model

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

3D

C

[Figure 226]

[Figure 227]

VAE

[Figure 228]

[Figure 229]

[Figure 230]

3D VAE

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

3D/4D Vision Foundation Model

[Figure 235]

|𝑀|
|---|

x𝑡𝑟𝑎𝑗 Mask Reshape

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Static Point Cloud

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

LLM for Generating Scene Prompt C Features Concatenation

|[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>…<br><br>|
|---|
|[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>…|

[Figure 253]

|Video Latent x𝑡𝑟𝑎𝑗 with Trajectory|
|---|

…

Reshaped Mask 𝑀 with Hole

[Figure 254]

|[Figure 255]|
|---|

VDM’s Sampling Process

Video Diffusion Model (VDM)

###### Rendering

| |
|---|

N Noise Interpolation FLF(xො0𝑁) Fused Latent by FLF

Intra-Step Recursive Refinement

[Figure 256]

IRR FLF DSG

[Figure 257]

[Figure 258]

[Figure 259]

Flow-Gated Latent Fusion

[Figure 260]

[Figure 261]

Unguided Velocity Field v𝑜𝑟𝑖

Dual-Path Self-Corrective Guidance

[Figure 262]

…

[Figure 263]

[Figure 264]

|v𝑡𝑟𝑎𝑗|
|---|

Corrected Velocity Field v𝑐𝑜𝑟𝑟 Initial Prediction (Uncontrolled) Guided Prediction (Noisy)

Trajectory-Guided Velocity Field

| |
|---|

[Figure 265]

Corrected Prediction (Controlled and High-Quality )

Models with Frozen Weights

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Dynamic Point Cloud

[Figure 270]

[Figure 271]

[Figure 272]

Inference Results

|xො0𝑁|
|---|

Trajectory-Warped Frames (with Masks)

The Single-Step Denoising Process

Sing-Step Denoised output

- Figure 2. Overview of WorldForge. Given a single image or video frames, a vision foundation model reconstructs a scene point cloud, which is warped and rendered along a user-specified trajectory to produce a guidance video. The input image (or first frame) is also converted into a textual prompt and latent representation for an image-to-video diffusion model. Trajectory control is injected through a training-free strategy comprising IRR, FLF, and DSG (detailed in Sec. 3.2–3.4), enabling precise control and high-quality synthesis without additional training.

#### 3.2. Intra-Step Recursive Refinement

where ωCFG is the guidance weight, with c and ϕ denoting the conditional and unconditional inputs, respectively. This interpolates conditional and unconditional scores to steer the sampling trajectory. Additionally, other works, such as APG, have sought to address issues like color oversaturation by adjusting the scaling of the guidance term. Our approach extends this principle through a self-referential guidance mechanism that dynamically adjusts the guided prediction using the model’s own unguided output at each step.

To enable precise trajectory injection during VDM’s inference processing, we introduce Intra-Step Recursive Refinement (IRR). As noted in Sec. 3.1.1, the denoising process

produces an intermediate variable xˆ(0t), a coarse estimate of the final output and the baseline for later steps,where t

denotes the current timestep. IRR modifies xˆ(0t) to impose trajectory constraints, ensuring that generation follows the desired path.

IRR operates within the updates of Eq. (1) and Eq. (2). Given the one-step denoised sample xˆ(0t) from Eq. (1), we fuse it with the trajectory latent xtraj, obtained by VAEencoding the warped frames I′tar of Eq. (4) into latent space. We then add Gaussian noise ϵ to obtain the modified latent x′t:

##### 3.1.2. Trajectory Control via Depth-Based Warping

Our framework controls trajectories using a depth-based warping-and-repainting strategy. First, a depth-prediction network estimates camera poses and depth maps (Pq,Dq) from single image I or image sequence {Iq}Nq=1 via a function f : {Iq}Nq=1 → {Pq,Dq}. Next, a warping operator W uses these estimates to project a source frame Isrc with depth Dsrc from pose Psrc to a target pose Ptar. This yields a partial target view I′tar and a validity mask Mtar indicating visible pixels:

x′t = (1 − w(σ))F(xˆ(0t),xtraj) + w(σ) · ϵ, (5)

where F(xˆ(0t),xtraj) = M · xtraj + (1 − M) · xˆ(0t) copies observable warped content from xtraj into the correspond-

ing locations of xˆ(0t) using the binary mask M from Eq. (4). This clean-space fusion on xˆ(0t), which differs from prior inpainting works like Lugmayr et al. [51] that operate in the noisy xt−1 space, is critical for our subsequent FLF module. ϵ = xT ∼ N(0,I) is a randomly sampled Gaussian noise used to re-noise the fused latent F(xˆ(0t),xtraj) so that it re-enters the denoising schedule with the injected trajectory. The re-noising is controlled by a scheduler w(σ), and

###### (I′tar,Mtar) = W(Isrc,Dsrc,Psrc,Ptar). (4)

The resulting warped frames and masks guide the VDMs along the target poses Ptar. This guidance is limited to regions visible in the source views. With these preliminaries, we use trajectory control to guide video generation.

the specific strategy can differ for various diffusion and flow models. The re-noised latent x′t is then fed into the network ϵθ, replacing the original xt in Eq. (1) and Eq. (2) for the next sampling step. In summary, IRR embeds a micro predictor–corrector at each denoising step. By updating xˆ(0t) with explicit trajectory cues xtraj, it continually corrects the sampling path and ensures that synthesis follows the target trajectory precisely.

#### 3.3. Flow-Gated Latent Fusion

In the IRR process, overwriting all latent channels with trajectory information degrades visual quality because VAE latents encode both motion and appearance.This observation is supported by Xiao et al. [85], which used PCA to statistically identify channels with distinct motion properties. The indiscriminate update in Eq. (5) injects noise into appearance-focused channels. To address this, we propose Flow-Gated Latent Fusion (FLF), a method that identifies and updates latent channels with high motion relevance. Unlike Xiao et al. [85], we use an optical-flow-based score that directly describes motion to filter channels, requiring no gradient-based optimization.

To select motion-related channels, we use an opticalflow-based scoring scheme since flow directly reflects interframe motion. First, for each channel c of the latent xˆ(0t) at timestep t (the xˆ(0t) denotes the one-step denoised prediction at timestep t), we compute the optical flow between consecutive frames to get a predicted flow Fpred(t,c). The resulting map for each channel has a shape of [2,τ,H,W] (flow vectors, frame pairs, spatial dimensions). Second, we compute a GT reference flow Fgt(t,c) from the target trajectory latent xtraj in the same manner. Finally, we compare the two flows within the visible regions defined by the mask M(c).

The comparison relies on three popular optical flow metrics [71]: Masked End-point Error (M-EPE), which measures the Euclidean distance between predicted and GT flow vectors; Masked Angular Error (M-AE), which measures their angular difference; and Outlier Percentage (Fl-all), which calculates the fraction of unreliable pixels. We combine the normalized metrics to calculate a motion similarity score S(t,c) for each channel in each timestep t:

γk 1 − Normk(t,c) , (6)

S(t,c) =

k∈{E, A, F}

where Norm(kt,c) ∈ [0,1] are the normalized errors from MEPE, M-AE, and Fl-all, and γk are the weighting factors. The calculation and normalization methods for these metrics are provided in the Supplementary. Higher S(t,c) means better flow alignment and stronger motion evidence.

To adaptively set the motion similarity threshold for each scene, we select motion-relevant channels using a dynamic

threshold δ(t) = µ(St) − λ(t)σS(t), where µ(St) and σS(t) are the mean and standard deviation of all channel scores at

step t. We schedule λ(t) to create a loose-to-tight selection over time. This matches the generative process, where early steps define broad structures and later steps handle fine details. Consequently, we disable flow-gating in the early steps (e.g., the first 5) to use all channels for structural integrity, then progressively reduce the number of guided channels in later steps to preserve high-fidelity details. Specific schedule settings are detailed in the Supplementary.

Finally, the latent update selectively fuses the trajectory latent xtraj into the selected high motion-relevance channels:

M(c)x(trajc)+ 1−M(c) x ˆ(0t,c), if S(t,c)≥δ(t) xˆ(0t,c), otherwise.

FLF(xˆ(0t),xtraj)=

(7) This FLF operator replaces the operator F in Eq. (5),

resulting in a more precise fusion rule.

In summary, FLF provides fine-grained trajectory control while preserving model priors and synthesis quality. Unlike methods that restart the sampling schedule [88], iterate in noisy space [51], or globally update the entire latent [46], FLF integrates with our IRR framework to apply selective, per-step guidance, effectively decoupling motion and appearance for precise control. Fig. 4 shows the statistics based on optical flow, demonstrating that each channel exhibits different motion relevance.

#### 3.4. Dual-Path Self-Corrective Guidance

Trajectory latents xtraj obtained by warping often contain distortions from depth errors, occlusions, or misalignments, which degrades synthesis quality. To mitigate this, we draw inspiration from CFG [23]. Conceptually, in the context of a flow model, CFG treats the unconditional prediction as a “bad” direction vuncon and the conditional one as a “good” direction vcon [32]. It then finds a “better” direction by pushing the “good” one away from the “bad” one. Based on this idea, we propose Dual-Path Self-Corrective Guidance (DSG). At each iteration, IRR produces two velocity fields. The unguided velocity vtori (from the original latent xt) stays on the data manifold with high fidelity but ignores the trajectory, which we consider a “bad” direction for control. The guided velocity vttraj (from the corrected latent x′t) may be noisy but follows the trajectory, which we consider a “good” direction. DSG uses the difference between them to find a “better” path.

In essence, DSG robustly balances precise trajectory control with high visual fidelity. Its adaptive cosineweighting is crucial for handling the large angular difference between the guided and unguided paths. It suppresses artifacts by applying strong corrections when the paths diverge, while preserving the model’s natural predictions when they align. This ensures the final corrected ve-

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

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

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Crafter View-

[Figure 312]

Solver View

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

ViewExtrapolater

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

See3DExtrapolater Trajectory-

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

TrajectoryAttention

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

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

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

Crafter

TrajectoryCrafter

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Attention NVS-

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

ReCamMaster

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Trajectory

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

Time Time

- Figure 3. Qualitative comparison for 3D novel view synthesis and 4D trajectory-controlled re-rendering. (Left) For 3D scene generation from a single image, our method produces more consistent and plausible content compared to SOTA baselines. (Right) For 4D rerendering, our approach leverages model priors to avoid common artifacts like floating heads and flattened faces, yielding more realistic results. Overall, our training-free guidance demonstrates superior performance in both static and dynamic scenes. Zoom in for more details.

locity, vtcorr, steers the sample along the target motion path while maintaining the quality of the model’s priors.

However, we empirically find the difference between our vttraj and vtori is far greater than that between the vcon and vuncon in CFG. In extensive tests, we observed that the cosine similarity αt = (vttraj · vtori)/(∥vttraj∥ · ∥vtori∥) between our two paths is typically between 0.3–0.6 (an angle of 50°–70°). In contrast, the cosine similarity between vcon and vuncon is nearly 1 (an angle close to 0°). This discrepancy arises because CFG inputs differ only by a prompt. Our trajectory guidance, however, significantly updates the network input x′t. This leads to a large directional divergence between the guided (vttraj) and unguided (vtori) velocity fields. Therefore, directly applying the CFG fails in our case. To address this large angular difference, we modify the guidance formula to only use the component of the “good” direction that is orthogonal to the “bad” direction. This is achieved by projecting vttraj onto vtori (after normalizing them to equal length) and taking the difference, which avoids the adverse effects of their large directional divergence, This principle of isolating and using the orthogonal guidance component was also explored in APG [64] to eliminate oversaturation. However, the task and input settings we address are completely different. We extend this principle to our trajectory guidance task, which suffers from a much larger velocity fields angular divergence than APG:

vtcorr = vttraj + ρ · βt vttraj − αt · vtori , (8)

where ρ controls guidance strength, αt is the cosine similarity, and βt = 1 − αt2 is its sine counterpart. The role of βt is to adaptively scale the guidance: it amplifies the correction when the paths diverge (low αt, high βt) and reduces it when they agree, preserving the model’s natural prediction.

### 4. Experiments

In this section, we present a comprehensive evaluation of our proposed training-free framework. We first outline the implementation details in Sec. 4.1. Subsequently, we demonstrate the performance of our method on 3D scene generation and 4D trajectory control in Sec. 4.2. Finally, we conduct a series of ablation studies in Sec. 4.3 to validate the effectiveness of each component of our approach.

#### 4.1. Implementation Details

Our framework is a training-free method that steers pretrained VDMs for precise camera control, with no additional training or fine-tuning. At inference time, our IRR is applied during approximately the first 35-45% of the steps. For example, when using the Wan2.1 [75] model with a UniPC [99] sampler to generate a video in 50 steps, our IRR is applied during approximately the first 20 steps, but users can fine-tune this number to 15-25 based on the scene for optimal results.

Setup. Experiments primarily use the Wan2.1 Image-toVideo (I2V-14B) model [75]. Generation runs on a single GPU with ≥69GB VRAM, producing videos up to

[Figure 415]

1280×720. The per-pass sequence length depends on the chosen VDM’s capacity; wider-view videos are obtained by concatenation. We also evaluate on SVD [8], which runs on a 24GB GPU for 25-frame inference, with qualitative results shown in Fig. 6. However, it is worth noting that SVD’s implicit world knowledge is limited by its parameter count and training data, which restricts our method’s performance. To further validate the transferability of our method, we also applied it to the recent LongCat-Video [70] model, again achieving results that surpass SOTA methods.

- 3D Static Scene
- 4D Dynamic Scene Average

FilteringFrequency(%)

Test Datasets and Metrics. For single-view 3D scene generation, we use data from LLFF [56], Tanks and Temples [39], MipNeRF 360 [7], and diverse images such as portraits and AI-generated images (e.g., from Pixabay) to validate generalization, testing on 70+ single views from 40+ scenes for quantitative comparison. We report FID [22] and CLIPsim [62]. As our single-view task does not have GT views, because it is a generation task rather than a reconstruction task, reference-based metrics like PSNR cannot measure the quality of “imagined” unseen scenes [86, 91]. For 4D trajectory control, we test on 50+ videos generated from 30+ selected videos using different camera paths (sourced from DAVIS [34], movie clips, and VDM generations), reporting FVD [73] and CLIP-Vsim. Trajectory accuracy for both tasks is measured via ATE, RPE-T, and RPE-R, which are standard metrics for trajectory and geometric consistency [21, 55].

Channel Index

Figure 4. FLF channel-wise flow statistics. The Y-axis shows filtering frequency; a “filtered out” channel has a low optical flow score (poor motion correlation). Statistics were gathered by tracking indices at each step across 40+ static and 30+ dynamic scenes. The results confirm distinct, stable roles: channel 13 is most frequently filtered out (low motion relevance), while channel 8 is never filtered out (high motion relevance). 4D dynamic scenes show more diverse scores than 3D scenes, reflecting greater motion complexity. This validates our selective guidance approach.

Generation Quality Trajectory Accuracy Training

-Free FID ↓ CLIPsim ↑ ATE ↓ RPE-T ↓ RPE-R ↓

See3D [52] 123.26 0.941 0.091 0.089 0.250 ✗ ViewCrafter [95] 117.50 0.930 0.236 0.315 0.728 ✗ ViewExtrapolator [46] 125.50 0.930 0.183 0.260 0.882 ✓ TrajectoryAttention [86] 122.37 0.920 0.159 0.238 0.532 ✗ TrajectoryCrafter [94] 111.49 0.910 0.090 0.152 0.267 ✗ NVS-Solver [91] 118.64 0.937 0.224 0.268 1.056 ✓

WorldForge (Ours) 96.08 0.948 0.077 0.086 0.221 ✓

#### 4.2. 3D and 4D Trajectory-Controlled Generation

We compare our method against state-of-the-art baselines on both 3D static scene generation and 4D dynamic video control. For 3D novel view synthesis, we evaluate against both training-based [52, 86, 94, 95] and training-free [46, 91] methods. For 4D trajectory control, baselines include ReCamMaster [3] and others [46, 86, 94]. Since ReCamMaster [3] uses a Text-to-Video model, it cannot accept the same video-depth-based warped input to ensure an identical trajectory as the other methods. Therefore, to ensure a fair comparison, we only compare its qualitative visual results.

- Table 1. Quantitative comparison with existing methods on 3D

static scenes. We evaluate generation quality (FID, CLIPsim) and trajectory accuracy (ATE, RPE-T, RPE-R). All methods use official code with identical inputs. ↑: Higher is better, ↓: Lower is better. Our method achieves the best or second-best results.

Generation Quality Trajectory Accuracy Training

-Free FVD ↓CLIP-Vsim ↑ATE ↓RPE-T ↓RPE-R ↓

ViewExtrapolator [46] 108.48 0.913 1.040 1.208 4.750 ✓ TrajectoryAttention [86] 106.94 0.911 0.605 1.238 3.560 ✗ TrajectoryCrafter [94] 97.31 0.923 0.431 1.078 8.950 ✗

WorldForge (Ours) 93.17 0.938 0.527 0.826 2.690 ✓

- Table 2. Quantitative comparison with existing methods on 4D dy-

As shown in Fig. 3, Table 1, and Table 2, our trainingfree method consistently achieves superior results under identical evaluation settings. On 3D static scenes, it outperforms both training-based and training-free baselines on public datasets. On 4D clips with diverse camera paths (e.g., arcs, dolly zooms), it yields higher visual fidelity, tighter trajectory alignment, and more coherent scene completion, matching or surpassing costly training-based approaches. In both settings, our method plausibly reconstructs unseen regions where baselines often produce distortions.

namic scenes. We evaluate generation quality (FVD, CLIP-Vsim) and trajectory accuracy (ATE, RPE-T, RPE-R). All methods use official code with identical inputs.

ance between controllability and fidelity. Furthermore, it serves as a versatile post-production tool for video editing tasks like object addition, removal, replacement, and virtual try-on. Additional visual results and detailed discussions on computational efficiency, robustness across flow estimators, and failure cases are provided in the Supplementary.

Our approach particularly excels in difficult cases, handling human-centric scenes that require high consistency and synthesizing photorealistic 360◦ views from a single input. By preserving model priors, it strikes a strong bal-

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

Figure 5. Ablation of the proposed components. IRR enables trajectory injection; without it, the model defaults to prompt-only free generation, and FLF/DSG cannot be applied. FLF decouples trajectory cues from noisy content; removing it introduces noise from warped frames. DSG guides sampling toward highquality, trajectory-consistent results; without it, detail and plausibility drop. If the standard CFG formulation is applied in DSG, the large angular difference between the two velocity fields causes severe artifacts and errors. The full model achieves the best fidelity and control, demonstrating their complementary effects.

[Figure 568]

[Figure 569]

Figure 6. Ablation across different VDMs. To verify our method’s transferability, we port it to the U-Net–based SVD model [8] and compare it against other SVD-based methods. Our guidance achieves excellent results on native SVD. Furthermore, we applied our method to the recent LongCat-Video [70] model. Leveraging its rich world priors, our method again achieves SOTA results.

Depth Model. Relying on the VDM’s strong world priors, our method effectively mitigates many warping-induced distortions. We experimented with different depth estimators [27, 42, 60, 78] and found our approach maintains robust performance, which allows for plug-and-play integration with various depth models. This integration demonstrates that our method will benefit from future advancements in depth estimation models.

Static Dynamic FID ↓ CLIPsim ↑ FVD ↓ CLIP-Vsim ↑

w/o DSG 109.43 0.943 95.69 0.937 w/o FLF 112.69 0.945 99.79 0.932 w/o DSG&FLF 113.12 0.943 103.17 0.931 DSG (Using CFG Formulation) 120.91 0.936 109.1 0.919

Complete Model (Ours) 96.08 0.948 93.17 0.938

More detailed comparative analysis of model performance, as well as efficiency analysis and metric definitions, are provided in the Supplementary.

Table 3. Quantitative ablation study of our core components. We report generation quality metrics for both static (FID, CLIPsim) and dynamic (FVD, CLIP-Vsim) scenes. All components are shown to be essential for the best performance.

### 5. Conclusion

#### 4.3. Ablation Experiments

We present WorldForge, a training-free framework for trajectory-controllable generation in static 3D and dynamic 4D scenes. Our method effectively balances visual quality, generalization, and precise control in video synthesis. At its heart is a unified, inference-time guidance strategy—comprising IRR, FLF, and DSG. By decoupling motion from appearance and correcting trajectory drift, our framework injects fine-grained control while preserving the rich world priors of the base model. Extensive experiments show state-of-the-art performance on both 3D and 4D generation tasks, offering a new path for exploring spatial intelligence in large-scale generative systems.

Component Analysis. As shown in Fig. 5, we remove IRR, FLF, and DSG in turn. Removing IRR disables trajectory guidance at inference time, resulting in failure to follow the target path. Without FLF, i.e., lacking motion/appearance separation, model priors become entangled, leading to unnatural outputs. Removing DSG introduces noise from warped trajectories into the generation process, causing artifacts and degrading visual quality. Using the standard CFG formulation (Eq. 3) in DSG also fails due to the large angular divergence, causing severe artifacts and errors. The complete model yields the best results, showing that all components are essential and work synergistically to enable robust and precise control.

Although our framework generates high-quality static and dynamic scenes, it currently cannot meet real-time requirements due to its iterative guidance. Future work will focus on applying our method to more powerful generative models and distilling existing ones. This will aim to generate high-resolution, trajectory-controlled video sequences in just a few sampling steps.

Video Model. To test transferability, we integrated our method onto SVD [8] and LongCat-Video [70]. As shown in Fig. 6, our method effectively adapts VDMs into trajectory-controllable scene generation models, its performance scaling positively with the base model’s capabilities.

### Acknowledgement

This work was supported by the National Natural Science Foundation of China (No. 6250070674) and the Zhejiang Leading Innovative and Entrepreneur Team Introduction Program (2024R01007).

### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22875– 22889, 2025. 1
- [2] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. In International Conference on Learning Representations (ICLR), 2025. 1
- [3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14834–14844, 2025. 2, 3, 7, 6
- [4] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. In International Conference on Learning Representations (ICLR), 2025. 2
- [5] Lichen Bai, Shitong Shao, Zikai Zhou, Zipeng Qi, Zhiqiang Xu, Haoyi Xiong, and Zeke Xie. Zigzag diffusion sampling: Diffusion models can self-improve via self-reflection. In International Conference on Learning Representations (ICLR),

2025. 3

- [6] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15791–15801, 2025. 2
- [7] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5470–5479, 2022. 7
- [8] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 2, 3, 7, 8, 6
- [9] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning (ICML), 2024. 2

- [10] Xudong Cai, Yongcai Wang, Zhaoxin Fan, Deng Haoran, Shuo Wang, Wanting Li, Deying Li, Lun Luo, Minhang Wang, and Jintao Xu. Dust to tower: Coarse-to-fine photorealistic scene reconstruction from sparse uncalibrated images. arXiv preprint arXiv:2412.19518, 2024. 3
- [11] Yukang Cao, Jiahao Lu, Zhisheng Huang, Zhuowen Shen, Chengfeng Zhao, Fangzhou Hong, Zhaoxi Chen, Xin Li, Wenping Wang, Yuan Liu, et al. Reconstructing 4d spatial intelligence: A survey. arXiv preprint arXiv:2507.21045,

2025. 1

- [12] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6299–6308, 2017. 2
- [13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 7
- [14] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 27713–27724, 2025. 2
- [15] Gunnar Farneb¨ack. Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis, pages 363–370, 2003. 3, 4, 9
- [16] Simone Foti, Bongjin Koo, Danail Stoyanov, and Matthew J Clarkson. 3d shape variational autoencoder latent disentanglement via mini-batch feature swapping for bodies and faces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18730– 18739, 2022. 2
- [17] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin Brualla, Pratul Srinivasan, Jonathan Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 37:75468–75494, 2024. 3
- [18] Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin Patrick Murphy, and Tim Salimans. Diffusion models and gaussian flow matching: Two sides of the same coin. In The Fourth Blogpost Track at ICLR 2025,

2025. 3, 1, 2

- [19] Google DeepMind. Veo 3 tech report. Google Developers Blog, 2025. 1
- [20] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, et al. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. In SIGGRAPH 2025 Conference Papers (SIGGRAPH), pages 1–12, 2025. 3
- [21] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 7
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a

- two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems (NeurIPS), 30, 2017. 7, 2
- [23] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop on Deep Generative Models and Downstream Applications, 2021. 2, 3, 5
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33:6840–6851, 2020. 3, 1
- [25] Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR),

2022. 3

- [26] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8153–8163, 2024. 2
- [27] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2005–2015, 2025. 8, 7
- [28] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson WH Lau, Wangmeng Zuo, et al. Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation. ACM Transactions on Graphics (TOG), 44(6): 1–15, 2025. 3
- [29] Junha Hyung, Kinam Kim, Susung Hong, Min-Jung Kim, and Jaegul Choo. Spatiotemporal skip guidance for enhanced video diffusion sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11006–11015, 2025. 3
- [30] Hyeonho Jeong, Suhyeon Lee, and Jong Chull Ye. Reanglea-video: 4d video generation as video-to-video translation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11164–11175, 2025. 2
- [31] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems (NeurIPS), 35:26565–26577, 2022. 3, 1
- [32] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems (NeurIPS), 37:52996–53021,

2024. 3, 5

- [33] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42(4), 2023. 3
- [34] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In Asian conference on computer vision (ACCV), pages 123– 141, 2018. 7

- [35] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems (NeurIPS), 36:65484–65516, 2023. 1, 2
- [36] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 34:21696–21707,

2021. 1

- [37] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2
- [38] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026,

2023. 7

- [39] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (TOG), 36

(4):1–13, 2017. 7

- [40] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [41] Min-Seop Kwak, Donghoon Ahn, In`es Hyeonsu Kim, JinHwa Kim, and Seungryong Kim. Geometry-aware score distillation via 3d consistent noising and gradient consistency modeling. arXiv preprint arXiv:2406.16695, 2024. 3
- [42] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10486–10496, 2025. 8, 7
- [43] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N Plataniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navigating 3d scenes from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 798–810, 2025. 3
- [44] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 2
- [45] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3, 1
- [46] Kunhao Liu, Ling Shao, and Shijian Lu. Novel view extrapolation with video diffusion priors. arXiv preprint arXiv:2411.14208, 2024. 2, 3, 5, 7, 6
- [47] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations (ICLR),

2022. 3

- [48] Tianqi Liu, Zihao Huang, Zhaoxi Chen, Guangcong Wang, Shoukang Hu, Liao Shen, Huiqiang Sun, Zhiguo Cao, Wei

- Li, and Ziwei Liu. Free4d: Tuning-free 4d scene generation with spatial-temporal consistency. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 25571–25582, 2025. 2, 3
- [49] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 1
- [50] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems (NeurIPS), 35:5775–5787, 2022. 1
- [51] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022. 4, 5
- [52] Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang. You see it, you got it: Learning 3d creation on pose-free videos at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2016–2029,

2025. 2, 3, 7, 6

- [53] Jingwei Ma, Erika Lu, Roni Paiss, Shiran Zada, Aleksander Holynski, Tali Dekel, Brian Curless, Michael Rubinstein, and Forrester Cole. Vidpanos: Generative panoramic videos from casual panning videos. In SIGGRAPH Asia 2024 Conference Papers (SIGGRAPH ASIA), pages 1–11, 2024. 2
- [54] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 4117–4125, 2024. 2
- [55] Yue Ma, Kunyu Feng, Xinhua Zhang, Hongyu Liu, David Junhao Zhang, Jinbo Xing, Yinhan Zhang, Ayden Yang, Zeyu Wang, and Qifeng Chen. Follow-your-creation: Empowering 4d creation through video inpainting. arXiv preprint arXiv:2506.04590, 2025. 2, 3, 7
- [56] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 38(4):1–14, 2019. 7
- [57] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), pages 405–421, 2020. 3
- [58] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 4296–4304, 2024. 3
- [59] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022. 3

- [60] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10106–10116, 2024. 2, 8, 7
- [61] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations (ICLR),

2023. 3

- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Mearning (ICML), pages 8748–8763, 2021. 7, 2
- [63] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6121– 6132, 2025. 2
- [64] Seyedmorteza Sadat, Otmar Hilliges, and Romann M Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In International Conference on Learning Representations (ICLR), 2025. 6
- [65] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In International Conference on Learning Representations (ICLR), 2024. 1
- [66] Chenxi Song, Shigang Wang, Jian Wei, and Yan Zhao. Fewarnet: An efficient few-shot view synthesis network based on trend regularization. IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 34(10):9264–9280,

2024. 3

- [67] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [68] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1
- [69] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2826, 2016. 2
- [70] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025. 7, 8, 6, 9
- [71] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision (ECCV), pages 402–419, 2020. 5
- [72] Fengrui Tian, Tianjiao Ding, Jinqi Luo, Hancheng Min, and

- Ren´e Vidal. Voyaging into unbounded dynamic scenes from a single view. arXiv preprint arXiv:2507.04183, 2025. 3
- [73] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7, 2
- [74] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision (ECCV), pages 313– 331, 2024. 2, 3
- [75] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 6, 7, 9
- [76] Hanyang Wang, Fangfu Liu, Jiawei Chi, and Yueqi Duan. Videoscene: Distilling video diffusion model to generate 3d scenes in one step. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16475–16485. IEEE, 2025. 1
- [77] Haiping Wang, Yuan Liu, Ziwei Liu, Wenping Wang, Zhen Dong, and Bisheng Yang. Vistadream: Sampling multiview consistent images for single-view scene reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 26772–26782, 2025. 3
- [78] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5294–5306, 2025. 2, 8, 7
- [79] Qian Wang, Weiqi Li, Chong Mou, Xinhua Cheng, and Jian Zhang. 360dvd: Controllable panorama video generation with 360-degree video diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6913–6923, 2024. 2
- [80] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH 2024 Conference Papers (SIGGRAPH), 2024. 2, 3
- [81] Min Wei, Jingkai Zhou, Junyao Sun, and Xuesong Zhang. Adversarial score distillation: When score distillation meets gan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8131– 8141, 2024. 3
- [82] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26024–26035, 2025. 1
- [83] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern

- Recognition Conference (CVPR), pages 21469–21480, 2025. 3
- [84] FU Xiao, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for multientity motion in video generation. In International Conference on Learning Representations (ICLR), 2024. 2
- [85] Zeqi Xiao, Yifan Zhou, Shuai Yang, and Xingang Pan. Video diffusion models are training-free motion interpreter and controller. In Advances in Neural Information Processing Systems (NeurIPS), pages 76115–76138, 2024. 2, 5
- [86] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for fine-grained video motion control. In International Conference on Learning Representations (ICLR), 2025. 2, 3, 7, 6
- [87] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision (ECCV), pages 399– 417, 2024. 2
- [88] Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, and Tommi Jaakkola. Restart sampling for improving generative processess. Advances in Neural Information Processing Systems (NeurIPS), 36:76806–76838, 2023. 3, 5
- [89] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 6
- [90] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In European conference on computer vision (ECCV), pages 767–783, 2018. 3
- [91] Meng You, Zhiyu Zhu, Hui Liu, and Junhui Hou. Nvssolver: Video diffusion model as zero-shot novel view synthesizer. In International Conference on Learning Representations (ICLR), 2025. 2, 3, 7, 6
- [92] Han Yu, Jiashuo Liu, Xingxuan Zhang, Jiayun Wu, and Peng Cui. A survey on evaluation of out-of-distribution generalization. arXiv preprint arXiv:2403.01874, 2024. 2
- [93] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, L´aszl´o Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 37:45256–45280, 2024. 3
- [94] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 100–111, 2025. 1, 2, 3, 7, 6, 8, 9
- [95] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. IEEE Trans-

- actions on Pattern Analysis and Machine Intelligence, pages 1–18, 2025. 1, 2, 3, 7, 6
- [96] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19447– 19456, 2024. 3
- [97] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2050–2062, 2025. 2
- [98] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, 2023. 3
- [99] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 36:49842–49869,

2023. 3, 6, 1

## Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control Supplementary Material

### Contents

- 1. Proof of The Equivalence between Diffusion and Flow Models 1
- 2. Evaluation Metrics 2

- 2.1. Static Scene Evaluation . . . . . . . . . . . 2
- 2.2. Dynamic Scene Evaluation . . . . . . . . . . 2
- 2.3. Camera Trajectory Evaluation . . . . . . . . 2
- 2.4. Evaluation Details . . . . . . . . . . . . . . 3

- 3. Implementation Details 3

- 3.1. Details for FLF Estimation and Motion Scoring 3
- 3.2. Hyperparameter Settings . . . . . . . . . . . 4

- 4. More Experimental Results 4

- 4.1. Efficiency and Runtime Analysis . . . . . . 4
- 4.2. Ablation of DSG and Naive CFG . . . . . . 5
- 4.3. Ablation on Video Diffusion Models . . . . . 7
- 4.4. Ablation on Depth Estimation Models . . . . 7
- 4.5. Applications in Video Editing . . . . . . . . 7
- 4.6. Generation on Challenging Scenes . . . . . . 7
- 4.7. Robustness across Optical Flow Estimators . 9
- 4.8. Limitations and Failure Cases . . . . . . . . 9
- 4.9. More Cases . . . . . . . . . . . . . . . . . . 9

### 1. Proof of The Equivalence between Diffusion and Flow Models

We consider Flow Matching [45, 49] as a special case of diffusion modeling [18, 35]. In the following, we will first outline the formulation of diffusion models and then substitute the specific parameterization of Flow Matching to demonstrate their compatibility.

Given a random variable x0 drawn from an unknown data distribution q0(x0), a Diffusion Probabilistic Model (DPM) [24, 50, 68] defines a forward process that gradually transforms the data into a simple prior distribution, typically a Gaussian distribution. The conditional distribution of the noised variable xt at time t given the initial data x0 is defined as a Gaussian transition kernel [36]:

qt(xt|x0) = N(xt|αtx0,σt2I). (1)

Equivalently, a sample xt at any time t ∈ [0,T] can be expressed through a reparameterization [18, 36]:

xt = αtx0 + σtϵ, ϵ ∼ N(0,I). (2)

Here, αt and σt are scalar functions of time, known as the noise schedule, that control the signal-to-noise ratio. Typically, αt decreases over time while σt increases, satisfying a condition such as αt2 + σt2 = 1 in Variance Preserving (VP) SDEs [24, 68]. Kingma et al. [36] proves that the following stochastic differential equation (SDE) has the same transition distribution in Eq. (1) for any t ∈ [0,T]:

dxt = f(t)xtdt + g(t)dwt, x0 ∼ q0(x0), (3)

where wt is a standard Wiener process. The drift coefficient f(t) and the diffusion coefficient g(t) can be derived using schedule parameters αt and σt [36]:

dσt2 dt − 2

dlog αt dt

dlog αt dt

, g2(t) =

σt2. (4)

f(t) =

The generative process of diffusion models involves reversing this forward process. This can be achieved via a corresponding reverse-time SDE [68]. For more efficient generation, one can utilize the associated probability flow ordinary differential equation (PF-ODE), which shares the same marginal distributions as at each time t as that of the SDE [68]. This PF-ODE is given by:

dxt dt

- 1

- 2

g2(t)∇xt

log pt(xt). (5)

= f(t)xt −

By relating the score function ∇xt

log pt(xt) to the noise term via ∇xt

θ(xt,t)

log pt(xt) ≈ −ϵ

σt , where ϵθ is a neural network trained to predict the noise, the ODE becomes [31, 99]:

g2(t) 2σt

dxt dt

ϵθ(xt,t). (6)

= f(t)xt +

Now, let us consider the forward process in Flow Matching [45, 49]. The path from a data point x0 to a noise sample ϵ is defined by a simple linear interpolation:

xt = (1 − t)x0 + t · ϵ, ϵ ∼ N(0,I), (7)

where t ∈ [0,1]. By comparing Eq. (7) with the general form of the diffusion forward process in Eq. (2), we can establish a direct correspondence by setting the diffusion schedule parameters as:

αt = 1 − t and σt = t.

Substituting this specific parameterization into the definitions for f(t) and g(t) in Eq. (4), we derive the corresponding coefficients for this Flow Matching SDE:

= −1 1 − t

dlog(1 − t) dt

, (8)

fFM(t) =

d(t2) dt − 2 −1 1 − t

2t 1 − t

gFM2 (t) =

t2 =

. (9)

Next, we insert these specific coefficients fFM(t) and gFM2 (t) into the PF-ODE formulation from Eq. (6). To analyze the underlying dynamics, we consider the ideal case where the score is perfectly known, which is equivalent to replacing the model prediction ϵθ(xt,t) with the groundtruth noise ϵ. This yields:

gFM2 (t) 2σt

dxt dt

= fFM(t)xt +

##### ϵ

= −1 1 − t

2t 2t · (1 − t)

xt +

##### ϵ

ϵ − xt 1 − t

=

ϵ − [(1 − t)x0 + t · ϵ] 1 − t

=

(1 − t)ϵ − (1 − t)x0 1 − t

=

= ϵ − x0. (10) This resultant vector field, dxt

dt = ϵ − x0, is precisely the time derivative of the Flow Matching path defined in Eq. (7). This equivalence demonstrates that the process prescribed by Flow Matching is a specific instance of the diffusion models, corresponding to the linear noise schedule αt = 1 − t and σt = t. Therefore, Flow Matching can be formally viewed as a subset of the broader diffusion modeling framework [18, 35].

### 2. Evaluation Metrics

We employ seven complementary metrics to comprehensively evaluate video generation quality: FID and CLIPsim similarity for static scenes, FVD and CLIP-Vsim for dynamic scenes, and ATE, RPE-T, and RPE-R for camera trajectory consistency. These metrics provide objective quantitative assessment across multiple dimensions including image realism, semantic consistency, temporal coherence, and camera motion fidelity.

#### 2.1. Static Scene Evaluation

Fr´echet Inception Distance (FID). FID [22] measures image generation quality by comparing the distribution of real and generated images in the Inception-V3 feature space. We use an ImageNet-pretrained Inception-V3 [69] model and extract 2048-dimensional features from the pool3 layer. The FID score is computed as:

FID = ∥µr − µg∥2 + Tr(Σr + Σg − 2(ΣrΣg)1/2) (11)

where µr and µg are the mean vectors of real and generated image features, and Σr and Σg are the corresponding covariance matrices.

CLIP Similarity. CLIP similarity [62] evaluates the semantic similarity between generated and real images using vision-language pre-trained representations. We employ the CLIP ViT-B/32 model trained on 400 million image-text pairs. The similarity score is calculated as:

1 N

CLIPsim =

N

cos(fr,i,fg,i) (12)

i=1

where fr,i and fg,i are the L2-normalized 512-dimensional CLIP features of the i-th real and generated image pair.

#### 2.2. Dynamic Scene Evaluation

Fr´echet Video Distance (FVD). FVD [73] measures distributional differences between real and generated video using pretrained spatio-temporal features. We use an I3D (Inflated 3D ConvNet) pretrained on Kinetics [12] and extract 1024-D features from the global average pooling layer for each video clip. Following FID, we compute the Fr´echet distance between the Gaussian fits of real and generated I3D features:

FVD = ∥µr−µg∥22+Tr Σr + Σg − 2(ΣrΣg)1/2 , (13)

with µr,µg and Σr,Σg estimated over clip-level I3D features.

Video CLIP Similarity (CLIP-Vsim). CLIP-Vsim extends CLIP similarity to the temporal domain by computing frame-level semantic consistency between generated and real videos. The score is calculated as:

  1

  (14)

Tj

M

1 M

CLIP-Vsim =

cos(fr,j,t,fg,j,t)

Tj

t=1

j=1

where M is the number of video pairs, Tj is the frame count of the j-th video pair, and fr,j,t, fg,j,t are the CLIP features of the t-th frame in the j-th video pair.

#### 2.3. Camera Trajectory Evaluation

Absolute Trajectory Error (ATE). Before evaluation, we align the estimated trajectory to the reference by a global Sim3 transform (scale, rotation, translation). Let the aligned pose components be ˜test,i and R˜ est,i. ATE measures global consistency by the Euclidean distance between corresponding camera positions:

ATEi = tref,i − ˜test,i 2,

n

1 n

ATE =

ATE2i.

i=1

(15)

Relative Pose Error — Translation (RPE-T). RPE-T evaluates local translation accuracy between consecutive

frames. Define relative motions via poses (index gap ∆ = 1):

∆Tref,i = T−ref1,iTref,i+1, ∆Test,i = T˜−est1,iT˜est,i+1. (16)

Let ∆tref,i and ∆test,i be the translation parts of these relative transforms. The per-step error and RMSE are:

RPE-Ti = ∆tref,i − ∆test,i 2,

(17)

n−1

1 n − 1

RPE-T2i.

RPE-T =

i=1

Relative Pose Error — Rotation (RPE-R). RPE-R assesses the accuracy of orientation changes between consecutive frames. Let the relative rotations be

∆Rref,i = R−ref1,iRref,i+1, ∆Rest,i = R˜ −est1,iR˜ est,i+1. (18)

The per-step angular error (degrees) and RMSE are:

trace ∆R⊤ref,i∆Rest,i − 1 2 ·

180 π

RPE-Ri = arccos

,

(19)

n−1

1 n − 1

RPE-R =

RPE-R2i.

i=1

#### 2.4. Evaluation Details

Preprocessing. For FID, images are resized to 299 × 299 and fed to Inception-V3 with standard ImageNet normalization. For FVD and CLIP-based metrics, frames are resized to 224 × 224 with the respective model normalizations. To align with I3D input requirements for FVD, we uniformly downsample the generated videos to 16 frames while strictly preserving the start and end frames to maintain boundary constraints. A similar temporal sampling strategy is applied for CLIP-based video metrics. For camera trajectory evaluation, images are resized to 720 × 480 and uniformly sampled to 20 frames.

Evaluation Protocol. To ensure robust feature covariance estimation for distributional metrics, we compute statistics over the aggregated evaluation set. Specifically, for FID on static scenes, we aggregate approximately 1,200 GT images from 40+ scenes as the reference distribution, comparing them against ≈ 2,200 generated novel views (40∼120 views per scene). For wild scenes without ground truth, we report the average FID over 5 representative scene groups using manually curated reference sets (20∼40 images per scene) to ensure content consistency. For FVD, we evaluate ≈ 700 generated video clips from 50+ scenes against an equal number of reference clips extracted from DAVIS and cinematic sequences.

All baseline methods are evaluated under this identical protocol. This standardized comparison ensures that the observed relative performance gaps reliably reflect the intrinsic differences in generation quality, verifying the superiority of our method across the evaluated metrics.

For trajectories, poses are recovered by SfM, the estimated trajectory is aligned to the reference by Sim3 to resolve scale, and metrics are computed using evo with alignment and scale correction enabled.

### 3. Implementation Details

This section provides a detailed breakdown of the FlowGated Latent Fusion (FLF) module,as introduced in Section 3.3 of the main paper. The goal of FLF is to identify and selectively update latent channels that are highly relevant to motion, thereby preserving visual details encoded in appearance-focused channels. To achieve this, at each denoising step i, FLF computes a motion similarity score S(t,c) for each latent channel c. Below, we detail how this score is calculated.

#### 3.1. Details for FLF Estimation and Motion Scoring

Optical Flow Estimation At each denoising step i, we compute optical flow maps for each channel c of both the

predicted latent xˆ(0t) and the target trajectory latent xtraj. The computation is performed frame-by-frame; that is, for each latent tensor, we calculate the dense optical flow between consecutive temporal frames using the Farneb¨ack algo-

rithm [15]. This process yields a predicted flow map, Fpred(t,c), and a ground-truth (GT) flow map, Fgt(t,c). At each pixel, the flow is a 2D vector (u∗,v∗) representing horizontal and vertical displacement. All subsequent metric calculations are performed over the set of valid (i.e., non-occluded) pixels, defined as Ω(t,c) = {(x,y,τ) | M(c)(x,y,τ) = 1}, where (x,y) are pixel coordinates and τ is the frame index. Since optical flow is computed between adjacent frames, for a latent tensor with Tl total frames, the index τ ranges from 1 to Tl − 1.

Metric Calculation The motion score S(t,c) is derived from three standard optical flow metrics that quantify the

error between the predicted flow Fpred(t,c) and the ground-truth flow Fgt(t,c) at each step i.

- • Masked End-point Error (M-EPE) measures the average Euclidean distance between the predicted and GT flow vectors over all valid pixels. Let err(x,y,τ) be the Euclidean error at a specific pixel:

err(x, y, τ) = Fpred(t,c)(x, y, τ) − Fgt(t,c)(x, y, τ)

2

. (20)

The M-EPE is then calculated as:

M-EPE(t,c) =

1 |Ω(t,c)|

(x,y,τ)∈Ω(t,c)

err(x, y, τ). (21)

- • Masked Angular Error (M-AE) calculates the average angular difference. We first define the cosine similarity

sim(x,y,τ) between the flow vectors:

sim(x, y, τ) = Fpred(t,c)(x, y, τ) · Fgt(t,c)(x, y, τ) ∥Fpred(t,c)(x, y, τ)∥ · ∥Fgt(t,c)(x, y, τ)∥

. (22)

The M-AE is derived by averaging the arccosine of this similarity:

1 |Ω(t,c)|

M-AE(t,c) =

arccos (sim(x, y, τ)) . (23)

(x,y,τ)∈Ω(t,c)

• Outlier Percentage (Fl-all) is the percentage of pixels in Ω(t,c) where the flow estimation is considered erroneous. Following standard benchmarks, a pixel is flagged as an outlier if its M-EPE exceeds 3 pixels or if its relative er-

ror, ∥Fpred(t,c) − Fgt(t,c)∥2/∥Fgt(t,c)∥2, is greater than 5%. We denote this outlier percentage as F(t,c).

Normalization and Weighting The three metrics exist on different scales, so we first normalize each to the range [0,1] before combining them. This corresponds to the Normk(t,c) terms used in the main text:

NormE(t,c) = min(M-EPE(t,c)/nE,1), NormA(t,c) = min(M-AE(t,c)/nA,1), NormF(t,c) = min(F(t,c)/nF,1),

(24)

where nE,nA, and nF are normalization constants chosen to reflect typical value ranges for each metric. The final motion score S(t,c) is a weighted sum of the inverted normalized errors, as defined in Eq. (25) (corresponding to Eq. (6) in the main text):

γk 1 − Normk(t,c) , (25)

S(t,c) =

k∈{E, A, F}

where the weights γk (where k ∈ {E,A,F} and k γk = 1) and the normalization constants are set based on com-

mon practices in optical flow evaluation to balance each metric’s contribution. In our experiments, we set nE = 10, nA = 30, and nF = 0.5. The weights in Eq. (25) are set to (γE,γA,γF) = (0.4,0.4,0.2).

#### 3.2. Hyperparameter Settings

To facilitate reproducibility, we provide the default hyperparameter settings used in our experiments, as listed in Table 1. These values are based on the implementation with the Wan 2.1 backbone. In practical applications, users may fine-tune these coefficients according to specific scene requirements to achieve optimal results.

Beyond the coefficients listed above, we specify several key operational parameters. Taking the Wan 2.1 model (which uses 50 sampling steps) as an example, the IntraStep Recursive Refinement (IRR) is applied by default during the first 20 sampling steps. For optical flow estimation

Table 1. Default coefficient settings used in our experiments (taking Wan 2.1 implementation as an example). While these values serve as a robust baseline, users can fine-tune them for specific scenes to maximize generation quality.

###### Reference Value Description

Eq. (6) in main paper

[0.4, 0.4, 0.2] Weights for optical flow metrics λ(t)

γk

Threshold for FLF channel filtering ρ

Eq. (6) in main paper

0.65

Eq. (8) in main paper

4.0 DSG guidance scale

within the FLF module, we utilize the classic Farneback algorithm [15] with its default settings (‘pyr scale‘=0.5, ‘levels‘=3, ‘winsize‘=15).

Critically, regarding the channel filtering in FLF, in addition to the threshold λ(t), we implement a progressively relaxed channel selection strategy to balance structural guidance and detail preservation: Phase 1 (Steps 0–5): Channel filtering is disabled. All channels are injected with guidance information. This is because the early-stage latent states are too noisy for reliable optical flow calculation, necessitating full guidance injection to establish initial structure. Phase 2 (Steps 6–10): We enforce a strict limit where a maximum of 2 channels are allowed to retain the original prediction (i.e., at least 14 channels are replaced with guided features). This ensures sufficient structural information is injected during the formative stages of generation. Phase 3 (Steps ≥ 11): The constraint is relaxed to allow a maximum of 6 channels to retain the original prediction. This looser constraint prevents excessive guidance from compromising fine texture details in the later stages.

### 4. More Experimental Results

This section provides additional experiments and results that complement the findings presented in the main paper. We include a detailed efficiency analysis, further ablation studies, and more qualitative examples to fully demonstrate the capabilities and robustness of our framework.

#### 4.1. Efficiency and Runtime Analysis

Our framework is training-free and operates entirely at inference time. Table 2 provides a detailed comparison of inference efficiency against several state-of-the-art methods on a single NVIDIA A100 GPU.

Our method incurs zero training cost, offering a significant advantage over resource-intensive fine-tuning approaches. The primary computational overhead stems from the IRR module, which effectively adds an extra sampling process, taking approximately the same time as a single standard sampling step. In contrast, the DSG module in-

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

- Figure 1. Qualitative ablation study of the DSG method. Substituting DSG with a standard CFG formulation fails to handle the large angular disparity between the two velocity fields, resulting in significant visual artifacts and errors. Removing the adaptive weighting

factor βt (denoted as DSG w/o βt) compromises guidance stability and introduces inconsistencies. In contrast, our full DSG framework stably generates high-fidelity and structurally consistent results.

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

MagaSAM-UnidepthDepthCrafterVGGT

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 630]|
|---|

[Figure 631]

|[Figure 632]|
|---|

[Figure 633]

[Figure 634]

Source (warped) Outputs

[Figure 635]

[Figure 636]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 637]|
|---|

|[Figure 638]|
|---|

|[Figure 639]|
|---|

|[Figure 640]|
|---|

[Figure 641]

[Figure 642]

[Figure 643]

| |
|---|

| |
|---|

| |
|---|

[Figure 644]

Input View

| |
|---|

[Figure 645]

Target View

|[Figure 646]<br><br>[Figure 647]<br><br>| |
|---|
<br><br>|
|---|

|[Figure 648]<br><br>[Figure 649]| |
|---|---|
|[Figure 650]<br><br>[Figure 651]<br><br>| |
|---|
<br><br>| |
| | |

|[Figure 652]<br><br>[Figure 653]<br><br>| |
|---|
<br><br>|
|---|

[Figure 654]

MasksWarpedImagesInferenceResults

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

|[Figure 659]<br><br>[Figure 660]|
|---|

|[Figure 661]<br><br>[Figure 662]|
|---|

| |
|---|

[Figure 663]

Input View

| |
|---|

[Figure 664]

Target View

- Figure 2. Depth-models ablation. Our method leverages the inherent world knowledge of VDMs to correct errors and fill missing regions even under challenging inputs (left). This strong self-correction ability ensures broad compatibility with different depth estimators (right). Despite variations or noise in depth-based warping, it reliably compensates through learned priors and produces realistic, high-quality results.

volves only a simple matrix operation per step, incurring negligible temporal cost. Similarly, the FLF module’s channel-wise optical flow estimation presents a substantially lower computational burden than the backbone model inference. A detailed breakdown of these component costs is reported in Table 3. Note that while absolute times may vary across hardware configurations, the relative temporal relationships remain reliable. Despite these additions, our framework maintains inference speeds comparable to, and often faster than, existing methods, as evidenced in Table 2.

This demonstrates that our framework achieves robust controllability without prohibitive computational costs, offering an efficient alternative to training-intensive pipelines.

#### 4.2. Ablation of DSG and Naive CFG

Through extensive experiments, we observe that the difference between our trajectory-guided velocity vttraj and the unguided velocity vtori is far greater than that between the conditional vcon and unconditional vuncon estimates in standard CFG [23]. Specifically, empirical analysis reveals that

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

- Figure 3. Other video effects enabled by our method. Beyond video re-cam, our flexible depth-based warping also supports various video editing operations, such as freezing the camera, stabilizing video, and editing video content. These extensions further broaden the practical scope of our approach.

Table 3. Computational cost breakdown of a single generation step. We report the runtime of each component on an NVIDIA A100, taking the generation of a 49-frame video at 832×480 resolution as an example. By default, we apply our guidance during the first 20 sampling steps. The primary overhead comes from the IRR module, while the DSG module incurs negligible cost.

Table 2. Efficiency comparison. We measure inference throughput on a single NVIDIA A100 across methods built on SVD [8], Wan 2.1 [75], CogVideoX [89], LongCat [70], and custom backbones. Our approach achieves competitive generation speed compared to prior approaches while avoiding any training overhead. SR denotes LongCat’s built-in super-resolution model, and D represents its distilled version. The best and second-best results are highlighted in bold and underlined, respectively.

Component Time (s)

VAE Encoding & Decoding 4.5 Backbone Inference (Transformer) 8.4

Inference Speed (frames/min)

Base Video Model

Training -Free

Resolution

FLF Module (Ours) 1.2 DSG Module (Ours) ≈ 0 IRR Module (Ours) 14.1

See3D [52] 576 × 1024 14.71 Custom ✗ ViewCrafter [95] 576 × 1024 13.89 Custom ✗ ViewExtrapolator [46] 576 × 1024 15.63 SVD ✓ TrajectoryAttention [86] 576 × 1024 4.55 SVD ✗ TrajectoryCrafter [94] 384 × 672 14.71 CogVideoX ✗ NVS-Solver [91] 576 × 1024 2.69 SVD ✓ ReCamMaster [3] 480 × 832 5.55 Wan 2.1 T2V ✗

Total Runtime 29.0

WorldForge, on Wan 2.1 [75] 720 × 1280 1.45 Wan 2.1 I2V ✓ WorldForge, on Wan 2.1 [75] 480 × 832 3.68 Wan 2.1 I2V ✓ WorldForge, on SVD [8] 576 × 1024 19.23 SVD ✓ WorldForge, on LongCat [70] 480 × 832 3.85 Longcat ✓ WorldForge, on LongCat [70] 720 × 1280 3.33 Longcat (SR) ✓ WorldForge, on LongCat [70] 480 × 832 16.67 Longcat (D) ✓ WorldForge, on LongCat [70]720 × 1280 10.20 Longcat (D+SR) ✓

visual comparison is presented in Fig. 1, where we compare our full framework against a standard CFG implementation and an ablated version without the adaptive weight βt. The results demonstrate that replacing DSG with a naive CFG formulation leads to severe visual artifacts and structural distortions, while removing the adaptive weighting factor reduces guidance stability. In contrast, our full DSG framework successfully maintains structural integrity and high perceptual quality while closely adhering to the intended camera path.

the angular difference in our setting typically ranges from 50◦ to 70◦, significantly exceeding the < 5◦ divergence found in typical CFG scenarios. To mitigate the adverse effects caused by this large angular discrepancy, we propose Dual-Path Self-Corrective Guidance (DSG). A direct

#### 4.3. Ablation on Video Diffusion Models

To evaluate the transferability of our proposed guidance mechanism and its performance across models of varying parameter scales, we conducted ablation studies by porting our entire framework to the U-Net-based Stable Video Diffusion (SVD) [8], which possesses fewer parameters, and the recently released LongCat model [70]. We performed minor hyperparameter fine-tuning to adapt our method to their respective architectures and sampling strategies. Subsequently, we conducted a fair comparison using identical inputs. Quantitative results are presented in Table 4. It is worth noting that due to SVD’s constraints in parameter count and architecture, it encapsulates fewer inherent world priors, which prevents it from fully exploiting the potential of our guidance algorithm. Comprehensive visualization results demonstrating the capabilities across these models are shown in Fig. 10 through Fig. 14.

Table 4. Quantitative comparison across different backbones. Using single-view 3D scene generation as a benchmark, we evaluate our method on SVD [8], Wan 2.1 [75], and LongCat [70]. The results demonstrate the scalability of our approach and its ability to generalize across different VDM architectures. Furthermore, the performance gains on advanced backbones indicate that our method effectively leverages the capabilities of the underlying model, promising improved generation quality as base models continue to evolve.

CLIP ↑ ATE ↓ RPE-T ↓ RPE-R ↓

WorldForge (on SVD [8]) 0.910 0.265 0.316 0.444 WorldForge (on Wan 2.1 [75]) 0.948 0.077 0.086 0.221 WorldForge (on LongCat [70]) 0.949 0.095 0.076 0.230

#### 4.4. Ablation on Depth Estimation Models

Our framework operates on a warp-and-repaint strategy. To assess the robustness and flexibility of our approach regarding depth estimation, we evaluated its performance using several state-of-the-art depth estimators: VGGT [78], UniDepth [60], Mega-SaM [42], and DepthCrafter [27]. As illustrated in Fig. 2, our method demonstrates broad compatibility, maintaining consistently high performance across all tested models. Even when depth-based warping yields challenging inputs characterized by noise, errors, or significant disocclusion regions, our framework effectively compensates for these imperfections. This resilience stems from the strong generative world priors inherent in the underlying VDM, which our guidance modules leverage to correct artifacts and plausibly inpaint missing areas during the repainting stage. This self-correction capability confirms that our framework functions in a plug-and-play manner with various depth estimation techniques and naturally scales with improvements in depth estimation performance.

#### 4.5. Applications in Video Editing

Beyond trajectory-controlled generation, our framework’s flexibility makes it a powerful tool for various video postproduction and editing tasks. This includes effects like video stabilization, camera freezing, and dynamic viewpoint switching.

Furthermore, by incorporating a flexible masking strategy, our framework can perform diverse content edits such as object removal, addition, subject replacement, and virtual try-on seamlessly. The general process for these edits involves first segmenting the target region in each frame using a tool like SAM [38]. The desired edit is then applied to the first frame (e.g., using Gemini [13]). Finally, this edited frame and the corresponding masks are processed by our pipeline to render a temporally consistent result. For adding new objects where none exist in the source video, a simple bounding box can be provided to guide the placement. Fig. 3 shows several qualitative examples of these video editing effects.

#### 4.6. Generation on Challenging Scenes

Our approach demonstrates robust performance in difficult cases where other methods may falter. We highlight two such scenarios: human-centric scenes and single-image 360° view generation.

Human-Centric Scenes. Human-centric scenes are challenging for novel view synthesis due to the need for high structural and temporal consistency. As shown in Fig. 4, some methods can struggle with these cases, sometimes introducing artifacts, unintended motion, or difficulty rendering plausible facial features. For instance, TrajectoryCrafter [94] may recover the coarse structure, but can introduce unnatural facial deformations. In contrast, our method’s use of strong generative priors and precise trajectory guidance helps maintain scene stationarity and consistency, producing more natural renderings that better preserve the subject’s appearance.

Large Camera Movements and 360◦ View Generation. Generating large camera movements (e.g., a 180◦ turn) or full 360◦ orbit views from a single image in a single pass is highly challenging for existing methods. It risks hallucination in invisible regions due to the limited field-of-view of the source observation. Our method effectively resolves this problem via an iterative multi-clip generation strategy. By using the last frame of the previous clip as the prior for the next, we successfully achieve large-range scene generation, such as 180◦ turns (as shown in Fig. 5). Furthermore, combined with our framework’s precise trajectory control, it enables the creation of coherent, object-centric orbit views of complex scenes (Fig. 6). We achieve the full 360◦ loop by generating a sequence where the final frame seamlessly connects to the first. This is made possible by our precise guidance, which maintains high image quality

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

- Figure 4. Static 3D generation on human-centric scenes. Existing methods struggle, particularly with motion-prone shots (left) and portrait close-ups (right). On the left, baselines introduce artifacts and unintended motion. On the right, most fail to produce plausible results; TrajectoryCrafter [94] recovers coarse structure but lacks detail and visual appeal. In contrast, our method maintains scene stationarity under trajectory guidance and produces natural, faithful renderings, achieving both precise control and high perceptual quality.

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

- Figure 5. Large camera movements (e.g., 180◦). Single-pass generation of large angles often suffers from poor quality. Our method effectively resolves this problem via iterative generation.

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

- Figure 6. 360◦ orbit views from a single real-world outdoor image. With precise trajectory control and realistic rendering, our method overcomes the viewpoint limitation of single-image generation and produces ultra-wide views of complex real scenes. Unlike panoramabased approaches, it directly supports object-centric trajectories and achieves higher visual quality.

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

w/o FLF w/o FLF w/o FLF

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

- Figure 7. Robustness in challenging scenarios. Our framework maintains structural integrity even under fast motion and complex occlusions.

[Figure 949]

w/o FLF w/o FLF w/o FLF

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

- Figure 8. Robustness across optical flow estimators. FLF consistently enhances quality with both Farneb¨ack [15] and RAFT, validating its flexibility and robustness.

and prevents the accumulation of errors over the entire longrange trajectory, thereby avoiding a common point of failure in other methods. Unlike traditional panoramic approaches, our method directly generates a continuous view along a given trajectory, offering more flexibility and strong visual quality, particularly for object-centric paths.

Challenging Scenarios. We also demonstrate the robustness against fast motion, occlusions, and non-rigid dynamics. As illustrated in Fig. 7, even when local flow precision drops, enabling FLF consistently yields better results than disabling it. This robustness relies on our adaptive designs: when flow is unstable (e.g., high variance), our dynamic threshold automatically adjusts. This allows more channels to receive control signals, preventing misclassification and maintaining structural stability.

#### 4.7. Robustness across Optical Flow Estimators

A potential concern is whether our framework heavily relies on a specific optical flow algorithm. To validate this, we replaced the lightweight Farneb¨ack algorithm [15] with the learning-based RAFT model. As shown in Fig. 8, our FlowGated Latent Fusion (FLF) consistently enhances generation quality regardless of the flow backbone. Furthermore, our multi-metric scoring system (evaluating magnitude, direction, and reliability) mitigates single-metric noise, ensuring that the lightweight Farneb¨ack choice is highly sufficient for our pipeline.

#### 4.8. Limitations and Failure Cases

While our framework achieves precise zero-shot camera control, we acknowledge that, similar to other depthwarping-based methods (e.g., TrajectoryCrafter [94]), our performance is inherently bottlenecked by the quality of the underlying depth estimation.

As illustrated in Fig. 9, in scenarios involving extremely complex scene dynamics or severe depth errors, structural distortions may occur. However, it is worth noting that our dynamic gating mechanism automatically reduces the number of filtered channels in such chaotic cases, ensuring that the outputs are typically no worse than the baseline model without FLF guidance. In future work, introducing explicit camera pose encoding or semantic priors could help resolve these fundamental ambiguities and further enhance robustness against depth failures.

#### 4.9. More Cases

To provide a more comprehensive evaluation of our method’s performance across different backbone architectures, we present additional qualitative results in Fig. 10, Fig. 11, Fig. 12, Fig. 13 and Fig. 14. As illustrated, our approach achieves superior visual fidelity and structural plausibility, consistently delivering state-of-the-art performance on both Wan 2.1 [75] and LongCat [70] models.

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

###### Figure 9. Failure cases. Erroneous depth estimation in highly complex scenes can diminish the control accuracy, leading to artifacts.

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

###### Figure 10. Additional qualitative results for single-view 3D scene generation (Case 1). Validated on Wan 2.1 and LongCat architectures, our method consistently produces 3D-consistent novel views with high visual fidelity.

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

###### Figure 11. Additional qualitative results for single-view 3D scene generation (Case 2). Validated on Wan 2.1 and LongCat architectures, our method consistently produces 3D-consistent novel views with high visual fidelity.

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

###### Figure 12. Additional qualitative results for dynamic video re-filming (Case 1). Validated on Wan 2.1 and LongCat architectures, our method enables effective camera control with superior realism and temporal smoothness.

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

###### Figure 13. Additional qualitative results for dynamic video re-filming (Case 2). Validated on Wan 2.1 and LongCat architectures, our method enables effective camera control with superior realism and temporal smoothness.

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

###### Figure 14. Additional qualitative results for dynamic video re-filming (Case 3). Validated on Wan 2.1 and LongCat architectures, our method enables effective camera control with superior realism and temporal smoothness.

