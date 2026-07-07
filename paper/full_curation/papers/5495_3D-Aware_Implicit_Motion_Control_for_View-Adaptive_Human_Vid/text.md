## 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation

Zhixue Fang1,∗ Xu He2,∗ Songlin Tang1,∗ Haoxian Zhang1,†, Qingfeng Li3 Xiaoqiang Liu1 Pengfei Wan1 Kun Gai1 1Kling Team, Kuaishou Technology 2Tsinghua University 3CASIA ∗Equal contribution †Project leader Corresponding author

https://hjrphoebus.github.io/3DiMo

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

DrivingVideoDrivingVideoDrivingVideoOutputVideoOutputVideoOutputVideoOutputVideoOutputVideoOutputVideo

DrivingVideoDrivingVideoDrivingVideoOutputVideo

# arXiv:2602.03796v2[cs.CV]14Feb2026

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

Camera tilts downward at a normal speed.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Camera pans to the right around the woman and rises.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Camera rotates in a circular path around the woman.

OutputVideo

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

Camera slowly arcs to the left around the woman.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Camera quickly arcs to the left around the girl.

###### OutputVideo

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Camera slowly arcs to the left around the man.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Camera moves upward while arcing right slowly.

###### OutputVideoOutputVideo

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Camera zooms out at a normal speed.

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

Camera pans to the right in a circular motion.

[Figure 81]

[Figure 82]

[Figure 83]

Camera slowly rises and rotates around the boy.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Camera quickly zooms out while moving upward.

OutputVideo

Camera moves slowly backward and descends.

Figure 1. 3DiMo can faithfully reproduce the 3D spatial motion from a 2D driving video, supporting flexible text-guided camera control.

1

### Abstract

Existing methods for human motion control in video generation typically rely on either 2D poses or explicit 3D parametric models (e.g., SMPL) as control signals. However, 2D poses rigidly bind motion to the driving viewpoint, precluding novel-view synthesis. Explicit 3D models, though structurally informative, suffer from inherent inaccuracies (e.g., depth ambiguity and inaccurate dynamics) which, when used as a strong constraint, override the powerful intrinsic 3D awareness of large-scale video generators. In this work, we revisit motion control from a 3D-aware perspective, advocating for an implicit, view-agnostic motion representation that naturally aligns with the generator’s spatial priors rather than depending on externally reconstructed constraints. We introduce 3DiMo, which jointly trains a motion encoder with a pretrained video generator to distill driving frames into compact, view-agnostic motion tokens, injected semantically via cross-attention. To foster 3D awareness, we train with view-rich supervision—singleview, multi-view, and moving-camera videos—forcing motion consistency across diverse viewpoints. Additionally, we use auxiliary geometric supervision that leverages SMPL only for early initialization and is annealed to zero, enabling the model to transition from external 3D guidance to learning genuine 3D spatial motion understanding from the data and the generator’s priors. Experiments confirm that 3DiMo faithfully reproduces driving motions with flexible, text-driven camera control, significantly surpassing existing methods in both motion fidelity and visual quality.

### 1. Introduction

Recent advances show that large-scale video generation models possess strong 3D spatial awareness and motion reasoning [29, 32], enabling text-guided novel-view synthesis and human reposing with consistent 3D geometry [11, 13, 30]. Meanwhile, controllable video generation has emerged as a research focus, with human motion control being one of its central challenges, which aims to animate a reference image according to motion cues from a driving video. Existing approaches typically extract 2D-rendered pose images from driving frames [4, 5, 7, 35, 37] and inject them via pixel-aligned conditioning [9, 34]. However, such 2D conditioning rigidly binds motion to the driving viewpoint, preventing the model from reasoning about motion in its inherently 3D nature. As a result, generated videos collapse to the 2D projection of the driving view, losing viewpoint flexibility and limiting applications such as novel-view human synthesis or cinematic camera motion.

In this work, we revisit the task of human motion control from a 3D-aware perspective. Our goal is to enable the model to reproduce the underlying 3D motion implied in 2D

driving frames, while maintaining independent, text-guided camera control during generation. To achieve similar goals, recent works attempt to explicitly separate motion and camera control via 3D reconstruction—recovering SMPL(X) [16, 21] sequences from driving videos and conditioning generation through mesh rendering [4] or projected keypoints [5] under predefined camera trajectories. This line of work indeed moves beyond 2D constraints by recognizing that human motion inherently occurs in 3D space. However, it still faces a fundamental limitation: the motion representation is fully determined by externally estimated parametric reconstructions such as SMPL. These estimates, while structurally stable, suffer from depth ambiguities [8] (e.g., forward tilting, incorrect inter-limb contact, or distorted Zaxis motion), and their inaccurate reconstruction of natural motion dynamics further limits expressiveness. When such biased 3D signals are injected into the generatorespecially through rigid projection-based 2D alignmentthey impose strong geometric constraints that override native 3D priors of large-scale video models, ultimately limiting the generator’s ability to produce spatially coherent and physically plausible motion.

To address these limitations, we propose a new paradigm of implicit 3D reasoning for motion control that leverages the video generator’s intrinsic spatial and motion understanding. We argue two key principles. First, we advocate for end-to-end learning of a motion encoder jointly with the generator, extracting implicit 3D motion representations directly from 2D frames while naturally aligning with the model’s spatial priors, which requires encoder designs that encourage view-agnostic motion discovery with semantic conditioning rather than rigid projection. Second, effective 3D awareness demands supervision beyond conventional same-view reconstruction, which merely learns 2D projection patterns. Instead, view-rich data across diverse viewpoints and camera trajectories forces the extraction of the essential 3D spatial motion. Unlike works such as Uni3C [4], which emphasize precise camera trajectory control in animation scenarios, our work focuses on modeling and reproducing 3D motion from 2D observations. In our framework, camera control is not an explicit objective but a natural byproduct of the model’s learned 3D awareness. We therefore leverage the generator’s native text-driven camera control—rather than predefined camera parameters—as it both aligns with the model’s intrinsic spatial understanding and serves as evidence of whether genuine 3D awareness has emerged in the learned motion representations.

Building on this paradigm, we present 3DiMo, an endto-end framework to learn 3D-aware implicit motion control for view-adaptive human video generation under viewrich supervision. Specifically, we design a Transformerbased motion encoder that distills 2D driving frames into compact 1D tokens, intentionally discarding spatial layout

to encourage viewpoint-agnostic, semantic motion abstraction. The encoder is jointly optimized with a pretrained video generator to align with the generator’s generative capability. The resulting motion tokens are injected through cross-attention, enabling flexible semantic conditioning in place of rigid projection-based alignment. To achieve genuine 3D awareness, we collect and train on a comprehensive view-rich dataset spanning single-view, multi-view, and moving-camera videos. Each clip produces motion cues that condition the generator, which is supervised to either reconstruct the same video or reproduce the motion from alternative viewpoints or camera trajectories guided by text prompts. This dual-objective training encourages the emergence of expressive, 3D-aware motion representations.

To accelerate spatial understanding during early training, we further introduce lightweight auxiliary decoders that provide geometric supervision by aligning motion features with parametric 3D reconstruction results (i.e., SMPL and MANO [25]). Although these external estimates are imperfect, they supply 3D human priors that offer a reliable initialization. As training progresses, the auxiliary loss is gradually annealed to zero, allowing the model to shift from externally guided geometry to the generator’s inherent 3D priors and the richness of view-rich data—ultimately enabling expressive and truly 3D-aware motion representations.

Our contributions can be summarized as:

- • 3D-aware motion control. We reformulate human motion control for video generation as a 3D-aware task that recovers underlying 3D motion from 2D frames while naturally supporting flexible text-driven camera control.
- • End-to-end implicit motion framework. We propose 3DiMo, an end-to-end framework that jointly learns a view-agnostic implicit motion encoder with a powerful pretrained DiT-based video generator. This design encourages motion representations that align with the generator’s intrinsic 3D spatial priors and enables semantically rich motion conditioning via cross-attention.
- • View-rich supervision for 3D learning. We collect a large-scale human motion dataset spanning singleview, multi-view, and moving-camera videos, enabling viewpoint-agnostic 3D motion learning aligned with the generator’s inherent 3D reasoning. The collected subset will be released to support future research.

Through extensive experiments, we demonstrate that 3DiMo faithfully reproduces driving motions while preserving 3D consistency across varying viewpoints, validating that the learned motion representations are both expressive and 3D-aware, effectively conditioning the DiT-based video model to generate high-fidelity human motion videos.

### 2. Related Work

Diffusion-Based Video Generation. Diffusion models have achieved remarkable success in high-fidelity image

and video synthesis. Latent Diffusion Models (LDMs) [3, 24] improve efficiency by operating in compressed latent spaces, while DiT-based architectures [22] further enhance scalability and spatiotemporal consistency for video generation. Recent advances [12, 13, 19, 30] show that large-scale pretrained video diffusion models exhibit strong awareness and reasoning capabilities over both dynamics and 3D space [29, 32]. In this work, we focus on learning an implicit 3D-aware motion representation that aligns with a pretrained video generator’s spatial and motion priorsthereby eliciting their intrinsic 3D understanding and enabling high-quality, spatially consistent human animation.

Motion Control for Human Image Animation. Human image animation aims to animate a reference image according to motion cues from a driving video, as pioneered by early works such as FOMM [26] and MRAA [27]. Recent diffusion-based approaches [9, 35] achieve impressive quality by injecting explicit motion signals (e.g., 2D poses or DensePose), but their 2D formulations inherently lose spatial information, causing depth ambiguities. To overcome this, 3D-based methods [4, 5, 8, 17, 37] introduce SMPL [16] or SMPL-X [21] models as control conditions—typically rendered or projected into 2D space, or mapped as camera-space joint trajectories for explicit 3D control. However, such approaches rely heavily on externally reconstructed representations, which, though structurally stable, lack the expressiveness and 3D reasoning priors present in large-scale pretrained video generators. In contrast, we learn an implicit, end-to-end motion representation aligned with these priors to achieve expressive and 3D-aware motion modeling. While X-Nemo [36] and X-UniMotion [28] explore implicit motion representations, they remain limited to 2D spatial patterns and cannot generalize to true 3D motion or camera control—challenges that our work directly addresses.

### 3. Our Approach

Given a reference image IR of a subject and a driving video VD = {IDt }Tt=0 providing motion cues, our proposed 3Daware motion control aims to transfer the driving video’s motion—which inherently exists in 3D space—to the reference subject, while preserving flexible, text-guided camera control. This task is highly challenging as human motion and camera trajectories are entangled within the 2D projections of driving frames, obscuring the underlying 3D motion essence that truly exist in physical space.

To achieve this goal, we leverage a pretrained DiT-based video generation model with rich 3D spatial and motion priors as our backbone, which generates videos from a reference image guided by text prompts (Sec. 3.1). The core of our framework lies in an implicit motion encoder jointly optimized with the pretrained video generator, which distills view-agnostic motion tokens from 2D driving frames

[Figure 91]

[Figure 92]

Training Framework

Training Under View-Rich Supervision

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

|[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>…<br><br>"%|
|---|

Supervision with Identical 3D Motion

[Figure 106]

###### ℒ

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

""#"

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Motion Encoder

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

"#$# #&

“Left side view of the woman. The camera arcs left to the back.”

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

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

“Front view. The camera remains static."

[Figure 143]

Same-View Reconstruction

[Figure 144]

+

[Figure 145]

[Figure 146]

…

VLM

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

###### DiTBlocks

[Figure 159]

[Figure 160]

###### 3DiMo

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

###### +

#&

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

“Front-right view. The camera arcs right."

[Figure 179]

Cross-View Reproduction

…

+

[Figure 180]

[Figure 181]

!!

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

Encoder

[Figure 192]

VAE

[Figure 193]

Inference

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

"% #&

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

…

[Figure 206]

[$#;$$]

[Figure 207]

[Figure 208]

!!

[Figure 209]

[Figure 210]

cross-attention

[Figure 211]

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

[Figure 224]

Motion Encoder

[Figure 225]

[Figure 226]

“Front view. The camera pulls away."

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

"$

[Figure 238]

ℰ#

[Figure 239]

[Figure 240]

|[Figure 241]|
|---|

[Figure 242]

[Figure 243]

[Figure 244]

augmentations

[Figure 245]

3DiMo

!! !"

[Figure 246]

[Figure 247]

|[Figure 248]<br><br>[Figure 249]|
|---|

[Figure 250]

|[Figure 251]|
|---|

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

|[Figure 263]|[Figure 264]|
|---|---|
| | |

[Figure 265]

[Figure 266]

[Figure 267]

ℰ"

…

|[Figure 268]<br><br>[Figure 269]|
|---|

[Figure 270]

[Figure 271]

Motion Encoder

Figure 2. Overview of 3DiMo. Our framework consists of end-to-end trained motion encoders—Eb for the body and Eh for hands—and an DiT-based video generator. Given a reference frame IR and a driving video VD, driving frames are first augmented with random perspective transformations before being encoded by the motion encoder to extract view-agnostic motion representations. These resulting features are then injected into the generator through cross-attention, enabling the model to synthesize a target sequence Vtgt that reenacts the same underlying 3D motion while preserving flexible text-driven camera control. To facilitate 3D-aware learning, we introduce earlystage auxiliary geometric supervision by regressing the encoded motion to external parametric reconstruction results θb and θh. During training, view-rich data is used to jointly supervise same-view reconstruction and cross-view motion reproduction, driving the emergence of expressive and 3D-aware motion representations. At inference, motion tokens extracted directly from 2D driving frames provide rich 3D spatial cues that can animate any reference character, supporting high-fidelity and view-adaptive motion-controlled video generation.

and injects them via cross-attention for semantical motion control compatible with text-driven camera manipulation (Sec. 3.2). To endow the learned motion representation with 3D awareness, we train our framework on a view-rich human video dataset encompassing diverse viewpoints and camera movements, supplemented with auxiliary decoders that provide early-stage geometric alignment supervision to accelerate spatial understanding (Sec. 3.3).

sion process [14, 15] that progressively adds noise to the target video latents, and the model is optimized using a vprediction objective.

Parametric 3D Human Model. SMPL [16] represents a full-body human mesh using a compact set of parameters, including shape coefficients βb and pose parameters θb that describe the articulated body configuration. Similarly, MANO [25] models the articulated hand using an analogous formulation, with hand-shape parameters βh and hand-pose parameters θh. Despite their well-known limitations in expressiveness and depth ambiguity, these parametric models provide robust 3D geometric priors that we leverage for early-stage auxiliary supervision.

#### 3.1. Preliminary

Video Generation Backbone. Our video generation model adopts the latent diffusion model (LDM) paradigm, utilizing a causal 3D VAE for video compression into latent space and a generative backbone for latent sequence modeling. The backbone is a DiT-based architecture [6] pretrained on text-to-video and image-to-video tasks, comprising multiple DiT blocks that interleave full self-attention and Feed-Forward Networks (FFN). The reference image is incorporated by concatenating its latent tokens with noised video tokens, facilitating cross-modal interaction among reference, video, and text tokens through the full selfattention. During training, we adopt a flow-based diffu-

#### 3.2. End-to-end Framework with Implicit ViewAgnostic Motion Control

As illustrated in Fig. 2, our framework features a motion encoders that extracts motion representations z from the driving video VD, which condition a pretrained DiT-based video generator. The reference image IR and accompanying text prompt T are also provided as token sequences to the video generator, which produces an output video depict-

[Figure 272]

[Figure 273]

[Figure 274]

ing the reference subject in IR reenacting the motion from VD under camera trajectories guided by T. Unlike previous methods that rely on external pose estimation, our framework jointly optimizes the motion encoder and the video generator in an end-to-end manner, learning semantically rich motion representations naturally aligned with the generator’s inherent spatial and motion priors.

[Figure 275]

###### Data Statistics

###### Data Collection Categorized by Camera

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Captured 100K UE-Rendered 60K

[Figure 280]

[Figure 281]

Single-View Videos

Internet

Internet 600K

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Implicit Motion Encoder. The core insight of our motion encoder design is to encourage view-agnostic representation learning that captures the semantically rich dynamics of 3D human motion, going beyond the superficial patterns observable in 2D projections. Following [33], we design our motion encoder as a Transformer-based 1D tokenizer. Each driving frame is patchified into visual tokens and concatenated with K(= 5 in our work) learnable latent tokens, which interact through several attention layers. Only the output latent tokens are retained as the motion representation. By compressing into compact 1D motion tokens, we enforce a semantic bottleneck that eliminates 2D structural information, including both appearance details and viewspecific pose configurations, while focusing on the intrinsic semantics of spatial motion.

[Figure 286]

Camera-Motion 80K Multi-View 80K

UE-Rendered

[Figure 287]

Multi-View Videos

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Single-View 600K

In-House Captured

Camera-Motion Videos

Figure 3. Our collected view-rich dataset combines internet videos, UE renderings, and in-house captures, covering camera categories including single-view, multi-view, and camera-motion sequences. High-quality large-scale single-view footage exposes the model to diverse human motions, while complementary multiview data provides consistent cross-view observations that are crucial for learning genuine 3D-aware motion representations.

To encourage view-agnostic motion representation learning, we apply random perspective transformations to the driving frames before motion encoding to introduce motioninvariant augmentations, which to some extent decouples the spatial motion from its view-specific 2D projection. Additionally, similar to [28, 36], we employ appearance augmentations (e.g., color jittering and lightweight spatial transforms) to prevent identity leakage from the driving frames. Through this design, the motion encoder is encouraged to focus purely on the intrinsic dynamics of 3D spatial motion while avoiding both appearance leakage and viewspecific pose constraints.

DiT-based generator.

Dual-Scale Motion Encoding. Considering that a single compact representation struggles to capture both global body movements and fine-grained hand dynamics, we follow [28] and employ two motion encoders: a body encoder Eb for coarse body motion and a hand encoder Eh for detailed gestures. The resulting motion tokens are concatenated and jointly injected into the generator via crossattention, enabling unified motion control over both fullbody articulation and fine-grained hand movements.

#### 3.3. 3D-Aware Training with View-Rich Data

View-Agnostic Cross Attention Conditioning. Instead of converting our motion representations into view-dependent

While our motion encoder with distilled 1D tokens effectively filters redundant 2D spatial structures and captures semantically rich motion dynamics for expressive motion generation, this design alone does not guarantee genuine 3D motion understanding. When trained solely through same-view reconstruction, the model can achieve satisfactory results by merely learning view-dependent 2D motion patterns, as reproducing motion under identical viewpoints requires no true spatial reasoning. This reveals a fundamental limitation: without more challenging supervision, the model lacks incentive to develop 3D awareness beyond the 2D projection domain. To overcome this issue, we introduce view-rich data supervision that imposes a more demanding learning objective, compelling the model to reason about motion as it truly occurs in 3D space, invariant to viewpoint changes rather than as isolated 2D observations.

- 2D spatially-aligned control using explicit camera parameters as in most existing works, we simply employ crossattention to inject motion representations directly into the generator. This achieves flexible semantic-level interaction between visual and motion modalities without rigid spatial constraints. To be specific, we append a cross-attention layer after each full self-attention in the DiT generator, where only video tokens attend to the motion tokens, while text tokens remain unchanged. Text-Guided Camera Control. Our view-agnostic motion representations and semantic-level conditioning naturally coexist with the generator’s native text-driven camera control capability. Consequently, beyond motion control, our framework readily supports flexible viewpoint manipulation by simply augmenting the text prompt T with camera-movement descriptions, which interact with visual tokens through the same native mechanism as in the original

View-Rich Dataset Construction. To enable comprehensive supervision for learning both expressive and 3D-aware

motion representations, we construct a large-scale dataset with diverse camera configurations. From the perspective of supervision objectives, our data serves three distinct purposes as shown in Fig. 3: 1) same-view reconstruction, where each motion-viewpoint pair is unique, enabling self-supervised learning of expressive motion dynamics; 2) multi-view motion reproduction, utilizing synchronized captures from fixed camera arrays for identical motions to enforce consist 3D motion learning across viewpoints; and 3) motion reproduction under moving cameras, featuring identical motions captured with different camera trajectories to decouple motion from viewpoint changes and support text-guided camera control.

To balance realism, diversity, and 3D consistency, we integrate three complementary data sources: 1) large-scale internet videos that provide diverse human motions for learning realistic dynamics, though limited to single-view supervision; 2) synthetic sequences rendered with Unreal Engine 5 (UE5) from [1, 18, 31], which offer precise motion captures under varied camera trajectories, despite potential domain gaps from real-world videos; 3) real-world multi-view captures, including both open-source datasets and our proprietary recordings, combining fixed multi-camera arrays and dynamic camera trajectories, which provide authentic

- 3D supervision within the real-world video domain. The detailed composition of our dataset is illustrated in Fig. 3. In practice, internet videos dominate in scale and drive the learning of natural, expressive motion patterns, whereas the multi-view and camera-trajectory data, though smaller in quantity, play a crucial role in fostering genuine 3D spatial understanding.

To obtain text descriptions for camera viewpoints and movements, we employ Qwen2.5-VL [2] to both annotate internet videos and convert predefined camera configurations from synthetic and real-world captured data into unified text prompts.

Training Strategies. With the constructed dataset, we enable 3D-aware training under view-rich supervision that includes both reconstruction and cross-view motion reproduction objectives. Specifically, given a driving video VD, we supervise the model output with either VD itself (reconstruction) or corresponding videos VD′ of the same motion captured from a different viewpoint or camera trajectory (cross-view reproduction). The reference image is taken as the first frame of the supervision target, which automatically aligns the generated motion with the reference subject’s facing direction—eliminating the need for explicit SMPL-toimage alignment or camera regression as in [4].

In practice, we adopt a progressive multi-stage training strategy with varying data mixtures based on supervision objectives. In the first stage, we exclusively use singleview data for self-reconstruction, exposing the model to diverse and expressive motion dynamics and enabling sta-

ble initialization of implicit motion learning. The second stage introduces a balanced mixture of reconstruction and cross-view motion reproduction, gradually transitioning the learned representations from 2D dynamics toward 3D spatial semantics. Finally, the third stage focuses entirely on multi-view and camera-motion data to strengthen the viewagnostic nature of the learned motion features and enhance compatibility with flexible text-guided camera control.

Geometric Supervision with Auxiliary Decoders. In our early experiments, we observed that direct end-to-end training often leads to slow and unstable convergence, especially after introducing cross-view supervision. This partly arises because the diffusion loss distributes uniformly across pixels, lacking targeted emphasis on motion-specific semantics. Moreover, the powerful DiT backbone tends to exploit its inherent motion priors to generate plausible videos from single images during early training, thereby reducing reliance on the encoded motion representations, which consequently receive weak gradient feedback.

To address this challenge, we introduce auxiliary geometric supervision to facilitate motion representation learning. Specifically, we employ a lightweight MLP-based geometric decoder Dg that processes the concatenated motion representations z = [zb;zh] to predict pose parameters θ = [θb;θh], using pseudo ground-truth annotations derived from off-the-shelf SMPL and MANO estimators [20, 23]. Notably, we exclude the global root orientation during supervision to ensure view-agnostic learning.

Despite limitations in depth ambiguity and expressiveness, this parametric 3D geometric supervision effectively transfers robust spatial motion priors through the lightweight, easily optimized auxiliary decoder, providing a well-initialized motion distribution for subsequent learning. In practice, we apply auxiliary supervision during the first stage and the early part of the second stage, with its loss weight annealed progressively as training proceeds. The supervision is then completely removed for the remaining steps of the second stage and the entirety of the third stage. This schedule allows the model to evolve from geometryguided initialization to learning motion representations that align with the DiT’s perceptual and generative capabilities, ultimately achieving superior 3D-aware motion understanding supported by view-rich supervision.

### 4. Experiments

#### 4.1. Experimental Setups

Implementation Details. We train our 3DiMo on the dataset described in Sec. 3.3, using 121-frame video clips resized to a target area of 480 × 854 pixels while preserving original aspect ratios. Training is performed with a total batch size of 64 using the Adam optimizer with a learning rate of 1e-5. The three training stages run for 10K, 15K,

Table 1. Quantitative evaluation and user study results of MOS with 95% confidence intervals. Top two are noted as first , second .

Quantitative Evaluation User Study Method

SSIM ↑ PSNR ↑ LPIPS ↓ FID ↓ FVD ↓ Accuracy ↑ Naturalness ↑ 3D Plausibility ↑ Overall ↑

AnimateAnyone 0.7325 17.21 0.2754 68.72 862.5 4.13±0.12 4.00±0.09 3.20±0.14 3.76±0.10 MimicMotion 0.7051 16.83 0.3286 62.45 628.2 3.84±0.14 4.15±0.06 3.01±0.12 3.84±0.10 MTVCrafter 0.7489 18.03 0.2542 57.21 379.6 3.69±0.09 3.98±0.10 3.69±0.12 4.19±0.09 Uni3C 0.7185 17.53 0.2639 41.28 321.9 3.72±0.06 4.03±0.06 3.97±0.10 4.24±0.06 Ours 0.7390 17.96 0.2206 36.92 297.4 4.28±0.08 4.18±0.06 4.05±0.09 4.38±0.08

[Figure 293]

and 5K steps, completing in approximately three days.

Reference Driving A.A. MTVCrafter Uni3C Ours

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

Evaluation Data and Metrics. We evaluate our method on 50 videos from TikTok dataset [10] and 100 videos collected from the internet. Following [9], we use PSNR, SSIM, LPIPS, and FID to measure per-frame visual quality, and adopt FVD to evaluate the overall video fidelity.

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

#### 4.2. Quantitative Evaluation

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

We compare 3DiMo with state-of-the-art human image animation methods, including both 2D pose-based approaches (AnimateAnyone [9], MimicMotion [35]) and 3D SMPLbased methods (Uni3C [4], MTVCrafter [5]), to validate the effectiveness of our implicit 3D-aware motion modeling and reenactment. Considering that most existing baselines do not support camera manipulation, we evaluate 3DiMo under prompts specifying a static camera.

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Figure 4. Visualization comparisons with baselines. Red and yellow bounding boxes highlight depth ambiguities and inaccurate poses, respectively. “A.A.” denotes AnimateAnyone. Our method produces accurate and 3D-plausible motion reenactment videos.

As shown in Tab. 1 (Col. 2-6), our method surpasses all baselines on LPIPS, FID, and FVD, indicating superior visual quality and motion control. Although our SSIM and PSNR are slightly lower than MTVCrafter, this is expected: these pixel-wise metrics are sensitive to minor viewpoint deviations while some evaluation videos contain weak, unintended camera motions. Competing methods mainly operate within a 2D alignment paradigm and simply reproduce these motions, while our text-driven static camera prompt suppresses such drift to maintain geometric consistency. This yields perceptually better results but introduces small pixel-wise discrepancies from the ground truth.

ther demonstrate our model’s capability in modeling 3D motion, Figure Fig. 1 showcases several motion control results under diverse text-guided camera configurations. Our approach inherits the DiT backbone’s native ability for textdriven camera manipulation, while simultaneously achieving precise video-driven motion control that preserves physical plausibility and spatial consistency across dynamic camera trajectories and varying viewpoints.

User Study. We further conduct a user study involving 30 participants, where each participant evaluates 10 crossidentity animation videos generated by each method. We collect mean opinion scores (MOS) based on a 5-point Likert scale across four aspects: motion accuracy, motion naturalness, 3D physical plausibility, and overall visual quality. The results, summarized in Tab. 1 (Col. 7-10), show that our method consistently outperforms all existing baselines, especially in motion naturalness and physical plausibility, which emphasize spatial relationships and realistic dynamics. These strongly support that our learned motion representation, aligned with the large-scale pretrained video generator’s spatial and motion priors, achieves more expressive and 3D-aware motion modeling and control compared to

#### 4.3. Qualitative Evaluation

Comparison with SOTAs. Fig. 4 presents the quantitative comparisons, where we configure our method with a static camera as described in Sec. 4.2. Our implicit 3Daware approach achieves precise motion control and expressive dynamics, producing high-fidelity and physically plausible human videos. In contrast, 2D pose-based methods often yield incorrect limb depth ordering due to their lack of geometric awareness, while SMPL-based approaches struggle to maintain accurate pose estimation and control under complex motions. More visualization results and extended comparisons are provided in the supplementary material.

More Results of View-Adaptive Motion Control. To fur-

[Figure 319]

attention mechanism.

[Figure 320]

Reference Driving Camera arcs to the right.

###### Ablation Results

Ours

As shown in Fig. 5, for a frontal one-hand-on-hip driving motion, the SMPL-based variant fails to maintain correct hand-hip contact from the side view. In contrast, our learned motion representation correctly preserves this physical relationship, effectively resolving the depth ambiguity that commonly occurs in parametric reconstructions. This demonstrates that our motion representation, distilled from the pretrained video generator, exhibits superior 3D spatial awareness aligned with real-world priors compared to offthe-shelf parametric reconstruction.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

| |
|---|
| |

[Figure 325]

w/ SMPL ctrl. w/ implicit ctrl.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Camera arcs to the left.

Multi-Stage View-Rich Learning. We further analyze the effect of our multi-stage training strategy under a leftward arcing camera trajectory. In the first stage with single-view reconstruction, the model learns diverse motion patterns but tends to collapse into 2D projections, often failing to follow text-guided camera motions. Introducing view-rich data in the second stage helps decouple motion and viewpoint, establishing initial 3D awareness; however, the camera motion sometimes only affects the background while the subject remains front-facing, indicating incomplete semantic interaction between motion and camera control. The third-stage refinement, trained solely on view-rich data, further strengthens this interaction, enabling the model to accurately follow camera trajectories while maintaining spatially consistent human motion and background rendering.

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

w/ stage 1 only w/ stage 1 & 2 full stages

Camera remains static.

[Figure 335]

[Figure 336]

w/o geo. superv. w/ channel concat.

full model

[Figure 337]

Camera remains static.

[Figure 338]

[Figure 339]

[Figure 340]

| |
|---|

w/o hand enc. w/ hand enc.

Figure 5. Visualizations of ablation results. Using SMPL poses as motion representation introduces typical depth ambiguity errors. Removing any view-rich data supervision impairs camera control. Removing auxiliary geometric supervision or using channel concatenation causes training instability and quality degradation. Without the hand encoder, fine-grained hand motions are lost.

Conditioning Mechanism. We replace cross-attention with channel concatenation as an alternative conditioning approach. This substitution shows significantly degraded motion control capability, demonstrating that crossattention is more suitable for semantic-rich interaction between our motion representations and the generator.

Auxiliary Geometric Supervision. Removing the auxiliary geometric supervision during early training leads to unstable convergence and collapsed motion control, demonstrating that this supervision provides crucial initialization for learning meaningful motion representations.

Table 2. Ablation results. Top two are noted as first , second .

Method SSIM ↑ PSNR ↑ LPIPS ↓ FID ↓ FVD ↓

w/ SMPL ctrl. 0.724 17.1 0.238 39.7 348.2 w/ stage 1 only 0.745 18.3 0.220 40.5 305.4 w/ stage 1 & 2 0.723 17.9 0.221 38.2 314.5 w/ channel concat. 0.703 16.8 0.304 48.2 395.6 w/o geo. superv. 0.684 15.8 0.347 51.3 383.1 w/o hand enc. 0.726 17.5 0.234 38.1 298.7 Full Model 0.739 18.0 0.221 36.9 297.4

Dual-scale motion encoders. Omitting the hand motion encoder results in loss of fine-grained hand control, confirming its necessity for complete motion representation.

Tab. 2 further shows that removing most components consistently degrades visual quality, confirming their necessity for producing high-fidelity human motion videos. Discarding the last two view-rich training stages yields a slight but negligible improvement in visual metrics; however, these stages are essential for endowing the model with genuine 3D-aware motion understanding—precisely the core capability required to address our proposed 3Daware motion control problem.

approaches that rely on externally predefined parameters.

#### 4.4. Ablation Study and Analysis

SMPL vs. Our Implicit Motion Representations. The core of our approach lies in the end-to-end learning of an implicit 3D-aware motion representation. For comparison, we also consider a variant that directly uses SMPL pose coefficients θbody as the motion representation, mapped through an MLP to match the token dimensionality before being injected into the generator via the same cross-

### 5. Conclusion

In this work, we present 3DiMo, an end-to-end framework for 3D-aware human motion control that learns to align with

a video generator’s intrinsic spatial priors rather than relying on explicit 3D parametric reconstruction. By jointly training a motion encoder with the pretrained video generation model and designing it to discard view-dependent layouts, 3DiMo aligns the two models and establishes the capacity for view-agnostic motion representation. When further trained under view-rich data supervision, the framework internalizes 3D spatial motion understanding directly from 2D observations. Combined with lightweight geometric initialization that is gradually annealed away, 3DiMo ultimately develops robust and expressive 3D-aware motion representations without relying on external estimates. Experiments demonstrate that 3DiMo faithfully reproduces driving motions under flexible text-driven camera control and consistently outperforms both 2D- and 3D-based baselines in motion fidelity and visual quality.

### References

- [1] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 6
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023. 3
- [4] Chenjie Cao, Jingkai Zhou, Shikai Li, Jingyun Liang, Chaohui Yu, Fan Wang, Xiangyang Xue, and Yanwei Fu. Uni3c: Unifying precisely 3d-enhanced camera and human motion controls for video generation. arXiv preprint arXiv:2504.14899, 2025. 2, 3, 6, 7
- [5] Yanbo Ding, Xirui Hu, Zhizhi Guo, Chi Zhang, and Yali Wang. Mtvcrafter: 4d motion tokenization for open-world human image animation. arXiv preprint arXiv:2505.10238,

2025. 2, 3, 7

- [6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 4

- [7] Xu He, Qiaochu Huang, Zhensong Zhang, Zhiwei Lin, Zhiyong Wu, Sicheng Yang, Minglei Li, Zhiyi Chen, Songcen Xu, and Xiaofei Wu. Co-speech gesture video generation via motion-decoupled diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2263–2273, 2024. 2
- [8] Xu He, Zhiyong Wu, Xiaoyu Li, Di Kang, Chaopeng Zhang, Jiangnan Ye, Liyang Chen, Xiangjun Gao, Han Zhang, and

- Haolin Zhuang. Magicman: Generative novel view synthesis of humans with 3d-aware diffusion and iterative refinement. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3437–3445, 2025. 2, 3
- [9] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 2, 3, 7
- [10] Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12753– 12762, 2021. 7
- [11] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [12] Kuaishou. Kling ai. https://klingai.kuaishou. com/, 2024. Accessed: 2025-05-19. 3
- [13] Xuanyi Li, Daquan Zhou, Chenxu Zhang, Shaodong Wei, Qibin Hou, and Ming-Ming Cheng. Sora generates videos with stunning geometrical consistency. arXiv preprint arXiv:2402.17403, 2024. 2, 3
- [14] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [15] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [16] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 2, 3, 4
- [17] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, and Tianshu Hu. Dreamactor-m1: Holistic, expressive and robust human image animation with hybrid guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11036–11046, 2025. 3
- [18] Yawen Luo, Xiaoyu Shi, Jianhong Bai, Menghan Xia, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Camclonemaster: Enabling reference-based camera control for video generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–10, 2025. 6
- [19] Midjourney. Midjourney. https://www.midjourney. com, 2024. Accessed: 2024. 3
- [20] Priyanka Patel and Michael J Black. Camerahmr: Aligning people with perspective. In 2025 International Conference on 3D Vision (3DV), pages 1562–1571. IEEE, 2025. 6
- [21] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pages 10975–10985, 2019. 2, 3

- [22] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 3

- [23] Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. Wilor: End-to-end 3d hand localization and reconstruction in-the-wild. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12242–12254, 2025. 6
- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [25] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 36(6), 2017. 3, 4
- [26] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019. 3
- [27] Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13653–13662, 2021. 3
- [28] Guoxian Song, Hongyi Xu, Xiaochen Zhao, You Xie, Tianpei Gu, Zenan Li, Chenxu Zhang, and Linjie Luo. Xunimotion: Animating human images with expressive, unified and identity-agnostic motion latents. arXiv preprint arXiv:2508.09383, 2025. 3, 5
- [29] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025. 2, 3
- [30] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3
- [31] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 6
- [32] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2, 3, 1
- [33] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances

- in Neural Information Processing Systems, 37:128940– 128966, 2024. 5
- [34] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2
- [35] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024. 2, 3, 7
- [36] Xiaochen Zhao, Hongyi Xu, Guoxian Song, You Xie, Chenxu Zhang, Xiu Li, Linjie Luo, Jinli Suo, and Yebin Liu. X-nemo: Expressive neural motion reenactment via disentangled latent attention. arXiv preprint arXiv:2507.23143,

2025. 3, 5

- [37] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision, pages 145–162. Springer, 2024. 2, 3

## 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation

### Supplementary Material

In this supplementary document, we provide additional details to support the main paper, organized as follows:

- • Section A: Ethical considerations regarding human video generation and our commitment to responsible research.
- • Section B: Demonstrations of broader applications, including single-image novel view synthesis, video stabilization, and automatic motion alignment.
- • Section C: Detailed specifications of our in-house data collection pipeline and the definition of camera trajectories.
- • Section D: A discussion on current limitations and potential directions for future work.
- • Section E: Technical implementation details.

### A. Ethical Considerations

The rapid progress in AI-driven human video generation offers significant potential for digital entertainment, virtual reality, and human-centric research. However, the ability to synthesize highly realistic human content also brings forth important ethical considerations, such as the potential for privacy violations, intellectual property concerns, and the risk of creating deceptive “deepfake” media.

As with many generative technologies, there is a possibility that these methods could be misused to create content without the consent of the individuals involved. Addressing these risks requires the collective development of ethical guidelines and legal frameworks. In this work, we are committed to responsible research practices. All processed data, models, and results are intended strictly for academic purposes and are not authorized for commercial use or the creation of harmful content. We believe that by adhering to these principles, the proper application of such techniques will continue to positively enhance research in computer graphics and artificial intelligence.

### B. Broader Applications

Benefiting from the implicit 3D motion reasoning and flexible text-driven camera control of 3DiMo, our approach generalizes effectively to specific downstream tasks, as shown in Fig. S1.

Human Novel View Synthesis From a Single Image. While traditional novel view synthesis (NVS) typically requires reconstructing 3D scenes from reference images to render new angles, 3DiMo achieves human-specific singleimage NVS through a straightforward inference strategy. We construct a driving video by repeating the reference frame (implying zero motion) and pair it with text prompts describing camera trajectories (e.g., “camera rotates in

a circular path around the woman”). Although pretrained I2V foundation models theoretically support this via prompts specifying camera movement alongside “static subject,” they suffer from significant limitations in practice. As noted by [32], these models tend to hallucinate motion, failing to keep the subject strictly stationary. Moreover, we observe that base I2V models often confuse camera control with background animation rather than performing true geometric view synthesis. By leveraging our view-agnostic motion representation and the model’s improved 3D awareness, 3DiMo overcomes these ambiguities to produce consistent novel-view generations.

Video Stabilization. Capturing stable footage during dynamic recording is often challenging. Video stabilization aims to smooth out camera jitters to obtain high-quality, steady sequences. In human-centric scenarios, 3DiMo effectively performs this task. By utilizing the first frame of the shaky video as the reference image and the full video as the driving signal, we can feed the model a prompt such as “camera remains static.” This instructs the generator to reconstruct the underlying human motion from a fixed viewpoint, effectively eliminating the original camera shake while preserving the subject’s dynamics.

Automatic Motion-Image Alignment. Conventional motion transfer methods, particularly 2D-based approaches, rigidly impose the absolute orientation of the driving video onto the reference subject. This often leads to unnatural transitions when the driving and reference subjects have different initial facing directions (e.g., a side-view driver controlling a front-view reference). In contrast, because 3DiMo extracts a view-agnostic implicit motion representation, it naturally aligns the driving motion with the reference subject’s initial orientation. Our model transfers the relative 3D dynamics rather than the absolute 2D projection, eliminating the need for manual camera calibration or explicit root-rotation alignment required by SMPL-based methods.

### C. In-House Data Acquisition Setup

Our in-house data capture involves a three-camera array positioned at diverse angles relative to the subject. For every captured performance, each camera is assigned a camera motion type sampled randomly from the following categories:

- • Static Variants: Static, Handheld Static.
- • Linear Translations: Move Forward, Move Back, Move Left, Move Right, Move Up, Move Down.
- • Zoom Actions: Zoom In, Zoom Out, Rapid Zoom In, Rapid Zoom Out, Handheld Zoom In, Aerial Pull-out.
- • Complex Trajectories: Vertigo In, Vertigo Out, Dy-

(a) Novel View Synthesis From a Single Image (b) Video Stabilization

[Figure 341]

JitteryDrivingVideoStabilizedOutput

Reference Image Output Novel View Output Novel View

[Figure 342]

(c) Automatic Motion-Appearance Alignment

[Figure 343]

Reference Driving Frame Output Frame Driving Frame Output Frame

Figure S1. Broader applications of 3DiMo. We demonstrate the versatility of our framework on three downstream tasks: (a) singleimage novel view synthesis by enforcing static motion; (b) video stabilization by suppressing camera jitter from the driving video; and (c) automatic motion-appearance alignment without explicit calibration.

### E. Implementation Details

namic Zoom Swing, Arc Left (variable angles, e.g., 30◦,45◦), and Arc Right (variable angles, e.g., 30◦,45◦).

We train our 3DiMo using 121-frame video clips resized to a target area of 480 × 854 pixels while preserving original aspect ratios. Training is performed with a total batch size of 64 using the Adam optimizer with a learning rate of 1e-5. The three training stages run for 10K, 15K, and 5K steps, completing in approximately three days. During training, the weight of the auxiliary geometric supervision is linearly annealed from 0.1 to 0 over the first 12K steps.

By pairing identical human motions with diverse, noncorrelated camera trajectories across three views, we maximize the supervision signal for view-agnostic motion learning.

### D. Limitations and Future Work

Despite the significant advancements 3DiMo achieves in view-adaptive human video generation, several limitations remain to be addressed in future research.

Resolution and Fine-Grained Details. Currently, our framework operates at a resolution of 480p. While this is sufficient for capturing global motion dynamics, it imposes a bottleneck on high-frequency details. Specifically, in full-body shots where the subject occupies a relatively small proportion of the frame, the limited pixel budget can lead to artifacts, such as blurred facial features or a lack of texture in hand details. Future iterations could address this by scaling up the framework to higher-resolution DiT backbones (e.g., 720p or 1080p) or incorporating cascaded super-resolution modules to enhance local details in smallscale regions.

Complex Human-Object Interactions. Since our motion encoders are explicitly designed to distill human body and hand dynamics, the current framework does not explicitly model the motion of external objects or props (e.g., a person holding a bag or riding a bicycle). Consequently, while the human motion is faithfully reproduced, the interaction with held objects may sometimes be hallucinated. Extending the implicit motion encoding mechanism to handle general dynamic objects or human-scene interactions represents a promising direction for future work.

