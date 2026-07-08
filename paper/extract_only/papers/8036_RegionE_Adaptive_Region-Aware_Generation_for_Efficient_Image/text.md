# arXiv:2510.25590v1[cs.CV]29Oct2025

[Figure 1]

## REGIONE: ADAPTIVE REGION-AWARE GENERATION FOR EFFICIENT IMAGE EDITING

###### Pengtao Chen1 Xianfang Zeng2‡ Maosen Zhao1 Mingzhu Shen3 Peng Ye1 Bangyin Xiang1 Zhibo Wang2 Wei Cheng2 Gang Yu2 Tao Chen1∗ 1 Fudan University 2 StepFun 3 Imperial College London Code: https://github.com/Peyton-Chen/RegionE

ABSTRACT

Recently, instruction-based image editing (IIE) has received widespread attention. In practice, IIE often modifies only specific regions of an image, while the remaining areas largely remain unchanged. Although these two types of regions differ significantly in generation difficulty and computational redundancy, existing IIE models do not account for this distinction, instead applying a uniform generation process across the entire image. This motivates us to propose RegionE, an adaptive, region-aware generation framework that accelerates IIE tasks without additional training. Specifically, the RegionE framework consists of three main components: 1) Adaptive Region Partition. We observed that the trajectory of unedited regions is straight, allowing for multi-step denoised predictions to be inferred in a single step. Therefore, in the early denoising stages, we partition the image into edited and unedited regions based on the difference between the final estimated result and the reference image. 2) Region-Aware Generation. After distinguishing the regions, we replace multi-step denoising with one-step prediction for unedited areas. For edited regions, the trajectory is curved, requiring local iterative denoising. To improve the efficiency and quality of local iterative generation, we propose the Region-Instruction KV Cache, which reduces computational cost while incorporating global information. 3) Adaptive Velocity Decay Cache. Observing that adjacent timesteps in edited regions exhibit strong velocity similarity, we further propose an adaptive velocity decay cache to accelerate the local denoising process. We applied RegionE to state-of-the-art IIE base models, including Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit. RegionE achieved acceleration factors of 2.57×, 2.41×, and 2.06×, respectively, with minimal quality loss (PSNR: 30.520–32.133). Evaluations by GPT-4o also confirmed that semantic and perceptual fidelity were well preserved.

1 INTRODUCTION

In recent years, diffusion models (Rombach et al., 2022) have achieved rapid progress in generative tasks, particularly in visual generation, where state-of-the-art models can synthesize highly realistic images. Within this context, the task of editing existing images according to user requirements has gradually emerged as an important direction (Kawar et al., 2023). Recently, diffusion-based foundation models, such as FLUX.1 Kontext (Labs et al., 2025), Qwen-Image-Edit (Wu et al., 2025), and Step1X-Edit (Liu et al., 2025d), have been developed. These models can perform precise image editing using only textual instructions, offering a novel solution for instruction-based image editing and providing more powerful tools for image post-processing (Choi et al., 2024).

Although diffusion-based IIE models can achieve impressive editing results, their high inference latency limits their use in real-time applications. Previous research on efficient diffusion inference has primarily focused on image generation. For instance, some studies reduce model parameters through pruning (Rombach et al., 2022; Castells et al., 2024), others decrease model bit-width via quantization (Shang et al., 2023; Zhao et al., 2025), and some employ distillation to reduce model size (Kim et al., 2023) and the number of timesteps (Sauer et al., 2024). In the two-stage

∗Corresponding author. ‡Project leader. Work was done when interned at StepFun.

Noise 𝒕 Data

Noise 𝒕 Data

###### Prompt: Add a hat to the cat.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

|[Figure 8]|
|---|

|[Figure 9]<br><br>Instruction Image|
|---|

[Figure 10]

onestepestimate

𝒗

real trajectory

𝒗

one step estimate

Instruction-base Editing

[Figure 11]

[Figure 12]

|[Figure 13]|
|---|

Curved

Straight

|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>Unedited Region<br><br>Edited|
|---|

Trajectory of the Unedited Region is Straight

Trajectory of the Edited Region is Curved

- Figure 1: Trajectories of different regions in the IIE task. In unedited regions, the trajectory is nearly linear, allowing early-stage velocity to provide a reliable estimate of the multi-step denoised images, including the final result. In contrast, edited regions exhibit curved trajectories, making the final image harder to predict. Despite this, the velocity between consecutive timesteps remains consistent.

inversion-based editing paradigm (Pan et al., 2023; Wang et al., 2025), redundancy in the inversion and denoising stages has been analyzed, leading to methods like EEdit (Yan et al., 2025) that accelerate both stages simultaneously. However, for the emerging denoising-only paradigm of IIE, the redundancy and feasibility of efficient inference remain largely unexplored.

Our study reveals that current IIE models exhibit two significant types of redundancy: 1) Spatial Generation Redundancy. Unlike image generation tasks, which require reconstructing the entire image, IIE models often need to modify only local regions specified by the instructions, while the remaining areas remain essentially unchanged. For example, as shown in Figure 1, the model edits only the region around the hat. Nevertheless, IIE models apply the same computational effort to both edited and unedited areas, resulting in significant redundancy in the latter. 2) Redundancy across diffusion timesteps. First, at neighboring timesteps, the key and value within the attention layers at the same network depth are highly similar. Second, in the middle stages of denoising, the velocity output by the diffusion transformer (DiT) at adjacent timesteps is also highly similar.

To mitigate spatial and temporal redundancy in IIE models, this paper introduces RegionE, a trainingfree, adaptive, and region-aware generative framework that accelerates the current IIE models. Firstly, we observed that the trajectories of edited regions are often more curved, making it difficult to accurately predict the final edited results at early timesteps, as shown in Figure 1. In contrast, unedited regions follow nearly linear trajectories, allowing more reliable predictions from the same early steps. Based on this observation, RegionE introduces an Adaptive Region Partition (ARP), which performs a one-step estimation for the final image in the early stage and compares its similarity with the reference (instruction) image. Regions with high similarity (minimal change after editing) are classified as unedited, whereas regions with low similarity are classified as edited. Then, we perform region-aware generation on the two separated parts. Specifically, We replace multi-step denoising with one-step estimation for the unedited areas and apply region-iterative denoising for edited areas. During edited region generation, RegionE discards unedited region tokens and instruction image tokens, and effectively reinjects global context into local generation through our proposed RegionInstruction KV Cache (RIKVCache), which leverages the similarity of key and value across timesteps. This process primarily addresses redundancy in spatial. Finally, regarding temporal redundancy, we find that the velocity outputs of DiT at adjacent timesteps are highly consistent in direction but decay in magnitude over time, with the decay dependent on the timestep. To exploit this property, RegionE introduces an Adaptive Velocity Decay Cache (AVDCache), which accurately models this pattern and further accelerates the region generation process. Experimental results demonstrate that RegionE achieves speedups of approximately 2.57×, 2.41×, and 2.06× on Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit, respectively, while maintaining PSNR values of 30.520, 32.133, and 31.115 before and after acceleration. Evaluations using GPT-4o further indicate that the perceptual differences are negligible, confirming that RegionE effectively eliminates redundancy in IIE tasks without compromising image quality.

The contributions of our paper are as follows:

- • We observe that in IIE tasks, unedited regions exhibit nearly linear generation trajectories, allowing early-stage velocities to provide reliable estimates for multi-step denoised images, including the

- final image. In contrast, edited regions follow more curved trajectories, making the final image harder to predict. Nevertheless, the velocity remains consistent across consecutive timesteps.
- • We propose RegionE, a training-free, efficient IIE method with adaptive, region-aware generation. It reduces spatial redundancy by performing early adaptive predictions for edited and unedited regions and generating each region locally in subsequent stages, while mitigating temporal redundancy via a velocity-decay cache across timesteps.
- • RegionE achieves 2.57×, 2.41×, and 2.06× end-to-end speedups on Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit, while maintaining PSNR (30.520, 32.133, 31.115) and SSIM (0.939, 0.917, 0.937). Evaluations with GPT-4o further confirm that no quality degradation occurs.

- 2 RELATED WORK

Efficient Diffusion Model. Although few efficient methods have been developed specifically for IIE models, a variety of acceleration techniques have been proposed for diffusion models more generally. From the perspective of parameter redundancy, researchers have introduced pruning methods such as Diff-Pruning Fang et al. (2023) and LD-Pruner (Castells et al., 2024), quantization methods such as PTQ4DM (Shang et al., 2023), FPQuant (Zhao et al., 2025), and SVDQuant (Li et al., 2024a), distillation methods such as BK-SDM (Kim et al., 2023) and CLEAR (Liu et al., 2024), sparse attention methods such as DiTFastAttn (Yuan et al., 2024; Zhang et al., 2025), SVG (Xi et al., 2025), Sparse-vDiT (Chen et al., 2025), and VORTA (Sun et al., 2025), and early-stopping strategies such as ES-DDPM (Lyu et al., 2022). From the perspective of temporal redundancy, methods like DeepCache (Ma et al., 2024), ∆-DiT (Chen et al., 2024), FORA (Selvaraju et al., 2024), and TeaCache (Liu et al., 2025a) reuse intermediate features across timesteps (Shen et al., 2024; Liu et al., 2025c;b), while approaches such as LCM (Luo et al., 2023) and ADD (Sauer et al., 2024) reduce the number of timesteps through model distillation. From the perspective of spatial redundancy, RAS (Liu et al., 2025e) observes that at each diffusion timestep, the model may focus only on semantically coherent regions; therefore, only those regions need to be updated, thereby accelerating image generation. Similarly, ToCa (Zou et al., 2024a) and DuCa (Zou et al., 2024b) note that during denoising, different tokens exhibit varying sensitivities, and dynamically updating only a subset of tokens at each timestep can further accelerate image generation. In contrast to the methods above, RegionE leverages the trajectory characteristics unique to IIE tasks, while simultaneously addressing both spatial and temporal redundancies in diffusion-based image editing to achieve acceleration.

Image Editing. Image editing is an essential task in the field of generative modeling. In the early U-Net (Ronneberger et al., 2015) era, ControlNet (Zhang et al., 2023b) introduced a robust editing solution through a repeat-structure design. As research advanced, inversion-based methods (Pan et al., 2023; Wang et al., 2025) gradually became the dominant approach. These methods apply noise to the original image in the latent space and then recover the edited result through a denoising process. However, this paradigm involves both inversion and denoising stages, which increases complexity. At the same time, IIE models began to emerge. Approaches such as InstructEdit (Wang et al., 2023), MagicBrush (Zhang et al., 2023a), and BrushEdit (Li et al., 2024b) employed modular pipelines, in which large language models generate prompts, spatial cues, or synthetic instruction–image pairs to guide diffusion-based editing. Most of these approaches, however, are task-specific and lack generality. More recently, a new class of IIE has been developed to improve general-purpose editing. These models rely solely on textual instructions, without requiring masks or task-specific designs, and still achieve effective editing performance. Concretely, they leverage MLLMs or advanced text encoders to provide richer semantic control signals, and feed both the target image and noise into a DiT (Peebles & Xie, 2023) architecture to enhance image alignment. In this work, we propose an adaptive, region-aware acceleration method for this emerging IIE models.

- 3 PRELIMINARY

Flow Matching & Rectified Flow. Flow matching (Lipman et al., 2022) has become a widely adopted training technique in advanced diffusion models. It facilitates the transfer from a source distribution π1 to a target distribution π0 by learning a time-dependent velocity field v(x,t). This

(d) L1 Sim. Between 𝒗 and 𝒗

###### (e) Cos Sim. Between 𝒗 and 𝒗

###### (c) Denoising Process

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

T Text Token N Noise Token I Instruction Token

𝒗

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Velocity 𝒗

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

𝒗 𝒗 ) Cosine(

###### N N N

N N N

𝑿

𝒗

𝒗 /𝒗

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

DiT Block

DiT Block

𝑿

###### ···

···

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Denoising

DiT Block

DiT Block

𝑡 = 1 𝑡 = 0

Denoising Process 𝑡

Denoising Process 𝑡

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

(f) Cos Sim. Between 𝒗 and 𝒗

(g) Key Sim. Cross Step (All) (h) Key Sim. Cross Step (𝑿 Part)

DiT Block

DiT Block

[Figure 49]

[Figure 50]

[Figure 51]

2×Tokens

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

###### T N N N

T N N N I I I

𝒗 𝒗 )Cosine(

𝑖 StepIndex

𝑖StepIndex

Prompt

Prompt 𝑿

[Figure 74]

|[Figure 75]<br><br>[Figure 76]|
|---|

[Figure 77]

Noise Image

Noise Image 𝑿

Instruction Image 𝑿

(a) Vanilla-DiT (b) Instruction-DiT

Step Index 𝑖

Step Index 𝑖

Step Interval （𝟐𝟏 − 𝒊)

- Figure 2: Comparison between traditional DiT and DiT in IIE (a, b). Symbolic visualization of the denoising process (c). L1 and cosine similarities of velocities between adjacent timesteps during denoising (d, e). Cosine similarity between velocities after t21 in edited and unedited regions with v21 (f). Cross-step key similarity (g) and cross-step similarity of instruction-related keys (h).

velocity field is used to construct the flow through the ordinary differential equation (ODE):

dϕt(x) dt

= v(ϕt(x),t),ϕ1(x) ∼ π1. (1)

Rectified Flow (Liu et al., 2022) simplifies this process through a linear assumption. Given that X1 follows a noise distribution π1 and X0 follows the target image distribution π0, the equation is

Xt = (1 − t)X0 + tX1,t ∈ [0,1]. (2) Differentiating both ends with respect to timestep t yields: dXt

dt = X1 − X0. The velocity of the rectified flow v(Xt,t), always points in the direction of X1 − X0. Therefore, the training loss is minimized by reducing the deviation between the velocity and X1 − X0:

L = Et ||(X1 − X0) − v(Xt,t)||2 . (3)

The inference process involves starting from X1 and iteratively solving for X0 in reverse, using the learned velocity v(Xt,t). In practice, we typically use a discrete Euler sampler, which discretizes the timestep ti(i ∈ NT,tT = 1,t0 = 0) and approximates:

,ti),∆ti,i−1 = ti − ti−1. (4)

i − ∆ti,i−1 · v(Xt

###### Xt

= Xt

i−1

i

After T iterations, the final target image X0 is obtained. This paper, therefore, targets the IIE task and optimizes the inference process of T iterations in Equation 4.

Instruction-Based Editing Model. Recent IIE models, such as Step1X-Edit (Liu et al., 2025d), FLUX.1 Kontext (Labs et al., 2025) and Qwen-Image (Wu et al., 2025), follow the same paradigm, as shown in Figure 2b. In these models, the velocity field is estimated using Instruction-DiT, the variants of DiT (Peebles & Xie, 2023). The input to Instruction-DiT consists of three types of tokens: text (prompt) tokens XP, noise tokens Xt

, and instruction tokens XI. The noise token corresponds to the generation of the target image, while the text token carries the instruction information. The instruction token is specific to the editing task, representing the part of the image to be edited. Notably, the counts of noise and instruction tokens are roughly comparable and substantially higher than that of text tokens. Temporally, the text and instruction tokens serve as static control signals throughout the denoising process, whereas the noise token evolves dynamically at each timestep. Since InstructionDiT is designed to predict only the noise component, the model’s output corresponds exclusively to the portion represented by the noise token. To simplify the expression, the Instruction-DiT mentioned below will be referred to simply as DiT.

i

Iteration

[Figure 78]

[Figure 79]

[Figure 80]

𝑿

Region-Instruction KV Cache

|[Figure 81]|𝑿|
|---|---|
| | |

[Figure 82]

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

𝑿

[Figure 86]

𝒗

Adaptive Velocity Decay Cache

𝑿

[Figure 87]

[Figure 88]

|[Figure 89]|
|---|

[Figure 90]

Adaptive Region Partition

𝑿

[Figure 91]

Instruction Image

[Figure 92]

···

𝒗𝒕𝑻

𝑿

###### S

S

[Figure 93]

[Figure 94]

|[Figure 95]|
|---|

𝑿

Prompt

𝑿𝒕𝑻 DiT

𝑿

𝑡

|[Figure 96]|
|---|

[Figure 97]

Timestep

|[Figure 98]|
|---|

[Figure 99]

𝑿 DiT

Gather Scatter

𝑿𝒕𝒇 DiT Region-Instruction KV Cache

···

###### S

Gather

|[Figure 100]<br><br>[Figure 101]|
|---|

[Figure 102]

[Figure 103]

𝑿

Solver

Prompt

###### Scatter

𝑿𝒕𝑻 𝟏

𝑿

𝑿

𝑿𝒕𝒔 𝟏

Prompt

|[Figure 104]|
|---|

|[Figure 105]|
|---|

One-step Estimation

[Figure 106]

[Figure 107]

Region-Instruction KV Cache

Few Steps

Forced Update

Eq.5

[Figure 108]

(final step of this stage) Few Steps

###### Most Steps

[Figure 109]

𝑿 𝑿

Stabilization Stage (STS) Region-Aware Generation Stage (RAGS) Smooth Stage (SMS)

###### Adaptive Region Partition (ARP) Region-Instruction KV Cache (RIKVCache) Adaptive Velocity Decay Cache (AVDCache)

E

Query

Query

E U

|[Figure 110]|
|---|

|[Figure 111]|
|---|

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

T E

- T E

- U I

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

−𝒗 ⋅ 𝑡 =

[Figure 116]

Timestep 𝑡

𝒗

DiT Block DiT Block

𝑿

[Figure 117]

DiT 𝑿

###### Update

###### ···

True

···

𝑿𝒕𝒊

𝑿

[Figure 118]

Accumulated Error (Eq.8)

w/ RIKV Cache

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

|[Figure 123]|Cosine Similarity|
|---|---|
| | |

DiT Block DiT Block

Key / Value

Key / Value

Cache

- T E

- U I

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

- T E

- U I

###### Cache

& Error=0

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

> 𝜼

DiT Block DiT Block

𝑿

> 𝜹

Erosion & Dilate

[Figure 128]

T E U I

T E

× Decay(Eq.7)Factor

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

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

0 2 3 7 8 1 4 5 6 9

E U

I

Instruction Token Unedited Noise Token Text Token

Edited Noise Token

Cached Real-time

𝒗

False

Final step of STS / Forced step of RAGS

Step of RAGS (Except forced)

Cache

T

Unedited Region Index Edited Region Index

- Figure 3: Overview of the RegionE. RegionE consists of three stages: STS, RAGS, and SMS. In the STS, no acceleration is applied due to unstable DiT outputs, and all KV values are cached at the final step. In the RAGS, an Adaptive Region Partition distinguishes between edited and unedited regions: unedited regions are denoised in one step, while edited regions are generated iteratively. This iterative generation process leverages RIKVCache for injecting global information and AVDCache for acceleration. Certain forced-update steps aggregate the full image to refresh RIKVCache with complete DiT computation. Finally, in the SMS, several full denoising steps are performed to eliminate artifacts along the boundaries between edited and unedited regions.
- 4 METHODOLOGY

This section introduces RegionE, a method that accelerates the IIE model without additional training. The workflow is shown in Figure 3. RegionE consists of three stages: the Stabilization Stage (STS), the Region-Aware Generation Stage (RAGS), and the Smooth Stage (SMS).

Stabilization Stage. In the early steps of denoising, the input Xt

to DiT is close to Gaussian noise (i.e., the signal-to-noise ratio is low). This leads to oscillations in DiT’s velocity estimation (see Figure 2d and 2e). Since the estimates at this stage are inherently unstable, it is not suitable for acceleration. Therefore, we keep the original sampling process unchanged. Additionally, at the last step of this stage, we save the Key and Value in each attention layer of DiT, denoted as KC and V C.

i

Region-Aware Generation Stage. This stage is the core component of RegionE and consists of three parts: adaptive region partition, region-aware generation, and adaptive velocity decay cache. The first two parts primarily address spatial redundancy in IIE, while the third further reduces temporal redundancy across timesteps.

Adaptive Region Partition. After the stabilization stage, the output of DiT becomes stable. As previously observed, the generation trajectories in the edited regions are curved, whereas those in the unedited regions are straight, as shown in Figure 1 and 2f. Therefore, for the unedited regions XtU

, we can accurately estimate XˆtU

i

at any timestep tf(f < i) using one-step estimation: XˆtU

f

= XtU

− vU(XtU

,ti) · ∆ti,f. (5)

i

i

f

When tf = 0, this corresponds to estimating the final unedited regions Xˆ0U, which is nearly identical to the true X0U. However, using Equation 5 for the edited region does not accurately estimate Xˆ0E. Based on this difference between the edited and unedited regions, we propose an adaptive region partition (ARP), as illustrated in the lower-left corner of Figure 3. Given the velocity vt

at the beginning of the region-aware generation stage and the noisy image Xt

i+1

, the final edited result Xˆ0 can be estimated in one step using Equation 5. This estimate is reliable in unedited regions but less accurate in edited ones. Since the unedited region undergoes minimal change before and after editing,

i

we can compute the cosine similarity between the estimated image Xˆ0 and the instruction image XI along the token dimension. Regions with sufficiently high similarity (> threshold η), that is, small changes before and after editing, are considered unedited regions, while the remainder is treated as the edited region. To account for potential segmentation noise, morphological opening and closing operations are applied to make the two regions more continuous and accurate.

Region-Aware Generation. After identifying the edited and unedited regions, we apply Equation 5 to the unedited region to directly estimate the denoised image XtU

at the next timestep tf in one step, thereby saving computation for the unedited region. For the edited region, our implementation is as follows: first, the input to DiT is changed from [XP,Xt

f

], so that DiT only estimates the velocity of the edited region vtE

###### ,XI] to [XP,XtE

i

i

. However, since DiT contains attention layers that involve global token interactions, completely discarding the XI and XtU

i

inputs can gradually inject bias into the estimation of vtE

i

during global attention. To compensate for this loss of information, we propose a Region-Instruction KV Cache (RIKVCache). Specifically, the input to DiT remains [XP,XtE

i

], but within the attention layers of DiT, it is modified as follows:

i

[QP,QE] · [KP,KE,KUC,KIC]T

) · [VP,VE,VUC,VIC]. (6)

√

softmax(

d

The lower corner labels P, E, U, and I represent prompt token, edited region token, unedited region token, and instruction token, respectively. The superscript C in the upper-right corner indicates that the value is taken from the cache of the previous complete computation. And the middle-lower part of Figure 3 visualizes this process. The feasibility of this approach is supported by the high similarity of the KV pairs between consecutive steps, as shown in Figure 2g and 2h.

Adaptive Velocity Decay Cache. As illustrated in the right part of Figure 1, although the trajectory of the edited region is curved, the velocities between consecutive timesteps are actually similar. Focusing on the intermediate denoising phase, we observe from Figure 2e that the velocity directions between adjacent steps are almost identical (cosine similarity approaches 1). At the same time, the magnitudes exhibit a gradual decay that varies across timesteps (Figure 2d). Based on this observation, we propose an adaptive velocity decay cache (AVDCache). Specifically, the AVDCache introduces a decay factor:

###### . (7) Here, (1 − ∆tt

||vt

i||/||vt

i+1|| = (1 − ∆tt

i+1,ti) · γt

i

i+1,ti) represents the sample-aware component under discrete Euler solver, while γt

i

represents the timestep-aware component. The solver entirely determines the former, while the latter is obtained by fitting on a randomly sampled dataset. Since the decay factor in Eq. 7 characterizes the intrinsic differences between diffusion model timesteps, we introduce the AVDCache criterion:

e

. (8)

Criterion = 1 −

(1 − ∆tt

i+1,ti) · γt

i

i=s

Here,ts and te denote the start and end timesteps of the cache, respectively, while the criterion measures the cumulative error of this process. The decision of whether to apply the cache is made using a threshold δ. The complete process is as follows:

###### ,XP) Criterion > δ vtE,C

DiT(XtE

vtE

(9)

=

i

· im=s(1 − ∆tt

m+1,tm) · γt

else.

i

m

s

The right-lower part of Figure 3 visualizes this process. In fact, AVDCache is an improved version of the existing residual cache methods, with further details and analysis provided in the supplementary.

After the above process, we obtain the generated results for both the edited and unedited regions. We then re-gather these results according to their spatial positions to reconstruct the complete image tokens. It is worth noting that the similarity of the KV Cache decreases as the timestep increases. To address this issue, we periodically enforce full-image gathering at certain timesteps within the region-aware generation stage, performing a complete DiT computation to update the RIKVCache.

Smooth Stage. Small gaps may appear at the boundaries between edited and unedited regions after stitching. Although these gaps are often imperceptible in most cases, to ensure the generality of our method, we perform several steps of unaccelerated denoising on the merged full image to smooth these discontinuities. Empirically, two denoising steps are sufficient to eliminate the gaps effectively.

- 5 EXPERIMENT

- 5.1 EXPERIMENTAL SETTINGS

Pretrained Model & Dataset. We evaluate RegionE on three open-source state-of-the-art IIE models: Step1X-Edit-v1p1 (Liu et al., 2025d), FLUX.1 Kontext (Labs et al., 2025), and Qwen-Image-Edit (Wu et al., 2025). Step1X-Edit adopts a CFG (classifier-free guidance) (Ho & Salimans, 2022) scale of 6, FLUX.1 Kontext uses a scale of 2.5, and Qwen-Image-Edit applies a scale of 4. All models are evaluated with 28 sampling steps. For evaluation, we follow the dataset protocols described in the respective technical reports. Specifically, we use 606 image prompt pairs covering 11 tasks from GEdit-Bench English (Liu et al., 2025d) for Step1X-Edit and Qwen-Image-Edit, and 1026 image prompt pairs spanning five tasks from KontextBench (Labs et al., 2025) for FLUX.1 Kontext.

Evaluation Metrics. We design a comprehensive evaluation framework to assess both the quality and efficiency of IIE models. For quality assessment, we adopt two complementary approaches. First, we evaluate reconstruction quality by measuring deviations before and after acceleration, using PSNR (Zhao et al., 2024), SSIM (Wang & Bovik, 2002), and LPIPS (Zhang et al., 2018) as metrics. Second, we conduct an editing evaluation using vision–language models (VLMs), specifically GPT4o, to assess image quality, semantic alignment, and overall performance (Ku et al., 2024), as shown in Table 1. Evaluation dimensions are denoted by the suffixes SC, PQ, and O, consistent with (Liu et al., 2025d) and (Wu et al., 2025). For efficiency evaluation, we report actual runtime latency as well as the relative speedup compared to the vanilla pretrained models.

Baseline. Currently, there are no acceleration methods designed explicitly for IIEmodels. Therefore, we adapt several effective acceleration techniques initially developed for diffusion models as baselines, since they are also applicable to diffusion-based IIE tasks. From the perspective of timestep redundancy, Steoskip performs larger jumps in the sampling steps, FORA (Selvaraju et al., 2024) employs block-level cache, and ∆-DiT (Chen et al., 2024) and TeaCache (Liu et al., 2025a) use residual cache. From the perspective of spatial redundancy, RAS (Liu et al., 2025e) and ToCa (Zou et al., 2024a) perform redundancy-reduction denoising at the token level.

Implementation Details. For all three models, RegionE uses six steps in the stabilization stage, enforces an update at step 16 in the region-aware generation stage, and adopts two steps in the smooth stage. For Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit, the segmentation thresholds η of ARP are 0.88, 0.93, and 0.80, respectively, while the decision thresholds δ of AVDCache are 0.02, 0.04, and 0.03, respectively. Latency is measured on a single NVIDIA H800 GPU, with each run editing one image at a time.

- 5.2 EXPERIMENTAL RESULTS ANALYSIS

We evaluate RegionE against several state-of-the-art acceleration methods on three prominent IIE models: Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit. Our evaluation encompasses quantitative metrics, efficiency measurements, and visualization, demonstrating that RegionE achieves a superior balance between acceleration and quality preservation. The quantitative results are shown in Table 1. Since both GEdit-Bench and KontextBench involve multiple editing tasks, the table reports results averaged over tasks, while the per-task quantitative results are provided in the supplementary.

Deviation Analysis Compared to Pre-trained Models. The ”Against Vanilla” evaluation reveals RegionE’s exceptional fidelity to original model outputs across all evaluation metrics, significantly outperforming competing acceleration methods. RegionE achieves the highest PSNR values: 30.520 dB (Step1X-Edit), 32.133 dB (FLUX.1 Kontext), and 31.115 dB (Qwen-Image-Edit), representing substantial improvements of 2-4 dB over the next-best methods, indicating minimal pixel-level deviation from the original outputs. The SSIM scores of 0.939, 0.917, and 0.937 demonstrate superior preservation of structural coherence across different model architectures. In contrast, the LPIPS scores of 0.054, 0.057, and 0.046 represent 25-50% improvements over competing methods, indicating dramatically reduced perceptual differences that would be virtually indistinguishable to users. This consistent performance across three diverse model architectures validates RegionE’s architectural agnosticism. RegionE consistently maintains stable, high-quality results.

GPT-4o Editing Quality Assessment. The GPT-4o evaluation provides additional quality validation through automated semantic and perceptual analysis across three dimensions, consistently

Table 1: Comparison of editing quality and efficiency between RegionE and the baseline. All the evaluations are carried out on a single NVIDIA H800 GPU. S denotes the strategy for reducing spatial redundancy, while T denotes the strategy for reducing temporal redundancy.

Against Vanilla GPT-4o Score Efficiency

Model Type

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 7.479 7.466 6.906 27.945 1.000 + Stepskip T 26.719 0.898 0.096 7.491 7.343 6.880 12.299 2.272 + FORA (Selvaraju et al., 2024) T 22.126 0.835 0.178 6.078 7.588 5.863 14.330 1.950 + ∆-DiT (Chen et al., 2024) T 24.659 0.874 0.122 7.432 7.233 6.795 12.728 2.196 + TeaCache (Liu et al., 2025a) T 28.262 0.924 0.072 7.455 7.361 6.866 11.212 2.493 + RAS (Liu et al., 2025e) S 26.819 0.892 0.100 7.339 7.072 6.615 15.239 1.834 + ToCa (Zou et al., 2024a) S 24.699 0.844 0.152 7.185 6.705 6.350 22.149 1.262 + Ours (RegionE) T & S 30.520 0.939 0.054 7.552 7.405 6.948 10.865 2.572

FLUX.1 Kontext (Labs et al., 2025) - - - 7.197 6.963 6.497 14.682 1.000 + Stepskip T 26.199 0.838 0.123 7.126 6.938 6.463 8.512 1.725 + FORA (Selvaraju et al., 2024) T 24.685 0.809 0.146 7.085 6.897 6.383 7.497 1.958 + ∆-DiT (Chen et al., 2024) T 20.227 0.723 0.225 7.055 6.918 6.411 6.751 2.175 + TeaCache (Liu et al., 2025a) T 28.307 0.869 0.097 7.233 6.846 6.455 6.203 2.367 + RAS (Liu et al., 2025e) S 26.217 0.829 0.132 7.216 6.785 6.460 8.219 1.786 + ToCa (Zou et al., 2024a) S 23.906 0.767 0.192 6.985 6.589 6.237 11.299 1.299 + Ours (RegionE) T & S 32.133 0.917 0.057 7.278 6.953 6.538 6.096 2.409

Qwen-Image-Edit (Wu et al., 2025) - - - 8.242 7.948 7.700 32.125 1.000 + Stepskip T 28.439 0.892 0.077 8.090 7.875 7.572 17.555 1.830 + FORA (Selvaraju et al., 2024) T 26.508 0.863 0.098 8.032 7.760 7.501 17.815 1.803 + ∆-DiT (Chen et al., 2024) T 25.020 0.821 0.116 7.964 7.718 7.417 17.470 1.839 + TeaCache (Liu et al., 2025a) T 28.314 0.900 0.075 8.084 7.841 7.563 16.445 1.954 + RAS (Liu et al., 2025e) S 27.251 0.879 0.090 8.152 7.680 7.515 22.327 1.439 + ToCa (Zou et al., 2024a) S OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) T & S 31.115 0.937 0.046 8.242 7.968 7.731 15.604 2.059

demonstrating RegionE’s superior performance. For semantic consistency (G-SC), RegionE achieves scores of 7.552, 7.278, and 8.242, matching or exceeding original models while maintaining substantial acceleration, with Qwen-Image-Edit showing perfect preservation (8.242) despite 2.059× speedup. The perceptual quality (G-PQ) scores of 7.405, 6.953, and 7.968 consistently outperform competing acceleration methods by 0.1 to 0.3 points, demonstrating the practical preservation of visual coherence through region-aware processing. Overall quality (G-O) scores of 6.948, 6.538, and 7.731 provide holistic assessment validation, with the alignment between GPT-4o assessments and quantitative metrics (PSNR, SSIM, LPIPS) strengthening confidence in RegionE’s comprehensive quality preservation across multiple evaluation dimensions and providing additional evidence of the hybrid temporal-spatial optimization approach’s effectiveness.

Efficiency Analysis. RegionE demonstrates substantial efficiency gains while maintaining superior quality, achieving an optimal balance between acceleration and performance preservation with impressive results across all evaluated models. The method achieves speedups of 2.572×, 2.409×, and 2.059× across Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit respectively, translating to significant absolute latency reductions: from 27.945s to 10.865s, from 14.682s to 6.096s, and from 32.125s to 15.604s respectively. RegionE occupies the optimal position on the efficiency-quality curve, maintaining the highest quality metrics while achieving competitive or superior acceleration compared to methods that sacrifice substantial quality for higher speedups.

Visualization. Figure 4 presents partial visualizations of different acceleration methods on Step1XEdit. Among the baselines, RegionE produces edited outputs closest to the vanilla setting at higher speedups, preserving both details and contours. The last column shows ARP predictions of spatial regions in RegionE, where unedited regions are masked. These masked regions closely match human perception. Additional visualizations for other tasks and models are provided in the supplementary.

The experimental results conclusively demonstrate the effectiveness of RegionE in addressing both spatial and temporal redundancies in IIE models. RegionE achieves superior quality preservation metrics while maintaining competitive acceleration. The consistent improvements across diverse model architectures validate the generalizability of the underlying insights about regional editing patterns and temporal similarities in diffusion-based IIE processes.

Table 2: Ablation study on cache design and stage design in RegionE.

Against Vanilla GPT-4o Score Efficiency

Variant

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑ RegionE 30.520 0.939 0.054 7.552 7.405 6.948 10.865 2.572

w/o RIKVCache 22.868 0.822 0.207 5.997 5.389 5.191 10.223 2.734 w/o AVDCache 31.139 0.946 0.046 7.570 7.482 7.023 16.122 1.733

Cache Design

w/o STS 21.441 0.814 0.161 7.045 6.758 6.325 7.149 3.909 w/o SMS 28.857 0.904 0.085 7.456 7.207 6.773 9.766 2.862 w/o Forced Step 28.452 0.915 0.080 7.536 7.305 6.925 10.204 2.739

Stage Design

Ours (RegionE)

Edited Region

Input Stepskip FORA △-DiT TeaCache RAS ToCa

Vanilla

|[Figure 149]<br><br>[Figure 150]|
|---|

|[Figure 151]<br><br>[Figure 152]|
|---|

|[Figure 153]<br><br>[Figure 154]|
|---|

|[Figure 155]<br><br>[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]|
|---|

|[Figure 159]<br><br>[Figure 160]|
|---|

|[Figure 161]<br><br>[Figure 162]|
|---|

|[Figure 163]<br><br>[Figure 164]|
|---|

|[Figure 165]<br><br>[Figure 166]|
|---|

|[Figure 167]<br><br>[Figure 168]|
|---|

Material Alter

Change the bear’s material to glass.

|[Figure 169]<br><br>[Figure 170]|
|---|

|[Figure 171]<br><br>[Figure 172]|
|---|

|[Figure 173]<br><br>[Figure 174]|
|---|

|[Figure 175]<br><br>[Figure 176]|
|---|

|[Figure 177]<br><br>[Figure 178]|
|---|

|[Figure 179]<br><br>[Figure 180]|
|---|

|[Figure 181]<br><br>[Figure 182]|
|---|

|[Figure 183]<br><br>[Figure 184]|
|---|

|[Figure 185]<br><br>[Figure 186]|
|---|

|[Figure 187]<br><br>[Figure 188]|
|---|

Text Change

Change the text 'hotwind' to 'cool breeze’.

|[Figure 189]<br><br>[Figure 190]|
|---|

|[Figure 191]<br><br>[Figure 192]|
|---|

|[Figure 193]<br><br>[Figure 194]|
|---|

|[Figure 195]<br><br>[Figure 196]|
|---|

|[Figure 197]<br><br>[Figure 198]|
|---|

|[Figure 199]<br><br>[Figure 200]|
|---|

|[Figure 201]<br><br>[Figure 202]|
|---|

|[Figure 203]<br><br>[Figure 204]|
|---|

|[Figure 205]<br><br>[Figure 206]|
|---|

|[Figure 207]<br><br>[Figure 208]|
|---|

Add a potted green plant to the right of the sofa.

Subject Add

|[Figure 209]<br><br>[Figure 210]|
|---|

|[Figure 211]<br><br>[Figure 212]|
|---|

|[Figure 213]<br><br>[Figure 214]|
|---|

|[Figure 215]<br><br>[Figure 216]|
|---|

|[Figure 217]<br><br>[Figure 218]|
|---|

|[Figure 219]<br><br>[Figure 220]|
|---|

|[Figure 221]<br><br>[Figure 222]|
|---|

|[Figure 223]<br><br>[Figure 224]|
|---|

|[Figure 225]<br><br>[Figure 226]|
|---|

|[Figure 227]<br><br>[Figure 228]|
|---|

Replace the two children with a fire truck.

Subject Replace

Figure 4: Examples of edited images by RegionE and baseline on Step1X-Edit-v1p1.

- 5.3 ABLATION STUDY

We conduct ablation studies to investigate the contributions of different components in RegionE, primarily on the Step1X-Edit-v1p1. The quantitative results are summarized in Table 2.

Cache Design. We propose two key components: RIKVCache and AVDCache. Removing RIKVCache, i.e., performing local attention within the edited region without injecting instruction information or context from the unedited region, results in a 2.734× speed-up. However, this comes at a significant cost to editing quality, with PSNR dropping from 30.520 to 22.868 and G-O decreasing from 6.948 to 5.191. This demonstrates that global context supervision is crucial even during region generation. In contrast, removing AVDCache results in a slight improvement in editing quality (G-O increases from 6.948 to 7.023), but without eliminating redundancy across timesteps, the acceleration is limited to 1.733. This indicates that AVDCache significantly improves inference efficiency with minimal degradation in quality.

Stage Design. We introduce two auxiliary stages: Stabilization Stage (STS) and Smooth Stage (SMS), as well as a forced step in the region-aware generation stage (RAGS). Removing STS causes substantial drops in editing quality (PSNR: 30.520 → 21.441, LPIPS: 0.054 → 0.161, G-O: 6.948 → 6.325). As discussed in Section 4, STS addresses the instability in speed estimation, and skipping it results in degraded performance. Removing SMS leads to smaller declines in both pixel-level (PSNR: 30.520 → 28.857, SSIM: 0.939 → 0.904) and perceptual metrics (G-O: 6.948 → 6.773), reflecting its role in bridging the gap between edited and unedited regions. Finally, when the forced step in RAGS was removed, since its role was to mitigate the decay of KV similarity over time, its removal led to a 2-point drop in PSNR, further validating its necessity.

6 CONCLUSION

Inspired by temporal and spatial redundancy in IIE, we propose RegionE, an adaptive, region-aware generation framework that accelerates the IIE process. Specifically, we perform early prediction on spatial regions using ARP and combine it with RIKVCache for region-wise editing to reduce spatial redundancy. We also use AVDCache to minimize temporal redundancy. Experiments show that RegionE achieves 2.57×, 2.41×, and 2.06× end-to-end speedups on Step1X-Edit and FLUX.1 Kontext, and Qwen-Image-Edit, respectively, while maintaining minimal bias (PSNR 30.52–32.13) and negligible quality loss (GPT-4o evaluation results remain comparable). These results demonstrate the effectiveness of RegionE in reducing redundancy in IIE.

REFERENCES

Jiazi Bu, Pengyang Ling, Yujie Zhou, Yibin Wang, Yuhang Zang, Tong Wu, Dahua Lin, and Jiaqi Wang. Dicache: Let diffusion model determine its own cache, 2025.

Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, and Shinkook Choi. Ld-pruner: Efficient pruning of latent diffusion models using task-agnostic insights. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 821–830, 2024.

Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. δ-dit: A training-free acceleration method tailored for diffusion transformers. arXiv preprint arXiv:2406.01125, 2024.

Pengtao Chen, Xianfang Zeng, Maosen Zhao, Peng Ye, Mingzhu Shen, Wei Cheng, Gang Yu, and Tao Chen. Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers, 2025.

Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision, pp. 206–235. Springer, 2024.

Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. In Advances in Neural Information Processing Systems, 2023.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6007–6017, 2023.

Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: Architecturally compressed stable diffusion for efficient text-to-image generation. In Workshop on Efficient Systems for Foundation Models@ ICML, 2023.

Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024a.

Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Junhao Zhuang, Ying Shan, Yuexian Zou, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. arXiv preprint arXiv:2412.10316,

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7353–7363, 2025a.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers, 2025b. URL https://arxiv. org/abs/2503.06923.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Fei Ren, Shaobo Wang, Kaixin Li, and Linfeng Zhang. Speca: Accelerating diffusion transformers with speculative feature caching. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 10024–10033. ACM, October 2025c. doi: 10.1145/3746027.3755331.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025d.

Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Clear: Conv-like linearization revs pre-trained diffusion transformers up. arXiv preprint arXiv:2412.16112, 2024.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Ziming Liu, Yifan Yang, Chengruidong Zhang, Yiqi Zhang, Lili Qiu, Yang You, and Yuqing Yang. Region-adaptive sampling for diffusion transformers. arXiv preprint arXiv:2502.10389, 2025e.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process. arXiv preprint arXiv:2205.12524, 2022.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15762–15772, 2024.

Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. Effective real image editing with accelerated iterative diffusion inversion, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computerassisted intervention, pp. 234–241. Springer, 2015.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2024.

Pratheba Selvaraju, Tianyu Ding, Tianyi Chen, Ilya Zharkov, and Luming Liang. Fora: Fast-forward caching in diffusion transformer acceleration. arXiv preprint arXiv:2407.01425, 2024.

Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1972–1981, 2023.

Mingzhu Shen, Pengtao Chen, Peng Ye, Guoxuan Xia, Tao Chen, Christos-Savvas Bouganis, and Yiren Zhao. MD-dit: Step-aware mixture-of-depths for efficient diffusion transformers. In Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024.

Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Zhao Jin, Jingyi Liao, Shunyu Liu, and Dacheng Tao. Vorta: Efficient video diffusion via routing sparse attention, 2025.

Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing, 2025.

Qian Wang, Biao Zhang, Michael Birsak, and Peter Wonka. Instructedit: Improving automatic masks for diffusion-based image editing with user instructions. arXiv preprint arXiv:2305.18047, 2023.

Zhou Wang and Alan C Bovik. A universal image quality index. IEEE signal processing letters, 9(3): 81–84, 2002.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, Jianfei Chen, Ion Stoica, Kurt Keutzer, and Song Han. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity, 2025.

Zexuan Yan, Yue Ma, Chang Zou, Wenteng Chen, Qifeng Chen, and Linfeng Zhang. Eedit: Rethinking the spatial and temporal redundancy for efficient image editing, 2025.

Zhihang Yuan, Hanling Zhang, Pu Lu, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattn: Attention compression for diffusion transformer models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 1196–1219. Curran Associates, Inc., 2024.

Hanling Zhang, Rundong Su, Zhihang Yuan, Pengtao Chen, Mingzhu Shen, Yibo Fan, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattnv2: Head-wise attention compression for multi-modality diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16399–16409, October 2025.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023a.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3836–3847, 2023b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Maosen Zhao, Pengtao Chen, Chong Yu, Yan Wen, Xudong Tan, and Tao Chen. Pioneering 4-bit fp

quantization for diffusion models: Mixup-sign quantization and timestep-aware fine-tuning, 2025. Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid

attention broadcast. arXiv preprint arXiv:2408.12588, 2024.

Xin Zhou, Dingkang Liang, Kaijin Chen, Tianrui Feng, Xiwu Chen, Hongkai Lin, Yikang Ding, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Less is enough: Training-free video diffusion acceleration via runtime-adaptive caching, 2025.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. arXiv preprint arXiv:2410.05317, 2024a.

Chang Zou, Evelyn Zhang, Runlin Guo, Haohang Xu, Conghui He, Xuming Hu, and Linfeng Zhang. Accelerating diffusion transformers with dual feature caching. arXiv preprint arXiv:2412.18911,

###### RegionE: Adaptive Region-Aware Generation for Efficient Image Editing Supplementary Material

We organize the supplementary material as follows:

- • Section A: Pseudocode of RegionE
- • Section B: Analysis of Adaptive Velocity Decay Cache
- • Section C: Per-Task Visualization Results in the Benchmark
- • Section D: Per-Task Quantitative Results in the Benchmark

- A PSEUDOCODE OF REGIONE

Algorithm 1 RegionE: Adaptive Region-Aware Generation for Efficient Image Editing Input: Diffusion transformer Φ(·), sampling step T, insturction image XI, text tokens XP, random

noise XT, total steps in stabilization stage tst, total steps in smooth stage tsm, threshold of adaptive region partition η, threshold of adaptive velocity decay cache δ, sorted forced steps list

tf list.

- 1: // Initialization
- 2: RIKVCache CKV = None, RIKVCache flag f = (False,False); AVDCache CA=None;
- 3: Accumulative Error e = 0; tf list.insert(0,T − tst); tf list.insert(−1,tsm − 1);

- 4: // Stabilization Stage
- 5: for i ← T to T − tst do
- 6: if i == T − tst then
- 7: f[0] = True ▷ first dimension represents storing, second dimension represents retrieving
- 8: end if
- 9: vt

i

,CKV = Φ([XP,Xt

i

,XI],CKV,f)

- 10: Xt

i−1

= Xt

i − (ti − ti−1) · vt

i

- 11: end for
- 12: // Region-Aware Generation Stage
- 13: ▷ Adaptive Region Partition
- 14: Xˆ0 = Xt

T−tst − vT−tst+1 · tT−tst

- 15: Eindex,Uindex = Erosion & Dilate(cos(Xˆ0,XI) > η)

- 16: ▷ Region-Aware Generation
- 17: for i ← 0 to len(tf list) − 2 do

- 18: prev = tf list[i]; next = tf list[i + 1]

- 19: XtE

prev

= Xt

prev

[Eindex];XtU

prev

= Xt

prev

[Uindex]

- 20: XˆtU

next+1

= XtU

prev

− vtU

prev+1 · (tprev − tnext+1) ▷ one-step estimation for unedited region

- 21: f[0] = False, f[1] = True ▷ iteritive denoising for edited region
- 22: for j ← prev to next + 1 do
- 23: ▷ Adaptive Velocity Decay Cache
- 24: Calculate e according to Eq.8
- 25: if e > δ then
- 26: vtE

j

,CKV = Φ([XP,XtE

j

,XI],CKV,f)

- 27: CA = vtE

j

- 28: XtE

j−1

= XtE

j

− (tj − tj−1) · vtE

j

- 29: else
- 30: vtE

j

= CA∗decay factor according to Eq.7

- 31: end if
- 32: end for
- 33: Xt

next+1

= gather(XtU

next

,XtE

next+1

)

- 34: f[0] = True,f[1] = False
- 35: vt

next+1

,CKV = Φ([XP,Xt

next+1

,XI],CKV,f)

- 36: Xt

next

= Xt

next+1 − (tnext − tnext+1) · vt

next+1

- 37: end for
- 38: // Smooth Stage
- 39: f[0] = False,f[1] = False
- 40: for i ← tsm − 1 to 1 do
- 41: vt

i

,CKV = Φ(Xt

i

,CKV,f)

- 42: Xt

i−1

= Xt

i − (ti − ti−1) · vt

i

- 43: end for Output: Target image after editing X0

###### B ANALYSIS OF ADAPTIVE VELOCITY DECAY CACHE

𝒗

𝒗

𝒗

𝒗

|[Figure 229]<br><br>[Figure 230]<br><br>DiT Block<br><br>[Figure 231]<br><br>[Figure 232]<br><br>DiT Block<br><br>[Figure 233]<br><br>[Figure 234]<br><br>DiT Block<br><br>···<br><br>[Figure 235]|
|---|

|[Figure 236]<br><br>[Figure 237]<br><br>DiT Block<br><br>[Figure 238]<br><br>[Figure 239]<br><br>DiT Block<br><br>[Figure 240]<br><br>[Figure 241]<br><br>DiT Block<br><br>···<br><br>[Figure 242]|
|---|

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

DiT Block

DiT Block

[Figure 247]

[Figure 248]

### ···

### ···

### ···

### ···

[Figure 249]

[Figure 250]

### ···

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

### ···

DiT Block

DiT Block

S S

∆ ∆

∆= 𝒗 − 𝑿

∆= 𝒗 − 𝑿

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

DiT Block

DiT Block

𝑿

𝑿

𝑿

𝑿

Full Computation Cached Step Cached Step

Full Computation

Figure 5: Pipeline Based on Residual Cache.

In current research on diffusion model caching, many studies focus on residual-based caches (Chen et al., 2024; Liu et al., 2025a; Zhou et al., 2025; Bu et al., 2025), which store the ∆ shown in Figure 5. Based on the sampling formula in Equation 4 and the definition of caching, we can derive the following expression: 

i − (ti − ti−1) · vt

###### Xt

= Xt

 

i−1

i

. (10)

i − Xt

∆ = vt

i

###### vt

= Xt

+ ∆

i−1

i−1

It can be solved as:

. (11) Similarly, for the timestep ti−2, we have:

= [1 − (ti − ti−1)] · vt

###### vt

i−1

i

. (12) Therefore, if we perform N steps of residual caching, as illustrated in Figure 5, we can obtain:

= [1 − (ti−1 − ti−2)] · vt

###### vt

i−2

i−1

N

[1 − (ti−m+1 − ti−m)] · vt

###### vt

=

i−N

i

m=1

N

. (13)

·vt

[1 − ∆ti−m+1,i−m]

=

i

m=1

Determined by Solver

This further indicates that the current residual cache and the velocity cache are equivalent. Since ∆ti−m+1,i−m is a minimal value approaching zero, the coefficient before vt

is less than one. Therefore, it can be seen that the current residual cache is essentially a decayed form of the velocity cache. Furthermore, we observe that the solver determines the decay coefficient in Equation 13. However, as shown in Figure 2d, the decay of velocity exhibits a timestep-dependent behavior. To account for this, we introduce an external timestep correction coefficient γt

i

. Notably, the AVDCache proposed in this paper reduces to Equation 13 when the correction coefficient γt

i

equals 1.

i

- C PER-TASK VISUALIZATION RESULTS IN THE BENCHMARK

Due to space limitations, we put the visualization results of some tasks in the manuscript. Here, we provide a visual comparison of additional tasks and models. Figure 6 and Figure 7 show the visualization results of 11 tasks on Step1X-Edit. Figure 9 and Figure 10 show the visualization results of 11 tasks on Qwen-Image-Edit. Figure 8 show the visualization results of 5 tasks on FLUX.1 Kontext.

DiT +TeaCache

+Stepskip +FORA

ToCa +Ours ( RegionE)Edited Region

Vanilla

+RAS

Input

△-

+

+

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

Background

change Color

Please change the background wall to a green forest with high.

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Change the background to an indoor setting.

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

Change the bed sheet color to sky blue.

alter Material

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

Alter the color of doughnut to silver.

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

Change the bear’s material to glass.

alter Motion

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Change the plane's material to feathers.

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

change Ps

Make the expression more sorrowful.

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Change the expression to a crying face.

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

human

Without altering or beautifying anything else, just shape my eyebrows to suit me.

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

Remove his abs and add more fat to his body.

Figure 6: Examples of edited images by RegionE and baseline on Step1X-Edit-v1p1.

DiT +TeaCache

+Stepskip +FORA

ToCa +Ours ( RegionE)Edited Region

Vanilla

+RAS

Input

△-

+

+

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

###### Change Subject

###### Style

Generate a cyberpunk-style photo.

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Convert this image into an anime style.

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

###### Add Subject

Add a potted green plant to the right of the sofa.

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

Add a flying baseball coming towards the player.

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

###### Remove Subject

Remove the umbrella.

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

Remove the freight train.

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

###### Replace Text

Replace the two children with a fire truck.

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

Replace the cat on the laptop with a robot.

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

###### Change Tone

Change the text 'hotwind' to 'cool breeze’.

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

Replace the text 'PROJECT' with 'PROMPT’.

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

###### Transfer

Change the time to nighttime.

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

Change the weather to snow.

Figure 7: Examples of edited images by RegionE and baseline on Step1X-Edit-v1p1.

DiT +TeaCache

+Stepskip +FORA

ToCa +Ours ( RegionE)Edited Region

Vanilla

+RAS

Input

△-

+

+

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

Reference Instruction

Character

The same man is now sitting on a throne.

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

Place this logo on the homescreen of an iphone as an app icon.

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

Editing-Global Instruction

Its now spring season with flowers blooming on the sidewalk.

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

Make this a real photo.

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

Editing-Local Style

Remove the text.

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

The man is now standing on the skis and riding away.

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

Reference Text

Make a photo of a cat smelling a flower in the style of this image.

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

Using this style make art of a castle made of ice cream on top of a cone.

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

Change the text from "Black Forest Labs" to "Make Pixels Beautiful“.

Editing

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

Replace all the text with "Black Forest Labs"

Figure 8: Examples of edited images by RegionE and baseline on FLUX.1 Kontext.

DiT +TeaCache

+Stepskip +FORA

RegionE)Edited Region

+Ours

Vanilla

+RAS

Input

△-

+

(

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

##### Background

##### change Color

Change the background to a starry sky.

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

Change the background to a green grassland.

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

Turn the color of dog to pink.

##### alter Material

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

Change the color of suit cases to silver.

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

Replace the bench’s material with marble.

##### alter Motion

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

Replace the doctor's coat with a Merino wool sweater.

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

##### change Ps

Change the action of cat to sleeping.

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

Make the action of the man to cheering.

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

##### human Style

Make his beard longer.

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

Make him look sad.

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

##### Change

Adjust the image style to a bubble-like aesthetic.

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

Edit this image into a bright and sunny style for use as an avatar.

DiT +TeaCache

+Stepskip +FORA

RegionE)Edited Region

+Ours

Vanilla

+RAS

Input

△-

+

(

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

#### Subject

#### Add Tone

Add a chair in the background.

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

Add a poolside lounge chair.

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

Please remove the woman outside the car while keeping everything else intact, then re-output the image for me.

#### Remove

#### Replace Subject

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

Remove the black railing.

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

#### Change Subject

Give me long hair—shoulder-length or waist-length.

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

Replace the laptop in front of the girl with a book.

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

Little Yue, can you replace the character "曹" with "叶" inside?

#### Transfer Text

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

Change the text 'SCHOOL' to 'COLLEGE’.

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

Change the nighttime scene in the image to daytime.

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

Change the weather to foggy.

- D PER-TASK QUANTITATIVE RESULTS IN THE BENCHMARK

In this section, we present the performance of RegionE and the baseline methods on each task in the benchmark. Table 3-Table 13 show the performance on the 11 tasks: motion-change, ps-human, color-alter, material-alter, subject-add, subject-remove, style-change, tone-transfer, subject-replace, text-change, and background-change. Table 14-Table18 show the performance on the five tasks: Character Reference, Style Reference, Text Editing, Instruction Editing-Global, and Instruction Editing-Local.

- Table 3: Comparison of RegionE and other baselines on the motion-change task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 4.350 7.950 4.444 27.950 1.000 + Stepskip 25.887 0.902 0.093 4.350 8.100 4.562 12.306 2.271 + FORA (Selvaraju et al., 2024) 20.935 0.818 0.189 2.175 7.575 2.385 14.339 1.949 + ∆-DiT (Chen et al., 2024) 24.549 0.876 0.121 4.350 7.975 4.445 12.730 2.196 + TeaCache (Liu et al., 2025a) 26.926 0.925 0.068 4.475 8.050 4.524 11.218 2.492 + RAS (Liu et al., 2025e) 25.888 0.889 0.109 4.025 7.375 4.012 15.253 1.832 + ToCa (Zou et al., 2024a) 24.428 0.843 0.165 3.775 6.975 3.578 22.225 1.258 + Ours (RegionE) 29.633 0.937 0.053 4.625 7.775 4.763 10.739 2.603

Qwen-Image-Edit (Wu et al., 2025) - - - 4.850 8.550 5.112 32.140 1.000 + Stepskip 27.791 0.905 0.066 4.725 8.625 5.029 17.566 1.830 + FORA (Selvaraju et al., 2024) 26.744 0.889 0.079 4.825 8.325 4.995 17.827 1.803 + ∆-DiT (Chen et al., 2024) 25.756 0.848 0.095 4.675 8.575 4.921 17.481 1.839 + TeaCache (Liu et al., 2025a) 26.776 0.911 0.070 5.025 8.500 5.251 16.389 1.961 + RAS (Liu et al., 2025e) 26.585 0.882 0.096 5.000 8.625 5.262 22.300 1.441 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 29.416 0.932 0.057 4.825 8.550 5.164 15.695 2.048

- Table 4: Comparison of RegionE and other baselines on the ps-human task of GEdit-Bench, evaluated in terms of quality and efficiency.

Against Vanilla GPT-4o Score Efficiency

Model

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 4.614 8.086 4.649 27.927 1.000 + Stepskip 29.220 0.916 0.069 4.600 8.086 4.728 12.296 2.271 + FORA (Selvaraju et al., 2024) 23.596 0.863 0.142 3.414 8.529 3.920 14.323 1.950 + ∆-DiT (Chen et al., 2024) 26.348 0.884 0.099 4.800 8.086 4.893 12.728 2.194 + TeaCache (Liu et al., 2025a) 31.428 0.942 0.047 5.114 7.929 5.191 11.208 2.492 + RAS (Liu et al., 2025e) 29.077 0.921 0.072 4.400 7.886 4.486 15.237 1.833 + ToCa (Zou et al., 2024a) 26.716 0.878 0.125 4.786 7.914 4.838 22.073 1.265

- + Ours (RegionE) 32.985 0.957 0.037 4.629 8.114 4.731 10.813 2.583

Qwen-Image-Edit (Wu et al., 2025) - - - 5.814 8.500 5.972 32.100 1.000 + Stepskip 32.080 0.936 0.040 5.757 8.414 5.904 17.553 1.829 + FORA (Selvaraju et al., 2024) 30.120 0.920 0.049 5.700 8.443 5.933 17.816 1.802 + ∆-DiT (Chen et al., 2024) 28.323 0.887 0.062 5.743 8.500 5.911 17.462 1.838 + TeaCache (Liu et al., 2025a) 32.347 0.948 0.038 5.714 8.400 5.833 16.360 1.962 + RAS (Liu et al., 2025e) 29.857 0.917 0.061 5.843 8.271 5.884 22.340 1.437 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM

- + Ours (RegionE) 33.550 0.963 0.029 6.086 8.486 6.227 15.473 2.075

- Table 5: Comparison of RegionE and other baselines on the color-alter task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.750 6.875 7.395 28.019 1.000 + Stepskip 27.291 0.919 0.080 8.325 6.975 7.349 12.330 2.273 + FORA (Selvaraju et al., 2024) 21.871 0.838 0.132 8.800 7.525 7.889 14.356 1.952 + ∆-DiT (Chen et al., 2024) 24.942 0.901 0.107 8.075 6.600 6.968 12.770 2.194 + TeaCache (Liu et al., 2025a) 28.084 0.938 0.050 8.525 6.950 7.345 11.242 2.492 + RAS (Liu et al., 2025e) 28.800 0.909 0.069 8.700 6.925 7.432 15.274 1.834 + ToCa (Zou et al., 2024a) 25.917 0.864 0.118 8.600 6.725 7.232 21.996 1.274

- + Ours (RegionE) 32.739 0.956 0.032 8.850 7.250 7.747 11.188 2.504

Qwen-Image-Edit (Wu et al., 2025) inf 1.000 0.000 9.250 7.525 8.170 32.082 1.000 + Stepskip 29.795 0.896 0.064 9.050 7.450 8.084 17.527 1.830 + FORA (Selvaraju et al., 2024) 28.035 0.879 0.078 8.875 7.350 7.872 17.795 1.803 + ∆-DiT (Chen et al., 2024) 25.892 0.835 0.094 9.025 7.375 8.021 17.479 1.835 + TeaCache (Liu et al., 2025a) 30.757 0.922 0.057 8.775 7.250 7.840 16.566 1.937 + RAS (Liu et al., 2025e) 29.132 0.909 0.060 9.150 7.050 7.860 22.356 1.435 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM

- + Ours (RegionE) 33.144 0.951 0.032 9.225 7.475 8.172 15.527 2.066

- Table 6: Comparison of RegionE and other baselines on the material-alter task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.300 6.575 7.226 27.880 1.000 + Stepskip 24.377 0.858 0.117 8.050 5.900 6.676 12.260 2.274 + FORA (Selvaraju et al., 2024) 20.406 0.763 0.224 7.175 6.875 6.579 14.286 1.952 + ∆-DiT (Chen et al., 2024) 21.995 0.829 0.154 8.025 5.975 6.695 12.685 2.198 + TeaCache (Liu et al., 2025a) 25.630 0.875 0.099 8.175 6.000 6.796 11.163 2.498 + RAS (Liu et al., 2025e) 24.302 0.844 0.141 8.275 5.700 6.633 15.202 1.834 + ToCa (Zou et al., 2024a) 22.503 0.793 0.186 7.850 5.450 6.352 22.306 1.250 + Ours (RegionE) 27.248 0.897 0.080 8.475 6.200 6.997 11.251 2.478

Qwen-Image-Edit (Wu et al., 2025) - - - 8.725 7.150 7.629 32.156 1.000 + Stepskip 26.300 0.870 0.093 8.650 6.875 7.557 17.578 1.829 + FORA (Selvaraju et al., 2024) 24.699 0.841 0.116 8.525 6.675 7.389 17.839 1.803 + ∆-DiT (Chen et al., 2024) 23.827 0.799 0.133 8.425 6.475 7.205 17.472 1.840 + TeaCache (Liu et al., 2025a) 26.788 0.876 0.092 8.725 6.775 7.564 16.485 1.951 + RAS (Liu et al., 2025e) 26.927 0.862 0.098 8.400 6.625 7.192 22.357 1.438 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 30.024 0.917 0.060 8.550 6.900 7.415 15.671 2.052

- Table 7: Comparison of RegionE and other baselines on the subject-add task of GEdit-Bench, evaluated in terms of quality and efficiency.

Against Vanilla GPT-4o Score Efficiency

Model

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.283 7.950 7.905 27.912 1.000 + Stepskip 25.692 0.892 0.085 8.583 8.083 8.142 12.290 2.271 + FORA (Selvaraju et al., 2024) 21.717 0.848 0.150 6.400 8.083 6.131 14.322 1.949 + ∆-DiT (Chen et al., 2024) 24.203 0.868 0.099 8.017 8.050 7.655 12.727 2.193 + TeaCache (Liu et al., 2025a) 26.413 0.914 0.073 8.600 8.067 8.177 11.204 2.491 + RAS (Liu et al., 2025e) 25.008 0.880 0.101 8.100 7.517 7.532 15.232 1.832 + ToCa (Zou et al., 2024a) 23.524 0.820 0.159 7.650 6.950 6.939 22.062 1.265 + Ours (RegionE) 28.514 0.923 0.058 8.383 7.950 7.858 10.528 2.651

Qwen-Image-Edit (Wu et al., 2025) - - - 9.117 8.017 8.381 32.081 1.000 + Stepskip 27.666 0.890 0.092 8.767 7.950 8.146 17.532 1.830 + FORA (Selvaraju et al., 2024) 26.871 0.879 0.093 9.017 7.933 8.313 17.810 1.801 + ∆-DiT (Chen et al., 2024) 25.559 0.849 0.108 8.617 7.817 7.967 17.452 1.838 + TeaCache (Liu et al., 2025a) 28.672 0.903 0.066 8.783 7.933 8.099 16.422 1.954 + RAS (Liu et al., 2025e) 27.398 0.891 0.081 9.100 7.933 8.267 22.278 1.440 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 30.763 0.938 0.050 8.983 8.233 8.441 15.295 2.097

- Table 8: Comparison of RegionE and other baselines on the subject-remove task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 7.351 7.947 6.973 27.954 1.000 + Stepskip 33.649 0.954 0.038 7.579 7.684 6.969 12.300 2.273 + FORA (Selvaraju et al., 2024) 30.330 0.943 0.062 5.474 7.895 5.285 14.330 1.951 + ∆-DiT (Chen et al., 2024) 31.847 0.948 0.047 7.930 7.684 7.319 12.724 2.197 + TeaCache (Liu et al., 2025a) 36.735 0.973 0.024 7.281 7.737 6.841 11.213 2.493 + RAS (Liu et al., 2025e) 32.966 0.936 0.052 7.211 7.860 6.861 15.236 1.835 + ToCa (Zou et al., 2024a) 29.806 0.894 0.095 7.175 7.088 6.481 22.378 1.249 + Ours (RegionE) 35.772 0.963 0.028 7.719 7.737 7.182 10.453 2.674

Qwen-Image-Edit (Wu et al., 2025) - - - 8.965 8.246 8.477 32.170 1.000 + Stepskip 32.187 0.913 0.048 9.035 8.298 8.558 17.572 1.831 + FORA (Selvaraju et al., 2024) 29.288 0.865 0.072 9.175 7.930 8.475 17.820 1.805 + ∆-DiT (Chen et al., 2024) 27.056 0.826 0.090 8.895 7.947 8.348 17.486 1.840 + TeaCache (Liu et al., 2025a) 31.687 0.899 0.051 8.895 8.228 8.441 16.434 1.958 + RAS (Liu et al., 2025e) 28.440 0.876 0.080 8.842 8.035 8.371 22.331 1.441 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 32.122 0.925 0.052 9.333 8.351 8.787 15.349 2.096

- Table 9: Comparison of RegionE and other baselines on the style-change task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.150 6.917 7.359 27.898 1.000 + Stepskip 21.064 0.828 0.185 8.183 6.583 7.199 12.277 2.272 + FORA (Selvaraju et al., 2024) 15.851 0.680 0.372 7.183 7.017 6.883 14.300 1.951 + ∆-DiT (Chen et al., 2024) 18.893 0.791 0.233 8.167 6.367 7.066 12.684 2.200 + TeaCache (Liu et al., 2025a) 21.695 0.857 0.156 8.000 6.733 7.213 11.187 2.494 + RAS (Liu et al., 2025e) 21.355 0.814 0.193 8.217 6.400 7.108 15.217 1.833 + ToCa (Zou et al., 2024a) 19.819 0.760 0.250 8.283 6.000 6.927 22.327 1.250 + Ours (RegionE) 25.449 0.900 0.102 8.267 6.617 7.251 11.797 2.365

Qwen-Image-Edit (Wu et al., 2025) - - - 8.267 7.133 7.526 32.115 1.000 + Stepskip 23.954 0.805 0.139 8.067 7.083 7.355 17.560 1.829 + FORA (Selvaraju et al., 2024) 21.784 0.745 0.185 8.017 7.100 7.385 17.807 1.804 + ∆-DiT (Chen et al., 2024) 20.552 0.662 0.219 8.117 7.033 7.395 17.455 1.840 + TeaCache (Liu et al., 2025a) 23.137 0.816 0.152 8.300 7.133 7.544 16.414 1.957 + RAS (Liu et al., 2025e) 24.073 0.772 0.169 7.983 7.000 7.348 22.275 1.442 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 27.980 0.897 0.073 8.233 7.250 7.583 16.822 1.909

- Table 10: Comparison of RegionE and other baselines on the tone-transfer task of GEdit-Bench, evaluated in terms of quality and efficiency.

Against Vanilla GPT-4o Score Efficiency

Model

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 6.950 7.325 6.679 27.874 1.000 + Stepskip 24.744 0.899 0.122 7.200 7.325 6.917 12.260 2.274 + FORA (Selvaraju et al., 2024) 19.078 0.786 0.251 6.900 8.125 7.088 14.293 1.950 + ∆-DiT (Chen et al., 2024) 22.104 0.862 0.164 7.000 7.250 6.852 12.678 2.199 + TeaCache (Liu et al., 2025a) 27.915 0.933 0.072 6.825 7.400 6.600 11.171 2.495 + RAS (Liu et al., 2025e) 26.455 0.895 0.111 7.200 7.000 6.688 15.187 1.835 + ToCa (Zou et al., 2024a) 23.954 0.840 0.159 6.500 6.550 5.991 22.408 1.244 + Ours (RegionE) 30.860 0.945 0.064 6.900 7.275 6.641 11.496 2.425

Qwen-Image-Edit (Wu et al., 2025) - - - 8.475 8.025 8.084 32.160 1.000 + Stepskip 29.715 0.862 0.092 8.150 8.000 7.820 17.562 1.831 + FORA (Selvaraju et al., 2024) 27.514 0.839 0.117 8.025 7.875 7.771 17.841 1.803 + ∆-DiT (Chen et al., 2024) 25.471 0.792 0.139 7.950 7.725 7.592 17.462 1.842 + TeaCache (Liu et al., 2025a) 30.064 0.910 0.061 8.375 8.125 8.033 16.381 1.963 + RAS (Liu et al., 2025e) 29.142 0.880 0.089 8.500 8.075 8.142 22.372 1.437 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 34.051 0.948 0.034 8.450 8.275 8.199 15.851 2.029

- Table 11: Comparison of RegionE and other baselines on the subject-replace task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.650 7.233 7.718 27.983 1.000 + Stepskip 25.233 0.875 0.111 8.683 6.867 7.548 12.325 2.270 + FORA (Selvaraju et al., 2024) 20.594 0.831 0.189 5.833 6.817 5.306 14.359 1.949 + ∆-DiT (Chen et al., 2024) 22.927 0.835 0.141 8.500 6.733 7.345 12.766 2.192 + TeaCache (Liu et al., 2025a) 25.856 0.915 0.088 8.417 7.183 7.536 11.245 2.488 + RAS (Liu et al., 2025e) 25.072 0.888 0.116 8.250 6.433 6.996 15.268 1.833 + ToCa (Zou et al., 2024a) 23.407 0.840 0.168 8.267 6.217 6.909 22.080 1.267 + Ours (RegionE) 28.654 0.935 0.064 8.517 7.167 7.585 10.647 2.628

Qwen-Image-Edit (Wu et al., 2025) inf 1.000 0.000 8.883 7.683 8.136 32.161 1.000 + Stepskip 26.344 0.897 0.076 8.783 7.733 8.128 17.575 1.830 + FORA (Selvaraju et al., 2024) 24.578 0.864 0.104 8.600 7.550 7.930 17.836 1.803 + ∆-DiT (Chen et al., 2024) 23.745 0.829 0.120 8.450 7.267 7.687 17.496 1.838 + TeaCache (Liu et al., 2025a) 25.993 0.891 0.084 8.700 7.733 8.120 16.391 1.962 + RAS (Liu et al., 2025e) 25.579 0.881 0.095 8.867 7.400 7.996 22.341 1.440 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 29.388 0.938 0.047 8.967 7.767 8.242 15.446 2.082

- Table 12: Comparison of RegionE and other baselines on the text-change task of GEdit-Bench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.293 8.091 7.900 28.027 1.000 + Stepskip 30.069 0.955 0.032 8.222 8.192 7.951 12.331 2.273 + FORA (Selvaraju et al., 2024) 26.368 0.941 0.049 7.000 7.899 6.926 14.375 1.950 + ∆-DiT (Chen et al., 2024) 28.615 0.953 0.032 8.515 8.222 8.171 12.768 2.195 + TeaCache (Liu et al., 2025a) 31.420 0.967 0.023 8.222 8.192 7.925 11.254 2.491 + RAS (Liu et al., 2025e) 28.434 0.939 0.042 7.929 7.970 7.649 15.270 1.835 + ToCa (Zou et al., 2024a) 26.305 0.902 0.078 7.949 7.707 7.609 21.723 1.290 + Ours (RegionE) 32.404 0.968 0.020 8.212 8.242 8.002 10.237 2.738

Qwen-Image-Edit (Wu et al., 2025) - - - 9.192 8.394 8.606 32.071 1.000 + Stepskip 29.577 0.929 0.047 8.828 8.222 8.202 17.519 1.831 + FORA (Selvaraju et al., 2024) 27.408 0.909 0.061 8.818 8.303 8.192 17.790 1.803 + ∆-DiT (Chen et al., 2024) 25.837 0.881 0.072 8.879 8.333 8.259 17.432 1.840 + TeaCache (Liu et al., 2025a) 29.126 0.932 0.047 8.778 8.222 8.184 16.539 1.939 + RAS (Liu et al., 2025e) 26.732 0.912 0.061 8.889 8.010 8.187 22.302 1.438 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM + Ours (RegionE) 31.357 0.950 0.033 8.838 8.313 8.260 14.813 2.165

- Table 13: Comparison of RegionE and other baselines on the background-change task of GEditBench, evaluated in terms of quality and efficiency.

Against Vanilla GPT-4o Score Efficiency

Model

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

Step1X-Edit (Liu et al., 2025d) - - - 8.575 7.175 7.722 27.886 1.000 + Stepskip 21.011 0.812 0.218 8.625 6.975 7.635 12.267 2.273 + FORA (Selvaraju et al., 2024) 15.897 0.719 0.372 6.500 7.125 6.104 14.285 1.952 + ∆-DiT (Chen et al., 2024) 18.641 0.781 0.271 8.375 6.625 7.338 12.687 2.198 + TeaCache (Liu et al., 2025a) 23.549 0.866 0.151 8.375 6.725 7.379 11.163 2.498 + RAS (Liu et al., 2025e) 25.468 0.840 0.169 8.425 6.725 7.372 15.206 1.834 + ToCa (Zou et al., 2024a) 22.925 0.768 0.255 8.200 6.175 6.995 22.631 1.232

- + Ours (RegionE) 29.076 0.917 0.091 8.500 7.125 7.675 11.324 2.463

Qwen-Image-Edit (Wu et al., 2025) - - - 9.125 8.200 8.603 32.221 1.000 + Stepskip 25.088 0.854 0.133 9.175 7.975 8.511 17.603 1.830 + FORA (Selvaraju et al., 2024) 22.473 0.802 0.184 8.775 7.875 8.252 17.815 1.809 + ∆-DiT (Chen et al., 2024) 21.263 0.750 0.210 8.825 7.850 8.281 17.552 1.836 + TeaCache (Liu et al., 2025a) 24.016 0.852 0.147 8.850 7.950 8.281 16.492 1.954 + RAS (Liu et al., 2025e) 26.550 0.858 0.128 9.100 7.450 8.153 22.411 1.438 + ToCa (Zou et al., 2024a) OOM OOM OOM OOM OOM OOM OOM OOM

- + Ours (RegionE) 30.462 0.939 0.053 9.175 8.050 8.547 16.694 1.930

- Table 14: Comparison of RegionE and other baselines on the Character Reference task of KontextBench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

FLUX.1 Kontext (Labs et al., 2025) - - - 7.549 6.642 6.664 14.677 1.000 + Stepskip 18.793 0.730 0.238 7.741 6.803 6.917 8.502 1.726 + FORA (Selvaraju et al., 2024) 17.898 0.697 0.275 7.617 6.788 6.813 7.494 1.958 + ∆-DiT (Chen et al., 2024) 15.560 0.604 0.387 7.668 6.451 6.704 6.737 2.178 + TeaCache (Liu et al., 2025a) 20.313 0.770 0.197 7.865 6.565 6.842 6.271 2.341 + RAS (Liu et al., 2025e) 21.320 0.752 0.214 7.632 6.352 6.657 8.211 1.788 + ToCa (Zou et al., 2024a) 19.596 0.679 0.298 7.570 6.047 6.454 11.279 1.301 + Ours (RegionE) 26.980 0.880 0.086 7.637 6.611 6.715 6.406 2.291

- Table 15: Comparison of RegionE and other baselines on the Instruction Editing-Global task of KontextBench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

FLUX.1 Kontext (Labs et al., 2025) - - - 7.023 6.798 6.380 14.688 1.000 + Stepskip 23.957 0.797 0.132 7.000 6.870 6.435 8.516 1.725 + FORA (Selvaraju et al., 2024) 22.611 0.760 0.159 7.092 6.840 6.497 7.506 1.957 + ∆-DiT (Chen et al., 2024) 18.687 0.659 0.252 7.073 6.882 6.574 6.754 2.175 + TeaCache (Liu et al., 2025a) 27.206 0.842 0.101 7.294 6.885 6.626 6.251 2.350 + RAS (Liu et al., 2025e) 24.845 0.778 0.157 7.302 6.866 6.668 8.221 1.787 + ToCa (Zou et al., 2024a) 23.030 0.711 0.218 7.179 6.588 6.483 11.412 1.287 + Ours (RegionE) 30.403 0.886 0.071 7.126 6.943 6.572 6.379 2.303

- Table 16: Comparison of RegionE and other baselines on the Instruction Editing-Local task of KontextBench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

FLUX.1 Kontext (Labs et al., 2025) - - - 6.779 6.909 5.817 14.677 1.000 + Stepskip 31.147 0.913 0.058 6.839 6.887 5.872 8.510 1.725 + FORA (Selvaraju et al., 2024) 29.279 0.895 0.072 6.800 6.901 5.873 7.491 1.959 + ∆-DiT (Chen et al., 2024) 23.390 0.824 0.130 6.822 6.829 5.846 6.751 2.174 + TeaCache (Liu et al., 2025a) 33.341 0.938 0.040 6.942 6.800 5.896 6.113 2.401 + RAS (Liu et al., 2025e) 30.088 0.907 0.063 6.945 6.921 5.972 8.219 1.786 + ToCa (Zou et al., 2024a) 26.996 0.855 0.112 6.851 6.635 5.790 11.231 1.307 + Ours (RegionE) 36.334 0.959 0.025 6.889 6.880 5.917 5.799 2.531

- Table 17: Comparison of RegionE and other baselines on the Style Reference task of KontextBench, evaluated in terms of quality and efficiency.

Model

Against Vanilla GPT-4o Score Efficiency

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

FLUX.1 Kontext (Labs et al., 2025) - - - 6.810 6.556 6.331 14.684 1.000 + Stepskip 18.606 0.678 0.290 6.333 6.381 5.947 8.501 1.727 + FORA (Selvaraju et al., 2024) 17.508 0.631 0.329 6.222 6.413 5.667 7.476 1.964 + ∆-DiT (Chen et al., 2024) 14.639 0.525 0.450 6.397 6.873 6.108 6.731 2.182 + TeaCache (Liu et al., 2025a) 19.781 0.712 0.264 6.444 6.286 5.832 6.261 2.345 + RAS (Liu et al., 2025e) 19.481 0.638 0.343 6.603 6.381 6.031 8.202 1.790 + ToCa (Zou et al., 2024a) 18.245 0.553 0.439 6.000 6.175 5.668 11.480 1.279 + Ours (RegionE) 24.433 0.811 0.165 6.921 6.571 6.277 6.411 2.291

- Table 18: Comparison of RegionE and other baselines on the Text Editing task of KontextBench, evaluated in terms of quality and efficiency.

Against Vanilla GPT-4o Score Efficiency

Model

PSNR↑ SSIM ↑ LPIPS ↓ G-SC ↑ G-PQ ↑ G-O ↑ Latency (s) ↓ Speedup ↑

FLUX.1 Kontext (Labs et al., 2025) - - - 7.826 7.913 7.295 14.697 1.000 + Stepskip 30.950 0.943 0.033 7.717 7.750 7.142 8.535 1.722 + FORA (Selvaraju et al., 2024) 28.976 0.915 0.044 7.696 7.543 7.067 7.520 1.954 + ∆-DiT (Chen et al., 2024) 23.931 0.839 0.085 7.315 7.554 6.823 6.779 2.168 + TeaCache (Liu et al., 2025a) 31.283 0.955 0.026 7.620 7.696 7.076 6.290 2.336 + RAS (Liu et al., 2025e) 27.504 0.913 0.048 7.598 7.402 6.971 8.238 1.784 + ToCa (Zou et al., 2024a) 25.342 0.857 0.093 7.326 7.500 6.791 11.206 1.312 + Ours (RegionE) 34.141 0.962 0.018 7.815 7.761 7.211 5.767 2.548

