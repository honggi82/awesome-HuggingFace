# arXiv:2403.12032v2[cs.CV]19Mar2024

## Generic 3D Diffusion Adapter Using Controlled Multi-View Editing

HANSHENG CHEN, Stanford University, USA RUOXI SHI, UC San Diego, USA YULIN LIU, UC San Diego, USA BOKUI SHEN, Apparate Labs, USA JIAYUAN GU, UC San Diego, USA GORDON WETZSTEIN, Stanford University, USA HAO SU, UC San Diego, USA LEONIDAS GUIBAS, Stanford University, USA Demo & code: https://lakonik.github.io/mvedit

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

###### MVEdit

MVEdit

###### “Turn her into a cyborg”

###### “As a Zelda cosplay, blue outfit”

###### “Tomb raider Lara Croft, high quality” …

Text-guided 3D-to-3D (3.8 min/29 steps) + texture super-res (37 sec/8 steps)

Instruct 3D-to-3D (4.4 min/32 steps) + texture super-res (41 sec/9 steps)

3D input

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

Zero123++ MVEdit

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### Initial 12 views×3 groups

[Figure 34]

[Figure 35]

[Figure 36]

Image-to-3D (1.9 min/12 steps) + Image-guided texture super-res (55 sec/10 steps)

Image input (text-to-image)

(1 min/(75 steps×6 passes))

[Figure 37]

[Figure 38]

[Figure 39]

MVEdit

[Figure 40]

[Figure 41]

…“red and white” … …“blue” … …“lEGO” …

“A blue Volkswagen Beetle

[Figure 42]

[Figure 43]

Text-guided re-texturing (1.8 min/24 steps)

StableSSDNeRF

GT3 racing car” MVEdit

MVEdit

###### Initial Text-to-3D (NeRF)

Text-guided 3D-to-3D (2.3 min/17 steps) + texture super-res (54 sec/12 steps)

Text input

Image-guided re-texturing (2 min/24 steps)

(1.4 sec/32 steps)

Fig. 1. Examples showcasing MVEdit’s generality across various 3D tasks, with associated inference times (on an RTX A6000) and the number of timesteps. For image-to-3D, note that the initial views by Zero123++ are not strictly 3D consistent (causing the failures in Fig. 9), an issue remedied by MVEdit.

Open-domain 3D object synthesis has been lagging behind image synthesis due to limited data and higher computational complexity. To bridge this gap, recent works have investigated multi-view diffusion but often fall short in either 3D consistency, visual quality, or efficiency. This paper proposes MVEdit, which functions as a 3D counterpart of SDEdit, employing ancestral sampling to jointly denoise multi-view images and output high-quality textured meshes. Built on off-the-shelf 2D diffusion models, MVEdit achieves

3D consistency through a training-free 3D Adapter, which lifts the 2D views of the last timestep into a coherent 3D representation, then conditions the 2D views of the next timestep using rendered views, without uncompromising visual quality. With an inference time of only 2-5 minutes, this framework achieves better trade-off between quality and speed than score distillation. MVEdit is highly versatile and extendable, with a wide range of applications including text/image-to-3D generation, 3D-to-3D editing, and high-quality texture synthesis. In particular, evaluations demonstrate state-of-the-art performance in both image-to-3D and text-guided texture generation tasks. Additionally, we introduce a method for fine-tuning 2D latent diffusion models on small 3D datasets with limited resources, enabling fast low-resolution text-to-3D initialization.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. This is the author’s version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record was published in , https://doi.org/ XXXXXXX.XXXXXXX.

CCS Concepts: • Computing methodologies → Computer graphics; Artificial intelligence.

Additional Key Words and Phrases: diffusion models, 3D generation and editing, texture synthesis, radiance fields, differentiable rendering

###### ACM Reference Format:

Hansheng Chen, Ruoxi Shi, Yulin Liu, Bokui Shen, Jiayuan Gu, Gordon Wetzstein, Hao Su, and Leonidas Guibas. 2024. Generic 3D Diffusion Adapter Using Controlled Multi-View Editing. In . ACM, New York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX

- 1 INTRODUCTION

Data-driven 3D object synthesis in an open domain has gained wide research interest at the intersection of computer graphics and artificial intelligence. Among the recent advances in generative modeling, diffusion models represent a significant leap in image generation and editing [Ho et al. 2020; Ho and Salimans 2021; Lugmayr et al. 2022; Po et al. 2024; Rombach et al. 2022; Zhang et al. 2023]. However, unlike 2D image models that benefit from massive datasets [Schuhmann et al. 2022] and a well-established grid representation, training a 3D-native diffusion model from scratch needs to grapple with the scarcity of large-scale datasets and the absence of a unified, neuralnetwork-friendly representation, and has therefore been limited to closed domains or lower resolution [Chen et al. 2023b; Dupont et al. 2022; Müller et al. 2023; Wang et al. 2023b; Zheng et al. 2023].

Multi-view diffusion has emerged as a promising approach to bridge the gap between 2D and 3D generation. Yet, when adapting pretrained image diffusion models into multi-view generators, precise 3D consistency is not often guaranteed due to the absence of a 3D-awaremodelarchitecture.Score distillation sampling (SDS) [Poole

et al. 2023] further enforces 3D awareness by optimizing a neural radiance field (NeRF) [Mildenhall et al. 2020] or mesh with multi-view diffusion priors, but they typically require hours-long optimization and often fall short in diversity and visual quality when compared to standard ancestral sampling (i.e., progressive denoising).

To address these challenges, we present a generic solution for adapting pre-trained image diffusion models for 3D-aware diffusion under the ancestral sampling paradigm. Inspired by ControlNet [Zhang et al. 2023], we introduce the Controlled Multi-View Editing (MVEdit) framework. Without fine-tuning, MVEdit simply extends the frozen base model by incorporating a novel trainingfree 3D Adapter. Inserted in between adjacent denoising steps, the 3D Adapter fuses multi-view 2D images into a coherent 3D representation, which in turn controls the subsequent 2D denoising steps without compromising image quality, thus enabling 3D-aware cross-view information exchange.

Analogous to the 2D SDEdit [Meng et al. 2022], MVEdit is a highly versatile 3D editor. Notably, when based on the popular Stable Diffusion image model [Rombach et al. 2022], MVEdit can leverage a wealth of community modules to accomplish a diverse array of 3D synthesis tasks based on multi-modal inputs.

Furthermore, MVEdit can utilize a real 3D-native generative model for geometry initialization. We therefore introduce StableSSDNeRF, a fast text-to-3D diffusion model fine-tuned from 2D Stable Diffusion, to complement MVEdit in high-quality domain-specific 3D generation.

To summarize, our main contributions are as follows:

- • We propose MVEdit, a generic framework for building 3D Adapters on top of image diffusion models, implementable on Stable Diffusion without the necessity for fine-tuning.
- • Utilizing MVEdit, we develop a versatile 3D toolkit and showcase its wide-ranging applicability in various 3D generation and editing tasks, as illustrated in Fig. 1.
- • Additionally, we introduce StableSSDNeRF, a fast, easy-to-finetune text-to-3D diffusion model for initializing MVEdit.

2 RELATED WORK

- 2.1 3D-Native Diffusion Models

We define 3D-native diffusion models as injecting noise directly into the 3D representations (or their latents) during the diffusion process. Early works [Bautista et al. 2022; Dupont et al. 2022] have explored training diffusion models on low-dimensional latent vectors of 3D representations, but are highly limited in model capacity. A more expressive approach is training diffusion models on triplane representations [Chan et al. 2022], which works reasonably well on closed-domain data [Chen et al. 2023b; Gupta et al. 2023; Shue et al. 2023; Wang et al. 2023b]. Directly working on 3D grid representations is more challenging due to the cubic computation cost [Müller

- et al. 2023], so an improved multi-stage sparse volume diffusion model is proposed in [Zheng et al. 2023] and also adopted in [Liu
- et al. 2024b]. In general, 3D-native diffusion models face the challenge of limited data, and sometimes the extra cost of converting existing data to 3D representations (e.g., NeRF). These challenges are partially addressed by our proposed StableSSDNeRF (Section 5).

- 2.2 Novel-/Multi-view Diffusion Models

Trained on multi-view images of 3D scenes, view diffusion models inject noise into the images (or their latents) and thus benefit from existing 2D diffusion research. [Watson et al. 2023] have demonstrated the feasibility of training a conditioned novel view generative model using purely 2D architectures. Subsequent works [Liu et al. 2023a; Long et al. 2024; Shi et al. 2023, 2024] achieve open-domain novel-/multi-view generation by fine-tuning the pre-trained 2D Stable Diffusion model [Rombach et al. 2022]. However, 3D consistency in these models is generally weak, as it is enforced only in a data-driven manner, lacking any inherent architectural bias.

To introduce 3D-awareness, [Anciukevicius et al. 2023; Tewari et al. 2023; Xu et al. 2024] lift image features into 3D NeRF to render the denoised views. However, they are prone to blurriness due to the information loss during the 2D-3D-2D conversion. [Chan et al. 2023; Liu et al. 2024a] propose 2D denoising networks conditioned on 3D projections, which generate crisp images but with slight

- 3D inconsistency. Inspired by the latter approach, MVEdit takes a significant step further by directly adopting pre-trained 2D diffusion models without fine-tuning, and enabling high-quality mesh output.

2.3 Diffusion Models with 3D Optimization

While the aforementioned approaches rely solely on feed-forward networks, optimization-based methods sometimes offer higher quality and greater flexibility, albeit at the cost of longer runtimes. [Poole et al. 2023] introduced the seminal Score Distillation Sampling (SDS),

which optimizes a NeRF using a pretrained image diffusion model

Denoised multi-view

Noisy multi-view

2D Net

- as a loss function. Some of its issues, such as limited resolution, the Janus problem, over-saturated colors, and mode-seeking behavior, have been addressed in subsequent works [Chen et al. 2023a; Lin et al. 2023; Qian et al. 2024; Sun et al. 2024; Wang et al. 2023a]. Despite improvements, SDS and its variants remain time-consuming and often yield a degraded distribution compared to ancestral sampling. [Haque et al. 2023; Zhou and Tulsiani 2023] alternate between ancestral sampling and optimization, which is also inefficient. A faster approach is seen in NerfDiff [Gu et al. 2023], which performs ancestral sampling only once and optimizes a NeRF within each timestep. However, if dealing with diverse open-domain objects, it would encounter the same blurriness issues due to NeRF disrupting the sampling process, a challenge to be addressed in this work.

(a) Basic 2D denoising network

Denoised

Noisy multi-view

2D Net

3D NeRF multi-view

(b) 3D-aware denoising network in the style of NerfDiff, DMV3D, MAS

###### Proposed architecture

Denoised multi-view

Noisy multi-view

###### 3D NeRF 2D Net

2D Net

(c) Blur-free 3D-aware denoising network with skip connection

DPMSolver

Denoised

Noisy multi-view

2D Net multi-view

3D NeRF

3 MVEDIT: CONTROLLED MULTI-VIEW EDITING

(d) Simplified 3D Adapter re-using last denoised RGB

As discussed in Section 2.2 and 2.3, although appending a 3D NeRF to the denoising network (Fig. 2 (b)) guarantees 3D consistency, it often leads to blurry results since NeRF typically averages the inconsistent multi-view inputs, resulting in inevitable loss. For latent diffusion models [Rombach et al. 2022], the additional VAE decoding and encoding process can further exacerbate this issue.

Fig. 2. Comparison among 3D-aware multi-view denoising architectures. Adding skip connection around the 3D NeRF in (c) mitigates the potential blurriness issue in (b), but requires two 2D UNet passes within the same denoising timestep when extending the off-the-shelf 2D Stable Diffusion; our simplified architecture in (d) re-uses the denoised multi-view images from the last denoising timestep to reconstruct the 3D NeRF.

To address the 3D consistency challenge without interrupting the information flow from the input noisy view to the denoised view, we propose a new architecture containing a skip connection around the 3D model (Fig. 2 (c)) and its simplified version (Fig. 2 (d)). Based on the simplified architecture, we introduce the MVEdit framework shown in Fig. 4, and provide a detailed elaboration below.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Arch. in Fig. 2 (b)

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- 3.1 Framework Overview

- 3.1.1 Preliminaries: SDEdit Using Single-Image Diffusion. Ignoring the red and orange flow in Fig. 4, the remaining blue flow depicts the original SDEdit sampling process using the base text-to-image

Arch. in

- Fig. 2 (d) (ours)
- Fig. 3. Comparison between the two architectures, based on the text-

guided 3D-to-3D pipeline with 𝑡start = 0.78𝑇. Rendered RGB images 𝑥RGBrend across different timesteps are shown to visualize the sampling process.

where𝑧 denotes the internal states of the solver. Recursive denoising can be executed by repeating Eq. (2) and Eq. (3) until reaching the denoised state 𝑥(0), thus completing the ancestral sampling process.

3.1.2 MVEdit Using Multi-View Diffusion. In MVEdit, we adapt the single-image diffusion model into a 3D-consistent multi-view diffusion model via a novel 3D Adapter, depicted as the red flow in

- Fig. 4. For each timestep, we first obtain the denoised images {𝑥ˆ𝑖} of all the predefined views with known camera parameters {𝑝𝑖}, where 𝑖 denotes the view index. Then, a 3D representation parameterized by 𝜙 can be reconstructed from these denoised views. In this paper, we employ optimization-based reconstruction approaches, using InstantNGP [Müller et al. 2022] for NeRF or DMTet [Shen et al. 2021] for mesh. Thus, the 3D parameters 𝜙ˆ can be estimated by

- 2D diffusion model. For latent diffusion models, we omit the VAE encoding/decoding process for brevity. Given an initial RGB image 𝑥init ∈ R𝐶×𝐻×𝑊 , SDEdit first perturbs the image with random noise 𝜖 ∼ N(0,𝐼) following the Gaussian diffusion process:

Init 𝑡𝑡 = 0.78𝑇𝑇 𝑡𝑡 = 0.54𝑇𝑇 𝑡𝑡 = 0.30𝑇𝑇 𝑡𝑡 = 0.06𝑇𝑇

“A zebra rocking horse, high quality” …

𝑥(𝑡) = 𝛼(𝑡)𝑥init + 𝜎(𝑡)𝜖, (1)

where 𝑡 ← 𝑡start ∈ [0,𝑇] is a user-specified starting timestep, 𝛼(𝑡),𝜎(𝑡) are scalars determined by the noise schedule, and 𝑥(𝑡) de-

notes the noisy image. For the denoising step, the UNet 𝜖ˆ 𝑥(𝑡),𝑐,𝑡

predicts the noise component 𝜖ˆ from the noisy image 𝑥(𝑡), the condition 𝑐 (i.e., text prompt), and the timestep 𝑡. Afterwards, we can derive the denoised image 𝑥ˆ from the predicted noise 𝜖ˆ:

𝑥(𝑡) − 𝜎(𝑡)𝜖ˆ 𝑥(𝑡),𝑐,𝑡 𝛼(𝑡)

𝑥ˆ =

. (2)

To move forward onto the next step, a generic diffusion ODE or SDE solver [Song et al. 2021] can be applied to yield a less noisy image 𝑥(𝑡−Δ𝑡) at a previous timestep 𝑡 − Δ𝑡. In this paper, we adopt the DPMSolver++ [Lu et al. 2022], and the solver step can be written as:

𝑥(𝑡−Δ𝑡) ← DPMSolver𝑧 𝑥,𝑡,𝑥 ˆ (𝑡) , (3)

Input Output

Initialization 𝑡𝑡 = 𝑡𝑡start 𝑡𝑡 = 𝑡𝑡start − Δ𝑡𝑡

𝑡𝑡 = 0

| | | | | |
|---|---|---|---|---|
|Noisy RGB𝑥𝑥 𝑡𝑡<br><br>| |Denoised RGB 𝑥𝑥ctrl<br><br>Noisy RGB𝑥𝑥 𝑡𝑡<br><br>| |Denoised RGB 𝑥𝑥ctrl|

…

Text prompt

Initial RGB𝑥𝑥init

Denoised RGB 𝑥𝑥ctrl

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Weighted blend B

Weighted blend B

Weighted blend B

DPM Solver

DPM Solver

Add noise

…

##### …

Exported mesh

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

ControlNets

ControlNets

…

View 1 View 1 View 1 View 1

NeRF/ Mesh

NeRF/ Mesh

NeRF/ Mesh

Mesh

Rendered RGBD𝑥𝑥rend

Rendered RGBD𝑥𝑥rend

Rendered RGBD𝑥𝑥rend

Progress=0% (Rendering resolution) NeRF=128 / mesh=512 30% NeRF=256 / mesh=512 60% Mesh=512 100%

Fig. 4. The initialization and ancestral sampling process of MVEdit. The original single-image SDEdit is shown in blue, the additional 3D Adapter in red, and extra conditioning in orange. For brevity, only the first view is depicted, and VAE encoding/decoding is omitted in cases involving latent diffusion.

where 𝑥RGBrend𝑖 denotes the RGB channels of the rendered image, 𝑥ˆ𝑖ctrl is the denoised image with reduced ControlNet weight, and 𝑤(𝑡)

minimizing the rendering loss against the denoised images {𝑥ˆ𝑖}: 𝜙ˆ = argmin

Lrend({𝑥ˆ𝑖,𝑝𝑖},𝜙). (4)

is a time-dependant blending weight. The blended image 𝑥ˆ𝑖blend is then treated as the denoised image to be fed into the DPMSolver.

𝜙

Details on the loss and optimization will be described in Section 3.2. With the reconstructed 3D representation, a new set of images with

RGBD channels {𝑥𝑖rend} can be rendered from the views. These strictly 3D-consistent renderings are the results of multi-view ag-

3.2 Robust NeRF/Mesh Optimization

The 3D Adapter faces the challenge of potentially inconsistent multiview inputs, especially at the early denoising stage. Existing surface optimization approaches, such as NeuS [Wang et al. 2021a], are not designed to address the inconsistency. Therefore, we have developed various techniques for the robust optimization of InstantNGP NeRF [Müller et al. 2022] and DMTet mesh [Shen et al. 2021], using enhanced regularization and progressive resolution.

gregation, and tend to be blurry at early denoising steps. By feeding 𝑥𝑖rend to the ControlNets [Zhang et al. 2023] as a conditioning signal, a sharper image 𝑥ˆ𝑖ctrl can be obtained via a second pass through the controlled UNet 𝜖ˆctrl 𝑥(𝑡),𝑐𝑖,𝑡,𝑥𝑖rend :

𝑥𝑖(𝑡) − 𝜎(𝑡)𝜖ˆctrl 𝑥𝑖(𝑡),𝑐𝑖,𝑡,𝑥𝑖rend 𝛼(𝑡)

𝑥ˆ𝑖ctrl =

. (5)

- 3.2.1 Rendering. For each NeRF optimization iteration, we randomly sample a 128×128 image patch from all camera views. Unlike [Poole et al. 2023] that computes the normal from NeRF density gradients, we compute patch-wise normal maps from the rendered depth maps, which we find to be faster and more robust. For mesh rendering, we obtain the surface color by querying the same InstantNGP neural field used in NeRF. For both NeRF and mesh, Lambertian shading is applied in the linear color space prior to tonemapping, with random point lights assigned to their respective views.
- 3.2.2 RGBA Losses. For both NeRF and mesh, we employ RGB and Alpha rendering losses to optimize the 3D parameters 𝜙 so

that the rendered views {𝑥𝑖rend} match the target denoised views {𝑥ˆ𝑖}. For RGB, we employ a combination of pixel-wise L1 loss and patch-wise LPIPS loss [Zhang et al. 2018]. For Alpha, we predict the target Alpha channel from {𝑥ˆ𝑖} using an off-the-shelf background removal network [Lee et al. 2022] as in Magic123 [Qian et al. 2024]. Additionally, we soften the predicted Alpha map using Gaussian blur to prevent NeRF from overfitting the initialization.

- 3.2.3 Normal Losses. To avoid bumpy surfaces, we apply an L1.5 total variation (TV) regularization loss on the rendered normal maps:

Therefore, 3D-consistent sampling can be achieved by replacing 𝑥ˆ𝑖 with 𝑥ˆ𝑖ctrl in the solver step in Eq. (3). Eq. (5) effectively formulates the two-pass architecture shown in Fig. 2 (c), where the skip connection is essentially re-feeding the noisy multi-view into the second UNet. In practice, running two passes within a single denoising step appears redundant. Therefore, we use the rendered views from the last denoising step to condition the UNet of the next denoising step, which corresponds to the simplified architecture in Fig. 2 (d).

Empirically, with Stable Diffusion [Rombach et al. 2022] as the base model, we find that off-the-shelf Tile (conditioned on blurry RGB images) and Depth (conditioned on depth maps) ControlNets can already handle RGB and depth conditioning for consistent multiview generation, eliminating the necessity of training a custom ControlNet. However, recursive self-conditioning may amplify some unfavorable bias within Stable Diffusion, such as color drifting or over-sharpening/smoothing. Therefore, we adopt time-dependant dynamic ControlNet weights. Notably, we reduce the𝑇𝑖𝑙𝑒 ControlNet weight when 𝑡 is large, otherwise the small denominator 𝛼(𝑡) in Eq. (5) at this time would significantly amplify any bias in the numerator. Reducing the ControlNet weight, however, leads to worse

- 3D consistency. To mitigate the consistency issue, we introduce an additional weighted blending operation for 𝑡 > 0.4𝑇 only:

#### LN = ∑︁ 𝑐ℎ𝑤

1.5

𝑤ℎ𝑤 · ∇ℎ𝑤𝑛𝑐ℎ𝑤rend

, (7)

𝑥ˆ𝑖blend = 𝑤(𝑡)𝑥RGBrend𝑖 + (1 − 𝑤(𝑡))𝑥ˆ𝑖ctrl, (6)

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

where 𝑛𝑐ℎ𝑤rend ∈ R denotes the value of the 𝐶 × 𝐻 ×𝑊 normal map

- at index (𝑐,ℎ,𝑤), ∇ℎ𝑤𝑛𝑐ℎ𝑤rend ∈ R2 is the gradient of the normal map w.r.t. (ℎ,𝑤), and 𝑤ℎ𝑤 ∈ [0, 1] is the value of a foreground mask with edge erosion. For image-to-3D, however, we can predict target

normal maps from the initial RGB images {𝑥𝑖init} using [Eftekhar et al. 2021], following [Sun et al. 2024]. In this case, we modify the

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

regularization loss in Eq. (7) into a normal regression loss:

LN = ∑︁ 𝑐ℎ𝑤

1.5

𝑤ℎ𝑤 · ∇ℎ𝑤𝑛𝑐ℎ𝑤rend − ∇ℎ𝑤𝑛ˆ𝑐ℎ𝑤

, (8)

where 𝑛ˆ𝑐ℎ𝑤 denotes the value of the predicted normal map at index (𝑐,ℎ,𝑤). Additionally, we also employ a patch-wise LPIPS loss between the high-pass components of both the rendered and predicted normal maps, akin to the patch-wise RGB loss.

0.96𝑇𝑇 0.87𝑇𝑇 0.78𝑇𝑇 0.69𝑇𝑇 0.48𝑇𝑇 Original

“Tomb raider Lara Croft, high quality” …

Fig. 5. Text-guided 3D-to-3D using the same seed but different 𝑡start.

- 3.2.4 Ray Entropy Loss for NeRF. To mitigate fuzzy NeRF geometry, we propose a novel ray entropy loss based on the probability of sample contribution. Unlike previous works [Kim et al. 2022; Metzer et al. 2023] that compute the entropy of opacity distribution or alpha map, we consider the ray density function:

the initial timestep 𝑡start of these pipelines is adjustable, allowing control over the extent of editing, as shown in Fig. 5.

- 4.1 3D Synthesis Pipelines

3D synthesis pipelines, which fully utilize robust NeRF/mesh optimization techniques, begin with 32 views surrounding the object. These are progressively reduced to 9 views, helping to alleviate the computational cost of multi-view denoising at later stages. NeRF is always adopted as the initial 3D representation, with its density field converted into a DMTet mesh representation upon reaching 60% completion. Various pipeline variants can then be constructed with unique input modalities and conditioning mechanisms.

- 4.1.1 Text-Guided 3D-to-3D. Given an input 3D object, we randomly sample 32 surrounding cameras and render the initial multiview images to initialize the NeRF. No additional modules are required, as Stable Diffusion is inherently conditioned on text prompts.
- 4.1.2 Instruct 3D-to-3D. Inspired by Instruct-NeRF2NeRF [Haque et al. 2023], we introduce the mesh-based Instruct 3D-to-3D pipeline. Extra image-conditioning is employed by feeding the initial multiview images into an InstructPix2Pix ControlNet [Brooks et al. 2023; Zhang et al. 2023].
- 4.1.3 Image-to-3D. Using Zero123++ [Shi et al. 2023] to generate initial multi-view images, MVEdit can lift these views into a high-quality mesh by resolving the initial 3D inconsistency. The original appearance can be preserved via image conditioning using IP-Adapter [Ye et al. 2023] and cross-image attention [Alaluf et al. 2023; Shi et al. 2023]. Since Zero123++ can only generate a fixed set of 6 views, we augment the initialization by mirroring the input and repeating the generation process three times, yielding a total of 36 images. The pose of the input view can also be estimated using correspondences to the generated views, so that we have 36 + 1 initial images in total. As the sampling process begins, this number is reduced to 32.

- 4.2 Re-Texturing Pipelines Given a frozen 3D mesh, MVEdit can generate high-quality textures from scratch (initialized with random Gaussian noise and 𝑡start =

𝑝(𝑠) = 𝑇 (𝑠)𝜎(𝑠), (9)

where 𝑠 denotes the distance, 𝜎(𝑠) is the volumetric density and 𝑇 (𝑠) = exp−∫ 𝑠

0 𝜎(𝑠) d𝑠 is the ray transmittance. The integral of𝑝(𝑠) equals the alpha value of the pixel, i.e., 𝑎 = ∫ +inf

0 𝑝(𝑠) d𝑠, which is less than 1. Therefore, the background probability is 1 − 𝑎 and a corresponding correction term needs to be added when computing the continuous entropy of the ray as the loss function:

∫ +inf

Lray = ∑︁

1 − 𝑎𝑟 𝑑

−𝑝𝑟 (𝑠) log𝑝𝑟 (𝑠) d𝑠 − (1 − 𝑎𝑟) log

, (10)

0

𝑟

background correction

where 𝑟 is the ray index, and 𝑑 is a user-defined “thickness” of an imaginative background shell, which can be adjusted to balance foreground-to-background ratio.

- 3.2.5 Mesh Smoothing Losses. As per common practice, we employ the Laplacian smoothing loss [Sorkine et al. 2004] and normal consistency loss to further regularize the mesh extracted from DMTet.
- 3.2.6 Implementation Details. The weighted sum of the aforementioned loss functions is utilized to optimize the 3D representation. At each denoising step, we carry forward the 3D representation from the previous step and perform additional iterations of Adam [Kingma and Ba 2015] optimization (96 for 3D or 48 for texture-only). During the ancestral sampling process, the rendering resolution progressively increases from 128 to 256, and finally to 512 when NeRF is converted into a mesh (for texture-only the resolution is consistently 512). When the rendering resolution is lower than the diffusion resolution 512, we employ RealESRGAN-small [Wang et al. 2021b] for efficient super-resolution.

- 4 MVEDIT APPLICATIONS AND PIPELINES

In this section, we present details on various MVEdit pipelines. Their respective applications are showcased in Fig. 1, with details on inference times and the number of timesteps. Same as SDEdit,

###### Solver

𝑇), or edit existing textures with a user-defined 𝑡start. The number of views is scheduled to decrease from 32 to 7. This process is faster as it only requires optimizing the texture field. In this paper, we demonstrate basic re-texturing pipelines using text and image guidance (the latter using IP-Adapter and cross-image attention), while more pipelines can also be customized.

Decoder

[Figure 79]

[Figure 80]

###### +LoRA

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

12x40x40

48x80x80

3x16x80x80

4x120x40 4x120x40 Latent code

- 4.3 Texture Super-Resolution Pipelines

The texture super-resolution pipelines require only 6 views throughout the sampling process. We employ the Tile ControlNet, originally trained for super-resolution, to condition the denoising UNet on the initial renderings. Consequently, the existing𝑇𝑖𝑙𝑒 ControlNet in our 3D Adapter can be disabled to avoid redundancy. Additionally, image guidance can be implemented using cross-image attention, facilitating low-level detail transfer from a high-resolution guidance image. Adopting the SDE-DPMSolver++ [Lu et al. 2022], these pipelines serve as a final boost to the 3D synthesis results.

### “A yellow sports car with black stripes”

Fig. 6. Architecture of StableSSDNeRF, consisting of a frozen Stable Diffusion UNet with LoRA fine-tuning, and a triplane latent decoder.

- Table 1. Comparison on image-to-3D generation. SyncDreamer and DreamCraft3D are not evaluated on the 248 objects due to slow inference.

Method

248 GSO images 33 in-the-wild images Infer. LPIPS↓ CLIP↑ FID↓ Img-3DAlign. ↑ 3D Plaus.↑ TextureDetails↑ time

SyncDreamer - - - 626 629 738 > 20 min One-2-3-45 0.199 0.832 89.4 812 815 797 45 sec DreamGaussian 0.171 0.862 57.6 734 728 740 2 min Wonder3D 0.240 0.871 55.7 848 903 829 3 min One-2-3-45++ 0.219 0.886 42.1 1172 1177 1178 1 min DreamCraft3D - - - 1189 1202 1210 > 2 h Ours (MVEdit) 0.139 0.914 29.3 1340 1339 1268 3.8 min

- Table 2. Comparison on text-guided texture generation. *Our ablation study without skip connections resembles the method of TexFusion.

- 5 STABLESSDNERF: FAST TEXT-TO-3D INITIALIZATION

Although text-to-3D generation is possible by chaining text-toimage and image-to-3D, we note that their ability in sculpting regular-shaped objects (e.g., cars) often lags behind 3D-native diffusion models trained specifically on category-level objects. However, as discussed in Section 2.1, training 3D-native diffusion models often faces the challenges of limited data, making it difficult to complete creative tasks such as text-to-3D. To this end, we propose to finetune the text-to-image Stable Diffusion model into a text-to-triplane 3D diffusion model using the single-stage training paradigm in SSDNeRF [Chen et al. 2023b], yielding the StableSSDNeRF.

Methods Aesthetic↑ CLIP↑ Infer. time TV/107

TEXTure 4.66 25.39 2.0 min 2.60 Text2Tex 4.72 24.44 11.2 min 2.15

As shown in Fig. 6, StableSSDNeRF adopts a similar architecture to [Gupta et al. 2023], with a triplane latent diffusion model and a triplane latent decoder. However, instead of training a triplane VAE from scratch to obtain the triplane latents, we employ the off-theshelf Stable Diffusion VAE encoder to obtain the image latents of orthographic views. These latents serve as the initial triplane latent for subsequent optimization, which aligns the triplane and image latent spaces initially, enabling the use of Stable Diffusion v2 as the backbone for triplane diffusion.

Ours (w/o skip, TexFusion)* 4.68 26.34 1.5 min 1.08 Ours (MVEdit) 4.83 26.12 1.6 min 1.59

6 RESULTS AND EVALUATION 6.1 Comparison on Image-to-3D Generation

We compare the image-to-3D results of our MVEdit against those from previous state-of-the-art image-to-3D mesh generators, utilizing two test sets: 248 rendered images of objects sampled from the GSO dataset [Downs et al. 2022], and 33 in-the-wild images, which include demo images from prior studies, AI-generated images, and images sourced from the Internet. To evaluate the quality of the generated textured meshes, we render them from novel views and calculate quality metrics for these renderings. For the GSO test set, we calculate the LPIPS scores [Zhang et al. 2018], CLIP similarities [Radford et al. 2021], and FID scores [Heusel et al. 2017], comparing the renderings of the generated meshes against the ground truth meshes. For the in-the-wild images without ground truths, we follow [Wu et al. 2024] and ask GPT-4V [OpenAI 2023] to compare the multi-view renderings from difference methods based on Image-3D Alignment, 3D Plausibility, and Texture Details. These comparisons allow us to compute the Elo scores [Elo 1967] of the

To fine-tune the model on 3D data, we adopt the LoRA approach [Hu et al. 2022] with a rank of 32 and freeze the base denoising UNet. Following the single-stage training of SSDNeRF, we jointly optimize the LoRA layers, the individual triplane latents, the triplane latent decoder (randomly initialized), and the triplane MLP layers. This optimization utilizes both the denoising mean-squared error (MSE) loss and the NeRF RGB rendering loss, the latter being a combination of pixel L1 loss and patch LPIPS loss, as detailed in Section 3.2.2. We fine-tune the model on the training split of ShapeNet-Cars [Chang et al. 2015; Sitzmann et al. 2019] containing 2458 objects, with text prompts generated by BLIP [Li et al. 2022] and 128×128 low resolution renderings. Using a batch size of 16 objects and 40k Adam iterations, training is completed in just 20 hours on two RTX3090 GPUs, making this approach particularly suitable for small-scale, domain-specific problems.

[Figure 90]

Input MVEdit (Ours) DreamCraft3D One-2-3-45++ Wonder3D

Fig. 7. Comparison of mesh-based image-to-3D methods on in-the-wild images. Please zoom in for detailed viewing.

evaluated methods, providing an automated alternative to costly user studies.

One-2-3-45++[Liu et al. 2024b] utilizes the same multi-view generator as ours (i.e., Zero123++) but employs a multi-view-conditioned 3D-native diffusion model to generate signed distance functions (SDF) for surface extraction, yet this results in overly smooth surfaces with occasional missing parts. DreamCraft3D[Sun et al. 2024], while capable of producing impressive geometric details through its hours-long distillation, generally yields noisy geometry and textures, sometimes even strong artifacts and the Janus problem. In contrast, our approach, while less detailed in geometry compared to SDS, is generally more robust and exhibits fewer artifacts or failures. This results in renderings that are visually more pleasing.

In Table 1, we present the results for One-2-3-45 [Liu et al. 2023b], DreamGaussian [Tang et al. 2024], Wonder3D [Long et al. 2024], One-2-3-45++[Liu et al. 2024b], and our own MVEdit (incorporating both image-to-3D and texture super-resolution) on the GSO test set. This comparison shows that MVEdit significantly outperforms the other methods on all metrics, while still offering a reasonable runtime. For the in-the-wild images, we extend our comparison to include SyncDreamer[Liu et al. 2024a] and DreamCraft3D [Sun et al. 2024]. Here, GPT-4V shows a distinct preference for our method, with MVEdit achieving Elo scores that exceed those of the SDS method DreamCraft3D, despite the latter’s extensive object generation time of over two hours.

- Fig. 7 further presents qualitative comparison among the top

6.2 Comparison on Text-Guided Texture Generation

competitors. Wonder3D [Long et al. 2024] generates multi-view images and normal maps for InstantNGP-based surface optimization, which can lead to broken structures due to multi-view inconsistency.

We randomly select 92 objects from a high-quality subset of Objaverse [Deitke et al. 2023] and employed BLIP [Li et al. 2022] to

“batman in a batman costume standing up with his hands in his pockets”

“a close up of a pair of headphones with a green cover”

“a close up of a toy figure of a man with a hat and a sword”

“there is a large army tank that is on a concrete surface”

“a close up of a robot with a skateboard on a white background”

“there is a toy car with a steering and a steering wheel”

“there is a toy airplane that is flying in the sky”

“a close up of a yellow scooter on a white background”

“there is a picture of a pair of shoes with a shoelace”

“a close up of a car with a roof on a white background”

Prompt (BLIP-generated)

[Figure 91]

TEXTure

Text2Tex

Ours (w/o skip, akin to TexFusion)

Ours (MVEdit)

- Fig. 8. Comparison on text-guided texture generation. Please zoom in for detailed viewing. Note that the BLIP-generated text prompts may not accurately reflect the actual geometry, so it is impossible to generate texture maps that align perfectly with the prompts.

[Figure 92]

[Figure 93]

[Figure 94]

generate text prompts from their rendered images. Using these textureless meshes and the generated prompts of these objects, we evaluate our MVEdit re-texturing pipeline against TEXTure [Richardson et al. 2023] and Text2Tex [Chen et al. 2023c]. TexFusion [Cao et al. 2023] is not directly compared due to the unavailability of official code, but it closely resembles a scenario in our ablation studies, which will be discussed in Section 6.3.1. We assess the quality of the generated textured meshes through rendered images, calculating Aesthetic [Schuhmann et al. 2022] and CLIP [Jain et al. 2022; Radford et al. 2021] scores as the metrics. It is important to note, as shown in a user study by [Wu et al. 2024], that Aesthetic scores more closely align with human preferences for texture details, whereas CLIP scores are less sensitive. Table 2 shows that MVEdit outperforms TEXTure and Text2Tex in both metrics by a clear margin and does so with greater speed.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Input MVEdit Reconstruction-only

Fig. 9. Ablation study on the effectiveness of MVEdit in resolving multi-view inconsistency. Without MVEdit diffusion, the reconstructiononly approach leads to broken thin structures and ambiguous textures.

Fig. 8 presents a quantitative comparison among the tested methods. Both TEXTure and Text2Tex generate slightly over-saturated colors and produce noisy artifacts. In contrast, MVEdit produces clean, detailed textures with a photorealistic appearance and strong text-image alignment.

Table 3. Quantitative ablation study on the effectiveness of MVEdit in resolving multi-view inconsistency.

Img-3D Align. ↑ 3D Plaus.↑

Texture Details↑

Methods

6.3 Ablation Studies

Ours (MVEdit) 1340 1339 1268 Ours (Reconstruction-only) 1275 1252 1241

- 6.3.1 Effectiveness of the 3D Adapter with a Skip Connection. To validate the effectiveness of our ControlNet-based 3D Adapter, we conduct an ablation study by removing the ControlNet, and set the blending weight 𝑤(𝑡) in Eq. 6 to 1 for all timesteps, effectively constructing an architecture without a skip connection, as shown in Fig. 2 (b). For text-guided texture generation, sampling without skip connections is fundamentally akin to TexFusion [Cao et al.

2023], which is known to yield textures with fewer details due to the information loss. This is confirmed by our quantitative results presented in Table 2, which show a notable decrease in the Aesthetic score and Total Variation. Qualitative comparisons in Fig. 8 further

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Textureless low-poly mesh

𝑡𝑡start = 0.84𝑇𝑇 “A realistic image of a camel standing in a natural pose, high quality” …

𝑡𝑡start = 0.96𝑇𝑇 “Turn it into a unicorn”

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Generated mesh using our image-to-3D pipeline

𝑡𝑡start = 0.69𝑇𝑇

𝑡𝑡start = 0.96𝑇𝑇 “Turn it into a stone chair”

“A chair covered in golden cloth, high quality” …

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Voxel character mesh

“Super Mario high poly 3D model , high quality” …

“What if he were in a zombie movie?”

𝑡𝑡start = 0.96𝑇𝑇

𝑡𝑡start = 0.72𝑇𝑇

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Stylized character mesh

“Make it a marble Roman sculpture”

𝑡𝑡start = 0.96𝑇𝑇

“A muscular man with white hair, high quality” …

𝑡𝑡start = 0.54𝑇𝑇

Fig. 10. Results of our text-guided 3D-to-3D and instruct 3D-to-3D pipelines.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

illustrate the visual gap between the two architectures. For 3D-to-3D editing, Fig. 3 shows that the skip connection plays a crucial role not only in producing crisp textures but also in enhancing geometric details (e.g., the ears and knees of the zebra).

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

- 6.3.2 Image-to-3D: MVEdit v.s. Reconstruction-Only. To validate that our image-to-3D pipeline effectively resolves the 3D inconsistency in the initial views generated by Zero123++, we conduct an ablation study by using only the initial views for robust NeRF/mesh optimization, thus bypassing the denoising UNet/DPMSolver and leaving only the reconstruction side. Quantitatively, the GPT-4V evaluation results in Table 3 reveal a clear gap between MVEdit and the reconstruction-only method, underscoring MVEdit’s effectiveness. Qualitatively, as observed in Fig.9, the reconstruction-only method tends to result in broken thin structures and less defined textures, a common consequence of multi-view misalignment.
- 6.3.3 Effectiveness of the Regularization Loss Functions. In Fig. 11, we showcase the results of instruct 3D-to-3D editing under three settings: the full MVEdit, the one without ray entropy loss, and the one without normal TV loss. It can be seen that: removing the ray entropy loss results in inflated geometry and less defined textures, a consequence of initializing DMTet with a fuzzy density field; removing the normal TV loss appears to have little impact on texture quality but leads to numerous holes in the geometry. Although the degradation in quality from these ablations is apparent to humans, especially when viewed interactively in 3D, we note that existing metrics, including Aesthetic score, CLIP score, and even the GPT-4V metrics, struggle to capture these differences. Therefore, we do not include quantitative evaluations for these ablation studies.

Full MVEdit

w/o ray entropy loss

w/o normal TV loss

Input

“As a Deadpool cosplay photo”

Fig. 11. Ablation study on the regularization loss functions, based on the instruct 3D-to-3D pipeline with 𝑡start = 1.0𝑇, using the same seed.

- 6.4 3D-to-3D Editing Results and Discussions

In Fig. 10, we showcase results from both the text-guided 3D-to-3D pipeline and the instruct 3D-to-3D pipeline (with texture superresolution), edited from four types of inputs: a textureless lowpoly mesh, a mesh generated by our image-to-3D pipeline, a voxel character mesh, and a stylized character mesh. As demonstrated in the figure, all inputs are adeptly handled, resulting in promptaccurate appearances, intricate textures, and detailed geometry, thereby highlighting the versatility of our 3D-to-3D pipelines.

- 6.5 Text-to-3D Generation Results and Discussions

In Fig. 12, we showcase results of text-to-3D generation using a combination of StableSSDNeRF and MVEdit pipelines. Thanks to the

[Figure 141]

[Figure 142]

[Figure 143]

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

StableSSDNeRF init. 3D-to-3D Re-texturing

StableSSDNeRF init. 3D-to-3D Re-texturing

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

“A green military vehicle”

“A black and white Porsche 911 police car”

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

StableSSDNeRF init. 3D-to-3D Re-texturing

StableSSDNeRF init. 3D-to-3D Re-texturing

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

“A Formula 1 racing car”

“A yellow Ferrari 458 GT3”

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

StableSSDNeRF init. 3D-to-3D Re-texturing

StableSSDNeRF init. 3D-to-3D Re-texturing

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

“A pink muscle car with black stripes”

“A yellow sports truck”

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

StableSSDNeRF init. 3D-to-3D Re-texturing

StableSSDNeRF init. 3D-to-3D Re-texturing

“A red and whiteNASCAR” (unseen concept)

“AFormula 1 race truck” (unusual combination)

Fig. 12. Results of text-to-3D generation using StableSSDNeRF and MVEdit pipelines.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

knowledge transfer from a large image diffusion model, StableSSDNeRF is able to follow never-seen prompts despite being fine-tuned only on low-resolution renderings of 2458 ShapeNet 3D Cars, generating the correct combination of colors and style. Notably, it can even generalize to completely unseen concept (NASCAR), or to unusual combinations (Formula 1 and truck). When further processed using the text-guided 3D-to-3D and re-texturing pipelines, conditioned on the same input prompts, our method successfully produces diverse, high-quality, photorealistic cars within just 4 minutes.

Input “Turn her into a cyborg”

Fig. 13. An example showcasing the diversity of the generated the samples, based on the instruct 3D-to-3D pipeline with 𝑡start = 1.0𝑇.

6.6 Sample Diversity

training-free 3D Adapter, leveraging off-the-shelf ControlNets and a robust NeRF/mesh optimization scheme, effectively addresses the challenge of achieving 3D-consistent multi-view ancestral sampling while generating sharp details. Additionally, we have developed StableSSDNeRF for domain-specific 3D initialization. Extensive quantitative and qualitative evaluations across a range of tasks have validated the effectiveness of the 3D Adapter design and the versatility of the associated pipelines, showcasing state-of-the-art performance in both image-to-3D and texture generation tasks.

UnlikeSDSapproachesthatexhibit a mode-seeking behavior, MVEdit can generate variations from the exact same input using different random seeds. An example is shown in Fig. 13.

- 7 CONCLUSION AND LIMITATIONS

In this work, we have bridged the gap between 2D and 3D content creation with the introduction of MVEdit, a generic approach for adapting 2D diffusion models into 3D diffusion pipelines. Our novel

Despite the achievements, the MVEdit 3D-to-3D pipelines still face the Janus problem when 𝑡start is close to 𝑇, unless controlled explicitly by directional text/image prompts. Furthermore, the offthe-shelf ControlNets, not being originally trained for our task, can introduce minor inconsistencies and sometimes impose their own biases. Future work could train improved 3D Adapters for strictly consistent and Janus-free multi-view ancestral sampling.

- 8 ACKNOWLEDGEMENTS

This project was in part supported by Vannevar Bush Faculty Fellowship, ARL grant W911NF-21-2-0104, Google, and Samsung. We thank the members of Geometric Computation Group, Stanford Computational Imaging Lab, and SU Lab for useful feedback and discussions. Special thanks to Yinghao Xu for sharing the data, code, and results for image-to-3D evaluation.

REFERENCES

Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. 2023. Cross-Image Attention for Zero-Shot Appearance Transfer. arXiv:2311.03335 [cs.CV]

Titas Anciukevicius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J. Mitra, and Paul Guerrero. 2023. RenderDiffusion: Image Diffusion for 3D Reconstruction, Inpainting and Generation. In CVPR.

Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, Afshin Dehghan, and Josh Susskind. 2022. GAUDI: A Neural Architect for Immersive 3D Scene Generation. In NeurIPS.

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR.

Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, and KangXue Yin. 2023. TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models. In ICCV.

Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. 2022. Efficient Geometry-aware 3D Generative Adversarial Networks. In CVPR.

Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. 2023. GeNVS: Generative Novel View Synthesis with 3D-Aware Diffusion Models. In ICCV.

Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. 2015. ShapeNet: An Information-Rich 3D Model Repository. Technical Report arXiv:1512.03012 [cs.GR]. Stanford University — Princeton University Toyota Technological Institute at Chicago.

Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner. 2023c. Text2Tex: Text-driven Texture Synthesis via Diffusion Models. In ICCV.

Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. 2023b. Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction. In ICCV.

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023a. Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation. In ICCV.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A Universe of Annotated 3D Objects. In CVPR.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. 2022. Google scanned objects: A high-quality dataset of 3d scanned household items. In ICRA. 2553–2560.

Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo Jimenez Rezende, and Dan Rosenbaum. 2022. From data to functa: Your data point is a function and you can treat it like one. In ICML.

Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. 2021. Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets From 3D Scans. In ICCV. 10786–10796.

Arpad E Elo. 1967. The proposed uscf rating system, its development, theory, and applications. Chess Life 22, 8 (1967), 242–247.

Jiatao Gu, Alex Trevithick, Kai-En Lin, Josh Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. 2023. NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion. In ICML.

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas Oğuz. 2023. 3DGen: Triplane Latent Diffusion for Textured Mesh Generation. arXiv:2303.05371 [cs.CV]

Ayaan Haque, Matthew Tancik, Alexei Efros, Aleksander Holynski, and Angjoo Kanazawa. 2023. Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions. In ICCV.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In NeurIPS.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In NeurIPS. Jonathan Ho and Tim Salimans. 2021. Classifier-Free Diffusion Guidance. In NeurIPS Workshop.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR. https://openreview.net/forum?id=nZeVKeeFYf9

Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, and Ben Poole. 2022. Zero-Shot Text-Guided Object Generation with Dream Fields. Mijeong Kim, Seonguk Seo, and Bohyung Han. 2022. InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering. In CVPR. Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR. Min Seok Lee, Wooseok Shin, and Sung Won Han. 2022. TRACER: Extreme Attention Guided Salient Object Tracing Network. In AAAI.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. In ICML.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3D: HighResolution Text-to-3D Content Creation. In CVPR.

Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. 2024b. One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion. In CVPR.

Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. 2023b. One2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. In NeurIPS.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023a. Zero-1-to-3: Zero-shot One Image to 3D Object. In ICCV. Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2024a. SyncDreamer: Generating Multiview-consistent Images from a Single-view Image. In ICLR.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang.

2024. Wonder3D: Single Image to 3D using Cross-Domain Diffusion. In CVPR.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022. DPMSolver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. In NeurIPS.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. Repaint: Inpainting using denoising diffusion probabilistic models. In CVPR. 11461–11471.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In ICLR.

Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. 2023. Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures. In CVPR. Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Norman Müller, , Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder, and Matthias Nießner. 2023. DiffRF: Rendering-Guided 3D Radiance Field Diffusion. In CVPR.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. ACM Transactions on Graphics 41, 4, Article 102 (July 2022), 15 pages. https://doi.org/10.1145/3528223. 3530127

OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL] Ryan Po, Wang Yifan, and Vladislav Golyanik et al. 2024. Compositional 3D Scene

Generation using Locally Conditioned Diffusion. In 3DV. Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In ICLR.

Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. 2024. Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors. In ICLR.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021.

Learning transferable visual models from natural language supervision. In ICML. 8748–8763.

Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. 2023. Texture: Text-guided texturing of 3d shapes. In SIGGRAPH.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. In CVPR. Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. 2022. LAION-5B: An open large-scale dataset for training next generation image-text models. In NeurIPS Workshop.

Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. 2021. Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis. In NeurIPS.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023. Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model. arXiv:2310.15110

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2024. MVDream: Multi-view Diffusion for 3D Generation. In ICLR.

J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 2023. 3D Neural Field Generation using Triplane Diffusion. In CVPR. Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. 2019. Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations. In NeurIPS.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In ICLR.

O. Sorkine, D. Cohen-Or, Y. Lipman, M. Alexa, C. Rössl, and H.-P. Seidel. 2004. Laplacian Surface Editing. In Proceedings of the 2004 Eurographics/ACM SIGGRAPH Symposium on Geometry Processing (Nice, France) (SGP ’04). Association for Computing Machinery, New York, NY, USA, 175–184. https://doi.org/10.1145/1057432.1057456

Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. 2024. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. In ICLR.

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2024. DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. In ICLR.

Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Joshua B. Tenenbaum, Frédo Durand, William T. Freeman, and Vincent Sitzmann. 2023. Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision. In NeurIPS.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021a. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. In NeurIPS. 27171–27183.

Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, and Baining Guo. 2023b. Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion. In CVPR. Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. 2021b. Real-ESRGAN: Training

Real-World Blind Super-Resolution with Pure Synthetic Data. In ICCV Workshop.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023a. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. In NeurIPS.

Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. 2023. Novel View Synthesis with Diffusion Models. In ICLR.

Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon Wetzstein. 2024. GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation. In CVPR.

Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, and Kai Zhang. 2024. DMV3D: Denoising Multi-View Diffusion using 3D Large Reconstruction Model. In ICLR.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721 Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. In ICCV. Richard Zhang, Phillip Isola, Alexei Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Xin-Yang Zheng, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu, and Heung-Yeung Shum. 2023. Locally Attentional SDF Diffusion for Controllable 3D Shape Generation. ACM Transactions on Graphics 42, 4 (2023).

Zhizhuo Zhou and Shubham Tulsiani. 2023. SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction. In CVPR.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

