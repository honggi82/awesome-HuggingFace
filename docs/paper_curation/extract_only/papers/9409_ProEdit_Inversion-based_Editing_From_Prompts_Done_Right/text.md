## ProEdit: Inversion-based Editing From Prompts Done Right

# arXiv:2512.22118v1[cs.CV]26Dec2025

Zhi Ouyang1,∗ Dian Zheng2,∗ Xiao-Ming Wu3 Jian-Jian Jiang1

Kun-Yu Lin4 Jingke Meng1, Wei-Shi Zheng1,5, 1Sun Yat-sen University 2CUHK MMLab 3College of Computing and Data Science, Nanyang Technological University 4The University of Hong Kong 5Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education, China

https://isee-laboratory.github.io/ProEdit/

Task 1: Image Editing Task 2: Video Editing

Red car Black car

A man with black shirt and hands down A man with black shirt and crossed arms

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

RF-SolverOursSourceVideo

Two white bears holding a rose One white bear holding a rose

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

An orange cat sitting on top of a fence A black cat sitting on top of a fence

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

Source Image

PnP PnP-Inv RF-Solver FireFLow UniEdit Ours

Figure 1. ProEdit for image and video editing. We propose a highly accurate, plug-and-play editing method for flow inversion that addresses the problem of excessive source image information injection, which prevents proper modification of attributes such as pose, number, and color. Our method has demonstrated impressive performance in both image editing and video editing tasks.

### Abstract

pect, we introduce KV-mix, which mixes KV features of the source and the target in the edited region, mitigating the influence of the source image on the editing region while maintaining background consistency. In the latent aspect, we propose Latents-Shift, which perturbs the edited region of the source latent, eliminating the influence of the inverted latent on the sampling. Extensive experiments on several image and video editing benchmarks demonstrate that our method achieves SOTA performance. In addition, our design is plug-and-play, which can be seamlessly integrated into existing inversion and editing methods, such as RFSolver, FireFlow and UniEdit.

Inversion-based visual editing provides an effective and training-free way to edit an image or a video based on user instructions. Existing methods typically inject source image information during the sampling process to maintain editing consistency. However, this sampling strategy overly relies on source information, which negatively affects the edits in the target image (e.g., failing to change the subject’s atributes like pose, number, or color as instructed). In this work, we propose ProEdit to address this issue both in the attention and the latent aspects. In the attention as-

∗ equal contributions.

corresponding authors. Code is available

Source feature

- (a) Previous
- (b) Ours

[Figure 31]

Inv.

[Figure 32]

MM-DiT

Orange Black

Inverted noise

Pure noise

Target feature

Mix feature

Mix

[Figure 33]

###### Inv. MM-DiT

[Figure 34]

Shift

Orange

Inverted noise

Black

- Figure 2. Framework comparison between (a) previous methods and (b) our method. To address the issue of excessive source image information injection, we introduce the Shift module for inverted noise and the Mix module for the attention injection, alleviating the editing failures caused by these issues.

### 1. Introduction

Inversion-based visual editing [7, 11, 21, 33–35, 48] has emerged as a highly effective and valuable research direction, offering a powerful, training-free paradigm for modifying images and videos according to user instructions, especially the flow-based editing methods, which provides better generative abilities with fewer sampling steps.

Most established inversion-based methods [8, 51] operate by first leveraging the inverted latents from the source image as the starting point. Then, they redo the sampling process by using the target prompt to guide the sampling process towards the target image or video. To maintain fidelity to the source content, they mostly employ the source injection strategy to re-introduce source-specific information or “nuggets” during the sampling process. However, as shown in Fig 1, this sampling strategy overly relies on source information, no matter in the latent aspect or the attention aspect, which negatively affects the edits in the target image, especially regarding subject attributes such as color, pose, and number.

In this work, we first conduct an in-depth investigation into the above problems and conclude that completely relying on the inverted latents for sampling and applying global attention feature injection introduces excessive source image information, leading to editing failure. Specifically, in the attention aspect, we find that the global attention feature injection strategy introduces excessive attribute-related information from the source attention, causing the model to overly focus on source information while neglecting text guidance. For the latent aspect, starting from the source image distribution creates an overly strong prior that easily leads the sampling process to reconstruct the source distribution.

Based on the above observation, we propose a novel inversion-based editing method, ProEdit, to eliminate the

negative impact of the source image from both the attention and the latent aspects. For the attention aspect, we introduce KV-mix. We first identify the edited regions based on source and target prompts, then mix the K and V features in these regions while fully injecting source KV features in non-edited areas to preserve background consistency. This mixing mechanism applies to all attention operations without manual adjustment of heads, layers, or blocks. For the latent aspect, we propose LatentsShift. Inspired by AdaIN from style transfer, which performs structure-preserving distribution transformations, we inject random noise into the source distribution of edited regions. This reduces the influence of source image attributes while maintaining structural and background consistency. As shown in Fig 2, our method successfully eliminates the negative effects of source image information from inversion noise and attention injection mechanisms, accurately modifying the subject’s attribute while maintaining the consistency of background and non-editing content.

Through extensive experiments, we demonstrate that 1) Effectiveness: ProEdit can eliminate the negative impact of the source image/video on the editing content, while preserving the non-edited content, achieving state-of-the-art (SOTA) performance in various editing tasks; 2) Plug-andplay: Our method is plug-and-play, enabling its seamless integration into a wide range of existing inversion and editing methods; 3) Attribute Correction: In attribute editing, where existing methods perform poorly, our approach showcases unprecedented performance. Our contributions can be summarized as follows:

- • We investigate the issue of excessive source image information injection in inversion-based editing and identify that this problem stems from both latent initialization and attention injection mechanisms, leading to failures.
- • We propose ProEdit, a novel training-free approach designed to solve the above problem, which can eliminate the negative impact from the source while maintaining background consistency.
- • Through extensive experiments, we proof that ProEdit is effective, plug-and-play and can be used in various types of image and video editing. Our code will be open-source to boost the developing of the generative community.

### 2. Related Work

#### 2.1. Text-to-Visual generation

Diffusion models have achieved significant success in the fields of text-to-visual generation, leading to the development of a series of outstanding foundational models [19, 37, 41, 44, 58]. Recently, the text-guided generation paradigm for both images and videos has been shifting from diffusion models based on U-Net [42] architecture to flow models based on DiT [37] architecture. Flow-based models,

such as FLUX [26] and HunyuanVideo [25], utilize the MMDiT [10] architecture and simulate a straight path between two distributions through a probability flow ordinary differential equation(ODE), enabling faster and better generation with fewer sampling steps. These T2I and T2V models also facilitate the editing of images and videos, where target images are generated based on source images and modified text prompts.

#### 2.2. Text-driven Editing

For visual editing tasks, early works focused on trainingbased methods [3, 20, 22, 23, 27, 29, 60]. These methods leverage generative models to achieve controllable image editing. As generative models have advanced, attention has shifted towards training-free editing methods, which offer greater flexibility and efficiency. Among them, inversionbased methods [49] have become an important research direction for applying diffusion models to image editing tasks. DDIM inversion [44], as a representative method, marked a significant advancement in inversion-based image editing within diffusion models, inspiring a series of high-precision solvers [32, 50, 57] aimed at minimizing inversion errors and improving sampling efficiency. Sampling-based methods introduce controlled randomness to enable more flexible editing [9, 21, 36, 53]. On the other hand, attentionbased methods achieve controllable image editing by altering the role of attention tokens [5, 24, 28, 46, 48, 55], and these methods have gradually expanded to video editing [4].

Following the trajectory of diffusion models, recent inversion methods based on flow models have mainly focused on improving inversion solvers [8, 18, 51, 54] and the joint attention mechanism [2, 56] in MM-DiT [10]. Although they have achieve good editing performance, these methods still overlook the negative impact of inversion strategies on the editing content. In this work, we reveal the negative impact of inversion on editing and propose ProEdit to eliminate this negative influence from both the attention and latent distribution perspectives. Notably, existing methods rely on selecting specific attention heads, layers, or block types when modifying the attention mechanism, which limits their alignment with the source image. Our method is the first to achieve this without requiring the selection of specific layers, heads, or block types.

### 3. Method

In this section, we first introduce the preliminary to understand our method in Section 3.1, then we conduct an investigation in Section 3.2, analyzing the reasons why the inversion-sampling paradigm faces challenges in removing the influence of the source image on the target image’s edited contents. Next, we introduce our proposed ProEdit method in Section 3.3 and 3.4. Our method eliminates the influence of the source image on the target image’s edited

contents from both the attention guidance and the initial latents in the sampling process, while maintaining the consistency of the background structure. Finally, we summarize our method’s editing process in Section 3.5.

#### 3.1. Preliminaries

First, we introduce the preliminary knowledge to better understand our method, including the training objective of flow-based generative models and the ODE solving process. Then, we derive the inversion ODE solving process based on flow models.

Generative models [19] aim to generate data X1 that follows the real data distribution π1 from noise X0 that follows a Gaussian distribution π0. Recently, flow matching [1, 30, 31] has emerged as a method that learns a velocity field vθ to transform noise into data along a straight trajectory. The training objective is to solve the following optimization problem:

0,Z1,t ∥(Z1 − Z0) − vθ(Zt,t)∥2 , Zt = tZ1 + (1 − t)Z0, t ∈ [0,1],

EZ

min

(1)

θ

where Z0 ∼ π0 is initialized from the source distribution, and Z1 ∼ π1 is generated at the end of the trajectory. The term Z1 − Z0 represents the target velocity. The model learns a velocity field to deterministically transform random samples of Gaussian noise into target data via an Ordinary Differential Equation (ODE) defined over the continuous time interval:

dZt = vθ(Zt,t)dt, t ∈ [0,1] (2)

This ODE can be discretized and numerically solved by solvers as follows:

,ti), (3) where i ∈ {0,...,N}, with t0 = 0 and tN = 1.

+ (ti+1 − ti)vθ(Zt

##### Zt

= Zt

i+1

i

i

Flow matching has deterministic trajectories. Its reverse process is obtained by reversing the learned flow trajectory. Starting from Z1 ∼ π1, the reverse ODE is given by reverse the velocity field:

dZt = −vθ(Zt,t)dt, t ∈ [1,0] (4)

Correspondingly, this ODE is discretized and solved using a numerical solver as follows:

,ti), (5)

i − (ti−1 − ti)vθ(Zt

##### Zt

= Zt

i−1

i

where i ∈ {N,...,0}, with tN = 1 and t0 = 0. This inverse process generates Z0 ∼ π0 by utilizing the symmetry of the velocity field v to ensure consistency with the forward process. Naturally, this inversion method is applied in visual reconstruction and visual editing.

An orange cat sitting on top of a fence A black cat sitting on top of a fence

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“orange”

[Figure 41]

[Figure 42]

RF-Solverw/oVinjection

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

“black”

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

“orange”

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

“black”

Inverted noise attention Sampling attention

- Figure 3. Excessive source image information injection phenomenon in RF-Solver. We validate it by visualizing the attention from source and target text tokens to the visual tokens during initial and sampling stage. In RF-Solver, the attention from the source text token to the visual tokens remains higher than that from the target text token. However, after removing attention injection, the attention from “black” and “orange” to visual tokens returns to similar levels, but some subject attributes (e.g., pose) change accordingly.

#### 3.2. Rethinking the Inversion-Sampling Paradigm

the image is inverted back to noise, it still retains substantial source image attributes. This causes editing to fail when the gap between target and source prompts is too large.

In this subsection, we conduct an investigation of the challenge in visual editing, and we conclude that previous works mostly rely on sampling with inverted noise and use a source attention injection mechanism to maintain background and structural consistency. This design often injects excessive source image information, leading to editing failure. The analysis is as follows.

Summary. The negative impact of the source image on the editing process can be attributed to two factors: global attention feature injection and the latent distribution injection. Therefore, this paper proposes ProEdit to address all these issues from corresponding aspects.

Attention Injection Problem. To maintain the overall structural consistency between the target and source images, current methods [8, 51] globally inject the value attention features V into specific time steps during the sampling process, as described by the following equation:

#### 3.3. KV-mix

Motivation. As analyzed before, previous methods use a global injection mechanism of visual and textual attention features to maintain consistency, but the excessive injection of source attention features negatively impacts the editing quality. Towards this end, our method aims to mitigate the problem with the insight that mixing source visual attention and target visual attention helps align with the target prompt while maintaining the consistency of non-edited content.

###### ztgt (l + 1) = Attn(Qttg,Ktgt ,Vst), (6)

where s denotes the attention features corresponding to the source prompt, while tg denotes the attention features corresponding to the target prompt. However, this global attention feature injection mechanism has a negative impact on the editing process. As shown in Fig 3, after adding the attention injection mechanism, although overall consistency is enhanced, the model focus far less on prompt “black” than on “orange” in the image, representing that the subject’s attributes are also forcibly injected into the target image. This issue increases the difficulty of attribute editing.

Method. Based on the above observations, we execute attention control on the visual components across all blocks, while consistently using the attention features of the target prompt for text attention to achieve effective editing guidance. To distinguish between the editing and non-editing regions, we obtain a mask M by processing the attention map to separate the editing region, for the detailed implementation please see the Supplementary File. For the non-editing region, we apply full injection of visual attention features to maintain background consistency. For the editing region, we use a mix of source and target visual attention features

Latent Distribution Injection Problem. As shown in Fig 3, the attention from “orange” to visual tokens is significantly higher than from “black”, indicating that although

DiT Blocks

Source Input

“orange cat” Source Prompt

푡  푡 

ModulationModulation

Target Input

Attention

×T

[Figure 63]

Mix Feature 푡  푡 

Mask

Source Image

Inverted Noise

푡  푡 

[Figure 64]

Target Output

Mask extraction

KV-Mix

(b) KV-Mix

Latents Shift

Inverted Noise

 (  )  (  )

[Figure 65]

Attention

Mask

푡  푡 

×T

AdaIN

Inverted Noise

푡 

Random Noise

 (  )  (  )

Noise

“black cat” Target Prompt

Edit Image

DiT Blocks

(c) Latents-Shift

###### (a) Pipeline

- Figure 4. Pipeline of our ProEdit. The mask extraction module identifies the edited region based on source and target prompts during the first inversion step. After obtaining the inverted noise, we apply Latents-Shift to perturb the initial distribution in the edited region, reducing source image information. In selected sampling steps, we fuse source and target attention features in the edited region while directly injecting source features in non-edited regions to achieve accurate attribute editing and background preservation simultaneously.

to preserve the consistency of non-editing content and improve the editing quality. After extensively exploring all plausible combinations of Q, K, and V , we found that the configuration shown in Eq.7 is most conducive to achieving consistent editing. Formally, our KV-mix design is as follows:

color and texture distributions while preserving structural consistency, we adapt this approach to image editing.

Method. As our goal is to eliminate the influence of source image information, we directly use random noise as the style image to shift the distribution of the inverted noise. We improve its formula to implement the shift of the latent distribution for the editing region as follows:

Kˆtgl = δKtgl + (1 − δ)Ksl, Vˆtgl = δVtgl + (1 − δ)Vsl, K˜tgl = M ⊙ Kˆtgl + (1 − M) ⊙ Ksl,

zT − µ(zT) σ(zT)

+ µ(zTr ),

z˜T = σ(zTr )

(8)

(7)

zˆT = M ⊙ (βz˜T + (1 − β)zT) + (1 − M) ⊙ zT,

V˜tgl = M ⊙ Vˆtgl + (1 − M) ⊙ Vsl, zt(l + 1) = Attn Qltg,K˜tgl ,V˜tgl ,

where β denotes the fusion ratio between the inverted noise and pure noise, controlling the level of shift in the inverted noise distribution. M denotes the edited region, which is inherited from KV-mix to achieve the shifted inverted noise distribution for the editing region.

where M denotes the edited region that is extracted from attention map and applied only to the visual branch. To enable controllable editing strength and preservation of non-edited content, we define the mixing strength δ as a ratio of mix for applying attention control in the edited region, which determines the level of non-edited content preservation during editing. This attention mechanism enables precise text control for consistent editing. Since we perform the KV-mix operation only within the visual tokens, KV-mix is applied in both Double and Single Attention blocks.

#### 3.5. Overall

The complete process of our pipeline can be summarized as Fig 4: During the inversion stage, the source image and source prompt are input into the model to conduct the inversion process, and Ksl and Vsl are cached in the fly. Then, the attention map is processed to obtain the mask of the editing region, and the inverted noise is output as the initial input for the sampling stage.

#### 3.4. Latents-Shift

In the sampling stage, the inverted noise first passes through the Latents-Shift module to obtain the fusion noise, which is then input into the model along with the target prompt for sampling. During the sampling process, the

Motivation. Here we aim to mitigate the problem of distribution injection while preserving the structure consistency. Inspired by AdaIN [14] in style transfer, which transfers

- Table 1. Text-driven image editing comparison on PIE-Bench. We report the peer-reviewed results of each baseline, and evaluate our proposed method using flow-based inversion methods RF-Solver, FireFlow, and UniEdit to demonstrate the effectiveness. The best and second-best results are shown in bold and underline respectively.

Method Model

Structure BG Preservation CLIP Sim.↑ NFE

Distance (×103)↓ PSNR↑ SSIM (×102)↑ Whole Edited

P2P [13] Diffusion 69.43 17.87 71.14 25.01 22.44 100 PnP [48] Diffusion 28.22 22.28 79.05 25.41 22.55 100 PnP-Inversion [21] Diffusion 24.29 22.46 79.68 25.41 22.62 100 EditFriendly [16] Diffusion – 24.55 81.57 23.97 21.03 90 MasaCtrl [5] Diffusion 28.38 22.17 79.67 23.96 21.16 100 InfEdit [55] Diffusion 13.78 28.51 85.66 25.03 22.22 72 RF-Inversion [43] Flow 40.60 20.82 71.92 25.20 22.11 56

RF-Solver [51] Flow 31.10 22.90 81.90 26.00 22.88 60 RF-Solver+Ours Flow 27.82 24.77 84.78 26.28 23.25 60

FireFlow [8] Flow 28.30 23.28 82.82 25.98 22.94 32 FireFlow+Ours Flow 27.51 24.78 85.19 26.28 23.24 32

UniEdit [18](α=0.6) Flow 10.14 29.54 90.42 25.80 22.33 28 UniEdit(α=0.6)+Ours Flow 9.22 30.08 90.87 25.78 22.30 28

UniEdit [18](α=0.8) Flow 26.85 24.10 84.86 26.97 23.51 37 UniEdit(α=0.8)+Ours Flow 24.27 24.82 85.87 27.08 23.64 37

- Table 2. Quantitative comparison on Color Editing. The best and second-best results are shown in bold and underline.

ProEdit with FateZero [39], Flatten [6], Tokenflow [12], and RF-Solver [51].

Datasets. For text-driven image editing, we evaluate our method based on the PIE-Bench [21], which contains 700 images with 10 different editing types. For text-driven video editing, we collected 55 text-video editing pairs with a resolution of 480×480, 540×960 or 960×540, consist of 40 to 120 frames, including the videos sourced from DAVIS dataset [38] and online platforms. The prompts are derived from ChatGPT or contributed by the authors.

BG Preservation CLIP Sim.↑

Method

SSIM (×102)↑ Whole Edited

RF-Solver 80.21 25.61 20.86 RF-Solver+Ours 86.63 27.30 22.88

FireFlow 80.14 26.03 21.02 FireFlow+Ours 86.53 27.32 22.55

Metrics. For text-driven image editing, to evaluate edit-irrelevant context preservation, we use structure distance [47], PSNR [17] and SSIM [52] for annotated unedited regions. The performance of the edits is assessed using CLIP similarity [40] for both the whole image and the edited regions. For text-driven video editing, we follow the metrics proposed in VBench [15, 59], including Subject Consistency, Motion Smoothness, Aesthetic Quality, and Imaging Quality.

UniEdit 85.39 26.81 21.74 UniEdit+Ours 85.21 27.32 22.56

source visual attention features Ksl and Vsl, obtained from the inversion, are injected through the KV-mix module, and the final model outputs the target image by multiple steps of sampling.

### 4. Experiments

Implementation. We primarily product experiments using FLUX.1-[dev] [26] for image editing and HunyuanVideo720p [25] for video editing. For image editing, we have made ProEdit plug-and-play for flow-based inversion methods: RF-Solver, FireFlow, and UniEdit. Notably, UniEdit uses α to denote the delay injection rate, and experiments were conducted with α = 0.6 and α = 0.8. In our experiments, unless otherwise specified, the delay rate α in UniEdit is set to 0.8. We set the sampling step for image editing to 15. For video editing, we have made ProEdit

#### 4.1. Setup

Baseline. We mainly compare our methods with previous state-of-the-art turning-free visual editing methods. For text-driven image editing, we compare our ProEdit with diffusion-based methods: P2P [13], PnP [48], PnP-Inversion [21], EditFriendly [16], MasaCtrl [5], and InfEdit [55], along with flow-based methods: RF-Inversion [43], RF-Solver [51], Fire-Flow [8], and UniEdit [18]. For text-driven video editing, we compare our

Source RF-Solver w/Ours FireFlow w/Ours UniEdit w/Ours

PnP PnP-Inversion

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

UmbrellaFlowerBenchSofaTigerCat+NightDaytime

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

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

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

- Figure 5. Qualitative comparison on image editing. With our method, various flow-based inversion methods achieve more appropriate editing while preserving the consistency of background and non-editing content.

plug-and-play for RF-Solver. We set the sampling step for video editing to 25.

#### 4.2. Text-driven Image Editing

Quantitative Evaluation. Table 1 presents the quantitative results for text-driven image editing. The results in the table show that with our ProEdit, flow-based inversion methods achieved superior results in image editing. Notably, ProEdit with the UniEdit inversion method achieves state-of-theart performance in both source content preservation and editing quality.

Qualitative Evaluation. We compare the performance of our method with several baselines across different types of editing requirements in Fig 5. The baseline methods often fail to maintain the consistency of non-editing attributes such as background and posture, or fail to achieve satisfactory editing results. In contrast, our method achieves highquality editing results while maintaining the consistency of non-editing content.

Color Editing. To validate that our method addresses the “latent distribution injection” issue overlooked by previous methods, We conducted experiments on the color editing task in PIE-Bench, which is significantly affected by the latent distribution. Table 2 shows the quantitative results for color editing. With our ProEdit, all flow-based inversion methods achieved impressive results. This supports the motivation behind our proposed Latents-Shift module. The AdaIN-based Latents-Shift helps the editing process break

free from the constraints imposed by the source image distribution. We further validate it by visualizing the attention map after adding Latents-Shift in Fig 6.

#### 4.3. Text-driven Video Editing

Quantitative Evaluation. Table 3 presents the quantitative results for text-driven video editing. For each metric, we report the average score of all videos. The results in the table show that with our ProEdit, flow-based inversion methods achieved superior results in video editing. This proves the versatility of our method for flow-based models, demonstrating its applicability to video editing tasks and its ability to improve editing performance.

Qualitative Evaluation. We compare the performance of our method with several methods in Fig 7. The baseline methods often fail to maintain the consistency of nonediting attributes such as background and posture, or fail to achieve satisfactory editing results. In contrast, ProEdit achieves high-quality editing while maintaining spatial and temporal consistency.

#### 4.4. Ablation Study

We conduct ablation study on PIE-Bench. First, we evaluate the effectiveness of each module we proposed and validate their synergistic effect. Then, we explore the best combinations within the attention mixing injecting mechanism.

The Synergistic Effect Analysis. We evaluate the effectiveness of the proposed KV-mix and Latents-Shift in Ta-

An orange cat sitting on top of a fence A black cat sitting on top of a fence

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

“orange”

[Figure 117]

[Figure 118]

Ours

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

“black”

Inverted noise attention Sampling attention

- Figure 6. Visualization of attention map after performing ProEdit. The initial distribution is shifted to target prompt and during the sampling, the model can accurately edit the image while maintaining non-editing attribute and background concsistent.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

SourceFlattenOursTokenFlowRF-Solver

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

+Crown

- Figure 7. Qualitative comparison on video editing. The video comprises 48 frames with a resolution of 540 × 960.

Table 4. Quantitative comparison for the ablation study. KVm, LS mean KV-mix, Latents-Shift in our method. The best and second-best results are shown in bold and underline respectively.

CLIP Sim.↑ Whole Edited

Method KV-m LS

26.00 22.88 RF-Solver ✓ 26.21 23.21 ✓ ✓ 26.28 23.25

- 25.98 22.94 FireFlow ✓ 26.22 23.18 ✓ ✓ 26.28 23.24

- 26.97 23.51 UniEdit ✓ 27.02 23.54 ✓ ✓ 27.08 23.64

serve a significant improvement in CLIP similarity due to the reduced influence of source features in the attention. After incorporating the Latents-Shift module, the CLIP similarity is further enhanced as the influence of the source image on the inversion noise latent distribution is eliminated. In summary, the various modules of ProEdit work synergistically to improve the editing results.

Table 3. Text-driven video editing comparison. We report the peer-reviewed results of each baseline, and evaluate our proposed method using flow-based inversion method RF-Solver. The best and second-best results are shown in bold and underline.

The Attention Feature Combination Effect Analysis. We evaluated the effectiveness of different attention feature combinations applied in the fusion injection mechanism using the RF-Solver inversion method on PIE-Bench to verify the superiority of our proposed KV-mix module. The quantitative results for different attention feature combinations is shown in Supplementary File. Note that the V attention feature is the most important attention feature for editing quality, so all the attention feature combinations we evaluated include V. Among the four combinations we evaluate, the KV combination achieved the best performance in both background consistency preservation and editing quality. Therefore, we adopted the KV fusion injection mechanism and designed the KV-mix module.

###### Method SC ↑ MS ↑ AQ ↑ IQ ↑

FateZero [39] 0.9612 0.9740 0.6004 0.6556 Flatten [6] 0.9690 0.9830 0.6318 0.6678 TokenFlow [12] 0.9697 0.9897 0.6436 0.6817 RF-Solver [51] 0.9708 0.9906 0.6497 0.6866 RF-Solver+Ours 0.9712 0.9920 0.6518 0.6936

ble 4. Note that without the modules we proposed, each method is evaluated using its original source code setup. When the KV-mix feature injection mechanism is applied to replace the original feature injection mechanism, we ob-

### 5. Conclusion

In this work, we identified the issue of excessive injection of source image information caused by the ”inverted latent with global injection” strategy used in existing flowbased inversion editing methods, which leads to sacrificing editing quality in order to maintain background consistency with the source image during the editing process. We introduce ProEdit, a novel, training-free method that addresses this issue by proposing the KV-mix and Latents-Shift modules from both the attention and latent perspectives, aiming to eliminate the negative impact of excessive source image information injection on editing quality. Extensive experiments show that ProEdit can be seamlessly integrated into existing flow-based inversion methods, achieving high background consistency and excellent editing quality simultaneously.

### References

- [1] Michael S. Albergo, Nicholas M. Boffi, and Eric VandenEijnden. Stochastic interpolants: A unifying framework for flows and diffusions. CoRR, 2023. 3
- [2] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. In CVPR, 2025. 3
- [3] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 3
- [4] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In CVPR, 2025. 3, 1
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In CVPR, 2023. 3, 6
- [6] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flowguided attention for consistent text-to-video editing. In ICLR, 2024. 6, 8
- [7] Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxspace: Disentangled semantic editing in rectified flow transformers. CoRR, 2024. 2
- [8] Yingying Deng, Xiangyu He, Changwang Mei, Peisong Wang, and Fan Tang. Fireflow: Fast inversion of rectified flow for image semantic editing. In ICML, 2025. 2, 3, 4, 6, 1
- [9] Wenkai Dong, Song Xue, Xiaoyue Duan, and Shumin Han. Prompt tuning inversion for text-driven image editing using diffusion models. In ICCV, 2023. 3
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-

fied flow transformers for high-resolution image synthesis. In ICML, 2024. 3

- [11] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. In ECCV, 2024. 2
- [12] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In ICLR, 2024. 6, 8
- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 6
- [14] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV,

2017. 5

- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 6
- [16] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In CVPR, 2024. 6
- [17] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 2008. 6
- [18] Guanlong Jiao, Biqing Huang, Kuan-Chieh Wang, and Renjie Liao. Uniedit-flow: Unleashing inversion and editing in the era of flow models. arXiv preprint arXiv:2504.13109,

2025. 3, 6, 1

- [19] Ajay Jain Jonathan Ho and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3
- [20] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In ECCV,

2024. 3

- [21] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In ICLR, 2024. 2, 3, 6, 1
- [22] Phillip Isola Jun-Yan Zhu, Taesung Park and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, 2017. 3
- [23] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. TPAMI, 2021. 3
- [24] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 3
- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3, 6
- [26] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3, 6
- [27] Hongyu Li, Manyuan Zhang, Dian Zheng, Ziyu Guo, Yimeng Jia, Kaituo Feng, Hao Yu, Yexin Liu, Yan Feng, Peng

- Pei, et al. Editthinker: Unlocking iterative reasoning for any image editor. arXiv preprint arXiv:2512.05965, 2025. 3
- [28] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. CoRR, 2023. 3
- [29] Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Ying Shan, Yuexian Zou, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. CoRR, 2024. 3
- [30] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [31] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learfning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3
- [32] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, 2025. 3
- [33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2
- [34] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. In WACV, 2025.
- [35] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 2
- [36] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In SIGGRAPH, 2023. 3
- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, 2023. 2
- [38] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 6
- [39] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In CVPR, 2023. 6, 8
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [42] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, 2015. 2
- [43] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic im-

- age inversion and editing using rectified stochastic differential equations. In ICLR, 2025. 6
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 3
- [45] Qwen Team. Qwen3 technical report, 2025. 2
- [46] Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. Add-it: Training-free object insertion in images with pretrained diffusion models. In ICLR,

2025. 3

- [47] Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing vit features for semantic appearance transfer. In CVPR, 2022. 6
- [48] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 2, 3, 6, 1
- [49] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: exact diffusion inversion via coupled transformations. In CVPR,

2023. 3

- [50] Fangyikang Wang, Hubery Yin, Yuejiang Dong, Huminhao Zhu, Zhang Chao, Hanbin Zhao, Hui Qian, and Chen Li. BELM: bidirectional explicit linear multi-step sampler for exact inversion in diffusion models. In NeurIPS, 2024. 3
- [51] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. In ICML, 2025. 2, 3, 4, 6, 8, 1
- [52] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing,

2004. 6

- [53] Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu Chang. Uncovering the disentanglement capability in texttoimage diffusion models. In CVPR, 2023. 3
- [54] Pengcheng Xu, Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Jiangning Zhang, Chengjie Wang, Yunsheng Wu, Charles Ling, and Boyu Wang. Unveil inversion and invariance in flow transformer for versatile image editing. In CVPR, 2025. 3
- [55] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with natural language. In CVPR, 2024. 3, 6
- [56] Yu Xu, Fan Tang, Juan Cao, Yuxin Zhang, Xiaoyu Kong, Jintao Li, Oliver Deussen, and Tong-Yee Lee. Headrouter: A training-free image editing framework for mmdits by adaptively routing attention heads. arXiv preprint arXiv:2411.15034, 2024. 3
- [57] Guoqiang Zhang, John P. Lewis, and W. Bastiaan Kleijn. Exact diffusion inversion via bidirectional integration approximation. In ECCV, 2024. 3
- [58] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2
- [59] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, WeiShi Zheng, et al. Vbench-2.0: Advancing video generation

benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025. 6

[60] Dian Zheng, Cheng Zhang, Xiao-Ming Wu, Cao Li, Chengfei Lv, Jian-Fang Hu, and Wei-Shi Zheng. Panorama generation from nfov image done right. In CVPR, 2025. 3

## ProEdit: Inversion-based Editing From Prompts Done Right Supplementary Material

### A. Extracting Mask From Attention Map

An orange cat sitting on top of a fence A black cat sitting on top of a fence Editing target: orange

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

When extracting the mask from the attention map, we adopted a strategy different from DitCtrl[4]. In our observation, the attention map of the last Double block effectively associates the relevant regions of text and image. Fig 8 presents an example showing that both the editing region mask extracted from the last Double block’s attention map and the mask extracted from the average of all attention maps successfully segment the editing and non-editing regions. Additionally, this mask extraction method reduces memory consumption. Therefore, we extract our editing region mask M from the attention map of the last Double block. Notably, we always extract the mask from the first step of inversion or the last step of sampling, as the images at these time steps are least affected by noise and have the best text-to-image correlation.

Source Image Last Double Block All Blocks Edited Image

Figure 8. A visual comparison of the editing region mask extracted from the last Double block and all blocks. Using ”orange” as the editing target, the editing region masks extracted from both the last Double block and all blocks effectively segment the editing region.

Table 5. Quantitative experiments on different attention feature combinations. The best and second-best results are shown in bold and underline respectively.

Due to the downsampling operation performed in the feature space relative to the pixel space, the extracted mask has a coarser granularity and cannot fully cover the boundary regions of the editing target in the pixel space. Therefore, we apply a diffusion operation to the mask, expanding it outward by one step to obtain a coarser mask that can fully cover the editing area. Given the strong global adaptability of generative models, this relatively coarse masking is sufficient for semantic alignment. The boundary coverage between the editing and non-editing regions helps smooth the edges and avoid image artifacts. The target object of the mask extraction can be identified by the noun of the editing object or through an externally provided mask for more flexible control.

BG Preservation CLIP Sim.↑

Method

PSNR↑ SSIM (×102)↑ Whole Edited

Q&V 24.04 82.24 26.16 23.04 Q&K&V 24.51 83.04 26.20 22.97 V 23.69 81.68 26.26 23.15 K&V 24.77 84.78 26.28 23.25

Among the four combinations evaluated, the KV combination demonstrated satisfactory results in both background consistency and editing quality. Therefore, we adopted the KV fusion injection mechanism and designed the KV-mix module.

### D. More Qualitative Results for Image Editing

### B. Implementation Details

Here we provide more qualitative results for image editing in Fig 9. In cases where other inversion-based editing methods[8, 18, 21, 48, 51] fail, result in insufficient edits, or fail to maintain consistency, our method successfully achieves semantically consistent editing and demonstrates impressive performance. It is worth noting that in our qualitative results (3rd, 5th, and 6th rows in Fig 9), our method is able to effectively preserve human characteristics when editing human-centric images.

For image and video editing, we set the mix strength δ = 0.9 to balance source content preservation and editing performance. The fusion ratio β is set to 0.25 to achieve the best editing results. At each timestep, the feature fusion injection mechanism is applied to all Double and Single blocks. We fine-tune the hyperparameters of the attention feature fusion injection steps to obtain better image and video editing results. We use the official implementations of all baseline methods and adjust their hyperparameters to achieve satisfactory performance.

### E. More Qualitative Results for Video Editing

### C. Quantitative Results of Attention Feature Combination Effect

Table 5 shows the quantitative results of different attention feature combinations in the fusion injection mechanism.

Here we provide more qualitative results for video editing in Fig 10. Our method demonstrates impressive performance across a wide range of video editing tasks, while maintaining temporal consistency and preserving the original motion patterns.

### F. Editing by Instruction

To lower the barrier for using our method and make it more user-friendly, we introduce a large language model Qwen38B[45] to enable editing based on editing instructions. Fig

- 11 shows the qualitative results of our method based on editing instruction. With the assistance of a large language model, our method can directly perform edits guided by editing instructions.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

mushroomwall+booklampmintleaves+horse+dresstreesmountain−−

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

source PnP PnP-Inv RF-Solver FireFlow UniEdit Ours

###### Figure 9. More qualitative comparison of image editing on PIE-Bench[21].

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Source Video

jeep blue jeep + Crown

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Target Video

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Source Video

car truck + roof rack

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Target Video

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Source Video

sunny rainy

deer cow

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Target Video

###### Figure 10. More video editing results.

Change the cat sitting on a wooden chair into a dog.

Change the color of the flowers from pink to red.

Change the birds sitting on a branch into origami birds.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Source Image Edited Image

Source Image Edited Image

Source Image Edited Image

Remove the paraglider flying over the mountain.

Add a hat to the cat.

Change the bulldog to rat.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Source Image Edited Image

Source Image Edited Image

Source Image Edited Image

Figure 11. Qualitative results of image editing based on editing instruction. The actual input editing instruction are shown above each source image and its corresponding edited image.

