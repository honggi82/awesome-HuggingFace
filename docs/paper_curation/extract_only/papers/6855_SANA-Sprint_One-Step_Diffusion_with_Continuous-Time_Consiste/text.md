### arXiv:2503.09641v4[cs.GR]29Sep2025

[Figure 1]

2025-9-30

#### SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

Junsong Chen1* Shuchen Xue5* Yuyang Zhao1† Jincheng Yu1† Sayak Paul4 Junyu Chen3 Dongyun Zou3 Han Cai1 Song Han1,2 Enze Xie1

1NVIDIA 2MIT 3Tsinghua University 4Huggingface 5Independent Researcher

*Equal contribution. †Core contribution.

Abstract: This paper presents SANA-Sprint, an efficient diffusion model for ultra-fast text-to-image (T2I) generation. SANA-Sprint is built on a pre-trained foundation model and augmented with hybrid distillation, dramatically reducing inference steps from 20 to 1-4. We introduce three key innovations: (1) We propose a training-free approach that transforms a pre-trained flow-matching model for continuous-time consistency distillation (sCM), eliminating costly training from scratch and achieving high training efficiency. Our hybrid distillation strategy combines sCM with latent adversarial distillation (LADD): sCM ensures alignment with the teacher model, while LADD enhances single-step generation fidelity. (2) SANA-Sprint is a unified step-adaptive model that achieves high-quality generation in 1-4 steps, eliminating step-specific training and improving efficiency. (3) We integrate ControlNet with SANA-Sprint for real-time interactive image generation, enabling instant visual feedback for user interaction. SANA-Sprint establishes a new Pareto frontier in speed-quality tradeoffs, achieving state-of-the-art performance with 7.59 FID and 0.74 GenEval in only 1 step outperforming FLUX-schnell (7.94 FID / 0.71 GenEval) while being 10× faster (0.1s vs 1.1s on H100). It also achieves 0.1s (T2I) and

- 0.25s (ControlNet) latency for 1024×1024 images on H100, and 0.31s (T2I) on an RTX 4090, showcasing its exceptional efficiency and potential for AI-powered consumer applications (AIPC). Code and pre-trained models will be open-sourced. Links: GitHub Code | HF Models | Project Page
- 1. Introduction

diffusion model, which increases computational overhead and imposes significant pressure on GPU memory, and requires careful tuning [20]. Consistency models, while stable, suffer quality erosion in ultra-few-step regimes (e.g., <4 steps), particularly in text-to-image tasks where trajectory truncation errors degrade semantic alignment. These challenges underscore the need for a distillation framework that harmonizes efficiency, flexibility, and quality.

The computational intensity of diffusion generative models [1, 2], typically requiring 50-100 iterative denoising steps, has driven significant innovation by time-step distillation to enable efficient inference. Current methodologies primarily coalesce into two dominant paradigms: (1) distribution-based distillations like GAN [3] (e.g., ADD [4], LADD [5]) and its variational score distillation (VSD) variants [6, 7, 8] leverage joint training to align single-step outputs with multi-step teacher’s distributions, and (2) trajectory-based distillations like Direct Distillation [9], Progressive Distillation [10, 11], Consistency Models (CMs) [12] (e.g. LCM [13], CTM [14], MCM [15], PCM [16], sCM [17]) learn ODE solution across reduced sampling intervals. Together, these methods achieve 10-100× image generation speedup while maintaining competitive generation quality, positioning distillation as a critical pathway toward practical deployment.

In this work, we present SANA-Sprint, an efficient diffusion model for one-step high-quality text-to-image (T2I) generation. Our approach builds on a pre-trained image generation model SANA and recent advancements in continuous-time consistency models (sCMs) [17], preserving the benefits of previous consistency-based models while mitigating the discretization errors of their discretetime counterparts. To achieve the one-step generation, we first transform SANA, a Flow Matching model, to the TrigFlow model, which is required for sCM distillation, through a lossless mathematical transformation. Then, to mitigate the instability of distillation, we adapt the QK norm in self- and cross-attention in SANA along with dense time embeddings to allow efficient knowledge transfer from the pre-trained models without retraining the teacher model. We further combine sCM with LADD’s adversarial distillation to enable fast convergence and highfidelity generation while retaining the advantages of sCMs. Note that, although validated primarily on SANA, our method can benefit other mainstream flow-matching models such as FLUX [21] and SD3 [22].

Despite their promise, key limitations hinder broader adoption. GAN-based methods suffer from training instability due to oscillatory adversarial dynamics and mode collapse. GANs face challenges due to the need to map noise to natural images without supervision, making unpaired learning more ill-posed than paired learning, as highlighted in [18, 19]. This instability is compounded by architectural rigidity, which demands meticulous hyperparameter tuning when adapting to new backbones or settings. VSDbased methods involve the joint training of an additional

© 2025 NVIDIA. All rights reserved.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

|Sana Baseline Deep Compression AutoEncoder + Linear DiT<br><br>+ Flow DPM-Solver Sana ﬁnal version)<br><br>+ Kernel Fusion<br><br>Flux-dev|S<br><br>9.6<br><br>21<br><br>24<br><br>41|ANA-Sprint|: One-Step D|iffusion with<br><br>4|Continuous-T<br><br>69|ime Consist|ency Distillat|ion|1023| |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

+

[Figure 2]

(

Params: 12B, GenEval: 0.71, FID:7.94

[Figure 3]

Text Encode: <0.05s

>80GB

Flux-Schnell 4 steps

>80GB

VAE: 0.15s Transformer: 1.94s

OOM

67GB

SANA 20 steps

Transformer: 1.18s 1.6x

VAE: 0.12s

bs=1 45GB bs=1

SANA-Sprint

bs=32

VAE: 0.12s

0.14s

8.4x

4 steps SANA-Sprint

bs=2

20GB bs=2

39.3x

VAE: 0.12s 0.03s

Params: 1.6B, GenEval: 0.74, FID:7.59

1 step

Flux Schnell 12B

SANA Sprint 1.6B

SANA Sprint 1.6B

SANA Sprint 0.6B

SDXL

64.7x

DMD2 0.9B

(a) Generation latency (s) on 1024x1024

(b) Training GPU Memory (GB)

- Figure 1 | (a) Our SANA-Sprint accelerate the inference speed for generating 1024 × 1024 images, achieving a remarkable speedup from FULX-Schnell’s 1.94 seconds to only 0.03 seconds. This represents a 64× improvement over the current state-of-the-art step-distilled model, FLUX-Schnell, as measured with a batch size of 1 on an NVIDIA A100 GPU. The ratio is calculated based on Transformer latency. (b) Additionally, our model demonstrates efficient GPU memory usage during training, outperforming other distillation methods in terms of memory cost. The GPU memory is measured using official code, 1024 × 1024 images and on a single A100 GPU.

67GB bs=32

34GB bs=32

1

SANA Sprint 1.6B

SANA Sprint 0.6B

Done

As a result, SANA-Sprint achieves excellent speed/quality tradeoff, benefiting from a hybrid objective, inheriting sCM’s diversity preservation and alignment with the teacher, while integrating LADD’s fidelity enhancement: experiments show a 0.6 lower FID and 0.4 higher CLIPScore at 2-step generations compared to standalone sCM, with 3.9 lower FID and 0.9 higher CLIP-Score over pure latent adversarial approaches. As shown in Fig. 1, SANASprint achieves state-of-the-art performance in FID and GenEval benchmark, surpassing recent advanced methods including SD3.5-Turbo, SDXL-DMD2, and Flux-schnell. Especially, SANA-Sprint is 64.7× faster than Flux-Schnell and exceeds in FID (7.59 vs 7.94) and GenEval (0.74 vs 0.71).

SANA-Sprint generates a 1024×1024 image in only 0.10s-0.18s on H100, achieving state-of-the-art 7.59 FID on MJHQ-30K and 0.74 GenEval score - surpassing FLUX-schnell (7.94 FID/0.71 GenEval) while being 10× faster.

• Real-Time Interactive Generation: By integrating ControlNet with SANA-Sprint, we enable real-time interactive image generation in 0.25s on H100. This facilitates immediate visual feedback in human-inthe-loop creative workflows, enabling better humancomputer interaction.

##### 2. Preliminaries

Moreover, SANA-Sprint demonstrates unprecedented inference speed—generating 1024×1024 images in 0.31 seconds on a laptop with consumer-grade GPUs (NVIDIA RTX 4090) and 0.1 seconds on H100 GPU, 8.4× speedup than teacher model SANA. This efficiency unlocks transformative applications that require instant visual feedback: in ControlNet-guided image generation/editing, by integrating with ControlNet, SANA-Sprint enables instant interaction with 250ms latency on H100. SANA-Sprint exhibits robust scalability and is potentially suitable for human-in-the-loop creative workflows, AIPC, and immersive AR/VR interfaces.

In summary, our key contributions are threefold:

- • Hybrid Distillation Framework: We designed an innovative hybrid distillation framework that seamlessly transforms the flow model into the Trigflow model, integrating continuous-time consistency models (sCM) with latent adversarial diffusion distillation (LADD). This framework leverages sCM’s diversity preservation and alignment with the teacher alongside LADD’s fidelity enhancement, enabling unified step-adaptive sampling.
- • Excellent Speed/Quality Tradeoff: SANA-Sprint achieves exceptional performance with only 1-4 steps.

###### 2.1. Diffusion Model and Its Variants

Diffusion models [1, 2] diffuse clean data sample 𝑥0 ∼ 𝑝𝑑𝑎𝑡𝑎 from data distribution to noisy data 𝑥𝑡 = 𝛼𝑡𝑥0 + 𝜎𝑡𝑧, where 𝑡 ∈ [0,𝑇] represents time within the interval, 𝑧 ∼ 𝒩(0,𝐼) is a standard Gaussian noise. The terminal distribution 𝑝𝑇 of 𝑥𝑇 exactly or approximately follows a Gaussian distribution. Typically, diffusion models train a noise prediction network 𝜖𝜃 using E𝑥

0,𝑧,𝑡[‖𝜖𝜃(𝑥𝑡,𝑡) − 𝑧‖2], which is equivalent to denoising score matching loss [2, 23]. The sampling process of diffusion models involves solving the probability flow ODE (PF-ODE) [2] d𝑥𝑡

d𝑡 = d log𝛼𝑡

d𝑡 − d log𝛼

d𝑡 𝑥𝑡 + (d𝜎𝑡

d𝑡 𝜎𝑡)𝜖𝜃(𝑥𝑡,𝑡) with the initial value 𝑥1 ∼ 𝒩(0,𝐼). Below, we will introduce two recent formulations of diffusion models that have received significant attention.

𝑡

Flow Matching [24, 25, 26] considers a linear interpolation noising process by defining 𝛼𝑡 = 1−𝑡,𝜎𝑡 = 𝑡,𝑇 = 1. The flow matching models train a velocity prediction network 𝑣𝜃 using E𝑥

0,𝑧,𝑡[𝑤(𝑡)‖𝑣𝜃(𝑥𝑡,𝑡) − (𝑧 − 𝑥0)‖2], where 𝑤(𝑡) is a weighting function. The sampling of flow models solves the PF-ODE d𝑥𝑡

d𝑡 = 𝑣𝜃(𝑥𝑡,𝑡) with the initial value 𝑥1 ∼ 𝒩(0,𝐼).

TrigFlow [17, 27] considers a spherical linear interpolation noising process by defining 𝛼𝑡 = cos(𝑡),𝜎𝑡 =

sin(𝑡),𝑇 = 𝜋2. Moreover, Trigflow assumes the noise 𝑧 ∼ 𝒩(0,𝜎𝑑2𝐼), where 𝜎𝑑 represents the standard deviation of data distribution 𝑝𝑑𝑎𝑡𝑎. The TrigFlow models train a velocity prediction network 𝐹𝜃 using E𝑥

𝜎𝑑,𝑡)−(cos(𝑡)𝑧 −sin(𝑡)𝑥0)‖2], where 𝑤(𝑡) is a weighting function. The sampling of TrigFlow models solves the PF-ODE defined by d𝑥𝑡

0,𝑧,𝑡[𝑤(𝑡)‖𝜎𝑑𝐹𝜃(𝑥𝑡

d𝑡 = 𝜎𝑑𝐹𝜃(𝑥𝑡

2 ∼ 𝒩(0,𝜎𝑑2𝐼).

𝜎𝑑,𝑡) starting from 𝑥𝜋

Diffusion, Flow Matching, and TrigFlow are all continuous-time generative models that differ in their interpolation schemes and velocity field parameterizations.

###### 2.2. Consistency Models

A consistency model (CM) [12] parameterizes a neural network 𝑓𝜃(𝑥𝑡,𝑡) which is trained to predict the solution 𝑥0 of the PF-ODE, which is the terminal clean data along the trajectory of the PF-ODE (regardless of its position in the trajectory), starting from the noisy observation 𝑥𝑡. The conventional approach parameterizes the CM using skip connections, bearing a close resemblance to [28, 29]

𝑓𝜃(𝑥𝑡,𝑡) = 𝑐skip(𝑡)𝑥𝑡 + 𝑐out(𝑡)𝐹𝜃(𝑥𝑡,𝑡), (1)

where 𝑐skip(𝑡) and 𝑐out(𝑡) are differentiable functions satisfying 𝑐skip(0) = 1 and 𝑐out(0) = 0 to ensure the boundary conditions 𝑓𝜃(𝑥0,0) = 𝑥0. 𝐹𝜃 indicates the pretrained diffusion/flow model and 𝑓𝜃 is the data prediction model. Based on the training approach, CMs can be categorized into two types: discrete-time [12, 13] and continuoustime [12, 17].

Discrete-time CMs are trained with the following objective

𝑙𝐶𝑀Δ𝑡 = E𝑥

𝑡,𝑡[𝑑(𝑓𝜃(𝑥𝑡,𝑡),𝑓𝜃−(𝑥𝑡−Δ𝑡,𝑡 − Δ𝑡))], (2)

where 𝜃− is the stopgrad version of 𝜃, 𝑤(𝑡) is the weighting function, Δ𝑡 is a small time interval, and 𝑥𝑡−Δ𝑡 is obtained from 𝑥𝑡 by running a numerical ODE solver. 𝑑(·,·) is a metric such as ℓ1, squared ℓ2, Pseudo-Huber loss, and the LPIPS loss [30].

Although discrete-time CMs work well in practice, the additional discretization errors brought by numerical ODE solvers are inevitable. Continuous-time CMs correspond to the limiting case of Δ𝑡 → 0 in Eq. (2). When choosing 𝑑(𝑥,𝑦) = ‖𝑥 − 𝑦‖22, the expression simplifies to:

= E𝑥𝑡,𝑡 [︁𝑤(𝑡)⟨𝑓𝜃(𝑥𝑡, 𝑡),

(𝑥𝑡, 𝑡)⟩]︁,

𝑙𝐶𝑀Δ𝑡 Δ𝑡

d𝑓𝜃− d𝑡

𝑙𝐶𝑀𝑐𝑜𝑛𝑡. := lim

Δ𝑡→0

(3)

d𝑡 = 𝜕𝑓𝜃−(𝑥𝑡,𝑡)

where d𝑓𝜃−(𝑥𝑡,𝑡)

###### 𝑓𝜃−(𝑥𝑡,𝑡)d𝑥𝑡

𝜕𝑡 +∇𝑥𝑡

d𝑡 . The infinitesimal step of d𝑥𝑡

d𝑡 replaces numerical ODE solvers, thereby eliminating discretization errors.

Specifically, under TrigFlow where 𝑐skip(𝑡) = cos(𝑡) and 𝑐out(𝑡) = −sin(𝑡), sCM’s parameterization and arithmetic coefficients are simplified to the following form:

𝑥𝑡 𝜎𝑑

𝑓𝜃(𝑥𝑡,𝑡) = cos(𝑡)𝑥𝑡 − sin(𝑡)𝜎𝑑𝐹𝜃(

,𝑡), (4) and the time derivative becomes:

= − cos(𝑡)(︂𝜎𝑑𝐹𝜃−(

)︂

d𝑓𝜃−(𝑥𝑡,𝑡) d𝑡

d𝑥𝑡 d𝑡

𝑥𝑡 𝜎𝑑

,𝑡) −

)︃.

− sin(𝑡)(︃𝑥𝑡 + 𝜎𝑑

(5)

𝜎𝑑,𝑡) d𝑡

d𝐹𝜃−(𝑥𝑡

##### 3. Method

sCM [17] simplify continuous-time CMs using the TrigFlow formulation. While this provides an elegant framework, most score-based generative models are based on diffusion or flow matching formulations. One possible approach is to develop separate training algorithms for continuous-time CMs under these formulations, but this requires distinct algorithm designs and hyperparameter tuning, increasing complexity. Alternatively, one could pretrain a dedicated TrigFlow model, as in [17], but this significantly increases computational cost.

To address these challenges, we propose a simple method to transform a pre-trained flow matching model into a TrigFlow model through straightforward mathematical input and output transformations. This approach makes it possible to strictly follow the training algorithm in [17], eliminating the need for separate algorithm designs while fully leveraging existing pre-trained models. The transformation process for general diffusion models can be carried out in a similar manner, which we omit here for simplicity.

###### 3.1. Training-Free Transformation to TrigFlow

Score-based generative models (diffusion, flow matching, and TrigFlow) can denoise data with proper data scales and signal-to-noise ratios (SNRs)1 aligned with training. However, flow matching cannot directly denoise TrigFlowscheduled data due to three mismatches: First, their time domains differ: TrigFlow uses [0, 𝜋2], while flow matching is defined on [0,1]. Second, their noise schedules are distinct—TrigFlow maintains cos2(𝑡Trig) + sin2(𝑡Trig) = 1, while flow matching yields 𝑡2FM + (1 − 𝑡FM)2 < 1, causing data scale discrepancies. Finally, their prediction targets differ: flow matching predicts 𝑧 − 𝑥0 with static coefficients (1,−1), whereas TrigFlow predicts cos(𝑡)𝑧 − sin(𝑡)𝑥0 with time-varying coefficients. These mismatches in temporal parameterization, SNR, and output necessitate explicit input/output transformations.

To clarify, we use the subscript Trig to denote noisy data under the TrigFlow framework and FM to denote noisy data under the flow matching framework. The following

2 𝑡

1For a diffusion model 𝑥𝑡 = 𝛼𝑡𝑥0 + 𝜎𝑡𝑧, SNR is defined as 𝛼

𝜎𝑡2

E6F0E7 F7E8E6 E6EDF2

Input Timestep Transform

TrigFlow

sin(tTrig) sin(tTrig) + cos(tTrig)

Fpretrain

Transform

[Figure 4]

tFM =

[Figure 5]

xt,FM

dx/dt

Input Latent Transform

Teacher

xt,Trig σd

Self (x0̂ ,JVP) Consistency

⋅ tFM2 + (1 − tFM)2

xt,FM =

sCM Loss

[Figure 6]

[Figure 7]

xt,FM

Output Transform

Fθ ̂(

Student

1 tFM2 + (1 − tFM)2

[(1 − 2tFM)xt,FM

xt,Trig σd

x0̂

,tTrig, y) =

fθ

[Figure 8]

+(1 − 2tFM + 2tFM2 )vθ(xt,FM,tFM, y)]

Discriminator Heads

[Figure 9]

[Figure 10]

xt,Trig

[Figure 11]

[Figure 12]

[Figure 13]

GAN Loss

[Figure 14]

Fake

[Figure 15]

xt,Trig ϵt′

[Figure 16]

x0

Teacher

| | |
|---|---|
| | |

DC-AE

[Figure 17]

[Figure 18]

Real

Fpretrain

ϵt

[Figure 19]

- Figure 2 | Training paradigm of SANA-Sprint. In SANA-Sprint, we use the student model for synthetic data

generation (𝑥^0) and JVP calculation, and we use the teacher model for velocity (d𝑥/d𝑡) compute and its feature for the GAN loss, which allows us train sCM and GAN together and have only one training model purely in the latent space. Details of training objective and TrigFlow Transformation are in Eq. (9), Eq. (11) and Sec. 3.1.

[Figure 20]

Input Timestep Transform

sin(tTrig) sin(tTrig) + cos(tTrig)

tFM =

,

Done

###### Table 1 | Comparison of original Flow-based SANA model and training-free transformation of TrigFlowbased SANA-Sprint model. We evaluate the FID and CLIP-Score before and after the transformation in Sec. 3.1.

proposition outlines the transformation from flow matching models to TrigFlow models, which is theoretically lossless.

Input Latent Transform

xt,Trig σd

⋅ tFM2 + (1 − tFM)2

xt,FM =

Output Transform

###### Fθ ̂(

1 tFM2 + (1 − tFM)2

[(1 − 2tFM)xt,FM

xt,Trig σd

,tTrig, y) =

Remark. We prioritize seamlessly transforming existing noise schedules, e.g.flow matching, into TrigFlow while integrating the sCM framework with minimal modifications. This approach avoids the need for pre-training a dedicated TrigFlow model, as in [17], although it involves a deviation from the unit variance principle in [17, 28].

Pipeline

Method FID CLIP-Score

t t2 t ]

|+(1 − 2 FM<br><br>|+2FM)↓vθ(xt|,FM, FM,y) ↑|
|---|---|---|
|Flow Euler 50 steps TrigFlow Euler 50 steps|5.81 5.73|28.810 28.806|

Self Consistency Loss. With the lossless transformation established, we can seamlessly adopt the training algorithm and pipeline of sCM without other modification. This allows us to directly follow the sCM training framework. Our final sCM loss is the following:

Proposition 3.1. Given a noisy data 𝑥𝑡,𝜎Trig

under TrigFlow noise schedule, a flow matching model can denoise it via 𝑣𝜃(𝑥𝑡,FM,𝑡FM,𝑦), where

𝑑

ℒsCM(𝜃,𝜑) = E𝑥𝑡,𝑡[︁

, 𝑡,𝑦)︁

⃦𝐹̂︁𝜃 (︁

𝑒𝑤𝜑(𝑡) 𝐷

𝑥𝑡 𝜎𝑑

sin (𝑡Trig) sin (𝑡Trig) + cos (𝑡Trig)

𝑡FM =

, (6)

− 𝐹̂︂𝜃− (︁

− 𝑤𝜑(𝑡)]︁

, 𝑡,𝑦)︁ − cos(𝑡)

d𝑓̂︂𝜃−(𝑥𝑡, 𝑡, 𝑦) d𝑡

2

𝑥𝑡 𝜎𝑑

⃦

𝜎𝑑 · √︀𝑡2FM + (1 − 𝑡FM)2. (7)

2

𝑥𝑡,FM = 𝑥𝑡,Trig

(9)

where 𝑓̂︂𝜃− is the parameterized sCM as in Eq. (4) after replacing 𝐹𝜃 with 𝐹̂︁𝜃 in Proposition. 3.1, 𝑥𝑡,𝑡 refers to 𝑥𝑡,Trig,𝑡Trig, and 𝑤𝜑(𝑡) is an adaptive weighting function to minimize variance across different timesteps following [17, 31].

Given 𝑣𝜃(𝑥𝑡,FM,𝑡FM,𝑦), the best estimator for the TrigFlow model 𝐹𝜃 is the following:

𝐹̂︁𝜃 (︁

, 𝑡Trig,𝑦)︁

𝑥𝑡,Trig 𝜎𝑑

√︀𝑡2FM + (1 − 𝑡FM)2 [︁(1 − 2𝑡FM)𝑥𝑡,FM

1

=

###### 3.2. Stabilizing Continuous-Time Distillation

(8)

+ (1 − 2𝑡FM + 2𝑡2FM)𝑣𝜃(𝑥𝑡,FM, 𝑡FM,𝑦)]︁.

To stabilize continuous-time consistency distillation, we address two key challenges: training instabilities and excessively large gradient norms that occur when scaling up the model size and increasing resolution, leading to model collapse. We achieve this by refining the time-embedding to be denser and integrating QK-Normalization into selfand cross-attention mechanisms. These modifications enable efficient training and improve stability, allowing for robust performance at higher resolutions and larger model sizes.

Furthermore, the transformation is lossless in theory.

The details and proof of Proposition. 3.1 are in Sec. D. The transformations of both input and output are all differentiable making it compatible with auto differentiation. As validated by Tab. 1, the transformation is lossless in both theory and practice. The training-free transformation is depicted in the gray box of Fig. 2.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-Sprint : One-Step Diffusion with Continuous-Time Consistency Distillation

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

10K iterations visualizations

15K iterations visualizations

(d) w/ or w/o schedule transfer

[Figure 28]

[Figure 29]

[Figure 30]

15K iterations visualizations

(a) QK-Norm vs Grad.

(b) Time embedding vs Grad.

(c) Timestep Embedding Similarity

- Figure 3 | Efficient Distillation via QK Normalization, Dense Timestep Embedding, and Training-free Schedule Transformation. (a) We compare gradient norms and visualizations with/without QK Normalization, showing its stabilizing effect. (b) Gradient norm curves for timestep scales (0∼1 vs. 0∼1000) highlight impacts on stability and stability and quality. (c) PCA-based similarity analysis of timestep embeddings. (d) Image results after 5,000 iterations of fine-tuning with (left) and without (right) the proposed schedule transfer (Sec. 3.1).

Done

[Figure 31]

[Figure 32]

timestep curve

Dense Time-Embedding. As analyzed in sCM [17], the instability issues in continuous-time CMs primarily stem from the unstable scale of d𝑓

###### 3.3. Improving Continuous-Time CMs with GAN

CTM [14] analyzes that CMs distill teacher information in a local manner, where at each iteration, the student model learns from local time intervals. This leads the model to learn cross timestep information under the implicit extrapolation, which can slow the convergence speed. To address this limitation, we introduce an additional adversarial loss [5] to provide direct global supervision across different timesteps, improving both the convergence speed and the output quality.

d𝑡 in Eq. (9). This instability can be traced back to the expression

𝜃

d𝐹𝜃−

d𝑥𝑡

d𝑡 +𝜕𝑡𝐹𝜃− in Eq. (5), which ultimately originates from the time derivative term 𝜕𝑡𝐹𝜃−:

d𝑡 = ∇𝑥𝑡

𝐹𝜃−

𝜕𝑐noise(𝑡) 𝜕𝑡

𝜕emb(𝑐noise) 𝜕𝑐noise ·

𝜕𝑡𝐹𝜃− = 𝜕𝐹𝜃− 𝜕emb(𝑐noise) ·

(10)

In previous flow matching models like SANA [32], SD3 [22], and FLUX [21], the noise coefficient 𝑐noise(𝑡) = 1000𝑡 amplifies the time derivative 𝜕𝑡𝐹𝜃− by a factor of 1000, leading to significant training fluctuations. To address this, we set 𝑐noise(𝑡) = 𝑡 and fine-tuned SANA for 5k iterations. As shown in Fig. 3 (b), this adjustment reduce excessively large gradient norms (originally exceeding 103) to more stable levels. Furthermore, PCA visualization in Fig. 3 (c) reveals that our dense time-embedding design results in more densely packed and similar embeddings for timesteps between 0∼1. This refinement improved training stability and accelerated convergence over 15k iterations.

GANs [3] consist of a generator 𝐺 and a discriminator 𝐷 that compete in a zero-sum game to produce realistic synthetic data. Diffusion-GANs [34] and LADD [5] extend this framework by enabling the discriminator to distinguish between noisy real and fake samples. Furthermore, LADD introduces a novel approach by utilizing a frozen teacher model as a feature extractor and training multiple discriminator heads on the teacher model. This methodology facilitates direct adversarial supervision in the latent space, as opposed to the traditional pixel space, leading to more efficient and effective training. Following LADD, we use a hinge loss [35] to train the student model and discriminator

QK-Normalization. When scaling up the model from 0.6B to 1.6B, we encounter similar issues with excessively large gradient norms, often exceeding 103, which lead to training collapse. To address this, we introduce RMS normalization [33] to the Query and Key in both selfand cross-attention modules of the teacher model during fine-tuning. This modification enhances training stability significantly, even with a brief fine-tuning process of only 5,000 iterations. By using the fine-tuned teacher model to initialize the student model, we achieve a more stable gradient norm, as shown in Fig. 3 (a), thereby making the distillation process viable where it was previously infeasible.

ℒ𝐺adv(𝜃)

= −E𝑥0,𝑠,𝑡[︁∑︁

𝐷𝜓,𝑘(𝐹𝜃pre,𝑘(^𝑥𝑓𝑠𝜃, 𝑠,𝑦))]︁,

(11)

𝑘

ℒ𝐷adv(𝜓)

= E𝑥0,𝑠[︁∑︁

ReLU(︁1 − 𝐷𝜓,𝑘(𝐹𝜃pre,𝑘(𝑥𝑠, 𝑠,𝑦)))︁]︁

𝑘

+ E𝑥0,𝑠,𝑡[︁∑︁

ReLU(︁1 + 𝐷𝜓,𝑘(𝐹𝜃pre,𝑘(^𝑥𝑠𝑓𝜃−, 𝑠,𝑦)))︁]︁,

𝑘

(12)

𝑠 ,𝑥^𝑓

where 𝑥𝑠,𝑥^𝑓

𝑠 are the noisy versions of 𝑥0,𝑥^𝑓

0 := 𝑓𝜃(𝑥𝑡,𝑡,𝑦),𝑥^𝑓

𝜃−

𝜃

𝜃

0 := 𝑓𝜃−(𝑥𝑡,𝑡,𝑦).

𝜃−

SDXL-PCM† 0.88s

SDXL-DMD2† 0.54s

SDXL-LCM 0.54s

SD3.5-Turbo 1.15s

Flux-Schnell 2.10s

###### SANA-Sprint 0.31s

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

- 1Step

A bucket bag made of blue suede with intricate golden paisley patterns. The handle of the bag is made of rubies and pearls.

- 2Steps

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

4Steps

3d digital art of an adorable ghost, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background

- Figure 4 | Visual comparison among SANA-Sprint and selected competing methods in different inference steps. † indicates that distinct models are required for different inference steps, and time below the method name is the latency of 4 steps tested on A100 GPU. SANA-Sprint produces images with superior realism and text alignment in all inference steps with the fastest speed.

The adversarial loss ℒadv is equivalent to the GAN loss shown at the bottom of Fig. 2. In summary, SANA-Sprint combines sCM loss with GAN loss: ℒ = ℒ𝑠𝐶𝑀 + 𝜆ℒadv, where 𝜆 = 0.5 by default, as in Tab. 5.

framework to obtain SANA-Sprint-ControlNet.

For ControlNet tasks, we extract Holistically-Nested Edge Detection (HED) scribbles from input images as conditions to guide image generation. Following PixArt’s [40] design principles, we train our SANA-ControlNet teacher model on 1024×1024 resolution images. During sampling, HED maps serve as additional conditioning inputs to the Transformer model, allowing precise control over image generation while maintaining structural details. Our experiments show that the distilled SANA-Sprint-ControlNet model retains the controllability of the teacher model and achieves fast inference speeds of approximately 200 ms on H100 machines, enabling near-real-time interaction. The effectiveness of our approach is demonstrated in Sec. F.3.

Additional Max-Time Weighting. In our early experiments, we adopt the timestep sampling distribution of sCM’s generator (student model) for GAN loss, given by 𝑡 = arctan (𝑒

𝜎𝑑), where 𝜏 ∼ 𝒩(𝑃mean,𝑃std2 ) with two hyperparameters 𝑃mean and 𝑃std. To further enhance oneand few-step generation performance and improve overall generation quality, we introduce an additional weighting at 𝑡 = 𝜋2. Specifically, with probability 𝑝, the training timestep is set to 𝜋2, while with probability 1−𝑝, it follows the original timestep sampling distribution of sCM’s generator. We find that this modification significantly improves the model’s capability for one- and few-step generation, as shown in Tab. 6.

𝜏

##### 4. Experiments

###### 3.4. Application: Real-Time Interactive Generation

###### 4.1. Experimental Setup

Extending the SANA-Sprint to image-to-image tasks is straightforward. We apply the SANA-Sprint training pipeline to ControlNet [41] tasks, which utilize both images and prompts as instructions. Our approach involves continuing the training of a pre-trained text-to-image diffusion model with a diffusion objective on a dataset adjusted for ControlNet tasks, resulting in the SANA-ControlNet model. We then distill this model using the SANA-Sprint

Our experiments employ a two-phase training strategy, with detailed settings and evaluation protocols outlined in Sec. F.1. The teacher models are pruned and fine-tuned from the larger SANA-1.5 4.8B model [42], followed by distillation using our proposed training paradigm. We evaluate performance using metrics including FID, CLIP Score on the MJHQ-30K [43], and GenEval [44].

- Table 2 | Comprehensive comparison of SANA-Sprint with SOTA approaches in efficiency and performance. The speed is tested on one A100 GPU with BF16 Precision. Throughput: Measured with batch=10. Latency: Measured with batch=1. We highlight the best and second best entries. † indicates that distinct models are required for different inference steps.

Inference Throughput Latency Params steps (samples/s) (s) (B) FID ↓ CLIP ↑ GenEval ↑

Methods

SDXL [36] 50 0.15 6.5 2.6 6.63 29.03 0.55 PixArt-Σ [37] 20 0.4 2.7 0.6 6.15 28.26 0.54 SD3-medium [38] 28 0.28 4.4 2.0 11.92 27.83 0.62 FLUX-dev [21] 50 0.04 23.0 12.0 10.15 27.47 0.67 Playground v3 [39] - 0.06 15.0 24 - - 0.76

Pre-trainModels

- SANA 0.6B [32] 20 1.7 0.9 0.6 5.81 28.36 0.64
- SANA 1.6B [32] 20 1.0 1.2 1.6 5.76 28.67 0.66

SDXL-LCM [13] 4 2.27 0.54 0.9 10.81 28.10 0.53 PixArt-LCM [40] 4 2.61 0.50 0.6 8.63 27.40 0.44 PCM [16]† 4 1.95 0.88 0.9 15.55 27.53 0.56 SD3.5-Turbo [22] 4 0.94 1.15 8.0 11.97 27.35 0.72 SDXL-DMD2 [20]† 4 2.27 0.54 0.9 6.82 28.84 0.60 FLUX-schnell [21] 4 0.5 2.10 12.0 7.94 28.14 0.71

- SANA-Sprint 0.6B 4 5.34 0.32 0.6 6.48 28.45 0.76

- SANA-Sprint 1.6B 4 5.20 0.31 1.6 6.54 28.45 0.77

SDXL-LCM [13] 2 2.89 0.40 0.9 18.11 27.51 0.44 PixArt-LCM [40] 2 3.52 0.31 0.6 10.33 27.24 0.42 SD3.5-Turbo [22] 2 1.61 0.68 8.0 51.47 25.59 0.53 PCM [16]† 2 2.62 0.56 0.9 14.70 27.66 0.55 SDXL-DMD2 [20]† 2 2.89 0.40 0.9 7.61 28.87 0.58 FLUX-schnell [21] 2 0.92 1.15 12.0 7.75 28.25 0.71

DistillationModels

- SANA-Sprint 0.6B 2 6.46 0.25 0.6 6.54 28.40 0.76

- SANA-Sprint 1.6B 2 5.68 0.24 1.6 6.50 28.45 0.77

SDXL-LCM [13] 1 3.36 0.32 0.9 50.51 24.45 0.28 PixArt-LCM [40] 1 4.26 0.25 0.6 73.35 23.99 0.41 PixArt-DMD [37]† 1 4.26 0.25 0.6 9.59 26.98 0.45 SD3.5-Turbo [22] 1 2.48 0.45 8.0 52.40 25.40 0.51 PCM [16]† 1 3.16 0.40 0.9 30.11 26.47 0.42 SDXL-DMD2 [20]† 1 3.36 0.32 0.9 7.10 28.93 0.59 FLUX-schnell [21] 1 1.58 0.68 12.0 7.26 28.49 0.69

- SANA-Sprint 0.6B 1 7.22 0.21 0.6 7.04 28.04 0.72

- SANA-Sprint 1.6B 1 6.71 0.21 1.6 7.69 28.27 0.76

###### 4.2. Efficiency and Performance Comparison

We compare SANA-Sprint with state-of-the-art text-toimage diffusion and timestep distillation methods in Tab. 2 and Fig. 4. Our SANA-Sprint models focus on timestep distillation, achieving high-quality generation with 1-4 inference steps, competing with the 20-step teacher model, as shown in Fig. 5. More details about the timestep setting are given in Sec. F.2.

Specifically, with 4 steps, SANA-Sprint 0.6B achieves

###### 5.34 samples/s throughput and 0.32s latency, with an FID of 6.48 and GenEval of 0.76. SANA-Sprint 1.6B has slightly lower throughput (5.20 samples/s) but improves GenEval to 0.77, outperforming larger models like FLUX-schnell (12B), which achieves only 0.5 samples/s with 2.10s latency. At 2 steps, SANA-Sprint models remain efficient: SANA-Sprint 0.6B reaches 6.46 samples/s with 0.25s latency (FID: 6.54), while SANA-Sprint 1.6B

achieves 5.68 samples/s with 0.24s latency (FID: 6.76). In single-step mode, SANA-Sprint 0.6B achieves 7.22 samples/s throughput and 0.21s latency, maintaining an FID of 7.04 and GenEval of 0.72, comparable to FLUX-schnell but with significantly higher efficiency.

These results demonstrate the practicality of SANA-Sprint for real-time applications, combining fast inference speeds with strong performance metrics.

4.3. Analysis In this section, we apply a 2-step sampling starting at 𝑡𝑚𝑎𝑥 = 𝜋/2 with an intermediate step 𝑡 = 1.0.

Schedule Transfer. To validate the effectiveness of our proposed schedule transfer in Sec. 3.1, we conduct ablation studies on a flow matching model SANA [32], comparing its performance with and without schedule transformation

- Table 3 | Comparison of loss combination.

Table 4 | Comparison of CFG training strategies.

Table 5 | sCM and LADD loss weighting.

Table 6 | Comparison of maxtime weighting strategy.

sCM LADD FID ↓ CLIP ↑ ✓ 8.93 27.51

sCM:LADD FID ↓ CLIP ↑

Max-Time FID ↓ CLIP ↑

CFG Setting FID ↓ CLIP ↑

1.0:1.0 8.81 27.93 1.0:0.5 8.43 27.85 1.0:0.1 8.90 27.76

0% maxT 9.44 27.65 50% maxT 8.32 27.94 70% maxT 8.11 28.02

w/o Embed 9.23 27.15 w/ Embed 8.72 28.09

✓ 12.20 27.00 ✓ ✓ 8.11 28.02

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

to TrigFlow [17]. As shown in Fig. 3 (d), removing schedule transfer leads to training divergence due to incorrect signals. In contrast, incorporating our schedule transfer enables the model to achieve decent results within 5,000 iterations, demonstrating its crucial role in efficiently adapting flow matching models to TrigFlow-based consistency models.

1Step2Steps4Steps TeacherModel

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

Influence of CFG Embedding. To clarify the influence of Classifier-Free Guidance (CFG) embedding in our model, we maintain the setting of incorporating CFG into the teacher model, as established in previous works [13, 17, 45]. Specifically, during the training of the student model, we uniformly sample the CFG scale of the teacher model from the set 4.0,4.5,5.0. To integrate CFG embedding [11] into the student model, we add it as an additional condition to the time embedding, multiplying the CFG scale by 0.1 to align with our denser timestep embeddings. We conduct experiments with and without CFG embedding to evaluate its role. As shown in Tab. 4, incorporating CFG embedding significantly improves CLIP score by 0.94.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

20Steps

Figure 5 | Visual comparison among SANA-Sprint with different inference steps and the teacher model SANA. SANA-Sprint can generate high-quality images with one or two steps and the images can be better when increasing steps.

##### 5. Related Work

We put a relatively brief overview of related work here, with a more comprehensive version in the appendix. Diffusion models have two primary paradigms for step distillation: trajectory-based and distribution-based methods. Trajectory-based approaches include direct distillation[9] and progressive distillation[10, 11]. Consistency models [12] include variants like LCM [13], CTM [14], MCM [15], PCM [16], and sCM [17]. Distribution-based methods involve GAN-based distillation [3] and VSD variants [6, 7, 8, 46, 47]. Recent improvements include adversarial training with DINOv2 [48][4], stabilization of VSD[49], and improved algorithms like SID [50] and SIM [51]. In real-time image generation, techniques like PaGoDA[52] and Imagine-Flash accelerate diffusion inference. Model compression strategies include BitsFusion[53] and Weight Dilation[54]. Mobile applications use MobileDiffusion[55], SnapFusion[56], and SnapGen[57]. SVDQuant[58] combined with SANA[32] enables fast image generation on consumer GPUs.

Effects of sCM and LADD. We evaluate the effectiveness of each component by comparing models trained with only the sCM loss or the LADD loss. As shown in Tab. 3, training with LADD alone results in instability and suboptimal performance, achieving a higher FID score of 12.20 and a lower CLIP score of 27.00. In contrast, combining both sCM and LADD losses improves model performance, yielding a lower FID score of 8.11 and a higher CLIP score of 28.02, demonstrating their complementary benefits. Using sCM alone achieves a FID score of 8.93 and a CLIP score of 27.51, indicating that while sCM is effective, adding LADD further enhances performance. The weighting ablations for sCM and LADD loss are shown in Tab. 5, with additional timestep distribution ablations provided in Sec. F.2

Additional Max-Time Weighting. We validate the proposed max-time weighting strategy in LADD (see Sec. 3.3) through experiments with both sCM and LADD losses. As shown in Tab. 6, this weighting significantly improves performance. We test the strategy at 0%, 50%, and 70% maxtime (𝑡 = 𝜋/2) probabilities, finding that 50% is the best balance, while higher probabilities provide only marginal gains. However, considering the qualitative results, we finally choose 50% as the default max-time weighting.

##### 6. Conclusion

In this paper, we introduced SANA-Sprint, an efficient diffusion model for ultra-fast one-step text-to-image generation while preserving multi-step sampling flexibility. By employing a hybrid distillation strategy combining continuous-time consistency distillation (sCM) and latent adversarial distillation (LADD), SANA-Sprint achieves SoTA performance with 7.04 FID and 0.72 GenEval in

one step, eliminating step-specific training. This unified step-adaptive model enables high-quality 1024×1024 image generation in only 0.1s on H100, setting a new SoTA in speed-quality tradeoffs.

Looking ahead, SANA-Sprint’s instant feedback unlocks real-time interactive applications, transforming diffusion models into responsive creative tools and AIPC. We will open-source our code and models to encourage further exploration in efficient, practical generative AI systems.

Acknowledgements. We would like to express our heartfelt gratitude to Cheng Lu from OpenAI for his invaluable guidance and insightful discussions on the implementation of the sCM part. We are also deeply thankful to Yujun Lin, Zhekai Zhang, and Muyang Li from MIT for their significant contributions and engaging discussions on the quantization parts, as well as to Lvmin Zhang from Stanford for his expertise and thoughtful input on the ControlNet part. Their collaborative efforts and constructive discussions have been instrumental in shaping this work.

##### A. Pseudo Code for Training-Free Transformation from Flow to Trigflow

In this section, we provide a concise implementation of the transformation from a trained flow matching model to a TrigFlow model without requiring additional training. This transformation is based on the theoretical equivalence established in Sec. D. The core idea is to first convert the TrigFlow timestep 𝑡Trig to its corresponding flow matching timestep 𝑡FM. Then, the input feature 𝑥Trig is scaled accordingly to obtain 𝑥FM. The output of the flow matching model is then transformed using a linear combination to produce the final TrigFlow output. The following pseudo code implements this transformation efficiently, ensuring consistency between the two formulations.

class TrigFlowModel(FlowMatchingModel):

def forward(self, x_trig, t_trig, c): t_fm = torch.sin(t_trig) / (torch.cos(t_trig) + torch.sin(t_trig)) x_fm = x_trig * torch.sqrt(t_fm**2 + (1 - t_fm)**2)

fm_model_out = super().forward(x_fm, t_fm, c) trig_model_out = ((1 - 2 * t_fm) * x_fm + (1 - 2 * t_fm + 2 * t_fm**2) * fm_model_out) /

torch.sqrt(t_fm**2 + (1 - t_fm)**2) return trig_model_out

##### B. Transformation Algorithm

We present an algorithm for training-free transformation from a flow matching model to its TrigFlow counterpart. Given a noisy sample, its corresponding TrigFlow timestep, and a pre-trained flow matching model, the algorithm computes the equivalent flow matching timestep, rescales the input, and applies a deterministic transformation to obtain the TrigFlow output. The detailed procedure is outlined in Algorithm 1.

- Algorithm 1 Training-Free Transformation to TrigFlow

- 1: Input: Noisy data 𝑥𝑡,𝜎Trig

𝑑

, timestep 𝑡Trig, condition 𝑦, flow matching model 𝑣𝜃(𝑥𝑡,FM,𝑡FM,𝑦)

- 2: Compute 𝑡FM from 𝑡Trig via 𝑡FM = sin (𝑡Trig) sin (𝑡Trig)+cos (𝑡Trig)

- 3: Compute 𝑥𝑡,FM from 𝑥𝑡,Trig via 𝑥𝑡,FM = 𝑥𝑡,Trig

𝜎𝑑 · √︀𝑡2FM + (1 − 𝑡FM)2

- 4: Evaluate 𝑣𝜃(𝑥𝑡,FM,𝑡FM,𝑦)
- 5: Transform the model output via 𝐹̂︁𝜃 (︁

𝑥𝑡,Trig

𝜎𝑑 , 𝑡Trig,𝑦)︁ = √ 1

𝑡2FM+(1−𝑡FM)2

[︁(1 − 2𝑡FM)𝑥𝑡,FM + (1 − 2𝑡FM + 2𝑡2FM)𝑣𝜃(𝑥𝑡,FM, 𝑡FM,𝑦)]︁

- 6: Output: Transformed result

##### C. Training Algorithm of SANA-Sprint

In this section, we present the detailed training algorithm for SANA-Sprint. To emphasize the differences from the standard sCM training algorithm, we highlight the modified steps in light blue. The following algorithm outlines the complete training procedure, including the key transformations and parameter updates specific to SANA-Sprint.

##### D. Proof of Proposition 3.1

Before presenting the formal proof, we first provide some context to understand the necessity of the transformation. In score-based generative models such as diffusion, flow matching, and TrigFlow, denoising is typically performed under certain conditions of data scales and signal-to-noise ratios (SNRs) that match the training setup. However, directly applying flow matching to denoise data generated by TrigFlow is not feasible due to mismatches in time parameterization, SNR, and output necessitating explicit transformations to align the input and output between the two models. The following proof provides the explicit transformation required to connect the TrigFlow-scheduled data to the flow-matching framework.

- Algorithm 2 Training Algorithm of SANA-Sprint

- 1: Input: dataset 𝒟 with std. 𝜎𝑑 = 0.5, pretrained flow model 𝐹pretrain with parameter 𝜃pretrain, student model 𝐹𝜃, discriminator head 𝐷𝜓, weighting 𝑤𝜑, learning rate 𝜂, generator distribution (𝑃mean, G,𝑃std, G), discriminator distribution (𝑃mean, D,𝑃std, D), constant 𝑐, warmup iteration 𝐻, max-time weighting 𝑝, condition 𝑦.
- 2: Init: transform 𝐹pretrain and 𝐹𝜃 to TrigFlow model using Algorithm 1, init student model and discriminator backbone with 𝜃pretrain, Iters ← 0.
- 3: repeat
- 4: update discriminator 𝜓:
- 5: 𝑥0 ∼ 𝒟, 𝑧 ∼ 𝒩(0,𝜎𝑑2𝐼), 𝜏 ∼ 𝒩(𝑃mean, G,𝑃std,2 G), 𝑡 ← arctan(𝑒

𝜏

𝜎𝑑)

- 6: if 𝑝 > 0, 𝜉 ∼ U[0,1], 𝑡 ← 𝜋2 if 𝜉 < 𝑝

- 7: 𝑥𝑡 ← cos(𝑡)𝑥0 + sin(𝑡)𝑧, 𝑥^𝑓

𝜃−

0 ← 𝑓𝜃−(𝑥𝑡,𝑡,𝑦)

- 8: 𝜏 ∼ 𝒩(𝑃mean, D,𝑃std,2 D), 𝑠 ← arctan(𝑒

𝜏

𝜎𝑑 )

- 9: 𝑥𝑠 ← cos(𝑠)𝑥0 + sin(𝑠)𝑧, 𝑥^𝑓

𝜃−

𝑠 ← cos(𝑠)^𝑥𝑓

𝜃−

0 + sin(𝑠)𝑧

- 10: ℒ𝐷adv(𝜓) ← E𝑥0,𝑠[︁∑︀

𝑘 ReLU(︁1 − 𝐷𝜓,𝑘(𝐹𝜃pre,𝑘(𝑥𝑠, 𝑠,𝑦)))︁]︁ + E𝑥0,𝑠,𝑡[︁∑︀

𝑘 ReLU(︁1 + 𝐷𝜓,𝑘(𝐹𝜃pre,𝑘(^𝑥𝑓𝑠𝜃−, 𝑠,𝑦)))︁]︁

- 11: 𝜓 ← 𝜓 − 𝜂∇𝜓ℒ𝐷adv(𝜓) ◁ Discriminator step
- 12: Iters ← Iters + 1
- 13: update student model 𝜃 and weighting 𝜑:
- 14: 𝑥0 ∼ 𝒟, 𝑧 ∼ 𝒩(0,𝜎𝑑2𝐼), 𝜏 ∼ 𝒩(𝑃mean, G,𝑃std,2 G), 𝑡 ← arctan(𝑒

𝜏

𝜎𝑑)

- 15: 𝑥𝑡 ← cos(𝑡)𝑥0 + sin(𝑡)𝑧
- 16: dd𝑥𝑡𝑡 ← 𝜎𝑑𝐹pretrain, cfg(𝑥𝑡

𝜎𝑑,𝑡)

- 17: 𝑟 ← min(1,Iters/𝐻) ◁ Tangent warmup
- 18: 𝑔 ← −cos2(𝑡)(𝜎𝑑𝐹𝜃− − d𝑥

𝑡

d𝑡 ) − 𝑟 · cos(𝑡)sin(𝑡)(𝑥𝑡 + 𝜎𝑑 d𝐹

𝜃−

d𝑡 ) ◁ JVP rearrangement

- 19: 𝑔 ← 𝑔/(‖𝑔‖ + 𝑐) ◁ Tangent normalization
- 20: ℒ(𝜃, 𝜑) ← 𝑒

𝑤𝜑(𝑡)

𝐷 ‖𝐹𝜃(𝑥𝑡

𝜎𝑑,𝑡) − 𝐹𝜃−(𝑥𝑡

𝜎𝑑,𝑡) − 𝑔‖22 − 𝑤𝜑(𝑡) ◁ sCM loss

- 21: if 𝑝 > 0, 𝜉 ∼ U[0,1], 𝑡 ← 𝜋2 if 𝜉 < 𝑝

- 22: 𝑥𝑡 ← cos(𝑡)𝑥0 + sin(𝑡)𝑧, 𝑥^𝑓

𝜃

0 ← 𝑓𝜃(𝑥𝑡,𝑡,𝑦)

- 23: 𝑥^𝑓

𝜃

𝑠 ← cos(𝑠)^𝑥𝑓

𝜃

0 + sin(𝑠)𝑧

- 24: ℒ(𝜃, 𝜑) ← ℒ(𝜃, 𝜑) − E𝑥

0,𝑠,𝑡[︁∑︀

𝑘 𝐷𝜓,𝑘(𝐹𝜃

pre,𝑘(^𝑥𝑓

𝜃

𝑠 ,𝑠,𝑦))]︁ ◁ GAN loss

- 25: (𝜃, 𝜑) ← (𝜃, 𝜑) − 𝜂∇𝜃,𝜑ℒ(𝜃, 𝜑) ◁ Generator step
- 26: Iters ← Iters + 1
- 27: until convergence

Proof. Under the TrigFlow framework, the noisy input sample is given by

𝑥𝑡,Trig 𝜎𝑑

𝑥0 𝜎𝑑

= cos(𝑡Trig)

𝑧 𝜎𝑑

+ sin(𝑡Trig)

. (13)

Since both 𝑥0 and 𝑧 originally have a standard deviation of 𝜎𝑑, we absorb 𝜎𝑑 into these variables so that they are normalized to have a standard deviation of 1. This normalization aligns with the conventions used in flow matching models. Note that the signal-to-noise ratios (SNRs) for flow matching models and TrigFlow models are given by

1 − 𝑡FM 𝑡FM

SNR(𝑡FM) = (

cos(𝑡Trig) sin(𝑡Trig)

)2, SNR(𝑡Trig) = (

1 tan(𝑡Trig)

)2 = (

)2. (14)

To ensure an equivalent SNR under the flow matching framework, we seek the corresponding time 𝑡FM that satisfies: (

1 − 𝑡FM 𝑡FM

1 tan(𝑡Trig)

)2 = (

)2. (15)

Solving this equation, we obtain the relationship between 𝑡FM and 𝑡Trig:

sin (𝑡Trig) sin (𝑡Trig) + cos(𝑡Trig)

𝑡𝐹𝑀 1 − 𝑡𝐹𝑀

, 𝑡𝑇𝑟𝑖𝑔 = arctan (

𝑡FM =

). (16)

Under this transformation, the SNRs of 𝑥𝑡,FM and 𝑥𝑡,Trig remain equal; however, their scales differ due to the following formulations:

𝑥𝑡,FM = (1 − 𝑡FM)𝑥0 + 𝑡FM𝑧, 𝑥𝑡,Trig = cos(𝑡Trig)𝑥0 + sin(𝑡Trig)𝑧, (17)

Since (1 − 𝑡FM)2 + 𝑡2FM is generally not equal to cos2(𝑡Trig) + sin2(𝑡Trig) = 1 (except when 𝑡FM = 0 or 1), a scale adjustment is needed. To align their scales, we introduce a scale factor function 𝜆(𝑡FM) that satisfies

𝜆(𝑡FM) · cos(𝑡Trig) = (1 − 𝑡FM),and 𝜆(𝑡FM) · sin(𝑡Trig) = 𝑡FM, (18) Therefore, the scale factor is determined as follows

= √︁𝑡2FM + (1 − 𝑡FM)2. (19)

1 − 𝑡FM cos(arctan ( 𝑡FM 1−𝑡FM))

𝑡FM sin(arctan ( 𝑡FM 1−𝑡FM))

𝜆(𝑡FM) =

=

The transformed 𝑥𝑡,Trig follows the same distribution as the flow matching model’s training distribution, achieving our desired objective. Next, we aim to determine the optimal estimator for the TrigFlow model 𝐹𝜃, given 𝑣𝜃(𝑥𝑡,FM,𝑡FM,𝑦). We first consider an ideal scenario where the model’s capacity is sufficiently large. In this case, the flow matching model reaches its optimal solution:

𝑣*(𝑥𝑡,FM,𝑡FM,𝑦) = E[𝑧 − 𝑥0|𝑥𝑡

,𝑦], (20)

FM

as the conditional expectation minimizes the mean squared error (MSE) loss. Similarly, the optimal solution of the TrigFlow model is given by

,𝑦]. (21) Noting that

𝐹*(𝑥𝑡,Trig,𝑡Trig,𝑦) = E[cos(𝑡Trig)𝑧 − sin (𝑡Trig)𝑥0|𝑥𝑡

Trig

1 − 𝑡FM √︀𝑡2FM + (1 − 𝑡FM)2

𝑡FM √︀𝑡2FM + (1 − 𝑡FM)2

cos(𝑡Trig) =

, sin (𝑡Trig) =

, (22)

we leverage the linearity of conditional expectation to derive 1 − 2𝑡FM √︀𝑡2FM + (1 − 𝑡FM)2 · 𝑥𝑡

1 − 2𝑡FM + 2𝑡2FM √︀𝑡2FM + (1 − 𝑡FM)2

+

E[𝑧 − 𝑥0|𝑥𝑡

###### ,𝑦]

FM

FM

1 − 2𝑡FM + 2𝑡2FM √︀𝑡2FM + (1 − 𝑡FM)2

1 − 2𝑡FM √︀𝑡2FM + (1 − 𝑡FM)2

E[(1 − 𝑡FM) · 𝑥0 + 𝑡FM · 𝑧|𝑥𝑡

,𝑦] +

E[𝑧 − 𝑥0|𝑥𝑡

###### ,𝑦]

=

FM

FM

(23)

1 − 𝑡FM √︀𝑡2FM + (1 − 𝑡FM)2

𝑡FM √︀𝑡2FM + (1 − 𝑡FM)2

=E[

###### ,𝑦]

𝑧 −

𝑥0|𝑥𝑡

FM

=E[cos(𝑡Trig)𝑧 − sin (𝑡Trig)𝑥0|𝑥𝑡

,𝑦].

Trig

Consequently, we obtain

√︀𝑡2FM + (1 − 𝑡FM)2[︁(1 − 2𝑡FM)𝑥𝑡,FM + (1 − 2𝑡FM + 2𝑡2FM)𝑣*(𝑥𝑡,FM,𝑡FM,𝑦)]︁. (24) Next, we consider a more realistic scenario where the model’s capacity is limited, leading to the learned velocity field

1

𝐹*(𝑥𝑡,Trig,𝑡Trig,𝑦) =

0,𝑧,𝑡[𝑤(𝑡)‖𝑣𝜃(𝑥𝑡,FM,𝑡FM,𝑦) − (𝑧 − 𝑥0)‖2]. (25) Under our parameterization, training the TrigFlow model amounts to minimizing

𝑣𝜃*(𝑥𝑡,FM,𝑡FM,𝑦) = min

E𝑥

𝜃

2]︃

[︃

√︀1 − 2𝑡FM + 2𝑡2FM

√︀ 1 − 2𝑡FM

𝑣𝜃(𝑥𝑡,FM, 𝑡FM,𝑦) − (cos(𝑡Trig)𝑧 − sin(𝑡Trig)𝑥0)⃦

min

· 𝑥𝑡FM +

E𝑥0,𝑧,𝑡

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

⃦

𝜃

[︃

(︃

)︃

2]︃

√︀1 − 2𝑡FM + 2𝑡2FM

√︀ 1 − 2𝑡FM

√︀ 1 − 𝑡FM

√︀ 𝑡FM

= min

· 𝑥𝑡FM +

𝑣𝜃(𝑥𝑡,FM, 𝑡FM, 𝑦) −

E𝑥0,𝑧,𝑡

𝑧 −

𝑥0

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

⃦

⃦

𝜃

(26)

Substituting 𝑥𝑡,FM = (1 − 𝑡FM)𝑥0 + 𝑡FM𝑧, the above expression simplifies to

2]︃

[︃

(︃

)︃

√︀1 − 2𝑡FM + 2𝑡2FM

√︀ 1 − 2𝑡FM

√︀ 1 − 𝑡FM

√︀ 𝑡FM

· 𝑥𝑡FM +

min

𝑣𝜃(𝑥𝑡,FM, 𝑡FM, 𝑦) −

E𝑥0,𝑧,𝑡

𝑧 −

𝑥0

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

⃦

⃦

𝜃

[︃

2]︃

(︃

)︃

√︀𝑡2FM + (1 − 𝑡FM)2

√︀𝑡2FM + (1 − 𝑡FM)2

√︀𝑡2FM + (1 − 𝑡FM)2

= min

𝑣𝜃(𝑥𝑡,FM, 𝑡FM, 𝑦) −

E𝑥0,𝑧,𝑡

𝑧 −

𝑥0

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

𝑡2FM + (1 − 𝑡FM)2

⃦

⃦

𝜃

[︀(︀

)︀

]︀

‖𝑣𝜃(𝑥𝑡,FM, 𝑡FM, 𝑦) − (𝑧 − 𝑥0)‖2

𝑡2FM + (1 − 𝑡FM)2

= min

E𝑥0,𝑧,𝑡

𝜃

(27)

Thus, training the TrigFlow model with our parameterization is equivalent to training the flow matching model, apart from differences in the loss weighting function 𝑤(𝑡) and the timestep sampling distribution 𝑝(𝑡).

| |
|---|

##### E. Full Related Work

Text to Image Generation Text-to-image generation has experienced transformative advancements in both efficiency and model design. The field gained early traction with Stable Diffusion [59], which set the stage for scalable highresolution synthesis. A pivotal shift occurred with Diffusion Transformers (DiT)[60], which replaced conventional U-Net architectures with transformer-based designs, unlocking improved scalability and computational efficiency. Building on this innovation, PixArt-𝛼[61] demonstrated competitive image quality while slashing training costs to only 10.8% of those required by Stable Diffusion v1.5 [59]. Recent breakthroughs have further pushed the boundaries of compositional generation. Large-scale models like FLUX [21] and Stable Diffusion 3 [38] have scaled up to ultra-highresolution synthesis and introduced multi-modal capabilities through frameworks such as the Multi-modal Diffusion Transformer (MM-DiT)[22]. Playground v3[39] achieved state-of-the-art image quality by seamlessly integrating diffusion models with Large Language Models (LLMs)[62], while PixArt-Σ[37] showcased direct 4K image generation using a compact 0.6B parameter model, emphasizing computational efficiency alongside high-quality outputs. Efficiencydriven innovations have also gained momentum. SANA [32] introduced high-resolution synthesis capabilities through deep compression autoencoding [63] and linear attention mechanisms, enabling deployment on consumer-grade hardware like laptop GPUs. Additionally, advancements in linear attention mechanisms for class-conditional generation [64, 65], diffusion models without attention [66, 67], and cascade structures [68, 69, 70] have further optimized computational requirements while maintaining performance. These developments collectively underscore the field’s rapid evolution toward more accessible, efficient, and versatile text-to-image generation technologies.

Diffusion Model Step Distillations Current methodologies primarily coalesce into two dominant paradigms: (1) trajectory-based distillation. Direct Distillation [9] directly learns noise-image mapping given by PF-ODE. Progressive Distillation [10, 11] makes the learning progress easier by progressively enlarging subintervals on the ODE trajectory. Consistency Models (CMs) [12] (e.g. LCM [13], CTM [14], MCM [15], PCM [16], sCM [17]) predict the solution 𝑥0 of the PF-ODE given 𝑥𝑡 via self-consistency. (2) distribution-based distillation. It can be further divided into GAN [3]-based distillation and its variational score distillation (VSD) variants [6, 7, 8, 46, 47]. ADD [4] explored distilling diffusion models using adversarial training with pretrained feature extractor like DINOv2 [48] in pixel space. LADD [5] further utilize teacher diffusion models as feature extractors enabling direct discrimination in latent space, drastically saving the computation and GPU memories. [49] stabilize VSD with regression loss. SID [50] and SIM [51] propose improved algorithms for VSD.

Real-Time Image Generation Recent advancements in real-time image generation have focused on improving the efficiency and quality of diffusion models. PaGoDA [52] introduces a progressive approach for one-step generation across resolutions. Imagine-Flash also uses a backward distillation to accelerate diffusion inference. In model compression, BitsFusion [53] quantizes Stable Diffusion’s UNet to 1.99 bits, and Weight Dilation [54] presents DilateQuant for enhanced performance. For mobile applications, MobileDiffusion [55] achieves sub-second generation times, with SnapFusion [56] and SnapGen [57] enabling 1024x1024 pixel image generation in about 1.4 seconds. SVDQuant [58] introduces 4-bit quantization for diffusion models, and when combined with SANA [32], enables fast generation of high-quality images on consumer GPUs, bridging the gap between model performance and real-time applications.

##### F. More Details

###### F.1. Experimental Setup

Model Architecture Following the pruning technology in SANA-1.5 [42], our teacher models are fine-tuned from SANA 0.6B and 1.6B, respectively. The architecture, training data, and other hyperparameters remain consistent with SANA-1.5 [42].

Training Details We conduct distributed training using PyTorch’s Distributed Data Parallel (DDP) across 32 NVIDIA A100 GPUs on 4 DGX nodes. Our two-phase strategy involves fine-tuning the teacher model with dense time embedding and QK normalization at a learning rate of 2e-5 for 5,000 iterations (global batch size of 1,024), as discussed in Sec. 3.1. Then, we perform timestep distillation through the proposed framework at a learning rate of 2e-6 with a global batch size of 512 for 20,000 iterations. As Flash Attention JVP kernel support is not available in PyTorch [17], we retain Linear Attention [32] to auto-compute the JVP.

Evaluation Protocol We use multiple metrics: FID, CLIP Score, and GenEval [44], comparing with state-of-the-art methods. FID and CLIP Score are evaluated on the MJHQ-30K dataset [43]. GenEval measures text-image alignment with 553 test prompts, emphasizing its ability to reflect alignment and show improvement potential. We also provide visualizations to compare state-of-the-art methods and highlight our performance.

###### F.2. More Ablations

Inference Timestep Search Fig. 6 illustrates the process of timestep optimization for inference across 1, 2, and 4 steps, comparing the performance of 0.6B and 1.6B models in terms of FID (top row) and CLIP-Scores (bottom row). The optimization follows a sequential search strategy: first, we determine the optimal 𝑡max for 1-step inference using arctan(𝑛/0.5), inspired by EDM [28], where 𝑛 is searched for the maximum timestep. Using this 𝑡max, we then search for the intermediate timestep 𝑡2𝑛𝑑 in 2-step inference. For 4-step inference, the timesteps for the first two steps are fixed to their previously optimized values, while the third (𝑡3𝑟𝑑) and fourth timesteps (𝑡4𝑡ℎ) are searched sequentially. In each case, the x-axis represents the timestep being optimized at the current step, ensuring that earlier steps use their best-found values to maximize overall performance. This hierarchical approach enables efficient timestep selection for multi-step inference settings.

Controlling the sCM Noise Distribution In the sCM-only experiments, we investigate the impact of different noise distribution parameter settings on model performance. The noise distribution is defined as 𝑡 = arctan (︁

)︁, where

𝑒𝜏 𝜎𝑑

𝜏 ∼ 𝒩(𝑃mean,𝑃std2 ). Starting from the initial parameters (−0.8,1.6) proposed in sCM [17], we experiment with various mean and standard deviation configurations to evaluate their effects. By tracking FID and CLIP-Score trends over

40k training iterations, we identify (𝑃mean,𝑃std) = (0.0,1.6), represented by the green curve in Fig. 7, as the optimal setting. This configuration consistently reduces FID while improving CLIP-Score, resulting in superior generation quality and text-image alignment. We also observe that extreme mean values, such as 𝑃mean = 0.6 or 𝑃mean = −0.8, lead to significant training instability and even failure in some cases. Consequently, we adopt (0.0,1.6) as the default parameter setting.

Controlling the LADD Discriminator Noise Distribution Generative features change with the noise level, offering structured feedback at high noise and texture-related feedback at low noise [5]. We compare the results for different mean and standard deviation settings in 𝑡 = arctan (︁

)︁, where 𝜏 ∼ 𝒩(𝑃mean,𝑃std2 ) for LADD discriminator. Building on the optimal mean and standard deviation settings (0.0,1.6) identified for sCM, we further explore the best noise configuration for the LADD’s discriminator. In Fig. 8 and Tab. 8, we visualize the probability distributions of 𝑡 sampled under different mean and standard deviation settings, as well as the corresponding FID and CLIP-Score results when applied in the LADD loss. Based on these analyses, we identify (−0.6,1.0) as the optimal setting, which achieves a more balanced feature distribution across high and low noise levels while maintaining stable training dynamics. Consequently, we adopt (−0.6,1.0) as the default configuration for LADD loss.

𝑒𝜏 𝜎𝑑

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-Sprint : One-Step Diffusion with Continuous-Time Consistency Distillation

0.6B 1.6B

0.6B 1.6B

0.6B 1.6B

7.80

10.5

7.8

6.8

6.72 6.73

6.72

6.70

7.38

7.67

6.66

9.15

6.65

7.11

7.35

10.24

8.5

6.9

6.6

6.96

7.05

6.59

8.73

6.54

6.54 6.49

7.48 7.49 7.58 7.61

6.52

6.51 6.51

[Figure 75]

6.55

6.09

9.47

7.59

7.04 7.15 7.29

6.5

6.0

6.5

1.546 1.551 1.554 1.557 1.568 π/2

1.0 1.1 1.2 1.3 1.4

0.4 0.5 0.6 0.7 0.8 0.9

FID comparison (1 step)

FID comparison (2 steps) FID comparison (4 steps)

26.2

0.6B 1.6B

0.6B 1.6B

0.6B 1.6B

28.5

28.40

28.40

28.4

28.1

28.08

28.08 28.08

28.46

28.06

28.45 28.45

28.45

28.44

28.04

28.37

28.04

22.1

28.43

28.02

28.4

28.3

28.0

28.33

28.35

28.32

28.00

28.34

28.38 28.38

28.38

28.38

27.94 27.91 27.95

27.99

28.37 28.37

18

28.32

28.32

5k 10k 17k 35k

28.30

28.4

28.3

27.9

0.4 0.5 0.6 0.7 0.8 0.9

1.0 1.1 1.2 1.3 1.4

1.546 1.551 1.554 1.557 1.568 π/2

CLIP score comparison (1 step) CLIP score comparison (2 steps) CLIP score comparison (4 steps)

- Figure 6 | Inference timesteps search. This figure illustrates the performance of timesteps search for achieving optimal results during inference with 0.6B and 1.6B models. The subplots compare FID (top row) and CLIP-Score (bottom row) across different timesteps for 1-step, 2-step, and 4-step inference settings. The x-axis represents the timestep being searched at the current step; for multi-step settings (e.g., 4 steps), the timesteps for earlier steps are fixed to their previously optimized values.

# Done

Table 7 | Inference timestep settings for both SANA-Sprint 0.6B and 1.6B models.

Inference Timestep search on 1, 2, 4 steps

1 step 2 steps 4 steps

| | | | |
|---|---|---|---|
|Timestep T|[𝜋/2,0.0]|[arctan(200/0.5),1.3,0.0]|[arctan(200/0.5),1.3,1.1,0.6,0.0]|

###### F.3. More Qualitative Results

SANA-Sprint-ControlNet Visualization Images In Fig. 9, we demonstrate the visualization capabilities of our SANA-Sprint-ControlNet, which efficiently achieves impressive results in only 0.4 seconds using a 2-step generation process, producing high-quality images at a resolution of 1024 × 1024 pixels. The visualization process begins with an input image, which is processed using a HED detection model to extract the scribe graph. This scribe graph, combined with a given prompt, is used to generate the corresponding image in the second column. The third column presents a blended image that combines the generated image with the scribe graph, highlighting the precise control of the model through boundary alignment. This visualization showcases the model’s ability to accurately interpret prompts and maintain robust control over generated images.

More Visualization Images In Fig. 11, we present images generated by our model using various prompts. SANASprint showcases comprehensive generation capabilities, including high-fidelity detail rendering, accurate semantic understanding, and reliable text generation, all achieved with only 2-step sampling. In particular, the model efficiently produces high-quality images of 1024 × 1024 pixels in only 0.24 seconds on an NVIDIA A100 GPU. The samples demonstrate versatility in various scenarios, from in tricate textures and complex compositions to accurate text rendering, highlighting the robust image quality of the model in both artistic and practical tasks.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

(-0.8, 1.6) (-0.4, 1.6) (0.0, 1.6) (0.2, 1.6) (0.2, 2.0) (0.4, 1.6)

(-0.8, 1.6) (-0.4, 1.6) (0.0, 1.6) (0.2, 1.6) (0.2, 2.0) (0.4, 1.6)

15.0

25.4

CLIP-Score

###### FID

12.0

25.2

9.0

25.0

8000 12000 16000 20000 24000 28000 32000 36000 40000

8000 12000 16000 20000 24000 28000 32000 36000 40000

(a) FID Comparison over Training Steps

(b) CLIP-Score Comparison over Training Steps

## Done

- Figure 7 | Controlling the sCM noise distribution. This figure compares FID and CLIP-Score across different noise

distribution settings over 40k training steps in sCM-only experiments. The green curve (𝑃mean,𝑃std) = (0.0,1.6) demonstrates optimal performance, achieving stable training dynamics and superior generation quality.

[Figure 76]

- Figure 8 | Controlling the LADD noise distribution. We vary the parameters of a logit-normal distribution for biasing the sampling of the LADD teacher noise level. When biasing towards very high noise levels (m = 0.4, s = 2), we observe unstable training.

[Figure 77]

sCM Mean Std on FID CLIP

Table 8 | Comparison of different noise distributions for LADD loss.

Mean, Std FID ↓ CLIP ↑

- (-0.6, 1.0) 9.48 28.08
- (-0.6, 2.0) 10.36 28.03

- (0.0, 1.0) 13.11 27.18
- (0.0, 2.0) 11.25 27.96 (0.4, 2.0) 9.77 28.00 (0.6, 1.0) 12.85 27.32

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-Sprint : One-Step Diffusion with Continuous-Time Consistency Distillation

prompt: portrait photo of a girl, photograph

prompt: portrait photo of a boy, photograph

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

prompt: A blue melting apple

prompt: Mystical Apple Made of Rubik’s Cubes

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

HED Signal Output Blender

HED Signal Output Blender

SANA-Sprint-ControlNet 2-step Output

SANA-Sprint-ControlNet 2-step Output

Input Image

- Figure 9 | Visualization of SANA-Sprint-ControlNet’s capabilities. The model outputs high-quality images of 1024 × 1024 pixels in only 2 steps and 0.3 seconds on an NVIDIA H100 GPU. The process involves processing the input image (first column) to extract a scribe graph, which, along with a prompt, generates an image (second column). The blended image (third column) highlights precise boundary alignment and control, demonstrating the model’s robust control capabilities.

Done

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

[Figure 90]

|[Figure 91]|
|---|

- Figure 10 | ControlNet Demo: Hand-Crafted Scribble to Stunning Image. Left: A hand-crafted scribble created with a brush. Right: The result generated by the Sana-Sprint-ControlNet model, strictly following the scribble and prompt. Inference Latency: The model achieves remarkable speed, generating the 1024 × 1024 images in only 1 step and 0.25 seconds on H100 GPU, as shown in the right red box. This demo showcases the model’s exceptional control and efficiency, adhering closely to the user’s input while producing visually appealing results.

Working

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

###### Figure 11 | Generated images with SANA-Sprint. The model outputs high-quality images of 1024 × 1024 pixels in 2 steps and 0.24 seconds on an NVIDIA A100 GPU, showcasing comprehensive generation capabilities with high-fidelity details and accurate text rendering, handling diverse scenarios with robust image quality.

##### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [2] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [3] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [4] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer, 2024.
- [5] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.
- [6] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [7] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36:8406–8441, 2023.
- [8] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525– 76546, 2023.
- [9] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.
- [10] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [11] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023.
- [12] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [13] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [14] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.
- [15] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. arXiv preprint arXiv:2403.06807, 2024.
- [16] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. arXiv preprint arXiv:2405.18407, 2024.
- [17] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.
- [18] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.
- [19] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. In European Conference on Computer Vision, pages 428–447. Springer, 2024.
- [20] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024.
- [21] Black Forest Labs. Flux, 2024.

- [22] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [23] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.
- [24] Stefano Peluchetti. Non-denoising forward-time diffusions, 2022.
- [25] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [27] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.
- [28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.
- [29] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [31] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024.
- [32] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.
- [33] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.
- [34] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. arXiv preprint arXiv:2206.02262, 2022.
- [35] Jae Hyun Lim and Jong Chul Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017.
- [36] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [37] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-𝜎: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.
- [38] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [39] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.
- [40] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-{∖delta}: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024.
- [41] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [42] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.

- [43] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.
- [44] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [45] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [46] Sirui Xie, Zhisheng Xiao, Diederik P Kingma, Tingbo Hou, Ying Nian Wu, Kevin Patrick Murphy, Tim Salimans, Ben Poole, and Ruiqi Gao. Em distillation for one-step diffusion models. arXiv preprint arXiv:2405.16852, 2024.
- [47] Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. Advances in Neural Information Processing Systems, 37:36046–36070, 2025.
- [48] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [49] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [50] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.
- [51] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. Advances in Neural Information Processing Systems, 37:115377–115408, 2025.
- [52] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Pagoda: Progressive growing of a one-step generator from a low-resolution diffusion teacher. Advances in Neural Information Processing Systems, 37:19167–19208, 2025.
- [53] Yang Sui, Yanyu Li, Anil Kag, Yerlan Idelbayev, Junli Cao, Ju Hu, Dhritiman Sagar, Bo Yuan, Sergey Tulyakov, and Jian Ren. Bitsfusion: 1.99 bits weight quantization of diffusion model. arXiv preprint arXiv:2406.04333, 2024.
- [54] Xuewen Liu, Zhikai Li, and Qingyi Gu. Dilatequant: Accurate and efficient diffusion quantization via weight dilation. arXiv preprint arXiv:2409.14307, 2024.
- [55] Yang Zhao, Yanwu Xu, Zhisheng Xiao, Haolin Jia, and Tingbo Hou. Mobilediffusion: Instant text-to-image generation on mobile devices. In European Conference on Computer Vision, pages 225–242. Springer, 2024.
- [56] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36:20662–20678, 2023.
- [57] Dongting Hu, Jierun Chen, Xijie Huang, Huseyin Coskun, Arpit Sahni, Aarush Gupta, Anujraaj Goyal, Dishani Lahiri, Rajesh Singh, Yerlan Idelbayev, et al. Snapgen: Taming high-resolution text-to-image models for mobile devices with efficient architectures and training. arXiv preprint arXiv:2412.09619, 2024.
- [58] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024.
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [60] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.
- [61] Junsong Chen, YU Jincheng, GE Chongjian, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-𝛼: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations, 2024.
- [62] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [63] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.
- [64] Han Cai, Muyang Li, Qinsheng Zhang, Ming-Yu Liu, and Song Han. Condition-aware neural network for controlled image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7194–7203, 2024.
- [65] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. arXiv preprint arXiv:2405.18428, 2024.
- [66] Jing Nathan Yan, Jiatao Gu, and Alexander M Rush. Diffusion models without attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8239–8249, 2024.
- [67] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024.
- [68] Pablo Pernias, Dominic Rampas, Mats L. Richter, Christopher J. Pal, and Marc Aubreville. Wuerstchen: An efficient architecture for large-scale text-to-image diffusion models, 2023.
- [69] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024.
- [70] Yuchuan Tian, Zhijun Tu, Hanting Chen, Jie Hu, Chao Xu, and Yunhe Wang. U-dits: Downsample tokens in u-shaped diffusion transformers. arXiv preprint arXiv:2405.02730, 2024.

