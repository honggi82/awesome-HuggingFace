## RectifiedHR: Enable Efficient High-Resolution Synthesis via Energy Rectification

# arXiv:2503.02537v4[cs.CV]9Apr2026

Zhen Yang1*‡ Guibao Shen1* Minyang Li1* Liang Hou2 Mushui Liu4 Luozhou Wang1 Xin Tao2 Ying-Cong Chen1,3† 1HKUST(GZ) 2Kuaishou Technology 3HKUST 4Zhejiang University

zheny.cs@gmail.com, yingcongchen@ust.hk

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

2048x2048 2048x2048

3072x3072

4096x2048 4096x2048 8192x8192

[Figure 7]

4096x4096

[Figure 8]

3072x3072

[Figure 9]

2048x4096

Figure 1. Generated images by RectifiedHR. The training-free RectifiedHR enables diffusion models to synthesize images at resolutions exceeding their original training resolution. Please zoom in for a closer view.

training-free and demonstrates efficient performance. Furthermore, we show that RectifiedHR is compatible with various diffusion model techniques, enabling advanced features such as image editing, customized generation, and video synthesis. Extensive comparisons with numerous baseline methods validate the superior effectiveness and efficiency of RectifiedHR. The code can be found here.

### Abstract

Diffusion models have achieved remarkable progress across various visual generation tasks. However, their performance significantly declines when generating content at resolutions higher than those used during training. Although numerous methods have been proposed to enable high-resolution generation, they all suffer from inefficiency. In this paper, we propose RectifiedHR, a straightforward and efficient solution for training-free high-resolution synthesis. Specifically, we propose a noise refresh strategy that unlocks the model’s training-free high-resolution synthesis capability and improves efficiency. Additionally, we are the first to observe the phenomenon of energy decay, which cause image blurriness during the high-resolution synthesis process. To address this issue, we introduce average latent energy analysis and find that tuning the classifier-free guidance hyperparameter can improve generation performance. Our method is entirely

### 1. Introduction

Recent advances in diffusion models [7, 12, 29, 32, 36, 39, 43, 46, 67] have significantly improved generation quality, enabling realistic editing [1, 4, 9, 27, 41, 42, 57, 61] and customized generation [2, 10, 13, 31, 48, 56]. However, these models struggle to generate images at resolutions beyond those seen during training, resulting in noticeable performance degradation. Training directly on high-resolution content is expensive, underscoring the need for methods that enhance resolution without requiring additional training.

* Equal contribution. † Corresponding author. ‡ This work was conducted during the author’s internship at Kling.

Currently, the naive approach is to directly input highresolution noise. However, this method leads to severe

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

##### 0 49 Timestep

- Figure 2. The visualization images corresponding to “predicted x0” at different time step t, abbreviated as ptx0. The figure visualizes the process of how ptx0 changes with the sampling steps, where the x-axis represents the timestep in the sampling process. The 11 images are

evenly extracted from 50 steps. Early steps primarily establish global structure, while later steps refine local details; toward the end, ptx0 exhibits RGB-like characteristics.

repeated pattern issues. To address this problem, many training-free high-resolution generation methods have been proposed, such as [2, 5, 11, 14–16, 22, 23, 25, 28, 30, 33, 34, 38, 52, 59, 65, 66]. However, these methods all share a common problem: they inevitably introduce additional computational overhead. For example, the sliding window operations introduced by [2, 11, 23, 30, 33, 34] have overlapping regions that result in redundant computations. Similarly, [34, 38, 52] require setting different prompts for small local regions of each image and need to incorporate a vision-language model. Additionally, [5, 28, 66] require multiple rounds of SDEdit [40] or complex classifier-free guidance (CFG) to gradually increase the resolution from a low-resolution image to a high-resolution image, thereby introducing more sampling steps or complex CFG calculations. All of these methods introduce additional computational overhead and complexity, significantly reducing the speed of high-resolution synthesis.

In general, our main contributions are as follows: (1) We propose RectifiedHR, an efficient, training-free framework for high-resolution synthesis that eliminates redundant computation and enables resolution scalability without requiring additional sampling steps. (2) We introduce noise refresh and energy rectification, pioneering the use of average latent energy analysis to address energy decay—an issue previously overlooked in high-resolution synthesis. (3) Our method surpasses existing baselines in both efficiency and quality, achieving faster inference while preserving superior fidelity. (4) We demonstrate that RectifiedHR can be seamlessly integrated with ControlNet, supporting a range of applications such as image editing, customized image generation, and video synthesis.

### 2. Related Work

#### 2.1. Text-guided image generation

With the scaling of models, data volume, and computational resources, text-guided image generation has witnessed unprecedented advancements, leading to the emergence of numerous diffusion models such as FLUX [29], LDM [46], SDXL [43], PixArt [7, 8], HunyuanDiT [32], SD3 [12], LCM [39], LuminaNext [67], and UltraPixel [45]. These models learn mappings from Gaussian noise to high-quality images through diverse training and sampling strategies, including DDPM [20], SGM [54], EDM [26], DDIM [53], flow matching [35], rectified flow [37], RDM [55], pyramidal flow [24] and PDDPM [49]. However, these methods typically require retraining and access to highresolution datasets to support high-resolution generation. Consequently, exploring training-free approaches for highresolution synthesis has become a key area of interest within the vision generation community. Our method is primarily designed to enable efficient, training-free high-resolution synthesis in a plug-and-play manner.

We propose a framework, RectifiedHR, to enable highresolution synthesis by progressively increasing resolution during sampling. The simplest baseline is to progressively increase the resolution in the latent space. However, naive resizing in latent space introduces noise and artifacts. We identify two critical issues and propose corresponding solutions: (1) Since the latent space is obtained by transforming RGB images via a VAE, RGB-based resizing becomes invalid in the latent space (Tab. 2, Method D). Moreover, as the latent comprises “predicted x0” and Gaussian noise, direct resizing distorts the noise distribution. To address this, we propose noise refresh, which independently resizes “predicted x0”—shown to exhibit RGB characteristics in late sampling (Fig. 2)—and injects fresh noise to maintain a valid latent distribution while increasing resolution. (2) We are the first to observe that resizing “predicted x0”: introduces spatial correlations, reducing pixel-wise independence, causing detail loss and blur, and leading to energy decay (Fig. 3a). To mitigate this, we propose energy rectification, which adjusts the CFG hyperparameter (Fig. 3b) to compensate for the energy decay and eliminate blur. Compared to [5, 28, 66], our method achieves high-resolution synthesis without additional sampling steps or complex CFG calculations, ensuring computational efficiency.

#### 2.2. Training-free high-resolution image generation

Due to the domain gap across different resolutions, directly applying diffusion models to high-resolution image generation often results in pattern repetition and poor semantic structure. MultiDiffusion [2] proposes a sliding window denoising scheme for panoramic image generation. How-

[Figure 21]

[Figure 22]

[Figure 23]

1.0

1.6 1.4 1.2 1.0 0.8 0.6

[Figure 24]

[Figure 25]

Original sampling process

[Figure 26]

ω=50

ω=1 ω=3 ω=5 ω=7 ω=10 ω=15 ω=30 ω=50

[Figure 27]

[Figure 28]

Our sampling process

[Figure 29]

AverageLatentEnergy

0.9 0.8 0.7 0.6 0.5

AverageLatentEnergy

[Figure 30]

[Figure 31]

ω=10

Energy Rectification

[Figure 32]

[Figure 33]

ω=1

0.4

0 10 20 30 40 50

0 10 20 30 40 50

Timestep

Timestep

(a) The energy decay phenomenon of our noise refresh sampling process is evaluated in comparison to the original sampling process across 100 random prompts.

(b) The evolution of average latent energy over timesteps during the generation of 1024×1024 resolution images from 100 random prompts under different classifierfree guidance hyperparameters.

- Figure 3. (a) The x-axis denotes the timesteps of the sampling process, and the y-axis indicates the average latent energy. The blue line shows the average latent energy of the original sampling process when generating 1024 × 1024-resolution images. The red line corresponds to our noise refresh sampling process, where noise refresh is applied at the 30th and 40th timesteps, and the resolution progressively increases from 1024 × 1024 to 2048 × 2048, and subsequently to 3072 × 3072. It can be observed that noise refresh induces a noticeable decay in average latent energy. From the left images, it is evident that after energy rectification, image details become more pronounced. (b) The x-axis represents the timestep, the y-axis represents the average latent energy, and ω denotes the hyperparameter for classifier-free guidance. It can be observed that the average latent energy increases as ω increases. From the right figures, one can observe how the generated images vary with increasing ω.

ever, this method suffers from severe pattern repetition, as it primarily focuses on the aggregation of local information. Improved variants based on the sliding window denoising scheme include SyncDiffusion [30], Demofusion [11], AccDiffusion [34], and CutDiffusion [33]. Specifically, SyncDiffusion incorporates global information by leveraging the gradient of perceptual loss from the predicted denoised images at each denoising step as guidance. Demofusion employs progressive upscaling, skip residuals, and dilated sampling mechanisms to support higher-resolution image generation. AccDiffusion introduces patch-content-aware prompts, while CutDiffusion adopts a coarse-to-fine strategy to mitigate pattern repetition. Nonetheless, these approaches share complex implementation logic and encounter efficiency bottlenecks due to redundant computation arising from overlapping sliding windows.

InfoScale [63], FAM [60], ScaleCrafter [16], FouriScale [22], HiDiffusion [65] and Attn-SF [25] modify the network architecture of the diffusion model, which may result in suboptimal performance. These methods perform high-resolution denoising throughout the entire sampling process, leading to slower inference compared to our approach, which progressively transitions from low to high resolution. Although HiDiffusion accelerates inference using window attention mechanisms, our method remains faster, as demonstrated by experimental results.

Upscale Guidance [23] and ElasticDiffusion [15] propose incorporating global and local denoising information into classifier-free guidance [19]. The global branch of Up-

scale Guidance and the overlapping window regions in the local branch of ElasticDiffusion involve higher computational complexity compared to our progressive resolution increase strategy. ResMaster [52] and HiPrompt [38] introduce multi-modal models to regenerate prompts and enrich image details; however, the use of such multi-modal models introduces substantial overhead, leading to efficiency issues.

DiffuseHigh [28], MegaFusion [59], FreCas [66], and AP-LDM [5] leverage the detail enhancement capabilities of SDEdit [40], progressively adding details from lowresolution to high-resolution images. In contrast to these methods, our approach neither increases sampling steps nor requires additional computations involving classifierfree guidance (CFG) variants, resulting in greater efficiency. Moreover, we identify the issue of energy decay and show that adjusting the classifier-free guidance parameter is sufficient to rectify the energy and achieve improved results.

### 3. Method

#### 3.1. Preliminaries

Diffusion models establish a mapping between Gaussian noise and images, enabling image generation by randomly sampling noise. In this paper, we assume 50 sampling steps, with the denoising process starting at step 0 and ending at step 49. We define Io as the RGB image. During training, the diffusion model first employs a VAE encoder E(·) to transform the RGB image into a lower-dimensional latent representation, denoted as x0. The forward diffusion process

RectifiedHR: Enable E!icient High Resolution Image Generation via Energy Rectification • 5

- Algorithm 1 Original Sampling Process Require: 𝑄𝑋 ↔ N(0,𝑊), 0 ↗ 𝑉 ↘ R

- 1: for 𝑁 in range(50) do
- 2: 𝑋˜(𝑄𝑄,𝑈, ≃) = 𝑋ˆ(𝑄𝑄,𝑈) + 𝑉 · [𝑋ˆ(𝑄𝑄,𝑈,𝑌) ↑ 𝑋ˆ(𝑄𝑄,𝑈, ≃)]
- 3: 𝑍𝑁𝑄0 ⇐ (𝑄𝑄 ↑

⇒1 ↑ 𝑎¯𝑄𝑋˜(𝑄𝑄,𝑈))/

⇒𝑎¯𝑄

- 4: 𝑄𝑄↑1 = ⇒𝑎¯𝑄↑1𝑍𝑁𝑄0 +

⇒1 ↑ 𝑎¯𝑄↑1𝑋˜(𝑄𝑄,𝑈)

- 5: end for
- 6: return 𝑄0

- Algorithm 2 Our Sampling Process

𝐿𝐿𝑀𝑁 and𝐿𝐿𝑂𝑃 de!ne the range of sampling timesteps to use noise refresh. 𝑀 denotes the number of noise refresh that need to be performed. The range of 𝑁 is all integers between 1 and 𝑀. 𝑂 is a hyperparameter that can be adjusted to obtain di"erent strategies to select𝐿𝑂.

- 457
- 458
- 459
- 460
- 461
- 462
- 463
- 464
- 465
- 466
- 467
- 468
- 469
- 470
- 471
- 472
- 473
- 474
- 475
- 476
- 477
- 478
- 479
- 480
- 481
- 482
- 483
- 484
- 485
- 486
- 487
- 488
- 489
- 490
- 491
- 492
- 493
- 494
- 495
- 496
- 497
- 498
- 499
- 500
- 501
- 502
- 503
- 504
- 505
- 506
- 507
- 508
- 509
- 510
- 511
- 512
- 513

- 514
- 515
- 516
- 517
- 518
- 519
- 520
- 521
- 522
- 523
- 524
- 525
- 526
- 527
- 528
- 529
- 530
- 531
- 532
- 533
- 534
- 535
- 536
- 537
- 538
- 539
- 540
- 541
- 542
- 543
- 544
- 545
- 546
- 547
- 548
- 549
- 550
- 551
- 552
- 553
- 554
- 555
- 556
- 557
- 558
- 559
- 560
- 561
- 562
- 563
- 564
- 565
- 566
- 567
- 568
- 569
- 570

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

0 T1 T2 50 Denoising Timestep

3.3 Energy rectification

(a) Original Sampling Process

Although the image resolution increases after using noise refresh, we !nd that the generated images exhibit signi!cant blurring if we do not conduct further process, as shown in the fourth column in Fig. 7. To analyze the cause of this phenomenon, we introduce relative latent energy formula as follows:

Energy Rectification

Energy Rectification

Require: 𝑄𝑋 ↔ N(0,𝑊), 0 ↗ 𝑉 ↘ R Require: ω𝑌𝑍𝑎𝑄𝑂𝑏 𝑂𝑍𝑐 = {{𝐿𝑂 : 𝑉𝑂}|𝑁 = 1...𝑀 ↑ 1}

[Figure 38]

[Figure 39]

Noise Refresh

- 1: for 𝑁 in range(50) do
- 2: 𝑋˜(𝑄𝑄,𝑈) = 𝑋ˆ(𝑄𝑄,𝑈, ≃) + 𝑉 · [𝑋ˆ(𝑄𝑄,𝑈,𝑌) ↑ 𝑋ˆ(𝑄𝑄,𝑈, ≃)]
- 3: 𝑍𝑁𝑄0 ⇐ (𝑄𝑄 ↑

⇒1 ↑ 𝑎¯𝑄𝑋˜(𝑄𝑄,𝑈))/

⇒𝑎¯𝑄

- 4: if 𝑁 in ω𝑌𝑍𝑎𝑄𝑂𝑏 𝑂𝑍𝑐.keys() then
- 5: 𝑍˜𝑁𝑄0 ⇐ 𝑃(𝑏𝑐𝑑𝑁𝑒𝑐(𝑓(𝑍𝑁𝑄0)))
- 6: 𝑋 ↔ N(0,𝑊𝑑˜𝑂 𝑃0

)

- 7: 𝑄𝑄↑1 = ⇒𝑎¯𝑄↑1𝑍˜𝑁𝑄0 +

⇒1 ↑ 𝑎¯𝑄↑1𝑋

- 8: 𝑉 = ω𝑌𝑍𝑎𝑄𝑂𝑏 𝑂𝑍𝑐 [𝑈]
- 9: else
- 10: 𝑄𝑄↑1 = ⇒𝑎¯𝑄↑1𝑍𝑁𝑄0 +

⇒1 ↑ 𝑎¯𝑄↑1𝑋˜(𝑄𝑄,𝑈)

- 11: end if
- 12: end for
- 13: return 𝑄0

𝑈 𝑉=1 𝑄𝑄2𝐿𝑀𝑁

𝑅 𝑂=1

𝑆 𝑇=1

𝑃𝑄𝑄2 =

, (8)

𝑅 → 𝑆 →𝑇

[Figure 40]

[Figure 41]

Noise Refresh

𝑄𝑄 represents the latent variable at time𝑈, where𝑅, 𝑆, and𝑇 denote the dimensions of the channel, height, and width of latent, respectively. The de!nition is very similar to the energy de!nition of an image, and is used to indicate the average energy of each element of a latent vector.

[Figure 42]

[Figure 43]

To analyze the issue of image blurring, we conduct an average energy experiment on 100 random prompts. As illustrated in Fig. 4, we !rst compare the relative latent energy di"erences between the noise refresh sampling process and the original sampling process. We observe a signi!cant energy decay phenomenon in the noise refresh sampling process, which is the reason why the naive implementation method produces noticeably blurred images. Subsequently, we conduct an experiment to analyze the impact of the hyperparameter 𝑉 in classi!er-free guidance on latent energy. As shown in Fig. 5, we !nd that as the classi!er-free guidance parameter 𝑉 increases, the energy exhibits a gradually increasing trend. Therefore, we can address the issue of energy decay and improve the quality of generated images by increasing 𝑉 to enhance the energy in the noise refresh sampling scheme. As demonstrated in the fourth column and !fth column in Fig. 7, after the energy is recti!ed with a larger classi!er-free guidance hyperparameter 𝑉, the blurry issue has been well addressed and the generated image shows remarkable clarity. We refer to this process of correcting energy decay as Energy Recti!cation.

0 T1 T2 50 Denoising Timestep

(b) Our Sampling Process

4 Experiment 4.1 Evaluation Setup

- Figure 4. Overview and Pseudo Code of RectifedHR. During sampling, we perform Noise Refresh at specific steps, resizing p˜tx0 in the RGB space, followed by Energy Rectification, where the classifier-free guidance parameter is appropriately increased to rectify energy decay in the sampling process and thereby recover missing image details.

We conduct experiments using SDXL [39] with 50 sampling steps as our base model, which is able to generate images at 1024 x 1024 resolution by default. Following the previous work, we randomly sample 1,000 prompts from the laion-5B [45] dataset as conditions to generate images. We compare our method with the following state-of-the-art: Demofusion [10], Di"useHigh [25], HiDi"usion[55], CutDi"usion [30], ElasticDi"usion [13], AP-LDM [5], FreCas [56], SDXL+BSRGAN [54] FouriScale [20], ScaleCrafter [14], and AccDiffusion [31]. Except for SDXL+BSRGAN, which requires to use the trained BSRGAN model, other methods are training-free. We !x inference steps and set the negative prompts as empty. In addition, we remove additional tricks such as FreeU [47] for a fair comparison. Quantitatively, we mainly generate high-resolution images at target resolutions of 2048 x 2048, 4x of the original resolution.

is then defined as:

We refer to ϵ˜(xt,t) as the predicted noise after applying classifier-free guidance.

As shown in Fig. 6b, the energy recti!cation operation is applied to the sampling process after noise refresh. To more automatically select 𝑉 in the classi!er-free guidance, we propose the following selection formula:

xt = √α¯tx0 + √1 − α¯tϵ. (1)

Sampling process for diffusion models. In this paper, we adopt the DDIM sampler [53] as the default. The deterministic sampling formulation of DDIM is given as follows:

Noise of varying intensity is added to x0 to produce different xt, where α¯t is a time-dependent scheduler parameter controlling the noise strength, and ϵ is randomly sampled Gaussian noise. The diffusion model ϵˆ(xt,t,c), parameterized by θ, is optimized to predict the added noise via the following training objective:

We employ four widely used quantitative evaluation metrics: FID (Frechet Inception Distance) [16], KID (Kernel Inception Distance) [3], IS (Inception Score) [44], and CLIP Score [40]. Speci!cally, FID𝑌, KID𝑌, and IS𝑌 require resizing images to 299x299 before calculation. However, this kind of evaluation is not reasonable for high-resolution image generation. Following the approach of previous works [10, 31], we randomly crop 10 patches of 1024x1024 (1x) from each generated high-resolution image to further calculate FID𝑒, KID𝑎, and IS𝑎. We set 𝐿𝐿𝑂𝑃 at 40, 𝐿𝐿𝑀𝑁 at 50, N at 1, 𝑉𝐿𝑂𝑃 at

𝑁 𝑀 )𝑊 + 𝑉𝐿𝑂𝑃 (9)

𝑉𝑂 = (𝑉𝐿𝑀𝑁 ↑ 𝑉𝐿𝑂𝑃) ↓ (

𝑉𝐿𝑀𝑁 and 𝑉𝐿𝑂𝑃 represent the range of 𝑉 in classi!er-free guidance during sampling process. 𝑀 denotes the number of noise refresh that needs to be performed. The range of 𝑁 is all integers between 1 and 𝑀. 𝑂 is a hyperparameter that can be adjusted to obtain di"erent strategies to select 𝑉𝑂.

√1 − α¯t · ϵ˜(xt,t) √α¯t

xt−1 = √α¯t−1

xt −

(4)

, Vol. 1, No. 1, Article . Publication date: May 2025.

predicted x0→ptx0

t,t,c ∥ϵ − ϵˆ(xt,t,c)∥22 , (2)

Ex

+ 1 − α¯t−1 · ϵ˜(xt,t).

min

θ

As illustrated in Eq. 4, at timestep t, we first predict the noise ϵ˜(xt,t) using the pre-trained neural network ϵˆ(·). We then compute a “predicted x0” at timestep t, denoted as ptx

where c denotes the conditioning signal for generation (e.g., a text prompt in T2I tasks). During inference, random noise is sampled in the latent space, and the diffusion model gradually transforms this noise into an image via a denoising process. Finally, the latent representation is passed through the decoder D(·) of the VAE to reconstruct the generated RGB image. The objective of high-resolution synthesis is to produce images at resolutions beyond those seen during training—for instance, resolutions exceeding 1024 × 1024 in our setting.

using the diffusion process defined in Eq. 4.

. Finally, xt−1 is derived from ϵ˜(xt,t) and ptx

0

0

In this paper, we propose RectifiedHR, which consists of noise refresh and energy rectification. The noise refresh module progressively increases the resolution during the sampling process, while the energy rectification module enhances the visual details of the generated contents.

#### 3.2. Noise refresh

Classifier-free guidance for diffusion models. Classifier-free guidance (CFG) [19] is currently widely adopted to enhance the quality of generated images by incorporating unconditional outputs at each denoising step. The formulation of classifier-free guidance is as follows:

To enable high-resolution synthesis, we propose a progressive resizing strategy during sampling. A straightforward baseline for implementing this strategy is to directly perform image-space interpolation in the latent space. However, this approach presents two key issues. First, since the latent space is obtained via VAE compression of the image, interpolation operations that work in RGB space are ineffective in the latent space, as demonstrated by Method D in the ablation study (Table 2). Second, because the latent space

ϵ˜(xt,t) = ϵˆ(xt,t,∅) + ω · [ˆϵ(xt,t,c) − ϵˆ(xt,t,∅)], (3)

where ω is the hyperparameter of classifier-free guidance, ϵˆ(xt,t,∅) and ϵˆ(xt,t,c) denote the predicted noises from the unconditional and conditional branches, respectively.

consists of ptx

and noise, directly resizing it alters the noise distribution, potentially shifting the latent representation outside the diffusion model’s valid domain. To address this, we visualize ptx

0

, as shown in Fig. 2, and observe that the image corresponding to ptx

0

exhibits RGB-like characteristics in the later stages of sampling. Therefore, we resize ptx

0

to enlarge the latent representation. To ensure the resized latent maintains a Gaussian distribution, we inject new Gaussian noise into ptx

0

. The method for enhancing the resolution of ptx

0

is as follows:

- 0

p˜tx

= E(resize(D(ptx

))), (5)

0

0

where E denotes the VAE encoder, D denotes the VAE decoder, and resize(·) refers to the operation of enlarging the RGB image. We adopt bilinear interpolation as the default resizing method. The procedure for re-adding noise is as follows:

xt−1 = √α¯t−1p˜tx

+ 1 − α¯t−1ϵ, (6)

0

where ϵ denotes a random Gaussian noise that shares the same shape as p˜tx

. We refer to this process as Noise Refresh.

0

As illustrated in Fig. 4b, the noise refresh operation is applied at specific time points Ti during the sampling process. To automate the selection of these timesteps T, we propose the following selection formula:

i − 1 N

)M

Ti = ⌊(Tmax − Tmin) ∗ (

+ Tmin⌋, (7)

T

where Tmax and Tmin define the range of sampling timesteps at which noise refresh is applied. N denotes the number of different resolutions in the denoising process, and N − 1 corresponds to the number of noise refresh operations performed. N is a positive integer, and the range of i includes all integers in [1,N). Specifically, we set T0 to 0 and Tmax to the total number of sampling steps. Tmin is treated as a hyperparameter. Since ptx

exhibits more prominent image features in the later stages of sampling, as shown in Fig. 2, Tmin is selected to fall within the later stage of the sampling process. Eq. 7 includes the linear interpolation form as well as the curve interpolation form.

0

#### 3.3. Energy rectification

Although noise refresh enables the diffusion model to generate high-resolution images, we observe that introducing noise refresh during the sampling process leads to blurriness in the generated content, as illustrated in the fourth row of Fig. 6. To investigate the cause of this phenomenon, we introduce the average latent energy formula as follows:

C i=1

H j=1

W k=1 x2t

E[x2t] =

, (8)

ijk

C × H × W

where xt represents the latent variable at time t, and C, H, and W denote the channel, height, and width dimensions of the latent, respectively. This definition closely resembles that of image energy and is used to quantify the average energy per element of the latent vector. To investigate the issue of image blurring, we conduct an average latent energy analysis on 100 random prompts. As illustrated in Fig. 3a, we first compare the average latent energy between the noise refresh sampling process and the original sampling process. We observe significant energy decay during the noise refresh sampling process, which explains why the naive implementation produces noticeably blurred images. Subsequently, we experimentally discover that the hyperparameter ω in classifier-free guidance influences the average latent energy.

- As shown in Fig. 3b, we find that increasing the classifierfree guidance parameter ω leads to a gradual increase in energy. Therefore, the issue of energy decay—and thus image quality degradation—can be mitigated by increasing ω to boost the energy in the noise refresh sampling scheme. As demonstrated in the left image of Fig. 3a, once energy is rectified by using a larger classifier-free guidance hyperparameter ω, the blurriness is substantially reduced, and the generated image exhibits significantly improved clarity. We refer to this process of correcting energy decay as Energy Rectification. However, we note that a larger ω is not always beneficial, as excessively high values may lead to overexposure. The goal of energy rectification is to align the energy level with that of the original diffusion model’s denoising process, rather than to maximize energy indiscriminately. The experiment analyzing the rectified average latent energy curve is provided in Sec. 7.8.
- As shown in Fig. 4b, the energy rectification operation is

applied during the sampling process following noise refresh. To automatically select an appropriate ω value for classifierfree guidance, we propose the following selection formula:

i N − 1

)M

+ ωmin, (9)

ωi = (ωmax − ωmin) ∗ (

ω

where ωmax and ωmin define the range of ω values used in classifier-free guidance during the sampling process. N denotes the number of different resolutions in the denoising process, and N−1 corresponds to the number of noise refresh operations performed. N is a positive integer, and the range of i includes all integers in [0,N). ωmin refers to the CFG hyperparameter at the original resolution supported by the diffusion model. Mω is a tunable hyperparameter that allows for different strategies in selecting ωi. The value of N used in Eq. 7 and Eq. 9 remains consistent throughout the sampling process. Eq. 9 includes the linear interpolation form as well as the curve interpolation form. We establish connection between energy rectification and SNR correction [21, 59, 66], showing that SNR correction is a form of energy rectification. The proof is provided in Sec. 7.4.

[Figure 44]

[Figure 45]

Ours SDXL+BSRGAN

| |Methods<br><br>|FIDr ↓|KIDr ↓<br><br>|ISr ↑|FIDc ↓|KIDc ↓<br><br>|ISc ↑|CLIP↑<br><br>|Time↓|User Study↑|
|---|---|---|---|---|---|---|---|---|---|---|
|20482048|FouriScale ScaleCrafter HiDiffusion CutDiffusion ElasticDiffusion AccDiffusion DiffuseHigh FreCas DemoFusion Ours|71.344 64.236 63.674 59.152 56.639 48.143 49.748 49.129 47.079 48.361<br><br>|0.010 0.007 0.007 0.007 0.010 0.002 0.003 0.003 0.002 0.002<br><br>|15.957 15.952 16.876 17.109 15.326 18.466 19.537 20.274 19.533 20.616<br><br>|53.990 45.861 41.930 38.004 37.649 32.747 27.667 27.002 26.441 25.347<br><br>|0.014 0.010 0.008 0.008 0.014 0.008 0.004 0.004 0.004 0.003<br><br>|20.625 22.252 23.165 23.444 19.867 24.778 27.876 29.843 27.843 28.126<br><br>|31.157 31.803 31.711 32.573 32.301 33.153 33.436 33.700 33.748 33.756<br><br>|59s 35s 18s 53s 150s 111s 37s 14s 79s 13s<br><br>|11.6%<br><br>13.6%<br><br>12.7%<br><br>-<br><br>13.8%<br><br><br>16.2% 32.2%<br><br>|
|40964096<br><br>|FouriScale ScaleCrafter HiDiffusion CutDiffusion ElasticDiffusion AccDiffusion DiffuseHigh FreCas DemoFusion Ours<br><br>|135.111 110.094 93.515 130.207 101.313 54.918 48.861 49.764 48.983 48.684<br><br>|0.046 0.028 0.024 0.055 0.056 0.005 0.003 0.003 0.003 0.003<br><br>|9.481 10.098 11.878 9.334 9.406 17.444 19.716 18.656 18.225 20.352<br><br>|129.895 112.105 120.170 113.033 111.102 60.362 40.267 39.047 38.136 35.718<br><br>|0.057 0.043 0.058 0.055 0.089 0.023 0.010 0.010 0.010 0.009<br><br>|9.792 11.421 11.272 10.961 7.627 16.370 21.550 21.700 20.786 20.819<br><br>|26.891 27.809 27.853 26.734 27.725 32.438 33.390 33.237 33.311 33.415<br><br>|489s 528s 71s 193s 400s 826s 190s 74s 605s 37s<br><br>|11.6% 13.6%<br>12.7%<br><br>-<br><br>13.8%<br><br><br>16.2% 32.2%<br><br>|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

××2048204840964096

Figure 5. Qualitative comparison between our method and SDXL+BSRGAN at a resolution of 2048 × 2048.

and third on ISc, its dominance on the other metrics, together with strong computational efficiency, demonstrates its overall effectiveness and robustness for high-resolution generation. When scaled to 4096 × 4096, RectifiedHR is roughly twice as fast as the next fastest approach. This speedup comes from preserving the original number of sampling steps and carefully tuning the CFG hyperparameter. In contrast, methods such as DiffuseHigh incur substantial overhead by adding extra sampling via repeated SDEdit and FreCas within heavier CFG pipelines. Notably, RectifiedHR achieves this speed without sacrificing quality, matching or exceeding baseline visual fidelity across resolutions, thereby striking a favorable speed–quality balance. User study also demonstrates the advantages of our approach. Details of the user study are presented in Sec. 7.2. Since the images of all resolutions were mixed together during the user study, the user study values in different resolutions are the same.

- Table 1. Comparison to SOTA methods at 2048×2048 and 4096× 4096 resolutions. Bold numbers indicate the best performance, while underlined numbers denote the second-best performance.

### 4. Experiments

#### 4.1. Evaluation Setup

Our experiments use SDXL [43] as the base model, which by default generates images at a resolution of 1024×1024. Furthermore, our method can also be applied to Stable Diffusion and transformer-based diffusion models such as WAN [58] and SD3 [12], as demonstrated in Fig. 7 and Sec. 7.5. The specific evaluation metrics and methods are provided in Sec. 7.1. We follow prior protocols and randomly select 1,000 prompts from LAION-5B [51] for text-to-image generation. The comparison includes state-of-the-art trainingfree methods: Demofusion [11], DiffuseHigh [28], HiDiffusion [65], CutDiffusion [33], ElasticDiffusion [15], FreCas [66], FouriScale [22], ScaleCrafter [16], and AccDiffusion [34]. Quantitative assessments focus on upsampling to 2048 × 2048 and 4096 × 4096 resolutions. All baseline methods are fairly and fully reproduced. For the 2048×2048

#### 4.3. Comparison with the super-resolution model

Training-free high-resolution image generation methods primarily exploit intrinsic properties of diffusion models to achieve super-resolution. Beyond the aforementioned approaches, another viable strategy adopts a two-stage pipeline that combines diffusion models with dedicated super-resolution models. For example, methods such as SDXL + BSRGAN first generate an image using a diffusion model, then apply a super-resolution model to upscale it to the target resolution. To further evaluate the differences between SDXL+BSRGAN and our method, we conduct additional qualitative comparisons. The experimental setup follows that described in Sec. 4.1. As shown in Fig. 5, we observe that when images generated by SDXL exceed the domain of the original training data—such as in cases involving distorted facial features—BSRGAN is unable to correct these artifacts, resulting in performance degradation. Furthermore, existing two-stage approaches rely on pre-trained super-resolution models constrained by fixedresolution training data. In contrast, our method inherently adapts to arbitrary resolutions without retraining. For example, as demonstrated in the 2048 × 4096 resolution scene, our approach remains effective, whereas BSRGAN cannot be applied.

- resolution setting, we set Tmin to 40, Tmax to 50, N to 2, ωmin to 5, ωmax to 30, MT to 1, and Mω to 1. For the 4096×4096
- resolution setting, we set Tmin to 40, Tmax to 50, N to 3, ωmin to 5, ωmax to 50, MT to 0.5, and Mω to 0.5. The above hyperparameters are obtained through a hyperparameter search, with detailed ablation studies provided in Sec. 7.6. More qualitative results are presented in Sec. 7.9 and Sec. 7.10.

#### 4.2. Quantitative Results

As shown in Tab. 1, RectifiedHR consistently surpasses competing methods at both 2048 × 2048 and 4096 × 4096. At 2048×2048, it leads 6/8 metrics, placing second on one and third on another; at 4096×4096, it leads 7/8 and places third on the remaining metric. At 2048 × 2048, our KIDr ranks third because this metric downsamples high-resolution images for evaluation, underrepresenting fine details—a known limitation [11, 34]. Although RectifiedHR ranks second

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

| |Methods<br><br>|Noise Refresh<br><br>|Energy Rectification|Resize Latent<br><br>|FIDr ↓|KIDr ↓|ISr ↑<br><br>|FIDc ↓<br><br>|KIDc ↓|ISc ↑<br><br>|CLIP↑|
|---|---|---|---|---|---|---|---|---|---|---|---|
|20482048<br><br>|A<br>B<br>C<br>D<br><br><br>Ours|× × ✓ × ✓<br><br>|× ✓ × ✓ ✓|× × × ✓ ×<br><br>|98.676 86.595 79.743 78.307 48.361<br><br>|0.030 0.021 0.021 0.019 0.002<br><br>|13.193 13.900 13.334 13.221 20.616|73.426 60.625 76.023 74.419 25.347<br><br>|0.029 0.021 0.035 0.034 0.003|17.867 19.921 11.840 11.883 28.126<br><br>|30.021 30.728 29.966 29.523 33.756<br><br>|
|40964096<br><br>|A<br><br>B<br><br>C<br><br>D<br><br><br>Ours<br><br>|× × ✓ × ✓|× ✓ × ✓ ✓<br><br>|× × × ✓ ×<br><br>|187.667 175.830 85.088 89.968 48.684<br><br>|0.088 0.079 0.026 0.033 0.003<br><br>|8.636 8.403 13.114 11.973 20.352|111.117 80.733 141.422 145.472 35.718<br><br>|0.057 0.034 0.091 0.103 0.009|13.383 15.791 5.465 6.312 20.819<br><br>|25.447 26.099 29.548 28.212 33.415|

Direct Inference Noise Refresh Energy Rectification (2048x2048)

Ours Noise Refresh Energy Rectification (2048x2048)

[Figure 56]

Noise Refresh Energy Rectification (2048x2048)

Noise Refresh Energy Rectification (2048x2048)

[Figure 57]

[Figure 58]

Direct Inference (1024x1024)

[Figure 59]

××2048204840964096

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Table 2. Quantitative results of the ablation studies. Method A denotes direct inference (without noise refresh and energy rectification), Method B excludes noise refresh, Method C excludes energy rectification, and Method D replaces noise refresh in our method with direct latent resizing. Ours refers to the full version of our proposed method.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(a) (b) (c) (d) (e)

Figure 6. Qualitative results of the ablation studies at 2048 × 2048 resolution. The orange and blue boxes indicate enlarged views of local regions within the high-resolution image. Zoom in for details.

#### 4.4. Ablation Study

To evaluate the effectiveness of each module in our method, we conduct both quantitative experiments (Tab. 2) and qualitative experiments (Fig. 6). The metric computation and all hyperparameter settings follow Sec. 4.1. Additionally, in scenarios without energy rectification, the classifier-free guidance hyperparameter ω is fixed at 5. For simplicity, this section compares the FIDc at the 4096 × 4096 resolution.

| |Visual Quality ↑<br><br>|Motion Quality ↑<br><br>|Temporal Consistency ↑|
|---|---|---|---|
|Direct Inference|65.31<br><br>|51.91|63.78|
|Ours|67.22|54.30<br><br>|64.26|

Comparing Method B in Tab.2 with Ours, the FIDc increases from 35.718 to 80.733 without noise refresh. As shown in Fig. 6c vs. Fig. 6e, this performance drop is due to the failure in generating correct semantic structures caused by the absence of noise refresh. Fig. 6d and Fig. 6e highlight the critical role of energy rectification in enhancing fine details. Comparing Method C in Tab. 2 with Ours, the FIDc rises sharply from 35.718 to 141.422 without energy rectification, demonstrating that energy decay severely degrades generation quality. This underscores the importance of energy rectification—despite its simplicity, it yields significant improvements. Comparing Method D in Tab. 2 with Ours, the FIDc improves from 145.472 to 35.718, revealing that directly resizing the latent is ineffective. This confirms that noise refresh is indispensable and cannot be replaced by na¨ıve latent resizing. We also conduct ablation studies on the hyperparameters related to Eq. 7 and Eq. 9, with detailed results provided in Sec. 7.6.

Table 3. Quantitative results of video generation.

Video Generation. RectifiedHR can be directly applied to video diffusion models such as WAN [58]. The officially supported maximum resolution for WAN 1.3B is 480 × 832. As shown in Fig. 7a and Tab. 3, directly generating high-resolution videos with WAN may lead to generation failure or prompt misalignment. However, integrating RectifiedHR enables WAN to produce high-quality, high-resolution videos reliably. More experimental results and details are presented in Sec. 7.11 and Sec. 7.7.

Image Editing. RectifiedHR can be applied to image editing tasks. In this section, we use SDXL as the base model with a default resolution of 1024 × 1024. Directly editing high-resolution images with OIR often results in ghosting artifacts, as illustrated in rows a, b, d, and e of Fig. 7b. Additionally, it can cause shape distortions and deformations, as shown in rows c and f. In contrast, the combination of OIR and RectifiedHR effectively mitigates these issues, as demonstrated in Fig. 7b.

### 5. More Applications

Customized Generation. RectifiedHR can be directly adapted to DreamBooth using SD1.4 with a default resolution of 512×512, as shown in Fig. 7c. The direct generation of high-resolution customized images often leads to severe repetitive pattern artifacts. Integrating RectifiedHR effectively addresses this problem.

This section highlights how RectifiedHR can enhance a variety of tasks, with a focus on demonstrating visual improvements. The experiments cover diverse tasks, models, and sampling methods to validate the effectiveness of our approach. While primarily evaluated on classic methods and models, RectifiedHR can also be seamlessly integrated into more advanced techniques. Sec. 7.7 provides detailed quantitative results and corresponding hyperparameter settings.

Controllable Generation. RectifiedHR can be seamlessly integrated with ControlNet [64] using SDXL at a

Prompt: A busy city street at night with moving cars.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

WAN

WANRectifiedHR WAN

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

WAN

+

Prompt: A cat playing with a ball of yarn.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

RectifiedHR

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

+

(a) Video Generation (Resolution: 960x1664).

Original Image OIR OIR + RectifiedHR DreamBooth + RectifiedHR

Prompts

DreamBooth DreamBooth + RectifiedHR DreamBooth

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Input images

rocks ! ship

- (a)
- (b)
- (c)

Resolution:2048x2048Resolution:3072x3072

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

wooden framed decorations ! portrait of the Statue of Liberty

full moon ! crescent moon

helmet ! SpiderMan mask

- (d)
- (e)
- (f)

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

flowers ! grapes

Prompt: a rb robot on the basketball court Prompt: a rb robot on the basketball court

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Input images

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

cat ! tiger

[Figure 135]

[Figure 136]

Prompt: a rc car on the train track Prompt: a rc car on the grass

(c) Customized Generation (Resolution: 1536x1536).

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

OriginalCannyOriginalPose

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

(d) Controllable Generation (Resolution: 3072x3072).

(b) Image Editing.

Figure 7. Applications. (a) Video Generation. (b) Image Editing. (c) Customized Generation. (d) Controllable Generation. Contents are best viewed when zoomed in.

default resolution of 1024 × 1024 to enable controllable generation. As shown in Fig. 7d, control signals may include pose, canny edges, and other modalities.

introduces a novel training-free pipeline that is both simple and effective, primarily incorporating noise refresh and energy rectification operations. Extensive comparisons demonstrate that RectifiedHR outperforms existing methods in both effectiveness and efficiency. Nonetheless, our method has certain limitations. During the noise refresh stage, it requires both decoding and encoding operations via the VAE, which impacts the overall runtime. In future work, we aim to investigate performing resizing operations directly in the latent space to further improve efficiency.

### 6. Conclusion and Future Work

We propose an efficient and straightforward method, RectifiedHR, for high-resolution synthesis. Specifically, we conduct an average latent energy analysis and, to the best of our knowledge, are the first to identify the energy decay phenomenon during high-resolution synthesis. Our approach

### References

- [1] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In ECCV, pages 707–723. Springer, 2022. 1
- [2] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 1, 2
- [3] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 12
- [4] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 18392–18402, 2023. 1
- [5] Boyuan Cao, Jiaxin Ye, Yujie Wei, and Hongming Shan. Ap-ldm: Attentive and progressive latent diffusion model for training-free high-resolution image generation. arXiv preprint arXiv:2410.06055, 2024. 2, 3
- [6] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 17
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 1, 2
- [8] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2025. 2
- [9] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. 2022. 1
- [10] Ganggui Ding, Canyu Zhao, Wen Wang, Zhen Yang, Zide Liu, Hao Chen, and Chunhua Shen. Freecustom: Tuning-free customized image generation for multi-concept composition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9089–9098, 2024. 1
- [11] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising highresolution image generation with no. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6159–6168, 2024. 2, 3, 6, 12
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 2, 6
- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618,

2022. 1

- [14] Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation. In European Conference on Computer Vision, pages 39–55. Springer, 2024. 2
- [15] Moayed Haji-Ali, Guha Balakrishnan, and Vicente Ordonez. Elasticdiffusion: Training-free arbitrary size image generation through global-local content separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6603–6612, 2024. 3, 6
- [16] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2023. 2, 3, 6
- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. 2021. 13, 15
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 12
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3, 4
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [21] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 5, 14
- [22] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. In European Conference on Computer Vision, pages 196–212. Springer, 2025. 2, 3, 6
- [23] Juno Hwang, Yong-Hyun Park, and Junghyo Jo. Upsample guidance: Scale up diffusion models without training. arXiv preprint arXiv:2404.01709, 2024. 2, 3, 14
- [24] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. 2
- [25] Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. Trainingfree diffusion model adaptation for variable-sized text-toimage synthesis. Advances in Neural Information Processing Systems, 36:70847–70860, 2023. 2, 3
- [26] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 2
- [27] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani.

- Imagic: Text-based real image editing with diffusion models. In CVPR, pages 6007–6017, 2023. 1
- [28] Younghyun Kim, Geunmin Hwang, Junyu Zhang, and Eunbyung Park. Diffusehigh: Training-free progressive highresolution image synthesis through structure guidance. arXiv preprint arXiv:2406.18459, 2024. 2, 3, 6
- [29] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2023. 1, 2
- [30] Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems, 36:50648–50660, 2023. 2, 3
- [31] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 1
- [32] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multiresolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 1, 2
- [33] Mingbao Lin, Zhihang Lin, Wengyi Zhan, Liujuan Cao, and Rongrong Ji. Cutdiffusion: A simple, fast, cheap, and strong diffusion extrapolation method. arXiv preprint arXiv:2404.15141, 2024. 2, 3, 6
- [34] Zhihang Lin, Mingbao Lin, Meng Zhao, and Rongrong Ji. Accdiffusion: An accurate method for higher-resolution image generation. In European Conference on Computer Vision, pages 38–53. Springer, 2025. 2, 3, 6, 12, 13
- [35] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [36] Mushui Liu, Yuhang Ma, Yang Zhen, Jun Dan, Yunlong Yu, Zeng Zhao, Zhipeng Hu, Bai Liu, and Changjie Fan. Llm4gen: Leveraging semantic representation of llms for text-to-image generation. arXiv preprint arXiv:2407.00737,

2024. 1

- [37] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2
- [38] Xinyu Liu, Yingqing He, Lanqing Guo, Xiang Li, Bu Jin, Peng Li, Yan Li, Chi-Min Chan, Qifeng Chen, Wei Xue, et al. Hiprompt: Tuning-free higher-resolution generation with hierarchical mllm prompts. arXiv preprint arXiv:2409.02919,

2024. 2, 3

- [39] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 1, 2
- [40] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2, 3
- [41] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. 2023. 1

- [42] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, pages 6038–6047,

2023. 1

- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for highresolution image synthesis. arXiv preprint arXiv:2307.01952,

2023. 1, 2, 6

- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 12
- [45] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024. 2
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510,

2023. 17

- [48] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 1
- [49] Dohoon Ryu and Jong Chul Ye. Pyramidal denoising diffusion probabilistic models. arXiv preprint arXiv:2208.01864,

2022. 2

- [50] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 12
- [51] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 6, 12, 13
- [52] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering highresolution image generation via structural and fine-grained guidance. arXiv preprint arXiv:2406.16476, 2024. 2, 3
- [53] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 2, 4

- [54] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based

- generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [55] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 2
- [56] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings,

2023. 1

- [57] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-toimage translation. In CVPR, pages 1921–1930, 2023. 1
- [58] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 6, 7, 17
- [59] Haoning Wu, Shaocheng Shen, Qiang Hu, Xiaoyun Zhang, Ya Zhang, and Yanfeng Wang. Megafusion: Extend diffusion models towards higher-resolution image generation without further tuning. arXiv preprint arXiv:2408.11001, 2024. 2, 3, 5, 14
- [60] Haosen Yang, Adrian Bulat, Isma Hadji, Hai X Pham, Xiatian Zhu, Georgios Tzimiropoulos, and Brais Martinez. Fam diffusion: Frequency and attention modulation for highresolution image generation with stable diffusion. arXiv preprint arXiv:2411.18552, 2024. 3
- [61] Zhen Yang, Ganggui Ding, Wen Wang, Hao Chen, Bohan Zhuang, and Chunhua Shen. Object-aware inversion and reassembly for image editing. arXiv preprint arXiv:2310.12149,

2023. 1, 17

- [62] Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14483–14494, 2025. 15
- [63] Guohui Zhang, Jiangtong Tan, Linjiang Huang, Zhonghang Yuan, Naishan Zheng, Jie Huang, and Feng Zhao. Infoscale: Unleashing training-free variable-scaled image generation via effective utilization of information. arXiv preprint arXiv:2509.01421, 2025. 3
- [64] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 7, 18
- [65] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Zhenyuan Chen, Yao Tang, Yuhao Chen, Wengang Cao, and Jiajun Liang. Hidiffusion: Unlocking high-resolution creativity and efficiency in low-resolution trained diffusion models. arXiv preprint arXiv:2311.17528, 2023. 2, 3, 6
- [66] Zhengqiang Zhang, Ruihuang Li, and Lei Zhang. Frecas: Efficient higher-resolution image generation via frequencyaware cascaded sampling. arXiv preprint arXiv:2410.18410,

2024. 2, 3, 5, 6, 14

- [67] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang,

Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024. 1, 2

### 7. Supplementary

#### 7.1. Implementation details

Although a limited number of samples may lead to lower values for metrics such as FID [18], we follow prior protocols and randomly select 1,000 prompts from LAION-5B [51] for text-to-image generation. Evaluations are conducted using 50 inference steps, empty negative prompts, and fixed random seeds.

We employ four widely used quantitative metrics: Fr´echet Inception Distance (FID) [18], Kernel Inception Distance (KID) [3], Inception Score (IS) [50], and CLIP Score [44]. FID and KID are computed using pytorch-fid, while CLIP Score and IS are computed using torchmetrics. The subscript r refers to resizing high-resolution images to 299 × 299 before evaluation, whereas the subscript c indicates that 10 patches of size 1024 × 1024 are randomly cropped from each generated high-resolution image and then resized to 299 × 299 for evaluation. Specifically, FIDr, KIDr, and ISr require resizing images to 299 × 299. However, such an evaluation is not ideal for high-resolution image generation. Following prior works [11, 34], we randomly crop 10 patches of size 1024 × 1024 from each generated high-resolution image to compute FIDs, KIDc, and ISc.

#### 7.2. User study details

[Figure 152]

Figure 8. The interface of one question in the user study

We conducted a user study to further demonstrate the effectiveness of our method. We selected 15 images in total, evenly distributed across three resolutions: 2048 × 2048, 4096 × 4096, and 2048 × 4096 (five images per resolution). 30 participants were involved in the study, where they were asked to evaluate the images provided and identify the best. The questionnaire is designed on the https://www.wjx.cn/ platform. The interface of the questionnaire is shown in Fig. 8.

The baselines in this study are consistent with those in Sec. 7.9, except for direct inference and DemoFusion. Direct inference was excluded because most of its generated images exhibited severe global distortions. The outputs of AccDiffusion and DemoFusion are highly similar under a fixed random seed. As [34] has quantitatively demonstrated the superiority of AccDiffusion, we retained AccDiffusion solely for conciseness in this study.

Fig. 9 shows the results of the user study. Our method (RectifiedHR) received 32.2% of the total votes, significantly exceeding the other competing methods. The second most selected method, FreCaS, accounted for only 16.2%, which is approximately half of RectifiedHR’s proportion. The remaining methods, including AccDiffusion (13.8%), ScaleCrafter (13.6%), HiDiffusion (12.7%), and FouriScale (11.5%), received relatively lower proportions of the total votes. These results demonstrate that more users are inclined to identify RectifiedHR as the best compared to existing approaches, validating the effectiveness of our method in subjective evaluation.

[Figure 153]

Figure 9. The results of the user study

#### 7.3. Quantitative Analysis of “Predicted x0”

To quantitatively validate this observation, as shown in Fig.10, we conduct additional experiments on the generation of ptx

0

using 100 random prompts sampled from LAION-5B [51], and analyze the CLIP Score [17] and Mean Squared Error (MSE). From Fig. 10a, we observe that after 30 denoising steps, the MSE between ptx

exhibits minimal change. In Fig. 10b, we find that the CLIP score between ptx

and ptx−1

0

0

and the corresponding prompt increases slowly beyond 30 denoising steps.

0

[Figure 154]

[Figure 155]

[Figure 156]

pxt−1

pxt

pxt

MSE( , ) Average_CLIP_Score( , prompt)

[Figure 157]

[Figure 158]

[Figure 159]

0

0

0

34 32 30 28 26 24 22 20

[Figure 160]

[Figure 161]

[Figure 162]

0.14 0.12 0.10 0.08 0.06 0.04 0.02 0.00

[Figure 163]

AverageCLIPScore

MSE

0 10 20 30 40 50 Timestep

0 10 20 30 40 50

[Figure 164]

Timestep

(a) (b)

Figure 10. The trend of the “predicted x0” at different timesteps t, denoted as ptx0, evaluated on 100 random prompts. (a) The average MSE between ptx0 and ptx−01. The x-axis represents the sampling timestep, and the y-axis denotes the average MSE. It can be observed that after approximately 30 steps, the rate of change in ptx0 slows significantly. (b) The trend of the average CLIP Score between ptx0 and the prompt across different timesteps. The x-axis represents the sampling timestep, and the y-axis denotes the average CLIP Score.

#### 7.4. The connection between energy rectification and Signal-to-Noise Ratio (SNR) correction

In the proof presented in this section, all symbols follow the definitions provided in the Method section of the main text. Any additional symbols not previously defined will be explicitly specified. This proof analyzes energy variation using the DDIM sampler as an example. The sampling formulation of DDIM is given as follows:

√1 − α¯tϵ˜(xt,t) √α¯t

xt−1 = √α¯t−1

xt −

+ 1 − α¯t−1 · ϵ˜(xt,t)

√α¯t−1√1 − α¯t √α¯t

α¯t−1 α¯t

xt + 1 − α¯t−1 −

=

ε ˜(xt,t).

(10)

To simplify the derivation, we assume that all quantities in the equation are scalar values. Based on the definition of average latent energy in Eq.8 of the main text, the average latent energy during the DDIM sampling process can be expressed as follows:

E[x2t−1] = E

+2 × E

√α¯t−1√1 − α¯t √α¯t

2

α¯t−1 α¯t

+ E 1 − α¯t−1 −

xt

√α¯t−1√1 − α¯t √α¯t

α¯t−1 α¯t

xt × E 1 − α¯t−1 −

ε ˜(xt,t)

2

ε ˜(xt,t) .

(11)

We assume that the predicted noise ϵ˜follows a standard normal distribution, such that E[˜ϵ(xt,t)] = 0. Under this assumption, the average latent energy of the DDIM sampler can be simplified as:

E[x2t−1] = E

α¯t−1 α¯t

xt

√α¯t−1√1 − α¯t √α¯t

2

+ E 1 − α¯t−1 −

ε ˜(xt,t)

2

. (12)

Several previous works [21, 23, 59, 66] define the Signal-to-Noise Ratio (SNR) at a given timestep of a diffusion model as follows:

α¯t 1 − α¯t

. (13)

SNRt =

Several works [21, 23, 59, 66] have observed that the SNR must be adjusted during the generation process at different resolutions. Suppose the diffusion model is originally designed for a resolution of H × W, and we aim to extend it to generate images at a higher resolution of H′ × W′, where H′ > H and W′ > W. According to the derivations in [59, 66], the adjusted formulation of αt is given as follows:

α¯t γ − (γ − 1)¯αt

α¯t′ =

. (14)

Here, the value of γ is typically defined as (H′/H · W′/W)2. By substituting the modified α¯t′ into Eq. 10, we obtain the SNR-corrected sampling formulation as follows:

=

α¯t−1 γ−(γ−1)¯αt−1 α¯t γ−(γ−1)¯αt

E[xt] +

=

γ − (γ − 1)¯αt γ − (γ − 1)¯αt−1

α¯t−1 α¯t

α¯′

t−1 1 − α¯′

α¯′

t α¯′

t−1 α¯′

E[xt−1] =

E[xt] + 1 − α¯′

E[˜ϵ(xt,t)]

t−1 −

t

t

   1 −

  E[˜ϵ(xt,t)]

αt−1

γ−(γ−1)¯αt−1 1 − α

t

αt−1 γ − (γ − 1)αt−1 −

γ−(γ−1)¯αt αt γ−(γ−1)αt

√α¯t−1√1 − α¯t √α¯t

γ γ − (γ − 1)¯αt−1

E[xt] +

E[˜ε(xt,t)].

1 − α¯t−1 −

The average latent energy under SNR correction can be derived as follows:

γ − (γ − 1)¯αt γ − (γ − 1)¯αt−1

E

=

2

α¯′

t−1 1 − α¯′

α¯′

t α¯′

t−1 α¯′

E[x2t−1] = E

+ E 1 − α¯′

t−1 −

xt

t

t

√α¯t−1√1 − α¯t √α¯t

2

α¯t−1 α¯t

γ γ − (γ − 1)¯αt−1

E 1 − α¯t−1 −

xt

+

2

ϵ ˜(xt,t)

2

ϵ ˜(xt,t)

.

(15)

(16)

Compared to the original energy formulation Eqa. 12, two additional coefficients appear: γ−(γ−1)¯α

γ−(γ−1)¯αt−1 and γ−(γ−γ1)¯α

. Since α¯t−1 and α¯t are very close, the first coefficient is approximately equal to 1. In the DDIM sampling formulation, α¯t is within the range [0, 1], which implies that the second coefficient falls within [1, γ]. As a result, after the SNR correction, the average latent energy increases. Therefore, SNR correction essentially serves as a mechanism for energy enhancement. In this sense, both energy rectification and SNR correction aim to increase the average latent energy. However, since our method allows for the flexible selection of hyperparameters, it can achieve superior performance.

t

t−1

#### 7.5. Applying RectifiedHR to Stable Diffusion 3

|Model:SD3<br><br>|CLIP-Score↑|DEQA-score↑|
|---|---|---|
|Direct-Inference<br><br>|0.275|3.311|
|RectifiedHR|0.289|3.621|

Table 4. The quantitative results of SD3.

To validate the effectiveness of our method on a transformer-based diffusion model, we apply it to stable-diffusion-3-medium using the diffusers library. As shown in Tab. 4, we provide additional quantitative results on SD3 (50 images, 2048×2048), and the test results mainly include CLIP-Score [17] and DEQA-Score [62].

#### 7.6. Ablation results on hyperparameters

In this section, we conduct ablation experiments on the hyperparameters in Eq.7 and Eq.9 of the main text using SDXL. The baseline hyperparameter settings follow those described in the Evaluation Setup section of the main text. We vary one hyperparameter at a time while keeping the others fixed at the two target resolutions to evaluate the impact of each parameter on performance, as defined in Eq.7 and Eq.9 of the main text. The evaluation procedure for FIDc, FIDr, ISc, and ISr follows the protocol outlined in Sec. 7.1.

In Eq.7 and Eq.9 of the main text, ωmin and Tmax are fixed and do not require ablation. The value of N in both equations is kept consistent. For the 2048 × 2048 resolution scene, with N set to 2, variations in MT and Mω do not significantly affect the results. Thus, only N, ωmax, and Tmin are ablated. The quantitative ablation results for the 2048 × 2048 resolution are shown in Fig. 11, Fig. 12, and Fig. 13. For the 4096 × 4096 resolution scene, N, ωmax, Tmin, MT, and Mω are ablated. The corresponding quantitative ablation results for the 4096 × 4096 resolution are presented in Fig. 14, Fig. 15, Fig. 16, Fig. 17, and Fig. 18. Based on these results, it can be concluded that the basic numerical settings used in this experiment represent the optimal solution.

In Eq.7 and Eq.9 of the main text, ωmin and Tmax are fixed and thus excluded from ablation. The value of N is kept consistent across both equations. For the 2048 × 2048 resolution setting, with N set to 2, variations in MT and Mω have minimal impact on performance. Therefore, only N, ωmax, and Tmin are subject to ablation. The corresponding quantitative ablation results are shown in Fig. 11, Fig. 12, and Fig. 13. For the 4096 × 4096 resolution setting, we ablate N, ωmax, Tmin, MT, and Mω. The corresponding results are presented in Fig. 14, Fig. 15, Fig.16, Fig.17, and Fig. 18. Based on these findings, we conclude that the default numerical settings used in our experiments yield the optimal performance.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Figure 11. The image illustrates the ablation study of ωmax in Eq.9 of the main text for the 2048 × 2048 resolution setting. The values of ωmax range over 20, 25, 30, 35, 40.

2048的wmax消融

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Figure 12. The image illustrates the ablation study of N in Eq.7 and Eq.9 of the main text for the 2048 × 2048 resolution setting. The values of N range over 2, 3, 4.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

2048的N消融

Figure 13. The image illustrates the ablation study of Tmin in Eq.7 of the main text for the 2048 × 2048 resolution setting. The values of Tmin range over 20, 25, 30, 35, 40.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

2048的T_min消融

Figure 14. The image illustrates the ablation study of ωmax in Eq.9 of the main text for the 4096 × 4096 resolution setting. The values of ωmax range over 30, 40, 50, 60, 70.

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Figure 15. The image illustrates the ablation study of Mω in Eq.9 of the main text for the 4096 × 4096 resolution setting. The values of Mω range over 0.5, 1, 2.

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

4096的wmax消融

Figure 16. The image illustrates the ablation study of MT in Eq.7 of the main text for the 4096 × 4096 resolution setting. The values of MT range over 0.5, 1, 2.

4096的Mcfg消融

4096的M_T消融

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Figure 17. The image illustrates the ablation study of N in Eq.7 and Eq.9 of the main text for the 4096 × 4096 resolution setting. The values of N range over 2, 3, 4.

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Figure 18. The image illustrates the ablation study of Tmin in Eq.7 of the main text for the 4096 × 4096 resolution setting. The values of Tmin range over 25, 30, 35, 40, 45.

4096的N消融

#### 7.7. Hyperparameter details and quantitative results for applying RectifiedHR to applications

The combination of RectifiedHR and WAN. RectifiedHR can be directly applied to video diffusion models such as WAN [58]. The officially supported maximum resolution for WAN 1.3B is 480 × 832 over 81 frames. Our goal is to generate videos at 960 × 1664 resolution using WAN 1.3B. The direct inference baseline refers to generating a 960 × 1664 resolution video directly using WAN 1.3B. In contrast, WAN+RectifiedHR refers to using RectifiedHR to generate the same-resolution video. The selected hyperparameters in Eq.7 and Eq.9 of the main text are: N = 2, ωmax = 10, ωmin = 5, Tmin = 30, Tmax = 50, MT = 1, and Mω = 1. Our quantitative experimental details follow [6] on 40 videos.

The combination of RectifiedHR and OIR. RectifiedHR can also be applied to image editing tasks. We employ SDXL as the base model and randomly select several high-resolution images from the OIR-Bench [61] dataset for qualitative comparison. Specifically, we compare two approaches: (1) direct single-object editing using OIR [61], and (2) OIR combined with RectifiedHR. While the OIR baseline directly edits high-resolution images, the combined method first downsamples the input to 1024 × 1024, performs editing via the OIR pipeline, and then applies RectifiedHR during the denoising phase to restore fine-grained image details. For the 2048 × 2048 resolution setting, the hyperparameters in Eq.7 and Eq.9 of the main text are: N = 2, ωmax = 30, ωmin = 5, Tmin = 40, Tmax = 50, MT = 1, and Mω = 1. For the 3072 × 3072 resolution setting, the hyperparameters are: N = 3, ωmax = 40, ωmin = 5, Tmin = 40, Tmax = 50, MT = 1, and Mω = 1.

4096的T_min消融

The combination of RectifiedHR and DreamBooth. RectifiedHR can be directly adapted to various customization methods, where it is seamlessly integrated into DreamBooth without modifying any of the training logic of DreamBooth [47]. The base model for the experiment is SD1.4, which supports a native resolution of 512 × 512 and a target resolution of 1536 × 1536. The hyperparameters selected in Eq.7 and Eq.9 of the main text are as follows: N is 3, ωmax is 30, ωmin is 5, Tmin is 40, Tmax is 50, MT is 1, and Mω is 1. Furthermore, as demonstrated in Tab. 5, we conduct a quantitative comparison between the RectifiedHR and direct inference, using the DreamBooth dataset for testing. The test metrics and process were fully aligned with the methodology in [47]. It can be observed that RectifiedHR outperforms direct inference in terms of quantitative metrics for high-resolution customization generation.

RectifiedHR can be directly adapted to various customization methods and is seamlessly integrated into DreamBooth [47] without modifying any part of its training logic. The base model used in this experiment is SD1.4, which natively supports a resolution of 512 × 512, with the target resolution set to 1536 × 1536. The selected hyperparameters in Eq.7 and Eq.9 of the main text are as follows: N = 3, ωmax = 30, ωmin = 5, Tmin = 40, Tmax = 50, MT = 1, and Mω = 1. Furthermore, as shown in Tab.5, we conduct a quantitative comparison between RectifiedHR and direct inference using the DreamBooth dataset for evaluation. The test metrics and protocol are fully aligned with the methodology described in [47]. The results demonstrate that RectifiedHR outperforms direct inference in terms of quantitative metrics for high-resolution customization generation.

|Direct Inference<br><br>|DINO ↑|CLIP-I ↑<br><br>|CLIP-T ↑|
|---|---|---|---|
|DreamBooth + RectifiedHR|0.625|0.761<br><br>|0.249|
|DreamBooth|0.400<br><br>|0.673|0.220|

Table 5. Quantitative comparison results between RectifiedHR and direct inference after DreamBooth training. The evaluation is conducted on a scene with a resolution of 1536 × 1536.

The combination of RectifiedHR and ControlNet. Our method can be seamlessly integrated with ControlNet [64] to operate directly during the inference stage, enabling image generation conditioned on various control signals while simultaneously enhancing its ability to produce high-resolution outputs. The base model used is SDXL. The selected hyperparameters in Eq.7 and Eq.9 of the main text are: N = 3, ωmax = 40, ωmin = 5, Tmin = 40, Tmax = 50, MT = 1, and Mω = 1.

[Figure 197]

Figure 19. Visualization of the average latent energy curve following energy rectification.

#### 7.8. Visualization of the energy rectification curve

To better visualize the average latent energy during the energy rectification process, we plot the corrected energy curves. We randomly select 100 prompts from LAION-5B for the experiments. As shown in Fig. 19, the blue line represents the energy curve at a resolution of 1024 × 1024. For the 2048 × 2048 resolution setting, we use the following hyperparameters: Tmin = 30, Tmax = 50, N = 2, ωmin = 5, ωmax = 30, MT = 1, and Mω = 1. The red line corresponds to our method with energy rectification for generating 2048 × 2048 resolution images, while the green line shows the result of our method without the energy rectification module. It can be observed that energy rectification effectively compensates for energy decay.

#### 7.9. Qualitative Results

As shown in Fig. 20, to clearly illustrate the differences between our method and existing baselines, we select a representative prompt for each of the three resolution scenarios and conduct qualitative comparisons against SDXL direct inference, AccDiffusion, DemoFusion, FouriScale, FreCas, HiDiffusion, and ScaleCrafter. AccDiffusion and DemoFusion tend to produce blurry details and lower visual quality, such as the peacock’s eyes and feathers in column b, and the bottle stoppers in column c. FouriScale and ScaleCrafter often generate deformed or blurred objects that fail to satisfy the prompt, such as feathers lacking peacock characteristics in column b, and a blurry bottle body missing the velvet element specified in the prompt in column c. HiDiffusion may introduce repetitive patterns, as seen in the duplicate heads in column b and the recurring motifs on the bottles in column c. FreCas can produce distorted details or fail to adhere to the prompt, such as the deformed and incorrect number of bottles in column c. In contrast, our method consistently achieves superior visual quality across all resolutions. In column a, our approach generates the clearest and most refined faces and is the only method that correctly captures the prompt’s description of the sun and moon intertwined. In column b, our peacock is the most detailed and visually accurate, with a color distribution and fine-grained features that closely align with the prompt’s reference to crystal eyes and delicate feather-like gears. In column c, our method demonstrates the highest fidelity in rendering the bottle stopper and floral patterns, and it uniquely preserves the white velvet background described in the prompt. These qualitative results highlight the effectiveness of our method in generating visually consistent, detailed, and prompt-faithful images across different resolution settings.

Resolution: 2048x2048

Resolution: 4096x4096 Resolution: 2048x4096

Prompt: A massive stained glass mural depicting a celestial sun and moon intertwined, glowing as sunlight passes through, casting colorful reflections on a polished marble floor, photorealistic with intricate glass textures, rich vibrant hues, and dramatic light dispersion.

Prompt: A highly detailed, antique brass clockwork automaton in the form of a majestic peacock, its gears and mechanisms visible as it struts across a polished mahogany table, photorealistic with intricate brass engravings, glowing crystal eyes, and delicate feather-like gears.

Prompt: An elegant glass perfume bottle with etched floral designs, golden stopper, and refracted light, on velvet, photorealistic with intricate reflections.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

DirectInferenceAccDiffusionDemoFusionFouriScaleFreCaSOursHiDiffusionScaleCrafter

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

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

(a) (b) (c)

Figure 20. Qualitative comparison across three different resolutions between our method and other training-free methods. The red box indicates an enlarged view of a local region within the high-resolution image.

#### 7.10. More Image Results

[Figure 246]

Figure 21. More image results

#### 7.11. More Video Results

[Figure 247]

Figure 22. More video results

