# arXiv:2503.18446v2[cs.CV]25Mar2025

#### Latent Space Super-Resolution for Higher-Resolution Image Generation with Diffusion Models

Jinho Jeong, Sangmin Han, Jinwoo Kim, Seon Joo Kim Yonsei University

{3587jjh, seonjookim}@yonsei.ac.kr

###### Pixelsmith LSRNA-Pixelsmith (Ours)

###### DemoFusion LSRNA-DemoFusion (Ours)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

| |
|---|

| |
|---|

[Figure 5]

[Figure 6]

1×

1×

| |
|---|

| |
|---|

|16× Resolution [4𝟎𝟗𝟔 × 4096]|
|---|

|16× Resolution<br><br>[4𝟎𝟗𝟔 × 4096]|
|---|

|16× Resolution<br><br>[4𝟎𝟗𝟔 × 4096]|
|---|

|16× Resolution<br><br>[4𝟎𝟗𝟔 × 4096]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Zoomed-in view Zoomed-in view Zoomed-in view Zoomed-in view

1507 sec (100%) 506 sec (34%)

505 sec (100%) 314 sec (62%)

Figure 1. Comparisons of 16× image generation with and without LSRNA framework. Our proposed LSRNA framework improves reference-based higher-resolution image generation, enhancing detail and sharpness beyond the native resolution of SDXL [39] (10242) while achieving faster generation speeds.

###### Abstract

In this paper, we propose LSRNA, a novel framework for higher-resolution (exceeding 1K) image generation using diffusion models by leveraging super-resolution directly in the latent space. Existing diffusion models struggle with scaling beyond their training resolutions, often leading to structural distortions or content repetition. Referencebased methods address the issues by upsampling a lowresolution reference to guide higher-resolution generation. However, they face significant challenges: upsampling in latent space often causes manifold deviation, which degrades output quality. On the other hand, upsampling in RGB space tends to produce overly smoothed out-

puts. To overcome these limitations, LSRNA combines Latent space Super-Resolution (LSR) for manifold alignment and Region-wise Noise Addition (RNA) to enhance high-frequency details. Our extensive experiments demonstrate that integrating LSRNA outperforms state-of-the-art reference-based methods across various resolutions and metrics, while showing the critical role of latent space upsampling in preserving detail and sharpness. The code is available at https://github.com/3587jjh/LSRNA.

###### 1. Introduction

Diffusion models have rapidly become powerful tools in generative modeling, starting from foundational models

[Figure 11]

[Figure 12]

[Figure 13]

| |
|---|

| |
|---|

| |
|---|

###### FinaloutputUpsampledReference

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

(a) Latent-Bicubic

(c) Latent-SR (Ours)

(b) RGB-Bicubic

- Figure 2. Comparison of DemoFusion with different upsampling strategies. All methods are directly upsampled to 16× resolution. (a) Latent space bicubic upsampling causes manifold deviation, degrading output quality. (b) RGB space bicubic upsampling produces outputs with reduced detail and sharpness. (c) Our learned latent-space upsampling aligns the manifold, resulting in sharp and detailed outputs. Best viewed ZOOMED-IN.

like DDPM [22] and DDIM [53]. Latent Diffusion Models (LDMs) [43] further revolutionized the field by reducing computational costs through operating in a lowerdimensional latent space. These advancements have been applied to various tasks for image processing such as generation [38, 41, 46], editing [3, 20, 27, 36], and superresolution [23, 47].

In parallel, with the rise of megapixel displays, demands for higher-resolution image generation (ranging from 2K to 8K) have grown drastically. However, existing diffusion models are limited to learning the distribution of fixed resolutions (e.g., 5122 for SD1.5), preventing them from generating images at unseen larger resolutions. Scaling diffusion models to larger resolutions presents several challenges. Training diffusion models from scratch [10, 15, 24, 57] or fine-tuning [7, 42, 63] for higher resolutions demands extensive data and computational resources. Meanwhile, directly generating larger images using pretrained models leads to content repetition and structural distortions [1, 19]. Alternatively, one can generate images at a pretrained resolution followed by super-resolution methods on RGB space. However, this approach tends to lack texture details and produce smoothed results [13, 19].

In response, recent studies have increasingly focused on training-free approaches that leverage pretrained diffusion

models. Among these, DemoFusion [13] stands out as a prominent method, laying the foundation for referencebased approaches [34, 50, 55]. By generating a reference latent (or image) at the original size and upsampling it to guide the higher-resolution generation process, these approaches have demonstrated robust performance in producing high-quality images. Nonetheless, what remains underexplored is the reference upsampling process, which makes it a key area for further investigation.

In this paper, we explore the significance of upsampled reference in reference-based image generation. Our first observation is that, as shown in Figure 2(a), the bicubicupsampled reference latent appears visually unappealing, exhibiting artifacts that suggest a deviation from the target resolution’s manifold. This deviation poses a considerable problem as it interferes with the subsequent generation process by providing suboptimal guidance, preventing the output from properly aligning with the desired manifold. Existing reference-based methods with latent space upsampling [13, 34] mitigate the issue by gradually increasing image resolution in stages (i.e., progressive upscaling). However, it leads to slower inference times and the potential accumulation of errors in the content.

Given these limitations, we explored RGB space upsampling as an alternative, inspired by reference-based RGB upsampling methods [50, 55]. This approach has the potential advantage of avoiding the artifacts observed in latent space upsampling, as RGB-based methods work directly in pixel space, where artifacts may be less pronounced. Specifically, we replace latent upsampling with RGB upsampling, where a reference latent is decoded into an RGB image, bicubic-upsampled, and then re-encoded back into the latent space. However, Figure 2(b) shows that the RGB space upsampling produces smooth and less detailed outputs compared to the original latent upsampling.

Based on these observations, we hypothesize that upsampling within the latent space plays a crucial role in preserving sharpness and detail essential for high-resolution image generation. However, the issue of manifold deviation caused by upsampling remains insufficiently addressed. To tackle this, we propose a novel approach: learning to map low-resolution latent representations onto the higher-resolution manifold. We introduce a lightweight Latent space Super-Resolution (LSR) module that operates directly within the latent space. Additionally, to facilitate the generation of high-frequency details in the subsequent generation process, we propose the Region-wise Noise Addition (RNA) module. Inspired by SDEdit [36], which improves image quality through uniform noise injection followed by denoising, RNA adaptively injects Gaussian noise into specific regions of the upsampled reference latent.

Combining the manifold alignment capability of LSR with the detail generation ability of RNA, our framework,

(a) Latent Upsampling

(b) RGB Upsampling (c) LSRNA (Ours)

|𝐳𝑇𝐿𝑅| | |
|---|---|---|
| | | |

| |𝐳0𝐿𝑅| |
|---|---|---|

|𝐳𝑇𝐿𝑅| | |
|---|---|---|
| | | |

|𝐳0𝐿𝑅| |
|---|---|

|𝐳𝑇𝐿𝑅| | |
|---|---|---|
| | | |

DM

DM

𝐳0𝐿𝑅

DM

U

Decode

| | |
|---|---|
|𝐳𝑇𝑀𝑅𝑎| |
| | |

| | |
|---|---|
|𝐳0𝑀𝑅| |
| | |

𝐱ො𝐿𝑅

DM

U

LSR

…

U

||𝐳𝑇𝐻𝑅𝑐|
|---|
|
|---|

||𝐳0𝐻𝑅|
|---|
|
|---|

||𝐳𝑇𝐻𝑅𝑏|
|---|
|
|---|

||𝐳0𝐻𝑅|
|---|
|
|---|

||𝐳𝑇𝐻𝑅𝑎|
|---|
|
|---|

||𝐳0𝐻𝑅|
|---|
|
|---|

DM

DM

DM

U Non-parametric Upsampler

- Figure 3. Framework Comparison. (a) Existing latent upsampling framework rely on progressive upsampling to address manifold deviation. (b) Existing RGB upsampling framework can directly upsample (optionally progressively), but produce smooth output. (c) Our framework enables latent upsampling without progressive upscaling with much fewer denoising steps (Tc < T) while producing detailed outputs (RNA omitted for simplicity). LR, MR, HR: low/mid/high resolution; DM: Diffusion Model.

LSRNA, is the first latent upsampling framework to eliminate the need for progressive upscaling as illustrated in Figure 3. Our high-quality guidance also allows for fewer subsequent denoising steps, enabling faster generation with minimal quality loss. We validate the effectiveness of LSRNA by incorporating it into the state-of-the-art latent upsampling-based method [13], pushing the boundaries of high-resolution image generation. Furthermore, we demonstrate that latent upsampling is crucial for preserving sharpness and detail by applying LSRNA to a very recent RGB upsampling-based model [55].

###### 2. Related Works

Latent Diffusion Models. Diffusion models, with foundations in DDPM [22], DDIM [53], and ADM [9], have become a dominant approach for image generation. Latent Diffusion Models (LDMs) [43] further advance the field by shifting the diffusion process into the latent space, using a pretrained autoencoder to compress images. This approach allows for faster computations and lower memory usage, supporting a wide range of applications [3, 16, 20, 37, 39, 44, 45, 59]. Notably, SD [43] and SDXL [39], both types of LDMs, have become widely adopted for their robustness and efficiency in generating images from text prompts. SD and SDXL are widely used for high-resolution image generation, laying the basis for the methods discussed in the following sections.

Higher-resolution Image Generation. While LDMs produce impressive results at their pretrained resolution, attempting to generate images beyond this scale often leads to repetitive patterns and structural inconsistencies, significantly lowering image quality. Various studies have explored ways for diffusion models to adapt to higher-

resolution images, either through training from scratch [10, 15, 24, 57] or fine-tuning [7, 42, 63] on larger-resolution datasets. However, these approaches still require substantial data collection and computational resources.

Consequently, numerous approaches [1, 13, 18, 19, 25, 34, 50, 55, 61] have leveraged pretrained models as alternatives without requiring additional training. ScaleCrafter [19], HiDiffusion [61], and FouriScale [25] use dilated convolutions in certain U-Net layers to tackle object repetition. Meanwhile, MultiDiffusion [1] independently denoises patches of the original resolution and aggregates them to produce a panoramic image. ElasticDiffusion [18] revisits classifier-free guidance, enhancing global coherence by separating global and local terms. DemoFusion [13] has emerged as a prominent method for its reference-based approach, generating a reference latent at the original resolution and upsampling it to guide the higher-resolution generation process, inspiring later studies [34, 50, 55].

While training-free methods effectively improve higherresolution image generation, the absence of model adaptation to higher resolutions prior to inference can result in suboptimal image quality. Self-Cascade [17] enables fast model adaptation with minimal parameters by utilizing learnable, time-aware feature upsampler modules to incorporate knowledge from a small set of newly acquired high-quality images. Our method achieves even faster adaptation, driven by a different motivation: mapping lowresolution latent onto the higher-resolution manifold.

###### 3. Preliminary

Latent Diffusion Model. In Latent Diffusion Model (LDM), an autoencoder is composed of an encoder E and a decoder D. Given an image x, the encoder E maps x to the latent space as z = E(x). The LDM then operates within this latent space, utilizing two processes, a forward process and a denoising process. In the forward process, Gaussian noise is incrementally added to z0(= z) across a fixed number of timesteps T, transforming it to approximate a Gaussian distribution N(0,I) as:

q(zt|zt−1) = N(zt; 1 − βtzt−1,βtI), (1)

where {βt}t=1...T is a predefined variance schedule. The denoising process begins with zT and gradually removes the added noise to recover the original latent z0. This is achieved through a learnable noise-prediction network [9, 11] parameterized by θ, described as:

pθ(zt−1|zt) = N(zt−1;µθ(zt,t),Σθ(zt,t)). (2)

After denoising, the decoder D transforms the restored latent back into an image xˆ.

Reference-based Image Generation. In reference-based approaches, an original-size reference is upsampled to guide the higher-resolution generation process. Additionally, this upsampling-then-generation process can be divided into smaller incremental steps, progressively increasing the resolution to enhance precision. For convenience, let LR denote the resolution of the current stage and HR denote the resolution of the next stage, considering the progressive upsampling procedure.

Existing reference-based methods belongs to either the latent upsampling framework or the RGB upsampling framework. To guide the subsequent generation process, the former employs the upsampled latent z′HR0 = Unp(zLR0 ) as guidance, where Unp represents a non-parametric upsampler (e.g., bicubic interpolation). In contrast, the latter first decodes zLR0 into an image xˆLR, which is then upsampled to xˆHR and used as guidance.

The denoising process for higher-resolution generation begins by injecting Gaussian noise into the upsampled guidance to initialize zHRT

, where Tinit ∈ [1,T] is the starting timestep. At each timestep t ∈ [1,Tinit], the denoised latent zHRt is updated with guidance, defined by:

init

zHRt ← Φt(zHRt ,gHR), (3)

where gHR represents the guidance (z′HR0 or xˆHR) specific to each framework, and Φt is an updating function that integrates the reference information. The specific denoising strategy for handling the larger latent size varies across reference-based methods (e.g., dividing it into overlapping patches [13]).

###### 4. Methodology

An overview of LSRNA framework is illustrated in Figure 4. LSRNA consists of Latent space Super-Resolution (LSR) module and Region-wise Noise Addition (RNA) module, providing better guidance. Given zLR0 (i.e., lowresolution reference latent), LSR maps the zLR0 into a higher-resolution manifold, resulting in gHR(= z′HR0 ). Then, RNA constructs region-adaptive Gaussian noise, which is added to the upsampled reference gHR, facilitating detail generation during the subsequent high-resolution generation process. The following sections provide a detailed explanation of each module.

###### 4.1. Latent Space Super-Resolution

Architecture. The LSR module consists of two main components: a backbone and an upsampler. The backbone takes the LR latent zLR0 ∈ Rh×w×C as input and extracts a feature map F ∈ Rh×w×C

′

. Here, any lightweight SR network [33, 62] can be used as backbone, modifying its input and output channels to the latent channel dimension C.

Decoding (to 𝐱ො𝑳𝑹)

Average Pooling

[Figure 20]

[Figure 21]

[Figure 22]

RNA

[Figure 23]

[Figure 24]

| |[Figure 25]|
|---|---|
| | |

[Figure 26]

LSR

|𝐳0𝐿𝑅|
|---|

[Figure 27]

[Figure 28]

|𝑬𝑟𝑒𝑠𝑖𝑧𝑒𝑑|
|---|

|𝐠𝐻𝑅(= 𝐳′0𝐻𝑅)|
|---|

[Figure 29]

|𝜖~𝒩(0,𝐼)|
|---|

Noise injection Guidance

|𝒕 = 0|
|---|

|𝒕 = 𝑇𝑖𝑛𝑖𝑡(< 𝑇)|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### … …

|𝐳𝟎𝐻𝑅|
|---|

|𝐳𝑇𝐻𝑅𝑖𝑛𝑖𝑡|
|---|

Figure 4. Overview of LSRNA. The proposed LSRNA enhances reference upsampling with Latent space Super-Resolution (LSR) and Region-wise Noise Addition (RNA). LSR directly maps the low-resolution reference latent onto the high-resolution manifold. RNA then injects region-adaptive noise into the mapped reference, guided by a canny edge map. RNA facilitates detail generation in the higher-resolution generation stage.

For the upsampler, we employ LIIF [8], an MLP-based upsampler that enables flexible upsampling to arbitrary resolutions with a single training. For a pixel q in the target HR latent gHR ∈ RH×W×C, let its spatial coordinate denoted as qcoord ∈ R2. The upsampler U takes the feature F and qcoord as input and predicts latent value for the pixel q as:

###### gHR(q) = U(F,qcoord;θU), (4)

where θU denotes the parameters of the upsampler. The entire gHR can be predicted by querying all pixels in gHR.

Training Setup. The LSR module takes an LR latent as input and outputs an HR latent. Ideally, obtaining LR-HR latent training pairs directly from the diffusion process would be advantageous. However, acquiring a ground-truth HR latent that both preserves the same content as the LR latent and aligns with the HR manifold is infeasible.

To address this, we use the real-world dataset [29] to obtain ground-truth HR RGB images. Specifically, we construct training pairs through two steps: (i) applying degradation (i.e., downsampling) to the HR images to generate LR RGB images, and (ii) separately encoding both the HR and LR RGB images using a pretrained encoder E. The design of steps (i) and (ii) mainly focuses on reducing the domain gap between training and inference, which is discussed in the appendix. To support multi-scale training, we introduce variability by randomly cropping HR images to various sizes and downscaling them by multiple factors during step (i). Our data preparation strategy produces a train-

ing dataset with a total of 4.7M LR-HR latent pairs. Using this setup, the LSR module is trained with L1 loss.

###### 4.2. Region-wise Noise Addition

In addition to LSR, we propose Region-wise Noise Addition (RNA) to further enhance finer details. The motivation for RNA arises from the observation that certain highfrequency regions in output images still have room for improvement. This may stem from the use of L1 training loss for the LSR, which tends to favor pixel-wise averaged solutions and often results in smoothed predictions [4, 12, 26]. We hypothesize that introducing controlled variations in the latent values according to region-wise property (e.g., frequency) can provide better guidance for generating finer details. To achieve this, we design RNA to adaptively add Gaussian noise to specific areas of the upsampled reference latent, focusing on detail-critical regions.

We adopt Canny edge detection [5] to identify highfrequency regions. It is preferred over other edge detection methods [14, 35] due to its ability to capture both weak and strong edges, as further discussed in the appendix. To obtain the edge map, we first decode the reference latent zLR0 ∈ Rh×w×C into an image xˆLR ∈ Rsh×sw×3, where s denotes the compression ratio of the autoencoder. Next, we apply Canny edge detection to xˆLR to obtain the edge map E. The edge map is then resized to match the target latent size using an average pooling, resulting in Eresized ∈ RH×W.

To flexibly adjust Gaussian noise, we normalize the edge map values using a linear transformation T to map the edge values to a predefined range [emin,emax]. The normalized edge map is then used to modulate Gaussian noise ϵ ∼ N(0,I). The resulting region-wise noise is added to the latent immediately after it is upsampled by the LSR module. This process can be expressed as:

gHR ← gHR + T (Eresized) · ϵ, (5) where gHR is the upsampled latent from the LSR module.

###### 5. Experiments

In this section, we present both quantitative and qualitative results, followed by ablation studies. Additional experiments and results, including the importance of latent space upsampling and the training details of the LSR module can be found in the appendix.

###### 5.1. Experiment Settings

We adopt several existing higher-resolution image generation methods, categorized by whether they use a reference. Non-reference-based methods include SDXL+BSRGAN [58], which generates a 1K resolution image using SDXL and upscales it with BSRGAN, and

SDXL (Direct) [39], which performs direct inference at the target resolution. Methods such as ScaleCrafter [19], FouriScale [25], and HiDiffusion [61] are also included, all of which utilize dilated convolutions for denoising. Reference-based methods include DemoFusion [13], the foundational reference-based approach, Pixelsmith [55], which use a slider mechanism for structural coherence, and Self-Cascade [17], which utilizes learnable time-aware upsample modules. To evaluate the effectiveness of our LSRNA framework, we integrate existing reference-based methods into our framework, resulting in two configurations: LSRNA-DemoFusion and LSRNA-Pixelsmith. Each configuration uses the denoising strategy of its respective model during the higher-resolution generation phase.

All methods are based on the SDXL, with inference and runtime measurements conducted on a single NVIDIA Tesla V100-SXM2 GPU. We set the denoising steps in the LSRNA framework to 30 DDIM steps, while existing methods use the standard 50, which is further discussed in the ablation study.

###### 5.2. Quantitative Results

Metrics. We employ four widely-used metrics: FID [21], KID [2], IS [48], and CLIP Score [40], to evaluate the quality and semantic alignment of generated images. FID and KID require resizing images to 2992 to match the input size of the Inception network [54]. However, the resizing can result in a loss of high-resolution details, potentially affecting the evaluation of high-resolution images. To address this limitation, inspired by Anyres-GAN [6], we also compute FID and KID on 50,000 randomly cropped 1K-resolution patches from both real and generated images. We term these patch-based metrics pFID and pKID, as they better capture the finer details present in high-resolution images.

Dataset Preparation. We sample a validation set of 400 images and a test set of 1,000 images from the OpenImages dataset [29], naming them OpenImages-Valid and OpenImages-Test. Captions for both the validation and test sets are generated using BLIP2 [32]. Since finer details in the generated patches may be inadvertently penalized when reference patches are derived from low-resolution ground-truth images, we filter both sets to include only high-resolution images (larger than 3K) to ensure reliable evaluation. We also note that although the training dataset for our LSR module is also sampled from OpenImages, we ensure that the training, validation, and test sets are completely independent, with no overlap in image IDs.

Comparison. As shown in Table 1, the proposed LSRNA framework meaningfully enhances the performance of both DemoFusion and Pixelsmith across resolutions and metrics. For DemoFusion, a latent upsampling-based method, integrating LSRNA eliminates the need for progressive upscaling and reduces denoising steps by providing high-quality

- Table 1. Quantitative comparison results on OpenImages-Test. The best and second-best performances are highlighted in red and blue, respectively. Methods above the dashed line are non-reference-based, while those below are reference-based. FouriScale is not measured above 2K due to out-of-memory on our V100 GPU.

Resolution Method FID (↓) KID (↓) pFID (↓) pKID (↓) IS (↑) CLIP (↑) Time (sec)

SDXL+BSRGAN [58] 84.32 0.0080 39.71 0.0090 30.11 0.303 16 SDXL (Direct) [39] 113.19 0.0222 64.20 0.0198 19.13 0.294 80

ScaleCrafter [19] 91.76 0.0103 45.60 0.0106 29.09 0.301 101 FouriScale [25] 104.30 0.0175 57.00 0.0167 23.90 0.300 134

HiDiffusion [61] 90.23 0.0106 43.76 0.0099 27.17 0.299 50 Self-Cascade [17] 83.50 0.0064 36.44 0.0070 31.56 0.305 90

2048×2048

DemoFusion [13] 85.02 0.0079 38.96 0.0087 32.54 0.302 205 LSRNA-DemoFusion (Ours) 83.58 0.0077 36.55 0.0069 31.74 0.303 115 Pixelsmith [55] 85.40 0.0091 39.51 0.0087 30.84 0.304 126

LSRNA-Pixelsmith (Ours) 83.90 0.0079 37.33 0.0074 30.75 0.302 86 SDXL+BSRGAN [58] 80.71 0.0058 39.81 0.0122 25.95 0.294 17

SDXL (Direct) [39] 134.57 0.0370 71.73 0.0230 12.57 0.276 245 ScaleCrafter [19] 93.32 0.0142 43.35 0.0130 23.12 0.288 419 HiDiffusion [61] 101.10 0.0177 51.66 0.0158 21.90 0.283 129 Self-Cascade [17] 78.34 0.0042 34.40 0.0076 25.93 0.294 210 DemoFusion [13] 83.63 0.0072 38.64 0.0082 27.74 0.294 648 LSRNA-DemoFusion (Ours) 80.53 0.0057 33.31 0.0064 27.17 0.297 223 Pixelsmith [55] 81.21 0.0069 40.86 0.0108 25.59 0.294 232

2048×4096

LSRNA-Pixelsmith (Ours) 78.47 0.0050 32.96 0.0062 28.04 0.295 157 SDXL+BSRGAN [58] 84.64 0.0081 37.04 0.0149 30.13 0.302 17

SDXL (Direct) [39] 217.88 0.0976 99.05 0.0468 9.15 0.270 786

ScaleCrafter [19] 110.49 0.0202 54.91 0.0196 21.80 0.293 1351

HiDiffusion [61] 128.28 0.0319 100.85 0.0564 19.62 0.280 240 Self-Cascade [17] 90.94 0.0106 43.91 0.0187 27.22 0.300 669 DemoFusion [13] 87.29 0.0089 32.89 0.0102 29.69 0.300 1507

4096×4096

LSRNA-DemoFusion (Ours) 85.03 0.0077 29.12 0.0085 31.50 0.304 506 Pixelsmith [55] 84.75 0.0086 32.34 0.0111 30.21 0.303 505

LSRNA-Pixelsmith (Ours) 84.19 0.0075 29.62 0.0090 31.74 0.302 313

guidance. These changes result in faster inference times and performance improvements in most cases, demonstrating LSRNA’s capability to optimize existing methods effectively. Furthermore, Pixelsmith, an RGB upsamplingbased method, also benefits from LSRNA integration. By shifting to latent upsampling-based guidance, Pixelsmith achieves improved performance across resolutions and metrics, while the reduced denoising depth introduced by LSRNA accelerates inference.

Compared to non-reference-based methods, referencebased methods achieve better performance, but at the cost of slower inference times. Non-reference-based methods, e.g., SDXL+BSRGAN and HiDiffusion, offer faster generation speeds, however, struggle to maintain semantic alignment and perceptual fidelity. Meanwhile, LSRNADemoFusion and LSRNA-Pixelsmith achieve state-of-the-

art performance across various metrics while maintaining fast generation times. These results highlight the efficacy and adaptability of the LSRNA framework.

###### 5.3. Qualitative Results

Figure 5 showcases the qualitative comparisons of three reference-based methods: DemoFusion, Pixelsmith, SelfCascade at 2K and 4K resolutions. DemoFusion, which utilizes latent upsampling, excels at generating fine details, as evidenced by the texture on the leaf at 2K and the intricate fur of the puppy at 4K. In contrast, Pixelsmith, an RGB upsampling-based method, struggles to preserve detail especially at 4K resolution. While it performs reasonably well at 2K, its reliance on RGB upsampling-based guidance results in smoother and less detailed outputs, as seen in the puppy’s fur at 4K. Self-Cascade also performs well

LSRNA-

LSRNA-

DemoFusion PixelSmith

Original (1K) Self-Cascade DemoFusion

PixelSmith

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

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

4096×40962048×2048

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

205 sec 115 sec 126 sec 86 sec

90 sec

Prompt: A bright red

apple on a smooth

white table, glistening skin, with a tiny stem and leaf.

LSRNA-

LSRNA-

Original (1K) Self-Cascade DemoFusion

DemoFusion PixelSmith

PixelSmith

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

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

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

1507 sec 506 sec 505 sec 313 sec

669 sec

Prompt: A top-down

view of a golden

retriever puppy curled up on a plain sage green surface, fluffy

fur radiating warmth.

Figure 5. Qualitative comparisons across reference-based methods at 2K and 4K resolutions.

|34<br><br>38<br><br>42<br><br>46<br><br>50<br><br>54<br><br>58<br><br>25 30 35 40 45 50<br><br>DemoFusion LSRNA-DemoFusion (w/o RNA) LSRNA-DemoFusion<br><br>|
|---|

138

at 2K resolution, generating coherent and visually appealing images. However, at 4K resolution, it produces structurally distorted outputs despite retaining sharpness and detail. Self-Cascade, as a latent upsampling-based method, preserves sharpness and fine details, however, struggles to scale effectively to higher resolutions due to its inability to address manifold deviations introduced during upsampling.

DemoFusion LSRNA-DemoFusion (w/o RNA) LSRNA-DemoFusion

136

134

pFID

FID

132

130

25 30 35 40 45 50

Denoising Steps Denoising Steps

Figure 6. Ablation study of denoising steps with DemoFusion.

###### 5.4. Ablation Studies

Integrating DemoFusion and Pixelsmith into LSRNA framework results in noticeable improvements in detail for both methods. LSRNA-DemoFusion enhances DemoFusion’s already strong capability to generate fine details, producing sharper textures and more refined outputs. Meanwhile, LSRNA-Pixelsmith shifts Pixelsmith’s guidance from RGB-based upsampling to latent-based upsampling. This addresses the smoothness and lack of detail inherent in RGB-based upsampling, resulting in richer and more intricate textures. These results underscore the effectiveness of latent upsampling in achieving both sharper details and improved textures. Additional experiments in the appendix further validate the advantages of latent upsampling over RGB upsampling, highlighting LSRNA’s impact on higher-resolution image generation.

Denoising steps and quality. Figure 6 plots how image quality varies with different denoising steps. As shown, both FID and pFID improve in the order DemoFusion → LSRNA-DemoFusion (w/o RNA) → LSRNADemoFusion, with our framework maintaining stable trends across denoising steps. A key reason for this stability is that our LSR module places the upsampled latent closer to the HR manifold, allowing fewer denoising steps to refine the image with minimal performance loss. In contrast, DemoFusion’s less aligned upsampled latent requires full noise injection (i.e., 50 steps) to align with the manifold. Although FID and pFID are similar across denoising steps for LSRNA, our inspection shows that steps below 30 produce unstable quality, leading us to adopt 30 steps.

|Region-adaptive Noise (𝑒𝑚𝑖𝑛 = 0)|
|---|

|Uniform Noise (𝑒𝑚𝑖𝑛 = 𝑒𝑚𝑎𝑥)|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Edge mapOriginal (1K)

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

| |
|---|

|𝑒𝑚𝑎𝑥 = 0.6|
|---|

|𝑒𝑚𝑎𝑥 = 1.8|
|---|

|𝑒𝑚𝑎𝑥 = 0.6|
|---|

|𝑒𝑚𝑎𝑥 = 1.8|
|---|

|𝑒𝑚𝑎𝑥 = 1.2|
|---|

|𝑒𝑚𝑎𝑥 = 1.2|
|---|

|𝑒𝑚𝑎𝑥 = 0.0|
|---|

|𝑒𝑚𝑎𝑥 = 0.0|
|---|

Figure 7. Qualitative comparison of Region-wise Noise Addition (RNA) and Uniform Noise Addition (UNA). RNA (left) effectively preserves low-frequency areas while enhancing high-frequency detail and texture. In contrast, UNA (right) introduce artifacts across various regions at higher noise levels.

- Table 2. LSR & RNA ablation on OpenImages-Valid (×9) with DemoFusion. The best results marked in bold.

FID (↓) KID (↓) pFID (↓) pKID (↓) Time (sec)

DemoFusion 131.95 0.0064 38.75 0.0075 660 LSRNA-DemoFusion (w/o RNA) 132.65 0.0065 37.10 0.0057 272 LSRNA-DemoFusion 132.01 0.0053 35.95 0.0057 272

Effectiveness of LSR & RNA. Table 2 compares DemoFusion and its ablations with LSR and RNA. Integrating LSR reduces inference time with minimal performance degradation, demonstrating that it effectively provides high-quality latent guidance by aligning low-resolution representations with the higher-resolution manifold. Adding RNA further refines the output by enhancing finer details in highfrequency regions, leading to improved performance especially on the patch-based metrics.

Importance of Region-Adaptiveness in RNA. As shown in Figure 7, Uniform Noise Addition (UNA) introduce artifacts across various regions, particularly at higher noise levels. By employing a Region-wise Noise Addition (RNA), we can effectively preserve low-frequency areas while facilitating high-frequency detail and texture.

Impact of RNA strength. Table 3 demonstrates the effect of the strength of RNA (i.e., emax) across various resolutions. The results indicate that emax = 1.2 yields the lowest pFID across all scales, suggesting that it effectively balances detail enhancement and content over-generation. Therefore, we select [emin,emax] = [0.0,1.2] for DemoFusion. A similar ablation study for Pixelsmith is presented in the appendix, where we adopt [emin,emax] = [0.4,0.8].

###### 6. Limitation

While RNA demonstrates robustness across different denoising steps (as seen in Figure 6), its optimal strength varies depending on the integrated method (e.g., DemoFusion or Pixelsmith) and the predefined noise schedule. This is because the guidance latent interacts with each method’s inherent denoising process during the second dif-

Table 3. Ablation study on RNA strength using OpenImagesValid with DemoFusion. The best results are marked in bold. emin is set to 0.

×4 (2048×2048) ×9 (3072×3072) ×16 (4096×4096) emax FID (↓) pFID (↓) emax FID (↓) pFID (↓) emax FID (↓) pFID (↓)

- 0.0 132.27 53.40 0.0 132.65 37.10 0.0 135.00 34.00

- 0.8 131.87 53.03 0.8 132.68 36.40 0.8 135.29 33.62

- 1.0 132.13 53.28 1.0 132.43 36.12 1.0 135.09 33.64

- 1.2 131.47 52.96 1.2 132.01 35.95 1.2 134.90 33.35 1.4 131.44 53.12 1.4 131.73 36.04 1.4 134.71 33.77

fusion stage. Moreover, although our approach learns to map onto the high-resolution manifold, the generation ability of LSRNA remains inherently limited by the capacity of the pretrained LDM.

###### 7. Conclusion

In this paper, we address the challenges in higher-resolution image generation with diffusion models by identifying manifold deviation in latent upsampling and insufficient texture detail in RGB upsampling. To tackle these issues, we introduce LSR, which aligns low-resolution latent representations with the higher-resolution manifold, and RNA, which adaptively enhances finer details in high-frequency regions. We then propose LSRNA, a novel framework that combines LSR and RNA. By incorporating LSRNA into both latent and RGB upsampling-based methods, we demonstrate its ability to outperform state-of-the-art methods across various resolutions and metrics. Our experimental results highlight the ability of LSRNA, pushing the boundaries of higher-resolution image generation.

###### Acknowledgement

This research was supported and funded by the Artificial Intelligence Graduate School Program under Grant 20200-01361 and by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) under Grant 20220-00124.

#### Latent Space Super-Resolution for Higher-Resolution Image Generation with Diffusion Models

##### Supplementary Material

###### A. Importance of Latent Space Upsampling

One of our key findings is that for reference-based higherresolution image generation methods [13, 17, 55], quality of output images differs significantly depending on whether the reference is upsampled in RGB space or latent space. We hypothesize that upsampling within the latent space plays a crucial role in preserving the sharpness and detail essential for higher-resolution image generation. In this section, we provide additional qualitative and quantitative experimental results to support our hypothesis.

###### A.1. Setting

We define several RGB upsampling variants of the existing models DemoFusion [13] and Pixelsmith [55] by modifying their reference upsampling strategies. First, we introduce DemoFusion-rgbBic for the DemoFusion model, where the reference is upsampled in RGB space using bicubic interpolation. For the Pixelsmith model, we define PixelsmithrgbLanc, which employs Lanczos interpolation [30] in RGB space for reference upsampling. Pixelsmith-rgbLanc corresponds to the original Pixelsmith model.

Building on these, we further define variants that perform super-resolution (SR) in RGB space using a separate SR network, namely DemoFusion-rgbSR and PixelsmithrgbSR. The SR network shares the same architecture and training settings as our LSR module (detailed in Section C.3), with the input and output channels is set to 3 to process RGB images.

In contrast to these variants, LSRNA-DemoFusion and LSRNA-Pixelsmith utilize latent space upsampling through our proposed LSRNA framework. While the original DemoFusion also performs bicubic upsampling in latent space and demonstrates strengths in preserving detail, we instead demonstrate the effectiveness of latent upsampling within the LSRNA framework.

###### A.2. Analysis

Our qualitative results, presented in Figures C and D for 16× resolution and Figures E and F for 64× resolution, demonstrate consistent trends when comparing RGB upsampling variants to latent upsampling using our LSRNA framework. Specifically, RGB upsampling methods, whether based on interpolation or super-resolution, produce smoother images that lack fine details. In contrast, latent upsampling yields sharper and more detailed results.

Our quantitative results in Table A further confirm these observations that latent upsampling approaches (i.e.,

Table A. RGB vs. Latent Space Upsampling on OpenImagesValid (×9). The best results marked in bold.

FID (↓) KID (↓) pFID (↓) pKID (↓)

DemoFusion-rgbBic 134.56 0.0084 37.44 0.0062 DemoFusion-rgbSR 134.55 0.0093 37.35 0.0061 LSRNA-DemoFusion 132.01 0.0053 35.95 0.0057 Pixelsmith-rgbLanc 134.31 0.0095 40.64 0.0084 Pixelsmith-rgbSR 134.34 0.0102 44.41 0.0110 LSRNA-Pixelsmith 132.17 0.0077 36.71 0.0057

LSRNA-DemoFusion and LSRNA-Pixelsmith) consistently outperform their RGB upsampling counterparts. The improvements are particularly evident in patch-based metrics like pFID and pKID, which focus on capturing finer details. These results underscore the critical role of latent space upsampling in enhancing local detail fidelity and textures.

We attribute these findings to the representational characteristics of the latent space. Unlike RGB space, the latent space encodes image features in a compressed form, capturing high-level information. Upsampling within this domain likely leverages these representations to better preserve fine details and sharpness. Conversely, RGB space upsampling is constrained by the raw pixel-level representation, which acts as a bottleneck and hinders the preservation of details and textures. Further exploration is needed to fully understand the underlying reasons behind these differences.

###### B. Experimental Details

###### B.1. Comparison

To ensure a fair comparison, all experiments across the main text and appendix are conducted with a consistent setup unless otherwise specified. This includes the use of a fixed random seed, tiled decoding, negative prompts derived from DemoFusion, a guidance scale of 7.5, xFormers [31] enabled with float16 precision, and FreeU [52] disabled. In addition, unnecessary visualization code like intermediate image reconstruction is disabled for runtime measurement.

We employ a DDIM [53] scheduler, with the η parameter set to 0 for reference-based methods, and η = 1 for non-reference-based methods, as η = 0 leads to noticeable degradation in quantitative performance for the latter. Existing methods use 50 DDIM steps for higher-resolution generation process, while LSRNA uses 30 steps.

###### B.2. Patch-Based Metrics

Conventional metrics like FID [21] and KID [2] involve resizing images to 2992, which can lead to a loss of highresolution details. To address this issue, inspired by AnyresGAN [6], we adopt patch-based metrics (pFID and pKID) that focus on local details and textures, which are critical for evaluating high-resolution image generation.

The patch-based metrics are computed by first cropping the original ground truth images to match the aspect ratio of the generated images, followed by resizing them to the same resolution using Lanczos interpolation. Next, 1K-sized patches are cropped from both the generated and ground truth images at 50,000 randomly selected locations. For a fair comparison, a fixed random seed is used to maintain consistency in crop locations both between generated and ground truth images and across different generation methods. These extracted patches are then used to calculate the FID and KID metrics, referred to as pFID and pKID.

###### C. LSR Training Details C.1. Data Preparation

To prepare LR-HR latent pairs for training the LSR module, we leverage the real-world dataset [29] to obtain groundtruth HR RGB images. We construct training pairs in two steps: (i) downsampling the HR RGB images to generate LR RGB images, and (ii) encoding the HR and LR RGB images independently using a pretrained encoder E.

Bridging the Domain Gap. To address the domain gap between training and inference, we simply adopt bicubic degradation over complex real-world degradations [56, 58] in step (i). This choice aligns with the inference scenario, where LR images (decoded from LR latents) typically exhibit minimal noise or artifacts. Bicubic degradation avoids the noise or artifacts introduced by real-world degradations while significantly reducing preprocessing time.

In step (ii), directly downsampling HR latents to create LR latents is avoided, as it can cause inconsistencies within the latent manifold. Instead, our approach ensures that LR and HR latents are encoded separately, preserving consistency within their respective manifolds.

Multi-Scale Preparation. To enable multi-scale training for the LSR module, we filter ground-truth HR RGB images with a minimum resolution of 1440 pixels in both height and width. For each HR image, a crop size is randomly selected between [1056,1440] in multiples of 96 (chosen to align with the downscaling factors and the encoder’s compression ratio of 8). The HR image is then divided into non-overlapping patches of the selected crop size, forming HR RGB patches. Each HR patch is subsequently encoded into the latent space.

To create LR counterparts, each HR RGB patch is downscaled by factors of ×2, ×3, and ×4. The resulting LR RGB

Table B. Quantitative comparison of image generation results by LSR training variants on OpenImages-Valid (×9). All training is conducted on a single NVIDIA Tesla V100-SXM2 GPU, using SwinIR [33] and RCAN [62] as backbones with LIIF [8] as the upsampler. Based on a balance of training efficiency and performance, we adopt the v1 configuration.

LSRNA-DemoFusion v1 (adpoted) v2 v3 v4

Params 1.29M 1.29M 1.29M 15.64M Backbone SwinIR (Light) SwinIR (Light) SwinIR (Light) RCAN

Initial learning rate 2 × 10−4 2 × 10−4 1 × 10−4 2 × 10−4 Batch size 32 32 16 32 Training iteration 200K 1000K 200K 200K

Training time 26h 129h 15h 26h FID (↓) 134.84 134.90 134.28 134.25 KID (↓) 0.0077 0.0077 0.0077 0.0074 pFID (↓) 33.47 33.35 33.75 34.34 pKID (↓) 0.0073 0.0072 0.0074 0.0076

patches are then encoded using the same encoder to generate LR latent representations. Our data preparation process results in a training dataset comprising a total of 4.7M LRHR latent pairs with diverse scale.

###### C.2. Batch Construction

Each LR-HR latent pair has varying sizes, necessitating alignment of the spatial dimensions between the input LR latent and target HR latent for batching. During dataloading, we further randomly crop the LR latents to a fixed size of 32×32 pixels. From the corresponding HR latents, 4096 pixels within the cropped region are randomly sampled to serve as the ground truth. Additionally, we avoid data augmentation techniques such as horizontal and vertical flips, as they can lead to deviations in the latent space manifold. Our batching strategy not only ensures efficient training but also enables the LSR to learn mappings from LR to HR latent representations across multiple scales.

###### C.3. Training

We adopt SwinIR [33] as the backbone for the LSR and LIIF [8] as the upsampler, modifying both input and output channel dimensions to match the latent space dimension of

###### 4. The optimizer used is Adam [28] with an initial learning rate of 2 × 10−4, scheduled with cosine annealing. Training is performed over 200K iterations with a batch size of

32. The loss function is defined by an L1 loss in the latent space. We leave the exploration of other loss functions (e.g., perceptual loss [60]) for future work.

Quantitative results of image generation by various LSR training settings are presented in Table B, while also demonstrating the efficiency of the LSR training. Although the LSR is trained on the paired dataset constructed with relatively sparse downscaling factors, it can generalize to arbitrary scaling factors during inference, enabled by our

Table C. Quantitative comparison of image generation results by Canny edge detection thresholds on OpenImages-Valid (×9) with DemoFusion. emax is set to 1.2 (default).

lower upper FID (↓) KID (↓) pFID (↓) pKID (↓)

0 255 132.01 0.0053 35.95 0.0057 30 180 132.18 0.0055 36.01 0.0057 50 200 132.54 0.0055 36.09 0.0057 60 150 132.19 0.0055 36.12 0.0057

Table D. Pixel-wise difference based on RNA strength. Differences are computed after applying RNA compared to no RNA. Histogram matching is applied before computing differences. The scene from the main Figure 7 is used.

emin emax Non-edge Edge Gap emin emax Non-edge Edge Gap

- 0 0.6 1.36 3.44 2.09 0.6 0.6 6.36 9.61 3.25

- 0 1.2 2.65 6.57 3.92 1.2 1.2 25.95 29.06 3.11 0 1.8 4.01 10.27 6.26 1.8 1.8 39.12 39.81 0.7

multi-scale training scheme and the generalization capability of LIIF. Our motivation for using LIIF upsampler instead of traditional fixed-scale upsampler [51] lies in its ability to handle arbitrary resolutions with a single LSR module trained once.

###### D. Edge Detection for RNA

RNA is designed to adaptively add Gaussian noise to specific areas of the upsampled reference latent, focusing on detail-critical regions (i.e., high-frequency regions). Our intuition behind RNA is that introducing irregularities in regions that would otherwise remain flat prompts the diffusion model to synthesize new details in those regions.

To identify these areas, we consider using edge detection algorithms. However, we find that common edge detectors such as Scharr [49], LoG [35], and Gabor [14], which primarily focus on precise object boundaries, tend to produce artifacts such as overgeneration around contours or jagged contours when used as the basis for RNA. We present qualitative results in Figure A using the Scharr edge map, which is known for effectively capturing weak edges. As shown, while strong edges (e.g., on the train’s window) are sparsely detected, weak edges (e.g., on the tree) appear with excessively low intensity. This results in artifacts or overenhanced details around strong edges when RNA is applied.

To address this, we adopt Canny edge detection [5], which allows us to prioritize weak edges by adjusting the lower and upper thresholds. By doing so, we can detect detailed regions rather than strictly connected edge lines, as demonstrated in Figure A. This region-based detection allows RNA to enhance local details effectively without introducing edge-based artifacts.

Table E. LSR & RNA ablation on OpenImages-Valid (×9). The best results marked in bold.

FID (↓) KID (↓) pFID (↓) pKID (↓) Time (sec)

DemoFusion 131.95 0.0064 38.75 0.0075 660 LSRNA-DemoFusion (w/o RNA) 132.65 0.0065 37.10 0.0057 272 LSRNA-DemoFusion 132.01 0.0053 35.95 0.0057 272 Pixelsmith 134.31 0.0095 40.64 0.0084 289 Pixelsmith-latentBic 142.23 0.0155 67.91 0.0275 291 LSRNA-Pixelsmith (w/o RNA) 137.71 0.0116 46.45 0.0112 181 LSRNA-Pixelsmith 132.17 0.0077 36.71 0.0057 182

###### E. Robustness of RNA

We demonstrate in Table C that generation performance is robust to variations in the Canny’s thresholds, as long as they are set to prioritize weak edges. In our implementation, we use a lower threshold of 0 and an upper threshold of 255. We further evaluate the robustness of RNA through additional experiments: Figure B indicates that RNA is quantitatively better than UNA (Uniform Noise Addition), while Table D shows that RNA indeed enhances details where the edge map is activated.

###### F. Additional Ablation Studies F.1. Effectiveness of LSR & RNA

We provide additional quantitative results to assess the impact of the LSR and RNA modules on both the DemoFusion and Pixelsmith models. For the original Pixelsmith, which performs upsampling in the RGB space unlike DemoFusion, we introduce an additional variant called PixelsmithlatentBic. This variant replaces the original RGB space upsampling with bicubic interpolation in the latent space.

The results are summarized in Table E. For DemoFusion, incorporating the LSR module enhances performance by providing high-quality latent guidance, improving image generation quality even with fewer denoising steps and without progressive upscaling. The addition of the RNA module further boosts performance by enriching finer details and textures in the generated images.

In case of Pixelsmith, replacing the original RGB upsampling with latent upsampling (i.e., Pixelsmith-latentBic) leads to significant performance degradation. However, applying the LSR module to perform super-resolution in the latent space leads to a noticeable improvement in performance. The RNA module further improves the results and ultimately surpasses the performance of the original model, demonstrating the adaptability and effectiveness of our LSR and RNA modules.

[Figure 74]

[Figure 75]

[Figure 76]

###### w/oRNAw/RNA

[Figure 77]

###### Canny edge map

Scharr edge map

###### Figure A. Qualitative results of RNA using Scharr edge map.

144

55

142

52

140

49

138

46

pFID

###### FID

136

43

134

40

Baseline (Region-wise)

132

37

Baseline (Region-wise) 34

130

0 0.2 0.4 0.6 0.8 Uniform Noise strength (𝒆𝒎𝒊𝒏 = 𝒆𝒎𝒂𝒙)

0 0.2 0.4 0.6 0.8

Uniform Noise strength (𝒆𝒎𝒊𝒏 = 𝒆𝒎𝒂𝒙)

- Figure B. Ablation study of UNA (Uniform Noise Addition) strength on OpenImages-Valid (×9) with DemoFusion. Dotted line shows our default RNA setting (emin = 0 and emax = 1.2).

Table F. Ablation Study on RNA strength with Pixelsmith on OpenImages-Valid (×9). The best results marked in bold.

emin emax FID (↓) pFID (↓) emin emax FID (↓) pFID (↓)

- 0.0 0.0 137.71 46.45 0.2 1.2 133.67 37.76

- 0.0 1.2 135.15 40.45 0.2 1.4 133.72 38.51 0.0 1.4 134.32 38.75 0.4 0.6 132.18 37.3 0.0 1.6 134.34 38.74 0.4 0.8 132.17 36.71 0.2 1.0 133.86 38.51 0.4 1.0 132.29 36.95

noise injection into the reference latent. The RNA strength determined from this validation process is consistently applied across all other experiments including those in the main text.

###### F.2. Impact of RNA Strength

Building on the RNA strength tuning results for LSRNADemoFusion presented in the main text, we further evaluate the impact of RNA strength on LSRNA-Pixelsmith, as shown in Table F. While LSRNA-DemoFusion achieves optimal performance with emin = 0 (and emax = 1.2), LSRNA-Pixelsmith performs best with emin = 0.4 and emax = 0.8. This difference in optimal RNA strength arises from the distinct roles played by the reference latent in the high-resolution generation process of each model. LSRNAPixelsmith likely requires a higher emin to ensure effective

[Figure 78]

[Figure 79]

### LSRNA-DemoFusionDemoFusion-rgbBicDemoFusion-rgbSR

| |
|---|

[Figure 80]

[Figure 81]

| |
|---|

[Figure 82]

[Figure 83]

| |
|---|

- Figure C. RGB vs. Latent Space Upsampling for DemoFusion on 16×. Prompt used is ”the sun is setting over the ocean on a cloudy day”. Best viewed ZOOMED-IN.

[Figure 84]

[Figure 85]

## Pixelsmith-rgbLancPixelsmith-rgbSRLSRNA-Pixelsmith

| |
|---|

[Figure 86]

[Figure 87]

| |
|---|

[Figure 88]

[Figure 89]

| |
|---|

- Figure D. RGB vs. Latent Space Upsampling for Pixelsmith on 16×. Prompt used is ”the sun is setting over the ocean on a cloudy day”. Best viewed ZOOMED-IN.

[Figure 90]

[Figure 91]

### LSRNA-DemoFusionDemoFusion-rgbBicDemoFusion-rgbSR

| |
|---|

[Figure 92]

[Figure 93]

| |
|---|

[Figure 94]

[Figure 95]

| |
|---|

- Figure E. RGB vs. Latent Space Upsampling for DemoFusion on 64×. Prompt used is ”A mysterious forest with tall, ancient trees and

[Figure 96]

[Figure 97]

### LSRNA-PixelsmithPixelsmith-rgbLancPixelsmith-rgbSR

| |
|---|

[Figure 98]

[Figure 99]

| |
|---|

[Figure 100]

[Figure 101]

| |
|---|

- Figure F. RGB vs. Latent Space Upsampling for Pixelsmith on 64×. Prompt used is ”A mysterious forest with tall, ancient trees and

###### References

- [1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2023. 2, 3
- [2] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 5, 2
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2, 3
- [4] Joan Bruna, Pablo Sprechmann, and Yann LeCun. Superresolution with deep convolutional sufficient statistics. arXiv preprint arXiv:1511.05666, 2015. 5
- [5] John Canny. A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 679–698, 1986. 5, 3
- [6] Lucy Chai, Michael Gharbi, Eli Shechtman, Phillip Isola, and Richard Zhang. Any-resolution training for highresolution image synthesis. In European Conference on Computer Vision, 2022. 5, 2
- [7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 2, 3
- [8] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8628–8638,

2021. 4, 2

- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021. 3
- [10] Zheng Ding, Mengqi Zhang, Jiajun Wu, and Zhuowen Tu. Patched denoising diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2023. 2, 3
- [11] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3
- [12] Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. Advances in Neural Information Processing Systems, 29,

2016. 5

- [13] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising highresolution image generation with no $$$. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6159–6168, 2024. 2, 3, 4, 5, 6, 1
- [14] Dennis Gabor. Theory of communication. part 1: The analysis of information. Journal of the Institution of Electrical Engineers-part III: radio and communication engineering, 93(26):429–441, 1946. 5, 3
- [15] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Joshua M Susskind, and Navdeep Jaitly. Matryoshka diffusion models. In Inter-

- national Conference on Learning Representations, 2023. 2, 3
- [16] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696–10706, 2022. 3
- [17] Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation. arXiv preprint arXiv:2402.10491, 2024. 3, 5, 6, 1
- [18] Moayed Haji-Ali, Guha Balakrishnan, and Vicente Ordonez. Elasticdiffusion: Training-free arbitrary size image generation through global-local content separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6603–6612, 2024. 3
- [19] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In International Conference on Learning Representations, 2023. 2, 3, 5, 6
- [20] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 3
- [21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017. 5, 2
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2, 3
- [23] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 2
- [24] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 2, 3
- [25] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. arXiv preprint arXiv:2403.12963, 2024. 3, 5, 6
- [26] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision, pages 694–711. Springer, 2016. 5
- [27] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 2
- [28] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 2

- [29] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. 4, 5, 2
- [30] Cornelius Lanczos. Evaluation of noisy data. Journal of the Society for Industrial and Applied Mathematics, Series B: Numerical Analysis, 1(1):76–85, 1964. 1
- [31] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, et al. xformers: A modular and hackable transformer modelling library, 2022. 1
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, pages 19730–

19742. PMLR, 2023. 5

- [33] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1833–1844,

2021. 4, 2

- [34] Zhihang Lin, Mingbao Lin, Meng Zhao, and Rongrong Ji. Accdiffusion: An accurate method for higher-resolution image generation. arXiv preprint arXiv:2407.10738, 2024. 2, 3
- [35] David Marr and Ellen Hildreth. Theory of edge detection. Proceedings of the Royal Society of London. Series B. Biological Sciences, 207(1167):187–217, 1980. 5, 3
- [36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [37] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 3
- [38] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 3, 5, 6
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 5

- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [42] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024. 2, 3
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2, 3
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3
- [45] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10,

2022. 3

- [46] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [47] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4713– 4726, 2022. 2
- [48] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems, 29, 2016. 5
- [49] Hanno Scharr. Optimal operators in digital image processing. 2000. 3
- [50] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering highresolution image generation via structural and fine-grained guidance. arXiv preprint arXiv:2406.16476, 2024. 2, 3
- [51] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1874–1883, 2016. 3
- [52] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4733–4743, 2024. 1
- [53] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 3, 1

- [54] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1–9,

2015. 5

- [55] Athanasios Tragakis, Marco Aversa, Chaitanya Kaul, Roderick Murray-Smith, and Daniele Faccio. Is one gpu enough? pushing image generation at higher-resolutions with foundation models. arXiv preprint arXiv:2406.07251, 2024. 2, 3, 5, 6, 1
- [56] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1905–1914,

2021. 2

- [57] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Yujun Lin, Zhekai Zhang, Muyang Li, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 2, 3
- [58] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4791– 4800, 2021. 5, 6, 2
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3
- [60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. 2
- [61] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Zhenyuan Chen, Yao Tang, Yuhao Chen, Wengang Cao, and Jiajun Liang. Hidiffusion: Unlocking high-resolution creativity and efficiency in low-resolution trained diffusion models. arXiv preprint arXiv:2311.17528, 2023. 3, 5, 6
- [62] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. In European Conference on Computer Vision, pages 286–301, 2018. 4, 2
- [63] Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, and Hang Xu. Any-sizediffusion: Toward efficient text-driven synthesis for any-size hd images. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7571–7578, 2024. 2, 3

