# arXiv:2504.08591v1[cs.CV]11Apr2025

### ZipIR: Latent Pyramid Diffusion Transformer for High-Resolution Image Restoration

Yongsheng Yu1,2*, Haitian Zheng2, Zhifei Zhang2, Jianming Zhang2, Yuqian Zhou2, Connelly Barnes2, Yuchen Liu2, Wei Xiong2, Zhe Lin2, Jiebo Luo1 1University of Rochester, 2Adobe Research

#### Abstract

Recent progress in generative models has significantly improved image restoration capabilities, particularly through powerful diffusion models that offer remarkable recovery of semantic details and local fidelity. However, deploying these models at ultra-high resolutions faces a critical tradeoff between quality and efficiency due to the computational demands of long-range attention mechanisms. To address this, we introduce ZipIR, a novel framework that enhances efficiency, scalability, and long-range modeling for high-res image restoration. ZipIR employs a highly compressed latent representation that compresses image 32×, effectively reducing the number of spatial tokens, and enabling the use of high-capacity models like the Diffusion Transformer (DiT). Toward this goal, we propose a Latent Pyramid VAE (LP-VAE) design that structures the latent space into subbands to ease diffusion training. Trained on full images up to 2K resolution, ZipIR surpasses existing diffusion-based methods, offering unmatched speed and quality in restoring high-resolution images from severely degraded inputs.

#### 1. Introduction

Recent advanced generative models, such as GANs [16] and diffusion models [19, 38], have dramatically improved image restoration (IR). These models leverage long-range context modeling [1, 44, 47], enhanced architectural designs [19, 30], and greater model capacity to effectively restore complex image structures from severely degraded or downsampled inputs. However, existing IR methods, often relying on UNet-based diffusion models [34, 38], are pretrained on an 8× compressed latent space. While being effective, they face efficiency challenges when restoring ultra high-resolution outputs, due to the quadratic computational demands associated with the number of spatial tokens.

Not only scaling up the resolution of models is challenging, but deploying such models for ultra high-resolution IR also presents significant challenges. At a high level,

*Work done during an internship at Adobe.

there seems to be a fundamental dilemma between quality and efficiency. On one hand, long-range attention modeling is crucial for both visual understanding [1, 21, 22, 35] and synthesis [5, 6, 9, 19], facilitating the recent success of the Diffusion Transformer (DiT)[31] in both image and video generation[17, 28]. On the other hand, such capacity comes with an extensive computational overhead with the number of spatial tokens, dramatically limiting the scalability of these methods for ultra high-resolution IR. As shown in Fig. 1, existing diffusion-based IR methods [50, 56] take approximately one minute to process a 2K image, and if tiled-based inference is employed, the runtime increases further. This limitation also impedes the exploration of more scalable models like DiT.

In this work, we introduce ZipIR, a novel framework designed to enhance model capacity, efficiency, and longrange modeling for high-quality, high-resolution diffusionbased image restoration. We start with building a highly compressed latent representation [13, 38] with a spatial downsampling factor of f = 32. Differing from existing methods [25, 46, 50, 56], our design effectively reduces the number of latent tokens and offers benefits: it enables the use of advanced models like DiT, facilitates training on the entire images rather than local patches for improved holistic modeling, and increases efficiency during both training and inference phases. As a result, ZipIR achieves up to 10 times faster inference than SeeSR [50] at 2K resolution and provides enhanced restoration for severely degraded inputs (downsampled by 20× or 16×).

However, designing the f32 latent space for image restoration introduces several challenges. First, a naively trained latent space is susceptible to minor perturbation and low-level degradation [29, 42], complicating the restoration process on latent space and leading to instability. Second, decoding from such a compressed latent code often distorts essential low-level details. To address these issues, we develop a novel Latent Pyramid VAE (LP-VAE) inspired by the Laplacian pyramid representation from image processing literature [3]. We train the latent space sequentially from lower to higher resolutions: early channels encode lower-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

ZipIR Output

Low Quality Input ZipIR Output

Low Quality Input ZipIR Output

Low Quality Input

(a) 20× super-resolution at 20482 px resolution.

(b) 16× super-resolution at 20482 px resolution.

(c) 8× restoration at 20482 px resolution.

| | | | | |Tra|inable P|arams| |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

SeeSR

Total Params

73.7s

SeeSR

SUPIR

53.0s 1.4×

SUPIR

Ours

6.9s 10.7×

Ours

0 500 1000 1500 2000 2500 3000 3500 4000

0 10 20 30 40 50 60 70 80

(e) Model scalability measured by the diffusion model parameters (in Millions).

(d) Inference time at 20482 px resolution (in seconds).

- Figure 1. Our ZipIR demonstrates a strong capacity in restoring severely degraded images, such as 20×, 16× downsampled or 8× degraded inputs to restore 20482 resolution output. Compared to different diffusion-based methods, ZipIR enjoys (d) an up to 10x running time advantage over SeeSR [50] while (e) maintaining a higher learning capacity for producing high-quality and ultra high-resolution images from severely degraded inputs.

resolution information, and subsequent channels capture residual details necessary for reconstructing high-resolution images. This sub-band decomposition effectively separates high-level image structures from low-level details. It ensures that the low-level degradation primarily affects the finer-level latent features, while the coarser-level codes remain consistent, thereby simplifying the learning process for the diffusion model.

Building upon LP-VAE, we finally designed a novel architecture based on DiT [31] to scale the capacity of the diffusion model for high-resolution IR. Trained on the entire image at up to 2K resolution and benefiting from the long-range modeling capacity, our method shows a stronger generation capacity, capable of upsampling across a wide range of scale factors (from 8× to 20×) directly at 2K resolution while achieving a significant speedup over previous diffusion-based image restoration methods including the recent SUPIR [56], without sacrificing the quality.

To summarize, we introduce ZipIR, a novel diffusionbased framework designed for high-quality and efficient high-resolution image restoration. Leveraging the highly

compact and structured LP-VAE latent space, along with a scaled-up diffusion model trained on full high-resolution images, ZipIR seamlessly reconstructs 2K images with both globally coherent structures and fine local fidelity from heavily degraded inputs, outperforming existing diffusionbased approaches in both efficiency and quality.

#### 2. Related Work

##### 2.1. High-Resolution Image Restoration

High-resolution image restoration (HR-IR) aims to enhance degraded images, often requiring models capable of generating fine-grained details at high fidelity. Early approaches to image restoration targeted specific degradations independently, such as super-resolution (SR) [12, 53], denoising [10, 60], and deblurring [51, 52], often relying on fixed assumptions about degradation patterns. While effective within constrained conditions, these methods lacked the flexibility to handle real-world complexities. Recently, blind IR methods have gained popularity [48, 56, 58], integrating multiple restoration tasks within unified frameworks that can generalize across diverse degradations, as exempli-

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Degradation-

aware Encoder

[Figure 12]

[Figure 13]

Ch: 64 ➔ 3

Ch: 12 ➔ 3

Ch: 3 ➔ 12

Ch: 3 ➔ 52

[Figure 14]

LP-VAE

| | |
|---|---|
|[Figure 15]<br><br>Diffusion Transformer<br><br>[Figure 16]<br><br>Gaussian Noise Map<br><br>Highly-compressed Latent<br><br>LQ Cond<br><br>Latent T<br><br>Text Pro<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]|ition Token<br><br>oken<br><br>mpt Token|
|[Figure 20]<br><br>[Figure 21]<br><br>|𝐷𝑐|
|---|
<br><br>[Figure 22]<br><br>[Figure 23]<br><br>𝐷𝑓<br><br>¼ resolution<br><br>full resolution<br><br>[Figure 24]<br><br>×|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>"Grey<br><br>realistic,<br><br>dramatic<br><br>portrait,<br><br>|

[Figure 28]

###### Fine Level

Encoder (f32)

[Figure 29]

Original image

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

LP-VAE

|𝐸𝑓|
|---|

Encoder (f32)

32×

LQ Condition

×T denoising

[Figure 34]

###### Coarse Level

[Figure 35]

[Figure 36]

[Figure 37]

¼ resolution

tokenization

|𝐸𝑐|
|---|

8×

8×

Crowned Crane,

yellow feathers,

lighting, close-up

photography style"

[Figure 38]

Cascaded Latent Maps

[Figure 39]

zfine

zmacro zcoarse

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

|𝐷𝑓∗|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

Detokenization

[Figure 52]

|Macro Level<br><br>1<br><br>32 resolution|
|---|

[Figure 53]

32

|[52,<br><br>𝐻 32<br><br>,<br><br>𝑊 32<br><br>]|
|---|

[Figure 54]

𝐻

𝑊

𝐻

𝑊

[3,

,

] [9,

,

]

32

32

32

32

(a) (b)

- Figure 2. Overview of ZipIR: (a) Latent Pyramid VAE (LP-VAE) compresses raw images into a 32× downsampled latent space through a pyramidal design that captures sub-band information across multiple scales (32×, 4×, original), ensuring a high compression rate while maintaining a well-structured latent space with improved representation invariance under degradation. (b) Our transformer diffusion model is trained and operates on the compressed latent space of the entire image, supporting resolutions up to 20482 pixels, enriching semantic understanding and synthesis of holistic structure. Furthermore, the pixel-space decoder Df∗ (Sec. 3.1) further enhances restoration quality.

fied by DiffBIR [25].

GAN-based methods have achieved realistic restoration results on real-world degraded images [48, 58]. However, these methods have limitations, particularly in preserving fine details under extreme scaling factors. Diffusion-based approaches like StableSR [46] and SUPIR [56], which leverage pre-trained models like Stable Diffusion [34, 38], have demonstrated notable improvements in restoration quality through multi-step processes, though these can be computationally intensive.

Scaling up restoration models has shown promise, especially with advancements in large-scale architectures [56]. Our proposed method leverages the scalability of diffusion transformers [31] to tackle the complex, high-dimensional nature of HR-IR.

##### 2.2. Efficient Diffusion Models

Diffusion models are powerful generative tools but face challenges with high computational demands and slow sampling speeds, limiting their practicality [19, 38]. Samplingefficient methods [27, 39, 40, 55] reduce the number of sampling steps, thereby shortening runtime, while model-based optimizations refine model architecture, using strategies like pruning [7, 14] and linear-complexity modules [15, 26] to create faster, more compact models. As diffusion models scale for high-resolution tasks, memory limitations and inference latency also become pressing issues. Our method addresses these with LP-VAE, a compact latent encoding approach that intensifies compression, reducing the spatial dimensions of feature maps and thus easing the computa-

tional load for high-resolution image restoration.

#### 3. Methodology

##### 3.1. Latent Pyramid VAE (LP-VAE)

To enable billion-scale DiT models to operate at 2K resolution and beyond, our priority is to optimize latent channel capacity and deepen the latent space mapping by adding more downsampling layers. This reduces the token count, lowering the quadratic complexity of DiT built on selfattention. As spatial compression increases, the spatial resolution of the latent representation shrinks, necessitating a corresponding increase in the latent channel count C to mitigate information loss. For an input image I ∈ R3×H×W, the encoder maps it to a latent code Z ∈ RC×

H f ×Wf . Despite the increased latent channels, raising the compression ratio still significantly impacts reconstruction quality, as evident from our ablation tests in Table 4.

Pyramid Cascade Encoders. Cascading networks have proven effective in other generative models [20, 32, 43], which allows different networks to independently learn representations at different resolutions, optimizing overall pipeline performance. Accordingly, our architecture employs a three-level pyramid VAE encoder to capture fine-level which encodes image high-frequency details, coarse-level features which encode lower-res structures, and macro-level semantics, with cascaded latent codes serving as a highly compressed image representation. The pyramid latent structure is shown on the left side of Fig. 2.

The fine and coarse-level encoders independently encode

representations from different resolutions. For f = 32, the fine-level encoder operates on the original image I without downsampling, producing a 52-channel latent encoding zfine ∈ R52×32H ×W32 . The coarse-level encoder captures lower-resolution features with an 4× downsampled input, I↓4 ∈ R3×H4 ×W4 , resulting in 9-channel latent encoding zcoarse ∈ R9×32H ×W32 . To incorporate macro-level semantics, we use downsampled I↓32 as a 3-channel image zmacro = I↓32σ−µ, where µ and σ are the mean and standard deviation calculated from the entire training dataset. Finally, the concatenated latent code across all levels, denoted by z = [zmacro;zcoarse;zfine], serves as the final highlycompressed 64-channel representation.

Progressive Training. We employ a progressive training approach. Training begins with the coarse-level encoder Ec, which requires a decoder Dc to reconstruct the 12-channel latent [zmacro;zcoarse] into pixel space. After completing this training stage, the coarse-level decoder Dc is discarded. Progressively, the next stage involves training the fine-level autoencoder to achieve full-level compression. The left side of Fig. 2 illustrates this stage, the fine-level decoder Df is trained to reconstruct from a 64-channel latent z back to pixel space, while the coarse-level encoder remains frozen.

For both training phases, we use a combination of discriminator loss and LPIPS loss as recommended in [13]. Based on empirical findings, we observed that attention layers did not significantly improve performance and added unnecessary overhead for both encoding and decoding. Therefore, our entire LP-VAE is designed as a pure convolutional network. Once training is complete, Ec and Ef serve as sub-networks of the LP-VAE Encoder, cascading three types of compressed mappings into a highly-compression representation z ∈ R64×32H ×W32 . Finally, a non-pyramidal decoder network Df decodes z to obtain the RGB image.

Pixel-aware Decoder-only Finetuning. Reconstructing from our highly compressed latents space achieves notable quality in high-resolution image reconstruction. However, without access to the full-resolution input, the decoder remains suboptimal for image processing applications, particularly restoration tasks that demand high pixel fidelity and detailed quality. Therefore, after obtaining the LPVAE with its encoder Ef, decoder Df, and the associated 64-channel latent space, we incorporate pixel-level details through skip connections to add spatial information during LP-VAE decoding, leading to a pixel-aware decoder Df∗.

To capture spatial features, we replicate an LP-VAE subencoder, Ef, initializing it with weights from the pretrained Ef. This degradation-aware feature extractor specifically handles degraded images, such as those blurred, noisy, or affected by JPEG artifacts. Additional residual layers are inserted between upsampling blocks in each layer of the LP-VAE decoder Df to pass multiscale spatial information from the degradation-aware feature extractor, effec-

tively capturing details from low-quality inputs. To train the pixel-aware decoder Df∗, we freeze the original Ef, but unlock the degradation-aware feature extractor and the entire decoder. With image reconstruction as the learning objective, this set-up enables the decoder to learn how to utilize low-quality images to complement the highly compressed latent code with pixel-level details. Note that decoder finetuning can occur after diffusion training to enhance quality, since the latent space is not altered by freezing Ef.

##### 3.2. Diffusion Transformer for Image Restoration

With the LP-VAE trained, we use its encoder to represent an input image I. Our model, a scaled-up diffusion transformer architecture of 3B parameters, G, is optimized for high-resolution image restoration. As illustrated in Fig. 2b, our framework uses two conditioning inputs: a low-quality image ILQ ∈ R3×H×W and a text embedding y, integrating both visual and semantic guidance for restoration.

Low-Quality Image Conditioning. Unlike traditional restoration methods relying on pixel-level low-quality (LQ) inputs, we resize ILQ to the target resolution, compress it using our LP-VAE Encoder, and concatenate it with a noisy latent zt. Despite sharing the same latent space, these latents differ significantly, necessitating separate, parameterindependent Patch Embedders for tokenization, which are arranged in parallel within the token sequence.

Text Semantic Guidance. Text embeddings aid in reconstructing degraded images by refining regions based on contextual cues [56]. We train G on paired text-image data, where the text prompt is a caption of the original image, encoded by T5 language model [36], and integrated via cross-attention layers within the Diffusion Transformer. To support classifier-free guidance, we randomly drop the text embedding with a probability of 0.05 during training. Additionally, we annotated low-quality images with negative prompts, enabling the model to produce clearer, more realistic outputs during inference with added negative text prompts. The effects of varying text prompt strengths are further analyzed in our supplementary material.

Learning HR-IR with DiT. We train our model on synthetic data degraded using methods similar to [48], resizing the generated low-quality images to match the highquality images for training. This approach aligns with our focus on high-resolution restoration while ensuring robustness to various degradation patterns. Leveraging the highlycompressed latent space, we only need to process a 64×64 latent map even for an original image resolution of 2048. This allows us to train DiT models on high-resolution images more effectively than existing methods [50, 56], facilitating the global-range semantic understand. During DiT training, we mix crop patches ranging from 5122 to 40962.

[Figure 55]

[Figure 56]

Input (128px)

| |
|---|

[Figure 57]

SeeSR (91.9s)

[Figure 58]

SUPIR (53s)

###### 16× SR, 2048px, 4Mpix

Ours (6.9s)

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Ours Input

SUPIR SeeSR SeeSR + Real-ESRGAN

- Figure 3. Our DiT-based ZipIR serves as a 16× upsampler, completing super-resolution from 1282 to 20482 in only 6.9 seconds. We compare it with SUPIR [56] and SeeSR [50], evaluating both their direct 2K inference and SeeSR’s default 512px output upscaled to 2K via Real-ESRGAN [48]. The gold dust on the face is real, not an artifact.
- 4. Experiments

##### 4.1. Experimental Settings

For training both LP-VAE and DiT, we use 300 million curated stock images paired with text as the data source. For the coarse-scale sub-model of LP-VAE, we use a batch size of 512 and run for 50K iterations. For the fine-scale submodel, we start with a batch size of 160 on 5122 cropped patches for 100K iterations, followed by a 1K patch adaptation with a batch size of 32 on 10242 cropped patches for 50K iterations. For DiT, we mix resolutions and aspect ratios to sample training images, similar to [34], using a batch size of 128 over 250K iterations, with the standard learning objective [31] guiding the training process. Inference for ZipIR employs the DDIM sampler with 25 denoising steps.

In recent benchmarking of IR, medium-resolution samples at 10242 or 5122 serve as HQ images, often from limited sets with fewer than 250 images, such as RealSet65 [57] and DrealSR [49]. These datasets are not ideally suited for distribution-based evaluation metrics like Fr´echet Inception Distance (FID) [18] due to their scale. To enable more robust benchmarking for IR at higher resolutions, we collected a comprehensive set of 3000 2K-resolution photos from Pexels [33], facilitating a thorough evaluation across tasks like mixture degradation restoration and superresolution across varying scale factors from 8× to 16×. In the Appendix, we present an analysis of the test set.

[Figure 64]

[Figure 65]

LQ Input (256px)

|[Figure 66]|
|---|

Blur: σ=1 Noise: σ=15 JPEG: q=65

[Figure 67]

SeeSR (73.7s)

[Figure 68]

SUPIR (53s)

8× IR, 2048px, 4Mpix

Ours (6.9s)

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Ours SUPIR SeeSR Input

Figure 4. Our DiT-based ZipIR functions as an 8× upsampler, enhancing images from 2562 to 20482 in just 6.9 seconds, while simultaneously restoring details through deblurring, denoising, and JPEG artifact removal. The input image is degraded with Gaussian blur (σ = 1), noise (σ = 15), and JPEG compression (q = 65). The reconstructed hand retains biological features without merging with the sack texture.

Quantitatively, we primarily use FID and Kernel Inception Distance (KID) [2] to measure output distribution realism. Given the HQ resolution of 2048, we additionally report Patch FID, inspired by [8, 37]. Text prompting for our model is provided via InternVL-26B [11], which generates consistent image caption. We continue to report PSNR, LPIPS [59] for benchmark purposes, despite the acknowledged misalignment of pixel-wise similarity metrics with human perception in evaluation [41, 56]. Noreference image quality metrics, such as MANIQA [54], are omitted from the main quantitative experiments in Table 1 because they downsample images to 2242, which may not adequately capture high-resolution restoration perfor-

mance. To provide a more comprehensive evaluation, we employ the real-world LQ dataset RealPhoto60 [56]. For a fair comparison—mirroring SUPIR [56], which downsamples its 1K results to 5122—we downsample our 1K results to 5122 (note that despite differences in model output resolutions, all methods use the same input images). This approach allows us to compare no-reference image quality metrics, including MANIQA [54], CLIP-IQA [45], and MUSIQA [23].

We conduct three main experiments to demonstrate our method’s performance and assess the contributions of each component. First, we benchmark traditional restoration on RealPhoto60 [56] and high-resolution image restoration on

Table 1. Quantitative comparison of image restoration methods under various degradation types. “Mixture degradation” denotes that the input image undergoes 8× downsampling, Gaussian blur with σ = 2, noise with σ = 40, and JPEG artifacts with p = 50.

Method PSNR↑ LPIPS↓ FID ↓ pFID ↓ KID×103 ↓

Real-ESRGAN [48] 25.55 0.5535 19.32 29.12 3.23 StableSR [46] 26.41 0.5683 24.40 54.22 8.47 DiffBIR [25] 25.92 0.4405 21.13 29.90 4.40 SeeSR [50] 25.22 0.4321 13.20 21.20 1.23 SUPIR [56] 23.85 0.4377 15.23 20.55 0.81 Ours 24.44 0.3978 9.89 18.17 0.63

SR(16)×

Real-ESRGAN [48] 27.47 0.4122 10.52 21.90 1.96 StableSR [46] 28.93 0.4238 5.17 19.51 0.29 DiffBIR [25] 28.03 0.3503 9.26 18.03 1.93 SeeSR [50] 27.77 0.3444 4.35 17.05 0.43 SUPIR [56] 26.35 0.3508 7.25 15.74 0.80 Ours 27.86 0.3374 3.24 13.95 0.02

SR(8)×

Real-ESRGAN [48] 22.24 0.5919 73.32 76.08 36.07 StableSR [46] 22.15 0.7593 123.87 172.62 73.25 DiffBIR [25] 22.45 0.5806 59.29 64.35 26.19 SeeSR [50] 22.06 0.6085 78.09 49.72 29.47 SUPIR [56] 21.65 0.6335 81.14 70.35 37.75 Ours 20.41 0.5791 35.10 31.08 11.23

Mix.Degradation

our newly proposed validation set of 3000 images (Section 4.2). Second, we analyze the inference efficiency and model parameters among a series of diffusion-based image restoration methods (Section 4.3). Third, an ablation study illustrates the effectiveness of each technical component by adding them incrementally (Section 4.4).

##### 4.2. Comparison with Existing Methods

Quantitative Evaluations. Table 1 illustrates the comparisons analysis of high-resolution (20482) image restoration across 16× to 8× super-resolution scale factor and a kind of mixture degradation by 8× downsampling, Gaussian blur σ = 2, noise σ = 40, JPEG artifacts p = 50. We evaluate the proposed ZipIR and recent advanced image restoration methods via PSNR, LPIPS [59], FID [18], Patch FID (pFID) and KID [2]. The method ZipIR demonstrates strong performance in high-resolution 16× and 8× scenarios. For the 16× super-resolution, ZipIR achieves notable LPIPS and FID improvements (0.3978 and 9.89, respectively), indicating superior perceptual quality and fidelity, while maintaining a competitive PSNR score. Its KID score (0.63×103) also emphasizes the reduced distributional discrepancy compared to other models. For the 8× superresolution task, ZipIR continues to show robustness, with the lowest FID (3.24) and best LPIPS (0.3374), affirming its quality consistency across different scales. Under mixed degradation, although the evaluation on pixel-wise similarity of ZipIR is lower, its LPIPS (0.5791), FID (35.10), Patch FID (31.08) and KID (11.23 × 103) reflect an ability to preserve perceptual quality and distributional consistency in challenging conditions.

Furthermore, in no-reference image quality assessment, as in Table 2, our method achieves the best or secondbest performance across all metrics. Specifically, while

- Table 2. Quantitative evaluation of real-world LQ images from RealPhoto60 [56] using no-reference image quality metrics.

Metrics BSRGAN Real-ESRGAN StableSR DiffBIR SeeSR SUPIR Ours CLIP-IQA 0.4119 0.5174 0.7654 0.6983 0.7721 0.8232 0.8154 MUSIQ 55.64 59.42 70.70 69.69 72.21 73.00 72.75 MANIQA 0.1585 0.2262 0.3035 0.2619 0.5596 0.4295 0.6681

- Table 3. Efficiency comparison of recent diffusion-based image restoration methods at 20482 resolution, including Neural Function Evaluations (NFEs), latency per denoising step, total inference time per image, and trainable parameters in diffusion models.

Model Type NFEs

Denoising Latency

(ms) Inf. Time #

Trainable Param.

SeeSR [50] UNet 50 1420 73.736s 0.5 B SUPIR [56] UNet 50 901 52.994s 1.2 B

Ours DiT 25 250 6.923s 3.1 B

CLIP-IQA [45] and MUSIQ [23] scores are slightly lower than those of SUPIR [56], they remain highly comparable (0.8154 vs. 0.8232 for CLIP-IQA [45] and 72.75 vs. 73.00 for MUSIQ [23]). Moreover, our method outperforms all others in MANIQA [54], highlighting the effectiveness of our approach in real-world LQ image restoration.

Qualitative Evaluations. Figures 3 and 4 present the visual comparison across existing the most advanced image restoration baselines. For facial portrait restoration at 16× SR, ZipIR produces sharper, more natural results than competing models, capturing intricate local details like the nose structure and piercings. In comparison, SUPIR [56] and SeeSR [50] fall short of ZipIR in preserving clarity, exhibiting noticeable distortions. We also evaluate an alternative approach where SeeSR processes its default resolution (512px), followed by upsampling with the efficient Real-ESRGAN [48]. While this results in slightly sharper outputs compared to SeeSR’s direct 2K inference, it still introduces artifacts, compromising overall visual fidelity. In the 8× IR task, as shown in Fig. 4, the LQ input suffers from blur σ = 1, noise σ = 15, and JPEG artifacts q = 65. SeeSR [50] introduces over-sharpening artifacts, while SUPIR [56] over-smooths textures, leading to unnatural hallucinations. In contrast, our ZipIR effectively restores fine-grained structures such as skin details, fabric texture, and citrus surfaces while minimizing artifacts.

- 4.3. Efficiency Analysis

Table 3 summarizes a comparison of ZipIR with several advanced baseline methods in terms of denoising latency, processing time per image, and trainable diffusion model parameters. ZipIR achieves a much lower denoising latency of 250 ms, outperforming all baselines, with the closest competitor, SUPIR, showing a latency of 901 ms. This efficiency is due to our proposed LP-VAE, which achieves a

[Figure 73]

[Figure 74]

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

Low-quality Input

w/o pixel-aware

w/ pixel-aware

w/ HR training w/o HR training w/ HR training w/o HR training

decoder

decoder

Figure 5. Effect of HR training for high-res restoration.

Figure 6. Comparison for w/ and w/o the pixel-aware decoder.

32× compression rate, significantly reducing the input token count in the diffusion transformer. Even at 2K resolution, each diffusion denoising step requires substantially less time. Consequently, ZipIR demonstrates exceptional efficiency in image processing, taking only 6.92 seconds per image–a significant improvement over other models like SeeSR (73.73 seconds) and SUPIR (52.99 seconds).

not guarantee robust improvement, as the latent code is susceptible to low-level perturbation and complicates the diffusion training. Next, by introducing our f32c64-LP-VAE, we achieve notable performance gains across all metrics, underscoring the impact of an optimized VAE design. Due to the lack of a pre-trained f32c64-SDVAE checkpoint, we trained it with the same settings as f32c64-LP-VAE for a fair comparison. Scaling up to a 3B DiT model, we incrementally add each of our proposed components. Notably, as we scale up the diffusion transformer, we observe significant boosts in perceptual quality and fidelity. Each addition, from Pyramid Cascade Encoders progressively enhances performance, with consistent reductions in FID and pFID alongside increases in PSNR.

Our LP-VAE introduces minimal overhead when encoding or decoding 2K images, highlighting its design efficiency. Despite ZipIR’s larger model size of 3.1 billion parameters, it performs inference 10.7 times faster than SeeSR, which has 1.4 billion parameters in the diffusion model. These results emphasize the superior efficiency and scalability of ZipIR for practical applications.

HR Training. Our high-compression LP-VAE encoder (f32) allows DiT models to be trained on global image above 2K resolution. We conduct a qualitative study to demonstrate the effect of the HR training technique. As illustrated in Fig. 5, HR training facilitates sharper and more accurate local details, such as the structure of accessories and textures of fur, compared to its counterpart.

##### 4.4. Effectiveness of Proposed Components

We quantitatively demonstrate the impact of our proposed components through an ablation study in Table 4. To facilitate the experiments, we sampled 100 images from a benchmark dataset of 3000 for the ablation study, performing 8× super-resolution from LQ 1282 px to HQ 10242 resolution. We report the metrics FID, Patch FID (pFID), and pixel-wise similarity PSNR. Starting with a baseline 0.68B DiT model paired with the original f8c4-SDVAE, we observe that switching directly to f32c64-SDVAE results in a decline in FID and pFID. This indicates that naively stacking networks or increasing channel dimensions in VAE does

Pixel-aware Decoder. The pixel-aware decoder is introduced as a complementary module to restore the spatial information of the input image at the pixel level. This proposed module enables ZipIR to capture spatial details directly from the original image, rather than relying solely on latent-level information. As shown in Fig. 6, the use of the pixel-aware decoder enhances clarity in textual and structural details, demonstrating its effectiveness.

- Table 4. Ablation on our model design, including latent space choices, model scaling, and different diffusion training schemes. For various VAEs, f represents the compression factor, while c denotes the dimensionality of the latent channels.

#### 5. Conclusion and Future Work

We present ZipIR, a framework that tackles efficiency, scalability, and quality in ultra-high-resolution image restoration. We developed the Latent Pyramid VAE (LP-VAE) to compress images into a structured latent space, enabling the training of the high-capacity Diffusion Transformer (DiT) on entire images. Tested on full images up to 2K resolution, ZipIR demonstrates a remarkable improvement over existing diffusion-based methods, highlighting the advantages of enhanced latent representation and scalable generative models for image restoration. We plan to explore even higher compression rates and larger capacity diffusion models for improved high-resolution image restoration.

Model FID ↓ pFID ↓ PSNR ↑ 0.68B DiT

+ f8c4-SDVAE 30.74 53.45 28.41 + f32c64-SDVAE 35.83 59.47 29.06 + f32c64-LP-VAE 28.14 51.73 28.50

3B DiT f32c64-LP-VAE

+ Pyramid Cascade Encoders 22.84 41.12 26.90 + 1K Patch Adapation 21.09 39.94 26.75 + Pixel-aware Decoder 20.95 38.73 27.94

+ HR Training 18.05 34.85 27.75

#### References

- [1] Dosovitskiy Alexey. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv: 2010.11929, 2020. 1
- [2] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In ICLR, 2018. 6, 7
- [3] Peter J Burt and Edward H Adelson. The laplacian pyramid as a compact image code. In Readings in computer vision, pages 671–679. Elsevier, 1987. 1
- [4] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, 2019. 12
- [5] Yuanhao Cai, Jing Lin, Xiaowan Hu, Haoqian Wang, Xin Yuan, Yulun Zhang, Radu Timofte, and Luc Van Gool. Mask-guided spectral-wise transformer for efficient hyperspectral image reconstruction. In CVPR, 2022. 1
- [6] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinexbased transformer for low-light image enhancement. In ICCV, 2023. 1
- [7] Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, and Shinkook Choi. Ld-pruner: Efficient pruning of latent diffusion models using task-agnostic insights. In CVPR, 2024. 3
- [8] Lucy Chai, Michael Gharbi, Eli Shechtman, Phillip Isola, and Richard Zhang. Any-resolution training for highresolution image synthesis. In ECCV, 2022. 6
- [9] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. Pixelsnail: An improved autoregressive generative model. In International conference on machine learning, pages 864–872. PMLR, 2018. 1
- [10] Yunjin Chen and Thomas Pock. Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration. TPAMI, 2016. 2
- [11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR,

2024. 6

- [12] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2):295–307, 2015. 2
- [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1, 4
- [14] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. In NeurIPS, 2023. 3
- [15] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Diffusion-rwkv: Scaling rwkvlike architectures for diffusion models. arXiv preprint arXiv:2404.04478, 2024. 3
- [16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and

- Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672–2680, 2014. 1
- [17] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. In European Conference on Computer Vision, pages 37–55. Springer, 2025. 1
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 5, 7
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arxiv:2006.11239,

2020. 1, 3

- [20] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 2022. 3
- [21] Hang Hua, Qing Liu, Lingzhi Zhang, Jing Shi, Zhifei Zhang, Yilin Wang, Jianming Zhang, and Jiebo Luo. Finecaption: Compositional image captioning focusing on wherever you want at any granularity. arXiv preprint arXiv:2411.15411,

2024. 1

- [22] Hang Hua, Yunlong Tang, Ziyun Zeng, Liangliang Cao, Zhengyuan Yang, Hangfeng He, Chenliang Xu, and Jiebo Luo. Mmcomposition: Revisiting the compositionality of pre-trained vision-language models. arXiv preprint arXiv:2410.09733, 2024. 1
- [23] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: multi-scale image quality transformer. In ICCV, 2021. 6, 7
- [24] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In ICCV, 2021. 12
- [25] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior, 2024. 1, 3, 7, 12
- [26] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024. 3
- [27] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusionbased text-to-image generation. In ICLR, 2024. 3
- [28] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. CoRR, abs/2401.03048, 2024. 1
- [29] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2574–2582, 2016. 1
- [30] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 1

- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 2, 3, 5
- [32] Pablo Pernias, Dominic Rampas, Mats L Richter, Christopher J Pal, and Marc Aubreville. W¨urstchen: An efficient architecture for large-scale text-to-image diffusion models. In ICLR, 2024. 3
- [33] Pexels. Pexels. https://www.pexels.com. 5
- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 1, 3, 5
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 13
- [36] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 4
- [37] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra high-resolution image synthesis to new peaks. In NeurIPS, 2024. 6
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 3, 12
- [39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 3
- [40] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 3
- [41] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In WACV, 2022. 6
- [42] C Szegedy. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. 1
- [43] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. In ICLR, 2024. 3
- [44] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 1
- [45] Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring CLIP for assessing the look and feel of images. In AAAI, 2023. 6, 7
- [46] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. IJCV, 2024. 1, 3, 7, 11
- [47] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7794–7803, 2018. 1

- [48] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In ICCVW, 2021. 2, 3, 4, 5, 7, 12
- [49] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In ECCV, 2020. 5, 12
- [50] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. In CVPR, 2024. 1, 2, 4, 5, 7, 12
- [51] Li Xu, Jimmy S Ren, Ce Liu, and Jiaya Jia. Deep convolutional neural network for image deconvolution. Advances in neural information processing systems, 27, 2014. 2
- [52] Li Xu, Xin Tao, and Jiaya Jia. Inverse kernels for fast spatial deconvolution. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 33–48. Springer, 2014. 2
- [53] Jianchao Yang, John Wright, Thomas Huang, and Yi Ma. Image super-resolution as sparse representation of raw image patches. In 2008 IEEE conference on computer vision and pattern recognition, pages 1–8. IEEE, 2008. 2
- [54] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In CVPR, 2022. 6, 7
- [55] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 3
- [56] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild. In CVPR, 2024. 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16
- [57] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. NeurIPS, 2023. 5
- [58] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In ICCV, 2021. 2, 3, 12
- [59] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6, 7
- [60] Yaping Zhao, Haitian Zheng, Zhongrui Wang, Jiebo Luo, and Edmund Y Lam. Manet: improving video denoising with a multi-alignment network. In ICIP, 2022. 2

### Supplementary – “ZipIR: Latent Pyramid Diffusion Transformer for High-Efficiency High-Resolution Image Restoration”

#### A. Additional Implementation Details

8000 ms

|Model<br><br>SeeS<br><br>SUPIR<br><br>ZipIR|R| | | |
|---|---|---|---|---|
| | | | | |
| |OOM OO|M OOM OO|M OOM OO|M|

Training Configuration. The optimization process employs AdamW with initial learning rate 5×10−5 (decaying to 5 × 10−6), weight decay 0.05, and betas (0.9,0.95). We implement DDPM loss with ϵ-prediction objective, coupled with linear noise schedule (βstart = 0.00085, βend = 0.012) and logit-normal time-step sampling for enhanced convergence.

6000 ms

Latency(ms)

4000 ms

2000 ms

Architecture Specifications. Our LP-VAE is constructed as a UNet-based architecture with 128 base channels. Specifically, the encoders Ec/Ef leverage residual blocks with channel multipliers [1,2,4,4]/[1,2,4,4,4,4] respectively, and these configurations are mirrored in the decoders Dc/Df. Each scale integrates two residual blocks powered by Swish activations. Our DiT adopts a 24-layer transformer, featuring a hidden dimension of 2048, 16 attention heads, and a strategy that incorporates adaptive layer normalization. Lastly, the f32c64-SDVAE used in ablation studies is modified from the LDM baseline, using zchannels = 64 and chmult = [1,2,4,4,4,4], thereby achieving a 32× spatial compression ratio.

0 ms

2048x2048 2944x2944 4096x4096 8192x8192

Figure 7. Denoising latency across ultra-high resolutions.

of-memory errors. Furthermore, ZipIR achieves lower inference latency at 40962 resolution compared to SeeSR at 20482.

#### C. Real-World HR Image Restoration

We collected randomly sampled image thumbnails from the internet, capturing diverse real-world degradations, and used them as LQ inputs for high-resolution image restoration experiments. As shown in Figures 16 and 17, our ZipIR effectively removes compression artifacts, reduces noise, and enhances fine details, producing high-resolution restored images with improved perceptual quality. These results demonstrate ZipIR’s robustness in handling in-thewild degradations across varying content and degradation

Pixel-aware Decoder. While previous work [46] adopts a similar approach by fine-tuning auxiliary networks for skip connections within the VAE decoder, we propose a joint optimization of the feature extractor and VAE decoder. As illustrated in Fig. 8, our pixel-aware decoder builds upon this design.

#### B. Ultra-High Image Super-Resolution

In the main text, we comprehensively present cases of 2K image restoration. To further evaluate whether our method can generalize effectively to ultra-high-resolution image super-resolution, such as 4K and even 8K, we conducted additional experiments. As shown in Figures 14-15, our ZipIR achieved a 16× upscale, enhancing 256-pixel and 512-pixel images to 4K and 8K resolutions, respectively. This demonstrates the capability of our method to handle ultra-high-resolution image super-resolution effectively.

[Figure 85]

[Figure 86]

[Figure 87]

Degradation-

aware feature extractor

[Figure 88]

[Figure 89]

Low-Quality Input

## 𝐷𝑓∗

[Figure 90]

DenoisedLatent

[Figure 91]

[Figure 92]

We have supplemented the evaluation of inference latency for ZipIR during diffusion denoising at each time step when synthesizing ultra-high-resolution images, including 4K and 8K resolutions. All efficiency evaluations were conducted on an A100-80G GPU. As shown in Fig. 7, our ZipIR demonstrates significant advantages across all resolutions. In contrast, the second-best method, SUPIR, fails to infer images larger than 29442 resolution due to out-

Pixel-aware Decoder

Figure 8. The pixel-aware decoder extracts high-res features from raw pixels with a degradation-aware feature extractor, enhancing the low-level fidelity of the decoded result during inference.

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

Ground Truth SD-VAE f32c64 Results LP-VAE Results Ground Truth SD-VAE f32c64 Results LP-VAE Results

Figure 9. Qualitative comparison of VAE image reconstruction. The first row visualizes the latent space representations, while the second row shows the reconstructed results.

types, highlighting its practical applicability in real-world scenarios.

#### D. Comparisons at a 4× Scale Factor

Our goal is not to achieve state-of-the-art performance on traditional settings (e.g., medium-resolution outputs at a 4× scale factor on previous benchmarks), but rather to enable efficient image restoration (IR) at ultra-high resolutions.

Nevertheless, for fair comparisons and a more comprehensive evaluation of our proposed method, we provide experiments on previous real-world image restoration benchmarks, including the DrealSR [49] and RealSR [49] datasets, for a 4× scale factor. Specifically, we restore LQ images of 2562 px to HQ resolutions of 10242 px. Following the experimental setup described in the main text, we adopt FID, patch FID (pFID), PSNR, SSIM, and LPIPS as evaluation metrics.

As shown in Table 5, even on traditional real-world im-

- Table 5. Comparative analysis of 4× image restoration methods on real-world low-quality datasets.

Dataset Method FID↓ pFID↓ PSNR↑ SSIM↑ LPIPS↓

BSRGAN [58] 78.17 95.84 25.35 0.7385 0.2809 Real-ESRGAN [48] 82.41 88.18 24.70 0.7384 0.2865 SwinIR [24] 78.43 84.59 24.86 0.7444 0.2732 SD Upscaler [38] 68.33 81.39 24.60 0.6644 0.3598

DiffBIR [25] 68.75 90.62 25.62 0.7149 0.3896 SeeSR [50] 68.96 79.42 25.06 0.7209 0.2874 SUPIR [56] 61.84 84.99 24.04 0.6673 0.3425

RealSR [4]

Ours 61.35 78.07 24.19 0.6999 0.2750

BSRGAN [58] 41.57 69.96 24.88 0.6969 0.2174 Real-ESRGAN [48] 45.10 71.51 24.05 0.6861 0.2273 SwinIR [24] 43.60 66.47 24.19 0.6905 0.2209 SD Upscaler [38] 33.96 68.83 23.91 0.6276 0.2992

DiffBIR [25] 36.33 62.19 25.03 0.6701 0.2175 SeeSR [50] 35.05 65.38 24.59 0.6701 0.2064 SUPIR [56] 38.93 69.42 23.97 0.6193 0.2884

DrealSR [49]

Ours 24.62 61.08 25.12 0.6843 0.2541

age restoration benchmarks, our ZipIR achieves the best FID (61.35 and 24.62) and pFID (78.07 and 61.08) across both datasets, demonstrating superior perceptual quality. On DrealSR, it also achieves the highest PSNR (25.12), reflecting exceptional clarity and detail. These results validate the robustness and effectiveness of ZipIR for real-world 4× image restoration.

#### E. VAE Reconstruction

To intuitively highlight the differences between our proposed LP-VAE and a straightforward deepening of SDVAE, we present a qualitative comparison of LP-VAE and SD-VAE f32c64 on 2048-resolution image reconstruction. Both LP-VAE and SD-VAE f32c64 perform 32× image compression and use a 64-channel dimensionality to represent the latent space. As shown in Figure 9, our proposed LP-VAE faithfully reconstructs the images, while SD-VAE f32c64 struggles to recover high-frequency details such as text and facial features.

#### F. Additional Qualitative Comparisons

In the main text, we provide only two qualitative comparisons. To offer a more comprehensive and intuitive evaluation of our method’s performance, we present additional qualitative results.

2K 3000-Sample Test Set. Following the experimental setup described in the main text, Figures 18-20 showcase 16× super-resolution. Our ZipIR faithfully reconstructs fine details, such as the nose ring in Figure 18, and textures, like the frog’s chin in Figure 20, while avoiding hallucination of unrealistic structures, as seen in the human chin in Figure 19.

Figures 21-22 illustrate 8× image restoration, where the inputs suffer from blur (σ=1), noise (σ=15), and JPEG ar-

- Table 6. Quantitative evaluation of configurations with and without text prompts, including varying CFG strengths under the ‘w/ text prompt‘ setting. The default configuration is ‘CFG strength = 3.5‘, and its values are equivalent to those for ‘w/ text prompt‘.

Metrics FID ↓ pFID ↓ PSNR ↑

Text Prompt CFG Strength

Default (3.5) 18.05 34.85 27.75 1.5 19.73 35.12 27.90 6.5 20.41 36.88 28.25

w/ text prompt

w/o text prompt – 19.97 35.40 28.04

tifacts (q=65). ZipIR effectively recovers realistic textures, such as the grass in Figure 21, and restores sharper details, exemplified by the chess piece in Figure 22.

512px RealPhoto60 [56]. As a qualitative counterpart to Table 2 in the main text, Figures 11 to 13 present visual comparisons on the RealPhoto60 test set. SeeSR and DiffBIR use their default training resolutions, while both SUPIR and our ZipIR process images at 1K resolution, following SUPIR’s approach, before downsampling to 512px.

#### G. Effectiveness of Text Prompt

Table 6 compares configurations with and without text prompts, highlighting the influence of CFG strength on FID, pFID, and PSNR. The default setting, with a CFG strength of 3.5, achieves a balanced performance, yielding an FID of 18.05, pFID of 34.85, and PSNR of 27.75. Reducing CFG strength to 1.5 slightly enhances PSNR but worsens FID and pFID, while increasing it to 5.5 maximizes PSNR (28.25) at the expense of perceptual fidelity.

Without text prompts, FID and pFID degrade, though PSNR (28.04) marginally surpasses the default. This underscores the role of text prompts in enhancing perceptual fidelity, while CFG strength tuning mediates the trade-off between fidelity and quality.

#### H. Test Set Construction

We curate a high-resolution test set of 3,000 images randomly sampled from Pexels, ensuring comprehensive coverage of real-world visual concepts. The dataset encompasses diverse semantic content, validated through CLIP [35] text similarity analysis, as illustrated in Fig. 10, spanning six major categories. Special attention is given to maintaining a balanced representation across visual domains while preserving natural image statistics.

[Figure 103]

Figure 10. Semantic category distribution of our 3,000-sample test dataset, classified using CLIP [35].

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

|[Figure 108]<br><br>Input (256px)|
|---|

ZipIR (Ours) 4096px, 16Mpix, 16× SR

- Figure 14. 4K Image Super-Resolution Result by ZipIR. The input is a 256px low-resolution image, and the output achieves a 4096px (16Mpix) resolution with 16× scaling.

[Figure 109]

|[Figure 110]<br><br>Input (512px)|
|---|

ZipIR (Ours) 8192px, 67Mpix, 16× SR

- Figure 15. 8K Image Super-Resolution Result by ZipIR. The input is a 512px low-resolution image, and the output achieves a 8192px (67Mpix) resolution with 16× scaling.

[Figure 111]

[Figure 112]

| |
|---|

Input

[Figure 113]

Input

[Figure 114]

ZipIR (Ours) Ours

[Figure 115]

[Figure 116]

| |
|---|

Input

[Figure 117]

Input

[Figure 118]

ZipIR (Ours) Ours

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

| |
|---|

| |
|---|

Input

Input

[Figure 123]

[Figure 124]

Input

Input

[Figure 125]

[Figure 126]

ZipIR (Ours) ZipIR (Ours)

Ours

Ours

- Figure 16. Real-world image restoration results by ZipIR. The inputs are low-resolution thumbnails sourced from the internet, featuring in-the-wild degradations.

|[Figure 127]<br><br>ZipIR (Ours)|
|---|

|[Figure 128]<br><br>Input|
|---|

|[Figure 129]<br><br>Input|
|---|

|[Figure 130]<br><br>ZipIR (Ours)|
|---|

[Figure 131]

[Figure 132]

Input ZipIR (Ours)

[Figure 133]

[Figure 134]

Input ZipIR (Ours)

- Figure 17. Real-world image restoration results by ZipIR. The inputs are low-resolution thumbnails sourced from the internet, featuring in-the-wild degradations.

[Figure 135]

[Figure 136]

Input (128px)

| |
|---|

| |
|---|

[Figure 137]

SeeSR (73.7s)

[Figure 138]

SUPIR (53s)

Ours (6.9s) 16× SR, 2048px, 4Mpix

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

Ours SUPIR SeeSR Real-ESRGAN Input

- Figure 18. Our DiT-based ZipIR achieves 16× super-resolution, enhancing images from 1282 to 20482 in just 6.9 seconds. Fine details, such as the nose ring and earrings, are faithfully restored without artifacts. Please zoom in for a detailed comparison.

[Figure 149]

[Figure 150]

Input (128px)

| |
|---|

[Figure 151]

SeeSR (73.7s)

[Figure 152]

SUPIR (53s)

Ours (6.9s) 16× SR, 2048px, 4Mpix

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Ours SUPIR SeeSR Input

- Figure 19. Our DiT-based ZipIR serves as a 16× upsampler, completing super-resolution from 1282 to 20482 in only 6.9 seconds. Fine details, such as the chin, are accurately restored without artifacts. Please zoom in for a detailed comparison.

[Figure 157]

[Figure 158]

Input (128px)

| |
|---|

| |
|---|

[Figure 159]

SeeSR (73.7s)

[Figure 160]

SUPIR (53s)

Ours (6.9s) 16× SR, 2048px, 4Mpix

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Ours SUPIR SeeSR Input

- Figure 20. Our DiT-based ZipIR serves as a 16× upsampler, completing super-resolution from 1282 to 20482 in only 6.9 seconds. The texture of the frog’s chin is faithfully reconstructed without blur. Please zoom in for a detailed comparison.

[Figure 169]

[Figure 170]

LQ Input (256px)

| |
|---|

| |
|---|

Blur: σ=1 Noise: σ=15 JPEG: q=65

[Figure 171]

SeeSR (73.7s)

[Figure 172]

SUPIR (53s)

8× IR, 2048px, 4Mpix

Ours (6.9s)

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Ours SUPIR SeeSR Input

- Figure 21. Our DiT-based ZipIR performs 8× super-resolution, enhancing images from 2562 to 20482 in just 6.9 seconds, while simultaneously restoring details through deblurring, denoising, and JPEG artifact removal. The input image is degraded with Gaussian blur (σ

= 1), noise (σ = 15), and JPEG compression (q = 65). Fine textures, such as the grass, are faithfully reconstructed. Zoom in for detailed comparison.

[Figure 181]

[Figure 182]

LQ Input (256px)

|[Figure 183]|
|---|

Blur: σ=1 Noise: σ=15 JPEG: q=65

[Figure 184]

SeeSR (73.7s)

[Figure 185]

SUPIR (53s)

8× IR, 2048px, 4Mpix

Ours (6.9s)

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Ours SUPIR SeeSR Input

- Figure 22. Our DiT-based ZipIR performs 8× super-resolution, enhancing images from 2562 to 20482 in just 6.9 seconds, while simultaneously restoring details through deblurring, denoising, and JPEG artifact removal. The input image is degraded with Gaussian blur (σ = 1), noise (σ = 15), and JPEG compression (q = 65). The generated chess piece base is sharp and free from blurring.

