# arXiv:2412.11815v2[cs.CV]4May2025

### ColorFlow: Retrieval-Augmented Image Sequence Colorization

Junhao Zhuang1,2∗ Xuan Ju2∗ Zhaoyang Zhang2† Yong Liu1,2 Shiyi Zhang1,2 Chun Yuan1‡ Ying Shan2‡

1Tsinghua University 2ARC Lab, Tencent PCG

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

[Figure 15]

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

[Figure 21]

[Figure 22]

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

## …

Reference Image Pool

Figure 1. ColorFlow is the first model designed for fine-grained ID preservation in image sequence colorization, utilizing contextual information. Given a reference image pool, ColorFlow accurately generates colors for various elements in black and white image sequences, including the hair color and attire of characters, ensuring color consistency with the reference images. [Best viewed in color with zoom-in].

#### Abstract

Automatic black-and-white image sequence colorization while preserving character and object identity (ID) is a complex task with significant market demand, such as in cartoon or comic series colorization. Despite advancements in visual colorization using large-scale generative models like diffusion models, challenges with controllability and identity consistency persist, making current solutions unsuitable for industrial application. To address this, we propose ColorFlow, a three-stage diffusion-based framework tailored for image sequence colorization in industrial applications. Unlike existing methods that require per-ID finetuning or explicit ID embedding extraction, we propose a novel robust and generalizable Retrieval Augmented Colorization pipeline for colorizing images with relevant color references. Our pipeline also features a dual-branch design: one branch for color identity extraction and the other

∗ Equal Contribution. † Project lead. ‡ Corresponding authors.

for colorization, leveraging the strengths of diffusion models. We utilize the self-attention mechanism in diffusion models for strong in-context learning and color identity matching. To evaluate our model, we introduce ColorFlowBench, a comprehensive benchmark for reference-based colorization. Results show that ColorFlow outperforms existing models across multiple metrics, setting a new standard in sequential image colorization and potentially benefiting the art industry. Code and model will be released.

#### 1. Introduction

Diffusion models have made substantial progress in generation, achieving state-of-the-art results in controllable image generation, including image inpainting[29, 84], image colorization[65, 76], and image editing [6]. This progress has sparked the growth of numerous downstream tasks. However, there has been limited attention to a problem where diffusion-based generation could significantly reduce

labor costs: reference-based image sequence colorization, which can be used in manga creation, animation production, and black-and-white film colorization.

Recently, with the unprecedented image generation capabilities of diffusion models, there is growing interest in colorization using diffusion models[8, 76, 80]. However, most efforts[39, 40, 65, 76, 80] only consider basic textto-image settings without reference to color, which is far from practical application. While some research[8, 37, 43] explores reference-based colorization, three key challenges remain: 1) selecting suitable reference image sets is complex and time-consuming, 2) models are limited to singleimage references, restricting multi-object representation, and 3) latent-diffusion[51] frameworks often produce structural distortions due to VAE compression, affecting texture and text details.

To tackle these challenges, we introduce ColorFlow , a novel three-stage framework for multi-reference image sequence colorization that preserves multiple color IDs. It consists of: a) Retrieval-Augmented Pipeline (RAP): Automatically matches patches from a black-and-white image to a reference pool, providing accurate color IDs with enhanced usability and eliminating the need for laborintensive reference image selection. b) In-context Colorization Pipeline (ICP): Employs a convolutional branch and a LoRA-fine-tuned diffusion model to integrate reference color information, leveraging contextual learning for precise multi-ID matching without requiring supervised ID correspondence training. c) Guided Super-Resolution Pipeline (GSRP): Minimizes output distortions by fusing high-quality black-and-white features with low-quality colored image features in the VAE encoder, utilizing residual connections to achieve high-fidelity color outputs.

To ensure a comprehensive evaluation of ColorFlow , we construct ColorFlow-Bench, a benchmark comprising 30 manga chapters, each containing 50 black-and-white images and 40 reference images. Experimental results indicate that ColorFlow achieves state-of-the-art performance across five metrics in both pixel-wise and image-wise evaluations. Compared to existing works, ColorFlow excels in preserving finer-grained color identities within image sequences, leading to significant enhancements in image quality. Our contributions are summarized as follows:

- • We introduce ColorFlow , a novel three-stage framework that effectively facilitates fine-grained color ID from retrieved multi-references through contextual learning, enabling high-quality colorization of image sequences.
- • We establish ColorFlow-Bench , a comprehensive benchmark specifically designed for multi-reference-based image sequence colorization.
- • Extensive evaluations demonstrate that our method surpasses existing approaches in both perceptual metrics and subjective user studies. Our model achieves over 37%

reduction in FID metrics compared to state-of-the-art colorization models. Additionally, our proposed model ranks first in user study scores for aesthetic quality, similarity to the reference, and sequential consistency.

#### 2. Related Work

Image Colorization [21, 78] aims to transform grayscale images (e.g., manga [49], line art [31], sketches [80], and grayscale natural images [76]) into their colored counterparts. To enhance controllability, various conditions are used to imply color information, including scribbles [14, 17, 20, 42, 49, 54, 58, 75, 78, 79, 82], reference images [1, 5, 8, 13, 19, 21, 23, 33–37, 44, 57, 60, 67– 72, 74, 77, 81, 86], palettes [4, 9, 59, 62, 68, 70], and text [4, 7, 10–12, 31, 47, 64, 65, 76, 80, 85]. Specifically, scribbles provide simple and freehand color strokes as color pattern hints. The Two-stage Sketch Colorization [78] employs a two-stage CNN-based framework that first applies strokes of color over the canvas, then corrects color inaccuracies and refines details. Reference image-based colorization transfers colors from a reference image that contains similar objects, scenes, or textures. ScreenVAE [71] and Comicolorization [21] compress color information from the reference image into a latent space, then inject the latent representation into the base colorization network. Palettebased models [9, 59] use the palette as a stylistic guide to inspire the overall color theme of the image. With the emergence of diffusion models [26, 56], text has become one of the most significant forms of guidance for image generation, and is thus widely used in image colorization. Text guidance uses a text prompt describing the desired color theme, object colors, or overall mood. ControlNet [80] adds additional trainable modules to pre-trained text-to-image diffusion models [51] and leverages the native text-to-image capabilities of diffusion models for colorization.

However, whether using palettes, text, or latent representations, these methods can only provide a rough color style and cannot guarantee the accurate color preservation of instances in black-and-white image. In contrast, ColorFlow achieves instance-level color preservation across frames in image sequences by introducing a retrieval-augmented pipeline and a context feature matching mechanism.

Image-to-image translation aims to establish a mapping from a source domain to a target domain (e.g., sketchto-image [80, 83], pose-to-image [28, 41], image inpainting [29, 84], and image editing [6, 30]). Recent advancements in diffusion models [15, 26, 52, 56] have made them dominant in this task. Approaches are mainly categorized into inference-based [3, 24] and training-based paradigms [29, 80]. Inference-based methods often use a dual-branch structure [30], where a source branch pre-

serves essential content and a target branch maps images with guidance. These branches interact through attention or latent feature integration, but often suffer from insufficient control. Training-based methods [48, 80, 83] are popular for their high quality and precise control. Stable Diffusion [52] adds depth control by directly concatenating control conditions with noisy input and fine-tuning the model end-to-end. ControlNet [80] uses a dual-branch design to add control conditions to a frozen pretrained diffusion model, enabling plug-and-play control while preserving high image generation quality.

Notably, none of these approaches specifically address identity preservation across frames in sequential image translation tasks, which limits their applicability in practical industrial scenarios involving sequential images. In contrast, ColorFlow is designed to tackle this limitation, providing robust instance identity preservation in image sequence colorization tasks across frames.

ID-Preservation is a trending topic in the field of image generation. Previous approaches can be classified into two primary categories: the first involves fine-tuning generative models to enable them to memorize one or more predefined concepts [22, 32, 53]; the second employs plug-andplay modules that have been trained on large-scale datasets, allowing the model to control the generation of desired concepts using given image content during the inference stage [38, 61, 73]. Generally, prior methods focus on a limited set of predefined concepts.

In contrast, we propose ColorFlow, which provides a robust and automated three-stage framework for sequential image colorization. ColorFlow effectively addresses the challenges of handling the dynamic and diverse characters, objects, and backgrounds present in comic sequences, making it well-suited for industrial applications.

#### 3. Method

Our aim is to colorize black-and-white images using colored images as references, ensuring consistency in characters, objects, and backgrounds throughout the image sequence. As illustrated in Figure 2, our framework consists of three main components: the Retrieval-Augmented Pipeline, the In-context Colorization Pipeline, and the Guided Super-Resolution Pipeline.

##### 3.1. Retrieval-Augmented Pipeline

The Retrieval-Augmented Pipeline (RAP) is designed to identify and extract relevant colored references to guide the colorization process. To accomplish this, we first divide the input black-and-white image into four overlapping patches: top-left, top-right, bottom-left, and bottom-right. Each patch covers three-quarters of the original image’s dimensions to ensure that important details are retained. For

each colored reference image, we create five patches: the same four overlapping patches and the complete image, providing a comprehensive set of reference data.

Next, we employ a pre-trained CLIP image encoder to generate image embeddings Ebw for the patches of the input image and Eref for the reference patches. These embeddings are defined as follows:

###### Ebw = fCLIP(Pbw) and Eref = fCLIP(Pref), (1)

where Pbw represents the black-and-white patch and Pref denotes the colored reference patch.

For each of the four patches from the input image, we compute the cosine similarity S between its embedding and the embeddings of the reference patches:

a · b ∥a∥ · ∥b∥

. (2)

S(a,b) =

where a and b are the embeddings of the query and reference patches, respectively. We define the top three similar patches for each query patch as follows:

Top3(Ebw(i)) = {Eref(j1),Eref(j2),Eref(j3) | jk ∈ arg max

S(Ebw(i),Eref(k)), k = 1,2,3},

k

(3) for i ∈ {0,1,2,3}, where Ebw(i) denotes the embedding of the i-th query patch and Eref(k) denotes the embeddings of the corresponding reference patches.

After identifying the top three similar patches for each query region, we combine these selected patches into a unified output image. The patches corresponding to the top-left, top-right, bottom-left, and bottom-right regions are stitched together to create the composite image Cbw, as illustrated in Figure 2. This spatial arrangement ensures the accurate placement of retrieved color information, enhancing the contextual relevance of the colorization process. In addition, we construct ( Ccolor ) by similarly stitching together the original colored versions corresponding to the black-and-white image patches. This forms a data pair with ( Cbw ) for subsequent colorization training. By effectively gathering the most contextually relevant color information, the Retrieval-Augmented Pipeline sets the stage for the next stages of our framework, ensuring that the generated colors are harmonious and consistent with reference images.

##### 3.2. In-context Colorization Pipeline

The In-context Colorization Pipeline is a fundamental component of our framework, designed to convert black-andwhite images into full-color versions by utilizing contextual information from retrieved patches. We introduce an auxiliary branch called the Colorization Guider, which aids in incorporating conditional information into the model. This

Retrieval-Augmented Pipeline

Crop Image Guided Super-Resolution Pipeline

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

[Figure 40]

[Figure 41]

Query

[Figure 42]

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

Reference Image Pool

[Figure 47]

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

Encoder

Decoder

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

[Figure 56]

C

T steps

Downsample

[Figure 57]

C

[Figure 58]

Colorization Guider

[Figure 59]

[Figure 60]

C

[Figure 61]

[Figure 62]

[Figure 63]

CNN CNN

C

Encoder

Reference

Image Pool

|[Figure 64]<br><br>|
|---|

[Figure 65]

Upsample

|[Figure 66]|
|---|

[Figure 67]

Downsample

Q K V

Q K V

Q K V

Q K V

Q K V

Q K V

[Figure 68]

| |
|---|

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

Next Image

In-context Colorization Pipeline

Original Image

- Figure 2. The overview of ColorFlow. This figure presents the three primary components of our framework: the Retrieval-Augmented Pipeline (RAP), the In-context Colorization Pipeline (ICP), and the Guided Super-Resolution Pipeline (GSRP). Each component is essential for maintaining the color identity of instances across black-and-white image sequences while ensuring high-quality colorization.

branch is initialized by replicating the weights of all convolutional layers from the U-Net of the diffusion model.

The inputs to the Colorization Guider consist of the noise latent variable Zt, the output of the variational autoencoder VAE(Cbw) for the composite image Cbw, and the downsampled mask M. These components are concatenated to form a comprehensive input for the model. Features from the Colorization Guider are integrated progressively into the U-Net of the diffusion model, enabling a dense, pixel-wise conditional embedding. Furthermore, we utilize a lightweight LoRA (Low-Rank Adaptation) approach to fine-tune the diffusion model for the colorization task. The loss function can be formalized as follows:

bw,ϵt∥ϵt − ϵθ({VAE(Cbw),M,Zt},t)∥22.

LColor = Et,C

(4) During training, Zt is derived from VAE(Ccolor) through the forward diffusion process.

This training objective allows the model to efficiently denoise the input latent space, gradually reconstructing the desired colored outputs from black-and-white inputs while being guided by reference images. Although we do not explicitly map instances from the colored reference images to those in the black-and-white images, the retrieval mechanism ensures that the reference images contain similar content. As a result, the model naturally learns to leverage contextual information from the retrieved references to accurately colorize the black-and-white images.

Timestep shifted sampling. Given that the colorization process is primarily determined during the higher timesteps, an emphasis on higher timesteps is important for generation.

We modify our sampling strategy by adjusting timestep t′:

eµ eµ + Tt − 1

t′ =

T, t ∼ U(0,T]. (5)

In this work, we set µ to 1.5. This adjustment enables the model to emphasize these higher timesteps, thereby enhancing the effectiveness of the colorization process.

Screenstyle augmentation. Xie et al. previously introduced ScreenVAE [71], which enables the automatic conversion of colored manga into Japanese black-and-white styles. In this work, we augment the input images by performing random linear interpolation between the grayscale images and the outputs generated by ScreenVAE. This augmentation strategy, illustrated in Fig. 4, helps the model better adapt to various styles and improves the overall performance of the colorization process.

Patch-Wise training strategy. To address the substantial computational demands of training on high-resolution stitched images, we introduce a patch-wise training strategy. During training, we randomly crop segments from reference image patches, ensuring that the entire black-and-white image area is always included. The corresponding masks, which indicate the coloring regions, are cropped in the same manner. To further enhance performance, we downsample the input images, reducing computational load while preserving crucial details. This strategy significantly shortens training time per iteration, promoting faster model convergence. During inference, we use the complete stitched im-

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| |
|---|

Reference

Image Patch

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| |
|---|

| | | |
|---|---|---|
| | | |
| | | |

Black and

| | |
|---|---|
| | |

White Image

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

Training

Inference

Mask

- Figure 3. Patch-Wise training strategy is designed to reduce the computational demands of training on high-resolution stitched images. The left box displays segmented stitched images from the training phase, with the corresponding masks also segmented accordingly. The right box presents the complete stitched image and masks for the inference phase.

[Figure 80]

- Figure 4. Screenstyle augmentation. From left to right: the colored manga, the grayscale manga, linear interpolations between the grayscale manga and the ScreenVAE [71] output with proportions of 0.66 and 0.33, the ScreenVAE output.

the issues related to downsampling and structural distortions, resulting in a higher quality final output.

#### 4. Experiments

##### 4.1. Dataset and Benchmark

Training data. The most direct application of sequence image colorization is in manga colorization. In this study, we compiled the largest manga colorization dataset to date, consisting of over 50,000 publicly available color manga chapter sequences sourced from various open online repositories, after filtering out black-and-white manga, resulting in more than 1.7 million images. For each manga frame, we randomly selected at least 20 additional frames from the corresponding manga chapter to construct a diverse reference image pool. Subsequently, we utilized the CLIP image encoder [50] to identify and retrieve the 12 most relevant reference image patches. This systematic recording of selections facilitates subsequent training while minimizing redundant computations.

age to maximize the availability of contextual information for colorization, as shown in Fig. 3.

##### 3.3. Guided Super-Resolution Pipeline

The Guided Super-Resolution Pipeline is designed to tackle the challenges associated with downsampling during the colorization and to reduce the structural distortions often seen in the output from the latent decoder D. These issues can significantly affect the quality of generated images. This pipeline takes as input a high-resolution blackand-white image Ibwhigh and a low-resolution colored output Icolorlow produced by the In-context Colorization Pipeline. The goal is to produce a high-resolution colored image Ipredhigh. To achieve this, we first upsample the low-resolution colored image Icolorlow to match the resolution of Ibwhigh using linear interpolation. The upsampled colored image and the original high-resolution black-and-white image are then processed through the VAE encoder E.

Evaluation Benchmark. Previous reference-based colorization datasets, such as AnimateDiffusion[8], use a single colored image as a reference for each test image, typically featuring only one character. This limits the evaluation of multiple color ID preservation. In contrast, ColorFlowBench is the first benchmark organized by manga series, allowing models to reference multiple pages from the same manga for colorization. Moreover, this design effectively assesses a model’s ability to preserve multiple color IDs, better aligning with industry needs. We establish the ColorFlow-Bench comprising 30 manga chapters that are not included in the training phase. Each chapter contains 40 reference images and 50 black-and-white manga pages, provided in two styles: screenstyle [71] and grayscale image. With a total of 2,700 manga pages, ColorFlowBench stands as the largest manga colorization benchmark. It covers 22 Japanese manga and 8 Western comics, categorized into 11 webtoons and 19 multi-grid manga, ensuring a diverse and reliable dataset. We evaluate the quality of the colorization and the fidelity of the colors to the original images using several metrics: CLIP Image Similar-

To enable effective feature integration, skip guidance is established between the encoder and decoder of the VAE. Intermediate features from both encoders are concatenated and passed to a fusion module F, which transmits the combined information to the corresponding layers in the decoder. This multi-scale approach enhances detail restoration, as illustrated in Fig. 2.

The overall loss function for this process is defined as:

LSR =E[|Ibwhigh − D(F(concat(Efeatures(Ibwhigh), Efeatures(Upsample(Icolorlow )))),E(Ibwhigh))|1],

(6) where Efeatures denotes the intermediate features extracted from the VAE encoder. This pipeline effectively addresses

- Table 1. Quantitative comparisons with state-of-the-art models for Reference Image-based Colorization. We compare two models without reference image input Manga Colorization V2 (MC-v2) [45] and AnimeColorDeOldify (ACDO) [16], and two reference imagebased colorization models, Example Based Manga Colorization (EBMC) [27] and ScreenVAE [71]. Best results are in bold.

Screenstyle Grayscale Image

Method Reference-based

CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑ CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑ MC-v2 [46] 0.8632 48.37 13.50 0.6987 4.753 0.8833 33.14 17.20 0.8396 4.845

ACDO [2] 0.8687 39.38 15.75 0.7672 4.540 0.8970 28.12 21.77 0.9516 4.686 EBMC [27] ✓ 0.8542 38.77 15.21 0.7592 4.605 0.8859 19.48 20.80 0.9474 4.702

ScreenVAE [71] ✓ 0.7328 98.52 9.12 0.5373 4.160 - - - Ours ✓ 0.9419 13.37 25.88 0.9541 4.924 0.9433 12.17 26.01 0.9579 5.011

ity (CLIP-IS) [50], Fr´echet Inception Distance (FID) [25], Peak Signal-to-Noise Ratio (PSNR) [66], Structural Similarity Index (SSIM) [63], and Aesthetic Score (AS) [55]. These metrics provide a thorough and holistic assessment of the colorization process, evaluating not only the aesthetic quality of the generated images but also their consistency with the original content.

##### 4.2. Implementation Details

Our colorization model is based on Stable Diffusion v1.5 [52]. We trained our model, along with all ablation models, for 150,000 steps using 8 NVIDIA A100 GPUs, with a learning rate of 1e-5. Additionally, the Guided Super-Resolution Pipeline was trained for 30,000 iterations under the same hardware configuration and learning rate. For inference, all methods were tested on NVIDIA Tesla A100 GPUs, consistent with their open-source code.

##### 4.3. Baseline Models

To ensure a fair comparison, we select the most recent and competitive approaches in manga colorization. Colorization without reference images includes Manga Colorization V2 (MC-v2) [45], which employs CycleGAN for automated manga colorization, and AnimeColorDeOldify (ACDO) [16], a DeOldify variant optimized for anime and manga. Colorization based on reference images features Example Based Manga Colorization (EBMC) [27], which uses a cGAN to combine color features from reference images with grayscale content; ScreenVAE [71], which utilizes variational autoencoders for colorization; and Style2Paints V4.5 [78], a software designed for coloring line art that matches color styles using reference images.

##### 4.4. Quantitative Comparisons

In Tab. 1, we present a comparison between ColorFlow and previous works using ColorFlow-Bench. Our results show that ColorFlow significantly outperforms other models across all metrics, including semantic alignment (CLIPIS, FID), pixel alignment (PSNR, SSIM), and Aesthetic Score (AS), highlighting its superior image colorization ac-

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Self-Attention Map Colorized Image Self-Attention Map Colorized Image

Figure 5. Visualization of the heatmap for the self-attention map of the selected colorization region (encircled in red).

curacy. Although methods like EBMC [27] and ScreenVAE [71] are capable of reference-based colorization, they fall short due to weak in-context learning and an inability to maintain consistent sequential colorization. In contrast, ColorFlow excels by utilizing diffusion models to effectively preserve color identity, as demonstrated by the selfattention map in Fig. 5.

##### 4.5. Qualitative Comparisons

To showcase ColorFlow’s generalization ability, we present qualitative results in four scenarios: manga colorization (Fig.6), cartoon colorization (Fig. 7), line art colorization (Fig. 8), and natural scenes colorization (Fig. 8).

- Fig. 6 compares the colorization results of ColorFlow

with previous methods. The no-reference model MCv2[45] lacks contextual awareness, leading to random colorization. EBMC and Style2Paints[78] use reference images but suffer from information loss, resulting in imprecise colorization. In contrast, ColorFlow integrates reference images effectively using image stitching, leveraging self-attention layers in diffusion models to maintain color consistency across manga frames.

- Fig. 7 and Fig. 8 demonstrate the exceptional perfor-

mance of ColorFlow across a diverse range of scenarios, including cartoon, line art, and natural images. These re-

[Figure 85]

- Figure 6. Comparison of our method with SOTA approaches in the manga colorization. Our method exhibits superior aesthetic quality, producing colors that more closely match the original image. [Best viewed in color with zoom-in]

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

Original Image Input Image MC-v2 EBMC ACDO Reference Images Ours

- Figure 7. Comparison of ColorFlow with other approaches in the animation storyboard colorization. Our method exhibits superior aesthetic quality, producing colors that more closely match the original image. [Best viewed in color with zoom-in]

sults emphasize the robustness and adaptability of our approach, demonstrating its strong generalization capabilities to effectively handle varying styles and content types.

##### 4.6. Abaltion Study

Pipeline components. In Tab. 2, we compare the impact of the Retrieval-Augmented Pipeline and the Guided SuperResolution Pipeline during training and inference. The results indicate that using the Retrieval-Augmented Pipeline

[Figure 100]

- Figure 8. Colorization results for line art and natural scenario.

for both training and inference, as well as the Guided SuperResolution Pipeline during training, is crucial for the performance of ColorFlow.

- Table 2. Ablation Study on the Influence of Retrieval-Augmented Pipeline (RAP) and Guided Super-Resolution Pipeline (GSRP).

Training Inference RAP RAP GSRP CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑

✓ ✓ 0.9326 15.98 24.48 0.9448 4.921

✓ 0.9233 18.32 24.16 0.9410 4.907 ✓ ✓ 0.9266 17.07 24.64 0.9464 4.914 ✓ ✓ 0.9322 17.85 20.12 0.8077 4.898 ✓ ✓ ✓ 0.9419 13.37 25.88 0.9541 4.924

Inference resolution. In Tab. 3, we conduct an ablation study on three different inference resolutions. Despite being trained only on a resolution of 512 × 800, the results demonstrate that ColorFlow has the ability to generalize across different resolutions.

Table 3. Ablation Study of Inference Resolution.

Width × Height (Pixel) CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑

512 × 800 0.9372 14.91 23.51 0.9414 4.868 1024 × 1600 0.9419 13.37 25.88 0.9541 4.924 1280 × 2000 0.9398 13.42 26.02 0.9580 4.929

LoRA rank. To demonstrate the necessity of partially retaining pretrained diffusion weights, we ablate on the rank of LoRA on the base diffusion model, where larger LoRA rank indicates bigger change on pretrained diffusion model weight. Tab. 4 shows that too big or too small LoRA rank all lead to performance decay, validating the choice of 64 for the optimal LoRA rank.

Sampling timesteps. In Tab. 5, we ablate on the design of timestep shift sampling. Since colorization is primarily

Table 4. Ablation Study of LoRA Rank.

Rank CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑

32 0.940 13.46 25.46 0.9521 4.920 64 0.9419 13.37 25.88 0.9541 4.924

128 0.9376 14.31 24.79 0.9461 4.930 192 0.9370 14.46 24.59 0.9440 4.914

performed at a higher timestep, we strengthen the sampling at higher timestep by a factor of µ. The results validate the effectiveness of adding timesteps sampling and using a factor of µ = 1.5.

Table 5. Ablation Study of Timesteps Sampling.

µ CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑ 0 0.9351 14.18 25.12 0.9501 4.927

1.5 0.9419 13.37 25.88 0.9541 4.924 3 0.9395 13.51 25.42 0.9509 4.917

##### 4.7. User Study

To perform a comprehensive comparison, we conducted a user study evaluating three key aspects: aesthetic quality, similarity to the original image, and consistency of color IDs in image sequences. In each trial, participants ranked their preferences among five sample groups. We assigned scores based on these rankings, with the first place receiving 5 points and the fifth place receiving 1 point. We then calculated the average score for each evaluation criterion. As detailed in Tab. 6, we gathered over 4,000 valid rankings. The results demonstrate that our colorization method is the preferred choice across all evaluation criteria.

Table 6. Results of the User Study. The table presents the average Score for different models based on aesthetic quality, similarity to the original image, and consistency in sequences

Ours EBMC MC-v2 ACDO ScreenVAE Aesthetic Quality ↑ 4.577 3.141 2.891 2.844 1.547

Similarity to Original ↑ 4.673 3.316 2.984 2.642 1.385 Consistency in Sequences ↑ 4.538 3.399 3.215 2.540 1.308

#### 5. Conclusion

In conclusion, this paper proposes ColorFlow for a novel task, Reference-based Image Sequence Colorization. The proposed method consists of a three-stage framework: the Retrieval-Augmented Pipeline, the In-context Colorization Pipeline, and the Super-Resolution Pipeline. Extensive quantitative and qualitative evaluation results on our proposed benchmark, ColorFlow-Bench, show the superior performance of ColorFlow. More discussion on limitations and future work will be discussed in supplementary files.

#### References

- [1] Kenta Akita, Yuki Morimoto, and Reiji Tsuruno. Colorization of line drawings with empty pupils. In Computer Graphics Forum, pages 601–610. Wiley Online Library, 2020. 2
- [2] AnimeColorDeOldify, 2020. https://github.com/Dakini/AnimeColorDeOldify. 6
- [3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022. 2
- [4] Hyojin Bahng, Seungjoo Yoo, Wonwoong Cho, David Keetae Park, Ziming Wu, Xiaojuan Ma, and Jaegul Choo. Coloring with words: Guiding image colorization through textbased palette generation. In Proceedings of the european conference on computer vision (eccv), pages 431–447, 2018. 2
- [5] Yunpeng Bai, Chao Dong, Zenghao Chai, Andong Wang, Zhengzhuo Xu, and Chun Yuan. Semantic-sparse colorization network for deep exemplar-based colorization. In European Conference on Computer Vision, pages 505–521. Springer, 2022. 2
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 1, 2
- [7] Ruizhi Cao, Haoran Mo, and Chengying Gao. Line art colorization based on explicit region segmentation. In Computer Graphics Forum, pages 1–10. Wiley Online Library,

2021. 2

- [8] Yu Cao, Xiangqiao Meng, PY Mok, Xueting Liu, TongYee Lee, and Ping Li. Animediffusion: Anime face line drawing colorization via diffusion models. arXiv preprint arXiv:2303.11137, 2023. 2, 5
- [9] Huiwen Chang, Ohad Fried, Yiming Liu, Stephen DiVerdi, and Adam Finkelstein. Palette-based photo recoloring. ACM Trans. Graph., 34(4):139–1, 2015. 2
- [10] Zheng Chang, Shuchen Weng, Yu Li, Si Li, and Boxin Shi. L-coder: Language-based colorization with color-object decoupling transformer. In European Conference on Computer Vision, pages 360–375. Springer, 2022. 2
- [11] Zheng Chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, and Boxin Shi. L-coins: Language-based colorization with instance awareness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19221–19230, 2023.
- [12] Jianbo Chen, Yelong Shen, Jianfeng Gao, Jingjing Liu, and Xiaodong Liu. Language-based image editing with recurrent attentive models. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8721–8729,

2018. 2

- [13] Shu-Yu Chen, Jia-Qi Zhang, Lin Gao, Yue He, Shihong Xia, Min Shi, and Fang-Lue Zhang. Active colorization for cartoon line drawings. IEEE Transactions on Visualization and Computer Graphics, 28(2):1198–1208, 2020. 2
- [14] Yuanzheng Ci, Xinzhu Ma, Zhihui Wang, Haojie Li, and Zhongxuan Luo. User-guided deep anime line art coloriza-

- tion with conditional adversarial networks. In Proceedings of the 26th ACM international conference on Multimedia, pages 1536–1544, 2018. 2
- [15] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 2
- [16] Dakini,AIEMMU,Abhinas Regmi. Anime/sketch/manga coloriser trained with deoldify, 2024. [Online; accessed 4Oct-2024]. 6
- [17] Zhi Dou, Ning Wang, Baopu Li, Zhihui Wang, Haojie Li, and Bin Liu. Dual color space guided sketch colorization. IEEE Transactions on Image Processing, 30:7292–7304, 2021. 2
- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1
- [19] Faming Fang, Tingting Wang, Tieyong Zeng, and Guixu Zhang. A superpixel-based variational model for image colorization. IEEE Transactions on Visualization and Computer Graphics, 26(10):2931–2943, 2019. 2
- [20] Kevin Frans. Outline colorization through tandem adversarial networks. arXiv preprint arXiv:1704.08834, 2017. 2
- [21] Chie Furusawa, Kazuyuki Hiroshiba, Keisuke Ogaki, and Yuri Odagiri. Comicolorization: semi-automatic manga colorization. In SIGGRAPH Asia 2017 Technical Briefs, New York, NY, USA, 2017. Association for Computing Machinery. 2
- [22] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [23] Mingming He, Dongdong Chen, Jing Liao, Pedro V Sander, and Lu Yuan. Deep exemplar-based colorization. ACM Transactions on Graphics (TOG), 37(4):1–16, 2018. 2
- [24] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2
- [25] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in Neural Information Processing Systems (NIPS), 30, 2017. 6
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NIPS), 33:6840–6851, 2020. 2
- [27] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134,

2017. 6

- [28] Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. Humansd: A native skeleton-guided

- diffusion model for human image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15988–15998, 2023. 2
- [29] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. arXiv preprint arXiv:2403.06976, 2024. 1, 2
- [30] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In The Twelfth International Conference on Learning Representations, 2024. 2
- [31] Hyunsu Kim, Ho Young Jhoo, Eunhyeok Park, and Sungjoo Yoo. Tag2pix: Line art colorization using text tag with secat and changing loss. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9056–9065,

2019. 2

- [32] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3
- [33] Junsoo Lee, Eungyeup Kim, Yunsung Lee, Dongjun Kim, Jaehyuk Chang, and Jaegul Choo. Reference-based sketch image colorization using augmented-self reference and dense semantic correspondence. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5801–5810, 2020. 2
- [34] Bo Li, Yu-Kun Lai, Matthew John, and Paul L Rosin. Automatic example-based image colorization using locationaware cross-scale matching. IEEE Transactions on Image Processing, 28(9):4606–4619, 2019.
- [35] Haoxuan Li, Bin Sheng, Ping Li, Riaz Ali, and CL Philip Chen. Globally and locally semantic colorization via exemplar-based broad-gan. IEEE Transactions on Image Processing, 30:8526–8539, 2021.
- [36] Yuan-kui Li, Yun-Hsuan Lien, and Yu-Shuen Wang. Stylestructure disentangled features and normalizing flows for diverse icon colorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11244–11253, 2022.
- [37] Zekun Li, Zhengyang Geng, Zhao Kang, Wenyu Chen, and Yibo Yang. Eliminating gradient conflict in reference-based line-art colorization. In European Conference on Computer Vision, pages 579–596. Springer, 2022. 2
- [38] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024. 3
- [39] Zhexin Liang, Zhaochen Li, Shangchen Zhou, Chongyi Li, and Chen Change Loy. Control color: Multimodal diffusion-based interactive image colorization. arXiv preprint arXiv:2402.10855, 2024. 2
- [40] Hanyuan Liu, Jinbo Xing, Minshan Xie, Chengze Li, and Tien-Tsin Wong. Improved diffusion-based image colorization via piggybacked models. arXiv preprint arXiv:2304.11105, 2023. 2

- [41] Xian Liu, Jian Ren, Aliaksandr Siarohin, Ivan Skorokhodov, Yanyu Li, Dahua Lin, Xihui Liu, Ziwei Liu, and Sergey Tulyakov. Hyperhuman: Hyper-realistic human generation with latent structural diffusion. arXiv preprint arXiv:2310.08579, 2023. 2
- [42] Yifan Liu, Zengchang Qin, Tao Wan, and Zhenbo Luo. Autopainter: Cartoon image generation from sketch by using conditional wasserstein generative adversarial networks. Neurocomputing, 311:78–87, 2018. 2
- [43] Zhiheng Liu, Ka Leong Cheng, Xi Chen, Jie Xiao, Hao Ouyang, Kai Zhu, Yu Liu, Yujun Shen, Qifeng Chen, and Ping Luo. Manganinja: Line art colorization with precise reference following. arXiv preprint arXiv:2501.08332, 2025. 2
- [44] Peng Lu, Jinbei Yu, Xujun Peng, Zhaoran Zhao, and Xiaojie Wang. Gray2colornet: Transfer more colors from reference image. In Proceedings of the 28th ACM international conference on multimedia, pages 3210–3218, 2020. 2
- [45] Maksim Golyadkin,Pupbani,Abhinas Regmi. Automatic colorization, 2024. [Online; accessed 4-Oct-2024]. 6
- [46] Manga Colorization V2, 2022. https://github.com/qweasdd/manga-colorization-v2. 6
- [47] Varun Manjunatha, Mohit Iyyer, Jordan Boyd-Graber, and Larry Davis. Learning to color from language. arXiv preprint arXiv:1804.06026, 2018. 2
- [48] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024. 3
- [49] Yingge Qu, Tien-Tsin Wong, and Pheng-Ann Heng. Manga colorization. ACM Transactions on Graphics (ToG), 25(3): 1214–1220, 2006. 2
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5, 6
- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 2
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 6, 1
- [53] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 3
- [54] Patsorn Sangkloy, Jingwan Lu, Chen Fang, Fisher Yu, and James Hays. Scribbler: Controlling deep image synthesis

- with sketch and color. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5400–5409, 2017. 2
- [55] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 6
- [56] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [57] Tsai-Ho Sun, Chien-Hsun Lai, Sai-Keung Wong, and YuShuen Wang. Adversarial colorization of icons based on contour and color conditions. In Proceedings of the 27th ACM International Conference on Multimedia, pages 683– 691, 2019. 2
- [58] Daniel S`ykora, John Dingliana, and Steven Collins. Lazybrush: Flexible painting tool for hand-drawn cartoons. In Computer Graphics Forum, pages 599–608. Wiley Online Library, 2009. 2
- [59] Chaitat Utintu, Pinaki Nath Chowdhury, Aneeshan Sain, Subhadeep Koley, Ayan Kumar Bhunia, and Yi-Zhe Song. Sketchdeco: Decorating b&w sketches with colour. arXiv preprint arXiv:2405.18716, 2024. 2
- [60] Hanzhang Wang, Deming Zhai, Xianming Liu, Junjun Jiang, and Wen Gao. Unsupervised deep exemplar colorization via pyramid dual non-local attention. IEEE Transactions on Image Processing, 2023. 2
- [61] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 3
- [62] Yi Wang, Menghan Xia, Lu Qi, Jing Shao, and Yu Qiao. Palgan: Image colorization with palette generative adversarial networks. In European Conference on Computer Vision, pages 271–288. Springer, 2022. 2
- [63] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [64] Shuchen Weng, Hao Wu, Zheng Chang, Jiajun Tang, Si Li, and Boxin Shi. L-code: Language-based colorization using color-object decoupled conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2677– 2684, 2022. 2
- [65] Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi, et al. L-cad: Language-based colorization with any-level descriptions using diffusion priors. Advances in Neural Information Processing Systems, 36, 2024. 1, 2
- [66] Wikipedia contributors. Peak signal-to-noise ratio Wikipedia, the free encyclopedia, 2024. [Online; accessed 4-March-2024]. 6
- [67] Shukai Wu, Xiao Yan, Weiming Liu, Shuchang Xu, and Sanyuan Zhang. Self-driven dual-path learning for reference-based line art colorization under limited data. IEEE Transactions on Circuits and Systems for Video Technology, 2023. 2

- [68] Shukai Wu, Yuhang Yang, Shuchang Xu, Weiming Liu, Xiao Yan, and Sanyuan Zhang. Flexicon: Flexible icon colorization via guided images and palettes. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8662–8673, 2023. 2
- [69] Wenqi Xian, Patsorn Sangkloy, Varun Agrawal, Amit Raj, Jingwan Lu, Chen Fang, Fisher Yu, and James Hays. Texturegan: Controlling deep image synthesis with texture patches. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8456–8465, 2018.
- [70] Chufeng Xiao, Chu Han, Zhuming Zhang, Jing Qin, TienTsin Wong, Guoqiang Han, and Shengfeng He. Examplebased colourization via dense encoding pyramids. In Computer Graphics Forum, pages 20–33. Wiley Online Library,

2020. 2

- [71] Minshan Xie, Chengze Li, Xueting Liu, and Tien-Tsin Wong. Manga filling style conversion with screentone variational autoencoder. ACM Transactions on Graphics (TOG), 39(6):1–15, 2020. 2, 4, 5, 6
- [72] Zhongyou Xu, Tingting Wang, Faming Fang, Yun Sheng, and Guixu Zhang. Stylization-based architecture for fast deep exemplar colorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9363–9372, 2020. 2
- [73] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [74] Wang Yin, Peng Lu, Zhaoran Zhao, and Xujun Peng. Yes,” attention is all you need”, for exemplar based colorization. In Proceedings of the 29th ACM international conference on multimedia, pages 2243–2251, 2021. 2
- [75] Jooyeol Yun, Sanghyeon Lee, Minho Park, and Jaegul Choo. icolorit: Towards propagating local hints to the right region in interactive colorization by leveraging vision transformer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1787–1796, 2023. 2
- [76] Nir Zabari, Aharon Azulay, Alexey Gorkor, Tavi Halperin, and Ohad Fried. Diffusing colors: Image colorization with text guided diffusion. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023. 1, 2
- [77] Jiangning Zhang, Chao Xu, Jian Li, Yue Han, Yabiao Wang, Ying Tai, and Yong Liu. Scsnet: An efficient paradigm for learning simultaneously image colorization and superresolution. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3271–3279, 2022. 2
- [78] Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. Two-stage sketch colorization. ACM Transactions on Graphics (TOG), 37(6):1–14, 2018. 2, 6
- [79] Lvmin Zhang, Chengze Li, Edgar Simo-Serra, Yi Ji, TienTsin Wong, and Chunping Liu. User-guided line art flat filling with split filling mechanism. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9889–9898, 2021. 2
- [80] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3

- [81] Qian Zhang, Bo Wang, Wei Wen, Hai Li, and Junhui Liu. Line art correlation matching feature transfer network for automatic animation colorization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3872–3881, 2021. 2
- [82] Richard Zhang, Jun-Yan Zhu, Phillip Isola, Xinyang Geng, Angela S Lin, Tianhe Yu, and Alexei A Efros. Real-time user-guided image colorization with learned deep priors. arXiv preprint arXiv:1705.02999, 2017. 2
- [83] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 2, 3
- [84] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. arXiv preprint arXiv:2312.03594, 2023. 1, 2
- [85] Changqing Zou, Haoran Mo, Chengying Gao, Ruofei Du, and Hongbo Fu. Language-based colorization of scene sketches. ACM Transactions on Graphics (TOG), 38(6):1– 16, 2019. 2
- [86] Chengyi Zou, Shuai Wan, Marc Gorriz Blanch, Luka Murn, Marta Mrak, Juil Sock, Fei Yang, and Luis Herranz. Lightweight deep exemplar colorization via semantic attention-guided laplacian pyramid. IEEE Transactions on Visualization and Computer Graphics, 2024. 2

### ColorFlow: Retrieval-Augmented Image Sequence Colorization Supplementary Material

This supplementary material provides further insights and additional results. It elaborates on ColorFlow’s performance across various artistic contexts, discussing retrieval effectiveness, the model’s limitations, and the ethical considerations associated with generating synthetic content.

#### 6. Additional Results

To highlight the robustness and versatility of ColorFlow, we present a series of visual results across diverse artistic contexts. Fig.10, 11, 12, and 13 demonstrate ColorFlow’s performance on black-and-white manga, line art, animation storyboards, and grayscale natural scenes, respectively. These examples collectively emphasize ColorFlow’s adaptability across various contexts, establishing it as a valuable tool for artists and content creators aiming to enhance their work through automated colorization.

#### 7. Effectiveness of CLIP Retrieval

We collected 10 anime characters from the internet, each with 10 images. A t-SNE visualization of the CLIP embeddings shows that images of the same character cluster together (see Figure 9). In top-1 retrieval tests, we achieved a 100% success rate with color images and 97% with grayscale. These results demonstrate that the CLIP image encoder effectively retrieves the corresponding characters, highlighting the robustness of the RAP.

#### 8. Limitations and Future Work

Despite the significant advancements achieved by ColorFlow in reference-based image sequence colorization, several limitations require careful consideration.

First, ColorFlow’s performance heavily relies on the quality of images in reference pool. If the artistic style of reference images is highly abstract or significantly different from the target style, the colorization accuracy may suffer.

Additionally, ColorFlow’s ability to generate images and preserve color identity is limited by its base model, Stable Diffusion 1.5 [52]. Although this model is effective, there is room for improvement by using more advanced architectures like Flux.1 or SD3 [18]. In future work, we plan to train ColorFlow on these next-generation models, which could enhance color fidelity and overall image quality.

We also intend to explore using ColorFlow for long video colorization. This would expand its use in multimedia production, allowing for consistent colorization across extended video frames.

Using the CLIP-ViT-H-14-laion2B-s32B-b79K model

[Figure 101]

[Figure 102]

Figure 9. Visualization of 10 characters, each represented by 10 images (left). The corresponding t-SNE plot illustrates the CLIP image embeddings (right).

#### 9. Ethical Considerations

While our research primarily emphasizes the technical advancements in image colorization, it is acknowledged there are ethical implications associated with the generation of synthetic content. We address the limitation in this section.

Our model is trained on data sourced from the internet, which may inadvertently reflect and amplify existing biases present in the training data. This concern can be salient, as artificial intelligence systems trained on biased datasets can perpetuate stereotypes and exacerbate inequalities, disproportionately affecting various demographic groups. To address this issue, we tried to ensure diversity in the training data to cover a wide range of styles, demographics, and cultural contexts. Moreover, we will monitor and evaluate the model for biased behavior, and fine-tune it using balanced datasets. Moreover, it is possible that the colorization method could be used maliciously, such as altering historical artifacts or creating misleading media. To address this issue, we will include watermarking or traceable signatures in outputs to indicate AI-generated content. We will also publish ethical usage guidelines for the model and monitor public applications of the model to identify and address misuse. By addressing these issues, our work can maintain ethical integrity while maximizing its positive impact.

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

Input Images

Reference Images

Ours

Input Images

Ours

Reference Images

###### Figure 10. Colorization results of black and white manga using ColorFlow. [Best viewed in color with zoom-in]

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

Input Images

Reference Images

Ours

Input Images

Ours

Reference Images

Figure 11. Colorization results of line art using ColorFlow. [Best viewed in color with zoom-in]

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

Input Images Reference Images Ours Input Images Reference Images Ours

- Figure 12. Colorization results of animation storyboard using ColorFlow. [Best viewed in color with zoom-in]

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

[Figure 149]

[Figure 150]

Input Images Reference Images Ours Input Images Reference Images Ours

- Figure 13. Colorization results of natural scenario using ColorFlow. [Best viewed in color with zoom-in]

