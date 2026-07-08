### SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration

## arXiv:2501.01320v4[cs.CV]22Mar2025

Jianyi Wang1,2∗ Zhijie Lin2 Meng Wei2 Yang Zhao2 Ceyuan Yang2 Fei Xiao2 Chen Change Loy1 Lu Jiang2

1Nanyang Technological University 2ByteDance

https://iceclear.github.io/projects/seedvr/

[Figure 1]

Figure 1. Speed and performance comparisons. SeedVR demonstrates impressive restoration capabilities, offering fine details and enhanced visual realism. Despite its 2.48B parameters, SeedVR is over 2× faster than existing diffusion-based video restoration approaches [20, 64, 82]. With delicate designs, SeedVR is as efficient as the Stable Diffusion Upscaler [2], even with five times the parameter count. (Zoom-in for best view)

#### Abstract

Video restoration poses non-trivial challenges in maintaining fidelity while recovering temporally consistent details from unknown degradations in the wild. Despite recent

∗ Work was done during Jianyi Wang’s internship at ByteDance in Singapore. (iceclearwjy@gmail.com)

advances in diffusion-based restoration, these methods often face limitations in generation capability and sampling efficiency. In this work, we present SeedVR, a diffusion transformer designed to handle real-world video restoration with arbitrary length and resolution. The core design of SeedVR lies in the shifted window attention that facilitates effective restoration on long video sequences. SeedVR further supports variable-sized windows near the boundary of

both spatial and temporal dimensions, overcoming the resolution constraints of traditional window attention. Equipped with contemporary practices, including causal video autoencoder, mixed image and video training, and progressive training, SeedVR achieves highly-competitive performance on both synthetic and real-world benchmarks, as well as AI-generated videos. Extensive experiments demonstrate SeedVR’s superiority over existing methods for generic video restoration.

#### 1. Introduction

Generic video restoration (VR) is a classical computer vision task, seeking to reconstruct high-quality (HQ) outputs from low-quality (LQ) input videos. A broad range of works have been proposed to tackle the challenges posed by complex and often unknown degradations [6, 56, 61, 81] encountered in real-world VR scenarios [4, 5, 32, 33, 55, 63].

More recently, diffusion-based image [54, 60, 70] and video restoration methods [12, 20, 64, 67, 82], often built on U-Net architectures with full-attention layers, have shown promise in addressing the issues found in previous approaches such as over-smoothing. However, the attention design in diffusion leads to significant computational costs and performance degradation when processing resolutions different from those used during training, limiting their applicability for restoring long-duration, high-resolution videos.

As such, previous VR approaches [12, 20, 64, 67, 82] rely on patch-based sampling [24, 54], i.e., dividing the input video into overlapping spatial-temporal patches and fusing these patches using a Gaussian kernel at each diffusion step. The large overlap (e.g., 50% of the patch size), required for ensuring a coherent output without visible patch boundaries, often leads to considerably slow inference speed. This inefficiency becomes even more pronounced when processing long videos at high resolutions. For instance, VEnhancer [20] takes 387 seconds to generate 31 frames at a resolution of 1344×768 with 50 sampling steps, even when using only temporal overlap. Likewise, Upscale-A-Video [82], using a spatial overlapping of 384×384 and a temporal overlapping of 2, takes 414 seconds to process the same video clip, rendering it less practical for real-world use.

In this work, we present SeedVR, a Diffusion Transformer (DiT) model designed for generic video restoration (VR) that tackles resolution constraints efficiently. We propose a design using large non-overlapping window attention in DiT, which we found effective for achieving competitive VR quality at a lower computational cost. Specifically, SeedVR uses MMDiT [17] as its backbone and replaces full self-attention with a window attention mechanism. While various window attention designs have been explored, we aim to keep our design as simple as possible and hence use the Swin attention [35], resulting in Swin-MMDiT. Unlike

previous methods [31, 73, 74], our Swin-MMDiT adopts a significantly larger attention window of 64 × 64 over an 8 × 8 compressed latent, compared to the 8 × 8 pixel space commonly used in window attention for low-level vision tasks [31, 73, 74]. When processing arbitrary input resolutions with Swin-MMDiT using a large window, we can no longer assume that the input spatial dimensions will be multiples of the window size. Additionally, the shifted window mechanism in Swin results in uneven 3D windows near the boundaries of the space-time volume. To address these, we design a 3D rotary position embedding [48] within each window to model the varying-sized windows.

To enhance the SeedVR training, we further incorporate several techniques inspired by recent work. First, building on Yu et al. [71], we develop a causal video variational autoencoder (CVVAE) that compresses time and space by factors of 4 and 8, respectively. This CVVAE significantly reduces the computational cost of VR, especially for high-resolution videos, while maintaining high reconstruction quality. Second, motivated by Dehghani et al. [15], we train SeedVR on images and videos with native and varying resolutions. Finally, we employ a multi-stage progressive training strategy to accelerate convergence on large-scale datasets. Extensive experimental results demonstrate that our SeedVR performs steadily well across various VSR benchmarks from different sources, serving as a strong baseline for VR in diverse real-world scenarios, as shown in Figure 1. See the supplementary materials for the videos.

To our knowledge, SeedVR is among the earliest explorations on training a large scalable diffusion transformer model designed for generic video restoration. The main contributions are as follows: 1) We tackle the key challenge in diffusion-based VR, i.e., handling inputs with arbitrary resolutions, by proposing simple yet effective diffusion transformer blocks based on a shifted window attention mechanism. 2) We further develop a casual video autoencoder, considerably improving both training and inference efficiency while achieving favorable video reconstruction quality. As shown in Figure 1, SeedVR is at least 2× faster than existing diffusion-based VR methods [20, 64, 82], despite having 2.48B parameters, which is over 3.5× more than UpscaleA-Video [82]. 3) By leveraging large-scale joint training on images and videos, along with multi-scale progressive training, SeedVR achieves state-of-the-art performance across diverse benchmarks, outperforming existing approaches by a large margin. Serving as the largest-ever diffusion transformer model towards generic VR, we believe SeedVR will push the frontiers of advanced VR and inspire future research in developing large vision models for real-world VR.

#### 2. Related Work

Attention Mechanism in Restoration. Early restoration approaches that adopted CNN-based architectures [4–

6, 22, 23, 46, 50, 55] typically struggled to capture longrange pixel dependencies due to limited receptive fields. Recent advances in transformer models have inspired a series of restoration methods [7, 11, 31, 33, 61, 83, 84] that introduce attention mechanisms into restoration networks, further improving performance on restoration benchmarks. To mitigate the quadratic complexity of the self-attention mechanism [52], many of these approaches [11, 31, 33, 83, 84] use window attention to reduce computational costs. For instance, SwinIR [31] adopts Swin Transformer [35] with a 8× 8 window attention. SRFormer [83] and SRFormerV2 [84] further increase the window size to 24 × 24 and 40 × 40, respectively, to enhance performance. Despite these improvements, limited window sizes still restrict the receptive field, especially in diffusion models where text embeddings interact with image embeddings within each window. As a result, existing diffusion-based restoration methods [20, 45, 54, 60, 70, 82] continue to rely on full attention to achieve effective restoration with text guidance.

In this study, we focus on investigating the window attention mechanism within a diffusion transformer for VR. By employing a substantially larger attention window, i.e., 64 × 64 in an 8× compressed latent space, our method interacts with text prompts and captures long-range dependencies. We also introduce variable-sized windows near the boundaries of each dimension, reducing resolution constraints. This design allows our approach to circumvent the reliance on tiled sampling strategies [24, 54] enables direct application to VR tasks with any length and resolution.

Diffusion Transformer. The development of DiT [40] has made diffusion transformer the prevailing architecture for diffusion models [8–10, 17, 19, 25, 27, 30, 34, 41, 65, 78]. To reduce the high computational cost of generating highresolution images and videos, common approaches include using separate temporal and spatial attention [78], applying token compression [8] and generating outputs in a multistage manner [25]. Instead of relying on the full attention mechanism, FIT [10] interleaves window attention and global attention with two types of transformer. While this method is efficient, it falls short of handling variable-sized inputs in VR. Inf-DiT [65] enables upscaling on images of varying shapes and resolutions by using local attention in an autoregressive manner, though it is limited by a finite receptive field. The approach most similar to ours is VideoPoet [27], which uses three types of 2D window attention for video super-resolution, each performing selfattention within a local window aligned along one of three axes. However, this method still struggles with arbitrary input shapes, as it requires a full attention operation along one axis. In contrast, our approach introduces a flexible 3D window attention that can be effectively applied to VR with varying resolutions.

Video Restoration. Most previous works [4, 5, 12, 29,

32, 33, 55, 69] focus primarily on synthetic data, resulting in limited effectiveness for real-world VR. Later approaches [6, 61, 77] have shifted towards real-world VR, yet still struggle to produce realistic textures due to limited generative capabilities. Motivated by recent advances in diffusion models [21, 39, 45, 47, 62], several diffuisonbased VR approaches [20, 64, 82] have emerged, showing impressive performance. While fine-tuning from a diffusion prior [45, 57] provides efficiency, these methods still inherit limitations inherent to diffusion priors. In particular, they use a basic autoencoder without temporal compression, resulting in inefficient training and inference. Additionally, their reliance on full attention imposes resolution constraints, further increasing the inference cost. Unlike existing diffusionbased VR approaches, we redesign the whole architecture with an efficient video autoencoder and a flexible window attention mechanism, achieving effective and efficient VR with arbitrary length and resolution.

#### 3. Methodology

We focus on effective VR with arbitrary lengths and resolutions, which is still underexplored. As depicted in Figure 1a, our approach employs a similar architecture following SD3 [17], where a pretrained autoencoder is applied to compress the input video into latent space, and the corresponding text prompt is encoded by three pretrained, frozen text encoders [13, 42, 43]. To relax the resolution constraints of the MMDiT block used in SD3, we introduce a SwinMMDiT block based on a shifted window mechanism, as detailed in Sec. 3.1. We further present our casual video autoencoder in Sec. 3.2, which significantly improves the training and inference efficiency compared to existing approaches [20, 64, 82]. In Sec. 3.3, we discuss the training strategies to effectively train our model on large-scale datasets.

##### 3.1. Shifted Window Based MM-DiT

MMDiT has been proven to be an effective transformer block by SD3 [17], where separate weights are applied to the two modalities, i.e., visual input and text, thus enabling a bidirectional flow of information between visual features and text tokens. However, the full attention nature of MMDiT makes it unsuitable for VR, which requires the capability of handling inputs with arbitrary lengths and resolutions. To this end, we introduce a shifted window attention mechanism into MMDiT, which we call as Swin-MMDiT. Given a video feature X ∈ RT×H×W×d and a text embedding Ctext ∈ RL×d, the video feature is first flattened to X′ ∈ RTHW×d, following the NaViT scheme [15]. For MMDiT [17], it directly extracts (QX′,KX′,VX′) and (Qtext,Ktext,Vtext) from X′ and Ctext, respectively. Full attention is then applied on the concatenation Cat(·) of the extracted features, i.e.,

# …

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

…

[Figure 12]

Swin-MMDiT-Block n

Caption

|Unflatten|
|---|
|Window Partition|
|Flatten|

T5 XXL

C CLIP-G/14

Add Noise

C

|Video Attention<br><br>… …<br><br>…<br><br>…<br><br>𝑄𝑋′<br><br>𝐶𝑎𝑡(𝐾…𝑋′,𝐾𝑡𝑒𝑥𝑡)|
|---|

CLIP-L/14

Patching & Flatten Linear Linear

| | |
|---|---|
|Shif|ted|
| | |

Pooled

DiffusionTransformer

MLP

- Swin-MMDiT-Block 1
- Swin-MMDiT-Block 2

|… …<br><br>…<br><br>…<br><br>𝑄𝑡𝑒𝑥𝑡<br><br>𝐶𝑎𝑡(𝐾……𝑋′,𝐾𝑡𝑒𝑥𝑡)<br><br>Text Attention|
|---|

+

…

MLP

[Figure 13]

Sinusoidal Encoding

Swin-MMDiT-Block n

Timestep

|Swin-MMDiT-Block n+1 Video Attention Text Attention<br><br>[Figure 14]<br><br>[Figure 15]|
|---|

Modulation Linear

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

…

Unpatching & Unflatten

(a) Overall Architecture (b) Swin-MMDiT Details

- Figure 2. Model architecture and the details of Swin-MMDiT of SeedVR. Our approach introduces a shifted window mechanism into the transformer block, bypassing the resolution constrain of vanilla attention. We further adopt large attention windows around the center and variable-sized windows near the boundary, enabling long-range dependency capturing given inputs of any length and size.

(Cat(QX′,Qtext),Cat(KX′,Ktext),Cat(VX′,Vtext)).

Instead of using the standard full attention, our SwinMMDiT employs two types of window attention: a regular window attention starting from the top-left unflattened pixel of X, and a shifted window attention, offset by half the window size from the regular windows.

As shown in Figure 1b, the first transformer block uses regular window attention with a t × h × w window. Specifically, the video feature X is first divided into (Tt + 1) × (Hh + 1) × (Ww + 1) windows, with some windows smaller than t × h × w. In Swin Transformer [35, 36], a cyclicshifting strategy with a masking mechanism is required to make the window size divisible by the feature map size. In contrast, our Swin-MMDiT benefits from the flexibility of NaViT and Flash attention [14]. Here, the partitioned window features are flattened into a concatenated 2D tensor, and attention is calculated within each window, eliminating the need for complex masking strategies on the 3D feature map. The subsequent transformer block applies shifted window attention, where windows are offset by (2t, h2, w2 ) before attention is calculated similarly to regular window attention. For attention calculations, we replace the absolute 2D positional frequency embeddings used in SD3 with 3D relative rotary positional embeddings (RoPE) [48] within each win-

dow, avoiding the resolution bias introduced by positional embeddings.

As shown in Figure 2, for simplicity, we use separate attention mechanisms for video and text features instead of the single multi-modality attention in MMDiT [17]. Specifically, the key and value of the video window features and text features are concatenated. We then compute attention by calculating the similarity between the concatenated key and value with the query of the video window and text features, respectively. This approach does not increase computational cost, and in practice, we observe no significant drop in performance.

##### 3.2. Causal Video VAE

To process video input, existing diffusion-based VR methods [20, 64, 82] typically fine-tune a pretrained image autoencoder for video by inserting 3D convolution layers. Without temporal compression, these video autoencoders are inefficient for both training and inference. Moreover, the limited number of latent channels ( i.e., 4) prevents these autoencoders from reconstructing videos with high quality. Instead of fine-tuning a pretrained image autoencoder, we train a video autoencoder from scratch with the following improvements: 1) We use a causal 3D residual block rather than a

|ResBlock3D<br><br>GroupNorm<br><br>CausalConv3D<br><br>GroupNorm<br><br>CausalConv3D|
|---|

|CausalConv3D| |ResBlock3D|ResBlock3D|Down| |ResBlock3D|ResBlock3D|Down| |ResBlock3D|ResBlock3D|Down| |ResBlock3D|ResBlock3D| |Attn.|ResBlock3D|ResBlock3D| |GroupNorm|CausalConv3D|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | |Spat.| | | |Spat.-Temp.| | | |Spat.-Temp.| | | | |Spat.| | | | | |

|CausalConv3D| |ResBlock3D|ResBlock3D|ResBlock3D|Up| |ResBlock3D|ResBlock3D|ResBlock3D|Up| |ResBlock3D|ResBlock3D|ResBlock3D|Up| |ResBlock3D|ResBlock3D|ResBlock3D| |Attn.|ResBlock3D|ResBlock3D| |GroupNorm|CausalConv3D|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |Spat.-Temp.| | | | |Spat.-Temp.| | | | |Spat.| | | | | |Spat.| | | | | |

Spat.-Temp.Down

Spat.-Temp.Down

Spat.-Temp.Up

Spat.-Temp.Up

CausalConv3D

CausalConv3D

CausalConv3D

CausalConv3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

ResBlock3D

GroupNorm

GroupNorm

Spat.Down

Spat.Attn.

Spat.Attn.

Spat.Up

- Figure 3. The model architecture of casual video autoencoder. In contrast to naively inflating an existing image autoenoder, we redesign a casual video VAE with spatial-temporal compression capability to achieve a strong reconstruction capability.

vanilla 3D block to capture spatial-temporal representations. In this way, our video autoencoder is capable of handling long videos by cutting them into clips. 2) We increase the latent channels to 16 following SD3 [17], to increase the model capacity for better reconstruction. 3) We apply a temporal compression factor of 4 for more efficient video encoding. The overall architecture is shown in Figure 3. We follow the common practice [17] to train our casual video VAE on a large dataset with ℓ1 loss, LPIPS loss [75] and GAN loss [18].

LQ conditions, which is crucial for training real-world VR models. Furthermore, eliminating the need to load pretrained VAE and text models saves GPU memory, allowing for a larger batch size for training.

Progressively Growing Up of Resolution and Duration. Our model is trained based on SD3-Medium [17] with 2.2B parameters. Although SD3-Medium handles high resolutions, e.g., 1024 × 1024, we found it challenging to directly adapt it into a VR model with our architecture at that resolution. Instead, we begin by tuning on short, low-resolution videos (5 frames at 256 × 256) and progressively increase the lengths and resolutions to (9 frames at 512 × 512) and eventually 21 frames at 768×768. The final model is trained on data with varying lengths and resolutions. We observe a rapid convergence with this progressive tuning strategy.

##### 3.3. Large-scale Training

Training a large-scale VR model on millions of highresolution videos is challenging and remains under-explored. Existing VR approaches [20, 64, 82] are trained on limited resources, which inevitably hinders their ability to generalize to more complex, real-world VR tasks. Besides model architectures, we extend our exploration to include more diverse training data and training strategies to scale up the training. Large-scale Mixed Data of Images and Videos. By virtue of the flexibility of our model architecture, we can train the model on image and video data simultaneously. To this end, we first collect a large-scale mixed dataset of images and videos. Specifically, we collect about 10 million images and 5 million videos. The images vary in resolution, with most exceeding 1024 × 1024 pixels. The videos are 720p, randomly cropped from higher-resolution videos to improve training efficiency. In practice, we observe that cropping yields better performance than resizing. To ensure highquality data, we further apply several evaluation metrics [1, 26, 53, 58] to filter out low-quality samples.

Injecting Noise to Condition. We follow existing methods [6, 56, 82] to create synthetic LQ-HQ image and video pairs for training. While effective, we observe a degradation gap between synthetic LQ videos and real-world ones, as synthetic videos typically exhibit much more severe degradations than those found in real-world videos. Simply lowering the degradation level for synthetic training data could weaken the model’s generative ability, so instead, we follow the strategy of injecting random noise to the latent LQ condition [3, 82]. This is done by diffusing the condition as CLQτ = ατCLQ + στϵ, where ϵ ∼ N(0,I), τ is the noise level associated with the early steps in the noise schedule defined by αt and σt.

Besides adding noise to the LQ condition, we enable the flexible use of the text encoder by randomly replacing the text input to each of the three text encoders with null prompts, similar to SD3 [17]. Although a similar approach could be applied to LQ conditions to enhance the model’s generative capability, we found that excessively strong generative ability often results in reduced output fidelity. Therefore, we opted not to include it in the final model.

Precomputing Latents and Text Embeddings. Training on high-resolution data poses challenges for training efficiency due to the slow encoding speed required to convert large videos into latent space with the pretrained VAE. In practice, encoding a 720p video with 21 frames takes approximately

- 2.9s on average, roughly as long as a single forward pass of the diffusion transformer model. In addition, encoding the low-quality (LQ) condition also requires VAE processing, doubling the encoding time per training iteration. By precomputing high-quality (HQ) and LQ video latent features along with text embeddings, we can achieve a 4× speed up in training. Applying diverse degradations on large-scale data also ensures sufficient random degradations applied to

#### 4. Experiments

Implementation Details. We train SeedVR on 256 NVIDIA H100-80G GPUs with around 150 720p frames per batch per GPU. We initialize the model parameters from SD3Medium [17] and train the full model following the strategies discussed in Sec. 3.3. We mostly follow the training

- Table 1. Quantitative comparisons on VSR benchmarks from diverse sources, i.e., synthetic (SPMCS, UDM10, REDS30, YouHQ40), real (VideoLQ), and AIGC (AIGC38) data. The best and second performances are marked in red and orange , respectively.

Datasets Metrics Real-ESRGAN [56] SD ×4 Upscaler [2] ResShift [74] RealViFormer [77] MGLD-VSR [64] Upscale-A-Video [82] VEhancer [20] Ours

|SPMCS<br><br>|PSNR ↑ SSIM ↑ LPIPS ↓ DISTS ↓<br><br>|22.55 0.637 0.406 0.189 3.355 62.78 0.451 8.566<br><br>|22.75 0.535 0.554 0.247 5.883 42.09 0.402 4.413<br><br>|23.14 0.598 0.547 0.261 6.246 55.11 0.598 5.342<br><br>| |24.19 0.663 0.378 0.186 3.431 62.09 0.424 7.664<br><br>|23.41 0.633 0.369 0.166 3.315 65.25 0.495 8.471<br><br>|21.69 0.519 0.508 0.229 3.272 65.01 0.507 6.237<br><br>|18.20 0.507 0.455 0.194 4.328 54.94 0.334 7.807|22.37 0.607 0.341 0.141 3.207 64.28 0.587 10.508<br><br>|
|---|---|---|---|---|---|---|---|---|---|---|
| |NIQE ↓ MUSIQ ↑ CLIP-IQA ↑ DOVER ↑<br><br>| | | | | | | | | |
|UDM10<br><br>|PSNR ↑ SSIM ↑ LPIPS ↓ DISTS ↓|24.78 0.763 0.270 0.156 4.365 54.18 0.398 7.958<br><br>|26.01 0.698 0.424 0.234 6.014 30.33 0.277 3.169<br><br>|25.56 0.743 0.417 0.211 5.941 51.34 0.537 5.111<br><br>| |26.70 0.796 0.285 0.166 3.922 55.60 0.397 7.259<br><br>|26.11 0.772 0.273 0.144 3.814 58.01 0.443 7.717<br><br>|24.62 0.712 0.323 0.178 3.494 58.31 0.458 9.238<br><br>|21.48 0.691 0.349 0.175 4.883 46.37 0.304 8.087|25.76 0.771 0.231 0.116 3.514 59.14 0.524 10.537<br><br>|
| |NIQE ↓ MUSIQ ↑ CLIP-IQA ↑ DOVER ↑<br><br>| | | | | | | | | |
|REDS30<br><br>|PSNR ↑ SSIM ↑ LPIPS ↓ DISTS ↓<br><br>|21.67 0.573 0.389 0.179 2.879 57.97 0.403 5.552|22.94 0.563 0.551 0.268 6.718 25.57 0.202 2.737<br><br>|22.72 0.572 0.509 0.234 6.258 47.50 0.554 3.712<br><br>| |23.34 0.615 0.328 0.154 3.032 58.60 0.392 5.229<br><br>|22.74 0.578 0.271 0.097 2.550 62.28 0.444 6.544<br><br>|21.44 0.514 0.397 0.181 2.561 56.39 0.398 5.234<br><br>|19.83 0.545 0.508 0.229 4.615 37.95 0.245 5.549|20.44 0.534 0.346 0.138 2.729 57.55 0.451 6.673<br><br>|
| |NIQE ↓ MUSIQ ↑ CLIP-IQA ↑ DOVER ↑<br><br>| | | | | | | | | |
|YouHQ40|PSNR ↑ SSIM ↑ LPIPS ↓ DISTS ↓<br><br>|22.31 0.605 0.342 0.169 3.721 56.45 0.371 10.92<br><br>|22.51 0.528 0.518 0.242 5.954 36.74 0.328 5.761<br><br>|22.67 0.579 0.432 0.215 5.458 54.96 0.590 7.618<br><br>| |23.26 0.606 0.362 0.193 3.172 61.88 0.438 9.483<br><br>|22.62 0.576 0.356 0.166 3.255 63.95 0.509 10.503<br><br>|21.32 0.503 0.404 0.196 3.000 64.450 0.471 9.957<br><br>|18.68 0.510 0.449 0.175 4.161 54.18 0.352 11.444<br><br>|21.15 0.554 0.298 0.118 2.913 67.45 0.635 12.788<br><br>|
| |NIQE ↓ MUSIQ ↑ CLIP-IQA ↑ DOVER ↑<br><br>| | | | | | | | | |

|VideoLQ<br><br>|NIQE ↓ MUSIQ ↑ CLIP-IQA ↑ DOVER ↑<br><br>|4.014 60.45 0.361 8.561<br><br>|4.584 43.64 0.296 4.349|4.829 59.69 0.487 6.749<br><br>| |4.007 57.50 0.312 6.823|3.888 59.50 0.350 7.325<br><br>|3.490 58.31 0.371 7.090<br><br>|4.264 52.59 0.289 8.719<br><br>|3.874 54.41 0.355 8.009|
|---|---|---|---|---|---|---|---|---|---|---|
|AIGC38|NIQE ↓ MUSIQ ↑ CLIP-IQA ↑ DOVER ↑<br><br>|4.942 58.39 0.442 12.275|4.399 56.72 0.554 10.547<br><br>|4.853 64.38 0.660 12.082<br><br>| |4.444 58.73 0.473 10.245<br><br>|4.162 62.03 0.528 11.008|4.124 63.15 0.497 12.857<br><br>|4.759 53.36 0.395 12.178|3.955 65.91 0.638 13.424<br><br>|

settings in SD3 [17] to train the diffusion transformer. We follow Upscale-A-Video [82] to synthesize training pairs. The entire training process requires about 30K H100-80G GPU hours. As for the training of our casual video VAE, we follow the standard settings in SD3 [17] and train on internal data with a resolution of 17 × 256 × 256. The model is trained on 32 NVIDIA H100-80G GPUs with a batch size of 5 per GPU for 115,000 iterations.

Experimental Settings. We use various metrics to evaluate both the frame quality and overall video quality. For synthetic datasets with LQ-HQ pairs, we employ fullreference metrics such as PSNR, SSIM, LPIPS [76], and DISTS [16], and no-reference metrics including NIQE [37], CLIP-IQA [53], MUSIQ [26], and DOVER [59]. For realworld and AIGC test data, we adopt no-reference metrics, i.e., NIQE, CLIP-IQA, MUSIQ, and DOVER due to the absence of ground truth. To ensure fair comparisons, all testing videos are processed to be 720p while maintaining the origi-

nal length. We follow previous work [82] to test on synthetic benchmarks i.e., SPMCS [68], UDM10 [49], REDS30 [38], and YouHQ40 [82], using the same degradations as training. Additionally, we evaluate the models on a real-world dataset (i.e., VideoLQ [6]) and an AIGC dataset (i.e., AIGC38) that collects 38 AI-generated videos.

##### 4.1. Comparison with Existing Methods

Quantitative Comparisons. As shown in Table 1, our method achieves significantly superior performance on 4 out of 6 benchmarks (i.e., SPMCS, UDM10, YouHQ40, and AIGC38). Similar to other diffusion-based methods, our SeedVR shows limitations on certain metrics like PSNR and SSIM [54, 70, 72] on these benchmarks because these metrics are primarily designed to measure pixel-level fidelity and structural similarity, while SeedVR focuses on perceptual quality. As can be observed, SeedVR obtains high DISTS, LPIPS, NIQE and DOVER scores, indicating the high per-

[Figure 21]

- Figure 4. Qualitative comparisons on both real-world videos in VideoLQ [6] dataset (the first and second row) and AIGC dataset (the third and fourth row). Our SeedVR is capable of generating realistic details. When compared to existing methods, it notably excels in its restoration capabilities, successfully removing the degradations while maintaining the textures of the buildings, the panda’s nose and the face of the terracotta warrior. (Zoom-in for best view).

ceptual quality of its generated results. It is worth noting that MGLD-VSR and RealViFormer are trained on REDS [38], which explains their strong performance on the corresponding test set, REDS30. Even so, our approach remains competitive, achieving the best DOVER score on REDS30. The consistent superiority across datasets from various sources demonstrates the effectiveness of our method.

Qualitative Comparisons. Figure 4 shows visual results

on both real-world [6] and AIGC videos. Our seedVR outperforms existing VR approaches by a large margin in both degradation removal and texture generation. Specifically, SeedVR effectively recovers detailed structures, such as the building architectures with severely degraded video inputs. For AIGC videos, SeedVR faithfully restores fine details, such as the panda’s nose and the terracotta warrior’s face in Figure 4, where other approaches produce blurred details.

- Table 2. Quantitative comparisons on VAE models commonly used in existing latent diffusion models [20, 28, 44, 45, 66, 80]. The best and second performances are marked in red and orange , respectively.

Methods (VAE)

Params (M)

Temporal Compression

Spatial Compression

Latent Channel

PSNR ↑ SSIM ↑ LPIPS ↓ rFVD ↓

|SD 2.1 [45] VEnhancer [20] Cosmos [44] OpenSora [80] OpenSoraPlan v1.3 [28] CV-VAE (SD3) [79] CogVideoX [66]|83.7 97.7 90.2 393.3 147.3 181.9 215.6<br><br>|4 4 4 4 4|8 8 8 8 8 8 8<br><br>|4 4 16 4 16 16 16<br><br>|29.50 30.81 32.34 27.70 30.41 33.21 34.30<br><br>|0.9050 0.9356 0.9484 0.8893 0.9280 0.9612 0.9650<br><br>|0.0998 0.0751 0.0847 0.1661 0.0976 0.0589 0.0623<br><br>|8.14 11.10 13.02 47.04 27.70 6.50 6.06<br><br>|
|---|---|---|---|---|---|---|---|---|
|Ours|250.6|4<br><br>|8|16<br><br>|33.83|0.9643|0.0517|1.85|

##### 4.2. Ablation Study

Effectiveness of Casual Video VAE. We first examine the significance of the proposed casual video VAE. As shown in Table 2, our VAE demonstrates better video reconstruction quality. Compared to state-of-the-art VAE models for video generation and restoration, our VAE reaches the lowest rFVD [51] score, 69.5% lower than the second best. Besides, our VAE achieves the best LPIPS score and competitive PSNR and SSIM relative to CogVideoX [66], indicating its superior reconstruction capability.

Window Size for Attention. Besides the powerful VAE, a key aspect of our design is the flexible window attention, which enables restoration at arbitrary resolutions. We measure the performance with different window sizes. Specifically, we train the model using different window sizes under the same settings for 12.5k iterations. Results show that smaller window sizes significantly increase training time. As shown in Table 3, training time rises considerably with smaller windows; for instance, with 1 × 8 × 8 window, the training time required is 455.49, which is 19.24 times longer than a 1 × 64 × 64 window. This increase is due to each window being assigned a text prompt in the attention computation, introducing text guidance while retaining flexibility for arbitrary resolutions. Therefore, using larger window sizes reduces the number of text tokens required for attention, improving both training and inference efficiency.

We further validate the performance under different window sizes on the YouHQ40 dataset [82] measured by DOVER [59]. From Table 4, we make the following observations: 1) The performance of full spatial attention declines as the temporal window length increases. We believe this is due to the high token count in full attention, which requires a much longer training period, far beyond 12.5k iterations, to converge fully. Larger temporal windows amplify this need. 2) Smaller spatial windows, e.g., 32 × 32, outperform full attention, but still show a performance drop as temporal length increases. We hypothesize that smaller temporal and spatial windows, like e.g., 1 × 32 × 32, allow for faster convergence, leading to better performance. However, smaller spatial windows may face difficulties in capturing tempo-

- Table 3. Training efficiency (sec/iter) with different window sizes.

|Temp. Win. Length|Spat. Win. Size| | | |
|---|---|---|---|---|
| |8 × 8<br><br>|16 × 16<br><br>|32 × 32|64 × 64<br><br>|
|t = 1 t = 5|455.49 345.78<br><br>|138.29 110.01|58.37 46.49<br><br>|23.68 20.29|

- Table 4. Ablation study on the performance of different window sizes. All baselines are trained on 16 NVIDIA A100-80G cards for 12.5k iterations. The comparison is conducted on YouHQ40 [82] synthetic data and measured by DOVER (↑) [59].

.

|Temp. Win. Length|Spat. Win. Size| | |
|---|---|---|---|
| |32 × 32|64 × 64<br><br>|Full|
|t = 1 t = 3 t = 5|11.947 11.476 10.558<br><br>|10.690 10.429 11.595|10.799 9.145 8.521<br><br>|

ral dependencies, requiring additional training to prevent performance degradation. 3) For a spatial window size of 64 × 64, performance is comparable with shorter temporal lengths, i.e., 1 and 3. Increasing the window length to 5 notably improves results, likely because the larger window size captures long-range dependencies and enhances semantic alignment between text prompts and restoration. These observations validate our design choice of using a 5 × 64 × 64 attention window.

- 5. Conclusion

We have presented SeedVR, a novel diffusion transformer model designed as foundational architecture to tackle highquality VR with arbitrary resolutions and lengths. SeedVR builds on the strengths of existing diffusion-based methods, yet addresses key limitations through a flexible architecture that combines a large attention window, a causal video autoencoder, and efficient training strategies. Extensive experiments demonstrate SeedVR’s superior ability to handle both synthetic and real-world degradations, with improved visual realism and detail consistency across frames. Notably, SeedVR is over twice as fast as existing methods despite its larger parameter size. In the future, we will improve the sampling efficiency and robustness of SeedVR.

Acknowledgement: This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-PhD-2022-01033[T]). We sincerely thank Zhibei Ma for data processing, Jiashi Li for hardware maintenance. We also give thanks to Shangchen Zhou and Zongsheng Yue for research discussions.

#### References

- [1] Laion-aesthetics. https://laion.ai/blog/laionaesthetics/, 2022. 5
- [2] Stable Diffusion x4 Upscaler. https://huggingface. co / stabilityai / stable - diffusion - x4 upscaler, 2023. 1, 6
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 5
- [4] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. BasicVSR: The search for essential components in video super-resolution and beyond. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2, 3
- [5] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. BasicVSR++: Improving video superresolution with enhanced propagation and alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3
- [6] Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2022. 2, 3, 5, 6, 7

- [7] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 3
- [8] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 3
- [9] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [10] Ting Chen and Lala Li. FIT: Far-reaching interleaved transformers. arXiv preprint arXiv:2305.12689, 2023. 3
- [11] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating more pixels in image super-resolution transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3

- [12] Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, and Tao Mei. Learning spatial adaptation and temporal coherence in diffusion models for video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3
- [13] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [14] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In Proceedings of International Conference on Learning Representations (ICLR), 2024. 4
- [15] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 3
- [16] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020. 6
- [17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of International Conference on Machine Learning (ICML),

2024. 2, 3, 4, 5, 6

- [18] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2014. 5
- [19] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. In Proceedings of the European Conference on Computer Vision (ECCV), 2025. 3
- [20] Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024. 1, 2, 3, 4, 5, 6, 8
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2020. 3
- [22] Yan Huang, Wei Wang, and Liang Wang. Bidirectional recurrent convolutional networks for multi-frame super-resolution. Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2015. 3
- [23] Takashi Isobe, Xu Jia, Shuhang Gu, Songjiang Li, Shengjin Wang, and Qi Tian. Video super-resolution with recurrent structure-detail network. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. 3
- [24] Alvaro´ Barbero Jim´enez. Mixture of diffusers for scene composition and high resolution image generation. arXiv preprint arXiv:2302.02412, 2023. 2, 3

- [25] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. 3
- [26] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 5, 6
- [27] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. VideoPoet: A large language model for zero-shot video generation. In Proceedings of International Conference on Machine Learning (ICML),

2024. 3

- [28] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan. https: //doi.org/10.5281/zenodo.10948109, 2024. 8
- [29] Dasong Li, Xiaoyu Shi, Yi Zhang, Ka Chun Cheung, Simon See, Xiaogang Wang, Hongwei Qin, and Hongsheng Li. A simple baseline for video restoration with grouped spatialtemporal shift. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [30] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-DiT: A powerful multiresolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 3
- [31] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. SwinIR: Image restoration using swin transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCV-W), 2021. 2, 3
- [32] Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V Gool. Recurrent video restoration transformer with guided deformable attention. Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 3
- [33] Jingyun Liang, Jiezhang Cao, Yuchen Fan, Kai Zhang, Rakesh Ranjan, Yawei Li, Radu Timofte, and Luc Van Gool. VRT: A video restoration transformer. IEEE Transactions on Image Processing (TIP), 2024. 2, 3
- [34] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 3
- [35] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2, 3, 4
- [36] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 4

- [37] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 2012. 6
- [38] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. NTIRE 2019 challenge on video deblurring and super-resolution: Dataset and study. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (CVPR-W), 2019. 6, 7
- [39] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In Proceedings of International Conference on Machine Learning (ICML), 2022. 3
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 3
- [41] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 3

- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of International Conference on Machine Learning (ICML), 2021. 3
- [43] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research (JMLR), 2020. 3
- [44] Fitsum Reda, Jinwei Gu, Xian Liu, Songwei Ge, Ting-Chun Wang, Haoxiang Wang, and Ming-Yu Liu. Cosmos tokenizer: A suite of image and video neural tokenizers. https:// github.com/NVIDIA/Cosmos-Tokenizer, 2024. 8
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3, 8
- [46] Mehdi SM Sajjadi, Raviteja Vemulapalli, and Matthew Brown. Frame-recurrent video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 3
- [47] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of International Conference on Machine Learning (ICML), 2015. 3
- [48] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024. 2, 4
- [49] Xin Tao, Hongyun Gao, Renjie Liao, Jue Wang, and Jiaya Jia. Detail-revealing deep video super-resolution. In Proceed-

- ings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2017. 6
- [50] Yapeng Tian, Yulun Zhang, Yun Fu, and Chenliang Xu. TDAN: Temporally-deformable alignment network for video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2020. 3

- [51] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 8
- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Proceedings of Advances in Neural Information Processing Systems (NeurIPS),

2017. 3

- [53] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence,

2023. 5, 6

- [54] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C.K. Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision (IJCV), 2024. 2, 3, 6
- [55] Xintao Wang, Kelvin CK Chan, Ke Yu, Chao Dong, and Chen Change Loy. Edvr: Video restoration with enhanced deformable convolutional networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (CVPR-W), 2019. 2, 3
- [56] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCV-W),

2021. 2, 5, 6

- [57] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional video synthesis with motion controllability. Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2024. 3
- [58] Haoning Wu, Chaofeng Chen, Jingwen Hou, Liang Liao, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Fast-vqa: Efficient end-to-end video quality assessment with fragment sampling. Proceedings of the European Conference on Computer Vision (ECCV), 2022. 5
- [59] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 6, 8
- [60] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. SeeSR: Towards semantics-aware real-world image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3
- [61] Liangbin Xie, Xintao Wang, Shuwei Shi, Jinjin Gu, Chao Dong, and Ying Shan. Mitigating artifacts in real-world

- video super-resolution models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023. 2, 3
- [62] S. Yang, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In Proceedings of International Conference on Learning Representations (ICLR), 2021. 3
- [63] Xi Yang, Wangmeng Xiang, Hui Zeng, and Lei Zhang. Realworld video super-resolution: A benchmark dataset and a decomposition based learning scheme. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2
- [64] Xi Yang, Chenhang He, Jianqi Ma, and Lei Zhang. Motionguided latent diffusion for temporally consistent real-world video super-resolution. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 1, 2, 3, 4, 5, 6
- [65] Zhuoyi Yang, Heyang Jiang, Wenyi Hong, Jiayan Teng, Wendi Zheng, Yuxiao Dong, Ming Ding, and Jie Tang. Inf-DiT: Upsampling any-resolution image with memory-efficient diffusion transformer. arXiv preprint arXiv:2405.04312, 2024. 3
- [66] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 8
- [67] Chang-Han Yeh, Chin-Yang Lin, Zhixiang Wang, ChiWei Hsiao, Ting-Hsuan Chen, and Yu-Lun Liu. Diffir2vrzero: Zero-shot video restoration with diffusion-based image restoration models. arXiv preprint arXiv:2407.01519, 2024. 2
- [68] Peng Yi, Zhongyuan Wang, Kui Jiang, Junjun Jiang, and Jiayi Ma. Progressive fusion video super-resolution network via exploiting non-local spatio-temporal correlations. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 6
- [69] Geunhyuk Youk, Jihyong Oh, and Munchurl Kim. FMA-Net: Flow-guided dynamic filtering and iterative feature refinement with multi-attention for joint video super-resolution and deblurring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [70] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3, 6
- [71] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In Proceedings of International Conference on Learning Representations (ICLR), 2024. 2
- [72] Zongsheng Yue and Chen Change Loy. Difface: Blind face restoration with diffused error contraction. arXiv preprint arXiv:2212.06512, 2022. 6

- [73] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image super-resolution by residual shifting. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2023. 2
- [74] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Efficient diffusion model for image restoration by residual shifting. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024. 2, 6
- [75] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

- (CVPR), 2018. 5

[76] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

- (CVPR), 2018. 6

- [77] Yuehan Zhang and Angela Yao. RealViformer: Investigating attention for real-world video super-resolution. Proceedings of the European Conference on Computer Vision (ECCV),

2024. 3, 6

- [78] Zhenghao Zhang, Junchao Liao, Menghao Li, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. arXiv preprint arXiv:2407.21705, 2024. 3
- [79] Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, and Ying Shan. Cv-vae: A compatible video vae for latent generative video models. Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2024. 8
- [80] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. https://github.com/hpcaitech/OpenSora, 2024. 8
- [81] Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li, and Chen Change Loy. Towards robust blind face restoration with codebook lookup transformer. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS),

2022. 2

- [82] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-A-Video: Temporalconsistent diffusion model for real-world video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 3, 4, 5, 6, 8
- [83] Yupeng Zhou, Zhen Li, Chun-Le Guo, Song Bai, Ming-Ming Cheng, and Qibin Hou. SRFormer: Permuted self-attention for single image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 3
- [84] Yupeng Zhou, Zhen Li, Chun-Le Guo, Li Liu, Ming-Ming Cheng, and Qibin Hou. SRFormerV2: Taking a closer look at permuted self-attention for image super-resolution. arXiv preprint arXiv:2303.09735, 2024. 3

