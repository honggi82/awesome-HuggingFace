# arXiv:2505.12667v2[cs.CV]22Sep2025

## Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking

Zihan Su1 Xuerui Qiu2 Hongbin Xu3 Tangyu Jiang1 Junhao Zhuang1 Chun Yuan1∗ Ming Li4∗ Shengfeng He5 Fei Richard Yu4

1 Tsinghua University 2 Institute of Automation, Chinese Academy of Sciences

3 South China University of Technology 4 Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ) 5 Singapore Management University zh-su24@mails.tsinghua.edu.cn

#### Abstract

The explosive growth of generative video models has amplified the demand for reliable copyright preservation of AI-generated content. Despite its popularity in image synthesis, invisible generative watermarking remains largely underexplored in video generation. To address this gap, we propose Safe-Sora, the first framework to embed graphical watermarks directly into the video generation process. Motivated by the observation that watermarking performance is closely tied to the visual similarity between the watermark and cover content, we introduce a hierarchical coarse-to-fine adaptive matching mechanism. Specifically, the watermark image is divided into patches, each assigned to the most visually similar video frame, and further localized to the optimal spatial region for seamless embedding. To enable spatiotemporal fusion of watermark patches across video frames, we develop a 3D wavelet transform-enhanced Mamba architecture with a novel spatiotemporal local scanning strategy, effectively modeling long-range dependencies during watermark embedding and retrieval. To the best of our knowledge, this is the first attempt to apply state space models to watermarking, opening new avenues for efficient and robust watermark protection. Extensive experiments demonstrate that Safe-Sora achieves state-of-the-art performance in terms of video quality, watermark fidelity, and robustness, which is largely attributed to our proposals. Code is publicly available at https://github.com/Sugewud/Safe-Sora

#### 1 Introduction

Recent advances in video generation models have significantly transformed digital content creation [1– 5]. VideoCrafter2 [2] delivers high-fidelity video generation results, while Open-Sora [6] enables efficient and scalable video generation. However, this rapid progress also raises growing concerns over copyright protection and ownership verification of generated videos.

Invisible watermarking has proven effective for copyright protection in image generation [7–14]. However, its extension to video generation remains relatively underexplored. Recent efforts such as VideoShield [15] and LVMark [16] embed watermarks by modifying latent noise or applying importance-based modulation strategies. Despite these advancements, existing approaches rely on embedding bitstring-based identifiers, which fall short of leveraging the high information capacity inherent in video content. Unlike static images, videos offer significantly greater embedding bandwidth, making them well-suited for graphical watermarks—e.g., logos or icons—that serve as more intuitive and visually recognizable evidence of ownership. Such designs enhance both the perceptual clarity and practical reliability of copyright verification.

∗Corresponding author.

Preprint. Under review.

Watermark Cover Image Watermarked Image

Extracted Watermark

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

LPIPS = 0.934 PSNR = 15.93 PSNR = 23.36

[Figure 5]

[Figure 6]

[Figure 7]

LPIPS = 0.556 PSNR = 33.12 PSNR = 31.16

- Figure 1: Impact of image-watermark similarity on watermarking performance. We used a pretrained classic image hiding network Balujanet [17] on 1,000 image pairs, each consisting of a graphical watermark from Logo-2k [18] and a cover image from ImageNet [19]. Image-Watermark similarity was quantified using 1-LPIPS and the quality of the watermarked image and extracted watermark was evaluated using PSNR. Higher PSNR and lower LPIPS indicate improved performance.

Recognizing the untapped potential of graphical watermarking in video generation, we propose Safe-Sora, the first framework, to the best of our knowledge, that embeds graphical watermarks elegantly into the video generation process. As illustrated in Fig. 1, we observe that watermarking performance significantly correlates with the visual similarity between the watermark and cover images. In particular, embedding becomes significantly more effective when the cover image shares high visual similarity with the watermark content. Motivated by this, we propose a hierarchical coarse-to-fine adaptive matching mechanism, which first divides the watermark image into patches and assigns each patch to the most similar video frame through an inter-frame automatic selection strategy. Subsequently, an intra-frame localization is performed to embed the patch into the most visually similar region within the selected frame. To address the challenge of fusing and extracting watermark information distributed across spatiotemporal locations, we further propose a 3D wavelet transform-enhanced Mamba architecture with a tailored scanning strategy. This design enables bidirectional modeling across frequency subbands in the 3D wavelet transform, effectively and efficiently capturing long-range dependencies in both space and time. To the best of our knowledge, this is the first application of state space models to generative watermarking.

In our experiments, we utilize the widely-used Panda-70M [20] dataset as the video source due to its extensive scale and diverse video categories. For graphical watermarks, we employ the Logo-2K+ [18] dataset, which offers a wide variety of real-world logos. The quantitative and qualitative comparisons with existing methods demonstrate that the proposed Safe-Sora achieves state-of-the-art performance in terms of video quality, watermark fidelity, and robustness. For instance, our method achieves a Fréchet Video Distance of 3.77, far lower than the second-best baseline’s 154.35, highlighting its superior temporal consistency. The ablation experiments showcase the effectiveness of our proposals.

Our primary contributions can be summarized as follows:

- • We introduce the first model specifically designed to embed graphical watermarks in video generation pipelines, directly addressing the pressing need for copyright protection of generated video content.
- • We propose a hierarchical coarse-to-fine adaptive matching mechanism that strategically embeds watermark patches into visually similar frames and spatial regions, enhancing overall watermarking performance.
- • We pioneer the application of state space models for watermarking through a novel 3D wavelet transform-enhanced Mamba architecture with a tailored scanning strategy, enabling enhanced fusion and extraction of watermark information across space and time.

#### 2 Related Work

##### 2.1 Video Diffusion Models

Diffusion models [21–25] are a class of generative models that synthesize data through a gradual denoising process, beginning from randomly sampled Gaussian noise. Latent Video Diffusion Models

Video Reconstruction Loss

: Freeze

[Figure 8]

E : Latent Encoder D : Latent Decoder

Multi-Scale Feature Injection

: Loss Flow : Multi-Scale Feature

Conv&Att.

Conv&Att.

Conv&Att.

[Figure 9]

[Figure 10]

[Figure 11]

###### E D

| | |
|---|---|
| | |
| | |

Frame-wiseConcat

PositionRecovery

DistortionLayer

[Figure 12]

3DSFMamba

|[Figure 13]<br><br>[Figure 14]|
|---|

3DSFMamba

Video Latent

2DSFMamba

2DSFMamba

2DSFMamba

2DSFMamba

2DSFMamba

2DSFMamba

Original Video

Down

Down

Down

Down

Up

Up

Up

PatchEmbedding

|[Figure 15]<br><br>[Figure 16]| |
|---|---|
| | |

Coarse-to-Fine Adaptive Patch Matching

Watermarked Video

Extracted Watermark

× N

× M

Up

Watermark

Watermark Reconstruction Loss

Coarse-to-Fine Adaptive Patch Matching

###### 3D Frequency Mamba

###### SFMamba

Forward Scanning Backward Scanning

Low

| | | | |
|---|---|---|---|
| |[Figure 17]| | |
| | | | |
| | | | |

| | | |
|---|---|---|
|[Figure 18]| | |
| | | |

| | | |
|---|---|---|
| | |[Figure 19]|
| | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 20]| | | |
| | | | |

......

DWConv SiLU

High High

High

LayerNorm

LLL

HHH

Frequency

3D Frequency Scanning

Fine Router Fine Router Fine Router Fine Router

Conv 1x1

FrequencyBranch

SpatialBranch

2/3D DWT

SiLU

SiLU

LLL LLH

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

LLH

HHL

......

SSM LayerNorm

2/3D Spatial Mamba

2/3D Frequency Mamba

HLL HLH

LHL

HLH

2/3D IDWT

LHL LHH

Coarse Router

SSM

Watermark

HHL HHH

HLL LHH

[Figure 25]

|[Figure 26]<br><br>[Figure 27]| |
|---|---|
| | |

[Figure 28]

[Figure 29]

| | |
|---|---|
|[Figure 30]<br><br>[Figure 31]| |

| | |
|---|---|
|[Figure 32]<br><br>[Figure 33]| |

| | | |
|---|---|---|
|[Figure 34]<br><br>[Figure 35]| |N-1|

| | | |
|---|---|---|
|[Figure 36]<br><br>[Figure 37]| |N|

| | | |
|---|---|---|
|[Figure 38]<br><br>[Figure 39]| |0|

| | | |
|---|---|---|
|[Figure 40]<br><br>[Figure 41]| |2|

Concat & Conv 1x1

[Figure 42]

3D Wavelet Transform

1 N-2

Patch Partition & Add Position Channel

Spatiotemporal Local Scanning Strategy

- Figure 2: Overview of our Safe-Sora framework. Our method consists of three main components:

(1) Coarse-to-Fine Adaptive Patch Matching: partitioning the watermark image into patches and optimally assigning them to appropriate video frames and regions, followed by patch embedding and upsampling to generate the watermark feature map; (2) Watermark Embedding: the watermark feature map is fused with multi-scale video features via a UNet with 2D SFMamba blocks, followed by a series of 3D SFMamba blocks that implement our spatiotemporal local scanning strategy, to produce the watermarked video; (3) Watermark Extraction: recovering the embedded watermark using an extraction network built with a distortion layer, a series of 3D SFMamba blocks, and position recovery. The difference between different types of Mamba blocks lies in their scanning strategies.

(LVDMs) [26] perform the diffusion process in the latent space to improve computational efficiency. VideoCrafter2 [2] builds high-quality video generation models by leveraging low-quality video data combined with synthesized high-quality images. Open-Sora [6] introduces the Spatial-Temporal Diffusion Transformer, an efficient video diffusion framework that separates spatial and temporal attention mechanisms. While LVDMs have shown strong performance in video generation, the integration of graphical watermarks into this framework has not been explored.

##### 2.2 Generative Video Watermarking

Digital watermarking has emerged as an essential technique for copyright protection, content authentication, and ownership verification across various media types. However, watermarking for video diffusion models represents a relatively unexplored area. VideoShield [15] pioneered this space by modifying latent noise during the diffusion process to embed binary watermark information. More recently, LVMark [16] introduced an importance-based weight modulation strategy to minimize visual quality degradation. Nevertheless, these existing approaches primarily focus on embedding low-capacity binary strings, without taking advantage of the high-capacity nature of video media, which is well-suited for embedding richer information such as graphical watermarks.

##### 2.3 State Space Models

State Space Models (SSMs) [27, 28] have emerged as efficient alternatives to transformers [29] for sequence modeling. The Mamba architecture [30] represents a significant advancement in SSMs by introducing selective state space modeling with data-dependent parameters, enabling dynamic resource allocation to important sequence elements while maintaining computational efficiency. Despite Mamba’s remarkable success in language processing tasks [31, 32] and its growing adoption in computer vision applications [33, 34], its potential for watermarking techniques has remained entirely unexplored until now.

#### 3 Graphical Watermarking for Video Generation

In this section, we present the pipeline of our Safe-Sora framework, which introduces a novel approach to embedding graphical watermarks directly within the video generation process (Fig. 2). We first partition the watermark image into patches and optimally assign them to appropriate video frames and regions (Section 3.1). These patches are then embedded and upsampled to generate the watermark feature map. To embed the watermark, this feature map is fused with multi-scale video features using a UNet built with 2D SFMamba blocks (Section 3.2), followed by a series of 3D SFMamba blocks that leverage our spatiotemporal local scanning strategy (Section 3.3), producing a watermarked video. To extract the watermark, the watermarked video is processed through an extraction network built with a degradation layer, a series of 3D SFMamba blocks, and position recovery. The training objectives are outlined in Section 3.4, while the preliminaries on latent video diffusion models, state space models, and wavelet transforms are detailed in Appendix A.

##### 3.1 Coarse-to-Fine Adaptive Patch Matching

Motivated by the observation that greater similarity between the watermark and cover content enhances watermarking performance (as shown in Fig. 1), we propose a coarse-to-fine adaptive patch matching mechanism to systematically identify the most semantically similar spatial-temporal regions in a video for watermark embedding, as illustrated in the bottom-left corner of Fig. 2.

First, to enable accurate localization of each patch during the final watermark recovery, we propose a simple yet effective method: the position channel. Specifically, we represent patch positions using binary encoding (e.g., using 8 bits to represent 256 patch positions). This binary code is then replicated to form an additional channel, introducing redundancy that enhances robustness against spatial distortions and degradation. Finally, this position channel is concatenated with the patch content, embedding positional information directly into the input and eliminating the need for additional positional processing during subsequent training.

Then, we adopt a two-stage process to adaptively determine the most suitable embedding location for each patch. The first stage operates at the frame level. We extract features from both patches and the latent representations of video frames using a convolution layer followed by ReLU and global average pooling (GAP). Similarity between each patch i and frame j is computed via dot product of these feature vectors, and normalized using Softmax:

wi,j = Softmax (GAP(ReLU(Conv(pi))) · GAP(ReLU(Conv(zj)))) . (1)

Here, wi,j denotes the similarity score between patch pi and the latent representation zj of frame j. Each patch is then assigned to the frame with the highest similarity score. To ensure balanced distribution, we impose a maximum capacity for each frame. If the top-ranked frame is full, the patch is redirected to the next highest available candidate. Having selected a frame, we proceed to the fine stage, which determines the optimal spatial position within that frame. Each frame is subdivided into spatial regions according to its patch capacity. Feature representations of these regions are computed similarly, and the similarity between patch i and region k in the assigned frame j is given by:

si,k = Softmax (GAP(ReLU(Conv(pi))) · GAP(ReLU(Conv(rj,k)))) , (2)

where si,k is the similarity score between the i-th patch and the k-th region rj,k in the latent representation of frame j. Note that we take full advantage of the inherent feature properties of latent variables in video generation models. Since latent variables can already be viewed as feature extractions of the original frames, we use only a single convolutional layer for feature extraction, which significantly reduces the computational overhead.

##### 3.2 Spatial-Frequency Mamba for Spatial Fusion

Mamba [30] has demonstrated strong capabilities in modeling long-range dependencies with high efficiency, making it well-suited for spatiotemporal modeling in video tasks. Meanwhile, frequency domain information has proven effective in enhancing watermark embedding by capturing structural patterns and resisting distortions [35, 36]. To incorporate both advantages, we propose the SpatialFrequency Mamba (SFMamba) block, as shown in Fig. 2.

SFMamba adopts a dual-stream design with separate spatial and frequency branches. It comes in two variants: a 2D version and a 3D version, differing primarily in the wavelet transform and scanning strategy. The 3D SFMamba will be introduced in Section 3.3. We next introduce the 2D SFMamba

Spatial Dimension

Low Frequency High Frequency

TimeDimension

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

- Frame 1
- Frame 2
- Frame 3
- Frame 4

LLL LLH HHL HHH

| | |
|---|---|
| | |

[Figure 51]

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

Forward Scanning Backward Scanning

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

- 3D Wavelet Transform

- Figure 3: For 3D frequency scanning, we propose a spatiotemporal local scanning strategy for 3D wavelet transform, which processes the frequency components hierarchically from low frequency to high frequency and high frequency to low frequency.

block for efficient spatial fusion of watermark and video content. It consists of separate 2D spatial and 2D frequency branches.

- 2D Spatial Branch. The spatial processing begins with a LayerNorm operation on the input feature map Fin, yielding normalized features FN. In the first path, FN undergoes a simple SiLU activation function. In the second path, FN passes through a 1×1 convolution layer, followed by our 2D spatial Mamba module. The 2D spatial branch output Fs is computed as:

Fs = SiLU(FN) ⊙ 2DSpatialMamba(Conv1×1(FN)). (3)

where ⊙ denotes element-wise multiplication of the two pathway outputs.

- 2D Frequency Branch. For frequency domain processing, we transform FN using a 2D Discrete Wavelet Transform (DWT), which decomposes the signal into four frequency subbands: LL (low-low), LH (low-high), HL (high-low), and HH (high-high). Each subband has spatial dimensions reduced by half compared to the original. Inspired by FreqMamba [37], we rearrange these components from top-left to bottom-right to restore the original resolution. The wavelet features are then divided into four blocks and scanned block by block. The output is projected back to the spatial domain via a 2D Inverse DWT (IDWT), followed by element-wise multiplication with SiLU(FN). The 2D frequency branch output Ff is computed as:

Ff = SiLU(FN) ⊙ IDWT(2DFreqMamba(DWT(FN))). (4)

The spatial branch output is enhanced with a residual connection from Fin. Finally, we concatenate the outputs from both branches and apply a 1×1 convolution to produce the integrated output.

- 3.3 3D Frequency Scanning for Spatiotemporal Interaction

To address the challenges of fusing and extracting watermark information distributed across spatiotemporal locations, we propose an efficient architecture—3D SFMamba, a 3D Wavelet Mamba transform-enhanced design with a customized scanning strategy. This architecture enables bidirectional modeling across frequency subbands within the 3D wavelet transform, effectively capturing long-range dependencies in both spatial and temporal domains to accurately recover watermark information embedded in the temporal dimension. 3D SFMamba consists of separate 3D spatial and

- 3D frequency branches.

- 3D Spatial Branch. The 3D spatial branch employs a vanilla 3D scanning strategy, which processes features across all three dimensions (temporal, height, width) to capture both spatial and temporal dependencies effectively.

- 3D Frequency Branch. In the frequency domain branch, input features Fin undergo a 3D Discrete Wavelet Transform (3D DWT), decomposing them into eight subbands: LLL, LLH, LHL, LHH, HLL, HLH, HHL, and HHH. Each subband has half the original dimensions in frame, height, and width. To address the complexity of 3D wavelet-transformed features, we propose a novel spatiotemporal local scanning strategy as shown in Fig. 3. This approach first rearranges the eight subbands to restore the original video resolution, then divides them into eight distinct parts for separate scanning. For forward scanning, the order follows LLL, LLH, LHL, HLL, LHH, HLH, HHL, and HHH—progressing systematically from low to high frequencies. Additionally, we implement a reverse scanning mechanism that processes the subbands in the opposite direction—from HHH to LLL—enabling the model to capture information from high to low frequencies. Within each part, we employ a spatial-first, temporal-second scanning pattern. This spatiotemporal local scanning

strategy is specifically designed for 3D wavelet transforms, allowing the model to process frequency information hierarchically across multiple scales.

##### 3.4 Training Objectives

Our training framework combines video reconstruction loss and watermark reconstruction loss. The video reconstruction loss uses mean squared error (MSE) to ensure the watermarked video Vˆ closely resembles the original video V:

Lvideo = MSE(V, Vˆ). (5)

Similarly, the watermark reconstruction loss measures the extraction accuracy by comparing the extracted watermark Wˆ with the original watermark W:

Lwatermark = MSE(W, Wˆ ). (6)

During training, we provide the correct positions to reconstruct the watermark image properly, while during testing, the model utilizes the embedded position channels to predict the correct arrangement of patches. The final loss function is:

Ltotal = Lvideo + λ Lwatermark, (7)

where the watermark weighting hyperparameter λ balances video quality against watermark fidelity.

#### 4 Experiments

##### 4.1 Experimental Setting

Datasets. For the video dataset, we use the Panda-70M [20] dataset for training, which is a largescale dataset containing 70 million high-quality videos across diverse content types. Specifically, we randomly download 10,000 videos from Panda-70M, sample 8 frames from each video, and resize each frame to a resolution of 320 × 512 for training purposes. For the watermark dataset, we use the Logo-2K dataset [18], which contains 167,140 watermark images at a resolution of 256 × 256, spanning a wide range of real-world logo classes. For the evaluation of text-to-video generation, we employ the VidProm [38] dataset as the source of prompts. The prompts in VidProm are generated by GPT-4 [39], and we randomly select 100 prompts from the dataset for evaluation.

Implementation Details. We use VideoCrafter2 [2] as our backbone model to generate videos at a resolution of 320 × 512. Our method is compatible with various video generation backbones, with additional results provided in Appendix C. The patch size is set to 16 × 16. Patch Embedding maps each patch to a 1024-dimensional feature space. The model is trained for 30 epochs on 4 NVIDIA RTX 4090 GPUs. We adopt the AdamW optimizer [40], with the initial learning rate set to 5e-4, which is gradually decayed to 1e-6 following a cosine decay schedule. The watermark embedding network uses M = 2 3D SFMamba Blocks, while the watermark extraction network uses N = 4 3D SFMamba Blocks. The hyperparameter λ in Eq. 7 is set to 0.75. The distortion layer simulates various real-world distortions, including H.264 video compression, rotation, and other common transformations. Since H.264 is non-differentiable, we follow DVMark [41] and use a 3D CNN to mimic its effects. For position recovery, we propose a confidence-guided greedy assignment algorithm, with detailed descriptions provided in Appendix B.

Baselines. To the best of our knowledge, no existing method embeds graphical watermarks directly into video generation models. To provide a comprehensive comparison, we select five representative state-of-the-art methods spanning three distinct paradigms of graphical watermarking: (1) Postprocessed image watermarking methods: Balujanet[17] – A classic image steganography network; UDH[42] – A classic graphical watermarking network; PUSNet [43] – A state-of-the-art image steganography network. (2) Generative image watermarking: Safe-SD [44] – A generative graphical watermarking approach. (3) Video steganography: Wengnet [45] – A method that hides one video within another. For a fair comparison, we retrain all baseline methods using the same training dataset as ours. For image-based methods, we embed a complete watermark image into each frame. For video-based methods, each frame of the secret video acts as a watermark and is embedded into the corresponding frame of the cover video.

Original Ours Balujanet Wengnet UDH PUSNet Safe-SD

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Frame Difference

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

×(5) Watermark

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

Difference

×(5)

“Kitten Max flies among clouds and stars.”

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Frame Difference

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

×(5) Watermark

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Difference

×(5)

“A gray and white cat is playing with a butterfly in a beautiful garden.”

- Figure 4: Qualitative comparison results on the first frame of each video. Difference maps show absolute differences between the watermarked and original videos, and between the recovered and original watermarks. More examples are shown in Fig. 10 of Appendix. Best viewed with zoom in.

##### 4.2 Comparison with State-of-the-art Methods

Qualitative Comparison. Fig. 4 shows the qualitative comparisons on the first frame of each video, while Fig. 5 presents visual results of Safe-Sora across multiple frames. As illustrated, Balujanet introduces clearly visible artifacts in the watermarked video, UDH suffers from stripe-like distortions, and Safe-SD presents noticeable color shifts. From the difference maps, it is evident that both WengNet and PUSNet introduce considerable degradation to both video quality and watermark fidelity. In contrast, our method produces watermarked videos with high visual fidelity, exhibiting minimal differences from the original videos. Moreover, the recovered watermark images closely resemble the originals, demonstrating high reconstruction accuracy.

Quantitative Comparison. To evaluate the accuracy of watermark recovery and the invisibility of the watermark (i.e., video quality), we adopt standard metrics including PSNR, MAE, RMSE, SSIM [46], and LPIPS [47]. To assess temporal consistency in videos, we employ tLP [48] and Fréchet Video Distance (FVD) [49]. Quantitative results are summarized in Tab 1. As shown in the table, our method achieves state-of-the-art performance across all evaluation metrics. We observe that image watermarking methods inject watermarks by embedding them independently into each frame, which leads to poor temporal consistency and higher FVD scores. In contrast, our method leverages Mamba’s long-range modeling capability across space and time, along with the proposed

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Frame Original

Frame

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

×(5) Watermarked

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Difference

“Spaceships traverse a vibrant cosmos filled with planets and stars.”

- Figure 5: Visual results of Safe-Sora on multiple frames. For each frame, we show the original image, the corresponding watermarked image, and their residual difference. Best viewed with zoom in.

- Table 1: Quantitative results on watermark quality and video quality metrics. Watermark quality is measured by comparing the recovered watermark image with the original watermark, while video quality is evaluated by comparing the watermarked video with the original video.

Watermark quality Video quality PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ tLP ↓ FVD ↓

Method

Balujanet 25.28 9.61 15.10 0.91 0.11 25.26 10.09 14.58 0.87 0.25 1.32 512.22 Wengnet 33.18 3.71 5.82 0.96 0.06 28.09 6.27 10.69 0.85 0.21 1.27 265.82

UDH 22.90 11.29 19.29 0.77 0.24 27.75 8.16 10.72 0.73 0.32 2.09 1075.62

PUSNet 28.86 7.45 9.57 0.93 0.12 29.98 4.50 8.72 0.92 0.11 0.98 154.35 Safe-SD 24.24 9.78 17.39 0.84 0.11 22.32 11.65 20.64 0.75 0.24 1.87 849.83 Ours 37.71 2.22 3.61 0.97 0.04 42.50 1.36 1.96 0.98 0.01 0.38 3.77

spatiotemporal local scanning strategy, resulting in superior temporal consistency. Specifically, our method achieves an FVD of 3.77, significantly outperforming all baselines.

##### 4.3 Robustness

To rigorously evaluate the robustness of our method, we apply a variety of distortion types. For random erasing, we randomly select an erasure ratio from the range [5%, 10%, 15%, 20%]. For Gaussian blur, we randomly choose a kernel size from 3, 5, 7. For Gaussian noise, we add noise with a standard deviation randomly sampled from a uniform distribution U(0,0.2). For rotation, the degree is randomly sampled from the range (−30◦,30◦). Specifically for video, we adopt H.264 compression with a fixed CRF value of 24. We use PSNR, SSIM, and LPIPS to evaluate the robustness of watermark reconstruction under these distortions. As shown in Fig. 6, our method consistently achieves the best performance across all types of attacks, demonstrating strong robustness. In particular, under H.264 compression, all baseline methods suffer a significant drop in performance, whereas our method maintains high watermark quality.

##### 4.4 Ablation Study

We conduct an ablation study on two key components— Coarse-to-Fine Adaptive Patch Matching and Spatiotemporal Local Scanning. Additional ablation studies can be found in Appendix D.

Impact of Coarse-to-Fine Adaptive Patch Matching. This strategy matches the most similar frame and spatial location for each watermark patch, based on similarity computed with the video latent representations. To evaluate the effectiveness of each component, we investigate three ablated variants of our method: w/o CFAPM, which completely removes the Coarse-to-Fine Adaptive Patch Matching mechanism; w/o RtL, which replaces the Routing by Latent strategy with a direct pixel-frame similarity computation; and w/o FS, which removes the Fine Stage responsible for spatial location refinement.

###### PSNR

###### SSIM

###### LPIPS

1.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.8

35

0.7

0.8

30

0.6

25

0.6

0.5

0.4

20

0.4

0.3

15

0.2

10

0.2

0.1

5

0.0

Erase Blur Noise Rotate H.264

Erase Blur Noise Rotate H.264

Erase Blur Noise Rotate H.264

Ours Balujanet Wengnet UDH PUSNet Safe-SD

- Figure 6: Watermark reconstruction quality under various distortions. Distortion settings include: Random Erasing (5%–20%), Gaussian Blur (kernel size 3/5/7), Gaussian Noise (σ ∼ U(0,0.2)), Rotation (-30°, 30°), and H.264 Compression (CRF = 24).

- Table 2: Comprehensive ablation study on key components of our method. CFAPM: Coarse-to-Fine Adaptive Patch Matching; RtL: Routing by Latent; FS: Fine Stage; SLS: Spatiotemporal Local Scanning; SFS: Spatial First Scanning within each subband.

Watermark quality Video quality

Method

PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ tLP ↓ FVD ↓ w/o CFAPM 36.71 2.53 3.99 0.96 0.05 39.68 1.94 2.76 0.97 0.03 1.14 16.87

w/o RtL 36.36 2.67 4.13 0.96 0.05 40.23 1.79 2.54 0.97 0.04 1.30 6.37 w/o FS 36.88 2.45 3.94 0.97 0.04 41.25 1.58 2.26 0.97 0.03 1.17 4.82

w/o SLS 35.96 2.98 4.02 0.94 0.08 38.42 1.98 2.12 0.92 0.03 1.01 13.16 w/o SFS 36.41 2.59 4.17 0.96 0.05 42.21 1.38 2.05 0.98 0.01 0.24 5.24

Ours 37.71 2.22 3.61 0.97 0.04 42.50 1.36 1.96 0.98 0.01 0.38 3.77

The results in Tab. 2 clearly demonstrate that each component of the CFAPM strategy plays a critical role in enhancing overall performance. Computing the similarity between watermark patches and video latents leverages the compressed semantic information encoded in the latent space, enabling more accurate matching; the fine stage further refines this process by identifying the most visually similar spatial location for each patch. Overall, the Coarse-to-Fine Adaptive Patch Matching mechanism consistently improves both watermark fidelity and video quality.

Impact of Spatiotemporal Local Scanning. This strategy traverses the eight subbands of the 3D wavelet transform in a frequency-aware hierarchical order. Within each subband, patches are selected following a spatial-first, temporal-second scanning pattern. To evaluate the effectiveness of this design, we ablate two key components: w/o SLS, which replaces the structured traversal with a vanilla 3D scanning strategy; and w/o SFS, which applies a temporal-first scanning order within each subband instead of the proposed spatial-first policy.

Results in Tab. 2 demonstrate that the full SLS strategy significantly improves both watermark and video quality. While the temporal-first scanning achieves slightly better tLP, it consistently underperforms in watermark fidelity metrics. In summary, SLS enables more effective fusion and extraction of watermark signals distributed across spatiotemporal regions, thereby enhancing the overall performance of watermark embedding.

#### 5 Conclusion

Our work introduces Safe-Sora, the first framework embedding graphical watermarks directly into generated video. We propose a hierarchical coarse-to-fine adaptive matching strategy that optimally maps watermark patches to visually similar frames and spatial regions. Our 3D wavelet transformenhanced Mamba architecture with a novel spatiotemporal local scanning strategy, effectively models spatiotemporal dependencies for watermark embedding and retrieval, pioneering the application of state space models to watermarking. Experiments demonstrate that Safe-Sora achieves superior performance in video quality, watermark fidelity, and robustness. This work establishes a foundation for copyright protection in generative video and opens new avenues for applying state space models to digital watermarking.

#### References

- [1] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.
- [2] H. Chen, Y. Zhang, X. Cun, M. Xia, X. Wang, C. Weng, and Y. Shan, “Videocrafter2: Overcoming data limitations for high-quality video diffusion models,” in CVPR, 2024, pp. 7310–7320.
- [3] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023.
- [4] X. Ma, Y. Wang, G. Jia, X. Chen, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao, “Latte: Latent diffusion transformer for video generation,” arXiv preprint arXiv:2401.03048, 2024.
- [5] J. Xing, M. Xia, Y. Zhang, H. Chen, W. Yu, H. Liu, G. Liu, X. Wang, Y. Shan, and T.-T. Wong, “Dynamicrafter: Animating open-domain images with video diffusion priors,” in ECCV. Springer, 2024, pp. 399–417.
- [6] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You, “Open-sora: Democratizing efficient video production for all,” arXiv preprint arXiv:2412.20404, 2024.
- [7] Y. Zhao, T. Pang, C. Du, X. Yang, N.-M. Cheung, and M. Lin, “A recipe for watermarking diffusion models,” arXiv preprint arXiv:2303.10137, 2023.
- [8] P. Fernandez, G. Couairon, H. Jégou, M. Douze, and T. Furon, “The stable signature: Rooting watermarks in latent diffusion models,” in ICCV, 2023, pp. 22466–22477.
- [9] R. Min, S. Li, H. Chen, and M. Cheng, “A watermark-conditioned diffusion model for ip protection,” in ECCV. Springer, 2024, pp. 104–120.
- [10] C. Xiong, C. Qin, G. Feng, and X. Zhang, “Flexible and secure watermarking for latent diffusion model,” in ACM MM, 2023, pp. 1668–1676.
- [11] L. Lei, K. Gai, J. Yu, and L. Zhu, “Diffusetrace: A transparent and flexible watermarking scheme for latent diffusion model,” arXiv preprint arXiv:2405.02696, 2024.
- [12] Z. Meng, B. Peng, and J. Dong, “Latent watermark: Inject and detect watermarks in latent diffusion space,” IEEE Transactions on Multimedia, 2025.
- [13] Z. Yang, K. Zeng, K. Chen, H. Fang, W. Zhang, and N. Yu, “Gaussian shading: Provable performancelossless image watermarking for diffusion models,” in CVPR, 2024, pp. 12162–12171.
- [14] H. Ci, P. Yang, Y. Song, and M. Z. Shou, “Ringid: Rethinking tree-ring watermarking for enhanced multi-key identification,” in ECCV. Springer, 2024, pp. 338–354.
- [15] R. Hu, J. Zhang, Y. Li, J. Li, Q. Guo, H. Qiu, and T. Zhang, “Videoshield: Regulating diffusion-based video generation models via watermarking,” arXiv preprint arXiv:2501.14195, 2025.
- [16] M. Jang, Y. Jang, J. Lee, K. Kawamura, F. Yang, and S. Kim, “Lvmark: Robust watermark for latent video diffusion models,” arXiv preprint arXiv:2412.09122, 2024.
- [17] S. Baluja, “Hiding images within images,” IEEE transactions on pattern analysis and machine intelligence, vol. 42, no. 7, pp. 1685–1697, 2019.
- [18] J. Wang, W. Min, S. Hou, S. Ma, Y. Zheng, H. Wang, and S. Jiang, “Logo-2K+: a large-scale logo dataset for scalable logo classification,” in AAAI, 2020.
- [19] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in CVPR. Ieee, 2009, pp. 248–255.
- [20] T.-S. Chen, A. Siarohin, W. Menapace, E. Deyneka, H.-w. Chao, B. E. Jeon, Y. Fang, H.-Y. Lee, J. Ren, M.-H. Yang, and S. Tulyakov, “Panda-70m: Captioning 70m videos with multiple cross-modality teachers,” in CVPR, 2024.
- [21] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020.
- [22] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in ICML. PMLR, 2021, pp. 8162–8171.

- [23] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
- [24] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in ICML. PMLR, 2015, pp. 2256–2265.
- [25] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” NeurIPS, vol. 34, pp. 8780–8794, 2021.
- [26] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10684–10695.
- [27] A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.
- [28] J. T. Smith, A. Warrington, and S. W. Linderman, “Simplified state space layers for sequence modeling,” arXiv preprint arXiv:2208.04933, 2022.
- [29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” NeurIPS, vol. 30, 2017.
- [30] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.
- [31] J. Wang, T. Gangavarapu, J. N. Yan, and A. M. Rush, “Mambabyte: Token-free selective state space model,” arXiv preprint arXiv:2401.13660, 2024.
- [32] R. Waleffe, W. Byeon, D. Riach, B. Norick, V. Korthikanti, T. Dao, A. Gu, A. Hatamizadeh, S. Singh, D. Narayanan et al., “An empirical study of mamba-based language models,” arXiv preprint arXiv:2406.07887, 2024.
- [33] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, and Y. Liu, “Vmamba: Visual state space model,” NeurIPS, vol. 37, pp. 103031–103063, 2024.
- [34] K. Li, X. Li, Y. Wang, Y. He, Y. Wang, L. Wang, and Y. Qiao, “Videomamba: State space model for efficient video understanding,” in ECCV. Springer, 2024, pp. 237–255.
- [35] S. A. Al-Taweel and P. Sumari, “Robust video watermarking based on 3d-dwt domain,” in TENCON 2009-2009 IEEE Region 10 Conference. IEEE, 2009, pp. 1–6.
- [36] X. Li and R. Wang, “A video watermarking scheme based on 3d-dwt and neural network,” in Ninth IEEE International Symposium on Multimedia Workshops (ISMW 2007). IEEE, 2007, pp. 110–115.
- [37] Z. Zhen, Y. Hu, and Z. Feng, “Freqmamba: Viewing mamba from a frequency perspective for image deraining,” arXiv preprint arXiv:2404.09476, 2024.
- [38] W. Wang and Y. Yang, “Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models,” in NeurIPS, 2024.
- [39] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [40] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.
- [41] X. Luo, Y. Li, H. Chang, C. Liu, P. Milanfar, and F. Yang, “Dvmark: a deep multiscale framework for video watermarking,” IEEE Transactions on Image Processing, 2023.
- [42] C. Zhang, P. Benz, A. Karjauv, G. Sun, and I. S. Kweon, “Udh: Universal deep hiding for steganography, watermarking, and light field messaging,” NeurIPS, vol. 33, pp. 10223–10234, 2020.
- [43] G. Li, S. Li, Z. Luo, Z. Qian, and X. Zhang, “Purified and unified steganographic network,” in CVPR, 2024, pp. 27569–27578.
- [44] Z. Ma, G. Jia, B. Qi, and B. Zhou, “Safe-sd: Safe and traceable stable diffusion with text prompt trigger for invisible generative watermarking,” in ACM MM, 2024, pp. 7113–7122.
- [45] X. Weng, Y. Li, L. Chi, and Y. Mu, “High-capacity convolutional video steganography with temporal residual modeling,” in Proceedings of the 2019 on international conference on multimedia retrieval, 2019, pp. 87–95.

- [46] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
- [47] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018, pp. 586–595.
- [48] M. Chu, Y. Xie, J. Mayer, L. Leal-Taixé, and N. Thuerey, “Learning temporal coherence via self-supervision for gan-based video generation,” ACM Transactions on Graphics (TOG), vol. 39, no. 4, pp. 75–1, 2020.
- [49] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv preprint arXiv:1812.01717, 2018.

### Technical Appendices

###### Generation

###### Identification

Decoder

[Figure 143]

Latent Video

"A campfire burning brightly"

Generative Model D

[Figure 144]

[Figure 145]

|[Figure 146]|
|---|

Distort

###### E

User

Watermarked Video

|[Figure 147]|
|---|

Watermarked Video

Distorted Video

Watermark Extractor

Watermark

###### F

Model Owner

Watermark

Feature Extractor

- Figure 7: Application Scenario of Safe-Sora: A user provides a text prompt to a video generation model. The model owner’s graphical watermark is embedded into the video through a feature extractor and decoder. Later, even if the video is distorted, a watermark extractor can recover the graphical watermark to verify authenticity and ensure copyright protection. A Preliminaries

##### A.1 Latent Video Diffusion Models

Latent Video Diffusion Models (LVDMs) extend the concept of latent diffusion models to the video domain. These models operate in a compressed latent space rather than pixel space to improve computational efficiency while maintaining generation quality. The process can be described in three key steps:

First, a video encoder E maps the input video x ∈ RF×H×W×3 to a latent representation z = E(x) ∈ RF×h×w×c, where F is the number of frames, and the spatial dimensions are reduced: h < H and w < W.

Second, a diffusion process gradually adds noise to the latent representation through a fixed Markov chain:

q(zt|zt−1) = N(zt; 1 − βtzt−1,βtI), (8)

q(zt|z0) = N(zt;√α¯tz0,(1 − α¯t)I), (9) where βt is the noise schedule, αt = 1 − βt, and α¯t = ts=1 αs.

Finally, a denoising network ϵθ is trained to predict the added noise at each time step. During generation, the reverse process starts from pure Gaussian noise zT ∼ N(0,I) and iteratively denoises to produce z0, which is then decoded to the final video xˆ = D(z0) using a decoder D.

For text-to-video generation, LVDMs incorporate a text encoder that processes a conditioning prompt, which guides the denoising process toward the desired content.

###### A.2 State Space Models State Space Models (SSMs) are continuous dynamical systems defined by the following equations:

dh(t) dt

= Ah(t) + Bx(t), (10) y(t) = Ch(t) + Dx(t), (11)

where x(t) is the input, h(t) is the hidden state, y(t) is the output, and {A,B,C,D} are the parameters of the system.

For discrete sequence modeling, these continuous equations are discretized:

ht = Ah¯ t−1 + Bx¯ t, (12) yt = Cht + Dxt, (13)

where A¯ and B¯ are the discretized versions of A and B.

Algorithm 1 Confidence-Guided Greedy Assignment for Watermark Position Recovery

- 1: Input: Watermark patches with position channel
- 2: Output: Reconstructed watermark image W

- Stage 1: Position Decoding

3: for each patch i do 4: Normalize position channel to [0,1] 5: Compute probability vector pi by averaging binary vectors in the position channel 6: Compute confidence ci = K1 Kj=1 |pji − 0.5| 7: Convert pi to binary ˆbi via thresholding 8: Decode ˆbi → position index posi ∈ [0,N − 1] 9: end for

- Stage 2: Confidence-Prioritized Assignment

10: Initialize watermark image W ← ∅ 11: Initialize unassigned patch pool U ← ∅ 12: for each patch i do 13: if posi is unoccupied in W then 14: Assign patch i to position posi in W 15: else if ci > confidence of current patch at posi then 16: Replace patch at posi with i in W 17: Add the replaced patch to U 18: else 19: Add patch i to U 20: end if 21: end for

- Stage 3: Greedy Reassignment of Unassigned Patches

- 22: Sort U by descending ci
- 23: for each patch j in U do
- 24: Find nearest vacant position pj to posj
- 25: Assign patch j to position pj in W
- 26: end for
- 27: return W

The Mamba architecture extends traditional SSMs by introducing input-dependent parameters:

A¯ ,B¯ = Projection(x), (14) ht = A¯ ⊙ ht−1 + B¯ ⊙ xt, (15) yt = Cht, (16)

This input-dependent parameterization allows Mamba to dynamically adapt its processing based on input content, making it effective for modeling complex sequential dependencies.

##### A.3 Wavelet Transforms

Wavelet transforms decompose signals into multiple frequency components with localized time information, making them useful for frequency domain watermarking.

For images, the 2D Discrete Wavelet Transform (DWT) decomposes an image into four sub-bands: approximation (LL), horizontal detail (LH), vertical detail (HL), and diagonal detail (HH).

The 3D Discrete Wavelet Transform extends the 2D DWT to the temporal domain for video processing.

- A video sequence is decomposed into eight sub-bands: LLL, LLH, LHL, LHH, HLL, HLH, HHL, and HHH, with L and H representing low and high frequencies across the frame, height, and width dimensions. Each sub-band has half the resolution of the original video in all dimensions. The 3D DWT provides a multi-level representation of videos, capturing both spatial and temporal characteristics, which is beneficial for video watermarking by allowing embedding in specific frequency bands while preserving perceptual quality.

Table 3: Quantitative comparison on VideoCrafter2 and Open-Sora backbones.

Watermark quality Video quality

Backbone

PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ tLP ↓ FVD ↓ VideoCrafter2 37.71 2.22 3.61 0.97 0.04 42.50 1.36 1.96 0.98 0.01 0.38 3.77

Open-Sora 35.42 2.93 4.70 0.96 0.06 44.15 1.31 1.75 0.97 0.01 0.31 3.04

Frame Watermaked frame Difference (×5) Watermark Recovered watermark Difference (×5)

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

“A flock of seagulls flies over the azure sea and above the red cliffs.”

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

“Numerous hot air balloons float above a snow-covered, peculiar landscape.”

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

“A magnificent waterfall cascades amidst the lush forest.”

Figure 8: Qualitative examples on Open-Sora backbone. Best viewed with zoom in.

#### B Robust Watermark Position Recovery Algorithm

To address rare cases where multiple watermark patches are decoded to the same spatial location due to distortion or attack, we propose a confidence-guided greedy assignment algorithm. This algorithm ensures reliable and unambiguous recovery of watermark positions by incorporating confidence estimation, conflict resolution, and greedy reassignment of unplaced patches.

The algorithm is as follows: first, compute the confidence score for each patch’s predicted position. Then, assign each patch to its corresponding position; in case of conflicts, give priority to the patch with higher confidence. Finally, assign the remaining unplaced patches in descending order of confidence to the nearest available positions. The detailed procedure is illustrated in Algorithm 1.

The confidence-guided greedy assignment algorithm effectively handles noisy or partial position corruption and significantly improves the robustness of watermark extraction.

#### C More Backbones

While our main experiments are conducted using VideoCrafter2 [2], a UNet-based video generation model, we further evaluate our method using Open-Sora [6], a DiT-based video generation model. Quantitative results are shown in Tab. 3, and qualitative examples are provided in Fig. 8. As can be seen, Open-Sora achieves comparable performance to VideoCrafter2 and produces videos with higher visual quality, but slightly lower watermark fidelity. These results demonstrate that our method is effective across different video generation models.

Table 4: Additional Ablation Studies. MSFI: Multi-Scale Feature Injection.

Watermark quality Video quality

Method

PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ PSNR ↑ MAE ↓ RMSE ↓ SSIM ↑ LPIPS ↓ tLP ↓ FVD ↓ w/o MSFI 36.56 2.56 4.06 0.96 0.05 39.39 2.02 2.84 0.97 0.03 1.19 14.11

Ours 37.71 2.22 3.61 0.97 0.04 42.50 1.36 1.96 0.98 0.01 0.38 3.77

Original w/o MSFI w/ MSFI Original w/o MSFI w/ MSFI

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

- Figure 9: Visual impact of Multi-Scale Feature Injection. We present difference maps (×5) between watermarked and original videos. After applying Multi-Scale Feature Injection, the differences are significantly reduced, leading to improved video quality.

#### D Additional Ablation Studies

To further assess the contribution of individual components in our framework, we perform extended ablation studies beyond the main experiments. In particular, we examine the impact of Multi-Scale Feature Injection, with quantitative results reported in Tab. 4 and qualitative comparisons shown in Fig. 9. The results demonstrate that incorporating the inherent multi-scale features of the VAE notably improves the visual quality of generated videos.

#### E Limitations

While our method demonstrates strong performance in embedding and recovering static graphical watermarks, it is currently limited to image-based watermarks such as logos or icons. Embedding more complex and information-rich video watermarks—e.g., animated sequences or temporally dynamic patterns—remains a challenge.

#### F Societal Impact

The ability to embed graphical watermarks directly into the video generation process carries important social and ethical implications. On the positive side, it provides a practical solution to the growing concerns over ownership verification and copyright protection in generative media. As synthetic content becomes increasingly widespread, methods like ours can help content creators assert their rights and trace misuse, thereby fostering accountability and transparency in digital media ecosystems.

However, like many watermarking techniques, our method may also be misused. For example, it could potentially be employed to falsely claim ownership over public material, or to embed unauthorized logos into generated videos. We strongly advocate for the responsible use of generative watermarking technologies and recommend that future research explores methods to verify the authenticity of embedded watermarks and prevent abuse.

Original Ours Balujanet Wengnet UDH PUSNet Safe-SD

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Frame Difference

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

×(5) Watermark

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

Difference

×(5)

“Batman overlooking the city at dusk.”

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Frame Difference

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

×(5) Watermark

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

Difference

×(5)

“ A modern beachside cabin with an ocean view. ”

- Figure 10: More qualitative examples on VideoCrafter2 backbone. Difference maps show absolute differences between the watermarked and original videos, and between the recovered and original watermarks. Best viewed with zoom in.

