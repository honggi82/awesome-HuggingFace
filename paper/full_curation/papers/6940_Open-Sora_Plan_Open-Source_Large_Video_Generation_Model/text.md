# arXiv:2412.00131v1[cs.CV]28Nov2024

### Open-Sora Plan: Open-Source Large Video Generation Model

Open-Sora Plan Team∗

##### Abstract

We introduce Open-Sora Plan, an open-source project that aims to contribute a large generation model for generating desired high-resolution videos with long durations based on various user inputs. Our project comprises multiple components for the entire video generation process, including a Wavelet-Flow Variational Autoencoder, a Joint Image-Video Skiparse Denoiser, and various condition controllers. Moreover, many assistant strategies for efficient training and inference are designed, and a multi-dimensional data curation pipeline is proposed for obtaining desired high-quality data. Benefiting from efficient thoughts, our Open-Sora Plan achieves impressive video generation results in both qualitative and quantitative evaluations. We hope our careful design and practical experience can inspire the video generation research community. All our codes and model weights are publicly available at https://github.com/PKU-YuanGroup/Open-Sora-Plan.

##### 1 Introduction

Driven by the recent progress of the diffusion model (Ho et al., 2020; Song et al., 2020) and transformer (Vaswani, 2017; Peebles and Xie, 2023) architecture, visual content generation demonstrates impressive creation capacity conditioned on given prompts, which attracts broad interests and emerging attempts. Since the image generation methods (Rombach et al., 2022b; Li et al., 2024c) achieve outstanding performance and are applied extensively, the video generation model is expected to make significant advancements to empower a variety of creative industries including entertainment, advertising, film, etc. Many early attempts (Guo et al., 2023; Xing et al., 2025) successfully generate video with low resolution and short frames, but few efforts challenge the high-quality and long-duration video generation due to the unimaginable computation and data cost.

However, the technique report of Sora (Brooks et al., 2024), the video generation model created by OpenAI, with impressive showcases is released suddenly, shocking the entire video generation community while pointing out a promising way to create remarkable videos. As one of the first open-source projects aiming to re-implement a powerful Sora-like video generation model, our Open-Sora Plan attracts wide attention and contributes many first attempts to the video generation community, which inspires many subsequent works.

In this work, we summarize our practical experiences in recent months and present the technical details of our Open-Sora Plan, which generates high-quality and long-duration videos queried by various categories of conditions including text prompts, multiple images, and structure control signals (canny, depth, sketch, etc.). As illustrated in Fig. 1, we divide the video generation model into three key components and propose improvements for each part:

• Wavelet-Flow Variational Autoencoder. To reduce memory usage and enhance training speed, we propose WF-VAE, a model that obtains multi-scale features in the frequency domain through multi-level wavelet transform. These features are then injected into a

∗See Contributions section for full author list.

Technical Report.

#### 模型框架

Pixel Space Low Dimension Latent Space Conditioning

Depth

[Figure 1]

[Figure 2]

[Figure 3]

Spacetime Patches Latent Noise

## 𝜀𝜀

[Figure 4]

Diffusion Process

[Figure 5]

𝓏𝓏𝑇𝑇

Canny Sketch

[Figure 6]

[Figure 7]

Diffusion Transformer

𝓧𝓧

×(𝑇𝑇 − 1)

[Figure 8]

[Figure 9]

Multi-head Attention

[Figure 10]

𝒟𝒟

Scale MLP

[Figure 11]

Condition Encoder

𝜏𝜏𝜃𝜃

𝓏𝓏 𝓏𝓏𝑇𝑇−1

𝓧𝓧

- Figure 1: The model architecture of the Open-Sora Plan consists of a VAE, a Diffusion Transformer, and conditional encoders. The conditional injection encoders enable precise manipulation of individual frames (whether it’s the first frame, a subset of frames, or all frames) using designated structural signals, such as images, canny edges, depth maps, and sketches.

- 1. 视觉编码器和解码器
- 2. Diffusion Transformer去噪模型
- 3. 条件注入模型：深度图、姿态图、草图、文本等控制条件

convolutional backbone using a pyramid structure. We also introduced the Causal Cache method to address the issue of latent space disruption caused by tiling inference.

- • Joint Image-Video Skiparse Denoiser. We first change the 2+1D Sora-like video generation denoiser to a 3D full attention structure, significantly enhancing the model’s ability to understand the world, including object motion, camera movement, physics, and human actions. Our denoiser is capable of creating both high-quality images and videos with specific designs. We also introduce a cheap but effective operation called Skiparse Attention for further reducing computation.
- • Condition Controllers. We design a frame-level image condition controller to introduce image conditions into the basic model for supporting various tasks including Image-to-Video, Video Transition, and Video Continuation in one framework. Additionally, we develop a novel network architecture to introduce structure conditions into our base model for controllable generation.

In addition, we carefully design a series of assistant strategies during all stages for training more efficiently and achieving more appreciated results in inference:

- • Min-Max Token Strategy. The Open-Sora Plan uses min-max tokens for training, which aggregates data of different resolutions and durations within the same bucket. This strategy unlocks efficient NPUs/GPUs computation and maximizes the effective usage of data.
- • Adaptive Gradient Clipping Strategy. We propose an adaptive gradient clipping strategy that detects outlier data based on the gradient norm at each step, preventing outliers from skewing the model’s gradient direction.
- • Prompt Refinement Strategy. We develop a prompt refiner that enables the model to reasonably expand input prompts while following semantics. Prompt refiner alleviates the issue of inconsistencies in prompt length and descriptive granularity during training and generation, significantly enhancing the stability of video motion and enriching details.

Moreover, we propose an efficient data curation pipeline to automatically filter and annotate visual data from uncleaned datasets:

- • Multi-dimensional Data Processor. Our data curation pipeline includes detecting jump cuts, clipping videos, filtering out fast or slow motion, cropping edge subtitles, filtering aesthetic scores, assessing video technical quality, and annotating captions.
- • LPIPS-Based Jump Cuts Detection. We implement a video cut detection method based on Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018) to prevent incorrect segmentation of fast-motion shots.

We notice that our Open-Sora Plan is an underway open-source project and we will make continuous efforts towards high-quality video generation. All latest news, codes, and model weights will be publicly updated at https://github.com/PKU-YuanGroup/Open-Sora-Plan.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

WT

Cin

[Figure 16]

[Figure 17]

###### Input Video

[Figure 18]

In In

[Figure 19]

WT

In

[Figure 20]

[Figure 21]

[Figure 22]

ConvIn(Cin)

[Figure 23]

DownBlock1

DownBlock2

[Figure 24]

[Figure 25]

ConvOut

[Figure 26]

WT

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

###### Multi-level In Haar WT

(b) Multi-Level Haar WT

###### Latent Representation

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

IWT

Output Video

Cout

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

ConvOut(Cout)

IWT

[Figure 41]

Out

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

UpBlock1

UpBlock2

ConvIn

[Figure 48]

[Figure 49]

Multi-level Haar IWT

| | |
|---|---|
|Ou|t|

IWT

Out

[Figure 50]

(c) Multi-Level Haar IWT

Out

[Figure 51]

[Figure 52]

Main Energy Flow Pathway Concat

In Inflow Block Out Outflow Block

(a) Architecture of WF-VAE

- Figure 2: Overview of WF-VAE. WF-VAE (Li et al., 2024b) consists of a backbone and a main energy path, with such a path injecting the main flow of video energy into the backbone through concatenations.

##### 2 Core Models of Open-Sora Plan

###### 2.1 Wavelet-Flow VAE

Preliminary. The multi-level Haar wavelet transform decomposes video signals by applying scaling filter h = √12[1,1] and wavelet filter g = √12[1,−1] along temporal and spatial dimensions. For a video signal V ∈ RC×T×H×W, where C, T, H, and W correspond to the number of channels, temporal frames, height, and width, the 3D Haar wavelet transform at layer l is defined as:

S(ijkl) = S(l−1) ∗ (fi ⊗ fj ⊗ fk), (1)

where fi,fj,fk ∈ h,g represent the filters applied along each dimension, and ∗ represents the convolution operation. The transform begins with S(0) = V, and for subsequent layers,

S(l) = S(hhhl−1), indicating that each layer operates on the low-frequency component from the previous layer. At each decomposition layer l, the transform produces eight sub-band compo-

nents: W(l) = {S(hhhl) ,S(hhgl) ,S(hghl) ,S(ghhl) ,S(hggl) ,S(gghl) ,S(ghgl) ,S(gggl) }. Here, S(hhhl) represents the lowfrequency component across all dimensions, while S(gggl) captures high-frequency details. To implement different downsampling rates in the temporal and spatial dimensions, a combination of 2D and 3D wavelet transforms can be implemented. Specifically, to obtain a compression rate of 4×8×8 (temporal×height×width), we can employ a combination of two-layer 3D wavelet transform followed by one-layer 2D wavelet transform.

Training Objective. Building upon the training strategies outlined in Rombach et al. (2022a), the proposed loss function integrates several components: reconstruction loss (including both L1 and perceptual losses (Zhang et al., 2018)), adversarial loss, and KL divergence regularization. As illustrated in Fig. 2, our model architecture emphasizes a low-frequency energy flow and enforces symmetry between the encoder and decoder. To preserve this architectural principle, we introduce a novel regularization term, denoted as LWL (WL loss), which ensures structural consistency by penalizing deviations from the expected energy flow:

LWL = |Wˆ (2) − W(2)| + |Wˆ (3) − W(3)|. (2) The overall loss function is defined as:

L = Lrecon + λadvLadv + λKLLKL + λWLLWL. (3)

where λadv, λKL, and λWL are weighting coefficients for the corresponding loss components. Following (Esser et al., 2021), we adopt dynamic adversarial loss weighting to balance the relative

gradient magnitudes of the adversarial and reconstruction losses:

[Lrecon]∥ ∥∇GL

∥∇GL

- 1

- 2

, (4)

λadv =

[Ladv]∥ + δ

where ∇GL[·] represents the gradient with respect to the final layer of the decoder, and δ = 10−6 is introduced for numerical stability.

Causal Cache. We substitute regular 3D convolutions with causal 3D convolutions (Yu et al., 2024) in WF-VAE with kt −1 temporal padding at the start, enabling unified processing of images and videos. We extract the first frame and process the remaining frames in chunks of size Tchunk for efficient inference of T-frame videos. We cache Tcache(m) tail frames between chunks, where:

Chunk 0 Chunk 1 Chunk 2

Causal padding Causal cache

[Figure 53]

[Figure 54]

Figure 3: Illustration of Causal Cache.

mTchunk st

+ 1⌋. (5)

Tcache(m) = kt + mTchunk − st⌊

This method necessitates that (T − kt) is divisible by st and (T − 1) mod st = 0. We given a illustrated sample for understanding in Fig. 3, with kt = 3,st = 1,Tchunk = 4, Tcache(m) = 2 frames are cached.

Training Details. We utilize the AdamW optimizer (Kingma and Ba, 2014; Loshchilov and Hutter, 2019) with parameters β1 = 0.9 and β2 = 0.999, maintaining a fixed learning rate of 1 × 10−5. Our training process consists of three stages: (i) In the first stage, following the methodology of (Chen et al., 2024b), we preprocess videos to contain 25 frames at a resolution of 256 × 256, with a total batch size of 8. (ii) We update the discriminator, increase the number of frames to 49 and halve the frames per second (FPS) to enhance motion dynamics. (iii) We find that a large λlpips adversely affects video stability; hence, we update the discriminator again and set λlpips to 0.1. The initial stage is trained for 800,000 steps, and the subsequent stages are each trained for 200,000 steps. The training process is conducted on 8 NPUs (Liao et al., 2021)/GPUs. We employ a 3D discriminator and initiate GAN training from the beginning.

- 2.2 Joint Image-Video Skiparse Denoiser 2.2.1 Model Overview

As shown in Fig. 4, we compress input images or videos from pixel space to latent space for denoising training with the diffusion model. Given an input latent x ∈ RB×C×T×H×W, we first split latent into small tokens by a 3D convolutional layer and flattened into a 1D sequence, with converting the latent dimension C to dimension D. We use kernel sizes kt = 1,kh = 2 and kw = 2, with strides matching the kernel sizes, resulting in a total of L = kTHW

tkhkw tokens. We further use mT5-XXL (Xue, 2020) as the text encoder to map text prompts to a high-dimensional feature space, and we also convert text feature to dimension D through a single MLP layer.

- 3D RoPE. We employ 3D rotational position encoding, which allows the model to directly compare relative differences between positions rather than relying on absolute positions. We define the computation process of nD RoPE. After “patchifying” operation, the latent X ∈ RB×L×D is divided

into n parts along the D dimension, e.g., X = [X1,...,Xn], where Xi ∈ RB×L×Dn , i ∈ [1,...,n], and we apply RoPE on partitioned tensor Xi. Assuming the RoPE operation (Su et al., 2024) is denoted as RoPE(Xi), we inject the relative position encoding of the i-th dimension into tensor Xi, and concatenate processed tensors along the D dimension to obtain the final result:

Xropei = RoPE(Xi), (6) Xfinal = Concat Xrope1 ,...,Xropen , (7)

where Concat(·) denotes the concatenate operation and Xfinal ∈ RB×L×D. When n = 1, it is equivalent to applying RoPE on a 1D sequence in large language models. When n = 2, it can be

Latent

Patchfy RoPE

Caption

Latent

Norm

Scale1, Shift1

| | |
|---|---|
| | |

Self-Attention

Timestep

Sparse/1

mT5-XXL

Patchfy

Position ids

Gate1

Sparse/4

RoPE3d

+

······

Cross-DiT-Block-Sparse/1

Cross-Attention T5 feature

+

Sparse/4

MLP

Cross-DiT-Block-Sparse/4

Norm

......

Scale2, Shift2

Sparse/1

Timestep

FFN

Cross-DiT-Block-Sparse/1

Gate2

UnPatchfy

+

UnPatchfy

Output

（a) Model Structure （b) Block Structure

- Figure 4: Overview of the Joint Image-Video Skiparse Denoiser. The model learns the denoising process in a low-dimensional latent space, which is compressed from input videos via our WaveletFlow VAE. Text prompts and timesteps are injected into each Cross-DiT block layer equipped with 3D RoPE. Our Skiparse attention is applied to every layer except the first and last two layers.

Latent

RoPE

Norm

Scale1, Shift1

Caption

Latent

| | |
|---|---|
| | |

Patchfy

Pooled CLIP Timestep

Self-Attention

| | |
|---|---|
|Spar|se/1|
| | |
|Spar|se/4|
| | |
|Spars|e/16|
| | |
|Spars|e/64|
|C| |
|Spars|e/16|
|C| |
|Spar|se/4|
|C| |
|Spar|se/1|
| | |

Gate1

OpenCLIP-G T5-XXL

Patchfy

Position ids

viewed as 2D RoPE applied along the height and width directions of an image. When n = 3, RoPE is successfully applied to video data by incorporating relative position encoding in both the temporal and spatial dimensions to enhance the representation of sequences.

T5 feature

+

pooled

RoPE3d

Norm Cross-Attention

Norm

Scale2, Shift2

MLP

Block Design. Inspired by large language model architectures (Dubey et al., 2024; Yang et al., 2024a; Jiang et al., 2023; Young et al., 2024), we adopt a pre-norm transformer block structure primarily comprising self-attention, cross-attention, and a feedforward network. Following (Peebles and Xie, 2023; Chen et al., 2023a), we map timesteps to two sets of scale, shift, and gate parameters through adaLN-Zero (Peebles and Xie, 2023). We then inject such two sets of values to self-attention and the FFN separately, and 3D RoPE is employed in self-attention layers. In version 1.2, we start to introduce Full 3D Attention instead of 2+1D Attention for significantly enhancing video motion smoothness and visual quality. However, the quadratic complexity of Full 3D Attention requires substantial computational resources, thus we propose a novel sparse attention mechanism. To ensure direct 3D interaction, we retain Full 3D Attention in the first and last two layers.

Cross-DiT-Block-Sparse/1

Gate2

+

+

Cross-DiT-Block-Sparse/4

Norm

MLP

......

Scale3, Shift3

FFN

Cross-DiT-Block-Sparse/1

Timestep

Gate3

+

UnPatchfy

UnPatchfy

Output

###### 2.2.2 Skiparse Attention

Pooled CLIP Timestep

The 2+1D Attention widely leveraged by former video generation methods calculates frame interactions only along the temporal dimension, theoretically and practically limiting video generation performance. Compared to 2+1D Attention, Full 3D Attention represents global c for allowing content from arbitrarily spatial and temporal positions to interact, which ap aligns well with real-world physics. However, Full 3D Attention is time-consuming and inefficient, visual information often contains considerable redundancy, making it unnecessary to establish on across all spatiotemporal tokens. An ideal spatiotemporal modeling approach should em on that minimizes the overhead from redundant visual information while capturing the co of the dynamic physical world. Reducing redundancy requires avoiding connections among tokens, yet global attention remains essential for modeling complex physical interactions.

Patchfy

Caption

Latent

| | |
|---|---|
|alculaSpar|se/1tion|
|proac|h|
|Spar|se/2|
| |as|
|Spara|se/4ttenti|
| | |
|employSpar|attentionse/8|
|C<br><br>mple al|xities l|
|Spar|se/4|
|C| |
|Spara S|se/2kipar|
|C<br><br>the roup|original<br><br>Skip|
|Spar|se/1|
|re|duces|

Latent T5 feature

OpenCLIP-G T5-XXL

Patchfy

Position ids

- Scale1, Shift1

Norm

Scale1, Shift1

Norm

Gate1 Gate1

- Scale2, Shift2

pooled

RoPE3d

QKV_Linear QKV_Linear QK_Norm QK_Norm

MLP

MM-DiT-Block-Sparse/1

RoPE Attention

Proj_out Proj_out

+

MM-DiT-Block-Sparse/4

To balance the computation efficiency and spatiotemporal modeling ability, we propose se (Skip-Sparse) Attention mechanism. Denoiser with Skiparse Attention only modifies al attention layers to two alternating sparse attention operations named Single Skip and G in Transformer Blocks. Giving a sparse ratio k, the sequence length in the attention operation to

MLP

......

+ +

MM-DiT-Block-Sparse/1

Norm

Norm

Timestep

Scale2, Shift2

UnPatchfy

Share FFN

UnPatchfy

5

Gate3 Gate3

+ +

Latent Output

T5 feature Output

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Space-Time Tokens

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

[Figure 85]

[Figure 86]

Flattened Tokens

Sparse Ratio = 2

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

Single Skip

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Group Skip

- Figure 5: Calculation process of Skiparse Attention with sparse ratio k = 2 for example. In our Skiparse Attention operation, we alternately perform the Single Skip and the Group Skip operations, reducing the sequence length to 1/k compared to the original size in each operation.

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

[Figure 149]

[Figure 150]

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

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

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

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

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

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

H

T

W

3D Full Attention

(equivalent to k=1)

2+1D Attention

(equivalent to k=HxW)

Skip + Window Attention

(Figure shows the case k = 2)

Skiparse Attention (Ours)

(Figure shows the case k = 2)

2N Block

2N+1 Block

- Figure 6: The interacted sequence scope of different attention mechanisms. Various attention mainly differ in the number and position of selected tokens during attention computations.

- 1 k compared to the original, and batch size increases by k-fold, lowering the theoretical complexity of self-attention to k1, while cross attention complexity remains unchanged. The Calculation process of two skip operations is shown Fig. 5. In Single Skip operation, the elements located at positions [0,k,2k,3k,...], [1,k + 1,2k + 1,3k + 1,...], ..., [k − 1,2k − 1,3k − 1,...] are bundled into a sequence, e.g., each token performs attention with tokens spaced k − 1 apart.

In Group Skip operation, the elements at positions [(0,1,...,k − 1),(k2,k2 + 1,...,k2 + k − 1),(2k2,2k2 + 1,...,2k2 + k − 1),...], [(k,k + 1,...,2k − 1),(k2 + k,k2 + k + 1,...,k2 + 2k − 1),(2k2 +k,2k2 +k+1,...,2k2 +2k−1),...], ..., [(k2 −k,k2 −k−1,...,k2 −1),(2k2 −k,2k2 − k−1,...,2k2 −1),(3k2 −k,3k2 −k−1,...,3k2 −1),...] are bundled as a sequence. Concretely, we first group adjacent tokens in segments of length k, then bundle these groups with other groups that are spaced k − 1 groups apart into a sequence. For instance, in [(0,1,...,k − 1),(k2,k2 + 1,...,k2 + k − 1),(2k2,2k2 + 1,...,2k2 + k − 1),...], each set of indices in parentheses represents a group, and each group is then connected with another group offset by k − 1 groups to form one sequence. We notice that the main difference between the Group Skip operation and traditional Skip + Window Attention is our operation involves not only grouping but also skipping, which is ignored by previous attempts. Concretely, Window Attention only groups adjacent tokens without connecting skipped groups into one sequence. The distinctions among these attention methods are illustrated in Fig. 6, with dark tokens representing the tokens involved in one attention calculation.

We further notice that the attention in 2+1D DiT corresponds to k = HW (Skip operation in Group Skip has no effect when T ≪ HW), while Full 3D DiT corresponds to k = 1. In Skiparse Attention, k is typically chosen to be close to 1, yet far smaller than HW, making the Skiparse Attention approach the effectiveness of Full 3D Attention while decreasing the computation cost.

Additionally, we propose the concept of Average Attention Distance (ADavg) to quantify how closely a given attention aligns with Full 3D Attention. This concept is defined as follows: If at least m attention calculations are required to establish a connection between any two tokens A&B, the attention distance A→B is m (Noticing that the attention distance between a token and itself is zero). Thus the ADavg for an attention mechanism is the mean of the attention distances across all token directions in input sequences, and ADavg reflects the modeling efficiency among all tokens for the corresponding attention method. To calculate the specific ADavg of different attention methods, we can first identify which tokens have an attention distance of 1, and tokens with an attention distance of 2 can be determined. Therefore, we give the ADavg and calculation process following:

For Full 3D Attention, each token can interact with any other token in one attention calculation, resulting in the ADavg = 1.

For 2+1D Attention, any two tokens can be directed with an attention distance between 1 and 2. In the 2N Block, attention operates over the (H,W) dimensions, where tokens within this region have an attention distance of 1. In the 2N + 1 Block, attention operates along the T dimension, and attention distance is also 1 for these tokens. The total number of tokens with an attention distance of

###### 1 is (HW + T − 1) − 1 = HW + T − 2. Therefore, ADavg of 2+1D Attention is:

1 THW

[1 × 0 + (HW + T − 2) × 1

ADavg =

+(THW − (HW + T − 1)) × 2]

(8)

1 T

1 HW

= 2 −

+

.

For Skip + Window Attention, aside from the token itself, there are THWk −1 tokens with an attention distance of 1 in the 2N Block, and k − 1 tokens with an attention distance of 1 in the 2N + 1 Block.

Thus, the total number of tokens with an attention distance of 1 is THWk + k − 2. Therefore, ADavg of Skip + Window Attention is:

1 THW

THW k

1 × 0 +

+ k − 2 × 1

ADavg =

THW k

(9)

+ THW −

+ k − 1 × 2

1 k

k THW

= 2 −

+

.

In Skiparse Attention, aside from the token itself, THWk − 1 tokens have an attention distance of 1 in the 2N Block, and THWk − 1 tokens have an attention distance of 1 in the 2N + 1 Block. Notably, THW

k2 − 1 tokens can establish an attention distance of 1 in both blocks and should not be counted twice. Therefore, ADavg in Skiparse Attention is:

2THW k − 2 −

THW k2 − 1 × 1

1 THW

1 × 0 +

ADavg =

2THW k −

THW k2 × 2

(10)

+ THW −

1 k2 −

1 THW ≈ 2 −

2 k

1 k2

2 k

= 2 −

+

+

.

We notice that the actual sequence length is k⌈THWk2 ⌉ rather than THWk in the Group Skip of the

###### 2N + 1 Block. Our calculation assumes the ideal case where k ≪ THW and THW mod k = 0,

yielding k⌈THWk2 ⌉ = k · THWk2 = THWk . In practical applications, excessively large k values are typically avoided, making this derivation a reasonably accurate approximation for general usage.

For the commonly used resolution of 93×512×512, using a causal VAE with a 4×8×8 compression rate and a convolutional layer with a 1×2×2 kernel for patch embedding, we obtain a latent shape of

- Table 1: Comparison of the different attention mechanisms. Across multiple comparison metrics, Skiparse Attention is closer to Full 3D Attention, giving it the best spatiotemporal modeling capability apart from Full 3D Attention.

Attention Mechanisms Speed

Modeling

Global Attention

Block

Average Attention Distance

Capability Computation Full 3D Attention Slow Strong All blocks Equal 1 2+1D Attention Fast Weak None block Not Equal 2 − (T1 + HW1 ) Skip + Window Attention Middle Weak Half blocks Not Equal 2 − (k1 + THWk ) Skiparse Attention Middle Strong All blocks Equal 2 − k2 + k12 , 1 < k ≪ THW

- Table 2: The average attention distance ADavg of different attention mechanisms. Results are calculated when the latent shape is 24 × 32 × 32.

Attention Mechanisms ADavg Full 3D Attention 1.000 2+1D Attention 1.957 Skip + Window Attention (k = 2) 1.500 Skip + Window Attention (k = 4) 1.750 Skip + Window Attention (k = 8) 1.875 Skiparse Attention (k = 2) 1.250 Skiparse Attention (k = 4) 1.563 Skiparse Attention (k = 8) 1.766

24 × 32 × 32 as input sequence for attention calculations. We summarize the characteristics of these attention types in Tab. 1, and ADavg for different attention methods when latent shape is 24×32×32 in Tab. 2. Considering the balance between computational load and Average Attention Distance, we use Skiparse Attention with k = 4 in our implementations.

###### 2.2.3 Training Details

Similar to previous works (Zheng et al., 2024; Chen et al., 2024a; Blattmann et al., 2023), we use a multi-stage approach for model training. Starting with training an image model, our joint denoiser learns a rich understanding of static visual features, as many effective visual patterns in images also apply to videos. Benefiting from the 3D DiT architecture, all parameters transfer seamlessly from images to videos. Thus, we adopt a progressive training strategy from images to videos. For all training stages, we use v-prediction diffusion loss with zero terminal SNR (Lin et al., 2024b). We use min-snr weighting strategy (Hang et al., 2023) with γ = 5.0 to accelerate the convergence process. The text encoder has a maximum input length of 512. We use AdamW Kingma and Ba (2014); Loshchilov and Hutter (2019) optimizer with parameters β1 = 0.9 and β2 = 0.999. Details of leveraged datasets in training stages are shown in Sec. 4

Text-to-Image Pretraining. The objective of this stage is to learn a visual prior that enables fast convergence when training on videos, reducing dependency on large-scale video datasets. Since the weights of Full 3D Attention can efficiently transfer to Skiparse Attention, we first train a Full 3D Attention model on 256 × 256 images to generate text-conditioned images, for approximately 150k steps. We then inherit the model weights and replace Full 3D Attention with Skiparse Attention, allowing tuning from a 3D dense attention model to a sparse attention model. The tuning process involves around 100k steps, a batch size of 1024, and a learning rate of 2e-5. Image datasets includes SAM, Anytext, and Human-images.

Text-to-Image&Video Pretraining. We jointly train on images and videos, with a maximum shape of 93 × 640 × 640. The pretraining process includes approximately 200k steps, a batch size of 1024, and a learning rate of 2e-5. Image data consists almost entirely of SAM from version 1.2.0, and the leveraged video dataset is the original Panda70M.

Text-to-Video Fine-tuning. The model nearly converges around 100k steps, with no substantial gains observed by 200k steps. Following the procedures in Sec. 4, we refine the data by cleaning and re-captioning. Fine-tuning is conducted with the filtered Panda70M and additional high-quality data

[Figure 503]

Video VAE Encoder

[Figure 504]

Text Encoder

A rocket ascends slowly into the sky.

Masked Video

Diffusion Transformer

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Video VAE Decoder

[Figure 511]

[Figure 512]

[Figure 513]

Reshape Encoding

Multi-head Attention

Scale MLP

[Figure 514]

Output

Mask

Reverse Diffusion Process

[Figure 515]

Noise

- Figure 7: Overview of our Image Condition Controller. Our Controller unifies multiple image conditional tasks including image-to-video, video transition, and video continuation in one framework when giving masks are changed.

Our Structure Condition Controller

T2V Transformer Block 1

[Figure 516]

T2V Transformer Block 2

[Figure 517]

T2V Transformer Block M-1

[Figure 518]

T2V transformer Block M

[Figure 519]

Time &Text

……

……

High-level Representation

Projector

Encoder

[Figure 520]

Zt

Patchify

to Tokens Zt-1

UnPatchify

Token-wise Transformation 1

[Figure 521]

Token-wise Transformation M-1

Token-wise Transformation 2

[Figure 522]

[Figure 523]

Token-wise Transformation M

[Figure 524]

Encoder Transformer ……

Block 1

3D-Convolution Downsample

[Figure 525]

[Figure 526]

Patchify to Tokens

[Figure 527]

[Figure 528]

Open-Sora Plan Pre-trained Model

Structural Control Signal

[Figure 529]

Encoder Transformer Block K

[Figure 530]

- Figure 8: Overview of our Structure Condition Controller. The structure Controller contains two light components including an encoder that focuses on extracting a high-level representation from the structural signals and a projector that transforms such representation into injection features. Finally, we directly add obtained injection features to the pre-trained model for structure control.

at a fixed resolution of 93 × 352 × 640. This process runs for 30k steps with a learning rate of 1e-5, utilizing 256 NPUs/GPUs with a total batch size of 1024.

- 2.3 Conditional Controllers

- 2.3.1 Image Condition Controller

Inspired by Stable Diffusion Inpainting (Rombach et al., 2022b), we regard the image conditional tasks as an inpainting task in the temporal dimension for a more flexible training paradigm.

The image condition model is initialized by our text-to-video weights. As shown in Fig. 7, it adds two additional inputs including given mask and masked video, which are concatenated with the latent noise and then fed into the Denoiser. For the given mask, instead of employing VAE for encoding, we adopt the “reshape” operation to align latent dimensions due to the temporal down-sampling in VAE will damage the control accuracy of masks. For the masked video, we multiply the original video by the given mask and then input the multiplied video into VAE for encoding.

Unlike previous works based on 2+1D Attention, which inject semantic features of images (usually extracted via CLIP (Radford et al., 2021)) into the UNet or DiT to enhance cross-frame stability (Blattmann et al., 2023; Xing et al., 2025; Xu et al., 2024a), we simply alter the input channels of the DiT without incorporating semantic features for control. We observe that leveraging various semantic injection methods can not noticeably improve the generated results while instead limiting the range of motion, thus we discard the image semantic injection module in our experiments.

Training Details. For training configuration, we adopt the same settings as the text-to-video model, including v-prediction, zero terminal SNR, and min-snr weighting strategy, with parameters consistent with the text-to-video model. We also use the AdamW optimizer with a constant learning rate of 1e-5 and utilize 256 NPUs a batch size fixed at 512.

Clear T2V I2V

Thanks to the flexibility of different mask types in our inpainting framework, we design a progressive training strategy that gradually increases the difficulty of training tasks as shown in Fig. 9, which strategy can lead to smoother training curves and improve motion consistency. The masks used during training are set as follows: (1) Clear: Retain all frames. (2) T2V: Discard all frames. (3) I2V: Retain only the first frame but discard the rest. (4) Transition: Retain only the first and last frames but discard the rest. (5) Continuation: Retain the first n frames but discard the rest. (6) Random: Retain n randomly

Transition Continuation Random

- Figure 9: Different types of masks for image-conditioned generation. Black masks indicate corresponding frames are retained, while white masks indicate frames are masked.

selected frames but discard the rest. Concretely, Our progressive training strategy includes two stages. In Stage 1, we train on multiple simple tasks at a low resolution. In Stage 2, we train the image-to-video and video transition tasks at a higher resolution.

- Stage 1: Any resolution and duration within 93 × 102400 (320 × 320), using unfiltered motion and aesthetic low-quality data. The task ratios at different steps are as follows:

- 1. T2V 10%, Continuation 40%, Random 40%, Clear 10%. Ensure that at least 50% of the frames are retained during continuation and random mask, training with 4 million samples.
- 2. T2V 10%, Continuation 40%, Random 40%, Clear 10%. Ensure that at least 25% of the frames are retained during continuation and random mask, training with 4 million samples.
- 3. T2V 10%, Continuation 40%, Random 40%, Clear 10%. Ensure that at least 12.5% of the frames are retained during continuation and random mask, training with 4 million samples.
- 4. T2V 10%, Continuation 25%, Random 60%, Clear 5%. Ensure that at least 12.5% of the frames are retained during continuation and random mask, training with 4 million samples.
- 5. T2V 10%, Continuation 25%, Random 60%, Clear 5%, training with 8 million samples.
- 6. T2V 10%, Continuation 10%, Random 20%, I2V 40%, Transition 20%, training with 16 million samples.
- 7. T2V 5%, Continuation 5%, Random 10%, I2V 40%, Transition 40%, training with 10 million samples.

- Stage 2: Any resolution and duration within 93×236544 (e.g., 480×480, 640×352, 352×640), using filtered motion and aesthetic high-quality data, ratios of different tasks are T2V 5%, Continuation 5%, Random 10%, I2V 40%, Transition 40%, training with 15 million samples.

After completing the two-stage training, we draw on the approach mentioned in Yang et al. (2024b), adding slight Gaussian noise to the conditional images to enhance generalization during fine-tuning, with utilizing 5 million filtered motion and aesthetic high-quality data.

###### 2.3.2 Structure Condition Controller

When imposing structural control on our retained text-to-image model, an intuitive idea is to use previous control methods Zhang et al. (2023); Mou et al. (2024); Li et al. (2024a); Guo et al. (2025) specified for the U-net-based base models. However, most of these methods are based on ControlNet Zhang et al. (2023), which copies half of the base model to process the control signals and will increase the hardware consumption by nearly 50%. The additional consumption is immense, as the original expense of our Open-Sora Plan base model is already extremely high. Although some works Mou et al. (2024); Peng et al. (2024) try to replace the heavy copy of the base model with a lighter network at the sacrifice of controllability, these will probably lead to poor alignment with the input structural signals and the generated video when used for our base model.

To more efficiently add structural control to our base model, we propose a novel Structure Condition Controller, as shown in Fig. 8. Specifically, we suppose the denoiser of our base model contains M transformer blocks. For the j-th 1 ≤ j ≤ M transformer block Tj in the base model, its output is a series of tokens Xj, which can be expressed as:

Xj = Tj(Xj−1). (11) Given a structural signal CS, the encoder E extracts the high-level representation R from CS:

###### R = E(CS). (12)

Then, the projector P, containing M transformations with the same process, transforms R into the injection feature F, including M elements, which can be expressed as:

P = [P1,P2,...PM], (13) F = [F1,F2,...,FM], (14)

Fj = Pj(R). (15)

Here Pj denotes the j transformation of P that transform R to Fj, the j-th element of F. To impose structural control on the base model, we can directly add Fj to Xj:

Xj = Xj + Fj. (16)

To satisfy the above equation, we should ensure the shape of Fj equals Xj. To achieve this, we use the following design of our encoder E and projector P. Specifically, in the encoder E, we first

downsample CS to make its shape the same as Zt with a tiny 3D convolution-based network. Then, we flatten CS to tokens with the same shape as Xj(1 ≤ j ≤ M). After that, to obtain R, these tokens are processed by K transformer blocks, which maintain the token’s shape. For the projector P, we only need to promise Pj will not change the token shape of R. Thus, we design Pj as a token-wise transformation with the same input and output shape, such as a linear FC-layer or two-layer MLP, which is efficient and can maintain the token shape.

Training Details. We utilize the Panda70M dataset to train our Structure Controller. Given a video clip, we use the specified signal extractors to extract the corresponding structural control signals. Specifically, we extract the canny, depth, and sketch, by canny detector Canny (1986), Midas Birkl et al. (2023), and PiDiNet Su et al. (2021), respectively. We train our Structure Controller for 20k steps, on 8 NPUs/GPUs, with a total batch size of 16, and a learning rate of 4e-6.

##### 3 Assistant Strategies

###### 3.1 Min-Max Token Strategy

To achieve efficient processing on hardware, deep neural networks are typically trained with batched inputs, meaning the shape of the training data is fixed. Traditional methods adopt two approaches including resizing images or padding images to a fixed size. However, both approaches have drawbacks, e.g., the former loses useful information, while the latter has low computational efficiency. Generally, there are three methods for training with variable token counts: Patch n’ Pack (Dehghani et al., 2024; Yang et al., 2024b), Bucket Sampler (Chen et al., 2023a, 2024a; Zheng et al., 2024), and Pad-Mask (Lu et al., 2024; Wang et al., 2024c).

Patch n’ Pack. By packing multiple samples, this method addresses the fixed sequence length limitation. Patch n’ Pack defines a new maximum length, and tokens from multiple data instances are packed into this new data. As a result, the original data is preserved while enabling training with arbitrary resolutions. However, this method introduces significant intrusion into the model code, making it difficult to adapt in fields where the model architecture is not yet stable.

Bucket Sampler. This method packs data of different resolutions into buckets and samples batches from the buckets to ensure all data in a batch have the same resolution. It incurs minimal intrusion into the model code, primarily requiring modifications to the data sampling strategy.

Pad-Mask. This method sets a maximum resolution, pads all data to this resolution, and generates a corresponding mask to exclude loss from the masked areas. While conceptually simple, it has low computational efficiency.

We believe current video generation models are still in an exploratory phase. Patch n’ Pack incurs significant intrusion into the model code, leading to unnecessary development costs. Pad-mask has low computational efficiency, which wastes resources in dense computations like video. The bucket strategy, while requiring no changes to the model code, leads to greater loss oscillation as token count variation increases (with more resolution types), indicating higher training instability. Given a maximum token m, resolution stride s, and a set of possible resolution ratios R = r1h,r1w , r2h,r2w ,..., rnh,rnw , we propose the Min-Max Token strategy for tacking mentioned issues. We notice that s = 8 × 2 is the multiples of spatial downsampling rate in VAE and convolution stride in denoiser, and there are five common resolutions: 11, 34, 43, 169 and 169 in practical needs. For each ratio rih,riw in R, rih and riw are required to be coprime positive integers. The height h and width w are defined as h = rih · k · s and w = riw · k · s, where is the scaling factor k to be determined. The total token count n satisfies the constraint n = h · w ≤ m. Substituting the expressions for h and w, we get:

ni = rih · k · s · (riw · k · s) = rih · riw · k2 · s2, (17) so the constraint becomes:

rih · riw · k2 · s2 ≤ m. (18) Taking the square root of both sides, to ensure k is an integer, we obtain the upper bound result for k:

m rih · riw · s2

ki =

The set of minimum token n is then expressed as:

. (19)

n = min rih · riw · ki2 · s2 | rih,riw ∈ R . (20)

For example, the max token m is typically set as a square rootable number, such as 65536 (256×256), as it reliably supports a 1:1 aspect ratio. Given this, we configure s = 16, and aspect ratios of 3:4 and 9:16. The resulting min token n is 36864 (144 × 256).

As discussed above, we implement the Min-Max Token Training combined with the Bucket Sampler using a custom data sampler to maintain a consistent token count per global batch, though token counts vary across global batches. This approach allows NPUs/GPUs to maintain nearly identical compute times, reducing synchronization overhead. The method fully decouples data sampling code from model code, providing a plug-and-play sampling strategy for multi-resolution, multi-frame data.

###### 3.2 Adaptive Gradient Clipping Strategy

In distributed model training, we often observe loss spikes as shown in Fig. 10, significantly degrade output quality without causing NaN errors. Unlike typical NaN errors that disrupt training, these spikes temporarily increase loss values and are followed by a return to normal levels, which occur sporadically and adversely impact model performance. These spikes arise due to various issues, including abnormal outputs from the VAE encoder, desynchronization in multi-node communication, or outliers in training data leading to large gradient norms.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Figure 10: Plot of spikes in training loss. We observe loss spikes during training that could not be reproduced with a fixed seed.

We attempt many methods including applying gradient clipping, adjusting the β2 in optimizer, and reducing the learning rate, but none of these approaches resolve the issue, which appears randomly and cannot be reproduced even with a fixed seed. Playground v3 (Liu et al., 2024a) encounters the same issue and involves discarding an iteration if the gradient norm exceeds a fixed threshold. However, fixed thresholds may fail to adapt to decreasing gradient norms as training progresses. Therefore, we introduce an adaptive thresholding mechanism that leverages Exponential Moving Averages (EMA) for effective anomaly detection. Our approach mitigates the effects of spikes while preserving training stability and output quality.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
|[Figure 531]<br><br>| | | | |
| | | | | |
| | | | | |

[Figure 532]

[Figure 533]

(a)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

(c)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

(b)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

(d)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| |[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]| | | | |
| | | | | | |
| | | | | | |

(e)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

(f)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

(g)

(h)

- Figure 11: Logging abnormal iterations during training. We resume training at step 75k and display logs from step 75k to 76k, noting an anomaly around step 75.6k. (a) Diffusion model loss during training. (b) Abnormal local batches discarded per step. (c) Gradient norm upper bound plotted based on a 3-sigma criterion. (d) Maximum gradient norm among all local batches. (e) Variance of the maximum gradient norm. Note that most steps involve values close to 0. (f) Maximum value of all processed gradient norms. (g) EMA of the maximum gradient norm. (h) EMA of the variance of the maximum gradient norm.

Let gni denote the gradient norm on NPU/GPUi for i = 1,2,...,N, where N is the total number of NPUs/GPUs. We define the maximum gradient norm across all NPUs/GPUs as:

gnmax = maxN

gni. (21)

i=1

To ensure the threshold adapts to the training dynamics, we use the EMA of the maximum gradient norm emagn and its variance-based EMA emavar, which updated as follows:

emagn = α · emagn + (1 − α) · gnmax, (22) emavar = α · emavar + (1 − α) · (gnmax − emagn)2 , (23)

Table 3: Overview of utilized datasets for fine-tuning prompt refiner. Source Year Length Manual # Num

COCO Lin et al. (2014) 2014 Short Yes 12k DiffusionDB Wang et al. (2022b) 2022 Tags Yes 6k JourneyDB Sun et al. (2024) 2023 Medium No 3k Dense Captions (From Internet) 2024 Dense Yes 0.5k

where α is the update rate for EMA, we set it to 0.99. We can record whether each gradient norm is abnormal based on the 3-sigma rule, denoted as δi:

δi =

- 0, if gni − emagn > 3 ·

√emavar

- 1, otherwise

. (24)

Then, the number of normal gradient norm M can be obtained by summing the indicator functions of all NPUs/GPUs:

N

δi. (25)

M =

i=1

For each NPU/GPU, we define the final gradient update rule based on the detection result. If an anomaly is detected for NPU/GPUi, the gradient for that NPU/GPU is set to zero, or it will be multiplied by MN otherwise:

√emavar

0, if gni − emagn > 3 ·

gifinal =

. (26)

N M · gi, otherwise

After adjusting the gradients, we apply an all-reduce operation across NPUs/GPUs to synchronize the remaining non-zero gradients. In Fig. 11, we illustrate how the moving average gradient norm addresses abnormal data. Fig. 11 (d) and Fig. 11 (e) show a sudden increase in gradient norm on a specific NPU/GPU near step 75.6k, exceeding the moving average of the maximum gradient norm (seen in Fig. 11 (c)). Consequently, the gradient for this local batch is set to zero (logged in Fig. 11 (b)). We also record the post-discard maximum gradient to confirm successful handling. Finally, the processed maximum gradient norm (logged in Fig. 11 (f)) updates the moving average of the maximum gradient norm and its variance in Fig. 11 (g) and Fig. 11 (h). As shown in Fig. 11 (a), the training loss remains stable without spikes, demonstrating that this approach effectively prevents anomalous batches from affecting the training process without discarding entire iterations.

###### 3.3 Prompt Refiner

The training dataset for the video generation model is annotated by Vision Language Models (Chen et al., 2024f; Wang et al., 2024b), providing highly detailed descriptions of scenes and themes, with most annotations consisting of lengthy texts that differ substantially from typical user input. User input is generally less detailed and concise, containing fewer words (e.g., in VBench (Huang et al., 2024), most test texts contain fewer than 30 words, sometimes no more than 5 words). This discrepancy results in a significant gap compared to the textual conditions used in model training, leading to reduced video quality, semantic fidelity, and motion amplitude. To address this gap and enhance the model performance when facing shorter texts, we introduce an LLM to leverage its text expansion and creation capabilities to transform short captions into more elaborate descriptions.

Data preparation. We use GPT-4o to generate paired training texts, using specific prompts to instruct the LLM to supplement detailed actions, scene descriptions, cinematic language, lighting nuances, and environmental atmosphere. These original and LLM-augmented text pairs are then used to train the refiner model. Concretely, the instruct prompt is: rewrite the prompt:“prompt” to contain subject description action, scene description. (Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the prompt more dynamic, making sure it’s a fluent sentence. Our data composition for fine-tuning LLM is shown in Tab. 3. Specifically, COCO Lin et al. (2014) consists of manually annotated data, while JourneyDB Sun et al. (2024) contains labels generated by a visual language model (VLM).

Table 4: Data card of Open-Sora Plan v1.3. “*” denotes that the original team employs multiple models, including OFA (Wang et al., 2022a), mPLUG-Owl (Ye et al., 2023), and ChatGPT (OpenAI,

- 2023) to refine captions. “†” indicates that while we do not release captions generated with QWen2-VL and ShareGPT4Video, the original team has made their generated captions publicly available.

Data Caption

# Num

Domain Dataset Source Captioner

Available Available

SAM SAM LLaVA Yes Yes 11.1M

Anytext Anytext InternVL2 Yes Yes 1.8M Human LAION InternVL2 Yes Yes 0.1M Internal - QWen2-VL No No 5.0M

Image

VIDAL YouTube Shorts Multi-model∗ Yes Yes 2.8M Panda70M YouTube

QWen2-VL

Yes Yes† 21.2M

ShareGPT4Video

Video

Mixkit‡

QWen2-VL Pexels⋏ Yes Yes

StockVideo

0.8M

ShareGPT4Video

Pixabay⋎

‡ https://mixkit.co, ⋏ www.pexels.com, ⋎ https://pixabay.com

Training Details. We perform LoRA fine-tuning using LLaMA 3.1 8B2, completing within 1 hour on a single NPU/GPU. Fine-tuning is conducted for just 1 epoch with a batch size of 32 and a LoRA rank of 64. The AdamW optimizer is used with β1 = 0.9, β2 = 0.999, and a learning rate of 1.5e-4.

##### 4 Data Curation Pipeline

Dataset quality is closely linked to model performance. However, some current open-source datasets, such as WebVid (Bain et al., 2021b), Panda70M (Chen et al., 2024e), VIDAL (Zhu et al., 2023) and HD-VILA (Xue et al., 2022), fall short in data quality. Excessive low-quality data in training disrupts the gradient direction of model learning. In this section, we propose an efficient, structured data-processing pipeline to filter high-quality video clips from raw data. We also present dataset statistics to provide reliable direction for further data enhancement.

###### 4.1 Training Data

As shown in Tab. 4, we obtain 11 million image-text pairs from Pixart-Alpha (Chen et al., 2023a), with captions generated by LLaVA (Liu et al., 2024c). Additionally, we use the OCR dataset Anytext-3M (Tuo et al., 2023), which pairs each image with corresponding OCR characters. We filter Anytext-3M for English data, constituting about half of the entire dataset. Since SAM (Kirillov

- et al., 2023) data (as used in Pixart-Alpha) includes blurred faces, we selected 160k high-quality images from Laion-5B (Schuhmann et al., 2022) to enhance the quality of person-related content in generation. The selection criteria include high resolution, high aesthetic scores, the absence of watermarks, and the presence of people in the images.

For videos, we download approximately 21M horizontal videos from Panda70M (Chen et al., 2024e) using our filtering pipeline. For vertical data, we obtain around 3M vertical videos from VIDAL (Zhu

- et al., 2023), sourced from YouTube Shorts. Additionally, we scrape high-quality videos from CC0-licensed websites, such as Mixkit, Pexels, and Pixabay. These open-source video sites contain no content-related watermarks.

###### 4.2 Data Filtering Strategy

1. Video Slicing. Excessively long videos are not conducive to input processing, so we utilize copy stream method in ffmpeg3 to split videos into 16-second clips.

- 2https://huggingface.co/meta-llama/Llama-3.1-8B
- 3https://ffmpeg.org/

Table 5: Implementation details and discarded data number of different filtering steps.

Curation Step Tools Thresholds Remaining

Video Slicing - Each video is clipped to 16s 100%

Jump Cut LPIPS (Zhang et al., 2018) 32 ≤ frames number ≤ 512 97% Motion Calculation LPIPS (Zhang et al., 2018) 0.001 ≤ motion score ≤ 0.3 89%

OCR Cropping EasyOCR∗ 0.20 ≤ edge 89% Aesthetic Filtration Laion Aesthetic Predictor v2† 4.75 ≤ aesthetic score 49%

Low-level Quality Filtration DOVER (Wu et al., 2023) 0 ≤ technical score 44% Motion Double-Checking LPIPS (Zhang et al., 2018) 0.001 ≤ motion score ≤ 0.3 42%

∗ https://github.com/JaidedAI/EasyOCR † https://github.com/christophschuhmann/improved-aesthetic-predictor

- 2. Jump Cut and Motion Calculation. We calculate the Learned Perceptual Image Patch Similarity (LPIPS) Zhang et al. (2018) between consecutive frames. Outliers are identified as cut points, while the mean value represents motion. Specifically, we utilize the decord4 library to efficiently read video frames with skipping. After reading the video, we calculate the LPIPS values to obtain a set of semantic similarities between frames, denoted as l ∈ L, and compute its mean µ and variance σ. Then, we calculate the zero score of L: Z = {z = l−σµ|l ∈ L}, to obtain the set of potential anomaly indices P = {i|zi > zthreshold,zi ∈ Z}. We further filter the anomalies by Pfinal = {i|L[i] > lthreshold or (zi > zthreshold2 and L[i] > lthreshold2),i ∈ P} to obtain the final set of anomaly indices. Based on our experiments, we set the parameters

as zthreshold = 2.0,lthreshold = 0.35,zthreshold2 = 3.2,lthreshold2 = 0.2. To validate the efficacy of our method, we conduct a manual assessment of 2,000 videos. The result demonstrates that the accuracy meets our predetermined criteria.

- 3. OCR Cropping. We employ EasyOCR to detect subtitles in videos by sampling one frame per second. Based on our estimates for common video platforms, subtitles typically appear in the edge regions, with manual verification showing an average occurrence in 18% of these areas. Therefore, we set the maximum cropping range to 20% of both sides of video spatial size (H,W), i.e., cropped video has (0.6H,0.6W) size and 36% area compared to the original video in extreme cases. We then crop subtitles appearing in the setting range, leaving any text in the central area unprocessed. We consider that text appearing in certain contexts, such as advertisements, speeches, or library settings is reasonable. In summary, we do not assume that all text in a video should be filtered out since certain words contribute significance in specific contexts, and we leave further judgments to aesthetic considerations. We notice that the OCR step only crops text areas without discarding videos.
- 4. Aesthetic Filtration. We use the Laion aesthetic predictor to assess the aesthetic score of a video. The aesthetic predictor effectively filters out videos that are blurry, low-resolution, overly exposed, excessively dark, or contain prominent watermarks or logos. We set a threshold of 4.75 to filter videos, as this value effectively removes extensive text and retains high aesthetic quality. We uniformly sample five frames from each video and average their scores to obtain the final aesthetic score. This filtering process eliminates approximately 40% of videos that do not meet human aesthetic standards.
- 5. Low-level Quality Filtration. However, even when some data have high resolutions, their visual effects can still appear very blurry or exhibit a mosaic-like quality, which is attributed to two factors: (i) Low bitrate or DPI of the video. (ii) Usage of motion blur techniques in 24 FPS videos, which simulate dynamic effects by blurring the image between frames, resulting in smoother visual motion. For these videos with absolutely low quality, aesthetic filtering struggles to eliminate them since frames are resized to a resolution of 224. We aim to utilize a metric independent of the visual content that evaluates absolute video quality, focusing on issues including compression artifacts, low bitrate, and temporal jitter. Finally, we find the technical prediction score from DOVER (Wu et al., 2023), selecting videos with a technical score > 0, which filters out 5% of the videos.

4https://github.com/dmlc/decord

Panda70M

Stock Video

###### VIDAL-10M

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.16

0.25

0.25

0.14

0.20

0.12

0.20

0.10

Proportion

Proportion

Proportion

0.15

0.15

0.08

Anytext

Human-images

###### SAM

0.10

0.06

0.10

1.0

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

1.0

0.04

0.8

0.05

0.05

0.8

0.02

0.8

0.00

0.00

0.00

0.6

Proportion

Proportion

Proportion

0 10 20 30 40 50 Frame

0 10 20 30 40 Frame

0 5 10 15 20 Frame

0.6

0.6

Panda70M

Stock Video

###### VIDAL-10M

0.4

2.5

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

0.4

1.4

0.4

2.5

1.2

0.2

0.2

2.0

0.2

1.0

2.0

0.0

0.0

0.0

1.5

Proportion

Proportion

Proportion

2 3 4 5 6 7 Aesthetic Score

4 5 6 7 Aesthetic Score

3 4 5 6 7 Aesthetic Score

0.8

1.5

0.6

1.0

1.0

0.4

Anytext

Human-images

SAM

0.5

0.5

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

1.0

0.2

0.25

0.5

0.0

0.0

0.0

0.8

5.0 5.5 6.0 6.5 7.0 Aesthetic Score

5.0 5.5 6.0 6.5 7.0 7.5 Aesthetic Score

5.0 5.5 6.0 6.5 7.0 7.5 Aesthetic Score

0.20

0.4

Panda70M

Stock Video

VIDAL-10M

Proportion

Proportion

Proportion

0.6

0.7

0.15

0.7

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

0.3

0.7

0.6

0.6

0.4

0.10

0.6

0.2

0.5

0.5

0.5

0.2

0.05

0.1

0.4

0.4

Proportion

Proportion

Proportion

0.4

0.0

0.00

0.0

0.3

0.3

2250x15001500x22502000x15002247x15002248x15001500x20002254x15002249x15002265x15002667x1500

512x512

768x512512x512512x768910x512683x512512x683767x512769x512512x640512x682

0.3

0.2

0.2

0.2

Resolution

Resolution

Resolution

0.1

0.1

0.1

0.0

0.0

0.0

1280x7201920x10801024x5761280x5761024x720768x576768x4321536x8641024x4321920x864

1280x720640x3601920x1080852x4801080x19201280x5761024x5761024x7202048x10801080x2048

720x1280608x1080576x1024576x1280432x1024720x1024720x720432x1280406x720480x854

(a)

Resolution

Resolution

Resolution

(b)

- Figure 12: (a) Distribution statistics of image datasets. The first row is the aesthetic scores distribution of the data, and the second row is the resolution distribution of the data. (b) Distribution statistics of video datasets. The first row is the duration distribution of the data, the second row is the aesthetic score distribution of the data, and the third row is the resolution distribution of the data.

6. Motion Double-Checking. In our post-check, we find that the changes in subtitles may lead to inaccuracies in motion values because the OCR cropping step occurs after detecting motion values. Therefore, we recheck the motion values and filter out videos according to average frame similarities with L¯ < 0.001 or L¯ > 0.3, which account for 2%.

###### 4.3 Data Annotation

Dense captioning provides additional semantic information for each sample, enabling the model to learn specific correspondences between text and visual features. Supervised by dense caption during diffusion training, the model gradually builds a conceptual understanding of various objects and scenes. However, the cost of manual annotation for dense captions is prohibitive, so large image-language models (Wang et al., 2023; Yao et al., 2024; Chen et al., 2024f, 2023b; Lin et al., 2024a; Liu et al., 2024b; Wang et al., 2024b) and large video-language models (Lin et al., 2023; Chen et al., 2024c; Wang et al., 2024b; Xu et al., 2024c; Liu et al., 2024d; Wang et al., 2024a; Jin

- et al., 2024) are typically used for annotation. This capability allows the model to express complex concepts in dense captions more accurately during image and video generations.

For images, the SAM dataset has available captions generated by LLaVA. Although Anytext contains some OCR-recognized characters, these are insufficient to describe the entire image. Therefore, we use InternVL2 (Chen et al., 2024f) and QWen2-VL-7B (Wang et al., 2024b) to generate captions for the images. The descriptions are as detailed and diverse as possible. The annotation prompt is: Combine this rough caption: “{}”, analyze the image in a comprehensive and detailed manner. “{}” can be recognized in the image.

For videos, in early versions such as Open-Sora Plan v1.1, we use ShareGPT4Video-7B (Chen et al.,

- 2024c) to annotate a portion of the videos. Another portion is annotated with QWen2-VL-7B (Wang

- et al., 2024b), with the input prompt: Please describe the content of this video in as much detail as possible, including the objects, scenery, animals, characters, and camera movements within the video. Please start the description with the video content directly. Please describe the content of the video and the changes that occur, in chronological order.

However, 7B caption models often generate prefixes like “This image” or “The video”. We search all such irrelevant strings and remove them.

- 4.4 Data Statistics

Image Data. The filtered image data primarily includes Anytext, Human-images, and SAM. We have plotted the top-10 most frequent resolutions, along with histograms depicting the distribution of aesthetic scores, as shown in Fig. 12 (a). The plots indicate that the Anytext dataset has a unified resolution 512 × 512. In contrast, Human-images and SAM datasets exhibit more diverse scores and resolutions. Human-images dataset shows a range of scores and multiple resolutions, suggesting varied content, while SAM heavily favors high resolutions 2250×1500. Overall, Anytext is consistent, while Human-images and SAM offer greater diversity in both aesthetic scores and image resolutions.

Video Data. The filtered video data primarily includes Panda70M, VIDAL-10M, and several stock video websites (e.g., Pixabay, Pexels, Mixkit). We have plotted the top 10 most frequent resolutions, along with histograms depicting the distribution of video duration, aesthetic scores, and resolution across the three datasets, as shown in Fig. 12 (b). From the distribution plots, it is evident that both Panda70M and VIDAL-10M contain shorter average video durations and relatively lower aesthetic scores. In contrast, videos from stock video websites tend to have longer durations and higher aesthetic quality. Regarding resolution, the majority of videos across all three datasets are 1280×720, with VIDAL-10M being a vertical video dataset (height > width), while the other two datasets are predominantly landscape (width > height).

- 5 Results

- 5.1 Wavelet-Flow VAE

- Tab. 6 and Fig. 15 present both quantitative and qualitative comparisons with several open-source VAEs, including Allegro (Zhou et al., 2024), OD-VAE (Chen et al., 2024b), and CogVideoX (Yang et al., 2024b). The experiments utilize the Panda70M (Chen et al., 2024d) and WebVid-10M (Bain et al., 2021a) datasets. To comprehensively evaluate reconstruction performance, we adopt the Peak Signal-to-Noise Ratio (PSNR) (Hore and Ziou, 2010), Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018), and Structural Similarity Index Measure (SSIM) (Wang et al.,

2004) as the primary evaluation metrics. Furthermore, the reconstruction Fréchet Video Distance (rFVD) (Unterthiner et al., 2019) is employed to assess visual quality and temporal coherence.

As shown in Tab. 6, WF-VAE-S achieves a throughput of 11.11 videos per second when encoding 33-frame videos at 512×512 resolution. This throughput surpasses CV-VAE and OD-VAE by approximately 6× and 4×, respectively. The memory cost reduces by nearly 5× and 7× compared to these baselines while achieving superior reconstruction quality. For the larger WF-VAE-L model, the encoding throughput exceeds Allegro by 7.8×, with approximately 8× lower memory usage, while maintaining better evaluation metrics. These results demonstrate that the WF-VAE maintains state-of-the-art reconstruction performance while substantially reducing computational costs.

We assess the impact of lossy block-wise inference on reconstruction metrics using contemporary open-source VAE implementations Yang et al. (2024b); Chen et al. (2024b), as summarized in

- Tab. 7. Specifically, we measure reconstruction performance in terms of PSNR and LPIPS on the Panda70M dataset under both block-wise and direct inference conditions. the overlap-fusion-based tiling inference of OD-VAE results in substantial performance degradation. In contrast, CogVideoX exhibits only minor degradation due to its temporal block-wise inference with caching. Notably, our proposed Causal Cache mechanism delivers reconstruction results that are numerically identical to those of direct inference, thereby confirming its lossless reconstruction capability.

###### 5.2 Text-to-Video

We evaluate the quality of our video generation model using VBench Huang et al. (2024) and ChronoMagic-Bench-150 Yuan et al. (2024). VBench, a commonly used metric in video generation, deconstructs “video generation quality” into several clearly defined dimensions, allowing for a fine-grained, objective assessment. However, many metrics are overly detailed and yield uniformly high scores across models, offering limited reference value. Consequently, we select Object Class, Multiple Object, and Human Action dimensions to evaluate the semantic fidelity of generated objects and human actions. Aesthetic quality is used to assess spatial generation effects, while Spatial

- Table 6: Quantitative comparison with state-of-the-art VAEs on WebVid-10M dataset. Reconstruction metrics are evaluated on 33-frame videos at a resolution of 256×256. “T” and “Mem.” denote encoding throughput and Memory cost (GB), assessed on 33-frame videos at a resolution of 512×512. The highest result is highlighted in bold, and the second highest result is underlined.

Channel Model T↑ Mem. ↓ PSNR↑ LPIPS↓ rFVD↓

4

CV-VAE 1.85 25.00 30.76 0.0803 369.23 OD-VAE 2.63 31.19 30.69 0.0553 255.92

Allegro 0.71 54.35 32.18 0.0524 209.68 WF-VAE-S(Ours) 11.11 4.70 31.39 0.0517 188.04 WF-VAE-L(Ours) 5.55 7.00 32.32 0.0513 186.00

16

CogVideoX 1.02 35.01 35.76 0.0277 59.83 WF-VAE-L(Ours) 5.55 7.00 35.79 0.0230 54.36

- Table 7: Quantitative analysis of visual quality degradation induced by block-wise inference on Panda70M. BWI denotes Block-Wise Inference and experiments are conducted on 33 frames with 256×256 resolution. Values highlighted in red signify degradation in comparison to direct inference, whereas values highlighted in green indicate preservation of the quality.

###### Channel Method BWI PSNR↑ LPIPS↓

✗ 30.31 0.0439

OD-VAE

✓ 28.51 (-1.80) 0.0552(+0.011) WF-VAE-L (Ours)

4

✗ 32.10 0.0411

✓ 32.10(-0.00) 0.0411(-0.000)

✗ 35.79 0.0198

CogVideoX

✓ 35.41(-0.38) 0.0218(+0.002) WF-VAE-L (Ours)

16

✗ 35.87 0.0175

✓ 35.87(-0.00) 0.0175(-0.000)

relationship reflected the model’s understanding of spatial relationships. For motion amplitude, we adopted ChronoMagic-Bench since motion evaluation metrics in VBench are considered inadequate.

- Tab. 8 compares the performance of the Open-Sora Plan with other state-of-the-art models. Results indicate that the Open-Sora Plan performs exceptionally well in video generation quality, and it has significant advantages over other models in terms of aesthetic quality, smoothness, and scene restoration fidelity. In addition, our model can automatically optimize the text prompts to further improve the generation quality.

###### 5.3 Condition Controllers

Image-to-Video. The video generation capability of image-to-video depends significantly on the performance of the base model and the quality of the initial frame, resulting in challenges in establishing fully objective evaluation metrics. To illustrate the generation ability of Open-Sora Plan, we select several showcases, as shown in Fig. 19, demonstrating that our model exhibits excellent image-to-video generation capabilities and realistic motion dynamics. Furthermore, We compare the image-to-video results of several state-of-the-art methods in Fig. 18, highlighting that Open-Sora Plan strikes an exceptional balance between the control information of the initial frame and the text. Our method maintains semantic consistency while ensuring high visual quality, demonstrating superior expressiveness compared to other models.

Structure-to-Video. As shown in Fig. 13, our structure condition controller enables the Open-Sora Plan text-to-image model to generate high-quality videos whose any frames (first frame, a few frames, all frames, etc.) can be accurately controlled by given structural signals (canny, depth, sketch, etc.).

- Table 8: Quantitative comparison of Open-Sora Plan and other state-of-the-art methods. “*” donates we use our prompt refiner to get results.

Aesthetic

Object

Multiple CH GPT4o

Model

Size

Action

Spatial Scene

Quality Class Objects Score MTScore

OpenSora v1.2 1.2B 56.18 85.8 83.37 67.51 42.47 58.41 51.87 2.50 CogVideoX-2B 1.7B 58.78 89.0 78.00 53.91 38.59 48.48 38.60 3.09 CogVideoX-5B 5.6B 56.46 77.2 76.85 45.89 41.44 46.43 48.45 3.36 Mochi-1 10.0B 56.94 94.6 86.51 69.24 36.99 50.47 28.07 3.76

OpenSoraPlan v1.3 2.7B 59.00 81.8 70.97 44.46 28.56 35.87 71.00 2.64 OpenSoraPlan v1.3∗ 2.7B 60.70 86.4 84.72 49.63 52.92 44.57 68.39 2.95

[Figure 537]

[Figure 538]

|[Figure 539]|
|---|

|[Figure 540]|
|---|

|[Figure 541]|
|---|

|[Figure 542]|
|---|

|[Figure 543]|
|---|

[Figure 544]

[Figure 545]

[Figure 546]

A man is riding a steed, galloping across the sandy beach, with the glow of the setting sun casting on the sand.

[Figure 547]

|[Figure 548]|
|---|

|[Figure 549]|
|---|

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

A panda is sitting by the lake, playing a guitar.

|[Figure 554]|
|---|

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

A small dog wearing sunglasses is strolling on the beach, with waves lapping against the shore and the sky a bright blue.

- Figure 13: Our structure controller can generate high-quality videos conditioned by specified structural signals corresponding to arbitrary frames.

###### 5.4 Prompt Refiner

The Open-Sora Plan leverages a substantial proportion of synthetic labels during training, resulting in superior performance in dense captioning tasks compared to shorter prompts. However, the evaluation prompts or user inputs are often brief, limiting the ability to accurately assess the model’s true performance. Following DALL-E 3 (Betker et al.), we report evaluation results where our prompt refiner is employed for rewriting input prompts.

imaging_quality human_action

multiple_objects

appearance_style

object_class

aesthetic

During the evaluation, we observe notable improvements in most VBench Huang et al. (2024) metrics when using prompt refiner, particularly in action accuracy and object description. Fig. 14 provides a radar chart that visually highlights the effectiveness of the prompt refiner. Specifically, the performance in human action generation and spatial relationship depiction improved by more than 5%. The semantic adherence for single-object and multi-object generation increased by 15% and 10%, respectively. Additionally, the score for scenery generation increased by 25%. Furthermore, our prompt refiner can translate multilingual into English, allowing the diffusion model to leverage training data and text encoders in English while supporting various languages for inference.

overall_consistency

temporal_style w/o Refiner

w/ Refiner

scene spatial_relationship

Figure 14: Ablations results for leveraging the prompt refiner in VBench. Evaluated videos are generated in 480p.

##### 6 Limitation and Future Work

###### 6.1 Wavelet-Flow VAE

Our decoder architecture is modeled after the design proposed by Rombach et al. (2022a), resulting in a greater number of parameters in the decoder compared to the encoder. While the computational cost remains manageable, we consider these additional parameters to be redundant. Consequently, in future work, we plan to streamline the model to fully exploit the advantages of our architecture.

###### 6.2 Transformer Denoiser

The current 2B model in version 1.3.0 shows performance saturation during the later stages of training. However, our model performs poor in understanding physical laws (e.g., a cup overflowing with milk, a car moving forward, or a person walking), thus we have three hypotheses:

- • Joint training of images and videos. Models such as Open-Sora v1.2 (Zheng et al., 2024), EasyAnimate v4 (Xu et al., 2024b), and Vchitect-2.05 can easily generate high-visual-quality videos, possibly due to their direct inheritance of image weights (Pixart-Sigma (Chen et al., 2024a), HunyuanDiT (Li et al., 2024c), SD3 (Esser et al., 2024)). They train the model with a small amount of video data to learn how to flow along the temporal dimension based on 2D images. However, we train images from scratch with only 10M-level data, which is far from sufficient. In recent work on Allegro (Zhou et al., 2024), they fine-tuned a better text-to-image model based on the T2I weights from Open-Sora Plan v1.2, achieving improved text-to-video results. We have two hypotheses regarding the training strategy: (i) Start joint training from scratch, with images significantly outnumbering videos; (ii) First train a high-quality image model and then use joint training, with a higher proportion of videos at that stage. Considering the learning path and training costs, the second approach may offer more decoupling, while the first aligns better with scaling laws.
- • The model still needs to scale. By observing the differences between CogVideoX-2B (Yang et al., 2024b) and its 5B variant, we can discover that the 5B model understands more physical laws than the 2B model. We speculate that instead of spending excessive effort designing for smaller models, it may be more effective to leverage scaling laws to solve these issues. In the next version, we will scale up the model to explore the boundaries of video generation. We currently have two plans: (i) Continue using the Deepspeed (Rasley et al., 2020)/FSDP (Zhao et al., 2023) approach, sharding the EMA and text encoder across ranks with Zero3 (Rasley et al., 2020), which is sufficient for training 10-15B models. (ii) Adopting MindSpeed6/Megatron-LM (Shoeybi et al., 2019) for various parallel strategies, enabling us to scale the model up to 30B.
- • Supervised loss in training. Flow Matching (Lipman et al., 2022) avoids the stability issues in Denoising Diffusion Probabilistic Models (Ho et al., 2020) (DDPM) when the timestep approaches 0, addressing the zero-terminal signal-to-noise ratio problem (Lin et al., 2024b). Recent works (Zheng et al., 2024; Polyak et al., 2024; Esser et al., 2024) also show that the validation loss in Flow Matching indicates whether the model is converging in the right direction, which is crucial for assessing model training progress. Whether flow-based models are more suitable than v-prediction models requires further ablation studies.

In addition to expanding the model and data scale, we will also explore other efficient algorithm implementations and improved evaluation metrics:

• Exploring more efficient architectures. Although Skiparse Attention significantly reduces FLOPs during computation, these advantages are only noticeable with longer sequence lengths (e.g., resolutions above 480P). Since most pre-training is conducted at a lower resolution (e.g., around 320 pixels), the Skiparse Attention operation has not achieved the desired acceleration ratio in this phase. In the future, we will explore more efficient training strategies to address this issue.

- 5https://github.com/Vchitect/Vchitect-2.0
- 6https://gitee.com/ascend/MindSpeed

- • Introducing more parallelization strategies. In Movie Gen (Polyak et al., 2024), the role of various parallelization strategies in accelerating training for video generation models is highlighted. However, Open-Sora Plan v1.3.0 currently only employs data parallelism (DP). In the future, we plan to explore additional parallelization strategies to enhance training efficiency. Additionally, in Skiparse Attention, each token only needs to attend to at most the same k2 − k12 tokens throughout, without requiring access to other tokens. This operation naturally suits a sequence parallelization strategy. However, the efficient implementation of this sequence parallelization in code remains a topic for further exploration.

- • Establishing reliable evaluation metrics. Although works like Vbench (Huang et al.,

2024) and Chronomagic Bench (Yuan et al., 2024) have proposed metrics to automate the evaluation of video model outputs, these metrics still cannot fully replace human review (Polyak et al., 2024). Human evaluation is labor-intensive and incurs significant costs, making it less feasible at scale. Therefore, developing more accurate and reliable automated metrics remains a key area for future research, and we will prioritize this in our work.

###### 6.3 Data

Despite ongoing improvements to our training data, the current dataset still faces several significant limitations in terms of data diversity, temporal modeling, video quality, and cross-modal information. We discuss these limitations and outline the corresponding directions for future works:

- • Lack of Data Diversity and Complexity. The current dataset predominantly covers specific domains such as simple actions, human faces, and a narrow range of scene types. We randomly sampled 2,000 videos from Panda70M and conducted manual verification, finding that less than 1% featured cars in motion, and there were even fewer than 10 videos of people walking. Approximately 80% of the videos consist of half-body conversations with multiple people in front of the camera. Therefore, we speculate that the narrow data domain of Panda70M restricts the model’s ability to generate many scenarios. Consequently, it lacks the ability to generate complex, dynamic scenes involving realistic human movement, object deformations, and intricate natural environments. This limitation hinders the model’s capacity to produce diverse and complex video content. Future work will focus on expanding the dataset to encompass a broader spectrum of dynamic and realistic environments, including more complex human interactions and dynamic physical effects. This expansion aims to improve the model’s generalization ability and facilitate the generation of high-quality, varied dynamic videos.
- • Lack of Camera Movement, Video Style, and Motion Speed Annotations. The current dataset lacks annotations for key dynamic aspects of video content, such as camera movement, video style, and motion speed. These annotations are essential for capturing the varied visual characteristics and movement dynamics within videos. Without them, the dataset may not fully support tasks that require detailed understanding of these elements, limiting the model’s ability to handle diverse video content. In future work, we will include these annotations to enhance the dataset’s versatility and improve the model’s ability to generate more contextually rich video content.
- • Limitations in Video Resolution and Quality. Although the dataset includes videos at common resolutions (e.g., 720P), these resolutions are insufficient for high-quality video generation tasks, such as generating detailed virtual characters or complex, high-fidelity scenes. The resolution and quality of the current dataset become limiting factors when generating fine-grained details or realistic dynamic environments. To address this limitation, future work should aim to incorporate high-resolution videos (e.g., 1080P, 2K), which will enable the generation of higher-quality videos with enhanced visual detail and realism.
- • Lack of Cross-Modal Information. The dataset predominantly focuses on video imagery and lacks complementary modalities such as audio or other forms of multi-modal data. This absence of cross-modal information limits the flexibility and applicability of generative models, particularly in tasks that involve speech, emotions, or contextual understanding. Future research should focus on integrating multi-modal data into the dataset. This will enhance the model’s ability to generate richer, more contextually nuanced content, thereby improving the overall performance and versatility of the generative system.

##### 7 Conclusion

We present Open-Sora Plan, our open-source high-quality and long-duration video generation project in this work. In the framework aspect, we decompose the entire video generation model into a Wavelet-Flow Variational Autoencoder, a Joint Image-Video Skiparse Denoiser, and various condition controllers. In the strategy aspect, we carefully design a min-max token strategy for efficient training, an adaptive gradient clipping strategy for preventing outflow gradients, and a prompt refiner for obtaining more appreciative results. Furthermore, we propose a multi-dimensional data curation pipeline for automatic high-quality data exploitation. While our Open-Sora Plan achieving a remarkable milestone, we will make more effort to promote the progress of the high-quality video generation research area and open-source community.

##### Contributors and Acknowledgements

###### Contributors

Bin Lin1, Yunyang Ge1, Xinhua Cheng1, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, Tanghui Jia, Junwu Zhang, Zhenyu Tang, Yatian Pang, Bin She, Cen Yan, Zhiheng Hu, Xiaoyi Dong, Lin Chen, Zhang Pan, Xing Zhou, Shaoling Dong, Yonghong Tian, Li Yuan

Project Lead Li Yuan Acknowledgements

We sincerely appreciate Zesen Cheng, Chengshu Zhao, Zongying Lin, Yihang Liu, Ziang Wu, Peng Jin, Hao Li for their valuable supports for our Open-Sora Plan project.

##### References

Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021a.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738, 2021b.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions.

Reiner Birkl, Diana Wofk, and Matthias Müller. Midas v3. 1–a model zoo for robust monocular relative depth estimation. arXiv preprint arXiv:2307.14460, 2023.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.

John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 1986.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023a.

1Core contributors with equal contributions

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023b.

Liuhan Chen, Zongjian Li, Bin Lin, Bin Zhu, Qian Wang, Shenghai Yuan, Xing Zhou, Xinghua Cheng, and Li Yuan. Od-vae: An omni-dimensional video compressor for improving latent video diffusion model. arXiv preprint arXiv:2409.01199, 2024b.

Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024c.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13320–13331, 2024d.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024e.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024f.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2025.

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7441–7451, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th International Conference on Pattern Recognition, 2010.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700–13710, 2024.

DiederikP. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv: Learning,arXiv: Learning, 2014.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback. In European Conference on Computer Vision (ECCV), 2024a.

Zongjian Li, Bin Lin, Yang Ye, Liuhan Chen, Xinhua Cheng, Shenghai Yuan, and Li Yuan. Wf-vae: Enhancing video vae by wavelet-driven energy flow for latent video diffusion model, 2024b.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024c.

Heng Liao, Jiajin Tu, Jing Xia, Hu Liu, Xiping Zhou, Honghui Yuan, and Yuxing Hu. Ascend: a scalable and unified architecture for ubiquitous deep neural network computing: Industry track paper. In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 789–801. IEEE, 2021.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Yatian Pang, Munan Ning, et al. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024a.

Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5404–5411, 2024b.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024b.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024c.

Ruyang Liu, Haoran Tang, Haibo Liu, Yixiao Ge, Ying Shan, Chen Li, and Jiankun Yang. Ppllava: Varied video

sequence understanding with prompt guidance. arXiv preprint arXiv:2411.02327, 2024d. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai. Fit: Flexible vision

transformer for diffusion model. arXiv preprint arXiv:2402.12376, 2024.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024.

OpenAI. Gpt-4 technical report, 2023. William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF

International Conference on Computer Vision, pages 4195–4205, 2023. Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022a.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022b.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Zhuo Su, Wenzhe Liu, Zitong Yu, Dewen Hu, Qing Liao, Qi Tian, Matti Pietikäinen, and Li Liu. Pixel difference networks for efficient edge detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5117–5127, 2021.

Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024.

Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054, 2023.

Thomas Unterthiner, Sjoerdvan Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. International Conference on Learning Representations,International Conference on Learning Representations, 2019.

A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. Jiawei Wang, Liping Yuan, and Yuchen Zhang. Tarsier: Recipes for training and evaluating large video

description models. arXiv preprint arXiv:2407.00634, 2024a.

Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International conference on machine learning, pages 23318–23340. PMLR, 2022a.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

ZiDong Wang, Zeyu Lu, Di Huang, Cai Zhou, Wanli Ouyang, et al. Fitv2: Scalable and improved flexible vision transformer for diffusion model. arXiv preprint arXiv:2410.13925, 2024c.

Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. arXiv preprint arXiv:2210.14896, 2022b.

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025.

Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv

- preprint arXiv:2405.18991, 2024a.

Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv

- preprint arXiv:2405.18991, 2024b.

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024c.

Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

L Xue. mt5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934, 2020.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024a.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024b.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024.

Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion – tokenizer is key to visual generation, 2024.

Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. arXiv preprint arXiv:2406.18522, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024.

Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024.

Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023.

##### Appendix

CogVideo-X(Chn=16)

Ground Truth

###### WF- VAE- L(Chn=16)

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

Allegro

OD-VAE

WF- VAE- L

WF- VAE- S

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

Ground Truth

WF- VAE- L(Chn=16) CogVideo-X(Chn=16)

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

WF- VAE- L

WF- VAE- S

Allegro

OD-VAE

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

Figure 15: Qualitative comparison of state-of-the-art VAEs. Top: High-detail static scene reconstruction. Bottom: Dynamic scene reconstruction under motion blur.

||[Figure 588]|[Figure 589]|[Figure 590]|[Figure 591]|[Figure 592]|
|---|---|---|---|---|
|[Figure 593]|[Figure 594]|[Figure 595]|[Figure 596]|[Figure 597]|
|[Figure 598]|[Figure 599]|[Figure 600]|[Figure 601]|[Figure 602]|
|[Figure 603]|[Figure 604]|[Figure 605]|[Figure 606]|[Figure 607]|
|[Figure 608]|[Figure 609]|[Figure 610]|[Figure 611]|[Figure 612]|
|[Figure 613]|[Figure 614]|[Figure 615]|[Figure 616]|[Figure 617]|
|[Figure 618]|[Figure 619]|[Figure 620]|[Figure 621]|[Figure 622]|
<br><br>(c)Allegro(e)Pyramid(d)Vchitect-2<br><br>(f)CogVideoX<br><br>5B<br><br>(g)CogVideoX<br><br><br>2B<br><br>(b)Mochi-1<br><br>A cinematic wide portrait of a man with his face lit by the glow of a TV.<br><br>(c)Allegro(a)Ours(e)Pyramid(d)Vchitect-2<br><br>(f)CogVideoX<br><br>5B<br><br>(g)CogVideoX<br><br><br>2B<br><br>(b)Mochi-1<br><br>A close-up shot of a young woman driving a car, looking thoughtful ...<br><br>(a)Ours<br><br>|[Figure 623]|[Figure 624]|[Figure 625]|[Figure 626]|[Figure 627]|
|---|---|---|---|---|
|[Figure 628]|[Figure 629]|[Figure 630]|[Figure 631]|[Figure 632]|
|[Figure 633]|[Figure 634]|[Figure 635]|[Figure 636]|[Figure 637]|
|[Figure 638]|[Figure 639]|[Figure 640]|[Figure 641]|[Figure 642]|
|[Figure 643]|[Figure 644]|[Figure 645]|[Figure 646]|[Figure 647]|
|[Figure 648]|[Figure 649]|[Figure 650]|[Figure 651]|[Figure 652]|
|[Figure 653]|[Figure 654]|[Figure 655]|[Figure 656]|[Figure 657]|
|
|---|

(c)Allegro(b)Mochi-1(a)Oursct-2

###### Figure 16: Comparison among several state-of-the-art methods in Text-to-Video Task.

|Hyperrealistic monster that closes its mouth.<br><br>|[Figure 658]|[Figure 659]|[Figure 660]|[Figure 661]|[Figure 662]|
|---|---|---|---|---|
<br><br>A mother dog ... her eyes filled with warmth and care as she watches her little one eat.<br><br>|[Figure 663]|[Figure 664]|[Figure 665]|[Figure 666]|[Figure 667]|
|---|---|---|---|---|
<br><br>|[Figure 668]|[Figure 669]|[Figure 670]|[Figure 671]|[Figure 672]|
|---|---|---|---|---|
<br><br>A curious cat peering out from a cozy hiding spot.<br><br>|[Figure 673]|[Figure 674]|[Figure 675]|[Figure 676]|[Figure 677]|
|---|---|---|---|---|
<br><br>A rabbit in a magician's outfit, pulling a human-sized carrot out of a top hat.<br><br>|[Figure 678]<br><br>[Figure 679]|[Figure 680]<br><br>[Figure 681]|[Figure 682]<br><br>[Figure 683]|[Figure 684]<br><br>[Figure 685]|[Figure 686]<br><br>[Figure 687]|
|---|---|---|---|---|
<br><br>A tilt-down from a starry night sky, revealing a quiet forest clearing bathed in moonlight.<br><br>|[Figure 688]<br><br>[Figure 689]|[Figure 690]<br><br>[Figure 691]|[Figure 692]<br><br>[Figure 693]|[Figure 694]<br><br>[Figure 695]|[Figure 696]<br><br>[Figure 697]|
|---|---|---|---|---|
<br><br>A serene mountain lake reflects the starry night sky as a small boat glides silently across the water ...<br><br>|[Figure 698]|[Figure 699]|[Figure 700]|[Figure 701]|[Figure 702]|
|---|---|---|---|---|
<br><br>A sports car accelerating rapidly on an open highway, the engine roaring ...<br><br>|[Figure 703]<br><br>[Figure 704]|[Figure 705]<br><br>[Figure 706]|[Figure 707]<br><br>[Figure 708]|[Figure 709]<br><br>[Figure 710]|[Figure 711]<br><br>[Figure 712]|
|---|---|---|---|---|
<br><br>an extreme close up shot of a woman's eye, with her iris appearing as earth<br><br>|[Figure 713]<br><br>[Figure 714]|[Figure 715]<br><br>[Figure 716]|[Figure 717]<br><br>[Figure 718]|[Figure 719]<br><br>[Figure 720]|[Figure 721]<br><br>[Figure 722]|
|---|---|---|---|---|
<br><br>An oil painting of a natural forest environment with colorful maple trees and cinematic parallax animation.<br><br>|[Figure 723]|[Figure 724]|[Figure 725]|[Figure 726]|[Figure 727]|
|---|---|---|---|---|
<br><br>A pull-out from the surface of a bubbling pot, revealing the busy kitchen around it.<br><br>Figure 17: Text-to-Video Showcases.<br><br>3|
|---|

||[Figure 728]|
|---|
<br><br>(a)Ours(c)Pyramid(b)CogVideoX<br><br>|[Figure 729]|
|---|
<br><br>The video depicts a character with features resembling those of a monkey, dressed in intricate armor adorned with ornate designs and animal motifs. The character's facial expression is serious, and the setting appears to be a misty, mountainous landscape, reminiscent of a traditional Chinese scene with a pagoda in the background. The character's attire and appearance could be inspired by Sun Wukong, the Monkey King from Chinese mythology, given the combination of monkey-like features and warrior attire.<br><br>|[Figure 730]|[Figure 731]|[Figure 732]|[Figure 733]|[Figure 734]|
|---|---|---|---|---|
|[Figure 735]|[Figure 736]|[Figure 737]|[Figure 738]|[Figure 739]|
|[Figure 740]|[Figure 741]|[Figure 742]|[Figure 743]|[Figure 744]|
|[Figure 745]|[Figure 746]|[Figure 747]|[Figure 748]|[Figure 749]|
<br><br>(a)Ours(c)Pyramid(b)CogVideoX<br><br>The video begins with a close-up of a talking pig character, seated comfortably in a wooden chair, with a relaxed smile. His fur is detailed and realistic, his small tusks curling out from his snout. As he speaks casually, the camera slowly zooms out to reveal more of the stunning backdrop: floating islands hovering above a sea of clouds. In the distance, trees cover the islands, while soft sunlight bathes the entire scene in a warm glow. The pig's gentle expressions and friendly demeanor create a calm and welcoming atmosphere, as the background subtly shifts with the moving clouds.<br><br>|[Figure 750]|[Figure 751]|[Figure 752]|[Figure 753]|[Figure 754]|
|---|---|---|---|---|
|[Figure 755]|[Figure 756]|[Figure 757]|[Figure 758]|[Figure 759]|
|[Figure 760]|[Figure 761]|[Figure 762]|[Figure 763]|[Figure 764]|
|[Figure 765]|[Figure 766]|[Figure 767]|[Figure 768]|[Figure 769]|
<br><br>(c)EasyAnimateV4(c)EasyAnimateV4<br><br>Figure 18: Comparison among several state-of-the-art methods in Image-to-Video Task.<br><br>4|
|---|

###### urs

||[Figure 770]|[Figure 771]|[Figure 772]|[Figure 773]|[Figure 774]|
|---|---|---|---|---|
<br><br>|[Figure 775]|[Figure 776]|[Figure 777]|[Figure 778]|[Figure 779]|
|---|---|---|---|---|
<br><br>|[Figure 780]|[Figure 781]|[Figure 782]|[Figure 783]|[Figure 784]|
|---|---|---|---|---|
<br><br>|[Figure 785]|[Figure 786]|[Figure 787]|[Figure 788]|[Figure 789]|
|---|---|---|---|---|
<br><br>|[Figure 790]|[Figure 791]|[Figure 792]|[Figure 793]|[Figure 794]|
|---|---|---|---|---|
<br><br>a boat sits on the shore of a lake with mt fuji in the background<br><br>a bridge that is in the middle of a river<br><br>a bridge that is over a body of water<br><br>a modern living room with wood floors and a tv<br><br>a red panda eating bamboo in a zoo<br><br>a woman smiles while holding a yellow flower<br><br>a basket of french fries in a fryer<br><br>[Figure 795]<br><br>[Figure 796]<br><br>[Figure 797]<br><br>[Figure 798]<br><br>[Figure 799]<br><br>|[Figure 800]|[Figure 801]|[Figure 802]|[Figure 803]|[Figure 804]|
|---|---|---|---|---|
<br><br>|[Figure 805]|[Figure 806]|[Figure 807]|[Figure 808]|[Figure 809]|
|---|---|---|---|---|
<br><br>an aerial view of a futuristic building on a cliff overlooking a body of water<br><br>a panda bear is eating a piece of bamboo<br><br>|[Figure 810]|[Figure 811]|[Figure 812]|[Figure 813]|[Figure 814]|
|---|---|---|---|---|
<br><br>a penguin walking on a beach near the water<br><br>|[Figure 815]|[Figure 816]|[Figure 817]|[Figure 818]|[Figure 819]|
|---|---|---|---|---|
<br><br>Figure 19: Image-to-Video Showcases.<br><br>5|
|---|

