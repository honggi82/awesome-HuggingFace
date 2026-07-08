# arXiv:2404.09967v2[cs.CV]24May2024

## CTRL-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model

###### Han Lin∗ Jaemin Cho∗ Abhay Zala Mohit Bansal

UNC Chapel Hill {hanlincs, jmincho, aszala, mbansal}@cs.unc.edu https://ctrl-adapter.github.io

Video Control

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

#### CTRL-Adapter

Depth

Multi-Condition Video Control

[Figure 7]

[Figure 8]

[Figure 9]

Input Condition

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Depth

Small Image ControlNet

###### +

[Figure 14]

[Figure 15]

[Figure 16]

Human Pose

[Figure 17]

CTRL-Adapter

[Figure 18]

Sparse Frame Video Control

[Figure 19]

[Figure 20]

[Figure 21]

|No Condition|
|---|

Large Image or Video Diffusion Model

[Figure 22]

[Figure 23]

Condition-Controlled Image/Video

Scribble

Image Control

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

: Frozen : Trained

[Figure 31]

Depth Canny Edge

Normal Map

Figure 1: We propose CTRL-Adapter, an efficient and versatile framework for adding diverse controls to any diffusion model. CTRL-Adapter supports a variety of useful applications.

##### Abstract

ControlNets are widely used for adding spatial control to text-to-image diffusion models with different conditions, such as depth maps, scribbles/sketches, and human poses. However, when it comes to controllable video generation, ControlNets cannot be directly integrated into new backbones due to feature space mismatches, and training ControlNets for new backbones can be a significant burden for many users. Furthermore, applying ControlNets independently to different frames cannot effectively maintain object temporal consistency. To address these challenges, we introduce CTRL-Adapter, an efficient and versatile framework that adds diverse controls to any image/video diffusion model through the adaptation of pretrained ControlNets. CTRL-Adapter offers strong and diverse capabilities, including image and video control, sparse-frame video control, fine-grained patchlevel multi-condition control (via an MoE router), zero-shot adaptation to unseen conditions, and supports a variety of downstream tasks beyond spatial control, including video editing, video style transfer, and text-guided motion control. With six diverse U-Net/DiT-based image/video diffusion models (SDXL, PixArt-α, I2VGen-XL, SVD, Latte, Hotshot-XL), CTRL-Adapter matches the performance of pretrained ControlNets on COCO and achieves the state-of-the-art on DAVIS 2017 with significantly lower computation (< 10 GPU hours).

∗equal contribution

##### 1 Introduction

Recent diffusion models have achieved significant progress in generating high-fidelity images [61, 52, 65, 55] and videos [3, 18, 6, 41, 43] from text descriptions. As it is often hard to describe every image/video detail only with text, there have been many works to control diffusion models in a more fine-grained manner by providing additional condition inputs such as bounding boxes [39, 83], reference object images [63, 17, 36], and segmentation maps [16, 2, 86]. Among them, Zhang et al. [86] have released a variety of ControlNet checkpoints based on Stable Diffusion [61] v1.5 (SDv1.5), and the user community has shared many ControlNets trained with different input conditions. Until now, ControlNet has become one of the most popular methods for controllable image generation.

However, there are challenges when using the existing pretrained image ControlNets for controllable video generation. First, pretrained ControlNeta cannot be directly plugged into new backbone models, and the cost for training ControlNets for new backbone models is a big burden for many users due to high computational costs. For example, training a ControlNet for SDv1.5 takes 500-600 A100 GPU hours [87, 88]. Second, ControlNet was originally designed for controllable image generation; hence, applying pretrained image ControlNets directly to each video frame independently does not take the temporal consistency across frames into account.

To address this challenge, we design CTRL-Adapter, a novel, flexible framework that enables the efficient reuse of pretrained ControlNets for diverse controls with any new image/video diffusion models, by adapting pretrained ControlNets (and improving temporal alignment for videos). We illustrate the overall capabilities of CTRL-Adapter framework in Fig. 1. As shown in Fig. 3 left, CTRL-Adapter trains adapter layers [30, 84] to map the features of a pretrained image ControlNet to a target image/video diffusion model, while keeping the parameters of the ControlNet and the backbone diffusion model frozen. As shown in Fig. 3 right, each CTRL-Adapter consists of four modules: spatial convolution, temporal convolution, spatial attention, and temporal attention. The temporal convolution/attention modules effectively fuse the ControlNet features into image/video diffusion models for better temporal consistency. Additionally, to ensure robust adaptation of ControlNets to backbone models of different noise scales and sparse frame control conditions, we propose skipping the visual latent variable from the ControlNet inputs. We also introduce inverse timestep sampling to effectively adapt ControlNets to new backbones equipped with continuous diffusion timestep samplers. For more accurate control beyond a single condition, we designed a novel and powerful Mixture-of-Experts (MoE) router, which allows fine-grained, patch-level composition of spatial feature maps from multiple control conditions via CTRL-Adapters (see Sec. 3.3).

As shown in Table 1, CTRL-Adapter allows many useful capabilities, including image control, video control, video control with sparse frames, multi-condition control, and compatibility with different backbone models, while previous methods only support a small subset of them (see details in Sec. 2). We demonstrate the effectiveness of CTRL-Adapter through extensive experiments and analyses. It exhibits strong performance when adapting ControlNets (pretrained with SDv1.5) to various video and image diffusion backbones, including image-to-video generation – I2VGen-XL [89] and Stable Video Diffusion (SVD) [3], text-to-video generation – Latte [46] and Hotshot-XL [49], and text-toimage generation – SDXL [52] and PixArt-α [8]. The ability of CTRL-Adapter to seamlessly adapt to DiT-based models such as Latte and PixArt-α, which are structurally different from U-Net based ControlNets, demonstrates the flexibility of our framework design.

In Sec. 5.1 and Sec. 5.2, we first show that CTRL-Adapter matches the performance of a pretrained image ControlNet on COCO dataset [42] and outperforms previous methods in controllable video generation (achieving state-of-the-art performance on the DAVIS 2017 dataset [53]) with significantly lower training costs (less than 10 GPU hours, see Fig. 2). Next, we demonstrate that CTRL-Adapter enables more accurate video generation with multiple conditions compared to a single condition. Our fine-grained patch-level MoE router consistently outperforms both the equal weights baseline and the global weights MoE router (Sec. 5.3). In addition, we show that skipping the visual latent variable from ControlNet inputs allows video control only with a few frames of (i.e., sparse) conditions, eliminating the need for dense conditions across all frames (Sec. 5.4). We also highlight zero-shot adaption – CTRL-Adapter trained with one condition can easily adapt to another ControlNet trained with a different condition (Sec. 5.5). Moreover, our CTRL-Adapter can be flexibly applied to a variety of downstream tasks beyond spatial control, including video editing, video style transfer, and text-guided object motion control (Sec. 6). Lastly, we provide comprehensive ablations for CTRL-Adapter design choices and qualitative examples (Appendix E, Appendix F, and Appendix G).

Table 1: Overview of the capabilities supported by controllable image/video generation methods.

Image Video Video Control Multi-Condition Compatible w/

Method

Control Control w/ Sparse Frames Control Different Backbones Image Control Methods

ControlNet [86] ✔ ✘ ✘ ✘ ✘ Multi-ControlNet [86] ✔ ✘ ✘ ✔ ✘ T2I-Adapter [48] ✔ ✘ ✘ ✔ ✘ Uni-ControlNet [92] ✔ ✘ ✘ ✔ ✘ X-Adapter [56] ✔ ✘ ✘ ✘ ✔

Video Control Methods

ControlVideo [90] ✘ ✔ ✘ ✘ ✘ VideoComposer [77] ✘ ✔ ✘ ✔ ✘ SparseCtrl [21] ✘ ✔ ✔ ✘ ✘

CTRL-Adapter (Ours) ✔ ✔ ✔ ✔ ✔

Video Control w/ Depth Map

Image Control w/ Depth Map

OpticalFlowError()

I2VGen-XL + Ctrl-Adapter (Ours)

4.5

0.84

Video Composer Text2Video Zero ControlVideo

SSIM()

4.0

0.82

Control-A-Video

SDXL + Ctrl-Adapter (Ours)

3.5

0.80

SDXL ControlNet

SDXL + X-Adapter

0 5 10 15 700

0 10 20 30 40 50

Training GPU hours

Training GPU hours

Figure 2: Training speed of CTRL-Adapter for video (left) and image (right) control with depth maps, measured on A100 80GB GPUs. For both video and image controls, CTRL-Adapter trained for 10 GPU hours outperforms strong baselines, including SDXL, which is trained for 700 GPU hours.

##### 2 Related Works: Adding Control to Diffusion Models

There have been many works using different types of additional inputs to control the image/video diffusion models, such as bounding boxes [39, 83], reference object image [63, 17, 36], segmentation map [16, 2, 86], sketch [86], etc., and combinations of multiple conditions [33, 54, 92, 77]. As finetuning all the parameters of such image/video diffusion models is computationally expensive, several methods, such as ControlNet [86], have been proposed to add conditional control capability via parameter-efficient training [86, 64, 48].

X-Adapter [56] learns an adapter module to reuse ControlNets pretrained with a smaller image diffusion model (e.g., SDv1.5) for a bigger image diffusion model (e.g., SDXL). While they focus solely on learning an adapter for image control, CTRL-Adapter features architectural designs (e.g., temporal convolution/attention layers) for video generation as well. In addition, X-Adapter needs to be used with the source image diffusion model (SDv1.5) during both training and inference, whereas CTRL-Adapter does not require the smaller diffusion model for image or video generation, making it more memory and computationally efficient (see Appendix B.3 for details).

SparseCtrl [21] guides a video diffusion model with conditional inputs of few frames (instead of full frames), to alleviate the cost of collecting video conditions. Since SparseCtrl involves augmenting ControlNet with an additional channel for frame masks, it requires training a new variant of ControlNet from scratch. In contrast, we leverage existing image ControlNets more efficiently by propagating information through temporal layers in adapters and enabling sparse frame control via skipping the latents from ControlNet inputs (see Sec. 3.2 for details).

Furthermore, compared with previous works that are specially designed for specific condition controls on a single modality (image [86, 54] or video [31, 90]), our work presents a unified and versatile framework that supports diverse controls, including image control, video control, sparse frame control, with significantly lower computational costs by reusing pretrained ControlNets (outperforms strong baselines in less than 10 GPU hours, see Fig. 2). To the best of our knowledge, we are also the first work that extends multi-condition video control into fine-grained patch-level composition. Table 1 compares CTRL-Adapter with other relevant methods. See Appendix A.1 for extended related works.

[Figure 32]

###### Pretrained Image ControlNet

[Figure 33]

###### Pretrained Image/Video Diffusion Model

Input latents F ✕ C ✕ H’ ✕ W’

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

✕N

[Figure 39]

[Figure 40]

[Figure 41]

Timestep Encoder

Text Encoder

Image Encoder

✕N

|1| |
|---|---|
|2| |
|3| |
|4| |
| | |

zero convolution

Up/Down Sample

[Figure 42]

[Figure 43]

F ✕ C ✕ H’ ✕ W’

Conv Spatial

2D Conv

Conv

Add Time & Norm

[Figure 44]

- Input Block A

- H1 ✕ W1

Input Block B

- H2 ✕ W2

Input Block C

- H3 ✕ W3

Input Block D

- H4 ✕ W4

Middle Block

- H5 ✕ W5

- Input Block A

- H1 ✕ W1

Input Block B

- H2 ✕ W2

Input Block C

- H3 ✕ W3

Input Block D

- H4 ✕ W4

Middle Block

- H5 ✕ W5

[Figure 45]

✕3

✕3

[Figure 46]

2D Conv

✕N

[Figure 47]

✕3

[Figure 48]

✕3

3D Conv

Attn Temporal

[Figure 49]

Add Time & Norm

[Figure 50]

[Figure 51]

[Figure 52]

✕3

[Figure 53]

✕3

[Figure 54]

3D Conv

[Figure 55]

[Figure 56]

Choose (T2V or I2V)

###### CTRL-Adapter

✕N

[Figure 57]

✕3

[Figure 58]

✕3

[Figure 59]

[Figure 60]

FFN

[Figure 61]

Attn Spatial

[Figure 62]

[Figure 63]

Middle Adapter

Cross Attention

[Figure 64]

[Figure 65]

[Figure 66]

Spatial Self-Attn

zero convolution

[Figure 67]

✕ND

Output Block D H4 ✕ W4

Output Adapter D

zero convolution

✕3

[Figure 68]

✕3

✕N

FFN

[Figure 69]

Temporal

[Figure 70]

✕NC

Output Block C H3 ✕ W3

Output Adapter C

zero convolution

✕3

[Figure 71]

Cross Attention

✕3

Temporal Self-Attn

[Figure 72]

✕NB

Output Block B H2 ✕ W2

Output Adapter B

zero convolution

✕3

[Figure 73]

✕3

Output latents F ✕ C ✕ H’ ✕ W’

[Figure 74]

✕NA

Output Block A H1 ✕ W1

Output Adapter A

[Figure 75]

zero convolution

✕3

✕3

Forward Backward Timestep/Text/ Trainable Frozen Image Conditions

[Figure 76]

F ✕ C ✕ H’ ✕ W’

[Figure 77]

[Figure 78]

Figure 3: Left: CTRL-Adapter (colored orange) enables to reuse pretrained image ControlNets (colored blue) for new image/video diffusion models (colored green). Right: Architecture details of CTRL-Adapter. Temporal convolution and attention layers are skipped for image diffusion backbones.

##### 3 Method

###### 3.1 Preliminaries: Latent Diffusion Models and ControlNets

Latent Diffusion Models. Many recent video generation works utilize latent diffusion models (LDMs) [61] to learn the compact representations of videos. First, given a F-frame RGB video x ∈ RF×3×H×W, a video encoder (of a pretrained autoencoder) provides C-dimensional latent representation (i.e., latents): z = E(x) ∈ RF×C×H

′×W′, where height and width are spatially downsampled (H′ < H and W′ < W). Next, in the forward process, a noise scheduler (e.g., DDPM [28]) adds noise to the latents z. Then, in the backward pass, a diffusion model Fθ(zt,t,ctext/img) learns to gradually denoise the latents, given a diffusion timestep t, and a text prompt ctext (i.e., T2V) and/or an initial frame cimg (i.e., I2V) if provided. The diffusion model is trained with objective: LLDM = Ez,ϵ∼N(0,I),t∥ϵ − ϵθ(zt,t,ctext/img)∥22, where ϵ and ϵθ represent the added noise to latents and the predicted noise by Fθ respectively. We apply the same objective for CTRL-Adapter training.

ControlNets. ControlNet [86] is designed to add spatial controls (e.g., depth, sketch, segmentation maps, etc.) to image diffusion models. Specifically, given a pretrained backbone image diffusion model Fθ that consists of input/middle/output blocks, ControlNet has a similar architecture Fθ′, where the input/middle blocks parameters of θ′ are initialized from θ, and the output blocks consist of 1 × 1 convolution layers initialized with zeros. ControlNet takes the diffusion timestep t, text prompt ctext, control image cf (e.g., depth map), and the noisy latents zt as inputs, and the output features are merged into the backbone model Fθ for final image generation.

###### 3.2 CTRL-Adapter

We introduce CTRL-Adapter, a novel framework that enables the efficient reuse of existing image ControlNets (SDv1.5) for spatial control with new diffusion models. We mainly describe our method details in the video generation settings, since CTRL-Adapter can be flexibly adapted to image diffusion models by regarding images as single-frame videos.

Efficient adaptation of pretrained ControlNets. As shown in Fig. 3 (left), we train an adapter module (colored orange) to map the middle/output blocks of a pretrained ControlNet (colored blue) to the corresponding middle/output blocks of the target video diffusion model (colored green). If the target backbone does not have the same number of output blocks CTRL-Adapter maps the ControlNet features to the output block that handles the closest height and width of the latents. We keep all parameters in both the ControlNet and the target video diffusion model frozen. Therefore, training a CTRL-Adapter can be significantly more efficient than training a new video ControlNet.

CTRL-Adapter architecture. As shown in Fig. 3 (right), each block of CTRL-Adapter consists of four modules: spatial convolution, temporal convolution, spatial attention, and temporal attention. We set the values for N1,...,N4 and N as 1 by default. The temporal convolution and attention modules effectively fuse the ControlNet features to the video backbone models for better temporal consistency. Moreover, the spatial/temporal convolution modules incorporate the current denoising timestep t and spatial/temporal attention modules incorporate the conditions (i.e., text prompt/initial frame) ctext/img. This design allows CTRL-Adapter to dynamically adjust its features according to different denoising stages and the objects generated. In addition, we skip the temporal convolution/attention modules when adapting to image diffusion models. See Appendix B.1 for architecture details of the four modules, and Appendix E for detailed ablation studies on the design choices of CTRL-Adapter.

Adaptation to DiT-based image/video backbones. Our CTRL-Adapter can also adapt U-Net based ControlNets to DiT-based image/video generation backbones. One important observation we made is that the spatial features encoded in the U-Net of ControlNets and the DiT blocks are structurally different (see Fig. 22). Specifically, the representation from U-Net blocks exhibits coarse-to-fine, hierarchical patterns (e.g., earlier blocks output smaller size feature maps and control high-level information such as object presence, while later blocks output larger feature maps and control lower-level details like textures), while all DiT blocks handle the feature maps of same sizes. This indicates that mapping all middle/output blocks of ControlNet to DiT blocks might not be the optimal solution. Therefore, we choose to map the feature maps of the largest size in ControlNet (i.e., block A) to the DiT blocks via CTRL-Adapters, which are followed by zero-convolutions for channel dimension matching. To improve computational efficiency for DiT-based video generation models (i.e., Latte [46]), we only insert CTRL-Adapters into every other DiT block (i.e., blocks 2, 4, 6..., 28, see (a) in Fig. 17). See Appendix E.2 for more discussion on CTRL-Adapter designs for DiT.

Skipping the latent from ControlNet inputs: robust adaption to different noise scales & sparse frame conditions. Although the original ControlNets take the latent zt as part of their inputs, we find that skipping zt from ControlNet inputs is effective for CTRL-Adapter in certain settings, as illustrated in Fig. 4. (1) Different noise scales: while SDv1.5 samples noise ϵ from N(0,I), some recent diffusion models [29, 12, 3] sample noise ϵ of much bigger scale (e.g. SVD [3] sample noise from σ ∗ N(0,I), where σ ∼ LogNormal(0.7,1.6); σ ∈ [0,+∞] and E[σ] = 7.24). We find that adding larger-scale

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

CTRL-Adapter

CTRL-Adapter

[Figure 92]

[Figure 93]

Video UNet ControlNets

Video UNet ControlNets

###### Latents given to ControlNet Latents not given to ControlNet

Figure 4: Left (default): latent zt is given to ControlNet. Right: latent zt not given to ControlNet.

zt from the new backbone models to image conditions cf dilutes the cf and makes the ControlNet outputs less informative, whereas skipping zt enables the adaptation of such new backbone models. (2) Sparse frame conditions: when the image conditions are provided only for the subset of video frames (i.e., cf = ∅ for most frames f), ControlNet could rely on the information from zt and ignore cf during training. Skipping zt from ControlNet inputs also helps the CTRL-Adapter to more effectively handle such sparse frame conditions (see Table 7).

Inverse timestep sampling: robust adaptation to continuous diffusion timestep samplers. While SDv1.5 samples discrete timesteps t uniformly from {0,1,...1000}, some recent diffusion models [12, 45, 60] sample timesteps from continuous distributions, e.g., SVD [3] samples timesteps from a LogNormal distribution. This gap between discrete and continuous distributions means that we cannot assign the same timestep t to both the video diffusion model and the ControlNet. Therefore, we propose inverse timestep sampling, an algorithm that creates a timestep mapping between the continuous and discrete time distributions (see Algorithm 1 for PyTorch [1] code). The highlevel idea of this algorithm is inspired by inverse transform sampling [13]. Given the cumulative

distribution functions (CDFs) of the continuous timestep distribution Fcont. and the ControlNet timestep distribution FCNet, we first uniformly sample a value u between [0,1], and then returns the smallest timesteps tcont. ∈ [0,∞] ⊆ R,tCNet ∈ {0,1,...,1000} ⊆ N, such that Fcont.(tcont.) ≥ u,FCNet(tCNet) ≥ u. This procedure naturally creates a mapping between two distributions. See Appendix B.2 for details.

ControlNet Patch Feature(s) Learnable Query Feature

ControlNets

###### w1i,j, w2i,j, … ,wNi,j SM

###### w1, w2, … ,wN

[Figure 94]

Video UNet

Linear & SM

[Figure 95]

[Figure 96]

- w1

- w2

Learnable Emb.

Add & Norm

[Figure 97]

- (a) Unconditional Global Weights

- (b) Patch-Level MLP Weights (c) Patch-Level Q-Former Weights

CTRL-Adapter

FFN

[Figure 98]

w1i,j, w2i,j, … ,wNi,j

[Figure 99]

[Figure 100]

Add & Norm

Cross Attn V K Q

1 ✕ 1 1 ✕ 1

[Figure 101]

SM

[Figure 102]

wN

MLP 1

MLP N

…

1 ✕ C N ✕ C 1 ✕ C

1 ✕ C

Figure 5: Left: Framework for multi-condition video generation by combining multiple ControlNets. w1,w2,...,wN are the weights allocated to each ControlNet. Right: Three MoE router variants. (a) operates globally, while (b) and (c) operate on the fine-grained patch-level. C and N represent feature dimensions and number of ControlNet experts respectively. wki,j represents the router weights at position (i,j) of the kth ControlNet 2D feature map. SM stands for Softmax.

###### 3.3 Multi-Condition Generation via CTRL-Adapter Composition

Multi-ControlNet [86] is proposed for spatial control beyond a single condition. However, this method naively combines different conditions with equal weights during inference time without training. For more effective control composition, we first experiment with some simple extensions, such as replacing these fixed weights with unconditional global learnable weights via a lightweight MoE [68] router (see variant (a) in Fig. 5 right). Next, we further propose refining the global weights MoE router into a more fine-grained, patch-level MoE router. Variant (b) processes the patch-level features of each ControlNet into a scalar value independently, then uses softmax to assign ControlNet weights for each patch. Variant (c) takes in all N ControlNet features associated with a patch, using an architecture design inspired by Q-Former [37] to output expert weights. Comparisons of different variants are discussed in Sec. 5.3 and Appendix E.4.

##### 4 Experimental Setup

ControlNets and Target Diffusion Models. We use ControlNets trained with SD v1.5. For target diffusion models, we experiment with two I2V models – I2VGen-XL [89] and Stable Video Diffusion (SVD) [3], two T2V models – Latte [46] and Hotshot-XL [49], and two T2I models – SDXL [52] and PixArt-α [8]. Note that Latte and PixArt-α are generation models based on DiT instead of U-Net.

Training and Evaluation Datasets. We use 200K videos sampled from Panda-70M training set [9] and 300K images from the LAION POP [67] dataset for video and image CTRL-Adapters training respectively. During training, we extract various control conditions (e.g., depth map) on-the-fly to simplify the data-preparation process. Following previous works [31, 90], we evaluate video CTRLAdapters on DAVIS 2017 [53], and image CTRL-Adapters on COCO val2017 split [42]. Detailed training and inference setups for the experiments are provided in Appendix C and Appendix D.

Evaluation Metrics. We perform evaluation on two folds: visual quality and spatial control. Following previous works [54, 31], we use Frechet Inception Distance (FID) [26] to measure the visual quality of generated images/videos. For video datasets, following VideoControlNet [31], we report the L2 distance between the optical flow error [58] between the input and generated videos. For image datasets, following Uni-ControlNet [92], we report the Structural Similarity (SSIM) [78] and mean squared error (MSE) between generated images and ground truth images.

- Table 2: Evaluation of video generation with single control condition on DAVIS 2017 dataset. The best number in each column is bolded, and the second best is underscored.

Method

Depth Map Canny Edge FID (↓) Optical Flow Error (↓) FID (↓) Optical Flow Error (↓)

Text2Video-Zero [32] 19.46 4.09 17.80 3.77 ControlVideo [90] 27.84 4.03 25.58 3.73 Control-A-Video [10] 22.16 3.61 22.82 3.44 VideoComposer [77] 22.09 4.55 - -

Hotshot-XL backbone

SDXL ControlNet [73] 45.35 4.21 25.40 4.43 SDv1.5 ControlNet + CTRL-Adapter (Ours) 14.63 3.94 20.83 4.15

Latte backbone (DiT-Based) SDv1.5 ControlNet + CTRL-Adapter (Ours) 16.92 3.98 17.87 2.73

I2VGen-XL backbone SDv1.5 ControlNet + CTRL-Adapter (Ours) 7.43 3.20 6.42 3.37

SVD backbone SVD Temporal ControlNet [62] 4.91 4.84 - SDv1.5 ControlNet + CTRL-Adapter (Ours) 3.82 2.96 3.96 2.39

- Table 3: Evaluation of image generation with single control condition on COCO val2017 split. The best number in each column is bolded, and the second best is underscored.

Depth Map Canny Edge Soft Edge / HED FID (↓) MSE (↓) SSIM (↑) FID (↓) SSIM (↑) FID (↓) SSIM (↑)

Method

- SDv1.4 or v1.5 backbone
- SDv1.5 ControlNet [86] 21.25 87.57 - 18.90 0.4828 26.59 0.4719 T2I-Adapter [48] 21.35 89.82 - 18.98 0.4422 - GLIGEN [39] 21.46 88.22 - 24.74 0.4226 28.57 0.4015 Uni-ControlNet [92] 21.20 91.05 - 17.79 0.4911 17.86 0.5197

SDXL backbone

SDXL ControlNet [73] 17.91 86.95 0.8363 17.21 0.4458 - SDv1.5 ControlNet + X-Adapter [56] 20.71 90.08 0.7885 19.71 0.3002 - SDv1.5 ControlNet + CTRL-Adapter (Ours) 19.26 87.54 0.8534 21.04 0.5806 18.08 0.6454

PixArt-α backbone (DiT-Based)

PixArt-δ ControlNet [7] - - - - - 20.41 0.6938 SDv1.5 ControlNet + CTRL-Adapter (Ours) 22.54 84.78 0.8496 18.75 0.6359 17.52 0.6812

[Figure 103]

[Figure 104]

Input Image/ Prompt

Prompt: Majestic white dragon …

Prompt: A car ﬂies over a hill

##### 5 Results and Analysis

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Input Control Conditions

- 5.1 Video Generation with Single Condition

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Control Video

We compare SDv1.5 ControlNet + CTRL-Adapter built on Hotshot-XL, I2VGen-XL, SVD, and Latte with video control methods including Text2Video-Zero [32], Control-A-Video [10], ControlVideo [90], and VideoComposer [77]. As the spatial layers of Hotshot-XL are initialized with SDXL and remain frozen, the SDXL ControlNets are directly compatible with Hotshot-XL, so we include Hotshot-XL + SDXL ControlNet as a baseline. We also experiment with a temporal ControlNet [62] trained with SVD.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Control-AVideo

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Text2VideoZero

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Hotshot-XL + CTRL-Adapter

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Table 2 shows that in both depth map and canny edge input conditions, CTRL-Adapters on I2VGen-XL and SVD outperforms all previous strong video control methods in visual quality (FID) and spatial control (optical flow error) metrics. Note that it takes < 10 GPU hours for CTRL-Adapter to outperform the baselines (see Fig. 2). In Fig. 6 and Appendix H.1, we visualize the comparison between CTRL-Adapter and other video control baselines. We study visual quality-spatial control trade-off in Appendix F.2.

Latte + CTRL-Adapter

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

SVD + CTRL-Adapter

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

I2VGen-XL + CTRL-Adapter

Figure 6: Single-cond. video generation.

- Table 4: Comparison of different weighting methods (see Fig. 5 right part for details) for multicondition video generation. The control sources are abbreviated as D (depth map), C (canny edge), N (surface normal), S (softedge), Seg (semantic segmentation map), L (line art), and P (human pose).

D+C D+P D+C+N+S D+C+N+S+Seg+L+P

FID (↓) Flow Error (↓) FID (↓) Flow Error (↓) FID (↓) Flow Error (↓) FID (↓) Flow Error (↓) Baseline: Equal Weights 8.50 2.84 11.32 3.48 8.75 2.40 9.48 2.93

- (a) Unconditional Global Weights 9.14 2.89 10.98 3.32 8.39 2.36 8.18 2.48

- (b) Patch-Level MLP Weights 8.40 2.34 9.37 3.17 7.87 2.11 8.26 2.00

- (c) Patch-Level Q-Former Weights 7.54 2.39 9.22 3.22 7.72 2.31 8.00 2.08

###### 5.2 Image Generation with Single Condition

Input Control Conditions

SDXL + X-Adapter

SDXL + Ctrl-Adapter

PixArt-α + Ctrl-Adapter

We compare SDv1.5 ControlNet + CTRLAdapter with controllable image generation methods that use SDv1.4, SDv1.5, SDXL, and PixArt-α as backbones, including pretrained SDv1.5/SDXL ControlNets [86, 73], T2I-Adapter [48], GLIGEN [39], UniControlNet [92], X-Adapter [56], and PixArt-δ ControlNet [7].

SDXL ControlNet

Input Prompt

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

A metal statue of two women sitting on a bench

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Kitten looking puzzled sitting in a bathroom sink

As shown in Table 3, CTRL-Adapter outperforms baselines with SDv1.4/v1.5 backbones in almost all metrics. When compared to the baselines with SDXL backbones, CTRL-Adapter outperforms X-Adapter in most metrics, and matches (in FID/MSE with depth map inputs) or outperforms SDXL ControlNet (in SSIM with depth map and canny edge inputs). Note that SDXL ControlNet was trained for much longer than CTRL-Adapter (700 vs. 44 A100 GPU hours) and it takes less than 10 GPU hours for CTRL-Adapter to outperform the SDXL depth ControlNet in SSIM (see Fig. 2). In addition, when applied to DiT-based backbone (i.e., PixArt-α), CTRL-Adapter achieves good improvement in FID (17.52 ours vs. 20.41 PixArt-δ ControlNet on soft edge) and competitive SSIM score. In Fig. 6 and Fig. 7, we visualize the comparison between CTRL-Adapter and other image control baselines. See Appendix H.3 for more visualizations.

Figure 7: Generated images on COCO val2017 split.

PixArtSigma ControlNet

[Figure 147]

[Figure 148]

###### 5.3 Video Generation with Multiple Control Conditions

As described in Sec. 3.3, users can achieve multi-source control by simply combining the control features of multiple ControlNets via our CTRL-Adapter. Table 4 shows the result in two folds: firstly, patch-level MoE routers (i.e., variants b and c in Fig. 5) consistently outperforms the equal weights baseline as well as the unconditional global weights (i.e., variant a in Fig. 5), which proves the effectiveness of patch-level fine-grained control composition. Secondly, as shown in (b) and (c), control with more conditions almost always yields better spatial control and visual quality than control with a single condition. Fig. 28 and Fig. 29 show that multi-condition composition provides more accurate control compared to a single condition. Table 8 extends (a) by conditioning on image/text/timestep embeddings.

Input Sparse Conditions (User Scribble, Depth Map) I2VGen-XL + CTRL-Adapter

Input Image/Prompt

|No Condition|No Condition|
|---|---|
|No Condition|No Condition|

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Fly through tour of a museum with many paintings and sculptures and beautiful works of art in all styles

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Reﬂections in the window of a train traveling through the Tokyo suburbs

Figure 8: Video generation from sparse frame conditions with CTRL-Adapter on I2VGen-XL (which generates 16 frames in total). We only provide controls for the 1st, 6th, 11th, and 16th frames.

###### 5.4 Video Generation with Sparse Frames as Control Condition

We experiment CTRL-Adapter with providing sparse frame conditions using I2VGen-XL as backbone. During each training step, we first randomly select an integer k ∈ {1,...,N}, where N is equal to the total number of output frames (e.g., N = 16 for I2VGen-XL). Next, we randomly select k key

frames from N total frames. We then extract these key frames’ depth maps and user scribbles as control conditions. we do not give the latents z and only give the k frames to ControlNet. In Fig. 8, we can see that I2VGen-XL with our CTRL-Adapter can correctly generate videos that follow the control conditions for the given 4 sparse key frames and make reasonable interpolations on the frames without conditions. In Appendix E.3, we show that skipping the latent from ControlNet inputs is important in improving the sparse control capability.

###### 5.5 Zero-Shot Generalization on Unseen Conditions

ControlNet can be understood as an image feature extractor that maps different types of controls to the unified representation space of backbone generation models. This begs an interesting question: “Does CTRL-Adapter learn general feature mapping from one (smaller) backbone to another (larger) backbone?” To answer this question, we experiment by directly plugging CTRL-Adapter to ControlNets that are not seen during training. In Fig. 9, we observe the CTRL-Adapter trained on depth maps can adapt to normal map and soft edge ControlNets in a zero-shot manner. Quantitative analysis of different training strategies based on such observation is illustrated in Appendix F.1.

Input Image Depth Map ➡ Video Normal Map ➡ Video Soft Edge ➡ Video

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

Training condition Zero-shot inference with other conditions

- Figure 9: Zero-shot transfer of CTRL-Adapter trained only on depth maps to unseen conditions.

6 Downstream Tasks Beyond Spatial Control

Here, we aim to qualitatively explore how other types of ControlNets can be seamlessly integrated into our framework to enable a wide variety of downstream tasks beyond spatial control. As shown in Fig. 10, we can achieve video editing by combining image and video CTRL-Adapters with user edited prompts; video style transfer via shuffle ControlNet + CTRL-Adapter; and text-guided motion control for masked object via inpainting ControlNet + CTRL-Adapter. See Appendix H.4 for details.

Depth ControlNet + SDXL + CTRL-Adapter

Output Video Source Video Extracted Depth

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

A camel with rainbow fur walking

Prompt

Video Editing

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Output Video

Shuffle A sea turtle

swimming gracefully above a coral reef

Prompt

Video Style Transfer

Output Video

Inpainting ControlNet An elk with

impressive antlers grazing on the snow-covered ground

Prompt

Text-Guided Motion Control

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

A white and orange tabby alley cat is seen darting across a back street alley in a heavy rain, looking for shelter.

Object Mask

Shuffle ControlNet + Latte + CTRL-Adapter

Inpainting ControlNet + I2VGen-XL + CTRL-Adapter

Depth ControlNet + I2VGen-XL + CTRL-Adapter

- Figure 10: Illustration of additional downstream tasks supported by our CTRL-Adapter framework.

##### 7 Conclusion

We propose CTRL-Adapter, an efficient, powerful, and versatile framework that adds diverse controls to any image/video diffusion model. Training an CTRL-Adapter is significantly more efficient than training a ControlNet for a new backbone, and it can outperform or match strong baselines in visual quality and spatial control. CTRL-Adapter not only provides many useful capabilities including image/video control, sparse frame control, multi-condition control, and zero-shot adaption to unseen conditions, but also can be easily and flexibly integrated into a variety of downstream tasks.

##### Acknowledgments

This work was supported by DARPA ECOLE Program No. HR00112390060, NSF-AI Engage Institute DRL-2112635, DARPA Machine Commonsense (MCS) Grant N66001-19-2-4031, ARO Award W911NF2110220, ONR Grant N00014-23-1-2356, and a Bloomberg Data Science Ph.D. Fellowship. The views contained in this article are those of the authors and not of the funding agency.

##### References

- [1] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, M. Suo, P. Tillet, E. Wang, X. Wang, W. Wen, S. Zhang, X. Zhao, K. Zhou, R. Zou, A. Mathews, G. Chanan, P. Wu, and S. Chintala. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24). ACM, Apr. 2024. 5, 17, 40
- [2] O. Avrahami, T. Hayes, O. Gafni, S. Gupta, Y. Taigman, D. Parikh, D. Lischinski, O. Fried, and X. Yin. SpaText: Spatio-Textual Representation for Controllable Image Generation. In CVPR, nov 2023. 2, 3, 16
- [3] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 5, 6, 16, 17, 20, 24, 29, 40
- [4] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools, 2000. 20
- [5] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7291–7299, 2017. 20
- [6] H. Chen, Y. Zhang, X. Cun, M. Xia, X. Wang, C. Weng, and Y. Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 2
- [7] J. Chen, Y. Wu, S. Luo, E. Xie, S. Paul, P. Luo, H. Zhao, and Z. Li. Pixart-{\delta}: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024. 7, 8
- [8] J. Chen, J. YU, C. GE, L. Yao, E. Xie, Z. Wang, J. Kwok, P. Luo, H. Lu, and Z. Li. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 2, 6, 20, 23, 40
- [9] T.-S. Chen, A. Siarohin, W. Menapace, E. Deyneka, H.-w. Chao, B. E. Jeon, Y. Fang, H.-Y. Lee, J. Ren, M.-H. Yang, and S. Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In CVPR 2024, 2024. 6, 19, 20, 40
- [10] W. Chen, J. Wu, P. Xie, H. Wu, J. Li, X. Xia, X. Xiao, and L. Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023. 7
- [11] X. Dai, J. Hou, C.-Y. Ma, S. Tsai, J. Wang, R. Wang, P. Zhang, S. Vandenhende, X. Wang, A. Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 20
- [12] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, D. Podell, T. Dockhorn, Z. English, K. Lacey, A. Goodwin, Y. Marek, and R. Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 5
- [13] Estimation lemma. Estimation lemma — Wikipedia, the free encyclopedia, 2010. 5, 17
- [14] G. Farnebäck. Two-frame motion estimation based on polynomial expansion. In Image Analysis: 13th Scandinavian Conference, SCIA 2003 Halmstad, Sweden, June 29–July 2, 2003 Proceedings 13, pages 363–370. Springer, 2003. 20
- [15] G. Farnebäck. Two-frame motion estimation based on polynomial expansion. In Scandinavian Conference on Image Analysis, 2003. 20
- [16] O. Gafni, A. Polyak, O. Ashual, S. Sheynin, D. Parikh, and Y. Taigman. Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors. In ECCV, 2022. 2, 3, 16

- [17] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In ICLR, 2023. 2, 3, 16
- [18] R. Girdhar, M. Singh, A. Brown, Q. Duval, S. Azadi, S. S. Rambhatla, A. Shah, X. Yin, D. Parikh, and

I. Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 2

- [19] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 16
- [20] Q. Guo and D. Yue. Dit-visualization. https://github.com/guoqincode/DiT-Visualization,

2024. Exploring the differences between DiT-based and Unet-based diffusion models in feature aspects using code from diffusers, Plug-and-Play, and PixArt. 27

- [21] Y. Guo, C. Yang, A. Rao, M. Agrawala, D. Lin, and B. Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933, 2023. 3, 16, 18, 19
- [22] Y. Guo, C. Yang, A. Rao, Y. Wang, Y. Qiao, D. Lin, and B. Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In International Conference on Learning Representations, 2024. 16
- [23] A. Gupta, L. Yu, K. Sohn, X. Gu, M. Hahn, L. Fei-Fei, I. Essa, L. Jiang, and J. Lezama. Photorealistic video generation with diffusion models, 2023. 16
- [24] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. 21
- [25] Y. He, T. Yang, Y. Zhang, Y. Shan, and Q. Chen. Latent video diffusion models for high-fidelity long video generation, 2022. 16
- [26] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30,

2017. 6, 21

- [27] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 16
- [28] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4, 16
- [29] E. Hoogeboom, J. Heek, and T. Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 5
- [30] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly. Parameter-efficient transfer learning for nlp. In ICML, volume abs/1902.00751, 2019. 2
- [31] Z. Hu and D. Xu. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073, 2023. 3, 6, 16, 20, 21
- [32] L. Khachatryan, A. Movsisyan, V. Tadevosyan, R. Henschel, Z. Wang, S. Navasardyan, and H. Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In ICCV 2023, 2023. 7, 16
- [33] S. Kim, J. Lee, K. Hong, D. Kim, and N. Ahn. Diffblender: Scalable and composable multimodal text-to-image diffusion models. arXiv preprint arXiv:2305.15194, 2023. 3, 16
- [34] D. P. Kingma and M. Welling. Auto-encoding variational bayes. In ICLR, 2014. 16
- [35] B. Lefaudeux, F. Massa, D. Liskovich, W. Xiong, V. Caggiano, S. Naren, M. Xu, J. Hu, M. Tintore, S. Zhang, P. Labatut, D. Haziza, L. Wehrstedt, J. Reizenstein, and G. Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022. 19
- [36] D. Li, J. Li, and S. C. H. Hoi. BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing. In NeurIPS, 2023. 2, 3, 16
- [37] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023. 6

- [38] Y. Li, Z. Gan, Y. Shen, J. Liu, Y. Cheng, Y. Wu, L. Carin, D. Carlson, and J. Gao. Storygan: A sequential conditional gan for story visualization. In CVPR, 2019. 16
- [39] Y. Li, H. Liu, Q. Wu, F. Mu, J. Yang, J. Gao, C. Li, and Y. J. Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 2, 3, 7, 8, 16
- [40] Y. Li, M. R. Min, D. Shen, D. Carlson, and L. Carin. Video generation from text. In AAAI, 2017. 16
- [41] H. Lin, A. Zala, J. Cho, and M. Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023. 2
- [42] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2, 6, 21, 40
- [43] F. Long, Z. Qiu, T. Yao, and T. Mei. Videodrafter: Content-consistent multi-scene video generation with llm. arXiv preprint arXiv:2401.01256, 2024. 2
- [44] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018. 18
- [45] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, and S. Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740,

2024. 5

- [46] X. Ma, Y. Wang, G. Jia, X. Chen, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2, 5, 6, 20, 23, 24, 29, 40
- [47] W. Menapace, A. Siarohin, I. Skorokhodov, E. Deyneka, T.-S. Chen, A. Kag, Y. Fang, A. Stoliar, E. Ricci, J. Ren, and S. Tulyakov. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis, 2024. 16
- [48] C. Mou, X. Wang, L. Xie, Y. Wu, J. Zhang, Z. Qi, Y. Shan, and X. Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI 2024, 2023. 3, 7, 8, 16
- [49] J. Mullan, D. Crawbuck, and A. Sastry. Hotshot-XL, Oct. 2023. 2, 6, 16, 20, 29, 40
- [50] OpenAI. Video generation models as world simulators, 2024. 16
- [51] G. Parmar, R. Zhang, and J.-Y. Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022. 21
- [52] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. 2, 6, 17, 20, 40
- [53] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 2, 6, 20, 40
- [54] C. Qin, S. Zhang, N. Yu, Y. Feng, X. Yang, Y. Zhou, H. Wang, J. C. Niebles, C. Xiong, S. Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. In NeurIPS 2023,

2023. 3, 6, 16, 21

- [55] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical Text-Conditional Image Generation with CLIP Latents, 2022. 2
- [56] L. Ran, X. Cun, J.-W. Liu, R. Zhao, S. Zijie, X. Wang, J. Keppo, and M. Z. Shou. X-adapter: Adding universal compatibility of plugins for upgraded diffusion model. In CVPR, 2024. 3, 7, 8, 16, 18, 19, 20, 21
- [57] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 20
- [58] A. Ranjan and M. J. Black. Optical flow estimation using a spatial pyramid network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4161–4170, 2017. 6, 21
- [59] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020. 19

- [60] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021. 5
- [61] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4, 16, 17
- [62] C. Rowles. Stable Video Diffusion Temporal Controlnet. https://github.com/CiaraStrawberry/ svd-temporal-controlnet, 2023. 7
- [63] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In CVPR, 2023. 2, 3, 16
- [64] S. Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 2022. 3, 16
- [65] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes, T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. In NeurIPS, 2022. 2
- [66] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 20
- [67] C. Schuhmann and P. Bevan. Laion pop: 600,000 high-resolution images with detailed descriptions. https://huggingface.co/datasets/laion/laion-pop, 2023. 6, 19, 40
- [68] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean. Outrageously Large Neural Networks: the Sparsely-Gated Mixture-of-Experts Layer. In ICLR, 2017. 6
- [69] U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023. 16
- [70] I. Skorokhodov, S. Tulyakov, and M. Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In CVPR, 2022. 16
- [71] J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 16
- [72] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel. Plug-and-play diffusion features for text-driven imageto-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 27
- [73] P. von Platen, S. Patil, A. Lozhkov, P. Cuenca, N. Lambert, K. Rasul, M. Davaadorj, and T. Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers, 2022. 7, 8, 40
- [74] F.-Y. Wang, W. Chen, G. Song, H.-J. Ye, Y. Liu, and H. Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023. 16
- [75] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang. Modelscope text-to-video technical report,

2023. 16

- [76] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 20
- [77] X. Wang, H. Yuan, S. Zhang, D. Chen, J. Wang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024. 3, 7, 16, 21
- [78] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6, 21
- [79] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, Oct. 2020. Association for Computational Linguistics. 40
- [80] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision (ECCV), pages 418–434, 2018. 20

- [81] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021. 20
- [82] J. Xing, M. Xia, Y. Zhang, H. Chen, W. Yu, H. Liu, X. Wang, T.-T. Wong, and Y. Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023. 16
- [83] Z. Yang, J. Wang, Z. Gan, L. Li, K. Lin, C. Wu, N. Duan, Z. Liu, C. Liu, M. Zeng, and L. Wang. Reco: Region-controlled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2, 3, 16
- [84] M. B. Yi-Lin Sung, Jaemin Cho. Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks. In CVPR, 2022. 2
- [85] S. Yin, C. Wu, H. Yang, J. Wang, X. Wang, M. Ni, Z. Yang, L. Li, S. Liu, F. Yang, J. Fu, M. Gong, L. Wang, Z. Liu, H. Li, and N. Duan. NUWA-XL: Diffusion over diffusion for eXtremely long video generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1309–1320, Toronto, Canada, July 2023. Association for Computational Linguistics. 16
- [86] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3, 4, 6, 7, 8, 16, 17, 20, 40
- [87] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. https://huggingface.co/lllyasviel/sd-controlnet-depth, 2023. 2
- [88] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. https://huggingface.co/lllyasviel/sd-controlnet-canny, 2023. 2
- [89] S. Zhang, J. Wang, Y. Zhang, K. Zhao, H. Yuan, Z. Qin, X. Wang, D. Zhao, and J. Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145,

2023. 2, 6, 16, 17, 20, 24, 29, 40

- [90] Y. Zhang, Y. Wei, D. Jiang, X. Zhang, W. Zuo, and Q. Tian. Controlvideo: Training-free controllable text-to-video generation. In ICLR, 2024. 3, 6, 7, 16, 20
- [91] L. Zhao, X. Peng, Y. Tian, M. Kapadia, and D. Metaxas. Learning to forecast and refine residual motion for image-to-video generation. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018. 16
- [92] S. Zhao, D. Chen, Y.-C. Chen, J. Bao, S. Hao, L. Yuan, and K.-Y. K. Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 3, 6, 7, 8, 16, 21
- [93] D. Zhou, W. Wang, H. Yan, W. Lv, Y. Zhu, and J. Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 16

### Appendix

- A Background 16

- A.1 Extended Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 Extended Preliminaries: LDM and ControlNet . . . . . . . . . . . . . . . . . . . . 16

- B CTRL-Adapter Method and Architecture Details 17

- B.1 CTRL-Adapter Architecture Details . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 PyTorch Implementation for Inverse Timestep Sampling . . . . . . . . . . . . . . 17
- B.3 Comparison of CTRL-Adapter Variants and Related Methods . . . . . . . . . . . . 18

- C Training and Inference Details 18
- D Experimental Setup 20

- D.1 ControlNets and Target Diffusion Models . . . . . . . . . . . . . . . . . . . . . . 20
- D.2 Training Datasets for CTRL-Adapter . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.3 Input Conditions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.4 Evaluation Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.5 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E Variants of CTRL-Adapter Architecture Design 21

- E.1 CTRL-Adapter Design Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E.1.1 Combinations of components within each CTRL-Adapter . . . . . . . . . . 21
- E.1.2 Where to fuse CTRL-Adapter outputs in backbone diffusion . . . . . . . . 22
- E.1.3 Number of CTRL-Adapters in each output block position . . . . . . . . . . 22

- E.2 Adaptation to DiT-Based Backbones . . . . . . . . . . . . . . . . . . . . . . . . . 23
- E.3 Skipping Latent from ControlNet Inputs . . . . . . . . . . . . . . . . . . . . . . . 24
- E.4 Different weighing modules for multi-condition generation . . . . . . . . . . . . . 24

- F Additional Quantitative Analysis 25

- F.1 Train individual CTRL-Adapter v.s. train a unified CTRL-Adapter . . . . . . . . . . 25
- F.2 Trade-off between Visual Quality and Spatial Control . . . . . . . . . . . . . . . . 25

- G Additional Qualitative Analysis 27

- G.1 Visualization of Spatial Feature Maps . . . . . . . . . . . . . . . . . . . . . . . . 27
- G.2 Fast Training Convergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- H Additional Visualization Examples 29

- H.1 Video Generation Visualization Examples . . . . . . . . . . . . . . . . . . . . . . 29
- H.2 Multi-Condition Video Generation Visualization Examples . . . . . . . . . . . . . 33
- H.3 Image Generation Visualization Examples . . . . . . . . . . . . . . . . . . . . . . 34
- H.4 Visualization Examples for Additional Downstream Tasks . . . . . . . . . . . . . 37

- I Broader Impacts 40
- J Limitations 40
- K License 40

##### A Background

###### A.1 Extended Related Works

Text-to-video and image-to-video generation models. Generating videos from text descriptions or images (e.g., initial video frames) based on deep learning and has increasingly gained much attention. Early works for this task [40, 38, 91, 70] have commonly used variational autoencoders (VAEs) [34] and generative adversarial networks (GANs) [19], while most of recent video generation works are based on denoising diffusion models [28, 71]. Powered by large-scale training, recent video diffusion models demonstrate impressive performance in generating highly realistic videos from text descriptions [25, 27, 69, 93, 32, 74, 85, 75, 49, 50, 23, 47] or initial video frames (i.e., images) [3, 89, 22, 82].

Adding control to image/video diffusion models. While recent image/video diffusion models demonstrate impressive performance in generating highly realistic images/videos from text descriptions, it is hard to describe every detail of images/videos only with text or first frame image. Instead, there have been many works using different types of additional inputs to control the image/video diffusion models, such as bounding boxes [39, 83], reference object image [63, 17, 36], segmentation map [16, 2, 86], sketch [86], etc., and combinations of multiple conditions [33, 54, 92, 77]. As finetuning all the parameters of such image/video diffusion models is computationally expensive, several methods, such as ControlNet [86], have been proposed to add conditional control capability via parameter-efficient training [86, 64, 48]. X-Adapter [56] learns an adapter module to reuse ControlNets pretrained with a smaller image diffusion model (e.g., SDv1.5) for a bigger image diffusion model (e.g., SDXL). While they focus solely on learning an adapter for image control, CTRL-Adapter features architectural designs (e.g., temporal convolution/attention layers) for video generation as well. In addition, X-Adapter needs the smaller image diffusion model (SDv1.5) during training and inference, whereas CTRL-Adapter doesn’t need the smaller diffusion model at all (for image/video generation), hence being more memory and computationally efficient (see Appendix B.3 for details). SparseCtrl [21] guides a video diffusion model with conditional inputs of few frames (instead of full frames), to alleviate the cost of collecting video conditions. Since SparseCtrl involves augmenting ControlNet with an additional channel for frame masks, it requires training a new variant of ControlNet from scratch. In contrast, we leverage existing image ControlNets more efficiently by propagating information through temporal layers in adapters and enabling sparse frame control via skipping the latents from ControlNet inputs (see Sec. 3.2 for details). Furthermore, compared with previous works that are specially designed for specific condition controls on a single modality (image [86, 54] or video [31, 90]), our work presents a unified and versatile framework that supports diverse controls, including image control, video control, sparse frame control, and multi-source control, with significantly lower computational costs by reusing pretrained ControlNets (e.g., CTRLAdapter outperforms baselines in less than 10 GPU hours). Table 1 summarizes the comparison of CTRL-Adapter with related works.

###### A.2 Extended Preliminaries: LDM and ControlNet

Latent Diffusion Models. Many recent video generation works are based on latent diffusion models (LDMs) [61], where a diffusion model learns the temporal dynamics of compact latent representations of videos. First, given a F-frame RGB video x ∈ RF×3×H×W, a video encoder (of a pretrained autoencoder) provides C-dimensional latent representation (i.e., latents): z =

′×W′, where height and width are spatially downsampled (H′ < H and W′ < W). Next, in the forward process, a noise scheduler such as DDPM [28] gradually adds noise to the latents z: q(zt|zt−1) = N(zt;√1 − βtzt−1,βtI), where βt ∈ (0,1) is the variance schedule

- E(x) ∈ RF×C×H

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

TemporalSelf-Attn

AddTime&Norm

AddTime&Norm

Up/DownSample

SpatialSelf-Attn

CrossAttention

CrossAttention

Add&Norm

Add&Norm

Add&Norm

Add&Norm

Add&Norm

Add&Norm

2DConv

2DConv

3DConv

3DConv

Norm

Norm

FFN

FFN

FFN

FFN

###### Spatial Conv Temporal Conv Spatial Attention Temporal Attention

Figure 11: Detailed architecture of CTRL-Adapter blocks.

with t ∈ {1,...,T}. Then, in the backward pass, a diffusion model (usually a U-Net architecture)

- Fθ(zt,t,ctext/img) learns to gradually denoise the latents, given a diffusion timestep t, and a text prompt ctext (i.e., T2V) or an initial frame cimg (i.e., I2V) if provided. The diffusion model is trained

with following objective: LLDM = Ez,ϵ∼N(0,I),t∥ϵ − ϵθ(zt,t,ctext/img)∥22, where ϵ and ϵθ represent the added noise to latents and the predicted noise by Fθ respectively. We apply the same objective for CTRL-Adapter training.

ControlNets. ControlNet [86] is designed to add spatial controls to image diffusion models in the form of different guidance images (e.g., depth, sketch, segmentation maps, etc.). Specifically, given a pretrained backbone image diffusion model Fθ that consists of input/middle/output blocks, ControlNet has a similar architecture Fθ′, where the input/middle blocks parameters of θ′ are initialized from θ, and the output blocks consist of 1x1 convolution layers initialized with zeros. ControlNet takes the diffusion timestep t, text prompt ctext, control image cf (e.g., depth image), and the noisy latents zt as inputs, and provides the features that are merged into middle/output blocks of backbone image model Fθ to generate the final image. The authors of ControlNet have released a variety of ControlNet checkpoints based on Stable Diffusion [61] v1.5 (SDv1.5) and the user community have also shared many ControlNets trained with different input conditions based on SDv1.5. However, these ControlNets cannot be used with more recently released bigger and stronger image/video diffusion models, such as SDXL [52] and I2VGen-XL [89]. Moreover, the input/middle blocks of the ControlNet are in the same size with those of the diffusion backbones (i.e., if the backbone model gets bigger, ControlNet also gets bigger). Due to this, it becomes increasingly difficult to train new ControlNets for each bigger and newer model that is released over time. To address this, we introduce CTRL-Adapter for efficient adaption of existing ControlNets for new diffusion models.

##### B CTRL-Adapter Method and Architecture Details

###### B.1 CTRL-Adapter Architecture Details

In Fig. 11, we illustrate the detailed architecture of CTRL-Adapter blocks. See Fig. 3 for how the CTRL-Adapter blocks are used to adapt ControlNets to image/video diffusion models. Fig. 11 is an extended version of Fig. 3 (right) with more detailed visualizations, including skip connections, normalization layers in each module, and the linear projection layers (i.e., FFN) in each spatial/temporal attention modules.

###### B.2 PyTorch Implementation for Inverse Timestep Sampling

In Algorithm 1, we provide the PyTorch [1] implementation of inverse timestep sampling, described in Sec. 3.2. In the example, inverse time stamping adapts to the SVD [3] backbone.

During each training step, the procedure for this algorithm can be summarized as follows:

- • Sample a variable u from Uniform[0,1]. See line 19 in function inverse_timestamp_sample.
- • Sample noise scale σcont. via inverse transform sampling [13]; i.e., we derive the inverse cumula-

tive density function of σcont. and sample σcont. by sampling u: σcont. = Fcont.−1 (u). See function sample_sigma and line 21 in function inverse_timestamp_sample.

- • Given a preconditioning function gcont. that maps noise scale to timestep (typically associated with the continuous-time noise sampler), we can compute tcont. = gcont.(σcont.). See function sigma_to_timestep and line 23 in inverse_timestamp_sample.
- • Set the timesteps and noise scales for both ControlNet and our CTRL-Adapter as tCNet = round(1000u) and σCNet = u respectively, where 1000 represents the denoising timestep range over which the ControlNet is trained. See line 25 in inverse_timestamp_sample.

During inference, we follow the similar sampling strategy, with the only change in the first step. Instead of uniformly sample a single value for u, we uniformly sample k equidistant values for u within [0,1] and derive corresponding tcont./CNet and σcont./CNet as inputs for denoising steps, where k here is the number of denoising steps during inference.

Algorithm 1 PyTorch Implementation for Inverse Timestep Sampling

- 1 import torch
- 2
- 3 def sample_sigma(u, loc=0., scale=1.):
- 4 """Draw a noise scale (sigma) from the noise schedule of Karras et al. (2022)"""
- 5 sigma_min, sigma_max, rho = 0.002, 700, 7 # values used in the paper
- 6 min_inv_rho, max_inv_rho = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
- 7 sigma = (max_inv_rho + (1-u) * (min_inv_rho - max_inv_rho)) ** rho
- 8 return sigma
- 9
- 10 def sigma_to_timestep(sigma):
- 11 """Map noise scale to timestep. Here we use the function used in SVD."""
- 12 timestep = 0.25 * sigma.log()
- 13 return timestep
- 14
- 15 def inverse_timestamp_sample():
- 16 """Sample noise scales and timesteps for ControlNet and diffusion models
- 17 trained with continuous noise sampler. Here we use the setting used for SVD."""
- 18 # 1) sample u from Uniform[0,1]
- 19 u = torch.rand(1)
- 20 # 2) calculate sigma_svd from pre-defined log-normal distribution
- 21 sigma_svd = sample_sigma(u, loc=0.7, scale=1.6)
- 22 # 3) calculate timestep_svd from sigma_svd via pre-defined mapping function
- 23 timestep_svd = sigma_to_timestep(sigma_svd)
- 24 # 4) calculate timestep and sigma for controlnet
- 25 sigma_cnet, timestep_cnet = u, round(1000 * u)
- 26 return sigma_svd, timestep_svd, sigma_cnet, timestep_cnet

###### B.3 Comparison of CTRL-Adapter Variants and Related Methods

In Fig. 12, we compare the variants of CTRL-Adapter designs (with latent is given / not given to ControlNet; see Sec. 3.2 for details) and two related methods: SparseCtrl [21] and X-Adapter [56]. Unlike CTRL-Adapter that leverages the pretrained image ControlNets, SparseCtrl (Fig. 12 c) trains a video ControlNet with control conditions cf and frame masks m as inputs. While X-Adapter (Fig. 12 d) needs SDv1.5 U-Net as well as SDv1.5 ControlNet during training and inference, CTRL-Adapter doesn’t need to SDv1.5 U-Net at all.

##### C Training and Inference Details

Model architectures. Detailed illustration of our CTRL-Adapter architecture has been provided across several parts of our paper, including Sec. 3, Appendix B.1, Appendix E, and Appendix F. In addition, for all the backbone models used in this paper, we kept all their parameters frozen and made no modifications.

Training details. We use a learning rate of 5 × e−5; AdamW [44] optimizer with values for β1, β2, ϵ, and weight decay as 0.9, 0.999, 1 × e−8, and 1 × e−2 respectively. We set the max gradient norm as 1. All our experiments are trained on 4 A100 80GB GPUs with batch size of 1. Detailed study

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

CTRL-Adapter

CTRL-Adapter

[Figure 224]

[Figure 225]

Image ControlNets

Image ControlNets

Image/Video UNet

Image/Video UNet

Latents not given to ControlNet

###### (a) CTRL-Adapter (Default)

(b) CTRL-Adapter: Latents not given to ControlNet

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

X-Adapter

[Figure 242]

Video UNet

Video ControlNets

SDXL UNet SDv1.5 UNet

SDv1.5 ControlNet

(c) SparseCtrl

(d) X-Adapter

- Figure 12: Comparison of giving different inputs to ControlNet, where zt,cf, and t represent latents, input control features, and timesteps respectively. (a): Default CTRL-Adapter design. (b): Variant of CTRL-Adapter where latents zt are not given to ControlNet (see Sec. 3.2 for details). (c): SparseCtrl [21] trains a video ControlNet with control conditions cf and frame masks m as inputs. (d): X-Adapter [56] needs SDv1.5 U-Net as well as SDv1.5 ControlNet during training and inference, whereas CTRL-Adapter doesn’t need to SDv1.5 U-Net at all.

of per-GPU training memory for different model architecture variants are shown in Fig. 13. Please note that other than mixed-precision training with data type bfloat16, we didn’t use any additional methods to speed up the training/inference clock time, or to save GPU memory. To be more specific, we didn’t use any of the following methods: xformers [35], gradient checkpointing, 8bit Adam optimizer, and DeepSpeed [59]. In addition, to make our framework easy to use directly from raw input images/videos, we extract all control condition images/frames on-the-fly during training. We train the image and video CTRL-Adapters for 80k and 40k steps respectively, which can be finished in 24 hours measured by training clock time. The fast convergence of our method is shown in Fig. 2.

Inference details. All inference can be done on a single A6000 GPU with 48GB memory. During inference, we use the default hyper-parameters for each backbone model, including the number of frames to generate, the number of denoising steps, and classifier-free guidance scale, etc..

Safeguards. When we generate images during inference, we also activate the NSFW filter of the backbone models. This ensures that users are protected from unnecessary exposure to explicit or objectionable materials. For training, the datasets we used [9, 67] both filter out the image/video samples with harmful contents. For example, as stated in the “Risk mitigation” section of Panda70M paper, they used the internal automatic pipeline to filter out the video samples with harmful or violent language and texts that include drugs or hateful speech. They also use the NLTK framework to replace all people’s names with "person". LAION-POP dataset is also created by filtering out samples based on the safety tags (using a customized trained NSFW classifier that they built).

##### D Experimental Setup

###### D.1 ControlNets and Target Diffusion Models

ControlNets. We use ControlNets trained with SDv1.5.2 SDv1.5 has the most number of publicly released ControlNets and has a much smaller training cost compared to recent image/video diffusion models. Note that unlike X-Adapter [56], CTRL-Adapter does not need to load the source diffusion model (SDv1.5) during training or inference (see (a) and (d) in Fig. 12 for model architecture comparison).

Target diffusion models (where ControlNets are to be adapted). For video generation models, we experiment with two text-to-video generation models – Latte [46] and Hotshot-XL [49], and two image-to-video generation models – I2VGen-XL [89] and Stable Video Diffusion (SVD) [3]. For image generation model, we experiment with PixArt-α [8] and the base model in SDXL [52]. For all models, we use their default settings during training and inference (e.g., number of output frames, resolution, number of denoising steps, classifier-free guidance scale, etc.).

###### D.2 Training Datasets for CTRL-Adapter

Video datasets. For training CTRL-Adapter for video diffusion models, we download around 1.5M videos randomly sampled from the Panda-70M training set [9]. Following recent works [3, 11], we filter out videos of static scenes by removing videos whose average optical flow [14, 4] magnitude is below a certain threshold. Concretely, we use the Gunnar Farneback’s algorithm3 [15] at 2FPS, calculate the averaged the optical flow for each video and re-scale it between 0 and 1, and filter out videos whose average optical flow error is below a threshold of 0.25. This process gives us a total of 200K remaining videos.

Image datasets. For training CTRL-Adapter for image diffusion models, we use 300K images randomly sampled from LAION POP,4 which is a subset of LAION 5B [66] dataset and contains 600K images in total with aesthetic values of at least 0.5 and a minimum resolution of 768 pixels on the shortest side. As suggested by the authors, we use the image captions generated with CogVLM [76].

###### D.3 Input Conditions We extract various input conditions from the video and image datasets described above.

- • Depth map: As recommended in Midas5 [57], we employ dpt_swin2_large_384 for the best speed-performance trade-off.
- • Canny edge, surface normal, line art, softedge, and user sketching/scribbles: Following ControlNet [86], we utilize the same annotator implemented in the controlnet_aux6 library.
- • Semantic segmentation map: To obtain higher-quality segmentation maps than UPerNet [80] used in ControlNet, we employ SegFormer [81] segformer-b5-finetuned-ade-640-640 finetuned on ADE20k dataset at 640×640 resolution.
- • Human pose: We employ ViTPose [81] ViTPose_huge_simple_coco to improve both processing speed and estimation quality, compared to OpenPose [5] used in ControlNet.

###### D.4 Evaluation Datasets

Video datasets. Following previous works [31, 90], we evaluate our video ControlNet adapters on DAVIS 2017 [53], a public benchmark dataset also used in other controllable video generation works [31]. We first combine all video sequences from TrainVal, Test-Dev 2017 and Test-Challenge 2017. Then, we chunk each video into smaller clips, with the number of frames

- 2https://huggingface.co/lllyasviel/ControlNet
- 3https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- 4https://laion.ai/blog/laion-pop/
- 5https://github.com/isl-org/MiDaS
- 6https://github.com/huggingface/controlnet_aux

[Figure 243]

Training GPU Memory

0 20 40GB

[Figure 244]

Training GPU Memory

0 40 80GB

- Figure 13: Comparison of different architecture of CTRL-Adapter for image and video control, measured with visual quality (FID) and spatial control (MSE/optical flow error) metrics. The metrics are calculated from 1000 randomly selected COCO val2017 images and 150 videos from DAVIS 2017 dataset respectively. Left: image control on SDXL backbone. Right: video control on I2VGen-XL backbone. For both plots, data points in the bottom-left are ideal. SC, TC, SA, and TA: Spatial Convolution, Temporal Convolution, Spatial Attention, and Temporal Attention. ∗N represents the number of blocks in each CTRL-Adapter. The diameters of bubbles represent training GPU memory.

in each clip being the same as the default number of frames generated by each video backbone (e.g., 8 frames for Hotshot-XL, 16 frames for I2VGen-XL, and 14 frames for SVD). This process results in a total of 1281 video clips of 8 frames, 697 clips of 14 frames, and 608 video clips of 16 frames.

Image datasets. We evaluate our image ControlNet adapters on COCO val2017 split [42], which contains 5k images that cover diverse range of daily objects. We resize and center crop the images to 1024 by 1024 for SDXL evaluation.

###### D.5 Evaluation Metrics

Visual quality. Following previous works [54, 31], we use Frechet Inception Distance (FID) [26] to measure the distribution distance between our generated images/videos and input images/videos.7

Spatial control. For video datasets, following VideoControlNet [31], we report the L2 distance between the optical flow [58] of the input video and the generated video (Optical Flow Error). For image datasets, following Uni-ControlNet [92], we report the Structural Similarity (SSIM) [78]8 and mean squared error (MSE)9 between generated images and ground truth images.

##### E Variants of CTRL-Adapter Architecture Design

- E.1 CTRL-Adapter Design Ablations

- E.1.1 Combinations of components within each CTRL-Adapter

As described in Sec. 3.2, each CTRL-Adapter module consists of four components: spatial convolution (SC), temporal convolution (TC), spatial attention (SA), and temporal attention (TA). We experiment with different architecture combinations of the adapter components for image and video control, and plot the results in Fig. 13. Compared to X-Adapter [56], which uses a stack of three spatial convolution modules (i.e., ResNet [24] blocks) for adapters, and VideoComposer [77], which employs spatial convolution + temporal attention for spatiotemporal condition encoder, we explore a richer combination that enhances global understanding of spatial information through spatial attention and improves temporal ability via a combination of temporal convolution and temporal attention. For image control (Fig. 13 left), we find that the combining of SC+SA is more effective than stacking

7To be consistent with the numbers reported in Uni-ControlNet [92], we use pytorch-fid (https://github. com/mseitzer/pytorch-fid) in Table 3. For other results, we use clean-fid [51] (https://github.com/ GaParmar/clean-fid) which is more robust to aliasing artifacts.

- 8https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
- 9https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18

| |Training Steps<br><br>10k 20k 30k<br><br>| | | | | |
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

Training Steps

10k 20k 30k

5.5

OpticalFlowError()

5.0

###### FID()

4.5

4.0

3.5

3.0

Mid + Out ABCD (Default)

Out ABCD Out ABC Out AB Out A

Mid + Out ABCD (Default)

Out ABCD Out ABC Out AB Out A

Adapter Insert Positions

Adapter Insert Positions

- Figure 14: Comparison of inserting CTRL-Adapter to different U-Net blocks. ‘Mid’ represents the middle block, whereas ‘Out ABCD’ represents output blocks A, B, C, and D. The metrics are calculated from 150 videos from DAVIS 2017 dataset.

Output Block D H4 ✕ W4

[Figure 245]

Output Adapter D

[Figure 246]

zero convolution

(a) 3 Blocks (default)

Output Block D H4 ✕ W4

[Figure 247]

Output Adapter D

[Figure 248]

zero convolution

Output Block D H4 ✕ W4

[Figure 249]

Output Adapter D

[Figure 250]

zero convolution

Output Block D H4 ✕ W4

[Figure 251]

Output Adapter D

[Figure 252]

zero convolution

(b) 2 Blocks

Output Block D H4 ✕ W4

[Figure 253]

Output Block D H4 ✕ W4

[Figure 254]

Output Adapter D

[Figure 255]

zero convolution

Output Block D H4 ✕ W4

[Figure 256]

Output Adapter D

[Figure 257]

zero convolution

(c) 1 Block

Output Block D H4 ✕ W4

[Figure 258]

Output Block D H4 ✕ W4

[Figure 259]

- Figure 15: Comparison of inserting different numbers of CTRL-Adapters to the backbone diffusion U-Net’s output blocks. We use output block D here for illustration. We insert three CTRL-Adapters to the output blocks of the same feature map size by default.

SC or SA layers only. Stacking SC+SA twice further improves the visual quality (FID) slightly but hurts the spatial control (MSE) as a tradeoff. Stacking SC+SA three times hurts the performance due to insufficient training. We use the single SC+SA layer for image CTRL-Adapter by default. For video control (Fig. 13 right), we find that SC+TC+SA+TA shows the best balance of visual quality (FID) and spatial control (optical flow error). Notably, we find that the combinations with both temporal layers, SC+TC+SA+TA and (TC+TA)*2, achieve the lowest optical flow error. We use SC+TC+SA+TA for video CTRL-Adapter by default.

###### E.1.2 Where to fuse CTRL-Adapter outputs in backbone diffusion

We compare the integration of CTRL-Adapter outputs at different positions of video diffusion backbone model. As illustrated in Fig. 3, we experiment with integrating CTRL-Adapter outputs to different positions of I2VGen-XL’s U-Net: middle block, output block A, output block B, output block C, and output block D. Specifically, we compared our default design (Mid + Out ABCD) with four other variants (Out ABCD, Out ABC, Out AB, and Out A) that gradually remove CTRL-Adapters from the middle block and output blocks at positions from B to D. As shown in Fig. 14, removing the CTRL-Adapters from the middle block and the output block D does not lead to a noticeable increase in FID or optical flow error (i.e., the performances of ‘Mid+Out ABCD’, ‘Out ABCD’, and ‘Out ABC’ are similar in both left and right plots). However, Fig. 14 (right) shows that removing CTRL-Adapters from block C causes a significant increase in optical flow error. Therefore, we recommend users retain our CTRL-Adapters in the mid and output blocks A/B/C to ensure good performance.

###### E.1.3 Number of CTRL-Adapters in each output block position

As illustrated in Fig. 3, there are three output blocks for each feature map dimension in the video diffusion model (represented by ×3 in each output block). Here, we conduct an ablation study by adding CTRL-Adapters to only one or two of the three output blocks of the same feature size. The motivation is that using fewer CTRL-Adapters can almost linearly decrease the number of trainable parameters, thereby reducing GPU memory usage during training. We visualize the architectural changes with output block D as an example in Fig. 15. We insert CTRL-Adapters for three blocks as our default setting. As observed in Fig. 16, reducing the number of CTRL-Adapters increases the optical flow error. Therefore, we recommend adding CTRL-Adapters to each output block to maintain optimal performance.

| |Training Steps<br><br>10k 20k 30k<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3.7

OpticalFlowError()

3.6

3.5

3.4

3.3

3.2

3.1

3.0

1 block 2 blocks 3 blocks

Number of Adapters per Block

- Figure 16: Comparison of inserting different numbers of CTRL-Adapters to each U-Net output block. The metrics are calculated from 150 videos from DAVIS 2017 dataset. We insert 3 CTRL-Adapters to each output block by default.

Transformer Block 1

F ✕ C ✕ H’ ✕ W’

F ✕ C ✕ H’ ✕ W’

[Figure 260]

Pretrained DiT-Based Video Backbone

CTRL-Adapter

[Figure 261]

[Figure 262]

Transformer Block 2

[Figure 263]

Adapter Block 2

[Figure 264]

Adapter Block 14

[Figure 265]

Adapter Block 28

[Figure 266]

…

…

…

zero conv

- Transformer Block 13

[Figure 267]

- Transformer Block 14

[Figure 268]

…

- Transformer Block 27

[Figure 269]

- Transformer Block 28

[Figure 270]

zero conv

zero conv

Transformer Block 1

F ✕ C ✕ H’ ✕ W’

F ✕ C ✕ H’ ✕ W’

[Figure 271]

Pretrained DiT-Based Video Backbone

CTRL-Adapter

[Figure 272]

[Figure 273]

…

…

Transformer Block 14

[Figure 274]

…

Transformer Block 28

[Figure 275]

Adapter Block 1

[Figure 276]

zero conv

Adapter Block 14

[Figure 277]

zero conv

Transformer Block 1

F ✕ C ✕ H’ ✕ W’

F ✕ C ✕ H’ ✕ W’

[Figure 278]

Pretrained DiT-Based Video Backbone

CTRL-Adapter

[Figure 279]

[Figure 280]

Adapter Block 15

[Figure 281]

Adapter Block 28

[Figure 282]

…

…

Transformer Block 15

[Figure 283]

…

Transformer Block 28

[Figure 284]

zero conv

zero conv

(a) Insert Adapters Interleaved (Default) (b) Insert into First 14 DiT Blocks (c) Insert into Last 14 DiT Blocks

- Figure 17: Visualization of different routing methods for combining multiple ControlNet outputs. We use (a) as our default setting, and show the settings (b) and (c) as ablations.

###### E.2 Adaptation to DiT-Based Backbones

As illustrated in Sec. 3, we have observed that the spatial features encoded in the U-Net of ControlNets and the DiT blocks are structurally different (see Fig. 22 for visualization of such observation). Therefore, mapping all middle/output blocks of ControlNet to DiT blocks might not be the optimal solution. In Fig. 17, we implement three different strategies to insert CTRL-Adapters to the DiT blocks. Specifically, variant (a) inserts CTRL-Adapters interleavingly into the DiT blocks, while variant (b) and (c) insert CTRL-Adapters to the first 14 and the last 14 DiT blocks respectively. In Table 5, we perform quantitative analysis of these three variants on the DiT-based video generation model, Latte [46], with soft edge as control condition. As we can see, inserting CTRL-Adapters interleavingly into the DiT blocks gives the best performance. This is consistent with our finding: since all DiT blocks encode global information of the generated objects, it is optimal to treat these blocks equally, rather than inserting CTRL-Adapters only at the beginning or end. Between locations A and B, we use location A as our default setting because its feature map size (64 × 64) directly matches the features of the DiT blocks (also 64 × 64) without resizing.

After finalizing where to insert CTRL-Adapters, the next question is which block(s) of the ControlNet we should create CTRL-Adapters to map from. In Table 6, we implement several variants on the DiT-based image generation model, PixArt-α [8], including mapping from the block(s) at location A, location B, location C, and location D, respectively (see Fig. 3 for the definitions of these locations). As we can see, mapping from location A or location B gives the best performance. Again, this is consistent with our findings in Fig. 22, since feature maps at locations C and D are too coarse to be informative. Moreover, we implemented two additional variants: (1) combining the ControlNet features from locations A and B (i.e., Output Blocks A+B), and (2) mapping more blocks from the same location (i.e., the second and third columns in Table 6). However, neither of these approaches provides sufficient gain compared to mapping a single block from location A or B. Therefore, we use mapping one block from location A as our default setting in our main paper.

- Table 5: Ablation of inserting CTRL-Adapters to different DiT blocks in Latte [46]. Visualization of the three architecture variants (interleaved, first half, and second half) are shown in Fig. 17. We use soft edge as control condition here for evaluation.

(a) Interleave (default) (b) First Half (c) Second Half FID (↓) Optical Flow Error (↓) FID (↓) Optical Flow Error (↓) FID (↓) Optical Flow Error (↓)

18.32 2.98 19.66 3.09 23.18 3.31

- Table 6: Ablation study of mapping ControlNet features from different locations, and mapping different number of blocks from the same location to the DiT blocks. The best numbers in each row are bolded, and the best numbers in each column are underscored.

Insert Locations

1 Block (default) 2 Blocks 3 Blocks FID (↓) SSIM (↓) FID (↓) SSIM (↓) FID (↓) SSIM (↓)

- Output Block A (default) 17.90 0.6802 19.08 0.6971 19.28 0.6855

- Output Block B 18.23 0.6712 18.61 0.6720 21.47 0.6549 Output Blocks A+B 17.52 0.6812 - - - -

- Output Block C 22.22 0.5273 - - - -
- Output Block D 34.16 0.3506 - - - -

E.3 Skipping Latent from ControlNet Inputs

- Table 7: Skipping latent from ControlNet inputs helps CTRL-Adapter for (1) adaptation to backbone models with different noise scales and (2) video control with sparse frame conditions. We evaluate SVD and I2VGen-XL on depth maps and scribbles as control conditions respectively.

Method Latent z is given to ControlNet FID (↓) Optical Flow Error (↓) Adaptation to different noise scales

SVD [3] + CTRL-Adapter ✔ 4.48 2.77 SVD [3] + CTRL-Adapter ✘ 3.82 2.96

Sparse frame conditions

I2VGen-XL [89] + CTRL-Adapter ✔ 7.20 5.13 I2VGen-XL [89] + CTRL-Adapter ✘ 5.98 4.88

As described in Sec. 3.2, we find skipping the latent z from ControlNet inputs can help CTRL-Adapter to more robustly handle (1) adaption to the backbone with noise scales different from SDv1.5, such as SVD and (2) video control with sparse frame conditions. For the first scenario, we can see from

- Table 7 that skipping latents in SVD leads to better visual quality (lower FID), but slightly worse spatial control (higher optical flow error). This is reasonable since skipping the noisy latents can avoid introducing large noise into the ControlNet, but it also risks losing information encoded in the latents. For the second scenario, skipping latents results in both better visual quality and better spatial control, as adding dense noisy latents can make the sparse control conditions less informative.

E.4 Different weighing modules for multi-condition generation

For multi-condition generation described in Sec. 3.3, in addition to the simple unconditional global weights, we also experimented with learning a router module that takes additional inputs such as diffusion time steps and image/text embeddings and outputs weights for different ControlNets. Specifically, we introduce three variants based on (a.1) unconditional global weights, which are (a.2) MLP router - taking timestep as inputs; (a.3) MLP router - taking image/text embedding as inputs; and (a.4) MLP router - taking timestep and image/text embedding as inputs. The MoE router in these variants are constructed as a 3-layer MLP. We illustrate the five methods in Fig. 18.

- Table 8 show that all four global weighting schemes for fusing different ControlNet outputs perform effectively, and no specific method outperforms other methods with significant margins in all settings. With no surprise, patch-level MoE router performs consistently better than global MoE router in all control settings. Testing the effectiveness of incorporating text/image/timestep embeddings to patch-level MoE routers are left for future work.

- Table 8: Comparison of global weighting methods for multi-condition video generation (see Fig. 18 for visualization of the additional weighting methods (a.2, a.3, and a.4) developed based on (a.1) unconditional global weights). The control sources are abbreviated as D (depth map), C (canny edge), N (surface normal), S (softedge), Seg (semantic segmentation map), L (line art), and P (human pose).

D+C D+P D+C+N+S D+C+N+S+Seg+L+P FID (↓) Flow Error (↓) FID (↓) Flow Error (↓) FID (↓) Flow Error (↓) FID (↓) Flow Error (↓)

Baseline Equal Weights 8.50 2.84 11.32 3.48 8.75 2.40 9.48 2.93

Global MoE Router

- (a.1) Unconditional Global Weights 9.14 2.89 10.98 3.32 8.39 2.36 8.18 2.48

- (a.2) Timestep Emb. Weights 9.41 3.51 11.13 3.35 9.51 2.78 8.17 2.45
- (a.3) Text/Image Emb. Weights 8.73 3.16 11.35 3.37 7.91 2.76 8.83 2.48
- (a.4) Timestep + Text/Image Emb. Weights 8.64 3.31 10.69 3.43 8.09 2.69 8.51 2.43 Patch-Level MoE Router

- (b) MLP Weights 8.40 2.34 9.37 3.17 7.87 2.11 8.26 2.00

- (c) Q-Former Weights 7.54 2.39 9.22 3.22 7.72 2.31 8.00 2.08

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Timestep Encoder

Text Encoder

Image Encoder

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Learnable Emb.

w1 = w2 = … = wN = 1/N

MLP Softmax

MLP Softmax

MLP Softmax

Softmax

w1, w2,…,wN

w1, w2,…,wN

w1, w2,…,wN

w1, w2,…,wN

Equal Weight

(a.1) Unconditional Global Weights

(a.2) Timestep Emb. Weights

(a.3) Text/Img Emb. Weights

(a.4) Timestep + Text/Img Emb. Weights

Figure 18: Visualization of different global MoE routing methods.

##### F Additional Quantitative Analysis

###### F.1 Train individual CTRL-Adapter v.s. train a unified CTRL-Adapter

In our main paper, we train CTRL-Adapter for each control conditions. An interesting question to ask is: can we have a single and unified CTRL-Adapter that works for all control conditions? In this part, we perform such analysis with SDXL as the backbone model on a new training strategy. Specifically, during each training step, we randomly choose one control condition from a pool of 8 control conditions, including depth map, canny edge, soft edge, normal map, semantic segmentation, line art, user scribbles, and human pose. We train this variant with the same hyper-parameter settings as the depth map or canny edge CTRL-Adapter mentioned in our main paper.

As shown in Table 9, training a unified CTRL-Adapter across all control conditions does suffer from a performance decrease. However, this decrease is not very significant. Therefore, from a practical point of view, if a user has computational constraints but still needs to work on multiple control conditions, training a unified CTRL-Adapter can be a viable workaround.

###### F.2 Trade-off between Visual Quality and Spatial Control

In Fig. 19, Fig. 20, and Fig. 21, we show the visual quality (FID) and spatial control (SSIM/optical flow error) metrics with different numbers of denoising steps with spatial control (with the fusion of CTRL-Adapter outputs) on SDXL, SVD, and I2VGen-XL backbones respectively. Specifically, suppose we use N denoising steps during inference, a control guidance level of x% means that we fuse CTRL-Adapter features to the video diffusion U-Net during the first x% × N denoising steps, followed by (100 − x)% × N regular denoising steps. In all experiments, we find that increasing the number of denoising steps with spatial control improves the spatial control accuracies (SSIM/optical flow error) but hurts visual quality (FID).

- Table 9: Ablation of training a unified CTRL-Adapter v.s. training an individual CTRL-Adapter for each condition. The results are evaluated with SDXL as the backbone model.

Depth Map Canny Edge FID (↓) SSIM (↑) FID (↓) SSIM (↑)

Training Strategy

Individual CTRL-Adapter (default) 19.26 0.8534 21.04 0.5806 Unified CTRL-Adapter 19.95 0.8437 22.31 0.5684

[Figure 295]

[Figure 296]

- Figure 19: Trade-off between generated visual quality (FID) and spatial control accuracy (SSIM) on SDXL. Control guidance level of x represents that we apply CTRL-Adapter in the first x% of the denoising steps during inference. A control guidance level between 30% and 60% usually achieves the best balance between image quality and spatial control accuracy.

[Figure 297]

[Figure 298]

- Figure 20: Trade-off between generated visual quality (FID) and spatial control accuracy (Optical Flow Error) on SVD. Control guidance level of x represents that we apply CTRL-Adapter in the first x% of the denoising steps during inference. A control guidance level between 40% and 60% usually achieves the best balance between image quality and spatial control accuracy

20 40 60 80 100 Control Guidance Level (%)

- 7

- 8

- 9

- 10

FID()

3.2

3.3

OpticalFlowError()

Depth

20 40 60 80 100 Control Guidance Level (%)

- 6

- 7

- 8

FID()

3.4

3.5

3.6

OpticalFlowError()

Canny

- Figure 21: Trade-off between generated visual quality (FID) and spatial control accuracy (Optical Flow Error) on I2VGen-XL. Control guidance level of x represents that we apply CTRL-Adapter in the first x% of the denoising steps during inference. A control guidance level between 40% and 60% usually achieves the best balance between image quality and spatial control accuracy.

##### G Additional Qualitative Analysis

###### G.1 Visualization of Spatial Feature Maps

As mentioned in Sec. 3 and Appendix E.2, the spatial features encoded in the U-Net of ControlNets and the DiT blocks are structurally different. We visualize this difference in Fig. 22. For the DiT-based model, we use PixArt-α as a representative. We follow the visualization method mentioned in [72]. Specifically, we first extract the spatial features from different DiT blocks and U-Net middle/output blocks at the last denoising step during inference. For each block, we applied PCA to the extracted features and visualized the top three leading components.

As shown in Fig. 22, almost all 28 DiT blocks capture global and semantic information about the object "cactus". This observation is consistent with the findings in [20]. On the other hand, the U-Net blocks in ControlNet demonstrate a coarse-to-fine pattern as the feature map size increases. This indicates that mapping output blocks A/B of ControlNet to DiT blocks is a better option compared to using middle or output blocks C/D of the ControlNet.

###### Feature Maps Visualization In Each Transformer Block

Block 1 Block 2 Block 3 Block 4 Block 5 Block 6 Block 7

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Block 8 Block 9 Block 10 Block 11 Block 12 Block 13 Block 14

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

Output Image

Input Prompt

[Figure 313]

A small cactus with a happy face in the Sahara desert

Block 15 Block 16 Block 17 Block 18 Block 19 Block 20 Block 21

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Block 22 Block 23 Block 24 Block 25 Block 26 Block 27 Block 28

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

###### Input Prompt

###### Feature Maps Visualization In Each ControlNet U-Net Block

Mid Block

Output Blocks D

Output Blocks C

A small cactus with a happy face in the Sahara desert

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

Output Image

[Figure 335]

Canny Edge

Output Blocks B Output Blocks A

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

- Figure 22: Visualization of spatial feature maps in PixArt-α and canny edge ControlNet. We first extract the spatial features from different DiT blocks and U-Net middle/output blocks at the last denoising step during inference. For each block, we applied PCA to the extracted features and visualized the top three leading components. Almost all 28 DiT blocks capture global and semantic information about the object "cactus", while the U-Net blocks in ControlNet demonstrate a coarse-tofine pattern as the feature map size increases.

###### G.2 Fast Training Convergence

In addition to the quantitative results shown in Fig. 2, we provide a more straightforward visualization for SDXL depth ControlNet + CTRL-Adapter training. The training speed test is performed on 4 A100 80GB GPUs, with a batch size of 1 per GPU. As shown in Fig. 23, for relatively easy examples (i.e., bedroom, sandwich, bus), our CTRL-Adapter training can converge within 4.5 GPU hours (which is equivalent to around 1.125 hours measured in training clock time). For complex examples and those requiring fine details (i.e., surfing man, group of kids), our CTRL-Adapter can also converge within around 6 to 7.5 GPU hours (which is equivalent to 1.5 to 1.875 hours measured in training clock time), which proves the training efficiency of our CTRL-Adapter.

000000005037

000000000632 000000001425 000000001000

000000002157 000000004134 000000005060 000000004765

Train GPU Hours

Input Depth Map

Input Prompt

1.5 h 3.0 h 4.5 h 6.0 h 7.5 h

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

A bed room with a neatly made bed a window and a bookshelf

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

A bus stops to pick up some passengers

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

A man surﬁng on a large wave in the ocean

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

A group of kids posing for a picture on a tennis court

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

A sandwich with a bite taken on a plate

###### Figure 23: Training efficiency of CTRL-Adapter on SDXL backbone. Total training GPU hours are measured on 4 A100 80GB GPUs, with batch size per GPU equal to 1.

- H Additional Visualization Examples We provide more qualitative examples in this section.

###### H.1 Video Generation Visualization Examples

- In Fig. 24, we show video generation results on COCO val2017 split using depth map and canny edge as control conditions. We visualize baseline methods as well as CTRL-Adapters built on top of Hotshot-XL [49], SVD [3], I2VGen-XL [89], and Latte [46].
- In Fig. 25 and Fig. 26, we show video generation results with I2VGen-XL using depth map and canny edge extracted from videos from Sora10 and the internet.

In Fig. 27, we show video generation results with Latte using soft edge extracted from videos from Sora11 and the internet.

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

Input Image/ Prompt Prompt:Hike

Prompt: Camel

Prompt: Snowboard

Prompt: Ice hockey

Depth Map Canny Edge

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

###### Input Control Conditions

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

ControlVideo

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

Control-A-Video

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

Text2Video-Zero

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Hotshot-XL + CTRL-Adapter

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

###### Latte + CTRL-Adapter

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

SVD + CTRL-Adapter

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

I2VGen-XL + CTRL-Adapter

- Figure 24: Videos generated from different video control methods and CTRL-Adapter on DAVIS 2017, using depth map (left) and canny edge (right) conditions.

- 10https://openai.com/sora
- 11https://openai.com/sora

Image/ Prompt

A goldfish swimming

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Input

Condition

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

I2VGen-XL +

CTRL-Adapter

[Figure 454]

Input Image/ Prompt

This close-up shot of a Victoria crowned pigeon showcases its striking blue plumage and red chest. Its crest is made of delicate, lacy feathers, while its eye is a striking red color. The bird's head is tilted slightly to the side, giving the impression of it looking regal and majestic. The background is blurred, drawing attention to the bird's striking appearance.

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

Input Condition

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

I2VGen-XL + CTRL-Adapter

[Figure 467]

A white and orange tabby cat is seen happily darting through a dense garden, as if chasing something. Its eyes are wide and happy as it jogs forward, scanning the branches, flowers, and leaves as it walks. The path is narrow as it makes its way between all the plants. the scene is captured from a groundlevel angle, following the cat closely, giving a low and intimate perspective. The image is cinematic with warm tones and a grainy texture. The scattered daylight between the leaves and plants above creates a warm contrast, accentuating the cat's orange fur. The shot is clear and sharp, with a shallow depth of field.

Input Image/ Prompt

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Input Condition

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

I2VGen-XL + CTRL-Adapter

[Figure 480]

Input

A miniature Christmas village with snow-covered houses, glowing windows, decorated trees, festive snowmen, and tiny figurines in a quaint, holiday-themed diorama evoking a cozy, celebratory winter atmosphere

Image/ Prompt

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

Input Condition

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

I2VGen-XL + CTRL-Adapter

- Figure 25: Video generation with I2VGen-XL + CTRL-Adapter using depth map as a control

This close-up shot of a chameleon showcases its striking color changing capabilities. The background is blurred, drawing attention to the animal's striking appearance.

Image/ Prompt

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

Input

Condition

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

I2VGen-XL +

CTRL-Adapter

[Figure 506]

Input Image/ Prompt

A bird flying over a forest.

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

Input Condition

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

I2VGen-XL + CTRL-Adapter

[Figure 519]

Input Image/ Prompt

Reflections in the window of a train traveling through the Tokyo suburbs.

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

Input Condition

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

I2VGen-XL + CTRL-Adapter

[Figure 532]

Input

A woman wearing blue jeans and a white t-shirt taking a pleasant stroll in Mumbai India during a beautiful sunset

Image/ Prompt

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

Input Condition

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

I2VGen-XL + CTRL-Adapter

- Figure 26: Video generation with I2VGen-XL + CTRL-Adapter using canny edge as control

This close-up shot of a futuristic cybernetic german shepherd showcases its striking brown and black fur. its chest and head have robotic modifications while its eye is a striking black color with futuristic digital altercations. the dogs head is tilted slightly to the side, giving the impression of it looking regal and majestic. the neon background is blurred, drawing attention to the dogs striking appearance

Input Prompt

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

Input Condition

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

Latte + CTRL-Adapter

Input Prompt

A giant, towering cloud in the shape of a man looms over the earth. The cloud man shoots lighting bolts down to the earth

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

Input Condition

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

Latte + CTRL-Adapter

Input Prompt

In an ornate, historical hall, a massive tidal wave peaks and begins to crash. Two surfers, seizing the moment, skillfully navigate the face of the wave

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

Input Condition

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

Latte + CTRL-Adapter

Input Prompt

A 2d abstract japanese animation where drops of ink in water form into lifelike creatures that swim and interact with each other, creating an ethereal underwater world made entirely of flowing, merging colors

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

Input Condition

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

Latte + CTRL-Adapter

Input Prompt

A Shiba Inu dog wearing a beret and black turtleneck

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

Input Condition

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

Latte + CTRL-Adapter

- Figure 27: Video generation with Latte + CTRL-Adapter using soft edge as control condition.

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

32

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

###### H.2 Multi-Condition Video Generation Visualization Examples

Fig. 28 shows example videos generated with single and multiple conditions. While all videos correctly capture the high-level dynamics of ‘a woman wearing purple strolling during sunset’, the videos generated with more conditions show more robustness in several minor artifacts. For example, when only depth map is given (Fig. 28 a), the building behind the person is blurred. When depth map and human pose are given (Fig. 28 b), the color of the purse changes from white to purple. When four conditions (depth map, canny edge, human pose, and semantic segmentation) are given, such artifacts are removed (Fig. 28 c).

- In Fig. 29, we show multi-condition control examples with I2VGen-XL.

###### Input Image/Prompt

###### Input Conditions: Depth Map, Human Pose, Canny Edge, Segmentation Mask

[Figure 617]

Prompt: A woman wearing purple overalls and cowboy boots taking a pleasant stroll in Johannesburg South Africa during a beautiful sunset

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

(c) Depth Map + Human Pose + Canny Edge + (a) Depth Map (b) Depth Map + Human Pose Semantic Segmentation

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

- Figure 28: Video generation from single and multiple conditions with CTRL-Adapter on I2VGen-XL. (a) single condition: depth map; (b) 2 conditions: depth map + human pose; (c) 4 conditions: depth map + human pose + canny edge + semantic segmentation. Adding more conditions can help fix several minor artifacts (e.g., in (a) – building is blurred; in (b) – purse color changes). Aaboveskateboarderabench,mid-trick,wearsacasualairborne

outfit and a beanie, displaying focus and athletic skill

[Figure 642]

I2VGen-XL + CTRL-Adapter

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

Prompt: A small child and an adult standing in shallow ocean waters along the beach

[Figure 656]

[Figure 657]

[Figure 658]

Input Image/ Prompt A manPrompt:dancing

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

Input Conditions

Depth Map

Human Pose

Canny Edge

Semantic Segmentation

- Figure 29: Video generated with I2VGen-XL + CTRL-Adapter from 4 control conditions: depth map + human pose + canny edge + semantic segmentation map.

33

###### H.3 Image Generation Visualization Examples

- In Fig. 30 and Fig. 31, we show image generation results on COCO val2017 split using depth map and canny edge as control conditions.

In Fig. 32, Fig. 33, and Fig. 34, we show image generation results on prompts from Lexica12 using depth map, canny edge, and soft edge as control conditions.

SDXL ControlNet

CTRL-Adapter (Ours)

Input Prompt

###### Input Depth Map

X-Adapter

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

A police officer on a motorcycle drives through a parade.

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

A middle aged black woman is standing behind a table full of bananas.

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

A pair of children stand on a fence together

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

A man sitting in a restaurant photographs a sandwich.

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

A man in a reception hall holds a drink

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

A woman is skiing down a snowy hill

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

A vegetable

vendor

organizing his food for sale

- Figure 30: Image generation from different SDXL-based image control methods and CTRL-Adapter on COCO val2017 split using depth map as control condition.

12https://lexica.art/

SDXL

Input Prompt

###### Input Depth Map

###### CTRL-Adapter (Ours)

###### X-Adapter

###### ControlNet

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

A man in a wet suit is surfing.

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

A woman tennis player getting ready to be served the ball

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

Baseball player

number 22

holding batting gloves and wearing helmet

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

A person who is on their motorcycle in the air

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

A vase full of irises with a

pitcher on an end

table

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

Flowers are arranged in a vase sitting on a table

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

A dog sits on top of a bed in a

room

- Figure 31: Image generation from different SDXL-based image control methods and CTRL-Adapter on COCO val2017 split using canny edge as control condition.

a cute, happy hedgehog taking a bite from a piece of watermelon, eyes closed, cute ink sketch style illustration

happy Hulk standing in a beautiful field of flowers, colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, Kodak Portra 400, film grain

a cute mouse pilot wearing

A cute sheep with rainbow fur, photo

aviator goggles, unreal engine render, 8k

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

- Figure 32: Image generation with SDXL + CTRL-Adapter using depth map as a control condition.

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

Astronaut walking on water

Cute fluffy corgi dog in the city in anime style

Cute lady frog in dress and

crown dressed in gown in

cinematic environment

Cute and super adorable mouse in black and red chef coat and chef hat, holding a steaming entree.

- Figure 33: Image generation with SDXL + CTRL-Adapter using canny edge as a control condition.

Darth Vader in a beautiful field of flowers, colorful flowers everywhere, perfect lighting

A plate of cheesecake, pink flowers everywhere, cinematic lighting, food photography

A micro-tiny clay pot full of dirt with a beautiful daisy planted in it, shining in the autumn sun

A raccoon family having a nice meal, life-like

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

- Figure 34: Image generation with PixArt-α + CTRL-Adapter using soft edge as a control condition.

###### H.4 Visualization Examples for Additional Downstream Tasks

Here, we describe in detail how our CTRL-Adapter can be seamlessly integrated into a wide variety of downstream tasks including video editing, video style transfer, and text-guided motion control, as mentioned in Sec. 6. Additional visualizations are also shown in this part.

Video editing. Video editing can be achieved by combining image and video CTRL-Adapters. The procedure is as follows:

- • Firstly, given a source video, we first extract the control condition(s). We can either extract a single control condition (e.g., depth map), or multiple control conditions (e.g., depth map, canny edge, segmentation, etc.) to enhance performance (as we observe in Appendix H.2 that multi-condition control improves spatial control accuracy) .
- • Next, given a user-provided prompt together with the extracted control condition(s), we can use image CTRL-Adapter (i.e., SDXL + CTRL-Adapter) to generate the first frame of the video.
- • Finally, we can use video CTRL-Adapter (i.e., I2VGen-XL + CTRL-Adapter), with the generated first frame image, text prompt, and extracted control conditions as inputs for final video generation.

- In Fig. 35, we provide additional visualizations of the camel example in our main paper.

I2VGen-XL + CTRL-Adapter Output

SDXL + CTRL-Adapter Output

###### Prompt

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

A zebra stripped camel walking

Source Video

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

A camel with rainbow fur walking

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

A camel walking, van gogh-style

Extracted Control Condition

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

A camel walking, ink sketch style

- Figure 35: Video editing by combining SDXL and I2VGen-XL, where both models are equipped with spatial control via CTRL-Adapter. First, we extract conditions (e.g., depth map) from the original video. Next, we create the initial frame with SDXL + CTRL-Adapter. Lastly, we provide the newly generated initial frame and frame-wise conditions to I2VGen-XL + CTRL-Adapter to obtain the final edited video. This video editing framework can edit both object and background.

Text-Guided Motion Control. This task can be achieved by combining video CTRL-Adapter with inpainting ControlNet. We train such CTRL-Adapter as follows:

- • Firstly, for each training video, we randomly select a random block in the image, with the width and height of the block uniformly sampled from 0.25 to 0.75 of the image size.
- • Next, we color the block area of the video frames as black color (these processed frames can be regarded as control condition sequences like depth maps).
- • Finally, we can train CTRL-Adapter with the frozen inpainting ControlNet similar as other types of CTRL-Adapters mentioned in our main paper.

- In Fig. 36, we provide additional visualizations of text-guided image amination.

###### Prompt I2VGen-XL + CTRL-Adapter

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

A white and orange tabby cat is darting through a dense garden, as if chasing something

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

A medium sized friendly looking dog walks through an industrial parking lot

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

An elk with impressive antlers grazing on the snow-covered ground

- Figure 36: Text-guided motion control by combining inpainting ControlNet with I2VGen-XL + CTRL-Adapter. Specifically, inpainting ControlNet takes the masked frames as well as text prompt as inputs. The output feature maps of inpainting ControlNet are then given to CTRL-Adapter built on top of I2VGen-XL to generate the final video. Object(s) in the masked area can follow the motion described in the text prompt. The unmasked area can be either static or dynamic.

[Figure 792]

[Figure 793]

[Figure 794]

The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope

[Figure 795]

[Figure 796]

Video style transfer. This task can be achieved by combining video CTRL-Adapter with shuffle ControlNet. We train such CTRL-Adapter as follows:

- • Firstly, for the first frame of each training video, we apply the content shuffle detector implemented in the controlnet_aux13 library, to get a shuffled image.
- • Next, we repeat this shuffled image N times, with N equal to the number of output frames of the backbone video diffusion model. These repeated images can be regarded as control condition sequences like depth maps.
- • Finally, we can train CTRL-Adapter with the frozen shuffle ControlNet similar as other types of CTRL-Adapters mentioned in our main paper.

in Fig. 37, we provide additional visualizations of video style transfer.

###### Prompt Output Video

###### Shuffle

[Figure 797]

[Figure 798]

[Figure 799]

A miniature Christmas village with snow-covered houses, glowing windows, decorated trees, festive snowmen, and tiny ﬁgurines in a quaint, holiday-themed diorama evoking a cozy, celebratory winter atmosphere

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

Stop motion of a colorful paper ﬂower blooming

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

In an ornate, historical hall, a massive tidal wave peaks and begins to crash

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

A meticulously crafted diorama depicting a serene scene from Edo-period Japan

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

Beautiful, snowy Tokyo city is bustling

- Figure 37: Video style transfer by combining shuffle ControlNet with Latte + CTRL-Adapter. Specifically, shuffle ControlNet takes the shuffled image as well as text prompt as inputs. The output feature maps of shuffle ControlNet are then given to CTRL-Adapter built on top of Latte to generate the final video. The generated videos maintain similar style as the input image before shuffling.

13https://github.com/huggingface/controlnet_aux

##### I Broader Impacts

CTRL-Adapter is motivated by the fact that training ControlNets for new diffusion models, especially video diffusion models that need to consider temporal consistency, can be a huge burden for many users. As shown in Fig. 2, by adopting pretrained ControlNets, training CTRL-Adapter can be significantly faster than training other controllable image or video generation methods. For example, with the same type of compute (i.e., A100 80GB GPUs), CTRL-Adapter trained on SDXL depth ControlNet for 10 GPU hours can outperform SDXL ControlNet trained for 700 GPU hours. This drastically reduces the carbon emissions footprint by over 70 times. Therefore, we believe that our work can be a strong contribution to efficient and controllable image and video generation.

While our framework can benefit numerous applications in controllable generation, similar to other image and video generation frameworks, it can also be used for potentially harmful purposes (e.g., creating false information or misleading videos). Therefore, it should be used with caution in real-world applications.

##### J Limitations

Note that since CTRL-Adapter is a method to equip current open-source image and video diffusion models with better control, its performance, quality, and potential visual artifacts largely depend on the capabilities (e.g., motion styles and video length) of the backbone models used. For example, if a diffusion model cannot handle complex motions, CTRL-Adapter built on top of this backbone might lead to non-optimal complex motion control.

##### K License

We use standard licenses from the community and provide the following links to the licenses for the datasets, codes, and models that we used in this paper. For further information, please refer to the specific link.

PyTorch [1]: BSD-style HuggingFace Transformers [79]: Apache License 2.0 HuggingFace Diffusers [73]: Apache License 2.0 ControlNet [86]: Apache License 2.0 SDXL [52]: MiT License PixArt-α [8]: AGPL-3.0 License I2VGen-XL [89]: MiT License Stable Video Diffusion (SVD) [3]: MiT License Latte [46]: Apache License 2.0 Hotshot-XL [49]: Apache License 2.0 LAION dataset [67]: MiT License Panda70M dataset [9]: License COCO dataset [42]: CC BY 4.0 DAVIS 2017 dataset [53]: CC BY 4.0

