## MotionPro: A Precise Motion Controller for Image-to-Video Generation*

# arXiv:2505.20287v1[cs.CV]26May2025

Zhongwei Zhang1, Fuchen Long2, Zhaofan Qiu2, Yingwei Pan2†, Wu Liu1†, Ting Yao2, and Tao Mei2 1University of Science and Technology of China 2HiDream.ai Inc.

zhwzhang@mail.ustc.edu.cn, {longfuchen, qiuzhaofan, pandy}@hidream.ai liuwu@live.cn, {tiyao, tmei}@hidream.ai

#### Abstract

Animating images with interactive motion control has garnered popularity for image-to-video (I2V) generation. Modern approaches typically rely on large Gaussian kernels to extend motion trajectories as condition without explicitly defining movement region, leading to coarse motion control and failing to disentangle object and camera moving. To alleviate these, we present MotionPro, a precise motion controller that novelly leverages region-wise trajectory and motion mask to regulate fine-grained motion synthesis and identify target motion category (i.e., object or camera moving), respectively. Technically, MotionPro first estimates the flow maps on each training video via a tracking model, and then samples the region-wise trajectories to simulate inference scenario. Instead of extending flow through large Gaussian kernels, our region-wise trajectory approach enables more precise control by directly utilizing trajectories within local regions, thereby effectively characterizing finegrained movements. A motion mask is simultaneously derived from the predicted flow maps to capture the holistic motion dynamics of the movement regions. To pursue natural motion control, MotionPro further strengthens video denoising by incorporating both region-wise trajectories and motion mask through feature modulation. More remarkably, we meticulously construct a benchmark, i.e., MC-Bench, with 1.1K user-annotated image-trajectory pairs, for the evaluation of both fine-grained and object-level I2V motion control. Extensive experiments conducted on WebVid-10M and MC-Bench demonstrate the effectiveness of MotionPro. Please refer to our project page for more results: https: //zhw-zhang.github.io/MotionPro-page/.

#### 1. Introduction

In recent years, diffusion models [9, 12, 16–18, 20–22, 28, 29, 37, 38, 43, 46, 47, 54, 62] have shown significant

*This work was performed at HiDream.ai. †Co-corresponding author.

Input Control Generated Video

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Typicality

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

MotionPro

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

- (a) Fine‐grained motion control
- (b) Object‐level motion control

|[Figure 51]|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

TypicalityMotionPro

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

[Figure 87]

|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]|
|---|

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

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

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

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Figure 1. An illustration of (a) fine-grained and (b) object-level motion control by using typical Gaussian filtered trajectory and our MotionPro. The flow of generated videos are also visualized.

progress in revolutionizing text-to-video (T2V) generation. Although promising visual appearance can be attained by these advances, the controllable motion generation is still a grand challenge in video diffusion paradigm. There are several attempts [13, 27, 49, 61] to enhance controllable capacity of video synthesis with additional guidance (e.g., depth, edge or optical flow). Nevertheless, it might be impractical for users to conveniently provide such signals as input conditions. Hence, the focus of this paper is to capitalize on the user-friendly conditions (i.e., sparse trajectory and region mask) for enabling interactively controllable imageto-video (I2V) generation: given the reference image as the first frame, the motion in the synthesized video should be natural and well-aligned with the provided trajectory.

Pioneering practices [52, 57] of controllable I2V generation usually guide video denoising with the single condition of Gaussian filtered trajectory. In the training stage, the input trajectories are first sparsely sampled from the optical flow maps and then processed by a Gaussian filter with large kernel to mitigate pixel-level trajectory instability (e.g., 99 in DragNUWA [57]). The flow extension brought by Gaussian filtering inevitably results in the inaccuracy

of fine-grained motion details and limits the model capability for precise motion control. Therefore, the generated fine-grained movement (e.g., the turning-head of first case in Figure 1) is unnatural. Another issue is that the single condition of trajectory commonly fails to precisely identify the target motion category (i.e., object or camera moving). For instance, as depicted in Figure 1, the trajectory on the planet could be explained as two moving situations, i.e., the camera being pulled downwards with relative to static two planets (camera movement) or planet rising corresponding to static background (object movement). Solely relying on the trajectory might lead to the motion misinterpretation and thus hinder exactly controllable I2V generation. To address the above two issues, we shape a new paradigm of motion controller that capitalizes on region-level trajectory and motion mask to enhance video denoising for controllable I2V synthesis. Specifically, we spatially sample multiple local regions in the video optical flow maps and directly employ the trajectories in the sparse regions as input condition. In this way, the attained trajectories could maintain accurate motion details and enables adequate capture of fine-grained motion. Meanwhile, a motion mask is estimated on the optical flow maps which aims to globally emphasize the motion area, thereby specifying the target motion category and alleviating misinterpretation. To further regulate motion synthesis, we predict the affine parameters on the collaboration of trajectory and motion mask to modulate the video latent codes in denoising. As shown in Figure 1, our unique region-wise trajectory design and the employment of motion mask achieves the better fine-grained (e.g., turninghead) and object-level (e.g., planet-moving) motion.

By materializing the idea of facilitating controllable I2V generation with the proposed conditions, we present a novel framework, namely MotionPro, a precise region-wise motion control. Specifically, given the input video, MotionPro first estimates the sequence of visibility masks and optical flow maps by using an off-the-shelf optical tracking model. Next, the global visibility mask is obtained through computing the intersection of all visibility masks, and further multiplied with the flow map of each frame. Then, MotionPro splits the masked flow maps into multiple local regions (e.g., the region with the size of 8 × 8) and employs the trajectories on such sparsely-sampled regions as region-wise trajectory. Meanwhile, MotionPro attains the motion mask on the flow maps via thresholding mechanism for representing holistic motion. Given the region-wise trajectory and corresponding motion mask, the multi-scale features are learnt by a motion encoder, and further employed to predict scale and bias for video latent feature modulation. Moreover, MotionPro fine-tunes all attention modules in 3D-UNet via utilizing the Low-Rank Adaptation (LoRA) technique to pursue better motion-trajectory alignment.

The main contribution of this work is design of Motion-

Pro by leveraging region-wise trajectory and motion mask as the complementary control signals for precise controllable I2V diffusion. Beyond this, one benchmark, i.e., MCBench, with 1.1K user-annotated image-trajectory pairs, is carefully collected for evaluation. Extensive experiments further verify the superiority of MotionPro in terms of both video quality and motion-trajectory alignment.

#### 2. Related Work

Image-to-Video Diffusion Models. The remarkable progress achieved by text-to-video generation [2, 3, 6, 7, 14, 17, 18, 20–22, 27, 30, 32, 43, 47, 56] encourages the development of image-to-video (I2V) diffusion models. These advances [5, 11, 15, 41, 49, 53, 58, 60] treat static image as the input condition for temporal coherent video synthesis. VideoComposer [49] is one of the earlier works that integrates image condition into 3D-UNet through concatenating the clean image latent with the noisy video latents. Based on this recipe, DynamiCrafter [53] and SVD [5] additionally inject the CLIP [39] feature of reference image into video denoising to enhance the information guidance. To achieve high-resolution I2V generation, I2VGen-XL [60] introduces a cascading diffusion model to first animate image in the low resolution and further magnifies it via video refinement. Besides, there are several explorations [11, 58] that simultaneously utilize two images (i.e., the first and last frames) as more powerful references to elevate I2V generation. In this work, we choose the pre-trained I2V diffusion model SVD [5] as our base architecture for motion control.

Controllable Video Diffusion Models. Despite highquality video synthesis via I2V diffusion models, the controllable motion generation still remains an under-explored problem. The early controllable video diffusion techniques [8, 10, 13, 27, 49, 61] typically leverage the condition of depth, edge or optical flow, for particular motion generation. Nevertheless, it is usually impractical for users to conveniently obtain such kinds of signals. To address this issue, the studies exploring bounding box [24, 31, 48, 51, 55] or trajectory [34, 35, 50, 57] as additional condition for motion control start to emerge. In the direction of utilizing trajectory condition, DragNUWA [57] and DragAnything [52] exploits Gaussian filtered trajectory to regulate motion synthesis to mitigate pixel-level trajectory instability. However, this approach limits fine-grained motion controllability and often fails to disentangle object and camera motions when relying solely on trajectories as the conditional input. Recently, Motion-I2V [41] and MOFA-Video [35] proposed a two-stage motion control framework, which first densifies input trajectories through a sparse-to-dense network and subsequently regulates video denoising using the estimated dense trajectories. Similarly, MOFA-Video also introduces the concept of a movement region mask. However, the mask is solely employed for

Iterative Denoising Repeat

Training Pipeline

[Figure 136]

[Figure 137]

Input video

[Figure 138]

First Frame

[Figure 139]

Input Video

Generated Video

[Figure 140]

|[Figure 141]| |
|---|---|
| | |

|[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]|Region|
|---|---|
| |Sampling|

[Figure 169]

[Figure 170]

|[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]|
|---|

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

|VAE Enc.|Add Noise|
|---|---|
| | |

[Figure 186]

[Figure 187]

SVD

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

VAE Dec.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

*

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

c

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

|[Figure 253]|[Figure 254]<br><br>[Figure 255]| | |
|---|---|---|---|
| | | | |

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

| | | |
|---|---|---|
| | | |
| | | |

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

Dense Optical Thresholding Tracking

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

Input video

Region‐wise Trajectory Motion Mask

[Figure 303]

Spatial‐Temporal Transformer with LoRA

[Figure 304]

|[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]|[Figure 317]<br><br>[Figure 318]|
|---|---|

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

TrainingInference

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

Inference Pipeline

[Figure 332]

[Figure 333]

|[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]|
|---|

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

c

[Figure 349]

User

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

Padding

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

|SelfAttn.|
|---|

|LoRA| |
|---|---|
| | |

|CrossAttn.|
|---|

|TempAttn.|
|---|

|LoRA| |
|---|---|
| | |

|CrossAttn.|
|---|

|LoRA| |
|---|---|
| | |

[Figure 359]

|LoRA|
|---|

[Figure 360]

[Figure 361]

TempAttn.

[Figure 362]

CrossAttn.

LoRA CrossAttn.

[Figure 363]

[Figure 364]

SelfAttn.

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

LoRA

LoRA

LoRA

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Motion Encoder

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

Upload

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

Image & Draw

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

|[Figure 397]<br><br>[Figure 398]<br><br>[Figure 399]<br><br>[Figure 400]<br><br>[Figure 401]<br><br>[Figure 402]<br><br>[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]|
|---|

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

User Draw

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

c

[Figure 444]

[Figure 445]

Graphical User Interface

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

Trainable Module Channel‐wise Concatenation

LoRA Block

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Mask Repeat

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

Spatial‐Temporal Transformer

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

Motion Condition Generation

Element‐wise Multiplication

Spatial‐Temporal Convolution

Motion Condition Generation

Figure 2. An overview of (a) our MotionPro for controllable I2V generation and (b) pipeline of motion condition generation. During training, MotionPro first extracts the proposed region-wise trajectory and motion mask on the input video as the control signals. The multiscale features are then learnt on these signals by a motion encoder, and further injected into the 3D-UNet of SVD in a feature modulation manner. Meanwhile, LoRA layers are integrated into all attention modules in the transformer blocks to improve the optimization of motiontrajectory alignment. In the inference stage, the region-wise trajectory and motion mask are first derived from the user provided trajectory and brushed region, and then exploited as the guidance to calibrate I2V video generation.

flow masking as a post-processing step, rather than being integrated as a conditional input for motion-region-aware video generation. This limitation can occasionally lead to unnatural motions and localized distortions in the synthesized videos. Instead, our work focuses on a new recipe of leverages region-wise trajectories and motion masks for precisely controllable I2V generation. The proposal of MotionPro contributes by studying not only how to express the motion trajectory accurately, but also how to benefit natural and precise motion generation with the synergy of the region-wise trajectory and motion mask.

#### 3. Our Approach

Here, we introduce our MotionPro framework for controllable I2V generation. Figure 2 illustrates an overview of our architecture. Given a video clip at training, the newlyminted region-wise trajectory and motion mask are first extracted as the control signals. Next, multi-scale features are learnt on the concatenation of the trajectory and mask via a motion encoder. These features are further injected into the 3D-UNet of SVD [5] to regulate video denoising. In each feature scale of the 3D-UNet, a scale and bias are predicted through convolutional layers to modulate the feature of video latent codes. Besides, all attention modules are fine-tuned by LoRA [23] to attain better alignment between the synthesized motion and input trajectory.

##### 3.1. Preliminaries: Stable Video Diffusion

To leverage comprehensive motion prior embedded in the pre-trained diffusion models for video generation, we exploit the advanced I2V generation model, i.e., Stable Video Diffusion (SVD) [5] as the base architecture of our MotionPro. To better understand our proposal, we first revisit the training procedure of SVD. Formally, given an input video

clip x0 = {xi0}Li=1 with L frames, the clean video latent codes z0 = {z0i}Li=1 are first extracted via a variational auto-encoder (VAE). Then, the Gaussian noise n is added to z0 through forward diffusion procedure as:

z = z0 + n, (σ,n) ∼ p(σ,n), (1)

where z is the noised video latent codes and p(σ,n) = p(σ)N(0,σ2I). σ represents the noise level and p(σ) is the pre-determined distribution over σ. Following the training protocol of EDM [26], SVD leverages the 3D-UNet Fθ (with parameters θ) to predict the clean video latent codes zˆ0 with the condition of input noised latents z, noise level σ and the reference image cI:

###### zˆ0 = cskip(σ)z + cout(σ)Fθ(cin(σ)z, cI; cnoise(σ)), (2)

where cskip(σ), cout(σ), cin(σ) and cnoise(σ) are pre-defined hyper-parameters determined by noise level σ. In SVD, the information of reference frame is injected into 3D-UNet along two pathways: a) the channel-wise concatenation of noised video latent codes and first frame latent code; b) the cross-attention between video latent feature and image CLIP [39] embedding of first frame. The loss function is formulated via denoising score matching (DSM) as:

0,cI,(σ,n)∼p(σ,n) λσ∥zˆ0 − z0∥22 , (3)

L = Ez

where λσ is a weighting function. In the scenario of our work, besides the condition of reference first frame, we additionally exploit a new kind of region-wise trajectory and motion mask as the control signals to refine video denoising for motion control.

##### 3.2. Motion Condition Generation

Most existing controllable I2V approaches calibrate the video denoising with the sole guidance of Gaussian fil-

tered point-wise trajectory. Nevertheless, the flow extending brought by Gaussian filtering may result in inaccuracy of fine-grained motion details. Therefore, the ability of precise motion control could be limited. Besides, solely relying on the trajectory for motion control might not exactly express target motion category (i.e., camera or object moving), leading to motion misinterpretation in video generation. To alleviate these issues, we propose to directly sample trajectories from optical flow maps in multiple local regions as the region-wise trajectory. Such trajectory could preserve more precise motion details and thus manage to characterize fine-grained movement. Meanwhile, a motion mask is further derived from the flow maps to explicitly identify target motion category of the generated videos.

Region-wise Trajectory. As depicted in Figure 2, given

the input video clip x0 = {xi0}Li=1 with the size of L×H × W ×3, we first employ a dense optical tracking model, i.e.,

DOT [33] to estimate optical flow maps f = {fi}Li=1 and the sequence of visibility masks M = {Mi}Li=1:

fi,Mi = DOT(x10,xi0), i = 1,2,...,L, (4) where fi ∈ RH×W×2 and Mi ∈ {0,1}H×W is the optical flow map and the visibility mask between the first and the i-th frame, respectively. Then, we calculate the intersection on M to attain a global visibility mask Mg ∈ {0,1}H×W that indicates the locations having visible optical flow along temporal dimension as:

Mg =

L

Mi. (5)

i=1

Next, the masked flow maps fm = {fmi }Li=1 are computed by frame-wisely multiplying the flow maps f with

the global visibility mask Mg as follows:

fm = {fi · Mg}Li=1. (6)

We split the masked flow maps fm into multiple local regions and the spatial size of each region is k × k. The region-wise trajectories Ts ∈ RL×H×W×2 are finally sampled from the region-split fm with a region selection mask Msel ∈ {0,1}H

k ×Wk : Ts = {fmi · Pad(Msel)}Li=1, (7)

where Msel is uniformly sampled from {0,1} with the mask ratio rm, and Pad(·) denotes the padding function which fills the mask value into the k × k region around each position. Instead of exploiting a constant mask ratio for trajectory selection, we randomly choose rm in a range of [rmin,1.0] to simulate different real-world motion masking scenarios, which benefits the robust network optimization. In this way, we formulate a more precise signal by exploiting the trajectories in local region, enhancing the control ability of fine-grained motion in I2V models.

Spatial Convolution

| | | |
|---|---|---|
| | | |
| | | |

Temporal Convolution

Group Normalization

Zero Initialization

| |
|---|

* +

###### +

Adaptive Feature Modulation

Figure 3. An illustration of adaptive feature modulation.

Motion Mask. In addition to the region-wise trajectory for video denoising regulation, the motion mask aims to specify the motion category and benefit the global motion correlation. Given the flow maps f = {fi}Li=1 estimated by DOT, we first calculate the average flow magnitude favg ∈ RH×W along temporal dimension as: favg = 1

L · Li=1 ∥ fi ∥2. Then, we construct the motion mask Mmot ∈ {0,1}H×W from zero matrix, and set the value of the position where favg is greater than 1 as True. Mmot is finally repeated L times as the motion mask sequence Mmot ∈ {0,1}L×H×W×1 to align the temporal length of input video for subsequent motion control learning.

##### 3.3. Motion Control Learning

With the region-wise trajectory and motion mask, we aim to control motion generation with the input signals. Inspired by the recipe of feature adaptation in controllable image generation [59], we propose to exploit a lightweight motion encoder to estimate multi-scale features on the conditions, and utilize these features to adaptively modulate video latent feature in each corresponding scale. To further improve the alignment between input trajectory and generated video, we fine-tune all attention modules in the spatial-temporal transformer blocks of 3D-UNet via using LoRA [23].

Adaptive Feature Modulation. Given the attained region-wise trajectory Ts and motion mask Mmot, we first concatenate them along channel dimension to form the input condition. As shown in Figure 2, a lightweight motion encoder with a series of convolutional layers first encodes the input condition into multi-scale feature maps. In each scale, the learnt feature map is employed to modulate the video latent feature at the same scale in 3D-UNet. Figure 3 depicts an illustration of the adaptive feature modulation by using the feature map ls in s-th scale. Particularly, we estimate the scale γs and bias βs on ls via a spatial-temporal convolutional layer. Then, the normalized feature map of the input video latent feature hs is modulated via γs and βs, and further added back to itself in a skip-connection manner to form the output feature map h′s as:

h′s = GN(hs) · γs + βs + hs, (8)

where GN(·) denotes the group normalization. Note that we implement zero initialization on temporal convolutional

Table 1. Fine-grained motion control results on WebVid-10M.

Approach FVD (↓) FID (↓) Frame Consis. (↑)

DragNUWA [57] 96.65 13.19 0.9888 MOFA-Video [35] 87.70 12.18 0.9894 MotionPro 59.88 10.40 0.9895

layers to initialize γs and βs as zero at the beginning of training, guaranteeing the stability of model optimization.

LoRA Integration. To preserve rich motion prior learnt by the pre-trained video diffusion model and elevate the effectiveness of motion control, we employ LoRA layers in all attention modules of spatial-temporal transformer blocks as demonstrated in Figure 2. Specifically, the LoRA parameters ∆W act as a residue part of the original weights W:

###### W′ = W + ∆W = W + ABT, (9)

where W′ is the fused weights of attention module. A and B are trainable matrices in LoRA layers.

- 3.4. Inference Pipeline of MotionPro

Our MotionPro is a user-friendly I2V generation framework for interactive motion control. In the inference stage, as shown in Figure 2, users can readily brush the motion region on the uploaded reference image and draw the trajectory of moving direction as input control signals. In detail, the motion mask can be directly obtained from the user provided brush mask. Given the user trajectory which generally describes the movement of a single pixel, we pad the trajectory value in the k × k region around the pixel position to match the training paradigm. The padded trajectory in local region is exploited as the input region-wise trajectory. Finally, MotionPro regulates video denoising with the guidance of the two collaborative control signals through adaptive feature modulation. Both fine-grained and objectlevel motion control are facilitated by the synergy of the proposed region-wise trajectory and motion mask.

- 4. Experiments

- 4.1. Experimental Settings

Benchmarks. We empirically verify the merit of MotionPro on two benchmarks, i.e., WebVid-10M [1] and our proposed MC-Bench. The WebVid-10M dataset consists of 10.7M video-caption pairs. There are 5K videos in the validation set and we sample 1K videos for evaluation. For each video, trajectories sampled at a ratio of 15% along with the first frame serve as the input condition for fine-grained I2V motion generation. We follow the protocols in recent controllable I2V advance [35] and choose the Frechet Video Distance (FVD) [45], Frechet Image Distance (FID) [19], and Frame Consistency (Frame Consis.) [36] of CLIP [39] features as the evaluation metrics on WebVid-10M.

Table 2. Fine-grained motion control results on MC-Bench.

Approach MD-Img (↓) MD-Vid (↓) Frame Consis. (↑)

DragDiffusion [42] 14.70 13.84 0.9947 MOFA-Video [35] 13.94 10.50 0.9972 MotionPro 10.56 8.34 0.9962

In practical applications, users typically prefer to control video generation through a limited number of representative trajectories, often just one or two. The automatically sampled trajectories employed in WebVid-10M do not adequately represent this scenario, thereby potentially compromising the validity of the evaluation. Thus, we introduce MC-Bench, a new benchmark with 1.1K reference images and user-annotated trajectories, which is tailored for the evaluation of controllable I2V generation. More details about the new dataset are provided in the supplementary material. Due to the absence of ground-truth video, FVD and FID metrics are not applicable to MC-Bench. In addition to Frame Consistency, we utilize the Mean Distance (MD) to measure the alignment between generated motion and input trajectory. Two evaluation protocols are exploited for this target, i.e., MD-Img and MD-Vid. MD-Img is proposed by DragDiffusion [42] which estimates the framelevel mean Euclidean distance between trajectories of input and generated frames. To further validate the video-level trajectory accuracy via MD-Vid, we replace the image correspondence detection model DIFT [44] in MD-Img with the video tracking model CoTracker [25], which supplies a more precise trajectory reference.

Implementation Details. In MotionPro, we employ SVD [5] as our base architecture. Each training sample is 16-frames video clip and the sampling rate is 8 fps. We fix the resolution of each frame as 320 × 512, which is centrally cropped from the resized video. The local region size k is set as 8 and the minimal mask ratio rmin is set as 0.95 determined by cross validation. We set the rank of LoRA parameters as 32. The motion encoder and LoRA layers are trained via AdamW optimizer with the base learning rate 1 × 10−5. All experiments are conducted on 6 NVIDIA A800 GPUs with minibatch size 48.

##### 4.2. Evaluation on Fine-grained Motion Control

We first evaluate MotionPro on the fine-grained motion control for I2V generation. The performances on WebVid-10M and MC-Bench are summarized in Table 1 and Table 2, respectively. Our MotionPro consistently achieves better performances on WebVid-10M across different metrics. In particular, MotionPro attains the FVD of 59.88, outperforming the best competitor MOFA-Video by 27.82. The better FVD indicates the better alignment of data distribution between the generated and ground-truth videos. Such results basically verify the superiority of exploring precise regionwise trajectory to strengthen fine-grained motion dynamic learning. On MC-Bench, MotionPro leads to performance

###### Input Control DragNUWA DragDiffusion MOFA-Video MotionPro

[Figure 503]

[Figure 504]

[Figure 505]

- Figure 4. Examples of fine-grained motion control results on MC-Bench. The input control signals include the reference image, trajectory and motion mask. Best viewed with Acrobat Reader for the animated videos.

Table 3. Object-level motion control results on MC-Bench.

Approach MD-Img (↓) MD-Vid (↓) Frame Consis. (↑)

MOFA-Video [35] 15.56 12.04 0.9951 DragAnything [52] 12.30 11.37 0.9917 MotionPro 10.48 8.59 0.9943

boosts against baselines in terms of MD-Img and MD-Vid, showing better alignment between the user input trajectory and synthesized videos. Note that MOFA-Video exploits a two-stage controllable I2V framework that first densifies the input trajectories through conditional motion propagation (CMP), and then calibrates video diffusion process using the estimated dense trajectories. In contrast, MotionPro learns precise motion patterns by directly referring regionwise trajectory via adaptive feature modulation, thus enhancing the motion-trajectory alignment, as evidenced by the better MD-Img and MD-Vid performances. Besides, the CMP technique in MOFA-Video generally focuses on flow completion in the local region surrounding the input trajectory while neglecting potential movements in other areas. Thus, MOFA-Video tends to synthesize videos with less motion dynamics and obtains slightly higher Frame Consistency (approximately 0.001). To substantiate this, we calculate the average flow magnitude of videos generated by MOFA-Video, which achieves 4.95. In comparison, MotionPro attains a higher value of 8.95, verifying that our model achieves greater motion variability while maintaining better motion-trajectory alignment.

Figure 4 further showcases three I2V generation results controlled by the user input trajectory and region mask on MC-Bench. Generally, the videos synthesized by our MotionPro exhibits more natural movement and better alignment with input trajectory than the baseline methods. For instance, DragNUWA suffers from motion misinterpretation issue which wrongly generates videos with camera movement instead of object moving (e.g., the 1st and 2nd cases). The videos generated by MOFA-Video usually present unnatural object movement with local part distortion, e.g., the nose of raccoon in the 2nd case. We speculate that such distortion is caused by the lack of global region guidance in MOFA-Video, where the region mask is only employed for flow masking as post-processing. Our MotionPro, in comparison, integrates the information of motion mask into 3D-UNet on the fly to facilitate the modeling of holistic motion correlation. Thus, the synthesized videos by MotionPro reflect more rational fine-grained movement.

##### 4.3. Evaluation on Object-level Motion Control

Next, we conduct evaluation on object-level motion control for I2V generation. Table 3 lists the performances of different approaches on MC-Bench. Overall, MotionPro attains the best performances on the metrics of MD-Img and MDVid. Specifically, MotionPro obtains 10.48 of MD-Img and 8.59 of MD-Vid, reducing the Mean Distance of the best competitor DragAnything by 1.82 and 2.78, respectively.

###### Input Control MOFA-Video DragAnything MotionPro

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

- Figure 5. Examples of object-level motion control results on MC-Bench. The input control signals include reference image, trajectory and motion mask. MotionPro can successfully handle complicated (e.g., the round trip of sun in the 1st case) and counterintuitive (e.g., the train moving back in the 3rd case) motion-trajectory alignment. Best viewed with Acrobat Reader.

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

(a) (b)

- Figure 6. Performance comparisons of MD-Vid and Frame Consistency on MC-Bench under the settings of both fine-grained and objectlevel motion control by using different (a) local region size k and (b) minimal mask ratio rmin in MotionPro.

The improvements again confirm the merit of leveraging the duet of region-wise trajectory and motion mask for precise motion control. Similar performance trend on Frame Consistency can be also observed in the table.

Figure 5 shows the visual comparison of four objectlevel motion control results by using different approaches on MC-Bench. Compared to other baselines, videos generated by MotionPro can precisely match the input trajectory and maintain natural object-level motion dynamics. MOFA-Video still faces the challenge of local part distortion (e.g., only the train rear moving back in the 3rd case) and video generation with limited motion dynamics (e.g., the 4th case). Though DragAnything effectively aligns pixel movement with the input trajectory, certain instances (e.g., the 1st and 4th cases) misinterpret the trajectory as cam-

era motion rather than object movement. In contrast, MotionPro nicely capitalizes on trajectory information to refine video denoising, and specifies motion category with the region mask, endowing images with better object motion.

##### 4.4. Ablation Study on MotionPro

In this section, we perform ablation study to delve into the design of MotionPro for controllable I2V generation. Here, all experiments are conducted on MC-Bench.

Local Region Size. We first investigate the choice of local region size k for region-wise trajectory design in our MotionPro. Figure 6(a) compares the performances of MDVid and Frame Consistency on both fine-grained and objectlevel motion control by using different k. The variation of Frame Consistency is minor (less than 0.01) across differ-

Control k=16 k=8 Control k=1 k=2 k=4 k=8

[Figure 514]

[Figure 515]

- Figure 7. Visualization of controllable I2V generation results with different local region size k in MotionPro. Viewed with Acrobat Reader.

[Figure 516]

ent settings, and the MD-Vid decreases when using larger k. When k is small (e.g., 1 or 2), the kept trajectories are less in each local region and the control signals are weaken for motion control, leading to the inferior trajectory matching performance. Meanwhile, the improvement of MD-Vid is marginal when increasing k to 16. Specifically, using large k will extend the input trajectory over a large region, which affects the fine-grained motion control. Accordingly, we exploit k = 8 to extract the region-wise trajectory as the motion condition. Figure 7 further illustrates the I2V generation results with different k. As shown in this figure, the synthesized videos with k = 8 present more natural motion dynamics and more precise motion-trajectory alignment. Moreover, the unnatural fine-grained motion as shown in the case when k = 16 validates our analysis on the influence of overlarge region size.

Figure 8. MD-Vid (↓) among different multi-scale feature injection approaches on MC-Bench.

information exchange requires strict spatial-temporal alignment between each other. In contrast, there is no such requirement for feature modulation, as it indirectly utilizes estimated scale and bias for feature regulation. Consequently, such feature injection demonstrates enhanced capacity to extract relevant information from input signals, potentially leading to improved motion control performance.

Minimal Mask Ratio. To explore the effect of minimal mask ratio rmin in trajectory selection stage, we then measure the motion control performance by conducting different rmin in Figure 6(b). Overall, Frame Consistency is not sensitive when changing rmin on both fine-grained and object-level motion control settings. Meanwhile, the performance of MD-Vid becomes better with the increase of the mask ratio at the beginning. The results are expected since using small value of rmin will sample more trajectories for model training, which enlarges the gap between training and real-world inference (i.e., only using one or two trajectories). Conversely, employing a large value of mask ratio (e.g., 0.99) could make it difficult to optimize networks with scarce trajectory signals. Thus, we empirically set rmin as 0.95 to obtain the best motion-trajectory alignment in the generated videos.

#### 5. Conclusions

This paper explores the motion condition formulation and the motion-trajectory alignment in diffusion models for controllable I2V generation. In particular, we study the problem from the viewpoint of integrating accurate motion control signals into video denoising to regulate motion generation. To materialize our idea, we have devised MotionPro, which leverages the region-wise trajectory and motion mask as the condition to calibrate video generation in a feature modulation manner. The region-wise trajectory directly exploits the original trajectory information in each local region, characterizing more accurate motion details. The motion mask derived from the optical flow maps presents holistic motion and aims to identify exact motion category. The collaboration of the two signals regulates video denoising for natural motion synthesis with precise motion-trajectory alignment. Moreover, we have carefully construct a new benchmark, i.e., MC-Bench, with 1.1K user-annotated image-trajectory pairs for both fine-grained and object-level motion control evaluation. Extensive experiments on WebVid-10M and MC-Bench validate the superiority of our proposal over state-of-the-art approaches.

Multi-scale Feature Injection. We also investigate different multi-scale feature injection strategies in MotionPro. Figure 8 details the MD-Vid performance comparisons among different variants of our MotionPro. MotionProC concatenates the multi-scale features learnt by motion encoder with the video latent features along channel dimension in each scale. MotionPro+ replaces the channel-wise feature concatenation in MotionProC with the feature summation. In comparison, our proposal (MotionPro) injects the control signals into 3D-UNet via the adaptive feature modulation. Overall, MotionPro exhibits better MD-Vid performances against other two variants. In direct feature aggregation methods such as concatenation or summation,

Acknowledgments. This work was supported in part by the Beijing Municipal Science and Technology Project No.Z241100001324002 and Beijing Nova Program No.20240484681.

#### References

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval. In ICCV, 2021. 5, 12
- [2] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models. arXiv preprint arXiv:2405.04233, 2024. 2
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A Space-Time Diffusion Model for Video Generation. arXiv preprint arXiv:2401.12945, 2024. 2
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, and Yunxin Jiao. Improving Image Generation with Better Captions, 2023. 12
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 5
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. In CVPR, 2023. 2
- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video Generation Models as World Simulators.

2024. 2

- [8] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. StableVideo: Text-driven Consistency-aware Diffusion Video Editing. In ICCV, 2023. 2
- [9] Jingyuan Chen, Fuchen Long, Jie An, Zhaofan Qiu, Ting Yao, Jiebo Luo, and Tao Mei. Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-free Long Video Diffusion. In AAAI, 2025. 1
- [10] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. ControlA-Video: Controllable Text-to-Video Diffusion Models with Motion Prior and Reward Feedback Learning. arXiv preprint arXiv:2305.13840, 2023. 2
- [11] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction. In ICCV,

2023. 2

- [12] Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, and Tao Mei. Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video Super-Resolution. In CVPR, 2024. 1

- [13] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and Content-Guided Video Synthesis with Diffusion Models. In ICCV, 2023. 1, 2
- [14] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models. In ICCV, 2023. 2
- [15] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. In ECCV, 2024. 2
- [16] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, Haibin Huang, and Chongyang Ma. I2VAdapter: A General Image-to-Video Adapter for Diffusion Models. arXiv preprint arXiv:2312.16693, 2023. 1
- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In ICLR, 2024. 2
- [18] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic Video Generation with Diffusion Models. arXiv preprint arXiv:2312.06662, 2023. 1, 2
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In NeuIPS, 2017. 5
- [20] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen Video: High Definition Video Generation with Diffusion Models. In CVPR, 2022. 1, 2
- [21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video Diffusion Models. In NeurIPS, 2022.
- [22] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. CogVideo: Large-scale Pretraining for Text-toVideo Generation via Transformers. In ICLR, 2023. 1, 2
- [23] Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR, 2022. 3, 4
- [24] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. PEEKABOO: Interactive Video Generation via MaskedDiffusion. In CVPR, 2024. 2
- [25] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker: It is Better to Track Together. In ECCV, 2024. 5
- [26] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the Design Space of Diffusion-Based Generative Models. In NeurIPS, 2022. 3
- [27] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant

- Navasardyan, and Humphrey Shi. Text2Video-Zero: Textto-Image Diffusion Models are Zero-Shot Video Generators. In ICCV, 2023. 1, 2
- [28] Hengyuan Liu, Xiaodong Chen, Xinchen Liu, Xiaoyan Gu, and Wu Liu. AnimateAnywhere: Context-Controllable Human Video Generation with ID-Consistent One-shot Learning. In HCMA, 2024. 1
- [29] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. VideoStudio: Generating Consistent-Content and MultiScene Videos. In ECCV, 2024. 1
- [30] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation. In CVPR, 2023. 2
- [31] Wan-Duo Kurt Ma, J.P. Lewis, and W. Bastiaan Kleijn. TrailBlazer: Trajectory Control for Diffusion-Based Video Generation. arXiv preprint arXiv:2401.00896, 2023. 2
- [32] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent Diffusion Transformer for Video Generation. arXiv preprint arXiv:2401.03048, 2024. 2
- [33] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid. Dense Optical Tracking: Connecting the Dots. In CVPR,

2024. 4

- [34] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. ReVideo: Remake a Video with Motion and Content Control. In NeurIPS, 2024. 2
- [35] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model. In ECCV,

2024. 2, 5, 6, 12, 13

- [36] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. FateZero: Fusing Attentions for Zero-Shot Text-Based Video Editing. In ICCV, 2023. 5
- [37] Yurui Qian, Qi Cai, Yingwei Pan, Yehao Li, Ting Yao, Qibin Sun, and Tao Mei. Boosting Diffusion Models with Moving Average Sampling in Frequency Domain. In CVPR, 2024. 1
- [38] Mengxue Qu, Xiaodong Chen, Wu Liu, Alicia Li, and Yao Zhao. ChatVTG: Video Temporal Grounding via Chat with Video Dialogue Large Language Models. In CVPR, 2024. 1
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In ICML,

2021. 2, 3, 5

- [40] Diana Wofk Peter Wonka Matthias M¨uller Shariq Farooq Bhat, Reiner Birkl. ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth. In CVPR, 2023. 13
- [41] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling. In ACM SIGGRAPH, 2024. 2

- [42] Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent Y. F. Tan, and Song Bai. DragDiffusion: Harnessing Diffusion Models for Interactive Point-Based Image Editing. In CVPR, 2024. 5, 12, 13
- [43] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-Video: Text-to-Video Generation without Text-Video Data. In ICLR, 2023. 1, 2
- [44] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent Correspondence from Image Diffusion. In NeurIPS, 2023. 5
- [45] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new Metric for Video Generation. In ICLR DeepGenStruct Workshop, 2019. 5
- [46] Siqi Wan, Jingwen Chen, Yingwei Pan, Ting Yao, and Tao Mei. Incorporating Visual Correspondence into Diffusion Model for Virtual Try-On. In ICLR, 2025. 1
- [47] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. ModelScope Text-to-Video Technical Report. arXiv preprint arXiv:2308.06571, 2023. 1, 2
- [48] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating Rich and Controllable Motions for Video Synthesis. arXiv preprint arXiv:2402.01566, 2024. 2
- [49] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional Video Synthesis with Motion Controllability. In NeurIPS, 2023. 1, 2
- [50] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. In ACM SIGGRAPH, 2024. 2
- [51] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. MotionBooth: Motion-Aware Customized Text-to-Video Generation. In NeurIPS, 2024. 2
- [52] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. DragAnything: Motion Control for Anything using Entity Representation. In ECCV, 2024. 1, 2, 6, 12, 13
- [53] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors. In ECCV, 2024. 2
- [54] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, Chong-Wah Ngo, and Tao Mei. Hi3D: Pursuing HighResolution Image-to-3D Generation with Video Diffusion Models. In ACM MM, 2024. 1
- [55] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-Video: Customized Video Generation with UserDirected Camera Movement and Object Motion. In ACM SIGGRAPH, 2024. 2

- [56] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [57] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. DragNUWA: Fine-Grained Control in Video Generation by Integrating Text, Image, and Trajectory. arXiv preprint arXiv:2308.08089, 2023. 1, 2, 5, 12
- [58] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make Pixels Dance: HighDynamic Video Generation. In CVPR, 2024. 2
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. In ICCV, 2023. 4
- [60] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models. arXiv preprint arXiv:2311.04145, 2023. 2
- [61] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. ControlVideo: Training-Free Controllable Text-to-Video Generation. In ICLR, 2024. 1, 2
- [62] Zhongwei Zhang, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Ting Yao, Yang Cao, and Tao Mei. TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Video Diffusion Models. In CVPR, 2024. 1

#### 6. Supplementary Material

The supplementary material contains: 1) the dataset details of MC-Bench; 2) baseline choices and experimental details; 3) the human evaluation of motion control; 4) robustness of motion mask; 5) the application of camera control; 6) runtime comparison; 7) ablation on control signals.

##### 6.1. Dataset Details of MC-Bench

The proposed MC-Bench consists of 412 high-quality reference images and corresponding 1.1K user-annotated trajectories. We collect the reference images with different visual contents, including animal, human, vehicle, etc. There are 72 images sampled from the public DragBench [42] and we further extend it with 340 additional images. Specifically, all the self-collected images about human are automatically generated by DALL·E3 [4] to avoid the potential legal concerns. The remaining self-collected images are real photos which are first crawled on the Pexels platform and then filtered according to the visual quality. For each reference image, the annotator is required to brush the motion region and draw the movement trajectory according to user intention (i.e., fine-grained local part moving or global object moving). During trajectory annotation, all annotators are encouraged to ensure the trajectory diversity, including some complicated trajectories. Finally, the benchmark is annotated with 460 image-trajectory pairs for fine-grained motion control evaluation, and 680 image-trajectory pairs for object-level motion control evaluation, respectively. Figure 12 and Figure 11 further illustrate several visual examples (reference image, trajectory and motion mask) from MC-Bench for the two evaluations.

##### 6.2. Baseline Choices and Experimental Details

For the evaluation on WebVid-10M [1] of fine-grained motion control, we adopt the commonly-used protocol in recent controllable image-to-video (I2V) advance [35]. Specifically, for each video, we sample the optical flow at the ratio of 15% as the sparse trajectories, which are combined with the first frame as the input condition. Under this experimental setting, we choose DragNUWA [57] and MOFA-Video [35] as baselines for comparison. Notably, DragAnything [52] is deliberately designed for object-level motion control, which only accepts a single trajectory of object, making it inapplicable for fine-grained motion control. Therefore, DragAnything is not involved for comparison in this setting.

For the fine-grained motion control on MC-Bench, we compare our MotionPro with DragDiffusion [42] and MOFA-Video. DragNUWA is not included in this comparison since it only relies on trajectories and lacks the input of motion regions. Thus, DragNUWA usually suffers from the misinterpretation of object and camera movement, making the comparison unfair. The baseline of DragDiffusion is a

recent trajectory-guided image editing advance, which also offers convincing results for comparison. To adapt DragDiffusion for video generation, we divide the input trajectories into 15 segments and independently feed each segment into DragDiffusion to generate target frame. All the synthesized frames are concatenated as the final video.

In the evaluation of object-level motion control on MCBench, both MOFA-Video and DragAnything are employed as baselines for performance comparison. To facilitate DragAnything in disentangling object and camera moving, we add static points in regions outside the motion mask areas to help DragAnything generate object-level motion instead of camera moving for evaluation. It’s worth to noting that MotionPro learns object and camera motion control on “inthe-wild” video data (e.g., WebVid-10M) without applying special data filtering.

##### 6.3. Human Evaluation

In addition to the evaluation over automatic metrics, we also conduct human evaluation to investigate user preferences from three perspectives (i.e., motion quality, temporal coherence and trajectory alignment) across different controllable I2V approaches. In particular, we randomly sample 200 generated videos from both fine-grained and objectlevel motion control for evaluation. Through the Amazon MTurk platform, we invite 32 evaluators, and ask each evaluator to choose the best one from the generated videos by all models given the same inputs.

Table 4 shows the user preference ratios across different models on MC-Bench. Overall, our MotionPro clearly outperforms all baselines in terms of the three criteria on both fine-grained and object-level motion control. The results demonstrate the advantage of leveraging complementary region-wise trajectory and motion mask to benefit video synthesis with natural motion, desirable temporal coherence and precise motion-trajectory alignment.

##### 6.4. Ablation on control signals.

We also include two runs (MotionPro−traj: replaces regionwise trajectory with random trajectory, MotionPro−mask: disables motion mask with all-one masks). Their FVD (73.7 and 66.2) on WebVid-10M are inferior to our MotionPro (59.88), which validates the effectiveness of our two control signal designs for precise motion formulation.

##### 6.5. Robustness of motion mask

To be clear, motion mask in our MotionPro refers to the rough dynamic region and does not require preciselyaligned shape at inference. We show I2V results controlled by the same trajectory with various motion masks in Figure 10, which show strong robustness. Such generalization merit is attributed to the use of estimated motion mask (flow

Table 4. Human evaluation of user preference ratios (%) over both fine-grained and object-level motion control on MC-Bench.

Fine-grained Motion Control Object-level Motion Control DragDiffusion [42] MOFA-Video [35] MotionPro MOFA-Video [35] DragAnything [52] MotionPro

Evaluation Items

Motion Quality (↑) 3.12 21.88 75.00 12.50 18.75 68.75 Temporal Coherence (↑) 6.25 40.63 53.12 25.00 15.63 59.37 Trajectory Alignment (↑) 9.37 18.75 71.88 15.62 21.88 62.50

Input Image

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

Depth Estimation

[Figure 522]

Metric Depth Map

3D Point Cloud Reconstruction

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

Camera pose sequence

[Figure 549]

Sparse Trajectories

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

### ...

Motion Mask

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

MotionPro

[Figure 562]

[Figure 563]

...

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

Generated Video

Figure 9. An illustration of I2V camera control using the condition of camera pose sequence in our MotionPro.

map estimated by DOT) at training, rather than ground-truth precise motion mask.

##### 6.6. Application: Camera Control

Our learnt MotionPro naturally supports two applications of camera control without additional training. The first application is controlling object and camera motion simultaneously with multiple trajectories in I2V generation. Another application is the I2V camera control by exploiting the sequence of camera poses as input condition. To be clear, motion mask in our MotionPro refers to the rough dynamic region and does not require precisely-aligned shape at inference.

Simultaneous object and camera motion control. In this setting, we simply set the input motion mask as all-ones matrix, and feed multiple trajectories that reflect the object and background moving into MotionPro for I2V generation. The video cases are provided in the offline project page.

Camera control with camera poses. Figure 9 illustrates the process of camera control using the condition of camera pose sequence in MotionPro. Concurrently, given an input image and the camera pose sequence, we first estimate the metric depth map of the image using ZoeDepth [40]. Next, we lift the 2D pixels to 3D point cloud using the metric depth map. Through projecting the point cloud into 2D space given the camera pose, we can determine the corresponding 2D positions of the same 3D points under the new view. By calculating the 2D displacement of the pixels projected from the same 3D points in the original and new views, the camera pose sequence is then converted into the sparse trajectories. Finally, we feed the sparse trajectories

and all-ones motion mask into MotionPro for I2V synthesis. The video cases are provided in the offline project page.

##### 6.7. Runtime Comparison

For 16-frame video generation (resolution: 512 × 320, on single NVIDIA H100 GPU), the runtime of MotionPro is 17 sec, which is comparable to baselines (DragNUWA: 27, DragDiffusion: 320, MOFA-Video: 15, DragAnything: 32).

Figure 10. MotionPro conditioned on diverse mask shapes. Best viewed with Acrobat Reader.

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

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

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

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

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

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

[Figure 656]

[Figure 657]

[Figure 658]

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

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

- Figure 11. Visual examples from MC-Bench for object-level motion control evaluation. Each reference image is annotated with trajectory and motion mask for image-to-video generation.

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

###### Figure 12. Visual examples from MC-Bench for fine-grained motion control evaluation. Each reference image is annotated with trajectory and motion mask for image-to-video generation.

