# arXiv:2406.00908v1[cs.CV]3Jun2024

## ZeroSmooth: Training-free Diffuser Adaptation for High Frame Rate Video Generation

Shaoshu Yang1, 2, Yong Zhang3, , Xiaodong Cun3, Ying Shan3, and Ran He1, 2,

1School of Artificial Intelligence, University of Chinese Academy of Sciences 2New Laboratory of Pattern Recognition (NLPR), Institute of Automation, Chinese Academy of Sciences 3Tencent AI Lab 1https://ssyang2020.github.io/zerosmooth.github.io

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### SVD

- (7 fps 14 frames)

SVD ZeroSmooth

(28 fps 56 frames)

VideoCrafter2

- (8 fps 16 frames)

Image-to-Video

ZeroSmoothZeroSmooth

| | | |[Figure 7]| |[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]| | | | | | |[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]| | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Text-to-Video

| | | |[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]| | | |[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]| | | |[Figure 44]<br><br>[Figure 45]| | | | |[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

VideoCrafter2 ZeroSmooth (32 fps 64 frames)

Figure 1: Our method enables pretrained video diffusion models for high frame rate (4× more than during training) generation without extra training data and parameter updates.

### Abstract

Video generation has made remarkable progress in recent years, especially since the advent of the video diffusion models. Many video generation models can produce plausible synthetic videos, e.g., Stable Video Diffusion (SVD). However, most video models can only generate low frame rate videos due to the limited GPU memory as well as the difficulty of modeling a large set of frames. The training videos

Corresponding author

Preprint. Under review.

are always uniformly sampled at a specified interval for temporal compression. Previous methods promote the frame rate by either training a video interpolation model in pixel space as a postprocessing stage or training an interpolation model in latent space for a specific base video model. In this paper, we propose a trainingfree video interpolation method for generative video diffusion models, which is generalizable to different models in a plug-and-play manner. We investigate the non-linearity in the feature space of video diffusion models and transform a video model into a self-cascaded video diffusion model with incorporating the designed hidden state correction modules. The self-cascaded architecture and the correction module are proposed to retain the temporal consistency between key frames and the interpolated frames. Extensive evaluations are preformed on multiple popular video models to demonstrate the effectiveness of the propose method, especially that our training-free method is even comparable to trained interpolation models supported by huge compute resources and large-scale datasets.

### 1 Introduction

Video generation has undergone rapid evolution recently, benefiting from the development of diffusion models [15, 29]. There are numerous popular video diffusion models [17] that can generate plausible synthetic videos. They are trained on large-scale datasets, including videos and images, using extensive compute resources. Several models are trained in the pixel space, e.g., Imagen Video [14] and Make-A-Video [27]. The majority are trained in the latent space, e.g., Stable Video Diffusion (SVD) [2], Align-your-latent [3], VideoCrafter [5], LaVie [34], ModelScope [33], etc.

Although those video models can produce realistic videos, most of them cannot generate high frame rate videos with smooth transitions. One reason is that most video generation models use uniformly sampled videos as training data to avoid exceeding the GPU memory limit; i.e., models are trained to generate key frames. The other reason is that capturing the distribution of a long video is challenging. In this work, we focus on improving the frame rate of those generative video models.

Conventional video frame interpolation methods can be grouped into two categories: i.e., flowbased [8, 20, 26, 37] and kernel-based methods [21, 41]. The former follows a two-stage paradigm that first estimates flows between neighboring frames and then applies forward or backward warping to the pixels or latent features for intermediate frame generation. The performance relies on the accuracy of the estimated flow, which is still a quite challenging task. The latter considers pixel synthesis for the interpolated frame as local convolution over input frames and uses a network to predict a convolution kernel for each pixel. However, it has difficulty handling large spatial displacements.

Recently, several methods have applied diffusion models for video interpolation, e.g., LDMVFI [9], MCVD [31], VIDIM [18], and CBBD [23]. These methods treat video interpolation as a conditional generation or temporal inpainting task. Given key frames, the diffusion models are trained to generate intermediate frames through the denoising process. In some video generation methods, video interpolation is integrated into the video model as a module, e.g., ImagenVideo, Make-A-Video, LaVie, and Show-1. The interpolation module is deeply integrated with other modules, which makes it difficult to incorporate the interpolation module into other video models.

Although conventional methods, diffusion-based models, and bounded interpolation modules can generate intermediate frames for constructing a smooth video, all these methods require a training process involving millions or even billions of parameters. Furthermore, since conventional and diffusion-based models are always trained independently on real data, a domain gap exists when applying them to synthetic videos produced by generative video models. As for bounded interpolation modules, they need to be retrained if the base video model is updated, and they are not generalizable across different video models.

In this work, we propose a training-free method to improve the frame rate of various existing generative video models, e.g., SVD [2] and VideoCrafter [4]. The proposed method can be applied in a plug-and-play manner across various video models. We investigate the temporal correlation of features learned by the video UNet and observe that latent space back-projection fails to inject content and appearance to adjacent frames. Fortunately, we find the temporal correlation learned in transformer hidden states is strong. Back-projection in transformer hidden states achieves strong

visual content control while preserving great inter-frame consistency. Based on this observation, we transform the target video model into a self-cascaded architecture containing two branches (see Fig. 2). One branch retains the architecture of the target model for short video inference, while the other is adapted for long video inference by placing the proposed hidden state correction modules into the transformer blocks. The correction modules use both hidden states from the two branches as input to calibrate the hidden states of the long branch for inter-frame consistency. Additionally, we design a strategy for the controllability of the correction strength.

Our contributions are summarized as follows:

- • We propose a training-free method to enhance generative video models to produce videos with a higher frame rate, resulting in visually smooth videos. The method can be applied to various video models in a plug-and-play manner.
- • We propose the self-cascaded architecture and the hidden state correction modules for inter-frame consistency.
- • We conduct extensive experiments with various popular generative video models to demonstrate the effectiveness of the proposed method, inclusing SVD, VideoCrafter, and LaVie.

### 2 Related Works

Video Diffusion Models Diffusion models now prevail in the video generation community. They model the video data distribution with a sequence of iterative denoising steps [15, 28, 29]. Diffusion models are used for a variety of video tasks, including action category conditioned video generation [31, 38, 22, 11], text conditioned video generation [14, 3, 13], image conditioned video generation [2, 40], and video translation [36, 39]. Among them, VideoCrafter[4, 5], LaVie[34], and StableVideoDiffusion[2] approximate the data in a compact video latent space computed by a VAE[19]. Meanwhile, Imagen-Video [14], Make-A-Video [27], and PYoCo [11] learn the pixel space video distribution. Some efforts have been made in long video generation [32, 25]. Gen-L-Video [32] bootstraps video diffusers to generate beyond the video length limit. FreeNoise [25] proposes a noise schedule to alleviate content shifting.

Zero-shot Visual Restoration and Video Interpolation With the power of modern pre-trained diffusion models, zero-shot visual restoration has made considerable progress [10, 35, 7, 42]. ILVR [6] adopts a trained diffuser for zero-shot image super-resolution. DDNM [35] proposes theoretical insights into null-space and range-space decomposition of visual restoration. DDPG [10] combines the back-projection method and least square method using an optimization preconditioner. However, limited efforts have been made to deliver zero-shot visual restoration to the video domain. ScaleCrafter [12] proposes a tuning-free method for inferring at a higher resolution. In this paper, we investigate the potential of zero-shot methods for higher frame rate generation.

Video interpolation is a long-standing problem in computer vision. Traditional video interpolation models adopt a two-stage paradigm to estimate the optical flow and use it to aid frame interpolation [8, 20, 37, 26]. Recently, progress has been made in deploying new architectures [21, 41] and training paradigms [9, 18, 23]. VFIformer [21] adopts a transformer for frame interpolation. LDMVFI[9] and VIVDM[18] propose training diffusion models for the task.

### 3 Preliminary

##### 3.1 Video Diffusion Models

Given a video x0 ∼ p(x0) with S frames, let x0 = {x00,x10,··· ,xS0 }. Generative models approximate a sequence of diffusion denoising transitions [17, 15, 24]

q(zt|x0) = N(zt|αtx0,σt2I) (1) q(zs|zt,x0) = N(zs|µs|t(zt,x0),σs2|tI), (2)

where zt is the noisy data at a noise scale t = 1,2,··· ,T and 0 ≤ s < t ≤ T. Let λt = log(αt2/σt2) be the monotonically decreasing signal-to-noise ratio along noise scale t. The mean and variance of

output sequence 0 1 2 3

output sequence 0 2

[Figure 49]

Result blending

share

Temporal Transformer

Temporal Transformer

| |[Figure 50]|
|---|---|
|Output p|rojection|
| | |
|Atten|tion|
| | |
|[Figure 51]<br><br>[Figure 52]|[Figure 53]|
| | |

| |[Figure 54]|
|---|---|
|Output p|rojection|
| | |
|Atten|tion|
| | |
|[Figure 55]<br><br>[Figure 56]|[Figure 57]|
| | |
|Hidden state|correction|
| | |

Hidden state correction ZeroSmooth

share

Spatial Transformer

Spatial Transformer

Hidden state correction

ZeroSmooth

[Figure 58]

share

ResNet Block ResBlock

× N

QKV Projection QKV projection

× N

[Figure 59]

[Figure 60]

Base model ZeroSmooth model

0 2 0 1 2 3

[Figure 61]

Input sequence Input sequence

(b) ZeroSmooth

(a) Self-cascaded Video Diffusion Model

- Figure 2: An overview of our method. (a) We build cascaded video diffusion model by adapting the base generator to generate at a higher frame rate. (b) A sketch for hidden states correction in transformers in ZeroSmooth.

backward diffusion process q(zs|zt,x0) is

- αs

- αt

t−λs)αsx0 (3) σs2|t = (1 − eλ

t−λs(

µs|t(zt,x0) = eλ

)zt + (1 − eλ

t−λs)σt2. (4)

Starting from Gaussian noise zT ∼ N(zT|0,I), we train a video diffusion UNet to denoise zT iteratively. Specifically, it estimates pθ(zs|zt,y) = N(zs|µsθ|t(zt,y),Σsθ|t(zt),y), which approximates the reverse diffusion process with condition y, i.e. textual prompts. The diffusion UNet ϵθ(·) is parameterized to estimate the added noise in xt. One can rearrange the ϵθ(·) to compute the noiseless prediction x0|t = fθ(zt,y) that approximates x0 [28].

##### 3.2 Visual Restoration with Diffusion Models

The back-projection method [35, 10] is prevalent for zero-shot image restoration. It considers a linear measurement operator A, such that the degenerated sample is y = Ax0. Meanwhile, the high-resolution image can be decomposed into two parts

##### x0 = A†Ax0

+(I − A†A)x0

(5)

rangespace

nullspace

where A†Ax0 is in the range space of A, (I − A†A)x0 is in the null space of A. A† is the pseudo inverse of A such that AA†A = A. DDNM [35] proposes to back-project the observed part of a sample y to replace the range space part in denoising process

xˆ0|t = (I − A†A)x0|t + A†y (6)

where xˆ0|t is the corrected estimate. Therefore, the diffusion model only predicts in the null space of A and ensures the consistency to condition y in the output.

### 4 Method

We aim to adapt a video diffusion model to generate at a higher frame rate while preserving its original generation capability. Given a pretrained video diffusion model ϵθ(·) : Rc×t

0×h×w, which is trained on t0 frame videos. Our method creates ϵˆθ(·) : Rc×t×h×w → Rc×t×h×w from ϵθ(·) in a training-free manner, where t > t0. Once the key frames y are observed, generating the corresponding high frame rate video can be considered video frame interpolation (VFI). Meanwhile, the measurement operator At

0×h×w → Rc×t

0×t, which acquires key frames from a full video x0 is linear and follows the formulation of y = Ax0. An intuitive idea to achieve this is to apply DDNM [35] to the

t0-frame base video generated by ϵθ(·), interpolating between the key frames and get the t-frame high frame rate video. During this process, one can build a t-frame generator ϵˆθ(·) using the method proposed by Qiu et al. [25] After each denoising step, the key frames are back-projected to correct the estimate.

However, as seen in Fig. 4, we find it challenging for this intuitive approach to maintain temporal consistency between observed and interpolated frames. We attribute this to the weak temporal correlation of the video latent space learned by diffusers. Correcting the estimate in the latent space can hardly inject content and appearance into the interpolated frames. Alternatively, we propose to correct the hidden states of transformer modules where the temporal correlation is strong. Our method is summarized in three parts. In the following subsections, we introduce the overall architecture of our method, the hidden state correction method in transformers, and our remedy to alleviate joint distribution mismatch after correction.

##### 4.1 Self-cascaded Video Diffusion Model

- As shown in Fig. 2(a), we construct a cascaded video diffusion model consisting of ϵθ(·) and ϵˆθ(·). In our method, ϵθ(·) and ϵˆθ(·) share the same neural network modules and parameters. Therefore, we call this paradigm a self-cascaded video diffusion model. In order to correct the high frame rate estimate, we first generate key frame transformer hidden states. The hidden state distribution in a diffusion model varies along denoising timesteps. Instead of using only a final key frame video for back-projection like DDNM, our method uses the key frame hidden states at the corresponding step to match the data distribution at different timesteps.

Mathematically, we denote the input to spatial or temporal transformer as h ∈ Rt×l×d in ˆϵθ(·). l is the stacked spatial dimension (i.e., feature width times feature height), and d is the hidden

state channel. The hidden state input for the corresponding transformer in ϵθ(·) is hk ∈ Rt

0×l×d.

- As shown in Fig. 2(b), we apply back-projection to correct the hidden state estimation with key frame hidden states hk before computing attention. We denote the query, key and value with Q ∈ Rt×l×d,K ∈ Rt×l×d,V ∈ Rt×l×d. After hidden state correction, we use Qˆ ,Kˆ and Vˆ to represent them. Note that the self-cascaded video diffusion model can include multiple stages to achieve higher frame rate results. For instance, consider t = nt0, where n is the video interpolation

2t0×h×w from ϵθ(·). The self-cascaded model {ϵθ,ϵˆθ,ϵ˜θ} achieves n2 times more frames than the key frame generator.

2t0×h×w → Rc×n

scale. One can construct another model ϵ˜θ(·) : Rc×n

Temporal Attention for High Frame Rate Generation A pretrained video diffusion model is trained on videos of length t0. Therefore, t-frame video is unseen for the temporal transformer modules. Direct inference without adaptation will result in severe sample quality degradation. We propose ZeroSmooth temporal attention to adapt the modules for a longer sequence. Video diffusers use different categories of temporal attention. Text-to-video diffusion models, like LaVie[34] and

- VideoCrafter1 [4], use relative positional embedding (RPE). The sequence length should be kept unchanged when processing high frame rate input. We improve upon the attention fusion technique in FreeNoise [25] by adding RPE to every attention window, ensuring correct relative time perception. Image-to-video diffusion models, like StableVideoDiffusion [2], use absolute positional embedding (APE) to provide temporal distance between the current frame and the reference image. Therefore, we interpolate the position index to maintain the temporal distance in the high frame rate video.

##### 4.2 Hidden State Correction for Transformers

We adopt back-projection method in transformer modules of video diffusion to correct hidden states estimate. Without loss of generality, we consider a 2× frame interpolation in this subsection. The measurement for video frame interpolation can be defined in two ways. First, we can use sampling operator A ∈ Rt

0×2t0 to draw key frames from a full video. Second, one can use the interpolation operator {A1,A2} to perform temporal downsampling to get key frames, in which the align corners are different in A1 and A2. The measurement matrices are shown below for the 2× interpolation

case:

   (7)

  

1 0 0 0 ··· 0 0 0 0 1 0 ··· 0 0

A =

. . . . . . 0 0 0 0 ··· 1 0

  . (8)

  

  ,A2 =

  

1 0 0 0 ··· 0 0 0 0.5 0.5 0 ··· 0 0

0.5 0.5 0 0 ··· 0 0 0 0 0.5 0.5 ··· 0 0 . . . . . . 0 0 0 0 ··· 0.5 0.5

A1 =

. . . . . . 0 0 0 0 ··· 0.5 0

Intuitively, sampling operator A is less likely to cause blur in key frames since it directly copies corresponding frames. However, we find A1 and A2 especially useful for the key and value in spatial transformer modules. Within these modules, the query determines the scene structure of a frame while the key and value provide textures and appearances. As a result, using interpolation operator for back-projection in the spatial transformer key and value offers additional temporal consistency in visual details without causing blurry scene structures. Examples of hidden state correction is shown in Fig. 3(a). Mathmatically, the correction for temporal transformer hidden states is

hˆtemp = (I − A†A)htemp + A†hk, (9)

where htemp and hˆtemp are the temporal hidden states and the corrected ones. In spatial transformers, the correction is

Qˆ spatial = (I − A†A)Qspatial + A†Wqhk (10) Kˆ spatial = p>0.5Kˆ spatial1 + (1 − p>0.5)Kˆ spatial2 (11) Vˆ spatial = p>0.5Vˆ 1spatial + (1 − p>0.5)Vˆ 2spatial (12)

Kˆ spatial1 = (I − A†1A1)Kspatial + A†1Wkhk,Kˆ spatial2 = (I − A†2A2)Kspatial + A†2Wkhk

- (13)

Vˆ 1spatial = (I − A†1A1)Vspatial + A†1Wvhk,Vˆ 2spatial = (I − A†2A2)Vspatial + A†2Wvhk.

- (14)

where p is a random value drawn from standard Gaussian. p>0.5 equals 1 if p > 0.5 and equals 0 otherwise. We randomly use A1 and A2 to correct K and V, since using only a single interpolation function will cause biased time stamp in interpolated frames (i.e. the 2-nd frame will look like 1.5-th frame if use A1 only) due to the align corner.

##### 4.3 Controlling Correction Strength

In vanilla DDNM, if ϵˆθ(·) approximates the data distribution well, then A†y and (I − A†A)x0|t satisfy the marginal data distribution in the range-space and null-space respectively. Unfortunately,

when combining them together. xˆ0|t may fail to match the joint distribution. This problem also presents in our transformer hidden state back-projection. We propose a simple yet effective remedy for this shown in Fig. 2(b). Instead of using the output oˆ from corrected hidden states alone. Our method utilize a linear interpolation of the original output o and the corrected output oˆ as the final result. Mathmatically, the controlled corrected output is

o˜ = o + wt(oˆ − o). (15)

wt is a predefined control scale that lies between 0 and 1 for denoising timestep t. With Eqn.15, we are able to control how much a denoising step relies on the corrected output. By introducing O to the

final result, we alleviate the artifact caused by the joint distribution mismatch of hˆ.

- 5 Experiments Experiment Settings We deploy our method on three prevalent video diffusers including

- VideoCrafter2[5], LaVie[34] and StableVideoDiffusion[2]. The base model of VideoCrafter2 and

Frame 0 Frame 1 Frame 2 Frame 3 Frame 4 Frame 5

RPE temporal attention APE temporal attention

q kv q kv q kv q kv q kv q kv

|Attn.|
|---|

|window|
|---|

|1 +RPE|
|---|

w

E

- 0
- 1
- 2

Position: 0 1 2 3 4 5

|0|0|
|---|---|

|2|2|
|---|---|

|4|4|
|---|---|

Key frame

In 0 1 2

In

[Figure 62]

[Figure 63]

|0|0|
|---|---|

|1|1|
|---|---|

|2|2|
|---|---|

|3|3|
|---|---|

|4|4|
|---|---|

|5|5|
|---|---|

Vanilla

Positional embedding

|4|4|
|---|---|

|5|5|
|---|---|

|0|0|
|---|---|

|2|2|
|---|---|

|3|3|
|---|---|

|1|1|
|---|---|

|Attn.|
|---|

|window|
|---|

|1+RPE|
|---|

w

- 0
- 1
- 2
- 3
- 4
- 5 0 1 2 3 4 5

|2|2|
|---|---|

|0|0|
|---|---|

|1|1|
|---|---|

|3|3|
|---|---|

|4|4|
|---|---|

|5|5|
|---|---|

- Spatial A1

Temporal

- Spatial A2

w

|Attn.|
|---|

|window|
|---|

|2+RPE|
|---|

- 1
- 2

- 1
- 2

|0|
|---|

|0|
|---|

|4|
|---|

|4|
|---|

|0|
|---|

|0|
|---|

|1|
|---|

|4|
|---|

|4|
|---|

|5|
|---|

+

+

:

) +

…

:

) +

−

(

−

(

In

In 2

[Figure 64]

[Figure 65]

0 0.4 0.8 1.2 1.6

w 3

|Attn.|
|---|

|window|
|---|

|+RPE|
|---|

- 1
- 2

- 1
- 2

|1|
|---|

|0|
|---|

|5|
|---|

|4|
|---|

|1|
|---|

|0|
|---|

|1|
|---|

|5|
|---|

|4|
|---|

|5|
|---|

…

+

+

) +

:

) +

:

−

(

−

(

|2|2|
|---|---|

|0|0|
|---|---|

|5|5|
|---|---|

|1|1|
|---|---|

|3|3|
|---|---|

|4|4|
|---|---|

Positional embedding

- 1
- 2

- 1
- 2

|1|
|---|

|2|
|---|

|4|
|---|

|4|
|---|

|1|
|---|

|1|
|---|

|2|
|---|

|4|
|---|

|3|
|---|

|4|
|---|

+

+

: …

) +

:

) +

−

(

−

(

(a) Hidden State Correction Examples

(b) ZeroSmooth temporal Attention

- Figure 3: (a) Examples for hidden states correction in 2× higher frame rate generation case, showcasing the queries, keys and values are calibration in temporal transformer (Temporal), and in spatial transformers using different interpolation operators (Spatial A1, Spatial A2). (b) We adapt temporal transformers to generate longer sequences in different ways. For the temporal module with relative positional embedding (RPE), we use windowed attention and apply RPE within each window. For absolute positional embedding (APE) modules, we interpolate the position index to get APE before applying attention operation.

LaVie generates 16 frames 320×512 videos using textual prompt. StableVideoDiffusion generates 14 frames 576×1024 videos given a reference start frame. The testing tasks includes text-to-video generation and image-to-video generation. Specifically, we use self-cascaded VideoCrafter2 and LaVie for text-to-video generation, and use self-cascaded StableVideoDiffusion for image-to-video. For all the experiments, we use our method to inference in two untrained frame rate settings including 2× and 4× higher frame rate without any further model tuning. The noise in key frame is set to be identical to the base generation stage in every timesteps to alleviate the stochasticity of denoising. Please see the detailed inference hyperparameters in our appendix.

Testing Datasets and Metrics For text-to-video generation, we use UCF-101[30] as the testing dataset. UCF-101 is a video dataset that includes 101 action categories. To use it as a text-to-video benchmark, we follow the text prompt proposed by Gu et al.[11] For each category of action, we generate 20 videos and getting 2,020 videos in total for every method. For both VideoCrafter2 and LaVie, we generate 320× 512 videos with 16, 32 and 64 frames. VideoCrafter2 uses a frame-persecond condition and we set it to 8, 16, and 32 respectively. The sample quality is evaluated with Inception Score (IS) and Frechet Video Distance (FVD). The consistency to key frame condition is measured by SSIM and PSNR. To get these metrics, we compute the image similarity between key frames in the high frame rate video and the base video. For image-to-video generation, we use WebVid-10M[1] as the testing dataset. It is a high definition video dataset within the domain of StableVideoDiffusion. WebVid-10M contains over 10 million videos in various scenarios. We randomly sample 2,048 video clips from the dataset and use the first frame of each video to serve as reference image. We use StableVideoDiffusion to generate 576 × 1024 videos with 14, 28 and 56 frames. The motion strength is 180, and the frame-per-second is set to 7 in the model. For every method in image-to-video, we generate 2,048 videos. FVD, SSIM and PSNR are used to evaluate the video quality and the similarity of key frames compared to the base video condition.

##### 5.1 Comparison to Tuning-free Methods

We compare our method to the vanilla model (Direct inference), a tuning-free method for visual restoration via back-projection in latent space named DDNM [35]. Our results are shown in Tab.1. A qualitative comparison is shown in Fig.4. For the best visual experience, please see the videos in our supplementary material. We surpass the baselines in most of the metrics. The similarity metrics PSNR and SSIM show our method is able to preserve key frame when generating in high frame rate. Meanwhile, our method achieves the best FVD and IS. ZeroSmooth generates visual plausible and temporal consistency high frame rate videos. We show the method is effective in VideoCrafter2, LaVie and StableVideoDiffusion. It illustrates our method’s potential to be an model agnostic method and can be generalized to other video generative diffusers.

Model Method

FVD↓ IS↑ PSNR↑ SSIM↑ FVD↓ IS↑ PSNR↑ SSIM↑

Dir. Inf. 1202.3 30.31 10.90 0.426 1183.9 21.63 10.81 0.409 DDNM 1139.6 40.99 23.50 0.841 1100.4 35.90 22.21 0.813

VC2

##### Ours 1085.8 43.92 24.12 0.852 1020.6 43.77 22.68 0.837

Dir. Inf. 682.0 28.11 10.21 0.438 834.7 24.23 10.19 0.447 DDNM 729.5 35.59 24.39 0.824 721.0 29.42 26.47 0.817

LaVie

##### Ours 703.2 39.23 26.90 0.876 697.6 38.67 24.17 0.831

Dir. Inf. 1350.7 - 17.33 0.651 1090.2 - 15.75 0.604 DDNM 846.6 - 29.12 0.875 895.2 - 28.12 0.864

SVD

##### Ours 779.6 - 30.92 0.927 752.1 - 30.74 0.921

- Table 1: Quantitative comparisons between tuning-free methods to generate 2× and 4× higher frame rate.

Method

2× 4× FVD↓ IS↑ PSNR↑ SSIM↑ FVD↓ IS↑ PSNR↑ SSIM↑ LDMVFI 1304.6 41.40 34.60 0.948 - - - -

LaVie Intp. - - - - 1172.1 35.78 28.44 0.874 Ours 1085.8 43.92 24.12 0.852 1020.6 43.77 22.68 0.837

- Table 2: Quantitative comparisons between our method and training-based video interpolators.

The result of DDNM shows it is hard to apply back-projection in video latent space for temporal consistent video interpolation. Generating higher frame rate video directly is out of the domain of a pretrained video diffuser. Therefore inference directly deteriorates in visual quality and text-video alignment. Due to the stochasticity introduced by interpolated frames. Direct inference fails to maintain key frame contents even when the key frame noise is identical to that during the base video generation.

##### 5.2 Comparison to Training-based Methods

Despite ZeroSmooth is zero-shot method and no additional tuning or training data is required. We find our method competitive with those training-based video interpolation models. We compare our method with video interpolation models including LDMVFI[9] and LaVie interpolation model [34]. Since our method is aimed for text-to-video and image-to-video generation, but not for real world video interpolation. We use the methods to interpolate the generated videos of VideoCrafter2 base generator for a fair comparison. The results are shown in Tab.2. Our method surpasses training-based method in visual quality and is competitive in content preserving. Please see more visual comparisons in our supplementary material.

##### 5.3 Ablation Study

We conduct an ablation experiment of our method on VideoCrafter2 to generate 320 × 512 videos with 32 and 64 frames. It includes three crucial technical components of our method, including spatial transformers hidden state correction (+ Spatial), spatial hidden state correction with interpolation operator (+ Spatial A1/A2), temporal hidden state correction (+ Temporal), and controlling correction strength (+ CCS). The result is shown in Tab.3. It shows the technical components of ZeroSmooth ensures high quality and content preserving high frame rate video generation. Without the key designs of our method, the visual quality and consistency to key frame content degenerate dramatically.

frame 0 frame 12 frame 13 frame 26 frame 38 frame 39 frame 52 Key

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

frame

|[Figure 69]|[Figure 70]|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]|
|---|---|---|---|---|---|---|

SVD

|[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]|[Figure 81]|[Figure 82]|
|---|---|---|---|---|---|---|

SVD DDNM

|[Figure 83]|[Figure 84]|[Figure 85]|[Figure 86]|[Figure 87]|[Figure 88]|1111<br><br>[Figure 89]|
|---|---|---|---|---|---|---|

Ours

[Figure 90]

[Figure 91]

[Figure 92]

Key frame

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

VC2

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

VC2 DDNM

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Ours

|[Figure 114]|
|---|

||[Figure 115]|
|---|
| | |[Figure 116]| | | | |
|---|---|---|---|---|---|---|---|
|[Figure 117]|[Figure 118]|[Figure 119]|[Figure 120]| |[Figure 121]|[Figure 122]|[Figure 123]|
|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]| |[Figure 128]|[Figure 129]|[Figure 130]|
|[Figure 131]|[Figure 132]|[Figure 133]|[Figure 134]| |[Figure 135]|[Figure 136]|[Figure 137]|

Key frame

LaVie

LaVie DDNM

Ours

Figure 4: Visual comparison between our method and other tuning-free baselines. The green rectangles capture the abrupt changes between adjacent frames.

### 6 Conclusion

We propose a training-free method for promoting generative video diffusion models to generate videos with a high frame rate, i.e., producing more smooth videos. It can be applied to different video models in a plug-and-play manner. Since most video models are trained on videos consisting of sampled key frames due to limited GPU memory, their performance will degenerate sharply if generating a video with more frames than those used in training. We design the self-cascaded framework and the hidden state correction module to avoid the degeneration and achieve temporal consistency between frames. Extensive experiments are conducted with various video models to demonstrate the effectiveness of the proposed method.

Limitations The proposed method is designed for generative video diffusion models. The interpolation performance heavily depends on the capability of the target video model in frame consistency and visual quality. If the generated key frames are inconsistent or blurry, our method will fail to improve the video smoothness.

Setting

FVD↓ IS↑ PSNR↑ SSIM↑ FVD↓ IS↑ PSNR↑ SSIM↑

Vanilla 1202.3 30.31 10.90 0.426 1183.9 21.63 10.81 0.409 + Spatial 1251.2 42.40 22.96 0.739 1116.8 41.19 22.31 0.721 + Temopral 1221.0 43.25 24.03 0.826 1156.1 42.54 22.63 0.817 + Spatial A1/A2 1199.8 43.09 23.22 0.768 1108.4 42.43 22.14 0.755 + CCS 1085.8 43.95 24.12 0.852 1020.6 43.77 22.68 0.837

Table 3: Quantitative results of ablation experiments.

Boarder Impact The proposed interpolation method have a broad impact, enhancing video quality and enabling applications like slow-motion effects, and video compression. They’re beneficial in industries like virtual reality, gaming, and film production. However, they also raise ethical concerns, such as the potential for creating misleading deepfake videos, necessitating their responsible use.

### References

- [1] Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in time: A joint video and image encoder for end-to-end retrieval. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1728–1738 (2021)
- [2] Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V., Rombach, R.: Stable video diffusion: Scaling latent video diffusion models to large datasets (2023)
- [3] Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- [4] Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., Weng, C., Shan, Y.: Videocrafter1: Open diffusion models for high-quality video generation (2023)
- [5] Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: Videocrafter2: Overcoming data limitations for high-quality video diffusion models (2024)
- [6] Choi, J., Kim, S., Jeong, Y., Gwon, Y., Yoon, S.: Ilvr: Conditioning method for denoising diffusion probabilistic models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision

(2021)

- [7] Chung, H., Kim, J., Mccann, M.T., Klasky, M.L., Ye, J.C.: Diffusion posterior sampling for general noisy inverse problems. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=OnD9zGAGT0k
- [8] Danier, D., Zhang, F., Bull, D.: St-mfnet: A spatio-temporal multi-flow network for frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3521–3531 (June 2022)
- [9] Danier, D., Zhang, F., Bull, D.: Ldmvfi: Video frame interpolation with latent diffusion models. arXiv preprint arXiv:2303.09508 (2023)
- [10] Garber, T., Tirer, T.: Image restoration by denoising diffusion models with iteratively preconditioned guidance. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2024)
- [11] Ge, S., Nah, S., Liu, G., Poon, T., Tao, A., Catanzaro, B., Jacobs, D., Huang, J.B., Liu, M.Y., Balaji, Y.: Preserve your own correlation: A noise prior for video diffusion models. arXiv preprint arXiv:2305.10474

(2023)

- [12] He, Y., Yang, S., Chen, H., Cun, X., Xia, M., Zhang, Y., Wang, X., He, R., Chen, Q., Shan, Y.: Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In: The Twelfth International Conference on Learning Representations (2024)
- [13] He, Y., Yang, T., Zhang, Y., Shan, Y., Chen, Q.: Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221 (2022)

- [14] Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022)
- [15] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33, 6840–6851 (2020)
- [16] Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- [17] Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. arXiv preprint arXiv:2204.03458 (2022)
- [18] Jain, S., Watson, D., Tabellion, E., Hoły´nski, A., Poole, B., Kontkanen, J.: Video interpolation with diffusion models (2024)
- [19] Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)
- [20] Kong, L., Jiang, B., Luo, D., Chu, W., Huang, X., Tai, Y., Wang, C., Yang, J.: Ifrnet: Intermediate feature refine network for efficient frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- [21] Lu, L., Wu, R., Lin, H., Lu, J., Jia, J.: Video frame interpolation with transformer. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- [22] Luo, Z., Chen, D., Zhang, Y., Huang, Y., Wang, L., Shen, Y., Zhao, D., Zhou, J., Tan, T.: Videofusion: Decomposed diffusion models for high-quality video generation (2023)
- [23] Lyu, Z., Li, M., Jiao, J., Chen, C.: Frame interpolation with consecutive brownian bridge diffusion (2024)
- [24] Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: International Conference on Machine Learning. pp. 8162–8171. PMLR (2021)
- [25] Qiu, H., Xia, M., Zhang, Y., He, Y., Wang, X., Shan, Y., Liu, Z.: Freenoise: Tuning-free longer video diffusion via noise rescheduling (2023)
- [26] Reda, F., Kontkanen, J., Tabellion, E., Sun, D., Pantofaru, C., Curless, B.: Film: Frame interpolation for large motion. In: European Conference on Computer Vision (ECCV) (2022)
- [27] Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022)
- [28] Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502

(2020)

- [29] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020)
- [30] Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)
- [31] Voleti, V., Jolicoeur-Martineau, A., Pal, C.: Masked conditional video diffusion for prediction, generation, and interpolation. arXiv preprint arXiv:2205.09853 (2022)
- [32] Wang, F.Y., Chen, W., Song, G., Ye, H.J., Liu, Y., Li, H.: Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264 (2023)
- [33] Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: Modelscope text-to-video technical report

(2023)

- [34] Wang, Y., Chen, X., Ma, X., Zhou, S., Huang, Z., Wang, Y., Yang, C., He, Y., Yu, J., Yang, P., Guo, Y., Wu, T., Si, C., Jiang, Y., Chen, C., Loy, C.C., Dai, B., Lin, D., Qiao, Y., Liu, Z.: Lavie: High-quality video generation with cascaded latent diffusion models (2023)
- [35] Wang, Y., Yu, J., Zhang, J.: Zero-shot image restoration using denoising diffusion null-space model. The Eleventh International Conference on Learning Representations (2023)
- [36] Xing, J., Xia, M., Liu, Y., Zhang, Y., Zhang, Y., He, Y., Liu, H., Chen, H., Cun, X., Wang, X., et al.: Make-your-video: Customized video generation using textual and structural guidance. arXiv preprint arXiv:2306.00943 (2023)

- [37] Xue, T., Chen, B., Wu, J., Wei, D., Freeman, W.T.: Video enhancement with task-oriented flow. International Journal of Computer Vision 127(8), 1106–1125 (Feb 2019). https://doi.org/10.1007/s11263-01801144-2, http://dx.doi.org/10.1007/s11263-018-01144-2
- [38] Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021)
- [39] Yang, S., Zhou, Y., Liu, Z., , Loy, C.C.: Rerender a video: Zero-shot text-guided video-to-video translation. In: ACM SIGGRAPH Asia Conference Proceedings (2023)
- [40] Yu, J., Cun, X., Qi, C., Zhang, Y., Wang, X., Shan, Y., Zhang, J.: Animatezero: Video diffusion models are zero-shot image animators. arXiv preprint arXiv:2312.03793 (2023)
- [41] Zhang, G., Zhu, Y., Wang, H., Chen, Y., Wu, G., Wang, L.: Extracting motion and appearance via interframe attention for efficient video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5682–5692 (2023)
- [42] Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., Gool, L.V.: Denoising diffusion models for plug-and-play image restoration. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops (NTIRE) (2023)

### A Appendix

##### A.1 ZeroSmooth Temporal Attention

Detailed computation of ZeroSmooth temporal attention is shown in this subsection. As shown in the main text, we have different strategies for temporal attention modules with relative positional embedding (RPE) and absolute positional embedding (APE). Specifically, in our experiment, LaVie uses RPE in temporal transformers. StableVideoDiffusion uses APE in temporal transformers. VideoCrafter2 learns relative position information within its temporal convolution blocks and no positional embedding is used in temporal transformers. We treat VideoCrafter2 as it has empty RPE.

Absolute positional embedding We interpolate the position index before computing absolute positional embedding. In the base video model, the video length is t0. Consider a adapted model that generates t frames. StableVideoDiffusion uses sinusoidal absolute postition embedding. Mathmatically, the APE is then interpolated

pos · t0 100002i/d · t

, (16)

APE(pos,2i) = sin

pos · t0 100002i/d · t

. (17)

APE(pos,2i+1) = cos

pos ∈ {1,2,··· ,t} is the position of the current token. Then a linear mapping layer is applied to get the final absolute positional embedding U ∈ Rt×d. The APE is added to hidden state h before attention operation. We then use windowed attention where the window size equals to sequence length during model training to maintain the perception field of attention operation. Our method ensures there are overlaps between attention windows, such that we can apply the attention fusion technique proposed by Qiu et al[25]. to achieve smooth transition between adjacent attention windows. We use an overlap of ⌊t0/2⌋ in our experiment. The the i-th attention window starts at position istart = min(it0,t − t0) and ends at iend = min((i + 1)t0,t). We use the subscript i to denote the tokens within position istart and iend. The output of the i-th attention window is:

oi = Attention(hi + Ui) (18)

Wq(hi + Ui)(hi + Ui)TWkT

Wv(hi + Ui), (19) where Wq,Wk,Wv are the projection weight for query, key and value. The final output is

√

= Softmax

d

o = AttentionFusion(o1,o2,··· ,on). (20) AttentionFusion(·) is the attention fusion operator in [25]. n is the number of windows, one can compute it through n = ⌈(t − t0)/(t0 − ⌊t0/2⌋)⌉, where ⌈·⌉ is the ceil operator.

Relative positional embedding There are many forms of RPEs. Without loss of generality, we consider trainable relative positional embedding in the followings. Let pk ∈ Rt

0×d and pv ∈ Rt

0×d

denote the learned relative position. Then we also apply the windowed attention and attention fusion strategy. Mathmatically, the attention output is

Wqhi(hi + pk)TWk

Wv(hi + pv) (21) o = AttentionFusion(o1,o2,··· ,on) (22)

√

oi = Softmax

d

##### A.2 Implementation Details

We describe the implementation details that is not included in the main text in this subsection, showcasing the adaptation methods for all the used models including VideoCrafter2, LaVie and StableVideoDiffusion.

Modules to apply ZeroSmooth Modern video diffusion UNet includes 6 categories of neural network modules. ZeroSmooth is applied to spatial self-attention, spatial cross-attention, temporal self-attention and temporal cross-attention. The remaining modules including spatial ResNet block

output sequence 0 2 4 6

output sequence 0 1 2 3

output sequence 0 4

4 5 6 7

share

share

Temporal Transformer

Temporal Transformer

Temporal Transformer

Hidden state correction

Hidden state correction ZeroSmooth

ZeroSmooth

share

share

Spatial Transformer

Spatial Transformer

Spatial Transformer

Hidden state correction

Hidden state correction

ZeroSmooth

ZeroSmooth

share

share

ResNet Block ResBlock

ResBlock

× N

× N

× N

[Figure 138]

[Figure 139]

[Figure 140]

Base model ZeroSmooth model

ZeroSmooth model

0 1 2 3 4 5 6 7

0 4 0 2 4 6

Input sequence Input sequence

Input sequence

Figure 5: An illustration of three stages self-cascaded model.

and temporal ResNet block are kept the same to the base generator. We use our method to all the modules in the selected categories for all denoising timesteps. When computing the key and the value, cross-attention modules uses text prompts or image embedding in text-to-video and image-to-video respectively. Since the condition is repeated for all frames, our hidden state correction will not change the value of these conditions.

Stochasticity control We use a noise scheduling strategy to reduce the stochasticity during denoising sampling. Consider a N-stage self-cascaded ZeroSmooth model that includes

N−1t0×hwc → Rn

0×hwc → Rt

0×hwc,ϵ2θ(·) : Rnt

0×hwc → Rnt

0×hwc,··· ,ϵNθ (·) : Rn

{ϵ1θ(·) : Rt

N−1t0×hwc}. n is the frame interpolation scale. Let ϵN ∈ Rn

N−1t0×hwc denote the additional

standard Gaussian noise introduced to the denoising iteration of ϵNθ (·) for a specific denoising timestep. Let the lower script i denote the noise in i-th frame. Then, the additional noised used for

ϵsθ(·) is sampled from ϵN via

ϵsi = ϵNinN−s. (23) This strategy ensures the noise at the corresponding frame to be the same in all self-cascaded stages. VideoCrafter2 VideoCrafter2 uses an additional frame-per-second condition to control the motion speed presents in the video. Given a fps condition for the base generator. We multiply it with the interpolation scale to get the fps condition of a spefic self-cascaded stage.

StableVideoDiffusion StableVideoDiffusion uses two additional conditions including fps and motion bucket id. Motion bucket id controls the the motion amplitude in the video. Since the overall motion is kept unchanged in high frame video, the motion bucket id is changed in all stages. We apply a similar strategy to set fps like in VideoCrafter2.

Unlike text-to-image video diffusers, StableVideoDiffusion concatenate a noisy reference image to every frame to provide a guidance for visual appearance. The noise strength added to the reference image is callded augmentation strength. In our method, we use a uniform augmentation strength for all stages. StableVideoDiffusion uses classifier-free guidance [16] during sampling. The guidance scale is linear monotonically decreasing along the frame number. To adapt this feature in ZeroSmooth, we fix the highest and the lowest guidance scale and use linear interpolation to get the guidance scale for the high frame rate video.

Self-cascaded model Our experiment includes 2× and 4× higher frame rate generation. We use two stages for the 2× experiment and use three stages for the 4× experiment. Every stage will generate 2× higher frame rate videos. A illustration of the three stages model is shown in Fig.5. We pass the corrected states of the second stage hˆ,Qˆ ,Kˆ ,Vˆ to calibrate the hidden states in the third stage.

##### Hyperparameter VideoCrafter2 LaVie StableVideoDiffusion

Frame resolution 320 × 512 320 × 512 576 × 1024 Frame number (16, 32, 64) (16, 32, 64) (14, 28, 56) FPS condition 8 - 7 Motion bucket id - - 180 Aug. strength - - 0.02 Inference steps 50 50 25 Eta 1.0 1.0 Temp. attn. overlap 8 8 7 Sampler DDIM samplre DDIM sampler Euler discrete sampler Guidance scale 12.0 7.5 7.5

Table 4: Hyperparameters of ZeroSmooth.

Controlling correction strength The weight controlling correction strength is defined as

wt = 0.8 t/T, (24)

where T is the maximum denoising timestep. We find in the early stage of denoising, the model is more tolerant to hidden states correction. Therefore we build a monotonically decreasing wt. In the late stage of denoising, the strength of hidden states correction becomes small.

Correcting color tone An additional normalization is applied to avoid abrupt color tone change in the interpolated frames. We find this helps in some special cases. Specifically, after each denoising iteration. We use adaptive instance normalization to calibrate the statistics of frames in the high frame rate video to match the corresponding key frames.

##### A.3 Experiment Settings

The hyperparameter used for the experiment is shown in Tab.4. Aug. strength denotes noise augmentation strength used for StableVideoDiffusion in image-to-video. FPS condition is an input for VideoCrafter2 and StableVideoDiffusion. Temporal attention overlap determines how many frames are shared between attention windows in temporal transformers. Eta controls the noise added for every denoising step in DDIM sampler. We use a stochastic samping trajectory for StableVideoDiffusion, no additional randomness is added after the initialization during sampling.

When using LDMVFI, we apply DDIM sampler. The inference timestep to 50 and eta is set to 0. Classifier-free guidance scale is 0 for LDMVFI. The model achieves 2× frame interpolation. When using LaVie interpolation model, we also use DDIM sampler. The inference timestep is 25 and eta is 0. Classifier-free guidance scale for LaVie interpolator is 4.

We deploy all the models in fp16 precision during inference. 32 NVIDIA V100 32GB GPUs are used to sample the results. Due to the memory limit, we use flash attention to reduce the VRAM requirements. The transformer hidden states in the previous stage are offloaded to CPU memory and is thrown to VRAM when it is need.

##### A.4 Additional Visual Comparisons

More visual comparisons between tuning-free methods and comparions between ZeroSmooth and training-based methods are shown below.

frame 1 frame 2 frame 3 frame 4 frame 5

|[Figure 141]<br><br>[Figure 142]|
|---|

|[Figure 143]<br><br>[Figure 144]|
|---|

[Figure 145]

Key frame

[Figure 146]

|[Figure 147]|[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]|[Figure 151]|[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]|[Figure 155]<br><br>[Figure 156]|
|---|---|---|---|---|

[Figure 157]

VC2

|[Figure 158]<br><br>[Figure 159]|[Figure 160]|[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]|[Figure 164]|[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]|
|---|---|---|---|---|

[Figure 168]

VC2 DDNM

[Figure 169]

|[Figure 170]<br><br>[Figure 171]|[Figure 172]<br><br>[Figure 173]|[Figure 174]|[Figure 175]<br><br>[Figure 176]|[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]|
|---|---|---|---|---|

[Figure 180]

Ours

[Figure 181]

[Figure 182]

“A knight riding on a horse through the countryside”

|[Figure 183]<br><br>[Figure 184]|
|---|

|[Figure 185]<br><br>[Figure 186]|
|---|

[Figure 187]

Key frame

[Figure 188]

|[Figure 189]|[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]|[Figure 193]|[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]|[Figure 197]<br><br>[Figure 198]|
|---|---|---|---|---|

[Figure 199]

VC2

|[Figure 200]|[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]|[Figure 204]|[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]|[Figure 208]<br><br>[Figure 209]|
|---|---|---|---|---|

[Figure 210]

VC2 DDNM

[Figure 211]

|[Figure 212]|[Figure 213]<br><br>[Figure 214]|[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]|[Figure 218]|[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]|
|---|---|---|---|---|

[Figure 222]

Ours

[Figure 223]

[Figure 224]

[Figure 225]

#### “A panda eats a pizza, wearing space suit, closeup, photorealistic”

Figure 6: Visual comparison between our method and other tuning-free baselines on VideoCrafter2.

frame 1 frame 2 frame 3 frame 4 frame 5

|[Figure 226]<br><br>[Figure 227]|
|---|

|[Figure 228]<br><br>[Figure 229]|
|---|

[Figure 230]

Key frame

[Figure 231]

|[Figure 232]|[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]|[Figure 236]|[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]|[Figure 240]<br><br>[Figure 241]|
|---|---|---|---|---|

[Figure 242]

LaVie

|[Figure 243]<br><br>[Figure 244]|[Figure 245]|[Figure 246]<br><br>[Figure 247]|[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]|[Figure 251]<br><br>[Figure 252]|
|---|---|---|---|---|

[Figure 253]

LaVie DDNM

[Figure 254]

[Figure 255]

|[Figure 256]|[Figure 257]<br><br>[Figure 258]|[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]|[Figure 262]<br><br>[Figure 263]|[Figure 264]<br><br>[Figure 265]|
|---|---|---|---|---|

[Figure 266]

Ours

[Figure 267]

[Figure 268]

“The lotus blossoms in a fairytale pond, time-lapse photography, DLSR”

|[Figure 269]<br><br>[Figure 270]|
|---|

|[Figure 271]<br><br>[Figure 272]|
|---|

[Figure 273]

Key frame

[Figure 274]

|[Figure 275]|[Figure 276]<br><br>[Figure 277]|[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]|[Figure 281]<br><br>[Figure 282]|[Figure 283]<br><br>[Figure 284]|
|---|---|---|---|---|

[Figure 285]

LaVie

|[Figure 286]<br><br>[Figure 287]|[Figure 288]<br><br>[Figure 289]|[Figure 290]|[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]|[Figure 294]<br><br>[Figure 295]|
|---|---|---|---|---|

[Figure 296]

LaVie DDNM

[Figure 297]

[Figure 298]

|[Figure 299]<br><br>[Figure 300]|[Figure 301]<br><br>[Figure 302]|[Figure 303]|[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]|[Figure 307]<br><br>[Figure 308]|
|---|---|---|---|---|

[Figure 309]

Ours

[Figure 310]

[Figure 311]

astronaut on the mars waving a flag, photorealistic, 4k

Figure 7: Visual comparison between our method and other tuning-free baselines on LaVie[34].

frame 1 frame 2 frame 3 frame 4 frame 5

|[Figure 312]<br><br>[Figure 313]|
|---|

|[Figure 314]<br><br>[Figure 315]|
|---|

[Figure 316]

Key frame

[Figure 317]

|[Figure 318]|[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]|[Figure 322]<br><br>[Figure 323]|[Figure 324]|[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]|
|---|---|---|---|---|

[Figure 328]

SVD

|[Figure 329]<br><br>[Figure 330]|[Figure 331]|[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]|[Figure 335]<br><br>[Figure 336]|[Figure 337]<br><br>[Figure 338]|
|---|---|---|---|---|

[Figure 339]

[Figure 340]

SVD DDNM

[Figure 341]

|[Figure 342]|[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]|[Figure 346]|[Figure 347]<br><br>[Figure 348]|[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]|
|---|---|---|---|---|

[Figure 352]

Ours

|[Figure 353]<br><br>[Figure 354]|
|---|

|[Figure 355]<br><br>[Figure 356]|
|---|

[Figure 357]

Key frame

[Figure 358]

|[Figure 359]<br><br>[Figure 360]|[Figure 361]<br><br>[Figure 362]|[Figure 363]|[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]|[Figure 367]<br><br>[Figure 368]|
|---|---|---|---|---|

[Figure 369]

SVD

|[Figure 370]<br><br>[Figure 371]|[Figure 372]<br><br>[Figure 373]|[Figure 374]<br><br>[Figure 375]|[Figure 376]|[Figure 377]<br><br>[Figure 378]<br><br>[Figure 379]|
|---|---|---|---|---|

[Figure 380]

[Figure 381]

SVD DDNM

[Figure 382]

|[Figure 383]<br><br>[Figure 384]|[Figure 385]<br><br>[Figure 386]|[Figure 387]|[Figure 388]<br><br>[Figure 389]|[Figure 390]<br><br>[Figure 391]<br><br>[Figure 392]|
|---|---|---|---|---|

[Figure 393]

Ours

Figure 8: Visual comparison between our method and other tuning-free baselines on StableVideoDiffusion.

frame 1 frame 2 frame 3 frame 4 frame 5

|[Figure 394]<br><br>[Figure 395]|
|---|

|[Figure 396]<br><br>[Figure 397]|
|---|

|[Figure 398]<br><br>[Figure 399]|
|---|

[Figure 400]

Key frame

[Figure 401]

|[Figure 402]<br><br>[Figure 403]|[Figure 404]<br><br>[Figure 405]|[Figure 406]|[Figure 407]<br><br>[Figure 408]|[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]|
|---|---|---|---|---|

[Figure 412]

LDMVFI

|[Figure 413]|[Figure 414]<br><br>[Figure 415]|[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]|[Figure 419]<br><br>[Figure 420]|[Figure 421]<br><br>[Figure 422]|
|---|---|---|---|---|

[Figure 423]

Ours

[Figure 424]

[Figure 425]

“Impressionist style, a yellow rubber duck floating on the wave on the sunset”

|[Figure 426]<br><br>[Figure 427]|
|---|

|[Figure 428]<br><br>[Figure 429]|
|---|

|[Figure 430]<br><br>[Figure 431]|
|---|

[Figure 432]

Key frame

[Figure 433]

|[Figure 434]<br><br>[Figure 435]|[Figure 436]<br><br>[Figure 437]|[Figure 438]|[Figure 439]<br><br>[Figure 440]|[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]|
|---|---|---|---|---|

[Figure 444]

LDMVFI

|[Figure 445]<br><br>[Figure 446]|[Figure 447]|[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]|[Figure 451]<br><br>[Figure 452]|[Figure 453]<br><br>[Figure 454]|
|---|---|---|---|---|

[Figure 455]

Ours

[Figure 456]

[Figure 457]

“astronaut riding a horse on the mars, photorealistic, 4k”

Figure 9: Visual comparison between our method and LDMVFI [9].

frame 1 frame 2 frame 3 frame 4 frame 5

|[Figure 458]<br><br>[Figure 459]|
|---|

|[Figure 460]<br><br>[Figure 461]|
|---|

[Figure 462]

Key frame

[Figure 463]

|[Figure 464]<br><br>[Figure 465]|[Figure 466]|[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]|[Figure 470]|[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]|
|---|---|---|---|---|

[Figure 474]

LaVie Interp.

[Figure 475]

|[Figure 476]|[Figure 477]<br><br>[Figure 478]|[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]|[Figure 482]|[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]|
|---|---|---|---|---|

[Figure 486]

Ours

[Figure 487]

[Figure 488]

“A pink llama wearing sunglasses, walking slowly near a caffe in Florence.”

|[Figure 489]<br><br>[Figure 490]|
|---|

|[Figure 491]<br><br>[Figure 492]|
|---|

[Figure 493]

Key frame

[Figure 494]

|[Figure 495]<br><br>[Figure 496]|[Figure 497]|[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]|[Figure 501]<br><br>[Figure 502]|[Figure 503]<br><br>[Figure 504]|
|---|---|---|---|---|

[Figure 505]

LaVie Interp.

[Figure 506]

|[Figure 507]<br><br>[Figure 508]|[Figure 509]|[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]|[Figure 513]|[Figure 514]<br><br>[Figure 515]<br><br>[Figure 516]|
|---|---|---|---|---|

[Figure 517]

Ours

[Figure 518]

[Figure 519]

“A young woman with glasses is jogging in the park wearing a pink headband.”

Figure 10: Visual comparison between our method and LaVie interpolation model [34].

